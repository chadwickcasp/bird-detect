"""Export COCO-pretrained YOLO26n to ncnn and LiteRT and record each tensor contract.

Run from the repository root:

    .venv/bin/python -m ml.export_pretrained

The record is built by reading the exported files, not by restating documentation.
"""

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import ai_edge_litert
import cv2
import ncnn
import numpy as np
import ultralytics
from ai_edge_litert.interpreter import Interpreter
from ultralytics import YOLO

from ml.project_paths import EXPORT_RECORD_PATH, FIXTURES_DIR, WEIGHTS_DIR

IMAGE_SIZE = 640
COCO_BIRD_CLASS = 14
# Thresholds shared by the host reference and the phone decoder.
CONFIDENCE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.45
LETTERBOX_PAD_VALUE = 114
BOX_UNITS_FIXTURE = FIXTURES_DIR / "bird_gull.jpg"


def hash_file(path: Path) -> str:
    """Return the SHA-256 hex digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_ncnn_io_blobs(param_path: Path) -> tuple[str, str]:
    """Return the input blob and the single unconsumed output blob of an ncnn graph."""
    lines = param_path.read_text().splitlines()[2:]
    input_blob = None
    produced = []
    consumed = set()
    for line in lines:
        fields = line.split()
        layer_type, input_count, output_count = fields[0], int(fields[2]), int(fields[3])
        inputs = fields[4 : 4 + input_count]
        outputs = fields[4 + input_count : 4 + input_count + output_count]
        if layer_type == "Input":
            input_blob = outputs[0]
        consumed.update(inputs)
        produced.extend(outputs)
    graph_outputs = [blob for blob in produced if blob not in consumed]
    if input_blob is None or len(graph_outputs) != 1:
        raise ValueError(f"Expected one Input and one output blob in {param_path}, got {graph_outputs}")
    return input_blob, graph_outputs[0]


def describe_ncnn_model(model_dir: Path) -> dict:
    """Run a zero image through the ncnn export and report its blob names and output shape."""
    param_path = model_dir / "model.ncnn.param"
    bin_path = model_dir / "model.ncnn.bin"
    input_blob, output_blob = find_ncnn_io_blobs(param_path)

    net = ncnn.Net()
    net.load_param(str(param_path))
    net.load_model(str(bin_path))
    extractor = net.create_extractor()
    extractor.input(input_blob, ncnn.Mat(np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)))
    _, output = extractor.extract(output_blob)

    return {
        "param": param_path.name,
        "bin": bin_path.name,
        "param_sha256": hash_file(param_path),
        "bin_sha256": hash_file(bin_path),
        "bin_bytes": bin_path.stat().st_size,
        "input_blob": input_blob,
        "input_shape_chw": [3, IMAGE_SIZE, IMAGE_SIZE],
        "output_blob": output_blob,
        "output_shape_hw": [output.h, output.w],
    }


def describe_litert_model(tflite_path: Path) -> dict:
    """Report the signature tensors of the LiteRT export."""
    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    model_input = interpreter.get_input_details()[0]
    model_output = interpreter.get_output_details()[0]
    return {
        "file": tflite_path.name,
        "sha256": hash_file(tflite_path),
        "bytes": tflite_path.stat().st_size,
        "input_name": model_input["name"],
        "input_shape": model_input["shape"].tolist(),
        "input_dtype": np.dtype(model_input["dtype"]).name,
        "output_name": model_output["name"],
        "output_shape": model_output["shape"].tolist(),
        "output_dtype": np.dtype(model_output["dtype"]).name,
    }


def letterbox_image(image_path: Path) -> np.ndarray:
    """Return the image as a letterboxed RGB CHW float array scaled to 0-1, matching the phone preprocessing."""
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_LINEAR)
    pad_w = (IMAGE_SIZE - resized.shape[1]) / 2
    pad_h = (IMAGE_SIZE - resized.shape[0]) / 2
    padded = cv2.copyMakeBorder(
        resized,
        round(pad_h - 0.1),
        round(pad_h + 0.1),
        round(pad_w - 0.1),
        round(pad_w + 0.1),
        cv2.BORDER_CONSTANT,
        value=(LETTERBOX_PAD_VALUE,) * 3,
    )
    return np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255


def measure_box_units(ncnn_dir: Path, tflite_path: Path, image_path: Path) -> dict:
    """Run one bird image through both exports and report the unit of each export's box rows.

    Ultralytics normalizes LiteRT boxes by the input size and rescales them in its own runtime,
    so the raw tensors of the two exports can differ by a factor of IMAGE_SIZE.
    """
    image = letterbox_image(image_path)
    net = ncnn.Net()
    net.load_param(str(ncnn_dir / "model.ncnn.param"))
    net.load_model(str(ncnn_dir / "model.ncnn.bin"))
    extractor = net.create_extractor()
    input_blob, output_blob = find_ncnn_io_blobs(ncnn_dir / "model.ncnn.param")
    extractor.input(input_blob, ncnn.Mat(image))
    ncnn_output = np.array(extractor.extract(output_blob)[1])

    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], image[None])
    interpreter.invoke()
    litert_output = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0]

    best = int(ncnn_output[4 + COCO_BIRD_CLASS].argmax())
    scale = float(np.mean(ncnn_output[:4, best] / litert_output[:4, best]))
    if abs(scale - 1) < 0.01:
        litert_units = "input pixels"
    elif abs(scale - IMAGE_SIZE) < 1:
        litert_units = f"fraction of the {IMAGE_SIZE}-pixel input"
    else:
        raise ValueError(f"LiteRT boxes differ from ncnn boxes by an unexpected factor {scale:.3f}")
    return {"ncnn": "input pixels", "litert": litert_units, "ncnn_over_litert": round(scale, 3)}


def export_pretrained_yolo26n(weights_dir: Path, record_path: Path) -> dict:
    """Export yolo26n.pt to ncnn and LiteRT, then write and return the contract record."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = weights_dir / "yolo26n.pt"
    model = YOLO(str(checkpoint))
    if model.names[COCO_BIRD_CLASS] != "bird":
        raise ValueError(f"COCO class {COCO_BIRD_CLASS} is {model.names[COCO_BIRD_CLASS]!r}, not 'bird'")

    ncnn_dir = Path(model.export(format="ncnn", imgsz=IMAGE_SIZE))
    tflite_path = Path(model.export(format="litert", imgsz=IMAGE_SIZE))
    ncnn_record = describe_ncnn_model(ncnn_dir)
    litert_record = describe_litert_model(tflite_path)
    box_units = measure_box_units(ncnn_dir, tflite_path, BOX_UNITS_FIXTURE)

    candidate_count = ncnn_record["output_shape_hw"][1]
    class_count = len(model.names)
    if ncnn_record["output_shape_hw"] != [4 + class_count, candidate_count]:
        raise ValueError(f"Unexpected ncnn output shape {ncnn_record['output_shape_hw']}")
    if litert_record["output_shape"] != [1, 4 + class_count, candidate_count]:
        raise ValueError(f"Unexpected LiteRT output shape {litert_record['output_shape']}")

    record = {
        "model": "yolo26n",
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": hash_file(checkpoint),
        "exported_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "versions": {
            "ultralytics": ultralytics.__version__,
            "ncnn_python": ncnn.__version__,
            "ai_edge_litert": ai_edge_litert.__version__,
        },
        "head": "one-to-many (nms=None); NMS runs outside the model",
        "preprocess": {
            "image_size": IMAGE_SIZE,
            "channel_order": "RGB",
            "layout": "NCHW",
            "scale": "pixel / 255",
            "letterbox": "keep aspect, center padding",
            "pad_value": LETTERBOX_PAD_VALUE,
        },
        "output": {
            "rows": "0-3 box center x, center y, width, height (units in box_units); 4+ per-class sigmoid scores",
            "box_units": box_units,
            "class_count": class_count,
            "candidate_count": candidate_count,
            "bird_class_index": COCO_BIRD_CLASS,
        },
        "postprocess": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "nms_iou_threshold": NMS_IOU_THRESHOLD,
        },
        "ncnn": {"dir": ncnn_dir.name, **ncnn_record},
        "litert": litert_record,
    }
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR)
    parser.add_argument("--record", type=Path, default=EXPORT_RECORD_PATH)
    args = parser.parse_args()
    record = export_pretrained_yolo26n(args.weights_dir, args.record)
    print(json.dumps({"ncnn": record["ncnn"], "litert": record["litert"]}, indent=2))
    print(f"Wrote {args.record}")


if __name__ == "__main__":
    main()
