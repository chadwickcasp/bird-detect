"""Compare bird boxes across PyTorch, exported runtimes, and the phone.

Run from the repository root:

    .venv/bin/python -m ml.compare_predictions host
    .venv/bin/python -m ml.compare_predictions phone outputs/parity/phone_cpu.json

`host` predicts the parity fixtures with the PyTorch checkpoint, the Ultralytics ncnn runtime,
and the LiteRT export. It writes outputs/parity/host_predictions.json. `phone` checks a
predictions.json written by the Android app against the host ncnn boxes.

Every predictions file uses the same shape:

    {"source": "...", "images": {"bird_gull.jpg": {"width": W, "height": H,
        "boxes": [{"xyxy": [x0, y0, x1, y1], "score": s}]}}}

Box coordinates are original-image pixels.
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from ultralytics import YOLO

from ml.export_pretrained import (
    COCO_BIRD_CLASS,
    CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    NMS_IOU_THRESHOLD,
)
from ml.project_paths import FIXTURES_DIR, OUTPUTS_DIR, WEIGHTS_DIR

MATCH_IOU = 0.50
HOST_PREDICTIONS_PATH = OUTPUTS_DIR / "parity" / "host_predictions.json"
HOST_MODELS = {
    "pytorch": WEIGHTS_DIR / "yolo26n.pt",
    "ncnn": WEIGHTS_DIR / "yolo26n_ncnn_model",
    "litert": WEIGHTS_DIR / "yolo26n.tflite",
}


def list_fixture_images(fixtures_dir: Path) -> list[Path]:
    """Return the fixture images in name order."""
    return sorted(path for path in fixtures_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})


def predict_bird_boxes(model_path: Path, images: Sequence[Path]) -> dict:
    """Predict bird boxes for each image with one Ultralytics-loadable model."""
    model = YOLO(str(model_path), task="detect")
    predictions = {}
    for image in images:
        result = model.predict(
            str(image),
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            classes=[COCO_BIRD_CLASS],
            verbose=False,
        )[0]
        height, width = result.orig_shape
        boxes = [
            {"xyxy": [round(value, 2) for value in xyxy], "score": round(score, 4)}
            for xyxy, score in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist())
        ]
        predictions[image.name] = {"width": width, "height": height, "boxes": boxes}
    return {"source": f"ultralytics:{model_path.name}", "images": predictions}


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Return intersection over union for two xyxy boxes."""
    overlap_w = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    overlap_h = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    overlap = overlap_w * overlap_h
    union = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - overlap
    return overlap / union if union > 0 else 0.0


def match_bird_boxes(reference: Sequence[Mapping], candidate: Sequence[Mapping]) -> dict:
    """Greedily pair boxes by IoU and report matches, misses, and extras for one image."""
    unmatched = list(range(len(candidate)))
    pairs = []
    missed = 0
    for ref_box in sorted(reference, key=lambda box: -box["score"]):
        best_index, best_iou = None, MATCH_IOU
        for index in unmatched:
            iou = box_iou(ref_box["xyxy"], candidate[index]["xyxy"])
            if iou >= best_iou:
                best_index, best_iou = index, iou
        if best_index is None:
            missed += 1
            continue
        unmatched.remove(best_index)
        pairs.append({"iou": round(best_iou, 4), "score_diff": round(candidate[best_index]["score"] - ref_box["score"], 4)})
    return {"matched": pairs, "missed": missed, "extra": len(unmatched)}


def compare_bird_boxes(reference: Mapping, candidate: Mapping) -> dict:
    """Compare two predictions files image by image.

    Parity passes when every reference box has a candidate at IoU >= 0.50 and the candidate
    has no extra boxes. An image missing from the candidate fails.
    """
    per_image = {}
    for name, ref_image in reference["images"].items():
        cand_image = candidate["images"].get(name)
        if cand_image is None:
            per_image[name] = {"matched": [], "missed": len(ref_image["boxes"]), "extra": 0, "absent": True}
            continue
        per_image[name] = match_bird_boxes(ref_image["boxes"], cand_image["boxes"])
    passed = all(not image.get("absent") and image["missed"] == 0 and image["extra"] == 0 for image in per_image.values())
    return {
        "reference": reference["source"],
        "candidate": candidate["source"],
        "passed": passed,
        "images": per_image,
    }


def summarize_comparison(report: Mapping) -> str:
    """Return a short human-readable summary of one comparison."""
    lines = [f"{report['candidate']} vs {report['reference']}: {'PASS' if report['passed'] else 'FAIL'}"]
    for name, image in report["images"].items():
        ious = [pair["iou"] for pair in image["matched"]]
        diffs = [abs(pair["score_diff"]) for pair in image["matched"]]
        detail = f"matched={len(ious)} missed={image['missed']} extra={image['extra']}"
        if ious:
            detail += f" min_iou={min(ious):.3f} max_score_diff={max(diffs):.4f}"
        if image.get("absent"):
            detail += " (image absent)"
        lines.append(f"  {name}: {detail}")
    return "\n".join(lines)


def write_host_predictions(output_path: Path) -> dict:
    """Predict the fixtures with every host model, write them, and print the host comparisons."""
    images = list_fixture_images(FIXTURES_DIR)
    host = {name: predict_bird_boxes(path, images) for name, path in HOST_MODELS.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(host, indent=2) + "\n")
    print(summarize_comparison(compare_bird_boxes(host["pytorch"], host["ncnn"])))
    print(summarize_comparison(compare_bird_boxes(host["ncnn"], host["litert"])))
    print(f"Wrote {output_path}")
    return host


def check_phone_predictions(phone_path: Path, host_path: Path, reference_name: str) -> bool:
    """Compare a phone predictions file with one host runtime and return whether parity passed."""
    host = json.loads(host_path.read_text())
    phone = json.loads(phone_path.read_text())
    report = compare_bird_boxes(host[reference_name], phone)
    print(summarize_comparison(report))
    report_path = phone_path.with_name(phone_path.stem + "_parity.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {report_path}")
    return report["passed"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    host_parser = commands.add_parser("host", help="predict fixtures with PyTorch, ncnn, and LiteRT")
    host_parser.add_argument("--output", type=Path, default=HOST_PREDICTIONS_PATH)
    phone_parser = commands.add_parser("phone", help="check a phone predictions.json against the host")
    phone_parser.add_argument("predictions", type=Path)
    phone_parser.add_argument("--host", type=Path, default=HOST_PREDICTIONS_PATH)
    phone_parser.add_argument("--reference", choices=sorted(HOST_MODELS), default="ncnn")
    args = parser.parse_args()

    if args.command == "host":
        write_host_predictions(args.output)
        return
    if not check_phone_predictions(args.predictions, args.host, args.reference):
        sys.exit(1)


if __name__ == "__main__":
    main()
