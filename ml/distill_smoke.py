"""Check that the pinned Ultralytics distillation path trains and exports only the student.

Run from the repository root:

    .venv/bin/python -m ml.distill_smoke

This trains a one-class YOLO26s teacher for one epoch on the parity fixtures, distills it into
YOLO26n for one epoch, and exports the student to ncnn. The fixtures are both the training and
validation set. The run checks the API and the export. It is not an accuracy result.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

from ml.compare_predictions import HOST_PREDICTIONS_PATH, list_fixture_images
from ml.export_pretrained import IMAGE_SIZE, describe_ncnn_model
from ml.label_files import YoloBox, format_label_line
from ml.project_paths import FIXTURES_DIR, OUTPUTS_DIR, WEIGHTS_DIR

SMOKE_DIR = OUTPUTS_DIR / "distill_smoke"
DISTILL_WEIGHT = 6.0


def build_fixture_dataset(dataset_dir: Path, host_predictions_path: Path) -> Path:
    """Write a one-class YOLO dataset from the fixtures and the host PyTorch boxes. Return its yaml."""
    host = json.loads(host_predictions_path.read_text())["pytorch"]["images"]
    image_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for image in list_fixture_images(FIXTURES_DIR):
        shutil.copy(image, image_dir / image.name)
        prediction = host[image.name]
        width, height = prediction["width"], prediction["height"]
        lines = []
        for box in prediction["boxes"]:
            x0, y0, x1, y1 = box["xyxy"]
            yolo_box = YoloBox((x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height)
            lines.append(format_label_line(yolo_box))
        (label_dir / image.with_suffix(".txt").name).write_text("".join(f"{line}\n" for line in lines))
    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(f"path: {dataset_dir}\ntrain: images\nval: images\nnames:\n  0: bird\n")
    return data_yaml


def train_one_epoch(checkpoint: Path, data_yaml: Path, run_name: str, teacher: Path | None) -> Path:
    """Train one epoch on CPU and return the saved last.pt."""
    model = YOLO(str(checkpoint))
    model.train(
        data=str(data_yaml),
        epochs=1,
        imgsz=IMAGE_SIZE,
        batch=3,
        device="cpu",
        workers=0,
        seed=0,
        deterministic=True,
        project=str(SMOKE_DIR / "runs"),
        name=run_name,
        exist_ok=True,
        plots=False,
        distill_model=str(teacher) if teacher else None,
        dis=DISTILL_WEIGHT,
        verbose=False,
    )
    return SMOKE_DIR / "runs" / run_name / "weights" / "last.pt"


def count_parameters(checkpoint: Path) -> int:
    """Return the parameter count of the model saved in one checkpoint."""
    return sum(parameter.numel() for parameter in YOLO(str(checkpoint)).model.parameters())


def run_distill_smoke(weights_dir: Path) -> dict:
    """Run teacher training, distillation, and student export. Return and write the smoke record."""
    data_yaml = build_fixture_dataset(SMOKE_DIR / "dataset", HOST_PREDICTIONS_PATH)
    teacher = train_one_epoch(weights_dir / "yolo26s.pt", data_yaml, "teacher_yolo26s", teacher=None)
    student = train_one_epoch(weights_dir / "yolo26n.pt", data_yaml, "student_yolo26n_distilled", teacher=teacher)

    saved = torch.load(student, map_location="cpu", weights_only=False)
    student_model = saved["model"] if saved.get("model") is not None else saved["ema"]
    teacher_attached = getattr(student_model, "teacher_model", None) is not None
    export_dir = Path(YOLO(str(student)).export(format="ncnn", imgsz=IMAGE_SIZE))
    ncnn_record = describe_ncnn_model(export_dir)

    student_parameters = count_parameters(student)
    teacher_parameters = count_parameters(teacher)
    # A float32 ncnn bin stores about four bytes per student parameter; a teacher would add ~4x more.
    bytes_per_parameter = ncnn_record["bin_bytes"] / student_parameters
    record = {
        "ultralytics": ultralytics.__version__,
        "distill_weight_dis": DISTILL_WEIGHT,
        "saved_train_args": {key: saved["train_args"].get(key) for key in ("distill_model", "dis", "epochs", "imgsz")},
        "student_checkpoint": str(student.relative_to(OUTPUTS_DIR.parent)),
        "student_class": type(student_model).__name__,
        "teacher_attached_to_saved_student": teacher_attached,
        "student_parameters": student_parameters,
        "teacher_parameters": teacher_parameters,
        "ncnn_export": ncnn_record,
        "ncnn_bytes_per_student_parameter": round(bytes_per_parameter, 2),
        "export_contains_only_student": (not teacher_attached)
        and ncnn_record["output_shape_hw"][0] == 5
        and bytes_per_parameter < 4.5,
    }
    (SMOKE_DIR / "distill_smoke.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR)
    args = parser.parse_args()
    record = run_distill_smoke(args.weights_dir)
    print(json.dumps(record, indent=2))
    if not record["export_contains_only_student"]:
        raise SystemExit("Student export check failed")


if __name__ == "__main__":
    main()
