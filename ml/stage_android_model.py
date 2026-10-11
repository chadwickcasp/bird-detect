"""Copy the verified exports and the parity fixtures into the Android apps' assets.

Run from the repository root after ml/export_pretrained.py:

    .venv/bin/python -m ml.stage_android_model

The camera app reads assets/models/yolo26n/model.json for tensor names and decode settings.
That file is trimmed from outputs/exports/pretrained_yolo26n.json, so the app and the host
parity check use the same contract. The G4 smoke app gets the float LiteRT export and, when
g4/compile_tensor_g4.py produced one, the Tensor G4 compiled model. Staged assets are gitignored.
"""

import json
import shutil
from pathlib import Path

from ml.compare_predictions import list_fixture_images
from ml.project_paths import EXPORT_RECORD_PATH, FIXTURES_DIR, OUTPUTS_DIR, REPO_ROOT, WEIGHTS_DIR

APP_ASSETS_DIR = REPO_ROOT / "android" / "app" / "src" / "main" / "assets"
G4SMOKE_ASSETS_DIR = REPO_ROOT / "android" / "g4smoke" / "src" / "main" / "assets"
G4_COMPILE_DIR = OUTPUTS_DIR / "g4"
G4SMOKE_COMPILED_NAME = "yolo26n_Tensor_G4.tflite"


def build_app_contract(record: dict) -> dict:
    """Return the fields the camera app needs from a verified export record."""
    ncnn = record["ncnn"]
    return {
        "name": f"{record['model']} pretrained COCO",
        "param": ncnn["param"],
        "bin": ncnn["bin"],
        "input_blob": ncnn["input_blob"],
        "output_blob": ncnn["output_blob"],
        "input_size": record["preprocess"]["image_size"],
        "bird_class_index": record["output"]["bird_class_index"],
        "confidence_threshold": record["postprocess"]["confidence_threshold"],
        "nms_iou_threshold": record["postprocess"]["nms_iou_threshold"],
        "bin_sha256": ncnn["bin_sha256"],
        "ultralytics": record["versions"]["ultralytics"],
    }


def copy_fixtures(fixture_dir: Path) -> None:
    """Replace fixture_dir with a fresh copy of the parity fixtures."""
    shutil.rmtree(fixture_dir, ignore_errors=True)
    fixture_dir.mkdir(parents=True)
    for image in list_fixture_images(FIXTURES_DIR):
        shutil.copy2(image, fixture_dir / image.name)


def stage_camera_app_assets(record: dict, assets_dir: Path) -> None:
    """Copy the ncnn files, write model.json, and copy the fixtures for the camera app."""
    ncnn_dir = WEIGHTS_DIR / record["ncnn"]["dir"]
    model_dir = assets_dir / "models" / "yolo26n"
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in (record["ncnn"]["param"], record["ncnn"]["bin"]):
        shutil.copy2(ncnn_dir / name, model_dir / name)
    (model_dir / "model.json").write_text(json.dumps(build_app_contract(record), indent=2) + "\n")
    copy_fixtures(assets_dir / "fixtures")


def find_compiled_g4_model(record: dict, compile_dir: Path) -> Path | None:
    """Return the YOLO26n artifact that the latest g4/compile_tensor_g4.py run listed, or None."""
    compile_record_path = compile_dir / "g4_compile.json"
    if not compile_record_path.exists():
        return None
    results = json.loads(compile_record_path.read_text())["results"]
    detector = next((result for result in results if result["model"] == record["litert"]["file"]), None)
    if detector is None or detector["status"] != "compiled":
        return None
    return compile_dir / detector["artifacts"][0]


def stage_g4smoke_assets(record: dict, assets_dir: Path, compile_dir: Path) -> Path | None:
    """Copy the float LiteRT export, any compiled G4 model, and the fixtures; return the compiled source."""
    model_dir = assets_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(WEIGHTS_DIR / record["litert"]["file"], model_dir / "yolo26n.tflite")
    compiled = find_compiled_g4_model(record, compile_dir)
    staged_compiled = model_dir / G4SMOKE_COMPILED_NAME
    staged_compiled.unlink(missing_ok=True)
    if compiled:
        shutil.copy2(compiled, staged_compiled)
    copy_fixtures(assets_dir / "fixtures")
    return compiled


if __name__ == "__main__":
    export_record = json.loads(EXPORT_RECORD_PATH.read_text())
    stage_camera_app_assets(export_record, APP_ASSETS_DIR)
    compiled_source = stage_g4smoke_assets(export_record, G4SMOKE_ASSETS_DIR, G4_COMPILE_DIR)
    print(f"Staged {APP_ASSETS_DIR} and {G4SMOKE_ASSETS_DIR}")
    print(f"Compiled G4 model: {compiled_source or 'none; the smoke app requests NPU for the float model'}")
