"""Compile LiteRT models for the Google Tensor G4 TPU and save each compilation report.

Runs inside the g4/Dockerfile image, or any Linux x86_64 Python 3.11 environment with
ai-edge-litert==2.1.5 and ai-edge-litert-sdk-google-tensor==2.1.5 installed:

    GOOGLE_TENSOR_BACKEND_ENABLED=1 python g4/compile_tensor_g4.py

The default order compiles the MediaPipe selfie model first. It compiled fully for G4 in
LiteRT issue 7598, so a failure there points at the environment rather than the detector graph.
The pretrained YOLO26n export follows with the G4 truncation and sharding flags.
"""

import argparse
import importlib.metadata
import json
import platform
import sys
import traceback
import urllib.request
from pathlib import Path

from ai_edge_litert.aot import aot_compile
from ai_edge_litert.aot.vendors.google_tensor import google_tensor_backend  # noqa: F401  registers GOOGLE
from ai_edge_litert.aot.vendors.google_tensor import target as google_tensor_target

SELFIE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite"
)
G4_DETECTOR_FLAGS = {
    "google_tensor_truncation_type": "half",
    "google_tensor_sharding_intensity": "maximum",
}


def fetch_selfie_model(models_dir: Path) -> Path:
    """Download the MediaPipe selfie probe model once and return its path."""
    path = models_dir / "selfie_multiclass_256x256.tflite"
    if not path.exists():
        models_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SELFIE_MODEL_URL, path)
    return path


def remove_previous_artifacts(model_path: Path, output_dir: Path) -> None:
    """Delete compiled files an earlier run exported for this model, so the summary lists only this run's."""
    for path in output_dir.glob(f"{model_path.stem}_*.tflite"):
        path.unlink()


def compile_for_tensor_g4(model_path: Path, output_dir: Path, flags: dict[str, str]) -> dict:
    """Compile one .tflite for Tensor G4, save the report and artifacts, and return a summary.

    A failed compile still writes a report with the exception text.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    remove_previous_artifacts(model_path, output_dir)
    report_path = output_dir / f"{model_path.stem}_Tensor_G4_report.txt"
    summary = {"model": model_path.name, "flags": flags, "report": report_path.name, "artifacts": []}
    try:
        result = aot_compile.aot_compile(
            str(model_path),
            output_dir=str(output_dir / f"{model_path.stem}_aot"),
            target=[google_tensor_target.Target(google_tensor_target.SocModel.TENSOR_G4)],
            keep_going=False,
            **flags,
        )
        report = result.compilation_report()
        result.export(output_dir, model_name=model_path.stem)
        summary["artifacts"] = sorted(path.name for path in output_dir.glob(f"{model_path.stem}_*.tflite"))
        summary["failed_backends"] = [f"{backend.target_id}: {error}" for backend, error in result.failed_backends]
        summary["status"] = "compiled" if summary["artifacts"] else "no artifact"
    except Exception as error:  # boundary: record any compiler failure and keep going
        report = f"{type(error).__name__}: {error}\n\n{traceback.format_exc()}"
        summary["status"] = "failed"
    report_path.write_text(report)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/g4"))
    parser.add_argument("--detector", type=Path, default=Path("weights/yolo26n.tflite"))
    parser.add_argument("--skip-probe", action="store_true", help="skip the selfie probe model")
    args = parser.parse_args()

    summaries = []
    if not args.skip_probe:
        selfie = fetch_selfie_model(args.output_dir / "models")
        summaries.append(compile_for_tensor_g4(selfie, args.output_dir, flags={}))
    summaries.append(compile_for_tensor_g4(args.detector, args.output_dir, flags=G4_DETECTOR_FLAGS))

    record = {
        "host": {"machine": platform.machine(), "python": sys.version.split()[0]},
        "litert": {
            "ai_edge_litert": importlib.metadata.version("ai-edge-litert"),
            "sdk_google_tensor": importlib.metadata.version("ai-edge-litert-sdk-google-tensor"),
        },
        "results": summaries,
    }
    (args.output_dir / "g4_compile.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
