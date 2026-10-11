"""Run local YOLO checkpoints as immutable suggestions, never approved labels."""

import math
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

from ml.annotation_sources import file_digest, yolo_to_rectangle
from ml.annotation_store import ReviewConflict, atomic_json, validate_review
from ml.label_files import YoloBox
from ml.project_paths import REPO_ROOT, WEIGHTS_DIR

DEFAULT_CONFIDENCE = 0.15
IMAGE_SIZE = 640
IOU_THRESHOLD = 0.45


def checkpoint_options(custom=None):
    options = [
        ("yolo26n", "YOLO26n · phone baseline", WEIGHTS_DIR / "yolo26n.pt"),
        ("yolo26s", "YOLO26s · larger baseline", WEIGHTS_DIR / "yolo26s.pt"),
        ("distilled-smoke", "Distilled student · smoke test", REPO_ROOT / "outputs/distill_smoke/runs/student_yolo26n_distilled/weights/last.pt"),
        ("teacher-smoke", "Teacher · smoke test", REPO_ROOT / "outputs/distill_smoke/runs/teacher_yolo26s/weights/last.pt"),
    ]
    if custom is not None:
        checkpoint = Path(custom).resolve()
        if not checkpoint.is_file() or checkpoint.suffix != ".pt":
            raise ValueError("Choose an existing local .pt checkpoint.")
        options.insert(0, ("custom", f"Custom · {checkpoint.name}", checkpoint))
    return {identifier: {"id": identifier, "name": name, "path": path.resolve()}
            for identifier, name, path in options if path.is_file()}


def load_yolo(checkpoint):
    # YOLO is an optional heavy runtime; ordinary review and unit tests need not load it.
    from ultralytics import YOLO

    model = YOLO(str(checkpoint), task="detect")
    if model.task != "detect":
        raise ValueError("Choose an object detection checkpoint.")
    return model


def bird_class_ids(names):
    entries = enumerate(names) if isinstance(names, (list, tuple)) else names.items()
    classes = [int(identifier) for identifier, name in entries if str(name).strip().lower() == "bird"]
    if not classes:
        raise ValueError("This checkpoint has no class named bird.")
    return classes


class AnnotationPredictions:
    def __init__(self, store, checkpoints=None, loader=load_yolo):
        self.store = store
        self.checkpoints = checkpoint_options() if checkpoints is None else checkpoints
        self.loader = loader
        self.lock = threading.Lock()
        self.models = {}

    def configuration(self):
        return {"models": [{"id": item["id"], "name": item["name"]} for item in self.checkpoints.values()],
                "confidence": DEFAULT_CONFIDENCE, "image_size": IMAGE_SIZE}

    def run(self, identifier, payload):
        checkpoint = self.checkpoints.get(payload.get("model"))
        if checkpoint is None:
            raise ValueError("Choose an available local model.")
        confidence = payload.get("confidence", DEFAULT_CONFIDENCE)
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not .01 <= confidence <= .99:
            raise ValueError("Confidence must be between 0.01 and 0.99.")
        if not self.lock.acquire(blocking=False):
            raise ReviewConflict("Another model run is in progress. Try again shortly.")
        try:
            return self.predict_candidate(identifier, payload, checkpoint, confidence)
        finally:
            self.lock.release()

    def predict_candidate(self, identifier, payload, checkpoint, confidence):
        item = self.store.load(identifier)
        if payload.get("source_version") != item["source_version"]:
            raise ReviewConflict("This image or proposal changed. Reload before running the model.")
        path = checkpoint["path"]
        if not path.is_file():
            raise ValueError("The checkpoint is no longer available.")
        stamp = (path.stat().st_size, path.stat().st_mtime_ns)
        cached = self.models.get(checkpoint["id"])
        if cached is None or cached["stamp"] != stamp:
            cached = {"stamp": stamp, "sha256": file_digest(path), "model": self.loader(path)}
            self.models[checkpoint["id"]] = cached
        model = cached["model"]
        classes = bird_class_ids(model.names)
        image_path = self.store.sources.data_dir / item["path"]
        with Image.open(image_path) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        started = time.monotonic()
        result = model.predict(image, imgsz=IMAGE_SIZE, rect=False, conf=confidence,
                               iou=IOU_THRESHOLD, classes=classes, device="cpu", max_det=1000,
                               save=False, verbose=False)[0]
        elapsed = time.monotonic() - started
        if tuple(result.orig_shape) != (item["height"], item["width"]):
            raise ValueError("Prediction dimensions do not match the displayed image.")
        boxes = []
        for index, (coords, score) in enumerate(zip(result.boxes.xywhn.tolist(), result.boxes.conf.tolist())):
            rectangle = yolo_to_rectangle(YoloBox(*coords), index)
            rectangle["confidence"] = float(score)
            boxes.append(rectangle)
        validate_review({"boxes": boxes})
        if any(not math.isfinite(box["confidence"]) or not 0 <= box["confidence"] <= 1 for box in boxes):
            raise ValueError("The model returned an invalid confidence score.")
        current = self.store.load(identifier)
        if current["source_version"] != item["source_version"] or (path.stat().st_size, path.stat().st_mtime_ns) != stamp:
            raise ReviewConflict("The image, proposal, or checkpoint changed during prediction. Run again after reloading.")
        record = {"id": uuid.uuid4().hex, "image_id": identifier, "path": item["path"],
                  "source_version": item["source_version"], "review_revision": item["review"]["revision"],
                  "created_at": datetime.now(timezone.utc).isoformat(), "model": checkpoint["id"],
                  "model_name": checkpoint["name"], "checkpoint": str(path), "checkpoint_sha256": cached["sha256"],
                  "confidence_threshold": confidence, "bird_class_ids": classes, "image_size": IMAGE_SIZE,
                  "iou_threshold": IOU_THRESHOLD, "device": "cpu", "rect": False, "max_det": 1000,
                  "width": item["width"], "height": item["height"], "elapsed_seconds": elapsed, "boxes": boxes}
        atomic_json(self.store.state_path(identifier).parent / "model_runs" / f'{record["id"]}.json', record)
        return record
