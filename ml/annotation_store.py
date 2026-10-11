"""Versioned review drafts, explicit approval, and reviewed YOLO exports."""

import io
import json
import math
import re
import threading
import uuid
import zipfile
from datetime import datetime, timezone

from ml.annotation_sources import file_digest, yolo_to_rectangle
from ml.label_files import YoloBox, format_label_line, read_label_file

IMAGE_FLAGS = {"blur", "tiny", "noise", "backlit", "flock", "uncertain"}
BOX_FLAGS = {"occluded", "truncated", "groupof", "group", "inside", "depiction", "uncertain"}


class ReviewConflict(ValueError):
    pass


def atomic_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def validate_review(payload):
    boxes = payload.get("boxes")
    if not isinstance(boxes, list) or len(boxes) > 1000:
        raise ValueError("Supply a list of at most 1,000 boxes.")
    identifiers = set()
    cleaned = []
    for box in boxes:
        if not isinstance(box, dict) or box.get("label") != "bird":
            raise ValueError("This dataset uses the bird label only.")
        identifier = box.get("id")
        if not isinstance(identifier, str) or not 1 <= len(identifier) <= 80 or identifier in identifiers:
            raise ValueError("Every box must have a unique ID.")
        identifiers.add(identifier)
        values = [box.get(key) for key in ("x", "y", "width", "height")]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            raise ValueError("Box coordinates must be finite numbers.")
        x, y, width, height = values
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > 1.000001 or y + height > 1.000001:
            raise ValueError("Boxes must have a positive size and stay inside the image.")
        flags = box.get("flags", [])
        if not isinstance(flags, list) or any(flag not in BOX_FLAGS for flag in flags):
            raise ValueError("Unknown box flag.")
        cleaned.append({"id": identifier, "label": "bird", "x": x, "y": y, "width": min(width, 1 - x),
                        "height": min(height, 1 - y), "flags": sorted(set(flags))})
    notes = payload.get("notes", "")
    flags = payload.get("flags", [])
    if not isinstance(notes, str) or len(notes) > 4000:
        raise ValueError("Notes must be text of at most 4,000 characters.")
    if not isinstance(flags, list) or any(flag not in IMAGE_FLAGS for flag in flags):
        raise ValueError("Unknown image flag.")
    negative = payload.get("negative", False)
    if not isinstance(negative, bool) or (negative and cleaned):
        raise ValueError("A confirmed no-bird image cannot also contain bird boxes.")
    model_run_id = payload.get("model_run_id")
    if model_run_id is not None and (not isinstance(model_run_id, str) or not re.fullmatch(r"[a-f0-9]{32}", model_run_id)):
        raise ValueError("Invalid model run reference.")
    return {"boxes": cleaned, "notes": notes, "flags": sorted(set(flags)), "negative": negative, "model_run_id": model_run_id}


class AnnotationStore:
    def __init__(self, sources):
        self.sources = sources
        self.root = sources.data_dir / "annotation_review"
        self.lock = threading.RLock()

    def state_path(self, identifier):
        return self.root / "items" / identifier / "current.json"

    def current(self, identifier):
        path = self.state_path(identifier)
        return json.loads(path.read_text()) if path.exists() else None

    def queue(self):
        records = []
        for candidate in self.sources.scan_candidates().values():
            current = self.current(candidate["id"])
            status = "approved" if candidate["has_reviewed"] else current["status"] if current else "candidate"
            records.append({key: candidate[key] for key in ("id", "path", "name", "source", "available", "has_proposal")}
                           | {"status": status, "box_count": len(current["boxes"]) if current else None})
        return records

    def load(self, identifier):
        with self.lock:
            candidate = self.sources.load_candidate(identifier)
            current = self.current(identifier)
            if current:
                review = {key: current[key] for key in ("boxes", "notes", "flags", "negative", "revision", "status")}
            elif candidate["has_reviewed"]:
                boxes, errors = read_label_file(self.sources.reviewed_path(candidate))
                review = {"boxes": [yolo_to_rectangle(box, index) for index, box in enumerate(boxes)],
                          "notes": "", "flags": [], "negative": not boxes and not errors,
                          "revision": "initial", "status": "approved"}
                candidate["proposal_errors"] += errors
            else:
                review = {"boxes": candidate["boxes"], "notes": "", "flags": [], "negative": False,
                          "revision": "initial", "status": "candidate"}
            review["status"] = "approved" if candidate["has_reviewed"] else review["status"]
            review["model_run_id"] = current.get("model_run_id") if current else None
            review["source_changed"] = bool(current and current["source_version"] != candidate["source_version"])
            return {**candidate, "review": review}

    def save_review(self, identifier, payload):
        action = payload.get("action", "draft")
        if action not in {"draft", "approve", "needs_work", "reopen"}:
            raise ValueError("Unknown review action.")
        cleaned = validate_review(payload)
        with self.lock:
            loaded = self.load(identifier)
            current = loaded["review"]
            if payload.get("revision") != current["revision"] or payload.get("source_version") != loaded["source_version"] or payload.get("reviewed_digest") != loaded["reviewed_digest"]:
                raise ReviewConflict("This image or review changed elsewhere. Reload it before saving; your edits are preserved on this device.")
            if loaded["has_reviewed"] and action != "reopen":
                raise ReviewConflict("Reopen this approved image before editing its labels.")
            if cleaned["model_run_id"] and not (self.state_path(identifier).parent / "model_runs" / f'{cleaned["model_run_id"]}.json').is_file():
                raise ValueError("This model run does not belong to the image.")
            if action == "approve" and not cleaned["boxes"] and not cleaned["negative"]:
                raise ValueError("Confirm No birds before approving an image with no boxes.")
            if action == "approve" and loaded["proposal_errors"]:
                # A valid edited set can replace an invalid proposal; original errors remain recorded.
                cleaned["source_errors"] = loaded["proposal_errors"]
            status = {"draft": "draft", "approve": "approved", "needs_work": "needs_work", "reopen": "draft"}[action]
            label_path = self.sources.reviewed_path(loaded)
            previous_label = label_path.read_text() if label_path.exists() else None
            revision = uuid.uuid4().hex
            record = {**cleaned, "id": identifier, "path": loaded["path"], "revision": revision,
                      "parent_revision": current["revision"], "source_version": loaded["source_version"],
                      "status": status, "action": action, "reviewer": "human",
                      "updated_at": datetime.now(timezone.utc).isoformat(), "previous_reviewed_yolo": previous_label}
            versions = self.state_path(identifier).parent / "versions"
            atomic_json(versions / f"{revision}.json", record)
            if action == "approve":
                lines = [format_label_line(YoloBox(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2,
                                                  box["width"], box["height"])) for box in cleaned["boxes"]]
                label_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = label_path.with_suffix(".txt.tmp")
                temporary.write_text("".join(line + "\n" for line in lines))
                temporary.replace(label_path)
            elif action == "reopen":
                label_path.unlink(missing_ok=True)
            atomic_json(self.state_path(identifier), record)
            return {"review": record, "reviewed_digest": file_digest(label_path)}

    def history(self, identifier):
        self.sources.get_record(identifier)
        directory = self.state_path(identifier).parent / "versions"
        records = [json.loads(path.read_text()) for path in directory.glob("*.json")]
        return sorted(records, key=lambda record: record["updated_at"], reverse=True)

    def export_reviewed(self):
        stream = io.BytesIO()
        with self.lock, zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            manifest = []
            for candidate in self.sources.scan_candidates().values():
                path = self.sources.reviewed_path(candidate)
                if not path.exists():
                    continue
                boxes, errors = read_label_file(path)
                if errors:
                    raise ValueError("An approved label file is invalid. Reopen and correct it before export.")
                label_name = path.relative_to(self.sources.data_dir / "labels_reviewed").as_posix()
                archive.write(path, "labels/" + label_name)
                manifest.append({"image": candidate["path"], "label": "labels/" + label_name,
                                 "box_count": len(boxes), "review": self.current(candidate["id"])})
            archive.writestr("classes.txt", "bird\n")
            archive.writestr("manifest.json", json.dumps({"items": manifest, "format": "YOLO normalized xywh, class 0 = bird"}, indent=2))
        return stream.getvalue()
