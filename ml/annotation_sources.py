"""Discover local proposals and selected Open Images candidates for annotation."""

import hashlib
import json
import re
import threading
from pathlib import Path
from urllib.request import urlopen

import yaml
from PIL import Image, ImageOps

from ml.label_files import PROPOSED_LABELS, REVIEWED_LABELS, YoloBox, format_label_line, label_path_for, list_review_images, read_label_file
from ml.open_images_picker import build_catalog
from ml.open_images_catalog import load_catalog_metadata
from ml.record_capture import record_new_captures

BIRD_MID = "/m/015p6"


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def image_identifier(relative):
    return hashlib.sha256(relative.encode()).hexdigest()[:24]


def source_version(path, proposal):
    stat = path.stat()
    signature = f"{stat.st_size}:{stat.st_mtime_ns}:{file_digest(proposal)}"
    return hashlib.sha256(signature.encode()).hexdigest()


def yolo_to_rectangle(box, index):
    left = max(0, box.center_x - box.width / 2)
    top = max(0, box.center_y - box.height / 2)
    right = min(1, box.center_x + box.width / 2)
    bottom = min(1, box.center_y + box.height / 2)
    return {"id": f"box-{index + 1}", "label": "bird", "x": left, "y": top,
            "width": right - left, "height": bottom - top, "flags": []}


def source_rectangle(obj, index):
    box = obj["bounding_box"]
    return {"id": f"box-{index + 1}", "label": "bird", "x": float(box["xmin"]),
            "y": float(box["ymin"]), "width": float(box["xmax"]) - float(box["xmin"]),
            "height": float(box["ymax"]) - float(box["ymin"]),
            "flags": [str(flag).lower() for flag in obj.get("attributes", [])]}


class CandidateSources:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir).resolve()
        self.import_lock = threading.Lock()

    def selected_images(self):
        root = self.data_dir / "open_images_picker"
        selections_path = root / "selections.json"
        if not selections_path.exists():
            return {}
        selections = json.loads(selections_path.read_text())
        metadata = load_catalog_metadata(root)
        # Only build selected records: the full catalog need not be sorted for each queue scan.
        catalog = build_catalog({identifier: record for identifier, record in metadata.items()
                                 if selections.get(identifier, {}).get("decision") == "keep"})
        # Imported proposals retain their original source order and attributes after catalog updates.
        for index, photo in enumerate(catalog):
            relative = f'incoming/stills/open_images/train/{photo["id"]}.jpg'
            source_path = self.data_dir / "annotation_review" / "sources" / f"{image_identifier(relative)}.json"
            if source_path.exists():
                catalog[index] = json.loads(source_path.read_text())
        return {photo["id"]: photo for photo in catalog
                if selections.get(photo["id"], {}).get("decision") == "keep"}

    def scan_candidates(self):
        selected = self.selected_images()
        records = {}
        for photo in selected.values():
            relative = f'incoming/stills/open_images/train/{photo["id"]}.jpg'
            identifier = image_identifier(relative)
            records[identifier] = {"id": identifier, "path": relative, "name": photo["title"] or photo["id"],
                                   "source": "Open Images", "remote": photo}
        for path in list_review_images(self.data_dir):
            if not path.is_file() or not path.resolve().is_relative_to(self.data_dir):
                continue
            relative = path.relative_to(self.data_dir).as_posix()
            identifier = image_identifier(relative)
            record = records.get(identifier, {"id": identifier, "path": relative, "name": path.name, "source": "Local"})
            record["available"] = True
            records[identifier] = record
        for record in records.values():
            path = self.data_dir / record["path"]
            record["available"] = path.exists()
            record["has_proposal"] = label_path_for(path, self.data_dir, PROPOSED_LABELS).exists() or bool(record.get("remote", {}).get("boxes"))
            record["has_reviewed"] = label_path_for(path, self.data_dir, REVIEWED_LABELS).exists()
        return records

    def get_record(self, identifier):
        record = self.scan_candidates().get(identifier)
        if record is None:
            raise KeyError("This candidate is no longer in the queue. Refresh to check new arrivals.")
        return record

    def materialize_image(self, record):
        path = self.data_dir / record["path"]
        if not path.resolve().is_relative_to(self.data_dir):
            raise ValueError("Images must stay inside the project's data folder.")
        if path.exists():
            return path
        photo = record.get("remote")
        if photo is None or not re.fullmatch(r"[a-f0-9]{16}", photo["id"]):
            raise ValueError("Image unavailable.")
        with self.import_lock:
            if path.exists():
                return path
            url = f'https://open-images-dataset.s3.amazonaws.com/train/{photo["id"]}.jpg'
            with urlopen(url, timeout=25) as response:
                content = response.read(30 * 1024 * 1024 + 1)
            if len(content) > 30 * 1024 * 1024:
                raise ValueError("Image exceeds the 30 MB import limit.")
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".download")
            temporary.write_bytes(content)
            try:
                with Image.open(temporary) as image:
                    image.verify()
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            proposal = label_path_for(path, self.data_dir, PROPOSED_LABELS)
            if not proposal.exists():
                boxes = [source_rectangle(obj, index) for index, obj in enumerate(photo["boxes"])
                         if "Depiction" not in obj.get("attributes", [])]
                lines = [format_label_line(YoloBox(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2,
                                                  box["width"], box["height"])) for box in boxes]
                proposal.parent.mkdir(parents=True, exist_ok=True)
                proposal.write_text("".join(line + "\n" for line in lines))
            self.record_import(record)
        return path

    def import_upload(self, filename, content):
        directory = self.data_dir / "incoming" / "stills" / "uploads"
        if not directory.resolve().is_relative_to(self.data_dir):
            raise ValueError("Uploads must stay inside the project's data folder.")
        safe_name = "".join(character if character.isalnum() or character in "._-" else "_" for character in filename)
        with self.import_lock:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{hashlib.sha256(content).hexdigest()[:12]}-{safe_name}"
            if not path.exists():
                temporary = path.with_suffix(".upload")
                temporary.write_bytes(content)
                try:
                    with Image.open(temporary) as image:
                        image.verify()
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            record_new_captures(self.data_dir)
        return image_identifier(path.relative_to(self.data_dir).as_posix())

    def record_import(self, record):
        photo = record["remote"]
        directory = self.data_dir / "annotation_review" / "sources"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f'{record["id"]}.json').write_text(json.dumps(photo, indent=2) + "\n")
        provenance = self.data_dir / "provenance" / "sources.yaml"
        contents = yaml.safe_load(provenance.read_text()) if provenance.exists() else {"items": []}
        items = contents.setdefault("items", [])
        if any(item["path"] == record["path"] for item in items):
            return
        links = re.findall(r"href=['\"]([^'\"]+)", photo["source_metadata"].get("footnote_bottom_right", ""))
        license_url = next((url for url in links if "creativecommons.org/" in url), None)
        items.append({"path": record["path"], "session_id": f'open-images-train-{photo["id"]}',
                      "media_type": "still", "camera_make": None, "camera_model": None,
                      "lens": None, "focal_length_mm": None, "capture_time": None,
                      "author": photo["attribution"], "license": license_url,
                      "source_url": photo["source_url"], "source_split": "train",
                      "notes": "Selected Open Images candidate. Camera and location unverified. Source boxes require review."})
        provenance.parent.mkdir(parents=True, exist_ok=True)
        temporary = provenance.with_suffix(".yaml.tmp")
        temporary.write_text(yaml.safe_dump(contents, sort_keys=False, allow_unicode=True))
        temporary.replace(provenance)

    def load_candidate(self, identifier):
        record = self.get_record(identifier)
        path = self.materialize_image(record)
        proposal = label_path_for(path, self.data_dir, PROPOSED_LABELS)
        reviewed = label_path_for(path, self.data_dir, REVIEWED_LABELS)
        boxes, errors = read_label_file(proposal) if proposal.exists() else ([], [])
        with Image.open(path) as raw:
            orientation = raw.getexif().get(274, 1)
            image = ImageOps.exif_transpose(raw)
            width, height = image.size
        # Proposals are normalized in the decoded, upright image coordinate system.
        rectangles = [yolo_to_rectangle(box, index) for index, box in enumerate(boxes)]
        if record.get("remote"):
            source_objects = [obj for obj in record["remote"]["boxes"] if "Depiction" not in obj.get("attributes", [])]
            for rectangle, obj in zip(rectangles, source_objects):
                rectangle["flags"] = [str(flag).lower() for flag in obj.get("attributes", [])]
        return {**record, "width": width, "height": height, "orientation": orientation,
                "boxes": rectangles, "proposal_errors": errors,
                "source_version": source_version(path, proposal), "reviewed_digest": file_digest(reviewed),
                "attribution": record.get("remote", {}).get("attribution", "Your local image"),
                "source_url": record.get("remote", {}).get("source_url"),
                "image_url": f"/api/image/{identifier}", "has_reviewed": reviewed.exists()}

    def reviewed_path(self, candidate):
        return label_path_for(self.data_dir / candidate["path"], self.data_dir, REVIEWED_LABELS)
