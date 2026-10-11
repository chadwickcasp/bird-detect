"""Add new stills and clips to data/provenance/sources.yaml.

Run from the repository root after dropping files into data/incoming/:

    .venv/bin/python -m ml.record_capture

Stills get camera fields from EXIF. Clips get blank camera fields for you to fill in. Fields
you already set are never overwritten.
"""

import argparse
import re
from collections.abc import Mapping
from pathlib import Path

import yaml
from PIL import ExifTags, Image

from ml.label_files import IMAGE_SUFFIXES
from ml.project_paths import DATA_DIR

CLIP_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
PROVENANCE_FIELDS = (
    "session_id",
    "media_type",
    "camera_make",
    "camera_model",
    "lens",
    "focal_length_mm",
    "capture_time",
    "author",
    "license",
    "source_url",
    "notes",
)
EXIF_TAG_IDS = {name: tag for tag, name in ExifTags.TAGS.items()}


def provenance_path(data_dir: Path) -> Path:
    """Return the local provenance file path."""
    return data_dir / "provenance" / "sources.yaml"


def load_provenance(data_dir: Path) -> list[dict]:
    """Return the provenance entries, or an empty list when the file does not exist yet."""
    path = provenance_path(data_dir)
    if not path.exists():
        return []
    return (yaml.safe_load(path.read_text()) or {}).get("items") or []


def read_exif_fields(image_path: Path) -> dict:
    """Return camera fields from a still's EXIF block. Missing tags are None."""
    with Image.open(image_path) as image:
        exif = image.getexif()
        details = exif.get_ifd(ExifTags.IFD.Exif)
    capture_time = details.get(EXIF_TAG_IDS["DateTimeOriginal"]) or exif.get(EXIF_TAG_IDS["DateTime"])
    if capture_time:
        date, _, clock = str(capture_time).partition(" ")
        capture_time = f"{date.replace(':', '-')}T{clock}" if clock else date.replace(":", "-")
    focal_length = details.get(EXIF_TAG_IDS["FocalLength"])
    return {
        "camera_make": exif.get(EXIF_TAG_IDS["Make"]),
        "camera_model": exif.get(EXIF_TAG_IDS["Model"]),
        "lens": details.get(EXIF_TAG_IDS["LensModel"]),
        "focal_length_mm": round(float(focal_length), 2) if focal_length else None,
        "capture_time": capture_time,
    }


def default_session_id(fields: Mapping) -> str | None:
    """Return a date-and-camera session id, or None without a capture date.

    Two outings on one day with one camera share this id. Edit it when they should split.
    """
    if not fields.get("capture_time"):
        return None
    date = fields["capture_time"][:10].replace("-", "")
    camera = re.sub(r"[^a-z0-9]+", "-", str(fields.get("camera_model") or "unknown").lower()).strip("-")
    return f"{date}-{camera}"


def record_still_exif(image_path: Path) -> dict:
    """Return a new provenance entry for one still, filled from EXIF where possible."""
    fields = read_exif_fields(image_path)
    entry = dict.fromkeys(PROVENANCE_FIELDS)
    entry.update(fields)
    entry["media_type"] = "still"
    entry["session_id"] = default_session_id(fields)
    entry["notes"] = ""
    return entry


def record_new_captures(data_dir: Path) -> list[str]:
    """Append entries for unrecorded files in data/incoming/ and return their relative paths."""
    entries = load_provenance(data_dir)
    recorded = {entry["path"] for entry in entries}
    added = []
    incoming = data_dir / "incoming"
    for path in sorted(incoming.rglob("*")) if incoming.is_dir() else []:
        relative = path.relative_to(data_dir).as_posix()
        suffix = path.suffix.lower()
        if relative in recorded or suffix not in IMAGE_SUFFIXES | CLIP_SUFFIXES:
            continue
        if suffix in IMAGE_SUFFIXES:
            entry = record_still_exif(path)
        else:
            entry = dict.fromkeys(PROVENANCE_FIELDS)
            entry.update(media_type="clip", notes="")
        entries.append({"path": relative, **entry})
        added.append(relative)

    path = provenance_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"items": entries}, sort_keys=False, allow_unicode=True))
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    added = record_new_captures(args.data_dir)
    print(f"Added {len(added)} entries to {provenance_path(args.data_dir)}")
    missing_session = [entry["path"] for entry in load_provenance(args.data_dir) if not entry.get("session_id")]
    if missing_session:
        print(f"{len(missing_session)} entries still need a session_id, for example {missing_session[0]}")


if __name__ == "__main__":
    main()
