"""Import the complete training Bird boxes without downloading the photographs."""

import argparse
import csv
import hashlib
import html
import json
import re
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.request import urlopen

from ml.project_paths import DATA_DIR

BIRD_MID = "/m/015p6"
BOXES_URL = "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
IMAGES_URL = "https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv"
FULL_METADATA = "bird_metadata.json"
ATTRIBUTES = {"IsOccluded": "Occluded", "IsTruncated": "Truncated", "IsGroupOf": "GroupOf",
              "IsDepiction": "Depiction", "IsInside": "Inside"}


class DownloadLines:
    """Hash the entire source stream and report progress without storing a multi-GB CSV."""

    def __init__(self, response, label):
        self.response = response
        self.label = label
        self.digest = hashlib.sha256()
        self.size = 0
        self.reported_at = time.monotonic()

    def __iter__(self):
        for line in self.response:
            self.digest.update(line)
            self.size += len(line)
            if time.monotonic() - self.reported_at > 20:
                print(f"{self.label}: scanned {self.size / 1_000_000:.0f} MB", flush=True)
                self.reported_at = time.monotonic()
            yield line.decode("utf-8-sig" if self.size == len(line) else "utf-8")


def read_bird_boxes(lines):
    reader = csv.reader(lines)
    columns = next(reader)
    indices = {name: columns.index(name) for name in
               ["ImageID", "LabelName", "XMin", "XMax", "YMin", "YMax", *ATTRIBUTES]}
    metadata = {}
    for row in reader:
        if row[indices["LabelName"]] != BIRD_MID:
            continue
        identifier = row[indices["ImageID"]]
        if not re.fullmatch(r"[a-f0-9]{16}", identifier):
            raise ValueError("Invalid Open Images image ID.")
        xmin, xmax, ymin, ymax = [float(row[indices[key]]) for key in ("XMin", "XMax", "YMin", "YMax")]
        if not (0 <= xmin < xmax <= 1 and 0 <= ymin < ymax <= 1):
            raise ValueError(f"Invalid bird box in {identifier}.")
        record = metadata.setdefault(identifier, {"image": {}, "objects": []})
        record["objects"].append({"category": BIRD_MID, "text": "Bird",
                                  "bounding_box": {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax},
                                  "attributes": [attribute for key, attribute in ATTRIBUTES.items()
                                                 if row[indices[key]] == "1"],
                                  "unknown_attributes": [attribute for key, attribute in ATTRIBUTES.items()
                                                         if row[indices[key]] == "-1"]})
    return metadata


def attach_image_metadata(metadata, lines):
    for row in csv.DictReader(lines):
        identifier = row["ImageID"]
        if identifier not in metadata:
            continue
        if row["Subset"] != "train":
            raise ValueError("The training catalog contains a different official split.")
        author, license_url = html.escape(row["Author"]), html.escape(row["License"], quote=True)
        metadata[identifier]["image"] = {
            "url": row["Thumbnail300KURL"] or f"https://open-images-dataset.s3.amazonaws.com/train/{identifier}.jpg",
            "footnote_top_right": "Title: " + html.escape(row["Title"]),
            "footnote_bottom_right": f'{author} · <a href="{license_url}">Image license</a>',
            "original_url": row["OriginalURL"], "original_landing_url": row["OriginalLandingURL"],
            "license_url": row["License"], "author_profile_url": row["AuthorProfileURL"],
            "rotation": row["Rotation"],
        }
    missing = [identifier for identifier, record in metadata.items() if not record["image"]]
    if missing:
        raise ValueError(f"Attribution metadata missing for {len(missing)} bird images.")


def load_catalog_metadata(directory):
    """Retain gallery titles and attribution alongside full-precision official boxes."""
    directory = Path(directory)
    files = []
    for name in (FULL_METADATA, "gallery_metadata.json"):
        path = directory / name
        if path.exists():
            stat = path.stat()
            files.append((str(path.resolve()), stat.st_mtime_ns, stat.st_size))
    return read_cached_metadata(tuple(files))


@lru_cache(maxsize=4)
def read_cached_metadata(files):
    # Queue polling reuses the parsed catalog; replacing a source file invalidates its cache key.
    metadata = {}
    for filename, modified_at, size in files:
        additions = json.loads(Path(filename).read_text())
        for identifier, record in additions.items():
            if identifier in metadata:
                metadata[identifier] = {**metadata[identifier], "image": record["image"]}
            else:
                metadata[identifier] = record
    return metadata


def completed_image_ids(data_dir):
    """Read current decisions on every reload; a reopened draft becomes visible again."""
    data_dir = Path(data_dir)
    completed = set()
    for path in (data_dir / "annotation_review" / "items").glob("*/current.json"):
        record = json.loads(path.read_text())
        relative = record.get("path", "")
        match = re.fullmatch(r"incoming/stills/open_images/train/([a-f0-9]{16})\.jpg", relative)
        if match and record.get("status") in {"approved", "needs_work"}:
            completed.add(match[1])
    # Also respect approved label files that predate the versioned review UI.
    directory = data_dir / "labels_reviewed" / "incoming" / "stills" / "open_images" / "train"
    completed.update(path.stem for path in directory.glob("*.txt")
                     if re.fullmatch(r"[a-f0-9]{16}", path.stem))
    return completed


def import_training_catalog(directory):
    sources = []
    with urlopen(BOXES_URL, timeout=90) as response:
        lines = DownloadLines(response, "Bounding boxes")
        metadata = read_bird_boxes(lines)
        sources.append({"url": BOXES_URL, "bytes": lines.size, "sha256": lines.digest.hexdigest()})
    print(f"Found {len(metadata):,} Bird images; reading attribution metadata…", flush=True)
    with urlopen(IMAGES_URL, timeout=90) as response:
        lines = DownloadLines(response, "Image metadata")
        attach_image_metadata(metadata, lines)
        sources.append({"url": IMAGES_URL, "bytes": lines.size, "sha256": lines.digest.hexdigest()})
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"split": "train", "class_id": BIRD_MID, "images": len(metadata),
                "boxes": sum(len(record["objects"]) for record in metadata.values()), "sources": sources,
                "imported_at": datetime.now(timezone.utc).isoformat()}
    temporary = directory / (FULL_METADATA + ".tmp")
    temporary.write_text(json.dumps(metadata) + "\n")
    temporary.replace(directory / FULL_METADATA)
    (directory / "bird_catalog_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR / "open_images_picker")
    args = parser.parse_args()
    import_training_catalog(args.data_dir)


if __name__ == "__main__":
    main()
