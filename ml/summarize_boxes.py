"""Count reviewed images, birds, negatives, sessions, and bird box sizes.

Run from the repository root:

    .venv/bin/python -m ml.summarize_boxes

Box size uses the model-space side length sqrt(width * height) after letterboxing the image
into 640x640. The bins are <16, 16-<32, 32-<64, and >=64 model pixels.
"""

import argparse
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from ml.export_pretrained import IMAGE_SIZE
from ml.label_files import REVIEWED_LABELS, image_path_for, read_label_file
from ml.project_paths import DATA_DIR
from ml.record_capture import load_provenance

SIZE_BIN_EDGES = (16, 32, 64)
SIZE_BIN_NAMES = ("<16", "16-<32", "32-<64", ">=64")


def session_for_image(image_relative: str, entries: Sequence[dict]) -> str:
    """Return the session id of a still, or of the clip an extracted frame came from."""
    for entry in entries:
        if entry["path"] == image_relative:
            return entry.get("session_id") or "unassigned"
    parts = Path(image_relative).parts
    if parts[0] == "frames" and len(parts) >= 3:
        clip_stem = parts[1]
        for entry in entries:
            if entry.get("media_type") == "clip" and Path(entry["path"]).stem == clip_stem:
                return entry.get("session_id") or "unassigned"
    return "unrecorded"


def bin_model_side(side: float) -> str:
    """Return the size bin name for one model-space side length."""
    for edge, name in zip(SIZE_BIN_EDGES, SIZE_BIN_NAMES):
        if side < edge:
            return name
    return SIZE_BIN_NAMES[-1]


def summarize_reviewed_boxes(data_dir: Path) -> dict:
    """Return counts of reviewed images, birds, negatives, sessions, and size bins."""
    entries = load_provenance(data_dir)
    reviewed_root = data_dir / REVIEWED_LABELS
    label_paths = sorted(reviewed_root.rglob("*.txt")) if reviewed_root.is_dir() else []

    images = birds = negatives = 0
    size_bins = Counter()
    images_per_session = Counter()
    birds_per_session = Counter()
    for label_path in label_paths:
        image_path = image_path_for(label_path, data_dir, REVIEWED_LABELS)
        boxes, errors = read_label_file(label_path)
        if image_path is None or errors:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        scale = IMAGE_SIZE / max(width, height)
        session = session_for_image(image_path.relative_to(data_dir).as_posix(), entries)

        images += 1
        images_per_session[session] += 1
        if not boxes:
            negatives += 1
        for box in boxes:
            side = math.sqrt(box.width * width * box.height * height) * scale
            size_bins[bin_model_side(side)] += 1
        birds += len(boxes)
        birds_per_session[session] += len(boxes)

    return {
        "images": images,
        "birds": birds,
        "negatives": negatives,
        "size_bins": {name: size_bins[name] for name in SIZE_BIN_NAMES},
        "sessions": {
            session: {"images": images_per_session[session], "birds": birds_per_session[session]}
            for session in sorted(images_per_session)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    summary = summarize_reviewed_boxes(args.data_dir)
    if summary["images"] == 0:
        print("No valid reviewed labels yet.")
        return
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
