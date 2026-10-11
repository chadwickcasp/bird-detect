"""Locate review images and read one-class YOLO label files under data/.

A label file mirrors its image path below a labels root. For example, the reviewed label for
data/frames/S01/clip_f000030.jpg is data/labels_reviewed/frames/S01/clip_f000030.txt.
"""

from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
REVIEW_IMAGE_DIRS = ("incoming/stills", "frames")
PROPOSED_LABELS = "labels_proposed"
REVIEWED_LABELS = "labels_reviewed"
BIRD_CLASS = 0
# format_label_line rounds center and size separately, so an edge box can land just past 0 or 1.
EDGE_TOLERANCE = 1e-4


@dataclass(frozen=True)
class YoloBox:
    """One normalized YOLO box: center x, center y, width, height, each in 0-1."""

    center_x: float
    center_y: float
    width: float
    height: float


def list_review_images(data_dir: Path) -> list[Path]:
    """Return every still and extracted frame that needs a reviewed label, in path order."""
    images = []
    for relative_dir in REVIEW_IMAGE_DIRS:
        image_dir = data_dir / relative_dir
        if image_dir.is_dir():
            images.extend(path for path in image_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    return sorted(images)


def label_path_for(image_path: Path, data_dir: Path, labels_dir_name: str) -> Path:
    """Return the label file that mirrors one image below data/<labels_dir_name>/."""
    relative = image_path.relative_to(data_dir).with_suffix(".txt")
    return data_dir / labels_dir_name / relative


def image_path_for(label_path: Path, data_dir: Path, labels_dir_name: str) -> Path | None:
    """Return the image a label file mirrors, or None when no image exists."""
    image_stem = data_dir / label_path.relative_to(data_dir / labels_dir_name).with_suffix("")
    if not image_stem.parent.is_dir():
        return None
    # Listing the directory returns the on-disk spelling, even on case-insensitive filesystems.
    for candidate in sorted(image_stem.parent.iterdir()):
        if candidate.stem == image_stem.name and candidate.suffix.lower() in IMAGE_SUFFIXES:
            return candidate
    return None


def parse_label_line(line: str) -> YoloBox:
    """Parse one label line and raise ValueError when it is not a valid bird box."""
    fields = line.split()
    if len(fields) != 5:
        raise ValueError(f"expected 5 fields, got {len(fields)}")
    if int(fields[0]) != BIRD_CLASS:
        raise ValueError(f"class must be {BIRD_CLASS}, got {fields[0]}")
    box = YoloBox(*(float(value) for value in fields[1:]))
    if box.width <= 0 or box.height <= 0:
        raise ValueError("width and height must be positive")
    low, high = -EDGE_TOLERANCE, 1 + EDGE_TOLERANCE
    if not (low <= box.center_x - box.width / 2 and box.center_x + box.width / 2 <= high):
        raise ValueError("box extends outside the image horizontally")
    if not (low <= box.center_y - box.height / 2 and box.center_y + box.height / 2 <= high):
        raise ValueError("box extends outside the image vertically")
    return box


def read_label_file(label_path: Path) -> tuple[list[YoloBox], list[str]]:
    """Return the valid boxes and one error message per invalid line. An empty file is a negative."""
    boxes = []
    errors = []
    for line_number, line in enumerate(label_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            boxes.append(parse_label_line(line))
        except ValueError as error:
            errors.append(f"{label_path}:{line_number}: {error}")
    return boxes, errors


def format_label_line(box: YoloBox) -> str:
    """Return one label line for a bird box."""
    return f"{BIRD_CLASS} {box.center_x:.6f} {box.center_y:.6f} {box.width:.6f} {box.height:.6f}"
