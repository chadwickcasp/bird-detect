"""Check reviewed one-class YOLO labels against the review images.

Run from the repository root:

    .venv/bin/python -m ml.check_labels

Exits with status 1 when a reviewed file has an invalid box or has no matching image. Images
without a reviewed file are reported as pending review, which is not an error.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ml.label_files import REVIEWED_LABELS, image_path_for, label_path_for, list_review_images, read_label_file
from ml.project_paths import DATA_DIR


@dataclass
class LabelCheckReport:
    """Results of one reviewed-label check."""

    reviewed_files: int = 0
    negatives: int = 0
    boxes: int = 0
    invalid_lines: list[str] = field(default_factory=list)
    labels_without_image: list[Path] = field(default_factory=list)
    images_pending_review: list[Path] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.invalid_lines and not self.labels_without_image


def check_reviewed_labels(data_dir: Path) -> LabelCheckReport:
    """Check every reviewed label file and list review images that have none."""
    report = LabelCheckReport()
    reviewed_root = data_dir / REVIEWED_LABELS
    label_paths = sorted(reviewed_root.rglob("*.txt")) if reviewed_root.is_dir() else []
    for label_path in label_paths:
        report.reviewed_files += 1
        if image_path_for(label_path, data_dir, REVIEWED_LABELS) is None:
            report.labels_without_image.append(label_path)
        boxes, errors = read_label_file(label_path)
        report.invalid_lines.extend(errors)
        report.boxes += len(boxes)
        if not boxes and not errors:
            report.negatives += 1
    for image_path in list_review_images(data_dir):
        if not label_path_for(image_path, data_dir, REVIEWED_LABELS).exists():
            report.images_pending_review.append(image_path)
    return report


def print_label_check(report: LabelCheckReport) -> None:
    """Print a short summary of a label check."""
    if report.reviewed_files == 0:
        print("No reviewed labels yet; nothing to check.")
    else:
        print(f"Reviewed files: {report.reviewed_files} ({report.negatives} negatives, {report.boxes} bird boxes)")
    for line in report.invalid_lines:
        print(f"INVALID {line}")
    for path in report.labels_without_image:
        print(f"NO IMAGE {path}")
    if report.images_pending_review:
        print(f"Pending review: {len(report.images_pending_review)} images, for example {report.images_pending_review[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()
    report = check_reviewed_labels(args.data_dir)
    print_label_check(report)
    if not report.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
