"""Write pretrained YOLO26n bird boxes as proposals for human review.

Run from the repository root:

    .venv/bin/python -m ml.propose_bird_boxes

Proposals go to data/labels_proposed/ only. A reviewer corrects them, adds missed birds, and
saves the result under data/labels_reviewed/. An empty proposal file means the model found no
bird. It is not a reviewed negative.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from ml.export_pretrained import COCO_BIRD_CLASS, IMAGE_SIZE, NMS_IOU_THRESHOLD
from ml.label_files import PROPOSED_LABELS, YoloBox, format_label_line, label_path_for, list_review_images
from ml.project_paths import DATA_DIR, WEIGHTS_DIR

# Lower than the app's operating threshold so the reviewer sees borderline birds.
PROPOSAL_CONFIDENCE = 0.15


def propose_bird_boxes(data_dir: Path, checkpoint: Path, confidence: float, overwrite: bool) -> int:
    """Write one proposal file per review image and return the number of files written."""
    model = YOLO(str(checkpoint))
    written = 0
    for image_path in list_review_images(data_dir):
        label_path = label_path_for(image_path, data_dir, PROPOSED_LABELS)
        if label_path.exists() and not overwrite:
            continue
        result = model.predict(
            str(image_path),
            imgsz=IMAGE_SIZE,
            conf=confidence,
            iou=NMS_IOU_THRESHOLD,
            classes=[COCO_BIRD_CLASS],
            verbose=False,
        )[0]
        lines = [format_label_line(YoloBox(*xywhn)) for xywhn in result.boxes.xywhn.tolist()]
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("".join(f"{line}\n" for line in lines))
        written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=WEIGHTS_DIR / "yolo26n.pt")
    parser.add_argument("--confidence", type=float, default=PROPOSAL_CONFIDENCE)
    parser.add_argument("--overwrite", action="store_true", help="replace existing proposal files")
    args = parser.parse_args()
    written = propose_bird_boxes(args.data_dir, args.checkpoint, args.confidence, args.overwrite)
    print(f"Wrote {written} proposal files under {args.data_dir / PROPOSED_LABELS}")


if __name__ == "__main__":
    main()
