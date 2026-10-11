"""Extract sparse frames from clips in data/incoming/clips/ for labeling.

Run from the repository root:

    .venv/bin/python -m ml.extract_clip_frames --every-seconds 1.0

Frames land in data/frames/<clip stem>/<clip stem>_f<frame index>.jpg. Each frame inherits its
clip's session_id from data/provenance/sources.yaml. Existing frames are kept.
"""

import argparse
from pathlib import Path

import cv2

from ml.project_paths import DATA_DIR
from ml.record_capture import CLIP_SUFFIXES


def extract_sparse_frames(clip_path: Path, frames_root: Path, every_seconds: float) -> list[Path]:
    """Write one frame per `every_seconds` of one clip and return the frame paths written."""
    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise OSError(f"Cannot open {clip_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps * every_seconds))
    output_dir = frames_root / clip_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    frame_index = 0
    try:
        while True:
            ok = capture.grab()
            if not ok:
                break
            if frame_index % step == 0:
                frame_path = output_dir / f"{clip_path.stem}_f{frame_index:06d}.jpg"
                if not frame_path.exists():
                    _, frame = capture.retrieve()
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    written.append(frame_path)
            frame_index += 1
    finally:
        capture.release()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--every-seconds", type=float, default=1.0)
    args = parser.parse_args()
    clips_dir = args.data_dir / "incoming" / "clips"
    clips = sorted(path for path in clips_dir.glob("*") if path.suffix.lower() in CLIP_SUFFIXES) if clips_dir.is_dir() else []
    if not clips:
        print(f"No clips in {clips_dir}")
        return
    for clip in clips:
        written = extract_sparse_frames(clip, args.data_dir / "frames", args.every_seconds)
        print(f"{clip.name}: wrote {len(written)} frames")


if __name__ == "__main__":
    main()
