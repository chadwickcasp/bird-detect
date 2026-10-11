from pathlib import Path

from PIL import Image

from ml.check_labels import check_reviewed_labels
from ml.summarize_boxes import summarize_reviewed_boxes


def write_image(path: Path, size: tuple[int, int] = (1280, 720)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "skyblue").save(path)


def write_label(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_invalid_box_fails_and_empty_label_counts_as_negative(tmp_path: Path) -> None:
    write_image(tmp_path / "incoming/stills/bird.jpg")
    write_image(tmp_path / "incoming/stills/sky.jpg")
    write_image(tmp_path / "incoming/stills/unreviewed.jpg")
    write_label(tmp_path / "labels_reviewed/incoming/stills/bird.txt", "0 0.5 0.5 0.1 0.1\n0 0.98 0.5 0.1 0.1\n")
    write_label(tmp_path / "labels_reviewed/incoming/stills/sky.txt", "")

    report = check_reviewed_labels(tmp_path)

    assert not report.passed
    assert len(report.invalid_lines) == 1
    assert "horizontally" in report.invalid_lines[0]
    assert report.negatives == 1
    assert report.boxes == 1
    assert [path.name for path in report.images_pending_review] == ["unreviewed.jpg"]


def test_rounded_box_touching_left_and_top_edge_passes(tmp_path: Path) -> None:
    write_image(tmp_path / "incoming/stills/edge.jpg")
    # The left and top edges round to -0.000001.
    write_label(tmp_path / "labels_reviewed/incoming/stills/edge.txt", "0 0.049999 0.049999 0.100000 0.100000\n")

    report = check_reviewed_labels(tmp_path)

    assert report.passed
    assert report.boxes == 1


def test_empty_reviewed_directory_passes(tmp_path: Path) -> None:
    write_image(tmp_path / "incoming/stills/bird.jpg")

    report = check_reviewed_labels(tmp_path)

    assert report.passed
    assert report.reviewed_files == 0


def test_summary_bins_model_space_size(tmp_path: Path) -> None:
    write_image(tmp_path / "incoming/stills/bird.jpg", size=(1920, 1080))
    write_image(tmp_path / "incoming/stills/sky.jpg", size=(1920, 1080))
    # 0.03125 * 1920 = 60 source pixels square, which is 20 model pixels after a 1/3 resize.
    write_label(tmp_path / "labels_reviewed/incoming/stills/bird.txt", "0 0.5 0.5 0.03125 0.0555556\n")
    write_label(tmp_path / "labels_reviewed/incoming/stills/sky.txt", "")

    summary = summarize_reviewed_boxes(tmp_path)

    assert summary["images"] == 2
    assert summary["negatives"] == 1
    assert summary["size_bins"]["16-<32"] == 1
    assert summary["sessions"] == {"unrecorded": {"images": 2, "birds": 1}}
