import json
import tempfile
import unittest
from pathlib import Path

from ml.open_images_picker import ReviewStore, build_catalog


class PickerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.catalog = [{"id": "0123456789abcdef", "split": "train", "boxes": [{"text": "Bird"}],
                         "camera_model": None, "location": None, "box_review_status": "unreviewed_source_annotations"}]

    def test_review_survives_restart_and_undo_preserves_history(self):
        store = ReviewStore(self.directory, self.catalog)
        store.save({"id": self.catalog[0]["id"], "decision": "keep", "notes": "Tiny silhouette"})
        restarted = ReviewStore(self.directory, self.catalog)
        self.assertEqual(restarted.picks()[0]["review"]["notes"], "Tiny silhouette")
        self.assertIsNone(restarted.picks()[0]["camera_model"])
        self.assertEqual(restarted.picks()[0]["box_review_status"], "unreviewed_source_annotations")
        restarted.save({"id": self.catalog[0]["id"], "decision": "unreviewed"})
        self.assertEqual(restarted.picks(), [])
        self.assertEqual(len(store.events_path.read_text().splitlines()), 2)
        self.assertEqual(json.loads((self.directory / "selections.json").read_text())[self.catalog[0]["id"]]["decision"], "unreviewed")

    def test_invalid_reviews_do_not_write_artifacts(self):
        store = ReviewStore(self.directory, self.catalog)
        for payload in [{"id": "unknown", "decision": "keep"},
                        {"id": self.catalog[0]["id"], "decision": "approved_boxes"},
                        {"id": self.catalog[0]["id"], "decision": "keep", "notes": "x" * 2001}]:
            with self.assertRaises(ValueError):
                store.save(payload)
        self.assertFalse(store.events_path.exists())

    def test_catalog_keeps_only_bird_boxes_and_uses_titles_as_hints(self):
        metadata = {
            "a": {"image": {"url": "https://c2.staticflickr.com/example.jpg", "footnote_top_right": "Title: Perched bird"}, "objects": []},
            "b": {"image": {"url": "https://c2.staticflickr.com/example.jpg", "footnote_top_right": "Title: Bird in flight"},
                  "objects": [{"category": "/m/015p6", "text": "Bird"}, {"category": "tree", "text": "Tree"}]},
        }
        photos = build_catalog(metadata)
        self.assertEqual(photos[0]["id"], "b")
        self.assertTrue(photos[0]["flight_title_hint"])
        self.assertEqual(len(photos[0]["boxes"]), 1)
        self.assertEqual(photos[0]["thumbnail_url"], "https://live.staticflickr.com/example.jpg")


if __name__ == "__main__":
    unittest.main()
