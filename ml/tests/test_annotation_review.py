import json
import math
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image

from ml.annotation_sources import CandidateSources
from ml.annotation_store import AnnotationStore, ReviewConflict, validate_review
from ml.check_labels import check_reviewed_labels


class AnnotationReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        image = self.root / "incoming/stills/frame.jpg"
        image.parent.mkdir(parents=True)
        Image.new("RGB", (100, 80), "blue").save(image)
        self.proposal = self.root / "labels_proposed/incoming/stills/frame.txt"
        self.proposal.parent.mkdir(parents=True)
        self.proposal.write_text("0 0.5 0.5 0.4 0.5\n")
        self.sources = CandidateSources(self.root)
        self.store = AnnotationStore(self.sources)
        self.identifier = next(iter(self.sources.scan_candidates()))

    def payload(self, action="draft"):
        item = self.store.load(self.identifier)
        return {**item["review"], "source_version": item["source_version"], "reviewed_digest": item["reviewed_digest"], "action": action}

    def test_draft_never_promotes_and_is_recovered_after_restart(self):
        payload = self.payload()
        payload["boxes"][0]["x"] = 0.1
        self.store.save_review(self.identifier, payload)
        self.assertEqual(self.proposal.read_text(), "0 0.5 0.5 0.4 0.5\n")
        self.assertFalse((self.root / "labels_reviewed").exists())
        restarted = AnnotationStore(self.sources)
        self.assertEqual(restarted.load(self.identifier)["review"]["boxes"][0]["x"], 0.1)
        self.assertEqual(len(restarted.history(self.identifier)), 1)

    def test_approval_writes_valid_yolo_and_export_excludes_drafts(self):
        self.store.save_review(self.identifier, self.payload("approve"))
        report = check_reviewed_labels(self.root)
        self.assertTrue(report.passed)
        self.assertEqual(report.boxes, 1)
        with zipfile.ZipFile(BytesIO(self.store.export_reviewed())) as archive:
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(len(manifest["items"]), 1)
            self.assertIn("labels/incoming/stills/frame.txt", archive.namelist())
        with self.assertRaises(ReviewConflict):
            self.store.save_review(self.identifier, self.payload("draft"))

    def test_reopening_archives_previous_approval_and_withdraws_it(self):
        self.store.save_review(self.identifier, self.payload("approve"))
        self.store.save_review(self.identifier, self.payload("reopen"))
        self.assertEqual(self.store.queue()[0]["status"], "draft")
        self.assertFalse(self.sources.reviewed_path(self.store.load(self.identifier)).exists())
        self.assertTrue(self.store.history(self.identifier)[0]["previous_reviewed_yolo"].startswith("0 "))
        with zipfile.ZipFile(BytesIO(self.store.export_reviewed())) as archive:
            self.assertEqual(json.loads(archive.read("manifest.json"))["items"], [])

    def test_zero_box_approval_requires_explicit_negative(self):
        payload = self.payload("approve")
        payload["boxes"] = []
        with self.assertRaises(ValueError):
            self.store.save_review(self.identifier, payload)
        payload["negative"] = True
        self.store.save_review(self.identifier, payload)
        self.assertEqual(check_reviewed_labels(self.root).negatives, 1)

    def test_stale_review_and_changed_proposal_are_rejected(self):
        first, stale = self.payload(), self.payload()
        self.store.save_review(self.identifier, first)
        with self.assertRaises(ReviewConflict):
            self.store.save_review(self.identifier, stale)
        current = self.payload()
        self.proposal.write_text("0 0.4 0.4 0.2 0.2\n")
        with self.assertRaises(ReviewConflict):
            self.store.save_review(self.identifier, current)
        self.assertTrue(self.store.load(self.identifier)["review"]["source_changed"])

    def test_outside_nonfinite_and_unknown_class_are_rejected(self):
        for change in [{"x": -0.1}, {"width": 0}, {"height": math.nan}, {"label": "person"}, {"width": 2}]:
            payload = self.payload()
            payload["boxes"][0].update(change)
            with self.assertRaises(ValueError):
                validate_review(payload)

    def test_queue_refresh_discovers_new_frames_and_uploads_deduplicate(self):
        image = self.root / "frames/clip/second.jpg"
        image.parent.mkdir(parents=True)
        Image.new("RGB", (20, 20)).save(image)
        self.assertEqual(len(self.store.queue()), 2)
        content = image.read_bytes()
        first = self.sources.import_upload("../../phone.jpg", content)
        second = self.sources.import_upload("../../phone.jpg", content)
        self.assertEqual(first, second)
        self.assertEqual(len(self.store.queue()), 3)
        self.assertTrue(self.sources.get_record(first)["path"].startswith("incoming/stills/uploads/"))


if __name__ == "__main__":
    unittest.main()
