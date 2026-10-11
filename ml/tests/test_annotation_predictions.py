import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from ml.annotation_predictions import AnnotationPredictions, bird_class_ids
from ml.annotation_sources import CandidateSources
from ml.annotation_store import AnnotationStore, ReviewConflict


class Values:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeModel:
    names = {0: "bird"}

    def __init__(self):
        self.calls = []
        self.after_predict = None

    def predict(self, image, **settings):
        self.calls.append((image.size, settings))
        if self.after_predict:
            self.after_predict()
        return [SimpleNamespace(orig_shape=(image.height, image.width),
                                boxes=SimpleNamespace(xywhn=Values([[.5, .5, .4, .4]]), conf=Values([.82])))]


class AnnotationPredictionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        image = self.root / "incoming/stills/phone.jpg"
        image.parent.mkdir(parents=True)
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (100, 80), "blue").save(image, exif=exif)
        self.store = AnnotationStore(CandidateSources(self.root))
        self.identifier = self.store.queue()[0]["id"]
        self.checkpoint = self.root / "bird.pt"
        self.checkpoint.write_bytes(b"test checkpoint")
        self.model = FakeModel()
        options = {"local": {"id": "local", "name": "Test bird", "path": self.checkpoint}}
        self.predictions = AnnotationPredictions(self.store, options, loader=lambda path: self.model)

    def request(self, **overrides):
        return {"model": "local", "confidence": .15,
                "source_version": self.store.load(self.identifier)["source_version"], **overrides}

    def save_payload(self, action="draft"):
        item = self.store.load(self.identifier)
        return {**item["review"], "action": action, "source_version": item["source_version"], "reviewed_digest": item["reviewed_digest"]}

    def test_prediction_uses_upright_image_and_records_scores_without_labels(self):
        run = self.predictions.run(self.identifier, self.request())
        self.assertEqual(self.model.calls[0][0], (80, 100))
        settings = self.model.calls[0][1]
        self.assertEqual(settings["classes"], [0])
        self.assertEqual(settings["imgsz"], 640)
        self.assertFalse(settings["rect"])
        self.assertEqual(run["boxes"][0]["confidence"], .82)
        self.assertEqual(len(run["checkpoint_sha256"]), 64)
        self.assertFalse((self.root / "labels_proposed").exists())
        self.assertFalse((self.root / "labels_reviewed").exists())
        self.assertEqual(self.store.queue()[0]["status"], "candidate")
        artifact = self.store.state_path(self.identifier).parent / "model_runs" / f'{run["id"]}.json'
        self.assertEqual(json.loads(artifact.read_text())["id"], run["id"])

    def test_approved_and_withheld_reviews_are_unchanged_by_model_runs(self):
        payload = self.save_payload("needs_work")
        self.store.save_review(self.identifier, payload)
        before = self.store.state_path(self.identifier).read_bytes()
        self.predictions.run(self.identifier, self.request())
        self.assertEqual(before, self.store.state_path(self.identifier).read_bytes())
        payload = self.save_payload("approve")
        payload["negative"] = True
        self.store.save_review(self.identifier, payload)
        label = self.store.sources.reviewed_path(self.store.load(self.identifier))
        before = label.read_bytes(), self.store.state_path(self.identifier).read_bytes()
        self.predictions.run(self.identifier, self.request())
        self.assertEqual(before, (label.read_bytes(), self.store.state_path(self.identifier).read_bytes()))

    def test_using_run_in_a_draft_keeps_provenance_and_still_requires_approval(self):
        run = self.predictions.run(self.identifier, self.request())
        payload = self.save_payload()
        payload.update(boxes=run["boxes"], model_run_id=run["id"])
        self.store.save_review(self.identifier, payload)
        self.assertEqual(self.store.load(self.identifier)["review"]["model_run_id"], run["id"])
        self.assertFalse((self.root / "labels_reviewed").exists())
        payload = self.save_payload("approve")
        self.store.save_review(self.identifier, payload)
        self.assertTrue(self.store.sources.reviewed_path(self.store.load(self.identifier)).exists())

    def test_class_mapping_handles_coco_and_single_bird_checkpoints(self):
        self.assertEqual(bird_class_ids({0: "person", 14: "bird"}), [14])
        self.assertEqual(bird_class_ids(["bird"]), [0])
        with self.assertRaises(ValueError):
            bird_class_ids({0: "person"})

    def test_unknown_models_scores_stale_inputs_and_busy_runs_rejected(self):
        for request in [self.request(model="../../file"), self.request(confidence=float("nan")), self.request(confidence=0), self.request(confidence=True)]:
            with self.assertRaises(ValueError):
                self.predictions.run(self.identifier, request)
        with self.assertRaises(ReviewConflict):
            self.predictions.run(self.identifier, self.request(source_version="old"))
        self.predictions.lock.acquire()
        try:
            with self.assertRaises(ReviewConflict):
                self.predictions.run(self.identifier, self.request())
        finally:
            self.predictions.lock.release()
        payload = self.save_payload()
        payload["model_run_id"] = "a" * 32
        with self.assertRaises(ValueError):
            self.store.save_review(self.identifier, payload)

    def test_changed_source_during_prediction_is_rejected(self):
        image = self.root / "incoming/stills/phone.jpg"
        self.model.after_predict = lambda: Image.new("RGB", (100, 80)).save(image)
        with self.assertRaises(ReviewConflict):
            self.predictions.run(self.identifier, self.request())
