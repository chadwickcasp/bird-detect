import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.request import urlopen

from ml.annotation_sources import CandidateSources, image_identifier
from ml.open_images_catalog import attach_image_metadata, completed_image_ids, load_catalog_metadata, read_bird_boxes
from ml.open_images_picker import PickerServer, ReviewStore, build_catalog

FIRST = "0123456789abcdef"
SECOND = "abcdef0123456789"
THIRD = "1111111111111111"
BOX_CSV = """ImageID,LabelName,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
0123456789abcdef,/m/015p6,0.1,0.4,0.2,0.6,1,0,0,0,-1
0123456789abcdef,/m/015p6,0.5,0.8,0.1,0.3,0,0,1,0,0
abcdef0123456789,/m/015p6,0.1,0.4,0.2,0.6,0,0,0,0,0
1111111111111111,/m/tree,0.1,0.4,0.2,0.6,0,0,0,0,0
"""
IMAGE_CSV = """ImageID,Subset,OriginalURL,OriginalLandingURL,License,AuthorProfileURL,Author,Title,Thumbnail300KURL,Rotation
0123456789abcdef,train,https://example.com/a.jpg,https://example.com/a,https://creativecommons.org/licenses/by/2.0/,https://example.com/author,A & B,Birds flying,https://example.com/thumb.jpg,0
abcdef0123456789,train,https://example.com/b.jpg,https://example.com/b,https://creativecommons.org/licenses/by/2.0/,,C,Perched bird,,0
"""


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.picker_root = self.root / "open_images_picker"
        self.picker_root.mkdir()

    def write_review(self, identifier, status):
        relative = f"incoming/stills/open_images/train/{identifier}.jpg"
        path = self.root / "annotation_review" / "items" / image_identifier(relative) / "current.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"path": relative, "status": status}))
        return path

    def metadata(self):
        metadata = read_bird_boxes(io.StringIO(BOX_CSV))
        attach_image_metadata(metadata, io.StringIO(IMAGE_CSV))
        return metadata

    def test_complete_boxes_keep_attributes_and_attribution(self):
        metadata = self.metadata()
        self.assertEqual(set(metadata), {FIRST, SECOND})
        self.assertEqual(len(metadata[FIRST]["objects"]), 2)
        self.assertEqual(metadata[FIRST]["objects"][0]["unknown_attributes"], ["Inside"])
        photos = build_catalog(metadata)
        self.assertEqual(photos[0]["id"], FIRST)
        self.assertIn("A & B", photos[0]["attribution"])
        self.assertEqual(photos[0]["boxes"][1]["attributes"], ["GroupOf"])
        self.assertIn(SECOND, photos[1]["thumbnail_url"])
        with self.assertRaises(ValueError):
            attach_image_metadata(read_bird_boxes(io.StringIO(BOX_CSV)), io.StringIO(IMAGE_CSV.splitlines()[0] + "\n"))

    def test_new_catalog_preserves_original_gallery_and_selections(self):
        metadata = self.metadata()
        gallery = {FIRST: {**metadata[FIRST], "image": {**metadata[FIRST]["image"], "footnote_top_right": "Original title"}}}
        (self.picker_root / "gallery_metadata.json").write_text(json.dumps(gallery))
        (self.picker_root / "bird_metadata.json").write_text(json.dumps(metadata))
        merged = load_catalog_metadata(self.picker_root)
        self.assertEqual(merged[FIRST]["image"], gallery[FIRST]["image"])
        self.assertEqual(merged[FIRST]["objects"], metadata[FIRST]["objects"])
        self.assertIn(SECOND, merged)
        store = ReviewStore(self.picker_root, build_catalog(merged))
        store.save({"id": SECOND, "decision": "keep"})
        candidates = CandidateSources(self.root).scan_candidates()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(next(iter(candidates.values()))["remote"]["id"], SECOND)
        self.assertFalse((self.root / "incoming").exists())
        relative = f"incoming/stills/open_images/train/{SECOND}.jpg"
        source = self.root / "annotation_review" / "sources" / f"{image_identifier(relative)}.json"
        source.parent.mkdir(parents=True)
        imported_photo = {**build_catalog(merged)[1], "boxes": [{"category": "/m/015p6", "text": "Original source"}]}
        source.write_text(json.dumps(imported_photo))
        candidate = next(iter(CandidateSources(self.root).scan_candidates().values()))
        self.assertEqual(candidate["remote"]["boxes"], imported_photo["boxes"])
        # Cache must notice a source replacement, not only a restart.
        metadata[THIRD] = metadata[SECOND]
        (self.picker_root / "bird_metadata.json").write_text(json.dumps(metadata))
        self.assertIn(THIRD, load_catalog_metadata(self.picker_root))

    def test_completed_filter_respects_needs_work_legacy_labels_and_reopening(self):
        self.write_review(FIRST, "approved")
        needs_work = self.write_review(SECOND, "needs_work")
        self.write_review(THIRD, "draft")
        self.assertEqual(completed_image_ids(self.root), {FIRST, SECOND})
        before = needs_work.read_bytes()
        self.write_review(FIRST, "draft")
        self.assertEqual(completed_image_ids(self.root), {SECOND})
        legacy = self.root / "labels_reviewed" / "incoming" / "stills" / "open_images" / "train" / f"{FIRST}.txt"
        legacy.parent.mkdir(parents=True)
        legacy.write_text("")
        self.assertEqual(completed_image_ids(self.root), {FIRST, SECOND})
        self.assertEqual(needs_work.read_bytes(), before)

    def test_reload_filters_grid_but_exports_all_saved_picks(self):
        metadata = self.metadata()
        (self.picker_root / "bird_metadata.json").write_text(json.dumps(metadata))
        catalog = build_catalog(metadata)
        store = ReviewStore(self.picker_root, catalog)
        for identifier in (FIRST, SECOND):
            store.save({"id": identifier, "decision": "keep"})
        saved = (self.picker_root / "selections.json").read_bytes()
        server = PickerServer(("127.0.0.1", 0), catalog, store, self.root)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        base = f"http://127.0.0.1:{server.server_port}"
        with urlopen(base + "/api/catalog") as response:
            self.assertEqual(len(json.load(response)), 2)
        self.write_review(FIRST, "approved")
        self.write_review(SECOND, "needs_work")
        with urlopen(base + "/api/catalog") as response:
            self.assertEqual(json.load(response), [])
        with urlopen(base + "/api/export") as response:
            self.assertEqual(len(json.load(response)["picks"]), 2)
        self.write_review(SECOND, "draft")
        with urlopen(base + "/api/catalog") as response:
            self.assertEqual([photo["id"] for photo in json.load(response)], [SECOND])
        with urlopen(base + "/api/catalog-info") as response:
            self.assertEqual(json.load(response), {"total": 2, "hidden": 1, "full_catalog": True})
        self.assertEqual((self.picker_root / "selections.json").read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
