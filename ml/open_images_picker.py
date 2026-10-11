"""A local visual shortlist for the Open Images bird gallery."""

import argparse
import json
import re
import threading
from datetime import datetime, timezone
from html import unescape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from ml.project_paths import DATA_DIR
from ml.open_images_catalog import BOXES_URL, FULL_METADATA, load_catalog_metadata, completed_image_ids

METADATA_URL = "https://storage.googleapis.com/openimages/web_v6/visualizer/annotations_detection_train/_m_015p6.json"
VIEWER_URL = "https://storage.googleapis.com/openimages/web/visualizer/index.html"
UI_DIR = Path(__file__).with_name("picker_ui")
FLIGHT_TITLE = re.compile(r"\bfly(?:ing)?\b|\bflight\b|\bsoar|\bairborne\b", re.I)


def plain_text(value):
    return unescape(re.sub(r"<[^>]*>", "", value)).strip()


def build_catalog(metadata):
    photos = []
    for image_id, record in metadata.items():
        image = record["image"]
        title = plain_text(image.get("footnote_top_right", "")).removeprefix("Title: ")
        boxes = [obj for obj in record.get("objects", []) if obj.get("category") == "/m/015p6"]
        photos.append({
            "id": image_id, "split": "train", "title": title,
            "image_url": f"https://open-images-dataset.s3.amazonaws.com/train/{image_id}.jpg",
            "thumbnail_url": re.sub(r"https?://[^/]*staticflickr.com/", "https://live.staticflickr.com/", image["url"]),
            "source_url": f"{VIEWER_URL}?type=detection&set=train&c=%2Fm%2F015p6&id={image_id}",
            "attribution": plain_text(image.get("footnote_bottom_right", "")),
            "source_metadata": image,
            "boxes": boxes, "flight_title_hint": bool(FLIGHT_TITLE.search(title)),
            "camera_model": None, "location": None,
            "box_review_status": "unreviewed_source_annotations",
        })
    return sorted(photos, key=lambda photo: (not photo["flight_title_hint"], photo["id"]))


class ReviewStore:
    def __init__(self, directory, catalog):
        self.directory = directory
        self.catalog = {photo["id"]: photo for photo in catalog}
        self.lock = threading.Lock()
        self.events_path = directory / "review_history.jsonl"
        self.state = {}
        if self.events_path.exists():
            for line in self.events_path.read_text().splitlines():
                event = json.loads(line)
                self.state[event["id"]] = event

    def save(self, payload):
        image_id = payload.get("id")
        decision = payload.get("decision")
        notes = payload.get("notes", "")
        if image_id not in self.catalog or decision not in {"keep", "skip", "unsure", "unreviewed"}:
            raise ValueError("Choose a photo from this gallery and a valid decision.")
        if not isinstance(notes, str) or len(notes) > 2000:
            raise ValueError("Notes must be text of at most 2,000 characters.")
        event = {"id": image_id, "split": "train", "decision": decision,
                 "notes": notes, "reviewer": "human", "updated_at": datetime.now(timezone.utc).isoformat()}
        with self.lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            with self.events_path.open("a") as stream:
                stream.write(json.dumps(event) + "\n")
                stream.flush()
            self.state[image_id] = event
            snapshot = self.directory / "selections.json.tmp"
            snapshot.write_text(json.dumps(self.state, indent=2) + "\n")
            snapshot.replace(self.directory / "selections.json")
        return event

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def picks(self):
        return [{**self.catalog[image_id], "review": event}
                for image_id, event in self.snapshot().items() if event["decision"] == "keep" and image_id in self.catalog]


class PickerServer(ThreadingHTTPServer):
    def __init__(self, address, catalog, store, project_data_dir=None):
        super().__init__(address, PickerHandler)
        self.catalog = catalog
        self.store = store
        self.project_data_dir = Path(project_data_dir) if project_data_dir is not None else store.directory.parent
        self.catalog_source = BOXES_URL if (store.directory / FULL_METADATA).exists() else METADATA_URL

    def visible_catalog(self):
        completed = completed_image_ids(self.project_data_dir)
        return [photo for photo in self.catalog if photo["id"] not in completed]


class PickerHandler(BaseHTTPRequestHandler):
    def respond(self, body, content_type="application/json", status=200, filename=None):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/catalog":
            self.respond(json.dumps(self.server.visible_catalog()))
        elif path == "/api/catalog-info":
            completed = completed_image_ids(self.server.project_data_dir)
            self.respond(json.dumps({"total": len(self.server.catalog),
                                     "hidden": sum(photo["id"] in completed for photo in self.server.catalog),
                                     "full_catalog": (self.server.store.directory / FULL_METADATA).exists()}))
        elif path == "/api/state":
            self.respond(json.dumps(self.server.store.snapshot()))
        elif path == "/api/export":
            self.respond(json.dumps({"source": self.server.catalog_source, "picks": self.server.store.picks()}, indent=2),
                         filename="bird-picks.json")
        elif path == "/api/export-ids":
            self.respond("".join(f'{photo["split"]}/{photo["id"]}\n' for photo in self.server.store.picks()),
                         "text/plain", filename="bird-picks.txt")
        elif path in {"/", "/app.js", "/style.css"}:
            name = "index.html" if path == "/" else path[1:]
            mime = {"index.html": "text/html", "app.js": "text/javascript", "style.css": "text/css"}[name]
            self.respond((UI_DIR / name).read_bytes(), mime + "; charset=utf-8")
        elif path == "/favicon.ico":
            self.respond(b"", "image/x-icon", 204)
        else:
            self.respond(json.dumps({"error": "Not found"}), status=404)

    def do_POST(self):
        if self.path != "/api/review":
            self.respond(json.dumps({"error": "Not found"}), status=404)
            return
        origin = self.headers.get("Origin")
        if origin and origin != f'http://{self.headers.get("Host")}':
            self.respond(json.dumps({"error": "Use the local picker to save."}), status=403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 10000:
                raise ValueError("Invalid request size.")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Expected a review object.")
            event = self.server.store.save(payload)
            self.respond(json.dumps(event))
        except (ValueError, TypeError) as error:
            self.respond(json.dumps({"error": str(error)}), status=400)
        except OSError:
            self.respond(json.dumps({"error": "Could not save. Your decision has not been confirmed."}), status=500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR / "open_images_picker")
    parser.add_argument("--project-data-dir", type=Path, default=DATA_DIR,
                        help="read annotation decisions from this project's data folder")
    args = parser.parse_args()
    metadata_path = args.data_dir / "gallery_metadata.json"
    if not metadata_path.exists() and not (args.data_dir / FULL_METADATA).exists():
        args.data_dir.mkdir(parents=True, exist_ok=True)
        with urlopen(METADATA_URL, timeout=30) as response:
            metadata_path.write_bytes(response.read())
    catalog = build_catalog(load_catalog_metadata(args.data_dir))
    store = ReviewStore(args.data_dir, catalog)
    server = PickerServer(("127.0.0.1", args.port), catalog, store, args.project_data_dir)
    print(f"Bird picker: http://127.0.0.1:{args.port} · {len(catalog)} photos", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
