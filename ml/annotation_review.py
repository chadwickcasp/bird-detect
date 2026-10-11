"""Serve the mobile bird annotation editor and versioned review API."""

import argparse
import base64
import io
import json
import secrets
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse

from PIL import Image, ImageOps

from ml.annotation_predictions import AnnotationPredictions, checkpoint_options
from ml.annotation_sources import CandidateSources
from ml.annotation_store import AnnotationStore, ReviewConflict
from ml.label_files import IMAGE_SUFFIXES
from ml.project_paths import DATA_DIR

UI_DIR = Path(__file__).with_name("review_ui")
STATIC_FILES = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"),
                "/canvas.js": ("canvas.js", "text/javascript"), "/geometry.js": ("geometry.js", "text/javascript"),
                "/style.css": ("style.css", "text/css"), "/manifest.webmanifest": ("manifest.webmanifest", "application/manifest+json")}


class AnnotationServer(ThreadingHTTPServer):
    def __init__(self, address, store, token, predictions=None):
        super().__init__(address, AnnotationHandler)
        self.store = store
        self.token = token
        self.predictions = predictions or AnnotationPredictions(store)


class AnnotationHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Avoid writing phone access tokens into request logs.
        pass

    def reply(self, content, mime="application/json", status=200, filename=None, cookie=None):
        data = content.encode() if isinstance(content, str) else content
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        if cookie:
            self.send_header("Set-Cookie", cookie)
        self.end_headers()
        self.wfile.write(data)

    def authenticated(self):
        cookies = SimpleCookie()
        try:
            cookies.load(self.headers.get("Cookie", ""))
        except Exception:
            return False
        supplied = cookies.get("bird_review")
        return bool(supplied and secrets.compare_digest(supplied.value, self.server.token))

    def require_access(self):
        if self.authenticated():
            return True
        self.reply(json.dumps({"error": "Open the phone access link from this computer to unlock the editor."}), status=401)
        return False

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        supplied = parse_qs(parsed.query).get("access", [""])[0]
        if path == "/" and supplied and secrets.compare_digest(supplied, self.server.token):
            self.reply((UI_DIR / "index.html").read_bytes(), "text/html; charset=utf-8",
                       cookie=f"bird_review={self.server.token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=2592000")
            return
        if path in STATIC_FILES:
            name, mime = STATIC_FILES[path]
            self.reply((UI_DIR / name).read_bytes(), mime + "; charset=utf-8")
            return
        if path == "/favicon.ico":
            self.reply(b"", "image/x-icon", 204)
            return
        if not self.require_access():
            return
        try:
            if path == "/api/queue":
                self.reply(json.dumps(self.server.store.queue()))
            elif path == "/api/models":
                self.reply(json.dumps(self.server.predictions.configuration()))
            elif path == "/api/export":
                self.reply(self.server.store.export_reviewed(), "application/zip", filename="reviewed-bird-labels.zip")
            elif path.startswith("/api/item/"):
                identifier = path.removeprefix("/api/item/")
                self.reply(json.dumps(self.server.store.load(identifier)))
            elif path.startswith("/api/history/"):
                self.reply(json.dumps(self.server.store.history(path.removeprefix("/api/history/"))))
            elif path.startswith("/api/image/"):
                identifier = path.removeprefix("/api/image/")
                candidate = self.server.store.sources.get_record(identifier)
                image_path = self.server.store.sources.materialize_image(candidate)
                with Image.open(image_path) as raw:
                    image = ImageOps.exif_transpose(raw).convert("RGB")
                    output = io.BytesIO()
                    image.save(output, format="JPEG", quality=95)
                self.reply(output.getvalue(), "image/jpeg")
            else:
                self.reply(json.dumps({"error": "Not found"}), status=404)
        except KeyError as error:
            self.reply(json.dumps({"error": str(error)}), status=404)
        except (ValueError, OSError, URLError) as error:
            self.reply(json.dumps({"error": f"Could not load this candidate: {error}"}), status=422)

    def do_POST(self):
        if not self.require_access():
            return
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            self.reply(json.dumps({"error": "Save from the review editor."}), status=403)
            return
        path = urlparse(self.path).path
        if path == "/api/upload":
            self.upload_image()
            return
        if not path.startswith(("/api/review/", "/api/predict/")):
            self.reply(json.dumps({"error": "Not found"}), status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1024 * 1024:
                raise ValueError("Invalid request size.")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("Expected a review object.")
            if path.startswith("/api/predict/"):
                self.reply(json.dumps(self.server.predictions.run(path.removeprefix("/api/predict/"), payload)))
            else:
                self.reply(json.dumps(self.server.store.save_review(path.removeprefix("/api/review/"), payload)))
        except ReviewConflict as error:
            self.reply(json.dumps({"error": str(error)}), status=409)
        except (ValueError, TypeError, KeyError) as error:
            self.reply(json.dumps({"error": str(error)}), status=400)
        except OSError:
            self.reply(json.dumps({"error": "Could not write this review. Your draft remains on this device."}), status=500)
        except (RuntimeError, ImportError) as error:
            self.reply(json.dumps({"error": f"Model unavailable: {error}"}), status=503)

    def upload_image(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 42 * 1024 * 1024:
                raise ValueError("Choose an image smaller than 30 MB.")
            payload = json.loads(self.rfile.read(length))
            filename = Path(payload["name"]).name
            if Path(filename).suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError("Choose a JPEG or PNG image.")
            content = base64.b64decode(payload["content"], validate=True)
            if len(content) > 30 * 1024 * 1024:
                raise ValueError("Choose an image smaller than 30 MB.")
            identifier = self.server.store.sources.import_upload(filename, content)
            self.reply(json.dumps({"id": identifier}))
        except (ValueError, KeyError, TypeError, OSError) as error:
            self.reply(json.dumps({"error": str(error)}), status=400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "0.0.0.0"])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, help="add a local fine-tuned .pt checkpoint as the default model")
    args = parser.parse_args()
    root = args.data_dir / "annotation_review"
    root.mkdir(parents=True, exist_ok=True)
    token_path = root / "access_token"
    if not token_path.exists():
        token_path.write_text(secrets.token_urlsafe(32))
        token_path.chmod(0o600)
    token = token_path.read_text().strip()
    store = AnnotationStore(CandidateSources(args.data_dir))
    predictions = AnnotationPredictions(store, checkpoint_options(args.checkpoint))
    server = AnnotationServer((args.host, args.port), store, token, predictions)
    print(f"Bird review: http://127.0.0.1:{args.port}/?access={token}", flush=True)
    if args.host == "0.0.0.0":
        print("Phone access: replace 127.0.0.1 with this computer's Wi-Fi address. Keep the access link private.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
