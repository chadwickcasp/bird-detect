# Mobile bird annotation review

The review editor runs locally alongside the Open Images picker. It reads your saved Keep selections and local stills or extracted frames. A Keep selection is a candidate for review; it is not an approved training label.

## Open it on a phone

Start the editor from the repository root:

```bash
.venv/bin/python -m ml.annotation_review --host 0.0.0.0 --port 8768
```

The startup message prints a private access URL. Keep the `?access=...` part, replace `127.0.0.1` with the Mac's Wi-Fi address, and open that link on your phone. Both devices must use the same network, and the Mac must remain awake with the server running. This is a local-network app; it does not provide public internet hosting. Guest networks can prevent devices from reaching one another.

The link unlocks photos and review data in that browser. Keep it private. The key lives in `data/annotation_review/access_token`; it is excluded from Git with the rest of `data/`. The page removes the key from the address bar after opening. For computer-only use, omit `--host 0.0.0.0`.

## Review a frame

1. Open **Queue** and choose an unreviewed candidate, draft, or image marked Needs work. Search by name or frame path. New arrivals are checked every 15 seconds, and **Refresh** checks immediately.
2. Pinch with two fingers on the image to zoom and pan. **Pan** moves the view with one finger. **Fit** returns to the full frame. **Focus** centers the selected box; the percentage button zooms further. The percentage is relative to the fitted view, rather than a promise of one display pixel per source pixel.
3. In **Select**, drag a box to move it, or drag a corner to resize it. Overlapping boxes can be selected through the Box menu. **Add box** draws a missed bird. **Hide boxes** helps inspect the underlying photo.
4. Use Left, Top, Width, and Height for source-image pixel coordinates. **Fine tune** moves a box or adjusts a chosen edge in 1, 5, or 10 pixel steps. Boxes stay within the image. **Undo**, **Redo**, **Delete**, and **Duplicate box** are available.
5. Expand **More review tools** for source comparison, image and box flags, notes, label import, and history. **Compare source boxes** shows the original proposals as dashed yellow boxes. Tags include blur, tiny birds, noise, backlighting, flock, uncertainty, occlusion, truncation, and groups.
6. **Approve & next** writes reviewed labels and advances. **Needs work** saves that status and advances. **Skip** saves edits as a draft and advances without approval. For a negative image, explicitly check **Confirmed no birds** before approving; checking it clears draft boxes and can be undone.

The dataset currently has one object class, `bird`. Label every recognizable bird individually. A source Group box is a review warning: replace it with individual boxes where possible, or leave the frame Needs work if it cannot be annotated under the project's policy. Flags and notes help record artifacts; they do not alter the original photograph or automatically exclude it from export.

Desktop shortcuts: Delete removes the selected box, arrow keys adjust it, Shift uses five pixels, and Ctrl/Cmd+Z undoes (Shift+Z redoes). Shortcuts do not intercept typing in notes or fields.

## Drafts, approval, and recovery

Edits autosave after a short pause. A backup on the current device preserves unsaved edits when a request fails. This is draft recovery, not a fully offline app: initial photo loading, queue refresh, saving to the Mac, and exports need the server.

Concurrent or stale saves are rejected instead of overwriting newer work. After a conflict, **Reload image** obtains the current version; **Recover device draft** copies the old device edits into that version for another review. **Save device draft** downloads a backup when recovery is needed.

Approved images are read-only. **Reopen review** archives the previous approval and withdraws its label file from reviewed exports until you approve again. **History** restores a saved version into a draft; it does not approve it automatically. Draft saves never write to `labels_reviewed`.

## Start with the YOLO model

Expand **Start with YOLO**, choose a model and minimum score, then tap **Run YOLO**. The prediction runs on the Mac, so the phone needs only its browser. The first run loads the checkpoint; later runs reuse it.

Predictions appear as dashed purple boxes with actual model scores. They are a preview: running a model does not alter review boxes, approved labels, source proposals, or Needs work decisions. Toggle **Show model preview** to compare it with your current boxes. Scores are model outputs, not calibrated probabilities of correctness.

**Use model boxes** replaces the current draft's boxes, retains image flags and notes, and autosaves the result as a draft. If boxes already exist, the UI asks before replacing them. **Undo** restores the previous boxes and model-run reference. Add missed birds, delete false detections, and adjust boundaries before approving. No model detections does not confirm a negative; **Confirmed no birds** remains a separate human decision.

On mobile, a successful run returns you to the image. While a new preview is pending, the bottom action says **Use model boxes**; after applying it, the action returns to **Approve & next**. Dismiss a preview to keep and approve your existing boxes.

An approved frame can be used for a read-only model preview. To change its reviewed boxes, explicitly **Reopen review** first. Complex frames remain Needs work unless you deliberately apply boxes or edit and approve them. There is no batch replacement or automatic approval.

Available local models are YOLO26n (the default phone baseline), YOLO26s (a larger baseline), and the student and teacher checkpoints from the earlier smoke test, when those files exist. The smoke checkpoints were trained for one epoch on three fixtures and are marked as smoke tests; they are not validated models trained on your reviewed images.

To add a later bird checkpoint, restart the server with its local path:

```bash
.venv/bin/python -m ml.annotation_review --host 0.0.0.0 --port 8768 --checkpoint /absolute/path/to/best.pt
```

That checkpoint becomes the default Custom option. The model must be an Ultralytics-compatible detection checkpoint with a class named `bird`. The service derives the bird class ID from the model's names, supporting both COCO class 14 and a fine-tuned class-0 bird model. Prediction uses an EXIF-upright image, a fixed 640 × 640 input, CPU execution, and a default minimum score of 0.15. Changing the threshold reruns prediction; it does not train the model.

Every successful run saves an immutable JSON artifact in `data/annotation_review/items/<image-id>/model_runs/`, including boxes, scores, checkpoint SHA-256, parameters, source version, and timing. Drafts started from that run retain its ID through history and export. A stale image or proposal, unavailable model, or concurrent model request produces an error without replacing labels.

The implementation uses the local pinned Ultralytics runtime. Its [prediction API documentation](https://docs.ultralytics.com/modes/predict/) describes the input, class-filtering, threshold, and fixed-shape parameters.

## New candidates and export

- Add stills under `data/incoming/stills/` or frames under `data/frames/`. Matching proposed YOLO files live below `data/labels_proposed/` at the same mirrored path.
- **Queue → Add images** uploads JPEG or PNG photos from a phone, up to 30 MB each. Uploads retain the original bytes and use content-based filename prefixes to avoid overwrites. Capture metadata is recorded using the existing provenance workflow.
- Saved Open Images Keep selections appear automatically. Opening one downloads that photo and source boxes; the whole selection is not downloaded at once. Source attribution, license link, URL, and official train split are recorded. Camera model and geographic location remain unverified. These photos are not automatically Pixel 9 or San Francisco data.
- **Import YOLO** replaces the current draft boxes with a class-0 YOLO text file. It does not overwrite the source proposals. An empty import still needs explicit no-bird confirmation before approval.
- **Queue → Export reviewed** downloads a ZIP containing approved YOLO labels, `classes.txt`, and a manifest with image paths and review metadata. It does not include the photo bytes. Drafts and Needs work frames are excluded.

The display uses EXIF-upright images. Proposal coordinates must refer to that same upright image. A later train/validation/test split still needs to account for related photos, capture sessions, licenses, and any official source split.

## Implementation

| File or folder | Purpose |
| --- | --- |
| `ml/annotation_review.py` | Local server, protected APIs, image display, uploads |
| `ml/annotation_sources.py` | Candidate discovery, individual Open Images imports, provenance |
| `ml/annotation_store.py` | Revision checks, drafts, approval, history, label export |
| `ml/annotation_predictions.py` | Cached local YOLO inference, class mapping, immutable prediction artifacts |
| `ml/review_ui/` | Phone and desktop editor; plain HTML, CSS, and JavaScript |
| `data/annotation_review/items/<id>/current.json` | Current review state |
| `data/annotation_review/items/<id>/versions/` | Immutable saved review versions |
| `data/labels_reviewed/` | Approved labels at mirrored image paths |

The editor shares the existing project's data conventions and Python environment. No frontend package installation or external AI service is needed.

## Design decisions

The mobile concept uses white panels, navy text (`#172d38`), teal actions (`#087e81`), muted labels (`#687981`), and a dark image canvas (`#13232e`). The implementation preserves its header, candidate identity, large image, two tool rows, box inspector, and approval bar. Fields use 16 px text; nudge buttons have at least 44 × 44 px targets. On narrow phones and desktop, fine-tune selectors move above the arrows to keep the targets usable.

Real source photos replace the concept's illustrative bird. Photos fit without cropping, so portrait images show side margins. Additional capabilities are under More review tools to keep the initial view readable. Desktop uses a separate inspector column. On shorter phones, notes and secondary tools require normal page scrolling; image gestures apply only inside the canvas.

The concept's visible labels are preserved: Review birds, Queue, Select, Pan, Add box, Fit, Undo, Redo, Hide boxes, Delete, Focus, coordinate fields, Fine tune, Needs work, Approve & next, and Skip. Candidate state correctly says Not reviewed until a draft exists. File names, dimensions, box numbers, coordinates, zoom, and save status reflect actual data.
