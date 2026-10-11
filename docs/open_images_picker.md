# Open Images visual picker

Scan the grid and click **Keep**. Click a photo or **Preview** to enlarge it. **My picks** shows the shortlist. Selections save in this project, so Codex can read them when you ask it to use your picks. No image IDs need to be copied.

The preview provides Keep, Skip, Unsure, optional notes, source attribution, and the original annotation link. K, S, and U save a decision and advance. Arrow keys browse without making a decision. Undo reverses the most recent decision in this browser session. Notes are saved when you choose Keep, Skip, or Unsure.

**Show bird boxes** displays the source annotations without cropping the photographs. Export picks downloads JSON containing the selections, source metadata, notes, and bird boxes. The local `/api/export-ids` endpoint also exports the `train/image_id` format accepted by the official Open Images downloader.

## Run

From the repository root:

```sh
python3 -m ml.open_images_picker --port 8766
```

Open `http://127.0.0.1:8766/`. The server uses Python's standard library. Browser previews load real public photographs as needed. It binds only to this computer.

To build or refresh the full training catalog, run `python3 -m ml.open_images_catalog`, then restart the picker. The importer streams the official bounding-box and image-attribution CSVs, retaining only Bird records rather than storing the multi-GB source CSVs or downloading all photographs. It publishes the catalog only after both source files have been read and every image has attribution metadata. Existing gallery records retain their original titles and attribution; new imports use the full-precision CSV boxes. Previously imported proposals retain their recorded source metadata and box order. The annotation review UI uses this same expanded catalog for new Keep selections.

On each UI reload, the picker reads current annotation decisions and hides **Approved** and **Needs work** images throughout the grid, including My picks. Drafts remain visible, and explicitly reopening a review makes it eligible again. Existing approved label files are also respected. The hidden photos' selections, decision history, downloaded images, and labels are preserved; both export endpoints still include every saved Keep selection. My picks counts only the photos visible in the picker. The source line reports the full catalog size and the number hidden.

## Saved selections and scope

- `data/open_images_picker/selections.json`: current decisions, keyed by image ID.
- `data/open_images_picker/review_history.jsonl`: append-only decision history.
- `data/open_images_picker/gallery_metadata.json`: source gallery metadata.
- `data/open_images_picker/bird_metadata.json`: complete training Bird catalog.
- `data/open_images_picker/bird_catalog_manifest.json`: official source URLs, SHA-256 hashes, import date, and counts.

These files are ignored by Git. Selections are a shortlist, separate from reviewed training labels. Keeping a photo does not approve its boxes or confirm its camera, species, or location.

The full training annotations imported on October 11, 2026 contain **18,525 Bird images and 47,921 Bird boxes**. These include the original 1,000-image gallery sample; validation and test images are not added to this training picker. If the full catalog is absent, the picker falls back to the gallery sample. Titles containing flight-related words appear first as a convenience; this is not an automatic flight classifier, and the remaining photos can contain flying birds too. Camera model and location remain unknown. Open Images is older than Pixel 9, so these are supplementary bird examples rather than confirmed Pixel 9 imagery. Group, occlusion, truncation, and depiction attributes remain attached to their source boxes in the export. Photographs and boxes need checking before training.

Source: [Open Images gallery metadata](https://storage.googleapis.com/openimages/web_v6/visualizer/annotations_detection_train/_m_015p6.json), [official download documentation](https://storage.googleapis.com/openimages/web/download_v7.html).

Implementation: `ml/open_images_catalog.py` imports and caches the expanded catalog and reads completed annotation decisions; `ml/open_images_picker.py` serves the filtered grid and preserved exports; `ml/annotation_sources.py` discovers kept candidates from the same catalog.

## Interface reference and verification

The built-in imagegen tool produced the layout reference at `/Users/chadcasper/.codex/generated_images/01a1273c-ae52-73b0-88f4-eff5ecc53578/exec-8196786d-16a7-4148-8806-9a1b1d94db66.png`. Brief: a white four-column bird picker, navy system typography, teal Keep and Export controls, My picks, box toggle, Undo, and an enlarged review dialog. Generated reference photos are illustrative and are not used by the app.

Design tokens: white background; text #172d38; muted text #687981; accent #087e81; borders #d9e1e6; 8px photo containers; 20px grid gaps; Arial/Helvetica typography; 42px desktop heading; 16px controls. Images are contained without a tint or crop. The compact utility is plain HTML/CSS/JavaScript served by `ml.open_images_picker`, rather than introducing a frontend framework.

Intentional reference differences: real Open Images photos replace generated examples; uncropped portrait photos have neutral side margins; 24 initial photos allow useful scanning rather than only eight; dynamic saved-state messages and sample-size disclosure are required for an honest review workflow; the Undo control is disabled until a decision exists. The three-column intermediate breakpoint and single-column phone view preserve readable previews.

Browser verification used the Codex in-app browser, with screenshots at the reference's native 1505×1045 size and at 390×844. Both the reference and the final screenshots were inspected with `view_image`. Compared heading and subtitle copy, four-column grid anatomy, typography hierarchy, white/navy/teal palette, photo containment, and button placement. Above-the-fold copy matches the reference; the dynamic pick count and Kept state reflect real saved selections. No material visual mismatches remain beyond the intentional differences above. The implementation was faithfully verified against the reference.

Functional checks: real photo loading; Keep; My picks; persistence after reload; enlarged preview; box toggle; Unsure plus notes; Undo; exported JSON containing the selected ID, split, and bird boxes. Functional mutations used an isolated temporary review store. Three focused Python tests cover restart persistence, reversible decisions and audit history, invalid review rejection, and catalog preservation. Phone layout has no horizontal overflow. Fixed hidden controls being made visible by the shared button style.

Final browser screenshots: `outputs/open_images_picker/desktop.jpg` and `outputs/open_images_picker/mobile.jpg`.
