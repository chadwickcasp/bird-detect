# Data conventions

Status: October 10, 2026. This page describes where your photos, clips, and labels go. Selected Open Images photos can be reviewed as candidates; no project train/validation/test split is frozen.

## Folders

Everything below lives in `data/`. Git ignores that folder.

| Folder | Contents |
| --- | --- |
| `data/incoming/stills/` | Still photos, with and without birds |
| `data/incoming/clips/` | Short videos |
| `data/frames/<clip stem>/` | Sparse frames extracted from one clip |
| `data/labels_proposed/` | Model-proposed boxes. Never edited by hand. |
| `data/labels_reviewed/` | Boxes a person has checked. Only these count as labels. |
| `data/provenance/sources.yaml` | One entry per still or clip |
| `data/annotation_review/` | Mobile review drafts, immutable history, source records, and private access key |

A label file mirrors its image path. The reviewed label for `data/frames/S01/S01_f000030.jpg` is `data/labels_reviewed/frames/S01/S01_f000030.txt`.

## Provenance

Copy [`configs/provenance.example.yaml`](../configs/provenance.example.yaml) to `data/provenance/sources.yaml`, or let `ml.record_capture` create it. Every still and clip gets one entry.

- `session_id` groups everything captured in one outing with one setup. Frames, crops, and augmentations inherit it. Later splits are assigned by session, never by frame.
- `media_type` is `still` or `clip`.
- `camera_make`, `camera_model`, `lens`, and `focal_length_mm` describe the capture. Pixel 9 Pro rear-camera footage is the phone domain. Frames from other cameras stay in the same folders and keep their own camera fields.
- `author`, `license`, and `source_url` record ownership. Public images need the page URL and the exact license. A dataset's terms are recorded separately from this repository's license.

For stills, `ml.record_capture` fills camera fields from EXIF. It also proposes a session id from the capture date and camera model. Two outings on one day with the same camera share that id until you edit it. Clips get blank camera fields because video metadata varies by app.

## Labels

Labels are one-class YOLO text files. Each line is one bird:

```text
0 center_x center_y width height
```

All four numbers are fractions of the image width or height. Class `0` is `bird`. An empty reviewed file means a person checked the image and found no bird. That is a negative.

Label every bird a person can recognize as a bird in the frame. A speck you cannot tell from noise, an insect, or a distant aircraft is not labeled. Write a short note in `notes` when a session has many ambiguous specks. These rules apply to every frame used for evaluation.

## Workflow

1. Drop files into `data/incoming/stills/` and `data/incoming/clips/`.
2. Record them and fill in any blank `session_id`, `author`, and `license`:

   ```bash
   .venv/bin/python -m ml.record_capture
   ```

3. Extract one frame per second from each clip. Neighboring frames are nearly identical, so sparse frames spend review time better:

   ```bash
   .venv/bin/python -m ml.extract_clip_frames --every-seconds 1.0
   ```

4. Propose boxes with pretrained YOLO26n. The proposal threshold is 0.15, lower than the app's 0.25, so borderline birds appear for review:

   ```bash
   .venv/bin/python -m ml.propose_bird_boxes
   ```

5. Review each proposal in the [mobile review editor](annotation_review.md). Fix boxes, delete false boxes, and add missed birds. Drafts autosave; **Approve & next** writes the result under `data/labels_reviewed/` at the mirrored path. Explicitly confirm no birds for a negative. Small and blurred birds are the ones the model misses most often. An external YOLO-format editor such as [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) can also write reviewed labels at the same paths.
6. Check the reviewed labels and look at the counts:

   ```bash
   .venv/bin/python -m ml.check_labels
   .venv/bin/python -m ml.summarize_boxes
   ```

   `check_labels` exits with status 1 for an invalid box or a label without an image. Images without a reviewed file are listed as pending.

7. Open [`notebooks/inspect_birds.ipynb`](../notebooks/inspect_birds.ipynb) to compare proposed and reviewed boxes side by side.

## Box size bins

`ml.summarize_boxes` reports bird size in model pixels. The side length is `sqrt(width x height)` after the image is letterboxed into 640x640. A 1920x1080 frame is scaled by 1/3, so a 48x48 source box becomes 16x16.

The bins are `<16`, `16-<32`, `32-<64`, and `>=64`. These are project bins, not COCO size categories. Freeze them before any final test.

## Public data

The mobile editor imports selected Open Images candidates individually when opened and records their source, attribution, license link, and official train split in `sources.yaml`. Their camera model and location remain unverified. Preserve official split boundaries and check related images before assigning project splits. Other public imports must also record licenses and page URLs. The scope document's earlier FBD-SV-2024 suggestion is research-limited; it is not automatically ingested by this editor.
