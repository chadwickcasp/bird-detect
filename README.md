# Bird Detect

An Android bird detection MVP focused on local camera inference and measured accuracy–latency tradeoffs on a Google Pixel 9 Pro.

**Status:** the hours 0–3 gate passed on October 10, 2026. COCO-pretrained YOLO26n runs on the Pixel 9 Pro through ncnn on CPU and Vulkan GPU. Its boxes match the host on all four backends. The app reads the rear camera, a still image, or a clip. The Tensor G4 compile is blocked because the compiler plugin is not publicly downloadable. No fine-tuned model, collected dataset, or sustained benchmark exists yet.

## MVP plan

- Detect birds with bounding boxes and confidence scores in rear-camera video or local clips.
- Establish a COCO-pretrained YOLO26n baseline using the ncnn C++ runtime on Android.
- Fine-tune a YOLO26n student and YOLO26s teacher using public bird data and phone footage.
- Test teacher–student distillation at a fixed input size.
- Compare CPU and Vulkan GPU inference, then test a Tensor G4-targeted LiteRT AOT artifact on the phone's TPU.
- Compare exported models on held-out recording sessions and measure sustained on-device latency and throughput.

Start with the [concise implementation plan](docs/implementation_plan.md) for the workflow diagram, proposed app design, repository layout, and ML workflow. See the [full MVP scope](docs/android_bird_detection_mvp.md) for the data plan, experiment matrix, evaluation protocol, integration gates, and approximate 20-hour work budget. The model choice was updated to YOLO26 on October 10, 2026. Most environment observations are dated October 7; G4 support was rechecked October 9. Verify the selected exports and device execution during implementation.

The [mobile HTML version](docs/implementation_plan.html) presents the same plan with responsive text, stacked model comparisons, and section navigation. Open it in a browser; its layout requires no external assets or internet connection.

The [Gemini extensions review](docs/gemini_extensions_review.md) preserves the October 10 proposal and evaluates it against this scope, recommending size-based recall, clutter-negative evaluation, and a sustained endurance run while deferring tracking and additional platforms.

The [implementation plan review](docs/implementation_plan_review.md) is a separate October 10 reading of the earlier YOLOv8 scope, written before any application code. Both reviews retain their historical model-specific discussion; follow the revised implementation plan for YOLO26 and verify its export contract directly.

## Getting started

Run every command from the repository root.

### Host environment

The pinned versions and install notes are in `ml/requirements.txt`.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r ml/requirements.txt
uv pip install --python .venv/bin/python --no-deps ncnn==1.0.20260526
```

### Export and host parity

```bash
.venv/bin/python -m ml.export_pretrained
.venv/bin/python -m ml.compare_predictions host
```

The export writes `weights/yolo26n_ncnn_model/`, `weights/yolo26n.tflite`, and the tensor record `outputs/exports/pretrained_yolo26n.json`. The record lists each runtime's tensor names, output shape, and box units. ncnn boxes are in input pixels. LiteRT boxes are fractions of the 640-pixel input.

### Android camera app

The app needs Android Studio's JDK, NDK 29.0.14206865, and CMake 3.31.5. Point `android/local.properties` at your SDK with `sdk.dir=...`.

Pair this Mac with the phone once. After that, installs go over Wi-Fi. The phone and the Mac need the same network.

On the phone, open Developer options. Turn on Wireless debugging. Tap Pair device with pairing code. The dialog shows an address and a six-digit code. Run the pair command before the dialog closes:

```bash
android/connect_phone.sh pair 192.168.1.20:37123 123456
```

Use the address and code from the dialog. This Mac stays paired after that.

Turn Wireless debugging on when you want to install. `android/connect_phone.sh` finds the phone. If discovery finds nothing, read IP address & Port on that same screen. Pass that address to `android/connect_phone.sh connect 192.168.1.20:34567`. That port is the connection port. The pairing dialog uses a different port.

```bash
.venv/bin/python -m ml.stage_android_model
android/fetch_native_deps.sh
(cd android && ./gradlew :app:assembleRelease)
android/connect_phone.sh install
android/connect_phone.sh adb shell am start -n com.birddetect/.MainActivity --ei source 1 --ei backend 3 --ez run_fixtures true
```

`install` sends `android/app/build/outputs/apk/release/app-release.apk`. Pass another apk path to install a different package. Prefix other adb commands with `android/connect_phone.sh adb`. The SDK `adb` is not on the default `PATH`.

The launch extras are optional. `source` is 0 for Camera, 1 for Image, and 2 for Clip. `backend` is 0 for CPU FP32, 1 for CPU FP16, 2 for GPU FP32, and 3 for GPU FP16. `run_fixtures` writes a predictions file for the parity fixtures after the model loads. [docs/phone_baseline.md](docs/phone_baseline.md) shows how to pull that file and compare it with the host.

### Bird data

Drop stills and clips into `data/incoming/`, then follow [docs/data_conventions.md](docs/data_conventions.md). It covers provenance, frame extraction, YOLO26n box proposals, human review, and the label checks. `notebooks/inspect_birds.ipynb` shows reviewed boxes and box sizes.

The [mobile annotation editor](docs/annotation_review.md) reviews local frames and saved Open Images picks with zoom, touch box editing, pixel adjustments, draft recovery, approval history, and reviewed-label export. Start it with `.venv/bin/python -m ml.annotation_review --host 0.0.0.0 --port 8768` and use the private startup link on the same Wi-Fi.

### Tensor G4

[docs/g4_smoke.md](docs/g4_smoke.md) has the compile container, the phone smoke app in `android/g4smoke`, and the rerun steps for when the Tensor ML SDK tarball is available.

Keep datasets, model artifacts, and local machine configuration outside version control. Record their provenance and settings so a run can be repeated.

## Repository contents

- `docs/android_bird_detection_mvp.md` — technical scope and experimental plan.
- `docs/implementation_plan.md` — concise build plan, basic app design, and ML workflow.
- `docs/implementation_plan.html` — phone-friendly, standalone reading version.
- `docs/gemini_extensions_review.md` — shared Gemini proposal, technical corrections, and prioritized extension recommendations.
- `docs/implementation_plan_review.md` — pre-implementation review of the plan, with decisions to lock before coding.
- `docs/phone_baseline.md` — Pixel 9 Pro parity, Mali GPU workarounds, and camera spot readings.
- `docs/g4_smoke.md` — Tensor G4 compile attempt and phone smoke test.
- `docs/data_conventions.md` — data layout, provenance, and the labeling workflow.
- `docs/third_party.md` — copied code, native packages, fixtures, and weights with their licenses.
- `ml/` — export, parity, data, labeling, and distillation scripts, with tests in `ml/tests/`.
- `android/app/` — ncnn camera, image, and clip app.
- `android/g4smoke/` — LiteRT Tensor G4 smoke app.
- `g4/` — linux/amd64 container and script for the Tensor G4 compiler.
- `notebooks/inspect_birds.ipynb` — reviewed-box inspection notebook.
- `configs/provenance.example.yaml` — template for `data/provenance/sources.yaml`.
- `LICENSE` — GNU Affero General Public License, version 3.
- `.gitignore` — Android, native build, Python/ML, local data, and secret exclusions.

## License

Original repository materials are licensed under **AGPL-3.0-only**. See [LICENSE](LICENSE) for the full terms. External code, datasets, images, and model weights retain their own licenses; their inclusion or use must be tracked separately.
