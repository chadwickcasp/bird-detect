# Phone baseline: pretrained YOLO26n on the Pixel 9 Pro

Recorded October 10, 2026. This is the hours 0–3 gate from the implementation plan. It shows that the ncnn export runs on the phone and matches the host boxes. It is not the sustained benchmark.

## Device and build

- Phone: Pixel 9 Pro (`caiman`), Android 17, build `CP3A.260905.009`.
- GPU: Mali-G715, reported by ncnn with FP16 and cooperative-matrix support.
- Camera: rear camera id `0`, sensor orientation 90, capture 1920x1080.
- App: release build signed with the debug key, arm64-v8a only.
- Runtime: ncnn 20260526 Vulkan, opencv-mobile 4.13.0, NDK 29.0.14206865, CMake 3.31.5.
- Model: `yolo26n.pt` exported by Ultralytics 8.4.175. The tensor record is `outputs/exports/pretrained_yolo26n.json`.

## Fixture parity

All four backends pass. The reference is the Ultralytics ncnn runtime on the Mac. A pass means every host box has a phone box at IoU 0.50 or higher, and the phone has no extra boxes.

| Backend that ran | Gull IoU | Rotated gull IoU | Empty sky | Largest score difference |
| --- | --- | --- | --- | --- |
| `ncnn-cpu-fp32` | 0.990 | 0.991 | no boxes | 0.029 |
| `ncnn-cpu-fp16` | 0.994 | 0.992 | no boxes | 0.027 |
| `ncnn-gpu-fp32` | 0.990 | 0.991 | no boxes | 0.029 |
| `ncnn-gpu-fp16` | 0.989 | 0.991 | no boxes | 0.032 |

The phone files are `outputs/parity/phone_ncnn-*.json`. Each one has a matching `_parity.json` report.

## Two Mali GPU faults

The first GPU run failed parity. Two separate ncnn options caused the failures. `BirdDetector::load` now turns both off whenever the GPU is used.

Light mode corrupted every frame after the first. The score rows stayed correct. The box rows were the correct boxes times one constant per frame, such as 0.316 or 5.93. Raw tensor dumps showed this pattern. With `lightmode = false`, the GPU FP32 boxes match the CPU boxes. The same model on the Mac's Vulkan driver stays correct with light mode on.

Cooperative-matrix FP16 saturated the output. Every score came out as 1.0, and about 100 empty boxes passed the threshold on each fixture. With `use_cooperative_matrix = false`, GPU FP16 passes. The Mac GPU reports no cooperative-matrix support, so it never takes that path.

Both faults are recorded for this phone, driver, and ncnn release only. Recheck them when any of the three changes.

## Camera spot readings

These readings come from the app's status line while the camera pointed at a desk. The phone was charging, and the runs were short. Treat the numbers as a sanity check, not as the sustained benchmark.

| Backend that ran | Processed frames per second | Inference time |
| --- | --- | --- |
| `ncnn-cpu-fp32` | about 11 | 69–77 ms |
| `ncnn-cpu-fp16` | 16–19 | 31–58 ms |
| `ncnn-gpu-fp32` | about 7 | 95–133 ms |
| `ncnn-gpu-fp16` | 10–11 | 59–80 ms |

GPU FP16 is not faster than CPU FP16 in these readings. The camera preview and the GPU share the same SoC, so the planned sustained run must settle this.

## Still and clip sources

A JPEG with EXIF orientation 6 was decoded upright. The detector found the gull at score 0.83, and the box sat on the bird. A clip with a 90° rotation tag played upright, and its overlay followed the bird. GPU FP16 processed about 14 frames per second on that clip.

Screenshots are in `outputs/phone/`.

## How to repeat

Pair the phone once. Connect over Wi-Fi with `android/connect_phone.sh`. The README has the pairing command.

```bash
.venv/bin/python -m ml.stage_android_model
android/fetch_native_deps.sh
(cd android && ./gradlew assembleRelease)
android/connect_phone.sh install
android/connect_phone.sh adb shell am start -n com.birddetect/.MainActivity --ei source 1 --ei backend 3 --ez run_fixtures true
android/connect_phone.sh adb pull /sdcard/Android/data/com.birddetect/files/predictions_ncnn-gpu-fp16.json outputs/parity/phone_ncnn-gpu-fp16.json
.venv/bin/python -m ml.compare_predictions phone outputs/parity/phone_ncnn-gpu-fp16.json
```

The `backend` extra uses the spinner order: 0 is CPU FP32, 1 is CPU FP16, 2 is GPU FP32, and 3 is GPU FP16. The `source` extra uses 0 for Camera, 1 for Image, and 2 for Clip.
