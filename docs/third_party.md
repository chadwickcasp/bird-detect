# Third-party code and assets

This file lists external code and media that the repository copies or downloads. Each item keeps its own license.

## ncnn Android camera example

The Android camera code starts from [nihui/ncnn-android-yolov8](https://github.com/nihui/ncnn-android-yolov8) at commit `f1ac75ec54ccb3817a9eba620fe51da8bdcf87ca` (May 27, 2026). The repository has no LICENSE file. Its source files carry the Tencent BSD 3-Clause notice, and the copied files keep that notice.

Copied files:

- `android/app/src/main/jni/ndkcamera.h` and `ndkcamera.cpp`
- `android/gradlew`, `android/gradlew.bat`, and `android/gradle/wrapper/` (Gradle 8.12)

Changes from the example:

- `ndkcamera.cpp` requests 1920x1080 frames instead of 640x480. The capture size and the sensor looper id sit in one file-scope block after the includes.
- `ndkcamera.cpp` skips a frame when no preview window is attached yet. The example crashed on the Pixel 9 Pro when a frame arrived before the surface.
- `android/app/src/main/jni/bird_detect_jni.cpp` is adapted from the example's `yolov8ncnn.cpp`. It keeps the camera bridge and replaces the model switch with one YOLO26 bird detector, a bitmap entry point, and a stats readout.
- The segmentation, pose, classification, and oriented-box tasks are dropped.
- The YOLOv8 distribution-focal decoder is replaced by `yolo26_det.cpp`. YOLO26 outputs decoded boxes, so the 16-bin decoding does not apply.
- The Turnip Vulkan driver loader is omitted. The Pixel 9 Pro GPU is a Mali-G715, not an Adreno.

## Prebuilt native libraries

`android/fetch_native_deps.sh` downloads these packages into `android/app/src/main/jni/`. They are gitignored.

| Package | Source | License |
| --- | --- | --- |
| ncnn 20260526 Android Vulkan | [Tencent/ncnn release](https://github.com/Tencent/ncnn/releases/tag/20260526) | BSD 3-Clause |
| opencv-mobile 4.13.0 Android | [nihui/opencv-mobile v35](https://github.com/nihui/opencv-mobile/releases/tag/v35) | Apache 2.0 |

## Parity fixtures

The images in `ml/fixtures/` keep their Wikimedia Commons licenses. [`ml/fixtures/README.md`](../ml/fixtures/README.md) records each author and license.

## Models

The COCO-pretrained `yolo26n.pt` and `yolo26s.pt` checkpoints come from Ultralytics under AGPL-3.0. Exports of those checkpoints stay in `weights/`, which is gitignored.
