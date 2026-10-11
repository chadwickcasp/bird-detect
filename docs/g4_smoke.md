# Tensor G4 compile and phone smoke test

No YOLO26n layer has run on the Tensor G4 NPU yet. The Linux compiler never produced a G4 model. The on-device compiler, tried October 11, partitioned the whole graph. The phone then refused the app access to the Edge TPU.

The ncnn camera app stays the baseline. Its results are in [phone_baseline.md](phone_baseline.md).

## Compiler access

The Google Tensor compiler plugin is not publicly downloadable. The `ai-edge-litert-sdk-google-tensor` package fetches it from `redirector.gvt1.com/edgedl/tensor-ml-sdk/sdk/releases/<version>/litert_plugin_compiler.tar.gz`. That URL returned 404 for versions 5.0.0, 6.0.0, 6.1.0, 6.1.1, 6.2.0, 6.3.0, and 7.0.0.

Package 2.1.5 is the version that LiteRT issue 7598 confirmed for G4. It installs in one of two ways:

- `GOOGLE_TENSOR_SDK_BETA` points at a local copy of the access-controlled tarball.
- `SKIP_SDK_DOWNLOAD=1` installs the package with no compiler.

[g4/Dockerfile](../g4/Dockerfile) uses the tarball when the `SDK_TARBALL` build argument is set. Otherwise it skips the download.

## Container compile attempt

The image ran as linux/amd64 under emulation on the Mac. Emulation was not the blocker. [g4/compile_tensor_g4.py](../g4/compile_tensor_g4.py) compiled two models in order:

1. The MediaPipe selfie model, as a probe that is known to compile for G4.
2. `weights/yolo26n.tflite`, with `google_tensor_truncation_type=half` and `google_tensor_sharding_intensity=maximum`.

Both stopped at the `apply_plugin` step with `ValueError: apply_plugin failed to apply plugin`. The log first warns `SDK files directory not found at .../ai_edge_litert_sdk_google_tensor/data`. The missing compiler causes that warning.

The run left these files in `outputs/g4/`:

- `g4_compile.json` records the host (x86_64, Python 3.11), the LiteRT versions (2.1.5), the flags, and `"status": "failed"` for both models.
- `g4_compile_run.log` is the console output.
- `*_Tensor_G4_report.txt` holds each traceback.
- `*_aot/*_apply_plugin.tflite` files are zero bytes. The compiler created them before it failed. They are not compiled models.

## Phone smoke test

The `android/g4smoke` app ran on the Pixel 9 Pro (build CP3A.260905.009) with LiteRT 2.1.1. It bundles `libLiteRtDispatch_GoogleTensor.so` from the LiteRT 2.1.1 release.

The dispatch library loaded. Logcat reported `GoogleTensor Dispatch API version 0.1.0, SB API version 0.21.0, vendor id: sb_pixel`. LiteRT listed the accelerators as NPU, GPU, and CPU.

The app ran two passes over the three fixtures:

| Pass | Model | Requested accelerator | Run time per image |
| --- | --- | --- | --- |
| `litert-cpu` | float `yolo26n.tflite` | CPU | 106 to 107 ms |
| `litert-npu-request-uncompiled` | float `yolo26n.tflite` | NPU | 103 to 122 ms |

The NPU request is not an NPU result. The float model has no G4 bytecode, so LiteRT executed it on the CPU. Its boxes and scores match the CPU pass exactly, and its time is the same.

Both passes match the host LiteRT reference. The rule is the same one used in phone_baseline.md. Every host box needs a phone box at IoU 0.50 or higher, and the phone may have no extra boxes.

| Fixture | Matched | Min IoU | Max score difference |
| --- | --- | --- | --- |
| `bird_gull.jpg` | 1 | 0.992 | 0.010 |
| `bird_gull_rot90.jpg` | 1 | 0.993 | 0.038 |
| `empty_sky.jpg` | 0 boxes on both sides | n/a | n/a |

## LiteRT box units

The LiteRT export emits box rows as fractions of the 640-pixel input. The ncnn export emits them in input pixels. [ml/export_pretrained.py](../ml/export_pretrained.py) measures this on `bird_gull.jpg` and writes it to `output.box_units` in `outputs/exports/pretrained_yolo26n.json`. The measured ncnn-to-LiteRT ratio was 640.6.

`Yolo26Decoding.BOX_ROW_SCALE` multiplies the LiteRT box rows by 640 before decoding. Without it the first phone run produced boxes such as `[0.26, 0, 1.71, 0]`.

## On-device compiler

Recorded October 11, 2026. LiteRT publishes an arm64 compiler, `libLiteRtCompilerPlugin_google_tensor.so`, in the v2.1.6 JIT runtime zip. [android/g4smoke/fetch_compiler_plugin.sh](../android/g4smoke/fetch_compiler_plugin.sh) copies it next to the dispatch library. The smoke app points `CompilerPluginLibraryDir` at that directory and requests NPU for the float model. The pass label is `litert-npu-on-device-compile`.

The app runtime, the dispatch library, and the compiler plugin are all LiteRT 2.1.6. With the 2.1.1 dispatch library, the 2.1.6 runtime aborted inside `CompiledModel.create`. The matching 2.1.6 dispatch library turns that into a recorded exception.

The plugin loaded. It selected all 414 ops of YOLO26n into one partition, with `soc_model=Unspecified`, and logged `Starting GoogleTensor Compilation`. Compilation then failed:

```text
vendor.google.edgetpu_app_service: com.birddetect.g4smoke is not in the EdgeTPU allowed list or signature mismatched.
libedgetpu_client: Compile failed, error code 16, message: Current application should not be allowed to access EdgeTPU.
```

The vendor service keeps an allowlist of app signatures. This sideloaded app is not on it. No compiler flag changes that. `g4smoke.json` records the NPU pass as `failed` with `LiteRtException: Failed to compile model`.

The CPU pass still matches the host LiteRT reference. The minimum box overlap was 0.992. The empty-sky image had no boxes.

## Play Protect

During the first install, logcat showed Play Protect's on-device classifier giving the sideloaded `com.birddetect.g4smoke` package a `HARMFUL` verdict. The install still succeeded and the app ran. The cause was not investigated.

## Rerun once the tarball is available

Put `litert_plugin_compiler.tar.gz` in `g4/`. From the repository root:

```bash
docker build --platform linux/amd64 --build-arg SDK_TARBALL=litert_plugin_compiler.tar.gz -t bird-detect-g4 g4
docker run --rm --platform linux/amd64 -v "$PWD":/work bird-detect-g4
```

A successful compile writes a `yolo26n_*.tflite` file to `outputs/g4/` and lists it under `artifacts` in `g4_compile.json`. The staging script copies the first such file into the smoke app as `models/yolo26n_Tensor_G4.tflite`. Stage it and rerun the phone app. Connect over Wi-Fi with `android/connect_phone.sh` before the install. The README has the one-time pairing command.

```bash
.venv/bin/python -m ml.stage_android_model
(cd android && ./g4smoke/fetch_dispatch_library.sh && ./g4smoke/fetch_compiler_plugin.sh && ./gradlew :g4smoke:assembleRelease)
android/connect_phone.sh install android/g4smoke/build/outputs/apk/release/g4smoke-release.apk
android/connect_phone.sh adb shell am start -n com.birddetect.g4smoke/.SmokeActivity
android/connect_phone.sh adb pull /sdcard/Android/data/com.birddetect.g4smoke/files/ outputs/g4/phone/
.venv/bin/python -m ml.compare_predictions phone outputs/g4/phone/predictions_litert-npu.json --reference litert
```

With a compiled model, `g4smoke.json` sets `compiled_for_npu` to true and labels the pass `litert-npu`. An NPU result also needs a logcat line showing that the dispatch delegate took the graph. Matching CPU timing would mean the run fell back to the CPU.

The smoke app runtime and its dispatch library are LiteRT 2.1.6. A model from the 2.1.5 Linux compiler is an unverified pairing with that runtime. If the phone rejects the model, align the runtime and compiler versions first.
