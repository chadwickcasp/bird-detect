# Gemini extension proposal and repository review

Reviewed October 10, 2026. Status: recommendations only; no application, model, dataset, or benchmark was implemented by this review.

Model update later October 10: the [implementation plan](implementation_plan.md) now uses YOLO26n with a YOLO26s teacher. This document preserves the source proposal and earlier YOLOv8 review; use the revised plan for current model/export choices. The field-evaluation recommendations still apply.

Source: [shared Gemini Flash conversation](https://share.gemini.google/k2NQR1r2pKEC), titled “C-UAS Perception Challenges and Solutions,” created and published October 10, 2026. The shared page resolves to [this conversation](https://gemini.google.com/share/8686934a9399). The source proposal is preserved below as a normalized outline, followed by an independent evaluation. Formatting and equations have been made readable in Markdown; this is not a verbatim transcript.

Repository context: [concise implementation plan](implementation_plan.md) and [detailed MVP scope](android_bird_detection_mvp.md). Both describe proposed work. “Already planned” below never means implemented. The user confirmed that accuracy and distillation should remain the weekend priority. The review retains the existing Pixel 9 Pro, Android, and roughly 20-hour budget as its baseline.

## Recommendation

Adopt the field-evaluation ideas from Module 5, clarify the camera and timing requirements in Modules 2–3, and defer tracking. Keep YOLOv8n at 640×640 with ncnn CPU/Vulkan as the initial implementation. Preserve the existing gated Tensor G4 experiment; Gemini's generic NPU advice does not replace it.

The best additions are a defined bird-size recall report, a repeatable clutter-negative test, and one 20-minute endurance run on the selected configuration. They strengthen the existing experiment without requiring another model family, mobile platform, or tracking subsystem. Budget roughly 2–4 additional active hours if these cannot be folded into the planned data/evaluation work; this is an estimate, not a measured implementation cost.

## Source proposal: normalized outline

Gemini's proposal is titled **“C-UAS Edge Perception Prototype: Mobile Smartphone Feasibility & Performance Sprint.”** It frames a phone-based aerial-object detection prototype using birds as a proxy, with small-object recall, thermal behavior, and temporal stability as the main questions. It proposes a complete camera → inference → tracking → display loop for a weekend sprint.

### Suggested architecture and stack

```text
1080p camera stream at 60 FPS
  → preprocessing / RGB tensor
  → CoreML or TFLite accelerated YOLOv8n (FP16 / INT8)
  → bounding-box postprocessing, NMS, confidence threshold
  → lightweight 2D Kalman filter
  → overlay with FPS, latency, and bounding boxes
```

- Hardware: iPhone 12 or newer, or Android devices such as Snapdragon 8 Gen 1+ / Pixel 6+.
- Runtime: CoreML/Metal on iOS; TFLite/NNAPI/GPU delegate on Android.
- Model: YOLOv8n at 640×640, or NanoDet-m, restricted to a bird detection class.

### Module 1: Model optimization and export

Convert pretrained YOLOv8n to phone-friendly artifacts:

- A CoreML `.mlpackage` export with FP16 precision.
- A TFLite `.tflite` export using INT8 dynamic-range or full-integer quantization.
- A dummy-input benchmark for latency and memory footprint.

The source describes these as optimized for smartphone NPUs. That is an intended outcome, not verified accelerator compatibility.

### Module 2: Native frame capture

Use AVFoundation on iOS or CameraX on Android, with asynchronous processing off the main UI thread:

- Fixed focus at infinity, expressed in the source as `focalLength = max`.
- Fixed exposure time to reduce motion blur.
- Zero-copy YUV/NV12-to-RGB tensor conversion.
- Frame skipping when the NPU queue is full, preserving preview frame rate.

### Module 3: Inference and display

- Run inference on camera buffers.
- Apply NMS and map boxes from the 640×640 model space to the viewport.
- Display inference latency, end-to-end processing FPS, and memory usage.

### Module 4: Temporal smoothing and association

- Add tracking to reduce box jitter and suppress isolated false detections from clutter.
- Use a constant-velocity Kalman filter with state `(x, y, vx, vy)`.
- Require an unspecified `N` consecutive detections before confirming a track.

The source describes eliminating single-frame false positives; that is a proposed benefit requiring measurement, not a guarantee.

### Module 5: Field evaluation

- Measure confidence degradation as apparent bird size falls from 64×64 pixels to below 16×16 pixels.
- Run continuous inference for 20 minutes, recording thermal throttling onset, throughput decline, and battery drain.
- Pan across treelines, powerlines, and buildings and record false alarms per minute.

### Suggested weekend sequence

| Phase | Source milestone | Deliverables |
| --- | --- | --- |
| Saturday morning | Setup/export | Quantized native artifact; offline shape and postprocessing checks |
| Saturday afternoon | Integration | Background camera inference, boxes, telemetry |
| Sunday morning | Tracking | Kalman filter, validation on recorded motion clips |
| Sunday afternoon | Outdoor benchmark | Bird footage, thermal behavior, apparent-size limits |

The shared page also offers follow-up prompts for export code, a Kalman filter, and a field-test protocol. Those prompts are not additional completed deliverables.

## Fit against this repository

| Proposal | Existing plan | Decision and reason |
| --- | --- | --- |
| YOLOv8n, 640×640, bird boxes | Core A0/B0/T0/D0 comparison | **Keep.** Already aligned; retain the pretrained-versus-fine-tuned class mapping. |
| CoreML/iOS | Pixel 9 Pro Android only | **Skip this weekend.** Adds another app/runtime with no value to the selected-device comparison. |
| Replace ncnn with TFLite | ncnn baseline plus gated LiteRT/G4 experiment | **Do not replace.** Establish one working native path before adding runtime comparisons. |
| FP16/INT8 first | Float baseline; precision and PTQ experiments separately | **Defer quantization.** Export parity and actual backend execution must precede speed claims. |
| Dummy-input benchmarking | Device pipeline timing and held-out quality evaluation | **Use as a smoke test.** Useful for load/shape failures; insufficient as a performance result. |
| Async capture and latest-frame processing | Explicitly planned | **Keep and specify.** Bounded work, buffer ownership, timestamps, and observable drops matter. |
| CameraX rewrite | Reuse an existing native ncnn camera example where practical | **Conditional.** Use CameraX if it simplifies integration; do not make a camera rewrite a prerequisite. |
| Mandatory 1080p/60 | Actual capture configuration still unverified | **Make optional.** Start with a supported configuration; test higher rates after measuring the baseline. |
| Fixed focus/exposure | Capture settings recorded, rear 1× baseline | **Small optional experiment.** Correct the focus API and compare against automatic settings. |
| Zero-copy conversion | Reuse buffers and include transfer costs | **Defer specialized plumbing.** Profile a correct path first. |
| Boxes, NMS, coordinate mapping | Explicitly planned | **Keep.** Rotation, letterbox padding, and preview cropping also need validation. |
| Live latency/FPS/memory | Basic HUD and saved benchmark logs planned | **Keep, with clear metric names.** Saved measurements are the primary evidence. |
| Kalman tracking and confirmation | Tracking explicitly excluded | **Follow-up.** New state, tuning, annotations, and metrics compete with the ML experiment. |
| Small-object degradation | Small/blurred/in-flight slices already planned | **Add a precise protocol.** Measure missed birds, not just scores of detections. |
| 20-minute thermal run | Warm-up and paired five-minute trials planned | **Add one finalist run.** Retain paired trials for comparisons. |
| Clutter false alarms/minute | Negatives and FP/100 frames already planned | **Add controlled negative clips.** Define the denominator and distinguish repeated boxes from events. |

## Corrections before implementation

1. **Infinity focus is not maximum focal length.** Android Camera2 uses `LENS_FOCUS_DISTANCE = 0.0f` for infinity on supported hardware. Focal length describes a different optical property. Check manual-focus support and autofocus mode, then verify the actual capture result. Start with automatic focus/exposure; test manual settings only if footage reveals a problem. Shorter exposure can trade blur for noise or underexposure. [Camera2 request reference](https://developer.android.com/reference/android/hardware/camera2/CaptureRequest#LENS_FOCUS_DISTANCE).

2. **Dynamic-range and full-integer quantization are different experiments.** Dynamic-range conversion quantizes weights and may dynamically quantize activations for supported operators. Full-integer conversion needs representative calibration data and compatible operators/I/O. Neither an INT8 filename nor a successful export proves Tensor G4 TPU execution. Establish a float export first; use the repository's separate deployment gate for accelerator claims. [LiteRT quantization guide](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization).

3. **NNAPI is not the default new acceleration route.** Android documents its deprecation in Android 15. Keep the repo's ncnn CPU/Vulkan baseline and separate LiteRT G4 investigation. This review has not reproduced the G4 recipe or verified libraries on the phone. [Android NNAPI documentation](https://developer.android.com/ndk/guides/neuralnetworks).

4. **Latest-frame processing is independent of an NPU.** If using CameraX, `STRATEGY_KEEP_ONLY_LATEST` prevents analysis backlog; release each `ImageProxy` after its data is no longer needed. Do not add another unbounded queue after the analyzer. CameraX can convert YUV to RGBA, but that is not evidence of a zero-copy path into ncnn. Profile conversion and transfer costs before adding specialized buffer sharing. [CameraX image analysis](https://developer.android.com/media/camera/camerax/analyze).

5. **Camera FPS, processed FPS, and result age answer different questions.** A 60 FPS preview does not mean 60 inferences per second. Save preprocessing, inference, postprocessing, and result-ready latency separately. Measure capture-to-result age only with compatible timestamp clocks. Do not label a CPU/backend change as quantization speedup or a throughput dip as proven thermal throttling without supporting observations.

6. **A Kalman filter does not perform data association by itself.** Track confirmation can suppress brief true sightings as well as false detections. Persistent clutter can survive confirmation. A follow-up would need matching, initialization, expiry, elapsed-time handling for dropped frames, and separate raw/filtered metrics. Smoother display boxes do not establish improved detector AP.

7. **Bird evidence supports bird-detection claims.** Similar apparent scale or motion can motivate other research, but this dataset cannot establish drone detection accuracy, range, or operational counter-UAS performance. Describe the deliverable as a mobile bird-detection and edge-performance study.

## Additions worth making concrete

### Priority 1: Recall by apparent bird size

Extend the existing evaluator instead of building another training pipeline. Proposed location: `ml/evaluate.py`, with a named `summarize_size_slices` function; keep slice settings in `configs/evaluation.yaml`. These names are proposals, not existing files.

Record each ground-truth box's width/height in original-frame coordinates and after the actual resize into the 640×640 tensor. Use model-space equivalent side length `s = sqrt(width × height)` and predeclare bins `<16`, `16–<32`, `32–<64`, and `>=64` pixels. Also retain width/height because narrow birds can have the same area as square boxes. These are project-specific bins, not a claim to use COCO's standard size categories.

For example, letterboxing a 1920×1080 image into 640×640 uses a scale of 1/3: a 48×48 source box becomes 16×16 model pixels. A source-pixel cutoff without preprocessing context is ambiguous.

Report recall at IoU 0.50 and a validation-selected score threshold, plus bird-instance, frame, and session counts in each bin. Count false negatives explicitly; retain global precision/AP and existing slice reports. Empty bins are unavailable results. Low-count bins are exploratory. Synthetic resizing can supplement debugging, but does not reproduce distant optics or real motion blur. Do not promise a universal minimum detectable size or detection range.

**Done when:** A0 and B0 have comparable size-slice tables and representative misses; include T0/D0 if trained. Size bins and thresholds are frozen before final testing.

### Priority 2: Clutter-negative evaluation

Add verified no-bird clips containing sky/clouds, foliage/treelines, powerlines, and buildings to the planned capture sessions. Include both a stationary view and a slow pan when practical. Keep recording sessions together in the existing train/validation/test split scheme.

Use the existing frame evaluator for FP/100 evaluated frames at the fixed operating threshold. For real-time replay, also report false detection boxes per minute of source footage, evaluated frame count, processed FPS, and dropped frames. A repeated false box in ten frames contributes ten box detections; call it a box rate, not ten distinct false-alarm events. An event metric would need a separate grouping rule. Compare frame-level quality on identical sampled frames so a slower backend cannot appear more accurate simply by skipping difficult frames.

**Done when:** each evaluated model has a negative-clip table with durations/denominators and examples of recurring mistakes. This fits the existing failure-analysis work without requiring tracking.

### Priority 3: Sustained endurance run

After the paired five-minute benchmarks select a candidate, run that configuration continuously for 20 minutes with camera and display enabled. Save approximately one-second summaries: processed FPS, pipeline latency percentiles with sample counts, frame drops, memory, available thermal status, and battery level. Preserve the per-frame timings behind aggregates. Note brightness, charging state, starting battery/thermal state, camera mode, ambient conditions, and runtime/precision.

Plot throughput and latency against elapsed time and compare early/late windows. Battery percentage change is only a coarse whole-device observation; it does not isolate model power or establish energy per inference. If slowdown coincides with a thermal status change, report that association; do not infer a specific temperature or throttling mechanism from FPS alone.

**Done when:** one trace and a short interpretation document sustained behavior, including stability if no slowdown occurs. Twenty minutes is an observation window, not proof of all-day endurance. Repeat or extend later if the deployment question requires it.

### Priority 4: Capture controls, only if needed

Record the selected camera/lens, actual frame size/rate, exposure, focus behavior, and rotation before changing settings. If focus hunting or blur dominates the validation footage, run a small automatic-versus-manual comparison on separate development clips. Keep test capture conditions fixed. Do not add 60 FPS, a telephoto switch, manual exposure, and model changes to the same comparison.

## Weekend sequencing and cut line

The source's Sunday tracking slot should remain available for the repository's fine-tuning/distillation and evaluation work. It otherwise displaces a central project goal. Retain the existing 20-hour order:

| Existing slot | Recommended treatment of the extensions |
| --- | --- |
| Hours 0–3: device/export gate | Verify the CPU camera path, custom export parity, and timing. Add bounded frame handling; no mandatory INT8 or 60 FPS. |
| Hours 3–7: data/splits | Capture clutter negatives and a range of bird sizes as part of the existing sessions. Freeze slice definitions and session splits. |
| Hours 7–12: A0/B0/T0/D0 | Preserve the matched ML comparisons. Distill only if teacher quality and a working integration justify it. |
| Hours 12–15: artifact integration | Check deployed predictions, coordinate transforms, and actual backend. |
| Hours 15–17: measurement | Retain paired five-minute trials; add one 20-minute finalist run if instrumentation is ready. |
| Hours 17–20: report/demo | Include size recall, negative-clip errors, endurance trace, and limitations in the existing report. |

These additions fit only if capture and telemetry are folded into work already planned. Otherwise allow roughly 2–4 extra active hours or reduce extension scope. Cut optional capture controls first, then extra configurations/endurance repeats. Do not cut held-out evaluation or export parity to make room for tracking. G4 bring-up remains separately gated and may already exceed the original budget.

If native integration fails the existing 2–3-hour gate, reduce scope to a working baseline and measured A0/B0 comparison before adding any extension. A smaller completed experiment is preferable to unmeasured claims across many backends.

## Follow-ups after the measured baseline

- Temporal tracking, if visible jitter or intermittent detections are a demonstrated user problem. Evaluate confirmation delay, lost tracks, false persistence, and raw versus displayed results on labeled sequences.
- INT8 PTQ, if a verified backend offers a useful integer path and quality remains acceptable.
- Specialized zero-copy preprocessing, if profiling identifies copying/conversion as a major bottleneck.
- A different input resolution or model family, as a separate accuracy/latency experiment after the fixed-640 comparison.
- iOS/CoreML only when a second target device becomes an explicit project requirement.

The source proposal is useful primarily as a field-testing checklist. The repository already has the stronger experimental structure: session-based splits, ordinary fine-tuning controls, a teacher-quality gate, deployed-artifact evaluation, and paired backend measurements. Preserve that structure while adding clearer measurements of where the bird detector fails.
