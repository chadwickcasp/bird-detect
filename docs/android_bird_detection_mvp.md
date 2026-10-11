# Android bird detection: 20-hour MVP scope

Prepared October 7, 2026. Status: scoped; no app, dataset, trained model, or measured result exists yet.

## Purpose

Demonstrate the complete engineering loop: define a detection task, establish a credible pretrained baseline, adapt it with phone footage, distill a larger teacher into a smaller student, export the student, run native inference on a physical Android phone, and measure the accuracy/latency tradeoff on held-out data.

The recommended subject is birds, with a single `bird` detection class and an airborne-bird evaluation slice. The planned device is a Google Pixel 9 Pro; physical device connection and capture configuration have not been verified. The alternative subject is a tossed tennis ball if reliable bird footage cannot be collected quickly.

The scope is approximately 20 hours of active work over one or two days. The distillation schedule assumes access to suitable training compute; its availability has not been checked. Training duration and achievable phone performance remain unknown.

## Task and boundaries

Input: RGB video from the Android rear camera or a locally stored clip.

Output: bird bounding boxes and confidences, with a baseline/adapted-model switch and basic timing display.

Evaluation: replay the same labeled frames through the deployed model; use live video to demonstrate operation and collect sustained performance measurements.

Include visible birds of different apparent sizes, backgrounds, and motion/blur conditions. Fast motion is represented by image displacement and blur; physical object speed alone does not define image difficulty. Start with birds a human can reliably annotate. Include no-bird frames and challenging backgrounds.

Version one excludes species recognition, multi-object tracking, segmentation, a backend, cloud inference, accounts, custom UI design, and custom model architecture research. Use an existing distillation implementation.

## Initial model/runtime choice

Revised October 10, 2026: use COCO-pretrained `yolo26n.pt` as the phone student and `yolo26s.pt` as the initial teacher. YOLO26 supplies mobile export support, a lighter DFL-free detection head, and a documented same-family distillation path. Measure bird accuracy and physical-phone performance before claiming an advantage. The [concise implementation plan](implementation_plan.md) includes the workflow diagram. [YOLO26 documentation](https://docs.ultralytics.com/models/yolo26/).

Train/fine-tune in Python. Export a model artifact. Use ncnn's C++ runtime on Android, with a thin native-app wrapper. Establish the CPU baseline, then make CPU-versus-Vulkan-GPU benchmarking a core experiment when device/model checks pass. Do not assume the GPU wins for a small model.

The compression experiment is a fine-tuned YOLO26s teacher to YOLO26n student, holding input size at 640. Ultralytics documents this pair and built-in same-family feature distillation through `distill_model`. Confirm this in the pinned package with a tiny one-class training/export smoke run before committing the training budget. The feature guidance is used during training; inference exports contain only the student. The teacher must first outperform the ordinary student on validation data. Larger YOLO26 teachers are follow-ups; cross-family distillation requires a separate implementation. [Distillation guide](https://docs.ultralytics.com/guides/knowledge-distillation).

TensorRT is an NVIDIA GPU runtime and cannot execute on the Pixel's Tensor G4. QAT is a training technique that simulates quantization; it still needs an export path matching the deployment runtime's integer operators and quantization rules. Training on an NVIDIA GPU does not require deploying with TensorRT.

LiteRT supplies the direct G4 AOT/TPU experiment and a paired CPU/GPU reference; INT8 remains optional. Its documented Interpreter GPU delegate executes floating-point operations even for quantized models, so INT8 file size does not establish integer GPU acceleration. The top-level Tensor SDK page lists G5/G6, but the compiler source exposes G4 and Google maintainers provide a G4 compilation/runtime recipe. Use that direct route first and validate the particular detector and physical phone. These details were rechecked October 9, 2026.

Pin the training package and the Android example/runtime revisions once an export-compatible combination is verified. Do not silently use the example's bundled weights as one baseline and a different upstream checkpoint as another.

Important integration gate: reuse the YOLOv8 Android example's camera, native bridge, and Vulkan controls, but replace its detector decoding for the standard YOLO26 export. Its raw 16-bin distribution-focal decoding and manual graph conversion edits are not the YOLO26 contract; YOLO26 removes DFL. Verify tensor names/shapes, box decoding, score semantics, class count, channel order, normalization, square letterboxing, and coordinate mapping on fixed images before training. Record this contract in artifact metadata. Start with the default one-to-many head and external CPU NMS for all core comparisons. The NMS-free head (`nms=False`) is an optional separate experiment; verify export/runtime support and remeasure accuracy before adopting it.

The pretrained model's COCO bird class is index 14; a single-class fine-tune normally uses index 0. Convert both to the same evaluation class ID. A single-class head also changes parameter count and postprocessing work slightly. Record that change; do not attribute a resulting speed difference solely to improved learned weights. For a strict weights-only ablation, preserve the class-head layout.

## Hardware acceleration experiment

Run this comparison independently of training ablations, initially using A0 and then the selected student. Hold checkpoint, input size, data, preprocessing, decoder, confidence/NMS thresholds, and display behavior constant. Use a fixed backend for each A0/B0/T0/D0 training comparison.

| ID | Runtime/backend | Goal |
| --- | --- | --- |
| H0 | ncnn CPU, explicit floating-point settings | Reference |
| H1 | ncnn Vulkan GPU, matched precision where supported | Isolate the backend effect |
| H2 | ncnn Vulkan GPU with FP16 storage/arithmetic where supported | Measure an additional precision change |
| L0/L1 (extension) | LiteRT CPU versus GPU, same compatible export | Compare backends within a second runtime |
| N0 (G4 experiment) | LiteRT AOT artifact targeting Tensor_G4, executed through Google Tensor TPU dispatch | Validate compilation, on-phone execution, quality, and actual device cost |

### Vulkan GPU implementation

- Build ncnn with Vulkan enabled and enumerate its available GPU devices before offering GPU selection.
- Configure the backend before loading the model. Recreate the model when switching backend and keep inference off the UI thread.
- Set and record precision options explicitly. If H0/H1 cannot use comparable precision, identify the comparison as backend plus precision; keep H2 separate.
- Verify predictions on the same fixed images, then evaluate each exported/backend configuration. GPU support does not prove every model layer runs there; record unsupported-layer CPU work.
- Include input/output transfers and execution completion in timing. Reuse buffers where possible and distinguish model execution from preprocessing/postprocessing and camera/display overhead.
- Show the actual backend in the app. Preserve a usable CPU fallback, but log it visibly in benchmark records rather than reporting a requested GPU run as hardware execution.

The [ncnn Vulkan guidance](https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/FAQ-ncnn-vulkan.md) documents Android support, CPU fallback for unsupported layers, and the possibility of slower GPU execution. Small-model overhead and shared camera/rendering work must be measured on this phone.

### Direct Tensor G4 AOT/TPU experiment

Target the Pixel 9 Pro's actual SoC with `SocModel.TENSOR_G4` (serialized target `Tensor_G4`), rather than relying on a generic Tensor or G5 artifact. [LiteRT's Google Tensor target source](https://github.com/google-ai-edge/LiteRT/blob/main/litert/python/aot/vendors/google_tensor/target.py) includes this target. The top-level SDK support list is narrower; that alone does not rule out an experimental G4 path.

In [LiteRT issue #7598](https://github.com/google-ai-edge/LiteRT/issues/7598), a Google collaborator supplied G4-specific flags for a YOLO11n compilation failure. The reporter subsequently confirmed successful detector compilation. A Google engineer also confirmed a compatible G4 runtime/library and public-OTA setup. These establish a concrete starting route; YOLO26 compilation, execution, and performance on our Pixel 9 Pro remain unverified.

Planned compilation, after obtaining and pinning the compiler package in a Linux x86_64 environment:

```python
from ai_edge_litert.aot import aot_compile
from ai_edge_litert.aot.vendors.google_tensor import google_tensor_backend
from ai_edge_litert.aot.vendors.google_tensor import target

compiled = aot_compile.aot_compile(
    "bird_detector.tflite",
    target=[target.Target(target.SocModel.TENSOR_G4)],
    keep_going=False,
    google_tensor_truncation_type="half",
    google_tensor_sharding_intensity="maximum",
)
print(compiled.compilation_report())
```

The explicit backend import registers Google Tensor in the reported package version. Check the pinned package's import/API behavior rather than assuming current source and older wheels are identical. Keep the original float export as a CPU/GPU reference. The `half` flag changes floating-point precision, so evaluate prediction parity and AP again; this comparison includes compilation/precision changes. The [compiler flag reference](https://developers.google.com/edge/tensor-sdk/compilation-flags) documents the truncation and sharding controls. INT8 is not a prerequisite established by this floating-point recipe.

Deployment steps:

1. After the CPU baseline works, compile and run a known-good vision model to separate environment/runtime problems from detector-graph problems. Then compile a pretrained YOLO26n one-to-many `.tflite` export and inspect the compilation report for actual offload coverage and failed partitions. Run this smoke test before full training where practical, then repeat for the selected trained student. It does not block the core accuracy/distillation comparison.
2. Verify compiler/SDK access and authentic library availability. The engineer-confirmed compatibility recipe uses compiler 2.1.5 with Android LiteRT 2.1.1 and the [v2.1.1 NPU runtime library archive](https://github.com/google-ai-edge/LiteRT/releases/tag/v2.1.1). Treat this as a historical starting combination to reproduce, not an assertion that arbitrary versions interoperate.
3. Bundle the arm64 Google Tensor dispatch library (`libLiteRtDispatch_GoogleTensor.so`) and configure LiteRT's dispatch-library directory and NPU execution. For a Pixel-only app, the engineer confirms direct packaging in `jniLibs` is acceptable. Verify and record the Pixel 9 Pro's Android build and runtime prerequisites; do not copy a Pixel 9a build identifier as a requirement for a different phone.
4. Load the G4-compiled artifact on the physical phone. Use NPU-only execution for the initial smoke test where supported, then inspect runtime partition/accelerator logs. Successful AOT compilation is not proof of device execution or speedup. Report CPU/GPU fallback and any partial delegation explicitly.
5. Keep box decoding/NMS on CPU initially and include that cost in pipeline timing. Compare against LiteRT CPU/GPU references as well as the original ncnn system, identifying runtime and precision changes. Add the app's NPU selector only after this gate passes.

If compiler access, the system prerequisite, model coverage, or runtime loading blocks progress, save the exact failure and retain the working CPU/GPU paths. NNAPI is a secondary legacy investigation, not the main G4 route. NPU bring-up may extend the original work budget.

### Benchmark and decision

Use release builds, warm-up, approximately five-minute paired replay trials, and comparable starting thermal conditions; alternate execution order. Compare AP50/AP50-95, p50/p95 pipeline latency, processed FPS, dropped frames, memory, startup/compilation time, and early/late throughput. Record thermal status; add energy per processed frame only when instrumentation can measure it. Recheck with live camera/display enabled.

Plot accuracy against p95 latency with runtime/backend/precision labels, and throughput against elapsed time. Select the configuration with acceptable accuracy and the best sustained pipeline performance. Training gains, runtime changes, and precision changes must remain distinguishable. GPU checks reuse the integration/benchmark slots below; a second runtime and NPU investigation may extend the 20-hour budget.

## Data plan

Preferred data mixture: roughly 500-1,000 selected annotated public training frames plus 200-400 manually checked phone frames from 6-10 short clips, including negative frames. These are budget targets, not a statistical guarantee. Avoid spending the annotation budget on nearly identical neighboring frames. Keep enough separate phone sessions for validation and final test; favor capture diversity over simply increasing the frame count.

Start with the rear 1x camera, SDR and a fixed target frame rate, and record the actual frame dimensions, lens/zoom, exposure behavior, and capture path. Prefer saving frames from the same camera analysis stream used for inference. Stock-camera video is an acceptable bootstrap but should be checked against app frames. Camera-model matching alone does not match lens, object scale, motion blur, image processing, codec, backgrounds, or live-stream behavior. The two verified Pixel 8 Pro photos below use the telephoto lens, which does not establish a match to a 1x Pixel 9 stream.

Reserve complete recording sessions/clips for training, validation, and final test. Prefer different backgrounds and a separate capture session for the test set. Never randomly scatter adjacent video frames across these splits. All crops or augmented versions of a frame inherit its original split.

Use assisted labeling only with manual review. Include missed birds, not just the boxes proposed by the baseline. Label every in-scope visible bird in an evaluation frame. Record visibility/annotation rules so ambiguous tiny specks are treated consistently.

Keep the final test set untouched during model, threshold, and resolution selection. The dataset is a small local-domain pilot; report the number of clips, sessions, images, and instances and avoid broad generalization claims.

Public source shortlist, checked October 7, 2026:

| Source | What is verified | Use in this project |
| --- | --- | --- |
| [FBD-SV-2024 author data page](https://figshare.com/s/1ca0193680f894a65371) and [processing repository](https://github.com/Ziwei89/FBD-SV-2024_github) | 483 clips, 28,694 frames, annotated flying birds; 400 training clips and 83 test clips; archive listed as 1.67 GB. The paper identifies a Hikvision DS-2DE7232IW-A camera at 1280x720, 25 fps. | Best task match found. Select a small subset from the training portion; preserve clip IDs and annotation/frame alignment. It is surveillance footage, with a different capture domain from the phone. |
| [Red Bird](https://commons.wikimedia.org/wiki/File:Red_Bird_(53391757289).jpg) | Google Pixel 8 Pro EXIF, 18 mm f/2.8 lens; individual photo under CC BY 2.0. | Small supplementary still-image example; cropped photo, no boxes or temporal information. |
| [Oriental Magpie-Robin](https://commons.wikimedia.org/wiki/File:Spotlight_on_the_Oriental_Magpie-Robin-_Bangladesh%27s_Feathered_Jewel.jpg) | Google Pixel 8 Pro EXIF, 18 mm f/2.8 lens; CC BY 4.0; metadata lists Lightroom editing. | Supplementary still only; requires manual annotation. |

No ready-made labeled Pixel 9 Pro flying-bird dataset was verified. Searching Commons' device category and public repositories did not establish one; this is a search result, not a claim of nonexistence. The closest verified Pixel examples are individual Pixel 8 Pro still photos, not a sufficient fine-tuning collection.

FBD's Figshare metadata says CC BY 4.0, while the description separately limits use to learning/research and excludes commercial use. Keep its use within that stated research scope and preserve source attribution; do not resolve the conflicting metadata into a claim of unrestricted commercial permission. Article reuse terms are separate from dataset terms. Keep a source/author/license record for every public image or clip used.

The archive has not been downloaded and labels have not been inspected locally. Validate extraction indices, box bounds, difficult/visibility rules, and official split files before ingestion. Public access through an author-provided share link is verified; download completion is not.

Secondary fallback: MVA2023 SOD4SB supplies annotated small-bird images, but the training archive is about 9.4 GB. Verify image-use terms separately from the repository's code license. Do not use challenge test data for training.

## Experiment matrix

| ID | Model | Input | Purpose |
| --- | --- | --- | --- |
| A0 | Pretrained YOLO26n, bird detections | 640 | Credible deployed baseline |
| B0 | Ordinary fine-tuned YOLO26n | 640 | Student control trained on the chosen public + phone mixture |
| T0 | Fine-tuned YOLO26s teacher | 640 | Establish the larger model's accuracy/cost and validate teacher quality |
| D0 | YOLO26n trained with T0 guidance | 640 | Improve student accuracy without enlarging its inference architecture |
| Q0 (stretch) | Best student, INT8 PTQ | 640 | Test storage and actual integer-backend speed versus its float equivalent |

The core A0/B0/T0/D0 comparison uses the same phone, runtime/backend, precision, capture settings, preprocessing, and comparable postprocessing settings. B0 and D0 use the same student initialization, labeled data, epoch budget, optimizer recipe, and augmentation settings; only the distillation guidance changes. Record that D0 uses more training compute. Select checkpoints using validation data. If time permits, repeat the student comparison with another seed; otherwise identify single-run uncertainty.

Distillation does not shrink B0 further: D0 has the same inference architecture as B0. The intended result is better quality at B0's cost and a smaller/faster model than T0. Both comparisons are needed to attribute the result correctly. Gains on this small dataset are not assured.

If a public-only versus phone-adapted claim matters, add a public-only nano control with matched training budget. A0 versus B0 alone shows the combined effect of task-specific fine-tuning, data choice, and any class-head change; it does not isolate the contribution of phone data.

For Q0, calibrate using representative training/validation phone frames, including negatives and difficult lighting; never use final test data. First prove that the deployed artifact executes useful integer kernels and benchmark it against the same model in float on that runtime/backend. Switching from ncnn GPU to LiteRT CPU combines runtime/backend and precision effects and must be reported as a system comparison.

Add QAT only if PTQ gives useful device acceleration but loses unacceptable accuracy, and a compatible YOLO training-to-integer-export path passes a short smoke run. Compare PTQ and QAT at the same architecture/precision/backend; QAT's role is recovering quantization loss, not removing parameters. Generic TensorFlow/Keras QAT documentation is not a turnkey recipe for a PyTorch YOLO detector. Quantization is optional in the 20-hour scope; custom QAT plumbing, structured pruning, and tracking are follow-ups.

Hypotheses: domain-specific fine-tuning improves detection over A0; teacher guidance improves D0 over B0 at comparable deployed cost; D0 retains much of T0's quality with lower device cost. These are hypotheses to test, not promised outcomes.

## Evaluation protocol

### Detection quality

- Primary: single-class AP50 on the frozen test set, computed from saved predictions in original-image coordinates.
- Companion: AP50-95, precision/recall at a validation-selected confidence threshold, and false positives per 100 evaluated frames, including negatives.
- Slice: small birds and blurred/in-flight birds, using rules fixed before inspecting final results. Report slice counts.
- Compare the Python reference and exported/native model on identical images. Score deployed predictions, not only the training checkpoint.
- Save per-frame boxes, scores, class mapping, original frame dimensions, frame IDs, and checkpoint/export identifiers. Do not filter evaluation frames according to whether a model detected something.

For timing, the operating confidence/NMS thresholds may differ from the low score floor used to construct an AP curve. Record both. Tune operating thresholds on validation data and report the realized test precision/recall and false-positive rate without retuning on test.

### Runtime

- Use a release build on the physical phone. Record phone model, Android version, backend, precision, threads, input size, checkpoint hash, and model load time.
- Separately time preprocessing, model execution, and postprocessing. Record result-ready latency, processed FPS, and dropped frames.
- Record deployed model bytes, parameter count, and observed app memory. INT8 weight storage, total artifact size, activation memory, and speed are different measurements.
- If measuring capture-to-result age, use comparable monotonic timestamps and state exactly which capture stages are included. A fast model can still deliver stale results if frames queue.
- Warm up, then run approximately five-minute sustained trials. Report p50 and p95 latency and early-versus-late throughput. Repeat paired A/B trials when possible, alternating order and starting from comparable thermal conditions.
- Use the same replay data for controlled timing comparisons, plus a live-camera run for demonstration. Ensure GPU timing includes execution completion.
- Keep display/rendering overhead separate or include it consistently. The existing demo's FPS overlay is not a complete benchmark.

### Provisional success criteria

Choose and record the target before model selection. Example targets, not measured results:

1. B0 improves test AP50 by at least 5 percentage points while p95 pipeline latency increases by no more than 10%; or
2. D0 improves AP50 by at least 2 points over B0 while p95 latency stays within 10%; and/or
3. D0 reduces p95 latency by at least 25% relative to T0 while staying within 3 AP50 points.

An initial 15 processed FPS goal is a planning target until the physical phone baseline is measured. Sustained throughput and latency must both be reported. If an experiment fails the target, document the failure and explain the tradeoff; do not change the test set or redefine success after seeing it. With a small test set, modest AP changes may reflect sampling or training noise; report clip counts and avoid claiming statistical significance from one run.

## Active-work budget

| Hours | Deliverable |
| --- | --- |
| 0-3 | YOLO26n CPU camera baseline running on phone; adapt the standard-export decoder, check Vulkan GPU support and parity, and smoke-test the pinned distillation API |
| 3-7 | Select public frames; collect/label phone footage; freeze session-level splits |
| 7-12 | Establish A0 accuracy; train B0 and T0, then D0 if T0 is a useful teacher; training may overlap data/integration work |
| 12-15 | Integrate student/teacher artifacts; verify prediction parity and exported-model quality |
| 15-17 | Sustained paired model and CPU/GPU benchmarks; optional PTQ only if the core result is complete |
| 17-20 | Produce a comparison table, failure examples, and a short demo |

Gate: if YOLO26n native inference cannot be brought up within the first 2-3 hours, reduce the app shell or test the same checkpoint through LiteRT CPU before collecting a large dataset. Record any runtime change and establish paired CPU/GPU controls before training. G4 work starts after CPU bring-up, may extend the budget, and is repeated on the selected trained export. A working baseline with trustworthy measurements remains useful.

Local setup observed on October 7: Android Studio and Android SDK are present; the usual SDK NDK/CMake directories were absent. Native toolchain setup may take part of the initial budget. No phone was identified or verified. No packages have been installed for this project.

## Project deliverables

- A 30-60 second demo showing local camera inference and model switching.
- A small accuracy-versus-latency table for A0, B0, T0, and D0, with hardware, precision, artifact size, and dataset details.
- Three annotated failure examples: blur, small object, and confusing background/negative.
- A reproducible record of checkpoints, export settings, splits, and benchmark conditions.
- A concise account of what was implemented or changed in the reused app/runtime.

The strongest narrative is a measured decision: what constraint was set, what changed, what improved, what got worse, and which configuration was selected. A functioning baseline or a negative experiment is still evidence if reported accurately.

## Sources

- [Ultralytics YOLO26 model documentation and output modes](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics ncnn export guide](https://docs.ultralytics.com/integrations/ncnn/)
- [Ultralytics COCO class list](https://docs.ultralytics.com/datasets/detect/coco/)
- [Tencent ncnn runtime](https://github.com/Tencent/ncnn)
- [ncnn Vulkan support and fallback guidance](https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/FAQ-ncnn-vulkan.md)
- [Android YOLOv8/ncnn camera example and conversion instructions](https://github.com/nihui/ncnn-android-yolov8)
- [SOD4SB challenge dataset repository](https://github.com/IIM-TTIJ/MVA2023SmallObjectDetection4SpottingBirds)
- [FBD-SV-2024 paper and camera/capture methodology](https://pmc.ncbi.nlm.nih.gov/articles/PMC11954963/)
- [FBD-SV-2024 data and reuse terms](https://figshare.com/s/1ca0193680f894a65371)
- [Ultralytics knowledge distillation guide](https://docs.ultralytics.com/guides/knowledge-distillation)
- [Ultralytics distillation implementation reference](https://docs.ultralytics.com/reference/nn/distill_model)
- [NVIDIA TensorRT hardware support](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)
- [Google Pixel hardware specifications](https://support.google.com/pixelphone/answer/7158570)
- [Google Tensor SDK supported SoCs](https://developers.google.com/edge/litert/next/tensor-sdk)
- [LiteRT Google Tensor target enum, including G4](https://github.com/google-ai-edge/LiteRT/blob/main/litert/python/aot/vendors/google_tensor/target.py)
- [Google-maintainer G4 YOLO compilation and runtime recipe](https://github.com/google-ai-edge/LiteRT/issues/7598)
- [Google Tensor compiler flags](https://developers.google.com/edge/tensor-sdk/compilation-flags)
- [Android NNAPI deprecation and device discovery](https://developer.android.com/ndk/guides/neuralnetworks)
- [LiteRT GPU quantized-model execution](https://developers.google.com/edge/litert/performance/gpu)
- [TensorFlow QAT overview, scope, and compatibility](https://www.tensorflow.org/model_optimization/guide/quantization/training)
