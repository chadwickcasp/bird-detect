// Bird Detect: one loaded ncnn YOLO26 model with an explicit backend and precision.

#include "bird_detector.h"

#include <benchmark.h>
#include <gpu.h>

int BirdDetector::load(AAssetManager* mgr, const ModelContract& model_contract, bool request_gpu, bool use_fp16)
{
    contract = model_contract;
    net.clear();
    net.opt = ncnn::Option();

    gpu_active = false;
#if NCNN_VULKAN
    gpu_active = request_gpu && ncnn::get_gpu_count() > 0;
    net.opt.use_vulkan_compute = gpu_active;
#endif
    // ncnn enables FP16 storage and arithmetic by default where supported; set them explicitly
    // so the FP32 setting is a true reference.
    fp16_active = use_fp16;
    net.opt.use_fp16_packed = use_fp16;
    net.opt.use_fp16_storage = use_fp16;
    net.opt.use_fp16_arithmetic = use_fp16;
    net.opt.use_bf16_storage = false;

    if (gpu_active)
    {
        // Pixel 9 Pro Mali-G715 with ncnn 20260526 fails fixture parity without both settings
        // (docs/phone_baseline.md). With light mode, every frame after the first has its box rows
        // scaled by a changing factor. Cooperative-matrix FP16 saturates every score.
        net.opt.lightmode = false;
        net.opt.use_cooperative_matrix = false;
    }

    if (net.load_param(mgr, contract.param_path.c_str()) != 0)
        return -1;
    if (net.load_model(mgr, contract.bin_path.c_str()) != 0)
        return -1;
    return 0;
}

int BirdDetector::detect(const unsigned char* pixels, int pixel_type, int width, int height, int stride,
                         std::vector<BirdBox>& boxes, StageTimings& timings)
{
    const double start = ncnn::get_current_time();
    Letterbox letterbox;
    ncnn::Mat input = letterbox_to_input(pixels, pixel_type, width, height, stride, contract.input_size, &letterbox);
    const double preprocessed = ncnn::get_current_time();

    ncnn::Extractor extractor = net.create_extractor();
    extractor.input(contract.input_blob.c_str(), input);
    ncnn::Mat output;
    // extract() waits for GPU work and downloads the result, so the inference time includes both.
    const int status = extractor.extract(contract.output_blob.c_str(), output);
    const double inferred = ncnn::get_current_time();
    if (status != 0)
        return status;

    decode_yolo26_detections(output, letterbox, contract.decode, boxes);
    const double decoded = ncnn::get_current_time();

    timings.preprocess_ms = preprocessed - start;
    timings.inference_ms = inferred - preprocessed;
    timings.postprocess_ms = decoded - inferred;
    return 0;
}
