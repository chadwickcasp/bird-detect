// Bird Detect: one loaded ncnn YOLO26 model with an explicit backend and precision.

#ifndef BIRD_DETECTOR_H
#define BIRD_DETECTOR_H

#include <string>
#include <vector>

#include <android/asset_manager.h>

#include <net.h>

#include "yolo26_det.h"

// Exported tensor names and decode settings from the model's record.
struct ModelContract
{
    std::string param_path;
    std::string bin_path;
    std::string input_blob;
    std::string output_blob;
    int input_size;
    DecodeSettings decode;
};

// Milliseconds spent in each stage of the most recent detection.
struct StageTimings
{
    double preprocess_ms;
    double inference_ms;
    double postprocess_ms;
};

class BirdDetector
{
public:
    // Loads the model. A GPU request falls back to CPU when ncnn finds no Vulkan device.
    // Returns 0 on success.
    int load(AAssetManager* mgr, const ModelContract& contract, bool request_gpu, bool use_fp16);

    // Detects birds in one upright image. `pixel_type` is an ncnn::Mat::PIXEL_* conversion to RGB.
    int detect(const unsigned char* pixels, int pixel_type, int width, int height, int stride,
               std::vector<BirdBox>& boxes, StageTimings& timings);

    bool uses_gpu() const { return gpu_active; }
    bool uses_fp16() const { return fp16_active; }

private:
    ncnn::Net net;
    ModelContract contract;
    bool gpu_active = false;
    bool fp16_active = false;
};

#endif // BIRD_DETECTOR_H
