// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
//
// Bird Detect: adapted from nihui/ncnn-android-yolov8 yolov8ncnn.cpp. The camera bridge is kept.
// The YOLOv8 task/model switch is replaced by one YOLO26 bird detector, a bitmap entry point for
// picked stills and clip frames, and a stats readout for the status line.

#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <android/native_window_jni.h>

#include <jni.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <benchmark.h>
#include <gpu.h>

#include "bird_detector.h"
#include "ndkcamera.h"

// Timing of the most recent detection from any source, plus a smoothed processed-frame rate.
struct PipelineStats
{
    StageTimings last = {0, 0, 0};
    int box_count = 0;
    double last_finish_ms = 0;
    float processed_fps = 0;
};

class BirdCameraWindow : public NdkCameraWindow
{
public:
    virtual void on_image_render(cv::Mat& rgb) const;
};

static const char* LOG_TAG = "BirdDetect";
static const float FPS_SMOOTHING = 0.9f;
// g_lock is held for a whole inference. g_stats_lock is separate so readStats on the UI thread never waits on one.
static ncnn::Mutex g_lock;
static std::unique_ptr<BirdDetector> g_detector;
static std::unique_ptr<BirdCameraWindow> g_camera;
static ncnn::Mutex g_stats_lock;
static PipelineStats g_stats;

static std::string to_string(JNIEnv* env, jstring value)
{
    const char* chars = env->GetStringUTFChars(value, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(value, chars);
    return result;
}

static void record_detection(const StageTimings& timings, int box_count)
{
    ncnn::MutexLockGuard guard(g_stats_lock);
    const double now = ncnn::get_current_time();
    if (g_stats.last_finish_ms > 0)
    {
        const float fps = 1000.f / (float)(now - g_stats.last_finish_ms);
        g_stats.processed_fps = g_stats.processed_fps == 0 ? fps
                                                           : FPS_SMOOTHING * g_stats.processed_fps + (1 - FPS_SMOOTHING) * fps;
    }
    g_stats.last_finish_ms = now;
    g_stats.last = timings;
    g_stats.box_count = box_count;
}

static void draw_bird_boxes(cv::Mat& rgb, const std::vector<BirdBox>& boxes)
{
    const cv::Scalar color(0, 255, 0);
    for (const BirdBox& box : boxes)
    {
        cv::rectangle(rgb, cv::Point((int)box.x0, (int)box.y0), cv::Point((int)box.x1, (int)box.y1), color, 3);
        char text[16];
        snprintf(text, sizeof(text), "bird %.2f", box.score);
        cv::putText(rgb, text, cv::Point((int)box.x0, std::max(20, (int)box.y0 - 6)), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    color, 2);
    }
}

void BirdCameraWindow::on_image_render(cv::Mat& rgb) const
{
    ncnn::MutexLockGuard guard(g_lock);
    if (!g_detector)
        return;

    std::vector<BirdBox> boxes;
    StageTimings timings;
    if (g_detector->detect(rgb.data, ncnn::Mat::PIXEL_RGB, rgb.cols, rgb.rows, (int)rgb.step, boxes, timings) != 0)
        return;
    record_detection(timings, (int)boxes.size());
    draw_bird_boxes(rgb, boxes);
}

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    ncnn::create_gpu_instance();
    g_camera = std::make_unique<BirdCameraWindow>();
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    {
        ncnn::MutexLockGuard guard(g_lock);
        g_detector.reset();
    }
    g_camera.reset();
    ncnn::destroy_gpu_instance();
}

// Returns -1 on failure, 0 when running on CPU, and 1 when running on the Vulkan GPU.
JNIEXPORT jint JNICALL Java_com_birddetect_BirdDetectNative_loadModel(
    JNIEnv* env, jobject thiz, jobject asset_manager, jstring param_path, jstring bin_path, jstring input_blob,
    jstring output_blob, jint input_size, jint class_index, jfloat confidence_threshold, jfloat nms_iou_threshold,
    jboolean request_gpu, jboolean use_fp16)
{
    ModelContract contract;
    contract.param_path = to_string(env, param_path);
    contract.bin_path = to_string(env, bin_path);
    contract.input_blob = to_string(env, input_blob);
    contract.output_blob = to_string(env, output_blob);
    contract.input_size = input_size;
    contract.decode = {class_index, confidence_threshold, nms_iou_threshold};

    auto detector = std::make_unique<BirdDetector>();
    AAssetManager* mgr = AAssetManager_fromJava(env, asset_manager);
    if (detector->load(mgr, contract, request_gpu, use_fp16) != 0)
    {
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "loadModel failed for %s", contract.param_path.c_str());
        return -1;
    }
    const int backend = detector->uses_gpu() ? 1 : 0;
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "loadModel %s backend=%s fp16=%d gpu_count=%d",
                        contract.param_path.c_str(), backend ? "gpu" : "cpu", (int)use_fp16, ncnn::get_gpu_count());

    {
        ncnn::MutexLockGuard guard(g_lock);
        g_detector = std::move(detector);
    }
    ncnn::MutexLockGuard stats_guard(g_stats_lock);
    g_stats = PipelineStats();
    return backend;
}

JNIEXPORT jboolean JNICALL Java_com_birddetect_BirdDetectNative_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1)
        return JNI_FALSE;
    g_camera->open((int)facing);
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG, "openCamera facing=%d sensor_orientation=%d", (int)facing,
                        g_camera->camera_orientation);
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_birddetect_BirdDetectNative_closeCamera(JNIEnv* env, jobject thiz)
{
    g_camera->close();
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_birddetect_BirdDetectNative_setOutputWindow(JNIEnv* env, jobject thiz,
                                                                               jobject surface)
{
    ANativeWindow* window = surface ? ANativeWindow_fromSurface(env, surface) : nullptr;
    g_camera->set_window(window);
    if (window)
        ANativeWindow_release(window);
    return JNI_TRUE;
}

// Detects birds in an upright ARGB_8888 bitmap. Returns x0, y0, x1, y1, score per box in bitmap
// pixels, or null when no model is loaded or the bitmap cannot be read.
JNIEXPORT jfloatArray JNICALL Java_com_birddetect_BirdDetectNative_detectBitmap(JNIEnv* env, jobject thiz,
                                                                               jobject bitmap)
{
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) != ANDROID_BITMAP_RESULT_SUCCESS ||
        info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return nullptr;

    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS)
        return nullptr;

    std::vector<BirdBox> boxes;
    StageTimings timings;
    int status = -1;
    {
        ncnn::MutexLockGuard guard(g_lock);
        if (g_detector)
        {
            status = g_detector->detect((const unsigned char*)pixels, ncnn::Mat::PIXEL_RGBA2RGB, (int)info.width,
                                        (int)info.height, (int)info.stride, boxes, timings);
            if (status == 0)
                record_detection(timings, (int)boxes.size());
        }
    }
    AndroidBitmap_unlockPixels(env, bitmap);
    if (status != 0)
        return nullptr;

    std::vector<float> flat;
    flat.reserve(boxes.size() * 5);
    for (const BirdBox& box : boxes)
        flat.insert(flat.end(), {box.x0, box.y0, box.x1, box.y1, box.score});
    jfloatArray result = env->NewFloatArray((jsize)flat.size());
    env->SetFloatArrayRegion(result, 0, (jsize)flat.size(), flat.data());
    return result;
}

// Returns preprocess, inference, postprocess, and total milliseconds of the latest detection,
// then the smoothed processed frames per second and the latest box count.
JNIEXPORT jfloatArray JNICALL Java_com_birddetect_BirdDetectNative_readStats(JNIEnv* env, jobject thiz)
{
    float values[6];
    {
        ncnn::MutexLockGuard guard(g_stats_lock);
        const StageTimings& t = g_stats.last;
        values[0] = (float)t.preprocess_ms;
        values[1] = (float)t.inference_ms;
        values[2] = (float)t.postprocess_ms;
        values[3] = (float)(t.preprocess_ms + t.inference_ms + t.postprocess_ms);
        values[4] = g_stats.processed_fps;
        values[5] = (float)g_stats.box_count;
    }
    jfloatArray result = env->NewFloatArray(6);
    env->SetFloatArrayRegion(result, 0, 6, values);
    return result;
}

}
