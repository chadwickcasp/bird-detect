// Bird Detect: YOLO26 one-to-many preprocessing and decoding for one target class.

#ifndef YOLO26_DET_H
#define YOLO26_DET_H

#include <vector>

#include <mat.h>

// One detection in source-image pixels.
struct BirdBox
{
    float x0;
    float y0;
    float x1;
    float y1;
    float score;
};

// How a source image was placed inside the square model input.
struct Letterbox
{
    float scale;
    int pad_left;
    int pad_top;
    int source_width;
    int source_height;
};

// Decoder settings read from the exported model record.
struct DecodeSettings
{
    int class_index;
    float confidence_threshold;
    float nms_iou_threshold;
};

// Resizes `pixels` with its aspect ratio kept, pads the rest of an input_size square with 114,
// and scales values to 0-1. `pixel_type` is an ncnn::Mat::PIXEL_* conversion to RGB.
ncnn::Mat letterbox_to_input(const unsigned char* pixels, int pixel_type, int width, int height, int stride,
                             int input_size, Letterbox* letterbox);

// Reads a (4 + classes) x candidates output: rows 0-3 are center x, center y, width, height in
// input pixels and the remaining rows are per-class sigmoid scores. A candidate is kept when its
// highest-scoring class is `class_index` and that score passes the threshold, matching
// Ultralytics' single-label NMS. Boxes are suppressed by IoU and mapped back to source pixels.
void decode_yolo26_detections(const ncnn::Mat& output, const Letterbox& letterbox, const DecodeSettings& settings,
                              std::vector<BirdBox>& boxes);

#endif // YOLO26_DET_H
