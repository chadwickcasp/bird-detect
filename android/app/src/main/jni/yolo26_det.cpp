// Bird Detect: YOLO26 one-to-many preprocessing and decoding for one target class.

#include "yolo26_det.h"

#include <algorithm>
#include <cmath>

static const float LETTERBOX_PAD_VALUE = 114.f;

static float box_area(const BirdBox& box)
{
    return (box.x1 - box.x0) * (box.y1 - box.y0);
}

static float box_iou(const BirdBox& a, const BirdBox& b)
{
    const float overlap_w = std::max(0.f, std::min(a.x1, b.x1) - std::max(a.x0, b.x0));
    const float overlap_h = std::max(0.f, std::min(a.y1, b.y1) - std::max(a.y0, b.y0));
    const float overlap = overlap_w * overlap_h;
    const float union_area = box_area(a) + box_area(b) - overlap;
    return union_area > 0.f ? overlap / union_area : 0.f;
}

// Keeps boxes in descending score order, dropping any that overlap a kept box above the threshold.
static void suppress_overlaps(std::vector<BirdBox>& boxes, float iou_threshold)
{
    std::sort(boxes.begin(), boxes.end(), [](const BirdBox& a, const BirdBox& b) { return a.score > b.score; });

    std::vector<BirdBox> kept;
    for (const BirdBox& box : boxes)
    {
        bool overlaps = false;
        for (const BirdBox& previous : kept)
        {
            if (box_iou(box, previous) > iou_threshold)
            {
                overlaps = true;
                break;
            }
        }
        if (!overlaps)
            kept.push_back(box);
    }
    boxes.swap(kept);
}

ncnn::Mat letterbox_to_input(const unsigned char* pixels, int pixel_type, int width, int height, int stride,
                             int input_size, Letterbox* letterbox)
{
    const float scale = std::min((float)input_size / width, (float)input_size / height);
    const int resized_w = (int)std::round(width * scale);
    const int resized_h = (int)std::round(height * scale);

    // Same split as Ultralytics LetterBox: the odd pixel of padding goes to the right/bottom.
    const float pad_w = (input_size - resized_w) / 2.f;
    const float pad_h = (input_size - resized_h) / 2.f;
    const int pad_left = (int)std::round(pad_w - 0.1f);
    const int pad_right = (int)std::round(pad_w + 0.1f);
    const int pad_top = (int)std::round(pad_h - 0.1f);
    const int pad_bottom = (int)std::round(pad_h + 0.1f);

    ncnn::Mat resized = ncnn::Mat::from_pixels_resize(pixels, pixel_type, width, height, stride, resized_w, resized_h);
    ncnn::Mat input;
    ncnn::copy_make_border(resized, input, pad_top, pad_bottom, pad_left, pad_right, ncnn::BORDER_CONSTANT,
                           LETTERBOX_PAD_VALUE);

    const float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    input.substract_mean_normalize(0, norm);

    letterbox->scale = scale;
    letterbox->pad_left = pad_left;
    letterbox->pad_top = pad_top;
    letterbox->source_width = width;
    letterbox->source_height = height;
    return input;
}

void decode_yolo26_detections(const ncnn::Mat& output, const Letterbox& letterbox, const DecodeSettings& settings,
                              std::vector<BirdBox>& boxes)
{
    boxes.clear();
    const int candidate_count = output.w;
    const int class_count = output.h - 4;
    const float* center_x = output.row(0);
    const float* center_y = output.row(1);
    const float* box_w = output.row(2);
    const float* box_h = output.row(3);

    for (int i = 0; i < candidate_count; i++)
    {
        int best_class = 0;
        float best_score = output.row(4)[i];
        for (int c = 1; c < class_count; c++)
        {
            const float score = output.row(4 + c)[i];
            if (score > best_score)
            {
                best_score = score;
                best_class = c;
            }
        }
        if (best_class != settings.class_index || best_score <= settings.confidence_threshold)
            continue;

        BirdBox box;
        box.x0 = center_x[i] - box_w[i] / 2;
        box.y0 = center_y[i] - box_h[i] / 2;
        box.x1 = center_x[i] + box_w[i] / 2;
        box.y1 = center_y[i] + box_h[i] / 2;
        box.score = best_score;
        boxes.push_back(box);
    }

    suppress_overlaps(boxes, settings.nms_iou_threshold);

    const float max_x = (float)letterbox.source_width;
    const float max_y = (float)letterbox.source_height;
    for (BirdBox& box : boxes)
    {
        box.x0 = std::clamp((box.x0 - letterbox.pad_left) / letterbox.scale, 0.f, max_x);
        box.y0 = std::clamp((box.y0 - letterbox.pad_top) / letterbox.scale, 0.f, max_y);
        box.x1 = std::clamp((box.x1 - letterbox.pad_left) / letterbox.scale, 0.f, max_x);
        box.y1 = std::clamp((box.y1 - letterbox.pad_top) / letterbox.scale, 0.f, max_y);
    }
}
