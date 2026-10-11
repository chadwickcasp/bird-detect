package com.birddetect.g4smoke;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;

import java.util.ArrayList;
import java.util.List;

/**
 * Java copy of the camera app's yolo26_det.cpp for the LiteRT export.
 * Input is NCHW float RGB / 255 letterboxed with 114. Output is (4 + classes, 8400): center x, center y,
 * width, height as fractions of the input, then per-class sigmoid scores.
 */
final class Yolo26Decoding
{
    static final int INPUT_SIZE = 640;
    static final int CANDIDATE_COUNT = 8400;
    /** The LiteRT export's box rows are fractions of the input (output.box_units in the export record). */
    static final float BOX_ROW_SCALE = INPUT_SIZE;
    static final int BIRD_CLASS = 14;
    static final float CONFIDENCE_THRESHOLD = 0.25f;
    static final float NMS_IOU_THRESHOLD = 0.45f;
    private static final int PAD_GRAY = 114;

    /** Letterboxed input plus the placement needed to map boxes back to the source image. */
    static final class LetterboxedInput
    {
        final float[] nchw;
        final float scale;
        final int padLeft;
        final int padTop;

        LetterboxedInput(float[] nchw, float scale, int padLeft, int padTop)
        {
            this.nchw = nchw;
            this.scale = scale;
            this.padLeft = padLeft;
            this.padTop = padTop;
        }
    }

    private Yolo26Decoding()
    {
    }

    static LetterboxedInput letterbox(Bitmap image)
    {
        float scale = Math.min((float) INPUT_SIZE / image.getWidth(), (float) INPUT_SIZE / image.getHeight());
        int resizedW = Math.round(image.getWidth() * scale);
        int resizedH = Math.round(image.getHeight() * scale);
        int padLeft = Math.round((INPUT_SIZE - resizedW) / 2f - 0.1f);
        int padTop = Math.round((INPUT_SIZE - resizedH) / 2f - 0.1f);

        Bitmap square = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(square);
        canvas.drawColor(Color.rgb(PAD_GRAY, PAD_GRAY, PAD_GRAY));
        canvas.drawBitmap(image, null, new Rect(padLeft, padTop, padLeft + resizedW, padTop + resizedH),
                new Paint(Paint.FILTER_BITMAP_FLAG));

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        square.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        int plane = INPUT_SIZE * INPUT_SIZE;
        float[] nchw = new float[3 * plane];
        for (int i = 0; i < plane; i++)
        {
            nchw[i] = Color.red(pixels[i]) / 255f;
            nchw[plane + i] = Color.green(pixels[i]) / 255f;
            nchw[2 * plane + i] = Color.blue(pixels[i]) / 255f;
        }
        return new LetterboxedInput(nchw, scale, padLeft, padTop);
    }

    /** Returns x0, y0, x1, y1, score per bird in source-image pixels after argmax, threshold, and NMS. */
    static List<float[]> decodeBirds(float[] output, LetterboxedInput placement, int sourceWidth, int sourceHeight)
    {
        int candidates = CANDIDATE_COUNT;
        int classCount = output.length / candidates - 4;
        List<float[]> boxes = new ArrayList<>();
        for (int i = 0; i < candidates; i++)
        {
            int bestClass = 0;
            float bestScore = output[4 * candidates + i];
            for (int c = 1; c < classCount; c++)
            {
                float score = output[(4 + c) * candidates + i];
                if (score > bestScore)
                {
                    bestScore = score;
                    bestClass = c;
                }
            }
            if (bestClass != BIRD_CLASS || bestScore <= CONFIDENCE_THRESHOLD)
                continue;
            float cx = output[i] * BOX_ROW_SCALE, cy = output[candidates + i] * BOX_ROW_SCALE;
            float w = output[2 * candidates + i] * BOX_ROW_SCALE, h = output[3 * candidates + i] * BOX_ROW_SCALE;
            boxes.add(new float[]{cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, bestScore});
        }

        boxes.sort((a, b) -> Float.compare(b[4], a[4]));
        List<float[]> kept = new ArrayList<>();
        for (float[] box : boxes)
        {
            boolean overlaps = false;
            for (float[] previous : kept)
                overlaps |= iou(box, previous) > NMS_IOU_THRESHOLD;
            if (!overlaps)
                kept.add(box);
        }

        for (float[] box : kept)
        {
            box[0] = clamp((box[0] - placement.padLeft) / placement.scale, sourceWidth);
            box[1] = clamp((box[1] - placement.padTop) / placement.scale, sourceHeight);
            box[2] = clamp((box[2] - placement.padLeft) / placement.scale, sourceWidth);
            box[3] = clamp((box[3] - placement.padTop) / placement.scale, sourceHeight);
        }
        return kept;
    }

    static float iou(float[] a, float[] b)
    {
        float overlapW = Math.max(0f, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
        float overlapH = Math.max(0f, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
        float overlap = overlapW * overlapH;
        float union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap;
        return union > 0f ? overlap / union : 0f;
    }

    private static float clamp(float value, int max)
    {
        return Math.max(0f, Math.min(value, max));
    }
}
