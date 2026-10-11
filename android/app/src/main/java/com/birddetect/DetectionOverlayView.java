package com.birddetect;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.util.AttributeSet;
import android.view.View;

import java.util.Locale;

/**
 * Draws bird boxes over an image or clip shown fit-center in the same parent.
 * Boxes arrive in source-frame pixels and are mapped with the same fit-center scale.
 */
public class DetectionOverlayView extends View
{
    private final Paint boxPaint = new Paint();
    private final Paint labelPaint = new Paint();
    private float[] boxes = new float[0];
    private int sourceWidth = 1;
    private int sourceHeight = 1;

    public DetectionOverlayView(Context context, AttributeSet attrs)
    {
        super(context, attrs);
        boxPaint.setColor(Color.GREEN);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(5f);
        labelPaint.setColor(Color.GREEN);
        labelPaint.setTextSize(40f);
        labelPaint.setAntiAlias(true);
    }

    /** Shows boxes given as x0, y0, x1, y1, score per bird in a sourceWidth x sourceHeight frame. */
    public void showDetections(float[] detections, int width, int height)
    {
        boxes = detections == null ? new float[0] : detections;
        sourceWidth = width;
        sourceHeight = height;
        invalidate();
    }

    public void clearDetections()
    {
        boxes = new float[0];
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas)
    {
        float scale = Math.min((float) getWidth() / sourceWidth, (float) getHeight() / sourceHeight);
        float offsetX = (getWidth() - sourceWidth * scale) / 2f;
        float offsetY = (getHeight() - sourceHeight * scale) / 2f;
        for (int i = 0; i + 4 < boxes.length; i += 5)
        {
            float left = offsetX + boxes[i] * scale;
            float top = offsetY + boxes[i + 1] * scale;
            canvas.drawRect(left, top, offsetX + boxes[i + 2] * scale, offsetY + boxes[i + 3] * scale, boxPaint);
            canvas.drawText(String.format(Locale.US, "bird %.2f", boxes[i + 4]), left, Math.max(40f, top - 8f), labelPaint);
        }
    }
}
