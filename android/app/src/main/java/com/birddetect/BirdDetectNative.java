package com.birddetect;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.view.Surface;

/** JNI bridge to the ncnn bird detector and the native rear-camera preview. */
public class BirdDetectNative
{
    public static final int BACKEND_FAILED = -1;
    public static final int BACKEND_CPU = 0;
    public static final int BACKEND_GPU = 1;

    /** Loads a model; returns BACKEND_FAILED, BACKEND_CPU, or BACKEND_GPU for the backend that actually runs. */
    public native int loadModel(AssetManager mgr, String paramPath, String binPath, String inputBlob,
                                String outputBlob, int inputSize, int classIndex, float confidenceThreshold,
                                float nmsIouThreshold, boolean requestGpu, boolean useFp16);

    /** Opens a camera; facing 0 is front and 1 is back. Frames render to the output window with boxes drawn. */
    public native boolean openCamera(int facing);

    public native boolean closeCamera();

    /** Pass null when the surface is destroyed so camera frames stop rendering into it. */
    public native boolean setOutputWindow(Surface surface);

    /** Returns x0, y0, x1, y1, score per bird in bitmap pixels, or null without a model. Needs ARGB_8888. */
    public native float[] detectBitmap(Bitmap bitmap);

    /** Returns preprocess, inference, postprocess, total ms, processed FPS, and box count of the latest detection. */
    public native float[] readStats();

    static {
        System.loadLibrary("birddetect");
    }
}
