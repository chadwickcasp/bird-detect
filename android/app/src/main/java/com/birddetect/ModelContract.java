package com.birddetect;

import android.content.res.AssetManager;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;

/**
 * Tensor names and decode settings for one bundled model, read from its model.json.
 * ml/stage_android_model.py writes that file from the verified export record.
 */
public class ModelContract
{
    public final String name;
    public final String paramPath;
    public final String binPath;
    public final String inputBlob;
    public final String outputBlob;
    public final int inputSize;
    public final int classIndex;
    public final float confidenceThreshold;
    public final float nmsIouThreshold;

    private ModelContract(String modelDir, JSONObject json) throws JSONException
    {
        name = json.getString("name");
        paramPath = modelDir + "/" + json.getString("param");
        binPath = modelDir + "/" + json.getString("bin");
        inputBlob = json.getString("input_blob");
        outputBlob = json.getString("output_blob");
        inputSize = json.getInt("input_size");
        classIndex = json.getInt("bird_class_index");
        confidenceThreshold = (float) json.getDouble("confidence_threshold");
        nmsIouThreshold = (float) json.getDouble("nms_iou_threshold");
    }

    public static ModelContract read(AssetManager assets, String modelDir) throws IOException, JSONException
    {
        try (InputStream stream = assets.open(modelDir + "/model.json"))
        {
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            byte[] chunk = new byte[4096];
            for (int read; (read = stream.read(chunk)) > 0; )
                bytes.write(chunk, 0, read);
            return new ModelContract(modelDir, new JSONObject(bytes.toString(StandardCharsets.UTF_8.name())));
        }
    }

    public int load(BirdDetectNative detector, AssetManager assets, boolean requestGpu, boolean useFp16)
    {
        return detector.loadModel(assets, paramPath, binPath, inputBlob, outputBlob, inputSize, classIndex,
                confidenceThreshold, nmsIouThreshold, requestGpu, useFp16);
    }
}
