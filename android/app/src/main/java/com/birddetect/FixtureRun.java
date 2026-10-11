package com.birddetect;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

/**
 * Detects birds in the bundled parity fixtures and writes predictions_<backend>.json in the
 * format ml/compare_predictions.py reads. Pull it with adb from the app's external files dir.
 */
public class FixtureRun
{
    public static final String FIXTURES_DIR = "fixtures";

    /** The written file plus the last fixture and its boxes, for display. */
    public static class Result
    {
        public final File predictionsFile;
        public final Bitmap lastImage;
        public final float[] lastBoxes;

        Result(File predictionsFile, Bitmap lastImage, float[] lastBoxes)
        {
            this.predictionsFile = predictionsFile;
            this.lastImage = lastImage;
            this.lastBoxes = lastBoxes;
        }
    }

    /** Runs every fixture through the loaded model; backendLabel is the backend that actually ran, e.g. "ncnn-gpu-fp16". */
    public static Result predictFixtures(BirdDetectNative detector, AssetManager assets, File outputDir,
                                         String backendLabel) throws IOException, JSONException
    {
        String[] names = assets.list(FIXTURES_DIR);
        Arrays.sort(names);
        JSONObject images = new JSONObject();
        Bitmap image = null;
        float[] boxes = null;
        for (String name : names)
        {
            image = decodeFixture(assets, FIXTURES_DIR + "/" + name);
            boxes = detector.detectBitmap(image);
            if (boxes == null)
                throw new IOException("detector returned no result for " + name);
            images.put(name, describeImage(image, boxes));
        }

        JSONObject predictions = new JSONObject();
        predictions.put("source", "phone:" + backendLabel);
        predictions.put("device", Build.MANUFACTURER + " " + Build.MODEL);
        predictions.put("android_build", Build.DISPLAY);
        predictions.put("images", images);

        File file = new File(outputDir, "predictions_" + backendLabel + ".json");
        try (FileOutputStream out = new FileOutputStream(file))
        {
            out.write(predictions.toString(2).getBytes(StandardCharsets.UTF_8));
        }
        return new Result(file, image, boxes);
    }

    private static Bitmap decodeFixture(AssetManager assets, String path) throws IOException
    {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        options.inScaled = false;
        try (InputStream stream = assets.open(path))
        {
            Bitmap bitmap = BitmapFactory.decodeStream(stream, null, options);
            if (bitmap == null)
                throw new IOException("cannot decode " + path);
            return bitmap;
        }
    }

    private static JSONObject describeImage(Bitmap image, float[] boxes) throws JSONException
    {
        JSONArray boxList = new JSONArray();
        for (int i = 0; i + 4 < boxes.length; i += 5)
        {
            JSONArray xyxy = new JSONArray();
            for (int k = 0; k < 4; k++)
                xyxy.put(Math.round(boxes[i + k] * 100) / 100.0);
            JSONObject box = new JSONObject();
            box.put("xyxy", xyxy);
            box.put("score", Math.round(boxes[i + 4] * 10000) / 10000.0);
            boxList.put(box);
        }
        JSONObject described = new JSONObject();
        described.put("width", image.getWidth());
        described.put("height", image.getHeight());
        described.put("boxes", boxList);
        return described;
    }
}
