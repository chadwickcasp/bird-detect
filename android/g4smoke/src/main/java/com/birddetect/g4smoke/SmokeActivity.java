package com.birddetect.g4smoke;

import android.app.Activity;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import com.google.ai.edge.litert.Accelerator;
import com.google.ai.edge.litert.CompiledModel;
import com.google.ai.edge.litert.Environment;
import com.google.ai.edge.litert.TensorBuffer;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Runs YOLO26n on the parity fixtures with LiteRT CPU, then requests NPU-only execution.
 * Writes predictions_litert-cpu.json, predictions_litert-npu.json when an ahead-of-time model ran,
 * predictions_litert-npu-on-device-compile.json when the on-device compiler plugin is present, and
 * g4smoke.json with the device, accelerators, timings, and any error. Results also go to logcat tag G4Smoke.
 */
public class SmokeActivity extends Activity
{
    private static final String TAG = "G4Smoke";
    private static final String FLOAT_MODEL = "models/yolo26n.tflite";
    private static final String COMPILED_MODEL = "models/yolo26n_Tensor_G4.tflite";
    private static final String FIXTURES_DIR = "fixtures";
    private static final String LITERT_VERSION = "2.1.6";
    private static final String DISPATCH_LIBRARY = "libLiteRtDispatch_GoogleTensor.so";
    private static final String DISPATCH_LIBRARY_VERSION = "2.1.6";
    /** From the LiteRT v2.1.6 JIT zip. The runtime zip that supplies the dispatch library has no compiler. */
    private static final String COMPILER_PLUGIN = "libLiteRtCompilerPlugin_google_tensor.so";
    private static final String COMPILER_PLUGIN_VERSION = "2.1.6";
    private static final AtomicBoolean smokeTestStarted = new AtomicBoolean(false);

    private TextView statusText;

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        statusText = new TextView(this);
        statusText.setPadding(32, 96, 32, 32);
        statusText.setText("running...");
        setContentView(statusText);
        if (!smokeTestStarted.compareAndSet(false, true))
        {
            statusText.setText("a smoke test is already running");
            return;
        }
        new Thread(this::runSmokeTest, "g4smoke").start();
    }

    private void runSmokeTest()
    {
        JSONObject record = new JSONObject();
        try
        {
            String nativeLibraryDir = getApplicationInfo().nativeLibraryDir;
            boolean onDeviceCompiler = new File(nativeLibraryDir, COMPILER_PLUGIN).exists();
            record.put("device", Build.MANUFACTURER + " " + Build.MODEL);
            record.put("soc", Build.SOC_MANUFACTURER + " " + Build.SOC_MODEL);
            record.put("android_build", Build.DISPLAY);
            record.put("litert", LITERT_VERSION);
            record.put("dispatch_library_present", new File(nativeLibraryDir, DISPATCH_LIBRARY).exists());
            record.put("dispatch_library", DISPATCH_LIBRARY_VERSION);
            record.put("on_device_compiler_present", onDeviceCompiler);
            record.put("compiler_plugin", onDeviceCompiler ? COMPILER_PLUGIN_VERSION : "absent");

            try (Environment environment = Environment.create(Map.of(
                    Environment.Option.DispatchLibraryDir, nativeLibraryDir,
                    Environment.Option.CompilerPluginLibraryDir, nativeLibraryDir)))
            {
                record.put("available_accelerators", String.valueOf(environment.getAvailableAccelerators()));
                record.put("cpu", runAndRecord(environment, FLOAT_MODEL, Accelerator.CPU, "litert-cpu"));
                // The NPU request can abort the process inside LiteRT. Keep the CPU result on disk first.
                writeQuietly("g4smoke.json", record);
                boolean compiledForNpu = hasAsset(COMPILED_MODEL);
                record.put("compiled_for_npu", compiledForNpu);
                record.put("npu", runAndRecord(environment, npuModel(compiledForNpu), Accelerator.NPU,
                        npuLabel(compiledForNpu, onDeviceCompiler)));
            }
        }
        catch (Exception e)
        {
            Log.e(TAG, "smoke test setup failed", e);
            putQuietly(record, "setup_error", e.toString());
        }
        writeQuietly("g4smoke.json", record);
        String summary = record.toString().replace(",\"", ",\n\"");
        Log.i(TAG, summary);
        runOnUiThread(() -> statusText.setText(summary));
    }

    /** An ahead-of-time model is an NPU result. The on-device plugin is only a compile request until logcat shows it. */
    private static String npuLabel(boolean compiledForNpu, boolean onDeviceCompiler)
    {
        if (compiledForNpu)
            return "litert-npu";
        if (onDeviceCompiler)
            return "litert-npu-on-device-compile";
        return "litert-npu-request-uncompiled";
    }

    private static String npuModel(boolean compiledForNpu)
    {
        return compiledForNpu ? COMPILED_MODEL : FLOAT_MODEL;
    }

    /** Runs every fixture with one accelerator and returns status, model, timings, and any error text. */
    private JSONObject runAndRecord(Environment environment, String modelPath, Accelerator accelerator, String label)
            throws JSONException
    {
        JSONObject run = new JSONObject();
        run.put("model", modelPath);
        run.put("accelerator", accelerator.name());
        deleteQuietly("predictions_" + label + ".json");
        try
        {
            JSONObject predictions = predictFixtures(environment, modelPath, accelerator, label, run);
            writeQuietly("predictions_" + label + ".json", predictions);
            run.put("status", "ran");
        }
        catch (Exception e)
        {
            Log.e(TAG, label + " failed", e);
            run.put("status", "failed");
            run.put("error", e.toString());
        }
        return run;
    }

    private JSONObject predictFixtures(Environment environment, String modelPath, Accelerator accelerator,
                                       String label, JSONObject run) throws Exception
    {
        JSONObject images = new JSONObject();
        JSONArray runMs = new JSONArray();
        try (CompiledModel model = CompiledModel.create(getAssets(), modelPath, new CompiledModel.Options(accelerator), environment))
        {
            List<TensorBuffer> inputs = model.createInputBuffers();
            List<TensorBuffer> outputs = null;
            try
            {
                outputs = model.createOutputBuffers();
                String[] names = getAssets().list(FIXTURES_DIR);
                if (names == null || names.length == 0)
                    throw new IOException("assets/" + FIXTURES_DIR + " has no images");
                Arrays.sort(names);
                for (String name : names)
                {
                    Bitmap image = decodeFixture(getAssets(), FIXTURES_DIR + "/" + name);
                    Yolo26Decoding.LetterboxedInput input = Yolo26Decoding.letterbox(image);
                    inputs.get(0).writeFloat(input.nchw);
                    long start = System.nanoTime();
                    model.run(inputs, outputs);
                    runMs.put((System.nanoTime() - start) / 1e6);
                    float[] output = outputs.get(0).readFloat();
                    List<float[]> boxes = Yolo26Decoding.decodeBirds(output, input, image.getWidth(), image.getHeight());
                    images.put(name, describeImage(image, boxes));
                    Log.i(TAG, label + " " + name + " boxes=" + boxes.size());
                }
            }
            finally
            {
                closeBuffers(inputs);
                closeBuffers(outputs);
            }
        }
        run.put("run_ms", runMs);

        JSONObject predictions = new JSONObject();
        predictions.put("source", "phone:" + label);
        predictions.put("device", Build.MANUFACTURER + " " + Build.MODEL);
        predictions.put("android_build", Build.DISPLAY);
        predictions.put("images", images);
        return predictions;
    }

    private static void closeBuffers(List<TensorBuffer> buffers)
    {
        if (buffers == null)
            return;
        for (TensorBuffer buffer : buffers)
            buffer.close();
    }

    private boolean hasAsset(String path)
    {
        try (InputStream ignored = getAssets().open(path))
        {
            return true;
        }
        catch (IOException e)
        {
            return false;
        }
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

    private static JSONObject describeImage(Bitmap image, List<float[]> boxes) throws JSONException
    {
        JSONArray boxList = new JSONArray();
        for (float[] box : boxes)
        {
            JSONArray xyxy = new JSONArray();
            for (int k = 0; k < 4; k++)
                xyxy.put(Math.round(box[k] * 100) / 100.0);
            JSONObject entry = new JSONObject();
            entry.put("xyxy", xyxy);
            entry.put("score", Math.round(box[4] * 10000) / 10000.0);
            boxList.put(entry);
        }
        JSONObject described = new JSONObject();
        described.put("width", image.getWidth());
        described.put("height", image.getHeight());
        described.put("boxes", boxList);
        return described;
    }

    private void deleteQuietly(String fileName)
    {
        File file = externalFile(fileName);
        if (file != null && file.exists() && !file.delete())
            Log.w(TAG, "cannot remove previous " + file.getName());
    }

    private File externalFile(String fileName)
    {
        File dir = getExternalFilesDir(null);
        if (dir == null)
        {
            Log.e(TAG, "external files dir is unavailable; skipped " + fileName);
            return null;
        }
        return new File(dir, fileName);
    }

    /** Writes through a temporary file so a failed write leaves the previous result in place. */
    private void writeQuietly(String fileName, JSONObject json)
    {
        File file = externalFile(fileName);
        if (file == null)
            return;
        File temporary = new File(file.getParentFile(), fileName + ".tmp");
        try
        {
            byte[] bytes = json.toString(2).getBytes(StandardCharsets.UTF_8);
            try (FileOutputStream out = new FileOutputStream(temporary))
            {
                out.write(bytes);
            }
            if (!temporary.renameTo(file))
                throw new IOException("cannot replace " + file.getName());
        }
        catch (IOException | JSONException e)
        {
            Log.e(TAG, "cannot write " + file, e);
            if (!temporary.delete() && temporary.exists())
                Log.w(TAG, "cannot remove " + temporary.getName());
        }
    }

    private static void putQuietly(JSONObject json, String key, String value)
    {
        try
        {
            json.put(key, value);
        }
        catch (JSONException e)
        {
            Log.e(TAG, "cannot record " + key, e);
        }
    }
}
