package com.birddetect;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.ImageDecoder;
import android.graphics.SurfaceTexture;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.util.Size;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.TextureView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.Spinner;
import android.widget.TextView;

import java.io.IOException;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * One screen with three bird sources: the rear camera, a picked still, and a picked clip.
 * The status line reports the backend that actually ran, never only the one requested.
 */
public class MainActivity extends Activity implements SurfaceHolder.Callback, ClipPlayer.Listener
{
    private static final String TAG = "BirdDetect";
    private static final String MODEL_DIR = "models/yolo26n";
    private static final int SOURCE_CAMERA = 0;
    private static final int SOURCE_IMAGE = 1;
    private static final int SOURCE_CLIP = 2;
    private static final int REQUEST_CAMERA = 100;
    private static final int REQUEST_PICK_IMAGE = 101;
    private static final int REQUEST_PICK_CLIP = 102;
    private static final int CAMERA_FACING_BACK = 1;
    private static final int MAX_IMAGE_SIDE = 4096;
    private static final long STATUS_INTERVAL_MS = 500;
    private static final int STATUS_LOG_EVERY = 10;
    /** Launch extras for adb-driven runs: am start -n com.birddetect/.MainActivity --ei source 0 --ei backend 3 --ez run_fixtures true */
    private static final String EXTRA_SOURCE = "source";
    private static final String EXTRA_BACKEND = "backend";
    private static final String EXTRA_RUN_FIXTURES = "run_fixtures";
    /** {requestGpu, useFp16} per entry of R.array.backend_names. */
    private static final boolean[][] BACKEND_CHOICES = {{false, false}, {false, true}, {true, false}, {true, true}};

    private final BirdDetectNative detector = new BirdDetectNative();
    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private ModelContract contract;
    private String backendLabel = "";
    private String notice = "";
    private int source = SOURCE_CAMERA;
    private Uri pendingClip;
    private boolean cameraOpen;
    private boolean runFixturesAfterLoad;
    private int statusRefreshCount;

    private SurfaceView cameraView;
    private ImageView imageView;
    private TextureView clipView;
    private DetectionOverlayView overlayView;
    private FrameLayout previewFrame;
    private Button playPauseButton;
    private TextView statusText;
    private Spinner sourceSpinner;
    private Spinner backendSpinner;
    private ClipPlayer clipPlayer;

    private final Runnable refreshStatus = new Runnable()
    {
        @Override
        public void run()
        {
            showStats();
            mainHandler.postDelayed(this, STATUS_INTERVAL_MS);
        }
    };

    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        previewFrame = findViewById(R.id.previewFrame);
        cameraView = findViewById(R.id.cameraView);
        imageView = findViewById(R.id.imageView);
        clipView = findViewById(R.id.clipView);
        overlayView = findViewById(R.id.overlayView);
        playPauseButton = findViewById(R.id.playPauseButton);
        statusText = findViewById(R.id.statusText);
        sourceSpinner = findViewById(R.id.sourceSpinner);
        backendSpinner = findViewById(R.id.backendSpinner);
        clipPlayer = new ClipPlayer(this, clipView, previewFrame, worker, detector, this);

        cameraView.getHolder().addCallback(this);
        clipView.setSurfaceTextureListener(new OpenPendingClip());
        findViewById(R.id.chooseButton).setOnClickListener(v -> chooseMedia());
        playPauseButton.setOnClickListener(v -> toggleClipPlayback());
        findViewById(R.id.fixturesButton).setOnClickListener(v -> runFixtures());
        bindSpinner(R.id.sourceSpinner, this::switchSource);
        bindSpinner(R.id.backendSpinner, this::loadModel);

        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
            requestPermissions(new String[]{Manifest.permission.CAMERA}, REQUEST_CAMERA);
        applyLaunchExtras(getIntent());
    }

    @Override
    protected void onNewIntent(Intent intent)
    {
        super.onNewIntent(intent);
        applyLaunchExtras(intent);
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if (source == SOURCE_CAMERA)
            openRearCamera();
        mainHandler.post(refreshStatus);
    }

    @Override
    public void onPause()
    {
        super.onPause();
        mainHandler.removeCallbacks(refreshStatus);
        closeRearCamera();
        clipPlayer.pause();
        playPauseButton.setText(R.string.play);
    }

    @Override
    public void onDestroy()
    {
        super.onDestroy();
        clipPlayer.release();
        worker.shutdown();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] results)
    {
        if (requestCode == REQUEST_CAMERA && source == SOURCE_CAMERA)
            openRearCamera();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        if (resultCode != RESULT_OK || data == null || data.getData() == null)
            return;
        if (requestCode == REQUEST_PICK_IMAGE)
            detectPickedImage(data.getData());
        else if (requestCode == REQUEST_PICK_CLIP)
            playPickedClip(data.getData());
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder)
    {
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height)
    {
        detector.setOutputWindow(holder.getSurface());
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder)
    {
        detector.setOutputWindow(null);
    }

    @Override
    public void onClipDetections(float[] boxes, int frameWidth, int frameHeight)
    {
        overlayView.showDetections(boxes, frameWidth, frameHeight);
    }

    /** Selects the source and backend from the launch extras and, when asked, runs the fixtures once that model is loaded. */
    private void applyLaunchExtras(Intent intent)
    {
        sourceSpinner.setSelection(intent.getIntExtra(EXTRA_SOURCE, sourceSpinner.getSelectedItemPosition()));
        int backend = intent.getIntExtra(EXTRA_BACKEND, backendSpinner.getSelectedItemPosition());
        runFixturesAfterLoad = intent.getBooleanExtra(EXTRA_RUN_FIXTURES, false);
        if (backend != backendSpinner.getSelectedItemPosition())
            backendSpinner.setSelection(backend);
        else if (runFixturesAfterLoad && !backendLabel.isEmpty())
            runFixtures();
    }

    private interface IndexAction
    {
        void apply(int index);
    }

    private void bindSpinner(int spinnerId, IndexAction action)
    {
        Spinner spinner = findViewById(spinnerId);
        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener()
        {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id)
            {
                action.apply(position);
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent)
            {
            }
        });
    }

    /** Loads the bundled model with one backend choice and records the backend that actually ran. */
    private void loadModel(int backendChoice)
    {
        boolean requestGpu = BACKEND_CHOICES[backendChoice][0];
        boolean useFp16 = BACKEND_CHOICES[backendChoice][1];
        backendLabel = "";
        showNotice("loading model...");
        worker.execute(() -> {
            String label;
            try
            {
                if (contract == null)
                    contract = ModelContract.read(getAssets(), MODEL_DIR);
                int backend = contract.load(detector, getAssets(), requestGpu, useFp16);
                label = backend == BirdDetectNative.BACKEND_FAILED ? ""
                        : String.format(Locale.US, "ncnn-%s-%s", backend == BirdDetectNative.BACKEND_GPU ? "gpu" : "cpu",
                        useFp16 ? "fp16" : "fp32");
            }
            catch (Exception e)
            {
                Log.e(TAG, "model load failed", e);
                label = "";
            }
            final String loaded = label;
            mainHandler.post(() -> {
                backendLabel = loaded;
                Log.i(TAG, "requested gpu=" + requestGpu + " fp16=" + useFp16 + " ran=" + loaded);
                showNotice(loaded.isEmpty() ? "model load failed; see logcat" : "");
                if (runFixturesAfterLoad && !loaded.isEmpty())
                    runFixtures();
            });
        });
    }

    private void switchSource(int newSource)
    {
        source = newSource;
        closeRearCamera();
        clipPlayer.release();
        overlayView.clearDetections();
        cameraView.setVisibility(source == SOURCE_CAMERA ? View.VISIBLE : View.GONE);
        imageView.setVisibility(source == SOURCE_IMAGE ? View.VISIBLE : View.GONE);
        clipView.setVisibility(source == SOURCE_CLIP ? View.VISIBLE : View.GONE);
        overlayView.setVisibility(source == SOURCE_CAMERA ? View.GONE : View.VISIBLE);
        playPauseButton.setEnabled(source == SOURCE_CLIP);
        if (source == SOURCE_CAMERA)
            openRearCamera();
    }

    private void openRearCamera()
    {
        if (cameraOpen || checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
            return;
        cameraOpen = detector.openCamera(CAMERA_FACING_BACK);
    }

    private void closeRearCamera()
    {
        detector.closeCamera();
        cameraOpen = false;
    }

    private void chooseMedia()
    {
        if (source == SOURCE_CAMERA)
            return;
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType(source == SOURCE_IMAGE ? "image/*" : "video/*");
        startActivityForResult(intent, source == SOURCE_IMAGE ? REQUEST_PICK_IMAGE : REQUEST_PICK_CLIP);
    }

    /** Decodes the still upright (EXIF orientation applied), detects once, and shows the boxes. */
    private void detectPickedImage(Uri image)
    {
        worker.execute(() -> {
            try
            {
                Bitmap bitmap = ImageDecoder.decodeBitmap(ImageDecoder.createSource(getContentResolver(), image),
                        (decoder, info, src) -> {
                            decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
                            Size size = info.getSize();
                            float scale = Math.min(1f, (float) MAX_IMAGE_SIDE / Math.max(size.getWidth(), size.getHeight()));
                            if (scale < 1f)
                                decoder.setTargetSize(Math.round(size.getWidth() * scale), Math.round(size.getHeight() * scale));
                        });
                Bitmap argb = bitmap.getConfig() == Bitmap.Config.ARGB_8888 ? bitmap : bitmap.copy(Bitmap.Config.ARGB_8888, false);
                float[] boxes = detector.detectBitmap(argb);
                mainHandler.post(() -> showStill(argb, boxes));
            }
            catch (IOException e)
            {
                Log.e(TAG, "cannot decode picked image", e);
                mainHandler.post(() -> showNotice("cannot decode image: " + e.getMessage()));
            }
        });
    }

    private void showStill(Bitmap image, float[] boxes)
    {
        imageView.setImageBitmap(image);
        overlayView.showDetections(boxes, image.getWidth(), image.getHeight());
    }

    private void playPickedClip(Uri clip)
    {
        pendingClip = clip;
        if (clipView.isAvailable())
            openPendingClip();
    }

    private void openPendingClip()
    {
        if (pendingClip == null)
            return;
        try
        {
            clipPlayer.open(pendingClip);
            playPauseButton.setText(R.string.pause);
        }
        catch (IOException e)
        {
            Log.e(TAG, "cannot open clip", e);
            showNotice("cannot open clip: " + e.getMessage());
        }
        pendingClip = null;
    }

    private void toggleClipPlayback()
    {
        if (clipPlayer.isPlaying())
        {
            clipPlayer.pause();
            playPauseButton.setText(R.string.play);
        }
        else
        {
            clipPlayer.resume();
            playPauseButton.setText(R.string.pause);
        }
    }

    /** Runs the parity fixtures with the loaded backend and writes predictions for adb pull. */
    private void runFixtures()
    {
        runFixturesAfterLoad = false;
        if (backendLabel.isEmpty())
            return;
        if (source == SOURCE_CAMERA)
            sourceSpinner.setSelection(SOURCE_IMAGE);
        final String label = backendLabel;
        worker.execute(() -> {
            try
            {
                FixtureRun.Result result = FixtureRun.predictFixtures(detector, getAssets(), getExternalFilesDir(null), label);
                Log.i(TAG, "fixture predictions " + result.predictionsFile.getAbsolutePath());
                mainHandler.post(() -> {
                    showStill(result.lastImage, result.lastBoxes);
                    showNotice("wrote " + result.predictionsFile.getName());
                });
            }
            catch (Exception e)
            {
                Log.e(TAG, "fixture run failed", e);
                mainHandler.post(() -> showNotice("fixture run failed: " + e.getMessage()));
            }
        });
    }

    private void showNotice(String message)
    {
        notice = message;
        statusText.setText(message);
    }

    private void showStats()
    {
        if (backendLabel.isEmpty())
            return;
        float[] stats = detector.readStats();
        String rate = source == SOURCE_IMAGE ? "" : String.format(Locale.US, " | %.1f FPS", stats[4]);
        String status = String.format(Locale.US, "%s%s | %.1f ms (pre %.1f, infer %.1f, post %.1f) | birds %d\n%s\n%s",
                backendLabel, rate, stats[3], stats[0], stats[1], stats[2], (int) stats[5], Build.MODEL, notice);
        statusText.setText(status);
        if (++statusRefreshCount % STATUS_LOG_EVERY == 0)
            Log.i(TAG, "status " + status.replace('\n', ' '));
    }

    private class OpenPendingClip implements TextureView.SurfaceTextureListener
    {
        @Override
        public void onSurfaceTextureAvailable(SurfaceTexture surface, int width, int height)
        {
            openPendingClip();
        }

        @Override
        public void onSurfaceTextureSizeChanged(SurfaceTexture surface, int width, int height)
        {
        }

        @Override
        public boolean onSurfaceTextureDestroyed(SurfaceTexture surface)
        {
            clipPlayer.release();
            return true;
        }

        @Override
        public void onSurfaceTextureUpdated(SurfaceTexture surface)
        {
        }
    }
}
