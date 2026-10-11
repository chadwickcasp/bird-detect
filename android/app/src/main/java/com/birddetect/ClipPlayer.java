package com.birddetect;

import android.content.Context;
import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.media.MediaPlayer;
import android.net.Uri;
import android.os.Handler;
import android.os.Looper;
import android.view.Choreographer;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.widget.FrameLayout;

import java.io.IOException;
import java.util.concurrent.Executor;

/**
 * Plays a picked clip on a TextureView and detects birds in the latest shown frame.
 * A new frame is copied only while the detector is idle, so slow inference drops frames instead of queueing them.
 */
public class ClipPlayer implements Choreographer.FrameCallback
{
    /** Receives boxes in analysis-frame pixels on the main thread. */
    public interface Listener
    {
        void onClipDetections(float[] boxes, int frameWidth, int frameHeight);
    }

    private static final int MAX_ANALYSIS_SIDE = 1920;

    private final Context context;
    private final TextureView textureView;
    private final View container;
    private final Executor worker;
    private final BirdDetectNative detector;
    private final Listener listener;
    private final Handler mainHandler = new Handler(Looper.getMainLooper());
    private MediaPlayer player;
    private Surface playerSurface;
    private Bitmap analysisFrame;
    private boolean detectorBusy;
    private boolean frameLoopRunning;

    public ClipPlayer(Context context, TextureView textureView, View container, Executor worker,
                      BirdDetectNative detector, Listener listener)
    {
        this.context = context;
        this.textureView = textureView;
        this.container = container;
        this.worker = worker;
        this.detector = detector;
        this.listener = listener;
    }

    /** Starts looping playback of the clip; the TextureView must already be available. */
    public void open(Uri clip) throws IOException
    {
        release();
        int[] displaySize = readDisplaySize(clip);
        fitTextureView(displaySize[0], displaySize[1]);
        analysisFrame = createAnalysisFrame(displaySize[0], displaySize[1]);

        player = new MediaPlayer();
        player.setDataSource(context, clip);
        playerSurface = new Surface(textureView.getSurfaceTexture());
        player.setSurface(playerSurface);
        player.setLooping(true);
        player.setOnPreparedListener(prepared -> resume());
        player.prepareAsync();
    }

    public boolean isPlaying()
    {
        return player != null && player.isPlaying();
    }

    public void pause()
    {
        if (player != null && player.isPlaying())
            player.pause();
        stopFrameLoop();
    }

    public void resume()
    {
        if (player == null)
            return;
        player.start();
        if (!frameLoopRunning)
        {
            frameLoopRunning = true;
            Choreographer.getInstance().postFrameCallback(this);
        }
    }

    public void release()
    {
        stopFrameLoop();
        if (player != null)
        {
            player.release();
            player = null;
        }
        if (playerSurface != null)
        {
            playerSurface.release();
            playerSurface = null;
        }
    }

    @Override
    public void doFrame(long frameTimeNanos)
    {
        if (!frameLoopRunning)
            return;
        Choreographer.getInstance().postFrameCallback(this);
        if (detectorBusy || player == null || !player.isPlaying() || !textureView.isAvailable())
            return;

        final Bitmap frame = textureView.getBitmap(analysisFrame);
        detectorBusy = true;
        worker.execute(() -> {
            float[] boxes = detector.detectBitmap(frame);
            mainHandler.post(() -> {
                detectorBusy = false;
                if (frameLoopRunning)
                    listener.onClipDetections(boxes, frame.getWidth(), frame.getHeight());
            });
        });
    }

    private void stopFrameLoop()
    {
        frameLoopRunning = false;
        Choreographer.getInstance().removeFrameCallback(this);
    }

    /** Returns the clip's upright width and height after its rotation tag. */
    private int[] readDisplaySize(Uri clip) throws IOException
    {
        try (MediaMetadataRetriever retriever = new MediaMetadataRetriever())
        {
            retriever.setDataSource(context, clip);
            int width = Integer.parseInt(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_WIDTH));
            int height = Integer.parseInt(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_HEIGHT));
            String rotation = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_ROTATION);
            boolean sideways = rotation != null && (rotation.equals("90") || rotation.equals("270"));
            return sideways ? new int[]{height, width} : new int[]{width, height};
        }
        catch (RuntimeException e)
        {
            throw new IOException("cannot read clip size", e);
        }
    }

    /** Sizes the TextureView to the clip aspect, centered in the container, so the overlay's fit-center mapping matches. */
    private void fitTextureView(int clipWidth, int clipHeight)
    {
        float scale = Math.min((float) container.getWidth() / clipWidth, (float) container.getHeight() / clipHeight);
        FrameLayout.LayoutParams params = new FrameLayout.LayoutParams(
                Math.round(clipWidth * scale), Math.round(clipHeight * scale), android.view.Gravity.CENTER);
        textureView.setLayoutParams(params);
    }

    private static Bitmap createAnalysisFrame(int clipWidth, int clipHeight)
    {
        float scale = Math.min(1f, (float) MAX_ANALYSIS_SIDE / Math.max(clipWidth, clipHeight));
        return Bitmap.createBitmap(Math.round(clipWidth * scale), Math.round(clipHeight * scale),
                Bitmap.Config.ARGB_8888);
    }
}
