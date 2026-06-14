package com.tristanpenman.qwenvlrknn;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.Rect;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

/**
 * Thin Java surface over the JNI wrapper around this repo's Qwen-VL RKNN core.
 *
 * <p>This is a process-wide singleton: only one model pair can be
 * loaded at a time. Call {@link #init} before {@link #run}, then
 * optionally {@link #loadImage} to attach an image to subsequent
 * prompts that contain the {@code <image>} marker. Call
 * {@link #cleanup} when done.
 */
public final class RknnLlm {
  static {
    System.loadLibrary("qwen-vl-rknn-jni");
  }

  private RknnLlm() {}

  private static volatile boolean initialised = false;
  private static volatile boolean hasImage = false;

  /**
   * Load the vision encoder and LLM from the given filesystem paths.
   * If a model is already loaded it is released first.
   *
   * @return 0 on success, non-zero on failure.
   */
  public static synchronized int init(@NonNull final String encoderPath,
                                      @NonNull final String llmPath) {
    if (initialised) {
      nativeCleanup();
      initialised = false;
      hasImage = false;
    }
    final int rc = nativeInit(encoderPath, llmPath);
    if (rc == 0) {
      initialised = true;
    }
    return rc;
  }

  /** Whether a model pair has been successfully loaded. */
  public static boolean isInitialised() {
    return initialised;
  }

  /** Whether an image has been encoded and is ready for multimodal queries. */
  public static boolean hasImage() {
    return hasImage;
  }

  /**
   * Load an image from the given filesystem path, preprocess it
   * (centre-pad to a square on a mid-grey background, resize to the
   * encoder's expected input dimensions), encode it, and cache the
   * embeddings for use by subsequent {@link #run} calls whose prompt
   * contains the {@code <image>} marker.
   *
   * @return 0 on success, non-zero on failure.
   */
  public static synchronized int loadImage(@NonNull final String imagePath) {
    if (!initialised) {
      return -1;
    }
    final int[] dims = nativeGetImageInputSize();
    if (dims == null || dims.length != 2 || dims[0] <= 0 || dims[1] <= 0) {
      return -1;
    }
    final int targetWidth = dims[0];
    final int targetHeight = dims[1];

    final Bitmap original = BitmapFactory.decodeFile(imagePath);
    if (original == null) {
      return -1;
    }

    Bitmap preprocessed = null;
    try {
      preprocessed = expandAndResize(original, targetWidth, targetHeight);
      final byte[] pixels = bitmapToRgb(preprocessed);
      final int rc = nativeSetImage(pixels, targetWidth, targetHeight);
      hasImage = (rc == 0);
      return rc;
    } finally {
      if (preprocessed != null && preprocessed != original) {
        preprocessed.recycle();
      }
      original.recycle();
    }
  }

  /**
   * Run a single inference with the given prompt. If an image has
   * been loaded and the prompt contains {@code <image>}, runs
   * multimodal inference; otherwise plain-text. Blocks until the
   * generation completes, dispatching callbacks on the calling
   * thread as text is produced.
   *
   * @param prompt   the user prompt to send to the model.
   * @param callback receives streaming events; may be {@code null}.
   * @return 0 on success, non-zero on failure.
   */
  public static int run(@NonNull final String prompt,
                        @Nullable final RknnLlmCallback callback) {
    return nativeRun(prompt, callback);
  }

  /**
   * Release resources held by the loaded models. Safe to call
   * repeatedly or before {@link #init}.
   */
  public static synchronized void cleanup() {
    if (initialised) {
      nativeCleanup();
      initialised = false;
      hasImage = false;
    }
  }

  private static Bitmap expandAndResize(Bitmap src, int targetWidth, int targetHeight) {
    final int srcWidth = src.getWidth();
    final int srcHeight = src.getHeight();
    final int size = Math.max(srcWidth, srcHeight);

    // Centre-pad to a square on a mid-grey background (matches the
    // expand2square step in the upstream multimodal demo).
    final Bitmap square = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);
    final Canvas canvas = new Canvas(square);
    canvas.drawColor(0xFF7F7F7F);
    final int xOffset = (size - srcWidth) / 2;
    final int yOffset = (size - srcHeight) / 2;
    canvas.drawBitmap(src, null,
        new Rect(xOffset, yOffset, xOffset + srcWidth, yOffset + srcHeight),
        new Paint(Paint.FILTER_BITMAP_FLAG));

    if (size == targetWidth && size == targetHeight) {
      return square;
    }
    final Bitmap resized = Bitmap.createScaledBitmap(square, targetWidth, targetHeight, true);
    if (resized != square) {
      square.recycle();
    }
    return resized;
  }

  private static byte[] bitmapToRgb(Bitmap bitmap) {
    final int width = bitmap.getWidth();
    final int height = bitmap.getHeight();
    final int[] argb = new int[width * height];
    bitmap.getPixels(argb, 0, width, 0, 0, width, height);
    final byte[] rgb = new byte[width * height * 3];
    for (int i = 0; i < argb.length; i++) {
      final int p = argb[i];
      rgb[i * 3]     = (byte) ((p >> 16) & 0xFF);  // R
      rgb[i * 3 + 1] = (byte) ((p >> 8) & 0xFF);   // G
      rgb[i * 3 + 2] = (byte) (p & 0xFF);          // B
    }
    return rgb;
  }

  private static native int nativeInit(String encoderPath, String llmPath);
  private static native int[] nativeGetImageInputSize();
  private static native int nativeSetImage(byte[] rgbPixels, int width, int height);
  private static native int nativeRun(String prompt, RknnLlmCallback callback);
  private static native void nativeCleanup();
}
