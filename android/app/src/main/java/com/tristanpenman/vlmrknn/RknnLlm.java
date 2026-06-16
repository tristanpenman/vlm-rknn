package com.tristanpenman.vlmrknn;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;

/**
 * Thin Java surface over the JNI wrapper around this repo's VLM RKNN core.
 *
 * <p>This is a process-wide singleton: only one model pair can be
 * loaded at a time. Call {@link #init} before {@link #run}, then
 * optionally {@link #loadImage} to attach an image to subsequent
 * prompts that contain the active model's image marker. Call
 * {@link #cleanup} when done.
 */
public final class RknnLlm {
  static {
    System.loadLibrary("vlm-rknn-jni");
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
   * Load an image from the given filesystem path, preprocess it using
   * the active native model profile, encode it, and cache the embeddings
   * for use by subsequent {@link #run} calls whose prompt contains the
   * model's image marker.
   *
   * @return 0 on success, non-zero on failure.
   */
  public static synchronized int loadImage(@NonNull final String imagePath) {
    if (!initialised) {
      return -1;
    }
    final int rc = nativeSetImage(imagePath);
    hasImage = (rc == 0);
    return rc;
  }

  /**
   * Run a single inference with the given prompt. If an image has
   * been loaded and the prompt contains the active model's image marker, runs
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

  private static native int nativeInit(String encoderPath, String llmPath);
  private static native int nativeSetImage(String imagePath);
  private static native int nativeRun(String prompt, RknnLlmCallback callback);
  private static native void nativeCleanup();
}
