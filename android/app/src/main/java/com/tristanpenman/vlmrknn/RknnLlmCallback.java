package com.tristanpenman.vlmrknn;

import androidx.annotation.NonNull;

/**
 * Receives streaming events from {@link RknnLlm#run}. Methods are
 * invoked synchronously on the thread that called {@code run}, so
 * implementations should not block.
 *
 * <p>The contract is: zero or more {@link #onText} calls, followed
 * by exactly one of {@link #onFinish} or {@link #onError}.
 */
public interface RknnLlmCallback {

  /** A chunk of generated text is available. */
  void onText(@NonNull String chunk);

  /** Generation completed successfully. */
  void onFinish();

  /** Generation failed. */
  void onError();
}
