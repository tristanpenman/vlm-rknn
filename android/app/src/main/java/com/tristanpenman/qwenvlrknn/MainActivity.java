package com.tristanpenman.qwenvlrknn;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends AppCompatActivity {

  private final ExecutorService executor = Executors.newSingleThreadExecutor();
  private final Handler mainHandler = new Handler(Looper.getMainLooper());

  private static final int PREVIEW_TARGET_PX = 512;

  private EditText encoderPathInput;
  private EditText llmPathInput;
  private EditText imagePathInput;
  private EditText promptInput;
  private TextView responseOutput;
  private TextView statusLabel;
  private Button initButton;
  private Button loadImageButton;
  private Button runButton;
  private ImageView imagePreview;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_main);

    encoderPathInput = findViewById(R.id.encoder_path);
    llmPathInput = findViewById(R.id.llm_path);
    imagePathInput = findViewById(R.id.image_path);
    promptInput = findViewById(R.id.prompt_input);
    responseOutput = findViewById(R.id.response_output);
    statusLabel = findViewById(R.id.status_label);
    initButton = findViewById(R.id.init_button);
    loadImageButton = findViewById(R.id.load_image_button);
    runButton = findViewById(R.id.run_button);
    imagePreview = findViewById(R.id.image_preview);

    initButton.setOnClickListener(v -> onInitClicked());
    loadImageButton.setOnClickListener(v -> onLoadImageClicked());
    runButton.setOnClickListener(v -> onRunClicked());

    loadImageButton.setEnabled(RknnLlm.isInitialised());
    runButton.setEnabled(RknnLlm.isInitialised());

  }

  @Override
  protected void onDestroy() {
    super.onDestroy();
    if (isFinishing()) {
      RknnLlm.cleanup();
    }
    executor.shutdown();
  }

  private void onInitClicked() {
    String encoderPath = encoderPathInput.getText().toString().trim();
    if (encoderPath.isEmpty()) {
      encoderPath = getString(R.string.hint_encoder_path);
    }
    String llmPath = llmPathInput.getText().toString().trim();
    if (llmPath.isEmpty()) {
      llmPath = getString(R.string.hint_llm_path);
    }

    setBusy(true, getString(R.string.status_loading_model));

    final String finalEncoderPath = encoderPath;
    final String finalLlmPath = llmPath;
    executor.execute(() -> {
      final int rc = RknnLlm.init(finalEncoderPath, finalLlmPath);
      mainHandler.post(() -> {
        if (rc == 0) {
          setBusy(false, getString(R.string.status_model_loaded));
        } else {
          RknnLlm.cleanup();
          setBusy(false, getString(R.string.status_init_failed, rc));
        }
      });
    });
  }

  private void onLoadImageClicked() {
    if (!RknnLlm.isInitialised()) {
      statusLabel.setText(R.string.status_not_initialised);
      return;
    }
    String imagePath = imagePathInput.getText().toString().trim();
    if (imagePath.isEmpty()) {
      imagePath = getString(R.string.hint_image_path);
    }

    setBusy(true, getString(R.string.status_loading_image));

    final String finalImagePath = imagePath;
    executor.execute(() -> {
      final int rc = RknnLlm.loadImage(finalImagePath);
      final Bitmap preview = (rc == 0) ? decodePreview(finalImagePath) : null;
      mainHandler.post(() -> {
        if (rc == 0) {
          if (preview != null) {
            imagePreview.setImageBitmap(preview);
            imagePreview.setVisibility(View.VISIBLE);
          }
          setBusy(false, getString(R.string.status_image_loaded));
        } else {
          imagePreview.setImageDrawable(null);
          imagePreview.setVisibility(View.GONE);
          setBusy(false, getString(R.string.status_image_failed, rc));
        }
      });
    });
  }

  private static Bitmap decodePreview(String imagePath) {
    final BitmapFactory.Options bounds = new BitmapFactory.Options();
    bounds.inJustDecodeBounds = true;
    BitmapFactory.decodeFile(imagePath, bounds);
    if (bounds.outWidth <= 0 || bounds.outHeight <= 0) {
      return null;
    }
    int sample = 1;
    int longest = Math.max(bounds.outWidth, bounds.outHeight);
    while (longest / (sample * 2) >= PREVIEW_TARGET_PX) {
      sample *= 2;
    }
    final BitmapFactory.Options opts = new BitmapFactory.Options();
    opts.inSampleSize = sample;
    return BitmapFactory.decodeFile(imagePath, opts);
  }

  private void onRunClicked() {
    if (!RknnLlm.isInitialised()) {
      statusLabel.setText(R.string.status_not_initialised);
      return;
    }
    final String prompt = promptInput.getText().toString();
    if (prompt.isEmpty()) {
      statusLabel.setText(R.string.status_prompt_required);
      return;
    }

    setBusy(true, getString(R.string.status_running));
    responseOutput.setText("");

    final RknnLlmCallback callback = new RknnLlmCallback() {
      @Override
      public void onText(@NonNull String chunk) {
        mainHandler.post(() -> responseOutput.append(chunk));
      }

      @Override
      public void onFinish() {
        mainHandler.post(() -> setBusy(false, getString(R.string.status_done)));
      }

      @Override
      public void onError() {
        mainHandler.post(() -> setBusy(false, getString(R.string.status_run_failed)));
      }
    };

    executor.execute(() -> {
      final int rc = RknnLlm.run(prompt, callback);
      if (rc != 0) {
        mainHandler.post(() -> setBusy(false, getString(R.string.status_run_failed)));
      }
    });
  }

  private void setBusy(boolean busy, CharSequence statusText) {
    final boolean ready = RknnLlm.isInitialised();
    initButton.setEnabled(!busy);
    loadImageButton.setEnabled(!busy && ready);
    runButton.setEnabled(!busy && ready);
    statusLabel.setText(statusText);
  }
}
