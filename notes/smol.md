# Llama and SmolVLM2 Support Plan

This document tracks the remaining work to move the project from a Qwen-VL-specific demo toward a small model-family runtime that can support text-only llama models and SmolVLM2-style multimodal models on Rockchip devices.

## Goals

- Keep the existing RKNN/RKLLM Qwen-VL path working.
- Add text-only llama support through RKLLM without requiring a vision encoder.
- Add SmolVLM2 support once a converted RKNN vision encoder and RKLLM decoder pair is available.
- Make model-specific prompt tokens, image placeholders, preprocessing, and validation explicit instead of hard-coded.

## Current State

- The CLI now uses named arguments for model and input paths: `--vision`, `--llm`, `--image`, and optional `--prompt`.
- `ModelFamily` currently only contains Qwen-VL variants.
- `Session::init()` always initializes both the RKLLM text decoder and RKNN vision encoder.
- `Session::is_ready()` requires both the decoder and vision encoder to be ready.
- Multimodal inference is selected by checking for the literal `<image>` marker.
- RKLLM multimodal tokens are hard-coded to Qwen-style values:
  - `<|vision_start|>`
  - `<|vision_end|>`
  - `<|image_pad|>`
- Image preprocessing is fixed to the current Qwen-VL path: BGR to RGB, pad to square on grey, resize to the encoder input size.
- Android JNI and Java wrappers still assume one vision encoder path plus one language model path.

## Phase 1: Model Profiles

Introduce model profiles as the single source of truth for family-specific behavior.

Proposed data:

```cpp
enum class ModelFamily {
    QwenVL2,
    QwenVL2_5,
    QwenVL3,
    Llama,
    SmolVLM2,
};

struct ModelProfile {
    bool uses_vision_encoder;
    bool supports_multimodal;
    const char* image_placeholder;
    const char* img_start;
    const char* img_end;
    const char* img_content;
    ImagePreprocessProfile image_preprocess;
};
```

Initial behavior:

- Qwen profiles keep the existing behavior.
- Llama has `uses_vision_encoder = false` and `supports_multimodal = false`.
- SmolVLM2 is added as experimental until its converted RKNN/RKLLM model pair is verified.

Tasks:

- Extend `parse_model_family()` with `llama`, `smolvlm2`, and useful aliases.
- Replace the current Qwen-only `ModelProfile` with a profile that includes text-only and multimodal capability flags.
- Update `Session::describe()` so it reports model family and whether a vision encoder is required.

## Phase 2: Text-Only Llama

Add llama support before SmolVLM2 because it exercises the RKLLM-only path with minimal changes.

Tasks:

- Make `ModelConfig::vision_encoder_path` optional for text-only profiles.
- Change `Session::init()` so it skips `init_vision_encoder()` when the active profile does not use vision.
- Change `Session::is_ready()` so llama only requires the RKLLM decoder handle.
- Change CLI validation so `--vision` and `--image` are only required for profiles that use the vision encoder.
- Allow llama invocations such as:

```bash
./build-native/qwen-vl-rknn \
  --model-family llama \
  --llm /path/to/llama.rkllm \
  --prompt "Explain RKNN in one paragraph."
```

- Start the REPL for text-only llama when `--prompt` is omitted.
- Ensure text-only decoding always uses `RKLLM_INPUT_PROMPT` with a null image embedding.

Verification:

- CLI rejects `--model-family llama --image ...` only if the combination is unsupported or ambiguous.
- CLI accepts llama without `--vision` and without `--image`.
- Existing Qwen-VL command lines still require `--vision` and `--image`.

## Phase 3: Multimodal Prompt Handling

Replace hard-coded `<image>` checks with profile-specific marker logic.

Tasks:

- Add a helper such as `Session::prompt_contains_image(const std::string&)`.
- Use `profile.image_placeholder` instead of the literal `<image>`.
- Validate that multimodal prompts have an encoded image before calling `rkllm_run`.
- Keep plain text prompts working for multimodal models.
- Decide whether the CLI should automatically prepend the image placeholder for multimodal runs when the user provides a prompt without one.

Open decision:

- For Qwen-VL compatibility, preserving explicit `<image>` prompts is safest.
- For user ergonomics, an optional `--auto-image-prefix` flag could be added later.

## Phase 4: Image Preprocessing Profiles

Move image preprocessing out of CLI-specific assumptions and into model profiles.

Proposed data:

```cpp
enum class ResizeMode {
    PadToSquare,
    Stretch,
    CenterCrop,
};

struct ImagePreprocessProfile {
    ResizeMode resize_mode;
    bool rgb;
    bool normalize_in_host;
    float pad_r;
    float pad_g;
    float pad_b;
    std::array<float, 3> mean;
    std::array<float, 3> std;
};
```

Tasks:

- Extract CLI image preprocessing into reusable core helpers.
- Preserve the current Qwen-VL behavior as the Qwen profile.
- Add a SmolVLM2 profile, but leave normalization/layout fields guarded by explicit TODOs until confirmed from the converted model.
- Keep Android preprocessing aligned with the C++ path, or move more preprocessing into native code to avoid duplicating model-specific behavior in Java.

SmolVLM2 questions to answer:

- Does the exported RKNN vision model expect raw RGB bytes, normalized floats, or preprocessing baked into the graph?
- Is the input layout NHWC or NCHW?
- Does the conversion include SigLIP resizing and normalization?
- What image size does the converted model require?

## Phase 5: SmolVLM2 Runtime Support

SmolVLM2 should not be marked fully supported until a converted model pair is available and tested.

Known architecture constraints:

- SmolVLM2 is a multimodal model using a SigLIP-style image encoder and SmolLM2 text decoder.
- SmolVLM2 can support image, multi-image, video, and text tasks in its upstream Transformers form.
- This project should start with single-image plus text support only.
- Video and multi-image support should be separate follow-up work.

Tasks:

- Add `ModelFamily::SmolVLM2`.
- Add an experimental SmolVLM2 profile with placeholder fields for RKLLM image tokens.
- Identify the exact RKLLM multimodal token strings produced by the conversion process.
- Confirm whether RKLLM accepts SmolVLM2 through `RKLLM_INPUT_MULTIMODAL` in the same shape as Qwen-VL.
- Confirm `n_image_tokens`, embedding width, and output tensor ordering from RKNN metadata.
- Add startup logging for all RKNN input and output tensor attributes to make converted model mismatches visible.
- Document the expected files:

```text
smolvlm2_vision.rknn
smolvlm2_language.rkllm
```

Initial CLI target:

```bash
./build-native/qwen-vl-rknn \
  --model-family smolvlm2 \
  --vision /path/to/smolvlm2_vision.rknn \
  --llm /path/to/smolvlm2_language.rkllm \
  --image data/cell.png \
  --prompt "<image>What is shown?"
```

The final prompt marker may differ. It must come from the converted tokenizer or RKLLM model requirements, not from guessing.

## Phase 6: Android API and UI

Android should expose the same model-family choices as the CLI.

Tasks:

- Change or overload `RknnLlm.init()` to accept a model family:

```java
init(String modelFamily, String encoderPath, String llmPath)
```

- Allow `encoderPath` to be empty for llama.
- Add a model-family selector to the test frontend.
- Disable image loading when the selected profile is text-only.
- Move image preprocessing to native code if profile-specific preprocessing becomes too complex to keep mirrored in Java.
- Update JNI validation to use profile-specific image placeholder checks instead of hard-coded `<image>`.

## Phase 7: Tests

The current host-side tests are placeholders and should be corrected as part of this work.

Tasks:

- Fix the current `is_ready()` expectation: constructing a `Session` should not make it ready before `init()`.
- Add tests for model-family parsing.
- Add tests for profile properties:
  - llama does not require a vision encoder.
  - Qwen-VL requires a vision encoder.
  - SmolVLM2 is registered as experimental.
- Add tests for CLI validation once parser logic is split into a testable helper.
- Add image preprocessing tests for square, portrait, and landscape inputs.
- Add a smoke-test hook that can be enabled when tiny or fixture RKNN/RKLLM models are available.

## Phase 8: Documentation

Tasks:

- Update `README.md` with separate examples for:
  - Qwen-VL
  - text-only llama
  - experimental SmolVLM2
- Add a support matrix:

| Family | RKLLM decoder | RKNN vision | Text-only | Single image | Status |
|---|---:|---:|---:|---:|---|
| Qwen2-VL | yes | yes | yes | yes | current target |
| Qwen2.5-VL | yes | yes | yes | yes | profile present, needs verification |
| Qwen3-VL | yes | yes | yes | yes | profile present, needs verification |
| Llama | yes | no | yes | no | planned |
| SmolVLM2 | yes | yes | yes | planned | experimental until converted model is tested |

- Document converted model requirements and exact source model revisions.
- Document unsupported modes explicitly:
  - SmolVLM2 video input
  - SmolVLM2 multi-image input
  - non-Rockchip backends

## Verification Gate

After any C++ or build-system change, attempt both project build commands before finalizing:

```bash
docker compose run --rm native ./scripts/build-native.sh Release
docker compose run --rm android ./scripts/build-android.sh Release
```

If Docker or the build environment is unavailable, report the exact command attempted and the environment failure.

SmolVLM2 should only move from experimental to supported after:

- The converted RKNN vision encoder initializes successfully.
- RKNN tensor metadata matches the profile.
- The RKLLM decoder initializes with the chosen image token strings.
- A single-image prompt produces plausible output on an RK3588 device.
- Native and Android builds have both been attempted.
