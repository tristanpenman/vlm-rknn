# Gemma 4

These notes outline what is required to support Gemma 4 in this project.

The vendored RKLLM 1.3.1 runtime already supports Gemma 4. Rockchip's [`rknn-llm`](https://github.com/airockchip/rknn-llm) repository has an end-to-end demo for `gemma-4-E2B-it` with a vision encoder, an audio encoder and an `.rkllm` model. Most of the work is therefore converting the models and adding a new entry to the `ModelProfile` table in `cpp/src/vlm_rknn.cc`, following the same route as Gemma 3. No new runtime work is needed for text and image input.

## Differences from Gemma 3

| Aspect              | Gemma 3 (current code)                                       | Gemma 4 (Rockchip demo)                                                              |
|---------------------|--------------------------------------------------------------|--------------------------------------------------------------------------------------|
| Image tags          | `<start_of_image>` / `<end_of_image>` / `<image_soft_token>` | `<\|image>` / `<image\|>` / `<\|image\|>`                                            |
| Image input         | Fixed 896×896, stretched                                     | 448×448 by default; the export rounds up to multiples of 48 (448 → 480)              |
| Image tokens        | Fixed at 256, checked at init                                | Depends on resolution: Gemma 4 has a configurable token budget (70/140/280/560/1120) |
| Vision model inputs | 1 (pixels)                                                   | The ONNX export takes **2**: patchified pixels plus `pixel_position_ids` `[1, N, 2]` |
| Normalisation       | Inside the model                                             | Inside the model (the export only casts to float32)                                  |
| `base_domain_id`    | 0                                                            | Demo uses 1 on RK3588/RK3576 (set by platform, not model)                            |
| Chat template       | None                                                         | None (left to the runtime's template from the `.rkllm` file)                         |
| Audio               | n/a                                                          | Separate RKNN audio encoder; RKLLM 1.3.1 adds `multimodal_input.audio`               |

## Image preprocessing

Google's canonical Gemma 4 and Rockchip's fixed-shape RKNN export have different image input contracts. They should not be treated as the same preprocessing pipeline.

The canonical Hugging Face processor converts images to RGB and resizes them with bicubic interpolation while preserving their aspect ratio. The target dimensions are selected from a visual-token budget of 70, 140, 280, 560 or 1120 tokens. Both dimensions are multiples of 48 because the processor creates 16×16 patches and the vision model pools each 3×3 patch block. Pixels are rescaled from `[0, 255]` to `[0, 1]` without further mean/std normalisation. The processor then patchifies the image, generates two-dimensional position IDs and pads the patch sequence rather than padding the decoded image.

Rockchip adapts this variable-resolution interface to a static RKNN graph. Its Gemma 4 exporter defaults to a nominal 448×448 input and rounds each dimension up to a multiple of 48, producing a 480×480 graph with fixed position IDs. The RKNN conversion uses mean `[0, 0, 0]` and standard deviation `[255, 255, 255]`, so `/255` rescaling is performed inside the RKNN artifact. Rockchip's deployment demo converts BGR to RGB, pads non-square images with RGB `(127.5, 127.5, 127.5)`, and resizes that square to the graph's input dimensions.

This repository's `kGemma4RknnImagePreprocess` profile follows the Rockchip approach - RGB conversion, mid-grey square padding, resizing to the dimensions reported by the RKNN model, and no host-side normalisation. It is compatible only with an artifact produced by Rockchip's conversion scripts. Supporting canonical variable-resolution input would require dynamic image dimensions, position IDs, the actual image-token count and patch-sequence padding. This is out of scope for this project.

## Plan

The phases are listed in the order they should be done. Phase 2 can be written before Phase 1 is finished, but it cannot be validated until a converted `.rknn`/`.rkllm` pair runs on an RK3588.

### Phase 1: Model Conversion

This lives in `python/vlm_rknn/`.

1. **Update the toolchain.**
   - Add the `rkllm_toolkit` 1.3.1 wheel; only 1.2.3 and 1.3.0 are in `python/wheels/`.
   - Pin `transformers==5.8.0` and `onnx==1.22.0` in `python/requirements.txt`, as Rockchip's demo requires.
   - Check that `Dockerfile.python` still builds.
2. **Add `convert_gemma4.py`.**
   - Reuse the download, cache and `build()` logic from `convert_gemma3.py`, ideally by pulling it into a shared module.
   - Target `google/gemma-4-E2B-it` first, since that is the variant Rockchip has validated.
   - Start with FP16, as their demo does. Try W8A8 or hybrid quantization afterwards.
3. **Export the vision encoder.**
   - Port or vendor Rockchip's `export_vision_gemma4.py` and `export_vision_rknn.py` (ONNX → RKNN).
   - Choose one fixed resolution per `.rknn` file, since RKNN wants static shapes.
   - **Key decision:** fold `pixel_position_ids` into the graph as a constant for that resolution. The C++ side then stays single-input and uint8 NHWC, as it is today. Otherwise the C++ side has to patchify the image and feed position IDs itself.
4. Optionally, export the audio encoder (`export_audio_gemma4.py`). This can be deferred to Phase 5.

### Phase 2: C++ Text and Image Support

These changes are confined to the model-family plumbing.

1. **Add the new family.**
   - Add `ModelFamily::kGemma4` in `cpp/src/vlm_rknn.h`.
   - Add the parse aliases `gemma4`, `gemma-4` and `gemma_4`, plus `modelFamilyName`, in `cpp/src/vlm_rknn.cc`.
   - Update the usage and help strings in `cpp/src/main.cc`.
2. **Add a `kGemma4` `ModelProfile`** with:
   - the new image tags;
   - `expectedImageTokens = 0`, i.e. read the token count from the RKNN output shape, as SmolVLM and Qwen do;
   - `baseDomainId = 1`, to be checked on the device;
   - `useChatTemplate = false`.
3. **Add a preprocessing profile.**
   - Use a distinct `kGemma4RknnImagePreprocess` profile rather than aliasing the Qwen profile, even though both currently use `kPadToSquare` with a 127.5 grey pad.
   - Set `normalizeInHost = false`; Rockchip's RKNN conversion performs `/255` rescaling inside the model.
   - Keep this explicitly scoped to Rockchip's fixed-shape export. Canonical Gemma 4 preserves aspect ratio and pads patches, not image pixels.
4. **Generalise the init checks.**
   - The hard-coded Gemma 3 check for an 896×896 input should become profile-driven, or be skipped for Gemma 4.
   - The single-input/single-output check can stay if position IDs are folded in. If not, `encode()` needs a second input tensor.
5. **Tokenizer.**
   - `--tokenizer-model` uses SentencePiece and is only used to estimate token counts.
   - Check whether Gemma 4 ships a `tokenizer.model`. If it does not, document the flag as unsupported for this family.
6. **Tests.** Add parse and family assertions to `cpp/tests/vlm_rknn_test.cc`, mirroring the Gemma 3 ones.

### Phase 3: Device Validation

- Compare the RKNN vision output against PyTorch on the same 480×480 input, using cosine similarity per token.
- Run greedy decoding on `data/*.png` and compare with Hugging Face Transformers.
- Measure memory and speed for E2B FP16 against W8A8, and record them as in [`hybrid-quantization.md`](./hybrid-quantization.md).
- Only then try larger variants (E4B, and so on). These may not convert at all, since Rockchip only demonstrates E2B.

### Phase 4: Integration Surfaces

- **Android app:** add a "Gemma 4" entry and model paths to `android/app/src/main/res/values/strings.xml`. The JNI layer passes the family name through, so it should need no changes.
- **Server and INI:** `model_family = gemma4` should just work. Add an example to `data/server.example.ini`.
- **Docs:** add a README Models section. The Gemma 3 README section also links to `notes/gemma3.md`, which does not exist yet.

### Phase 5: Audio (Optional)

This is the only part that adds a new capability rather than a new profile, and should be scoped separately.

- A second `rknn_context` for the audio encoder, plus a config key and flag (`--audio` / `audio =`).
- Reading WAV files and computing log-mel features natively. Rockchip's demo has `gemma4_audio_preprocess()` and mel filter helpers that could be ported. The demo feeds FP16 tensors to the audio model.
- An `<audio>` placeholder in prompts, populating `multimodal_input.audio` with a token count of `((len + 1) / 2 + 1) / 2`.
- API and UI work in the server and the Android app.

## Open Questions

1. **Vision export shape.** Can position IDs be folded into the graph so the C++ side stays single-input? This decides how much `encode()` changes.
2. **Token count at 480×480.** One reading of the export script gives 900 tokens with no separate pooler. That conflicts with Gemma 4's stated pooling kernel of 3 and its 70–1120 token budgets. Reading the count from the RKNN output makes the C++ side safe either way, but the converted model needs checking.
3. **Calibration for quantization.** Rockchip's calibration script (`make_input_embeds_for_quantize.py`) only supports Qwen2-VL, so quantizing Gemma 4 with real image data may need a custom calibration set.

## References

- [Google Gemma 4 image understanding](https://ai.google.dev/gemma/docs/capabilities/vision/image)
- [Gemma 4 processor configuration](https://huggingface.co/google/gemma-4-E4B/blob/main/processor_config.json)
- [Hugging Face Gemma 4 image processor](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/image_processing_gemma4.py)
- [airockchip/rknn-llm](https://github.com/airockchip/rknn-llm)
- [rknn-llm multimodal demo](https://github.com/airockchip/rknn-llm/tree/main/examples/multimodal_model_demo)
- [`export_vision_gemma4.py`](https://github.com/airockchip/rknn-llm/blob/main/examples/multimodal_model_demo/export/export_vision_gemma4.py)
- [`export_vision_rknn.py`](https://github.com/airockchip/rknn-llm/blob/main/examples/multimodal_model_demo/export/export_vision_rknn.py)
- [Demo `main.cpp`](https://github.com/airockchip/rknn-llm/blob/main/examples/multimodal_model_demo/deploy/src/main.cpp)
- [Keras Hub Gemma 4 image converter](https://keras.io/keras_hub/api/models/gemma4/gemma4_image_converter)
- [Gemma 4 architecture walkthrough](https://programmerraja.is-a.dev/notes/AI-ML/Generative-AI/Gemma-4)
