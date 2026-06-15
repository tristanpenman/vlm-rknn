# Qwen 3 Support

Qwen3-VL should be treated as a distinct model family rather than a drop-in replacement for Qwen2-VL.

## Required Changes

1. Add a model-family/profile layer in `ModelConfig`.

   The implementation should distinguish at least:

   - `qwen2_vl`
   - `qwen2_5_vl`
   - `qwen3_vl`

   The selected profile should own image special tokens, prompt/chat template rules, image placeholder syntax, preprocessing settings, default context length, and whether thinking mode is supported.

2. Stop hard-coding Qwen2-VL assumptions in `cpp/src/qwen_vl_rknn.cc`.

   The current RKLLM setup hard-codes:

   ```cpp
   params.img_start = "<|vision_start|>";
   params.img_end = "<|vision_end|>";
   params.img_content = "<|image_pad|>";
   ```

   These should come from the selected model profile.

3. Add Qwen3 thinking-mode control.

   The bundled RKLLM header already exposes `RKLLMInput::enable_thinking`, so the CLI should probably support:

   - `--model-family qwen3-vl`
   - `--thinking`
   - `--no-thinking`

4. Make image preprocessing model-specific.

   Current preprocessing is fixed:

   - RGB
   - square pad with `127.5`
   - resize to RKNN input dimensions
   - uint8 NHWC

   Qwen2, Qwen2.5, and Qwen3 VL models may differ in expected resize policy, dynamic resolution handling, patch layout, and token count. This belongs in the profile or in a small preprocessing strategy object.

5. Harden vision encoder metadata handling.

   Current code infers image token count from the first output tensor shape by selecting the first dimension greater than 1. That is fragile for Qwen3-VL. The implementation should explicitly validate:

   - number of RKNN inputs
   - output tensor count
   - output tensor dtype/layout
   - final flattened embedding length expected by RKLLM
   - whether multiple outputs need concatenation or structured passing

6. Add model switching by reloading the session, not by mutating live handles.

   A practical API would be:

   ```cpp
   int Session::reload(ModelConfig config);
   ```

   This should:

   - destroy the RKLLM handle
   - destroy the RKNN context
   - replace the config
   - initialize the decoder
   - initialize the encoder
   - clear cached state

   On RK3588, keeping Qwen2 and Qwen3 loaded simultaneously is probably not worth the memory cost.

7. Add CLI support for profiles/manifests.

   Avoid forcing users to remember every model-specific option. Prefer a manifest-based flow:

   ```bash
   qwen-vl-rknn --model models/qwen2-vl-2b/model.json image.png "Describe this"
   qwen-vl-rknn --model models/qwen3-vl-2b/model.json --thinking image.png "Reason about this"
   ```

   Example manifest:

   ```json
   {
     "family": "qwen3_vl",
     "vision_encoder": "vision.rknn",
     "language_model": "model.rkllm",
     "max_context_len": 8192,
     "preprocess": {
       "pad_color": [127.5, 127.5, 127.5]
     }
   }
   ```

## Switching Between Qwen2 and Qwen3

For one-shot CLI usage, choose the model with `--model` or `--model-family`.

For the REPL, add commands such as:

- `:model /path/to/qwen2.json`
- `:model /path/to/qwen3.json`
- `:thinking on`
- `:thinking off`

Internally, model switching should call `Session::reload(...)`.

## Main Risk

The biggest unknown is not the C++ wrapper. It is whether there is a matching Qwen3-VL `.rkllm` and vision `.rknn` pair that RKLLM/RKNN can consume correctly on RK3588. If Rockchip's RKLLM runtime does not support the Qwen3-VL multimodal path for the converted model, the repository changes above will not be enough.

The local RKLLM header already exposes Qwen3 thinking control in `thirdparty/rkllm/include/rkllm.h`. Public Qwen3-VL technical material also describes model-family architecture changes including enhanced interleaved M-RoPE and DeepStack-style vision integration, so Qwen3-VL should be validated as its own profile.
