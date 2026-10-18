# Plan

- Gemma 4 support (see [`notes/gemma4.md`](./notes/gemma4.md))
  + Model conversion
    + ~~Add the `rkllm_toolkit` 1.3.1 wheel and pin `transformers` / `onnx` to the versions Rockchip's demo requires~~
    + ~~Add `convert_gemma4.py` for `google/gemma-4-E2B-it`, sharing download and build logic with `convert_gemma3.py`~~
    + ~~Document the conversion recipe~~
    + Export the vision encoder to RKNN at a fixed resolution, folding `pixel_position_ids` into the graph
  + C++ text and image support
    + ~~Add the `kGemma4` model family, parse aliases, and CLI help text~~
    + ~~Add parse and family assertions to the host tests~~
    + Make the Gemma 3 vision input checks profile-driven
    + Validate tensor layout assumptions for RKNN model metadata (`NCHW` vs `NHWC`, output shape ordering)
  + Device validation
    + Compare RKNN vision output against the PyTorch reference
    + Confirm preprocessing matches the Hugging Face reference (resize mode, pad colour, normalisation)
    + Compare greedy decoding against Hugging Face Transformers
    + Capture a sample run (image + prompt + transcript) in the README
  + ~~Integration~~
    + ~~Add Gemma 4 to the Android app's model list~~
    + ~~Add a Gemma 4 example to `data/server.example.ini`~~
  + Audio (optional)
    + Export the Gemma 4 audio encoder to RKNN
    + Add native WAV loading and log-mel feature extraction
    + Pass audio embeddings through `multimodal_input.audio`
    + Expose audio input in the CLI, server, and Android app

- Build and CI
  + Publish prebuilt binaries from tagged releases
  + Run the host tests in CI
  + Add a smoke test that runs the binary against a tiny fixture model

- CLI and runtime
  + Handle `rkllm_clear_kv_cache` for a `clear` REPL command
  + ~~Install a `SIGINT` handler that destroys the RKLLM handle cleanly~~
  + ~~Handle image sources other than files on disk~~
  + Add RAII wrappers for RKNN primitives

- R&D
  + KV-caching: verify prefix reuse on device by comparing `prefill_time_ms` with and without a loaded prompt cache, and extend prompt caching to the HTTP server (see [`notes/kv-caching.md`](./notes/kv-caching.md))
  + Cross-attention: support Llama 3.2 Vision via `rkllm_set_cross_attn_params`, which depends on a custom-converted Mllama RKLLM decoder (see [`notes/cross-attention.md`](./notes/cross-attention.md))
