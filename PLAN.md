# Plan

- Build and tooling
  + ~~Project scaffolding and CMake layout~~
  + ~~Wire up RKNN and RKLLM libraries from `thirdparty/`~~
  + ~~Add OpenCV via FetchContent (trimmed module set)~~
  + ~~Docker images and helper scripts for native and Android builds~~
  + ~~GitHub Actions for ShellCheck~~
  + ~~Strip symbols from release builds~~
  + Add a CI job that runs the native build end-to-end
  + Publish prebuilt binaries from tagged releases

- Vision encoder
  + ~~Load RKNN model from disk~~
  + ~~Honour `--cores` and set the NPU core mask~~
  + Query model input/output info and populate `VisionEncoder`
  + Free `input_attrs` / `output_attrs` in `cleanup_vision_encoder`
  + Implement vision encoder portion of `Session::run()`

- Image preprocessing
  + ~~Load image with OpenCV, BGR → RGB~~
  + ~~Pad to square with neutral background (`expand2square`)~~
  + ~~Resize to model input dimensions~~
  + Confirm preprocessing matches the Qwen-VL reference (mean/std normalisation, pad colour)

- Text decoder (RKLLM)
  + ~~Configure `RKLLMParam` with Qwen-VL `<|vision_start|>` / `<|vision_end|>` / `<|image_pad|>` tokens~~
  + ~~Initialise the RKLLM handle and register the streaming callback~~
  + Invoke `rkllm_run` with encoded image vector
  + Update debug logging to use existing logger rather than `printf`
  + Handle `rkllm_clear_kv_cache` for a `clear` REPL command

- CLI and UX
  + ~~Argument parsing for `--verbose`, `--cores`, etc~~
  + ~~Run the supplied prompt once and exit when `[prompt]` is provided~~
  + ~~Drop into an interactive REPL when no prompt is given~~
  + Expose `--max-new-tokens` and `--max-context-len` flags that flow through to `ModelConfig`
  + Install a `SIGINT` handler that destroys the RKLLM handle cleanly

- Tests
  + Fix the broken reference to `qwen_vl_rknn::target_device()`
  + Replace the placeholder readiness check for `is_ready()`
  + Add a host-side unit test for `expand2square` covering portrait, landscape, and already-square inputs
  + Add a smoke test that runs the binary against a tiny fixture model

- On-device verification
  + Source or convert a Qwen-VL vision encoder (`.rknn`) and language model (`.rkllm`) suitable for RK3588
  + Document the conversion recipe
  + Capture a sample run (image + prompt + transcript) in the README
  + Measure load and per-token latency for the default model and record baseline numbers

- Documentation
  + ~~Top-level README with build instructions and project layout~~
  + ~~AGENTS.md describing the project for assistants~~
  + Add a "Running" section to the README with concrete CLI examples once `Session::run` is functional
  + Document the expected `thirdparty/` layout and where to download the runtime libraries
