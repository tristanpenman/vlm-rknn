# Backends

What is the most practical way to add support for qwen models that run on macOS and CUDA? Would ggml work?

## Challenges

This project is explicitly built around Rockchip RKNN/RKLLM and Rockchip targets, not generic GPU backends. In particular, it hardwires RKNN/RKLLM runtime libs in CMake.

## Pathways

Most practical architecture would introduce a backend interface based on the current `Session` class.

This would keep the existing Rockchip backend (RKNN/RKLLM), making it a subclasses of the backend interface. Then we can add a new gglm-based backend to support macOS (via Metal) and NVIDIA (via CUDA)

## GGML

GGML should be appropriate due to existing support for ecosystems like llama.cpp/GGUF tooling.

Caveats: for Qwen-VL multimodal, the text model path is usually straightforward, but the vision encoder / projector integration is the part that will need to be verified per model variant. Practically speaking, ggml is viable, but we should choose a Qwen-VL variant with known GGUF/llama.cpp compatibility.

## Next Steps

* Refactor: isolate current RK backend behind an interface
* Text-only ggml backend (prove macOS+CUDA runtime plumbing)
* Add multimodal path (image preprocessing + vision encoder bridging)
* Update CI/build to include new targets
