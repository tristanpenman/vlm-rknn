# RKLLM

RKLLM is a bit of a black box. I would like to have a clean decoder implementation based on the original Hugging Face weights, similar to what I did with [Marian-RKNN](https://github.com/tristanpenman/marian-rknn) which only relies on RKNN.

Reverse engineering the `.rkllm` container and `librkllmrt.so` can inform that implementation. Compatibility with the `.rkllm` container is a nice to have, but the main goal is to side-step it completely.

## What Transfers from Marian-RKNN

The following parts of the Marian-RKNN approach remain useful:

- Convert model weights and selected operations into forms accepted by the RKNN runtime.
- Use fixed and explicit tensor interfaces rather than depending on dynamic graph behaviour.
- Compare intermediate results against a CPU reference at every model boundary.
- Keep tokenization, decoding policy, and sampling under application control.
- Use `rknn_matmul_api` directly for large linear operations where appropriate.
- Establish an FP16 implementation before adding quantization.

Marian-RKNN exports fixed-shape encoder and decoder ONNX graphs to RKNN. During greedy decoding it runs the complete fixed-length decoder for every generated token, selects the hidden state for the current position, applies the LM head separately, and feeds the selected token into the next invocation. Its conversion and inference implementations are useful references:

- [RKNN conversion](https://github.com/tristanpenman/marian-rknn/blob/master/python/marian_rknn/rknn_convert.py)
- [Native inference](https://github.com/tristanpenman/marian-rknn/blob/master/cpp/src/marian_rknn.cc)

This approach is practical for a relatively small translation model with short sequences. Recomputing the complete context for every token would be prohibitively expensive for a larger causal language model.

Figuring how to do this for models such as Qwen may unlock performance improvements in Marian too.

## Decoder Requirements

A Qwen or Llama-style decoder introduces several requirements that the current Marian implementation does not address:

- Persistent key/value caches for every transformer layer.
- Separate prefill and single-token decode workloads.
- Causal masking and position tracking.
- Rotary position embeddings, including Qwen multimodal M-RoPE.
- Grouped-query attention.
- Much larger weights and intermediate tensors.
- Efficient handling of context growth and cache eviction.

It is technically possible to export a monolithic ONNX decoder with all key/value caches represented as graph inputs and outputs. This is unlikely to be a good runtime design because it exposes many large cache tensors at every invocation and may force expensive memory transfers. Standard RKNN graphs also do not provide an obvious portable mechanism for retaining mutable key/value state internally across runs.

## Runtime Architecture

A practical initial implementation would use the host CPU for orchestration and smaller element-wise operations while offloading large matrix multiplications to the NPU:

```text
tokenizer and chat template
        |
token embedding
        |
for each transformer layer:
    RMSNorm                     CPU/NEON initially
    Q/K/V projections           rknn_matmul
    RoPE and KV-cache update    CPU/NEON
    attention                   CPU/NEON initially
    output projection           rknn_matmul
    RMSNorm and gated MLP       CPU plus rknn_matmul
        |
LM head                         rknn_matmul
        |
sampling and detokenization
```

Prefill should process multiple tokens using larger matrix multiplications. Autoregressive decode uses `M=1` matrix multiplications and may require different kernel choices and memory layouts. Treating these as separate execution paths is important for performance.

Attention can initially remain on the CPU to make cache ownership and correctness explicit. It can be profiled and migrated later if it becomes the bottleneck. This also avoids making the first prototype dependent on a monolithic compiler-generated graph.

## RKLLM Observations

Inspection of the bundled stripped `librkllmrt.so`:

- It contains llama.cpp and GGML-related strings and metadata names.
- It contains tokenizer, chat-template, RoPE, M-RoPE, and grouped model metadata.
- It contains explicit prefill, generation, and KV-cache management paths.
- It contains Rockchip matmul types and NPU matmul failure messages.

The Linux AArch64 library does not dynamically depend on the separately bundled `librknnrt.so`, while containing RKNN runtime version and implementation strings. This suggests that relevant RKNN runtime code is integrated into the RKLLM library.

These observations suggest that RKLLM is closer to a llama.cpp-derived host runtime driving specialised Rockchip NPU operations than to a single conventional RKNN graph. This is an inference from binary inspection rather than a documented implementation guarantee.

Rockchip's public exporter also loads a Hugging Face model directly and produces a proprietary `.rkllm` package instead of passing the complete language model through ONNX.

See the [official multimodal RKLLM export example](https://github.com/airockchip/rknn-llm/blob/main/examples/multimodal_model_demo/export/export_rkllm.py). This may shed some light on how this works.

## Targets

There are two related but distinct projects:

1. Implement a decoder backend from the original Hugging Face or safetensors weights.
2. Decode the `.rkllm` container and reproduce the behaviour of `librkllmrt.so`.

The first is the shortest route to removing the RKLLM runtime dependency. It also provides known model semantics and makes intermediate validation straightforward.

The second project is useful for understanding Rockchip's quantization, graph partitioning, cache layouts, and scheduling, but it adds proprietary file-format and binary-analysis work that will be arduous.

Useful binary-analysis techniques include:

- Compare `.rkllm` files generated from deliberately small model variants and one-variable configuration changes.
- Inspect file headers, offsets, alignment, embedded strings, and tensor metadata.
- Trace file access, memory mapping, device access, and ioctls during `rkllm_init` and `rkllm_run`.
- Compare prefill and decode traces.
- Use the public RKLLM API's token, embedding, logits, and hidden-state modes as controlled probes.
- Compare runtime outputs and intermediate reference tensors using deterministic greedy decoding.

## First Milestone

Start with a small text-only Qwen2 or Qwen2.5 decoder, such as a 0.5B model, using FP16 weights:

1. Load safetensors and tokenizer metadata into a simple internal model package.
2. Establish a Transformers or llama.cpp CPU reference.
3. Implement one transformer layer and compare its residual stream, Q/K/V tensors, attention output, and MLP output against the reference.
4. Offload Q/K/V, attention-output, gate/up/down, and LM-head matrix multiplications through `rknn_matmul_api`.
5. Implement an explicit FP16 key/value cache.
6. Add a batched prefill path and a separate single-token decode path.
7. Validate full greedy generation token by token.
8. Add W8A8 or W4A16 quantization only after FP16 logits agree sufficiently with the reference.
9. Inject the existing vision encoder output at the image-token positions.
10. Add Qwen multimodal position handling and validate M-RoPE.

The vision encoder should remain an ordinary RKNN model. The replacement decoder only needs to accept its image embeddings with the correct placeholder-token expansion and position encoding.

## Integration

The current decoder dependency is already concentrated in a small part of the codebase. `Session::init_text_decoder()` constructs the RKLLM session, and `Session::decode()` submits either prompt or multimodal input. The vision path independently produces a float embedding through the RKNN runtime.

A backend interface should eventually replace the RKLLM-specific `TextDecoder` handle. An initial interface might expose:

```cpp
class DecoderBackend
{
public:
    virtual ~DecoderBackend() = default;
    virtual int init(const ModelConfig& config) = 0;
    virtual int generate(
        const std::string& prompt,
        const float* image_embeddings,
        size_t image_token_count,
        Session::OutputCallback callback) = 0;
    virtual void clear_cache() = 0;
};
```

The existing RKLLM implementation can remain as one backend while the replacement is developed and compared against it. This permits differential testing on the same device and avoids coupling the new decoder work to the existing vision implementation.

## Main Risks

- RK3588 `M=1` matmul performance may be substantially different from batched prefill performance.
- Moving the key/value cache between CPU and NPU memory can dominate decode time.
- Quantization error can accumulate across many layers even when individual operations appear close.
- Compiler-supported operators and layouts may make an ONNX-heavy implementation less predictable than direct matmul orchestration.
- M-RoPE and image-token position construction must exactly match the selected Qwen-VL model family.
- Matching RKLLM performance will require more than matching model outputs; memory layout, cache reuse, threading, and NPU scheduling are likely important.

The initial success criterion should therefore be correct deterministic generation without `librkllmrt.so`. Performance and proprietary `.rkllm` compatibility should be treated as later milestones.
