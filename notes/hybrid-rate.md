# RKLLM Hybrid Quantization and Conversion Memory

The RKLLM Toolkit's `hybrid_rate` setting controls the proportion of eligible model blocks that use an alternative group-wise quantization format instead of the primary quantization format. For example, a W8A8 conversion with `hybrid_rate=0.25` can combine ordinary W8A8 blocks with group-wise `W8A8_G512` blocks. The setting is intended to balance model accuracy, performance, and size by applying the alternative representation to blocks selected through quantization-sensitivity analysis. It is not a GPU/CPU offload ratio and does not move that proportion of the model into system RAM.

Sensitivity analysis and mixed-quantization processing increase conversion-time memory usage. RKLLM conversion can consequently require substantially more memory than the final model. Model file size and parameter count provide only a lower-bound estimate because the converter also creates transformed weights, quantized copies, calibration data, and temporary CUDA workspaces.

The toolkit implementation is proprietary, so there is no exact public formula for its peak memory requirement. Capacity planning should combine a rough estimate with measurement of an actual conversion.

## Interpreting CUDA Out-of-Memory Errors

The total-capacity value in a CUDA error is not the amount still available. For example, the Gemma 3 4B conversion produced this state before failing:

```text
Total capacity:        15.47 GiB
Free before request:  691.88 MiB
Requested allocation:   2.50 GiB
Process usage:         14.70 GiB
```

The immediate lower bound for that conversion stage is therefore approximately:

```text
15.47 GiB - 0.68 GiB + 2.50 GiB = 17.29 GiB
```

This only describes the allocation that failed. The converter might request more memory after that allocation succeeds, so 17.29 GiB is not a complete peak-memory measurement.

A useful capacity-planning rule is to leave 15% headroom:

```text
recommended VRAM = observed lower bound / 0.85
                 = 17.29 GiB / 0.85
                 = approximately 20.3 GiB
```

For this workload, a 24 GiB GPU is therefore a safer choice than a 16 GiB GPU. System RAM does not normally count as available CUDA device memory.

The allocator's suggestion to enable `expandable_segments` is relevant only when PyTorch has a large amount of reserved but unallocated memory. In the observed failure, only about 62 MiB was reserved but unused while the converter requested 2.50 GiB. The failure was caused by insufficient capacity rather than meaningful allocator fragmentation.

## Estimating Memory From Parameter Count

For a model with four billion parameters, weights alone require approximately:

```text
FP32: 4 billion * 4 bytes = 14.9 GiB
FP16: 4 billion * 2 bytes =  7.5 GiB
INT8: 4 billion * 1 byte  =  3.7 GiB
```

An RKLLM W8A8 conversion may need several of the following allocations at the same time:

- Source FP16 or FP32 weights.
- Transformed working weights.
- Quantized output weights.
- Quantization scales and statistics.
- Layer inputs and outputs used for calibration or sensitivity analysis.
- Per-layer CUDA workspaces.
- CUDA context and PyTorch allocator overhead.

A rough estimate for the observed Gemma 3 4B conversion is:

```text
 7.5 GiB  FP16 source weights
 3.7 GiB  INT8 output weights
 2-5 GiB  transformed weights and temporary tensors
 2.5 GiB  observed workspace request
 1-2 GiB  CUDA, graph, and safety overhead
---------
16.7-20.7 GiB approximate peak
```

This estimate is useful for selecting hardware, but it is not a guarantee. The closed-source toolkit controls tensor lifetimes and may use FP32 intermediates even when the model is loaded as FP16.

## Measuring Peak Memory

Monitor device memory from a second host terminal while conversion runs:

```bash
nvidia-smi dmon -s m -d 1
```

The framebuffer-memory columns show device usage once per second. A shorter sampling interval may be needed to capture brief allocation spikes. Measuring a successful run on a GPU with sufficient memory is the only way to discover the complete peak; an unsuccessful run establishes only a lower bound.

PyTorch memory counters can help diagnose allocations made through PyTorch, but they may omit memory allocated directly by CUDA or native toolkit components. Device-level measurements from `nvidia-smi` are therefore preferable for total capacity planning.

## Effect of `hybrid_rate`

`hybrid_rate` is a mixed-quantization setting. It is not a GPU/CPU offload ratio and does not move that proportion of the model into system RAM.

For example, W8A8 conversion with `hybrid_rate=0.25` can mix ordinary W8A8 blocks with group-wise `W8A8_G512` blocks. RKLLM first performs sensitivity analysis to decide which blocks should use the alternative representation. This analysis may retain original weights, candidate quantized tensors, temporary dequantized tensors, layer activations, scales, and sensitivity scores at the same time.

Consequently, enabling mixed quantization increases temporary peak memory:

```text
hybrid_rate = 0
    -> uniform quantization
    -> no mixed-block sensitivity-selection pass
    -> lowest conversion-memory requirement

hybrid_rate > 0
    -> mixed quantization
    -> sensitivity analysis and candidate tensors
    -> higher temporary peak memory
```

The increase is not necessarily proportional to the configured rate. The toolkit may evaluate many candidate blocks before selecting the final fraction. Even `hybrid_rate=0.25` can therefore cause a substantial peak-memory increase.

For a memory-constrained conversion, start with:

```bash
python -m vlm_rknn.convert_gemma3 \
  --device cuda \
  --dtype float16 \
  --hybrid-rate 0 \
  --max-context 2048
```

Loading with `float16` reduces source-weight memory relative to `float32`, but it may not reduce every build-stage intermediate. Reducing `max_context` can reduce context-dependent buffers, although weight conversion may remain the dominant cost. If conversion still exceeds device capacity, use CPU conversion or a GPU with more VRAM.

## Toolkit Memory Controls

RKLLM Toolkit 1.3.0 does not expose a supported sensitivity batch size, layer-at-a-time mode, CPU-offload option, or GPU-memory limit. Lowering `hybrid_rate` changes the number of blocks ultimately selected, but it may not reduce the analysis peak because the toolkit can still evaluate every candidate block. `optimization_level=0` is documented as prioritizing speed and memory and may reduce the later model-building peak, but sensitivity analysis occurs before that stage and may be unaffected.

The toolkit contains internal `hybrid_topk` and `hybrid_thresh` attributes. These appear to control selection rather than the number or size of tensors used during sensitivity analysis. They are undocumented implementation details and should not be treated as stable memory controls.

## Two-Stage Quantization-Parameter Workflow

The most promising way to retain hybrid quantization on a memory-constrained GPU is to separate sensitivity analysis from the final conversion. RKLLM's public `build()` method accepts `extra_qparams`, while its internal API contains `get_hybrid_block()`, `get_hybrid_layer()`, and `save_qparams()` methods. The toolkit also reports that saved `.qparams` files can be reloaded through `extra_qparams` when reconverting a model.

The intended workflow is:

```text
Stage 1: CPU or larger GPU
    -> load the exact source checkpoint
    -> calculate hybrid quantization sensitivity
    -> select the hybrid blocks
    -> save the resulting .qparams file

Stage 2: memory-constrained GPU
    -> load the same source checkpoint
    -> provide the saved file through extra_qparams
    -> reuse the quantization decisions and parameters
    -> build and export the RKLLM model
```

Both stages must use the same source checkpoint, RKLLM Toolkit version, target platform, primary quantization format, quantization algorithm, calibration dataset, and `hybrid_rate`. Quantization parameters are model- and configuration-specific and should not be reused after changing tensor shapes, model weights, or relevant quantization settings.

Conceptually, the second build would look like this:

```python
llm.build(
    do_quantization=True,
    optimization_level=0,
    quantized_dtype="w8a8",
    quantized_algorithm="normal",
    target_platform="rk3588",
    num_npu_core=3,
    extra_qparams="gemma-3-4b-it.qparams",
    dataset=None,
    hybrid_rate=0.25,
    max_context=2048,
)
```

This approach is experimental. Rockchip does not document a public standalone command that performs only hybrid sensitivity analysis and saves its result. Stage 1 may therefore require a complete CPU conversion, a conversion on a larger GPU, or direct use of undocumented Cython methods. It also remains to be verified whether loading `.qparams` skips the complete hybrid sensitivity pass or only part of the quantization work. Peak VRAM should be measured during stage 2 before relying on the workflow.

The repository's Gemma converter currently passes `extra_qparams=None` and does not expose a command-line option for loading or saving `.qparams`. Supporting this workflow requires extending the converter and validating that the resulting model has the intended mixture of W8A8 and group-wise blocks.
