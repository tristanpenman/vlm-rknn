# Calibration

The shared converter in [`rkllm_conversion.py`](../python/vlm_rknn/rkllm_conversion.py) accepts an optional calibration dataset and forwards it to RKLLM Toolkit's `build()`. The Gemma 3 converter exposes this through `--calibration-dataset`; omitting it preserves `dataset=None`. These converter changes are implemented, including readable-file checks, a conversion-summary row, and rejection for FP16 export.

This note describes calibration support using the repository's existing conversion helpers and Python environments, including the remaining dataset work. The scope is Gemma 3 text calibration; the Gemma 4 converter currently exports FP16 only. Tokenizer behavior and the effect of calibration on each supported quantization mode still need validation.

## Converter changes

### `python/vlm_rknn/rkllm_conversion.py`

In `convert_rkllm()`:

- Add a keyword argument `dataset: Path | None = None`. Defaulting to `None` keeps existing callers, including `convert_gemma4.py`, working unchanged.
- Replace the hard-coded `dataset=None` in `llm.build()` with `dataset=str(dataset) if dataset else None`.
- In the `fp16` branch, raise a `ValueError` when a dataset is given, alongside the existing checks for `dtype` and `hybrid_rate`. FP16 export disables quantization, so accepting a calibration dataset would be misleading.
- Check that a supplied dataset path is a readable file before loading the model.
- Add a `("Calibration", dataset or "none")` row to the conversion summary.

### `python/vlm_rknn/convert_gemma3.py`

- Add `--calibration-dataset` (`type=Path`) and pass it through as `dataset=args.calibration_dataset`.
- Default to `None` to preserve the current conversion behavior. A bundled default can be considered once a dataset has been created and validated.

### Dataset file

The `build()` docstring in `rkllm/api/rkllm.py` inside the bundled [rkllm_toolkit-1.3.1-cp311-cp311-linux_x86_64.whl](../python/wheels/rkllm_toolkit-1.3.1-cp311-cp311-linux_x86_64.whl) documents text calibration datasets as a JSON list of objects with `input` and `target` fields:

```json
[
  {"input": "<formatted prompt>", "target": "<reference completion>"}
]
```

- Create a dataset from prompts relevant to the model's intended use, using that schema. The public docstring does not specify chat-template or special-token handling.
- A suggested location is `python/vlm_rknn/data/gemma3_calibration_dataset.json`.
- Check how the toolkit handles chat templates and special tokens before storing formatted prompts. Do not assume that the host baseline's tokenization matches the toolkit's calibration path.
- The Python services in [`docker-compose.yml`](../docker-compose.yml) bind-mount the repository at `/workspace`, so a dataset stored in the repository would be available inside those containers without changing their mounts.

### Optional calibration generator

A manually prepared dataset can be supplied without a generator. To automate target generation, add a generator such as `python/vlm_rknn/generate_gemma3_calibration.py`, following the package's conventions:

- Run it as `python -m vlm_rknn.generate_gemma3_calibration` from the GPU shell documented in [`README.md`](../python/README.md).
- Reuse `require_huggingface_login()` and `HF_MODEL` from `convert_gemma3.py`, or move the authentication helper into a shared module.
- Download the original checkpoint with `download_hf_model(HF_MODEL, CACHE_DIR / "gemma-3-4b-it")`.
- Generate the `target` completions from the original checkpoint with greedy decoding and record the generation settings.
- Use `validate_gemma3.py` as a reference for device and dtype selection and generation. Its baseline loads a Q8_0 GGUF, so its model-loading path should not be reused for a generator intended to use the original checkpoint.
- Verify special-token handling against the toolkit before deciding whether serialized prompts should include a beginning-of-sequence token.

The dataset and generator do not exist yet. Validate dataset loading, prompt formatting, and target completions with the installed toolkit before relying on them. Check each quantization mode that will be documented as supporting calibration; accepting the `dataset` argument alone does not demonstrate a quality improvement.

### `python/README.md`

- Add `--calibration-dataset` to the Gemma 3 options table once implemented.
- Document the dataset format, path resolution from the current directory, and validated prompt formatting. Include the bundled dataset location and target regeneration command if those are added.
- State that FP16 conversion does not use a calibration dataset.

## Optional and related changes

- **`README.md`**: add an entry for the generator under "Python Scripts" once it exists.
- **[`hybrid-quantization.md`](./hybrid-quantization.md)**: it lists the calibration dataset among the settings that must match between stages, and its example `build()` call passes `dataset=None`. Update the example when demonstrating a calibrated workflow.
- **Output naming**: consider distinguishing calibrated and uncalibrated exports in their filenames. The current Gemma 3 default output name contains the model, quantization format, and target, but no calibration information.
