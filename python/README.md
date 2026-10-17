# Python scripts

The `vlm_rknn` Python package contains utilities for preparing models for use with the C++ RKNN/RKLLM runtime. It includes converters for Gemma 3 and Gemma 4 language models.

## Prerequisites

Gemma 3 model has restricted access on Hugging Face. Request access to [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) and accept Google's usage license.

### Python Shell

From the repository root, build the CPU conversion environment, start its Ollama server, and open a shell with:

```bash
./scripts/python-shell.sh
```

For CUDA conversion, use the GPU environment:

```bash
./scripts/python-shell-gpu.sh
```

The GPU service requires an NVIDIA driver and NVIDIA Container Toolkit on the host. From the GPU container, verify CUDA access with:

```bash
python -c \
  'import torch; print(torch.version.cuda); print(torch.cuda.is_available())'
```

Inside the selected container, authenticate with the Hugging Face CLI before running the converter:

```bash
hf auth login
```

The containers bind-mount the repository at `/workspace`. Ollama models, Hugging Face downloads, and RKLLM scratch files are stored beneath `/workspace/.cache`, which corresponds to the repository's `.cache` directory.

### RKLLM Toolkit

Three RKLLM Toolkit wheels are included in [`python/wheels`](./wheels):

- `rkllm_toolkit-1.3.1-cp311-cp311-linux_x86_64.whl`
- `rkllm_toolkit-1.3.0-cp311-cp311-linux_x86_64.whl`
- `rkllm_toolkit-1.2.3-cp311-cp311-linux_x86_64.whl`

The Python images install version 1.3.1 by default. To use an older version, edit [`requirements.txt`](./requirements.txt) so its RKLLM Toolkit entry points to the desired wheel under `/opt/wheels`, then rebuild the selected Python image. The requirements also pin `transformers==5.8.0` and `onnx==1.22.0` for Gemma 4 conversion.

## Gemma 3

Ollama stores `gemma3:4b-it-q8_0` as a Q8_0 GGUF model, but RKLLM Toolkit does not accept GGUF input. Its W8A8 and W4A16 formats also use different quantization schemes from GGUF Q8_0.

The converter therefore does not dequantize and requantize Ollama's lossy weights. Instead, it:

1. Asks Ollama which GGUF blob backs `gemma3:4b-it-q8_0`.
2. Validates that the blob contains Q8_0 tensors.
3. Downloads the matching original `google/gemma-3-4b-it` weights from Hugging Face.
4. Loads the original weights with RKLLM Toolkit.
5. Quantizes and exports them for the selected Rockchip target.

This avoids an unnecessary Q8_0 to FP16 to W8A8 conversion.

### Usage

Run the converter from either Python container:

```bash
python -m vlm_rknn.convert_gemma3
```

The default invocation validates the installed Ollama model, downloads the original Hugging Face weights to `.cache/gemma-3-4b-it`, and creates `gemma-3-4b-it-w8a8-rk3588.rkllm` in the current directory. If `XDG_CACHE_HOME` is set, it is used instead of `.cache`.

The converter prompts to pull the Ollama model if it is not already installed. Use `--skip-ollama-validation` to skip the Ollama lookup and GGUF validation. This option does not skip the Hugging Face download or RKLLM conversion.

Relative paths are resolved from the current directory, and missing output directories are created automatically.

### Options

| Option                     | Default                                  | Description                                       |
|----------------------------|------------------------------------------|---------------------------------------------------|
| `--ollama-model`           | `gemma3:4b-it-q8_0`                      | Ollama model to validate                          |
| `--hf-dir`                 | `.cache/gemma-3-4b-it`                   | Hugging Face model directory                      |
| `--output` or `-o`         | `gemma-3-4b-it-w8a8-rk3588.rkllm`        | Destination RKLLM file                            |
| `--target`                 | `rk3588`                                 | Target platform: `rk3588` or `rk3576`             |
| `--quantization`           | `w8a8`                                   | `w8a8`, `w8a8_gx`, `w4a16`, or `w4a16_gx`         |
| `--npu-cores`              | `3`                                      | Number of target NPU cores                        |
| `--max-context`            | `4096`                                   | Maximum context length                            |
| `--hybrid-rate`            | `0.25`                                   | Mixed-block quantization ratio                    |
| `--device`                 | `cpu`                                    | Conversion device: `cpu` or `cuda`                |
| `--dtype`                  | `bfloat16` (CUDA), `float32` (CPU)       | Source dtype: `float16`, `bfloat16`, or `float32` |
| `--optimization-level`     | `1`                                      | `0` prioritizes memory, `1` prioritizes accuracy  |
| `--skip-ollama-validation` | disabled                                 | Skip Ollama GGUF validation                       |

W8A8 variants use RKLLM Toolkit's `normal` algorithm. W4A16 variants use group-wise refined quantization (`grq`), which divides weights into small groups to reduce sensitivity to outliers.

For example, this command creates a W4A16 model for RK3576 with an 8192-token context using CUDA:

```bash
python -m vlm_rknn.convert_gemma3 \
  --target rk3576 \
  --quantization w4a16 \
  --max-context 8192 \
  --device cuda \
  --output ./models/gemma-3-4b-it-w4a16-rk3576.rkllm
```

Use `--device cuda` only in the GPU container with working CUDA access. Model weights and generated RKLLM files remain subject to their respective model and toolkit license terms.

CUDA conversion defaults to `bfloat16`. RKLLM Toolkit's own default is `float32`, which materializes the 4.30 B-parameter Gemma 3 checkpoint as 16.0 GiB of weights and does not fit on a 16 GiB card. The published checkpoint is already `bfloat16`, as recorded by `torch_dtype` in its `config.json`, so `bfloat16` halves the resident weights to 8.0 GiB without down-converting any stored value. Prefer it over `float16` on Ampere and later, where `bfloat16` is native, because Gemma's weight and activation outliers can exceed `float16` range. Pascal and earlier GPUs have no `bfloat16` support and need `--dtype float16`. Pass `--dtype float32` to restore the toolkit default when sufficient VRAM is available.

`--optimization-level 0` reduces the model-building peak; RKLLM documents level `0` as prioritizing speed and memory and level `1` as prioritizing accuracy. Mixed-quantization sensitivity analysis runs before that stage, so `--hybrid-rate 0` is the lever for its separate peak.

RKLLM Toolkit 1.3.0 declares exact dependencies on PyTorch 2.6.0 and torchvision 0.21.0, whose CUDA 12.6 wheels do not support Blackwell GPUs. The GPU image intentionally overrides those pins with PyTorch 2.7.1 and torchvision 0.22.1 from the CUDA 12.8 index so that RTX 50-series GPUs, including the RTX 5060 Ti, have `sm_120` kernels. This combination is outside Rockchip's declared dependency constraints.

## Gemma 4

### Usage

Run this in the Python container to download `google/gemma-4-E2B-it` and export an FP16 RKLLM model for RK3588:

```bash
python -m vlm_rknn.convert_gemma4
```

### Options

The default output is `gemma-4-E2B-it-fp16-rk3588.rkllm`. Use `--target rk3576` for RK3576; the default NPU core count adjusts to the target. `--hf-dir`, `--output`, `--npu-cores`, `--max-context`, `--optimization-level`, and `--device` are also available. This converter covers the language model only. Image input requires a separate RKNN vision encoder, as described in [`notes/gemma4.md`](../notes/gemma4.md).

## Model evaluation

The `vlm_rknn.evaluate_model` module runs a prompt and expected-answer test suite against any model command. It starts the command once per sample, scores its standard output, and records the output, expected answer, standard error, return code, score, and latency.

Create prompt and expected-answer files, then reference them from a JSON manifest. Paths are resolved relative to the manifest:

```text
evaluation/
├── manifest.json
├── prompts/
│   └── capital.txt
└── expected/
    └── capital.txt
```

```json
{
  "samples": [
    {
      "id": "capital-of-france",
      "prompt": "prompts/capital.txt",
      "expected": "expected/capital.txt",
      "evaluator": "exact_match"
    }
  ]
}
```

Pass the manifest and model command to the evaluator. By default, each prompt is sent to the command on standard input:

```bash
python -m vlm_rknn.evaluate_model evaluation/manifest.json \
  --command 'ollama run gemma3:4b-it-q8_0'
```

If the model command expects a file, include `{prompt_file}` in the command. The evaluator replaces it with a temporary UTF-8 prompt file for each sample:

```bash
python -m vlm_rknn.evaluate_model evaluation/manifest.json \
  --command './model-runner --prompt-file {prompt_file}'
```

Each sample may select one of these evaluators:

| Evaluator     | Scoring                                                                |
|---------------|------------------------------------------------------------------------|
| `exact_match` | Case-insensitive exact match after normalizing whitespace              |
| `token_f1`    | F1 score based on overlapping word and punctuation tokens              |
| `rouge_l`     | F1 score based on the longest common token subsequence                 |
| `json`        | Parsed JSON equality, or the proportion of matching `required_fields`  |

The default evaluator is `exact_match`. A sample passes when its command exits successfully and its score meets `threshold`. The default threshold is `1.0` for `exact_match` and `json`, and `0.8` for `token_f1` and `rouge_l`.

For JSON output, a sample can restrict comparison to selected fields:

```json
{
  "id": "structured-answer",
  "prompt": "prompts/structured.txt",
  "expected": "expected/structured.json",
  "evaluator": "json",
  "required_fields": ["answer", "units"],
  "threshold": 1.0
}
```

### Evaluation options

| Option             | Default              | Description                                                       |
|--------------------|----------------------|-------------------------------------------------------------------|
| `--command`        | required             | Model command; reads standard input or uses `{prompt_file}`       |
| `--output-dir`     | `evaluation-results` | Directory for `report.json` and `report.md`                       |
| `--timeout`        | `120`                | Per-sample timeout in seconds                                     |
| `--min-pass-rate`  | `0.0`                | Required overall pass rate from `0` to `1`                        |
| `--min-mean-score` | `0.0`                | Required overall mean score from `0` to `1`                       |

Overall thresholds can make an evaluation fail when model quality falls below an acceptable level:

```bash
python -m vlm_rknn.evaluate_model evaluation/manifest.json \
  --command 'ollama run gemma3:4b-it-q8_0' \
  --timeout 300 \
  --min-pass-rate 0.9 \
  --min-mean-score 0.85 \
  --output-dir evaluation-results/gemma3
```

The command exits with status `0` when both overall thresholds are met, `1` when either threshold is missed, and `2` for invalid input or setup errors. A timed-out sample is recorded as failed with return code `124`; any other non-zero model return code also fails the sample.
