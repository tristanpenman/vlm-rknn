# Copyright (c) 2026 Tristan Penman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run Gemma 3 on the host to produce a baseline for the on-device model.

By default this uses the model corresponding to Ollama's:

    gemma3:4b-it-q8_0

Inference happens inside this Python process, using Transformers with CUDA
when a GPU is visible, so no Ollama server or external runner is involved.
The Q8_0 GGUF is loaded directly through the Transformers GGUF reader, which
means the baseline runs the already-quantized int8 weights rather than the
original Hugging Face checkpoint.

Important:
    The GGUF reader dequantizes Q8_0 blocks back to dense tensors, so weight
    *values* match Ollama exactly but the arithmetic is bfloat16/float32
    rather than llama.cpp's int8 kernels. That makes this a reference for
    answer quality, not a bit-exact reproduction of an Ollama transcript.

    Ollama's blob does carry the Gemma 3 vision tower, but Transformers
    loads a GGUF as a text decoder and drops those tensors. The baseline
    therefore covers text prompts only.

The manifest format is the one used by ``evaluate_model``, and the report is
written in the same shape, so a host baseline and a device run can be
compared sample by sample.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .convert_gemma3 import OLLAMA_MODEL, find_ollama_gguf
from .evaluate_model import EVALUATORS, Result, read_text
from .rkllm_conversion import CACHE_DIR

GGUF_CACHE_DIR = CACHE_DIR / "gguf-baseline"

HF_REPO = "ggml-org/gemma-3-4b-it-GGUF"
HF_FILE = "gemma-3-4b-it-Q8_0.gguf"

# Used when the GGUF does not carry a chat template of its own.
GEMMA3_TURN = "<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"

END_OF_TURN = "<end_of_turn>"

EMBEDDING_TENSOR = "token_embd.weight"


@dataclass
class BaselineResult(Result):
    """An ``evaluate_model`` result plus host generation statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: float = 0.0


def describe_gguf(path: Path) -> dict[str, Any]:
    """Summarize a GGUF's architecture and quantization mix for the report."""

    from gguf import GGMLQuantizationType, GGUFReader

    summary: dict[str, Any] = {"path": str(path)}

    reader = GGUFReader(str(path))

    field = reader.fields.get("general.architecture")
    if field is not None:
        try:
            summary["architecture"] = field.contents()
        except Exception:
            # gguf-py API has changed a little between releases.
            summary["architecture"] = str(field)

    quantizations: dict[str, int] = {}
    for tensor in reader.tensors:
        try:
            name = GGMLQuantizationType(tensor.tensor_type).name
        except (TypeError, ValueError):
            name = f"type_{int(tensor.tensor_type)}"
        quantizations[name] = quantizations.get(name, 0) + 1

        if tensor.name == EMBEDDING_TENSOR:
            # GGUF stores the fastest-moving dimension first, so the row
            # count, and therefore the vocabulary size, is the second one.
            summary["vocab_size"] = int(tensor.shape[1])

    summary["tensors"] = sum(quantizations.values())
    summary["quantizations"] = dict(sorted(quantizations.items()))

    architecture = summary.get("architecture")
    if architecture is not None and architecture != "gemma3":
        print(
            f"warning: GGUF architecture is {architecture!r}, not 'gemma3'.",
            file=sys.stderr,
        )

    if not any(name.startswith("Q8") for name in quantizations):
        print(
            "warning: GGUF contains no Q8 tensors, so this baseline is not "
            f"int8. Quantizations found: {', '.join(summary['quantizations'])}",
            file=sys.stderr,
        )

    return summary


def cache_gguf(path: Path, name: str) -> Path:
    """
    Expose a GGUF under a ``.gguf`` filename in a directory of its own.

    Ollama stores blobs as extensionless ``sha256-...`` files alongside every
    other blob it has pulled. Transformers expects to be handed a directory
    plus a filename, so link the blob somewhere tidy instead.
    """

    if path.suffix == ".gguf":
        return path

    GGUF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    link = GGUF_CACHE_DIR / name

    if link.is_symlink():
        if link.resolve() == path.resolve():
            return link
        link.unlink()
    elif link.exists():
        raise RuntimeError(
            f"GGUF cache path already exists and is not a link: {link}"
        )

    link.symlink_to(path.resolve())
    return link


def download_hf_gguf(repo: str, filename: str) -> Path:
    """Download a single GGUF file from Hugging Face."""

    from huggingface_hub import hf_hub_download

    print(f"Downloading {filename} from {repo}...")
    return Path(hf_hub_download(repo_id=repo, filename=filename))


def resolve_gguf(args: argparse.Namespace) -> tuple[Path, str]:
    """Locate the GGUF to run, returning its path and where it came from."""

    if args.gguf is not None:
        path = args.gguf.expanduser()
        if not path.is_file():
            raise ValueError(f"GGUF file does not exist: {path}")
        return cache_gguf(path, path.name), "file"

    if args.source in {"auto", "ollama"}:
        try:
            path = find_ollama_gguf(args.ollama_model)
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            if args.source == "ollama":
                raise
            print(
                f"warning: could not use Ollama model "
                f"{args.ollama_model!r} ({error}); falling back to "
                f"{args.hf_repo}.",
                file=sys.stderr,
            )
        else:
            slug = re.sub(r"[^A-Za-z0-9._-]+", "-", args.ollama_model)
            return cache_gguf(path, f"{slug}.gguf"), f"ollama:{args.ollama_model}"

    path = download_hf_gguf(args.hf_repo, args.hf_file)
    return cache_gguf(path, path.name), f"huggingface:{args.hf_repo}"


def require_runtime() -> None:
    """Fail early, and usefully, when the inference dependencies are absent."""

    for module, hint in (
        ("torch", "torch is only installed in the GPU image"),
        ("transformers", "transformers loads the GGUF"),
        ("accelerate", "transformers requires accelerate to load a GGUF"),
    ):
        try:
            __import__(module)
        except ImportError:
            raise ImportError(
                f"{module} is not installed ({hint}).\n\n"
                "Run this script from the GPU shell, which has the full "
                "dependency set:\n\n"
                "    ./scripts/python-shell-gpu.sh"
            ) from None


def select_device(requested: str) -> str:
    import torch

    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but no GPU is visible")
        return requested

    return "cuda" if torch.cuda.is_available() else "cpu"


def select_dtype(requested: str, device: str):
    """
    Choose a compute dtype.

    Gemma 3 overflows in float16: activations exceed its range, the hidden
    states go to NaN, and generation degenerates into padding. Only bfloat16
    and float32 are safe, so float16 is never chosen automatically.
    """

    import torch

    if requested == "float16":
        print(
            "warning: Gemma 3 overflows in float16 and will emit empty "
            "answers. Use bfloat16 or float32 instead.",
            file=sys.stderr,
        )

    if requested != "auto":
        return getattr(torch, requested)

    if device != "cuda":
        return torch.float32

    # Pre-Ampere cards, such as the GTX 10 series, have no native bfloat16,
    # but torch still runs it by converting, which is both correct and far
    # quicker than keeping a 4B model in float32.
    if torch.cuda.get_device_capability()[0] < 8:
        print(
            "note: this GPU has no native bfloat16, so bfloat16 ops are "
            "converted rather than run directly."
        )

    return torch.bfloat16


class Gemma3Baseline:
    """Loads a Gemma 3 GGUF once, then answers prompts greedily by default."""

    def __init__(
        self,
        gguf_path: Path,
        device: str,
        dtype,
        system: str | None,
        vocab_size: int | None = None,
    ):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.system = system

        directory, filename = str(gguf_path.parent), gguf_path.name

        print(f"Loading {gguf_path} on {device} as {dtype}...")
        started = time.perf_counter()

        config = AutoConfig.from_pretrained(directory, gguf_file=filename)

        # Gemma 3 GGUFs do not always carry a vocab_size, in which case
        # Transformers counts the tokenizer's tokens instead. Ollama's blob
        # lists one more token than the embedding has rows, which makes the
        # load fail on a size mismatch, so prefer the embedding's own shape.
        if vocab_size is not None and config.vocab_size != vocab_size:
            print(
                f"Overriding config vocab_size {config.vocab_size} with "
                f"{vocab_size} from {EMBEDDING_TENSOR}"
            )
            config.vocab_size = vocab_size

        self.tokenizer = AutoTokenizer.from_pretrained(directory, gguf_file=filename)
        self.model = AutoModelForCausalLM.from_pretrained(
            directory, gguf_file=filename, config=config, dtype=dtype
        )
        self.model.to(device)
        self.model.eval()

        print(f"Loaded in {time.perf_counter() - started:.1f} s")

        self.stop_ids = self._stop_ids()

        # Nothing is padded at a batch size of one, but generate() still wants
        # an id. Test for None rather than truthiness, because <pad> is id 0.
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.stop_ids[0] if self.stop_ids else 0
        self.pad_id = pad

    def _stop_ids(self) -> list[int]:
        ids: set[int] = set()

        eos = self.tokenizer.eos_token_id
        if isinstance(eos, int):
            ids.add(eos)
        elif eos:
            ids.update(eos)

        end_of_turn = self.tokenizer.convert_tokens_to_ids(END_OF_TURN)
        if end_of_turn is not None and end_of_turn != self.tokenizer.unk_token_id:
            ids.add(int(end_of_turn))

        return sorted(ids)

    def _encode(self, prompt: str):
        """Apply the model's chat template, or Gemma 3's turn format."""

        torch = self.torch

        # Gemma 3 has no system role, so a system prompt leads the user turn.
        content = f"{self.system}\n\n{prompt}" if self.system else prompt

        if getattr(self.tokenizer, "chat_template", None):
            ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:
            ids = self.tokenizer(
                GEMMA3_TURN.format(content=content),
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids

        # Gemma 3 expects a leading BOS, but tokenizers rebuilt from a GGUF
        # often have add_bos_token off, and chat templates usually write BOS
        # themselves. Add it only when it is actually missing.
        bos = self.tokenizer.bos_token_id
        if bos is not None and (not ids.shape[-1] or int(ids[0, 0]) != bos):
            prefix = torch.tensor([[bos]], dtype=ids.dtype, device=ids.device)
            ids = torch.cat([prefix, ids], dim=-1)

        return ids

    def generate(
        self, prompt: str, max_new_tokens: int, temperature: float, top_p: float
    ) -> tuple[str, int, int, float]:
        torch = self.torch

        input_ids = self._encode(prompt).to(self.device)
        sampling = temperature > 0

        if self.device == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()

        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=sampling,
                temperature=temperature if sampling else None,
                top_p=top_p if sampling else None,
                eos_token_id=self.stop_ids or None,
                pad_token_id=self.pad_id,
            )

        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        prompt_tokens = int(input_ids.shape[-1])
        completion = output[0][prompt_tokens:].tolist()

        # `<end_of_turn>` is not flagged as special in GGUF-derived tokenizers,
        # so trim the answer at the first stop token rather than relying on
        # `skip_special_tokens` to hide it.
        stop = set(self.stop_ids)
        answer = list(itertools.takewhile(lambda token: token not in stop, completion))
        text = self.tokenizer.decode(answer, skip_special_tokens=True).strip()

        return text, prompt_tokens, len(completion), elapsed


def markdown_report(report: dict[str, Any]) -> str:
    model, summary = report["model"], report["summary"]
    lines = [
        "# Gemma 3 host baseline",
        "",
        f"- Source: {model['source']}",
        f"- GGUF: {model['gguf']['path']}",
        f"- Device: {model['device']} ({model['dtype']})",
        f"- Samples: {summary['samples']}",
        f"- Passed: {summary['passed']}",
        f"- Pass rate: {summary['pass_rate']:.1%}",
        f"- Mean score: {summary['mean_score']:.3f}",
        f"- Mean latency: {summary['mean_latency_seconds']:.3f} s",
        f"- Mean throughput: {summary['mean_tokens_per_second']:.1f} tokens/s",
        "",
        "| Sample | Evaluator | Score | Passed | Latency | Tokens/s |",
        "| --- | --- | ---: | :---: | ---: | ---: |",
    ]
    for result in report["results"]:
        sample_id = str(result["id"]).replace("|", "\\|")
        lines.append(
            f"| {sample_id} | {result['evaluator']} | {result['score']:.3f} | "
            f"{'yes' if result['passed'] else 'no'} | "
            f"{result['latency_seconds']:.3f} s | "
            f"{result['tokens_per_second']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def write_report(
    output_dir: Path, report: dict[str, Any], results: list[BaselineResult]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "report.md").write_text(markdown_report(report), encoding="utf-8")

    # Answers are written out individually so that a known-good host run can
    # be reused as the expected output of a device evaluation.
    answers_dir = output_dir / "answers"
    answers_dir.mkdir(exist_ok=True)
    for result in results:
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", result.id)
        (answers_dir / f"{name}.txt").write_text(result.actual + "\n", encoding="utf-8")


def summarize(results: list[BaselineResult]) -> dict[str, Any]:
    count = len(results)
    completion_tokens = sum(result.completion_tokens for result in results)
    elapsed = sum(result.latency_seconds for result in results)
    return {
        "samples": count,
        "passed": sum(result.passed for result in results),
        "pass_rate": sum(result.passed for result in results) / count,
        "mean_score": sum(result.score for result in results) / count,
        "mean_latency_seconds": elapsed / count,
        "mean_tokens_per_second": completion_tokens / elapsed if elapsed else 0.0,
        "completion_tokens": completion_tokens,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])

    parser.add_argument(
        "manifest", type=Path, help="JSON manifest containing a samples list"
    )

    parser.add_argument(
        "--source",
        default="auto",
        choices=["auto", "ollama", "huggingface"],
        help="Where to obtain the GGUF; 'auto' tries Ollama first",
    )

    parser.add_argument(
        "--gguf",
        type=Path,
        default=None,
        help="Path to an int8 GGUF to use instead of downloading one",
    )

    parser.add_argument(
        "--ollama-model",
        default=OLLAMA_MODEL,
        help=f"Ollama model to run (default: {OLLAMA_MODEL})",
    )

    parser.add_argument("--hf-repo", default=HF_REPO)
    parser.add_argument("--hf-file", default=HF_FILE)

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Inference device (default: CUDA when a GPU is visible)",
    )

    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Compute dtype (default: bfloat16 on CUDA, float32 on CPU)",
    )

    parser.add_argument("--system", default=None, help="System prompt")
    parser.add_argument("--max-new-tokens", type=int, default=512)

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; 0 selects greedy decoding (default: 0)",
    )

    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--output-dir", type=Path, default=Path("baseline-results"))
    parser.add_argument("--min-pass-rate", type=float, default=0.0)
    parser.add_argument("--min-mean-score", type=float, default=0.0)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        manifest_path = args.manifest.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        samples = manifest["samples"]
        if not isinstance(samples, list) or not samples:
            raise ValueError("manifest 'samples' must be a non-empty list")
        if not 0 <= args.min_pass_rate <= 1 or not 0 <= args.min_mean_score <= 1:
            raise ValueError("quality thresholds must be between 0 and 1")

        require_runtime()

        gguf_path, source = resolve_gguf(args)
        gguf_summary = describe_gguf(gguf_path)

        device = select_device(args.device)
        dtype = select_dtype(args.dtype, device)
        baseline = Gemma3Baseline(
            gguf_path, device, dtype, args.system, gguf_summary.get("vocab_size")
        )

        results: list[BaselineResult] = []
        interrupted = False

        for index, sample in enumerate(samples, 1):
            sample_id = str(sample.get("id", index))
            evaluator_name = sample.get("evaluator", "exact_match")
            if evaluator_name not in EVALUATORS:
                raise ValueError(f"sample {sample_id}: unknown evaluator {evaluator_name!r}")

            prompt = read_text(manifest_path.parent, sample["prompt"], "prompt")
            expected = read_text(manifest_path.parent, sample["expected"], "expected").strip()

            try:
                actual, prompt_tokens, completion_tokens, latency = baseline.generate(
                    prompt, args.max_new_tokens, args.temperature, args.top_p
                )
            except KeyboardInterrupt:
                print("\nInterrupted; writing a partial report.", file=sys.stderr)
                interrupted = True
                break

            score = EVALUATORS[evaluator_name](actual, expected, sample)
            threshold = float(
                sample.get("threshold", 1.0 if evaluator_name in {"exact_match", "json"} else 0.8)
            )
            results.append(
                BaselineResult(
                    sample_id, evaluator_name, score, score >= threshold, latency,
                    actual, expected, "", 0,
                    prompt_tokens, completion_tokens,
                    completion_tokens / latency if latency else 0.0,
                )
            )
            print(
                f"[{index}/{len(samples)}] {sample_id}: {score:.3f} "
                f"{'PASS' if results[-1].passed else 'FAIL'} "
                f"({completion_tokens} tokens, {latency:.2f} s)"
            )

        if not results:
            raise ValueError("no samples were evaluated")

        summary = summarize(results)
        report = {
            "manifest": str(manifest_path),
            "model": {
                "source": source,
                "gguf": gguf_summary,
                "device": device,
                "dtype": str(dtype).removeprefix("torch."),
                "system": args.system,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "decoding": "greedy" if args.temperature <= 0 else "sampling",
            },
            "summary": summary,
            "results": [asdict(result) for result in results],
        }
        write_report(args.output_dir, report, results)
        print(f"Reports written to {args.output_dir.resolve()}")

        if interrupted:
            return 130
        return int(
            summary["pass_rate"] < args.min_pass_rate
            or summary["mean_score"] < args.min_mean_score
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except (OSError, KeyError, TypeError, ValueError, ImportError,
            RuntimeError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
