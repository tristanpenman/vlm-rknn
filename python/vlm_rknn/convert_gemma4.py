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

"""Convert google/gemma-4-E2B-it language weights to RKLLM FP16 format.

This exports only the language model. Image input also needs a separate RKNN
vision encoder, as described in notes/gemma4.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .rkllm_conversion import CACHE_DIR, convert_rkllm, download_hf_model


HF_MODEL = "google/gemma-4-E2B-it"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-dir", type=Path, default=CACHE_DIR / "gemma-4-E2B-it",
        help="Directory for the Hugging Face checkpoint",
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        default=Path("gemma-4-E2B-it-fp16-rk3588.rkllm"),
        help="Destination RKLLM file",
    )
    parser.add_argument("--target", choices=["rk3588", "rk3576"], default="rk3588")
    parser.add_argument("--npu-cores", type=int, default=None)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--optimization-level", type=int, choices=[0, 1], default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    max_cores = 3 if args.target == "rk3588" else 2
    npu_cores = args.npu_cores if args.npu_cores is not None else max_cores
    if not 1 <= npu_cores <= max_cores:
        parser.error(f"--npu-cores must be between 1 and {max_cores} for {args.target}")
    if not 32 <= args.max_context <= 16384 or args.max_context % 32:
        parser.error("--max-context must be a multiple of 32 between 32 and 16384")

    model_path = download_hf_model(HF_MODEL, args.hf_dir)
    convert_rkllm(
        model_path=model_path,
        output_path=args.output,
        target=args.target,
        quantization="fp16",
        npu_cores=npu_cores,
        max_context=args.max_context,
        hybrid_rate=0,
        device=args.device,
        dtype="float16",
        optimization_level=args.optimization_level,
    )


if __name__ == "__main__":
    main()
