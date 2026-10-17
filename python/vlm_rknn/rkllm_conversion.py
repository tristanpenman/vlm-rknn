# Copyright (c) 2026 Tristan Penman
# Licensed under the Apache License, Version 2.0 (the "License");

"""Shared Hugging Face download and RKLLM conversion helpers."""

from __future__ import annotations

import os
from pathlib import Path

CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", ".cache"))
RKLLM_CACHE_DIR = CACHE_DIR / "rkllm-toolkit"


def download_hf_model(model_id: str, output_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    print(f"Downloading {model_id}...")
    return Path(snapshot_download(repo_id=model_id, local_dir=str(output_dir)))


def configure_rkllm_cache(cache_dir: Path) -> Path:
    """Redirect toolkit scratch files to a writable cache beside linked libraries."""
    import rkllm.base.opfbs as opfbs

    source_lib_dir = Path(opfbs.root_dir) / "lib"
    cache_dir = cache_dir.resolve()
    cache_lib_dir = cache_dir / "lib"
    cache_lib_dir.mkdir(parents=True, exist_ok=True)

    for source in source_lib_dir.iterdir():
        if not source.is_file():
            continue

        link = cache_lib_dir / source.name
        if link.is_symlink() and link.resolve() == source.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise RuntimeError(
                "RKLLM cache library path already exists and is not the "
                f"expected link: {link}"
            )
        link.symlink_to(source.resolve())

    opfbs.root_dir = str(cache_dir)
    return cache_dir


def convert_rkllm(
    model_path: Path,
    output_path: Path,
    *,
    target: str,
    quantization: str,
    npu_cores: int,
    max_context: int,
    hybrid_rate: float,
    device: str,
    dtype: str,
    optimization_level: int,
) -> None:
    """Load, build, and export an RKLLM model.

    ``fp16`` keeps the loaded float16 weights instead of running quantization.
    """
    from rkllm.api import RKLLM

    if quantization == "fp16":
        if dtype != "float16":
            raise ValueError("FP16 export requires --dtype float16")
        if hybrid_rate != 0:
            raise ValueError("FP16 export requires --hybrid-rate 0")
        do_quantization = False
        # The toolkit still accepts these build arguments when quantization is off.
        quantized_dtype = "w8a8"
        algorithm = "normal"
    elif quantization in {"w8a8", "w8a8_gx"}:
        do_quantization = True
        quantized_dtype = quantization
        algorithm = "normal"
    elif quantization in {"w4a16", "w4a16_gx"}:
        do_quantization = True
        quantized_dtype = quantization
        algorithm = "grq"
    else:
        raise ValueError(f"Unsupported quantization {quantization!r}")

    rkllm_cache_dir = configure_rkllm_cache(RKLLM_CACHE_DIR)
    print("\nRKLLM conversion\n----------------")
    for label, value in (
        ("Source", model_path), ("Output", output_path),
        ("Platform", target), ("Quantization", quantization),
        ("Algorithm", algorithm if do_quantization else "none"),
        ("NPU cores", npu_cores), ("Max context", max_context),
        ("Hybrid rate", hybrid_rate), ("Optimization", optimization_level),
        ("Converter", device), ("Source dtype", dtype),
        ("Scratch", rkllm_cache_dir),
    ):
        print(f"{label + ':':14}{value}")
    print()

    llm = RKLLM()
    ret = llm.load_huggingface(model=str(model_path), device=device, dtype=dtype)
    if ret != 0:
        raise RuntimeError(f"RKLLM load_huggingface() failed: {ret}")

    ret = llm.build(
        do_quantization=do_quantization,
        optimization_level=optimization_level,
        quantized_dtype=quantized_dtype,
        quantized_algorithm=algorithm,
        target_platform=target,
        num_npu_core=npu_cores,
        max_context=max_context,
        extra_qparams=None,
        dataset=None,
        hybrid_rate=hybrid_rate,
    )
    if ret != 0:
        raise RuntimeError(f"RKLLM build() failed: {ret}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ret = llm.export_rkllm(str(output_path))
    if ret != 0:
        raise RuntimeError(f"RKLLM export_rkllm() failed: {ret}")
    print(f"\nCreated: {output_path}")
