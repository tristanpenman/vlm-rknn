# Vendored Rockchip runtimes

This directory contains the prebuilt Rockchip runtime libraries and matching C API headers used by the native and Android builds. The directory layout follows the convention used by Rockchip's examples. Platform directories are named `Linux` and `Android`, with shared headers under `include`.

## RKLLM

- Version: 1.3.1
- Release: [`release-v1.3.1`](https://github.com/airockchip/rknn-llm/releases/tag/release-v1.3.1)
- Commit: [`f7390530443bf84f0394255a449d7cbe81e69d1c`](https://github.com/airockchip/rknn-llm/commit/f7390530443bf84f0394255a449d7cbe81e69d1c)
- Source paths: `rkllm-runtime/{Linux,Android}/librkllm_api`

## RKNPU2

- Version: 2.3.2
- Release: [`v2.3.2`](https://github.com/airockchip/rknn-toolkit2/releases/tag/v2.3.2)
- Commit: [`42aa1d426c0a9e0869b6374edba009f7208a1926`](https://github.com/airockchip/rknn-toolkit2/commit/42aa1d426c0a9e0869b6374edba009f7208a1926)
- Source paths: `rknpu2/runtime/{Linux,Android}/librknn_api`
