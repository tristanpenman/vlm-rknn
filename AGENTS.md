# Agents

## Project Context

This repo contains a starter CMake project for running Qwen-VL style multimodal models on Rockchip devices via RKNN/RKLLM.

## Structure

The `cpp` directory contains C++ source code for a Qwen-VL demo.

The `thirdparty` directory contains RKNN and RKLLM libraries.

## Build

To build for Android:

```
docker compose run --rm android ./scripts/build-android.sh Release
```

To build for native Linux:

```
docker compose run --rm native ./scripts/build-native.sh Release
```

## Caveats

The build is intended to be run via Docker. The target is an RK3588 device, so the output of these builds cannot be run locally.
