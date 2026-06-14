# Agents

## Project Context

This repo contains a starter CMake project for running Qwen-VL style multimodal models on Rockchip devices via RKNN/RKLLM.

## Structure

The `cpp` directory contains C++ source code for a Qwen-VL demo.

The `thirdparty` directory contains RKNN and RKLLM libraries.

## Guidelines

Do not try to run CMake directly. Use the build steps documented below.

After making any C++ or build-system change, always check the Docker Compose builds before finalising the task. While iterating on changes, building for `native` is fine. To finalise changes, ensure both the native and Android builds have been attempted:

```bash
docker compose run --rm native ./scripts/build-native.sh Release
docker compose run --rm android ./scripts/build-android.sh Release
```

If a Docker Compose build cannot be run or fails for an environment reason, report that explicitly along with the command that was attempted.

## Build

To build for Android:

```bash
docker compose run --rm android ./scripts/build-android.sh Release
```

To build for native Linux:

```bash
docker compose run --rm native ./scripts/build-native.sh Release
```

## Caveats

The build is intended to be run via Docker. The target is an RK3588 device, so the output of these builds cannot be run locally.
