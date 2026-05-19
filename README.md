# Qwen-VL RKNN

This repo contains a starter CMake project for running Qwen-VL style multimodal models on Rockchip devices via RKNN/RKLLM.

The layout is modeled after my [Marian RKNN](https://github.com/tristanpenman/marian-rknn) implementation and is intentionally small while the model runtime is being brought up.

The project targets Rockchip Linux and Android devices.

OpenCV is being integrated as a fetched third-party dependency for image loading and preprocessing. The initial CMake integration builds a small OpenCV module set (`core,imgproc,imgcodecs`) by default.

## Layout

- `src/` - C++ placeholder library and CLI entry point.
- `tests/` - Placeholder CTest-based executable tests.
- `scripts/` - Host, Rockchip Linux, and Android build helpers.
- `Dockerfile.native` - Linux/aarch64 cross-build environment for Rockchip boards.
- `Dockerfile.android` - Android NDK build environment.
- `docker-compose.yml` - Convenience services for the development containers.

## Rockchip Linux cross-build

```bash
docker compose run --rm native ./scripts/build-native.sh Release
```

Set `QWEN_VL_RKNN_ENABLE_RKNN=ON` when the RKNN headers and runtime are available under `thirdparty/rknpu2`, or pass `-DRKNN_INCLUDE_DIR` and `-DRKNN_RUNTIME_LIB` through your own CMake invocation.

OpenCV is fetched during CMake configure when `QWEN_VL_RKNN_ENABLE_OPENCV=ON` (the default). Override the fetched release (default `4.13.0`) with `-DQWEN_VL_RKNN_OPENCV_GIT_TAG=<tag-or-commit>` or trim/expand modules with `-DQWEN_VL_RKNN_OPENCV_MODULES=<comma-separated-modules>`. OpenCV sample projects are disabled by default, including Android samples; opt in with `-DQWEN_VL_RKNN_OPENCV_BUILD_SAMPLES=ON`. The default OpenCV image codec configuration keeps PNG and JPEG enabled and disables optional non-PNG/JPEG codecs.

## Android build

```bash
docker compose run --rm android ./scripts/build-android.sh Release
```

The Android helper expects the NDK from the Android container or an `ANDROID_NDK_HOME` path supplied by the caller.

## License

This code is released under the Apache License 2.0 license. See the [LICENSE](./LICENSE) file for more information.
