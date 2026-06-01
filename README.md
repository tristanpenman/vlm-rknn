# Qwen-VL RKNN

This repo contains a starter CMake project for running Qwen-VL style multimodal models on Rockchip devices via RKNN/RKLLM.

The layout is modeled after my [Marian RKNN](https://github.com/tristanpenman/marian-rknn) implementation and is intentionally small while the model runtime is being brought up.

## Background

### Qwen

Qwen is a family of open-weight large language models from Alibaba's Tongyi team, released under permissive licences and widely used as a base for downstream fine-tuning and on-device deployment.

### Qwen-VL

Qwen-VL extends Qwen with a vision encoder, producing a multimodal model that takes images and text and emits text. This repo runs Qwen-VL on Rockchip NPUs, with the language model executed via RKLLM and the vision encoder via RKNN.

## Project Structure

This project targets Rockchip Linux and Android devices based on the Rockchip RK3588. CMake is used for both platforms.

### Layout

- `cpp/src/` - C++ library sources and CLI entry point.
- `cpp/tests/` - CTest-based executable tests.
- `cmake/` - CMake helper modules (e.g. third-party fetch configuration).
- `scripts/` - Linux and Android build helper scripts.
- `thirdparty/` - Bundled RKNN and RKLLM headers and prebuilt runtime libraries.
- `CMakeLists.txt` - CMake build configuration.
- `Dockerfile.android` - Android NDK build environment.
- `Dockerfile.native` - Linux/aarch64 cross-build environment.
- `docker-compose.yml` - Convenience services for the development containers.

### Dependencies

OpenCV is being integrated as a fetched third-party dependency for image loading and preprocessing. The initial CMake integration builds a small OpenCV module set (`core,imgproc,imgcodecs`) by default.

## Native Implementation

Use the native build wrapper script:

```bash
docker compose run --rm native ./scripts/build-native.sh Release
```

The RKNN and RKLLM headers and runtime libraries are expected under `thirdparty/rknpu2` and `thirdparty/rkllm` respectively.

Override the defaults using one or more of the following when invoking CMake:
* `-DRKNN_INCLUDE_DIR`
* `-DRKNN_RUNTIME_LIB`
* `-DRKLLM_INCLUDE_DIR`
* `-DRKLLM_RUNTIME_LIB`

OpenCV is fetched when configuring CMake. Override the fetched release (default `4.13.0`) with `-DQWEN_VL_RKNN_OPENCV_GIT_TAG=<tag-or-commit>`.

You may also trim or expand the set of modules to be built with `-DQWEN_VL_RKNN_OPENCV_MODULES=<comma-separated-modules>`.

OpenCV sample projects are disabled by default. The default OpenCV image codec configuration keeps PNG and JPEG enabled and disables optional non-PNG/JPEG codecs.

## Android 14

Use the Android build wrapper script:

```bash
docker compose run --rm android ./scripts/build-android.sh Release
```

The Android helper expects the NDK from the Android container or an `ANDROID_NDK_HOME` path supplied by the caller.

## Contributing

Contributions are welcome. I will make an effort to review any bona fide contributions.

You are also welcome to raise GitHub issues against this repo, however please note this is merely a hobby project. I cannot offer any guarantee that issues will be responded to in a timely fashion.

## License

This code is released under the Apache License 2.0 license. See the [LICENSE](./LICENSE) file for more information.
