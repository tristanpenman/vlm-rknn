# Qwen-VL RKNN

This repo contains a starter CMake project for running Qwen-VL style multimodal models on Rockchip devices via RKNN/RKLLM.

The layout is modeled after my [Marian RKNN](https://github.com/tristanpenman/marian-rknn) implementation and is intentionally small while the model runtime is being brought up.

### Contents

* [Background](#background)
  * [Qwen](#qwen)
  * [Qwen-VL](#qwen-vl)
* [Project Structure](#project-structure)
  * [Layout](#layout)
  * [Dependencies](#dependencies)
* [Native Implementation](#native-implementation)
* [Android 14](#android-14)
* [Models](#models)
  * [Qwen2-VL-2b](#qwen2-vl-2b)
  * [Qwen2-VL-7b](#qwen2-vl-7b)
* [Contributing](#contributing)
* [License](#license)

## Background

A Qwen-VL model is a vision-language model from Alibaba’s Qwen family.

VL is short for "Vision-Language". What this means is that input to the model can be in the form of images, text, and sometimes video. However the output will always be text.

Immediate applications include describing or answering questions about images, reading text in screenshots or documents, and parsing graphical user interfaces.

### Qwen

Qwen is a family of open-weight large language models from Alibaba's Tongyi team, released under permissive licences and widely used as a base for downstream fine-tuning and on-device deployment.

### Qwen-VL

Qwen-VL extends Qwen with a vision encoder, producing a multimodal model that takes images and text as input and emits text. Qwen-VL models are extensions of corresponding Qwen base models:

| Model      | Description                                       |
|------------|---------------------------------------------------|
| Qwen       | Alibaba’s LLM family                              |
| Qwen-VL    | Qwen with vision-language capability              |
| Qwen2-VL   | Newer vision-language generation                  |
| Qwen2.5-VL | Stronger newer generation                         |
| Qwen3-VL   | Later generation with further multimodal upgrades |

This repo runs Qwen-VL on Rockchip NPUs, with the language model executed via RKLLM and the vision encoder via RKNN. It aims to provide full support for Qwen2, Qwen2.5 and Qwen3 models.

## Project Structure

This project targets Rockchip Linux and Android devices based on the Rockchip RK3588. CMake is used for both platforms.

### Layout

- `cmake/` - CMake helper modules (e.g. third-party fetch configuration).
- `cpp/src/` - C++ library sources and CLI entry point.
- `cpp/tests/` - CTest-based executable tests.
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

## Models

### Qwen2-VL-2B

Qwen2-VL is distributed in several sizes, ranging from 2B (2 billion parameters) to 7B and 72B parameters. I recommend starting with the 2B model, which weighs in at about 4.5GB. This fits with plenty of room to spare on a 16GB device, such as the Khadas Edge2.

You can download compatible Qwen2-VL models from Hugging Face:

👉 [Qwen2-VL-2B-rkllm](https://huggingface.co/3ib0n/Qwen2-VL-2B-rkllm)

Fetch the `Qwen2-VL-2B-Instruct.rkllm` and `qwen2_vl_2b_vision_rk3588.rknn` model files.

### Qwen2-VL-7B

Alternatively, you can fetch the 7B version. This weighs in at around 9.6GB. These can also be fetched from Hugging Face:

👉 [Qwen2-VL-7B-rkllm](https://huggingface.co/3ib0n/Qwen2-VL-7B-rkllm)

Fetch the `Qwen2-VL-7B-Instruct.rkllm` and `qwen2_vl_7b_vision_rk3588.rknn` model files.

## Contributing

Contributions are welcome. I will make an effort to review any bona fide contributions.

You are also welcome to raise GitHub issues against this repo, however please note this is merely a hobby project. I cannot offer any guarantee that issues will be responded to in a timely fashion.

## License

This code is released under the Apache License 2.0.

See the [LICENSE](./LICENSE) file for more information.
