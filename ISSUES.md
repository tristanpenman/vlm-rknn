# Issues

Findings from a source and documentation scan on 2026-06-17. Grouped by severity. Line references are approximate and may drift as the code changes.

## Bugs / breakage

### 1. `build-native.sh` passes an undefined CMake option

`scripts/build-native.sh` passes `-DVLM_RKNN_ENABLE_RKNN="${VLM_RKNN_ENABLE_RKNN:-OFF}"`, but no such option exists in `CMakeLists.txt` (only `VLM_RKNN_ENABLE_OPENCV`). CMake emits a "manually-specified variables were not used" warning and the flag has no effect — RKNN is always required (the build `FATAL_ERROR`s if the runtime libs are missing). `build-android.sh` does not pass this flag, so the two scripts are inconsistent. Remove the flag or wire up a real option.

## Dead / duplicate code

### 2. `expand2square()` is unused and duplicates `pad_to_square()`

`cpp/src/cv2_utils.{h,cc}` define `expand2square()`, but nothing calls it (`grep` finds only the definition). The live code path uses `pad_to_square()`, a near-identical copy in the anonymous namespace of `cpp/src/vlm_rknn.cc`. The file is still compiled into both `vlm_rknn_core` (CMakeLists.txt) and the Android JNI lib (`android/app/src/main/cpp/CMakeLists.txt`). Either delete `cv2_utils.*` and drop it from both build files, or consolidate the two square-padding implementations into one.

### 3. `Logger::Writer::m_level` is stored but never read

`cpp/src/logger.h` keeps `m_level` as a member and `logger.cc` initializes it, but only `m_enabled` is used. This is dead state and may trip `-Wunused-private-field` under Clang.

## Behavior concerns

### 4. Vision path always runs a hard-coded warm-up decode

`cpp/src/main.cc` (~308-316) unconditionally runs a decode with `"<image>What is in the image?"` before handling the user's `--prompt` or starting the REPL. This emits real model output the user did not ask for and consumes tokens/context on every multimodal run. If this is intended as a warm-up it should be silent (suppressed output) or gated behind `--verbose`; otherwise it looks like leftover debug code.

### 5. `--cores 1` silently maps to AUTO

In `Session::init_vision_encoder()` only `core_num == 2` (`RKNN_NPU_CORE_0_1`) and `== 3` (`RKNN_NPU_CORE_0_1_2`) are explicit; every other accepted value (i.e. `1`) falls through to `RKNN_NPU_CORE_AUTO` rather than a single core (`RKNN_NPU_CORE_0`). The validation in `main.cc` accepts 1–3, so `--cores 1` does not do what a user would expect. Either map 1 to a single core or document that 1 means "auto".

### 6. `std::optional<int>` compared directly against `int`

In `main.cc` the parsed values are compared as e.g. `num_cores <= 0`, `max_new_tokens <= 0`, `max_context_len <= 0` where the left side is a `std::optional<int>`. This compiles via `std::optional`'s heterogeneous comparison operators and is correct only because the optional was just assigned (engaged). It is fragile/surprising — prefer comparing `*value` or the freshly parsed `int` before storing into the optional.

## Documentation

### 7. README "Models" section omits SmolVLM2

The intro lists SmolVLM2 as currently supported, but the Models section only documents Qwen2-VL-2B/7B downloads — there is no SmolVLM2 model source or example. Also minor naming inconsistency: the support list says "Qwen-VL2" while the rest of the doc uses "Qwen2-VL".

## CI gaps

### 8. Host/unit tests are built but never run

Both build scripts pass `-DBUILD_TESTING=ON` and CMake builds `vlm-rknn-tests`, but no workflow runs `ctest`. The native build is a cross-compile to aarch64, so the tests can't run on the x86 GitHub runner. There is currently no job that executes the test binary anywhere. (PLAN.md already lists "Add a CI job that runs the native build end-to-end" as a TODO.)
