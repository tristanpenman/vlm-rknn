# Issues

Findings from a source and documentation scan on 2026-06-17. Grouped by severity. Line references are approximate and may drift as the code changes.

## Bugs / breakage

### 1. `build-native.sh` passes an undefined CMake option

`scripts/build-native.sh` passes `-DVLM_RKNN_ENABLE_RKNN="${VLM_RKNN_ENABLE_RKNN:-OFF}"`, but no such option exists in `CMakeLists.txt` (only `VLM_RKNN_ENABLE_OPENCV`). CMake emits a "manually-specified variables were not used" warning and the flag has no effect — RKNN is always required (the build `FATAL_ERROR`s if the runtime libs are missing). `build-android.sh` does not pass this flag, so the two scripts are inconsistent. Remove the flag or wire up a real option.

## Documentation

### 2. README "Models" section omits SmolVLM2

The intro lists SmolVLM2 as currently supported, but the Models section only documents Qwen2-VL-2B/7B downloads — there is no SmolVLM2 model source or example. Also minor naming inconsistency: the support list says "Qwen-VL2" while the rest of the doc uses "Qwen2-VL".

## CI gaps

### 3. Host/unit tests are built but never run

Both build scripts pass `-DBUILD_TESTING=ON` and CMake builds `vlm-rknn-tests`, but no workflow runs `ctest`. The native build is a cross-compile to aarch64, so the tests can't run on the x86 GitHub runner. There is currently no job that executes the test binary anywhere. (PLAN.md already lists "Add a CI job that runs the native build end-to-end" as a TODO.)
