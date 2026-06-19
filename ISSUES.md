# Issues

Findings from a source and documentation scan on 2026-06-17. Grouped by severity. Line references are approximate and may drift as the code changes.

## Bugs / breakage

### 1. `build-native.sh` passes an undefined CMake option

`scripts/build-native.sh` passes `-DVLM_RKNN_ENABLE_RKNN="${VLM_RKNN_ENABLE_RKNN:-OFF}"`, but no such option exists in `CMakeLists.txt` (only `VLM_RKNN_ENABLE_OPENCV`). CMake emits a "manually-specified variables were not used" warning and the flag has no effect — RKNN is always required (the build `FATAL_ERROR`s if the runtime libs are missing). `build-android.sh` does not pass this flag, so the two scripts are inconsistent. Remove the flag or wire up a real option.

## Behavior concerns

### 2. `--cores 1` silently maps to AUTO

In `Session::init_vision_encoder()` only `core_num == 2` (`RKNN_NPU_CORE_0_1`) and `== 3` (`RKNN_NPU_CORE_0_1_2`) are explicit; every other accepted value (i.e. `1`) falls through to `RKNN_NPU_CORE_AUTO` rather than a single core (`RKNN_NPU_CORE_0`). The validation in `main.cc` accepts 1–3, so `--cores 1` does not do what a user would expect. Either map 1 to a single core or document that 1 means "auto".

### 3. `std::optional<int>` compared directly against `int`

In `main.cc` the parsed values are compared as e.g. `num_cores <= 0`, `max_new_tokens <= 0`, `max_context_len <= 0` where the left side is a `std::optional<int>`. This compiles via `std::optional`'s heterogeneous comparison operators and is correct only because the optional was just assigned (engaged). It is fragile/surprising — prefer comparing `*value` or the freshly parsed `int` before storing into the optional.

## Documentation

### 4. README "Models" section omits SmolVLM2

The intro lists SmolVLM2 as currently supported, but the Models section only documents Qwen2-VL-2B/7B downloads — there is no SmolVLM2 model source or example. Also minor naming inconsistency: the support list says "Qwen-VL2" while the rest of the doc uses "Qwen2-VL".

## CI gaps

### 5. Host/unit tests are built but never run

Both build scripts pass `-DBUILD_TESTING=ON` and CMake builds `vlm-rknn-tests`, but no workflow runs `ctest`. The native build is a cross-compile to aarch64, so the tests can't run on the x86 GitHub runner. There is currently no job that executes the test binary anywhere. (PLAN.md already lists "Add a CI job that runs the native build end-to-end" as a TODO.)
