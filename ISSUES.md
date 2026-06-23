# Issues

Findings from a source and documentation scan on 2026-06-17. Grouped by severity. Line references are approximate and may drift as the code changes.

## Documentation

### 1. README "Models" section omits SmolVLM2

The intro lists SmolVLM2 as currently supported, but the Models section only documents Qwen2-VL-2B/7B downloads — there is no SmolVLM2 model source or example. Also minor naming inconsistency: the support list says "Qwen-VL2" while the rest of the doc uses "Qwen2-VL".

## CI gaps

### 2. Host/unit tests are built but never run

Both build scripts pass `-DBUILD_TESTING=ON` and CMake builds `vlm-rknn-tests`, but no workflow runs `ctest`. The native build is a cross-compile to aarch64, so the tests can't run on the x86 GitHub runner. There is currently no job that executes the test binary anywhere. (PLAN.md already lists "Add a CI job that runs the native build end-to-end" as a TODO.)
