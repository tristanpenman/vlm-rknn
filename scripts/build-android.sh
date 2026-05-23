#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "${1:-}" == "docker" ]]; then
  shift
  cd "${ROOT_DIR}"
  exec docker compose run --rm --remove-orphans --build android ./scripts/build-android.sh "$@"
fi

BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-android}"
BUILD_TYPE="${1:-Release}"
NDK_PATH="${ANDROID_NDK_HOME:-}"

if [[ -z "${NDK_PATH}" ]]; then
  cat <<USAGE
Usage: ANDROID_NDK_HOME=<ndk-path> $0 [build-type]

Example:
  ANDROID_NDK_HOME=/opt/android-sdk/ndk/29.0.14206865 $0 Release
USAGE
  exit 1
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-34 \
  -DCMAKE_TOOLCHAIN_FILE="${NDK_PATH}/build/cmake/android.toolchain.cmake" \
  -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
  -DBUILD_TESTING=OFF \
  -DQWEN_VL_RKNN_ENABLE_RKNN="${QWEN_VL_RKNN_ENABLE_RKNN:-OFF}"
cmake --build "${BUILD_DIR}" -- -j"$(nproc)"

echo "Android build finished: ${BUILD_DIR}/qwen-vl-rknn"
