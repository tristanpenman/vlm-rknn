#!/usr/bin/env bash
set -euo pipefail

# Push the Android build of vlm-rknn-server to a device and start it.
#
# Usage:
#   ./scripts/run-android.sh <device-ip> [remote-path]
#
# Arguments:
#   device-ip    IP address (or host) of the target device, as used by `adb connect`.
#   remote-path  Directory to push files to on the device (default /data/local/tmp).
#
# Environment:
#   MODEL_SIZE   Qwen2-VL model to download and run: 2b (default) or 7b.
#   PORT         Port the server should listen on (default 8080).

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build-android}"
CACHE_DIR="${ROOT_DIR}/.cache"
MODEL_SIZE="${MODEL_SIZE:-2b}"
PORT="${PORT:-8080}"

DEVICE_IP="${1:-}"
REMOTE_DIR="${2:-/data/local/tmp}"

if [[ -z "${DEVICE_IP}" ]]; then
  cat <<USAGE
Usage: $0 <device-ip> [remote-path]

Arguments:
  device-ip    IP address (or host) of the target device, as used by adb connect.
  remote-path  Directory to push files to on the device (default /data/local/tmp).

Environment:
  MODEL_SIZE   Qwen2-VL model to download and run: 2b (default) or 7b.
  PORT         Port the server should listen on (default 8080).
USAGE
  exit 1
fi

# -----------------------------------------------------------------------------
# Preconditions
# -----------------------------------------------------------------------------

echo "=== Check preconditions ==="

SERVER_BIN="${BUILD_DIR}/vlm-rknn-server"
if [[ ! -x "${SERVER_BIN}" ]]; then
  echo "Error: ${SERVER_BIN} not found." >&2
  echo "Run the Android build first: ./scripts/build-android.sh docker" >&2
  exit 1
fi

echo "Server binary: ${SERVER_BIN}"

for tool in adb curl; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "Error: required tool '${tool}' is not installed or not on PATH." >&2
    exit 1
  fi
done

echo "All preconditions satisfied."

# -----------------------------------------------------------------------------
# Model selection
# -----------------------------------------------------------------------------

echo "=== Model selection ==="

case "${MODEL_SIZE}" in
  2b)
    HF_REPO="3ib0n/Qwen2-VL-2B-rkllm"
    LLM_FILE="Qwen2-VL-2B-Instruct.rkllm"
    VISION_FILE="qwen2_vl_2b_vision_rk3588.rknn"
    ;;
  7b)
    HF_REPO="3ib0n/Qwen2-VL-7B-rkllm"
    LLM_FILE="Qwen2-VL-7B-Instruct.rkllm"
    VISION_FILE="qwen2_vl_7b_vision_rk3588.rknn"
    ;;
  *)
    echo "Error: unsupported MODEL_SIZE '${MODEL_SIZE}' (expected 2b or 7b)." >&2
    exit 1
    ;;
esac

HF_BASE="https://huggingface.co/${HF_REPO}/resolve/main"
MODEL_CACHE="${CACHE_DIR}/qwen2-vl-${MODEL_SIZE}"

echo "Selected model size: ${MODEL_SIZE}"
echo "LLM file: ${LLM_FILE}"
echo "Vision file: ${VISION_FILE}"

# -----------------------------------------------------------------------------
# Download models
# -----------------------------------------------------------------------------

echo "=== Download models ==="

mkdir -p "${MODEL_CACHE}"

download() {
  local file="$1"
  local dest="${MODEL_CACHE}/${file}"
  if [[ -s "${dest}" ]]; then
    echo "Already cached: ${dest}"
    return
  fi
  echo "Downloading ${file} ..."
  curl -fL --progress-bar -o "${dest}.tmp" "${HF_BASE}/${file}"
  mv "${dest}.tmp" "${dest}"
}

download "${LLM_FILE}"
download "${VISION_FILE}"

# -----------------------------------------------------------------------------
# ADB connect
# -----------------------------------------------------------------------------

echo "=== Connect to device ==="

adb connect "${DEVICE_IP}" >/dev/null || true

# Resolve the serial: prefer an explicit ip:port match, fall back to the device.
SERIAL=""
if adb devices | awk '{print $1}' | grep -q "^${DEVICE_IP}:"; then
  SERIAL="$(adb devices | awk '{print $1}' | grep "^${DEVICE_IP}:" | head -n1)"
elif adb devices | awk 'NR>1 && $2=="device" {print $1}' | grep -qx "${DEVICE_IP}"; then
  SERIAL="${DEVICE_IP}"
fi

if [[ -z "${SERIAL}" ]]; then
  echo "Error: device ${DEVICE_IP} is not connected." >&2
  echo "Connected devices:" >&2
  adb devices >&2
  exit 1
fi

echo "Using device: ${SERIAL}"
ADB=(adb -s "${SERIAL}")

# -----------------------------------------------------------------------------
# Generate server configuration
# -----------------------------------------------------------------------------

echo "=== Generate server configuration ==="

REMOTE_MODEL_DIR="${REMOTE_DIR}/models/qwen2-vl"
REMOTE_LIB_DIR="${REMOTE_DIR}/lib"

INI_FILE="${CACHE_DIR}/server.android.ini"
cat > "${INI_FILE}" <<INI
[qwen2-vl]
model_family=qwen2-vl
vision=${REMOTE_MODEL_DIR}/${VISION_FILE}
llm=${REMOTE_MODEL_DIR}/${LLM_FILE}
max_new_tokens=300
INI

# -----------------------------------------------------------------------------
# Sync helpers
# -----------------------------------------------------------------------------

# Local file size in bytes (portable across macOS and Linux).
local_size() {
  stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

# Push a model file only if the device copy is missing or a different size.
# Large model files are expensive to transfer, so we compare byte size, which
# is instant on both ends and a strong signal for these immutable files.
sync_model() {
  local src="$1"
  local dst="$2"
  local name lsize rsize
  name="$(basename "${dst}")"

  lsize="$(local_size "${src}")"
  # A missing remote file makes stat exit non-zero; tolerate it (rsize stays
  # empty) so set -o pipefail / set -e don't abort the script on first push.
  rsize="$("${ADB[@]}" shell "stat -c%s '${dst}' 2>/dev/null" | tr -d '\r' || true)"

  if [[ "${rsize}" == "${lsize}" ]]; then
    echo "Up to date (${lsize} bytes): ${dst}"
    return
  fi

  echo "Pushing ${name} (local ${lsize} bytes, remote ${rsize:-missing}) ..."
  "${ADB[@]}" push "${src}" "${dst}"
}

# -----------------------------------------------------------------------------
# Push files
# -----------------------------------------------------------------------------

echo "=== Push files to device ==="

"${ADB[@]}" shell mkdir -p "${REMOTE_MODEL_DIR}" "${REMOTE_LIB_DIR}"

"${ADB[@]}" push "${SERVER_BIN}" "${REMOTE_DIR}/vlm-rknn-server"
"${ADB[@]}" push "${INI_FILE}" "${REMOTE_DIR}/server.ini"

"${ADB[@]}" push "${ROOT_DIR}/thirdparty/rknpu2/lib-android/arm64-v8a/librknnrt.so" "${REMOTE_LIB_DIR}/"
"${ADB[@]}" push "${ROOT_DIR}/thirdparty/rkllm/lib-android/arm64-v8a/librkllmrt.so" "${REMOTE_LIB_DIR}/"
"${ADB[@]}" push "${ROOT_DIR}/thirdparty/rkllm/lib-android/arm64-v8a/libomp.so" "${REMOTE_LIB_DIR}/"

# Model files are large; only push when missing or changed on the device.
sync_model "${MODEL_CACHE}/${LLM_FILE}" "${REMOTE_MODEL_DIR}/${LLM_FILE}"
sync_model "${MODEL_CACHE}/${VISION_FILE}" "${REMOTE_MODEL_DIR}/${VISION_FILE}"

"${ADB[@]}" shell chmod 755 "${REMOTE_DIR}/vlm-rknn-server"

# -----------------------------------------------------------------------------
# Start the server
# -----------------------------------------------------------------------------

echo "=== Start server on ${SERIAL} (port ${PORT}) ==="

# Force a pseudo-terminal (-t -t) so the server's stdout is line-buffered and
# streams live; without a tty it stays block-buffered and nothing appears until
# the process exits. Merge stderr into stdout so both are shown.
"${ADB[@]}" shell -t -t "cd ${REMOTE_DIR} && LD_LIBRARY_PATH=${REMOTE_LIB_DIR} ./vlm-rknn-server --ini-file ${REMOTE_DIR}/server.ini --port ${PORT} 2>&1"
