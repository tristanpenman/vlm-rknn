#include "rknn_utils.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <rknn_api.h>

#include "logger.h"

namespace {

std::optional<std::string> find_loaded_library_path(std::string_view library_name)
{
    std::ifstream maps("/proc/self/maps");
    std::string line;
    while (std::getline(maps, line)) {
        if (line.find(library_name) == std::string::npos) {
            continue;
        }

        const auto path_start = line.find('/');
        if (path_start == std::string::npos) {
            continue;
        }

        return line.substr(path_start);
    }

    return std::nullopt;
}

std::optional<std::string> read_embedded_version_string(
    const std::string& library_path,
    std::string_view marker)
{
    std::ifstream library(library_path, std::ios::binary);
    if (!library) {
        return std::nullopt;
    }

    std::string carry;
    std::vector<char> buffer(64 * 1024);
    while (library) {
        library.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto bytes_read = library.gcount();
        if (bytes_read <= 0) {
            break;
        }

        std::string chunk = carry;
        chunk.append(buffer.data(), static_cast<size_t>(bytes_read));
        const auto marker_pos = chunk.find(marker);
        if (marker_pos != std::string::npos) {
            const auto value_start = marker_pos + marker.size();
            auto value_end = value_start;
            const auto value_limit = std::min(chunk.size(), value_start + 160);
            while (value_end < value_limit && chunk[value_end] != '\0' && chunk[value_end] != '\n') {
                ++value_end;
            }
            return std::string(marker) + chunk.substr(value_start, value_end - value_start);
        }

        const auto carry_size = std::min(chunk.size(), marker.size() + 160);
        carry = chunk.substr(chunk.size() - carry_size);
    }

    return std::nullopt;
}

}  // namespace

const char* rknn_error_message(int ret)
{
    switch (ret) {
    case RKNN_SUCC:
        return "RKNN_SUCC (0): execute succeeded";
    case RKNN_ERR_FAIL:
        return "RKNN_ERR_FAIL (-1): execute failed";
    case RKNN_ERR_TIMEOUT:
        return "RKNN_ERR_TIMEOUT (-2): execute timed out";
    case RKNN_ERR_DEVICE_UNAVAILABLE:
        return "RKNN_ERR_DEVICE_UNAVAILABLE (-3): device is unavailable";
    case RKNN_ERR_MALLOC_FAIL:
        return "RKNN_ERR_MALLOC_FAIL (-4): memory allocation failed";
    case RKNN_ERR_PARAM_INVALID:
        return "RKNN_ERR_PARAM_INVALID (-5): parameter is invalid";
    case RKNN_ERR_MODEL_INVALID:
        return "RKNN_ERR_MODEL_INVALID (-6): model is invalid";
    case RKNN_ERR_CTX_INVALID:
        return "RKNN_ERR_CTX_INVALID (-7): context is invalid";
    case RKNN_ERR_INPUT_INVALID:
        return "RKNN_ERR_INPUT_INVALID (-8): input is invalid";
    case RKNN_ERR_OUTPUT_INVALID:
        return "RKNN_ERR_OUTPUT_INVALID (-9): output is invalid";
    case RKNN_ERR_DEVICE_UNMATCH:
        return "RKNN_ERR_DEVICE_UNMATCH (-10): SDK and NPU driver or firmware do not match";
    case RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL:
        return "RKNN_ERR_INCOMPATILE_PRE_COMPILE_MODEL (-11): pre-compiled model is incompatible with current driver";
    case RKNN_ERR_INCOMPATILE_OPTIMIZATION_LEVEL_VERSION:
        return "RKNN_ERR_INCOMPATIBLE_OPTIMIZATION_LEVEL_VERSION (-12): model optimization level is incompatible with current driver";
    case RKNN_ERR_TARGET_PLATFORM_UNMATCH:
        return "RKNN_ERR_TARGET_PLATFORM_UNMATCH (-13): model target platform does not match current platform";
    default:
        return "Unknown RKNN error code";
    }
}

void log_rknn_version(rknn_context ctx)
{
    rknn_sdk_version rknn_version;
    memset(&rknn_version, 0, sizeof(rknn_version));
    const int ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &rknn_version, sizeof(rknn_version));
    if (ret == RKNN_SUCC) {
        LOG(INFO) << "RKNN version: api=" << rknn_version.api_version
                  << " driver=" << rknn_version.drv_version;
    } else {
        LOG(WARNING) << "Failed to query RKNN version, error=" << rknn_error_message(ret);
    }
}

void log_rkllm_version()
{
    const auto library_path = find_loaded_library_path("librkllmrt.so");
    if (!library_path.has_value()) {
        LOG(WARNING) << "RKLLM version: unavailable (librkllmrt.so is not visible in /proc/self/maps)";
        return;
    }

    const auto version = read_embedded_version_string(
        *library_path,
        "rknn llm lib version: ");
    if (version.has_value()) {
        LOG(INFO) << "RKLLM version: " << *version;
        return;
    }

    const auto sdk_version = read_embedded_version_string(
        *library_path,
        "RKLLM SDK (version: ");
    if (sdk_version.has_value()) {
        LOG(INFO) << "RKLLM version: " << *sdk_version;
        return;
    }

    LOG(WARNING) << "RKLLM version: unavailable (no embedded version marker found in "
                 << *library_path << ")";
}
