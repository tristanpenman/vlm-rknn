#include "rknn_utils.h"

#include <rknn_api.h>

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
