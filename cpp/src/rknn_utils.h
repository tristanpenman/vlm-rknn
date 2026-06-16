#pragma once

#include <rknn_api.h>

const char* rknn_error_message(int ret);
void log_rknn_version(rknn_context ctx);
void log_rkllm_version();
