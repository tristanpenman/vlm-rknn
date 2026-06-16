#pragma once

#include <string>

#include <rknn_api.h>

const char* rknn_error_message(int ret);
std::string tensor_attr_to_string(const rknn_tensor_attr& attr);
void log_rknn_version(rknn_context ctx);
void log_rkllm_version();
