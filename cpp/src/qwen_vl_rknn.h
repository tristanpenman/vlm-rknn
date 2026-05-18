#pragma once

#include <string>

#include <rknn_api.h>
#include <rkllm.h>

namespace qwen_vl_rknn {

struct Encoder {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
};

struct ModelConfig {
    // Model paths
    std::string vision_encoder_path;
    std::string language_model_path;

    // Language model parameters
    int max_new_tokens = 128;
    int max_context_len = 2048;
};

class Session {
public:
    explicit Session(ModelConfig config);

    int init();
    const ModelConfig& config() const noexcept;
    bool is_ready() const noexcept;
    std::string describe() const;

private:
    static int callback(RKLLMResult *result, void *userdata, LLMCallState state);

    ModelConfig config_;
    Encoder encoder_;
    LLMHandle handle_;
};

std::string target_device();

}  // namespace qwen_vl_rknn
