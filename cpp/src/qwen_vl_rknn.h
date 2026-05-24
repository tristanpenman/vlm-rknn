#pragma once

#include <optional>
#include <string>

#include <rknn_api.h>
#include <rkllm.h>

namespace qwen_vl_rknn {

struct TextDecoder {
    LLMHandle handle;
};

struct VisionEncoder {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;

    int model_channel;
    int model_width;
    int model_height;
};

struct ModelConfig {
    std::string vision_encoder_path;
    std::string language_model_path;

    int max_new_tokens = 128;
    int max_context_len = 2048;

    std::optional<int> num_cores;
};

class Session {
    int init_vision_encoder();
    int init_text_decoder();

    void cleanup_vision_encoder();
    void cleanup_text_decoder();

public:
    explicit Session(ModelConfig config);

    ~Session();

    int init();
    const ModelConfig& config() const noexcept;
    bool is_ready() const noexcept;
    std::string describe() const;

    const VisionEncoder& vision_encoder() const
    {
        return encoder_;
    }

    const TextDecoder& text_decoder() const
    {
        return decoder_;
    }

    void run(void* img_data, float* out_result);

private:
    static int callback(RKLLMResult *result, void *userdata, LLMCallState state);

    ModelConfig config_;
    VisionEncoder encoder_;
    TextDecoder decoder_;
};

}  // namespace qwen_vl_rknn
