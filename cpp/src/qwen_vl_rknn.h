#pragma once

#include <functional>
#include <optional>
#include <string>
#include <string_view>

#include <rknn_api.h>
#include <rkllm.h>

namespace qwen_vl_rknn {

struct TextDecoder {
    LLMHandle handle = nullptr;
};

struct VisionEncoder {
    rknn_context rknn_ctx = 0;
    rknn_input_output_num io_num {};
    rknn_tensor_attr* input_attrs = nullptr;
    rknn_tensor_attr* output_attrs = nullptr;

    int model_channel = 0;
    int model_width = 0;
    int model_height = 0;

    int model_image_token = 0;
    int model_embed_size = 0;
};

enum class ModelFamily {
    QwenVL2,
    QwenVL2_5,
    QwenVL3,
    Llama,
    SmolVLM2,
};

struct ModelConfig {
    ModelFamily model_family = ModelFamily::QwenVL2;

    std::optional<std::string> vision_encoder_path;
    std::string language_model_path;

    int max_new_tokens = 128;
    int max_context_len = 2048;

    std::optional<int> num_cores;
};

bool parse_model_family(std::string_view value, ModelFamily& family);
const char* model_family_name(ModelFamily family);
bool model_family_uses_vision_encoder(ModelFamily family);
bool model_family_supports_multimodal(ModelFamily family);

class Session {
public:
    using OutputCallback = std::function<void(const char* text, LLMCallState state)>;

private:
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
    void set_output_callback(OutputCallback callback);

    const VisionEncoder& vision_encoder() const
    {
        return encoder_;
    }

    const TextDecoder& text_decoder() const
    {
        return decoder_;
    }

    // run the vision encoder and fill the output buffer with the resulting embedding
    int encode(void* img_data, float* out_result);

    // run the text decoder with the given prompt; results are delivered via the RKLLM callback
    int decode(const std::string& prompt, float* img_vec);

private:
    static int callback(RKLLMResult *result, void *userdata, LLMCallState state);

    ModelConfig config_;
    VisionEncoder encoder_;
    TextDecoder decoder_;

    std::string last_decoded_text_;
    OutputCallback output_callback_;
};

}  // namespace qwen_vl_rknn
