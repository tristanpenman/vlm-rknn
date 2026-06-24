#pragma once

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>
#include <rknn_api.h>
#include <rkllm.h>

namespace vlm_rknn {

struct TextDecoder
{
    LLMHandle handle = nullptr;
};

struct VisionEncoder
{
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

enum class ModelFamily
{
    kQwenVL2,
    kQwenVL2_5,
    kQwenVL3,
    kLlama,
    kSmolVLM2,
};

enum class ResizeMode
{
    kPadToSquare,
    kStretch,
    kCenterCrop,
};

struct ImagePreprocessProfile
{
    ResizeMode resize_mode;
    bool rgb;
    bool normalize_in_host;
    float pad_r;
    float pad_g;
    float pad_b;
    std::array<float, 3> mean;
    std::array<float, 3> std;
};

struct ModelConfig
{
    ModelFamily model_family = ModelFamily::kQwenVL2;

    std::optional<std::string> vision_encoder_path;
    std::string language_model_path;

    int max_new_tokens = 128;
    int max_context_len = 2048;

    std::optional<int> num_cores;
};

// A ModelConfig together with the identifier it was defined under (the INI
// section name). The server uses the identifier to route requests.
struct NamedModelConfig
{
    std::string model_id;
    ModelConfig config;
};

// Parse an INI document into an ordered list of named model configurations.
//
// Each [section] defines one model, keyed by the section name. The recognised
// keys mirror the command-line options, with underscores instead of hyphens:
//   model_family, vision, llm, max_new_tokens, max_context_len, cores
// `llm` is required, and `vision` is required for model families that use a
// vision encoder (and rejected for those that do not). The order of the result
// matches the order of the sections, so the first entry is the default model.
//
// On failure, returns false and populates `error` with a human-readable message.
bool parse_model_configs_from_ini(
    const std::string& ini_text,
    std::vector<NamedModelConfig>& out,
    std::string& error);

bool parse_model_family(std::string_view value, ModelFamily& family);
const char* model_family_name(ModelFamily family);
const char* model_family_image_placeholder(ModelFamily family);
const ImagePreprocessProfile& model_family_image_preprocess_profile(ModelFamily family);
bool model_family_uses_vision_encoder(ModelFamily family);
bool model_family_supports_multimodal(ModelFamily family);
int preprocess_image_for_vision_encoder(
    ModelFamily family,
    const cv::Mat& image_bgr,
    cv::Size target_size,
    cv::Mat& output);

class Session
{
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
    bool prompt_contains_image(const std::string& prompt) const;
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

}  // namespace vlm_rknn
