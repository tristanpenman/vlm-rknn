#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <sstream>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "logger.h"
#include "vlm_rknn.h"
#include "rknn_utils.h"

namespace vlm_rknn {

namespace {

struct ModelProfile {
    bool uses_vision_encoder;
    bool supports_multimodal;
    int32_t base_domain_id;
    bool use_chat_template;
    const char* image_placeholder;
    const char* img_start;
    const char* img_end;
    const char* img_content;
    ImagePreprocessProfile image_preprocess;
};

const ModelProfile& model_profile_for(ModelFamily family)
{
    static constexpr ImagePreprocessProfile qwen_vl_image_preprocess {
        ResizeMode::PadToSquare,
        true,
        false,
        127.5f,
        127.5f,
        127.5f,
        {0.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 1.0f},
    };

    static constexpr ImagePreprocessProfile unused_image_preprocess {
        ResizeMode::PadToSquare,
        true,
        false,
        0.0f,
        0.0f,
        0.0f,
        {0.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 1.0f},
    };

    static constexpr ModelProfile qwen2_vl {
        true,
        true,
        0,
        false,
        "<image>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        qwen_vl_image_preprocess,
    };

    // TODO: Change if necessary when the actual RKLLM models are available
    static constexpr ModelProfile qwen2_5_vl {
        true,
        true,
        0,
        false,
        "<image>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        qwen_vl_image_preprocess,
    };

    // TODO: Change if necessary
    static constexpr ModelProfile qwen3_vl {
        true,
        true,
        0,
        false,
        "<image>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        qwen_vl_image_preprocess,
    };

    static constexpr ModelProfile llama {
        false,
        false,
        0,
        false,
        "",
        "",
        "",
        "",
        unused_image_preprocess,
    };

    static constexpr ImagePreprocessProfile smolvlm2_image_preprocess {
        ResizeMode::PadToSquare,
        true,
        false,
        127.5f,
        127.5f,
        127.5f,
        {0.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 1.0f},
    };

    // The SmolVLM2-256M-NPU reference currently uses the same RKLLM multimodal
    // marker strings as the Qwen-VL path.
    static constexpr ModelProfile smolvlm2 {
        true,
        true,
        1,
        true,
        "<image>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        smolvlm2_image_preprocess,
    };

    switch (family) {
    case ModelFamily::QwenVL2:
        return qwen2_vl;
    case ModelFamily::QwenVL2_5:
        return qwen2_5_vl;
    case ModelFamily::QwenVL3:
        return qwen3_vl;
    case ModelFamily::Llama:
        return llama;
    case ModelFamily::SmolVLM2:
        return smolvlm2;
    }

    return qwen2_vl;
}

bool infer_embedding_shape_from_attr(
    const rknn_tensor_attr& attr,
    int& image_tokens,
    int& embed_size)
{
    for (uint32_t i = 0; i + 1 < attr.n_dims && i + 1 < RKNN_MAX_DIMS; ++i) {
        if (attr.dims[i] > 1 && attr.dims[i + 1] > 1) {
            image_tokens = attr.dims[i];
            embed_size = attr.dims[i + 1];
            return true;
        }
    }
    return false;
}

cv::Mat pad_to_square(const cv::Mat& image, const cv::Scalar& background_color)
{
    const int width = image.cols;
    const int height = image.rows;
    if (width == height) {
        return image.clone();
    }

    const int size = std::max(width, height);
    cv::Mat square(size, size, image.type(), background_color);
    const int x_offset = (size - width) / 2;
    const int y_offset = (size - height) / 2;
    image.copyTo(square(cv::Rect(x_offset, y_offset, width, height)));
    return square;
}

cv::Mat center_crop_to_aspect(const cv::Mat& image, double target_aspect)
{
    const double source_aspect = static_cast<double>(image.cols) / static_cast<double>(image.rows);
    if (source_aspect > target_aspect) {
        const int crop_width = static_cast<int>(image.rows * target_aspect);
        const int x_offset = (image.cols - crop_width) / 2;
        return image(cv::Rect(x_offset, 0, crop_width, image.rows)).clone();
    }

    const int crop_height = static_cast<int>(image.cols / target_aspect);
    const int y_offset = (image.rows - crop_height) / 2;
    return image(cv::Rect(0, y_offset, image.cols, crop_height)).clone();
}

}  // namespace

bool parse_model_family(std::string_view value, ModelFamily& family)
{
    if (value == "qwen2-vl" || value == "qwen2_vl" || value == "qwen2") {
        family = ModelFamily::QwenVL2;
        return true;
    }

    if (value == "qwen2.5-vl" || value == "qwen2_5_vl" ||
        value == "qwen25-vl" || value == "qwen2.5") {
        family = ModelFamily::QwenVL2_5;
        return true;
    }

    if (value == "qwen3-vl" || value == "qwen3_vl" || value == "qwen3") {
        family = ModelFamily::QwenVL3;
        return true;
    }

    if (value == "llama" || value == "llama3" || value == "llama3.2") {
        family = ModelFamily::Llama;
        return true;
    }

    if (value == "smolvlm2" || value == "smol-vlm2" || value == "smol_vlm2") {
        family = ModelFamily::SmolVLM2;
        return true;
    }

    return false;
}

const char* model_family_name(ModelFamily family)
{
    switch (family) {
    case ModelFamily::QwenVL2:
        return "qwen2-vl";
    case ModelFamily::QwenVL2_5:
        return "qwen2.5-vl";
    case ModelFamily::QwenVL3:
        return "qwen3-vl";
    case ModelFamily::Llama:
        return "llama";
    case ModelFamily::SmolVLM2:
        return "smolvlm2";
    }

    return "qwen2-vl";
}

const char* model_family_image_placeholder(ModelFamily family)
{
    return model_profile_for(family).image_placeholder;
}

const ImagePreprocessProfile& model_family_image_preprocess_profile(ModelFamily family)
{
    return model_profile_for(family).image_preprocess;
}

bool model_family_uses_vision_encoder(ModelFamily family)
{
    return model_profile_for(family).uses_vision_encoder;
}

bool model_family_supports_multimodal(ModelFamily family)
{
    return model_profile_for(family).supports_multimodal;
}

int preprocess_image_for_vision_encoder(
    ModelFamily family,
    const cv::Mat& image_bgr,
    cv::Size target_size,
    cv::Mat& output)
{
    if (image_bgr.empty()) {
        LOG(ERROR) << "Image input is empty";
        return -1;
    }
    if (target_size.width <= 0 || target_size.height <= 0) {
        LOG(ERROR) << "Invalid target image size: " << target_size.width << "x" << target_size.height;
        return -1;
    }

    const auto& profile = model_family_image_preprocess_profile(family);
    if (profile.normalize_in_host) {
        LOG(ERROR) << "Host-side image normalization is not supported by the current uint8 RKNN input path";
        return -1;
    }

    cv::Mat color;
    if (profile.rgb) {
        if (image_bgr.channels() == 3) {
            cv::cvtColor(image_bgr, color, cv::COLOR_BGR2RGB);
        } else if (image_bgr.channels() == 4) {
            cv::cvtColor(image_bgr, color, cv::COLOR_BGRA2RGB);
        } else {
            LOG(ERROR) << "Unsupported image channel count for RGB conversion: " << image_bgr.channels();
            return -1;
        }
    } else {
        if (image_bgr.channels() != 3) {
            LOG(ERROR) << "Unsupported image channel count: " << image_bgr.channels();
            return -1;
        }
        color = image_bgr.clone();
    }

    cv::Mat resized;
    switch (profile.resize_mode) {
    case ResizeMode::PadToSquare: {
        cv::Mat square = pad_to_square(color, cv::Scalar(profile.pad_r, profile.pad_g, profile.pad_b));
        cv::resize(square, resized, target_size, 0, 0, cv::INTER_LINEAR);
        break;
    }
    case ResizeMode::Stretch:
        cv::resize(color, resized, target_size, 0, 0, cv::INTER_LINEAR);
        break;
    case ResizeMode::CenterCrop: {
        const double target_aspect = static_cast<double>(target_size.width) / static_cast<double>(target_size.height);
        cv::Mat cropped = center_crop_to_aspect(color, target_aspect);
        cv::resize(cropped, resized, target_size, 0, 0, cv::INTER_LINEAR);
        break;
    }
    }

    output = resized.isContinuous() ? resized : resized.clone();
    return 0;
}

int Session::init_vision_encoder()
{
    memset(&encoder_, 0, sizeof(encoder_));

    // Initialize vision encoder
    rknn_context ctx = 0;
    if (!config_.vision_encoder_path.has_value() || config_.vision_encoder_path->empty()) {
        LOG(ERROR) << "Vision encoder path is required for " << model_family_name(config_.model_family);
        return -1;
    }
    int ret = rknn_init(&ctx, (void*)config_.vision_encoder_path->c_str(), 0, 0, NULL);
    if (ret < 0) {
        LOG(ERROR) << "Failed to initialize RKNN, error=" << rknn_error_message(ret);
        return -1;
    }

    log_rknn_version(ctx);

    // Set RKNN core mask if specified in the configuration
    if (config_.num_cores.has_value()) {
        int core_num = config_.num_cores.value();
        if (core_num == 2) {
            ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1);
        } else if (core_num == 3) {
            ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2);
        } else {
            ret = rknn_set_core_mask(ctx, RKNN_NPU_CORE_AUTO);
        }
        if (ret != 0) {
            LOG(ERROR) << "Failed to set RKNN core mask, error=" << rknn_error_message(ret);
            return -1;
        }
    }

    // Query model I/O information
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &encoder_.io_num, sizeof(encoder_.io_num));
    if (ret != RKNN_SUCC) {
        LOG(ERROR) << "Failed to query RKNN input/output count, error=" << rknn_error_message(ret);
        return -1;
    }

    if (encoder_.io_num.n_input <= 0 || encoder_.io_num.n_output <= 0) {
        LOG(ERROR) << "Invalid RKNN I/O count: n_input=" << encoder_.io_num.n_input
                   << " n_output=" << encoder_.io_num.n_output;
        return -1;
    }
    if (Logger::verbose()) {
        LOG(VERBOSE) << "RKNN model I/O count: inputs=" << encoder_.io_num.n_input
                     << " outputs=" << encoder_.io_num.n_output;
    }

    encoder_.input_attrs = static_cast<rknn_tensor_attr*>(calloc(
        encoder_.io_num.n_input,
        sizeof(rknn_tensor_attr)));
    if (encoder_.input_attrs == nullptr) {
        LOG(ERROR) << "Failed to allocate input tensor attribute buffers";
        return -1;
    }

    encoder_.output_attrs = static_cast<rknn_tensor_attr*>(calloc(
        encoder_.io_num.n_output,
        sizeof(rknn_tensor_attr)));
    if (encoder_.output_attrs == nullptr) {
        LOG(ERROR) << "Failed to allocate output tensor attribute buffers";
        return -1;
    }

    for (auto i = 0; i < encoder_.io_num.n_input; ++i) {
        encoder_.input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &encoder_.input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            LOG(ERROR) << "Failed to query RKNN input attr for index " << i
                       << ", error=" << rknn_error_message(ret);
            return -1;
        }
        if (Logger::verbose()) {
            LOG(VERBOSE) << "RKNN input tensor: " << tensor_attr_to_string(encoder_.input_attrs[i]);
        }
    }

    for (auto i = 0; i < encoder_.io_num.n_output; ++i) {
        encoder_.output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &encoder_.output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            LOG(ERROR) << "Failed to query RKNN output attr for index " << i
                       << ", error=" << rknn_error_message(ret);
            return -1;
        }
        if (Logger::verbose()) {
            LOG(VERBOSE) << "RKNN output tensor: " << tensor_attr_to_string(encoder_.output_attrs[i]);
        }
    }

    if (!infer_embedding_shape_from_attr(
            encoder_.output_attrs[0],
            encoder_.model_image_token,
            encoder_.model_embed_size)) {
        LOG(ERROR) << "Could not infer image token and embedding dimensions from first RKNN output tensor";
        return -1;
    }

    if (encoder_.input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        encoder_.model_channel = encoder_.input_attrs[0].dims[1];
        encoder_.model_height = encoder_.input_attrs[0].dims[2];
        encoder_.model_width = encoder_.input_attrs[0].dims[3];
    } else {
        encoder_.model_height = encoder_.input_attrs[0].dims[1];
        encoder_.model_width = encoder_.input_attrs[0].dims[2];
        encoder_.model_channel = encoder_.input_attrs[0].dims[3];
    }
    if (Logger::verbose()) {
        LOG(VERBOSE) << "RKNN vision input shape: height=" << encoder_.model_height
                     << " width=" << encoder_.model_width
                     << " channel=" << encoder_.model_channel;
        LOG(VERBOSE) << "RKNN image embedding shape: tokens=" << encoder_.model_image_token
                     << " embed_size=" << encoder_.model_embed_size
                     << " output_tensors=" << encoder_.io_num.n_output;
    }

    // Save context for later use
    encoder_.rknn_ctx = ctx;

    return 0;
}

int Session::init_text_decoder()
{
    // Prepare RKLLM parameters based on the configuration
    const auto& profile = model_profile_for(config_.model_family);
    RKLLMParam params = rkllm_createDefaultParam();
    params.model_path = config_.language_model_path.c_str();
    params.top_k = 1;
    params.max_new_tokens = config_.max_new_tokens;
    params.max_context_len = config_.max_context_len;
    params.skip_special_token = true;
    params.extend_param.base_domain_id = profile.base_domain_id;
    params.img_start = profile.img_start;
    params.img_end = profile.img_end;
    params.img_content = profile.img_content;

    // Initialize RKLLM session
    int ret = rkllm_init(&decoder_.handle, &params, &callback);
    if (ret != 0) {
        LOG(ERROR) << "Failed to initialize RKLLM, error=" << ret;
        return -1;
    }

    if (profile.use_chat_template) {
        ret = rkllm_set_chat_template(
            decoder_.handle,
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
            "<|im_start|>user\n",
            "<|im_end|>\n<|im_start|>assistant\n");
        if (ret != 0) {
            LOG(ERROR) << "Failed to set RKLLM chat template, error=" << ret;
            return -1;
        }
    }

    log_rkllm_version();

    return 0;
}

void Session::cleanup_vision_encoder()
{
    if (encoder_.input_attrs) {
        free(encoder_.input_attrs);
        encoder_.input_attrs = nullptr;
    }

    if (encoder_.output_attrs) {
        free(encoder_.output_attrs);
        encoder_.output_attrs = nullptr;
    }

    if (encoder_.rknn_ctx) {
        rknn_destroy(encoder_.rknn_ctx);
        encoder_.rknn_ctx = 0;
    }
}

void Session::cleanup_text_decoder()
{
    if (decoder_.handle) {
        rkllm_destroy(decoder_.handle);
        decoder_.handle = nullptr;
    }
}

Session::Session(ModelConfig config)
  : config_(std::move(config)) {}

Session::~Session()
{
    cleanup_vision_encoder();
    cleanup_text_decoder();
}

int Session::init()
{
    if (init_text_decoder() != 0) {
        return -1;
    }

    if (model_family_uses_vision_encoder(config_.model_family)) {
        if (init_vision_encoder() != 0) {
            return -1;
        }
    }

    return 0;
}

const ModelConfig& Session::config() const noexcept
{
    return config_;
}

bool Session::is_ready() const noexcept
{
    if (decoder_.handle == nullptr) {
        return false;
    }

    if (model_family_uses_vision_encoder(config_.model_family)) {
        return encoder_.rknn_ctx != 0;
    }

    return true;
}

std::string Session::describe() const
{
    const auto& profile = model_profile_for(config_.model_family);

    std::ostringstream stream;
    stream << "VLM RKNN session";
    stream << " target=" << VLM_RKNN_TARGET;
    stream << " model_family=" << model_family_name(config_.model_family);
    stream << " requires_vision_encoder=" << (profile.uses_vision_encoder ? "yes" : "no");
    stream << " vision_encoder_path=" << (config_.vision_encoder_path.has_value() && !config_.vision_encoder_path->empty() ? *config_.vision_encoder_path : "<unset>");
    stream << " language_model_path=" << (config_.language_model_path.empty() ? "<unset>" : config_.language_model_path);
    return stream.str();
}

bool Session::prompt_contains_image(const std::string& prompt) const
{
    const auto& profile = model_profile_for(config_.model_family);
    return profile.supports_multimodal &&
        profile.image_placeholder[0] != '\0' &&
        prompt.find(profile.image_placeholder) != std::string::npos;
}

void Session::set_output_callback(OutputCallback callback)
{
    output_callback_ = std::move(callback);
}

int Session::encode(void* img_data, float* out_result)
{
    if (encoder_.rknn_ctx == 0) {
        LOG(ERROR) << "Vision encoder has not been initialized";
        return -1;
    }

    if (img_data == nullptr) {
        LOG(ERROR) << "Image input buffer is null";
        return -1;
    }

    if (out_result == nullptr) {
        LOG(ERROR) << "Image embedding output buffer is null";
        return -1;
    }

    if (encoder_.io_num.n_input != 1) {
        LOG(ERROR) << "Unsupported RKNN input count: " << encoder_.io_num.n_input;
        return -1;
    }

    const auto& input_attr = encoder_.input_attrs[0];
    rknn_input input;
    memset(&input, 0, sizeof(input));
    input.index = 0;
    input.buf = img_data;
    input.size = input_attr.n_elems;
    input.pass_through = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;

    int ret = rknn_inputs_set(encoder_.rknn_ctx, 1, &input);
    if (ret != RKNN_SUCC) {
        LOG(ERROR) << "Failed to set RKNN input, error=" << rknn_error_message(ret);
        return ret;
    }

    ret = rknn_run(encoder_.rknn_ctx, nullptr);
    if (ret != RKNN_SUCC) {
        LOG(ERROR) << "Failed to run RKNN encoder, error=" << rknn_error_message(ret);
        return ret;
    }

    std::vector<rknn_output> outputs(encoder_.io_num.n_output);
    for (uint32_t i = 0; i < encoder_.io_num.n_output; ++i) {
        memset(&outputs[i], 0, sizeof(rknn_output));
        outputs[i].index = i;
        outputs[i].want_float = 1;
        outputs[i].is_prealloc = 0;
    }

    ret = rknn_outputs_get(encoder_.rknn_ctx, encoder_.io_num.n_output, outputs.data(), nullptr);
    if (ret != RKNN_SUCC) {
        LOG(ERROR) << "Failed to get RKNN output, error=" << rknn_error_message(ret);
        return ret;
    }

    if (encoder_.io_num.n_output == 1) {
        if (outputs[0].buf == nullptr) {
            LOG(ERROR) << "RKNN output buffer is null for index 0";
            ret = -1;
        } else {
            memcpy(out_result, outputs[0].buf, outputs[0].size);
        }
    } else {
        for (uint32_t i = 0; i < encoder_.io_num.n_output; ++i) {
            int output_tokens = 0;
            int output_embed_size = 0;
            if (!infer_embedding_shape_from_attr(encoder_.output_attrs[i], output_tokens, output_embed_size)) {
                LOG(ERROR) << "Could not infer embedding shape for RKNN output index " << i;
                ret = -1;
                break;
            }
            if (output_tokens != encoder_.model_image_token ||
                output_embed_size != encoder_.model_embed_size) {
                LOG(ERROR) << "Unsupported RKNN output shape at index " << i
                           << ": expected tokens=" << encoder_.model_image_token
                           << " embed_size=" << encoder_.model_embed_size
                           << " but got tokens=" << output_tokens
                           << " embed_size=" << output_embed_size;
                ret = -1;
                break;
            }
            if (outputs[i].buf == nullptr) {
                LOG(ERROR) << "RKNN output buffer is null for index " << i;
                ret = -1;
                break;
            }
        }

        if (ret == RKNN_SUCC) {
            const size_t output_count = encoder_.io_num.n_output;
            const size_t image_tokens = encoder_.model_image_token;
            const size_t embed_size = encoder_.model_embed_size;
            for (size_t token = 0; token < image_tokens; ++token) {
                for (size_t output_index = 0; output_index < output_count; ++output_index) {
                    const auto* output = static_cast<const float*>(outputs[output_index].buf);
                    memcpy(
                        out_result + token * output_count * embed_size + output_index * embed_size,
                        output + token * embed_size,
                        embed_size * sizeof(float));
                }
            }
        }
    }

    int release_ret = rknn_outputs_release(encoder_.rknn_ctx, encoder_.io_num.n_output, outputs.data());
    if (release_ret != RKNN_SUCC) {
        LOG(ERROR) << "Failed to release RKNN output buffers, error=" << rknn_error_message(release_ret);
        return release_ret;
    }

    if (ret != RKNN_SUCC) {
        return ret;
    }

    return 0;
}

int Session::decode(const std::string& prompt, float* img_vec)
{
    const auto& profile = model_profile_for(config_.model_family);

    last_decoded_text_.clear();

    RKLLMInferParam rkllm_infer_params;
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));
    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
    rkllm_infer_params.keep_history = profile.use_chat_template ? 1 : 0;

    RKLLMInput rkllm_input;
    memset(&rkllm_input, 0, sizeof(RKLLMInput));
    if (!prompt_contains_image(prompt)) {
        rkllm_input.input_type = RKLLM_INPUT_PROMPT;
        rkllm_input.role = "user";
        rkllm_input.prompt_input = (char*)prompt.c_str();
    } else {
        if (img_vec == nullptr) {
            LOG(ERROR) << "Prompt references " << profile.image_placeholder << " but no image embedding was provided";
            return -1;
        }
        const size_t n_image_tokens = encoder_.model_image_token;
        const size_t image_height = encoder_.model_height;
        const size_t image_width = encoder_.model_width;
        if (n_image_tokens == 0 || image_height == 0 || image_width == 0) {
            LOG(ERROR) << "Vision encoder metadata is not available for multimodal decoding";
            return -1;
        }
        rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL;
        rkllm_input.role = "user";
        rkllm_input.multimodal_input.prompt = (char*)prompt.c_str();
        rkllm_input.multimodal_input.image_embed = img_vec;
        rkllm_input.multimodal_input.n_image_tokens = n_image_tokens;
        rkllm_input.multimodal_input.n_image = 1;
        rkllm_input.multimodal_input.image_height = image_height;
        rkllm_input.multimodal_input.image_width = image_width;
    }

    int ret = rkllm_run(decoder_.handle, &rkllm_input, &rkllm_infer_params, this);
    if (ret != 0) {
        LOG(ERROR) << "Failed to run RKLLM, error=" << ret;
        return ret;
    }

    return 0;
}

int Session::callback(RKLLMResult *result, void *userdata, LLMCallState state)
{
    Session *session = static_cast<Session*>(userdata);

    if (state == RKLLM_RUN_FINISH) {
        LOG(INFO) << "RKLLM run finished";
        if (session->output_callback_) {
            session->output_callback_(nullptr, state);
        }
    } else if (state == RKLLM_RUN_ERROR) {
        LOG(ERROR) << "RKLLM run error";
        if (session->output_callback_) {
            session->output_callback_(nullptr, state);
        }
    } else if (state == RKLLM_RUN_NORMAL) {
        session->last_decoded_text_ += result->text;
        LOG(INFO) << session->last_decoded_text_;
        if (session->output_callback_) {
            session->output_callback_(result->text, state);
        }
    }

    return 0;
}

}  // namespace vlm_rknn
