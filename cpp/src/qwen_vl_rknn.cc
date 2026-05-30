#include <cstring>
#include <string>
#include <sstream>
#include <utility>
#include <vector>

#include "file_utils.h"
#include "logger.h"
#include "qwen_vl_rknn.h"

namespace qwen_vl_rknn {

int Session::init_vision_encoder()
{
    memset(&encoder_, 0, sizeof(encoder_));

    // Load RKNN Model
    char* model;
    int model_len = read_data_from_file(config_.vision_encoder_path.c_str(), &model);
    if (model == NULL) {
        LOG(ERROR) << "Failed to read data from file!";
        return -1;
    }

    // Initialize vision encoder
    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0) {
        LOG(ERROR) << "Failed to initialize RKNN, error=" << ret;
        return -1;
    }

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
    }

    if (ret != 0) {
        LOG(ERROR) << "Failed to set RKNN core mask, error=" << ret;
        return -1;
    }

    // Save context for later use
    encoder_.rknn_ctx = ctx;

    // Query model I/O information.
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &encoder_.io_num, sizeof(encoder_.io_num));
    if (ret != RKNN_SUCC) {
        LOG(ERROR) << "Failed to query RKNN input/output count, error=" << ret;
        return -1;
    }

    if (encoder_.io_num.n_input <= 0 || encoder_.io_num.n_output <= 0) {
        LOG(ERROR) << "Invalid RKNN I/O count: n_input=" << encoder_.io_num.n_input
                   << " n_output=" << encoder_.io_num.n_output;
        return -1;
    }

    encoder_.input_attrs = static_cast<rknn_tensor_attr*>(calloc(
        encoder_.io_num.n_input,
        sizeof(rknn_tensor_attr)));
    encoder_.output_attrs = static_cast<rknn_tensor_attr*>(calloc(
        encoder_.io_num.n_output,
        sizeof(rknn_tensor_attr)));

    if (encoder_.input_attrs == nullptr || encoder_.output_attrs == nullptr) {
        LOG(ERROR) << "Failed to allocate tensor attribute buffers";
        return -1;
    }

    for (uint32_t i = 0; i < encoder_.io_num.n_input; ++i) {
        encoder_.input_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &encoder_.input_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            LOG(ERROR) << "Failed to query RKNN input attr for index " << i << ", error=" << ret;
            return -1;
        }
    }

    for (uint32_t i = 0; i < encoder_.io_num.n_output; ++i) {
        encoder_.output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &encoder_.output_attrs[i], sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            LOG(ERROR) << "Failed to query RKNN output attr for index " << i << ", error=" << ret;
            return -1;
        }
    }

    // Populate VisionEncoder metadata from input/output tensor shapes.
    const auto& input0 = encoder_.input_attrs[0];
    if (input0.n_dims < 4) {
        LOG(ERROR) << "Unexpected input rank: " << input0.n_dims << " (expected >= 4)";
        return -1;
    }
    encoder_.model_channel = input0.dims[1];
    encoder_.model_height = input0.dims[2];
    encoder_.model_width = input0.dims[3];

    const auto& output0 = encoder_.output_attrs[0];
    if (output0.n_dims < 3) {
        LOG(ERROR) << "Unexpected output rank: " << output0.n_dims << " (expected >= 3)";
        return -1;
    }
    encoder_.model_image_token = output0.dims[1];
    encoder_.model_embed_size = output0.dims[2];

    return 0;
}

int Session::init_text_decoder()
{
    // Prepare RKLLM parameters based on the configuration
    RKLLMParam params = rkllm_createDefaultParam();
    params.model_path = config_.language_model_path.c_str();
    params.top_k = 1;
    params.max_new_tokens = config_.max_new_tokens;
    params.max_context_len = config_.max_context_len;
    params.skip_special_token = true;
    params.img_start = "<|vision_start|>";
    params.img_end = "<|vision_end|>";
    params.img_content = "<|image_pad|>";

    // Initialize RKLLM session
    int ret = rkllm_init(&decoder_.handle, &params, &callback);
    if (ret != 0) {
        LOG(ERROR) << "Failed to initialize RKLLM, error=" << ret;
        return -1;
    }

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

    if (init_vision_encoder() != 0) {
        return -1;
    }

    return 0;
}

const ModelConfig& Session::config() const noexcept
{
    return config_;
}

bool Session::is_ready() const noexcept
{
    return decoder_.handle != nullptr && encoder_.rknn_ctx != 0;
}

std::string Session::describe() const
{
    std::ostringstream stream;
    stream << "Qwen-VL RKNN session";
    stream << " target=" << QWEN_VL_RKNN_TARGET;
    stream << " vision_encoder_path=" << (config_.vision_encoder_path.empty() ? "<unset>" : config_.vision_encoder_path);
    stream << " language_model_path=" << (config_.language_model_path.empty() ? "<unset>" : config_.language_model_path);
    return stream.str();
}

int Session::encode(void* img_data, float* out_result)
{
    // TODO

    return -1;
}

int Session::decode(const std::string& prompt, char* output_buffer, size_t buffer_size, float* img_vec)
{
    const size_t n_image_tokens = encoder_.model_image_token;
    const size_t image_height = encoder_.model_height;
    const size_t image_width = encoder_.model_width;

    RKLLMInferParam rkllm_infer_params;
    memset(&rkllm_infer_params, 0, sizeof(RKLLMInferParam));
    rkllm_infer_params.mode = RKLLM_INFER_GENERATE;
    rkllm_infer_params.keep_history = 0;

    RKLLMInput rkllm_input;
    memset(&rkllm_input, 0, sizeof(RKLLMInput));
    if (prompt.find("<image>") == std::string::npos) {
        rkllm_input.input_type = RKLLM_INPUT_PROMPT;
        rkllm_input.role = "user";
        rkllm_input.prompt_input = (char*)prompt.c_str();
    } else {
        rkllm_input.input_type = RKLLM_INPUT_MULTIMODAL;
        rkllm_input.role = "user";
        rkllm_input.multimodal_input.prompt = (char*)prompt.c_str();
        rkllm_input.multimodal_input.image_embed = img_vec;
        rkllm_input.multimodal_input.n_image_tokens = n_image_tokens;
        rkllm_input.multimodal_input.n_image = 1;
        rkllm_input.multimodal_input.image_height = image_height;
        rkllm_input.multimodal_input.image_width = image_width;
    }

    if (rkllm_run(decoder_.handle, &rkllm_input, &rkllm_infer_params, NULL) != 0) {
        LOG(ERROR) << "Failed to run RKLLM";
        return -1;
    }

    return 0;
}

int Session::callback(RKLLMResult *result, void *userdata, LLMCallState state)
{
    if (state == RKLLM_RUN_FINISH) {
        LOG(INFO) << "RKLLM run finished";
    } else if (state == RKLLM_RUN_ERROR) {
        LOG(ERROR) << "RKLLM run error";
    } else if (state == RKLLM_RUN_NORMAL) {
        LOG(INFO) << result->text;
    }

    return 0;
}

}  // namespace qwen_vl_rknn
