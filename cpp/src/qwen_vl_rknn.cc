#include <sstream>
#include <utility>

#include "file_utils.h"
#include "qwen_vl_rknn.h"

namespace qwen_vl_rknn {

Session::Session(ModelConfig config)
  : config_(std::move(config)) {}

int Session::callback(RKLLMResult *result, void *userdata, LLMCallState state)
{
    if (state == RKLLM_RUN_FINISH) {
        printf("\n");
    } else if (state == RKLLM_RUN_ERROR) {
        printf("\\run error\n");
    } else if (state == RKLLM_RUN_NORMAL) {
        printf("%s", result->text);
    }

    return 0;
}

int Session::init()
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
    int ret = rkllm_init(&handle_, &params, &callback);
    if (ret != 0) {
        printf("rkllm_init fail! ret=%d\n", ret);
        return -1;
    }

    // Load RKNN Model
    char* model;
    int model_len = read_data_from_file(config_.vision_encoder_path.c_str(), &model);
    if (model == NULL) {
        printf("load_model fail!\n");
        return -1;
    }

    // Initialize vision encoder
    rknn_context ctx = 0;
    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Save context for later use
    encoder_.rknn_ctx = ctx;

    return 0;
}

const ModelConfig& Session::config() const noexcept
{
    return config_;
}

bool Session::is_ready() const noexcept
{
    return !config_.vision_encoder_path.empty() && !config_.language_model_path.empty();
}

std::string Session::describe() const
{
    std::ostringstream stream;
    stream << "Qwen-VL RKNN session";
    stream << " target=" << target_device();
    stream << " vision_encoder_path=" << (config_.vision_encoder_path.empty() ? "<unset>" : config_.vision_encoder_path);
    stream << " language_model_path=" << (config_.language_model_path.empty() ? "<unset>" : config_.language_model_path);
    return stream.str();
}

std::string target_device()
{
    return QWEN_VL_RKNN_TARGET;
}

}  // namespace qwen_vl_rknn
