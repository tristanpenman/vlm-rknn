#include <iostream>
#include <string>

#include "qwen_vl_rknn.h"

namespace {

void expect(bool condition, const std::string& message)
{
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

}  // namespace

int main()
{
    qwen_vl_rknn::Session empty_session({});
    expect(!empty_session.is_ready(), "empty session should not be ready");

    qwen_vl_rknn::ModelConfig config;
    expect(config.model_family == qwen_vl_rknn::ModelFamily::QwenVL2,
           "default model family should be Qwen2-VL");
    config.vision_encoder_path = "/models/qwen-vl/vision_encoder.rknn";
    config.language_model_path = "/models/qwen-vl/language_model.rkllm";
    qwen_vl_rknn::Session session(config);

    expect(!session.is_ready(), "session should not be ready before init");
    expect(session.config().vision_encoder_path.has_value(), "vision_encoder_path should be set");
    expect(*session.config().vision_encoder_path == "/models/qwen-vl/vision_encoder.rknn", "vision_encoder_path should be retained");
    expect(session.config().language_model_path == "/models/qwen-vl/language_model.rkllm", "language_model_path should be retained");
    expect(session.describe().find("Qwen-VL RKNN session") != std::string::npos,
           "description should contain project name");
    expect(session.describe().find("target=") != std::string::npos, "description should contain target device");
    expect(session.describe().find("model_family=qwen2-vl") != std::string::npos,
           "description should contain model family");
    expect(session.describe().find("requires_vision_encoder=yes") != std::string::npos,
           "description should contain vision encoder requirement");

    qwen_vl_rknn::ModelFamily family;
    expect(qwen_vl_rknn::parse_model_family("llama", family), "llama should parse");
    expect(family == qwen_vl_rknn::ModelFamily::Llama, "llama should map to Llama family");
    expect(qwen_vl_rknn::parse_model_family("smol-vlm2", family), "smol-vlm2 alias should parse");
    expect(family == qwen_vl_rknn::ModelFamily::SmolVLM2, "smol-vlm2 should map to SmolVLM2 family");
    expect(!qwen_vl_rknn::parse_model_family("not-a-family", family), "unknown family should not parse");

    expect(!qwen_vl_rknn::model_family_uses_vision_encoder(qwen_vl_rknn::ModelFamily::Llama),
           "llama should not require a vision encoder");
    expect(!qwen_vl_rknn::model_family_supports_multimodal(qwen_vl_rknn::ModelFamily::Llama),
           "llama should not support multimodal input");
    expect(qwen_vl_rknn::model_family_uses_vision_encoder(qwen_vl_rknn::ModelFamily::QwenVL2),
           "Qwen2-VL should require a vision encoder");
    expect(qwen_vl_rknn::model_family_uses_vision_encoder(qwen_vl_rknn::ModelFamily::SmolVLM2),
           "SmolVLM2 should require a vision encoder");
    expect(qwen_vl_rknn::model_family_supports_multimodal(qwen_vl_rknn::ModelFamily::SmolVLM2),
           "SmolVLM2 should be registered as multimodal");
    expect(std::string(qwen_vl_rknn::model_family_image_placeholder(qwen_vl_rknn::ModelFamily::QwenVL2)) == "<image>",
           "Qwen2-VL image placeholder should be exposed");
    expect(std::string(qwen_vl_rknn::model_family_image_placeholder(qwen_vl_rknn::ModelFamily::Llama)).empty(),
           "llama image placeholder should be empty");

    expect(session.prompt_contains_image("<image>Describe this."), "Qwen2-VL should detect its image placeholder");
    expect(!session.prompt_contains_image("Describe this."), "Qwen2-VL should allow plain text prompts");

    qwen_vl_rknn::ModelConfig llama_config;
    llama_config.model_family = qwen_vl_rknn::ModelFamily::Llama;
    llama_config.language_model_path = "/models/llama.rkllm";
    qwen_vl_rknn::Session llama_session(llama_config);
    expect(!llama_session.is_ready(), "llama session should not be ready before init");
    expect(llama_session.describe().find("model_family=llama") != std::string::npos,
           "description should contain llama model family");
    expect(llama_session.describe().find("requires_vision_encoder=no") != std::string::npos,
           "description should report llama does not require a vision encoder");
    expect(!llama_session.prompt_contains_image("<image>Describe this."),
           "llama should not treat image placeholders as multimodal prompts");

    std::cout << "placeholder tests passed" << '\n';
    return 0;
}
