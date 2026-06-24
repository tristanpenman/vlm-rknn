#include <iostream>
#include <string>

#include "vlm_rknn.h"

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
    vlm_rknn::Session empty_session({});
    expect(!empty_session.is_ready(), "empty session should not be ready");

    vlm_rknn::ModelConfig config;
    expect(config.model_family == vlm_rknn::ModelFamily::kQwenVL2,
           "default model family should be Qwen2-VL");
    config.vision_encoder_path = "/models/qwen-vl/vision_encoder.rknn";
    config.language_model_path = "/models/qwen-vl/language_model.rkllm";
    vlm_rknn::Session session(config);

    expect(!session.is_ready(), "session should not be ready before init");
    expect(session.config().vision_encoder_path.has_value(), "vision_encoder_path should be set");
    expect(*session.config().vision_encoder_path == "/models/qwen-vl/vision_encoder.rknn", "vision_encoder_path should be retained");
    expect(session.config().language_model_path == "/models/qwen-vl/language_model.rkllm", "language_model_path should be retained");
    expect(session.describe().find("VLM RKNN session") != std::string::npos,
           "description should contain project name");
    expect(session.describe().find("target=") != std::string::npos, "description should contain target device");
    expect(session.describe().find("model_family=qwen2-vl") != std::string::npos,
           "description should contain model family");
    expect(session.describe().find("requires_vision_encoder=yes") != std::string::npos,
           "description should contain vision encoder requirement");

    vlm_rknn::ModelFamily family;
    expect(vlm_rknn::parse_model_family("llama", family), "llama should parse");
    expect(family == vlm_rknn::ModelFamily::kLlama, "llama should map to Llama family");
    expect(vlm_rknn::parse_model_family("smol-vlm2", family), "smol-vlm2 alias should parse");
    expect(family == vlm_rknn::ModelFamily::kSmolVLM2, "smol-vlm2 should map to SmolVLM2 family");
    expect(!vlm_rknn::parse_model_family("not-a-family", family), "unknown family should not parse");

    expect(!vlm_rknn::model_family_uses_vision_encoder(vlm_rknn::ModelFamily::kLlama),
           "llama should not require a vision encoder");
    expect(!vlm_rknn::model_family_supports_multimodal(vlm_rknn::ModelFamily::kLlama),
           "llama should not support multimodal input");
    expect(vlm_rknn::model_family_uses_vision_encoder(vlm_rknn::ModelFamily::kQwenVL2),
           "Qwen2-VL should require a vision encoder");
    expect(vlm_rknn::model_family_uses_vision_encoder(vlm_rknn::ModelFamily::kSmolVLM2),
           "SmolVLM2 should require a vision encoder");
    expect(vlm_rknn::model_family_supports_multimodal(vlm_rknn::ModelFamily::kSmolVLM2),
           "SmolVLM2 should be registered as multimodal");
    expect(std::string(vlm_rknn::model_family_image_placeholder(vlm_rknn::ModelFamily::kQwenVL2)) == "<image>",
           "Qwen2-VL image placeholder should be exposed");
    expect(std::string(vlm_rknn::model_family_image_placeholder(vlm_rknn::ModelFamily::kLlama)).empty(),
           "llama image placeholder should be empty");

    expect(session.prompt_contains_image("<image>Describe this."), "Qwen2-VL should detect its image placeholder");
    expect(!session.prompt_contains_image("Describe this."), "Qwen2-VL should allow plain text prompts");

    const auto& qwen_preprocess =
        vlm_rknn::model_family_image_preprocess_profile(vlm_rknn::ModelFamily::kQwenVL2);
    expect(qwen_preprocess.resize_mode == vlm_rknn::ResizeMode::kPadToSquare,
           "Qwen2-VL should pad images to square");
    expect(qwen_preprocess.rgb, "Qwen2-VL should request RGB encoder input");
    expect(!qwen_preprocess.normalize_in_host, "Qwen2-VL should not normalize in host code");

    const auto& smol_preprocess =
        vlm_rknn::model_family_image_preprocess_profile(vlm_rknn::ModelFamily::kSmolVLM2);
    expect(smol_preprocess.resize_mode == vlm_rknn::ResizeMode::kPadToSquare,
           "SmolVLM2 profile should have an explicit resize mode");

    cv::Mat bgr_pixel(1, 1, CV_8UC3, cv::Scalar(10, 20, 30));
    cv::Mat preprocessed_pixel;
    expect(vlm_rknn::preprocess_image_for_vision_encoder(
               vlm_rknn::ModelFamily::kQwenVL2,
               bgr_pixel,
               cv::Size(1, 1),
               preprocessed_pixel) == 0,
           "Qwen2-VL preprocessing should succeed");
    expect(preprocessed_pixel.cols == 1 && preprocessed_pixel.rows == 1,
           "preprocessed image should match requested dimensions");
    const cv::Vec3b rgb_pixel = preprocessed_pixel.at<cv::Vec3b>(0, 0);
    expect(rgb_pixel[0] == 30 && rgb_pixel[1] == 20 && rgb_pixel[2] == 10,
           "Qwen2-VL preprocessing should convert BGR to RGB");

    cv::Mat wide_bgr(1, 2, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat padded;
    expect(vlm_rknn::preprocess_image_for_vision_encoder(
               vlm_rknn::ModelFamily::kQwenVL2,
               wide_bgr,
               cv::Size(2, 2),
               padded) == 0,
           "Qwen2-VL preprocessing should pad non-square images");
    expect(padded.cols == 2 && padded.rows == 2, "padded image should match requested dimensions");
    expect(padded.at<cv::Vec3b>(1, 0)[0] >= 127 && padded.at<cv::Vec3b>(1, 0)[0] <= 128,
           "Qwen2-VL padding should use the profile background color");

    vlm_rknn::ModelConfig llama_config;
    llama_config.model_family = vlm_rknn::ModelFamily::kLlama;
    llama_config.language_model_path = "/models/llama.rkllm";
    vlm_rknn::Session llama_session(llama_config);
    expect(!llama_session.is_ready(), "llama session should not be ready before init");
    expect(llama_session.describe().find("model_family=llama") != std::string::npos,
           "description should contain llama model family");
    expect(llama_session.describe().find("requires_vision_encoder=no") != std::string::npos,
           "description should report llama does not require a vision encoder");
    expect(!llama_session.prompt_contains_image("<image>Describe this."),
           "llama should not treat image placeholders as multimodal prompts");

    // INI-based model configuration.
    {
        const std::string ini =
            "[qwen]\n"
            "model_family=qwen2-vl\n"
            "vision=/models/qwen/vision.rknn\n"
            "llm=/models/qwen/language.rkllm\n"
            "max_new_tokens=300\n"
            "\n"
            "; a text-only model\n"
            "[chat]\n"
            "model_family=llama\n"
            "llm=/models/llama.rkllm\n"
            "cores=2\n";

        std::vector<vlm_rknn::NamedModelConfig> configs;
        std::string error;
        expect(vlm_rknn::parse_model_configs_from_ini(ini, configs, error),
               "valid INI should parse: " + error);
        expect(configs.size() == 2, "two models should be parsed");
        expect(configs[0].model_id == "qwen", "first model should be the default 'qwen'");
        expect(configs[0].config.model_family == vlm_rknn::ModelFamily::kQwenVL2,
               "qwen model family should parse");
        expect(configs[0].config.vision_encoder_path.has_value() &&
                   *configs[0].config.vision_encoder_path == "/models/qwen/vision.rknn",
               "qwen vision path should be retained");
        expect(configs[0].config.language_model_path == "/models/qwen/language.rkllm",
               "qwen llm path should be retained");
        expect(configs[0].config.max_new_tokens == 300, "qwen max_new_tokens should be 300");
        expect(configs[1].model_id == "chat", "second model should be 'chat'");
        expect(configs[1].config.model_family == vlm_rknn::ModelFamily::kLlama,
               "chat model family should be llama");
        expect(!configs[1].config.vision_encoder_path.has_value(),
               "llama model should have no vision path");
        expect(configs[1].config.num_cores.has_value() && *configs[1].config.num_cores == 2,
               "chat cores should be 2");
    }

    {
        std::vector<vlm_rknn::NamedModelConfig> configs;
        std::string error;
        expect(!vlm_rknn::parse_model_configs_from_ini("", configs, error),
               "empty INI should fail");
        expect(!vlm_rknn::parse_model_configs_from_ini(
                   "[m]\nmodel_family=qwen2-vl\nvision=/v.rknn\n", configs, error),
               "missing llm should fail");
        expect(!vlm_rknn::parse_model_configs_from_ini(
                   "[m]\nllm=/m.rkllm\n", configs, error),
               "vision model without vision encoder should fail");
        expect(!vlm_rknn::parse_model_configs_from_ini(
                   "[m]\nmodel_family=llama\nllm=/m.rkllm\nvision=/v.rknn\n", configs, error),
               "vision key for llama should fail");
        expect(!vlm_rknn::parse_model_configs_from_ini(
                   "[m]\nmodel_family=llama\nllm=/m.rkllm\nbogus=1\n", configs, error),
               "unknown key should fail");
        expect(!vlm_rknn::parse_model_configs_from_ini(
                   "[m]\nmodel_family=llama\nllm=/m.rkllm\ncores=5\n", configs, error),
               "out-of-range cores should fail");
        expect(!vlm_rknn::parse_model_configs_from_ini(
                   "[m]\nllm=/a.rkllm\nmodel_family=llama\n[m]\nllm=/b.rkllm\nmodel_family=llama\n",
                   configs, error),
               "duplicate section should fail");
    }

    {
        // A text-only single-model INI should parse and default correctly.
        std::vector<vlm_rknn::NamedModelConfig> configs;
        std::string error;
        expect(vlm_rknn::parse_model_configs_from_ini(
                   "[only]\nmodel_family=llama\nllm=/m.rkllm\n", configs, error),
               "single llama model should parse: " + error);
        expect(configs.size() == 1 && configs[0].model_id == "only",
               "single model should be parsed and named");
        expect(configs[0].config.max_new_tokens == 128 && configs[0].config.max_context_len == 2048,
               "defaults should apply when keys are omitted");
    }

    std::cout << "placeholder tests passed" << '\n';
    return 0;
}
