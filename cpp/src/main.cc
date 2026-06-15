// Copyright (c) 2025 by Rockchip Electronics Co., Ltd. All Rights Reserved.
// Copyright (c) 2026 Tristan Penman
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstring>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "cv2_utils.h"
#include "logger.h"
#include "qwen_vl_rknn.h"

namespace {

void print_usage(const char* program)
{
    LOG(ERROR) << "Usage: " << program
               << " [-v|--verbose] [--cores <num_cores>]"
               << " [--model-family <qwen2-vl|qwen2.5-vl|qwen3-vl|llama|smolvlm2>]"
               << " [--max-new-tokens <tokens>] [--max-context-len <tokens>]"
               << " --vision <vision_encoder_path> --llm <language_model_path>"
               << " --image <image_path> [--prompt <prompt>]";
    LOG(ERROR) << "If --prompt is omitted, an interactive REPL is started.";
}

bool get_option_value(int argc, char** argv, int& index, const char* option, const char*& value)
{
    if (index + 1 >= argc) {
        LOG(WARNING) << option << " option requires an argument";
        return false;
    }
    value = argv[++index];
    return true;
}

bool parse_model_family(const char* value, qwen_vl_rknn::ModelFamily& family)
{
    if (strcmp(value, "qwen2-vl") == 0 || strcmp(value, "qwen2_vl") == 0 || strcmp(value, "qwen2") == 0) {
        family = qwen_vl_rknn::ModelFamily::QwenVL2;
        return true;
    }

    if (strcmp(value, "qwen2.5-vl") == 0 || strcmp(value, "qwen2_5_vl") == 0 ||
        strcmp(value, "qwen25-vl") == 0 || strcmp(value, "qwen2.5") == 0) {
        family = qwen_vl_rknn::ModelFamily::QwenVL2_5;
        return true;
    }

    if (strcmp(value, "qwen3-vl") == 0 || strcmp(value, "qwen3_vl") == 0 || strcmp(value, "qwen3") == 0) {
        family = qwen_vl_rknn::ModelFamily::QwenVL3;
        return true;
    }

    if (strcmp(value, "llama") == 0 || strcmp(value, "llama3") == 0 || strcmp(value, "llama3.2") == 0) {
        family = qwen_vl_rknn::ModelFamily::Llama;
        return true;
    }

    if (strcmp(value, "smolvlm2") == 0 || strcmp(value, "smol-vlm2") == 0 || strcmp(value, "smol_vlm2") == 0) {
        family = qwen_vl_rknn::ModelFamily::SmolVLM2;
        return true;
    }

    return false;
}

}  // namespace

int main(int argc, char** argv)
{
    Logger::configure(std::cout);

    std::optional<int> num_cores;
    std::optional<int> max_new_tokens;
    std::optional<int> max_context_len;
    std::optional<qwen_vl_rknn::ModelFamily> model_family;
    std::optional<std::string> vision_encoder_path;
    std::optional<std::string> language_model_path;
    std::optional<std::string> image_path;
    std::optional<std::string> prompt;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            Logger::configure(std::cout, Logger::Level::Verbose);
            continue;
        }
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
        if (strcmp(argv[i], "--cores") == 0) {
            const char* value = nullptr;
            if (!get_option_value(argc, argv, i, "--cores", value)) {
                return -1;
            }
            num_cores = std::atoi(value);
            if (num_cores <= 0 || num_cores > 3) {
                LOG(WARNING) << "Invalid number of cores specified: " << value;
                return -1;
            }
            continue;
        }
        if (strcmp(argv[i], "--model-family") == 0) {
            const char* value = nullptr;
            if (!get_option_value(argc, argv, i, "--model-family", value)) {
                LOG(WARNING) << "--model-family option requires one of: qwen2-vl, qwen2.5-vl, qwen3-vl, llama, smolvlm2";
                return -1;
            }
            qwen_vl_rknn::ModelFamily parsed_family;
            if (!parse_model_family(value, parsed_family)) {
                LOG(WARNING) << "Invalid model family specified: " << value;
                LOG(WARNING) << "Expected one of: qwen2-vl, qwen2.5-vl, qwen3-vl, llama, smolvlm2";
                return -1;
            }
            model_family = parsed_family;
            continue;
        }
        if (strcmp(argv[i], "--max-new-tokens") == 0) {
            const char* value = nullptr;
            if (!get_option_value(argc, argv, i, "--max-new-tokens", value)) {
                return -1;
            }
            max_new_tokens = std::atoi(value);
            if (max_new_tokens <= 0) {
                LOG(WARNING) << "Invalid max new token count specified: " << value;
                return -1;
            }
            continue;
        }
        if (strcmp(argv[i], "--max-context-len") == 0) {
            const char* value = nullptr;
            if (!get_option_value(argc, argv, i, "--max-context-len", value)) {
                return -1;
            }
            max_context_len = std::atoi(value);
            if (max_context_len <= 0) {
                LOG(WARNING) << "Invalid max context length specified: " << value;
                return -1;
            }
            continue;
        }
        if (strcmp(argv[i], "--vision") == 0) {
            const char* value = nullptr;
            if (!get_option_value(argc, argv, i, "--vision", value)) {
                return -1;
            }
            vision_encoder_path = value;
            continue;
        }
        if (strcmp(argv[i], "--llm") == 0) {
            const char* value = nullptr;
            if (!get_option_value(argc, argv, i, "--llm", value)) {
                return -1;
            }
            language_model_path = value;
            continue;
        }
        if (strcmp(argv[i], "--image") == 0) {
            const char* value = nullptr;
            if (!get_option_value(argc, argv, i, "--image", value)) {
                return -1;
            }
            image_path = value;
            continue;
        }
        if (strcmp(argv[i], "--prompt") == 0) {
            const char* value = nullptr;
            if (!get_option_value(argc, argv, i, "--prompt", value)) {
                return -1;
            }
            prompt = value;
            continue;
        }

        LOG(WARNING) << "Unexpected positional argument or unknown option: " << argv[i];
        print_usage(argv[0]);
        return -1;
    }

    if (!vision_encoder_path.has_value() || vision_encoder_path->empty()) {
        LOG(WARNING) << "Missing required --vision <vision_encoder_path> argument";
        print_usage(argv[0]);
        return 1;
    }
    if (!language_model_path.has_value() || language_model_path->empty()) {
        LOG(WARNING) << "Missing required --llm <language_model_path> argument";
        print_usage(argv[0]);
        return 1;
    }
    if (!image_path.has_value() || image_path->empty()) {
        LOG(WARNING) << "Missing required --image <image_path> argument";
        print_usage(argv[0]);
        return 1;
    }

    qwen_vl_rknn::ModelConfig config;
    config.vision_encoder_path = *vision_encoder_path;
    config.language_model_path = *language_model_path;
    if (model_family.has_value()) {
        config.model_family = model_family.value();
    }
    if (num_cores.has_value()) {
        config.num_cores = num_cores.value();
    }
    if (max_new_tokens.has_value()) {
        config.max_new_tokens = max_new_tokens.value();
    }
    if (max_context_len.has_value()) {
        config.max_context_len = max_context_len.value();
    }

    qwen_vl_rknn::Session session(config);

    if (session.init() != 0) {
        LOG(ERROR) << "Session initialization failed.";
        return -1;
    }

    if (!session.is_ready()) {
        LOG(ERROR) << "Session is not ready. Something went wrong.";
        return -1;
    }

    LOG(INFO) << "Session initialized successfully.";
    LOG(INFO) << session.describe();

    // Load the image using OpenCV
    cv::Mat img = cv::imread(*image_path);
    if (img.empty()) {
        LOG(ERROR) << "Could not open or find the image: " << *image_path;
        return -1;
    }

    // The image is read in BGR format, convert to RGB format
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Expand the image into a square and fill it with the specified background color
    cv::Scalar background_color(127.5, 127.5, 127.5);
    cv::Mat square_img = expand2square(img, background_color);

    // Get image dimensions from the model configuration
    const auto& encoder = session.vision_encoder();
    auto image_width = encoder.model_width;
    auto image_height = encoder.model_height;

    // Resize the image
    LOG(INFO) << "Resizing image to " << image_width << "x" << image_height;
    cv::Mat resized_img;
    cv::Size new_size(image_width, image_height);
    cv::resize(square_img, resized_img, new_size, 0, 0, cv::INTER_LINEAR);

    // Determine size of the embedding that will be passed to the decoder
    auto n_image_tokens = encoder.model_image_token;
    auto image_embed_len = encoder.model_embed_size;
    auto n_embed_output = encoder.io_num.n_output;
    auto rkllm_image_embed_len = n_image_tokens * image_embed_len * n_embed_output;
    LOG(INFO) << "Image embedding size: " << rkllm_image_embed_len;

    // allocate memory for the image embedding output
    std::vector<float> img_vec;
    img_vec.resize(rkllm_image_embed_len);

    LOG(INFO) << "Running encoder...";
    int ret = session.encode(resized_img.data, img_vec.data());
    if (ret != 0) {
        LOG(ERROR) << "Failed to run model, error=" << ret;
        return -1;
    }

    LOG(INFO) << "Encoder ran successfully. Warming up decoder with multimodal input...";
    ret = session.decode("<image>What is in the image?", img_vec.data());
    if (ret != 0) {
        LOG(ERROR) << "Failed to run decoder, error=" << ret;
        return -1;
    }

    LOG(INFO) << "Decoder ran successfully!";
    if (prompt.has_value()) {
        LOG(INFO) << "Running decoder with prompt: " << *prompt;
        ret = session.decode(*prompt, img_vec.data());
        if (ret != 0) {
            LOG(ERROR) << "Failed to run decoder with prompt, error=" << ret;
            return -1;
        }
    } else {
        LOG(INFO) << "No prompt provided; starting interactive REPL.";
        LOG(INFO) << "Type ':quit' or ':exit' to end the session.";
        std::string prompt;
        while (true) {
            std::cout << "> " << std::flush;
            if (!std::getline(std::cin, prompt)) {
                LOG(INFO) << "EOF received, exiting REPL.";
                break;
            }
            if (prompt == ":quit" || prompt == ":exit") {
                LOG(INFO) << "Exiting REPL.";
                break;
            }
            if (prompt.empty()) {
                continue;
            }
            ret = session.decode(prompt, img_vec.data());
            if (ret != 0) {
                LOG(ERROR) << "Failed to run decoder with prompt, error=" << ret;
                return -1;
            }
        }
    }

    return 0;
}
