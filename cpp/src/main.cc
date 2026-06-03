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
               << " [--max-new-tokens <tokens>] [--max-context-len <tokens>] <vision_encoder_path>"
               << " <language_model_path> <image_path> [prompt]";
    LOG(ERROR) << "If [prompt] is omitted, an interactive REPL is started.";
}

}  // namespace

int main(int argc, char** argv)
{
    Logger::configure(std::cout);

    std::optional<int> num_cores;
    std::optional<int> max_new_tokens;
    std::optional<int> max_context_len;
    std::vector<const char*> positional_args;
    positional_args.reserve(static_cast<size_t>(argc - 1));
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            Logger::configure(std::cout, Logger::Level::Verbose);
            continue;
        }
        if (strcmp(argv[i], "--cores") == 0) {
            if (i + 1 >= argc) {
                LOG(WARNING) << "--cores option requires an argument specifying the number of cores to use";
                return -1;
            }
            num_cores = std::atoi(argv[++i]);
            if (num_cores <= 0 || num_cores > 3) {
                LOG(WARNING) << "Invalid number of cores specified: " << argv[i];
                return -1;
            }
            continue;
        }
        if (strcmp(argv[i], "--max-new-tokens") == 0) {
            if (i + 1 >= argc) {
                LOG(WARNING) << "--max-new-tokens option requires an argument specifying the token count";
                return -1;
            }
            max_new_tokens = std::atoi(argv[++i]);
            if (max_new_tokens <= 0) {
                LOG(WARNING) << "Invalid max new token count specified: " << argv[i];
                return -1;
            }
            continue;
        }
        if (strcmp(argv[i], "--max-context-len") == 0) {
            if (i + 1 >= argc) {
                LOG(WARNING) << "--max-context-len option requires an argument specifying the token count";
                return -1;
            }
            max_context_len = std::atoi(argv[++i]);
            if (max_context_len <= 0) {
                LOG(WARNING) << "Invalid max context length specified: " << argv[i];
                return -1;
            }
            continue;
        }
        positional_args.push_back(argv[i]);
    }

    if (positional_args.size() < 3) {
        print_usage(argv[0]);
        return 1;
    }
    if (positional_args.size() >= 4) {
        LOG(INFO) << "Prompt placeholder: " << positional_args[3];
    }

    qwen_vl_rknn::ModelConfig config;
    config.vision_encoder_path = positional_args[0];
    config.language_model_path = positional_args[1];
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
    const auto image_path = positional_args[2];
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        LOG(ERROR) << "Could not open or find the image: " << image_path;
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
    if (positional_args.size() >= 4) {
        LOG(INFO) << "Running decoder with prompt: " << positional_args[3];
        ret = session.decode(positional_args[3], img_vec.data());
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
