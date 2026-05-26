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
               << " [-v|--verbose] [--cores <num_cores>] <vision_encoder_path>"
               << " <language_model_path> <image_path> [prompt]";
}

}  // namespace

int main(int argc, char** argv)
{
    bool verbose = false;
    std::optional<int> num_cores;
    std::vector<const char*> positional_args;
    positional_args.reserve(static_cast<size_t>(argc - 1));
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
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
        positional_args.push_back(argv[i]);
    }

    if (positional_args.size() < 3) {
        print_usage(argv[0]);
        return 1;
    }
    if (positional_args.size() >= 4) {
        LOG(INFO) << "Prompt placeholder: " << positional_args[3];
    }

    Logger::configure(std::cout, verbose ? Logger::Level::Verbose : Logger::Level::Info);
    LOG(INFO) << "Qwen-VL RKNN Demo";

    qwen_vl_rknn::ModelConfig config;
    config.vision_encoder_path = positional_args[0];
    config.language_model_path = positional_args[1];
    if (num_cores.has_value()) {
        config.num_cores = num_cores.value();
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
    size_t image_width = encoder.model_width;
    size_t image_height = encoder.model_height;

    // Resize the image
    LOG(INFO) << "Resizing image to " << image_width << "x" << image_height;
    cv::Mat resized_img;
    cv::Size new_size(image_width, image_height);
    cv::resize(square_img, resized_img, new_size, 0, 0, cv::INTER_LINEAR);

    // Determine size of the embedding that will be passed to the decoder
    size_t n_image_tokens = encoder.model_image_token;
    size_t image_embed_len = encoder.model_embed_size;
    size_t n_embed_output = encoder.io_num.n_output;
    size_t rkllm_image_embed_len = n_image_tokens * image_embed_len * n_embed_output;
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
    ret = session.decode("<image>What is in the image?", nullptr, 0, img_vec.data());
    if (ret != 0) {
        LOG(ERROR) << "Failed to run decoder, error=" << ret;
        return -1;
    }

    LOG(INFO) << "Decoder ran successfully!";
    if (positional_args.size() >= 4) {
        LOG(INFO) << "Running decoder with prompt: " << positional_args[3];
        ret = session.decode(positional_args[3], nullptr, 0, img_vec.data());
        if (ret != 0) {
            LOG(ERROR) << "Failed to run decoder with prompt, error=" << ret;
            return -1;
        }
    }

    return 0;
}
