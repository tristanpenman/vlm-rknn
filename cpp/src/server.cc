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

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include "logger.h"
#include "vlm_rknn.h"

namespace {

using Json = nlohmann::json;

struct ServerOptions {
    std::string host = "0.0.0.0";
    int port = 8080;
    vlm_rknn::ModelConfig model;
};

struct QueryResult {
    int status = 200;
    Json body;
};

void print_usage(const char* program)
{
    LOG(ERROR) << "Usage: " << program
               << " [-v|--verbose] [--host <address>] [--port <port>] [--cores <num_cores>]"
               << " [--model-family <qwen2-vl|qwen2.5-vl|qwen3-vl|llama|smolvlm2>]"
               << " [--max-new-tokens <tokens>] [--max-context-len <tokens>]"
               << " --llm <language_model_path>"
               << " [--vision <vision_encoder_path>]";
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

bool parse_int_option(const char* option, const char* value, int min_value, int max_value, int& parsed)
{
    errno = 0;
    char* end = nullptr;
    const long result = std::strtol(value, &end, 10);
    if (value == end || *end != '\0' || errno == ERANGE || result < min_value || result > max_value) {
        LOG(WARNING) << "Invalid value for " << option << ": " << value
                     << " (expected " << min_value << "-" << max_value << ")";
        return false;
    }

    parsed = static_cast<int>(result);
    return true;
}

bool parse_options(int argc, char** argv, ServerOptions& options)
{
    std::optional<std::string> vision_encoder_path;
    std::optional<std::string> language_model_path;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            Logger::configure(std::cout, Logger::Level::Verbose);
            continue;
        }
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return false;
        }
        if (strcmp(argv[i], "--host") == 0) {
            const char* value = nullptr;
            if (!get_option_value(argc, argv, i, "--host", value)) {
                return false;
            }
            options.host = value;
            continue;
        }
        if (strcmp(argv[i], "--port") == 0) {
            const char* value = nullptr;
            if (!get_option_value(argc, argv, i, "--port", value)) {
                return false;
            }
            if (!parse_int_option("--port", value, 1, 65535, options.port)) {
                return false;
            }
            continue;
        }
        if (strcmp(argv[i], "--cores") == 0) {
            const char* value = nullptr;
            if (!get_option_value(argc, argv, i, "--cores", value)) {
                return false;
            }
            int parsed = 0;
            if (!parse_int_option("--cores", value, 1, 3, parsed)) {
                return false;
            }
            options.model.num_cores = parsed;
            continue;
        }
        if (strcmp(argv[i], "--model-family") == 0) {
            const char* value = nullptr;
            if (!get_option_value(argc, argv, i, "--model-family", value)) {
                return false;
            }
            if (!vlm_rknn::parse_model_family(value, options.model.model_family)) {
                LOG(WARNING) << "Invalid model family specified: " << value;
                return false;
            }
            continue;
        }
        if (strcmp(argv[i], "--max-new-tokens") == 0) {
            const char* value = nullptr;
            if (!get_option_value(argc, argv, i, "--max-new-tokens", value)) {
                return false;
            }
            if (!parse_int_option("--max-new-tokens", value, 1, INT_MAX, options.model.max_new_tokens)) {
                return false;
            }
            continue;
        }
        if (strcmp(argv[i], "--max-context-len") == 0) {
            const char* value = nullptr;
            if (!get_option_value(argc, argv, i, "--max-context-len", value)) {
                return false;
            }
            if (!parse_int_option("--max-context-len", value, 1, INT_MAX, options.model.max_context_len)) {
                return false;
            }
            continue;
        }
        if (strcmp(argv[i], "--vision") == 0) {
            const char* value = nullptr;
            if (!get_option_value(argc, argv, i, "--vision", value)) {
                return false;
            }
            vision_encoder_path = value;
            continue;
        }
        if (strcmp(argv[i], "--llm") == 0) {
            const char* value = nullptr;
            if (!get_option_value(argc, argv, i, "--llm", value)) {
                return false;
            }
            language_model_path = value;
            continue;
        }

        LOG(WARNING) << "Unexpected positional argument or unknown option: " << argv[i];
        return false;
    }

    if (!language_model_path.has_value() || language_model_path->empty()) {
        LOG(WARNING) << "Missing required --llm <language_model_path> argument";
        return false;
    }
    options.model.language_model_path = *language_model_path;

    const bool uses_vision_encoder = vlm_rknn::model_family_uses_vision_encoder(options.model.model_family);
    if (uses_vision_encoder) {
        if (!vision_encoder_path.has_value() || vision_encoder_path->empty()) {
            LOG(WARNING) << "Missing required --vision <vision_encoder_path> argument for "
                         << vlm_rknn::model_family_name(options.model.model_family);
            return false;
        }
        options.model.vision_encoder_path = *vision_encoder_path;
    } else if (vision_encoder_path.has_value() && !vision_encoder_path->empty()) {
        LOG(WARNING) << "--vision is not supported for "
                     << vlm_rknn::model_family_name(options.model.model_family);
        return false;
    }

    return true;
}

Json error_body(const std::string& message)
{
    return Json{{"error", message}};
}

void send_json(httplib::Response& response, int status, const Json& body)
{
    response.status = status;
    response.set_content(body.dump(), "application/json");
}

bool encode_image(vlm_rknn::Session& session, const std::string& image_path, std::vector<float>& image_embedding)
{
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        LOG(ERROR) << "Could not open or find the image: " << image_path;
        return false;
    }

    const auto& encoder = session.vision_encoder();
    cv::Mat preprocessed;
    int ret = vlm_rknn::preprocess_image_for_vision_encoder(
        session.config().model_family,
        image,
        cv::Size(encoder.model_width, encoder.model_height),
        preprocessed);
    if (ret != 0) {
        LOG(ERROR) << "Failed to preprocess image, error=" << ret;
        return false;
    }

    const size_t image_embed_len = static_cast<size_t>(encoder.model_image_token)
        * static_cast<size_t>(encoder.model_embed_size)
        * static_cast<size_t>(encoder.io_num.n_output);
    image_embedding.assign(image_embed_len, 0.0f);

    ret = session.encode(preprocessed.data, image_embedding.data());
    if (ret != 0) {
        LOG(ERROR) << "Failed to encode image, error=" << ret;
        image_embedding.clear();
        return false;
    }

    return true;
}

QueryResult run_query(vlm_rknn::Session& session, std::mutex& session_mutex, const Json& request)
{
    if (!request.is_object()) {
        return {400, error_body("request body must be a JSON object")};
    }
    if (!request.contains("prompt") || !request["prompt"].is_string()) {
        return {400, error_body("request field 'prompt' must be a string")};
    }

    const std::string prompt = request["prompt"].get<std::string>();
    const std::optional<std::string> image_path =
        request.contains("image") && request["image"].is_string()
            ? std::make_optional(request["image"].get<std::string>())
            : std::nullopt;

    std::lock_guard<std::mutex> lock(session_mutex);
    if (!session.is_ready()) {
        return {503, error_body("session is not ready")};
    }

    std::vector<float> image_embedding;
    float* image_embedding_data = nullptr;
    if (session.prompt_contains_image(prompt)) {
        if (!image_path.has_value() || image_path->empty()) {
            return {400, error_body("prompt references an image but request field 'image' was not provided")};
        }
        if (!encode_image(session, *image_path, image_embedding)) {
            return {500, error_body("image encoding failed")};
        }
        image_embedding_data = image_embedding.data();
    }

    std::string output;
    session.set_output_callback([&output](const char* text, LLMCallState state) {
        if (state == RKLLM_RUN_NORMAL && text != nullptr) {
            output += text;
        }
    });

    const int ret = session.decode(prompt, image_embedding_data);
    session.set_output_callback(nullptr);
    if (ret != 0) {
        return {500, error_body("decoder run failed")};
    }

    return {200, Json{{"text", output}}};
}

}  // namespace

int main(int argc, char** argv)
{
    Logger::configure(std::cout);

    ServerOptions options;
    if (!parse_options(argc, argv, options)) {
        print_usage(argv[0]);
        return 1;
    }

    vlm_rknn::Session session(options.model);
    if (session.init() != 0 || !session.is_ready()) {
        LOG(ERROR) << "Session initialization failed.";
        return 1;
    }

    std::mutex session_mutex;
    httplib::Server server;

    server.Get("/health", [&session, &session_mutex](const httplib::Request&, httplib::Response& response) {
        std::lock_guard<std::mutex> lock(session_mutex);
        send_json(response, 200, Json{{"ready", session.is_ready()}});
    });

    server.Post("/query", [&session, &session_mutex](const httplib::Request& request, httplib::Response& response) {
        Json body;
        try {
            body = Json::parse(request.body);
        } catch (const Json::parse_error& error) {
            send_json(response, 400, error_body(std::string("invalid JSON: ") + error.what()));
            return;
        }

        const QueryResult result = run_query(session, session_mutex, body);
        send_json(response, result.status, result.body);
    });

    LOG(INFO) << "Session initialized successfully.";
    LOG(INFO) << session.describe();
    LOG(INFO) << "Starting HTTP server on " << options.host << ":" << options.port;

    if (!server.listen(options.host, options.port)) {
        LOG(ERROR) << "Failed to listen on " << options.host << ":" << options.port;
        return 1;
    }

    return 0;
}
