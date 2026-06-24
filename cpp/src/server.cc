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
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include "logger.h"
#include "registry.h"
#include "vlm_rknn.h"

namespace {

using Json = nlohmann::json;

struct ServerOptions
{
    std::string host = "0.0.0.0";
    int port = 8080;
    std::string ini_file_path;
    int max_loaded_models = 1;
};

struct QueryResult
{
    int status = 200;
    Json body;
};

void print_usage(const char* program)
{
    LOG(ERROR) << "Usage: " << program
               << " [-v|--verbose] [--host <address>] [--port <port>]"
               << " [--max-loaded-models <count>]"
               << " --ini-file <path>";
    LOG(ERROR) << "The INI file defines one or more models, one per [model_id] section.";
    LOG(ERROR) << "Recognised keys: model_family, vision, llm, max_new_tokens, max_context_len, cores.";
    LOG(ERROR) << "The first model in the file is the default; requests may select another via 'model_id'.";
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
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            Logger::configure(std::cout, Logger::Level::kVerbose);
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
        if (strcmp(argv[i], "--max-loaded-models") == 0) {
            const char* value = nullptr;
            if (!get_option_value(argc, argv, i, "--max-loaded-models", value)) {
                return false;
            }
            if (!parse_int_option("--max-loaded-models", value, 1, INT_MAX, options.max_loaded_models)) {
                return false;
            }
            continue;
        }
        if (strcmp(argv[i], "--ini-file") == 0) {
            const char* value = nullptr;
            if (!get_option_value(argc, argv, i, "--ini-file", value)) {
                return false;
            }
            options.ini_file_path = value;
            continue;
        }

        LOG(WARNING) << "Unexpected positional argument or unknown option: " << argv[i];
        return false;
    }

    if (options.ini_file_path.empty()) {
        LOG(WARNING) << "Missing required --ini-file <path> argument";
        return false;
    }

    return true;
}

bool read_file(const std::string& path, std::string& contents)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        LOG(ERROR) << "Could not open INI file: " << path;
        return false;
    }

    std::ostringstream stream;
    stream << file.rdbuf();
    contents = stream.str();
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

// Run a query against an already-resolved, ready session. The caller is
// responsible for holding the registry mutex for the duration of the call.
QueryResult run_query(vlm_rknn::Session& session, const Json& request)
{
    if (!request.contains("prompt") || !request["prompt"].is_string()) {
        return {400, error_body("request field 'prompt' must be a string")};
    }

    const std::string prompt = request["prompt"].get<std::string>();
    const std::optional<std::string> image_path =
        request.contains("image") && request["image"].is_string()
            ? std::make_optional(request["image"].get<std::string>())
            : std::nullopt;

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

    std::string ini_text;
    if (!read_file(options.ini_file_path, ini_text)) {
        return 1;
    }

    std::vector<vlm_rknn::NamedModelConfig> configs;
    std::string parse_error;
    if (!vlm_rknn::parse_model_configs_from_ini(ini_text, configs, parse_error)) {
        LOG(ERROR) << "Failed to parse " << options.ini_file_path << ": " << parse_error;
        return 1;
    }

    LOG(INFO) << "Loaded " << configs.size() << " model definition(s) from " << options.ini_file_path;
    for (const auto& named : configs) {
        LOG(INFO) << "  " << named.model_id
                  << " (" << vlm_rknn::model_family_name(named.config.model_family) << ")"
                  << (named.model_id == configs.front().model_id ? " [default]" : "");
    }

    Registry registry(std::move(configs), static_cast<size_t>(options.max_loaded_models));
    std::mutex registry_mutex;

    // Eagerly load the default model so the server is ready to serve immediately.
    {
        std::lock_guard<std::mutex> lock(registry_mutex);
        vlm_rknn::Session* default_session = registry.acquire(registry.default_model_id());
        if (default_session == nullptr) {
            LOG(ERROR) << "Failed to load default model '" << registry.default_model_id() << "'";
            return 1;
        }
        LOG(INFO) << "Default model ready: " << default_session->describe();
    }

    httplib::Server server;

    server.Get("/health", [&registry, &registry_mutex](const httplib::Request&, httplib::Response& response) {
        std::lock_guard<std::mutex> lock(registry_mutex);
        send_json(response, 200, Json{{"ready", registry.default_ready()}});
    });

    server.Get("/models", [&registry](const httplib::Request&, httplib::Response& response) {
        Json models = Json::array();
        for (const auto& id : registry.model_ids()) {
            models.push_back(id);
        }
        send_json(response, 200, Json{{"models", models}, {"default", registry.default_model_id()}});
    });

    server.Post("/query", [&registry, &registry_mutex](const httplib::Request& request, httplib::Response& response) {
        Json body;
        try {
            body = Json::parse(request.body);
        } catch (const Json::parse_error& error) {
            send_json(response, 400, error_body(std::string("invalid JSON: ") + error.what()));
            return;
        }

        if (!body.is_object()) {
            send_json(response, 400, error_body("request body must be a JSON object"));
            return;
        }

        std::string model_id = registry.default_model_id();
        if (body.contains("model_id")) {
            if (!body["model_id"].is_string()) {
                send_json(response, 400, error_body("request field 'model_id' must be a string"));
                return;
            }
            model_id = body["model_id"].get<std::string>();
        }

        std::lock_guard<std::mutex> lock(registry_mutex);
        if (!registry.has_model(model_id)) {
            send_json(response, 400, error_body("unknown model_id: " + model_id));
            return;
        }

        vlm_rknn::Session* session = registry.acquire(model_id);
        if (session == nullptr) {
            send_json(response, 503, error_body("failed to load model '" + model_id + "'"));
            return;
        }

        const QueryResult result = run_query(*session, body);
        send_json(response, result.status, result.body);
    });

    LOG(INFO) << "Starting HTTP server on " << options.host << ":" << options.port
              << " (max " << options.max_loaded_models << " model(s) resident)";

    if (!server.listen(options.host, options.port)) {
        LOG(ERROR) << "Failed to listen on " << options.host << ":" << options.port;
        return 1;
    }

    return 0;
}
