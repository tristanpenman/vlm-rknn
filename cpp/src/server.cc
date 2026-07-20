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

#include <cctype>
#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
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
    std::string iniFilePath;
    std::string publicPath;
    int maxLoadedModels = 1;
};

struct QueryResult
{
    int status = 200;
    Json body;
};

void printUsage(const char* program)
{
    std::cout << "Usage: " << program
              << " [-v|--verbose] [--host <address>] [--port <port>]"
              << " [--max-loaded-models <count>]"
              << " [--public <directory>]"
              << " --ini-file <path>\n";
    std::cout << "The INI file defines one or more models, one per [model_id] section.\n";
    std::cout << "Recognised keys: model_family, vision, llm, max_new_tokens, max_context_len, cores.\n";
    std::cout << "The first model in the file is the default; requests may select another via 'model_id'.\n";
}

bool getOptionValue(int argc, char** argv, int& index, const char* option, const char*& value)
{
    if (index + 1 >= argc) {
        std::cout << option << " option requires an argument\n";
        return false;
    }
    value = argv[++index];
    return true;
}

bool parseIntOption(const char* option, const char* value, int minValue, int maxValue, int& parsed)
{
    errno = 0;
    char* end = nullptr;
    const long result = std::strtol(value, &end, 10);
    if (value == end || *end != '\0' || errno == ERANGE || result < minValue || result > maxValue) {
        std::cout << "Invalid value for " << option << ": " << value
                  << " (expected " << minValue << "-" << maxValue << ")\n";
        return false;
    }

    parsed = static_cast<int>(result);
    return true;
}

bool parseOptions(int argc, char** argv, ServerOptions& options)
{
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            Logger::configure(std::cout, Logger::Level::kVerbose);
            continue;
        }
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return false;
        }
        if (strcmp(argv[i], "--host") == 0) {
            const char* value = nullptr;
            if (!getOptionValue(argc, argv, i, "--host", value)) {
                return false;
            }
            options.host = value;
            continue;
        }
        if (strcmp(argv[i], "--port") == 0) {
            const char* value = nullptr;
            if (!getOptionValue(argc, argv, i, "--port", value)) {
                return false;
            }
            if (!parseIntOption("--port", value, 1, 65535, options.port)) {
                return false;
            }
            continue;
        }
        if (strcmp(argv[i], "--max-loaded-models") == 0) {
            const char* value = nullptr;
            if (!getOptionValue(argc, argv, i, "--max-loaded-models", value)) {
                return false;
            }
            if (!parseIntOption("--max-loaded-models", value, 1, INT_MAX, options.maxLoadedModels)) {
                return false;
            }
            continue;
        }
        if (strcmp(argv[i], "--ini-file") == 0) {
            const char* value = nullptr;
            if (!getOptionValue(argc, argv, i, "--ini-file", value)) {
                return false;
            }
            options.iniFilePath = value;
            continue;
        }
        if (strcmp(argv[i], "--public") == 0) {
            const char* value = nullptr;
            if (!getOptionValue(argc, argv, i, "--public", value)) {
                return false;
            }
            options.publicPath = value;
            continue;
        }

        std::cout << "Unexpected positional argument or unknown option: " << argv[i] << "\n";
        return false;
    }

    if (options.iniFilePath.empty()) {
        std::cout << "Missing required --ini-file <path> argument\n";
        return false;
    }

    return true;
}

bool readFile(const std::string& path, std::string& contents)
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

Json errorBody(const std::string& message)
{
    return Json{{"error", message}};
}

void sendJson(httplib::Response& response, int status, const Json& body)
{
    response.status = status;
    response.set_content(body.dump(), "application/json");
}

bool hasUnsafePathComponent(const std::string& path)
{
    if (path.empty() || path.front() != '/'
        || path.find('\\') != std::string::npos || path.find('\0') != std::string::npos) {
        return true;
    }

    std::size_t offset = 0;
    while (offset <= path.size()) {
        const std::size_t end = path.find('/', offset);
        const std::string component = path.substr(offset, end - offset);
        if (component == "." || component == "..") {
            return true;
        }
        if (end == std::string::npos) {
            break;
        }
        offset = end + 1;
    }
    return false;
}

bool isWithinDirectory(const std::filesystem::path& directory, const std::filesystem::path& path)
{
    auto directoryPart = directory.begin();
    auto pathPart = path.begin();
    for (; directoryPart != directory.end(); ++directoryPart, ++pathPart) {
        if (pathPart == path.end() || *directoryPart != *pathPart) {
            return false;
        }
    }
    return true;
}

// Maximum size, in bytes, of an uploaded (base64-decoded) image.
constexpr std::size_t kMaxUploadedImageBytes = 1024 * 1024;

// Decode a standard base64 string (no URL-safe alphabet). Whitespace is
// tolerated and '=' padding terminates decoding. Returns false on an invalid
// character.
bool base64Decode(const std::string& input, std::string& output)
{
    const auto decodeChar = [](unsigned char c) -> int {
        if (c >= 'A' && c <= 'Z') {
            return c - 'A';
        }
        if (c >= 'a' && c <= 'z') {
            return c - 'a' + 26;
        }
        if (c >= '0' && c <= '9') {
            return c - '0' + 52;
        }
        if (c == '+') {
            return 62;
        }
        if (c == '/') {
            return 63;
        }
        return -1;
    };

    output.clear();
    int val = 0;
    int bits = -8;
    for (const unsigned char c : input) {
        if (c == '=') {
            break;
        }
        if (std::isspace(c)) {
            continue;
        }
        const int d = decodeChar(c);
        if (d < 0) {
            return false;
        }
        val = (val << 6) | d;
        bits += 6;
        if (bits >= 0) {
            output.push_back(static_cast<char>((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return true;
}

bool encodeImage(vlm_rknn::Session& session, const cv::Mat& image, std::vector<float>& imageEmbedding)
{
    const auto& encoder = session.visionEncoder();
    cv::Mat preprocessed;
    int ret = vlm_rknn::preprocessImageForVisionEncoder(
        session.config().modelFamily,
        image,
        cv::Size(encoder.modelWidth, encoder.modelHeight),
        preprocessed);
    if (ret != 0) {
        LOG(ERROR) << "Failed to preprocess image, error=" << ret;
        return false;
    }

    const std::size_t imageEmbedLen = static_cast<std::size_t>(encoder.modelImageToken)
        * static_cast<std::size_t>(encoder.modelEmbedSize)
        * static_cast<std::size_t>(encoder.ioNum.n_output);
    imageEmbedding.assign(imageEmbedLen, 0.0f);

    ret = session.encode(preprocessed.data, imageEmbedding.data());
    if (ret != 0) {
        LOG(ERROR) << "Failed to encode image, error=" << ret;
        imageEmbedding.clear();
        return false;
    }

    return true;
}

// Run a query against an already-resolved, ready session. The caller is
// responsible for holding the registry mutex for the duration of the call.
QueryResult runQuery(vlm_rknn::Session& session, const Json& request)
{
    if (!request.contains("prompt") || !request["prompt"].is_string()) {
        return {400, errorBody("request field 'prompt' must be a string")};
    }

    const std::string prompt = request["prompt"].get<std::string>();

    if (!session.isReady()) {
        return {503, errorBody("session is not ready")};
    }

    const bool hasImageData =
        request.contains("image_data") && !request["image_data"].is_null();

    std::vector<float> imageEmbedding;
    float* imageEmbeddingData = nullptr;
    if (session.promptContainsImage(prompt)) {
        cv::Mat image;
        // Images must be uploaded as base64-encoded bytes via 'image_data'.
        // Referencing a server-side path is deliberately not supported, as it
        // would let a client read arbitrary files from the server's disk.
        if (!hasImageData) {
            return {400, errorBody("prompt references an image but 'image_data' was not provided")};
        }
        if (!request["image_data"].is_string()) {
            return {400, errorBody("request field 'image_data' must be a base64-encoded string")};
        }
        std::string imageBytes;
        if (!base64Decode(request["image_data"].get<std::string>(), imageBytes)) {
            return {400, errorBody("request field 'image_data' is not valid base64")};
        }
        if (imageBytes.empty()) {
            return {400, errorBody("request field 'image_data' decoded to zero bytes")};
        }
        if (imageBytes.size() > kMaxUploadedImageBytes) {
            return {413, errorBody("uploaded image exceeds the maximum size of 1 MB")};
        }
        const std::vector<uchar> buffer(imageBytes.begin(), imageBytes.end());
        image = cv::imdecode(buffer, cv::IMREAD_COLOR);
        if (image.empty()) {
            return {400, errorBody("could not decode uploaded image data")};
        }

        if (!encodeImage(session, image, imageEmbedding)) {
            return {500, errorBody("image encoding failed")};
        }
        imageEmbeddingData = imageEmbedding.data();
    }

    std::string output;
    session.setOutputCallback([&output](const char* text, LLMCallState state) {
        if (state == RKLLM_RUN_NORMAL && text != nullptr) {
            output += text;
        }
    });

    const int ret = session.decode(prompt, imageEmbeddingData);
    session.setOutputCallback(nullptr);
    if (ret != 0) {
        return {500, errorBody("decoder run failed")};
    }

    return {200, Json{{"text", output}}};
}

}  // namespace

int main(int argc, char** argv)
{
    Logger::configure(std::cout);

    ServerOptions options;
    if (!parseOptions(argc, argv, options)) {
        printUsage(argv[0]);
        return 1;
    }

    std::string iniText;
    if (!readFile(options.iniFilePath, iniText)) {
        return 1;
    }

    std::vector<vlm_rknn::NamedModelConfig> configs;
    std::string parseError;
    if (!vlm_rknn::parseModelConfigsFromIni(iniText, configs, parseError)) {
        LOG(ERROR) << "Failed to parse " << options.iniFilePath << ": " << parseError;
        return 1;
    }

    LOG(INFO) << "Loaded " << configs.size() << " model definition(s) from " << options.iniFilePath;
    for (const auto& named : configs) {
        LOG(INFO) << "  " << named.modelId
                  << " (" << vlm_rknn::modelFamilyName(named.config.modelFamily) << ")"
                  << (named.modelId == configs.front().modelId ? " [default]" : "");
    }

    Registry registry(std::move(configs), static_cast<std::size_t>(options.maxLoadedModels));
    std::mutex registryMutex;

    // Eagerly load the default model so the server is ready to serve immediately.
    {
        std::lock_guard<std::mutex> lock(registryMutex);
        vlm_rknn::Session* defaultSession = registry.acquire(registry.defaultModelId());
        if (defaultSession == nullptr) {
            LOG(ERROR) << "Failed to load default model '" << registry.defaultModelId() << "'";
            return 1;
        }
        LOG(INFO) << "Default model ready: " << defaultSession->describe();
    }

    httplib::Server server;

    if (!options.publicPath.empty()) {
        std::error_code error;
        const std::filesystem::path publicDirectory = std::filesystem::canonical(options.publicPath, error);
        if (error || !std::filesystem::is_directory(publicDirectory)) {
            LOG(ERROR) << "Could not serve static content from directory: " << options.publicPath;
            return 1;
        }

        server.set_pre_routing_handler(
            [publicDirectory](const httplib::Request& request, httplib::Response& response) {
                if (hasUnsafePathComponent(request.path)) {
                    response.status = 400;
                    return httplib::Server::HandlerResponse::Handled;
                }

                std::error_code pathError;
                const std::filesystem::path requestPath =
                    std::filesystem::weakly_canonical(publicDirectory / request.path.substr(1), pathError);
                if (pathError || !isWithinDirectory(publicDirectory, requestPath)) {
                    response.status = 400;
                    return httplib::Server::HandlerResponse::Handled;
                }
                return httplib::Server::HandlerResponse::Unhandled;
            });

        if (!server.set_mount_point("/", publicDirectory.string())) {
            LOG(ERROR) << "Could not serve static content from directory: " << options.publicPath;
            return 1;
        }
        LOG(INFO) << "Serving static content from " << publicDirectory.string();
    }

    server.Get("/health", [&registry, &registryMutex](const httplib::Request&, httplib::Response& response) {
        std::lock_guard<std::mutex> lock(registryMutex);
        sendJson(response, 200, Json{{"ready", registry.defaultReady()}});
    });

    server.Get("/models", [&registry](const httplib::Request&, httplib::Response& response) {
        Json models = Json::array();
        for (const auto& id : registry.modelIds()) {
            models.push_back(id);
        }
        sendJson(response, 200, Json{{"models", models}, {"default", registry.defaultModelId()}});
    });

    server.Post("/query", [&registry, &registryMutex](const httplib::Request& request, httplib::Response& response) {
        Json body;
        try {
            body = Json::parse(request.body);
        } catch (const Json::parse_error& error) {
            sendJson(response, 400, errorBody(std::string("invalid JSON: ") + error.what()));
            return;
        }

        if (!body.is_object()) {
            sendJson(response, 400, errorBody("request body must be a JSON object"));
            return;
        }

        std::string modelId = registry.defaultModelId();
        if (body.contains("model_id")) {
            if (!body["model_id"].is_string()) {
                sendJson(response, 400, errorBody("request field 'model_id' must be a string"));
                return;
            }
            modelId = body["model_id"].get<std::string>();
        }

        std::lock_guard<std::mutex> lock(registryMutex);
        if (!registry.hasModel(modelId)) {
            sendJson(response, 400, errorBody("unknown model_id: " + modelId));
            return;
        }

        vlm_rknn::Session* session = registry.acquire(modelId);
        if (session == nullptr) {
            sendJson(response, 503, errorBody("failed to load model '" + modelId + "'"));
            return;
        }

        const QueryResult result = runQuery(*session, body);
        sendJson(response, result.status, result.body);
    });

    LOG(INFO) << "Starting HTTP server on " << options.host << ":" << options.port
              << " (max " << options.maxLoadedModels << " model(s) resident)";

    if (!server.listen(options.host, options.port)) {
        LOG(ERROR) << "Failed to listen on " << options.host << ":" << options.port;
        return 1;
    }

    return 0;
}
