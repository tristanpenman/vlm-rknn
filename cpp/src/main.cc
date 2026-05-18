#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "logger.h"
#include "qwen_vl_rknn.h"

namespace {

void print_usage(const char* program)
{
    std::cerr << "Usage: " << program << " <vision_encoder_path> <language_model_path> [prompt]" << '\n';
}

}  // namespace

int main(int argc, char** argv)
{
    bool verbose = false;
    std::vector<const char*> positional_args;
    positional_args.reserve(static_cast<size_t>(argc - 1));
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
            continue;
        }
        positional_args.push_back(argv[i]);
    }

    if (positional_args.size() < 2) {
        print_usage(argv[0]);
        return 1;
    }

    Logger::configure(std::cout, verbose ? Logger::Level::Verbose : Logger::Level::Info);
    LOG(INFO) << "Marian RKNN Translator Demo";

    qwen_vl_rknn::ModelConfig config;
    config.vision_encoder_path = positional_args[0];
    config.language_model_path = positional_args[1];

    qwen_vl_rknn::Session session(config);
    std::cout << session.describe() << '\n';

    if (positional_args.size() >= 3) {
        std::cout << "Prompt placeholder: " << positional_args[2] << '\n';
    }

    std::cout << "RKNN inference integration is not implemented yet." << '\n';
    return session.is_ready() ? 0 : 1;
}
