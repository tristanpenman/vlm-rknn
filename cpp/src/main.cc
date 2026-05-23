#include <cstring>
#include <iostream>
#include <optional>
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

    if (positional_args.size() < 2) {
        print_usage(argv[0]);
        return 1;
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
    std::cout << session.describe() << '\n';

    if (positional_args.size() >= 3) {
        std::cout << "Prompt placeholder: " << positional_args[2] << '\n';
    }

    std::cout << "RKNN inference integration is not implemented yet." << '\n';
    return session.is_ready() ? 0 : 1;
}
