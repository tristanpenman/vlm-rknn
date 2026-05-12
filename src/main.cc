#include <iostream>
#include <string>

#include "qwen_vl_rknn.h"

namespace {

void print_usage(const char* program)
{
    std::cerr << "Usage: " << program << " <model_dir> [prompt]" << '\n';
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    qwen_vl_rknn::ModelConfig config;
    config.model_dir = argv[1];

    qwen_vl_rknn::Session session(config);
    std::cout << session.describe() << '\n';

    if (argc >= 3) {
        std::cout << "Prompt placeholder: " << argv[2] << '\n';
    }

    std::cout << "RKNN inference integration is not implemented yet." << '\n';
    return session.is_ready() ? 0 : 1;
}
