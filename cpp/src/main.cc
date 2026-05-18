#include <iostream>
#include <string>

#include "qwen_vl_rknn.h"

namespace {

void print_usage(const char* program)
{
    std::cerr << "Usage: " << program << " <vision_encoder_path> <language_model_path> [prompt]" << '\n';
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    qwen_vl_rknn::ModelConfig config;
    config.vision_encoder_path = argv[1];
    config.language_model_path = argv[2];

    qwen_vl_rknn::Session session(config);
    std::cout << session.describe() << '\n';

    if (argc >= 4) {
        std::cout << "Prompt placeholder: " << argv[3] << '\n';
    }

    std::cout << "RKNN inference integration is not implemented yet." << '\n';
    return session.is_ready() ? 0 : 1;
}
