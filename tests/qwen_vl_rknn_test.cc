#include <iostream>
#include <string>

#include "qwen_vl_rknn.h"

namespace {

void expect(bool condition, const std::string& message)
{
    if (!condition) {
        std::cerr << "FAILED: " << message << '\n';
        std::exit(1);
    }
}

}  // namespace

int main()
{
    qwen_vl_rknn::Session empty_session({});
    expect(!empty_session.is_ready(), "empty session should not be ready");

    qwen_vl_rknn::ModelConfig config;
    config.model_dir = "/models/qwen-vl";
    qwen_vl_rknn::Session session(config);

    expect(session.is_ready(), "session with model_dir should be ready");
    expect(session.config().model_dir == "/models/qwen-vl", "model_dir should be retained");
    expect(session.describe().find("Qwen-VL RKNN session") != std::string::npos,
           "description should contain project name");
    expect(!qwen_vl_rknn::target_device().empty(), "target device should be configured");

    std::cout << "placeholder tests passed" << '\n';
    return 0;
}
