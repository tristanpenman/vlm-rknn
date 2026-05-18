#pragma once

#include <string>

namespace qwen_vl_rknn {

struct ModelConfig {
    std::string vision_encoder_path;
    std::string language_model_path;
};

class Session {
public:
    explicit Session(ModelConfig config);

    const ModelConfig& config() const noexcept;
    bool is_ready() const noexcept;
    std::string describe() const;

private:
    ModelConfig config_;
};

std::string target_device();

}  // namespace qwen_vl_rknn
