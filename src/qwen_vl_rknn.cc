#include "qwen_vl_rknn.h"

#include <sstream>
#include <utility>

namespace qwen_vl_rknn {

Session::Session(ModelConfig config) : config_(std::move(config)) {}

const ModelConfig& Session::config() const noexcept
{
    return config_;
}

bool Session::is_ready() const noexcept
{
    return !config_.model_dir.empty();
}

std::string Session::describe() const
{
    std::ostringstream stream;
    stream << "Qwen-VL RKNN session";
    stream << " target=" << target_device();
    stream << " model_dir=" << (config_.model_dir.empty() ? "<unset>" : config_.model_dir);
    return stream.str();
}

std::string target_device()
{
    return QWEN_VL_RKNN_TARGET;
}

}  // namespace qwen_vl_rknn
