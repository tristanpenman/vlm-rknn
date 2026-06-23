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

#include <stdexcept>
#include <utility>

#include "logger.h"
#include "registry.h"

Registry::Registry(std::vector<vlm_rknn::NamedModelConfig> configs, size_t capacity)
  : configs_(std::move(configs))
  , capacity_(capacity == 0 ? 1 : capacity)
{
    if (configs_.empty()) {
        throw std::invalid_argument("At least one model configuration is required");
    }

    for (const auto& named : configs_) {
        config_by_id_.emplace(named.model_id, &named.config);
    }

    default_id_ = configs_.front().model_id;
}

const std::string& Registry::default_model_id() const noexcept
{
    return default_id_;
}

bool Registry::has_model(const std::string& id) const
{
    return config_by_id_.find(id) != config_by_id_.end();
}

std::vector<std::string> Registry::model_ids() const
{
    std::vector<std::string> ids;
    ids.reserve(configs_.size());
    for (const auto& named : configs_) {
        ids.push_back(named.model_id);
    }
    return ids;
}

bool Registry::default_ready() const
{
    const auto it = loaded_.find(default_id_);
    return it != loaded_.end() && it->second->is_ready();
}

vlm_rknn::Session* Registry::acquire(const std::string& id)
{
    const auto config_it = config_by_id_.find(id);
    if (config_it == config_by_id_.end()) {
        return nullptr;
    }

    const auto loaded_it = loaded_.find(id);
    if (loaded_it != loaded_.end()) {
        touch(id);
        return loaded_it->second.get();
    }

    while (loaded_.size() >= capacity_) {
        evict_lru();
    }

    auto session = std::make_unique<vlm_rknn::Session>(*config_it->second);
    if (session->init() != 0 || !session->is_ready()) {
        LOG(ERROR) << "Failed to initialize model '" << id << "'";
        return nullptr;
    }

    vlm_rknn::Session* ptr = session.get();
    loaded_.emplace(id, std::move(session));
    lru_.push_front(id);
    LOG(INFO) << "Loaded model '" << id << "' (" << loaded_.size() << "/" << capacity_ << " resident)";
    return ptr;
}

void Registry::touch(const std::string& id)
{
    lru_.remove(id);
    lru_.push_front(id);
}

void Registry::evict_lru()
{
    if (lru_.empty()) {
        return;
    }
    const std::string victim = lru_.back();
    lru_.pop_back();
    loaded_.erase(victim);
    LOG(INFO) << "Evicted model '" << victim << "' to free NPU memory";
}
