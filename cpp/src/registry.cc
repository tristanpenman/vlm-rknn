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

#include "registry.h"

#include <stdexcept>
#include <utility>

#include "logger.h"

Registry::Registry(std::vector<vlm_rknn::NamedModelConfig> configs, std::size_t capacity)
    : configs_(std::move(configs))
    , capacity_(capacity == 0 ? 1 : capacity)
{
    if (configs_.empty()) {
        throw std::invalid_argument("At least one model configuration is required");
    }

    for (const auto& named : configs_) {
        configById_.emplace(named.modelId, &named.config);
    }

    defaultId_ = configs_.front().modelId;
}

const std::string& Registry::defaultModelId() const noexcept
{
    return defaultId_;
}

bool Registry::hasModel(const std::string& id) const
{
    return configById_.find(id) != configById_.end();
}

std::vector<std::string> Registry::modelIds() const
{
    std::vector<std::string> ids;
    ids.reserve(configs_.size());
    for (const auto& named : configs_) {
        ids.push_back(named.modelId);
    }
    return ids;
}

bool Registry::defaultReady() const
{
    const auto it = loaded_.find(defaultId_);
    return it != loaded_.end() && it->second->isReady();
}

vlm_rknn::Session* Registry::acquire(const std::string& id)
{
    const auto configIt = configById_.find(id);
    if (configIt == configById_.end()) {
        return nullptr;
    }

    const auto loadedIt = loaded_.find(id);
    if (loadedIt != loaded_.end()) {
        touch(id);
        return loadedIt->second.get();
    }

    while (loaded_.size() >= capacity_) {
        evictLru();
    }

    auto session = std::make_unique<vlm_rknn::Session>(*configIt->second);
    if (session->init() != 0 || !session->isReady()) {
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

void Registry::evictLru()
{
    if (lru_.empty()) {
        return;
    }
    const std::string victim = lru_.back();
    lru_.pop_back();
    loaded_.erase(victim);
    LOG(INFO) << "Evicted model '" << victim << "' to free NPU memory";
}
