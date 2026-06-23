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

#pragma once

#include <cstddef>
#include <list>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "vlm_rknn.h"

//
// A registry of model configurations backed by an LRU cache.
//
// The default model is loaded eagerly at startup; other models are loaded on
// demand. At most `capacity` models are kept resident in NPU memory at a time,
// with the least recently used model evicted when room is needed.
//
// The registry is not internally synchronised. Callers must serialise access,
// while the server holds a single mutex around request handling.
//
class Registry
{
public:
    Registry(std::vector<vlm_rknn::NamedModelConfig> configs, size_t capacity);

    const std::string& default_model_id() const noexcept;
    bool has_model(const std::string& id) const;
    std::vector<std::string> model_ids() const;
    bool default_ready() const;

    // Return a ready session for `id`, loading it (and evicting the least
    // recently used model if necessary) when it is not already resident.
    // Returns nullptr if `id` is unknown or initialization fails.
    vlm_rknn::Session* acquire(const std::string& id);

private:
    void touch(const std::string& id);
    void evict_lru();

    std::vector<vlm_rknn::NamedModelConfig> configs_;
    std::unordered_map<std::string, const vlm_rknn::ModelConfig*> config_by_id_;
    size_t capacity_;
    std::list<std::string> lru_;  // front = most recently used, back = least
    std::unordered_map<std::string, std::unique_ptr<vlm_rknn::Session>> loaded_;
    std::string default_id_;
};
