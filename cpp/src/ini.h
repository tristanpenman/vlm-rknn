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

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ini {

// A single [section] from an INI document, holding key/value entries in the
// order they appeared. Keys within a section are unique (enforced by parse()).
struct Section {
    std::string name;
    std::vector<std::pair<std::string, std::string>> entries;

    // Returns the value for the given key, or std::nullopt if the key is absent.
    std::optional<std::string> get(const std::string& key) const;
};

// A parsed INI document. Sections are kept in the order they appeared, so the
// first section in the file is sections.front().
struct Document {
    std::vector<Section> sections;
};

// Parse INI text into a Document.
//
// Supported syntax:
//   * [section] headers
//   * key=value entries (whitespace around key and value is trimmed)
//   * ';' and '#' line comments
//   * blank lines
//
// Values are taken verbatim after trimming surrounding whitespace; quotes are
// not interpreted. On failure, returns false and populates `error` with a
// human-readable message that includes the offending line number.
bool parse(const std::string& text, Document& out, std::string& error);

}  // namespace ini
