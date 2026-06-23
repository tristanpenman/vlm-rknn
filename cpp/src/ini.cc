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

#include "ini.h"

#include <cctype>
#include <sstream>

namespace ini {

namespace {

std::string trim(const std::string& value)
{
    size_t begin = 0;
    size_t end = value.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(value[begin]))) {
        ++begin;
    }
    while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(begin, end - begin);
}

}  // namespace

std::optional<std::string> Section::get(const std::string& key) const
{
    for (const auto& entry : entries) {
        if (entry.first == key) {
            return entry.second;
        }
    }
    return std::nullopt;
}

bool parse(const std::string& text, Document& out, std::string& error)
{
    out.sections.clear();

    std::istringstream stream(text);
    std::string raw_line;
    int line_number = 0;
    Section* current = nullptr;

    while (std::getline(stream, raw_line)) {
        ++line_number;

        // Tolerate Windows line endings.
        if (!raw_line.empty() && raw_line.back() == '\r') {
            raw_line.pop_back();
        }

        const std::string line = trim(raw_line);
        if (line.empty() || line[0] == ';' || line[0] == '#') {
            continue;
        }

        if (line[0] == '[') {
            if (line.back() != ']') {
                error = "line " + std::to_string(line_number) + ": malformed section header: " + line;
                return false;
            }
            const std::string name = trim(line.substr(1, line.size() - 2));
            if (name.empty()) {
                error = "line " + std::to_string(line_number) + ": empty section name";
                return false;
            }
            for (const auto& section : out.sections) {
                if (section.name == name) {
                    error = "line " + std::to_string(line_number) + ": duplicate section: " + name;
                    return false;
                }
            }
            out.sections.push_back(Section{name, {}});
            current = &out.sections.back();
            continue;
        }

        const size_t equals = line.find('=');
        if (equals == std::string::npos) {
            error = "line " + std::to_string(line_number) + ": expected key=value: " + line;
            return false;
        }

        const std::string key = trim(line.substr(0, equals));
        const std::string value = trim(line.substr(equals + 1));
        if (key.empty()) {
            error = "line " + std::to_string(line_number) + ": empty key";
            return false;
        }
        if (current == nullptr) {
            error = "line " + std::to_string(line_number) + ": key '" + key + "' appears before any section";
            return false;
        }
        if (current->get(key).has_value()) {
            error = "line " + std::to_string(line_number) + ": duplicate key '" + key
                + "' in section [" + current->name + "]";
            return false;
        }

        current->entries.emplace_back(key, value);
    }

    return true;
}

}  // namespace ini
