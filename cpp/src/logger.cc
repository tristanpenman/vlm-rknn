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

#include <iostream>
#include <mutex>
#include <utility>

#include "logger.h"

using namespace std;

atomic<ostream*> Logger::os_ = nullptr;
atomic<Logger::Level> Logger::min_level_ = Level::kInfo;
mutex Logger::mutex_;

namespace {
const char* level_label(const Logger::Level level)
{
    switch (level) {
    case Logger::Level::kInfo:
        return "I";
    case Logger::Level::kWarning:
        return "W";
    case Logger::Level::kError:
        return "E";
    case Logger::Level::kVerbose:
        return "V";
    default:
        return "U";
    }
}
}  // namespace

Logger::Logger(string name)
  : name_(std::move(name))
{
}

void Logger::configure()
{
    os_ = &cout;
}

void Logger::configure(ostream &os)
{
    os_ = &os;
}

void Logger::configure(const Level min_level)
{
    min_level_ = min_level;
}

void Logger::configure(ostream &os, const Level min_level)
{
    os_ = &os;
    min_level_ = min_level;
}

//
// Logger::Writer implementation
//

Logger::Writer::Writer(Logger &logger, const Level level)
  : logger_(logger)
{
    enabled_ = os_.load() && level >= min_level_.load();
    if (!enabled_) {
        return;
    }

    ss_ << "[" << level_label(level) << "]";
    if (!logger_.name_.empty()) {
        ss_ << "[" << logger_.name_ << "]";
    }
    ss_ << " ";
}

Logger::Writer::~Writer()
{
    if (!enabled_) {
        return;
    }

    ostream* os = os_.load();
    if (!os) {
        return;
    }

    lock_guard lock(mutex_);
    *os << ss_.str() << '\n';
}
