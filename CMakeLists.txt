cmake_minimum_required(VERSION 3.20)
project(qwen-vl-rknn VERSION 0.1.0 LANGUAGES CXX)

set(COPYRIGHT "Copyright (c) 2026 Tristan Penman")
set(IDENTIFIER "com.tristanpenman.qwen-vl-rknn")

# Force out-of-source builds
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# C++17 is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# FetchContent: Easier than Git submodules
include(FetchContent)

# Default to Debug for single-config generators if user didn't specify anything
get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT is_multi_config AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

option(QWEN_VL_RKNN_BUILD_CLI "Build the qwen-vl-rknn command-line demo" ON)
option(QWEN_VL_RKNN_ENABLE_RKNN "Require Rockchip RKNN headers/runtime while configuring" OFF)

set(QWEN_VL_RKNN_TARGET "rk3588" CACHE STRING "Rockchip target device family")

set(RKNN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rknpu2/include" CACHE PATH
    "Path containing rknn_api.h")
set(RKNN_RUNTIME_LIB "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rknpu2/lib/aarch64/librknnrt.so" CACHE FILEPATH
    "Path to librknnrt.so for the target Rockchip device")

if(QWEN_VL_RKNN_ENABLE_RKNN)
    if(NOT EXISTS "${RKNN_INCLUDE_DIR}/rknn_api.h")
        message(FATAL_ERROR
            "RKNN headers not found in ${RKNN_INCLUDE_DIR}. "
            "Set -DRKNN_INCLUDE_DIR=<path-containing-rknn_api.h> or disable "
            "-DQWEN_VL_RKNN_ENABLE_RKNN=OFF for host-only development.")
    endif()

    if(NOT EXISTS "${RKNN_RUNTIME_LIB}")
        message(FATAL_ERROR
            "RKNN runtime library not found: ${RKNN_RUNTIME_LIB}. "
            "Set -DRKNN_RUNTIME_LIB=<path-to-librknnrt.so> or disable "
            "-DQWEN_VL_RKNN_ENABLE_RKNN=OFF for host-only development.")
    endif()
endif()

add_library(qwen_vl_rknn_core STATIC
    cpp/src/qwen_vl_rknn.cc
)
add_library(qwen-vl-rknn::core ALIAS qwen_vl_rknn_core)

target_include_directories(qwen_vl_rknn_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src
)

target_compile_definitions(qwen_vl_rknn_core
    PUBLIC
        QWEN_VL_RKNN_TARGET="${QWEN_VL_RKNN_TARGET}"
)

if(QWEN_VL_RKNN_ENABLE_RKNN)
    target_compile_definitions(qwen_vl_rknn_core PUBLIC QWEN_VL_RKNN_ENABLE_RKNN=1)
    target_include_directories(qwen_vl_rknn_core PUBLIC "${RKNN_INCLUDE_DIR}")
    target_link_libraries(qwen_vl_rknn_core PUBLIC "${RKNN_RUNTIME_LIB}")
endif()

if(QWEN_VL_RKNN_BUILD_CLI)
    add_executable(qwen-vl-rknn
        cpp/src/main.cc
    )
    target_link_libraries(qwen-vl-rknn PRIVATE qwen-vl-rknn::core)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(qwen-vl-rknn-tests
        cpp/tests/qwen_vl_rknn_test.cc
    )
    target_link_libraries(qwen-vl-rknn-tests PRIVATE qwen-vl-rknn::core)
    add_test(NAME qwen-vl-rknn COMMAND qwen-vl-rknn-tests)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Generator: ${CMAKE_GENERATOR} (multi-config: ${is_multi_config})")
message(STATUS "Rockchip target: ${QWEN_VL_RKNN_TARGET}")
message(STATUS "RKNN integration enabled: ${QWEN_VL_RKNN_ENABLE_RKNN}")
