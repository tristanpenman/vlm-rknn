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

# Avoid language extensions to ensure consistent behavior across compilers
set(CMAKE_CXX_EXTENSIONS OFF)

# FetchContent: Easier than Git submodules
include(FetchContent)

# Default to Debug for single-config generators if user didn't specify anything
get_property(is_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(NOT is_multi_config AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# -----------------------------------------------------------------------------
# build config
# -----------------------------------------------------------------------------

option(QWEN_VL_RKNN_ENABLE_OPENCV "Fetch and link OpenCV as a third-party dependency" ON)

set(QWEN_VL_RKNN_TARGET "rk3588" CACHE STRING "Rockchip target device family")

if(ANDROID)
    set(QWEN_VL_RKNN_ARCH ${CMAKE_ANDROID_ARCH_ABI})
    set(DEFAULT_RKNN_RUNTIME_LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rknpu2/lib-android/${QWEN_VL_RKNN_ARCH}/librknnrt.so)
    set(DEFAULT_RKLLM_RUNTIME_LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rkllm/lib-android/${QWEN_VL_RKNN_ARCH}/librkllmrt.so)
else()
    set(QWEN_VL_RKNN_ARCH aarch64)
    set(DEFAULT_RKNN_RUNTIME_LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rknpu2/lib/${QWEN_VL_RKNN_ARCH}/librknnrt.so)
    set(DEFAULT_RKLLM_RUNTIME_LIB
        ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rkllm/lib/${QWEN_VL_RKNN_ARCH}/librkllmrt.so)
endif()

# -----------------------------------------------------------------------------
# rknn runtime
# -----------------------------------------------------------------------------

set(RKNN_RUNTIME_LIB ${DEFAULT_RKNN_RUNTIME_LIB} CACHE FILEPATH
    "Path to librknnrt.so for the selected target")
set(RKNN_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rknpu2/include CACHE PATH
    "Path containing rknn_api.h")

if(NOT EXISTS ${RKNN_RUNTIME_LIB})
    message(FATAL_ERROR
        "RKNN runtime library not found: ${RKNN_RUNTIME_LIB}. "
        "Set -DRKNN_RUNTIME_LIB=<path-to-librknnrt.so> (for Android use arm64-v8a runtime).")
endif()

if(NOT EXISTS ${RKNN_INCLUDE_DIR}/rknn_api.h)
    message(FATAL_ERROR
        "RKNN headers not found in ${RKNN_INCLUDE_DIR}. "
        "Set -DRKNN_INCLUDE_DIR=<path-containing-rknn_api.h>.")
endif()

# -----------------------------------------------------------------------------
# rkllm runtime
# -----------------------------------------------------------------------------

set(RKLLM_RUNTIME_LIB ${DEFAULT_RKLLM_RUNTIME_LIB} CACHE FILEPATH
    "Path to librkllmrt.so for the selected target")
set(RKLLM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/rkllm/include CACHE PATH
    "Path containing rkllm.h")

if(NOT EXISTS ${RKLLM_RUNTIME_LIB})
    message(FATAL_ERROR
        "RKLLM runtime library not found: ${RKLLM_RUNTIME_LIB}. "
        "Set -DRKLLM_RUNTIME_LIB=<path-to-librkllmrt.so> (for Android use arm64-v8a runtime).")
endif()

if(NOT EXISTS ${RKLLM_INCLUDE_DIR}/rkllm.h)
    message(FATAL_ERROR
        "RKLLM headers not found in ${RKLLM_INCLUDE_DIR}. "
        "Set -DRKLLM_INCLUDE_DIR=<path-containing-rkllm.h>.")
endif()

# -----------------------------------------------------------------------------
# OpenCV
# -----------------------------------------------------------------------------

if(QWEN_VL_RKNN_ENABLE_OPENCV)
    include(cmake/thirdparty/OpenCV.cmake)
endif()

# -----------------------------------------------------------------------------
# qwen_vl_rknn_core
# -----------------------------------------------------------------------------

add_library(qwen_vl_rknn_core STATIC
    cpp/src/file_utils.cc
    cpp/src/logger.cc
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

target_include_directories(qwen_vl_rknn_core PUBLIC
    "${RKNN_INCLUDE_DIR}"
    "${RKLLM_INCLUDE_DIR}"
)

target_link_libraries(qwen_vl_rknn_core PUBLIC
    "${RKNN_RUNTIME_LIB}"
    "${RKLLM_RUNTIME_LIB}"
)

if(QWEN_VL_RKNN_ENABLE_OPENCV)
    target_link_libraries(qwen_vl_rknn_core PUBLIC ${QWEN_VL_RKNN_OPENCV_TARGETS})
    target_compile_definitions(qwen_vl_rknn_core PUBLIC QWEN_VL_RKNN_ENABLE_OPENCV=1)
endif()

# -----------------------------------------------------------------------------
# qwen-vl-rknn
# -----------------------------------------------------------------------------

add_executable(qwen-vl-rknn
    cpp/src/main.cc
)

# OpenCV's CMake sets CMAKE_RUNTIME_OUTPUT_DIRECTORY to <build>/bin
set_target_properties(qwen-vl-rknn PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

target_link_libraries(qwen-vl-rknn PRIVATE qwen-vl-rknn::core)

# Strip debug symbols from release builds
target_link_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Release>:-Wl,--strip-all>
)

# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

include(CTest)
if(BUILD_TESTING AND NOT ANDROID)
    add_executable(qwen-vl-rknn-tests
        cpp/tests/qwen_vl_rknn_test.cc
    )
    target_link_libraries(qwen-vl-rknn-tests PRIVATE qwen-vl-rknn::core)
    add_test(NAME qwen-vl-rknn COMMAND qwen-vl-rknn-tests)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Generator: ${CMAKE_GENERATOR} (multi-config: ${is_multi_config})")
message(STATUS "Rockchip target: ${QWEN_VL_RKNN_TARGET}")
message(STATUS "OpenCV enabled: ${QWEN_VL_RKNN_ENABLE_OPENCV}")
message(STATUS "RKNN runtime library: ${RKNN_RUNTIME_LIB}")
