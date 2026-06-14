cmake_minimum_required(VERSION 3.22.1)
project(qwen-vl-rknn-android)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(REPO_ROOT "${CMAKE_SOURCE_DIR}/../../../..")

if(NOT DEFINED RKLLM_INCLUDE_DIR)
  set(RKLLM_INCLUDE_DIR "${REPO_ROOT}/thirdparty/rkllm/include")
endif()

if(NOT DEFINED RKLLM_RUNTIME_LIB)
  set(RKLLM_RUNTIME_LIB "${REPO_ROOT}/thirdparty/rkllm/lib-android/${ANDROID_ABI}/librkllmrt.so")
endif()

if(NOT EXISTS "${RKLLM_RUNTIME_LIB}")
  message(FATAL_ERROR
    "Could not find librkllmrt.so at ${RKLLM_RUNTIME_LIB}. "
    "Set RKLLM_RUNTIME_LIB or rkllmRuntimeLib to the Android RKLLM runtime.")
endif()

if(NOT DEFINED RKNN_INCLUDE_DIR)
  set(RKNN_INCLUDE_DIR "${REPO_ROOT}/thirdparty/rknpu2/include")
endif()

if(NOT DEFINED RKNN_RUNTIME_LIB)
  set(RKNN_RUNTIME_LIB "${REPO_ROOT}/thirdparty/rknpu2/lib-android/${ANDROID_ABI}/librknnrt.so")
endif()

if(NOT EXISTS "${RKNN_RUNTIME_LIB}")
  message(FATAL_ERROR
    "Could not find librknnrt.so at ${RKNN_RUNTIME_LIB}. "
    "Set RKNN_RUNTIME_LIB or rknnRuntimeLib to the Android RKNN runtime.")
endif()

include_directories(
    "${REPO_ROOT}/cpp/src"
    "${RKLLM_INCLUDE_DIR}"
    "${RKNN_INCLUDE_DIR}")

add_library(qwen-vl-rknn-jni SHARED
    rknn_jni.cpp
    "${REPO_ROOT}/cpp/src/logger.cc"
    "${REPO_ROOT}/cpp/src/qwen_vl_rknn.cc")

target_compile_definitions(qwen-vl-rknn-jni PRIVATE
    QWEN_VL_RKNN_TARGET="rk3588")

target_link_libraries(qwen-vl-rknn-jni
    "${RKLLM_RUNTIME_LIB}"
    "${RKNN_RUNTIME_LIB}"
    log)
