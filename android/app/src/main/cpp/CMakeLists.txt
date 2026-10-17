cmake_minimum_required(VERSION 3.28)
project(vlm-rknn-android)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

set(REPO_ROOT "${CMAKE_SOURCE_DIR}/../../../../..")

if(NOT DEFINED RKLLM_INCLUDE_DIR)
  set(RKLLM_INCLUDE_DIR "${REPO_ROOT}/thirdparty/rkllm/include")
endif()

if(NOT DEFINED RKLLM_RUNTIME_LIB)
  set(RKLLM_RUNTIME_LIB "${REPO_ROOT}/thirdparty/rkllm/Android/${ANDROID_ABI}/librkllmrt.so")
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
  set(RKNN_RUNTIME_LIB "${REPO_ROOT}/thirdparty/rknpu2/Android/${ANDROID_ABI}/librknnrt.so")
endif()

if(NOT EXISTS "${RKNN_RUNTIME_LIB}")
  message(FATAL_ERROR
    "Could not find librknnrt.so at ${RKNN_RUNTIME_LIB}. "
    "Set RKNN_RUNTIME_LIB or rknnRuntimeLib to the Android RKNN runtime.")
endif()

include("${REPO_ROOT}/cmake/OpenCV.cmake")

set(SPM_ENABLE_SHARED OFF CACHE BOOL INTERNAL)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL INTERNAL)
FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece
    GIT_TAG 11051e3b73b3a6222a52acd720e39805dc7545ab
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(sentencepiece)

target_compile_options(sentencepiece-static PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-Wno-stringop-overflow>
)

include_directories(
    "${REPO_ROOT}/cpp/src"
    "${RKLLM_INCLUDE_DIR}"
    "${RKNN_INCLUDE_DIR}"
    "${sentencepiece_SOURCE_DIR}/src"
    ${VLM_RKNN_OPENCV_INCLUDE_DIRS})

add_library(vlm-rknn-jni SHARED
    rknn_jni.cpp
    "${REPO_ROOT}/cpp/src/ini.cc"
    "${REPO_ROOT}/cpp/src/logger.cc"
    "${REPO_ROOT}/cpp/src/vlm_rknn.cc"
    "${REPO_ROOT}/cpp/src/rknn_utils.cc")

target_compile_definitions(vlm-rknn-jni PRIVATE
    VLM_RKNN_TARGET="rk3588")

target_link_libraries(vlm-rknn-jni
    "${RKLLM_RUNTIME_LIB}"
    "${RKNN_RUNTIME_LIB}"
    sentencepiece
    ${VLM_RKNN_OPENCV_TARGETS}
    log)
