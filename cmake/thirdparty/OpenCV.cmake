# OpenCV third-party dependency configuration.
#
# This intentionally keeps the fetched OpenCV build small for the first
# integration step. Add modules to QWEN_VL_RKNN_OPENCV_MODULES as the runtime
# starts using more image/video functionality.

set(QWEN_VL_RKNN_OPENCV_GIT_TAG "4.13.0" CACHE STRING
    "OpenCV git tag, branch, or commit fetched when QWEN_VL_RKNN_ENABLE_OPENCV is ON")
set(QWEN_VL_RKNN_OPENCV_MODULES "core,imgproc,imgcodecs" CACHE STRING
    "Comma-separated OpenCV modules to build via FetchContent")

# Keep OpenCV focused on embeddable C++ libraries for Rockchip Linux/Android
# targets. These cache entries are consumed by OpenCV's own CMake project.
set(BUILD_LIST "${QWEN_VL_RKNN_OPENCV_MODULES}" CACHE STRING "OpenCV modules to build" FORCE)
set(BUILD_opencv_apps OFF CACHE BOOL "Build OpenCV command-line applications" FORCE)
set(BUILD_opencv_js OFF CACHE BOOL "Build OpenCV.js bindings" FORCE)
set(BUILD_JAVA OFF CACHE BOOL "Build OpenCV Java bindings" FORCE)
set(BUILD_opencv_java OFF CACHE BOOL "Build OpenCV Java bindings" FORCE)
set(BUILD_opencv_python2 OFF CACHE BOOL "Build OpenCV Python 2 bindings" FORCE)
set(BUILD_opencv_python3 OFF CACHE BOOL "Build OpenCV Python 3 bindings" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "Build OpenCV tests" FORCE)
set(BUILD_PERF_TESTS OFF CACHE BOOL "Build OpenCV performance tests" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Build OpenCV examples" FORCE)
set(BUILD_DOCS OFF CACHE BOOL "Build OpenCV documentation" FORCE)
set(OPENCV_ENABLE_NONFREE OFF CACHE BOOL "Enable OpenCV non-free algorithms" FORCE)

# Avoid probing or linking optional desktop/media stacks that are not needed for
# the initial embedded image-preprocessing dependency.
set(WITH_FFMPEG OFF CACHE BOOL "Use FFmpeg" FORCE)
set(WITH_GSTREAMER OFF CACHE BOOL "Use GStreamer" FORCE)
set(WITH_GTK OFF CACHE BOOL "Use GTK" FORCE)
set(WITH_QT OFF CACHE BOOL "Use Qt" FORCE)
set(WITH_OPENEXR OFF CACHE BOOL "Use OpenEXR" FORCE)
set(WITH_IPP OFF CACHE BOOL "Use Intel IPP" FORCE)
set(WITH_ITT OFF CACHE BOOL "Use Intel ITT tracing" FORCE)
set(WITH_OPENCL OFF CACHE BOOL "Use OpenCL" FORCE)

FetchContent_Declare(
    opencv
    GIT_REPOSITORY https://github.com/opencv/opencv.git
    GIT_TAG ${QWEN_VL_RKNN_OPENCV_GIT_TAG}
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)

FetchContent_MakeAvailable(opencv)

set(QWEN_VL_RKNN_OPENCV_TARGETS
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
)
