cmake_minimum_required(VERSION 3.5)

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif ()
project(cudathing LANGUAGES CXX CUDA)

# --------------------------------------------------
# CUDA
# --------------------------------------------------
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 121 CACHE STRING "" FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ../executable_debug)

# --------------------------------------------------
# FetchContent
# --------------------------------------------------
include(FetchContent)

# Declare opencv_contrib FIRST (opencv needs it)
FetchContent_Declare(
        opencv_contrib
        GIT_REPOSITORY https://github.com/opencv/opencv_contrib.git
        GIT_TAG 4.13.0
)



# --------------------------------------------------
# Make opencv_contrib available BEFORE opencv
# --------------------------------------------------
FetchContent_MakeAvailable(opencv_contrib)

# --------------------------------------------------
# OpenCV configuration (MUST be CACHE vars!)
# --------------------------------------------------
set(OPENCV_EXTRA_MODULES_PATH
        "${opencv_contrib_SOURCE_DIR}/modules"
        CACHE PATH "" FORCE
)

set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(CUDNN_INCLUDE_DIR "C:/Program Files/NVIDIA/CUDNN/v9.17/include/13.1" CACHE PATH "" FORCE)
set(CUDNN_LIBRARY "C:/Program Files/NVIDIA/CUDNN/v9.17/lib/13.1/x64/cudnn.lib" CACHE FILEPATH "" FORCE)

#set(BUILD_opencv_world ON CACHE BOOL "" FORCE)
set(BUILD_opencv_cudacodec ON CACHE BOOL "" FORCE)
set(WITH_IPP OFF CACHE BOOL "" FORCE)
set(WITH_CUDA ON CACHE BOOL "" FORCE)
set(WITH_CUBLAS ON CACHE BOOL "" FORCE)
set(WITH_CUDNN ON CACHE BOOL "" FORCE)  # Unless you actually have cuDNN
set(WITH_EIGEN ON CACHE BOOL "" FORCE)
set(CUDA_FAST_MATH ON CACHE BOOL "" FORCE)
set(ENABLE_FAST_MATH ON CACHE BOOL "" FORCE)


set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(WITH_OPENBLAS OFF CACHE BOOL "" FORCE)
set(WITH_LAPACK OFF CACHE BOOL "" FORCE)
set(OPENCV_ENABLE_NONFREE OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_apps OFF CACHE BOOL "" FORCE)
set(BUILD_ANDROID_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_ANDROID_PROJECTS OFF CACHE BOOL "" FORCE)

set(WITH_NVCUVID OFF CACHE BOOL "" FORCE)
set(WITH_NVCUVENC OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_python2 OFF CACHE BOOL "" FORCE)
set(BUILD_opencv_python3 OFF CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
        opencv
        GIT_REPOSITORY https://github.com/opencv/opencv.git
        GIT_TAG 4.13.0
        GIT_SHALLOW TRUE
        GIT_PROGRESS TRUE
)
# --------------------------------------------------
# Build OpenCV
# --------------------------------------------------
FetchContent_MakeAvailable(opencv)

    #macro to find opencv modules
macro(collect_include_dirs result curdir)
        file(GLOB children ${curdir}/*)
        set(dirlist "")
        foreach(child ${children})
            if(IS_DIRECTORY ${child})
                list(APPEND dirlist ${child}/include)
            endif()
        endforeach()
        set(${result} ${dirlist})
endmacro()


collect_include_dirs(OPENCV_INCLUDE_DIRS ${OpenCV_SOURCE_DIR}/modules)
list(APPEND OPENCV_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR} ${OpenCV_SOURCE_DIR}/include)
# --------------------------------------------------
# Your CUDA target
# --------------------------------------------------
set(OPENCV_SOURCES
        opencv_main.cu
)
set(CUDA_SOURCES
cudamain.cu
)


add_executable(opencvbenchmark ${OPENCV_SOURCES}
        opencv_main.cu)

add_executable(cudabenchmark ${CUDA_SOURCES}
        cudamain.cu)

set_target_properties(cudabenchmark PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(opencvbenchmark PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)

target_link_libraries(opencvbenchmark PRIVATE
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        opencv_highgui
        opencv_dnn
        opencv_cudaarithm
        opencv_cudaimgproc
        CUDA::cudart
)
target_link_libraries(cudabenchmark PRIVATE
        CUDA::cudart
)

target_include_directories(opencvbenchmark PRIVATE ${OPENCV_INCLUDE_DIRS})
target_include_directories(cudabenchmark PRIVATE ${OPENCV_INCLUDE_DIRS})