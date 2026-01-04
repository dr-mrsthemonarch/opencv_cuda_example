cmake_minimum_required(VERSION 3.5)

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif ()
project(cudathing LANGUAGES CXX CUDA)

# --------------------------------------------------
# CUDA
# --------------------------------------------------
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 120 CACHE STRING "" FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ../executable_debug)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ../executable_release)

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

find_package(OpenMP REQUIRED)

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
set(CUDA_SOURCES
        cudamain.cu
)
set(CUDA_CPU_SOURCES
        main.cpp
        cpp_benchmark.cpp
        gpu_benchmark.cu
)


add_executable(opencvbenchmark ${OPENCV_SOURCES})

add_executable(cudabenchmark ${CUDA_SOURCES})

add_executable(cuda_cpu  ${CUDA_CPU_SOURCES})

add_executable(cuda_sanity cuda_sanity.cu)

set_target_properties(cudabenchmark PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(opencvbenchmark PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(cuda_cpu PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
set_target_properties(cuda_sanity PROPERTIES
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

if(OpenMP_CXX_FOUND)
    target_link_libraries(cuda_cpu PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(cuda_cpu PRIVATE
        CUDA::cudart
)

target_include_directories(opencvbenchmark PRIVATE ${OPENCV_INCLUDE_DIRS})
target_include_directories(cudabenchmark PRIVATE ${OPENCV_INCLUDE_DIRS})


if(MSVC)
    # CPU-only files optimization flags (NOT passed to NVCC)
    target_compile_options(cuda_cpu PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/O2>
            $<$<COMPILE_LANGUAGE:CXX>:/Oi>
            $<$<COMPILE_LANGUAGE:CXX>:/Ot>
            $<$<COMPILE_LANGUAGE:CXX>:/GL>
            $<$<COMPILE_LANGUAGE:CXX>:/fp:fast>
            $<$<COMPILE_LANGUAGE:CXX>:/arch:AVX2>
            $<$<COMPILE_LANGUAGE:CXX>:/openmp:llvm>
    )

    # Linker optimizations (only for final linking)
    target_link_options(cuda_cpu PRIVATE
            $<$<CONFIG:Release>:/LTCG>
            $<$<CONFIG:Release>:/OPT:REF>
            $<$<CONFIG:Release>:/OPT:ICF>
    )

    # Definitions
    target_compile_definitions(cuda_cpu PRIVATE
            _USE_MATH_DEFINES
            NOMINMAX
    )

    message(STATUS "Using MSVC with AVX2 optimizations for Ryzen 3800X")

elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Clang-cl
    target_compile_options(cuda_cpu PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-O3>
            $<$<COMPILE_LANGUAGE:CXX>:-march=znver2>
            $<$<COMPILE_LANGUAGE:CXX>:-mtune=znver2>
            $<$<COMPILE_LANGUAGE:CXX>:-ffast-math>
            $<$<COMPILE_LANGUAGE:CXX>:-mavx2>
            $<$<COMPILE_LANGUAGE:CXX>:-mfma>
    )
    message(STATUS "Using Clang-cl with Zen 2 optimizations")
endif()

## Separate CPU source files from CUDA source files
#set_source_files_properties(
#        main.cpp
#        cpp_benchmark.cpp
#        PROPERTIES
#        LANGUAGE CXX
#)
#
#set_source_files_properties(
#        gpu_benchmark.cuf
#        PROPERTIES
#        LANGUAGE CUDA
#)

# Build configuration
set(CMAKE_CXX_FLAGS_RELEASE "/MD /DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "/MDd /Zi /Ob0 /Od /RTC1")

message(STATUS "=== Build Configuration ===")
message(STATUS "CUDA Architecture: 121 (RTX 5070Ti)")
message(STATUS "CPU Optimization: AVX2, OpenMP LLVM")
message(STATUS "CUDA Optimization: --use_fast_math, -O3")
message(STATUS "==========================")

