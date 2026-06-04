cmake_minimum_required(VERSION 3.16)

#--------Auto-detect architecture-----------
if(NOT CMAKE_SYSTEM_PROCESSOR)
    execute_process(COMMAND uname -m OUTPUT_VARIABLE CMAKE_SYSTEM_PROCESSOR OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
message(STATUS "System Processor: ${CMAKE_SYSTEM_PROCESSOR}")

project(calibrator_int8 LANGUAGES CXX)

#--------- compile standard-----------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

#-----------OpenCV-----------
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#-----------CUDA (runtime only, no .cu files)-----------
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

#--------TensorRT (auto-detect architecture)-----------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|arm64")
    # Jetson: system-installed TensorRT
    include_directories(/usr/local/include/aarch64-linux-gnu)
    link_directories(/usr/local/lib/aarch64-linux-gnu)
    message(STATUS "TensorRT: Jetson aarch64 paths")
elseif(DEFINED ENV{TENSORRT_DIR})
    # x86_64: TENSORRT_DIR from environment
    set(TENSORRT_DIR $ENV{TENSORRT_DIR})
    include_directories(${TENSORRT_DIR}/include)
    link_directories(${TENSORRT_DIR}/lib)
    message(STATUS "TensorRT: x86_64 from $ENV{TENSORRT_DIR}")
else()
    message(FATAL_ERROR "TensorRT not found! Set TENSORRT_DIR env var or install system-wide.")
endif()

#-----------project include-----------
include_directories(${PROJECT_SOURCE_DIR}/include)

#----------project source-----------
file(GLOB_RECURSE CPP_FILES src/*.cc)

#----------build executable----------
add_executable(calibrator_int8 ${CPP_FILES})

#----------link libraries----------
target_link_libraries(calibrator_int8 PRIVATE
    ${OpenCV_LIBS}
    nvinfer
    nvonnxparser
    nvinfer_plugin
    cudart
    pthread
)
