cmake_minimum_required(VERSION 3.16)

#--------Auto-detect architecture-----------
# CMAKE_SYSTEM_PROCESSOR may be empty before project() - fallback to uname
if(NOT CMAKE_SYSTEM_PROCESSOR)
    execute_process(COMMAND uname -m OUTPUT_VARIABLE CMAKE_SYSTEM_PROCESSOR OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()
message(STATUS "System Processor: ${CMAKE_SYSTEM_PROCESSOR}")

# set CUDA compile path (architecture-adaptive)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|arm64")
    # Jetson: CUDA typically at /usr/local/cuda-11.8
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.8/bin/nvcc)
else()
    # x86_64: auto-detect from PATH or common install locations
    find_program(NVCC_EXECUTABLE nvcc
        PATHS /usr/local/cuda/bin
              /usr/local/cuda-11.8/bin
    )
    if(NVCC_EXECUTABLE)
        set(CMAKE_CUDA_COMPILER ${NVCC_EXECUTABLE})
    else()
        message(FATAL_ERROR "nvcc not found! Please install CUDA or set CMAKE_CUDA_COMPILER manually.")
    endif()
endif()

project(yolov11 LANGUAGES CXX CUDA)

#--------- compile standard-----------
set(CMAKE_CXX_STANDARD 17)
# CUDA17 requires CMake 3.17+; use 14 for CUDA device code, C++17 still applies to host code
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# -pthread only for C++ host code; nvcc does not understand it
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

#-----------OpenCV-----------
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#-----------CUDA-----------
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

#--------TensorRT (auto-detect architecture)-----------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|arm64")
    # Jetson: system-installed TensorRT
    include_directories(/usr/local/include/aarch64-linux-gnu)
    link_directories(/usr/local/lib/aarch64-linux-gnu)
    message(STATUS "TensorRT: Jetson arm|aarch64|arm64 paths")
elseif(DEFINED ENV{TENSORRT_DIR})
    # x86_64: TENSORRT_DIR from environment (e.g. /home/xcmg/TensorRT-8.6.1.6)
    set(TENSORRT_DIR $ENV{TENSORRT_DIR})
    include_directories(${TENSORRT_DIR}/include)
    link_directories(${TENSORRT_DIR}/lib)
    message(STATUS "TensorRT: x86_64 from $ENV{TENSORRT_DIR}")
else()
    message(FATAL_ERROR "TensorRT not found! Set TENSORRT_DIR env var or install system-wide.")
endif()

#-----------project inlcude-----------
include_directories(${PROJECT_SOURCE_DIR}/include)
#----------project source-----------
file(GLOB_RECURSE CPP_FILES src/*.cc)
file(GLOB_RECURSE CUDA_FILES src/*.cu)

# ----------build executable----------
add_executable(yolov11 ${CPP_FILES} ${CUDA_FILES})

#----------set CUDA property----------
set_property(SOURCE ${CUDA_FILES} PROPERTY LANGUAGE CUDA)

#----------add cuda options----------
target_compile_options(yolov11 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>    
)

# ---------GPU architecture (auto-detect)----------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|arm64")
    # Jetson Xavier / Orin: Volta(7.2) / Ampere(8.7)
    set(CUDA_ARCH_LIST "72;87")
else()
    # x86_64: RTX 4070 = Ada Lovelace(89), RTX 30xx = Ampere(86)
    set(CUDA_ARCH_LIST "86;89")
endif()

# CUDA_ARCHITECTURES auto-generates -gencode flags, no need for manual --gpu-architecture
set_target_properties(yolov11 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

#----------link libraries----------
target_link_libraries(yolov11 PRIVATE
    ${OpenCV_LIBS}
    nvinfer
    nvonnxparser
    nvinfer_plugin
    cudart
    pthread
)

#----------add CUDA define---------
target_compile_definitions(yolov11 PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:__CUDACC__>
)

#----------GPU-specific optimization flags (use compile_OPTIONS, not definitions)--------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|arm64")
    # Jetson: limit register usage for power efficiency
    target_compile_options(yolov11 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--ftz=true>
        $<$<COMPILE_LANGUAGE:CUDA>:--prec-div=false>
        $<$<COMPILE_LANGUAGE:CUDA>:--prec-sqrt=false>
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>
        $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=32>
    )
else()
    # x86_64: full register file, optimize for throughput
    target_compile_options(yolov11 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--ftz=true>
        $<$<COMPILE_LANGUAGE:CUDA>:--prec-div=false>
        $<$<COMPILE_LANGUAGE:CUDA>:--prec-sqrt=false>
        $<$<COMPILE_LANGUAGE:CUDA>:--fmad=true>
    )
endif()

#-----------make sure CUDA file compile correctly----------
set_source_files_properties(${CUDA_FILES} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

#---------add option profile---------
option(ENABLE_CUDA_PROFILING "Enable CUDA profiling" OFF)
if (ENABLE_CUDA_PROFILING)
    target_compile_definitions(yolov11 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:ENABLE_PROFILING>
    )
endif()
