cmake_minimum_required(VERSION 3.20)
project(peleng_core CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG)

if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmake
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    find_package(pybind11 REQUIRED PATHS "${pybind11_DIR}")
endif()

pybind11_add_module(peleng_core
    src/peleng_core/bindings.cpp
)

target_include_directories(peleng_core PRIVATE src/peleng_core)

if(DEFINED ENV{NVDEC_SDK_PATH})
    set(NVDEC_SDK_DIR "$ENV{NVDEC_SDK_PATH}")
else()
    set(NVDEC_SDK_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/Video_Codec_SDK_13.1.24")
endif()

if(DEFINED ENV{CUDA_PATH})
    set(CUDA_TOOLKIT_DIR "$ENV{CUDA_PATH}")
else()
    set(CUDA_TOOLKIT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4")
endif()

if(EXISTS "${NVDEC_SDK_DIR}/Interface/cuviddec.h" AND EXISTS "${CUDA_TOOLKIT_DIR}/include/cuda.h")
    message(STATUS "Found NVIDIA Video Codec SDK 13.1.24 at ${NVDEC_SDK_DIR}")
    target_compile_definitions(peleng_core PRIVATE PELENG_HAS_NVDEC_SDK=1)
    target_include_directories(peleng_core PRIVATE
        "${NVDEC_SDK_DIR}/Interface"
        "${CUDA_TOOLKIT_DIR}/include"
    )
    target_link_directories(peleng_core PRIVATE
        "${NVDEC_SDK_DIR}/Lib/win/x64"
        "${CUDA_TOOLKIT_DIR}/lib/x64"
    )
    target_link_libraries(peleng_core PRIVATE nvcuvid nvencodeapi cuda)
endif()
