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
