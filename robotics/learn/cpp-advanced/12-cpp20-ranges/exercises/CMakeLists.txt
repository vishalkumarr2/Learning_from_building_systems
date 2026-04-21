cmake_minimum_required(VERSION 3.16)
project(module12_ranges LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

# Ranges require GCC 10+ (partial) or GCC 12+ (full) or Clang 14+
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10")
        message(WARNING "GCC ${CMAKE_CXX_COMPILER_VERSION} does not support ranges. Need GCC 10+.")
        return()
    endif()
endif()

add_executable(ex01_range_basics ex01_range_basics.cpp)
add_executable(ex02_projections ex02_projections.cpp)
add_executable(ex03_custom_view ex03_custom_view.cpp)
add_executable(ex04_range_algorithms ex04_range_algorithms.cpp)
add_executable(ex05_lazy_pipeline ex05_lazy_pipeline.cpp)
