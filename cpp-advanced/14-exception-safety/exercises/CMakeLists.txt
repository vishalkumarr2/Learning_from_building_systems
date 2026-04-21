cmake_minimum_required(VERSION 3.16)
project(module14_exception_safety LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

# GCC 9 compatibility
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10")
    add_compile_options(-fconcepts)
endif()

add_executable(ex01_guarantee_levels ex01_guarantee_levels.cpp)
add_executable(ex02_copy_and_swap ex02_copy_and_swap.cpp)
add_executable(ex03_scope_guard ex03_scope_guard.cpp)
add_executable(ex04_noexcept_impact ex04_noexcept_impact.cpp)
add_executable(ex05_exception_safe_container ex05_exception_safe_container.cpp)
