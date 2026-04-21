cmake_minimum_required(VERSION 3.16)
project(module11_coroutines LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

# Coroutines require GCC 10+ with -fcoroutines flag, or Clang 14+
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10")
        message(WARNING "GCC ${CMAKE_CXX_COMPILER_VERSION} does not support coroutines. Need GCC 10+.")
        return()
    endif()
    add_compile_options(-fcoroutines)
endif()

add_executable(ex01_generator ex01_generator.cpp)
add_executable(ex02_async_task ex02_async_task.cpp)
add_executable(ex03_coroutine_scheduler ex03_coroutine_scheduler.cpp)
add_executable(ex04_symmetric_transfer ex04_symmetric_transfer.cpp)
