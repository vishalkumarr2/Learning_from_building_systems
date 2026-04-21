cmake_minimum_required(VERSION 3.16)
project(module13_format LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

# std::format requires GCC 13+ or Clang 17+
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "13")
        message(WARNING "GCC ${CMAKE_CXX_COMPILER_VERSION} does not fully support <format>. Need GCC 13+.")
        return()
    endif()
endif()
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "17")
        message(WARNING "Clang ${CMAKE_CXX_COMPILER_VERSION} does not fully support <format>. Need Clang 17+.")
        return()
    endif()
endif()

add_executable(ex01_format_basics ex01_format_basics.cpp)
add_executable(ex02_custom_formatter ex02_custom_formatter.cpp)
add_executable(ex03_format_benchmark ex03_format_benchmark.cpp)
add_executable(ex04_safe_logging ex04_safe_logging.cpp)
target_link_libraries(ex04_safe_logging PRIVATE pthread)
