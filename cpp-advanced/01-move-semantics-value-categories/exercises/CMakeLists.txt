cmake_minimum_required(VERSION 3.16)
project(week1_exercises LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Strict warnings
add_compile_options(-Wall -Wextra -Wpedantic)

# GCC 9 needs -fconcepts for Concepts TS support
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10")
    add_compile_options(-fconcepts)
endif()

# Sanitizers for debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR NOT CMAKE_BUILD_TYPE)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

# --- Exercises (ex01-03 work with GCC 9 C++20) ---
add_executable(ex01_value_categories ex01_value_categories.cpp)
add_executable(ex02_rule_of_five ex02_rule_of_five.cpp)
add_executable(ex03_perfect_forwarding ex03_perfect_forwarding.cpp)
add_executable(ex04_constexpr_crc32 ex04_constexpr_crc32.cpp)
add_executable(ex05_concepts ex05_concepts.cpp)
add_executable(ex06_message_descriptor ex06_message_descriptor.cpp)
