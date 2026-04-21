cmake_minimum_required(VERSION 3.16)
project(module15_pmr_allocators LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

# GCC 9 compatibility
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10")
    add_compile_options(-fconcepts)
endif()

add_executable(ex01_monotonic_buffer ex01_monotonic_buffer.cpp)
add_executable(ex02_pool_resource ex02_pool_resource.cpp)
add_executable(ex03_custom_memory_resource ex03_custom_memory_resource.cpp)
add_executable(ex04_realtime_arena ex04_realtime_arena.cpp)
add_executable(ex05_pmr_containers ex05_pmr_containers.cpp)

# Link pthreads for ex02 (synchronized_pool_resource)
find_package(Threads REQUIRED)
target_link_libraries(ex02_pool_resource Threads::Threads)
target_link_libraries(ex04_realtime_arena Threads::Threads)
