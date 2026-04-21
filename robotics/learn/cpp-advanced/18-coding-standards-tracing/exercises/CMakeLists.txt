cmake_minimum_required(VERSION 3.16)
project(coding_standards_tracing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Debug by default — exercises rely on assert()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Warnings: strict
add_compile_options(-Wall -Wextra -Wpedantic -Werror
                    -Wconversion -Wsign-conversion
                    -Wdouble-promotion -Wformat=2
                    -Wnull-dereference -Wold-style-cast
                    -Wshadow -Wunused)

# --- Exercises ---
add_executable(ex01_coding_standards ex01_coding_standards.cpp)
add_executable(ex02_static_analysis_traps ex02_static_analysis_traps.cpp)
add_executable(ex03_cache_profiling ex03_cache_profiling.cpp)
add_executable(ex04_callgraph_hotspot ex04_callgraph_hotspot.cpp)
add_executable(ex05_syscall_reduction ex05_syscall_reduction.cpp)
add_executable(ex06_tracepoint_framework ex06_tracepoint_framework.cpp)
target_link_libraries(ex06_tracepoint_framework PRIVATE pthread)
add_executable(ex07_watchpoint_corruption ex07_watchpoint_corruption.cpp)
add_executable(ex08_flamegraph_optimization ex08_flamegraph_optimization.cpp)

# --- Puzzles ---
add_executable(puzzle01_false_sharing ../puzzles/puzzle01_false_sharing.cpp)
target_link_libraries(puzzle01_false_sharing PRIVATE pthread)
add_executable(puzzle02_invisible_allocation ../puzzles/puzzle02_invisible_allocation.cpp)
add_executable(puzzle03_lying_benchmark ../puzzles/puzzle03_lying_benchmark.cpp)
