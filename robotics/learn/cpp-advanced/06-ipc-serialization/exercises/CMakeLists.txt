cmake_minimum_required(VERSION 3.16)
project(week06_ipc_serialization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

# ── Exercises ────────────────────────────────────────────
add_executable(ex01 ex01_shared_memory.cpp)
target_link_libraries(ex01 PRIVATE rt pthread)

add_executable(ex02 ex02_serialization_bench.cpp)
target_link_libraries(ex02 PRIVATE rt pthread)

add_executable(ex03 ex03_type_erasure.cpp)
target_link_libraries(ex03 PRIVATE pthread)

add_executable(ex04 ex04_plugin_loader.cpp)
target_link_libraries(ex04 PRIVATE dl pthread)

add_executable(ex05 ex05_telemetry_pipeline.cpp)
target_link_libraries(ex05 PRIVATE rt pthread)

# ── Puzzles ──────────────────────────────────────────────
add_executable(puzzle01 ../puzzles/puzzle01_placement_new.cpp)
target_link_libraries(puzzle01 PRIVATE rt pthread)

add_executable(puzzle02 ../puzzles/puzzle02_dlopen_leak.cpp)
target_link_libraries(puzzle02 PRIVATE dl rt pthread)
