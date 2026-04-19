cmake_minimum_required(VERSION 3.20)
project(week05_build_test_tooling
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "Week 5: Production Build Systems, Testing & Tooling"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─── Compiler Warnings ─────────────────────────────────────────
add_compile_options(
    -Wall -Wextra -Wpedantic -Wshadow
    -Wconversion -Wsign-conversion
    -Wnon-virtual-dtor -Wold-style-cast
    -Woverloaded-virtual -Wnull-dereference
    -Wformat=2
)

# ─── Threads ────────────────────────────────────────────────────
find_package(Threads REQUIRED)

# ─── GoogleTest via FetchContent ────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
    GIT_SHALLOW    TRUE
)
# Prevent GoogleTest from overriding our compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)
enable_testing()

# ─── Exercise 01: Full CMake Project ───────────────────────────
add_subdirectory(ex01_cmake_project)

# ─── Exercise 02: GTest Suite ──────────────────────────────────
add_executable(ex02_gtest_suite ex02_gtest_suite.cpp)
target_link_libraries(ex02_gtest_suite PRIVATE
    GTest::gtest
    GTest::gtest_main
    Threads::Threads
)
gtest_discover_tests(ex02_gtest_suite)

# ─── Exercise 03: Fuzzer Harness ───────────────────────────────
# Two build modes:
#   1. Normal test (always built):
add_executable(ex03_fuzzer_test ex03_fuzzer_harness.cpp)
target_compile_definitions(ex03_fuzzer_test PRIVATE STANDALONE_TEST)
target_link_libraries(ex03_fuzzer_test PRIVATE Threads::Threads)

#   2. libFuzzer mode (only with clang + fuzzer sanitizer):
option(ENABLE_FUZZING "Build with libFuzzer" OFF)
if(ENABLE_FUZZING)
    add_executable(ex03_fuzzer ex03_fuzzer_harness.cpp)
    target_compile_options(ex03_fuzzer PRIVATE -fsanitize=fuzzer,address)
    target_link_options(ex03_fuzzer PRIVATE -fsanitize=fuzzer,address)
endif()

# ─── Exercise 04: Profiling Guide ──────────────────────────────
add_executable(ex04_profiling_guide ex04_profiling_guide.cpp)
target_link_libraries(ex04_profiling_guide PRIVATE Threads::Threads)

# ─── Puzzle 01: Coverage Lie ────────────────────────────────────
add_executable(puzzle01_coverage_lie
    ${CMAKE_CURRENT_SOURCE_DIR}/../puzzles/puzzle01_coverage_lie.cpp)

# ─── Puzzle 02: ASan vs TSan ────────────────────────────────────
add_executable(puzzle02_asan_vs_tsan
    ${CMAKE_CURRENT_SOURCE_DIR}/../puzzles/puzzle02_asan_vs_tsan.cpp)
target_link_libraries(puzzle02_asan_vs_tsan PRIVATE Threads::Threads)
