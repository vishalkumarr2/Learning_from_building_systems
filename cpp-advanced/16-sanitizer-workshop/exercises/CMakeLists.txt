cmake_minimum_required(VERSION 3.16)
project(module16_sanitizer_workshop LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic)

# GCC 9 compatibility
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10")
    add_compile_options(-fconcepts)
endif()

# Default build: no sanitizers (for baseline comparison)
add_executable(ex01_asan_bugs ex01_asan_bugs.cpp)
add_executable(ex02_ubsan_bugs ex02_ubsan_bugs.cpp)
add_executable(ex03_tsan_bugs ex03_tsan_bugs.cpp)
target_link_libraries(ex03_tsan_bugs PRIVATE pthread)
add_executable(ex04_sanitizer_ci ex04_sanitizer_ci.cpp)
target_link_libraries(ex04_sanitizer_ci PRIVATE pthread)
add_executable(ex05_sanitizer_detective ex05_sanitizer_detective.cpp)
target_link_libraries(ex05_sanitizer_detective PRIVATE pthread)

# Puzzles (no sanitizers for default build)
add_executable(puzzle01_false_positive ../puzzles/puzzle01_false_positive.cpp)
target_link_libraries(puzzle01_false_positive PRIVATE pthread)
add_executable(puzzle02_sanitizer_blind_spots ../puzzles/puzzle02_sanitizer_blind_spots.cpp)
target_link_libraries(puzzle02_sanitizer_blind_spots PRIVATE pthread)

# ============================================================
# ASan targets
# ============================================================
add_executable(ex01_asan_bugs_asan ex01_asan_bugs.cpp)
target_compile_options(ex01_asan_bugs_asan PRIVATE -fsanitize=address -fno-omit-frame-pointer)
target_link_options(ex01_asan_bugs_asan PRIVATE -fsanitize=address)

add_executable(ex04_ci_asan ex04_sanitizer_ci.cpp)
target_compile_options(ex04_ci_asan PRIVATE -fsanitize=address -fno-omit-frame-pointer)
target_link_options(ex04_ci_asan PRIVATE -fsanitize=address)
target_link_libraries(ex04_ci_asan PRIVATE pthread)

add_executable(ex05_detective_asan ex05_sanitizer_detective.cpp)
target_compile_options(ex05_detective_asan PRIVATE -fsanitize=address -fno-omit-frame-pointer)
target_link_options(ex05_detective_asan PRIVATE -fsanitize=address)
target_link_libraries(ex05_detective_asan PRIVATE pthread)

# ============================================================
# UBSan targets
# ============================================================
add_executable(ex02_ubsan_bugs_ubsan ex02_ubsan_bugs.cpp)
target_compile_options(ex02_ubsan_bugs_ubsan PRIVATE -fsanitize=undefined)
target_link_options(ex02_ubsan_bugs_ubsan PRIVATE -fsanitize=undefined)

add_executable(ex04_ci_ubsan ex04_sanitizer_ci.cpp)
target_compile_options(ex04_ci_ubsan PRIVATE -fsanitize=undefined)
target_link_options(ex04_ci_ubsan PRIVATE -fsanitize=undefined)
target_link_libraries(ex04_ci_ubsan PRIVATE pthread)

add_executable(ex05_detective_ubsan ex05_sanitizer_detective.cpp)
target_compile_options(ex05_detective_ubsan PRIVATE -fsanitize=undefined)
target_link_options(ex05_detective_ubsan PRIVATE -fsanitize=undefined)
target_link_libraries(ex05_detective_ubsan PRIVATE pthread)

# ============================================================
# TSan targets (CANNOT combine with ASan!)
# ============================================================
add_executable(ex03_tsan_bugs_tsan ex03_tsan_bugs.cpp)
target_compile_options(ex03_tsan_bugs_tsan PRIVATE -fsanitize=thread)
target_link_options(ex03_tsan_bugs_tsan PRIVATE -fsanitize=thread)
target_link_libraries(ex03_tsan_bugs_tsan PRIVATE pthread)

add_executable(ex04_ci_tsan ex04_sanitizer_ci.cpp)
target_compile_options(ex04_ci_tsan PRIVATE -fsanitize=thread)
target_link_options(ex04_ci_tsan PRIVATE -fsanitize=thread)
target_link_libraries(ex04_ci_tsan PRIVATE pthread)

add_executable(ex05_detective_tsan ex05_sanitizer_detective.cpp)
target_compile_options(ex05_detective_tsan PRIVATE -fsanitize=thread)
target_link_options(ex05_detective_tsan PRIVATE -fsanitize=thread)
target_link_libraries(ex05_detective_tsan PRIVATE pthread)

# TSan puzzle targets
add_executable(puzzle01_false_positive_tsan ../puzzles/puzzle01_false_positive.cpp)
target_compile_options(puzzle01_false_positive_tsan PRIVATE -fsanitize=thread)
target_link_options(puzzle01_false_positive_tsan PRIVATE -fsanitize=thread)
target_link_libraries(puzzle01_false_positive_tsan PRIVATE pthread)
