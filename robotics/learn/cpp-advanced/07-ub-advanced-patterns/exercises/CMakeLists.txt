cmake_minimum_required(VERSION 3.16)
project(week07_ub_advanced_patterns LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Strict warnings
add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wnon-virtual-dtor)

find_package(Threads REQUIRED)

# --- Exercise list ---
set(EXERCISES
    ex01_ub_catalog
    ex02_crtp
    ex03_policy_design
    ex04_structured_logging
    ex05_ub_detector_challenge
)

# --- Puzzles ---
set(PUZZLES
    puzzle01_compiler_exploits_ub
    puzzle02_temporary_lifetime
)

# Regular (no-sanitizer) targets
foreach(ex IN LISTS EXERCISES)
    add_executable(${ex} ${ex}.cpp)
    target_link_libraries(${ex} PRIVATE Threads::Threads)
endforeach()

foreach(pz IN LISTS PUZZLES)
    add_executable(${pz} ../puzzles/${pz}.cpp)
    target_link_libraries(${pz} PRIVATE Threads::Threads)
endforeach()

# --- Sanitizer targets ---
# UBSan
foreach(ex IN LISTS EXERCISES)
    add_executable(${ex}_ubsan ${ex}.cpp)
    target_link_libraries(${ex}_ubsan PRIVATE Threads::Threads)
    target_compile_options(${ex}_ubsan PRIVATE -fsanitize=undefined -fno-omit-frame-pointer)
    target_link_options(${ex}_ubsan PRIVATE -fsanitize=undefined)
endforeach()

# ASan + UBSan combined
foreach(ex IN LISTS EXERCISES)
    add_executable(${ex}_asan ${ex}.cpp)
    target_link_libraries(${ex}_asan PRIVATE Threads::Threads)
    target_compile_options(${ex}_asan PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${ex}_asan PRIVATE -fsanitize=address,undefined)
endforeach()
