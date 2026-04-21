cmake_minimum_required(VERSION 3.20)
project(week4_safety_critical LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Safety-critical level warnings
add_compile_options(
    -Wall -Wextra -Wpedantic -Werror
    -Wshadow -Wconversion -Wdouble-promotion
    -Wformat=2 -Wnull-dereference
    -Woverloaded-virtual -Wnon-virtual-dtor
    -fstack-protector-strong
    -D_FORTIFY_SOURCE=2
)

# Sanitizers for debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR NOT CMAKE_BUILD_TYPE)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=address,undefined)
endif()

# --- Exercises ---
add_executable(ex01_jpl_rules ex01_jpl_rules.cpp)
add_executable(ex02_defensive_programming ex02_defensive_programming.cpp)
add_executable(ex03_static_analysis ex03_static_analysis.cpp)
# ex03 has intentionally buggy code in namespaces — suppress -Werror for it
target_compile_options(ex03_static_analysis PRIVATE -Wno-error=conversion)
add_executable(ex04_watchdog ex04_watchdog.cpp)
target_link_libraries(ex04_watchdog PRIVATE pthread)
add_executable(ex05_watchdog_health_project ex05_watchdog_health_project.cpp)
target_link_libraries(ex05_watchdog_health_project PRIVATE pthread)

# --- Puzzles ---
add_executable(puzzle01_integer_overflow ../puzzles/puzzle01_integer_overflow.cpp)
add_executable(puzzle02_watchdog_cheat ../puzzles/puzzle02_watchdog_cheat.cpp)
target_link_libraries(puzzle02_watchdog_cheat PRIVATE pthread)
add_executable(puzzle03_signed_unsigned ../puzzles/puzzle03_signed_unsigned.cpp)

# Puzzle 03 needs -Wno-sign-compare to compile (demonstrates the traps)
target_compile_options(puzzle03_signed_unsigned PRIVATE -Wno-sign-compare -Wno-error)
# Puzzle 01 needs -Wno-conversion for the intentional overflow demo
target_compile_options(puzzle01_integer_overflow PRIVATE -Wno-conversion -Wno-error)
