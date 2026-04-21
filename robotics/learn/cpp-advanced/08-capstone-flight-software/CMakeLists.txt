cmake_minimum_required(VERSION 3.16)
project(flight_sw LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Compiler warnings (JPL-style strict) ──────────────────────────────
add_compile_options(
    -Wall -Wextra -Wpedantic -Wconversion -Wshadow
    -Wnon-virtual-dtor -Wold-style-cast -Wcast-align
    -Woverloaded-virtual -Wsign-conversion -Wnull-dereference
    -Wdouble-promotion -Wformat=2
)

# ── Main executable ───────────────────────────────────────────────────
add_executable(flight_sw src/main.cpp)
target_include_directories(flight_sw PRIVATE include)
target_link_libraries(flight_sw PRIVATE pthread)

# ── Sanitizer presets (activated via CMAKE_CXX_FLAGS) ─────────────────
# Usage:
#   cmake -B build-asan -DCMAKE_BUILD_TYPE=Debug -DSANITIZER=asan
#   cmake -B build-tsan -DCMAKE_BUILD_TYPE=Debug -DSANITIZER=tsan
#   cmake -B build-ubsan -DCMAKE_BUILD_TYPE=Debug -DSANITIZER=ubsan
if(SANITIZER STREQUAL "asan")
    message(STATUS "Enabling AddressSanitizer")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
elseif(SANITIZER STREQUAL "tsan")
    message(STATUS "Enabling ThreadSanitizer")
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
elseif(SANITIZER STREQUAL "ubsan")
    message(STATUS "Enabling UBSanitizer")
    add_compile_options(-fsanitize=undefined -fno-sanitize-recover=all)
    add_link_options(-fsanitize=undefined -fno-sanitize-recover=all)
endif()

# ── Tests (GTest via FetchContent, optional) ──────────────────────────
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    include(FetchContent)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
        GIT_SHALLOW    TRUE
    )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

    FetchContent_GetProperties(googletest)
    if(NOT googletest_POPULATED)
        FetchContent_Populate(googletest)
        # Only add if fetch succeeded (offline builds skip tests gracefully)
        if(EXISTS "${googletest_SOURCE_DIR}/CMakeLists.txt")
            add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR}
                             EXCLUDE_FROM_ALL)

            add_executable(flight_sw_tests tests/test_main.cpp)
            target_include_directories(flight_sw_tests PRIVATE include)
            target_link_libraries(flight_sw_tests PRIVATE gtest gtest_main pthread)

            include(CTest)
            enable_testing()
            add_test(NAME flight_sw_tests COMMAND flight_sw_tests)

            message(STATUS "GTest found — flight_sw_tests target enabled")
        else()
            message(STATUS "GTest fetch failed — skipping test target (offline?)")
        endif()
    endif()
endif()
