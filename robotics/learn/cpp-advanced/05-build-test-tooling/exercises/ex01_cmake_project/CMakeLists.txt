cmake_minimum_required(VERSION 3.20)
project(rt_core
    VERSION 1.0.0
    LANGUAGES CXX
    DESCRIPTION "Real-Time Core Library — SPSC Queue & Cyclic Executive"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Find Dependencies ──────────────────────────────────────────
find_package(Threads REQUIRED)

# ─── rt_core Library ────────────────────────────────────────────
add_library(rt_core
    src/cyclic_executive.cpp
)

# Header-only portion (spsc_queue.hpp) + compiled headers
target_include_directories(rt_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(rt_core PUBLIC cxx_std_20)
target_link_libraries(rt_core PUBLIC Threads::Threads)

# Strict warnings for this target
target_compile_options(rt_core PRIVATE
    -Wall -Wextra -Wpedantic -Wshadow
    -Wconversion -Wsign-conversion
)

# ─── rt_tests Executable ───────────────────────────────────────
# Only build tests if we're the top-level project or GTest is available
if(TARGET GTest::gtest)
    add_executable(rt_tests
        tests/test_spsc_queue.cpp
        tests/test_cyclic_executive.cpp
    )
    target_link_libraries(rt_tests PRIVATE
        rt_core
        GTest::gtest
        GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(rt_tests)
endif()

# ─── Install Rules ──────────────────────────────────────────────
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS rt_core
    EXPORT rt_core-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/rt_core
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Export targets for find_package(rt_core)
install(EXPORT rt_core-targets
    FILE rt_core-targets.cmake
    NAMESPACE rt_core::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rt_core
)

# Generate package config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/rt_core-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/rt_core-config.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rt_core
    NO_SET_AND_CHECK_MACRO
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/rt_core-config-version.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/rt_core-config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/rt_core-config-version.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rt_core
)
