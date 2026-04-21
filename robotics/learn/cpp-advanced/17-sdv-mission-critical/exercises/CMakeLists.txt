cmake_minimum_required(VERSION 3.16)
project(sdv_mission_critical LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Debug by default — exercises rely on assert()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Warnings: strict for safety-critical code
add_compile_options(-Wall -Wextra -Wpedantic -Werror
                    -Wconversion -Wsign-conversion
                    -Wdouble-promotion -Wformat=2
                    -Wnull-dereference -Wold-style-cast
                    -Wshadow -Wunused)

# ThreadSanitizer for lock-free exercises
option(ENABLE_TSAN "Enable ThreadSanitizer" OFF)
if(ENABLE_TSAN)
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
endif()

add_executable(ex01_e2e_protection ex01_e2e_protection.cpp)
add_executable(ex02_degradation_fsm ex02_degradation_fsm.cpp)
add_executable(ex03_cyclic_executor ex03_cyclic_executor.cpp)
target_link_libraries(ex03_cyclic_executor PRIVATE pthread)
add_executable(ex04_triple_buffer ex04_triple_buffer.cpp)
target_link_libraries(ex04_triple_buffer PRIVATE pthread)
add_executable(ex05_dtc_manager ex05_dtc_manager.cpp)
add_executable(ex06_ota_state_machine ex06_ota_state_machine.cpp)
add_executable(ex07_secoc_authenticator ex07_secoc_authenticator.cpp)
add_executable(ex08_service_health_monitor ex08_service_health_monitor.cpp)
target_link_libraries(ex08_service_health_monitor PRIVATE pthread)
