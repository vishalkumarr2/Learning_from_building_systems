cmake_minimum_required(VERSION 3.16)
project(week2_error_handling_memory_model LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(Threads REQUIRED)

# --- Normal targets ---
set(EXERCISES
    ex01_exception_cost
    ex02_expected
    ex03_variant_state_machine
    ex04_memory_model
    ex05_acquire_release
    ex06_compare_exchange
    ex07_spsc_queue
)

foreach(ex IN LISTS EXERCISES)
    add_executable(${ex} ${ex}.cpp)
    target_link_libraries(${ex} PRIVATE Threads::Threads)
endforeach()

# --- ThreadSanitizer targets for concurrency exercises ---
set(TSAN_EXERCISES
    ex04_memory_model
    ex05_acquire_release
    ex06_compare_exchange
    ex07_spsc_queue
)

foreach(ex IN LISTS TSAN_EXERCISES)
    add_executable(tsan_${ex} ${ex}.cpp)
    target_link_libraries(tsan_${ex} PRIVATE Threads::Threads)
    target_compile_options(tsan_${ex} PRIVATE -fsanitize=thread -g)
    target_link_options(tsan_${ex} PRIVATE -fsanitize=thread)
endforeach()
