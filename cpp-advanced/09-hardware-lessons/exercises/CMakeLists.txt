cmake_minimum_required(VERSION 3.16)
project(HardwareLessons LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic -O2)

find_package(Threads REQUIRED)

set(EXERCISES
    hw01_clock_drift
    hw02_gpio_signals
    hw03_dma_cache
    hw04_scheduler
    hw05_memory_hierarchy
)

foreach(ex IN LISTS EXERCISES)
    add_executable(${ex} ${ex}.cpp)
    target_link_libraries(${ex} PRIVATE Threads::Threads)
endforeach()
