cmake_minimum_required(VERSION 3.20)
project(safety_lessons LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -Wpedantic -O2)

find_package(Threads REQUIRED)

set(EXERCISES
    sf01_mcdc_coverage
    sf02_misra_compliance
    sf03_fault_tolerance
    sf04_formal_lite
    sf05_incident_analysis
)

foreach(exercise ${EXERCISES})
    add_executable(${exercise} ${exercise}.cpp)
    target_link_libraries(${exercise} PRIVATE Threads::Threads)
endforeach()
