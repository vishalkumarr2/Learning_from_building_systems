cmake_minimum_required(VERSION 3.16)
project(week3_rt_linux_programming LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic)

find_package(Threads REQUIRED)

# --- Exercise targets ---
set(EXERCISES
    ex01_rt_forbidden
    ex02_thread_affinity
    ex03_cyclic_executive
    ex04_pool_allocator
    ex05_rt_cyclic_with_telemetry
)

foreach(ex IN LISTS EXERCISES)
    add_executable(${ex} ${ex}.cpp)
    target_link_libraries(${ex} PRIVATE Threads::Threads)
endforeach()

# --- Puzzle targets ---
set(PUZZLES
    ../puzzles/puzzle01_clock_nanosleep
    ../puzzles/puzzle02_priority_inversion
    ../puzzles/puzzle03_page_fault
)

foreach(pz IN LISTS PUZZLES)
    get_filename_component(pz_name ${pz} NAME)
    add_executable(${pz_name} ${pz}.cpp)
    target_link_libraries(${pz_name} PRIVATE Threads::Threads)
endforeach()

# --- Notes ---
# Some exercises (ex02, ex03, ex05) benefit from elevated privileges.
# To grant CAP_SYS_NICE without running as root:
#   sudo setcap cap_sys_nice+ep ./ex02_thread_affinity
#   sudo setcap cap_sys_nice+ep ./ex03_cyclic_executive
#   sudo setcap cap_sys_nice+ep ./ex05_rt_cyclic_with_telemetry
#
# ex01 and ex04 run fine as a normal user.
# puzzle02 needs cap_sys_nice for SCHED_FIFO threads.
# puzzle03 needs cap_ipc_lock for mlockall (or run as root).
