# Advanced C++ for Real-Time, Safety-Critical & Production Systems

**8 weeks · 1–2 hrs/day · Hands-on every day · Linux target**

From "knows templates" to "can write flight-software-grade C++."

## Quick Start

```bash
# Build and run any week's exercises:
cd 01-move-semantics-value-categories/exercises
mkdir -p build && cd build
cmake .. && make -j$(nproc)
./ex01_value_categories

# Or use the top-level build script:
./build_all.sh        # build everything
./build_all.sh 03     # build only week 03
```

## Structure

```
learn/cpp-advanced/
├── STUDY-PLAN.md                          ← Master plan (read this first)
├── README.md                              ← You are here
├── build_all.sh                           ← Build helper
│
├── 01-move-semantics-value-categories/    ← Week 1
│   ├── notes.md                           ← Teaching material
│   ├── exercises/                         ← Compilable exercises
│   │   ├── CMakeLists.txt
│   │   ├── ex01_value_categories.cpp
│   │   ├── ex02_rule_of_five.cpp
│   │   ├── ex03_perfect_forwarding.cpp
│   │   ├── ex04_constexpr_crc32.cpp
│   │   ├── ex05_concepts.cpp
│   │   └── ex06_message_descriptor.cpp
│   └── puzzles/                           ← "99% fail" brain teasers
│       ├── puzzle01_named_rref.cpp
│       ├── puzzle02_moved_from_vector.cpp
│       ├── puzzle03_forwarding_deduction.cpp
│       ├── puzzle04_constexpr_runtime.cpp
│       └── puzzle05_concept_overload.cpp
│
├── 02-error-handling-memory-model/        ← Week 2
│   ├── notes.md
│   ├── exercises/ (7 exercises)
│   └── puzzles/   (5 puzzles)
│
├── 03-rt-linux-programming/              ← Week 3
│   ├── notes.md
│   ├── exercises/ (5 exercises)
│   └── puzzles/   (3 puzzles)
│
├── 04-safety-critical-patterns/          ← Week 4
│   ├── notes.md
│   ├── exercises/ (5 exercises)
│   └── puzzles/   (3 puzzles)
│
├── 05-build-test-tooling/                ← Week 5
│   ├── notes.md
│   ├── exercises/ (5 exercises + cmake project)
│   └── puzzles/   (2 puzzles)
│
├── 06-ipc-serialization/                 ← Week 6
│   ├── notes.md
│   ├── exercises/ (5 exercises)
│   └── puzzles/   (2 puzzles)
│
├── 07-ub-advanced-patterns/              ← Week 7
│   ├── notes.md
│   ├── exercises/ (5 exercises)
│   └── puzzles/   (2 puzzles)
│
├── 08-capstone-flight-software/          ← Week 8 (capstone project)
│   ├── README.md
│   ├── CMakeLists.txt
│   ├── include/flight_sw/               ← 8 header-only components
│   ├── src/main.cpp
│   └── tests/test_main.cpp
│
├── 09-hardware-lessons/                  ← ⚡ Hardware deep dives
│   └── exercises/ (5 exercises)
│
└── 10-safety-lessons/                    ← 🛡️ Safety-critical deep dives
    └── exercises/ (5 exercises)
```

## Content Summary

| # | Topic | Exercises | Puzzles | Notes |
|---|-------|-----------|---------|-------|
| 01 | Move semantics, value categories, constexpr, concepts | 6 | 5 | ✅ |
| 02 | Error handling, memory model, atomics, SPSC queue | 7 | 5 | ✅ |
| 03 | RT Linux: cyclic executive, pool allocators, PREEMPT_RT | 5 | 3 | ✅ |
| 04 | Safety-critical: JPL rules, watchdog, health monitor | 5 | 3 | ✅ |
| 05 | CMake, GTest, fuzzing, profiling, CI/CD | 5 | 2 | ✅ |
| 06 | IPC, shared memory, serialization, type erasure, plugins | 5 | 2 | ✅ |
| 07 | UB deep dive, CRTP, policy design, structured logging | 5 | 2 | ✅ |
| 08 | Capstone: mini flight software (all components) | — | — | ✅ |
| 09 | ⚡ Hardware: clocks, GPIO, DMA, scheduler, memory | 5 | — | — |
| 10 | 🛡️ Safety: MC/DC, MISRA, TMR, formal methods, incidents | 5 | — | — |
| **Total** | | **53** | **22** | **8** |

## Prerequisites

- GCC 9+ or Clang 10+ (C++20 support)
- CMake 3.16+
- Linux (any distro, PREEMPT_RT optional)
- pthreads, librt, libdl (standard on Linux)

## How to Use

1. **Read STUDY-PLAN.md** — the master plan with all exercises embedded
2. **Each day**: Read notes.md (20 min) → Code exercises (40-60 min) → Solve puzzles (10-20 min)
3. **Each week**: Complete the mini-project
4. **Week 8**: Build the capstone combining everything
5. **Hardware & Safety lessons**: Work through alongside or after the main 8 weeks

## Marker Legend

| Marker | Meaning |
|--------|---------|
| 💀 | "99% fail" puzzle — tests deep understanding |
| 🤖 | ROS/robotics tie-in — connects to warehouse robot work |
| 🧓 | Senior lesson — 20-year veteran wisdom |
| ⚡ | Hardware lesson — what the electrons do |
| 🛡️ | Safety lesson — when code can kill |
