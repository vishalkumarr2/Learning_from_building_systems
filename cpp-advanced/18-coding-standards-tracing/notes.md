# Module 18: Coding Standards, Profiling & System Tracing

## Overview

Module 16 taught you **what** sanitizers catch and how to read their reports.
This module teaches the **broader quality ecosystem**: how to prevent bugs
before they happen (coding standards + static analysis), how to find performance
bottlenecks (profiling), how to trace system behaviour in production (LTTng,
strace, perf), and how to debug the toughest problems (advanced GDB).

**Key insight**: Sanitizers are *reactive* — they find bugs after you write them.
Coding standards + static analysis are *preventive*. Profiling and tracing are
*diagnostic*. A production-grade workflow uses ALL three layers.

```
Layer 1 — PREVENTION:  Coding standards → static analysis → compiler warnings
Layer 2 — DETECTION:   Sanitizers (ASan, TSan, UBSan) → Module 16
Layer 3 — DIAGNOSIS:   Profiling (perf, callgrind) → Tracing (LTTng, strace)
Layer 4 — DEBUG:       GDB advanced → core dumps → post-mortem
```

---

## Part 1: C++ Coding Standards & Static Analysis

### 1.1 Why Coding Standards Matter

Coding standards reduce bugs by **banning patterns that are known to cause defects**.
They are not about style (tabs vs spaces) — they encode decades of hard-won
knowledge about what goes wrong in C++ code.

| Standard | Scope | Target Domain | Key Feature |
|----------|-------|---------------|-------------|
| **CppCoreGuidelines** | General | All C++ | Broad, modern, tool-enforceable |
| **MISRA C++:2023** | Safety | Automotive, medical, aerospace | Decidable, every rule provable |
| **CERT C++** | Security | Network, systems | Focuses on exploitability |
| **AUTOSAR C++14** | Safety | Automotive ECU | Based on MISRA, adds automotive rules |
| **SEI CERT C** | Security | C code (not C++) | Applicable to C-style code in C++ |
| **JSF++ AV** | Safety | Joint Strike Fighter | Military aerospace, very strict |

**For robot software** (our context): CppCoreGuidelines + selected MISRA rules
give the best cost/benefit ratio. Full MISRA compliance is only needed for
ISO 26262 / IEC 61508 certification.

---

### 1.2 CppCoreGuidelines — The Essential Rules

The full guidelines have 500+ rules. Here are the ones that catch the most bugs:

#### Resource Management (R)

```cpp
// R.1: Manage resources automatically using RAII
// BAD — raw pointer ownership
void bad() {
    auto* p = new Widget();
    do_something(*p);  // if this throws → LEAK
    delete p;
}

// GOOD — unique_ptr enforces cleanup
void good() {
    auto p = std::make_unique<Widget>();
    do_something(*p);  // exception-safe, no leak possible
}
```

```cpp
// R.3: A raw pointer (T*) is non-owning
// Rule: if you see T*, the pointed-to object is managed elsewhere.
// Ownership MUST use unique_ptr or shared_ptr.

void process(Widget* w);  // non-owning: caller keeps ownership
auto owner = std::make_unique<Widget>();
process(owner.get());     // explicitly non-owning
```

```cpp
// R.5: Prefer scoped objects, don't heap-allocate unnecessarily
// BAD — heap allocation for no reason
auto* data = new std::vector<int>{1, 2, 3};
// ...
delete data;

// GOOD — stack allocation
std::vector<int> data{1, 2, 3};
```

#### Lifetime Safety (F, C)

```cpp
// F.42: Return a T* to indicate a position (only)
// Never return a pointer to indicate ownership transfer.

// F.43: Never return a pointer or reference to a local object
// BAD
int& bad() {
    int local = 42;
    return local;  // dangling reference — undefined behavior
}

// C.31: All resources acquired by a class must be released by the destructor
// This IS the Rule of Zero / Rule of Five.
```

#### Concurrency (CP)

```cpp
// CP.1: Assume code will run in a multi-threaded environment
// Even single-threaded code may become multi-threaded later.

// CP.2: Avoid data races
// A data race = two threads access the same memory, at least one writes,
// no synchronization. This is UNDEFINED BEHAVIOR (not just a bug).

// CP.20: Use RAII for locking, never plain lock()/unlock()
// BAD
mutex_.lock();
do_work();  // if this throws → deadlock (mutex never unlocked)
mutex_.unlock();

// GOOD
{
    std::lock_guard<std::mutex> lock(mutex_);
    do_work();  // unlock guaranteed even on exception
}

// CP.44: Remember to name your lock_guards and unique_locks
// BAD — anonymous temporary, unlocks immediately!
std::lock_guard<std::mutex>{mutex_};  // ← UNLOCKS RIGHT HERE
do_work();  // ← UNPROTECTED!

// GOOD — named variable lives until scope end
std::lock_guard<std::mutex> lock{mutex_};
do_work();  // ← protected
```

#### Type Safety (I, ES)

```cpp
// ES.48: Avoid casts
// If you must cast, prefer static_cast over C-style casts.
// C-style casts can silently do reinterpret_cast.

// I.11: Never transfer ownership by a raw pointer
// Use unique_ptr for single ownership, shared_ptr for shared.

// ES.46: Avoid lossy narrowing conversions
// BAD
int64_t big = 1LL << 40;
int32_t small = big;  // silent truncation

// GOOD
int32_t small = static_cast<int32_t>(big);  // explicit
// Even better: use gsl::narrow<> which throws on data loss
```

---

### 1.3 MISRA C++:2023 — Safety-Critical Rules

MISRA rules are classified as **Required**, **Advisory**, or **Mandatory**.
Here are the most impactful Required rules:

| Rule | Category | What it bans | Why |
|------|----------|-------------|-----|
| 0.1.2 | Mandatory | Unreachable code | Dead code masks bugs |
| 4.6.1 | Required | Implicit narrowing conversions | Silent data loss |
| 6.0.1 | Required | `goto` | Unstructured control flow |
| 6.7.2 | Required | Global non-const variables | Hidden coupling |
| 6.8.2 | Required | Single-use variables | Unnecessarily complex |
| 7.0.5 | Required | C-style casts | Bypasses type system |
| 7.6.1 | Required | `reinterpret_cast` | Undefined behavior risk |
| 8.2.5 | Required | Virtual functions in constructors | Surprising dispatch |
| 8.14.1 | Required | `const_cast` to remove const | Breaks type safety |
| 9.3.1 | Required | `malloc`/`free` in C++ | Use RAII instead |
| 10.3.1 | Required | Empty throw (`throw;`) outside catch | Calls `std::terminate` |
| 12.3.1 | Required | NULL macro | Use `nullptr` |
| 15.0.2 | Required | Uninitialized variables | Undefined behavior |
| 19.0.1 | Required | `#define` for constants | Use `constexpr` |
| 21.10.1 | Required | `signal()` for signal handling | Race conditions |

**Key insight**: Many MISRA rules can be enforced automatically with
clang-tidy + cppcheck. You don't need to memorize them — configure your tools.

---

### 1.4 CERT C++ Secure Coding

CERT rules focus on **security** — what an attacker can exploit:

| Rule | What | Exploit |
|------|------|---------|
| STR50-CPP | Guarantee null termination | Buffer overflow |
| MEM50-CPP | Don't access freed memory | Use-after-free → RCE |
| MEM51-CPP | Properly deallocate memory | Memory corruption |
| INT50-CPP | Don't cast to smaller type | Integer truncation |
| ERR50-CPP | Don't call `exit()` in destructors | Stack unwinding break |
| CON50-CPP | Don't destroy a locked mutex | Undefined behavior |
| DCL50-CPP | Don't define a C-style variadic function | Type confusion |
| EXP50-CPP | Don't depend on order of evaluation | Sequencing bugs |

---

### 1.5 Enforcing Standards with clang-tidy

clang-tidy is the primary tool for automated coding standard enforcement:

```bash
# Install (Ubuntu 20.04)
sudo apt install clang-tidy-10  # or later

# Run all checks
clang-tidy -checks='*' source.cpp -- -std=c++2a

# Run specific check categories
clang-tidy -checks='cppcoreguidelines-*,modernize-*,bugprone-*' source.cpp

# Auto-fix what it can
clang-tidy -checks='modernize-*' -fix source.cpp

# Use with compile_commands.json (from CMake)
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
clang-tidy -p build/ source.cpp
```

#### Essential clang-tidy check categories:

| Category | Catches | Example |
|----------|---------|---------|
| `bugprone-*` | Likely bugs | Dangling handles, infinite loops, misused `move` |
| `cppcoreguidelines-*` | CppCoreGuidelines violations | Owning raw pointers, C-arrays |
| `modernize-*` | Pre-C++11 patterns | `NULL` → `nullptr`, raw loops → algorithms |
| `performance-*` | Performance issues | Unnecessary copies, move-eligible returns |
| `readability-*` | Readability | Magic numbers, inconsistent naming |
| `cert-*` | CERT rules | Security-related issues |
| `misc-*` | Miscellaneous | Unused parameters, redundant expressions |
| `clang-analyzer-*` | Deep analysis | Null deref paths, dead stores |

#### Configuration file (`.clang-tidy`):

```yaml
---
Checks: >
  -*,
  bugprone-*,
  cppcoreguidelines-*,
  modernize-*,
  performance-*,
  readability-*,
  cert-*,
  -modernize-use-trailing-return-type,
  -readability-magic-numbers,
  -cppcoreguidelines-avoid-magic-numbers
WarningsAsErrors: 'bugprone-*,cert-*'
HeaderFilterRegex: '.*'
CheckOptions:
  - key: readability-identifier-naming.ClassCase
    value: CamelCase
  - key: readability-identifier-naming.FunctionCase
    value: camelBack
  - key: readability-identifier-naming.VariableCase
    value: lower_case
  - key: readability-identifier-naming.MemberPrefix
    value: ''
  - key: readability-identifier-naming.MemberSuffix
    value: '_'
```

---

### 1.6 Enforcing Standards with cppcheck

cppcheck does pattern-based analysis (no Clang AST dependency):

```bash
# Install
sudo apt install cppcheck

# Basic analysis
cppcheck --enable=all --std=c++20 source.cpp

# With suppression file
cppcheck --enable=all --suppressions-list=cppcheck.supp source.cpp

# Generate XML report (for CI integration)
cppcheck --enable=all --xml source.cpp 2> cppcheck_report.xml

# Check a whole directory
cppcheck --enable=all -I include/ src/
```

#### What cppcheck catches that clang-tidy sometimes misses:

| Issue | Example |
|-------|---------|
| Null pointer dereference paths | Complex conditional chains |
| Buffer overflows | Array index out of bounds |
| Uninitialized variables | Conditional initialization paths |
| Memory leaks | Non-RAII allocation patterns |
| Resource leaks | File descriptors, sockets |
| Redundant conditions | `if (p != NULL)` after `p = new X` |
| Portability issues | Different behavior across compilers |

#### Suppression file (`cppcheck.supp`):

```
// Suppress specific findings
unusedFunction:src/test_helper.cpp
uninitvar:third_party/*.cpp
// Suppress by ID
memleak:src/legacy.cpp:42
```

---

### 1.7 Compiler Warnings as a First Line of Defence

The cheapest static analysis is your compiler's warning flags:

```bash
# Recommended minimum for any project
-Wall -Wextra -Wpedantic

# Safety-critical / production quality
-Wall -Wextra -Wpedantic -Werror \
-Wconversion -Wsign-conversion \
-Wdouble-promotion -Wformat=2 \
-Wnull-dereference -Wold-style-cast \
-Wshadow -Wunused

# GCC-specific extras
-Wlogical-op -Wduplicated-cond -Wduplicated-branches \
-Wuseless-cast -Wrestrict

# Clang-specific extras
-Wmost -Weverything  # (warning: very noisy)
```

#### What each flag catches:

| Flag | Catches | Example |
|------|---------|---------|
| `-Wconversion` | Implicit narrowing | `int x = 3.14;` |
| `-Wsign-conversion` | Signed ↔ unsigned | `unsigned u = -1;` |
| `-Wshadow` | Variable shadowing | Inner `x` hides outer `x` |
| `-Wdouble-promotion` | Float→double promotion | `float f; printf("%f", f);` |
| `-Wold-style-cast` | C-style casts | `(int)ptr` instead of `static_cast` |
| `-Wnull-dereference` | Potential null deref | GCC path analysis |
| `-Wformat=2` | Printf format mismatches | `printf("%d", "hello")` |
| `-Wduplicated-cond` | Duplicate `if` conditions | Copy-paste bugs |

#### include-what-you-use (IWYU)

```bash
# Install
sudo apt install iwyu

# Run (needs compile_commands.json)
iwyu_tool.py -p build/ source.cpp

# IWYU output example:
# source.cpp should add:    #include <algorithm>
# source.cpp should remove:  #include <iostream>  // not used
```

IWYU ensures each file includes exactly what it uses — no transitive include
dependencies, no unused headers. This improves compile times and makes
dependencies explicit.

---

## Part 2: Profiling & Performance Analysis

### 2.1 perf — Linux Performance Counters

`perf` is the standard Linux profiling tool. It uses hardware performance
counters (PMU — Performance Monitoring Unit) built into the CPU.

```bash
# Install
sudo apt install linux-tools-common linux-tools-$(uname -r)

# Allow non-root profiling (set once)
echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

#### perf stat — Count events

```bash
# Count cache misses, branch mispredictions, instructions
perf stat ./my_program

# Example output:
#    1,234,567,890  instructions       #  2.50 IPC
#       45,678,901  cache-misses       #  3.7% of all cache refs
#       12,345,678  branch-misses      #  1.2% of all branches
#            2.345  seconds time elapsed

# Specific events
perf stat -e cache-misses,cache-references,L1-dcache-load-misses ./my_program

# Repeat N times for statistical significance
perf stat -r 10 ./my_program
```

**Key metric**: IPC (Instructions Per Cycle). An IPC < 1.0 usually means
the CPU is stalling on memory access (cache misses). IPC > 2.0 is good.

#### perf record + perf report — Profile

```bash
# Record call stacks at 99 Hz (use prime numbers to avoid aliasing)
perf record -g -F 99 ./my_program

# Show interactive profile
perf report

# Show annotated source (requires -g debug info)
perf annotate

# Key perf report columns:
# Overhead — % of total samples in this function
# Children — % of total samples in this function + its callees
# Self     — % of total samples ONLY in this function (excludes callees)
```

#### perf record tips:

```bash
# Record for a running process
perf record -g -p $(pidof my_process) -- sleep 10

# Record specific events
perf record -e cache-misses -g ./my_program

# Record with call graph (dwarf = most reliable, needs -g in compilation)
perf record --call-graph dwarf -F 99 ./my_program
```

---

### 2.2 Valgrind Callgrind — Call Graph Profiling

Callgrind simulates the CPU's cache hierarchy and counts instruction costs:

```bash
# Profile
valgrind --tool=callgrind ./my_program

# Output: callgrind.out.<pid>

# Annotate source (text)
callgrind_annotate callgrind.out.12345

# Visualize with KCachegrind (GUI)
kcachegrind callgrind.out.12345
```

#### Callgrind vs perf:

| Feature | perf | Callgrind |
|---------|------|-----------|
| Speed | ~1x (hardware counters) | ~20-100x (simulation) |
| Accuracy | Statistical sampling | Exact instruction count |
| Cache model | Real hardware | Simulated (may differ) |
| Call graph | Yes (sampling) | Yes (exact) |
| Thread support | Yes | Limited |
| Root required | Usually no | No |

**Rule of thumb**: Use `perf` for quick profiling, `callgrind` when you need
exact call counts or cache simulation details.

---

### 2.3 Valgrind Cachegrind — Cache Simulation

```bash
# Run cache simulation
valgrind --tool=cachegrind ./my_program

# Output example:
# ==12345== D1  miss rate:  4.2% (  3.8% rd + 6.1% wr)
# ==12345== LLd miss rate:  0.8% (  0.7% rd + 1.2% wr)

# Annotate per-line cache misses
cg_annotate cachegrind.out.12345
```

**What to look for**:
- D1 miss rate > 5% → data cache pressure, likely array access pattern issue
- LLd (Last-Level cache) miss rate > 2% → memory bandwidth bottleneck
- High `Dw` (data writes) miss rate → false sharing in multi-threaded code

---

### 2.4 Flamegraphs

Flamegraphs visualize profiling data as stacked function calls where width
represents time. Created by Brendan Gregg.

```bash
# Install
git clone https://github.com/brendangregg/FlameGraph.git

# Generate from perf data
perf record -g -F 99 ./my_program
perf script > out.perf
FlameGraph/stackcollapse-perf.pl out.perf > out.folded
FlameGraph/flamegraph.pl out.folded > flamegraph.svg

# Open in browser
firefox flamegraph.svg
```

#### Reading a flamegraph:

```
    ┌──────────── main() ────────────────────┐
    │  ┌─── process_data() ────────────────┐ │
    │  │  ┌── sort_items() ──────────┐     │ │  ← WIDEST = HOTTEST
    │  │  │  ┌─ compare() ─────┐     │     │ │
    │  │  │  └─────────────────┘     │     │ │
    │  │  └──────────────────────────┘     │ │
    │  │  ┌── validate() ─┐               │ │
    │  │  └────────────────┘               │ │
    │  └───────────────────────────────────┘ │
    └────────────────────────────────────────┘
```

- **Width** = time spent. `sort_items()` is the hottest function.
- **Height** = call depth. Read bottom (caller) to top (callee).
- **Colour** is random (for visual separation), not meaningful.
- Look for **wide plateaus** — those are your optimization targets.

---

### 2.5 CPU Cache Effects on C++ Performance

Cache-friendly code can be 10-100x faster than cache-unfriendly code for
the same algorithmic complexity:

#### Row-major vs column-major (the classic example)

```cpp
// Cache-friendly: sequential access (row-major in C++)
for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
        matrix[i][j] *= 2;  // stride = sizeof(element)

// Cache-HOSTILE: column-major access
for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i)
        matrix[i][j] *= 2;  // stride = N * sizeof(element)
```

For N=4096, column-major is ~10x slower due to L1 cache thrashing.

#### Struct of Arrays (SoA) vs Array of Structs (AoS)

```cpp
// AoS — cache-unfriendly if you only access one field
struct Particle { float x, y, z, mass, charge, radius; };
std::vector<Particle> particles(N);

// Process only positions → loads mass, charge, radius into cache too (waste!)
for (auto& p : particles) p.x += p.vx * dt;

// SoA — cache-friendly for single-field access
struct Particles {
    std::vector<float> x, y, z, mass, charge, radius;
};
Particles ps;
ps.x.resize(N); ps.vx.resize(N);

// Process only x → only x data in cache (no waste)
for (size_t i = 0; i < N; ++i) ps.x[i] += ps.vx[i] * dt;
```

#### False sharing

```cpp
// BAD: Two threads writing to adjacent cache lines
struct alignas(8) Counters {
    std::atomic<int> count_a;  // thread 1 writes here
    std::atomic<int> count_b;  // thread 2 writes here
    // Both fit in ONE 64-byte cache line → ping-pong between cores
};

// GOOD: Pad to separate cache lines
struct Counters {
    alignas(64) std::atomic<int> count_a;  // own cache line
    alignas(64) std::atomic<int> count_b;  // own cache line
};
```

False sharing can cause a 10x slowdown because every write by one thread
invalidates the other thread's copy of the cache line.

---

## Part 3: System Tracing

### 3.1 LTTng — Linux Tracing Toolkit next generation

LTTng is a high-performance tracing framework for Linux. It can trace:
- **Kernel** events (syscalls, scheduling, interrupts, block I/O)
- **Userspace** events (your application's tracepoints)

LTTng's overhead is ~100ns per tracepoint — low enough for production use.
This is 10-100x lower than printf-debugging or syslog.

#### Architecture:

```
┌─────────────────────────────────────────┐
│  Your Application                       │
│  ┌─────────────────────────────────┐    │
│  │ TRACEPOINT(my_app, request_start│    │
│  │   , size_t, req_id             │    │
│  │   , int, priority              │    │
│  │ )                               │    │
│  └───────────┬─────────────────────┘    │
│              │ ~100ns per tracepoint    │
│  ┌───────────▼──────────────────────┐   │
│  │  LTTng-UST (Userspace Tracer)   │   │
│  └───────────┬──────────────────────┘   │
└──────────────┼──────────────────────────┘
               │ shared memory ring buffer
┌──────────────▼──────────────────────────┐
│  lttng-sessiond (Session Daemon)        │
│  ┌──────────────────────────────────┐   │
│  │ lttng-consumerd → trace files    │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│  Analysis Tools                         │
│  babeltrace2, Trace Compass, LTTng Live │
└─────────────────────────────────────────┘
```

#### Installation:

```bash
# Ubuntu 20.04+
sudo apt install lttng-tools lttng-modules-dkms liblttng-ust-dev babeltrace2

# Verify
lttng version
```

#### Defining userspace tracepoints:

Create a tracepoint provider header (`my_tp.h`):

```c
/* my_tp.h */
#undef TRACEPOINT_PROVIDER
#define TRACEPOINT_PROVIDER my_app

#undef TRACEPOINT_INCLUDE
#define TRACEPOINT_INCLUDE "./my_tp.h"

#if !defined(_MY_TP_H) || defined(TRACEPOINT_HEADER_MULTI_READ)
#define _MY_TP_H

#include <lttng/tracepoint.h>

TRACEPOINT_EVENT(
    my_app,            /* provider name */
    request_start,     /* event name */
    TP_ARGS(
        size_t, req_id,
        int, priority,
        const char*, endpoint
    ),
    TP_FIELDS(
        ctf_integer(size_t, req_id, req_id)
        ctf_integer(int, priority, priority)
        ctf_string(endpoint, endpoint)
    )
)

TRACEPOINT_EVENT(
    my_app,
    request_end,
    TP_ARGS(
        size_t, req_id,
        int, status_code,
        uint64_t, duration_ns
    ),
    TP_FIELDS(
        ctf_integer(size_t, req_id, req_id)
        ctf_integer(int, status_code, status_code)
        ctf_integer(uint64_t, duration_ns, duration_ns)
    )
)

#endif /* _MY_TP_H */

#include <lttng/tracepoint-event.h>
```

Create the tracepoint provider source (`my_tp.c`):

```c
/* my_tp.c */
#define TRACEPOINT_CREATE_PROBES
#define TRACEPOINT_DEFINE
#include "my_tp.h"
```

Use in your C++ code:

```cpp
#include "my_tp.h"

void handle_request(size_t id, int priority, const char* endpoint) {
    tracepoint(my_app, request_start, id, priority, endpoint);
    // ... do work ...
    auto start = std::chrono::steady_clock::now();
    process(id);
    auto elapsed = std::chrono::steady_clock::now() - start;
    uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
    tracepoint(my_app, request_end, id, 200, ns);
}
```

Compile:

```bash
gcc -c my_tp.c -I.
g++ -std=c++2a -c main.cpp -I.
g++ main.o my_tp.o -ldl -llttng-ust -o my_app
```

#### Recording traces:

```bash
# Create a session
lttng create my-session --output=/tmp/my-trace

# Enable userspace events
lttng enable-event --userspace 'my_app:*'

# Enable kernel events (optional, needs root)
sudo lttng enable-event --kernel sched_switch,sched_wakeup

# Start tracing
lttng start

# Run your application
./my_app

# Stop and destroy session
lttng stop
lttng destroy

# View trace (text)
babeltrace2 /tmp/my-trace

# Example output:
# [10:30:01.123456789] my_app:request_start: req_id=42 priority=3 endpoint="/api/data"
# [10:30:01.124567890] my_app:request_end:   req_id=42 status_code=200 duration_ns=1111101
```

#### LTTng for ROS systems:

```bash
# Trace ROS 2 DDS events + your custom tracepoints
lttng create ros-trace
lttng enable-event --userspace 'ros2:*'
lttng enable-event --userspace 'my_navigation:*'
lttng enable-event --kernel sched_switch  # see thread scheduling
lttng start

ros2 run my_package my_node
# ... reproduce issue ...

lttng stop && lttng destroy
babeltrace2 /tmp/ros-trace | grep -E 'callback|timer|my_navigation'
```

---

### 3.2 strace — System Call Tracing

strace intercepts all system calls made by a process. Invaluable for:
- Understanding what files a program opens
- Finding why a program hangs (stuck in `read()`, `futex()`, etc.)
- Measuring syscall latency
- Debugging permission errors

```bash
# Trace all syscalls
strace ./my_program

# Trace specific syscall categories
strace -e trace=file ./my_program      # open, read, write, stat, etc.
strace -e trace=network ./my_program   # socket, connect, send, recv
strace -e trace=process ./my_program   # fork, exec, exit
strace -e trace=memory ./my_program    # mmap, brk, mprotect

# Attach to running process
strace -p $(pidof my_process)

# Count syscalls (summary)
strace -c ./my_program
# Example output:
# % time     seconds  calls  errors syscall
# ------ ----------- ------ ------ --------
#  45.23   0.002345   12345      0 write
#  30.12   0.001567    8901      0 read
#  10.45   0.000543    4567     23 open
#   5.67   0.000294    2345      0 close

# With timestamps (microsecond resolution)
strace -T -t ./my_program
# 10:30:01 write(1, "hello\n", 6) = 6 <0.000015>
#                                       ^^^^^^^^^^^ syscall duration

# Follow child processes
strace -f ./my_program

# Output to file (stderr is trace output)
strace -o trace.log ./my_program
```

#### strace patterns for robot software:

```bash
# Find which config files a ROS node reads
strace -e trace=open,openat rosrun my_package my_node 2>&1 | grep -v ENOENT

# Find why a node hangs on startup
strace -e trace=futex,read,poll rosrun my_package my_node
# If stuck in futex() → waiting for a lock (mutex contention or deadlock)
# If stuck in poll()  → waiting for network data (topic not published)

# Measure I/O syscall counts for a log-heavy node
strace -c -e trace=write rosrun my_package my_node
```

---

### 3.3 ltrace — Library Call Tracing

ltrace traces dynamic library calls (like strace for libc/libstdc++):

```bash
# Trace library calls
ltrace ./my_program

# Example output:
# malloc(64)                = 0x5555557b0260
# memcpy(0x5555557b0260, "hello", 5) = 0x5555557b0260
# free(0x5555557b0260)

# Trace specific libraries
ltrace -e 'malloc+free' ./my_program

# Count calls
ltrace -c ./my_program
```

**Use case**: Finding unexpected allocations in a real-time code path.
If you see `malloc` calls from a function that should be allocation-free,
you have a latency problem.

---

### 3.4 ftrace & trace-cmd

ftrace is the Linux kernel's built-in tracer. trace-cmd is its CLI:

```bash
# Install
sudo apt install trace-cmd

# Record function tracer for 5 seconds
sudo trace-cmd record -p function -l 'sched_*' sleep 5
sudo trace-cmd report | head -50

# Record function graph (call tree with timing)
sudo trace-cmd record -p function_graph -l 'ext4_*' -- dd if=/dev/zero of=/tmp/test bs=4k count=1000
sudo trace-cmd report

# Trace scheduling events (who preempted whom)
sudo trace-cmd record -e sched:sched_switch -e sched:sched_wakeup sleep 5
sudo trace-cmd report
```

#### Why ftrace matters for real-time:

In an OKS-style robot system, if your 100Hz control loop occasionally takes
15ms instead of 10ms, ftrace can show you:
- Which kernel thread preempted your RT thread
- How long the preemption lasted
- Whether it was a scheduling issue or I/O stall

```bash
# Trace scheduling of your RT thread
sudo trace-cmd record -e sched:sched_switch \
    -f 'next_comm == "my_rt_thread" || prev_comm == "my_rt_thread"' \
    sleep 10
```

---

### 3.5 eBPF & bpftrace

eBPF (extended Berkeley Packet Filter) runs sandboxed programs in the kernel.
bpftrace provides a one-liner scripting interface:

```bash
# Install
sudo apt install bpftrace  # Ubuntu 20.04+

# Count syscalls by type
sudo bpftrace -e 'tracepoint:raw_syscalls:sys_enter { @[comm] = count(); }'

# Trace malloc sizes
sudo bpftrace -e 'uprobe:/lib/x86_64-linux-gnu/libc.so.6:malloc {
    @sizes = hist(arg0); }'

# Latency of read() syscall
sudo bpftrace -e '
tracepoint:syscalls:sys_enter_read { @start[tid] = nsecs; }
tracepoint:syscalls:sys_exit_read /@start[tid]/ {
    @read_ns = hist(nsecs - @start[tid]);
    delete(@start[tid]);
}'

# Function call latency in YOUR binary
sudo bpftrace -e 'uprobe:./my_program:process_frame {
    @start[tid] = nsecs;
}
uretprobe:./my_program:process_frame /@start[tid]/ {
    @latency_us = hist((nsecs - @start[tid]) / 1000);
    delete(@start[tid]);
}'
```

**eBPF advantages over strace/ftrace**:
- Near-zero overhead (JIT-compiled in kernel)
- Can aggregate data in-kernel (histograms, counts)
- No context-switch overhead per event
- Programmable (complex logic without dumping everything)

---

## Part 4: Advanced Debugging

### 4.1 GDB — Beyond Basics

Most developers use GDB for breakpoint → step → print. These advanced features
solve problems that basic debugging cannot:

#### Conditional breakpoints

```bash
# Break only when a condition is true
(gdb) break process_frame if frame_id == 42

# Break only after N hits (skip startup)
(gdb) break main.cpp:100
(gdb) ignore 1 1000    # skip first 1000 hits of breakpoint 1

# Break when a variable changes (hardware watchpoint)
(gdb) watch counter     # break on ANY write to counter
(gdb) rwatch buffer[10] # break on ANY read from buffer[10]
(gdb) awatch flags      # break on read OR write

# Watch with condition
(gdb) watch counter if counter > 100
```

#### Watchpoints (hardware-assisted)

Watchpoints are the **killer feature** for finding memory corruption:

```bash
# Find who overwrites a variable
(gdb) break main
(gdb) run
(gdb) watch *(int*)0x7fffffffde4c    # watch specific address
(gdb) continue
# GDB stops at the EXACT instruction that modifies the address

# Watch a struct member
(gdb) watch my_object.state_
# Stops whenever state_ changes, shows old and new values
```

Hardware watchpoints use CPU debug registers (limited to 4 on x86).
Software watchpoints work for more locations but are extremely slow.

#### Catchpoints

```bash
# Break on exception throw
(gdb) catch throw
(gdb) catch throw std::runtime_error  # specific type

# Break on exception catch
(gdb) catch catch

# Break on syscall
(gdb) catch syscall write

# Break on fork/exec
(gdb) catch fork
(gdb) catch exec
```

#### Reverse debugging

GDB can record execution and step backwards:

```bash
(gdb) break main
(gdb) run
(gdb) record              # start recording
(gdb) continue            # run until crash/breakpoint
# Now you can go BACKWARDS:
(gdb) reverse-continue    # run backwards until previous breakpoint
(gdb) reverse-step        # step backwards one line
(gdb) reverse-next        # step backwards over function calls
(gdb) reverse-finish      # run backwards until function entry

# Find when a variable was last changed
(gdb) watch -l my_var
(gdb) reverse-continue    # finds previous write to my_var
```

**Limitation**: Recording slows execution 10-100x. Best used with small
reproduction cases, not full robot systems.

#### Pretty-printers

Make GDB display STL containers and custom types readably:

```bash
# STL pretty printers (usually auto-loaded)
(gdb) print my_vector
# $1 = std::vector of length 3, capacity 4 = {1, 2, 3}

# Custom pretty-printer (Python, in ~/.gdbinit or .gdbinit)
```

```python
# ~/.gdbinit or project .gdbinit
import gdb.printing

class PosePrinter:
    """Pretty-print Pose2D(x, y, theta)"""
    def __init__(self, val):
        self.val = val

    def to_string(self):
        x = float(self.val['x_'])
        y = float(self.val['y_'])
        theta = float(self.val['theta_'])
        return f"Pose2D(x={x:.3f}, y={y:.3f}, θ={theta:.4f})"

def build_printer():
    pp = gdb.printing.RegexpCollectionPrettyPrinter("my_project")
    pp.add_printer('Pose2D', '^Pose2D$', PosePrinter)
    return pp

gdb.printing.register_pretty_printer(gdb.current_objfile(), build_printer())
```

#### Remote debugging (for robot targets)

```bash
# On the robot (target)
gdbserver :2345 ./my_node

# On your dev machine
gdb ./my_node
(gdb) target remote robot_ip:2345
(gdb) break main
(gdb) continue
```

---

### 4.2 Core Dump Analysis

When a program crashes, the kernel can save a core dump — a snapshot of
the process's memory at the moment of death:

```bash
# Enable core dumps
ulimit -c unlimited

# Set core dump pattern (system-wide)
echo '/tmp/core.%e.%p.%t' | sudo tee /proc/sys/kernel/core_pattern

# Run the crashing program
./my_program    # → segfault → /tmp/core.my_program.12345.1619280000

# Analyze with GDB
gdb ./my_program /tmp/core.my_program.12345.1619280000
(gdb) bt                  # backtrace — shows where it crashed
(gdb) frame 3             # switch to frame 3 in the backtrace
(gdb) info locals         # show local variables in that frame
(gdb) print *this         # if inside a member function
(gdb) info threads        # show all threads at crash time
(gdb) thread 2            # switch to thread 2
(gdb) bt                  # backtrace for thread 2
```

#### Automated core dump analysis in CI:

```bash
#!/bin/bash
# Run tests, check for core dumps
ulimit -c unlimited
./run_tests

for core in /tmp/core.*; do
    echo "=== CRASH DETECTED ==="
    gdb -batch -ex "bt full" -ex "info threads" -ex "thread apply all bt" \
        ./my_test_binary "$core"
done
```

---

### 4.3 Post-Mortem Tools

When you don't have GDB or a core dump, these tools extract information from
the binary itself:

```bash
# addr2line — convert address to file:line
addr2line -e ./my_program -f 0x4011a3
# process_frame
# /home/user/src/main.cpp:42

# nm — list symbols
nm ./my_program | grep ' T '    # exported (Text) symbols
nm ./my_program | grep process  # find a specific symbol
nm -C ./my_program              # demangle C++ names

# objdump — disassembly
objdump -d ./my_program | less
objdump -d -S ./my_program      # interleave source (needs -g)

# readelf — ELF header info
readelf -h ./my_program         # file header
readelf -S ./my_program         # section headers
readelf --debug-dump=line ./my_program  # line number info

# c++filt — demangle a single symbol
echo '_ZN5MyApp12process_dataERKSt6vectorIiSaIiEE' | c++filt
# MyApp::process_data(std::vector<int, std::allocator<int>> const&)
```

---

## Part 5: Putting It All Together

### 5.1 CI Pipeline Design for Quality Tools

A production CI pipeline should run these tools in parallel:

```yaml
# .github/workflows/quality.yml
name: Quality Pipeline

on: [push, pull_request]

jobs:
  # ── Static analysis (fast, run first) ──
  static-analysis:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: clang-tidy
        run: |
          cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
          clang-tidy -p build/ src/*.cpp
      - name: cppcheck
        run: cppcheck --enable=all --error-exitcode=1 src/

  # ── Sanitizers (parallel, separate jobs) ──
  asan:
    runs-on: ubuntu-latest
    steps:
      - run: cmake -B build -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -g"
      - run: cmake --build build && cd build && ctest

  tsan:
    runs-on: ubuntu-latest
    steps:
      - run: cmake -B build -DCMAKE_CXX_FLAGS="-fsanitize=thread -g"
      - run: cmake --build build && cd build && ctest

  # ── Profiling (nightly, expensive) ──
  profile:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    steps:
      - run: cmake -B build -DCMAKE_BUILD_TYPE=Release
      - run: cmake --build build
      - run: |
          perf stat ./build/my_benchmark
          valgrind --tool=callgrind --callgrind-out-file=callgrind.out ./build/my_benchmark
          # Compare against baseline
```

### 5.2 Suppression Management for Large Codebases

In a real robot codebase (ROS + custom code + third-party), expect noise:

```
project/
├── sanitizer/
│   ├── asan.supp           # ASan suppressions
│   ├── tsan.supp           # TSan suppressions
│   └── lsan.supp           # Leak suppressions
├── .clang-tidy             # clang-tidy config
├── cppcheck.supp           # cppcheck suppressions
└── CMakeLists.txt          # Sanitizer targets
```

**Rule**: Every suppression MUST have a comment explaining:
1. Why it's suppressed (false positive? third-party? benign?)
2. A link to the upstream issue if applicable
3. When it can be removed

```
# tsan.supp
# Benign race in ros::init() counter — reported upstream as ros/ros_comm#2134
race:ros::init
# False positive: atomic<bool> with relaxed ordering — TSan doesn't model this
race:StatusFlags::is_ready
# Third-party: libcurl internal threading — we can't fix this
race:libcurl*
```

### 5.3 Tool Selection Quick Reference

| Symptom | First tool | Second tool | Third tool |
|---------|-----------|------------|-----------|
| Crash (segfault) | Core dump + GDB | ASan rebuild | Valgrind memcheck |
| Data corruption | ASan | GDB watchpoints | UBSan |
| Race condition | TSan | GDB with thread commands | LTTng |
| Deadlock | TSan | `strace -e futex` | GDB `info threads` |
| Slow execution | `perf stat` | `perf record` → flamegraph | Callgrind |
| Cache misses | `perf stat` cache events | Cachegrind | SoA refactor |
| Syscall overhead | `strace -c` | `bpftrace` | Buffering |
| Memory leak | ASan (LSan) | Valgrind memcheck | `massif` |
| Undefined behavior | UBSan | ASan | Code review |
| Coding standard | clang-tidy | cppcheck | `-Wall -Werror` |

---

## Exercises in This Module

| Exercise | Focus | Tools |
|----------|-------|-------|
| ex01 | Coding standards violations — detect and fix | clang-tidy, compiler warnings |
| ex02 | Static analysis traps — patterns that hide bugs | cppcheck, code review |
| ex03 | Cache profiling — AoS vs SoA, row vs column | perf stat, cachegrind |
| ex04 | Call-graph hotspot — find the bottleneck | callgrind, perf record |
| ex05 | Syscall audit — reduce I/O overhead | strace -c |
| ex06 | Tracepoint framework — build LTTng-style tracing | LTTng concepts |
| ex07 | Watchpoint hunting — find memory corruption | GDB watchpoints |
| ex08 | Flamegraph-driven optimization | perf record + flamegraph |
| puzzle01 | The observant profiler — cache line effects | perf + false sharing |
| puzzle02 | The invisible allocation — RT latency | ltrace + perf |
| puzzle03 | The lying benchmark — measurement traps | perf stat |
