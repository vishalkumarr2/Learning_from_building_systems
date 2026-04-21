# Module 16: Sanitizer Workshop — Practical Bug-Hunting with C++ Sanitizers

## Overview

Sanitizers are compiler-based tools that instrument your code at compile time and detect
bugs at runtime. They are the single most effective bug-finding tool for C++ code —
faster and more comprehensive than Valgrind, and used by every major C++ project
(Chrome, LLVM, Firefox, Linux kernel test suites).

**Key insight**: Sanitizers catch bugs at RUNTIME. You need tests that exercise the
code paths. A sanitizer with no test coverage catches nothing.

---

## The Sanitizer Family

| Sanitizer | Flag | Catches | Overhead | GCC | Clang |
|-----------|------|---------|----------|-----|-------|
| **ASan** (AddressSanitizer) | `-fsanitize=address` | Memory errors | ~2x | ✓ | ✓ |
| **UBSan** (UndefinedBehaviorSanitizer) | `-fsanitize=undefined` | Undefined behavior | ~minimal | ✓ | ✓ |
| **TSan** (ThreadSanitizer) | `-fsanitize=thread` | Data races | ~5-15x | ✓ | ✓ |
| **MSan** (MemorySanitizer) | `-fsanitize=memory` | Uninitialized reads | ~3x | ✗ | ✓ |
| **LSan** (LeakSanitizer) | Included in ASan | Memory leaks | ~minimal | ✓ | ✓ |

**Critical constraint**: ASan and TSan cannot be combined (different runtimes).
Run them as separate CI jobs.

---

## How Sanitizers Work

1. **Compile-time instrumentation**: The compiler inserts extra checks around memory
   accesses, arithmetic operations, thread synchronization, etc.
2. **Runtime library**: A runtime component (linked into the binary) manages shadow
   memory, tracks allocations, monitors thread operations.
3. **Error reporting**: When a violation is detected, the sanitizer prints a detailed
   report with stack traces and terminates (or continues, depending on configuration).

This is fundamentally different from Valgrind, which uses binary translation (no
recompilation needed, but 10-50x slower).

---

## ASan (AddressSanitizer)

### What it catches:
- **Heap-buffer-overflow**: Writing/reading past the end of a `malloc`'d / `new`'d buffer
- **Stack-buffer-overflow**: Writing/reading past stack arrays
- **Use-after-free**: Accessing memory after `delete` / `free`
- **Use-after-scope**: Reference to local variable outlives its scope
- **Double-free**: Calling `free`/`delete` on the same pointer twice
- **Memory leaks** (via integrated LSan): Allocations with no corresponding free
- **Use-after-return**: Returning pointer to stack-allocated memory (with runtime flag)
- **Container-overflow**: Out-of-bounds access on STL containers (with annotations)

### Compiler flags:
```bash
g++ -std=c++2a -fsanitize=address -fno-omit-frame-pointer -g -O1 source.cpp -o binary
```

- `-fno-omit-frame-pointer`: Ensures accurate stack traces
- `-g`: Debug info for line numbers in reports
- `-O1`: Recommended optimization level (catches more than `-O0`, less noise than `-O2`)

### Runtime options (environment variables):
```bash
export ASAN_OPTIONS="detect_stack_use_after_return=1:detect_leaks=1:halt_on_error=0"
```

Key options:
- `detect_stack_use_after_return=1` — enables use-after-return detection (extra overhead)
- `detect_leaks=1` — enable leak checking (default on Linux)
- `halt_on_error=0` — continue after first error (find multiple bugs in one run)
- `malloc_context_size=30` — deeper stack traces for allocations
- `suppressions=/path/to/asan.supp` — suppress known false positives

### How ASan works (shadow memory):
ASan maps every 8 bytes of application memory to 1 byte of "shadow memory".
The shadow byte encodes which of the 8 bytes are accessible. Each `malloc` adds
"redzones" around the allocation (poisoned shadow bytes). Each `free` poisons
the entire region and adds it to a quarantine (delayed reuse).

---

## UBSan (UndefinedBehaviorSanitizer)

### What it catches:
- **Signed integer overflow**: `INT_MAX + 1`
- **Shift exponent errors**: Shift by negative or >= bit width
- **Null pointer dereference**: Dereferencing `nullptr`
- **Division by zero**: Integer division by zero
- **Alignment violations**: Accessing data through misaligned pointers
- **Float-cast overflow**: Floating point value too large for target integer type
- **Unreachable code**: Reaching `__builtin_unreachable()`
- **Invalid bool/enum**: Loading invalid values for bool or enum types
- **VLA bound**: Non-positive VLA size

### Compiler flags:
```bash
g++ -std=c++2a -fsanitize=undefined -g source.cpp -o binary
```

### Sub-sanitizers (can be enabled individually):
```bash
-fsanitize=signed-integer-overflow
-fsanitize=null
-fsanitize=alignment
-fsanitize=shift
-fsanitize=float-cast-overflow
-fsanitize=vla-bound
```

### Runtime options:
```bash
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1"
```

- `print_stacktrace=1` — include stack trace in error reports
- `halt_on_error=1` — abort on first UB (default is to continue)

### UBSan + ASan combination:
UBSan can be combined with ASan:
```bash
g++ -fsanitize=address,undefined -fno-omit-frame-pointer -g source.cpp
```

---

## TSan (ThreadSanitizer)

### What it catches:
- **Data races**: Two threads access the same memory location, at least one is a write,
  with no synchronization between them
- **Lock-order inversion**: Thread 1 locks A then B, Thread 2 locks B then A (deadlock)
- **Signal-unsafe code**: Calling non-async-signal-safe functions in signal handlers

### Compiler flags:
```bash
g++ -std=c++2a -fsanitize=thread -g -O1 source.cpp -o binary -pthread
```

**IMPORTANT**: Must compile ALL source files with `-fsanitize=thread`, including libraries.
Mixing instrumented and non-instrumented code produces false positives.

### Runtime options:
```bash
export TSAN_OPTIONS="second_deadlock_stack=1:halt_on_error=0:history_size=7"
```

- `second_deadlock_stack=1` — show both stacks in deadlock reports
- `halt_on_error=0` — continue after finding a race
- `history_size=7` — more memory for race detection (0-7, higher = more memory)
- `suppressions=/path/to/tsan.supp` — suppress known races

### TSan limitations:
- 5-15x slowdown, 5-10x memory overhead
- Cannot combine with ASan (different memory layouts)
- Maximum 8192 threads
- May miss races that don't manifest in a particular execution

### Suppression file format:
```
# tsan.supp
race:third_party_lib::SomeFunction
race:legacy_module.cpp
deadlock:known_safe_lock_pattern
```

---

## MSan (MemorySanitizer) — Clang Only

MSan detects use of uninitialized memory. **Not available in GCC.**

```bash
clang++ -fsanitize=memory -fno-omit-frame-pointer -g source.cpp -o binary
```

MSan requires ALL linked libraries to be instrumented. This makes it the hardest
sanitizer to deploy (you need to build libc++, etc. with MSan).

---

## LSan (LeakSanitizer)

LSan is integrated into ASan and runs at program exit:
```bash
g++ -fsanitize=address source.cpp -o binary
ASAN_OPTIONS=detect_leaks=1 ./binary
```

Or standalone (Clang only):
```bash
clang++ -fsanitize=leak source.cpp -o binary
```

---

## Sanitizer Suppressions

When you have false positives or third-party code you can't fix:

**ASan suppression file** (`asan.supp`):
```
interceptor_via_fun:third_party_function
leak:ThirdPartyLib::Allocate
```

**TSan suppression file** (`tsan.supp`):
```
race:ThirdPartyLib*
mutex:AnnotateMutex
```

**UBSan suppression file** — use function attribute instead:
```cpp
__attribute__((no_sanitize("undefined")))
void legacy_function_with_known_ub() { ... }
```

Apply at runtime:
```bash
ASAN_OPTIONS=suppressions=asan.supp ./binary
TSAN_OPTIONS=suppressions=tsan.supp ./binary
```

---

## Integration with CI/CD

Best practice: run each sanitizer as a separate CI job:

```yaml
# .github/workflows/sanitizers.yml
jobs:
  asan:
    runs-on: ubuntu-latest
    steps:
      - run: cmake -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer" ..
      - run: make && ctest

  ubsan:
    runs-on: ubuntu-latest
    steps:
      - run: cmake -DCMAKE_CXX_FLAGS="-fsanitize=undefined" ..
      - run: make && ctest

  tsan:
    runs-on: ubuntu-latest
    steps:
      - run: cmake -DCMAKE_CXX_FLAGS="-fsanitize=thread" ..
      - run: make && ctest
```

---

## Sanitizers vs Valgrind

| Feature | Sanitizers | Valgrind |
|---------|-----------|----------|
| Speed | 2-15x overhead | 10-50x overhead |
| Recompilation needed | Yes | No |
| Memory errors | ASan | Memcheck |
| Thread errors | TSan | Helgrind/DRD |
| UB detection | UBSan | Limited |
| Uninitialized reads | MSan (Clang) | Memcheck |
| Platform | Linux, macOS, others | Linux primarily |
| False positives | Very rare | Occasional |
| CI suitability | Excellent (fast) | Poor (slow) |

**Rule of thumb**: Use sanitizers for CI, Valgrind for one-off deep analysis of
existing binaries you can't recompile.

---

## ROS Integration

Running ROS nodes under sanitizers:

```bash
# Build your ROS package with sanitizers
catkin_make -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer -g"

# Run a node with ASan
ASAN_OPTIONS="detect_leaks=0:halt_on_error=0" rosrun my_package my_node

# Suppress noise from ROS libraries
ASAN_OPTIONS="suppressions=$HOME/ros_asan.supp" rosrun my_package my_node
```

Common suppressions needed for ROS:
```
# ros_asan.supp
leak:ros::init
leak:ros::NodeHandle
leak:pluginlib
```

For TSan with ROS, many races in ROS infrastructure are benign but noisy.
Use a suppression file and focus on YOUR code's races.

---

## Real-World Catches

Sanitizers have found critical bugs in major projects:

- **Chrome**: Thousands of security vulnerabilities found by ASan/MSan in continuous fuzzing
- **LLVM**: Regular ASan/UBSan runs catch bugs before they reach releases
- **Firefox**: TSan found data races that caused intermittent crashes
- **Linux kernel**: KASAN (kernel ASan) catches slab-out-of-bounds, use-after-free in drivers
- **ROS 2**: TSan revealed races in rmw implementations and executor scheduling

---

## Limitations

1. **Runtime only**: Sanitizers only catch bugs in code paths that actually execute.
   Dead code bugs are invisible. You need good test coverage.
2. **Not combinable**: ASan + TSan cannot coexist. Run separately.
3. **MSan is Clang-only**: GCC users miss uninitialized reads (use Valgrind instead).
4. **Performance overhead**: TSan's 5-15x makes it impractical for real-time systems.
5. **Cannot catch logic errors**: Code that is technically correct but semantically wrong.
6. **ABA problem**: Lock-free algorithms with ABA issues are invisible to TSan.
7. **Signed overflow wrapping**: If overflow is intentional (hash functions), use
   `-fwrapv` or cast to unsigned.

---

## Quick Reference: Compilation Commands

```bash
# ASan (memory bugs)
g++ -std=c++2a -fsanitize=address -fno-omit-frame-pointer -g -O1 src.cpp -o bin_asan

# UBSan (undefined behavior)
g++ -std=c++2a -fsanitize=undefined -g src.cpp -o bin_ubsan

# TSan (data races) — needs -pthread
g++ -std=c++2a -fsanitize=thread -g -O1 src.cpp -o bin_tsan -pthread

# ASan + UBSan combined
g++ -std=c++2a -fsanitize=address,undefined -fno-omit-frame-pointer -g src.cpp -o bin_combo

# Run with options
ASAN_OPTIONS="detect_leaks=1:halt_on_error=0" ./bin_asan
UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1" ./bin_ubsan
TSAN_OPTIONS="second_deadlock_stack=1" ./bin_tsan
```

---

## Exercises in This Module

| Exercise | Sanitizer | Focus |
|----------|-----------|-------|
| ex01 | ASan | 8 classic memory bugs |
| ex02 | UBSan | 8 undefined behavior patterns |
| ex03 | TSan | 5 data races + deadlock |
| ex04 | All three | Realistic mini-project, find & fix |
| ex05 | Detective | Real-world-like subtle bugs |
| puzzle01 | TSan | False positive + suppression |
| puzzle02 | None | What sanitizers CAN'T catch |
