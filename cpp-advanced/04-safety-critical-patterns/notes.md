# Week 4: Safety-Critical Patterns & JPL Rules

## Overview

Safety-critical software is code where failure means loss of life, destruction of expensive hardware, or catastrophic environmental damage. This week we study the patterns, rules, and disciplines that make C++ viable in these domains — and the disasters that happen when they're ignored.

The key insight: **safety-critical code is not about being clever. It's about being provably boring.**

---

## 1. JPL Power of 10 Rules — Applied to C++

Gerard Holzmann at NASA's Jet Propulsion Laboratory distilled decades of flight software experience into 10 rules. These rules are deliberately restrictive — they trade expressiveness for verifiability.

### Rule 1: Restrict All Code to Very Simple Control Flow — No Goto, No Recursion

**What it prevents:** Unbounded stack growth, inability to prove termination, difficulty in static analysis.

**C++ application:** Ban `goto`, `longjmp`, and all recursive functions. Every call graph must be a DAG. Convert recursive algorithms to iterative versions using explicit stacks.

```cpp
// BANNED: recursive fibonacci
int fib(int n) { return n <= 1 ? n : fib(n-1) + fib(n-2); }

// COMPLIANT: iterative with bounded loop
int fib(int n) {
    assert(n >= 0 && n <= 46); // prevent overflow in int32
    int a = 0, b = 1;
    for (int i = 0; i < n; ++i) {
        int tmp = a + b;
        a = b;
        b = tmp;
    }
    return a;
}
```

**Why recursion kills:** In the Mars Pathfinder mission (1997), a priority inversion bug led to unexpected stack depths. JPL's rule exists because you cannot bound stack usage when recursion depth depends on runtime data.

### Rule 2: All Loops Must Have Fixed Upper Bounds

**What it prevents:** Infinite loops, livelock, inability to compute WCET (Worst Case Execution Time).

**C++ application:** Every `for` and `while` must have a provable upper bound. Use countdown variables or explicit iteration limits.

```cpp
// BANNED: unbounded retry
while (!device.ready()) { /* spin forever? */ }

// COMPLIANT: bounded retry
constexpr int MAX_RETRIES = 1000;
for (int attempt = 0; attempt < MAX_RETRIES; ++attempt) {
    if (device.ready()) break;
}
```

### Rule 3: No Dynamic Memory Allocation After Initialization

**What it prevents:** Heap fragmentation, out-of-memory in long-running systems, non-deterministic allocation time.

**C++ application:** All `new`/`malloc` must happen during startup. After init completes, use PMR (polymorphic memory resources) with monotonic or pool allocators backed by pre-allocated buffers.

```cpp
#include <memory_resource>
alignas(64) std::byte buffer[65536]; // pre-allocated at startup
std::pmr::monotonic_buffer_resource pool(buffer, sizeof(buffer));
// After init: all allocations come from pool, never from heap
std::pmr::vector<SensorReading> readings(&pool);
```

### Rule 4: No Function Should Be Longer Than 60 Lines (Printed on Paper)

**What it prevents:** Functions too complex to review, test, or reason about.

**C++ application:** Strict limit. One function = one responsibility. If you need a scroll bar to read it, it's too long.

### Rule 5: Use a Minimum of Two Runtime Assertions Per Function (On Average)

**What it prevents:** Silent corruption, undefined behavior propagating silently.

**C++ application:** Assertions are **executable specifications**, not debug aids. They document what must be true and crash loudly when violated. Use `assert()` for debug, `static_assert` for compile-time, and custom `ENSURE`/`REQUIRE` macros for production.

```cpp
void set_motor_speed(int rpm) {
    assert(rpm >= -MAX_RPM && rpm <= MAX_RPM);  // precondition
    // ... implementation ...
    assert(actual_rpm_within_tolerance(rpm));     // postcondition
}
```

### Rule 6: Restrict the Scope of Data to the Smallest Possible

**What it prevents:** Unexpected aliasing, action at a distance, globals causing untraceable state changes.

**C++ application:** No global mutable state. Period. Pass dependencies explicitly. Use `const` everywhere possible. Prefer value semantics.

**Toyota case study connection:** The Toyota unintended acceleration code had **7,134 global variables**. Any function could modify any of them at any time, making reasoning about system state impossible.

### Rule 7: Check Return Values of All Non-Void Functions, or Cast to Void

**What it prevents:** Silently ignored errors, resource leaks, missed failure conditions.

**C++ application:** Every return value must be checked. In C++17, use `[[nodiscard]]` attribute. If you genuinely don't care, explicitly cast to `(void)`.

```cpp
[[nodiscard]] ErrorCode init_sensor();

// BANNED: ignoring return value
init_sensor();

// COMPLIANT:
auto err = init_sensor();
if (err != ErrorCode::OK) handle_error(err);
```

### Rule 8: Limit Preprocessor Use to File Inclusion and Simple Macros

**What it prevents:** Macro side effects, textual substitution bugs, inability to parse code statically.

**C++ application:** Use `constexpr`, `consteval`, `inline`, templates, and `if constexpr` instead of macros. The only acceptable macros: include guards, logging macros with `__FILE__` and `__LINE__`, and assertion macros.

### Rule 9: Restrict Pointer Use — No More Than One Level of Dereferencing

**What it prevents:** Pointer arithmetic bugs, aliasing violations, null dereference chains.

**C++ application:** Prefer references over pointers. No `**ptr`. Use `std::span` for arrays, `std::optional` for nullable values, `std::variant` for unions.

### Rule 10: Compile with All Warnings Enabled; Use Static Analysis on Day One

**What it prevents:** Entire categories of bugs caught before they ever run.

**C++ application:**
```bash
-Wall -Wextra -Wpedantic -Werror -Wshadow -Wconversion
-Wdouble-promotion -Wformat=2 -Wnull-dereference
-fstack-protector-strong -D_FORTIFY_SOURCE=2
```

---

## 2. Toyota Unintended Acceleration — A Case Study in Everything Wrong

In 2009–2010, Toyota vehicles accelerated uncontrollably, killing people. NASA and embedded systems expert Michael Barr investigated. The findings are a catalog of safety-critical anti-patterns:

### The Code

- **7,134 global variables** — any function could mutate shared state with no synchronization
- **Stack overflow undetected** — the stack and heap grew toward each other with no guard page. Stack corruption silently overwrote heap data, including the throttle control variable
- **Single-bit errors unprotected** — critical variables like `targetThrottleAngle` were stored in a single memory location with no redundancy (no CRC, no triple-modular redundancy, no safety variable pattern)
- **Repurposed watchdog** — the watchdog timer existed but was fed ("petted") by a low-priority background task that only proved the RTOS scheduler was running, not that the throttle control task was executing correctly
- **No MISRA compliance** — Toyota claimed MISRA compliance but the code violated rules extensively
- **Spaghetti code** — deeply nested functions, 9000+ line files, no clear module boundaries

### Lessons

1. **A watchdog that doesn't verify actual work is worthless** — Toyota's watchdog proved the scheduler ran, not that the throttle task executed
2. **Global state is the enemy** — 7134 globals made formal verification impossible
3. **Stack must be bounded and monitored** — use MPU/MMU guard pages, paint the stack, check high-water mark
4. **Critical variables must be redundant** — store in 2+ locations, compare before use

---

## 3. Defensive Programming — Design by Contract

Bertrand Meyer's Design by Contract (DBC) formalizes the relationship between caller and callee:

- **Precondition (requires):** What the caller guarantees before the call
- **Postcondition (ensures):** What the callee guarantees after the call
- **Class invariant:** What must be true before and after every public method

### C++ Implementation

C++ doesn't have native DBC (C++20 contracts were removed, C++26 may reintroduce them). We build our own:

```cpp
#define REQUIRE(cond) do { if (!(cond)) { \
    std::cerr << "PRECONDITION FAILED: " #cond \
              << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
    std::abort(); }} while(0)

#define ENSURE(cond) do { if (!(cond)) { \
    std::cerr << "POSTCONDITION FAILED: " #cond \
              << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
    std::abort(); }} while(0)
```

**Key principle:** Assertions are not error handling. They catch programmer mistakes, not runtime errors. A failed assertion means a bug exists in the code. The correct response is to crash loudly and fix the bug, not to "handle" it gracefully.

### Saturating Arithmetic

In safety-critical code, integer overflow is undefined behavior that has killed people (Ariane 5). Saturating arithmetic clamps results to the representable range:

```cpp
int32_t sat_add(int32_t a, int32_t b) {
    int64_t result = static_cast<int64_t>(a) + b;
    if (result > INT32_MAX) return INT32_MAX;
    if (result < INT32_MIN) return INT32_MIN;
    return static_cast<int32_t>(result);
}
```

---

## 4. Static Analysis Tools for C++

### Compiler Flags That Matter

| Flag | What It Catches |
|------|----------------|
| `-Wall -Wextra` | Basic warnings: unused vars, implicit conversions, missing returns |
| `-Wshadow` | Variable shadowing — a frequent source of subtle bugs |
| `-Wconversion` | Implicit narrowing conversions (int→short, double→float) |
| `-Wformat=2` | Format string mismatches (printf family) |
| `-Werror` | Treat warnings as errors — forces clean builds |
| `-fstack-protector-strong` | Stack canaries on functions with local arrays or address-taken vars |
| `-D_FORTIFY_SOURCE=2` | Compile-time and runtime buffer overflow checks for libc functions |
| `-fsanitize=address,undefined` | Runtime detection of memory errors and UB |

### clang-tidy Checks for Safety

```yaml
Checks: >
  bugprone-*,
  cert-*,
  cppcoreguidelines-*,
  misc-*,
  modernize-*,
  performance-*,
  readability-*,
  -modernize-use-trailing-return-type,
  -readability-magic-numbers
```

Critical checks:
- `bugprone-use-after-move` — catches moved-from access
- `bugprone-dangling-handle` — string_view/span pointing to destroyed temporary
- `cppcoreguidelines-pro-bounds-*` — array bounds violations
- `cert-err58-cpp` — static object exceptions during init
- `misc-redundant-expression` — `x == x` (often a typo for `x == y`)

---

## 5. Watchdog Patterns

A watchdog timer (WDT) is a hardware or software timer that must be periodically reset ("petted" or "kicked"). If the timer expires, it means the system is stuck and the watchdog takes corrective action (reset, safe mode, shutdown).

### Hardware Watchdog

Built into the MCU. Cannot be disabled by software (if configured correctly). The most reliable last resort.

```
Timeline:
[pet]---[pet]---[pet]---[HANG]---timeout---[HARDWARE RESET]
```

### Software Watchdog

Implemented in a high-priority timer ISR. Checks that the main application loop completes within a deadline.

**Critical requirement:** The watchdog must verify that **meaningful work** was done, not just that a pet function was called. This is the Toyota lesson.

### Heartbeat Pattern

Each subsystem sends periodic heartbeats. A health monitor checks that all heartbeats arrive within deadlines. If a subsystem misses its deadline:

1. **Warning phase** — log the miss, increment counter
2. **Degraded mode** — disable the failing subsystem, continue with reduced capability
3. **Shutdown** — if too many subsystems fail, enter safe state

```
SubsystemA: [♥]--[♥]--[♥]--[♥]--...                    (healthy)
SubsystemB: [♥]--[♥]--[♥]--[MISS]--[MISS]--[SHUTDOWN]  (failed)
SubsystemC: [♥]--[♥]--[♥]--[♥]--[♥]--...              (healthy)
```

### The Dead Man's Switch

A system must actively prove it's alive. Silence is interpreted as failure. This is the fundamental watchdog principle:

- **Active proof of liveness:** The monitored task must perform a sequence of checks and only then pet the watchdog
- **Verification of progress:** The pet must include evidence of forward progress (e.g., incrementing a work counter)
- **Multiple checkpoint verification:** The watchdog check requires that multiple stages of the processing pipeline have executed

---

## 6. MISRA C++ Overview

MISRA (Motor Industry Software Reliability Association) defines coding rules for safety-critical C/C++. MISRA C++:2023 is the current edition.

### What's Banned and Why

| Rule | Banned | Why |
|------|--------|-----|
| Exceptions | `throw`/`catch` in safety code | Non-deterministic control flow, stack unwinding cost unknown |
| Dynamic allocation | `new`/`delete` after init | Fragmentation, OOM unpredictable |
| RTTI | `dynamic_cast`, `typeid` | Runtime overhead, implementation-defined behavior |
| Multiple inheritance | Diamond inheritance | Ambiguity, layout complexity |
| Union type punning | `union` for type conversion | UB if wrong member read |
| `goto` | All uses | Spaghetti control flow |
| Implicit conversions | Narrowing, sign changes | Silent data loss |
| C-style casts | `(int)x` | Hides dangerous conversions |
| `#define` for constants | Macros vs constexpr | No scope, no type, textual replacement bugs |

### What's Required

- Every `switch` must have a `default` case
- Every non-void function must have a `return` on all paths
- All variables must be initialized at declaration
- Pointer arithmetic is prohibited (use array indexing)
- All loops must have a single exit point (controversial but enforced)

---

## 7. Safety Variable Pattern

For critical control variables, store the value and its bitwise complement:

```cpp
struct SafeVar {
    int32_t value;
    int32_t complement; // must equal ~value

    void set(int32_t v) {
        value = v;
        complement = ~v;
    }

    bool is_valid() const {
        return complement == ~value;
    }

    int32_t get() const {
        assert(is_valid());
        return value;
    }
};
```

If a single-bit flip corrupts `value`, `is_valid()` will detect it. This is what Toyota should have used for `targetThrottleAngle`.

---

## 8. Putting It All Together

Safety-critical C++ is about layers of defense:

1. **Coding rules** (JPL/MISRA) — prevent entire categories of bugs at the source
2. **Static analysis** (clang-tidy, compiler warnings) — catch at compile time what rules miss
3. **Runtime assertions** (DBC) — catch at runtime what static analysis misses
4. **Watchdog monitoring** — detect at system level when a component fails
5. **Safe mode / graceful degradation** — when failure is detected, fail safely

No single layer is sufficient. The Swiss cheese model applies: each layer has holes, but stacking them makes it unlikely a bug passes through all layers.

### The Discipline

Writing safety-critical code is not about talent or cleverness. It's about discipline:
- Follow the rules even when they feel restrictive
- Run the tools even when you're confident
- Add the assertions even when they seem obvious
- Test the failure paths even when they seem unlikely

Because in safety-critical systems, "unlikely" is not "impossible," and impossible is the only acceptable failure rate.
