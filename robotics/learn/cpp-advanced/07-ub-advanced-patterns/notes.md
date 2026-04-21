# Week 7: Undefined Behavior Deep Dive & Advanced Patterns

## 1. The 20 Most Dangerous Forms of Undefined Behavior

Undefined Behavior (UB) is not "implementation-defined" or "unspecified" — it means the C++ standard imposes **no requirements whatsoever** on what happens. The compiler is free to assume UB never occurs and optimize accordingly. This is not theoretical: modern compilers aggressively exploit UB assumptions for optimization, and the results can be catastrophic.

### Category A: Memory UB

**UB-01: Use-After-Free.** Accessing heap memory after `delete` or `free()`. The allocator may have returned that memory to someone else, or the page may be unmapped. With AddressSanitizer (ASan), this is caught immediately. Without it, you get silent corruption or a delayed crash hundreds of lines later. [CWE-416]

**UB-02: Buffer Overflow (Out-of-Bounds Access).** Reading or writing past the end of an array or buffer. This is the single most exploited vulnerability class in history. Stack buffer overflows enable return-address overwrites; heap overflows corrupt allocator metadata. `operator[]` on `std::vector` does NOT bounds-check; use `.at()` or ASan. [CWE-787]

**UB-03: Null Pointer Dereference.** Dereferencing a null pointer is UB, not a guaranteed segfault. The compiler may remove null checks if it can prove the pointer was already dereferenced (see Puzzle 1). On embedded systems without an MMU, null deref silently reads address 0x0. [CWE-476]

**UB-04: Dangling Reference.** A reference bound to a local variable that has gone out of scope. Unlike pointers, references cannot be null-checked — they're assumed valid. Returning `const std::string&` from a function that builds the string locally is a classic instance.

**UB-05: Uninitialized Read.** Reading from an uninitialized variable of automatic storage duration. The compiler may propagate "undef" through the entire computation, making ALL downstream values undefined. MemorySanitizer (MSan) catches this, but it's the hardest sanitizer to deploy because it requires an instrumented libc.

### Category B: Arithmetic UB

**UB-06: Signed Integer Overflow.** `INT_MAX + 1` is UB for signed integers (but well-defined wrapping for unsigned). GCC with `-O2` will assume signed overflow never happens and may transform `if (x + 1 > x)` into `if (true)`. Use `__builtin_add_overflow()` or `<numeric>` safe arithmetic.

**UB-07: Division by Zero.** `x / 0` and `x % 0` for integer types. Floating-point division by zero is NOT UB — it produces `inf` or `NaN` per IEEE 754. Integer division by zero may raise SIGFPE on x86 or silently produce garbage on ARM.

**UB-08: Shift UB.** Shifting by a negative amount, shifting by ≥ the bit-width, or shifting a negative signed value left. `1 << 32` is UB on a 32-bit int. `1 << 31` was UB in C++17 and earlier; C++20 mandates two's complement, making it well-defined as `INT_MIN`.

### Category C: Type System UB

**UB-09: Strict Aliasing Violation.** Accessing an object through a pointer of incompatible type. The classic: casting `float*` to `int*` to inspect the bit pattern. The compiler assumes pointers to different types don't alias, enabling powerful optimizations. Use `std::memcpy` or `std::bit_cast` (C++20) instead.

**UB-10: Type Punning via Union.** In C99/C11, reading a different union member than last written is explicitly permitted (not just implementation-defined) as long as the value is a valid representation. In C++, it's UB (with narrow exceptions for common initial sequences). Use `std::bit_cast` or `memcpy`.

**UB-11: Alignment Violation.** Accessing data through a misaligned pointer. `reinterpret_cast<int*>(char_ptr + 1)` may work on x86 (with a performance penalty) but will SIGBUS on ARM/RISC-V. Use `alignas` and `std::assume_aligned` (C++20).

### Category D: Sequence & Evaluation UB

**UB-12: Unsequenced Modification.** `i = i++ + ++i` — modifying a scalar variable twice without an intervening sequence point (C++17 refined this with "sequenced before" rules, but many forms remain UB). Rule: never modify and read the same variable in one expression without a clear sequence.

**UB-13: Order of Evaluation of Function Arguments.** `f(g(), h())` — the order of `g()` and `h()` is unspecified (not UB, but surprising). If `g()` and `h()` have side effects on shared state, results vary between compilers. C++17 fixed some cases (e.g., `a.b(c)` evaluates `a.b` before `c`).

### Category E: Lifetime UB

**UB-14: Returning Reference to Local.** `int& f() { int x = 42; return x; }` — the reference dangles immediately. GCC warns (`-Wreturn-local-addr`), but templates can hide this.

**UB-15: Temporary Lifetime Extension Failure.** `const std::string& r = get_string();` extends the temporary's lifetime ONLY if `get_string()` returns by value. If it returns `const std::string&`, no extension happens and `r` may dangle. See Puzzle 2 for detailed scenarios.

**UB-16: Use After `std::move`.** Reading a moved-from object is not technically UB (it's in a "valid but unspecified" state), but calling anything except assignment or destruction on it is usually a bug. For `std::unique_ptr`, the moved-from state IS specified (null), but for `std::string` it could be empty or contain the original data.

### Category F: Miscellaneous Lethal UB

**UB-17: Infinite Loop Without Side Effects.** A loop like `while (true) {}` with no observable side effects (no I/O, no volatile access, no atomics) is UB in C++11 and later. The compiler may assume it terminates and delete it entirely.

**UB-18: Violating `[[noreturn]]` by Returning.** If a function marked `[[noreturn]]` does return, the behavior is undefined.

**UB-19: Double Free / Double Delete.** Freeing the same memory twice corrupts the allocator's internal data structures. Can manifest as a crash in a completely unrelated allocation. ASan catches this instantly.

**UB-20: ODR Violation (One Definition Rule).** Having two different definitions of the same entity across translation units. The linker picks one silently. Different optimization levels, different `#defines`, or template specializations in different TUs can trigger this. No sanitizer catches it at runtime — only careful code review and `-flto` help.

---

## 2. CRTP: Curiously Recurring Template Pattern

CRTP enables **static polymorphism** — dispatching to derived class methods at compile time, with zero runtime overhead (no vtable, no indirect call).

```cpp
template <typename Derived>
struct SensorBase {
    double read() { return static_cast<Derived*>(this)->read_impl(); }
};

struct IMU : SensorBase<IMU> {
    double read_impl() { return 9.81; }
};
```

**When to use CRTP vs virtual:**
- CRTP: hot path, embedded, real-time, millions of calls per second, type known at compile time
- Virtual: plugin systems, runtime configuration, heterogeneous containers, ABI stability

The key trade-off: CRTP requires the concrete type at compile time, so you can't store `SensorBase<IMU>` and `SensorBase<Lidar>` in the same container without type erasure.

**CRTP Mixin pattern** — adding behavior to classes via CRTP base:

```cpp
template <typename Derived>
struct Counted {
    static inline int count = 0;
    Counted() { ++count; }
    ~Counted() { --count; }
    static int instance_count() { return count; }
};

struct Widget : Counted<Widget> {};  // Widget now has instance_count()
```

---

## 3. Policy-Based Design

Alexandrescu's policy-based design uses templates to compose behavior from small, orthogonal policy classes:

```cpp
template <typename IntegrationPolicy, typename SaturationPolicy>
class PIDController : private IntegrationPolicy, private SaturationPolicy {
    double compute(double error, double dt) {
        integral_ = IntegrationPolicy::integrate(integral_, error, dt);
        integral_ = SaturationPolicy::clamp(integral_, -limit_, limit_);
        return kp_ * error + ki_ * integral_;
    }
};
```

Each policy is a small struct with a single static or non-virtual method. The compiler inlines everything — zero overhead compared to hand-written code, but fully composable.

This is the C++ answer to the Strategy pattern when runtime selection isn't needed.

---

## 4. Structured Logging for Production & Real-Time

Traditional logging (`printf`, `spdlog`, `std::cout`) is **not RT-safe** because:
- Format strings cause allocation
- File I/O causes syscalls (unbounded latency)
- Mutex contention from multiple threads

The RT-safe pattern: **SPSC ring buffer with deferred formatting**.

```
┌──────────────────────────────────────────────────────────────┐
│                 RT-SAFE LOGGING PIPELINE                      │
│                                                              │
│  RT Thread          Ring Buffer           Background Thread  │
│  ┌─────────┐       ┌──────────────┐      ┌───────────────┐  │
│  │ record() │──────▶│ TraceBuffer  │─────▶│ drain_and_    │  │
│  │ (no      │ push  │ {ts, id,     │ pop  │ format()      │  │
│  │  alloc,  │       │  payload}    │      │ (allocation   │  │
│  │  no      │       │              │      │  OK here)     │  │
│  │  syscall)│       │ Atomic idx   │      │               │  │
│  └─────────┘       └──────────────┘      └───────┬───────┘  │
│                                                   │          │
│                                           ┌───────▼───────┐  │
│                                           │  File / Net   │  │
│                                           │  (non-RT)     │  │
│                                           └───────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

Key design:
- `TraceBuffer<N>`: fixed-size circular buffer, power-of-2 size for fast modulo
- Each entry: `{uint64_t timestamp_ns, uint16_t event_id, uint64_t payload}` — 18 bytes, padded to 32
- `record()`: single atomic fetch_add on write index, no mutex, no allocation
- `dump()`: copy the buffer, sort by timestamp, format offline
- `LogLevel` as `std::atomic<int>`: check level before recording, changeable at runtime

This is the "flight data recorder" pattern used in avionics and robotics: always capture the last N thousand events, and when something goes wrong, you have the trace.

---

## 5. How Compilers Exploit UB for Optimization

This is the crucial insight: UB is not just "anything might happen" — it's **a contract between you and the compiler**. You promise certain things never happen; the compiler uses those promises to generate faster code.

```
┌─────────────────────────────────────────────────────────────┐
│           COMPILER UB EXPLOITATION PIPELINE                  │
│                                                             │
│  Source Code                                                │
│      │                                                      │
│      ▼                                                      │
│  ┌──────────────────────────────────┐                       │
│  │ 1. PARSE & BUILD IR             │                       │
│  │    (SSA form, all UB present)   │                       │
│  └──────────────┬───────────────────┘                       │
│                 ▼                                            │
│  ┌──────────────────────────────────┐                       │
│  │ 2. ASSUME UB NEVER HAPPENS      │                       │
│  │                                  │                       │
│  │  • signed overflow → impossible  │  ◀── This is where   │
│  │  • null deref → impossible       │      your bugs become │
│  │  • aliasing rules hold           │      "optimizations"  │
│  │  • loops terminate               │                       │
│  └──────────────┬───────────────────┘                       │
│                 ▼                                            │
│  ┌──────────────────────────────────┐                       │
│  │ 3. OPTIMIZE BASED ON ASSUMPTIONS│                       │
│  │                                  │                       │
│  │  • Delete "dead" null checks    │                       │
│  │  • Vectorize "bounded" loops    │                       │
│  │  • Reorder aliased loads/stores │                       │
│  │  • Fold "can't overflow" math   │                       │
│  └──────────────┬───────────────────┘                       │
│                 ▼                                            │
│  ┌──────────────────────────────────┐                       │
│  │ 4. EMIT MACHINE CODE            │                       │
│  │    (runs "wrong" if UB existed) │                       │
│  └──────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────┘
```

### Concrete Example 1: Null Check Elimination

```cpp
void process(int* p) {
    int x = *p;        // Dereference — compiler assumes p != null
    if (p == nullptr)  // Dead code! Compiler proved p != null above
        handle_error();
    use(x);
}
```

At `-O2`, GCC/Clang will **delete the entire `if` block**. The dereference on line 2 is only valid if `p` is non-null, so the compiler assumes it is, making the null check on line 3 provably false.

### Concrete Example 2: Signed Overflow Loop Transformation

```cpp
bool check(int x) {
    return x + 1 > x;  // Always true if no overflow
}
// GCC -O2 compiles this to: return true;
```

Since signed overflow is UB, the compiler assumes `x + 1` never overflows, therefore `x + 1 > x` is always true. The function becomes `return true;` — a single `mov eax, 1; ret`.

### Concrete Example 3: Strict Aliasing Load Elimination

```cpp
void update(int* ip, float* fp) {
    *ip = 42;
    *fp = 3.14f;
    printf("%d\n", *ip);  // Compiler can print 42 without reloading
}
```

Since `int*` and `float*` cannot alias (strict aliasing rule), the compiler knows the store to `*fp` cannot affect `*ip`. It skips the reload of `*ip` and directly uses the value `42`. If `ip` and `fp` actually point to the same memory (via a cast), the printed value is `42` not the bit pattern of `3.14f` — silent, terrifying wrongness.

---

## Summary

| Topic | Key Takeaway |
|-------|-------------|
| Memory UB | Use ASan + MSan in CI; never ship without sanitizer runs |
| Arithmetic UB | Use unsigned for wrapping; builtins for checked arithmetic |
| Type UB | Use `std::bit_cast` / `memcpy` — never raw casts for type punning |
| CRTP | Static polymorphism for hot paths; virtual for flexibility |
| Policy Design | Compile-time strategy pattern — zero overhead composition |
| RT-safe logging | SPSC ring buffer; format offline; no allocation on hot path |
| Compiler + UB | The compiler is your adversary if you write UB — it will exploit every loophole |

**Build command for this week's exercises:**
```bash
cd exercises && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)
```

Run with sanitizers: each exercise has a sanitizer target (e.g., `ex01_ub_catalog_asan`).
