# Advanced C++ for Real-Time, Safety-Critical & Production Systems
### 8 weeks · 1–2 hrs/day · Hands-on daily · Linux target
### From "knows templates" to "can write flight-software-grade C++"

---

## Philosophy

```
Every day:  READ (20 min) → CODE (40–60 min) → PUZZLE (10–20 min)
Every week: One mini-project that ships a working artifact
```

**No passive reading days.** Every concept has a coding exercise.
**"99% fail" puzzles** are marked with 💀 — these test deep understanding, not syntax.
**ROS tie-ins** are marked with 🤖 — connect to your robot work.

---

## Materials Structure

```
learn/cpp-advanced/
├── STUDY-PLAN.md                          ← YOU ARE HERE
├── README.md                              ← Navigation & content summary
├── build_all.sh                           ← Build all modules
├── 01-move-semantics-value-categories/
│   ├── notes.md
│   ├── exercises/
│   └── puzzles/
├── 02-error-handling-memory-model/
│   ├── notes.md
│   ├── exercises/
│   └── puzzles/
├── 03-rt-linux-programming/
│   ├── notes.md
│   ├── exercises/
│   └── puzzles/
├── 04-safety-critical-patterns/
│   ├── notes.md
│   ├── exercises/
│   └── puzzles/
├── 05-build-test-tooling/
│   ├── notes.md
│   ├── exercises/
│   └── puzzles/
├── 06-ipc-serialization/
│   ├── notes.md
│   ├── exercises/
│   └── puzzles/
├── 07-ub-advanced-patterns/
│   ├── notes.md
│   ├── exercises/
│   └── puzzles/
├── 08-capstone-flight-software/
│   ├── include/flight_sw/   (8 headers)
│   ├── src/main.cpp
│   └── tests/test_main.cpp
├── 09-hardware-lessons/
│   └── exercises/  (hw01–hw05)
├── 10-safety-lessons/
│   └── exercises/  (sf01–sf05)
├── 11-cpp20-coroutines/           ← NEW: requires GCC 10+
│   ├── notes.md
│   ├── exercises/ (ex01–ex04)
│   └── puzzles/
├── 12-cpp20-ranges/               ← NEW: requires GCC 10+
│   ├── notes.md
│   ├── exercises/ (ex01–ex05)
│   └── puzzles/
├── 13-cpp20-format/               ← NEW: requires GCC 13+
│   ├── notes.md
│   ├── exercises/ (ex01–ex04)
│   └── puzzles/
├── 14-exception-safety/           ← NEW: GCC 9+ compatible
│   ├── notes.md
│   ├── exercises/ (ex01–ex05)
│   └── puzzles/
├── 15-pmr-allocators/             ← NEW: GCC 9+ compatible
│   ├── notes.md
│   ├── exercises/ (ex01–ex05)
│   └── puzzles/
├── 16-sanitizer-workshop/         ← NEW: GCC 9+ compatible
│   ├── notes.md
│   ├── exercises/ (ex01–ex05, multi-sanitizer builds)
│   └── puzzles/
└── 17-sdv-mission-critical/       ← NEW: C++20, optional pthread
    ├── notes.md
    ├── exercises/ (ex01–ex08: E2E, degradation, cyclic exec,
    │   triple buffer, DTC, OTA, SecOC, watchdog)
    └── puzzles/
├── 18-coding-standards-tracing/   ← NEW: profiling, LTTng-style tracing
│   ├── notes.md
│   ├── exercises/ (ex01–ex08: standards, static analysis, cache,
│   │   callgraph, syscall, tracepoints, watchpoints, flamegraph)
│   └── puzzles/ (false sharing, invisible alloc, lying benchmark)
```

---

## Key References (free, high-quality)

| Resource | What | URL |
|---|---|---|
| Shuhao Wu RT Linux series | RT app dev on Linux (4 parts) | shuhaowu.com/blog/2022/ |
| Preshing on Programming | Lock-free, memory ordering | preshing.com |
| Regehr UB Guide | Undefined behavior deep dive | blog.regehr.org/archives/213 |
| cppquiz.org | 189 tricky standard questions | cppquiz.org |
| JPL Power of 10 | 10 rules for safety-critical C | spinroot.com/gerard/pdf/P10.pdf |
| JSF AV C++ Rules | Joint Strike Fighter C++ standard | stroustrup.com/JSF-AV-rules.pdf |
| CppCoreGuidelines | ISO C++ guidelines | isocpp.github.io/CppCoreGuidelines |
| cppreference | The standard, readable | en.cppreference.com |
| Godbolt Compiler Explorer | See what the compiler does | godbolt.org |

---

## Week 1 — Move Semantics, Value Categories & Compile-Time C++
*Goal: Write zero-copy, compile-time-evaluated code*

### Day 1: Value categories (lvalue, rvalue, xvalue, prvalue, glvalue) — 1.5 hrs

**Read (20 min):**
- cppreference: Value categories
- The 5-category system: every expression has a type AND a value category

**Code (50 min):**
```cpp
// Exercise 1: Predict the value category of each expression
int x = 42;
int& ref = x;
int&& rref = std::move(x);
// What is the value category of: x? ref? rref? std::move(x)? 42? x+1?

// Exercise 2: Write a function that accepts ONLY rvalues
template<typename T>
void sink(T&& val) requires std::is_rvalue_reference_v<T&&>;

// Exercise 3: Demonstrate that a named rvalue reference IS an lvalue
void prove_named_rref_is_lvalue();
```

**💀 Puzzle (10 min):**
```cpp
// What does this print? 99% get it wrong.
#include <iostream>
void f(int& x)  { std::cout << "lvalue\n"; }
void f(int&& x) { std::cout << "rvalue\n"; }

template<typename T>
void g(T&& x) { f(x); }  // What gets called?

int main() {
    int a = 1;
    g(a);        // ?
    g(1);        // ? — THIS ONE surprises people
}
```

**Checkpoint:**
- [ ] Can explain why `g(1)` calls `f(int&)` not `f(int&&)`
- [ ] Can draw the value category taxonomy from memory

---

### Day 2: Move semantics — the real cost model — 1.5 hrs

**Read (20 min):**
- What move actually does at the assembly level (use godbolt.org)
- The "moved-from state" — what's guaranteed, what's not

**Code (50 min):**
```cpp
// Exercise 1: Implement a class with all 5 special members
// Rule of 5 — make it own a raw resource (fd, mmap, etc.)
class MappedFile {
    int fd_ = -1;
    void* addr_ = nullptr;
    size_t size_ = 0;
public:
    // Implement: ctor, dtor, copy ctor, copy assign, move ctor, move assign
    // The move ctor must leave source in a valid but empty state
};

// Exercise 2: Benchmark copy vs move of std::vector<std::string> with 1M elements
// Use std::chrono::steady_clock — expect 1000x+ difference

// Exercise 3: Show a case where RVO (Return Value Optimization) makes
// move unnecessary — and a case where it can't apply
```

**💀 Puzzle:**
```cpp
// Does this compile? If yes, what happens?
std::vector<std::unique_ptr<int>> v;
v.push_back(std::make_unique<int>(42));
auto v2 = v;  // ???
```

**🤖 ROS tie-in:**
- ROS1 `sensor_msgs::PointCloud2` — why does it copy on publish? What would a move-aware transport look like?

**Checkpoint:**
- [ ] Can implement Rule of 5 without looking anything up
- [ ] Can explain what state a moved-from `std::string` is in

---

### Day 3: Perfect forwarding & universal references — 1.5 hrs

**Read (20 min):**
- Scott Meyers' "universal reference" rules
- `std::forward` — why it exists, what it does

**Code (50 min):**
```cpp
// Exercise 1: Write a factory function that perfect-forwards to any constructor
template<typename T, typename... Args>
std::unique_ptr<T> make(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// Exercise 2: Write an emplace_back equivalent for a fixed-size ring buffer
template<typename T, size_t N>
class RingBuffer {
    std::array<std::aligned_storage_t<sizeof(T), alignof(T)>, N> storage_;
    size_t head_ = 0, tail_ = 0, count_ = 0;
public:
    template<typename... Args>
    bool try_emplace(Args&&... args);  // construct in-place, return false if full
};

// Exercise 3: Show the difference between T&& and auto&& in a for loop
// Why can range-for with auto&& bind to temporaries AND lvalues?
```

**💀 Puzzle:**
```cpp
// What types are deduced? Most people get at least one wrong.
template<typename T> void f(T&& x);

int a = 1;
const int b = 2;
int& c = a;

f(a);       // T = ?  x type = ?
f(b);       // T = ?  x type = ?
f(c);       // T = ?  x type = ?
f(42);      // T = ?  x type = ?
f(std::move(a)); // T = ?  x type = ?
```

**Checkpoint:**
- [ ] Can explain reference collapsing rules from memory
- [ ] Can implement a perfect-forwarding factory

---

### Day 4: constexpr, consteval, constinit — 1.5 hrs

**Read (20 min):**
- `constexpr` evolution: C++11 → C++14 → C++17 → C++20 → C++23
- `consteval` — guaranteed compile-time (C++20)
- `constinit` — static init order fiasco killer (C++20)

**Code (50 min):**
```cpp
// Exercise 1: CRC32 lookup table at compile time
constexpr std::array<uint32_t, 256> make_crc32_table() {
    // Implement the full CRC32 table generation at compile time
    // Verify with static_assert that table[0] == 0 and table[1] == correct value
}

// Exercise 2: Compile-time string hash for switch statements
constexpr uint64_t fnv1a(std::string_view sv) {
    // Implement FNV-1a hash
}
// Use it: switch(fnv1a(input)) { case fnv1a("start"): ... }

// Exercise 3: consteval vs constexpr — show a case where consteval
// catches a bug that constexpr wouldn't
consteval int safe_divide(int a, int b) {
    if (b == 0) throw "division by zero"; // compile error!
    return a / b;
}
```

**💀 Puzzle:**
```cpp
// Does this compile? Why or why not?
constexpr int f(int n) {
    if (n <= 1) return 1;
    return n * f(n - 1);
}
constexpr int a = f(5);    // ?
int runtime_val = 10;
int b = f(runtime_val);    // ?
consteval int g(int n) { return f(n); }
int c = g(runtime_val);    // ?
```

**🤖 ROS tie-in:**
- Compute message CRCs at compile time — zero runtime cost for message validation
- `constinit` for static ROS parameters to avoid the static init order fiasco

**Checkpoint:**
- [ ] Can write a non-trivial constexpr algorithm (sort, hash, CRC)
- [ ] Can explain when constexpr actually runs at compile time vs runtime

---

### Day 5: Concepts and constrained templates (C++20) — 1.5 hrs

**Read (20 min):**
- Concepts: what they replace (SFINAE, `enable_if`, tag dispatch)
- Writing constraints that produce readable error messages

**Code (50 min):**
```cpp
// Exercise 1: Define a Sensor concept for your robot work
template<typename T>
concept Sensor = requires(T t) {
    { t.read() } -> std::convertible_to<double>;
    { t.is_valid() } -> std::same_as<bool>;
    { T::sample_rate_hz } -> std::convertible_to<unsigned>;
};

// Exercise 2: Write a constrained template that only accepts trivially copyable types
// (important for shared memory IPC and DMA buffers)
template<typename T>
concept DmaCompatible = std::is_trivially_copyable_v<T>
                     && (sizeof(T) % alignof(std::max_align_t) == 0);

// Exercise 3: Rewrite this SFINAE horror using concepts
template<typename T, typename = std::enable_if_t<
    std::is_arithmetic_v<T> && !std::is_same_v<T, bool>>>
T clamp_to_range(T val, T lo, T hi);
// → Clean concept version
```

**💀 Puzzle:**
```cpp
// Overload resolution with concepts: which function gets called?
template<typename T> concept Integral = std::is_integral_v<T>;
template<typename T> concept SignedIntegral = Integral<T> && std::is_signed_v<T>;

void f(Integral auto x)       { std::cout << "1\n"; }
void f(SignedIntegral auto x)  { std::cout << "2\n"; }

f(42);   // ? — concept subsumption decides
f(42u);  // ?
```

**Checkpoint:**
- [ ] Can write a concept from scratch
- [ ] Can explain concept subsumption (more constrained wins)

---

### Day 6-7: WEEK 1 MINI-PROJECT — Compile-Time Message Descriptor — 4 hrs

**Build:**
A compile-time message serialization framework:
```cpp
// Define messages at compile time
struct ImuMessage {
    static constexpr uint32_t id = fnv1a("ImuMessage");
    static constexpr size_t max_size = 48;
    float accel[3];
    float gyro[3];
    uint64_t timestamp_ns;
};

// Compile-time dispatch table
template<typename... Messages>
class MessageRegistry {
    // at compile time: verify no ID collisions
    // at compile time: compute max message size
    // at runtime: O(1) dispatch by ID using switch or array lookup
};
```

**Requirements:**
- [ ] All IDs computed at compile time (constexpr)
- [ ] Static assert catches duplicate IDs
- [ ] DmaCompatible concept enforced on all message types
- [ ] Zero heap allocation in the dispatch path
- [ ] Benchmark: dispatch 1M messages, measure ns/msg

---

## Week 2 — Error Handling Without Exceptions & The C++ Memory Model
*Goal: Handle errors in RT-safe ways, understand atomics from the ground up*

### Day 8: Why exceptions are banned in RT/safety-critical code — 1.5 hrs

**Read (20 min):**
- Exception overhead: stack unwinding, RTTI, non-deterministic timing
- JSF AV Rule 208: "Exceptions shall not be used"
- `-fno-exceptions` flag and what it breaks in the standard library

**Code (50 min):**
```cpp
// Exercise 1: Implement Expected<T, E> (like std::expected from C++23)
template<typename T, typename E>
class Expected {
    union { T value_; E error_; };
    bool has_value_;
public:
    // Implement: constructors, value(), error(), has_value()
    // Implement: and_then(), or_else(), transform() (monadic operations)
};

// Exercise 2: Error propagation macro (like Rust's ?)
#define TRY(expr) \
    ({ auto&& _r = (expr); \
       if (!_r.has_value()) return unexpected(_r.error()); \
       std::move(_r).value(); })

// Exercise 3: Show timing difference — exception throw vs error code
// Throw 10,000 exceptions vs return 10,000 error codes — measure wall time
```

**💀 Puzzle:**
```cpp
// This function is compiled with -fno-exceptions. What happens?
std::vector<int> v;
v.at(999);  // at() uses exceptions for out-of-bounds
// Answer: std::terminate() is called. Not a nice crash.

// How would you make operator[] safe without exceptions?
```

**Checkpoint:**
- [ ] Can implement a basic Expected type
- [ ] Can measure the timing cost of exception throw+catch

---

### Day 9: std::optional, std::variant, std::visit — 1.5 hrs

**Read (20 min):**
- `std::variant` as a type-safe union — no dynamic allocation
- `std::visit` and the visitor pattern

**Code (50 min):**
```cpp
// Exercise 1: Model robot states as a variant
using RobotState = std::variant<
    struct Idle {},
    struct Navigating { double target_x, target_y; },
    struct Charging { int battery_percent; },
    struct Error { int code; std::string_view message; }
>;

// Write a state machine using std::visit
void handle_state(RobotState& state, Event event);

// Exercise 2: Implement a command dispatch system
using Command = std::variant<MoveCmd, StopCmd, ChargeCmd, DiagnosticCmd>;
// Process commands without dynamic dispatch or exceptions

// Exercise 3: Measure sizeof(std::variant<A,B,C>) — understand the layout
// Compare with a manual tagged union
```

**💀 Puzzle:**
```cpp
// What does std::visit do with this? Does it compile?
std::variant<int, double, std::string> v = "hello";
std::visit([](auto&& arg) {
    arg += arg;  // ??? — does this work for all types?
}, v);
```

**Checkpoint:**
- [ ] Can model a state machine with variant + visit
- [ ] Know the sizeof overhead of variant vs manual union

---

### Day 10: C++ Memory Model — happens-before, sequenced-before — 1.5 hrs

**Read (20 min):**
- Preshing: "An Introduction to Lock-Free Programming"
- The C++ memory model: happens-before, synchronizes-with, sequenced-before
- Why "volatile" is NOT atomic and NOT a memory barrier

**Code (50 min):**
```cpp
// Exercise 1: Demonstrate a data race with TSan (Thread Sanitizer)
// Compile with: g++ -fsanitize=thread -g
int shared_counter = 0;
void thread_func() { for (int i = 0; i < 100000; ++i) ++shared_counter; }
// Run 2 threads — observe TSan report

// Exercise 2: Fix it three ways:
// (a) std::mutex
// (b) std::atomic<int> with seq_cst
// (c) std::atomic<int> with relaxed + load/store
// Benchmark all three

// Exercise 3: Show why volatile fails as synchronization
volatile bool ready = false;
volatile int data = 0;
// Thread 1: data = 42; ready = true;
// Thread 2: while (!ready); assert(data == 42);
// This CAN fail on ARM/POWER. Show why with memory model reasoning.
```

**💀 Puzzle:**
```cpp
// Classic: is this code correct on x86? On ARM?
std::atomic<bool> x{false}, y{false};
int r1, r2;

// Thread 1:              Thread 2:
x.store(true, relaxed);   y.store(true, relaxed);
r1 = y.load(relaxed);     r2 = x.load(relaxed);

// Can r1 == 0 && r2 == 0?
// On x86? On ARM? According to the standard?
```

**Checkpoint:**
- [ ] Can explain happens-before vs sequenced-before
- [ ] Can run TSan and interpret its output

---

### Day 11: Acquire-Release semantics — 1.5 hrs

**Read (20 min):**
- Preshing: "Acquire and Release Semantics"
- memory_order_acquire, memory_order_release, memory_order_acq_rel
- The "barrier sandwich" for locks

**Code (50 min):**
```cpp
// Exercise 1: Implement a simple spinlock using acquire/release
class Spinlock {
    std::atomic<bool> locked_{false};
public:
    void lock() {
        while (locked_.exchange(true, std::memory_order_acquire))
            ; // spin
    }
    void unlock() {
        locked_.store(false, std::memory_order_release);
    }
};

// Exercise 2: Producer-consumer with ONE atomic flag (no mutex)
std::atomic<bool> ready{false};
int payload = 0;
// Producer: payload = 42; ready.store(true, release);
// Consumer: while (!ready.load(acquire)); use(payload);
// Prove this is correct. Then prove the relaxed version is broken.

// Exercise 3: Implement a SeqLock (sequence lock) for read-heavy data
// Used in Linux kernel for clock reads
template<typename T>
class SeqLock {
    std::atomic<uint32_t> seq_{0};
    T data_;
public:
    void write(const T& val);  // increment seq (odd = writing), write, increment (even = done)
    T read() const;            // read seq, read data, read seq again, retry if changed
};
```

**💀 Puzzle:**
```cpp
// The "IRIW" problem — can two readers disagree about store order?
// Thread 1: x.store(1, release);
// Thread 2: y.store(1, release);
// Thread 3: r1=x.load(acquire); r2=y.load(acquire);
// Thread 4: r3=y.load(acquire); r4=x.load(acquire);
// Can r1==1,r2==0 AND r3==1,r4==0 simultaneously?
// Hint: yes with acquire/release, no with seq_cst. Why?
```

**Checkpoint:**
- [ ] Can implement a spinlock with correct memory ordering
- [ ] Can explain why seq_cst is needed for the IRIW problem

---

### Day 12: Atomic operations & compare-exchange — 1.5 hrs

**Read (20 min):**
- CAS (compare-and-swap) — the universal atomic primitive
- `compare_exchange_weak` vs `compare_exchange_strong`
- The ABA problem

**Code (50 min):**
```cpp
// Exercise 1: Lock-free stack (Treiber stack)
template<typename T>
class LockFreeStack {
    struct Node { T data; Node* next; };
    std::atomic<Node*> head_{nullptr};
public:
    void push(T val);  // CAS loop on head
    std::optional<T> pop();  // CAS loop, watch for ABA!
};

// Exercise 2: Atomic increment with CAS (don't use fetch_add)
int atomic_increment(std::atomic<int>& val) {
    int old = val.load(std::memory_order_relaxed);
    while (!val.compare_exchange_weak(old, old + 1,
            std::memory_order_relaxed)) {}
    return old;
}
// Why does weak work here? When would you need strong?

// Exercise 3: Implement an atomic max operation
template<typename T>
T atomic_fetch_max(std::atomic<T>& a, T val) {
    // CAS loop: update only if val > current
}
```

**💀 Puzzle:**
```cpp
// The ABA problem in action. What goes wrong?
// Stack has: A -> B -> C
// Thread 1: pop() reads head=A, next=B
//           (gets preempted here)
// Thread 2: pop() A, pop() B, push() A back
//           Stack is now: A -> C
// Thread 1: resumes, CAS(head, A, B) succeeds! (head==A, set to B)
//           But B was freed! Use-after-free.
// How do you fix this? (hazard pointers, epoch-based reclamation, tagged pointers)
```

**Checkpoint:**
- [ ] Can implement a lock-free stack
- [ ] Can explain the ABA problem and at least one solution

---

### Day 13-14: WEEK 2 MINI-PROJECT — Lock-Free SPSC Queue — 4 hrs

**Build:**
A Single-Producer, Single-Consumer (SPSC) bounded queue:
```cpp
template<typename T, size_t Capacity>
class SPSCQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "must be power of 2");
    std::array<T, Capacity> buffer_;
    alignas(64) std::atomic<size_t> write_pos_{0};  // cache line isolation
    alignas(64) std::atomic<size_t> read_pos_{0};
public:
    bool try_push(const T& val);
    bool try_pop(T& val);
    size_t size() const;
};
```

**Requirements:**
- [ ] No mutex, no allocation, no exceptions
- [ ] Cache-line padding to prevent false sharing
- [ ] Acquire/release semantics only (not seq_cst)
- [ ] Pass TSan with zero warnings
- [ ] Benchmark: >100M ops/sec on modern hardware
- [ ] Compare with `boost::lockfree::spsc_queue`

**🤖 ROS tie-in:**
- This is exactly what you'd use between a sensor ISR handler thread and a ROS publisher thread

---

## Week 3 — Real-Time Linux Programming
*Goal: Build a deterministic RT loop that hits 1kHz with <50μs jitter*

### Day 15: What is real-time? Hard/Soft/Firm — 1.5 hrs

**Read (20 min):**
- Shuhao Wu: "Real-time programming with Linux, part 1"
- Hard RT: aircraft control (deadline miss = crash)
- Soft RT: audio (deadline miss = pop/click)
- Your OKS robots: ~100Hz control loop = 10ms period

**Code (50 min):**
```cpp
// Exercise 1: Measure scheduling latency
// Create a thread, set SCHED_FIFO priority 90, measure wakeup jitter
#include <sched.h>
#include <time.h>

void measure_latency() {
    struct sched_param sp;
    sp.sched_priority = 90;
    sched_setscheduler(0, SCHED_FIFO, &sp);
    mlockall(MCL_CURRENT | MCL_FUTURE);

    struct timespec next;
    clock_gettime(CLOCK_MONOTONIC, &next);

    std::vector<int64_t> latencies;
    for (int i = 0; i < 10000; ++i) {
        next.tv_nsec += 1000000; // 1ms period
        // normalize
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr);
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        int64_t lat_ns = (now.tv_sec - next.tv_sec) * 1e9 + (now.tv_nsec - next.tv_nsec);
        latencies.push_back(lat_ns);
    }
    // Print: min, max, avg, p99, p99.9
}

// Exercise 2: Compare SCHED_OTHER vs SCHED_FIFO latency histograms
// Run the above with both schedulers, plot results

// Exercise 3: Run `stress -c $(nproc)` in background
// See how SCHED_OTHER degrades but SCHED_FIFO holds
```

**Checkpoint:**
- [ ] Can set SCHED_FIFO and mlockall on a thread
- [ ] Can measure and histogram scheduling latency

---

### Day 16: The RT-forbidden operations list — 1.5 hrs

**Read (20 min):**
- What you CANNOT do in a real-time thread:
  - `malloc/new/delete` (non-deterministic)
  - `printf/cout/spdlog` (locks, allocation, syscalls)
  - `mutex lock` (priority inversion without PI mutex)
  - Blocking I/O (file, network, pipe)
  - Page faults (hence `mlockall`)
  - System calls that may block (even `clock_gettime` can on some kernels)

**Code (50 min):**
```cpp
// Exercise 1: Demonstrate priority inversion
// Create 3 threads: Low (holds mutex), Medium (CPU hog), High (wants mutex)
// Show that High is blocked by Medium — classic Mars Pathfinder bug!

// Exercise 2: Fix with a priority-inheritance mutex
pthread_mutexattr_t attr;
pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);

// Exercise 3: Build a "RT-safe logger" that writes from a non-RT thread
// RT thread pushes log entries into SPSC queue (from Week 2!)
// Non-RT thread drains queue and writes to file
class RTLogger {
    SPSCQueue<LogEntry, 4096> queue_;
    std::thread writer_thread_;
public:
    void log(LogEntry entry);  // called from RT thread — zero allocation!
    void drain();              // called from writer thread — can do I/O
};
```

**💀 Puzzle:**
```cpp
// This RT thread "works fine in testing." What's the time bomb?
void rt_loop() {
    while (running) {
        auto data = sensor.read();
        auto result = controller.compute(data);
        actuator.write(result);

        std::string log_msg = "pos=" + std::to_string(result.pos); // 💣
        logger.info(log_msg);                                       // 💣💣
    }
}
// Answer: std::string + to_string = malloc in the hot path
// logger.info = likely locks a mutex + does I/O
```

**🤖 ROS tie-in:**
- `ros::Publisher::publish()` — does it allocate? (yes, serialization)
- This is why `realtime_tools` exists in ros_control

**Checkpoint:**
- [ ] Can list 6+ operations forbidden in RT path
- [ ] Can explain priority inversion and the Mars Pathfinder bug

---

### Day 17: Thread affinity, CPU isolation, PREEMPT_RT — 1.5 hrs

**Read (20 min):**
- CPU affinity: `pthread_setaffinity_np`, `isolcpus` kernel parameter
- PREEMPT_RT patch: what it does to the kernel
- `cyclictest` — the standard RT latency benchmark

**Code (50 min):**
```cpp
// Exercise 1: Pin your RT thread to a specific CPU core
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
CPU_SET(3, &cpuset);  // pin to core 3
pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);

// Exercise 2: Run cyclictest, interpret the results
// sudo cyclictest -t1 -p 90 -i 1000 -l 10000 --mlockall
// Parse: T: 0 Min: X Act: Y Avg: Z Max: W

// Exercise 3: Build a complete "RT thread launcher" utility
struct RTThreadConfig {
    int priority = 90;                    // 1-99
    int policy = SCHED_FIFO;
    int cpu_affinity = -1;               // -1 = no affinity
    bool lock_memory = true;
    size_t stack_size = 8 * 1024 * 1024;  // pre-fault stack
};

std::thread launch_rt_thread(RTThreadConfig config, std::function<void()> fn);
// Must: set scheduler, set affinity, mlockall, pre-fault stack
```

**Checkpoint:**
- [ ] Can pin a thread to a CPU core
- [ ] Can run and interpret cyclictest results

---

### Day 18: Cyclic executive pattern — 1.5 hrs

**Read (20 min):**
- Cyclic executive: fixed-rate loop, deterministic task scheduling
- Major frame / minor frame design
- Timing: `clock_nanosleep` vs busy-wait vs hardware timer

**Code (50 min):**
```cpp
// Exercise 1: Implement a 1kHz cyclic executive
class CyclicExecutive {
    using Task = std::function<void()>;
    struct ScheduledTask {
        Task fn;
        uint32_t period_ms;     // run every N ms
        uint32_t offset_ms;     // phase offset
    };
    std::vector<ScheduledTask> tasks_;
    uint32_t major_frame_ms_;
    std::atomic<bool> running_{true};

public:
    void add_task(Task fn, uint32_t period_ms, uint32_t offset_ms = 0);
    void run();  // the main RT loop
    void stop() { running_ = false; }

    // Overrun detection: if a minor frame takes longer than its budget
    struct Stats {
        uint64_t total_frames = 0;
        uint64_t overruns = 0;
        int64_t max_latency_ns = 0;
        int64_t max_execution_ns = 0;
    };
    Stats get_stats() const;
};

// Exercise 2: Schedule these tasks in a 10ms major frame:
// - IMU read:      1kHz (every 1ms)
// - Controller:    500Hz (every 2ms)
// - Odometry pub:  100Hz (every 10ms)
// - Heartbeat:     10Hz (every 100ms)

// Exercise 3: Add overrun detection and graceful degradation
// If a frame overruns, skip the lowest-priority task next frame
```

**🤖 ROS tie-in:**
- This is exactly what `ros_control`'s `controller_manager` does
- Your OKS robot's base controller runs a cyclic executive

**Checkpoint:**
- [ ] Can implement a cyclic executive with overrun detection
- [ ] Can schedule multiple tasks at different rates

---

### Day 19: Pool allocators & PMR (Polymorphic Memory Resources) — 1.5 hrs

**Read (20 min):**
- `std::pmr` — runtime-swappable allocators
- Pool allocator: pre-allocate N objects, O(1) alloc/dealloc
- Monotonic allocator: bump pointer, never free (great for per-frame allocation)

**Code (50 min):**
```cpp
// Exercise 1: Implement a fixed-size pool allocator
template<typename T, size_t N>
class PoolAllocator {
    union Slot { T obj; Slot* next; };
    std::array<Slot, N> pool_;
    Slot* free_list_ = nullptr;
public:
    PoolAllocator();
    T* allocate();    // O(1), deterministic
    void deallocate(T* p);  // O(1), deterministic
};

// Exercise 2: Use std::pmr with a monotonic buffer
std::array<std::byte, 4096> buffer;
std::pmr::monotonic_buffer_resource mbr(buffer.data(), buffer.size());
std::pmr::vector<int> v(&mbr);
// Fill vector — no heap allocation until buffer is exhausted
// What happens when it IS exhausted? (depends on upstream resource)

// Exercise 3: Benchmark: std::vector<int> vs pmr::vector<int> with pool
// Insert 1000 elements, measure allocation time
// Expected: pmr ~5-10x faster for small allocations
```

**💀 Puzzle:**
```cpp
// This look fine but has a subtle bug. What is it?
std::array<std::byte, 256> buf;
std::pmr::monotonic_buffer_resource mbr(buf.data(), buf.size());
std::pmr::string s("hello", &mbr);
auto s2 = s;  // What allocator does s2 use? 💣
// Answer: s2 uses the DEFAULT allocator (global new/delete), not mbr!
// pmr containers copy the allocator only if you explicitly pass it
```

**Checkpoint:**
- [ ] Can implement a pool allocator
- [ ] Can use std::pmr and explain its propagation rules

---

### Day 20-21: WEEK 3 MINI-PROJECT — RT Cyclic Executive with Telemetry — 4 hrs

**Build:**
A complete real-time application:
```
┌─────────────────────────────────────────────┐
│ RT Thread (SCHED_FIFO, pinned to core 3)    │
│                                              │
│  1kHz: read_sensor() → controller.compute() │
│  100Hz: publish_telemetry() → SPSC queue    │
│  10Hz:  watchdog_kick()                     │
│                                              │
│  Overrun detection + stats                  │
└──────────────┬──────────────────────────────┘
               │ SPSC Queue (lock-free)
┌──────────────▼──────────────────────────────┐
│ Non-RT Thread                               │
│  Drain telemetry → log to file / print      │
│  Health monitoring                          │
└─────────────────────────────────────────────┘
```

**Requirements:**
- [ ] RT thread: SCHED_FIFO prio 90, pinned CPU, mlockall
- [ ] Zero allocations in RT path (pool allocator for messages)
- [ ] SPSC queue from Week 2 for telemetry transport
- [ ] RTLogger from Day 16 for debug output
- [ ] Overrun detection with stats (max latency, p99)
- [ ] Run for 60 seconds under `stress -c $(nproc)`, report jitter histogram
- [ ] Target: max jitter < 100μs at 1kHz

---

## Week 4 — Safety-Critical Patterns & JPL Rules
*Goal: Write code that could survive a code review at NASA/JPL*

### Day 22: JPL Power of 10 Rules — applied to C++ — 1.5 hrs

**Read (20 min):**
- Power of 10 Rules (all 10, memorize them)
- How to adapt them to modern C++ (some are C-specific)

**The 10 Rules, C++ Adapted:**
1. No recursion (use iterative algorithms, `std::stack<>` if needed)
2. All loops have fixed upper bounds (use `for` with known limit, not `while(true)`)
3. No dynamic allocation after init (`constinit`, pool allocators, PMR)
4. Functions ≤ 60 lines
5. ≥ 2 assertions per function (preconditions, postconditions)
6. Smallest possible scope for all objects
7. Check all return values; validate all parameters
8. Minimal preprocessor (no token pasting, no variadic macros)
9. Restrict pointer use (prefer references, spans, unique_ptr)
10. All code compiled with max warnings + static analyzer, zero warnings

**Code (50 min):**
```cpp
// Exercise 1: Rewrite this function to comply with ALL 10 rules
// Original (violates rules 1, 3, 4, 5, 7):
std::vector<int> flatten(const std::vector<std::vector<int>>& nested) {
    std::vector<int> result;
    for (auto& inner : nested)
        for (auto& x : inner)
            result.push_back(x);
    return result;
}
// JPL version: fixed output buffer, assertions, bounded loops, return status

// Exercise 2: Write a bounded loop version of binary search
// Must have a provable upper bound on iterations (log2(N) + 1)

// Exercise 3: Add contracts (assert-based) to your SPSC queue from Week 2
// Pre: !is_full() for push, !is_empty() for pop
// Post: size increased/decreased by 1
// Invariant: write_pos - read_pos == size (modular)
```

**💀 Puzzle:**
```cpp
// Toyota unintended acceleration (2009): single bit flip in RAM
// corrupted the task stack, overwriting the throttle variable.
// 243 violations of Power of 10 were found in their firmware.
// Question: Which rule(s) would have prevented the stack overflow?
// Answer: Rule 1 (no recursion) + Rule 4 (small functions) = bounded stack use
```

**Checkpoint:**
- [ ] Can recite all 10 Power of 10 rules from memory
- [ ] Can refactor a function to comply with all 10

---

### Day 23: Defensive programming — assertions, contracts, invariants — 1.5 hrs

**Read (20 min):**
- Design by Contract: preconditions, postconditions, invariants
- `assert` vs custom assert with recovery action
- C++20 contracts (deferred to C++26) — `[[expects:]]`, `[[ensures:]]`

**Code (50 min):**
```cpp
// Exercise 1: Implement a production-grade assertion macro
// Must: log file:line, log the expression, allow custom recovery
#define RT_ASSERT(cond, recovery_action) \
    do { \
        if (!(cond)) [[unlikely]] { \
            rt_assert_handler(__FILE__, __LINE__, #cond); \
            recovery_action; \
        } \
    } while (0)

// Exercise 2: Class invariant checking
class PIDController {
    double kp_, ki_, kd_;
    double integral_ = 0.0;
    double prev_error_ = 0.0;
    static constexpr double MAX_INTEGRAL = 1000.0;

    void check_invariant() const {
        RT_ASSERT(kp_ >= 0.0, return);
        RT_ASSERT(ki_ >= 0.0, return);
        RT_ASSERT(std::isfinite(integral_), integral_ = 0.0);
        RT_ASSERT(std::abs(integral_) <= MAX_INTEGRAL, integral_ = 0.0);
    }
public:
    double compute(double error, double dt) {
        check_invariant();  // pre
        // ... computation ...
        check_invariant();  // post
        return output;
    }
};

// Exercise 3: Saturating arithmetic — never overflow, always clamp
template<typename T>
constexpr T sat_add(T a, T b) {
    // If a + b would overflow, return max; if underflow, return min
    // Must work at compile time (constexpr)
    // Must have assertions for NaN if floating point
}
```

**🤖 ROS tie-in:**
- Your OKS estimator crash (#98301): `est_vy` went non-finite
- A `check_invariant()` after every prediction step would have caught it immediately
- Rule: assert `std::isfinite()` on every state variable after every computation

**Checkpoint:**
- [ ] Can implement a custom assertion macro with recovery
- [ ] Can add pre/post/invariant checks to a class

---

### Day 24: Static analysis & compiler hardening — 1.5 hrs

**Read (20 min):**
- `clang-tidy` checks for RT/safety: `bugprone-*`, `cert-*`, `cppcoreguidelines-*`
- Compiler flags: `-Wall -Wextra -Wpedantic -Werror -Wconversion`
- Sanitizers: ASan, UBSan, TSan, MSan

**Code (50 min):**
```bash
# Exercise 1: Run clang-tidy on your SPSC queue, fix ALL findings
clang-tidy spsc_queue.hpp -checks='*' -- -std=c++20

# Exercise 2: Compile with ALL sanitizers, one at a time
g++ -fsanitize=address -g code.cpp         # ASan: buffer overflows, use-after-free
g++ -fsanitize=undefined -g code.cpp       # UBSan: signed overflow, null deref
g++ -fsanitize=thread -g code.cpp          # TSan: data races
g++ -fsanitize=memory -g code.cpp          # MSan: uninitialized reads (clang only)

# Exercise 3: Write a .clang-tidy config for safety-critical code
```

```yaml
# .clang-tidy for safety-critical C++
Checks: >
  bugprone-*,
  cert-*,
  cppcoreguidelines-*,
  -cppcoreguidelines-avoid-magic-numbers,
  misc-*,
  performance-*,
  readability-*,
  -readability-magic-numbers
WarningsAsErrors: >
  bugprone-use-after-move,
  bugprone-dangling-handle,
  cert-err58-cpp,
  cppcoreguidelines-pro-type-reinterpret-cast
```

**💀 Puzzle:**
```cpp
// UBSan catches this. What's the UB?
int x = INT_MAX;
int y = x + 1;  // signed overflow = UB!
// GCC may optimize away if(y < x) because "signed overflow can't happen"
// This is how the Linux kernel null-pointer check was removed (Day 10 reading)
```

**Checkpoint:**
- [ ] Can run clang-tidy with a custom config
- [ ] Can use all 4 sanitizers and interpret output

---

### Day 25: Watchdog, heartbeat, and redundancy patterns — 1.5 hrs

**Read (20 min):**
- Software watchdog: if the main loop doesn't kick within N ms → restart
- Heartbeat: periodic "I'm alive" signal to a monitor
- N-version programming: 2+ independent implementations, cross-check

**Code (50 min):**
```cpp
// Exercise 1: Implement a software watchdog timer
class Watchdog {
    std::chrono::milliseconds timeout_;
    std::chrono::steady_clock::time_point last_kick_;
    std::function<void()> on_timeout_;
    std::thread monitor_thread_;
    std::atomic<bool> running_{true};
public:
    Watchdog(std::chrono::milliseconds timeout, std::function<void()> on_timeout);
    void kick();  // called from RT thread — must be fast!
    // Monitor thread periodically checks if last_kick is too old
};

// Exercise 2: Implement a heartbeat publisher
// Every 100ms, publish a heartbeat containing:
// - monotonic timestamp
// - sequence number
// - mode (nominal, degraded, error)
// - worst-case latency since last heartbeat
struct Heartbeat {
    uint64_t timestamp_ns;
    uint32_t sequence;
    uint8_t mode;
    int64_t max_latency_ns;
};

// Exercise 3: Cross-check pattern
// Two independent implementations of atan2, compare results
// If they disagree by > epsilon, flag an error
double safe_atan2(double y, double x) {
    double r1 = std::atan2(y, x);
    double r2 = my_cordic_atan2(y, x);  // your own implementation
    RT_ASSERT(std::abs(r1 - r2) < 1e-6, return r1);
    return r1;
}
```

**🤖 ROS tie-in:**
- Your OKS robot heartbeat analysis (#batch_heartbeat_analysis.py)
- Watchdog on the base controller — if navigation stops publishing, emergency stop

**Checkpoint:**
- [ ] Can implement a software watchdog
- [ ] Can explain N-version programming

---

### Day 26-28: WEEK 4 MINI-PROJECT — Watchdog + Health Monitor — 4 hrs

**Build:**
Extend the Week 3 RT application with safety features:
- Software watchdog (restarts RT loop if it hangs)
- Heartbeat publisher (sequence, timestamp, health status)
- Mode manager (INIT → NOMINAL → DEGRADED → ERROR → SAFE)
- Assertion system with counters (track assertion failures)
- Overrun degradation (drop low-priority tasks when overrunning)

**Requirements:**
- [ ] All JPL Power of 10 rules followed
- [ ] clang-tidy clean with safety-critical config
- [ ] ASan + UBSan + TSan clean
- [ ] ≥ 2 assertions per function
- [ ] No allocation in RT path
- [ ] Mode transitions logged with timestamps
- [ ] Kill the RT thread mid-execution → watchdog catches it within 100ms

---

## Week 5 — Production Build Systems, Testing & Tooling
*Goal: CMake, GoogleTest, fuzzing, CI/CD for C++*

### Day 29: Modern CMake — targets, not variables — 1.5 hrs

**Code (60 min):**
```cmake
# Exercise 1: Create a proper CMake project structure
cmake_minimum_required(VERSION 3.20)
project(rt_framework VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Library target with PUBLIC/PRIVATE dependencies
add_library(rt_core
    src/spsc_queue.cpp
    src/cyclic_executive.cpp
    src/pool_allocator.cpp
)
target_include_directories(rt_core PUBLIC include)
target_compile_options(rt_core PRIVATE
    -Wall -Wextra -Wpedantic -Werror -Wconversion
)

# Sanitizer presets
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(ENABLE_ASAN)
    target_compile_options(rt_core PUBLIC -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(rt_core PUBLIC -fsanitize=address)
endif()

# Exercise 2: Cross-compile preset for ARM (like your Jetson)
# CMakePresets.json

# Exercise 3: Install rules + package config
install(TARGETS rt_core EXPORT rt_framework-targets)
install(EXPORT rt_framework-targets DESTINATION lib/cmake/rt_framework)
```

**Checkpoint:**
- [ ] Can create a modern CMake project with targets (not variables)
- [ ] Can set up sanitizer options and cross-compile presets

---

### Day 30: GoogleTest + GoogleMock — 1.5 hrs

**Code (60 min):**
```cpp
// Exercise 1: Test your SPSC queue with GTest
TEST(SPSCQueue, PushPopBasic) {
    SPSCQueue<int, 16> q;
    EXPECT_TRUE(q.try_push(42));
    int val;
    EXPECT_TRUE(q.try_pop(val));
    EXPECT_EQ(val, 42);
}

TEST(SPSCQueue, FullQueue) {
    SPSCQueue<int, 4> q;
    EXPECT_TRUE(q.try_push(1));
    EXPECT_TRUE(q.try_push(2));
    EXPECT_TRUE(q.try_push(3));
    EXPECT_FALSE(q.try_push(4));  // full! (N-1 usable)
}

// Exercise 2: Concurrent stress test
TEST(SPSCQueue, StressTest) {
    SPSCQueue<uint64_t, 1024> q;
    constexpr uint64_t N = 1'000'000;
    std::thread producer([&] {
        for (uint64_t i = 0; i < N; ++i)
            while (!q.try_push(i)) {}
    });
    uint64_t expected = 0;
    std::thread consumer([&] {
        for (uint64_t i = 0; i < N; ++i) {
            uint64_t val;
            while (!q.try_pop(val)) {}
            EXPECT_EQ(val, expected++);
        }
    });
    producer.join();
    consumer.join();
}

// Exercise 3: GMock — mock a Sensor interface
class MockSensor : public ISensor {
    MOCK_METHOD(double, read, (), (override));
    MOCK_METHOD(bool, is_valid, (), (const, override));
};
```

**Checkpoint:**
- [ ] Can write and run GTest tests with concurrent stress tests
- [ ] Can use GMock for dependency injection

---

### Day 31: Fuzzing with libFuzzer / AFL — 1.5 hrs

**Code (60 min):**
```cpp
// Exercise 1: Fuzz your message deserializer
// Build: clang++ -fsanitize=fuzzer,address -g fuzz_message.cpp
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // Feed random bytes to your message parser
    // libFuzzer + ASan will find buffer overflows, OOB reads, etc.
    auto result = parse_message(data, size);
    return 0;
}

// Exercise 2: Fuzz your Expected<T, E> type with random operations
// Exercise 3: Let the fuzzer run for 10 minutes — analyze crash reports
```

**💀 Puzzle:**
- Fuzzing found a bug in every C/C++ parser ever tested (including gcc, clang)
- The median time to find a crashing input with libFuzzer is ~5 minutes
- Question: If fuzzing is so effective, why don't most projects use it?
- Answer: Integration cost + CI budget + developer awareness

**Checkpoint:**
- [ ] Can set up and run a libFuzzer harness
- [ ] Can interpret fuzzer crash reports

---

### Day 32: Profiling — perf, Valgrind, flame graphs — 1.5 hrs

**Code (60 min):**
```bash
# Exercise 1: Profile your RT loop with perf
perf record -g ./rt_executive --duration 10
perf report

# Exercise 2: Generate a flame graph
perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg

# Exercise 3: Use Valgrind's callgrind for instruction-level profiling
valgrind --tool=callgrind ./rt_executive --duration 1
kcachegrind callgrind.out.*

# Exercise 4: Cache miss analysis
perf stat -e cache-misses,cache-references,L1-dcache-load-misses ./rt_executive
# Your SPSC queue should have very few cache misses due to cache line alignment
```

**Checkpoint:**
- [ ] Can use perf to profile and generate flame graphs
- [ ] Can identify cache misses and hot functions

---

### Day 33-35: WEEK 5 MINI-PROJECT — Testing & CI for RT Framework — 5 hrs

**Build:**
Full test suite + CI pipeline for your accumulated code:
- [ ] GTest unit tests for: SPSC queue, pool allocator, Expected, CyclicExecutive
- [ ] Concurrent stress tests with TSan
- [ ] Fuzzing harness for message parsing
- [ ] Performance benchmarks with Google Benchmark
- [ ] CMake build with presets: Debug, Release, ASan, TSan, UBSan, Fuzz
- [ ] GH Actions / Makefile CI that runs all of the above
- [ ] Code coverage report (lcov/gcovr) — target 80%+

---

## Week 6 — IPC, Serialization & Software-Defined Patterns
*Goal: Build inter-process communication for real systems*

### Day 36: Shared memory IPC — 1.5 hrs

**Code (60 min):**
```cpp
// Exercise 1: Shared memory ring buffer between two processes
// Use shm_open + mmap to create a shared memory region
// Place your SPSC queue's data there (trivially copyable types only!)

// Exercise 2: Demonstrate the DmaCompatible concept from Week 1
// Only types that pass DmaCompatible can go in shared memory

// Exercise 3: Measure latency: pipe vs socket vs shared memory
// Expected: shared memory ~100ns, others ~10μs+
```

---

### Day 37: Protobuf / FlatBuffers / Cap'n Proto — 1.5 hrs

**Code (60 min):**
```cpp
// Exercise 1: Define an IMU message in protobuf
// Serialize/deserialize, measure throughput (messages/sec)

// Exercise 2: Same message in FlatBuffers (zero-copy!)
// Compare: protobuf vs FlatBuffers serialization time
// FlatBuffers should be 10-100x faster because no copy on read

// Exercise 3: Write a benchmark comparing:
// - Raw memcpy (trivially copyable struct)
// - Protobuf
// - FlatBuffers
// - JSON (as a baseline to laugh at)
```

---

### Day 38: Type erasure & plugin architecture — 1.5 hrs

**Code (60 min):**
```cpp
// Exercise 1: Implement type erasure (like std::function but simpler)
class AnyCallable {
    struct Concept {
        virtual void invoke() = 0;
        virtual ~Concept() = default;
    };
    template<typename F>
    struct Model : Concept {
        F fn;
        void invoke() override { fn(); }
    };
    std::unique_ptr<Concept> impl_;
public:
    template<typename F>
    AnyCallable(F&& f) : impl_(std::make_unique<Model<F>>(std::forward<F>(f))) {}
    void operator()() { impl_->invoke(); }
};

// Exercise 2: Small buffer optimization — avoid the heap allocation
// Store small callables inline (e.g. 32 bytes)

// Exercise 3: Plugin loading with dlopen
// Load a .so at runtime, resolve a symbol, call it
using PluginFactory = IController* (*)();
void* handle = dlopen("./libmy_controller.so", RTLD_NOW);
auto factory = (PluginFactory)dlsym(handle, "create_controller");
```

**🤖 ROS tie-in:**
- ROS pluginlib works exactly this way (dlopen + factory pattern)
- Your `ros_control` controllers are loaded this way

---

### Day 39-42: WEEK 6 MINI-PROJECT — Telemetry Pipeline — 5 hrs

**Build:**
A complete telemetry system:
```
┌─────────────┐      shared memory      ┌─────────────┐
│ RT Process   │  ──── SPSC queue ────>  │ Logger Proc │
│ (sensor sim) │                         │ (file write) │
└─────────────┘                         └─────────────┘
        │                                       │
        │ FlatBuffers                    protobuf/JSON
        │                                       │
        ▼                                       ▼
   [Local display]                    [Network export]
```

**Requirements:**
- [ ] Two separate processes (not threads) communicating via shared memory
- [ ] RT process: simulated 1kHz sensor → serialize → SPSC queue in shm
- [ ] Logger process: drain queue → write to file in FlatBuffers format
- [ ] Measure end-to-end latency (sensor event → on disk): target < 500μs
- [ ] All code follows JPL rules, passes sanitizers

---

## Week 7 — Undefined Behavior Deep Dive & Advanced Patterns
*Goal: Become the person who catches UB that others miss*

### Day 43: UB catalog — the 20 most dangerous forms — 1.5 hrs

**Read (20 min):**
- Regehr: "A Guide to Undefined Behavior in C and C++ (Parts 1-3)"
- cppreference: Undefined behavior page

**💀 Puzzles (ALL session — 60 min):**
These are the exercises that 99% of C++ developers get wrong.

```cpp
// Puzzle 1: What does this print?
int i = 0;
int j = (i++) + (i++);
std::cout << j << "\n";
// Answer: UB! Unsequenced modification of i. Any result is valid.

// Puzzle 2: Is this legal?
int* p = new int(42);
delete p;
std::cout << p << "\n";  // just printing the pointer, not dereferencing!
// Answer: UB! Even using the value of a deleted pointer is UB.

// Puzzle 3: The signed overflow trap
for (int i = 1; i > 0; i *= 2)
    std::cout << i << " ";
// What does this print? How many iterations?
// Answer: UB on overflow. Compiler may assume infinite loop, or terminate.

// Puzzle 4: Strict aliasing violation
float f = 3.14f;
int i = *(int*)&f;  // type punning through pointer cast = UB!
// Correct way: std::bit_cast<int>(f)  (C++20)
// or: std::memcpy(&i, &f, sizeof(i))

// Puzzle 5: Returning reference to local
const std::string& getName() {
    std::string name = "robot";
    return name;  // dangling reference!
}
// Many compilers warn but don't error. The caller gets garbage.

// Puzzle 6: The null reference "can't happen"
void f(int& x) { if (&x == nullptr) return; /* ... */ }
// Compiler removes the null check because references can't be null.
// But what if someone did: f(*static_cast<int*>(nullptr))?

// Puzzle 7: Order of evaluation
std::map<int, int> m;
m[0] = m.size();  // Is m[0] == 0 or m[0] == 1?
// Answer: unspecified (not UB, but non-deterministic)

// Puzzle 8: Lifetime of temporaries
const std::string& r = std::string("hello");  // OK — lifetime extended!
const std::string& r2 = std::string("hello").substr(0, 3);  // DANGLING!
// Why? Lifetime extension only applies to the direct temporary.

// Puzzle 9: The infinite loop assumption (C++11)
void compute(int* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (data[i] == 0) return;
        data[i] = process(data[i]);
    }
    // Compiler may assume this function terminates
    // If it doesn't (infinite loop with no side effects) → UB
}

// Puzzle 10: memory_order_relaxed surprise
std::atomic<int> x{0}, y{0};
// Thread 1: x.store(1, relaxed); y.store(1, relaxed);
// Thread 2: while (y.load(relaxed) != 1); assert(x.load(relaxed) == 1);
// Can the assert fire? YES! relaxed doesn't guarantee ordering.
```

**Checkpoint:**
- [ ] Can explain at least 15 of the 20 common UB forms
- [ ] Can identify UB in code that "works fine in testing"

---

### Day 44: CRTP, Mixins, Policy-Based Design — 1.5 hrs

**Code (60 min):**
```cpp
// Exercise 1: CRTP for static polymorphism (zero-cost)
template<typename Derived>
class SensorBase {
public:
    double read() { return static_cast<Derived*>(this)->read_impl(); }
    bool is_valid() { return static_cast<Derived*>(this)->is_valid_impl(); }
};

class IMU : public SensorBase<IMU> {
    friend class SensorBase<IMU>;
    double read_impl() { /* actual hardware read */ return 0.0; }
    bool is_valid_impl() { return true; }
};
// No virtual dispatch! Compiler inlines everything.

// Exercise 2: Policy-based controller
template<typename IntegrationPolicy, typename SaturationPolicy>
class PIDController {
    IntegrationPolicy integrator_;
    SaturationPolicy saturator_;
    // The policies are injected as template parameters — zero overhead
};

struct TrapezoidalIntegration { double integrate(double error, double dt); };
struct ClampSaturation { double saturate(double value, double min, double max); };
using MyPID = PIDController<TrapezoidalIntegration, ClampSaturation>;

// Exercise 3: Benchmark CRTP vs virtual dispatch
// 10M calls — measure time difference
// Expected: CRTP 2-5x faster due to inlining
```

**Checkpoint:**
- [ ] Can implement CRTP for static polymorphism
- [ ] Can explain when CRTP is better than virtual

---

### Day 45: Structured logging for production — 1.5 hrs

**Code (60 min):**
```cpp
// Exercise 1: Set up spdlog with structured JSON output
// But remember: DON'T use spdlog in the RT path!
// Use the RT-safe logger pattern from Day 16.

// Exercise 2: Implement a log-level that can be changed at runtime
// Use an atomic<int> for the level — no mutex needed for reads
std::atomic<int> log_level{LOG_INFO};

// Exercise 3: Implement a lock-free trace ring buffer
// Fixed-size entries, overwrite oldest when full
// Can be dumped post-mortem after a crash (like a flight data recorder)
template<size_t N>
class TraceBuffer {
    struct Entry {
        uint64_t timestamp_ns;
        uint32_t event_id;
        uint32_t data;
    };
    std::array<Entry, N> buffer_;
    std::atomic<uint64_t> write_pos_{0};
public:
    void record(uint32_t event_id, uint32_t data);  // RT-safe!
    std::vector<Entry> dump() const;  // non-RT: copy out for analysis
};
```

**🤖 ROS tie-in:**
- Flight data recorder pattern = exactly what your OKS rosbag recording does
- If the ring buffer was dumped on crash, you'd always have the last N events

---

### Day 46-49: WEEK 7 MINI-PROJECT + REVIEW — UB Detector Exercises — 5 hrs

**Challenge set: 20 code snippets with hidden UB**
I'll provide 20 C++ programs (50-100 lines each). Each has 1-3 subtle bugs.
Find them ALL. Use sanitizers to verify.

Then: review ALL your previous code through the "UB lens":
- [ ] SPSC queue: any signed overflow? any data race? aliasing violation?
- [ ] Pool allocator: any use-after-free paths? alignment issues?
- [ ] Cyclic executive: any integer overflow in time calculations?
- [ ] Shared memory IPC: strict aliasing? alignment?

---

## Week 8 — Capstone: Mini Flight Software
*Goal: Combine everything into a flight-software-grade application*

### Day 50-56: CAPSTONE PROJECT — Mini Flight Software — 10+ hrs

**Build a simplified satellite/robot flight software system:**

```
┌─────────────────────────────────────────────────────────┐
│                    Mode Manager                         │
│  BOOT → INIT → NOMINAL → DEGRADED → SAFE → SHUTDOWN    │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              Cyclic Executive (1kHz)                     │
│                                                          │
│  1kHz:  sensor_read()       ── pool-allocated ──>        │
│  500Hz: controller_compute()                    SPSC     │
│  100Hz: telemetry_publish() ──────────────────> Queue    │
│  10Hz:  health_check()                                   │
│  1Hz:   heartbeat_publish()                              │
│                                                          │
│  Watchdog kick every iteration                           │
│  Overrun detection + task shedding                       │
└─────────────────────────────────────────────────────────┘
          │                              │
    ┌─────▼──────┐              ┌───────▼────────┐
    │ Sensor Sim  │              │ Telemetry Proc │
    │ (noise +    │              │ (FlatBuffers   │
    │  failures)  │              │  → file)       │
    └────────────┘              └────────────────┘
```

**Components:**
1. **Mode Manager** — state machine with std::variant
2. **Cyclic Executive** — SCHED_FIFO, pinned CPU, mlockall
3. **Sensor Simulator** — generates IMU-like data with configurable noise + failure injection
4. **PID Controller** — with saturation, anti-windup, CRTP-based
5. **Telemetry Pipeline** — SPSC queue → shared memory → logger process
6. **Watchdog** — catches hangs within 100ms
7. **Health Monitor** — tracks assertion failures, overruns, sensor failures
8. **Trace Buffer** — flight-data-recorder ring buffer, dumpable
9. **RT Logger** — SPSC queue to non-RT writer thread

**Quality Requirements:**
- [ ] ALL JPL Power of 10 rules followed
- [ ] clang-tidy clean (safety-critical config)
- [ ] ASan + UBSan + TSan clean
- [ ] ≥ 80% test coverage (GTest + stress tests)
- [ ] Fuzz test for message parsing
- [ ] Zero heap allocation in RT path (verify with ASan + custom allocator)
- [ ] Runs for 60s under `stress -c $(nproc)` with max jitter < 100μs
- [ ] Modern CMake with presets
- [ ] README with architecture diagram

---

# ═══════════════════════════════════════════════════════════
# SENIOR LESSONS — What 20 Years of C++ Teaches You
# ═══════════════════════════════════════════════════════════
# These are woven into the weekly schedule as bonus sessions.
# Each is a standalone lesson you can do on a weekend or
# insert between regular days. Marked with 🧓 (elder wisdom).
# ═══════════════════════════════════════════════════════════

---

## 🧓 Senior Lesson 1: The Linker Is Where Dreams Go to Die
*Insert after Week 1 or 2*

**The Hard Truth:**
A senior C++ dev spends more time fighting the linker than writing algorithms.
The compiler checks *syntax*. The linker checks *reality*.

### What juniors never learn

**1. ODR violations — the silent killer**
```cpp
// file_a.cpp
struct Config { int timeout = 100; bool verbose = false; };

// file_b.cpp
struct Config { int timeout = 100; int verbose = 0; };  // bool vs int!

// Both compile. Both link. The program silently uses ONE definition
// chosen by the linker. Which one? UNDEFINED BEHAVIOR.
// No warning. No error. Your program just does something random.
```

**Exercise:**
```cpp
// Create this exact ODR violation in two .cpp files
// Compile and link: g++ -o odr_test a.cpp b.cpp
// Run it — observe "it works"
// Now compile with: g++ -fsanitize=undefined -o odr_test a.cpp b.cpp
// Or use gold linker: g++ -fuse-ld=gold -Wl,--detect-odr-violations
// Observe: it was broken all along
```

**2. Symbol visibility — why your .so is 10x bigger than it should be**
```cpp
// Without visibility control, EVERY symbol is exported from your .so
// A 20-year dev's first line in any shared library:
#pragma GCC visibility push(hidden)  // hide everything by default

// Then explicitly export only the public API:
#define API_PUBLIC __attribute__((visibility("default")))

class API_PUBLIC RobotController {
public:
    void start();  // exported
private:
    void internal_stuff();  // hidden — not in the symbol table
};
```

**Exercise:**
```bash
# Build a shared library with and without visibility
g++ -shared -o libfoo.so foo.cpp
nm -D libfoo.so | wc -l  # count exported symbols — will be HUGE

g++ -shared -fvisibility=hidden -o libfoo_hidden.so foo.cpp
nm -D libfoo_hidden.so | wc -l  # much smaller

# The difference is: link time, load time, binary size, symbol collision risk
```

**3. Link order matters (and it shouldn't, but it does)**
```bash
# This fails:
g++ main.o -lm -o app libcustom.a
# This works:
g++ main.o libcustom.a -lm -o app
# Why? The linker is single-pass. It resolves symbols left-to-right.
# If libcustom.a needs symbols from -lm, -lm must come AFTER.
```

**💀 20-year puzzle:**
```cpp
// You have a static library libutil.a and a shared library libcore.so
// libcore.so has a function foo() that calls bar() in libutil.a
// Your main.cpp also calls bar() directly.
// Question: How many copies of bar() exist in the final executable?
// Answer: TWO. One baked into libcore.so at so-build-time, one in your exe.
// If bar() has static variables, they are DIFFERENT variables.
// This causes "impossible" bugs where a singleton has two instances.
```

**Checkpoint:**
- [ ] Can explain ODR violations and how to detect them
- [ ] Can set symbol visibility on a shared library
- [ ] Can explain why link order matters

---

## 🧓 Senior Lesson 2: ABI Stability — Why You Can't Just Change a Struct
*Insert after Week 2*

**The Hard Truth:**
Once you ship a `.so` or a public header, the binary layout of every class
is frozen forever. Change it, and every dependent binary silently corrupts.

### The Pimpl idiom — its REAL purpose

```cpp
// Juniors think Pimpl is about "hiding implementation details."
// Seniors know it's about ABI STABILITY.

// Version 1 of your library (shipped to 100 customers as libfoo.so.1):
class RobotArm {
    double joint_angles_[6];
    bool enabled_;
public:
    void move_to(double x, double y, double z);
};
// sizeof(RobotArm) == 56  (48 + 1 + 7 padding)

// Version 2 — you need to add a new field:
class RobotArm {
    double joint_angles_[6];
    bool enabled_;
    double max_velocity_;     // NEW! Shifts everything after it
public:
    void move_to(double x, double y, double z);
};
// sizeof(RobotArm) == 64 — CHANGED!
// Every binary compiled against v1 now reads memory wrong
// when it accesses an instance created by v2 code.
// Random crashes. Data corruption. Welcome to ABI hell.

// The Pimpl fix:
class RobotArm {
    struct Impl;
    std::unique_ptr<Impl> impl_;   // sizeof(RobotArm) == 8 — FOREVER
public:
    void move_to(double x, double y, double z);
};
// Now you can add fields to Impl freely — the ABI never changes.
```

**Exercise:**
```cpp
// 1. Build a shared library v1 with a class (no Pimpl)
// 2. Build an app linked against v1
// 3. Rebuild the library as v2 (add a field in the middle of the class)
// 4. Run the old app with new library (LD_LIBRARY_PATH)
// 5. Watch it crash or produce wrong results
// 6. Fix with Pimpl, repeat — now it works

// Key command:
// pahole MyClass  — shows the exact memory layout with padding
```

**The vtable ABI trap:**
```cpp
// Adding a virtual function to a class CHANGES THE VTABLE LAYOUT
// If anyone links against your .so and uses virtual dispatch,
// adding a virtual function is an ABI BREAK.

// Safe: Add virtual functions at the END of the class declaration
// Unsafe: Add them in the middle (shifts vtable indices)
// Also unsafe: Remove a virtual function, even if unused
// Also unsafe: Change the order of base classes

// This is why Qt, KDE, and Chrome use "reserved" vtable slots:
class Widget {
    virtual void event(Event*);
    virtual void paint();
    virtual void reserved1() {} // placeholder for future use
    virtual void reserved2() {}
    virtual void reserved3() {}
};
```

**💀 20-year puzzle:**
```cpp
// This code links fine on GCC but crashes on Clang (or vice versa).
// Why? Same standard, same language, different ABIs.
struct Foo {
    std::string name;
    int id;
};
// GCC: std::string uses COW (Copy-On-Write) = 8 bytes (pointer)
// Clang's libc++: std::string uses SSO = 24 bytes (inline buffer)
// sizeof(Foo) is DIFFERENT between compilers!
// Pass Foo across a library boundary compiled with different compilers = crash
// Rule: NEVER pass STL types across shared library boundaries
```

**Checkpoint:**
- [ ] Can explain why Pimpl exists (ABI, not just "hiding")
- [ ] Can show a vtable ABI break
- [ ] Know why STL types can't cross library boundaries

---

## 🧓 Senior Lesson 3: Data-Oriented Design — AoS vs SoA & Cache Reality
*Insert during Week 3 or 5*

**The Hard Truth:**
The CPU's L1 cache is 1000x faster than RAM. The #1 performance optimization
in any program is *cache-friendly data layout*. Not algorithms. Not clever tricks.
**Data layout.**

### The cost model nobody teaches

```
L1 cache hit:    ~1 ns     (4 cycles)
L2 cache hit:    ~4 ns     (12 cycles)
L3 cache hit:    ~12 ns    (40 cycles)
RAM (DRAM):      ~60 ns    (200 cycles)
Branch mispredict: ~15 ns  (50 cycles)

Virtual function call: 0-2 ns (if predicted, ~same as direct call)
std::function call: 1-3 ns (indirect call + possible allocation)
```

**The real lesson:** Virtual calls are NOT slow. Cache misses ARE slow.
A linked list is 50-100x slower than a vector for iteration — not because
of pointer chasing overhead, but because every node is a cache miss.

### Array of Structs (AoS) vs Struct of Arrays (SoA)

```cpp
// AoS — what you'd naturally write:
struct Particle {
    float x, y, z;         // position
    float vx, vy, vz;      // velocity
    float mass;
    uint32_t color;
    bool active;
    // 33 bytes + padding = 36 bytes per particle
};
std::vector<Particle> particles(100000);

// To update positions: for (auto& p : particles) p.x += p.vx * dt;
// Cache loads 64 bytes per line, but you only use 8 bytes (x, vx)
// Cache efficiency: 8/36 = 22% — you're wasting 78% of every cache line!

// SoA — what a 20-year dev writes:
struct Particles {
    std::vector<float> x, y, z;
    std::vector<float> vx, vy, vz;
    std::vector<float> mass;
    std::vector<uint32_t> color;
    std::vector<bool> active;  // careful: vector<bool> is special
};

// Now: for (size_t i = 0; i < n; ++i) x[i] += vx[i] * dt;
// Cache loads: x is contiguous, vx is contiguous
// Cache efficiency: 100% — every byte loaded is used
// Also: compiler can now auto-vectorize with SIMD (SSE/AVX)!
```

**Exercise:**
```cpp
// Benchmark AoS vs SoA for 1M particles, 1000 frames
// Measure: time per frame, cache misses (perf stat)
// Expected: SoA is 3-10x faster for position-only updates

// perf stat -e L1-dcache-load-misses,L1-dcache-loads \
//   ./benchmark_aos
// perf stat -e L1-dcache-load-misses,L1-dcache-loads \
//   ./benchmark_soa
```

### Entity Component System (ECS) — the game industry's answer

```cpp
// ECS: entities are just IDs. Components are SoA tables.
// Commonly used in: game engines, physics simulations, robotics

// Each "system" iterates over only the components it needs
// → Perfect cache utilization, easy parallelism

// Example: A Physics system only reads (Position, Velocity, Mass)
// It doesn't touch (Color, Name, AI_State) — those don't pollute the cache
```

**💀 20-year puzzle:**
```cpp
// Which is faster for summing values, and by how much?
// A) std::vector<int> (contiguous array)
// B) std::list<int> (linked list)
// C) std::unordered_set<int> (hash table)

// Answer: A is 50-200x faster than B, 10-50x faster than C.
// NOT because of algorithmic complexity (all O(n) for traversal).
// Because vector = 1 cache miss per ~16 ints (64-byte line / 4 bytes)
// list = 1 cache miss per EVERY node (nodes scattered in heap)

// A senior never uses std::list. EVER. std::deque if you need
// stable iterators, std::vector for everything else.

// Bonus: std::unordered_map<int,int> with default allocator is
// ~7x slower than a sorted std::vector<std::pair<int,int>> for
// small maps (n < 100) because of hash table node allocation.
// Abseil's flat_hash_map or a robin-hood map fixes this.
```

**🤖 ROS tie-in:**
- PointCloud2 is stored as AoS (x,y,z,intensity per point)
- PCL's internal format is... also AoS. Cache-hostile for processing.
- If you ever need to process 100k+ points per frame, convert to SoA first

**Checkpoint:**
- [ ] Can benchmark AoS vs SoA and explain the cache difference
- [ ] Know the real cost hierarchy (cache > algorithms > instructions)
- [ ] Can explain why `std::list` is almost never the right choice

---

## 🧓 Senior Lesson 4: Debugging Production Crashes — Core Dumps & Forensics
*Insert after Week 5*

**The Hard Truth:**
In production, you can't attach a debugger. You can't reproduce it easily.
You have a core dump, syslog, and your wits. Senior devs are forensics experts.

### Setting up for success (BEFORE the crash)

```bash
# 1. Enable core dumps on your system
ulimit -c unlimited
echo "/tmp/core.%e.%p" | sudo tee /proc/sys/kernel/core_pattern

# 2. Build with debug info EVEN in release
# -g = debug symbols, -O2 = still optimized
# This combination is standard for production builds at Google, Facebook, etc.
g++ -O2 -g -o my_app main.cpp

# 3. Keep the non-stripped binary alongside the stripped one
cp my_app my_app.debug
strip my_app  # ship this
# When you get a crash, load the .debug binary in gdb
```

### Reading a core dump

```bash
# Load the core dump
gdb ./my_app.debug /tmp/core.my_app.12345

# Essential commands:
(gdb) bt                    # backtrace — where did it crash?
(gdb) bt full               # backtrace with all local variables
(gdb) frame 3               # jump to frame 3 in the stack
(gdb) info registers        # CPU register state at crash
(gdb) x/16xw $rsp           # examine 16 words of stack memory
(gdb) print *this            # print the object that crashed
(gdb) info threads           # was it a multi-thread crash?
(gdb) thread apply all bt   # every thread's backtrace

# For optimized code where variables are "optimized out":
(gdb) info registers        # the value is probably in a register
(gdb) disassemble           # read the assembly to find it
```

**Exercise:**
```cpp
// 1. Write a program that crashes (null deref, or use-after-free)
// 2. Enable core dumps, run it, get the core
// 3. Open in gdb, reconstruct what happened from ONLY the core dump
// 4. Now do the same with an OPTIMIZED build (-O2 -g)
//    — notice that some variables say "<optimized out>"
//    — learn to read registers and disassembly to find them

// 5. The HARD one: double-free crash
// The crash backtrace is in malloc/free internals.
// The CAUSE is somewhere else (the first free).
// How do you find where the first free happened?
// Answer: MALLOC_CHECK_=3 ./my_app or ASan
```

### addr2line — convert addresses to source lines

```bash
# When you only have a stack trace from syslog/journal:
# "my_app[12345]: segfault at 0x0 ip 0x55a3b4c01234 sp 0x7ffd12345678"
addr2line -e my_app.debug -f -C 0x55a3b4c01234
# Output: RobotController::update() at robot_controller.cpp:142
```

### When GDB isn't enough — the "impossible" bugs

```cpp
// Bug class 1: Heap corruption (crash happens far from the cause)
// The crash is in malloc/free, but the bug is a buffer overflow
// 10,000 lines ago. ASan finds these. In production without ASan:
// Use MALLOC_CHECK_=3 (glibc) or jemalloc's debug mode.

// Bug class 2: The Heisenbug (disappears under observation)
// Adding printf makes it work. Removing printf makes it crash.
// Cause: timing-dependent race condition.
// The printf adds enough delay to change thread scheduling.
// Fix: run with TSan, or add strategic barriers.

// Bug class 3: Works in Debug, crashes in Release
// Cause: optimization-dependent UB (see Day 43)
// The optimizer assumes no UB and removes "unnecessary" checks.
// Example: a null check is removed because "references can't be null"
// but someone passed *(int*)0.
// Fix: compile with -O2 -fsanitize=undefined

// Bug class 4: "Corrupted stack" on embedded / RT systems
// Stack overflow. Your 8KB stack in the RT thread wasn't enough.
// The stack grows into the heap or other data, corrupting everything.
// Fix: guard pages (mmap PROT_NONE after the stack)
//      or: paint the stack with a known pattern, check periodically
```

**💀 20-year puzzle:**
```cpp
// You get a crash report from a deployed robot. The backtrace shows:
//   #0  0x00007f2a... in __libc_fatal (assert.c:...)
//   #1  0x00007f2a... in __malloc_assert (malloc.c:...)
//   #2  0x00007f2a... in _int_free (malloc.c:...)
//   #3  0x0000558a... in std::vector<int>::push_back(...)
//   #4  0x0000558a... in Controller::update(...)
//
// The crash is in push_back, but push_back isn't buggy.
// What actually happened?
//
// Answer: Something earlier corrupted malloc's internal data structures
// (a buffer overflow, double free, or use-after-free somewhere).
// malloc detected the corruption only when it tried to split/merge
// a chunk. The crash location is the SYMPTOM, not the CAUSE.
// You need ASan/Valgrind replay or MALLOC_CHECK to find the real bug.
```

**Checkpoint:**
- [ ] Can load a core dump in GDB and reconstruct the crash
- [ ] Can identify the 4 classes of "impossible" production bugs
- [ ] Can use addr2line on a raw crash address

---

## 🧓 Senior Lesson 5: The Standard Library's Dark Corners
*Insert during Week 4 or 7*

**The Hard Truth:**
The STL is full of performance traps and behavior that surprises even
experienced developers. A 20-year dev has been burned by all of these.

### Iterator invalidation — the #1 STL bug source

```cpp
// EVERY senior has caused this bug at least once:
std::vector<int> v = {1, 2, 3, 4, 5};
for (auto it = v.begin(); it != v.end(); ++it) {
    if (*it == 3) {
        v.erase(it);  // 💣 iterator invalidated!
        // it++ in the for-loop header is now UB
    }
}
// Correct:
for (auto it = v.begin(); it != v.end(); ) {
    if (*it == 3) it = v.erase(it);  // erase returns next valid iterator
    else ++it;
}
// Even more correct (C++20):
std::erase(v, 3);  // just do this

// The COMPLETE invalidation rules (memorize these):
// vector: insert/push_back invalidates ALL iterators if reallocation occurs
//         (size == capacity). erase invalidates from erase point onward.
// deque:  insert/erase in middle invalidates ALL iterators.
//         insert/erase at ends only invalidates if reallocation.
// list:   insert/erase only invalidates the erased iterator. Others safe.
// map/set: insert/erase only invalidates the erased iterator. Others safe.
// unordered_*: insert invalidates ALL if rehash. Erase invalidates erased only.
```

### Small String Optimization (SSO) — why short strings are fast

```cpp
// Most std::string implementations (GCC, Clang) store short strings
// INLINE in the object, no heap allocation.
// Typical SSO threshold: 15 bytes (GCC) or 22 bytes (libc++)

std::string short_str = "hello";      // NO heap allocation — fits in SSO buffer
std::string long_str(100, 'x');       // HEAP allocation needed

// Why this matters for performance:
// If all your robot's topic names are < 15 chars, string creation is O(1)
// If they're 16+ chars, every creation is a malloc (60ns penalty)

// Exercise: determine your platform's SSO threshold
for (size_t i = 0; i < 30; ++i) {
    std::string s(i, 'x');
    // If &s[0] points inside the string object → SSO
    // If &s[0] points to heap → heap allocated
    bool is_sso = (&s[0] >= reinterpret_cast<const char*>(&s) &&
                   &s[0] < reinterpret_cast<const char*>(&s) + sizeof(s));
    std::cout << i << " bytes: " << (is_sso ? "SSO" : "HEAP") << "\n";
}
```

### The `std::map` vs `std::unordered_map` vs Sorted Vector Decision

```cpp
// Junior: "hash map is O(1), tree map is O(log n), always use hash map"
// Senior: "it depends on N, access pattern, and cache behavior"

// For N < 100 elements:       use a sorted vector + binary search
// For N < 1000, read-heavy:   use std::map (cache-friendly tree walk)
// For N > 1000, lookup-heavy: use absl::flat_hash_map (open addressing)
// NEVER use std::unordered_map for performance-critical code:
//   - It uses chained hashing (linked list per bucket → cache hostile)
//   - Each node is a separate heap allocation
//   - Rehash copies everything

// Exercise: Benchmark all 4 for insert + lookup with N = 10, 100, 10K, 1M
// You'll find sorted vector wins until ~200 elements due to cache effects
```

### `std::shared_ptr` — the silent performance killer

```cpp
// shared_ptr has hidden costs that compound in hot paths:
// 1. Two allocations: one for the object, one for the control block
//    (use make_shared to merge into one allocation)
// 2. Atomic reference count: every copy/destroy is an atomic increment/decrement
//    On x86: lock xadd (30-50 cycles). On ARM: ldrex/strex loop.
// 3. Thread-safe destruction: the destructor calls atomic decrement + conditional delete

// In a tight loop passing shared_ptr by value:
void process(std::shared_ptr<Data> data);  // atomic inc + dec per call!
void process(const std::shared_ptr<Data>& data);  // zero atomic ops — pass by ref!
void process(Data& data);  // even better if you don't need ownership

// Exercise: Benchmark these three function signatures with 1M calls
// Expected: by-value shared_ptr is 10-100x slower than by-reference

// Senior rule: shared_ptr is for OWNERSHIP TRANSFER at boundaries.
// Inside a component, use raw pointers or references.
// If nobody needs to own it, don't wrap it in shared_ptr.
```

### `std::move` doesn't move — the name is a lie

```cpp
// std::move is a CAST, not an operation. It casts to rvalue reference.
// The actual move happens in the move constructor/assignment.

std::string s = "hello";
std::string s2 = std::move(s);  // s is now in "valid but unspecified state"

// But this DOESN'T move anything:
const std::string cs = "hello";
std::string cs2 = std::move(cs);  // COPIES! const rvalue → const lvalue ref!
// No warning. No error. Just a silent copy.

// Exercise: prove this with a class that logs its special members
struct Verbose {
    Verbose() { puts("default ctor"); }
    Verbose(const Verbose&) { puts("copy ctor"); }
    Verbose(Verbose&&) { puts("move ctor"); }
    Verbose& operator=(const Verbose&) { puts("copy assign"); return *this; }
    Verbose& operator=(Verbose&&) { puts("move assign"); return *this; }
};
const Verbose cv;
Verbose v2 = std::move(cv);  // "copy ctor" — NOT "move ctor"!
```

**Checkpoint:**
- [ ] Can recite iterator invalidation rules for vector, map, unordered_map
- [ ] Can determine SSO threshold on your platform
- [ ] Know when to use sorted vector vs map vs flat_hash_map
- [ ] Can explain why `std::move` of a const object copies

---

## 🧓 Senior Lesson 6: Reading Disassembly — What the Compiler Actually Did
*Insert during Week 5 or 7*

**The Hard Truth:**
The compiler doesn't generate the code you wrote. It generates code that
has the same *observable behavior* (the "as-if" rule). To understand
performance, you must read what it actually produced.

### Godbolt — every senior's daily driver

```cpp
// Exercise 1: Compare these on godbolt.org with -O2
int sum_loop(const int* arr, int n) {        // which generates SIMD?
    int s = 0;
    for (int i = 0; i < n; ++i) s += arr[i];
    return s;
}

int sum_range(const std::vector<int>& v) {   // does this auto-vectorize?
    int s = 0;
    for (int x : v) s += x;
    return s;
}

int sum_algorithm(const std::vector<int>& v) {
    return std::accumulate(v.begin(), v.end(), 0);   // same or different?
}
// Spoiler: on GCC -O2, all three generate identical SIMD assembly

// Exercise 2: Does the virtual call get devirtualized?
struct Base { virtual int value() = 0; };
struct Derived : Base { int value() override { return 42; } };
int get_value() {
    Derived d;
    Base& b = d;
    return b.value();  // virtual call... or is it?
}
// With -O2: the compiler sees the concrete type and devirtualizes!
// The "virtual call overhead" is ZERO in this case.
```

### Reading pattern: what to look for

```cpp
// 1. Is it vectorized? (look for xmm/ymm registers, movaps, addps)
// 2. Is the branch predicted? (look for cmov instead of jmp)
// 3. Was the function inlined? (look for call instruction)
// 4. Are there cache prefetches? (look for prefetcht0)
// 5. Did the compiler eliminate dead code? (count instructions)

// Exercise 3: Write code that you THINK should vectorize but doesn't
// Then figure out why (aliasing? non-contiguous? data-dependent branch?)
void add_arrays(float* dst, const float* a, const float* b, int n) {
    for (int i = 0; i < n; ++i)
        dst[i] = a[i] + b[i];  // Does this vectorize?
}
// Answer: MAYBE. If dst aliases a or b, the compiler can't vectorize.
// Fix: void add_arrays(float* __restrict__ dst, ...)
// __restrict__ promises no aliasing → compiler vectorizes aggressively
```

### The cost of "zero-cost" abstractions

```cpp
// Exercise 4: Compare on godbolt with -O2:
// Raw loop vs std::transform vs ranges::transform
auto raw = [](std::vector<int>& v) {
    for (auto& x : v) x *= 2;
};

auto stl = [](std::vector<int>& v) {
    std::transform(v.begin(), v.end(), v.begin(), [](int x) { return x*2; });
};

auto ranges = [](std::vector<int>& v) {
    auto view = v | std::views::transform([](int x) { return x*2; });
    std::ranges::copy(view, v.begin());
};
// Check: do they generate the same assembly?
// In practice with -O2: usually yes. "Zero-cost" IS actually zero-cost.
// But: add -fno-inline and watch the abstraction cost appear.
```

**💀 20-year puzzle:**
```cpp
// This function is 50x slower than expected. The godbolt output shows
// no SIMD despite float arithmetic in a tight loop. Why?
float compute(float* data, int n) {
    float result = 0.0f;
    for (int i = 0; i < n; ++i) {
        result += data[i];
    }
    return result;
}
// Answer: Floating-point addition is NOT associative.
// (a+b)+c != a+(b+c) due to rounding.
// The compiler CANNOT reorder additions → CANNOT vectorize.
// Fix: compile with -ffast-math, or manually unroll:
// float r0=0,r1=0,r2=0,r3=0;
// for(i=0; i<n; i+=4) { r0+=data[i]; r1+=data[i+1]; ... }
// result = r0+r1+r2+r3;
```

**Checkpoint:**
- [ ] Can read basic x86 assembly on Godbolt
- [ ] Can identify if a loop was vectorized, inlined, or devirtualized
- [ ] Know about `__restrict__` and `-ffast-math` tradeoffs

---

## 🧓 Senior Lesson 7: Legacy Code Survival — Working with C APIs & Callbacks
*Insert during Week 6*

**The Hard Truth:**
The real world is 70% C code wrapped in C++ veneers. A senior dev can
bridge C and C++ seamlessly without leaking resources.

### extern "C" — what it actually does

```cpp
// C++ name-mangles functions: int foo(int) → _Z3fooi
// C doesn't: int foo(int) → foo
// extern "C" tells the C++ compiler to use C linkage (no mangling)

extern "C" {
    // These functions can be called from C code or loaded with dlsym
    int initialize(const char* config_path);
    int shutdown();
    int process(const float* data, int n, float* out);
}

// GOTCHA: You can't use extern "C" with:
// - Overloaded functions (C has no overloading)
// - Templates
// - Namespaced functions (C has no namespaces)
// - Functions with default arguments (technically works, but C callers can't use them)
```

### The callback pattern — C void* to C++ objects

```cpp
// C libraries use function pointers + void* for callbacks:
// typedef void (*callback_t)(int event, void* user_data);
// void register_callback(callback_t cb, void* user_data);

// Senior pattern: wrap with a static member (thunk):
class EventHandler {
    int state_ = 0;

    static void c_callback(int event, void* user_data) {
        auto* self = static_cast<EventHandler*>(user_data);
        self->handle(event);  // dispatch to C++ method
    }

    void handle(int event) {
        state_ += event;
    }

public:
    void register_with_c_library() {
        register_callback(&c_callback, this);
    }
};

// DANGER: The C library holds a raw pointer to your C++ object.
// If the object is destroyed while the callback is registered → UB.
// Senior fix: RAII + deregister in destructor
~EventHandler() { deregister_callback(&c_callback, this); }
```

### C resource wrapping — the definitive pattern

```cpp
// Every C resource follows the same pattern:
// create/open → use → close/destroy

// Template RAII wrapper for ANY C resource:
template<typename T, auto Deleter>
class CResource {
    T handle_;
public:
    explicit CResource(T h) : handle_(h) {}
    ~CResource() { if (handle_) Deleter(handle_); }

    CResource(const CResource&) = delete;
    CResource& operator=(const CResource&) = delete;
    CResource(CResource&& o) noexcept : handle_(std::exchange(o.handle_, {})) {}
    CResource& operator=(CResource&& o) noexcept {
        if (this != &o) { if (handle_) Deleter(handle_); handle_ = std::exchange(o.handle_, {}); }
        return *this;
    }

    T get() const { return handle_; }
    explicit operator bool() const { return handle_ != T{}; }
};

// Usage:
using FileHandle = CResource<FILE*, fclose>;
using ShmHandle = CResource<int, close>;  // for file descriptors

FileHandle f(fopen("/tmp/test", "w"));
// fclose called automatically when f goes out of scope
```

**🤖 ROS tie-in:**
```cpp
// ROS1 is C++03 with Boost everywhere. ROS2 is modern C++ under the hood,
// but the C middleware (rmw_fastrtps, rmw_cyclonedds) uses plain C APIs.
// Your OKS robot's base controller talks to hardware via C APIs.
// This pattern is exactly how ros_control wraps hardware interfaces.
```

**Checkpoint:**
- [ ] Can write a C-callable wrapper for a C++ class
- [ ] Can wrap any C resource in RAII with the CResource pattern
- [ ] Know the void* callback dispatch pattern

---

## 🧓 Senior Lesson 8: Build Times at Scale — When Compilation Takes 45 Minutes
*Insert during Week 5*

**The Hard Truth:**
On a large project (500+ files, millions of LOC), a full rebuild takes
30-60 minutes. Incremental builds take 2-10 minutes. Compile time
IS development speed. Seniors obsess over it.

### The enemies of compile time

```cpp
// 1. Headers that include the world
// #include <algorithm>  — pulls in 60,000+ lines on GCC 12
// #include <iostream>   — pulls in 30,000+ lines
// #include <regex>      — pulls in 100,000+ lines (!!!)

// Exercise: measure include cost
echo '#include <vector>' | g++ -x c++ -std=c++20 -E - | wc -l    # ~15,000
echo '#include <iostream>' | g++ -x c++ -std=c++20 -E - | wc -l  # ~30,000
echo '#include <regex>' | g++ -x c++ -std=c++20 -E - | wc -l     # ~100,000+
echo '#include <string>' | g++ -x c++ -std=c++20 -E - | wc -l    # ~20,000

// 2. Templates in headers (instantiated in EVERY translation unit)
// 3. Inline functions in headers (re-compiled in EVERY .cpp)
// 4. Unnecessary includes (header A includes header B, nobody needs B)
```

### Senior build-time techniques

```cpp
// 1. Forward declarations — THE most effective technique
// Bad:
#include "robot_controller.h"  // full class definition, brings in everything
class Planner {
    RobotController* controller_;  // only need a pointer!
};

// Good:
class RobotController;  // forward declaration — zero include cost
class Planner {
    RobotController* controller_;  // works fine with just a declaration
};

// Rule: if you only use pointers/references to a type, forward-declare.
// Only #include when you call methods, use sizeof, or embed by value.

// 2. Precompiled headers (PCH)
# In CMake:
target_precompile_headers(my_target PRIVATE
    <vector>
    <string>
    <memory>
    <algorithm>
    "common_types.h"
)
# These headers are parsed ONCE, then reused as binary blob

// 3. Unity builds (jumbo compilation)
// Concatenate N .cpp files into one, compile as one translation unit
# In CMake:
set(CMAKE_UNITY_BUILD ON)
set(CMAKE_UNITY_BUILD_BATCH_SIZE 16)
// Pro: 2-5x faster builds, better inlining across files
// Con: ODR violations become more likely, harder to understand errors

// 4. ccache — compiler cache
$ ccache g++ -O2 -g -o foo foo.cpp
// First compile: normal time, result cached
// Second compile (same input): ~0.1s (just copies cached result)
// Put in CMake:
find_program(CCACHE ccache)
if(CCACHE)
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
endif()

// 5. Use -ftime-report to find what's slow
g++ -ftime-report -c slow_file.cpp
// Shows: parsing time, template instantiation time, code generation time
// If template instantiation dominates → move templates to .cpp with explicit instantiation
```

### Include-what-you-use (IWYU)

```bash
# Find unnecessary includes
include-what-you-use my_file.cpp
# Output tells you which includes to add and which to remove
# On large projects, removing 30% of includes is common
# Build time drops 20-40%
```

**💀 20-year puzzle:**
```cpp
// Your 200-file project takes 5 minutes for incremental build.
// You touch one header. Now ALL 200 files recompile. Why?
//
// Answer: That header is included (directly or transitively) by
// a header that every .cpp file includes (like "common.h" or "types.h").
// This is the "fragile header" problem.
//
// Fix: The header should contain ONLY forward declarations and trivial types.
// Move implementations to .cpp files. Use Pimpl for complex classes.
// Check with: g++ -M main.cpp | tr ' ' '\n' | sort | wc -l
//            (count how many headers main.cpp transitively includes)
```

**Checkpoint:**
- [ ] Can reduce compile time by 2x using forward declarations + PCH
- [ ] Can use `-ftime-report` to identify compile bottlenecks
- [ ] Can explain why touching one header recompiles the world

---

## 🧓 Senior Lesson 9: The "Works in Debug, Crashes in Release" Bug
*Insert during Week 7*

**The Hard Truth:**
Debug and Release are almost different languages. The optimizer assumes
your code is correct and transforms it aggressively. If you have UB,
debug mode hides it, release mode exposes it (or vice versa).

### Why debug and release behave differently

```cpp
// 1. Uninitialized variables
int x;  // debug: initialized to 0 (or 0xCDCDCDCD on MSVC)
         // release: whatever was in that stack location
if (x > 0) { crash(); }
// Debug: never crashes (x is 0)
// Release: crashes 50% of the time (x is random garbage)

// 2. Optimization removes "dead" code
int* p = get_pointer();  // might return null
*p = 42;                 // release: compiler assumes p != null (otherwise UB)
if (p == nullptr) {      // release: REMOVED! "can't be null, we just dereffed it"
    handle_error();
}
// Debug: handle_error() runs when p is null
// Release: handle_error() is never called. Program crashes on *p = 42.

// 3. Signed overflow
int x = INT_MAX;
if (x + 1 > x) {  // mathematically always true...
    // release: REMOVED because signed overflow is UB
    // compiler assumes x + 1 > x is always true → removes the check
    do_something();
}

// 4. Timing differences expose races
// Debug builds are 10-100x slower → threads interleave differently
// A race condition that never triggers in debug triggers every time in release

// 5. Guard pages and memory layout
// Debug: extra padding around allocations (catches buffer overflows)
// Release: objects packed tightly (overflow silently corrupts neighbors)
```

**Exercise:**
```cpp
// Create each of these 5 bug types:
// 1. Uninitialized variable that "works" in debug
// 2. Null check removed by optimizer
// 3. Signed overflow that changes behavior
// 4. Race condition only visible in release
// 5. Buffer overflow hidden by debug padding

// For each:
// - Compile debug (-O0 -g) and release (-O2 -g)
// - Show different behavior
// - Find with the appropriate sanitizer
```

**💀 20-year puzzle:**
```cpp
// This code works perfectly in debug and release for 3 years.
// One day, after upgrading from GCC 11 to GCC 13, it "starts crashing."
// The code hasn't changed. What happened?
void process(int* data, int len) {
    for (int i = 0; i <= len; ++i) {  // off-by-one: should be < len
        data[i] = transform(data[i]);
    }
}
// Answer: The old compiler generated bounds-safe code by coincidence.
// The new compiler version has a more aggressive optimizer that exploits
// the UB (buffer overflow) to vectorize the loop differently.
// The bug was ALWAYS there. The new compiler just stopped hiding it.
// Lesson: "it works" is not the same as "it's correct."
```

**Checkpoint:**
- [ ] Can explain 5 reasons debug and release behave differently
- [ ] Can create and diagnose a "works in debug, crashes in release" bug
- [ ] Understand that "it works" ≠ "it's correct"

---

## 🧓 Senior Lesson 10: Design Decisions You Only Learn from Shipping
*Standalone — review anytime*

These are not code exercises. These are **engineering judgment** lessons.
A 20-year dev knows these from painful experience.

### 1. "Don't use exceptions" vs "Do use exceptions"

```
Google, LLVM, Unreal Engine:   -fno-exceptions (banned)
Bloomberg, Qt, most startups:  exceptions (allowed)
Safety-critical (DO-178C):     banned
Real-time:                     banned (non-deterministic unwinding)

When a senior hears "should we use exceptions?" they ask:
- Is this real-time? → No exceptions
- Does it link with C code? → No exceptions (C doesn't have them)
- Is the team > 50 people? → Exceptions are likely banned (style guide)
- Is it a library used by others? → Exceptions at THE boundary only

The real answer: use Expected<T,E> internally, exceptions only at the
top-level boundary if your users expect them.
```

### 2. How much to template

```
Junior:  templates everything (800-line template metaprogramming headers)
Senior:  templates the hot path, virtual dispatch for cold paths

The rule: If it's called >1M times per second, template it.
If it's called at initialization or config time, use virtual dispatch.
The compile-time cost of templates is paid by EVERY developer on the team.
The runtime cost of virtual dispatch is paid by users, but is tiny (1-2ns).
```

### 3. When to use `std::string` vs `std::string_view` vs `const char*`

```cpp
// The rule a senior follows:
// - Function takes string input → std::string_view (zero-copy, works with both)
// - Function returns a NEW string → std::string (owns the data)
// - Returning a reference to internal data → std::string_view (but DANGEROUS)
// - C API boundary → const char* (with size parameter)
// - Storing in a data structure → std::string (owns the data)

// NEVER:
// - Return string_view to a local std::string (dangling!)
// - Store string_view for later use (the string might be gone)
// - Pass string_view to a function that stores it
```

### 4. Error handling strategy

```
Layer 1 (hardware/OS boundary): Check EVERY return value. errno. strerror.
Layer 2 (internal library):     Expected<T, ErrorCode>.
Layer 3 (domain logic):         Variant-based error algebraic types.
Layer 4 (app boundary):         Exception (if allowed) or error code.
Layer 5 (UI/user report):       Human-readable messages with context.

Never propagate errno to the UI. Never throw from a destructor.
Never ignore an error "because it never happens in practice."
```

### 5. Comments: what a 20-year dev actually comments

```cpp
// Juniors: comment WHAT the code does (useless — read the code)
// x++; // increment x              ← NEVER do this

// Seniors: comment WHY, TRADEOFFS, and INVARIANTS
//
// Use fences instead of seq_cst here because profiling showed
// the seq_cst load in the critical path costs 12ns on ARM (LDAR).
// Fence + relaxed load costs 3ns. See benchmark results in #PR-4231.
//
// INVARIANT: buffer_size_ is always a power of 2 (enforced by ctor).
// This allows using bitwise AND instead of modulo for wrapping.
//
// WARNING: Do NOT reorder these two stores. The reader thread
// observes `data` and `ready` flag. If `ready` is visible before
// `data` is written, reader gets garbage. Release on `ready` ensures
// `data` is visible first. See Preshing "Acquire and Release Semantics".
//
// TODO(viku): This CAS loop spins unbounded under contention.
// If more than 4 producers are expected, switch to a ticket lock.
// Current max observed spin was 340 iterations in stress test (2026-04-15).
```

### 6. The one-binary rule

```
Ship ONE binary that handles all configurations via runtime config.
Do NOT ship debug/release/test/staging as different binaries.

Reason: The binary that passes your tests must be the binary in production.
If you test the debug build and ship the release build, you tested nothing.

How: compile with -O2 -g (optimized + debug info).
Use a build-time flag for EXPENSIVE assertions only:
#ifdef PARANOID_CHECKS  // enabled in CI, disabled in production
    full_validation();
#endif
```

### 7. DRY is sometimes wrong

```cpp
// Junior: sees two 5-line blocks that look similar → extracts common function
// Senior: asks "will these EVOLVE together or separately?"

// If they're in different modules (e.g., navigation and charging)
// they may look the same today but diverge next month.
// Extracting a shared function creates a COUPLING between them.
// When one module changes, the shared function changes, and the
// other module breaks.

// Rule: Duplicate code in DIFFERENT domains is OK.
//       Duplicate code in the SAME domain must be extracted.
// The test: "If I change one, must I always change the other?"
//           Yes → extract. No → keep separate.
```

### 8. Debugging is twice as hard as writing code

```
    "Everyone knows that debugging is twice as hard as writing a program
     in the first place. So if you're as clever as you can be when you
     write it, how will you ever debug it?"
                                            — Brian Kernighan

A 20-year dev writes SIMPLE code because they know their future self
will have to debug it at 3 AM on a Friday with a production outage.

The cleverness budget is limited. Spend it on architecture, not on
"look how smart this template metaprogram is."
```

---

# ═══════════════════════════════════════════════════════════
# HARDWARE-LEVEL LESSONS — What the Electrons Actually Do
# ═══════════════════════════════════════════════════════════
# These teach you what's UNDER the C++ abstractions.
# A firmware/robotics engineer who knows these destroys bugs
# that pure-software devs can't even see.
# Marked with ⚡ (hardware wisdom).
# ═══════════════════════════════════════════════════════════

---

## ⚡ Hardware Lesson 1: Clock Trees, PLLs & Why Your Timer Is Wrong
*Insert during Week 3 (RT programming)*

**Why this matters:**
Every `clock_gettime()`, every `std::chrono::steady_clock`, every SPI transfer
timing — they all depend on the hardware clock tree. If you don't understand
it, your "1kHz loop" might be running at 999.7Hz and drifting 26ms/minute.

### How a CPU gets its clock

```
                    ┌──────────┐
  Crystal (HSE)     │          │    SYSCLK
  8 MHz ──────────> │   PLL    │ ──────────> 168 MHz (STM32F4)
                    │ x21      │              480 MHz (STM32H7)
  Internal RC (HSI) │          │              1.8 GHz (Cortex-A72, Jetson)
  16 MHz ──────┬──> └──────────┘
               │
               │     ┌──────────────────────────────────────┐
               │     │         Clock Tree (Prescalers)       │
               │     │                                       │
               │     │  SYSCLK ──/1──> AHB bus (168 MHz)    │
               │     │  AHB ────/2──> APB1 bus (42 MHz)     │
               │     │  AHB ────/1──> APB2 bus (84 MHz)     │
               │     │  APB1 ──────> UART, I2C, SPI1        │
               │     │  APB2 ──────> SPI2, ADC, Timers      │
               │     └──────────────────────────────────────┘
```

**The critical insight:**
```
Crystal oscillator: ±20 ppm accuracy  (20 parts per million)
Internal RC:        ±1% accuracy      (10,000 ppm!)

20 ppm = 20 μs error per second = 1.73 seconds drift per day
1%     = 10 ms error per second = 14.4 MINUTES drift per day

If your OKS robot uses the internal RC for timestamps,
two robots' clocks can drift 29 minutes/day relative to each other.
That's why GPS/NTP/PTP time sync exists.
```

**Exercise 1 — Read your system's clock sources:**
```bash
# On Linux (your Jetson / dev machine):
cat /sys/devices/system/clocksource/clocksource0/available_clocksource
# Typical: tsc hpet acpi_pm
# TSC = Time Stamp Counter (CPU cycles), fastest

cat /sys/devices/system/clocksource/clocksource0/current_clocksource
# Should be: tsc (on x86) or arch_sys_counter (on ARM)

# Measure clock resolution:
cat /proc/timer_list | head -50
# Shows: ktime resolution in nanoseconds

# In C++:
#include <chrono>
auto res = std::chrono::steady_clock::period::num;
auto den = std::chrono::steady_clock::period::den;
// res/den = clock tick period in seconds
// On modern Linux: 1 nanosecond resolution (but ~25ns actual precision)
```

**Exercise 2 — Clock drift measurement:**
```cpp
// Measure how much CLOCK_MONOTONIC drifts vs CLOCK_REALTIME over 60s
// REALTIME is NTP-adjusted, MONOTONIC is free-running
// The difference tells you your crystal's actual error

#include <time.h>
void measure_drift() {
    struct timespec mono_start, real_start, mono_end, real_end;
    clock_gettime(CLOCK_MONOTONIC, &mono_start);
    clock_gettime(CLOCK_REALTIME, &real_start);

    sleep(60);

    clock_gettime(CLOCK_MONOTONIC, &mono_end);
    clock_gettime(CLOCK_REALTIME, &real_end);

    double mono_elapsed = (mono_end.tv_sec - mono_start.tv_sec)
                        + (mono_end.tv_nsec - mono_start.tv_nsec) * 1e-9;
    double real_elapsed = (real_end.tv_sec - real_start.tv_sec)
                        + (real_end.tv_nsec - real_start.tv_nsec) * 1e-9;
    double drift_ppm = (mono_elapsed - real_elapsed) / real_elapsed * 1e6;
    printf("Drift: %.1f ppm (%.3f μs/s)\n", drift_ppm, drift_ppm);
}
```

**Exercise 3 — Timer hardware on STM32 (read along, code on your Jetson):**
```cpp
// On STM32 (your OKS robot's MCU), hardware timers are the backbone:
//
// TIM1-TIM14: each is a 16/32-bit counter driven by the clock tree
// Each timer has:
//   - Prescaler (PSC): divides input clock
//   - Auto-reload register (ARR): counter wraps at this value
//   - Capture/Compare registers (CCR): trigger events at specific counts
//
// Timer frequency = APB_timer_clock / (PSC + 1) / (ARR + 1)
//
// Example: 1kHz interrupt from 84 MHz APB2:
//   PSC = 83    → 84 MHz / 84 = 1 MHz tick
//   ARR = 999   → 1 MHz / 1000 = 1 kHz interrupt
//
// YOUR SPI bus runs on a timer-derived clock.
// If PSC is wrong, SPI runs at wrong speed → sensorbar communication fails.
// This is how your sensorbar duplicate data issue (#99185) is clock-adjacent.

// On Linux/Jetson — use hrtimer for high-resolution periodic wakeup:
// (This is what clock_nanosleep uses underneath)
// Check available resolution:
struct timespec res;
clock_getres(CLOCK_MONOTONIC, &res);
printf("Clock resolution: %ld ns\n", res.tv_nsec);
// Typical x86: 1ns resolution
// Typical ARM: 1ns or 52ns (depends on arch_timer frequency)
```

**💀 Hardware puzzle:**
```
Your robot's main loop runs at "1 kHz" but telemetry shows it's actually
at 999.2 Hz. Over 24 hours, it misses 69,120 cycles.

Q: The crystal is rated ±30 ppm. Can this explain the error?
A: 1000 Hz × 30 ppm = 0.03 Hz error. Expected: 999.97–1000.03 Hz.
   But you measure 999.2 Hz = 800 ppm error. Way too much for the crystal.
   → The prescaler or ARR is set wrong, OR the PLL multiplication factor
   is misconfigured, OR you're running from the internal RC oscillator
   instead of the crystal.
   
   Debug: read the RCC registers to check which clock source is active.
   On Linux: check /sys/devices/system/clocksource/
```

**🤖 ROS tie-in:**
- `ros::Time::now()` uses `CLOCK_REALTIME` — can JUMP when NTP corrects
- `ros::SteadyTime::now()` uses `CLOCK_MONOTONIC` — never jumps, only drifts
- Your OKS robot timestamps: if two sensor boards use different crystals,
  their timestamps drift. The estimator must handle this!
- PTP (Precision Time Protocol) sync on the SPI bus would fix inter-board drift

**Checkpoint:**
- [ ] Can explain clock tree: crystal → PLL → SYSCLK → bus prescalers
- [ ] Can measure your system's actual clock drift in ppm
- [ ] Know why internal RC vs crystal matters (1% vs 20ppm)

---

## ⚡ Hardware Lesson 2: Resistors, Pull-ups & Why Your I2C Hangs
*Insert during Week 3 or 6*

**Why this matters:**
Every GPIO, every I2C bus, every SPI chip-select, every UART line — they
all need correct resistor configurations. A missing pull-up resistor on I2C
will cause random communication failures that look like software bugs.

### Pull-up and pull-down resistors — the physical reality

```
VCC (3.3V)                    VCC (3.3V)
    │                              │
    ┣━━ 4.7 kΩ (pull-up)          ┣━━ 10 kΩ (pull-up)
    │                              │
    ├─── SDA ◄──── I2C bus         ├─── GPIO pin
    │                              │
   ═╧═ (open drain output)       ═╧═
   GND                           GND

Without pull-up:                 With pull-up:
    │                              │
   ???  (floating!)                ┣━━ 4.7 kΩ → VCC
    │                                   │
 Could be 0V, 3.3V, 1.7V,        Guaranteed HIGH when
 quantum noise... undefined!      nothing is driving LOW
```

**Why I2C NEEDS pull-ups:**
```
I2C uses "open-drain" outputs:
- A device can PULL the line LOW (connect to GND)
- A device can RELEASE the line (disconnect)
- NOTHING can DRIVE the line HIGH actively!
- The pull-up resistor pulls it HIGH when released

Without pull-up → line floats → random 0s and 1s → data corruption

Pull-up value matters:
- Too high (100 kΩ): slow rise time → bits blur together → errors
- Too low (100 Ω): too much current when line is LOW → chip damage
- Sweet spot: 2.2–10 kΩ depending on bus speed and capacitance

Rise time ≈ 0.35 × R × C
  R = pull-up resistance
  C = bus capacitance (depends on wire length + number of devices)

For 400 kHz I2C (fast mode):
  Max rise time = 300 ns
  Typical bus C = 100 pF (short bus, 2 devices)
  R = 300 ns / (0.35 × 100 pF) = 8.6 kΩ max
  → Use 4.7 kΩ (standard recommendation)

For 1 MHz I2C (fast mode plus):
  Max rise time = 120 ns
  R = 120 ns / (0.35 × 100 pF) = 3.4 kΩ max
  → Use 2.2 kΩ
```

**Exercise 1 — GPIO internal pull-ups (Linux sysfs / libgpiod):**
```bash
# Check if your Jetson GPIO has internal pull-ups enabled:
# Using gpiomon (libgpiod-tools):
sudo gpioinfo | grep -i pull

# Configure a GPIO with internal pull-up (device-tree or runtime):
# In device tree: bias-pull-up;
# In code:
```
```cpp
#include <gpiod.h>
struct gpiod_line_request_config config = {};
config.consumer = "my_app";
config.request_type = GPIOD_LINE_REQUEST_DIRECTION_INPUT;
config.flags = GPIOD_LINE_REQUEST_FLAG_BIAS_PULL_UP;
gpiod_line_request(line, &config, 0);
```

**Exercise 2 — SPI signal integrity:**
```
Your OKS sensorbar SPI bus:

MOSI ─────────────────── Sensorbar MCU
MISO ◄──────────────────
SCLK ─────────────────────
CS   ─────────────────────

Issues at high SPI speed (8+ MHz):
┌────────────────────────────────┐
│ Signal Integrity Problems:     │
│                                │
│ 1. Ringing on SCLK edges      │
│    → False clock edges         │
│    → Extra bits clocked in     │
│    → Data shifted by 1 bit     │
│                                │
│ 2. Crosstalk MOSI→MISO        │
│    → Noise on received data    │
│    → Bit errors                │
│                                │
│ 3. Ground bounce               │
│    → Multiple devices switch   │
│    → Ground voltage rises      │
│    → Logic levels shift        │
│                                │
│ Fix: Series resistors (33Ω)   │
│      on SCLK and MOSI close   │
│      to the driver. These      │
│      dampen reflections.       │
│                                │
│ Fix: Decoupling capacitors     │
│      (100nF ceramic) on VCC    │
│      near every IC.            │
└────────────────────────────────┘
```

**Exercise 3 — Voltage divider & ADC reading:**
```cpp
// Reading a battery voltage that's too high for the ADC (12V battery, 3.3V ADC):
//
//  12V ──┤ R1=30kΩ ├──┬── ADC pin (0–3.3V range)
//                      │
//                     ┤ R2=10kΩ ├── GND
//
// Voltage divider: V_adc = V_bat × R2 / (R1 + R2)
//                        = 12V × 10k / 40k = 3.0V  ✓ within ADC range
//
// ADC conversion (12-bit ADC, 3.3V reference):
// raw = (V_adc / V_ref) × 4095 = (3.0 / 3.3) × 4095 = 3722
// V_bat = raw × V_ref / 4095 × (R1 + R2) / R2
//       = 3722 × 3.3 / 4095 × 40k / 10k = 12.0V

constexpr double R1 = 30000.0;  // 30 kΩ
constexpr double R2 = 10000.0;  // 10 kΩ
constexpr double V_REF = 3.3;
constexpr int ADC_BITS = 12;
constexpr int ADC_MAX = (1 << ADC_BITS) - 1;  // 4095

double adc_to_battery_voltage(int raw_adc) {
    double v_adc = static_cast<double>(raw_adc) / ADC_MAX * V_REF;
    double v_bat = v_adc * (R1 + R2) / R2;
    return v_bat;
}
// Exercise: calculate R1, R2 for a 48V battery system (your OKS robot!)
// Constraint: R2 current < 1mA (for low power), V_adc < 3.0V (margin)
```

**💀 Hardware puzzle:**
```
Your I2C temperature sensor works perfectly for 3 hours, then starts
returning 0xFF for all reads. Reboot fixes it. What happened?

Answer: I2C bus lockup. A noise glitch during a read caused the sensor
to miss a clock pulse. It's now holding SDA LOW (in the middle of
sending a byte), waiting for a clock that already happened.
The master sees SDA stuck LOW and can't send a START condition.

Fix: Implement I2C bus recovery in your driver:
  1. Toggle SCL up to 9 times while SDA is LOW
  2. This clocks out the slave's incomplete byte
  3. Then send a STOP condition (SDA low→high while SCL high)
  4. Re-initialize the bus

Senior fix: Use a hardware I2C bus recovery circuit:
  TCA9406 (level translator with bus disconnect) or
  LTC4306 (I2C mux with stuck-bus recovery)
```

**🤖 ROS tie-in:**
- Your OKS sensorbar's SPI issues: series resistors on SCLK reduce ringing
- The `is_reliable` flag in sensorbar firmware — it checks SPI CRC, not data freshness
- Battery monitoring through voltage divider → Published on `/battery_state` topic

**Checkpoint:**
- [ ] Can calculate a voltage divider for a given ADC and battery voltage
- [ ] Can explain why I2C needs pull-ups and how to calculate the value
- [ ] Know what series resistors and decoupling caps do for SPI signal integrity

---

## ⚡ Hardware Lesson 3: DMA, Interrupts & Zero-Copy From Hardware
*Insert during Week 3 or 6*

**Why this matters:**
Every sensor read on your OKS robot uses DMA (Direct Memory Access).
Understanding DMA is the difference between your SPI running at 100%
efficiency with zero CPU cost vs burning 40% of your CPU on data copies.

### Interrupts — the hardware calling YOUR code

```
Normal execution:          With interrupt:
┌─────────────────┐        ┌─────────────────┐
│ main loop       │        │ main loop       │
│ ...             │        │ ...             │
│ controller()    │        │ control──┐      │
│ ...             │        │          ▼      │
│ publish()       │        │  ┌──────────┐   │
│ ...             │        │  │ ISR:     │   │
└─────────────────┘        │  │ read SPI │   │
                           │  │ set flag │   │
                           │  └────┬─────┘   │
                           │       ▼         │
                           │ ──ller()        │ ← resumes where it left off
                           │ ...             │
                           └─────────────────┘

ISR (Interrupt Service Routine) rules:
1. MUST be FAST (<1 μs for sensor ISR, <10 μs max)
2. NO allocation, NO printf, NO mutex
3. NO floating point (on ARM Cortex-M, FPU context not always saved)
4. Set a flag or enqueue data, then return
5. The main loop processes the data

This is the ISR → SPSC queue → main_loop pattern from Week 2+3!
```

**Exercise 1 — Interrupt latency measurement on Linux:**
```cpp
// On Linux, you don't write ISRs directly (the kernel does).
// But you CAN measure interrupt-to-userspace latency:

// Method: Use a GPIO interrupt and measure response time
#include <gpiod.h>
#include <time.h>

// 1. Configure a GPIO as interrupt source (rising edge)
// 2. In a tight loop, measure time between hardware edge and your read
// 3. This tells you: hardware interrupt → kernel ISR → wake userspace → your code

// Expected: 5–50 μs on vanilla Linux, 1–10 μs on PREEMPT_RT
// Your OKS robot needs < 5 μs to not miss SPI frames at 1 MHz clock

// On bare metal (STM32):
// Interrupt latency = 12 cycles (Cortex-M4) = 71 ns at 168 MHz
// This is why embedded beats Linux for hard real-time!
```

### DMA — hardware moves data while CPU sleeps

```
Without DMA (PIO — Programmed I/O):
┌───────────┐                  ┌─────────────┐
│    CPU     │  for each byte: │  SPI FIFO   │
│            │  1. wait ready  │             │
│  40% busy! │  2. read byte   │   ← sensor  │
│            │  3. store byte  │     data     │
│            │  4. repeat ×128 │             │
└───────────┘                  └─────────────┘
 CPU does ALL the work. 128 bytes × ~100ns = 12.8 μs of CPU time.

With DMA:
┌───────────┐                  ┌─────────────┐
│    CPU     │  1. configure   │  SPI FIFO   │
│            │     DMA once    │             │
│  0% busy! │                  │   ← sensor  │
│  (sleep)   │    ┌────────┐   │     data     │
│            │    │  DMA   │◄──┤             │
│            │    │ engine │──►│ RAM buffer  │
│ ──────────│    └────────┘   │             │
│ interrupt! │  2. DMA done    │             │
│ process!   │     interrupt   │             │
└───────────┘                  └─────────────┘
 CPU configures DMA, goes to sleep. DMA engine moves all 128 bytes.
 CPU wakes up ONCE when all data is ready. Zero CPU cycles for transfer.
```

**Exercise 2 — DMA and cache coherency (THE senior trap):**
```cpp
// On Cortex-A (Jetson, RPi): caches exist between CPU and RAM
//
// Problem:
//   1. CPU writes data to RAM buffer
//   2. DMA reads from RAM buffer → reads STALE data (not in cache!)
//   3. DMA writes results to RAM buffer
//   4. CPU reads RAM buffer → reads STALE data (cache has old copy!)
//
// This is the #1 embedded DMA bug that takes weeks to find.
//
// Solution on Linux:
//   Use dma_alloc_coherent() → non-cacheable memory
//   Or: manual cache operations:
//     - Before DMA read:  cache_flush(buffer)     (write cache → RAM)
//     - After DMA write:  cache_invalidate(buffer) (discard cache line)

// In userspace, you encounter this with mmap'd DMA buffers:
int fd = open("/dev/dma_device", O_RDWR);
void* buf = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0);
// This mmap is typically "uncached" or "write-combining"
// Reads from this buffer are SLOW (~60ns vs ~1ns for cached)
// But data is always fresh from hardware

// On Cortex-M (STM32): no cache on most models → no problem!
// On STM32H7: has cache! Must use __DSB() and SCB_CleanDCache_by_Addr()
```

**Exercise 3 — Implement a zero-copy SPI reader (conceptual + Linux):**
```cpp
// For your OKS sensorbar: DMA-based SPI read pattern
//
// Hardware setup:
//   SPI DMA reads 128 bytes every 1ms (sensorbar data frame)
//   Double buffer: DMA fills buffer A while CPU processes buffer B
//
//   ┌──────────┐      ┌──────────┐
//   │ Buffer A │◄─DMA─│   SPI    │
//   ├──────────┤      │ hardware │
//   │ Buffer B │◄─────│          │
//   └──────────┘      └──────────┘
//        │
//        ▼ CPU processes whichever buffer DMA isn't using
//
// In Linux userspace (using spidev):
#include <linux/spi/spidev.h>
#include <sys/ioctl.h>

struct spi_ioc_transfer xfer = {};
xfer.tx_buf = (unsigned long)tx_buf;
xfer.rx_buf = (unsigned long)rx_buf;
xfer.len = 128;
xfer.speed_hz = 8000000;  // 8 MHz SPI clock
xfer.bits_per_word = 8;
int ret = ioctl(fd, SPI_IOC_MESSAGE(1), &xfer);
// The kernel's SPI driver uses DMA internally for large transfers
// But: each ioctl is a syscall → context switch → non-deterministic
// For hard RT: use a kernel module or PREEMPT_RT + mlockall
```

**💀 Hardware puzzle:**
```
Your DMA transfer from SPI works 99.9% of the time.
Every ~1000 transfers, the last 4 bytes are corrupted (0x00000000).

Q: The DMA length is correct. The SPI clock is stable. What's wrong?

A: Cache coherency bug. The CPU read the buffer BEFORE the DMA finished
   writing the last cache line (64 bytes). The CPU's cache had stale data
   from the previous transfer. The first 124 bytes happened to match
   because DMA wrote them early enough to evict the cache line.
   The last 4 bytes were still cached from the previous (zeroed) transfer.

   Fix: Insert a memory barrier + cache invalidate AFTER the DMA
   complete interrupt, BEFORE the CPU accesses the buffer.
   On ARM: __DSB(); __ISB();
   On Linux userspace: the kernel should handle this, but some drivers
   have bugs. Check with: dmesg | grep -i "DMA\|cache\|coherent"
```

**🤖 ROS tie-in:**
- Your sensorbar SPI reads 128 bytes per frame at 1 kHz via DMA
- The duplicate data bug (#99185): if DMA doesn't complete before next frame,
  the CPU reads the old buffer → duplicate! Double-buffering prevents this.
- `ros_control` hardware_interface: the `read()` function should only
  copy the latest DMA buffer, not trigger a new SPI transfer.

**Checkpoint:**
- [ ] Can explain DMA transfer vs PIO and when each is used
- [ ] Understand the DMA + CPU cache coherency problem
- [ ] Know the double-buffer pattern for continuous sensor reads

---

## ⚡ Hardware Lesson 4: The Linux Scheduler Deep Dive
*Insert during Week 3 (extends the RT programming week)*

**Why this matters:**
Your RT application asks the kernel to wake it up every 1ms.
Understanding HOW the scheduler decides who runs next is the difference
between "it usually works" and "it provably meets deadlines."

### The three schedulers in Linux

```
┌──────────────────────────────────────────────────────────┐
│                Linux Scheduler Architecture               │
│                                                           │
│  SCHED_DEADLINE ──► Earliest Deadline First (EDF)        │
│  (priority 0)       Highest priority. Guaranteed CPU.    │
│                     Parameters: runtime, period, deadline │
│                                                           │
│  SCHED_FIFO ──────► Fixed Priority First-In First-Out    │
│  SCHED_RR           Priority 1–99. Higher runs first.    │
│  (priority 1-99)    FIFO: runs until it blocks/yields    │
│                     RR: time-sliced among same priority   │
│                                                           │
│  SCHED_OTHER ─────► CFS (Completely Fair Scheduler)      │
│  (priority 0)       Nice values -20 to +19               │
│                     Not real-time. "Fair" sharing.        │
│                     99.9% of processes use this.          │
└──────────────────────────────────────────────────────────┘

Priority hierarchy:
  DEADLINE > FIFO/RR > OTHER
  A SCHED_FIFO prio 1 thread ALWAYS preempts ANY SCHED_OTHER thread
  A SCHED_DEADLINE thread ALWAYS preempts ANY SCHED_FIFO thread
```

### SCHED_DEADLINE — the mathematically correct way

```cpp
// SCHED_FIFO has a problem: priority assignment is manual and fragile.
// Add a new thread → must recalculate all priorities.
// SCHED_DEADLINE uses EDF: each task declares its timing requirements.
// The kernel PROVES (at admission time) that all deadlines can be met.

#include <sched.h>
#include <linux/sched.h>

struct sched_attr {
    uint32_t size;
    uint32_t sched_policy;
    uint64_t sched_flags;
    int32_t  sched_nice;
    uint32_t sched_priority;
    // DEADLINE-specific:
    uint64_t sched_runtime;   // CPU time needed per period (ns)
    uint64_t sched_deadline;  // relative deadline (ns)
    uint64_t sched_period;    // period (ns)
};

// Example: 1kHz control loop that needs 200μs of CPU per period
struct sched_attr attr = {};
attr.size = sizeof(attr);
attr.sched_policy = SCHED_DEADLINE;
attr.sched_runtime  =  200'000;   // 200 μs of CPU time
attr.sched_deadline = 1'000'000;  // must finish within 1 ms
attr.sched_period   = 1'000'000;  // repeat every 1 ms

// The kernel will REJECT this if total declared utilization > ~95%
// (leaving 5% for non-RT work)
int ret = sched_setattr(0, &attr, 0);
if (ret == -1 && errno == EBUSY) {
    // Admission test FAILED — not enough CPU capacity
    // This is actually GREAT — you know at startup that deadlines can't be met
    // Instead of finding out at runtime with a crash
}
```

**Exercise 1 — Compare FIFO vs DEADLINE jitter:**
```cpp
// Run your 1kHz cyclic executive from Week 3 under both:
// 1. SCHED_FIFO priority 90
// 2. SCHED_DEADLINE (runtime=200μs, deadline=1ms, period=1ms)
//
// Under heavy load (stress -c $(nproc)):
// - FIFO: jitter depends on whether a higher-priority thread exists
// - DEADLINE: kernel guarantees your 200μs budget, jitter is bounded
//
// Measure: max latency, p99.9 latency, overruns
```

### Priority inversion — the Mars Pathfinder deep dive

```
The Mars Pathfinder bug (1997) — The most famous RT bug in history:

Three threads:
  [HIGH prio]   Telemetry task — reads bus, must not be delayed
  [MED prio]    Communications task — CPU-intensive data processing
  [LOW prio]    Meteorological task — holds a shared mutex

Timeline:
  1. LOW acquires mutex M
  2. HIGH wakes up, needs mutex M → blocks (waiting for LOW)
  3. MED wakes up, higher priority than LOW → preempts LOW
  4. MED runs for a long time (CPU-intensive)
  5. LOW can't run → can't release M → HIGH is still blocked
  6. HIGH misses its deadline → watchdog fires → system reboots

  HIGH is blocked by MED, even though HIGH > MED priority!
  Because MED prevents LOW from running to release the mutex.

Fix 1: Priority Inheritance (PI mutex)
  When HIGH blocks on mutex held by LOW:
  → LOW temporarily gets HIGH's priority
  → LOW preempts MED, finishes, releases mutex
  → HIGH runs. Nobody misses their deadline.

Fix 2: Priority Ceiling Protocol
  The mutex itself has a priority = highest priority of any thread that
  uses it. When LOW acquires the mutex, it immediately gets the ceiling
  priority. No other thread can preempt it while it holds the mutex.

The Pathfinder team patched it BY UPLOADING NEW CODE TO MARS.
They enabled VxWorks' priority inheritance option via a remote command.
```

**Exercise 2 — Reproduce and fix priority inversion:**
```cpp
// Create the Mars Pathfinder scenario:
// Thread LOW (prio 10):  lock(mutex M), sleep 100ms, unlock(mutex M)
// Thread MED (prio 50):  CPU spin for 200ms (no mutex)
// Thread HIGH (prio 90): try lock(mutex M), measure wait time

// Step 1: Use normal pthread mutex → observe HIGH waits >200ms
// Step 2: Use PI mutex:
pthread_mutexattr_t attr;
pthread_mutexattr_init(&attr);
pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT);
pthread_mutex_init(&mutex, &attr);
// → observe HIGH waits only ~100ms (LOW's critical section time)
```

### Context switch cost

```
Context switch = save all registers + switch page tables + flush TLB/pipeline

Measured costs (approximate, varies by hardware):
  x86-64 (modern):     1–3 μs (thread switch in same process)
  x86-64 process:      3–10 μs (different process, TLB flush)
  ARM Cortex-A72:      2–5 μs (thread switch)
  ARM Cortex-M4:       ~1 μs (ISR entry/exit, no MMU)

At 1 kHz with 3 task switches per period:
  3 × 3 μs = 9 μs overhead per period = 0.9% of 1ms budget
  Acceptable? Yes. But at 10 kHz → 9% overhead. Getting tight.
  At 100 kHz → 90% overhead. Can't use thread-per-task anymore!

This is why high-frequency control <100 μs period uses:
  - Single-thread cyclic executive (no context switches)
  - Or bare-metal ISR (Cortex-M: 71ns switch = 0.07% at 100 kHz)
```

**Exercise 3 — Measure YOUR system's context switch cost:**
```cpp
// Pipe benchmark: two threads, ping-pong a byte through a pipe
// Each round trip = 2 context switches

int pipefd[2];
pipe(pipefd);

auto start = std::chrono::steady_clock::now();
constexpr int N = 100000;

// Thread A:
for (int i = 0; i < N; ++i) {
    write(pipefd[1], "x", 1);
    char c;
    read(pipefd[0], &c, 1);
}

auto elapsed = std::chrono::steady_clock::now() - start;
double ns_per_switch = std::chrono::duration<double, std::nano>(elapsed).count()
                     / (N * 2);
printf("Context switch: %.0f ns\n", ns_per_switch);
// Expected: 1000–5000 ns on modern hardware
```

**Checkpoint:**
- [ ] Can explain CFS vs SCHED_FIFO vs SCHED_DEADLINE
- [ ] Can reproduce priority inversion and fix with PI mutex
- [ ] Can measure context switch cost on your system

---

## ⚡ Hardware Lesson 5: Memory Hierarchy — What Every Address Really Does
*Insert during Week 2 or 5*

**Why this matters:**
When you write `int x = array[i]`, the CPU doesn't just "read memory."
It triggers a cascade of hardware events across 4-5 levels of cache,
potentially stalls for 200 cycles, and might prefetch the wrong thing.

### The full picture of a memory access

```
Your code:  int x = data[i];

What actually happens:

1. CPU decodes the load instruction (1 cycle)

2. TLB lookup: virtual → physical address
   TLB hit:   ~1 cycle
   TLB miss:  page table walk → 10-100 cycles (4 levels of page tables!)
   Page fault: OS loads from disk → 10,000,000+ cycles (10ms!)

3. L1 cache lookup (32-64 KB, 4 cycle latency)
   Hit? → done, value in register
   Miss? → check L2

4. L2 cache lookup (256 KB - 1 MB, 12 cycle latency)
   Hit? → load to L1, value in register
   Miss? → check L3

5. L3 cache lookup (8-32 MB, shared among cores, 40 cycle latency)
   Hit? → load to L2 → L1 → register
   Miss? → go to RAM

6. RAM access (60-100 ns = ~200 cycles)
   Memory controller sends command to DRAM
   CAS latency: 12-20 ns (the DRAM row/column access)
   Total: 60-100 ns for the first byte
   Then: burst of 64 bytes (one cache line) at DDR bandwidth
   DDR4-3200: ~25 GB/s → 64 bytes = ~2.5 ns per cache line at full speed

7. Data arrives in L3 → L2 → L1 → register
   Your "simple read" took 200+ cycles if it was a cache miss!
```

### Cache line — the fundamental unit of data transfer

```
EVERYTHING moves in 64-byte cache lines (on almost all modern CPUs).

If you read array[0] (4 bytes), the hardware loads:
  array[0], array[1], array[2], ... array[15]  (64 bytes total)

This is why sequential access is fast:
  array[0] → cache miss (load 64 bytes)
  array[1] → HIT (already in cache line)
  array[2] → HIT
  ...
  array[15] → HIT
  array[16] → miss (new cache line)

And why random access is slow:
  array[random()] → miss (load 64 bytes, use 4)
  array[random()] → miss (different cache line)
  array[random()] → miss (different cache line)
  
  Utilization: 4/64 = 6.25% of loaded data actually used
```

**Exercise 1 — Measure cache line size and miss penalty:**
```cpp
// Classic benchmark: stride access pattern
// Access every Nth byte of a large array. Measure throughput.
// When N < 64: performance is flat (hitting same cache lines)
// When N >= 64: performance drops (every access is a new cache line)

constexpr size_t ARRAY_SIZE = 64 * 1024 * 1024;  // 64 MB
std::vector<char> array(ARRAY_SIZE, 0);

for (int stride : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
    auto start = std::chrono::steady_clock::now();
    volatile int sum = 0;
    for (size_t i = 0; i < ARRAY_SIZE; i += stride) {
        sum += array[i];
    }
    auto elapsed = /* ... */;
    double ns_per_access = elapsed_ns / (ARRAY_SIZE / stride);
    printf("Stride %4d: %.1f ns/access\n", stride, ns_per_access);
}
// You'll see: stride 1-32 → ~1ns, stride 64+ → ~5-60ns
// The jump at stride 64 reveals the cache line size
```

**Exercise 2 — False sharing:**
```cpp
// Two threads updating "independent" variables that share a cache line:
struct BadLayout {
    std::atomic<int> counter_a;  // offset 0
    std::atomic<int> counter_b;  // offset 4 — SAME cache line!
};

struct GoodLayout {
    alignas(64) std::atomic<int> counter_a;  // own cache line
    alignas(64) std::atomic<int> counter_b;  // own cache line
};

// Thread 1: increments counter_a 100M times
// Thread 2: increments counter_b 100M times
// With BadLayout: ~3 seconds (cache line bounces between cores)
// With GoodLayout: ~0.5 seconds (each core has its own cache line)
// SAME CODE, 6x performance difference, just from memory layout!
```

**Exercise 3 — TLB behavior with huge pages:**
```cpp
// Default page size: 4 KB → TLB covers ~2 MB (~512 entries × 4 KB)
// If your working set > 2 MB → TLB misses → page table walks → SLOW

// Huge pages: 2 MB → TLB covers ~1 GB (512 entries × 2 MB)
// For large data (sensor buffers, point clouds):

// In C++:
#include <sys/mman.h>
void* buf = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
// Falls back to normal pages if huge pages aren't available

// Enable huge pages on Linux:
// echo 512 > /proc/sys/vm/nr_hugepages  (allocate 512 × 2MB = 1GB)

// Exercise: Benchmark random access pattern on:
// 1. Normal 4KB pages
// 2. 2MB huge pages
// Expected: 10-30% speedup for random access patterns > 2MB working set
```

**💀 Hardware puzzle:**
```
You have two arrays, each 32 KB. You access them alternately:
  for (int i = 0; i < N; ++i) { sum += a[i] + b[i]; }

Performance is great.

Now you change b to be exactly 32 KB + 64 bytes offset from a.
Performance drops 50%. The code is IDENTICAL. Why?

Answer: L1 cache is 8-way set-associative, 32 KB.
When a and b are exactly 32 KB apart, a[i] and b[i] map to the
SAME cache set. With 8-way associativity, they keep evicting each
other. This is called "cache set thrashing" or "conflict miss."

Fix: Add padding to one array:
  alignas(64) char pad[64];  // shifts b to a different cache set
This is why seniors randomly add padding to structures.
```

**Checkpoint:**
- [ ] Can explain L1→L2→L3→RAM hierarchy with cycle counts
- [ ] Can measure cache line size experimentally
- [ ] Understand false sharing and can fix it with `alignas(64)`

---

# ═══════════════════════════════════════════════════════════
# SAFETY-CRITICAL DEEP DIVE — Beyond JPL Rules
# ═══════════════════════════════════════════════════════════
# Deeper treatment of certification, standards, and patterns
# used in aviation, automotive, medical, and space software.
# Marked with 🛡️ (safety shield).
# ═══════════════════════════════════════════════════════════

---

## 🛡️ Safety Lesson 1: DO-178C, ISO 26262, IEC 62304 — The Standards That Govern Life-Critical Code
*Insert during Week 4*

**The Big Picture:**
If your code can kill someone, it must meet a formal safety standard.
These standards don't tell you HOW to code — they tell you how to PROVE
your code is correct.

### The major standards

```
┌──────────────────────────────────────────────────────────────┐
│ Standard        │ Domain      │ Levels              │ Example │
├─────────────────┼─────────────┼─────────────────────┼─────────┤
│ DO-178C         │ Aviation    │ DAL A (catastrophic) │ Boeing  │
│                 │             │ DAL B (hazardous)    │ Airbus  │
│                 │             │ DAL C (major)        │         │
│                 │             │ DAL D (minor)        │         │
│                 │             │ DAL E (no effect)    │         │
│                 │             │                      │         │
│ ISO 26262       │ Automotive  │ ASIL-D (highest)    │ Tesla   │
│                 │             │ ASIL-C               │ Waymo   │
│                 │             │ ASIL-B               │         │
│                 │             │ ASIL-A (lowest)      │         │
│                 │             │                      │         │
│ IEC 62304       │ Medical     │ Class C (death)     │ Insulin │
│                 │             │ Class B (injury)     │  pumps  │
│                 │             │ Class A (no harm)    │         │
│                 │             │                      │         │
│ IEC 61508       │ Industrial  │ SIL-4 (highest)     │ Nuclear │
│ (generic)       │             │ SIL-1 (lowest)      │ Railway │
│                 │             │                      │         │
│ ECSS-E-ST-40C  │ Space (ESA) │ Criticality A-D     │ Ariane  │
│ NPR 7150.2     │ Space (NASA)│ Class A-D            │ Artemis │
└──────────────────────────────────────────────────────────────┘
```

### What DAL-A / ASIL-D actually requires

```
At the highest safety levels, you need ALL of these:

1. REQUIREMENTS TRACEABILITY
   Every line of code ← traces to a design requirement
         ← traces to a safety requirement
         ← traces to a hazard analysis
   "Why does this line exist?" must be answerable.

2. TEST COVERAGE (structural)
   DAL-A: MC/DC coverage (Modified Condition/Decision Coverage)
   Every condition in every decision must independently affect the outcome.
   
   if (a && b || c) { ... }
   
   You need tests where:
   - a=T→F changes the outcome (while b,c held constant)
   - b=T→F changes the outcome (while a,c held constant)
   - c=T→F changes the outcome (while a,b held constant)
   This is MUCH harder than just "100% branch coverage."
   
   Code coverage hierarchy:
   Statement coverage < Branch coverage < MC/DC coverage < Path coverage
   DAL-A requires MC/DC at minimum.

3. INDEPENDENT VERIFICATION
   The person who writes the code CANNOT verify it.
   A separate team reviews requirements, code, AND tests.

4. TOOL QUALIFICATION
   Your compiler must be qualified ("does GCC correctly compile this?")
   Your test framework must be qualified
   Even your requirements management tool must be qualified
   This is why DO-178C projects use certified compilers (e.g., CompCert)

5. FORMAL METHODS (increasingly used for DAL-A/ASIL-D)
   Mathematical proof that the code meets the specification
   Tools: SPARK/Ada, CompCert C, Frama-C, CBMC, TLA+
```

**Exercise:**
```cpp
// Write a function with MC/DC test cases:
enum class BrakeCommand { NONE, SOFT, HARD, EMERGENCY };

BrakeCommand compute_brake(bool obstacle_near, bool speed_over_limit,
                            bool driver_press_brake) {
    if (obstacle_near && speed_over_limit) {
        return BrakeCommand::EMERGENCY;
    } else if (obstacle_near || driver_press_brake) {
        return BrakeCommand::HARD;
    } else if (speed_over_limit) {
        return BrakeCommand::SOFT;
    }
    return BrakeCommand::NONE;
}

// Write the MC/DC test table:
// For the condition (obstacle_near && speed_over_limit):
//   Test 1: obstacle_near=T, speed_over_limit=T → EMERGENCY (baseline)
//   Test 2: obstacle_near=F, speed_over_limit=T → SOFT
//           (obstacle_near: T→F changed outcome: EMERGENCY→SOFT) ✓
//   Test 3: obstacle_near=T, speed_over_limit=F → HARD
//           (speed_over_limit: T→F changed outcome: EMERGENCY→HARD) ✓
//
// Continue for ALL conditions in ALL decisions...
// How many test cases do you need? (More than you think!)
```

**Checkpoint:**
- [ ] Can name the 4 major safety standards and their domains
- [ ] Can explain MC/DC coverage and write test cases for it
- [ ] Know what requirements traceability means in practice

---

## 🛡️ Safety Lesson 2: MISRA C++ & AUTOSAR C++14 — The Banned List
*Insert during Week 4*

**Why these exist:**
MISRA and AUTOSAR don't ban features because they're "bad" — they ban
features because they're UNPREDICTABLE or HARD TO VERIFY in the context
of safety-critical certification.

### Key bans and restrictions (MISRA C++ 2023 / AUTOSAR C++14)

```
═══════ BANNED completely ═══════

❌ Dynamic memory allocation after initialization
   Why: malloc/new can fail non-deterministically (fragmentation)
   Fix: Pool allocators, static arrays, PMR with fixed buffers

❌ Exceptions (throw, try, catch)
   Why: Non-deterministic stack unwinding, RTTI overhead
   Fix: Error codes, Expected<T,E>, status return values

❌ RTTI (dynamic_cast, typeid)
   Why: Hidden performance cost, vtable dependency
   Fix: Variant-based dispatch, CRTP, enum-based type tags

❌ Recursion
   Why: Unbounded stack usage, can't prove termination
   Fix: Iterative algorithms with explicit stack (std::stack)

❌ goto (except for error cleanup in C)
   Why: Spaghetti control flow, hard to verify
   Fix: RAII, early returns, structured loops

❌ C-style casts ((int)x)
   Why: Hides what kind of cast is happening
   Fix: static_cast, reinterpret_cast (explicit and grep-able)

❌ setjmp/longjmp
   Why: Bypasses destructors → resource leaks, UB
   Fix: Return error codes up the call stack

═══════ RESTRICTED (allowed with justification) ═══════

⚠️ Templates (limited template depth, no SFINAE tricks)
   Why: Hard to review, compilation errors are unreadable
   Rule: Max template instantiation depth of 10

⚠️ Operator overloading
   Why: Can make code hard to understand (what does + do for a Matrix?)
   Rule: Only for mathematical/domain types where behavior is obvious

⚠️ Multiple inheritance
   Why: Diamond problem, vtable complexity
   Rule: Only single implementation inheritance + multiple interface inheritance

⚠️ The preprocessor (#define, #ifdef)
   Rule: Only for include guards, platform detection, and conditional compilation
   No function-like macros, no token pasting

⚠️ Pointer arithmetic
   Rule: Only via std::span or array indexing. Raw pointer + int is banned.

⚠️ Union types
   Why: Type punning, no type safety
   Rule: Use std::variant instead, or tagged union with explicit discriminator
```

**Exercise:**
```cpp
// Rewrite this code to be MISRA-compliant:

// BEFORE (7 violations):
class ProximitySensor {
    double* readings;  // raw pointer
    int count;
public:
    ProximitySensor(int n) {
        readings = new double[n];  // dynamic alloc
        count = n;
    }
    ~ProximitySensor() { delete[] readings; }

    double getAvg() {
        double sum = 0;
        for (int i = 0; i <= count; i++)  // off by one
            sum += readings[i];
        return sum / (double)count;  // C-style cast
    }

    void process() {
        try {
            auto avg = getAvg();
            if (avg > 100.0) throw std::runtime_error("too far");
        } catch (...) {
            goto cleanup;  // goto!
        }
        cleanup:
        // ...
    }
};

// AFTER (MISRA-compliant): rewrite with
// - Fixed-size array, no heap allocation
// - No exceptions, no goto
// - static_cast, bounded loops
// - Assertions for preconditions
```

### The "undecidable subset" — why some features are banned at certain levels

```
Some MISRA rules exist because tools CAN'T verify the code:

Rule: "All non-null pointers shall be checked before dereference"
  Sounds easy, but with pointer aliasing? UNDECIDABLE.
  Static analysis tools over-report (false positives) or under-report.

Rule: "No recursion"
  Checking termination of arbitrary recursive functions is equivalent to
  the Halting Problem. IMPOSSIBLE in general.
  Banning recursion makes it POSSIBLE to statically bound stack usage.

Rule: "All loops must have a fixed upper bound"
  Loop termination is also undecidable in general.
  Fixed bounds make termination TRIVIALLY provable.

The standards don't ban features because engineers are stupid.
They ban features because VERIFICATION TOOLS can't handle them.
```

**Checkpoint:**
- [ ] Can list 7+ MISRA-banned features and explain WHY each is banned
- [ ] Can rewrite non-compliant code to comply
- [ ] Understand that bans serve verification tools, not just humans

---

## 🛡️ Safety Lesson 3: Fault Tolerance Patterns — When Hardware Fails
*Insert during Week 4 or 8*

**The Hard Truth:**
In safety-critical systems, you assume EVERYTHING will fail.
RAM will bit-flip. CPUs will hang. Sensors will lie. Actuators will stick.
Your code must detect it AND handle it.

### Single Event Upset (SEU) — cosmic rays flip bits

```
At sea level: ~1 bit flip per GB of RAM per month
At aircraft altitude (35,000 ft): ~100x more (cosmic ray flux)
In space (LEO): ~1000x more

Your OKS robot in a warehouse: SEU risk is LOW but non-zero.
The Toyota unintended acceleration: blamed on SEU in unprotected RAM.

Protection patterns:

1. ECC RAM (hardware)
   - Corrects 1 bit error, detects 2 bit errors per 64-bit word
   - Your Jetson Orin has ECC option — USE IT for production
   - STM32H7 has ECC on SRAM — enabled by default

2. Software redundancy (N-Modular Redundancy)
   - Store critical values 3 times, vote:
   
   uint32_t throttle_a, throttle_b, throttle_c;
   
   uint32_t safe_read_throttle() {
       if (throttle_a == throttle_b) return throttle_a;
       if (throttle_a == throttle_c) return throttle_a;
       if (throttle_b == throttle_c) return throttle_b;
       // All three disagree — FAULT! Enter safe mode.
       enter_safe_mode("throttle TMR failure");
       return 0;
   }
   
   // On every write:
   void safe_write_throttle(uint32_t val) {
       throttle_a = val;
       throttle_b = val;
       throttle_c = val;
   }

3. Checksum on critical data structures:
   struct CriticalState {
       double position[3];
       double velocity[3];
       double orientation[4];
       uint32_t checksum;  // CRC32 of the above
       
       void update_checksum() {
           checksum = crc32(this, offsetof(CriticalState, checksum));
       }
       bool verify_checksum() const {
           return checksum == crc32(this, offsetof(CriticalState, checksum));
       }
   };
   // Check on EVERY read. If corrupted → use backup copy or enter safe mode.

4. Control flow monitoring:
   // Verify the program executed the expected sequence of functions
   static uint32_t flow_counter = 0;
   
   void sensor_read() {
       flow_counter += 0x1111;  // expected: 0x1111 after this
       // ... actual work ...
   }
   void controller_compute() {
       if (flow_counter != 0x1111) enter_safe_mode("flow error");
       flow_counter += 0x2222;  // expected: 0x3333 after this
       // ... actual work ...
   }
   void actuator_write() {
       if (flow_counter != 0x3333) enter_safe_mode("flow error");
       flow_counter = 0;  // reset for next cycle
       // ... actual work ...
   }
   // A bit flip or wild jump that skips a function → detected!
```

**Exercise 1 — Implement Triple Modular Redundancy (TMR):**
```cpp
// A generic TMR wrapper for any trivially-copyable type
template<typename T>
class TMR {
    static_assert(std::is_trivially_copyable_v<T>);
    T a_, b_, c_;
public:
    void store(const T& val) { a_ = val; b_ = val; c_ = val; }
    
    std::optional<T> load() const {
        if (std::memcmp(&a_, &b_, sizeof(T)) == 0) return a_;
        if (std::memcmp(&a_, &c_, sizeof(T)) == 0) return a_;
        if (std::memcmp(&b_, &c_, sizeof(T)) == 0) return b_;
        return std::nullopt;  // all three disagree — FAULT
    }

    // Exercise: What happens if the pointer to this TMR object
    // itself is corrupted? (Meta-protection problem)
};
```

**Exercise 2 — Implement a safety-critical state machine:**
```cpp
// The key pattern: EXPLICIT safe state that the system falls into
// when anything goes wrong.

enum class SafetyState : uint8_t {
    INIT        = 0x00,
    OPERATIONAL = 0x5A,  // not sequential! bit patterns chosen for Hamming distance
    DEGRADED    = 0xA5,
    SAFE_STOP   = 0xC3,
    FAULT       = 0xFF,
};

// Why these specific values?
// INIT (0x00) → OPERATIONAL (0x5A): 4 bits differ
// OPERATIONAL (0x5A) → SAFE_STOP (0xC3): 6 bits differ
// A single bit flip CANNOT accidentally transition between valid states.
// This is "Hamming distance protection" for state variables.

// ANY unrecognized state value → enter FAULT
SafetyState validate_state(uint8_t raw) {
    switch (raw) {
        case 0x00: return SafetyState::INIT;
        case 0x5A: return SafetyState::OPERATIONAL;
        case 0xA5: return SafetyState::DEGRADED;
        case 0xC3: return SafetyState::SAFE_STOP;
        default:   return SafetyState::FAULT;  // bit flip → caught!
    }
}
```

**Exercise 3 — Stack overflow protection:**
```cpp
// On embedded, stack overflow = silent data corruption (no MMU to catch it)
// Pattern: stack painting + periodic check

// At startup, fill the stack with a known pattern:
constexpr uint32_t STACK_CANARY = 0xDEADBEEF;

void init_stack_canary(uint32_t* stack_bottom, size_t stack_words) {
    for (size_t i = 0; i < stack_words; ++i) {
        stack_bottom[i] = STACK_CANARY;
    }
}

// In the health check (10 Hz):
bool check_stack_overflow(const uint32_t* stack_bottom, size_t guard_words) {
    for (size_t i = 0; i < guard_words; ++i) {
        if (stack_bottom[i] != STACK_CANARY) {
            return true;  // OVERFLOW DETECTED!
        }
    }
    return false;
}

// On Linux: the kernel provides guard pages (PROT_NONE below the stack)
// Use: mmap(NULL, guard_size, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
// Place this BELOW your thread's stack. Overflow → SIGSEGV → caught!
```

**💀 Safety puzzle:**
```
Ariane 5 Flight 501 (1996):
  - 64-bit float converted to 16-bit signed integer
  - Value was 32,768+ (fit in 64-bit, not in 16-bit)
  - Integer overflow → diagnostic error → both IRUs (Inertial Reference Units) shut down
  - Both IRUs ran the SAME software → common-mode failure
  - Rocket self-destructed 37 seconds after launch
  - Cost: $370 million

The code was PROVEN CORRECT for Ariane 4.
Ariane 5 had a different trajectory → larger values → overflow.
The proof was correct — for the WRONG specification.

Lesson: Safety verification is only as good as the REQUIREMENTS.
If the spec says "lateral velocity ∈ [-32768, 32767]" but reality
exceeds it, the proof holds but the rocket explodes.
```

**🤖 ROS tie-in:**
```
Your OKS robot already does some of these:
- Heartbeat monitoring (Day 25 exercise)
- State machine (INIT → NOMINAL → ERROR)
- Watchdog timer

What it's probably NOT doing:
- TMR on critical state variables (position, velocity)
- CRC on state structs
- Control flow monitoring
- Hamming-distance-protected state enums
- Stack canary checking

Adding these to your base controller would catch the class of bugs
where sensorbar data corruption propagates to the estimator (#98301).
```

**Checkpoint:**
- [ ] Can implement TMR and explain when voting fails
- [ ] Can design a state machine with Hamming-distance-protected states
- [ ] Know the Ariane 5 failure and what lesson it teaches about specs

---

## 🛡️ Safety Lesson 4: Formal Methods Light — Static Proofs You Can Actually Use
*Insert during Week 7 or 8*

**Why this matters:**
Full formal verification (TLA+, SPARK/Ada) is expensive — often 10x the
development cost. But there's a middle ground: "lightweight formal methods"
that give you proof-like guarantees with practical effort.

### Level 1: assert + static_assert (you already know this)

```cpp
// Pre/postconditions that act as lightweight specifications
double safe_sqrt(double x) {
    assert(x >= 0.0 && "precondition: non-negative");
    assert(std::isfinite(x) && "precondition: finite");
    double result = std::sqrt(x);
    assert(std::isfinite(result) && "postcondition: finite");
    assert(result >= 0.0 && "postcondition: non-negative");
    return result;
}
```

### Level 2: Type-based proofs (making illegal states unrepresentable)

```cpp
// Instead of runtime checks, make illegal values IMPOSSIBLE TO CONSTRUCT

// BAD: runtime check
void set_speed(double speed_mps) {
    assert(speed_mps >= 0.0 && speed_mps <= 2.0);  // discoverable only at runtime
    actual_speed_ = speed_mps;
}

// GOOD: type-based proof
template<int MinCenti, int MaxCenti>
class BoundedFloat {
    float value_;
    explicit constexpr BoundedFloat(float v) : value_(v) {}
public:
    static constexpr std::optional<BoundedFloat> make(float v) {
        if (v < MinCenti / 100.0f || v > MaxCenti / 100.0f) return std::nullopt;
        return BoundedFloat(v);
    }
    constexpr float get() const { return value_; }

    // Arithmetic that preserves bounds:
    constexpr BoundedFloat<MinCenti, MaxCenti> 
    operator+(BoundedFloat<0, 0> zero) const { return *this; }
    // Adding zero is safe. Adding arbitrary values? Not proven safe.
};

using Speed = BoundedFloat<0, 200>;  // 0.00 to 2.00 m/s
// Speed::make(5.0) returns std::nullopt — CAN'T exceed limit!
// The TYPE proves the value is in range. No runtime check needed after construction.
```

### Level 3: CBMC — Bounded Model Checking for C/C++

```bash
# CBMC can MATHEMATICALLY PROVE your function has no:
# - Buffer overflows
# - Null pointer dereferences  
# - Arithmetic overflows
# - Assertion failures
# For ALL possible inputs (up to a bound).

# Install: apt install cbmc
# Usage:
cbmc --function safe_sqrt --bounds-check --pointer-check \
     --signed-overflow-check my_code.cpp

# CBMC explores ALL execution paths (symbolic execution).
# If it says "VERIFICATION SUCCESSFUL" → no bugs exist in that function.
# If it finds a bug → gives you a concrete counter-example (input that triggers it).

# Limitation: bounded depth (can't handle arbitrary recursion or huge loops)
# But for bounded functions (which JPL Rule 2 enforces!) → COMPLETE proofs.
```

**Exercise:**
```cpp
// 1. Install CBMC: sudo apt install cbmc
// 2. Write a bounded array access function:
int safe_access(int* arr, int size, int index) {
    assert(arr != nullptr);
    assert(size > 0 && size <= 1024);
    assert(index >= 0 && index < size);
    return arr[index];
}
// 3. Run: cbmc --function safe_access --bounds-check --pointer-check code.cpp
// 4. Observe: VERIFICATION SUCCESSFUL
// 5. Remove an assert → observe CBMC finds the vulnerability
```

### Level 4: Frama-C + ACSL (for C code in safety-critical contexts)

```c
// ACSL annotations let you write formal specifications as comments:
// The tool PROVES they hold for ALL inputs.

/*@ requires \valid(a + (0 .. n-1));
    requires n > 0 && n <= 1000;
    assigns \nothing;
    ensures \result >= a[0];
    ensures \forall integer i; 0 <= i < n ==> \result >= a[i];
    ensures \exists integer i; 0 <= i < n && \result == a[i];
*/
int array_max(const int* a, int n) {
    int max = a[0];
    /*@ loop invariant 1 <= i <= n;
        loop invariant max >= a[0];
        loop invariant \forall integer j; 0 <= j < i ==> max >= a[j];
        loop invariant \exists integer j; 0 <= j < i && max == a[j];
        loop assigns i, max;
        loop variant n - i;
    */
    for (int i = 1; i < n; i++) {
        if (a[i] > max) max = a[i];
    }
    return max;
}

// Run: frama-c -wp -wp-rte array_max.c
// Output: All verification conditions PROVED.
// This is a MATHEMATICAL PROOF that the function:
// - Never accesses out of bounds
// - Always returns the maximum element
// - Always returns an element that exists in the array
// No test suite, no matter how large, gives this guarantee.
```

**💀 Safety puzzle:**
```
CompCert C compiler (used in Airbus A380 flight control):
  - Mathematically PROVEN to preserve program semantics
  - If your source code is correct → the compiled binary is correct
  - Cost: millions of euros of research, 100,000+ lines of Coq proofs

GCC/Clang:
  - Extensively tested but NOT proven correct
  - Has had bugs where optimization CHANGED program behavior
  - Example: a null check was "optimized away" because the compiler
    assumed pointers were never null after dereference

Question: Your robot runs on GCC. Should you worry?
Answer: Yes, but:
  - GCC bugs are RARE in practice
  - Compile with -O1 or -O2 (not -O3 — more aggressive = more risk)
  - Always test the ACTUAL optimized binary, not the debug build
  - Use multiple compilers (GCC + Clang) — if both agree, more confident
  - For the really critical 100 lines: review the assembly (Godbolt)
```

**Checkpoint:**
- [ ] Can explain the 4 levels: assertions → types → CBMC → Frama-C
- [ ] Can run CBMC on a simple function and interpret "VERIFICATION SUCCESSFUL"
- [ ] Understand the tradeoff: formal methods = more confidence, more cost

---

## 🛡️ Safety Lesson 5: Real Incident Analyses — What Actually Went Wrong
*Standalone — read anytime for perspective*

These are REAL software failures that killed people or destroyed equipment.
Study them. Internalize them. They're the reason these rules exist.

### 1. Therac-25 Radiation Machine (1985-1987) — 6 deaths

```
Root cause: Race condition between operator input and radiation mode.
  - Operator typed FAST → keyboard input overran a buffer
  - Mode flag was set to "electron" but hardware was in "X-ray" mode
  - Machine delivered 100x intended radiation dose

Software failures:
  - No hardware interlocks (software was the ONLY safety check)
  - No independent verification of mode vs hardware state
  - Counter overflow: a flag was incremented, not set
    When it overflowed to 0, safety check was "passed"
  - No error reporting to operator (machine said "MALFUNCTION 54"
    → operator pressed P to proceed, not knowing it was fatal)

Lessons for you:
  - NEVER rely on software alone for safety (hardware interlocks!)
  - Use SET/CLEAR, not increment/decrement for boolean flags
  - Make error messages meaningful ("RADIATION OVERDOSE DETECTED"
    not "MALFUNCTION 54")
  - Validate hardware state independently of software state
```

### 2. Boeing 737 MAX MCAS (2018-2019) — 346 deaths

```
Root cause: Single sensor dependency + no pilot override.
  - MCAS (Maneuvering Characteristics Augmentation System) used
    ONE angle-of-attack sensor to determine nose-up condition
  - Faulty sensor → MCAS pushed nose down repeatedly
  - Pilots couldn't override because they didn't know MCAS existed
  - Second sensor was a PAID OPTION (!!)

Software & systems failures:
  - Single point of failure: one sensor → no redundancy
  - No "disagree" alert unless optional sensor was purchased
  - MCAS authority: could command 2.5° of stabilizer per activation
    (equivalent to full pilot authority)
  - No limit on total MCAS-commanded stabilizer movement
  - Pilot training didn't mention MCAS at all

Lessons:
  - NEVER make a safety sensor optional
  - Redundancy is not a feature. It's a REQUIREMENT.
  - Authority limits: no subsystem should have unlimited control
  - If the system can override the human, the human must know about it
  - Cross-check: if two sensors disagree, ALERT and REDUCE authority
```

### 3. Toyota Unintended Acceleration (2009-2010) — 89 deaths

```
NASA/NHTSA findings in the Toyota firmware:
  - 243 violations of MISRA-C coding standards
  - 7,134 global variables (should be near zero in safety-critical code)
  - Recursion depth up to 23 levels (JPL Rule 1 violation)
  - Stack overflow: 94% of stack used in normal operation (6% margin!)
  - No RTOS memory protection between tasks
  - Throttle control variable lived in unprotected RAM (SEU vulnerable)
  - Watchdog could be "petted" from a low-priority background task
    even if the main control task had crashed

The "kitchen sink" problem:
  - 280,000 lines of code in a single embedded system
  - No separation between critical (throttle) and non-critical (infotainment)
  - A bug in the infotainment code could corrupt throttle variables

Lessons:
  - Minimize global variables (each one is a coupling point)
  - Bound stack usage (Rule 1 + 4: no recursion + small functions)
  - Protect critical variables (TMR, CRC, separate memory region)
  - Watchdog must verify the ACTUAL safety function, not a dummy task
  - Separate critical and non-critical code (different processes, MMU protection)
```

### 4. Knight Capital Trading Glitch (2012) — $440M lost in 45 minutes

```
Not life-critical but instructive:
  - Old code had a feature flag "POWER_PEEL" intended for testing
  - Deployment to 8 servers: 7 updated, 1 missed (manual deployment!)
  - Flag was repurposed to enable new feature
  - Old server interpreted the flag differently → ran test logic in production
  - 4 million trades in 45 minutes, all losing money
  - Company went bankrupt within a week

Lessons (applicable to your OKS robot deployments):
  - AUTOMATE deployment (no manual "copy file to 8 servers")
  - Dead code removal: delete old feature flag code, don't repurpose
  - Canary deployment: roll out to 1 server, verify, then the rest
  - Kill switch: ability to halt the system within seconds
  - Integration test that verifies ALL servers run the SAME version
```

### Pattern summary: the 5 universal failure modes

```
Every safety incident in history falls into one or more of these:

1. SINGLE POINT OF FAILURE
   One sensor, one computer, one check → fails silently
   Fix: Redundancy + cross-checking + independent monitoring

2. UNCHECKED ASSUMPTIONS
   "This value can never exceed X" → it does
   Fix: Defensive checks + requirements validation + boundary testing

3. INSUFFICIENT SEPARATION
   Safety-critical and non-critical code share memory/resources
   Fix: Process isolation, memory protection, privilege separation

4. HUMAN FACTORS IGNORED
   Operator can't understand the error, can't override the system
   Fix: Clear error messages, manual override, pilot awareness

5. INADEQUATE TESTING OF THE DEPLOYED ARTIFACT
   "Tested OK" on a different binary/config/version than production
   Fix: Test the actual binary, automated deployment, version verification
```

**Checkpoint:**
- [ ] Can explain at least 3 real safety incidents and their root causes
- [ ] Can map each incident to one of the 5 universal failure modes
- [ ] Understand why coding standards (MISRA, Power of 10) exist — they're written in blood

---

## 🔄 Module 11: C++20 Coroutines
*Standalone — requires GCC 10+ or Clang 14+*
*Goal: Understand stackless coroutines for lazy pipelines and async patterns*

### Topics Covered
- Coroutine machinery: `co_await`, `co_yield`, `co_return`
- Promise type and `coroutine_handle`
- Awaitable/Awaiter concepts (`await_ready`, `await_suspend`, `await_resume`)
- Generator pattern — lazy sequences with iterator interface
- Async Task pattern — single-value futures with continuation
- Cooperative scheduler (round-robin, priority)
- Symmetric transfer for O(1) stack usage
- Coroutine lifetime pitfalls (dangling references across suspension points)

### Exercises
| File | What |
|------|------|
| `ex01_generator.cpp` | Build a `Generator<T>` from scratch — fibonacci, file lines, transforms |
| `ex02_async_task.cpp` | Build a `Task<T>` with continuation chaining |
| `ex03_coroutine_scheduler.cpp` | Cooperative round-robin + priority scheduler |
| `ex04_symmetric_transfer.cpp` | Demonstrate and benchmark symmetric transfer |

### Puzzles
| File | What |
|------|------|
| `puzzle01_lifetime_trap.cpp` | 💀 Dangling reference across `co_await` — find and fix |
| `puzzle02_generator_leak.cpp` | 💀 Resource leak on early `break` from generator loop |

---

## 📐 Module 12: C++20 Ranges
*Standalone — requires GCC 10+ (partial) or GCC 12+ (full)*
*Goal: Replace iterator-pair algorithms with composable, lazy view pipelines*

### Topics Covered
- Range concepts: `range`, `view`, `input_range`, `sized_range`, `contiguous_range`
- Views vs containers (lazy vs eager, O(1) construction)
- Pipe syntax `|` for composition
- Range adaptors: `filter`, `transform`, `take`, `drop`, `reverse`, `split`, `join`
- Range algorithms with projections — the killer feature
- Writing custom views with `view_interface`
- Sentinel vs end iterator
- Dangling views and borrowed ranges

### Exercises
| File | What |
|------|------|
| `ex01_range_basics.cpp` | Filter/transform/take pipeline, laziness proof |
| `ex02_projections.cpp` | Sort, find, max with projections — replaces verbose lambdas |
| `ex03_custom_view.cpp` | Build a `stride_view` with iterator + pipe adaptor |
| `ex04_range_algorithms.cpp` | `ranges::sort` vs `std::sort`, partition, rotate, unique |
| `ex05_lazy_pipeline.cpp` | Infinite sensor stream, lazy vs eager benchmark |

### Puzzles
| File | What |
|------|------|
| `puzzle01_dangling_view.cpp` | 💀 View over temporary — predict `ranges::dangling` |
| `puzzle02_single_pass_trap.cpp` | 💀 `istream_view` consumed once — explain and fix |

---

## 📝 Module 13: C++20 `std::format`
*Standalone — requires GCC 13+ or Clang 17+ (use `{fmt}` as polyfill for older)*
*Goal: Type-safe, fast, compile-time-checked string formatting*

### Topics Covered
- Format specification mini-language: fill, align, sign, width, precision, type
- `std::format` vs `std::format_to` vs `std::format_to_n` vs `std::formatted_size`
- Custom formatters via `std::formatter<T>` specialization
- Compile-time format string validation
- Performance vs printf vs iostream
- `{fmt}` library as polyfill

### Exercises
| File | What |
|------|------|
| `ex01_format_basics.cpp` | Format integers, floats, hex, binary, alignment |
| `ex02_custom_formatter.cpp` | Formatters for `Vector3d`, `Timestamp`, `HexDump` |
| `ex03_format_benchmark.cpp` | printf vs ostringstream vs std::format — 100K iterations |
| `ex04_safe_logging.cpp` | Thread-safe logger with compile-time validation |

### Puzzles
| File | What |
|------|------|
| `puzzle01_format_spec_quiz.cpp` | 💀 Predict the output of tricky format specs |
| `puzzle02_formatter_sfinae.cpp` | 💀 `Formattable` concept — better error messages |

---

## 🛟 Module 14: Exception Safety Guarantees
*Standalone — GCC 9+ compatible*
*Goal: Design APIs that never leak, never corrupt, always roll back on failure*

### Topics Covered
- The 4 guarantee levels: no-throw, strong, basic, no guarantee
- `noexcept` specifier and operator — impact on move operations
- Copy-and-swap idiom for strong guarantee
- RAII as the foundation of exception safety
- Scope guards using `std::uncaught_exceptions()`
- `std::move_if_noexcept` — why `noexcept` on moves matters for containers
- Exception safety in constructors and multi-step operations
- Commit-or-rollback pattern

### Exercises
| File | What |
|------|------|
| `ex01_guarantee_levels.cpp` | `BankAccount::transfer()` — all 4 levels demonstrated |
| `ex02_copy_and_swap.cpp` | `DynamicArray<T>` — naive (broken) vs copy-and-swap |
| `ex03_scope_guard.cpp` | `ScopeGuard`, `ScopeSuccess`, `ScopeFailure` with rollback |
| `ex04_noexcept_impact.cpp` | Benchmark: 100K push_backs with/without noexcept move |
| `ex05_exception_safe_container.cpp` | `SmallVector<T,N>` with SBO and strong guarantee |

### Puzzles
| File | What |
|------|------|
| `puzzle01_destructor_throw.cpp` | 💀 Throwing in destructor during unwinding → `std::terminate` |
| `puzzle02_move_or_copy.cpp` | 💀 Predict: does vector move or copy for each Widget variant? |

---

## 🧱 Module 15: PMR (Polymorphic Memory Resource) Allocators
*Standalone — GCC 9+ compatible*
*Goal: Eliminate heap allocation from real-time hot paths*

### Topics Covered
- Why `new`/`malloc` break real-time guarantees (locks, page faults, fragmentation)
- `std::pmr::memory_resource` — polymorphic, stateful, runtime-switchable
- 3 standard resources: `monotonic_buffer_resource`, `unsynchronized_pool_resource`, `synchronized_pool_resource`
- Null resource (testing), new-delete resource (default), upstream chaining
- PMR containers: `pmr::vector`, `pmr::string` — allocator propagation
- Writing custom `memory_resource` (TrackingResource, StackResource, BumpAllocator)
- Real-time arena pattern: pre-allocate at startup, use bump allocator in 1kHz loop
- The `pmr::vector<std::string>` vs `pmr::vector<pmr::string>` propagation trap

### Exercises
| File | What |
|------|------|
| `ex01_monotonic_buffer.cpp` | Stack-backed bump allocator, benchmark vs heap |
| `ex02_pool_resource.cpp` | Pool with size binning, memory reuse, multi-threaded |
| `ex03_custom_memory_resource.cpp` | TrackingResource + StackResource + BumpAllocator |
| `ex04_realtime_arena.cpp` | 1kHz control loop with zero-allocation hot path |
| `ex05_pmr_containers.cpp` | Propagation verification, the non-pmr inner container bug |

### Puzzles
| File | What |
|------|------|
| `puzzle01_arena_lifetime.cpp` | 💀 Dangling pointers after arena destruction |
| `puzzle02_propagation_trap.cpp` | 💀 Compile-time detection of non-pmr inner containers |

---

## 🔬 Module 16: Sanitizer Workshop
*Standalone — GCC 9+ compatible*
*Goal: Master ASan, UBSan, TSan for hunting real bugs*

### Topics Covered
- AddressSanitizer: buffer overflow, use-after-free, double-free, leak detection
- UndefinedBehaviorSanitizer: signed overflow, null deref, alignment, shift
- ThreadSanitizer: data races, lock-order-inversion, signal-unsafe code
- Compiler flags and runtime options (`ASAN_OPTIONS`, `TSAN_OPTIONS`)
- Cannot combine ASan + TSan — run separately
- Performance overhead: ASan ~2x, TSan ~5-15x, UBSan ~minimal
- CI integration: separate sanitizer jobs
- Sanitizer blind spots: logic errors, ABA, uninitialized reads

### Exercises
| File | What |
|------|------|
| `ex01_asan_bugs.cpp` | 8 ASan bugs — heap/stack overflow, UAF, double-free, leak |
| `ex02_ubsan_bugs.cpp` | 8 UBSan bugs — overflow, shift, null, alignment, float-cast |
| `ex03_tsan_bugs.cpp` | 5 TSan bugs — races, bitfield race, ABBA deadlock |
| `ex04_sanitizer_ci.cpp` | Realistic `MessageQueue<T>` with 3 hidden bugs + fixed version |
| `ex05_sanitizer_detective.cpp` | `RobotController` with 3 subtle bugs — pick the right sanitizer |

### Puzzles
| File | What |
|------|------|
| `puzzle01_false_positive.cpp` | 💀 SeqLock false positive — write a suppression |
| `puzzle02_sanitizer_blind_spots.cpp` | 💀 What sanitizers CAN'T catch — alternative strategies |

---

## Progression Summary

| Week | You Can Now... |
|------|---------------|
| 1 | Write zero-copy, type-safe, compile-time-evaluated C++ |
| 2 | Handle errors without exceptions, understand the memory model |
| 3 | Build a deterministic 1kHz RT loop on Linux |
| 4 | Write code that passes NASA/JPL code review |
| 5 | Test, fuzz, profile, and CI/CD C++ projects |
| 6 | Build IPC systems with shared memory + serialization |
| 7 | Catch UB that 99% of developers miss |
| 8 | Ship a flight-software-grade application |
| 🧓 | Think like a 20-year veteran — know the tradeoffs, not just the syntax |
| ⚡ | Understand what the hardware is doing under your abstractions |
| 🛡️ | Design for failure — because hardware WILL fail, code WILL have bugs |
| 🔄 | Write lazy async pipelines with C++20 coroutines |
| 📐 | Compose data transformations with zero-overhead ranges |
| 📝 | Format output safely without printf or iostream pain |
| 🛟 | Design exception-safe APIs with strong guarantee |
| 🧱 | Eliminate heap allocation from hot paths with PMR arenas |
| 🔬 | Hunt memory, UB, and race bugs with sanitizer instrumentation |

---

## Daily Habit Checklist

```
□ Read for 20 minutes (notes, reference, blog post)
□ Write ≥ 50 lines of code
□ Solve at least 1 puzzle / tricky question
□ Compile with -Wall -Wextra -Werror (zero warnings)
□ Run with at least 1 sanitizer
□ Commit your work
```

## Weekly Review Questions

Every Sunday, answer these without looking:
1. What was the hardest concept this week?
2. What mistake will I NOT make again?
3. What would I explain differently to someone else?
4. What's still fuzzy and needs more practice?
