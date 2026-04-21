# Week 2: Error Handling Without Exceptions & The C++ Memory Model

---

## Part A — Why Exceptions Are Banned in RT/Safety-Critical Code

If you've written C++ in application-land, `try/catch` feels natural. In real-time and safety-critical systems (automotive, robotics firmware, avionics), exceptions are almost universally banned. Here's why — and what you use instead.

### 1. Stack Unwinding Is Non-Deterministic

When an exception is thrown, the runtime must:
1. Walk the call stack frame-by-frame
2. Find a matching `catch` handler (potentially many frames up)
3. Invoke destructors for every automatic object in each unwound frame
4. De-allocate stack frames

The cost depends on **how deep the stack is at throw-time** — something you cannot bound at compile time.

```
throw site                        catch site
   |                                 |
   v                                 v
 [ frame 7 ] --dtor--> [ frame 6 ] --dtor--> ... --dtor--> [ frame 1 ]
              ~Widget()              ~Buffer()               ~Lock()
```

In a hard-RT loop running at 1 kHz, you have a 1 ms budget. Stack unwinding through 5+ frames with destructors can easily blow that. Worse, the cost **varies per invocation** — it's data-dependent, not just code-dependent.

### 2. RTTI Overhead

Exception matching uses Run-Time Type Information. The compiler emits type metadata for every class hierarchy involved in `catch` clauses. This:
- Bloats binary size (matters on MCUs with 256 KB flash)
- Adds indirection through type-info tables at catch time
- Is incompatible with `-fno-rtti` (which many embedded toolchains require)

### 3. MISRA C++ and AUTOSAR Prohibition

MISRA C++:2023 Rule 18.3.1: *"An exception shall not be thrown."* AUTOSAR AP R20-11 similarly prohibits exceptions in adaptive-platform code. These aren't style preferences — they're certification requirements. If your code needs ISO 26262 ASIL-D or DO-178C Level A, exceptions are a non-starter.

### 4. Hidden Control Flow

Exceptions create invisible `goto` paths. Any function call can jump to an arbitrary catch handler. In safety-critical code, **every control path must be analysed** — exceptions make that combinatorially harder.

---

## Part B — Error Handling Alternatives

### Error Codes (C-style)

The simplest pattern. Every function returns a status:

```cpp
enum class ErrCode { Ok, Timeout, BadParam, HwFault };

ErrCode read_sensor(float* out) {
    if (!hw_ready()) return ErrCode::HwFault;
    *out = hw_read();
    return ErrCode::Ok;
}
```

**Pros:** Zero overhead, fully deterministic, trivially analysable.
**Cons:** Caller can silently ignore the return. The "real" return must go through an out-parameter. Composing multiple fallible calls is verbose.

### std::optional — "Value or Nothing"

```cpp
std::optional<float> read_sensor() {
    if (!hw_ready()) return std::nullopt;
    return hw_read();
}
```

Good when the **only failure mode is "absent."** No room to say *why* it failed.

### std::variant as Sum Type — "Value or Error"

```cpp
using SensorResult = std::variant<float, ErrCode>;

SensorResult read_sensor() {
    if (!hw_ready()) return ErrCode::HwFault;
    return hw_read();
}
```

Now we can carry **which** error. But `std::visit` is clunky.

### Expected<T, E> — Railway-Oriented Programming

C++23 gives us `std::expected<T, E>`. Before that, use `tl::expected` or roll your own:

```cpp
std::expected<float, ErrCode> read_sensor() {
    if (!hw_ready()) return std::unexpected(ErrCode::HwFault);
    return hw_read();
}
```

The killer feature is `and_then` / `transform` chaining (railway-oriented programming):

```
 read_sensor() --[ok]--> calibrate() --[ok]--> filter() --[ok]--> publish()
       |                     |                    |
       +---[err]-------------+------[err]---------+-------> handle error once
```

Each step only runs if the previous succeeded. Errors skip forward like a railway switch. **One error-handling site at the end**, not interleaved checks:

```cpp
auto result = read_sensor()
    .and_then(calibrate)
    .and_then(filter)
    .transform(to_message);

if (!result) log_error(result.error());
```

This is the dominant pattern in modern safety-critical C++: deterministic, zero-overhead, composable, and every error path is explicit.

---

## Part C — The C++ Memory Model from First Principles

### What Is a "Memory Location"?

The C++ standard (§6.7.1) defines a memory location as:
- A scalar object (int, float, pointer), OR
- A maximal sequence of adjacent bit-fields of non-zero width

Two threads accessing **the same memory location** where at least one is a write, and they're not ordered by synchronization → **data race → undefined behavior.**

This is the entire foundation. Everything below exists to establish ordering.

### Sequenced-Before (Single Thread)

Within one thread, statements execute in order:

```
Thread 1:
  a = 1;          // A
  b = 2;          // B
  
  A sequenced-before B  (always)
```

### Happens-Before (Cross Thread)

The hard part. Two operations in different threads have **no inherent ordering** unless you create one:

```
Thread 1              Thread 2
---------             ---------
x = 42;               ???
flag = true;           if (flag)
                         use(x);   // is x guaranteed to be 42?
```

With plain (non-atomic) variables: **NO.** This is a data race. The compiler and CPU can both reorder.

### Why Reordering Happens: Store Buffers

Modern CPUs don't write directly to cache. They write to a **store buffer** first:

```
  CPU Core 0                        CPU Core 1
 +-----------+                     +-----------+
 | registers |                     | registers |
 +-----+-----+                     +-----+-----+
       |                                 |
 +-----v-----+                     +-----v-----+
 | STORE BUF |                     | STORE BUF |
 | x=42      |  (not yet visible)  |           |
 | flag=true  |                     |           |
 +-----+-----+                     +-----+-----+
       |                                 |
 +=====v=================================v=====+
 |              SHARED CACHE (L3)              |
 |  x = 0 (stale!)    flag = false (stale!)   |
 +==============================================+
```

Core 0 wrote `x=42` then `flag=true` into its store buffer. Core 1 reads from shared cache and might see `flag=true` (drained) but `x=0` (not yet drained). **The stores can become visible out of order.**

### MESI Cache Coherence Protocol

Caches maintain coherence via MESI states for each cache line:

```
 +----------+----------------------------------------------------+
 | State    | Meaning                                            |
 +----------+----------------------------------------------------+
 | Modified | I have the only copy, it's dirty (newer than RAM)  |
 | Exclusive| I have the only copy, it's clean                   |
 | Shared   | Multiple caches have clean copies                  |
 | Invalid  | My copy is stale, must re-fetch                    |
 +----------+----------------------------------------------------+

  Core 0 writes x:
    Core 0: x line -> Modified
    Core 1: x line -> Invalid  (snooped invalidation)
    
  Core 1 reads x:
    Core 1 sees Invalid -> requests from Core 0
    Core 0 flushes Modified line -> both go to Shared
```

MESI guarantees **eventual** coherence per cache line. But "eventual" isn't ordered. The store buffer sits *between* the core and the cache — that's where reordering sneaks in.

### Happens-Before Graph

```
  Thread 1                  Thread 2
  --------                  --------
  x.store(42, release)  
         \                
          \  synchronizes-with
           \              
            +---> flag.load(acquire)
                  if true:
                    x.load(relaxed) == 42  ✓ guaranteed
```

The `release` store **publishes** everything before it. The `acquire` load **subscribes** to everything the release published. Together they form a **happens-before** edge.

---

## Part D — Acquire-Release Semantics

### What the CPU Actually Does

`memory_order_release` on a store = **"drain my store buffer before (or as part of) making this store visible."**

`memory_order_acquire` on a load = **"don't let any of my subsequent reads/writes execute before this load completes."**

On x86, stores already have release semantics (Total Store Order). On ARM/RISC-V, the compiler emits actual fence instructions (`dmb ish` on ARM).

### Store Buffer Drain Diagram

```
 BEFORE release fence:
 +-------------+
 | Store Buffer|     Cache
 | x = 42      |    [stale]
 | y = 7       |    [stale]
 +-------------+

 release store of flag=true triggers drain:
 
 +-------------+
 | Store Buffer|     Cache
 | (empty)     |    x = 42  ✓
 |             |    y = 7   ✓
 |             |    flag=true ✓
 +-------------+

 Other cores doing acquire-load of flag will now
 also see x=42 and y=7  (happens-before guarantee)
```

### The Canonical Pattern

```cpp
std::atomic<bool> ready{false};
int data = 0;                      // non-atomic!

// Producer (Thread 1)
data = 42;                         // ordinary write
ready.store(true, std::memory_order_release);  // publish

// Consumer (Thread 2)
while (!ready.load(std::memory_order_acquire))  // subscribe
    ;  // spin
assert(data == 42);  // GUARANTEED — happens-before established
```

This is safe **without making `data` atomic**. The acquire-release pair on `ready` orders everything before the release against everything after the acquire.

---

## Part E — Compare-And-Swap (CAS)

### The CAS Loop Pattern

CAS is the fundamental lock-free building block. It atomically does:
> "If the value at address X is what I expect, replace it with my new value. Otherwise, tell me what it actually is."

```cpp
std::atomic<int> counter{0};

void increment() {
    int expected = counter.load(std::memory_order_relaxed);
    while (!counter.compare_exchange_weak(
               expected,              // read & updated on failure
               expected + 1,          // desired
               std::memory_order_acq_rel)) {
        // expected was updated to the current value — retry
    }
}
```

**Why `weak`?** On ARM/RISC-V, CAS is implemented as LL/SC (load-linked/store-conditional). `weak` allows spurious failure (the LL/SC window expired), which is fine inside a loop. `strong` adds a retry internally — wasteful in a CAS loop.

### The ABA Problem

```
 Thread 1:                    Thread 2:
 read A (value = ptr_X)       
                              pop X
                              push Y
                              push X back  (same address!)
 CAS(expected=ptr_X,          
     desired=ptr_Z)           
 => SUCCEEDS!  (X is back)   
 => But the stack changed underneath!
```

CAS only compares **the value**, not the history. If a value goes A→B→A, CAS thinks nothing changed. Solutions:
- **Tagged pointers:** Pack a version counter into the pointer (use upper bits on 64-bit)
- **Hazard pointers:** Protect nodes from reclamation while in use
- **Epoch-based reclamation:** Defer frees until safe

---

## Part F — Lock-Free SPSC Queue

### Ring Buffer Layout

A Single-Producer Single-Consumer queue is the simplest useful lock-free structure. One thread writes, one reads — no CAS needed, just acquire-release on the indices.

```
  Buffer (capacity = 8, only 7 usable to distinguish full from empty):

  Index:   0   1   2   3   4   5   6   7
         +---+---+---+---+---+---+---+---+
  Data:  |   | D | E | F |   |   |   |   |
         +---+---+---+---+---+---+---+---+
               ^           ^
               |           |
             head=1      tail=4
             (consumer)  (producer)

  Readable slots: [head, tail)  = {1, 2, 3} → D, E, F
  Writable slots: [tail, head) with wrap = {4, 5, 6, 7, 0}
  
  Full when:  (tail + 1) % cap == head
  Empty when: tail == head
```

### Why It Works Without CAS

- **Producer only writes `tail`** (after writing data into `buf[tail]`)
- **Consumer only writes `head`** (after reading data from `buf[head]`)
- Each index is **written by exactly one thread** → no contention

The ordering contract:

```
 Producer:                             Consumer:
 ---------                             ---------
 buf[tail] = item;     // ordinary     auto item = buf[head]; // ordinary
 tail.store(                           head.store(
   next_tail,                            next_head,
   memory_order_release);  // publish    memory_order_release);  // publish
                                       
 // reads head with acquire            // reads tail with acquire
 // to check "is there space?"         // to check "is there data?"
```

**Correctness argument:**
1. Producer stores data, then release-stores `tail`. Consumer acquire-loads `tail`, then reads data. The acquire-release pair guarantees the consumer sees the data.
2. Consumer reads data, then release-stores `head`. Producer acquire-loads `head`, then writes new data to the freed slot. The acquire-release pair guarantees the producer doesn't overwrite data the consumer hasn't read yet.
3. No two threads write the same index → no CAS, no ABA.

### Happens-Before Graph for One Push/Pop Cycle

```
  Producer Thread                Consumer Thread
  ---------------                ---------------
  buf[1] = 'D'        (A)
        |
        | seq-before
        v
  tail.store(2, rel)  (B) ----synchronizes-with----> tail.load(acq) (C)
                                                           |
                                                           | seq-before
                                                           v
                                                     read buf[1]   (D)

  A happens-before B  (sequenced-before)
  B happens-before C  (synchronizes-with: release/acquire)
  C happens-before D  (sequenced-before)
  
  Therefore: A happens-before D  ✓  (data is visible)
```

### Sizing and Cache Considerations

- **Power-of-2 capacity** — replace modulo with bitmask: `idx & (cap - 1)`
- **Pad head and tail to separate cache lines** — prevents false sharing:

```cpp
struct alignas(64) SPSCQueue {
    alignas(64) std::atomic<size_t> head{0};  // consumer cache line
    alignas(64) std::atomic<size_t> tail{0};  // producer cache line
    alignas(64) T buf[N];                      // data cache lines
};
```

Without the padding, `head` and `tail` sit on the same cache line. Every write by the producer invalidates the consumer's cache line (and vice versa) — destroying performance despite being logically contention-free.

---

## Summary Table

| Concept | Key Insight |
|---|---|
| No exceptions in RT | Non-deterministic timing, RTTI bloat, MISRA ban |
| Expected&lt;T,E&gt; | Deterministic, composable, zero-overhead error handling |
| Memory location | C++ unit of data-race analysis |
| Store buffer | Why writes become visible out of order |
| MESI | Cache-line level coherence, not ordering |
| Release | "Drain my store buffer, publish everything before me" |
| Acquire | "Don't move anything past me until I complete" |
| CAS | Atomic read-modify-write; ABA is the trap |
| SPSC queue | acquire-release on indices, no CAS needed |

---

## Exercises (in `exercises/` directory)

| File | Topic |
|---|---|
| `ex01_expected.cpp` | Implement `Expected<T,E>` with `and_then` chaining |
| `ex02_error_pipeline.cpp` | Build a 4-stage processing pipeline using Expected |
| `ex03_variant_visitor.cpp` | Error handling with `std::variant` and `std::visit` |
| `ex04_store_buffer_demo.cpp` | Demonstrate store-buffer reordering (run under TSan) |
| `ex05_acquire_release.cpp` | Publish/subscribe pattern with acquire-release |
| `ex06_cas_counter.cpp` | Lock-free counter using CAS loop |
| `ex07_spsc_queue.cpp` | Implement and test a lock-free SPSC ring buffer |
