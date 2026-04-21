# Module 15: PMR (Polymorphic Memory Resource) Allocators

## The Allocation Problem in Real-Time Systems

In real-time robotics and embedded systems, **deterministic timing** is non-negotiable.
Standard `new`/`malloc` is the enemy of determinism:

| Problem | Why It Kills RT |
|---------|----------------|
| **Mutex locks** | `malloc` uses internal locks → priority inversion, unbounded wait |
| **Page faults** | OS may need to map new pages → 1-100μs stall |
| **Fragmentation** | After hours of alloc/dealloc, no contiguous block available → OOM |
| **System calls** | `brk()`/`mmap()` for new heap pages → kernel entry |
| **Unpredictable latency** | Best case: 50ns. Worst case: 100μs+ (1000x variance) |

**Rule for RT**: All memory for hot paths must be pre-allocated at startup.

---

## The Traditional C++ Allocator Model (Pre-C++17)

The STL allocator model (`std::allocator<T>`) has fundamental problems:

```cpp
// 1. Type-parameterized → viral template propagation
std::vector<int, MyAlloc<int>> v;          // Allocator is part of the TYPE
std::vector<int, MyAlloc<int>> v2;         // Same type → can swap
std::vector<int, OtherAlloc<int>> v3;      // Different type! Can't swap with v.

// 2. Stateless by convention → can't point to a specific arena
// std::allocator<T> instances are interchangeable (no state)

// 3. Propagation nightmare → every nested container needs explicit allocator
std::vector<std::basic_string<char, std::char_traits<char>, MyAlloc<char>>,
            MyAlloc<std::basic_string<char, std::char_traits<char>, MyAlloc<char>>>>
// ^^^ This is insane. Nobody does this.
```

**Result**: Real-world code almost never uses custom STL allocators.

---

## The PMR Solution (C++17)

`<memory_resource>` introduces **polymorphic, stateful, runtime-switchable** allocators.

### Core Design

```
┌─────────────────────────────────────────────────┐
│  std::pmr::polymorphic_allocator<T>             │
│  (Thin wrapper, always same type regardless     │
│   of underlying resource)                        │
│                                                  │
│  Points to → std::pmr::memory_resource*         │
│              (Abstract base class)               │
└──────────────────────┬──────────────────────────┘
                       │ virtual dispatch
          ┌────────────┼────────────────────┐
          ▼            ▼                    ▼
   monotonic_     unsynchronized_     synchronized_
   buffer_        pool_              pool_
   resource       resource           resource
```

**Key insight**: `pmr::vector<int>` is always the same type, regardless of which
memory resource backs it. No more viral template propagation!

### `std::pmr::memory_resource` — The Base Class

```cpp
class memory_resource {
public:
    void* allocate(size_t bytes, size_t alignment = alignof(max_align_t));
    void  deallocate(void* p, size_t bytes, size_t alignment = alignof(max_align_t));
    bool  is_equal(const memory_resource& other) const noexcept;

protected:
    // Override these three:
    virtual void* do_allocate(size_t bytes, size_t alignment) = 0;
    virtual void  do_deallocate(void* p, size_t bytes, size_t alignment) = 0;
    virtual bool  do_is_equal(const memory_resource& other) const noexcept = 0;
};
```

---

## The 3 Standard Memory Resources

### 1. `monotonic_buffer_resource` — Bump Allocator (Fastest)

```cpp
char buffer[1024];
std::pmr::monotonic_buffer_resource mr(buffer, sizeof(buffer));

std::pmr::vector<int> v(&mr);
v.push_back(42);  // Allocated from buffer, pointer just bumps forward
```

**Properties**:
- Allocation = pointer increment (O(1), ~2 instructions)
- **Deallocation is a no-op** — memory is never returned to the resource
- `release()` frees ALL memory at once (resets to initial buffer)
- Falls back to upstream resource when buffer exhausted
- **Perfect for**: per-frame/per-cycle temporary allocations

**When to use**: You know the total allocation size upfront, or you'll release everything at once.

### 2. `unsynchronized_pool_resource` — Pool Allocator (Single-threaded)

```cpp
std::pmr::pool_options opts;
opts.max_blocks_per_chunk = 1024;
opts.largest_required_pool_block = 4096;

std::pmr::unsynchronized_pool_resource pool(opts, upstream);
```

**Properties**:
- Maintains pools of different block sizes (like `tcmalloc`)
- Allocation from matching pool = O(1) (pop from free list)
- Deallocation returns to pool for reuse (unlike monotonic)
- **Not thread-safe** — fastest pool option
- **Perfect for**: single-threaded hot paths with mixed alloc/dealloc

### 3. `synchronized_pool_resource` — Pool Allocator (Thread-safe)

Same as unsynchronized but with internal synchronization.
Use when multiple threads share the resource. Slower than unsynchronized.

---

## Global Resources & Utilities

```cpp
// Default resource (wraps ::operator new)
std::pmr::memory_resource* def = std::pmr::new_delete_resource();

// Throws std::bad_alloc on ANY allocation (for testing)
std::pmr::memory_resource* null = std::pmr::null_memory_resource();

// Set/get global default
std::pmr::memory_resource* old = std::pmr::set_default_resource(&my_resource);
std::pmr::memory_resource* cur = std::pmr::get_default_resource();
```

---

## PMR Containers

All standard containers have PMR aliases:

```cpp
#include <memory_resource>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <deque>
#include <list>
#include <set>

std::pmr::vector<int>                 v(&mr);
std::pmr::string                      s(&mr);
std::pmr::map<std::pmr::string, int>  m(&mr);
std::pmr::unordered_map<int, double>  um(&mr);
std::pmr::deque<float>                d(&mr);
std::pmr::list<int>                   l(&mr);
std::pmr::set<int>                    st(&mr);
```

### Critical: Nested Container Propagation

```cpp
// CORRECT: Inner strings use the SAME memory resource as the vector
std::pmr::vector<std::pmr::string> good(&mr);
good.emplace_back("hello");  // "hello" allocated from mr!

// BUG: Inner strings use the DEFAULT allocator (heap)
std::pmr::vector<std::string> bad(&mr);
bad.emplace_back("hello");  // "hello" allocated from ::operator new!
```

The inner container must also be a PMR type for propagation to work.

---

## Upstream Resources: Chaining Allocators

Resources can be chained. The upstream is the fallback when the primary is exhausted:

```
Stack buffer (fast, limited)
    └──▶ monotonic_buffer_resource
              └──▶ unsynchronized_pool_resource (upstream)
                        └──▶ new_delete_resource (upstream)
```

```cpp
char stack_buf[4096];

// Level 1: monotonic over stack buffer, falls back to pool
std::pmr::monotonic_buffer_resource mono(stack_buf, sizeof(stack_buf), &pool);

// Level 2: pool falls back to new_delete
std::pmr::unsynchronized_pool_resource pool(opts, std::pmr::new_delete_resource());
```

---

## Writing a Custom `memory_resource`

Override three virtual functions:

```cpp
class TrackingResource : public std::pmr::memory_resource {
    std::pmr::memory_resource* upstream_;
    size_t total_allocated_ = 0;
    size_t total_deallocated_ = 0;
    int    alloc_count_ = 0;

protected:
    void* do_allocate(size_t bytes, size_t alignment) override {
        void* p = upstream_->allocate(bytes, alignment);
        total_allocated_ += bytes;
        ++alloc_count_;
        return p;
    }

    void do_deallocate(void* p, size_t bytes, size_t alignment) override {
        upstream_->deallocate(p, bytes, alignment);
        total_deallocated_ += bytes;
    }

    bool do_is_equal(const memory_resource& other) const noexcept override {
        return this == &other;
    }
};
```

---

## Real-Time Patterns

### Pattern 1: Pre-allocated Arena for Hot Loop

```cpp
// STARTUP (non-RT)
constexpr size_t ARENA_SIZE = 1024 * 1024;  // 1 MB
alignas(std::max_align_t) char arena_buf[ARENA_SIZE];
std::pmr::monotonic_buffer_resource arena(arena_buf, ARENA_SIZE,
                                           std::pmr::null_memory_resource());
// null upstream → crash immediately if we overflow (better than silent heap use)

// HOT LOOP (RT, 1 kHz)
while (running) {
    // All allocations from pre-allocated buffer
    std::pmr::vector<SensorReading> readings(&arena);
    readings.reserve(MAX_SENSORS);
    process(readings);

    // Reset for next cycle — O(1), no deallocation
    arena.release();
}
```

### Pattern 2: Per-Frame Allocator (Game Engine Style)

```cpp
// Double-buffered arenas
char buf_a[FRAME_SIZE], buf_b[FRAME_SIZE];
std::pmr::monotonic_buffer_resource* current = new_arena(buf_a);
std::pmr::monotonic_buffer_resource* previous = new_arena(buf_b);

void on_frame() {
    std::swap(current, previous);
    current->release();
    // current is now fresh, previous still valid for GPU reads
}
```

---

## Performance: PMR vs Default Allocator

Typical benchmarks (10K small allocations):

| Allocator | Time | Relative |
|-----------|------|----------|
| `new`/`delete` | 500 μs | 1.0x (baseline) |
| `monotonic_buffer_resource` | 5 μs | **100x faster** |
| `unsynchronized_pool_resource` | 15 μs | **33x faster** |
| `synchronized_pool_resource` | 25 μs | **20x faster** |
| `monotonic` (stack buffer, no upstream) | 3 μs | **167x faster** |

The monotonic allocator is essentially free — it's a pointer bump.

---

## ROS 2 Tie-In

ROS 2 (`rclcpp`) supports custom allocators for real-time nodes:

```cpp
// rclcpp uses allocator interfaces for:
// - Message allocation in callbacks
// - Subscription/publisher internal buffers
// - Executor message handling

// The pattern: pre-allocate a PMR arena, pass to the executor
// This avoids heap allocation in the callback hot path
auto alloc = std::make_shared<rclcpp::memory_strategies::
    allocator_memory_strategy::AllocatorMemoryStrategy<MyPMRAllocator>>();
```

Key ROS 2 RT rules:
1. No allocation in timer/subscription callbacks
2. Pre-allocate message buffers
3. Use `realtime_tools::RealtimeBuffer` for inter-thread data
4. PMR arenas for any temporary containers in callbacks

---

## Common Pitfalls

### 1. Dangling References After Resource Destruction
```cpp
std::pmr::vector<int>* v;
{
    std::pmr::monotonic_buffer_resource mr;
    v = new std::pmr::vector<int>(&mr);
    v->push_back(42);
}  // mr destroyed! v's memory is now invalid!
v->at(0);  // UNDEFINED BEHAVIOR — silent corruption
```

### 2. Forgetting Upstream → Silent Heap Allocation
```cpp
// Default upstream is new_delete_resource() — silently falls back to heap
std::pmr::monotonic_buffer_resource mr(buf, 64);  // tiny buffer
// If you exceed 64 bytes, it heap-allocates. Use null_resource to catch this:
std::pmr::monotonic_buffer_resource mr(buf, 64, std::pmr::null_memory_resource());
```

### 3. Thread Safety
```cpp
// WRONG: sharing unsynchronized_pool across threads
std::pmr::unsynchronized_pool_resource pool;
std::thread t1([&]{ std::pmr::vector<int> v(&pool); /* ... */ });
std::thread t2([&]{ std::pmr::vector<int> v(&pool); /* ... */ });
// Data race! Use synchronized_pool_resource instead.
```

### 4. Non-PMR Inner Containers
```cpp
std::pmr::vector<std::string> v(&arena);  // std::string uses default new!
std::pmr::vector<std::pmr::string> v(&arena);  // CORRECT
```

---

## Comparison with Other Allocators

| Feature | PMR | tcmalloc | jemalloc | mimalloc | Custom Pool |
|---------|-----|----------|----------|----------|-------------|
| Standard C++ | ✅ | ❌ (replace malloc) | ❌ | ❌ | ❌ |
| Per-container control | ✅ | ❌ (global) | ❌ | ❌ | Depends |
| Deterministic timing | ✅ (monotonic) | ❌ | ❌ | ❌ | ✅ |
| Zero-overhead dealloc | ✅ (monotonic) | ❌ | ❌ | ❌ | Depends |
| Works with STL | ✅ | ✅ (transparent) | ✅ | ✅ | Complex |
| Composable/chainable | ✅ | ❌ | ❌ | ❌ | Manual |
| Thread-safe option | ✅ | ✅ | ✅ | ✅ | Manual |

**Bottom line**: PMR is the right tool when you need per-container, deterministic,
composable allocation within standard C++. Use `tcmalloc`/`jemalloc`/`mimalloc`
as global replacements when you want better average performance without per-container control.

---

## Key Takeaways

1. **PMR solves the allocator propagation problem** — one type for all allocators
2. **monotonic_buffer_resource is your RT workhorse** — bump allocator, release-all-at-once
3. **Chain resources** — stack → monotonic → pool → new_delete
4. **Use null_memory_resource as upstream** to catch unexpected overflow
5. **Inner containers must also be PMR** for propagation to work
6. **Pre-allocate everything at startup**, use arenas in hot paths
7. **PMR + null upstream = zero-allocation guarantee** at compile-test time
