# Module 11: C++20 Coroutines

## What Are Coroutines?

A **coroutine** is a function that can **suspend** its execution and later **resume** from where it left off. Unlike regular functions which run to completion once called, coroutines can pause mid-execution, return control to the caller, and pick up exactly where they stopped.

C++20 coroutines are **stackless** — the coroutine frame (local variables, suspension point) is allocated on the heap (or elided by the compiler), not on the call stack. This makes them extremely lightweight compared to threads or fibers.

### How the compiler transforms coroutines

Any function containing `co_await`, `co_yield`, or `co_return` is a coroutine. The compiler transforms it into a state machine:

```
Original coroutine:
    Task<int> compute() {
        int a = co_await fetch_data();
        int b = co_await process(a);
        co_return a + b;
    }

Compiler generates (conceptually):
    - Allocates a coroutine frame on the heap
    - Stores local variables (a, b) in the frame
    - Creates a state machine with suspension points
    - Each co_await becomes a state transition
```

The coroutine frame contains:
- The promise object
- Parameters (copies)
- Local variables
- Suspension point index (which `co_await` are we at?)
- Temporaries spanning suspension points

---

## The Three Coroutine Keywords

### `co_await expr`
Suspends the coroutine until the awaited expression is ready. The expression must be an **Awaitable**.

```cpp
auto result = co_await some_async_operation();
// Execution resumes here when the operation completes
```

### `co_yield expr`
Yields a value to the caller and suspends. Shorthand for:
```cpp
co_await promise.yield_value(expr);
```
Used primarily in **generators** — coroutines that produce a sequence of values lazily.

### `co_return expr` / `co_return`
Completes the coroutine, optionally returning a value. Calls `promise.return_value(expr)` or `promise.return_void()`.

---

## The Coroutine Machinery

### Promise Type

Every coroutine has an associated **promise type** that controls its behavior. The compiler finds it via:

```cpp
// For a coroutine returning ReturnType:
using promise_type = typename std::coroutine_traits<ReturnType, Args...>::promise_type;
// Usually defined as ReturnType::promise_type
```

The promise type must provide:

| Method | Purpose |
|--------|---------|
| `get_return_object()` | Creates the return object (e.g., `Generator<T>`, `Task<T>`) |
| `initial_suspend()` | Return awaitable: suspend at start? (`suspend_always` = lazy, `suspend_never` = eager) |
| `final_suspend() noexcept` | Return awaitable: suspend at end? (usually `suspend_always` so caller can inspect result) |
| `unhandled_exception()` | Called if coroutine throws — typically stores `std::current_exception()` |
| `return_value(T)` / `return_void()` | Called on `co_return` |
| `yield_value(T)` | Called on `co_yield` (optional, for generators) |

### Coroutine Handle

`std::coroutine_handle<Promise>` is a non-owning pointer to the coroutine frame:

```cpp
std::coroutine_handle<promise_type> handle;

handle.resume();    // Resume execution
handle.done();      // Check if coroutine has finished
handle.destroy();   // Destroy the coroutine frame (free memory)
handle.promise();   // Access the promise object
```

**Critical**: Someone must call `handle.destroy()` or the coroutine frame leaks. Typically the return object (Generator, Task) owns the handle and destroys it in its destructor.

---

## Awaitable / Awaiter Concepts

An **Awaitable** is anything you can `co_await`. The compiler converts it to an **Awaiter** with three methods:

```cpp
struct MyAwaiter {
    bool await_ready() const noexcept;
    // Return true if result is already available (skip suspension)

    void await_suspend(std::coroutine_handle<> h) noexcept;
    // Called when the coroutine suspends.
    // Can also return bool (false = don't actually suspend)
    // Or return coroutine_handle<> for symmetric transfer

    T await_resume() noexcept;
    // Called when the coroutine resumes. Returns the co_await result.
};
```

The standard provides two built-in awaiters:
- `std::suspend_always` — `await_ready()` returns false (always suspends)
- `std::suspend_never` — `await_ready()` returns true (never suspends)

---

## Generator Pattern (Lazy Sequences)

Generators produce values on-demand using `co_yield`:

```cpp
Generator<int> fibonacci() {
    int a = 0, b = 1;
    while (true) {
        co_yield a;
        auto next = a + b;
        a = b;
        b = next;
    }
}

// Consumer pulls values lazily:
for (int val : fibonacci()) {
    if (val > 1000) break;
    std::cout << val << "\n";
}
```

Key properties:
- **Lazy**: values computed only when requested
- **Infinite sequences**: the generator can run forever; consumer controls termination
- **Memory efficient**: only one value in flight at a time
- **initial_suspend** returns `suspend_always` so the generator doesn't run until first value is requested

---

## Async Task Pattern (Single-Value Future)

A `Task<T>` represents a computation that will eventually produce a single value:

```cpp
Task<std::string> fetch_url(std::string url) {
    auto response = co_await http_get(url);
    co_return response.body;
}

Task<int> compute() {
    auto data = co_await fetch_url("https://example.com");
    co_return data.size();
}
```

Key properties:
- **Lazy start**: `initial_suspend` returns `suspend_always`
- **Chainable**: `co_await`-ing one Task from another suspends the caller
- **Single value**: produces exactly one result (unlike generators)

---

## Symmetric Transfer and Tail-Call Optimization

### The Problem
When coroutine A awaits coroutine B, and B completes, B needs to resume A. Without symmetric transfer, this creates a chain of `resume()` calls on the stack:

```
main() → A.resume() → B.resume() → C.resume() → ...
```

With deep chains, this overflows the stack.

### The Solution: Symmetric Transfer
`await_suspend` can return a `coroutine_handle<>` instead of void:

```cpp
std::coroutine_handle<> await_suspend(std::coroutine_handle<> caller) noexcept {
    // Instead of: caller stored, then return to resume loop
    // Return the handle to resume next — compiler does tail-call
    return next_coroutine_to_resume;
}
```

The compiler generates a tail-call: instead of stacking resume() calls, it jumps directly to the next coroutine. Stack depth stays O(1).

---

## Coroutine Allocator Awareness

### Heap Allocation Elision (HALO)
The compiler can elide the heap allocation for the coroutine frame if:
- The coroutine's lifetime is fully enclosed by the caller
- The compiler can prove the frame size at compile time
- Optimization is enabled

This is similar to copy elision — the standard permits it but doesn't require it.

### Custom Allocation
Override `operator new`/`operator delete` in the promise type:

```cpp
struct promise_type {
    void* operator new(std::size_t size) {
        return my_pool_allocator::allocate(size);
    }
    void operator delete(void* ptr) {
        my_pool_allocator::deallocate(ptr);
    }
};
```

Useful for real-time systems where heap allocation is forbidden after initialization.

---

## Practical Use Cases

### 1. Async I/O
Replace callback-based async I/O with linear code:
```cpp
Task<Buffer> read_file(std::string path) {
    auto fd = co_await async_open(path);
    auto data = co_await async_read(fd);
    co_await async_close(fd);
    co_return data;
}
```

### 2. Lazy Pipelines
Chain transformations without materializing intermediate collections:
```cpp
Generator<int> filter_even(Generator<int> source) {
    for (int val : source) {
        if (val % 2 == 0) co_yield val;
    }
}
```

### 3. State Machines
Express complex state machines as sequential code with suspension points:
```cpp
Task<void> connection_handler(Socket sock) {
    auto handshake = co_await read_handshake(sock);
    if (!validate(handshake)) co_return;

    while (true) {
        auto request = co_await read_request(sock);
        if (request.is_close()) break;
        auto response = process(request);
        co_await write_response(sock, response);
    }
}
```

### 4. Cooperative Multitasking
A scheduler that round-robins between coroutines without threads:
```cpp
// Each coroutine yields back to the scheduler
scheduler.spawn(task_a());
scheduler.spawn(task_b());
scheduler.run();  // Runs all tasks cooperatively
```

---

## Coroutines vs. Threads

| Aspect | Coroutines | Threads |
|--------|-----------|---------|
| Scheduling | Cooperative (explicit `co_await`) | Preemptive (OS scheduler) |
| Context switch | ~nanoseconds (jump + restore registers) | ~microseconds (kernel transition) |
| Memory | ~200 bytes per coroutine frame | ~1-8 MB stack per thread |
| Concurrency | Single-threaded by default | True parallelism |
| Data races | No (single thread) | Yes (shared mutable state) |
| Scalability | Millions of coroutines | Thousands of threads |
| Blocking | One block stops everything | Only blocks that thread |

Coroutines excel for I/O-bound workloads with many concurrent operations. Threads excel for CPU-bound parallelism.

---

## ROS2 Tie-In: Replacing Callback Chains

ROS2's executor model dispatches callbacks (subscription, timer, service) in an event loop. Complex workflows require chaining callbacks, leading to "callback hell":

```cpp
// Current ROS2 callback chain:
void on_scan(LaserScan::SharedPtr msg) {
    auto result = process(msg);
    // Publish triggers another callback in a different node...
    publisher_->publish(result);
}
void on_result(Result::SharedPtr msg) {
    // Continue processing...
}
```

With coroutines, this could be linearized:
```cpp
// Hypothetical coroutine-based ROS2 workflow:
Task<void> scan_pipeline(Node& node) {
    while (rclcpp::ok()) {
        auto scan = co_await node.next_message<LaserScan>("/scan");
        auto result = process(scan);
        node.publish("/result", result);
        auto ack = co_await node.next_message<Ack>("/ack");
        // All sequential, no callback spaghetti
    }
}
```

This is the direction ROS2 is heading — `rclcpp` may adopt coroutine-friendly executors in future releases.

---

## Common Pitfalls

### 1. Dangling References
**The #1 coroutine bug.** Parameters and references can become dangling after suspension:

```cpp
Task<void> bad(const std::string& s) {
    co_await something();
    // s may be dangling! The caller's string could be destroyed.
    std::cout << s;  // UNDEFINED BEHAVIOR
}

void caller() {
    bad(std::string("temporary"));  // temporary destroyed before resume
}
```

**Fix**: Take parameters by value, or ensure the referenced object outlives the coroutine.

### 2. Forgetting to co_await
```cpp
Task<void> process() {
    expensive_async_operation();  // Oops — returned Task is discarded!
    // Use: co_await expensive_async_operation();
}
```

The Task object is created and immediately destroyed. The operation never runs. Use `[[nodiscard]]` on Task to get compiler warnings.

### 3. Coroutine Frame Lifetime
The coroutine frame lives until `destroy()` is called. If nobody destroys it, it leaks:

```cpp
void leak() {
    auto gen = fibonacci();  // Coroutine frame allocated
    // gen goes out of scope — destructor must call handle.destroy()
    // If Generator lacks a proper destructor, the frame leaks.
}
```

### 4. Exception in final_suspend
`final_suspend()` must be `noexcept`. If it throws, the program terminates.

### 5. Destroying a running coroutine
Calling `handle.destroy()` on a coroutine that is currently executing (not suspended) is undefined behavior.

---

## GCC/Clang Compiler Support

| Compiler | Version | Flag | Notes |
|----------|---------|------|-------|
| GCC | 10+ | `-std=c++20 -fcoroutines` | `-fcoroutines` flag required even with C++20 mode |
| GCC | 11+ | `-std=c++20 -fcoroutines` | Better optimization, fewer bugs |
| GCC | 13+ | `-std=c++20` | `-fcoroutines` no longer needed |
| Clang | 14+ | `-std=c++20` | Full support, no extra flags needed |
| Clang | 16+ | `-std=c++20` | Best optimization, symmetric transfer support |
| MSVC | 19.28+ | `/std:c++20` | Full support |

**This study module**: requires GCC 10+ with `-fcoroutines`, or Clang 14+.

GCC 9 (Ubuntu 20.04 default) does **NOT** support coroutines at all. Install GCC 10+:
```bash
sudo apt install g++-10
# Then compile with:
g++-10 -std=c++20 -fcoroutines -o ex01 ex01_generator.cpp
```

Or install a newer Clang:
```bash
sudo apt install clang-14
clang++-14 -std=c++20 -o ex01 ex01_generator.cpp
```

---

## Further Reading

- [cppreference: Coroutines](https://en.cppreference.com/w/cpp/language/coroutines)
- Lewis Baker's blog: "Asymmetric Transfer" — the definitive deep-dive
- David Mazières: "My tutorial and take on C++20 coroutines"
- P0973R0: Coroutines TS proposal
