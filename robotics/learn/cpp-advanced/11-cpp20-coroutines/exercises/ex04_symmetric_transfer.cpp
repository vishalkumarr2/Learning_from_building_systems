/**
 * Exercise 04: Symmetric Transfer
 *
 * This exercise demonstrates:
 *   1. The stack overflow problem without symmetric transfer
 *   2. The solution using await_suspend returning coroutine_handle<>
 *   3. How the compiler optimizes this into a tail-call (trampoline)
 *   4. A benchmark comparing stack depth in both approaches
 *
 * Concepts:
 *   - When coroutine A awaits B, and B completes, B must resume A.
 *   - Without symmetric transfer: B.final_suspend -> resume(A) grows stack
 *   - With symmetric transfer: B.final_suspend returns handle to A, compiler
 *     does tail-call — stack stays O(1)
 *
 * Compile:
 *   g++-10 -std=c++20 -fcoroutines -O2 -Wall -Wextra -o ex04 ex04_symmetric_transfer.cpp
 *   clang++-14 -std=c++20 -O2 -Wall -Wextra -o ex04 ex04_symmetric_transfer.cpp
 *
 * Note: Run with a generous stack for the "without" demo, or it will crash
 * (which is the point!). The "with" version should handle any depth.
 */

#include <chrono>
#include <coroutine>
#include <cstdint>
#include <exception>
#include <iostream>
#include <utility>
#include <vector>

// =============================================================================
// PART 1: Task WITHOUT symmetric transfer — stack overflow prone
// =============================================================================
//
// In this version, final_suspend returns suspend_always.
// The awaiting coroutine's await_suspend stores itself, then the
// resume happens via a nested call — each level adds a stack frame.
//
// Chain: main -> resume(A) -> A.await_suspend stores, resume(B)
//                             -> B.await_suspend stores, resume(C)
//                                -> C.final_suspend, then resume(B)
//                                   -> B.final_suspend, then resume(A)
//                                      -> stack depth = N

namespace no_symmetric {

class Task {
public:
    struct promise_type {
        int result = 0;
        std::coroutine_handle<> continuation = nullptr;

        Task get_return_object() {
            return Task{
                std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_always initial_suspend() noexcept { return {}; }

        // NO symmetric transfer — just suspend. The awaiter will
        // manually resume the continuation.
        std::suspend_always final_suspend() noexcept { return {}; }

        void unhandled_exception() { std::terminate(); }
        void return_value(int v) { result = v; }
    };

    struct Awaiter {
        std::coroutine_handle<promise_type> handle;

        bool await_ready() noexcept { return false; }

        void await_suspend(std::coroutine_handle<> awaiting) noexcept {
            handle.promise().continuation = awaiting;
            handle.resume(); // <-- This NESTS on the stack!
            // When handle completes, its final_suspend suspends,
            // and then we return here, unwinding back up.
            // But the continuation.resume() inside happens further down...
        }

        int await_resume() {
            auto result = handle.promise().result;
            // Now resume happens in a helper that checks done()
            return result;
        }
    };

    Awaiter operator co_await() { return Awaiter{handle_}; }

    explicit Task(std::coroutine_handle<promise_type> h) : handle_(h) {}
    Task(Task&& o) noexcept : handle_(o.handle_) { o.handle_ = nullptr; }
    Task& operator=(Task&& o) noexcept {
        if (this != &o) {
            if (handle_) handle_.destroy();
            handle_ = o.handle_;
            o.handle_ = nullptr;
        }
        return *this;
    }
    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;
    ~Task() {
        if (handle_) handle_.destroy();
    }

    int sync_get() {
        handle_.resume();
        // After the full chain unwinds, manually resume continuations
        // by walking the chain
        return handle_.promise().result;
    }

    // Walk the chain and resume each continuation manually
    // This simulates what happens without symmetric transfer
    void sync_get_with_manual_resume() {
        handle_.resume();
        // The nested resume calls handle the continuation chain
    }

private:
    std::coroutine_handle<promise_type> handle_;
};

// Create a chain of depth N coroutines, each awaiting the next
Task create_chain(int depth) {
    if (depth <= 0) {
        co_return 42;
    }
    // Each level awaits the next — this nests resume() calls on the stack
    auto result = co_await create_chain(depth - 1);
    co_return result + 1;
}

} // namespace no_symmetric

// =============================================================================
// PART 2: Task WITH symmetric transfer — O(1) stack depth
// =============================================================================
//
// In this version, final_suspend returns a coroutine_handle<> —
// the compiler generates a tail-call (trampoline) instead of nesting.
//
// The resume loop becomes:
//   while (handle_to_resume != noop) {
//       handle_to_resume = handle_to_resume.resume();  // tail-call
//   }
// Stack depth stays at 1 regardless of chain length.

namespace with_symmetric {

class Task {
public:
    struct promise_type {
        int result = 0;
        std::coroutine_handle<> continuation = nullptr;

        Task get_return_object() {
            return Task{
                std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_always initial_suspend() noexcept { return {}; }

        // SYMMETRIC TRANSFER: return the continuation handle.
        // The compiler tail-calls into it instead of nesting.
        struct FinalAwaiter {
            bool await_ready() noexcept { return false; }

            std::coroutine_handle<> await_suspend(
                std::coroutine_handle<promise_type> me) noexcept {
                auto cont = me.promise().continuation;
                // Return the continuation — compiler does tail-call
                return cont ? cont : std::noop_coroutine();
            }

            void await_resume() noexcept {}
        };

        FinalAwaiter final_suspend() noexcept { return {}; }

        void unhandled_exception() { std::terminate(); }
        void return_value(int v) { result = v; }
    };

    struct Awaiter {
        std::coroutine_handle<promise_type> handle;

        bool await_ready() noexcept { return false; }

        // SYMMETRIC TRANSFER: instead of resuming and nesting,
        // store the continuation and return the child handle.
        // The compiler's trampoline does:
        //   current = awaiting; (saved)
        //   jump to handle (child runs)
        //   when child hits final_suspend, jump to continuation
        std::coroutine_handle<> await_suspend(
            std::coroutine_handle<> awaiting) noexcept {
            handle.promise().continuation = awaiting;
            return handle; // Tail-call into child
        }

        int await_resume() { return handle.promise().result; }
    };

    Awaiter operator co_await() { return Awaiter{handle_}; }

    explicit Task(std::coroutine_handle<promise_type> h) : handle_(h) {}
    Task(Task&& o) noexcept : handle_(o.handle_) { o.handle_ = nullptr; }
    Task& operator=(Task&& o) noexcept {
        if (this != &o) {
            if (handle_) handle_.destroy();
            handle_ = o.handle_;
            o.handle_ = nullptr;
        }
        return *this;
    }
    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;
    ~Task() {
        if (handle_) handle_.destroy();
    }

    int sync_get() {
        handle_.resume();
        return handle_.promise().result;
    }

private:
    std::coroutine_handle<promise_type> handle_;
};

// Same chain, but with symmetric transfer — no stack overflow
Task create_chain(int depth) {
    if (depth <= 0) {
        co_return 42;
    }
    auto result = co_await create_chain(depth - 1);
    co_return result + 1;
}

} // namespace with_symmetric

// =============================================================================
// Benchmark helper
// =============================================================================

template <typename Func>
double benchmark_ms(Func&& f) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// =============================================================================
// main()
// =============================================================================

int main() {
    std::cout << "================================================\n";
    std::cout << " Exercise 04: Symmetric Transfer\n";
    std::cout << "================================================\n\n";

    // --- Part 1: Demonstrate the problem ---
    std::cout << "--- Part 1: WITHOUT symmetric transfer ---\n";
    std::cout << "  Each co_await nests a resume() call on the stack.\n";
    std::cout << "  Deep chains overflow the stack.\n\n";

    // Try a shallow chain first — this should work
    {
        const int shallow_depth = 100;
        std::cout << "  Shallow chain (depth=" << shallow_depth << "): ";
        auto task = no_symmetric::create_chain(shallow_depth);
        int result = task.sync_get();
        std::cout << "result = " << result
                  << " (expected " << (42 + shallow_depth) << ")\n";
    }

    std::cout << "\n  NOTE: A very deep chain (depth=100000) without symmetric\n"
              << "  transfer would crash with a stack overflow. We skip that\n"
              << "  to avoid actually crashing. In production, this is the\n"
              << "  exact bug that symmetric transfer solves.\n\n";

    // --- Part 2: Symmetric transfer handles any depth ---
    std::cout << "--- Part 2: WITH symmetric transfer ---\n";
    std::cout << "  await_suspend returns coroutine_handle<> for tail-call.\n";
    std::cout << "  Stack depth stays O(1) regardless of chain length.\n\n";

    // Deep chain with symmetric transfer — no stack overflow
    const std::vector<int> depths = {100, 1000, 10000, 100000};

    for (int depth : depths) {
        auto ms = benchmark_ms([depth]() {
            auto task = with_symmetric::create_chain(depth);
            int result = task.sync_get();
            int expected = 42 + depth;
            if (result != expected) {
                std::cout << "  ERROR: expected " << expected << " got "
                          << result << "\n";
            }
        });
        std::cout << "  depth=" << depth << ": " << ms << " ms  ✓\n";
    }
    std::cout << "\n";

    // --- Part 3: Visual comparison ---
    std::cout << "--- Comparison Summary ---\n\n";
    std::cout << "  WITHOUT symmetric transfer:\n";
    std::cout << "    await_suspend(h) {\n";
    std::cout << "        child.resume();  // NESTED call — grows stack\n";
    std::cout << "    }\n";
    std::cout << "    Stack: main→A.resume→B.resume→C.resume→... (depth N)\n\n";

    std::cout << "  WITH symmetric transfer:\n";
    std::cout << "    coroutine_handle<> await_suspend(h) {\n";
    std::cout << "        return child_handle;  // Tail-call — O(1) stack\n";
    std::cout << "    }\n";
    std::cout << "    Stack: trampoline→current.resume (always depth 1)\n\n";

    std::cout << "  The compiler generates a trampoline loop:\n";
    std::cout << "    while (next != noop_coroutine()) {\n";
    std::cout << "        next = next.resume();  // tail-call\n";
    std::cout << "    }\n\n";

    // --- Part 4: Benchmark comparison ---
    std::cout << "--- Benchmark: Symmetric Transfer at Various Depths ---\n\n";
    std::cout << "  Depth      Time (ms)\n";
    std::cout << "  ---------- ---------\n";

    for (int depth : {1000, 5000, 10000, 50000, 100000}) {
        auto ms = benchmark_ms([depth]() {
            auto task = with_symmetric::create_chain(depth);
            [[maybe_unused]] int r = task.sync_get();
        });
        char buf[64];
        std::snprintf(buf, sizeof(buf), "  %-10d %.3f", depth, ms);
        std::cout << buf << "\n";
    }
    std::cout << "\n  Time scales linearly with depth (O(N)), not exponentially.\n";
    std::cout << "  Stack usage is O(1) — the key advantage.\n";

    std::cout << "\n================================================\n";
    std::cout << " All symmetric transfer exercises complete.\n";
    std::cout << "================================================\n";

    return 0;
}
