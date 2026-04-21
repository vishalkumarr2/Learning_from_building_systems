/**
 * Puzzle 01: The Lifetime Trap — Dangling References in Coroutines
 *
 * This is the #1 most common coroutine bug. A coroutine captures a
 * reference to a local variable. After the coroutine suspends, the
 * caller's stack frame is destroyed. When the coroutine resumes,
 * it accesses a dangling reference — undefined behavior.
 *
 * CHALLENGE:
 *   1. Read the buggy code and identify where the UB occurs
 *   2. Uncomment the buggy version and observe the behavior (may crash,
 *      may print garbage, may "work" — that's the danger of UB)
 *   3. Study the fix and understand why it works
 *
 * Compile:
 *   g++-10 -std=c++20 -fcoroutines -Wall -Wextra -o puzzle01 puzzle01_lifetime_trap.cpp
 *   clang++-14 -std=c++20 -Wall -Wextra -o puzzle01 puzzle01_lifetime_trap.cpp
 */

#include <coroutine>
#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// =============================================================================
// Minimal Task for this puzzle
// =============================================================================

class Task {
public:
    struct promise_type {
        std::string result;
        std::coroutine_handle<> continuation = nullptr;

        Task get_return_object() {
            return Task{
                std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_always initial_suspend() noexcept { return {}; }

        struct FinalAwaiter {
            bool await_ready() noexcept { return false; }
            std::coroutine_handle<> await_suspend(
                std::coroutine_handle<promise_type> me) noexcept {
                auto c = me.promise().continuation;
                return c ? c : std::noop_coroutine();
            }
            void await_resume() noexcept {}
        };
        FinalAwaiter final_suspend() noexcept { return {}; }

        void unhandled_exception() { std::terminate(); }
        void return_value(std::string v) { result = std::move(v); }
    };

    struct Awaiter {
        std::coroutine_handle<promise_type> handle;
        bool await_ready() noexcept { return false; }
        std::coroutine_handle<> await_suspend(
            std::coroutine_handle<> awaiting) noexcept {
            handle.promise().continuation = awaiting;
            return handle;
        }
        std::string await_resume() {
            return std::move(handle.promise().result);
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

    std::string sync_get() {
        handle_.resume();
        return std::move(handle_.promise().result);
    }

private:
    std::coroutine_handle<promise_type> handle_;
};

// A simple awaitable that always suspends (simulates async work)
struct SuspendOnce {
    bool await_ready() noexcept { return false; }
    void await_suspend(std::coroutine_handle<>) noexcept {
        // In a real system, this would register the handle with an I/O
        // completion system. Here, the handle is just stored and will
        // be manually resumed.
    }
    void await_resume() noexcept {}
};

// =============================================================================
// THE BUG: Coroutine takes a const reference to a string
// =============================================================================
//
// The problem:
//   1. process_data() takes `data` by const reference
//   2. The caller creates a temporary string and passes it
//   3. process_data() hits co_await and suspends
//   4. The temporary string is destroyed (caller's scope ends)
//   5. process_data() resumes and reads `data` — DANGLING REFERENCE!
//
// Why this is tricky:
//   - With regular functions, const& extends temporary lifetime
//   - With coroutines, the suspension breaks this guarantee
//   - The compiler won't warn you (it can't see across suspension points)
//   - It might "work" in debug builds due to stack not being overwritten

/*
 * BUGGY VERSION — uncomment to see UB in action:
 *
 * Task process_data_BUGGY(const std::string& data) {
 *     std::cout << "  [BUGGY] Before suspend, data = '" << data << "'\n";
 *
 *     co_await SuspendOnce{};
 *     // ^^^ After this suspension point, `data` may be dangling!
 *     // The temporary string the caller passed has been destroyed.
 *
 *     // THIS IS UNDEFINED BEHAVIOR:
 *     std::cout << "  [BUGGY] After suspend, data = '" << data << "'\n";
 *     co_return "processed: " + data;  // Reading dangling reference!
 * }
 *
 * void trigger_bug() {
 *     // The temporary string "sensor_reading_42" is destroyed after
 *     // the coroutine suspends, but before it resumes.
 *     auto task = process_data_BUGGY(std::string("sensor_reading_42"));
 *     // ^ temporary destroyed here, while coroutine is suspended
 *
 *     auto result = task.sync_get();
 *     std::cout << "  [BUGGY] Result: " << result << "\n";
 * }
 */

// =============================================================================
// THE FIX: Take the string by value
// =============================================================================
//
// When a coroutine takes parameters by value, the compiler copies/moves
// them into the coroutine frame. The frame lives on the heap and persists
// across suspension points. No dangling references.

Task process_data_FIXED(std::string data) {
    // `data` is now a copy stored in the coroutine frame.
    // It lives as long as the coroutine frame — safe across suspension.
    std::cout << "  [FIXED] Before suspend, data = '" << data << "'\n";

    co_await SuspendOnce{};
    // After resumption, `data` is still valid — it's in the coroutine frame.

    std::cout << "  [FIXED] After suspend, data = '" << data << "'\n";
    co_return "processed: " + data; // Safe!
}

// =============================================================================
// Another common variant: capturing references in lambdas used in coroutines
// =============================================================================

Task lambda_trap_demo() {
    // BUG PATTERN (do NOT do this):
    // auto handler = [&local_var]() -> Task { co_await ...; use local_var; };
    //
    // The lambda captures by reference, but the coroutine frame outlives
    // the lambda's captures. After suspension, local_var is gone.

    // FIX: capture by value, or ensure the referenced object outlives
    // the coroutine.

    std::string safe_data = "captured_by_value";

    // Safe: capturing a copy in the coroutine itself
    std::cout << "  [Lambda] Using safely captured data: " << safe_data << "\n";
    co_return safe_data;
}

// =============================================================================
// Demonstration of the correct pattern
// =============================================================================

Task correct_pipeline() {
    // Pattern 1: Pass by value to coroutines
    auto r1 = co_await process_data_FIXED("temperature_reading");

    // Pattern 2: Use local variables (stored in coroutine frame)
    std::string local_data = "pressure_reading";
    // local_data lives in THIS coroutine's frame — safe across suspension
    co_await SuspendOnce{};
    std::cout << "  [Pipeline] Local data after suspend: " << local_data
              << "\n";

    // Pattern 3: Use shared_ptr for shared ownership
    // auto shared = std::make_shared<std::string>("shared_data");
    // co_await some_task(shared);  // shared_ptr keeps it alive

    co_return r1;
}

// =============================================================================
// main()
// =============================================================================

int main() {
    std::cout << "================================================\n";
    std::cout << " Puzzle 01: The Coroutine Lifetime Trap\n";
    std::cout << "================================================\n\n";

    std::cout << "--- The Bug ---\n";
    std::cout << R"(
  A coroutine takes a parameter by const reference:

    Task process(const std::string& data) {
        co_await something();    // SUSPENDS here
        use(data);               // data is DANGLING!
    }

  Called with a temporary:
    auto t = process(std::string("temp"));  // destroyed before resume!

  The temporary is destroyed while the coroutine is suspended.
  When it resumes, the reference is dangling. UNDEFINED BEHAVIOR.

  This is subtle because:
    - Regular functions: const& extends temporary lifetime
    - Coroutines: suspension breaks this guarantee
    - The compiler cannot warn you
    - It may "work" in debug builds (stack not overwritten yet)
)" << "\n";

    std::cout << "--- The Fix ---\n";
    std::cout << "  Take parameters by VALUE in coroutines.\n";
    std::cout << "  The compiler stores the copy in the coroutine frame.\n\n";

    // Demonstrate the fixed version
    std::cout << "--- Running fixed version ---\n";
    {
        auto task = process_data_FIXED("sensor_reading_42");
        auto result = task.sync_get();
        std::cout << "  Result: " << result << "\n\n";
    }

    // Demonstrate the full pipeline
    std::cout << "--- Correct pipeline ---\n";
    {
        auto task = correct_pipeline();
        auto result = task.sync_get();
        std::cout << "  Pipeline result: " << result << "\n\n";
    }

    std::cout << "--- Rules of Thumb ---\n";
    std::cout << R"(
  1. NEVER take parameters by reference in coroutines
     (unless you can guarantee the referenced object outlives the coroutine)

  2. ALWAYS take parameters by value — the compiler stores them
     in the coroutine frame (heap-allocated, safe across suspension)

  3. Be extra careful with:
     - const std::string& → use std::string
     - const std::vector<T>& → use std::vector<T>
     - std::string_view → use std::string (view may dangle!)
     - Lambda captures by reference
     - Pointers to stack variables

  4. Use std::shared_ptr when ownership must be shared between
     the caller and the coroutine

  5. [[clang::coro_lifetimebound]] attribute (Clang 17+) can help
     the compiler detect some of these issues at compile time
)" << "\n";

    std::cout << "================================================\n";
    std::cout << " Puzzle 01 complete.\n";
    std::cout << "================================================\n";

    return 0;
}
