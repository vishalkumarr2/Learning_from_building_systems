/**
 * Exercise 02: Task<T> — An Async Single-Value Coroutine
 *
 * This exercise builds a Task<T> class that:
 *   - Represents an async operation producing one value
 *   - Supports co_await to chain tasks together
 *   - Demonstrates suspension, resumption, and value propagation
 *   - Simulates an async pipeline: read → process → write
 *
 * Concepts demonstrated:
 *   - promise_type with return_value
 *   - Awaiter that resumes the waiting coroutine
 *   - Continuation-based resumption (awaiting coroutine stores itself)
 *   - Exception propagation through co_await chains
 *
 * Compile:
 *   g++-10 -std=c++20 -fcoroutines -Wall -Wextra -o ex02 ex02_async_task.cpp
 *   clang++-14 -std=c++20 -Wall -Wextra -o ex02 ex02_async_task.cpp
 */

#include <coroutine>
#include <exception>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// =============================================================================
// Task<T> — a single-value async coroutine
// =============================================================================

template <typename T>
class Task {
public:
    struct promise_type {
        // The result of the computation
        std::optional<T> result;
        std::exception_ptr exception;

        // The coroutine that is waiting for this task's result.
        // When this task completes, we resume the waiter.
        std::coroutine_handle<> continuation = nullptr;

        Task get_return_object() {
            return Task{
                std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        // Lazy start: don't run until someone co_awaits or explicitly resumes.
        std::suspend_always initial_suspend() noexcept { return {}; }

        // At final_suspend, resume whoever is waiting for our result.
        // We use a custom awaiter here instead of suspend_always so we can
        // resume the continuation.
        struct FinalAwaiter {
            bool await_ready() noexcept { return false; }

            // Symmetric transfer: return the continuation handle so the
            // compiler can tail-call it (avoids stack buildup).
            std::coroutine_handle<> await_suspend(
                std::coroutine_handle<promise_type> me) noexcept {
                auto cont = me.promise().continuation;
                if (cont) {
                    return cont; // Resume the waiter via symmetric transfer
                }
                return std::noop_coroutine(); // Nobody waiting — return to
                                              // caller
            }

            void await_resume() noexcept {}
        };

        FinalAwaiter final_suspend() noexcept { return {}; }

        void unhandled_exception() { exception = std::current_exception(); }

        // Called on co_return value;
        void return_value(T value) { result = std::move(value); }
    };

    // ---- Awaiter: allows co_await on a Task<T> ----
    // When coroutine A does `auto x = co_await taskB;`, this awaiter:
    //   1. Checks if taskB is already done (await_ready)
    //   2. Stores A as the continuation in taskB's promise
    //   3. Resumes taskB so it runs
    //   4. When taskB hits co_return, its final_suspend resumes A
    //   5. await_resume extracts the result

    struct Awaiter {
        std::coroutine_handle<promise_type> handle;

        bool await_ready() noexcept {
            // If the task already completed, no need to suspend
            return handle.done();
        }

        // Called when the awaiting coroutine suspends.
        // Store the awaiting coroutine as our continuation, then
        // resume this task so it starts/continues running.
        std::coroutine_handle<> await_suspend(
            std::coroutine_handle<> awaiting) noexcept {
            handle.promise().continuation = awaiting;
            return handle; // Symmetric transfer: run this task
        }

        // Called when the awaiting coroutine resumes.
        // Extract the result (or re-throw the exception).
        T await_resume() {
            if (handle.promise().exception) {
                std::rethrow_exception(handle.promise().exception);
            }
            return std::move(*handle.promise().result);
        }
    };

    // Allow co_await on Task<T>
    Awaiter operator co_await() {
        return Awaiter{handle_};
    }

    // ---- RAII handle management ----

    explicit Task(std::coroutine_handle<promise_type> h) : handle_(h) {}

    Task(Task&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    Task& operator=(Task&& other) noexcept {
        if (this != &other) {
            if (handle_) handle_.destroy();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;

    ~Task() {
        if (handle_) handle_.destroy();
    }

    // Synchronous execution: resume the task and return its result.
    // For use from non-coroutine code (like main()).
    T sync_get() {
        if (!handle_.done()) {
            handle_.resume();
        }
        if (handle_.promise().exception) {
            std::rethrow_exception(handle_.promise().exception);
        }
        return std::move(*handle_.promise().result);
    }

private:
    std::coroutine_handle<promise_type> handle_;
};

// =============================================================================
// Task<void> specialization
// =============================================================================

template <>
class Task<void> {
public:
    struct promise_type {
        std::exception_ptr exception;
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
                auto cont = me.promise().continuation;
                return cont ? cont : std::noop_coroutine();
            }
            void await_resume() noexcept {}
        };

        FinalAwaiter final_suspend() noexcept { return {}; }
        void unhandled_exception() { exception = std::current_exception(); }
        void return_void() {}
    };

    struct Awaiter {
        std::coroutine_handle<promise_type> handle;

        bool await_ready() noexcept { return handle.done(); }

        std::coroutine_handle<> await_suspend(
            std::coroutine_handle<> awaiting) noexcept {
            handle.promise().continuation = awaiting;
            return handle;
        }

        void await_resume() {
            if (handle.promise().exception) {
                std::rethrow_exception(handle.promise().exception);
            }
        }
    };

    Awaiter operator co_await() { return Awaiter{handle_}; }

    explicit Task(std::coroutine_handle<promise_type> h) : handle_(h) {}
    Task(Task&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    Task& operator=(Task&& other) noexcept {
        if (this != &other) {
            if (handle_) handle_.destroy();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }
    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;
    ~Task() {
        if (handle_) handle_.destroy();
    }

    void sync_run() {
        if (!handle_.done()) handle_.resume();
        if (handle_.promise().exception) {
            std::rethrow_exception(handle_.promise().exception);
        }
    }

private:
    std::coroutine_handle<promise_type> handle_;
};

// =============================================================================
// Example async operations (simulated — no real I/O)
// =============================================================================

// Simulates reading raw data from a source
Task<std::string> async_read(const std::string& source) {
    std::cout << "  [async_read] Reading from '" << source << "'...\n";
    // In a real system, this would co_await an I/O operation.
    // Here we just produce a result immediately.
    co_return "raw_data_from_" + source;
}

// Simulates processing data
Task<std::string> async_process(std::string data) {
    std::cout << "  [async_process] Processing '" << data << "'...\n";
    co_return "processed(" + data + ")";
}

// Simulates writing data to a destination
Task<int> async_write(const std::string& dest, std::string data) {
    std::cout << "  [async_write] Writing '" << data << "' to '" << dest
              << "'...\n";
    int bytes_written = static_cast<int>(data.size());
    co_return bytes_written;
}

// =============================================================================
// Pipeline: chains async operations with co_await
// =============================================================================
//
// This demonstrates the key advantage of coroutines: complex async workflows
// written as sequential code. No callbacks, no promise chains.

Task<int> async_pipeline(const std::string& source, const std::string& dest) {
    std::cout << "  [pipeline] Starting pipeline: " << source << " -> " << dest
              << "\n";

    // Step 1: Read (await suspends until read completes)
    auto raw_data = co_await async_read(source);

    // Step 2: Process (await suspends until processing completes)
    auto processed = co_await async_process(std::move(raw_data));

    // Step 3: Write (await suspends until write completes)
    auto bytes = co_await async_write(dest, std::move(processed));

    std::cout << "  [pipeline] Pipeline complete. Wrote " << bytes
              << " bytes.\n";
    co_return bytes;
}

// =============================================================================
// Demonstrate chaining multiple tasks
// =============================================================================

Task<int> multi_pipeline() {
    int total = 0;

    // Run three pipelines sequentially (each awaited in turn)
    total += co_await async_pipeline("sensor_data.bag", "analysis.csv");
    total += co_await async_pipeline("config.yaml", "config_backup.yaml");
    total += co_await async_pipeline("logs.txt", "logs_processed.txt");

    std::cout << "  [multi] All pipelines done. Total bytes: " << total << "\n";
    co_return total;
}

// =============================================================================
// Demonstrate exception propagation
// =============================================================================

Task<int> failing_task() {
    std::cout << "  [failing] About to throw...\n";
    throw std::runtime_error("Simulated async failure");
    co_return 42; // Never reached
}

Task<int> exception_demo() {
    std::cout << "  [exception_demo] About to await failing task...\n";
    try {
        auto result = co_await failing_task();
        std::cout << "  [exception_demo] Got result: " << result << "\n";
        co_return result;
    } catch (const std::exception& e) {
        std::cout << "  [exception_demo] Caught exception: " << e.what()
                  << "\n";
        co_return -1;
    }
}

// =============================================================================
// main()
// =============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << " Exercise 02: Task<T> Async Coroutine\n";
    std::cout << "========================================\n\n";

    // --- Simple single task ---
    std::cout << "--- Single async_read task ---\n";
    {
        auto task = async_read("robot_log.bag");
        auto result = task.sync_get();
        std::cout << "  Result: " << result << "\n\n";
    }

    // --- Full pipeline ---
    std::cout << "--- Async pipeline (read -> process -> write) ---\n";
    {
        auto task = async_pipeline("odometry.bag", "trajectory.csv");
        auto bytes = task.sync_get();
        std::cout << "  Pipeline returned: " << bytes << " bytes\n\n";
    }

    // --- Multi-pipeline chaining ---
    std::cout << "--- Multi-pipeline (3 sequential pipelines) ---\n";
    {
        auto task = multi_pipeline();
        auto total = task.sync_get();
        std::cout << "  Total bytes across all pipelines: " << total << "\n\n";
    }

    // --- Exception propagation ---
    std::cout << "--- Exception propagation through co_await ---\n";
    {
        auto task = exception_demo();
        auto result = task.sync_get();
        std::cout << "  Exception demo returned: " << result << "\n\n";
    }

    std::cout << "========================================\n";
    std::cout << " All Task<T> exercises complete.\n";
    std::cout << "========================================\n";

    return 0;
}
