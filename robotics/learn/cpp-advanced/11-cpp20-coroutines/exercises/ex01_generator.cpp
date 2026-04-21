/**
 * Exercise 01: Generator<T> — A Lazy Sequence Coroutine
 *
 * This exercise builds a Generator class from scratch that:
 *   - Uses co_yield to produce values lazily
 *   - Supports range-for iteration
 *   - Properly manages the coroutine frame lifetime
 *
 * Concepts demonstrated:
 *   - promise_type with yield_value and return_void
 *   - coroutine_handle ownership and RAII destruction
 *   - Iterator interface wrapping a coroutine
 *   - Lazy evaluation — values computed only when pulled
 *
 * Compile:
 *   g++-10 -std=c++20 -fcoroutines -Wall -Wextra -o ex01 ex01_generator.cpp
 *   clang++-14 -std=c++20 -Wall -Wextra -o ex01 ex01_generator.cpp
 */

#include <coroutine>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <utility>

// =============================================================================
// Generator<T> — full implementation
// =============================================================================

template <typename T>
class Generator {
public:
    // ---- Promise type: the coroutine machinery ----
    struct promise_type {
        // The most recently yielded value. We store it here so the
        // caller can read it via the coroutine handle.
        std::optional<T> current_value;

        // Store any unhandled exception from inside the coroutine
        std::exception_ptr exception;

        // Called at coroutine creation — returns the Generator object
        // that the caller receives.
        Generator get_return_object() {
            return Generator{
                std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        // Suspend immediately at start — the generator doesn't run until
        // the first value is requested (lazy start).
        std::suspend_always initial_suspend() noexcept { return {}; }

        // Suspend at the end so the caller can detect completion.
        // Must be noexcept — if this throws, std::terminate is called.
        std::suspend_always final_suspend() noexcept { return {}; }

        // Called when the coroutine body throws an exception.
        // We capture it for later re-throwing.
        void unhandled_exception() { exception = std::current_exception(); }

        // Called on co_yield expr. Stores the value and suspends.
        // Returns suspend_always so the caller gets control back.
        std::suspend_always yield_value(T value) {
            current_value = std::move(value);
            return {};
        }

        // Generator uses co_yield, not co_return with a value.
        // co_return; (without value) triggers return_void().
        void return_void() {}
    };

    // ---- Iterator: enables range-for loops ----
    // This is a simplified input iterator — enough for range-for.
    struct Iterator {
        std::coroutine_handle<promise_type> handle;
        bool done;

        Iterator() : handle(nullptr), done(true) {}
        explicit Iterator(std::coroutine_handle<promise_type> h)
            : handle(h), done(false) {
            advance(); // Fetch the first value
        }

        void advance() {
            if (handle && !handle.done()) {
                handle.resume();
                // After resume, check if coroutine finished
                if (handle.done()) {
                    done = true;
                    // Re-throw any stored exception
                    if (handle.promise().exception) {
                        std::rethrow_exception(handle.promise().exception);
                    }
                }
            } else {
                done = true;
            }
        }

        // Dereference: return the current yielded value
        const T& operator*() const { return *handle.promise().current_value; }

        // Prefix increment: resume the coroutine to get next value
        Iterator& operator++() {
            advance();
            return *this;
        }

        // Sentinel comparison for range-for
        bool operator==(const Iterator& other) const {
            return done == other.done;
        }
        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }
    };

    // ---- Generator RAII wrapper ----

    // Construct from a coroutine handle (only promise_type calls this)
    explicit Generator(std::coroutine_handle<promise_type> h) : handle_(h) {}

    // Move-only: coroutine handle is a unique resource
    Generator(Generator&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }
    Generator& operator=(Generator&& other) noexcept {
        if (this != &other) {
            if (handle_) handle_.destroy();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    // No copies — the coroutine frame is a unique resource
    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;

    // Destructor: destroy the coroutine frame to prevent leaks.
    // This is critical — without this, the heap-allocated frame leaks.
    ~Generator() {
        if (handle_) handle_.destroy();
    }

    // Range-for support
    Iterator begin() { return Iterator{handle_}; }
    Iterator end() { return Iterator{}; }

    // Manual pull interface (alternative to range-for)
    bool next() {
        if (handle_ && !handle_.done()) {
            handle_.resume();
            return !handle_.done();
        }
        return false;
    }

    // Get the current value (after calling next())
    const T& value() const { return *handle_.promise().current_value; }

private:
    std::coroutine_handle<promise_type> handle_;
};

// =============================================================================
// Fibonacci Generator
// =============================================================================
//
// An infinite sequence of Fibonacci numbers. The generator never terminates
// on its own — the consumer decides when to stop pulling values.

Generator<std::uint64_t> fibonacci() {
    std::uint64_t a = 0;
    std::uint64_t b = 1;
    while (true) {
        co_yield a; // Suspend here, yield current value
        auto next = a + b;
        a = b;
        b = next;
    }
    // Unreachable in practice, but the coroutine machinery requires
    // either co_return or falling off the end (which calls return_void).
}

// =============================================================================
// Range Generator — finite sequence [start, end) with step
// =============================================================================

Generator<int> range(int start, int end, int step = 1) {
    for (int i = start; i < end; i += step) {
        co_yield i;
    }
    // Falls off the end → calls return_void() → coroutine completes
}

// =============================================================================
// File Line Generator — lazily reads lines from a file
// =============================================================================
//
// Demonstrates a practical generator that wraps I/O.
// Each line is read only when the consumer requests it.

Generator<std::string> read_lines(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        // Coroutine can simply return early — calls return_void()
        std::cerr << "  [read_lines] Could not open: " << filename << "\n";
        co_return;
    }

    std::string line;
    while (std::getline(file, line)) {
        co_yield line;
    }
    // File closes automatically (RAII) when ifstream destructor runs
    // as the coroutine frame is destroyed.
}

// =============================================================================
// Transform Generator — maps a function over another generator
// =============================================================================
//
// Shows how generators compose into lazy pipelines.

template <typename T, typename Func>
Generator<T> transform(Generator<T> source, Func func) {
    for (auto&& value : source) {
        co_yield func(std::move(value));
    }
}

// =============================================================================
// main() — demonstrate all generators
// =============================================================================

int main() {
    std::cout << "========================================\n";
    std::cout << " Exercise 01: Generator<T> Coroutine\n";
    std::cout << "========================================\n\n";

    // --- Fibonacci ---
    std::cout << "--- Fibonacci (first 20 numbers) ---\n";
    int count = 0;
    for (auto val : fibonacci()) {
        std::cout << val << " ";
        if (++count >= 20) break; // Consumer controls termination
    }
    std::cout << "\n\n";

    // --- Range ---
    std::cout << "--- Range(0, 10, 2) ---\n";
    for (auto val : range(0, 10, 2)) {
        std::cout << val << " ";
    }
    std::cout << "\n\n";

    // --- File line generator ---
    // We'll create a small temp file to demonstrate
    std::cout << "--- File Line Generator ---\n";
    {
        // Create a test file
        const std::string test_file = "/tmp/coroutine_test_lines.txt";
        {
            std::ofstream out(test_file);
            out << "Line 1: Hello from coroutine generators\n";
            out << "Line 2: Each line is lazily read\n";
            out << "Line 3: Only when the consumer asks\n";
            out << "Line 4: No buffering the entire file\n";
            out << "Line 5: This is the last line\n";
        }

        // Read lazily — each line fetched on demand
        for (const auto& line : read_lines(test_file)) {
            std::cout << "  > " << line << "\n";
        }
    }
    std::cout << "\n";

    // --- Transform pipeline ---
    std::cout << "--- Transform: range(1,6) * 10 ---\n";
    auto doubled = transform<int>(range(1, 6), [](int x) { return x * 10; });
    for (auto val : doubled) {
        std::cout << val << " ";
    }
    std::cout << "\n\n";

    // --- Manual pull interface ---
    std::cout << "--- Manual pull interface ---\n";
    auto fib = fibonacci();
    for (int i = 0; i < 10; ++i) {
        if (fib.next()) {
            std::cout << "fib[" << i << "] = " << fib.value() << "\n";
        }
    }
    std::cout << "\n";

    // --- Early termination demo ---
    std::cout << "--- Early termination (break at fib > 100) ---\n";
    std::cout << "Fibonacci values <= 100: ";
    for (auto val : fibonacci()) {
        if (val > 100) break; // Generator destructor cleans up the frame
        std::cout << val << " ";
    }
    std::cout << "\n";
    std::cout << "(Generator properly destroyed — no leak)\n";

    std::cout << "\n========================================\n";
    std::cout << " All generator exercises complete.\n";
    std::cout << "========================================\n";

    return 0;
}
