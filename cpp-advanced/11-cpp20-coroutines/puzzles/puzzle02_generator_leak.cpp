/**
 * Puzzle 02: The Generator Resource Leak
 *
 * When a generator yields values that own heap resources, and the
 * consumer breaks out of the loop early, does the generator clean up?
 *
 * CHALLENGE:
 *   1. Read the code and predict: does the generator leak resources
 *      when the consumer breaks early?
 *   2. Run the code and verify your prediction
 *   3. Study how the Generator destructor ensures cleanup
 *   4. Understand the interplay between promise, handle, and RAII
 *
 * Compile:
 *   g++-10 -std=c++20 -fcoroutines -Wall -Wextra -o puzzle02 puzzle02_generator_leak.cpp
 *   clang++-14 -std=c++20 -Wall -Wextra -o puzzle02 puzzle02_generator_leak.cpp
 */

#include <coroutine>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// =============================================================================
// A resource that tracks its own lifetime
// =============================================================================

// Global counter to track resource leaks
static int g_resource_count = 0;
static int g_total_created = 0;
static int g_total_destroyed = 0;

struct TrackedResource {
    int id;
    std::string data;

    TrackedResource(int id_, std::string data_)
        : id(id_), data(std::move(data_)) {
        ++g_resource_count;
        ++g_total_created;
        std::cout << "    [Resource " << id << "] CREATED ('" << data
                  << "') — live count: " << g_resource_count << "\n";
    }

    ~TrackedResource() {
        --g_resource_count;
        ++g_total_destroyed;
        std::cout << "    [Resource " << id << "] DESTROYED ('" << data
                  << "') — live count: " << g_resource_count << "\n";
    }

    // Move-only
    TrackedResource(TrackedResource&& other) noexcept
        : id(other.id), data(std::move(other.data)) {
        other.id = -1; // Mark as moved-from
        // Don't adjust counters — ownership transferred, not new resource
    }
    TrackedResource& operator=(TrackedResource&& other) noexcept {
        if (this != &other) {
            // This object is being overwritten — its old resource is "lost"
            // (but we're moving, not destroying, so counter stays)
            id = other.id;
            data = std::move(other.data);
            other.id = -1;
        }
        return *this;
    }

    TrackedResource(const TrackedResource&) = delete;
    TrackedResource& operator=(const TrackedResource&) = delete;
};

// =============================================================================
// Generator<T> — same as ex01 but with explicit cleanup logging
// =============================================================================

template <typename T>
class Generator {
public:
    struct promise_type {
        std::optional<T> current_value;
        std::exception_ptr exception;

        Generator get_return_object() {
            return Generator{
                std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }

        void unhandled_exception() { exception = std::current_exception(); }

        std::suspend_always yield_value(T value) {
            current_value = std::move(value);
            return {};
        }

        void return_void() {}

        // THE KEY: when the promise is destroyed, any stored value
        // in current_value is also destroyed via optional's destructor.
        ~promise_type() {
            std::cout << "    [promise_type] Destroyed"
                      << (current_value.has_value() ? " (had pending value)"
                                                    : " (no pending value)")
                      << "\n";
        }
    };

    struct Iterator {
        std::coroutine_handle<promise_type> handle;
        bool done;

        Iterator() : handle(nullptr), done(true) {}
        explicit Iterator(std::coroutine_handle<promise_type> h)
            : handle(h), done(false) {
            advance();
        }

        void advance() {
            if (handle && !handle.done()) {
                handle.resume();
                if (handle.done()) {
                    done = true;
                    if (handle.promise().exception) {
                        std::rethrow_exception(handle.promise().exception);
                    }
                }
            } else {
                done = true;
            }
        }

        const T& operator*() const { return *handle.promise().current_value; }
        Iterator& operator++() {
            advance();
            return *this;
        }
        bool operator!=(const Iterator& other) const {
            return done != other.done;
        }
    };

    explicit Generator(std::coroutine_handle<promise_type> h) : handle_(h) {}

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

    Generator(const Generator&) = delete;
    Generator& operator=(const Generator&) = delete;

    // CRITICAL: The destructor calls handle_.destroy(), which destroys
    // the coroutine frame INCLUDING:
    //   - The promise object (and its current_value optional)
    //   - All local variables in the coroutine body
    //   - Any temporaries that were live at the suspension point
    //
    // This is the RAII guarantee: even if the consumer breaks early,
    // the Generator destructor runs, which destroys the coroutine frame,
    // which destroys everything inside it.
    ~Generator() {
        if (handle_) {
            std::cout << "    [Generator] Destroying coroutine frame...\n";
            handle_.destroy();
        }
    }

    Iterator begin() { return Iterator{handle_}; }
    Iterator end() { return Iterator{}; }

private:
    std::coroutine_handle<promise_type> handle_;
};

// =============================================================================
// Generator that produces TrackedResources
// =============================================================================

Generator<TrackedResource> resource_generator(int count) {
    std::cout << "    [Generator coroutine] Starting, will produce " << count
              << " resources\n";

    for (int i = 0; i < count; ++i) {
        // Create a resource — it's moved into the promise via yield_value
        TrackedResource res(i, "item_" + std::to_string(i));
        co_yield std::move(res);
        // After yield, the previous current_value in the promise is
        // replaced by the new one. The old one's destructor runs.
    }

    std::cout << "    [Generator coroutine] Finished all " << count
              << " items\n";
}

// =============================================================================
// SCENARIO 1: Consumer reads ALL values — no leak
// =============================================================================

void scenario_full_consumption() {
    std::cout << "\n  Scenario: Full consumption (read all 5 values)\n";
    std::cout << "  " << std::string(50, '-') << "\n";

    g_resource_count = 0;
    g_total_created = 0;
    g_total_destroyed = 0;

    {
        auto gen = resource_generator(5);
        for (const auto& res : gen) {
            std::cout << "    [Consumer] Got resource " << res.id << ": '"
                      << res.data << "'\n";
        }
    } // gen goes out of scope — destructor runs

    std::cout << "\n  Results: created=" << g_total_created
              << " destroyed=" << g_total_destroyed
              << " leaked=" << g_resource_count << "\n";
}

// =============================================================================
// SCENARIO 2: Consumer breaks early — does the generator leak?
// =============================================================================

void scenario_early_break() {
    std::cout << "\n  Scenario: Early break (read 2 of 5, then break)\n";
    std::cout << "  " << std::string(50, '-') << "\n";

    g_resource_count = 0;
    g_total_created = 0;
    g_total_destroyed = 0;

    {
        auto gen = resource_generator(5);
        int consumed = 0;
        for (const auto& res : gen) {
            std::cout << "    [Consumer] Got resource " << res.id << ": '"
                      << res.data << "'\n";
            if (++consumed >= 2) {
                std::cout << "    [Consumer] Breaking early!\n";
                break;
            }
        }
        std::cout << "    [Consumer] After break, gen still alive...\n";
    } // gen goes out of scope here — destructor destroys the coroutine frame

    std::cout << "\n  Results: created=" << g_total_created
              << " destroyed=" << g_total_destroyed
              << " leaked=" << g_resource_count << "\n";
}

// =============================================================================
// SCENARIO 3: What if Generator has NO destructor? (BAD — leak!)
// =============================================================================

// This is the "broken" generator that forgets to destroy the coroutine frame.
template <typename T>
class LeakyGenerator {
public:
    struct promise_type {
        std::optional<T> current_value;
        std::exception_ptr exception;

        LeakyGenerator get_return_object() {
            return LeakyGenerator{
                std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void unhandled_exception() { exception = std::current_exception(); }
        std::suspend_always yield_value(T value) {
            current_value = std::move(value);
            return {};
        }
        void return_void() {}
    };

    explicit LeakyGenerator(std::coroutine_handle<promise_type> h)
        : handle_(h) {}

    // BUG: No destructor! The coroutine frame is never freed.
    // The promise's current_value (and any live locals) leak.
    // ~LeakyGenerator() { /* missing handle_.destroy() */ }

    // Minimal pull interface
    bool next() {
        if (handle_ && !handle_.done()) {
            handle_.resume();
            return !handle_.done();
        }
        return false;
    }

    const T& value() const { return *handle_.promise().current_value; }

    // Also leaks if moved — but we're demonstrating the concept
    LeakyGenerator(LeakyGenerator&& o) noexcept : handle_(o.handle_) {
        o.handle_ = nullptr;
    }
    LeakyGenerator(const LeakyGenerator&) = delete;
    LeakyGenerator& operator=(const LeakyGenerator&) = delete;
    LeakyGenerator& operator=(LeakyGenerator&&) = delete;

    // Destructor intentionally missing handle_.destroy()!
    ~LeakyGenerator() {
        std::cout << "    [LeakyGenerator] Destructor called but does NOT "
                     "destroy coroutine frame!\n";
        // handle_.destroy();  // <-- This line is missing!
    }

private:
    std::coroutine_handle<promise_type> handle_;
};

LeakyGenerator<TrackedResource> leaky_resource_generator(int count) {
    for (int i = 0; i < count; ++i) {
        TrackedResource res(i, "leaky_" + std::to_string(i));
        co_yield std::move(res);
    }
}

void scenario_leak_demo() {
    std::cout << "\n  Scenario: LeakyGenerator (missing destroy) — LEAK!\n";
    std::cout << "  " << std::string(50, '-') << "\n";

    g_resource_count = 0;
    g_total_created = 0;
    g_total_destroyed = 0;

    {
        auto gen = leaky_resource_generator(3);
        gen.next(); // Consume first value
        std::cout << "    [Consumer] Got: " << gen.value().data << "\n";
        gen.next(); // Consume second value
        std::cout << "    [Consumer] Got: " << gen.value().data << "\n";
        std::cout << "    [Consumer] Stopping early...\n";
    } // gen destroyed — but handle_.destroy() is never called!

    std::cout << "\n  Results: created=" << g_total_created
              << " destroyed=" << g_total_destroyed
              << " LEAKED=" << g_resource_count << "\n";

    if (g_resource_count > 0) {
        std::cout << "  *** " << g_resource_count
                  << " resources leaked! The coroutine frame was never freed.\n";
        std::cout << "  *** The promise's current_value and any live locals "
                     "are lost.\n";
    }
}

// =============================================================================
// main()
// =============================================================================

int main() {
    std::cout << "================================================\n";
    std::cout << " Puzzle 02: Generator Resource Leak\n";
    std::cout << "================================================\n\n";

    std::cout << "--- Question ---\n";
    std::cout << "  A generator yields TrackedResources (heap objects).\n";
    std::cout << "  If the consumer breaks out of the loop early,\n";
    std::cout << "  does the generator properly clean up?\n\n";

    // Scenario 1: Full consumption
    scenario_full_consumption();

    // Scenario 2: Early break
    scenario_early_break();

    // Scenario 3: Leaky generator
    scenario_leak_demo();

    std::cout << "\n--- Lessons ---\n";
    std::cout << R"(
  1. A properly implemented Generator MUST call handle_.destroy()
     in its destructor. This destroys the coroutine frame, which
     in turn destroys:
     - The promise object (including current_value)
     - All local variables in the coroutine body
     - Any temporaries alive at the suspension point

  2. Early break from a range-for loop calls the Generator destructor
     (it's an automatic variable going out of scope). The destructor
     calls handle_.destroy(), which cleans up everything. NO LEAK.

  3. If the Generator lacks a destructor (or forgets handle_.destroy()),
     the coroutine frame is leaked. This includes:
     - The promise and its stored value
     - All coroutine-local variables
     - The coroutine frame itself (heap memory)

  4. This is the C++ RAII principle applied to coroutines:
     - The return object (Generator) owns the coroutine handle
     - The handle controls the coroutine frame lifetime
     - The Generator destructor is the cleanup point

  5. In summary:
     Generator::~Generator() → handle_.destroy() → frame destroyed
     → promise destroyed → current_value destroyed → resource freed
)" << "\n";

    std::cout << "================================================\n";
    std::cout << " Puzzle 02 complete.\n";
    std::cout << "================================================\n";

    return 0;
}
