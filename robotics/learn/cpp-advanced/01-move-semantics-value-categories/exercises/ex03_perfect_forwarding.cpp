// =============================================================================
// Exercise 3: Perfect Forwarding — RingBuffer<T, N> with try_emplace
// =============================================================================
// Demonstrates:
//   1. Perfect forwarding to construct objects in-place
//   2. Factory function make<T>(args...)
//   3. auto&& in range-for (forwarding reference binding)
//   4. Timing comparisons: copy vs forward
//
// Build: g++ -std=c++2a -O2 -Wall -Wextra -Wpedantic ex03_perfect_forwarding.cpp
// =============================================================================

#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// =============================================================================
// Part 1: RingBuffer with try_emplace using perfect forwarding
// =============================================================================

template<typename T, std::size_t N>
class RingBuffer {
    // Use aligned storage to avoid default-constructing all N elements
    alignas(T) std::byte storage_[N * sizeof(T)];
    std::size_t head_ = 0;  // next write position
    std::size_t tail_ = 0;  // next read position
    std::size_t count_ = 0;

    T* slot(std::size_t idx) noexcept {
        return std::launder(reinterpret_cast<T*>(&storage_[idx * sizeof(T)]));
    }
    const T* slot(std::size_t idx) const noexcept {
        return std::launder(reinterpret_cast<const T*>(&storage_[idx * sizeof(T)]));
    }

public:
    RingBuffer() = default;

    ~RingBuffer() {
        while (count_ > 0) {
            slot(tail_)->~T();
            tail_ = (tail_ + 1) % N;
            --count_;
        }
    }

    // Non-copyable (complex to implement correctly for manual-lifetime storage)
    RingBuffer(const RingBuffer&) = delete;
    RingBuffer& operator=(const RingBuffer&) = delete;

    [[nodiscard]] bool empty() const noexcept { return count_ == 0; }
    [[nodiscard]] bool full() const noexcept { return count_ == N; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] static constexpr std::size_t capacity() noexcept { return N; }

    // -------------------------------------------------------------------------
    // try_emplace: construct T in-place using perfect forwarding
    // Returns true if successful, false if the buffer is full.
    // -------------------------------------------------------------------------
    template<typename... Args>
    bool try_emplace(Args&&... args) {
        if (full()) return false;

        // Construct in-place using placement new + perfect forwarding
        // std::forward preserves the value category of each argument
        ::new (static_cast<void*>(slot(head_))) T(std::forward<Args>(args)...);

        head_ = (head_ + 1) % N;
        ++count_;
        return true;
    }

    // -------------------------------------------------------------------------
    // try_push: copy/move a value in (contrast with try_emplace)
    // -------------------------------------------------------------------------
    bool try_push(const T& val) {
        return try_emplace(val);  // forwards as const T& → copy ctor
    }

    bool try_push(T&& val) {
        return try_emplace(std::move(val));  // forwards as T&& → move ctor
    }

    // -------------------------------------------------------------------------
    // try_pop: move the front element out and return it
    // -------------------------------------------------------------------------
    [[nodiscard]] bool try_pop(T& out) {
        if (empty()) return false;

        out = std::move(*slot(tail_));
        slot(tail_)->~T();
        tail_ = (tail_ + 1) % N;
        --count_;
        return true;
    }

    // Peek at front
    [[nodiscard]] const T& front() const {
        assert(!empty());
        return *slot(tail_);
    }
};

// =============================================================================
// Part 2: Factory function with perfect forwarding
// =============================================================================

// Generic factory: perfect-forwards to any constructor
template<typename T, typename... Args>
std::unique_ptr<T> make(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// Test class to verify forwarding behavior
struct Verbose {
    std::string name;
    int value;

    Verbose(const std::string& n, int v) : name(n), value(v) {
        std::cout << "  Verbose(const string&, int): copied name \"" << name << "\"\n";
    }

    Verbose(std::string&& n, int v) : name(std::move(n)), value(v) {
        std::cout << "  Verbose(string&&, int): moved name \"" << name << "\"\n";
    }
};

// =============================================================================
// Part 3: auto&& in range-for
// =============================================================================

// Returns a vector by value — elements are rvalues of the temporary
std::vector<std::string> make_strings() {
    return {"alpha", "bravo", "charlie"};
}

void demo_auto_rref_range_for() {
    std::cout << "\n--- auto&& in range-for ---\n";

    std::vector<std::string> v = {"one", "two", "three"};

    // auto&& binds to lvalue elements (v is an lvalue, so elements are lvalues)
    std::cout << "Iterating lvalue vector with auto&&:\n";
    for (auto&& elem : v) {
        // elem is std::string& (lvalue reference)
        static_assert(std::is_lvalue_reference_v<decltype(elem)>,
            "auto&& on lvalue container yields lvalue ref");
        std::cout << "  " << elem << " (lvalue ref)\n";
    }

    // auto&& binds to rvalue elements (temporary vector)
    std::cout << "Iterating temporary vector with auto&&:\n";
    for (auto&& elem : make_strings()) {
        // elem is std::string& because the temporary is materialized and
        // range-for binds to it; elements of the materialized object are lvalues
        std::cout << "  " << elem << "\n";
    }

    // Contrast: T&& would NOT compile with an lvalue container
    // for (std::string&& elem : v) { }  // ERROR: can't bind rvalue ref to lvalue
}

// =============================================================================
// Part 4: Timing — copy vs forward
// =============================================================================

struct HeavyObject {
    std::string data;
    HeavyObject() : data(10000, 'x') {}
    HeavyObject(const std::string& d) : data(d) {}
    HeavyObject(std::string&& d) : data(std::move(d)) {}
    HeavyObject(const HeavyObject&) = default;
    HeavyObject(HeavyObject&&) noexcept = default;
    HeavyObject& operator=(const HeavyObject&) = default;
    HeavyObject& operator=(HeavyObject&&) noexcept = default;
};

void benchmark_copy_vs_forward() {
    std::cout << "\n--- Benchmark: copy vs emplace with forward ---\n";

    constexpr int ITERS = 100'000;

    // Benchmark 1: push via copy
    {
        RingBuffer<HeavyObject, 16> rb;
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < ITERS; ++i) {
            HeavyObject obj;
            rb.try_push(obj);  // COPY — obj is an lvalue
            HeavyObject out;
            (void)rb.try_pop(out);
        }
        auto elapsed = std::chrono::steady_clock::now() - start;
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        std::cout << "  Copy push:    " << us << " us (" << ITERS << " iterations)\n";
    }

    // Benchmark 2: push via move
    {
        RingBuffer<HeavyObject, 16> rb;
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < ITERS; ++i) {
            HeavyObject obj;
            rb.try_push(std::move(obj));  // MOVE — explicit consent
            HeavyObject out;
            (void)rb.try_pop(out);
        }
        auto elapsed = std::chrono::steady_clock::now() - start;
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        std::cout << "  Move push:    " << us << " us (" << ITERS << " iterations)\n";
    }

    // Benchmark 3: emplace (construct in-place — no copy OR move of the string)
    {
        RingBuffer<HeavyObject, 16> rb;
        std::string template_data(10000, 'x');
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < ITERS; ++i) {
            std::string tmp = template_data;  // must create the string somewhere
            rb.try_emplace(std::move(tmp));   // forward into constructor
            HeavyObject out;
            (void)rb.try_pop(out);
        }
        auto elapsed = std::chrono::steady_clock::now() - start;
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        std::cout << "  Emplace fwd:  " << us << " us (" << ITERS << " iterations)\n";
    }
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "=== Perfect Forwarding Exercises ===\n";

    // --- Part 1: RingBuffer ---
    std::cout << "\n--- RingBuffer<std::string, 4> ---\n";
    {
        RingBuffer<std::string, 4> rb;
        assert(rb.empty());
        assert(rb.capacity() == 4);

        // try_emplace constructs in-place — no string copy needed
        assert(rb.try_emplace("hello"));   // from const char* → string(const char*)
        assert(rb.try_emplace(5, 'x'));    // string(5, 'x') = "xxxxx"
        assert(rb.size() == 2);

        std::string val;
        assert(rb.try_pop(val));
        assert(val == "hello");
        assert(rb.try_pop(val));
        assert(val == "xxxxx");
        assert(rb.empty());

        // Fill to capacity
        for (int i = 0; i < 4; ++i)
            assert(rb.try_emplace("item_" + std::to_string(i)));
        assert(rb.full());
        assert(!rb.try_emplace("overflow"));  // should fail

        std::cout << "  RingBuffer tests passed!\n";
    }

    // --- Part 2: Factory function ---
    std::cout << "\n--- make<T>() factory ---\n";
    {
        std::string name = "sensor_imu";

        // This should COPY the string (name is lvalue)
        auto v1 = make<Verbose>(name, 100);
        assert(v1->name == "sensor_imu");

        // This should MOVE the string
        auto v2 = make<Verbose>(std::move(name), 200);
        assert(v2->name == "sensor_imu");
        // name is now in a valid but unspecified state

        std::cout << "  Factory tests passed!\n";
    }

    // --- Part 3: auto&& ---
    demo_auto_rref_range_for();

    // --- Part 4: Benchmark ---
    benchmark_copy_vs_forward();

    std::cout << "\nAll exercises complete!\n";
    return 0;
}
