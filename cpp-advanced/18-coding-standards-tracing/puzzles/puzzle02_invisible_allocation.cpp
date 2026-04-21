// Puzzle 2: Invisible Allocations in Real-Time Code
//
// Real-time systems (robotics, audio, trading) must avoid dynamic
// memory allocation in the hot path because malloc/free are unbounded.
//
// This puzzle shows code that LOOKS allocation-free but secretly
// allocates through:
//   - std::string concatenation
//   - std::vector growth
//   - Exception throwing
//   - std::function capture
//   - std::map insertion
//
// TASK:
//   1. Run with `ltrace -e malloc ./puzzle02_invisible_allocation`
//      to count allocations per function
//   2. Compare bad_ vs good_ versions
//   3. Use: valgrind --tool=massif ./puzzle02_invisible_allocation --bench
//
// Each test demonstrates one hidden allocation pattern and its fix.

#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <functional>
#include <vector>

namespace rt_alloc {

// ========================================================================
// Trap 1: String concatenation in logging
// ========================================================================

namespace trap1_string {

// BAD: builds a new string every call
std::string bad_format_status(int sensor_id, double value) {
    return "sensor_" + std::to_string(sensor_id) + ": " + std::to_string(value);
    // 3-4 heap allocations per call!
}

// GOOD: write into pre-allocated buffer
struct StatusBuffer {
    char data[128];
    size_t len = 0;

    void format(int sensor_id, double value) {
        len = static_cast<size_t>(
            std::snprintf(data, sizeof(data), "sensor_%d: %.6f", sensor_id, value));
    }

    const char* c_str() const { return data; }
};

void test() {
    std::string bad = bad_format_status(42, 3.14);
    assert(bad.find("sensor_42") != std::string::npos);

    StatusBuffer good;
    good.format(42, 3.14);
    assert(std::strstr(good.c_str(), "sensor_42") != nullptr);
}

}  // namespace trap1_string

// ========================================================================
// Trap 2: Vector push_back without reserve
// ========================================================================

namespace trap2_vector {

// BAD: vector grows and reallocates multiple times
std::vector<double> bad_collect_readings(size_t count) {
    std::vector<double> readings;
    for (size_t i = 0; i < count; ++i) {
        readings.push_back(static_cast<double>(i) * 0.1);
        // Allocates at size 1, 2, 4, 8, 16, 32, 64, 128, ...
    }
    return readings;
}

// GOOD: pre-allocated fixed buffer
template <size_t N>
class FixedBuffer {
    std::array<double, N> data_{};
    size_t count_ = 0;

public:
    bool push(double val) {
        if (count_ >= N) return false;
        data_[count_++] = val;
        return true;
    }

    size_t size() const { return count_; }
    double operator[](size_t i) const { return data_[i]; }
};

void test() {
    auto bad = bad_collect_readings(100);
    assert(bad.size() == 100);

    FixedBuffer<100> good;
    for (size_t i = 0; i < 100; ++i) {
        bool ok = good.push(static_cast<double>(i) * 0.1);
        assert(ok);
    }
    assert(good.size() == 100);

    // Values should match
    for (size_t i = 0; i < 100; ++i)
        assert(std::abs(bad[i] - good[i]) < 0.001);
}

}  // namespace trap2_vector

// ========================================================================
// Trap 3: std::function with capturing lambda (heap-allocates captures)
// ========================================================================

namespace trap3_function {

using Callback = std::function<double(double)>;

// BAD: std::function allocates when capture exceeds SBO (small buffer optimization)
// Most implementations have SBO of ~16-32 bytes
Callback bad_make_filter(double alpha, double beta, double gamma,
                         double delta, double epsilon) {
    // Captures 5 doubles = 40 bytes — exceeds typical SBO
    return [=](double input) {
        return input * alpha + beta * gamma - delta + epsilon;
    };
}

// GOOD: functor with no heap allocation
struct Filter {
    double alpha, beta, gamma, delta, epsilon;

    double operator()(double input) const {
        return input * alpha + beta * gamma - delta + epsilon;
    }
};

void test() {
    auto bad_fn = bad_make_filter(1.0, 2.0, 3.0, 4.0, 5.0);
    double bad_result = bad_fn(10.0);

    Filter good_fn{1.0, 2.0, 3.0, 4.0, 5.0};
    double good_result = good_fn(10.0);

    assert(std::abs(bad_result - good_result) < 0.001);
}

}  // namespace trap3_function

// ========================================================================
// Trap 4: Error handling with exceptions (allocates on throw)
// ========================================================================

namespace trap4_exception {

// BAD: exceptions allocate on the heap when thrown
double bad_safe_divide(double a, double b) {
    if (std::abs(b) < 1e-10) {
        throw std::runtime_error("division by zero");
        // std::runtime_error allocates a string on the heap
    }
    return a / b;
}

// GOOD: error code return — zero allocations
struct DivResult {
    double value;
    bool ok;
};

DivResult good_safe_divide(double a, double b) {
    if (std::abs(b) < 1e-10) {
        return {0.0, false};
    }
    return {a / b, true};
}

void test() {
    // Normal case
    double bad_result = bad_safe_divide(10.0, 2.0);
    auto good_result = good_safe_divide(10.0, 2.0);
    assert(good_result.ok);
    assert(std::abs(bad_result - good_result.value) < 0.001);

    // Error case — bad version throws
    bool caught = false;
    try {
        bad_safe_divide(1.0, 0.0);
    } catch (const std::runtime_error&) {
        caught = true;
    }
    assert(caught);

    // Good version returns error
    auto err = good_safe_divide(1.0, 0.0);
    assert(!err.ok);
}

}  // namespace trap4_exception

// ========================================================================
// Benchmark
// ========================================================================

void benchmark() {
    constexpr size_t kIter = 100000;
    volatile double sink = 0.0;

    auto time_it = [](const char* name, size_t iters, auto fn) {
        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < iters; ++i) fn(i);
        auto t1 = std::chrono::steady_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        std::printf("  %-30s %6ld us (%zu iters)\n", name, us, iters);
    };

    std::printf("\n[Trap 1: String formatting]\n");
    time_it("bad (string concat)", kIter,
            [&](size_t i) { sink += static_cast<double>(
                rt_alloc::trap1_string::bad_format_status(
                    static_cast<int>(i), 3.14).size()); });
    rt_alloc::trap1_string::StatusBuffer buf;
    time_it("good (snprintf buffer)", kIter,
            [&](size_t i) { buf.format(static_cast<int>(i), 3.14);
                sink += static_cast<double>(buf.len); });

    std::printf("\n[Trap 2: Vector growth]\n");
    time_it("bad (push_back no reserve)", 100,
            [&](size_t) { auto v = rt_alloc::trap2_vector::bad_collect_readings(1000);
                sink += static_cast<double>(v.size()); });
    time_it("good (fixed buffer)", 100,
            [&](size_t) { rt_alloc::trap2_vector::FixedBuffer<1000> fb;
                for (size_t j = 0; j < 1000; ++j) fb.push(static_cast<double>(j));
                sink += static_cast<double>(fb.size()); });

    std::printf("\n[Trap 3: std::function vs functor]\n");
    time_it("bad (std::function)", kIter,
            [&](size_t) { auto fn = rt_alloc::trap3_function::bad_make_filter(
                1.0, 2.0, 3.0, 4.0, 5.0); sink += fn(1.0); });
    rt_alloc::trap3_function::Filter filt{1.0, 2.0, 3.0, 4.0, 5.0};
    time_it("good (functor)", kIter,
            [&](size_t) { sink += filt(1.0); });

    (void)sink;
}

}  // namespace rt_alloc

// ========================================================================
// Main
// ========================================================================

int main(int argc, char** argv) {
    rt_alloc::trap1_string::test();
    rt_alloc::trap2_vector::test();
    rt_alloc::trap3_function::test();
    rt_alloc::trap4_exception::test();

    std::printf("puzzle02_invisible_allocation: ALL TESTS PASSED\n");

    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0)
            bench = true;
    }

    if (bench) {
        std::printf("\n=== ALLOCATION BENCHMARK ===\n");
        std::printf("Run with: ltrace -e malloc ./puzzle02_invisible_allocation --bench\n");
        rt_alloc::benchmark();
    }

    return 0;
}
