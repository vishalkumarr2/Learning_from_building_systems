// Module 12, Exercise 5: Lazy Processing Pipeline
// Compiler: GCC 12+ or Clang 14+ with -std=c++20
//
// Demonstrates:
//   - Infinite range via views::iota
//   - Filter → transform → batch (take N) → process
//   - Proof that nothing is computed until consumed
//   - Benchmark: lazy pipeline vs eager pipeline with intermediate vectors

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <ranges>
#include <vector>

namespace views = std::views;
namespace ranges = std::ranges;

// ============================================================
// Part 1: Infinite range + lazy pipeline
// ============================================================
void part1_infinite_range() {
    std::cout << "=== Part 1: Infinite Range with iota ===\n";

    // views::iota(0) produces 0, 1, 2, 3, ... forever
    // Nothing is allocated. No infinite vector in memory.

    // First 10 squares of odd numbers
    auto odd_squares = views::iota(0)
        | views::filter([](int x) { return x % 2 != 0; })
        | views::transform([](int x) { return x * x; })
        | views::take(10);

    std::cout << "First 10 odd squares: ";
    for (int x : odd_squares) {
        std::cout << x << " ";
    }
    std::cout << "\n\n";
    // Expected: 1 9 25 49 81 121 169 225 289 361
}

// ============================================================
// Part 2: Sensor data simulation — nothing computed until consumed
// ============================================================
void part2_lazy_proof() {
    std::cout << "=== Part 2: Lazy Evaluation Proof ===\n";

    int filter_calls = 0;
    int transform_calls = 0;

    // Simulated infinite sensor stream
    // iota(1) = 1, 2, 3, 4, ...  (sensor reading IDs)
    auto pipeline = views::iota(1)
        | views::filter([&filter_calls](int reading_id) {
              ++filter_calls;
              // Simulate: only readings divisible by 3 are "above threshold"
              return reading_id % 3 == 0;
          })
        | views::transform([&transform_calls](int reading_id) {
              ++transform_calls;
              // Simulate: calibration transform
              return reading_id * 1.5;
          })
        | views::take(5);

    std::cout << "Pipeline created.\n";
    std::cout << "  filter calls:    " << filter_calls << " (should be 0)\n";
    std::cout << "  transform calls: " << transform_calls << " (should be 0)\n\n";

    std::cout << "Consuming 5 elements:\n";
    for (double val : pipeline) {
        std::cout << "  → " << val << "\n";
    }

    std::cout << "\nAfter consumption:\n";
    std::cout << "  filter calls:    " << filter_calls << "\n";
    std::cout << "  transform calls: " << transform_calls << " (should be exactly 5)\n";
    // filter runs on 1..15 (to find 5 multiples of 3: 3,6,9,12,15)
    // transform runs on exactly 5 elements
    std::cout << "\n";
}

// ============================================================
// Part 3: Batched processing
// ============================================================
void part3_batched_processing() {
    std::cout << "=== Part 3: Batched Processing ===\n";

    // Simulate: process sensor data in batches of 10
    // from an infinite stream, take 3 batches

    constexpr int batch_size = 10;
    constexpr int num_batches = 3;

    auto sensor_stream = views::iota(1)
        | views::transform([](int id) {
              // Simulate noisy sensor reading
              return std::sin(static_cast<double>(id) * 0.1) * 100.0;
          });

    for (int batch = 0; batch < num_batches; ++batch) {
        auto batch_data = sensor_stream
            | views::drop(static_cast<std::size_t>(batch) * batch_size)
            | views::take(batch_size);

        // Compute batch statistics
        double sum = 0.0;
        double min_val = 1e9;
        double max_val = -1e9;
        int count = 0;

        for (double val : batch_data) {
            sum += val;
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
            ++count;
        }

        std::cout << "Batch " << batch << ": "
                  << "count=" << count
                  << ", mean=" << sum / count
                  << ", min=" << min_val
                  << ", max=" << max_val << "\n";
    }
    std::cout << "\n";
}

// ============================================================
// Part 4: Benchmark — lazy vs eager
// ============================================================

// Eager approach: create intermediate vectors at each step
std::vector<double> eager_pipeline(int n) {
    // Step 1: generate data
    std::vector<int> raw(static_cast<std::size_t>(n));
    std::iota(raw.begin(), raw.end(), 0);

    // Step 2: filter (creates new vector)
    std::vector<int> filtered;
    filtered.reserve(static_cast<std::size_t>(n) / 3);
    for (int x : raw) {
        if (x % 3 == 0) filtered.push_back(x);
    }

    // Step 3: transform (creates new vector)
    std::vector<double> transformed;
    transformed.reserve(filtered.size());
    for (int x : filtered) {
        transformed.push_back(std::sqrt(static_cast<double>(x)));
    }

    // Step 4: take first 1000
    if (transformed.size() > 1000) {
        transformed.resize(1000);
    }

    return transformed;
}

// Lazy approach: single pass, no intermediate allocations
std::vector<double> lazy_pipeline(int n) {
    auto view = views::iota(0, n)
        | views::filter([](int x) { return x % 3 == 0; })
        | views::transform([](int x) { return std::sqrt(static_cast<double>(x)); })
        | views::take(1000);

    return std::vector<double>(view.begin(), view.end());
}

void part4_benchmark() {
    std::cout << "=== Part 4: Benchmark — Lazy vs Eager ===\n";

    constexpr int N = 1'000'000;
    constexpr int RUNS = 50;

    // Warm up
    auto r1 = eager_pipeline(N);
    auto r2 = lazy_pipeline(N);

    // Verify same result
    bool same = (r1.size() == r2.size());
    if (same) {
        for (std::size_t i = 0; i < r1.size(); ++i) {
            if (std::abs(r1[i] - r2[i]) > 1e-10) { same = false; break; }
        }
    }
    std::cout << "Results match: " << (same ? "YES" : "NO") << "\n";

    // Benchmark eager
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; ++i) {
        auto result = eager_pipeline(N);
        // Prevent optimization
        if (result.empty()) std::cout << "impossible";
    }
    auto eager_time = std::chrono::high_resolution_clock::now() - start;

    // Benchmark lazy
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; ++i) {
        auto result = lazy_pipeline(N);
        if (result.empty()) std::cout << "impossible";
    }
    auto lazy_time = std::chrono::high_resolution_clock::now() - start;

    auto eager_us = std::chrono::duration_cast<std::chrono::microseconds>(eager_time).count();
    auto lazy_us  = std::chrono::duration_cast<std::chrono::microseconds>(lazy_time).count();

    std::cout << "N=" << N << ", " << RUNS << " runs each:\n";
    std::cout << "  Eager: " << eager_us / RUNS << " µs/run"
              << " (3 intermediate vectors)\n";
    std::cout << "  Lazy:  " << lazy_us / RUNS << " µs/run"
              << " (0 intermediate vectors)\n";
    std::cout << "  Speedup: " << static_cast<double>(eager_us) / static_cast<double>(lazy_us)
              << "x\n\n";

    // The lazy version should be faster because:
    // 1. No heap allocation for intermediate vectors
    // 2. Single pass through the data (better cache behavior)
    // 3. Early termination via take(1000) — doesn't process all N elements
}

int main() {
    part1_infinite_range();
    part2_lazy_proof();
    part3_batched_processing();
    part4_benchmark();
    return 0;
}
