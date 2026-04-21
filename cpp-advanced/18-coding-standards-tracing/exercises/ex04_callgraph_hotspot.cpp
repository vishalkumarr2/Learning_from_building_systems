// Exercise 4: Call-Graph Hotspot Detection
//
// This program has a performance bottleneck hidden in a call tree.
// Use profiling tools to find it WITHOUT reading the source first.
//
// TOOL PRACTICE:
//   1. valgrind --tool=callgrind ./ex04_callgraph_hotspot
//      kcachegrind callgrind.out.*    (or callgrind_annotate)
//
//   2. perf record -g ./ex04_callgraph_hotspot
//      perf report
//
//   3. Generate a flamegraph:
//      perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg
//
// TASK: Identify which function is the bottleneck, then look at the source
//       to understand WHY. The "optimized" version removes the bottleneck.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

// ========================================================================
// Simulated sensor data processing pipeline
// ========================================================================

namespace pipeline {

// Stage 1: Read sensor data (fast)
std::vector<double> read_sensors(size_t count) {
    std::vector<double> data(count);
    for (size_t i = 0; i < count; ++i)
        data[i] = std::sin(static_cast<double>(i) * 0.01) * 100.0;
    return data;
}

// Stage 2: Validate readings (fast)
std::vector<double> validate(const std::vector<double>& raw) {
    std::vector<double> valid;
    valid.reserve(raw.size());
    for (double v : raw) {
        if (std::isfinite(v) && std::abs(v) < 1000.0)
            valid.push_back(v);
    }
    return valid;
}

// Stage 3a: SLOW sort — insertion sort (O(n^2)) pretending to be a "filter"
// This is the HIDDEN BOTTLENECK
void slow_sort(std::vector<double>& data) {
    for (size_t i = 1; i < data.size(); ++i) {
        double key = data[i];
        size_t j = i;
        while (j > 0 && data[j - 1] > key) {
            data[j] = data[j - 1];
            --j;
        }
        data[j] = key;
    }
}

// Stage 3b: Fast sort — std::sort (O(n log n))
void fast_sort(std::vector<double>& data) {
    std::sort(data.begin(), data.end());
}

// Stage 4: Compute statistics (fast)
struct Stats {
    double mean = 0.0;
    double median = 0.0;
    double stddev = 0.0;
};

Stats compute_stats(const std::vector<double>& sorted_data) {
    if (sorted_data.empty()) return {};

    Stats s;
    double sum = 0.0;
    for (double v : sorted_data) sum += v;
    s.mean = sum / static_cast<double>(sorted_data.size());

    size_t mid = sorted_data.size() / 2;
    s.median = (sorted_data.size() % 2 == 0)
        ? (sorted_data[mid - 1] + sorted_data[mid]) / 2.0
        : sorted_data[mid];

    double sq_sum = 0.0;
    for (double v : sorted_data) {
        double diff = v - s.mean;
        sq_sum += diff * diff;
    }
    s.stddev = std::sqrt(sq_sum / static_cast<double>(sorted_data.size()));

    return s;
}

// Stage 5: Format output (fast)
std::string format_report(const Stats& s) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "mean=%.3f median=%.3f stddev=%.3f", s.mean, s.median, s.stddev);
    return std::string(buf);
}

// SLOW pipeline (bottleneck in stage 3a)
Stats run_slow_pipeline(size_t sensor_count) {
    auto raw = read_sensors(sensor_count);
    auto valid = validate(raw);
    slow_sort(valid);       // ← BOTTLENECK: O(n^2)
    return compute_stats(valid);
}

// FAST pipeline (fixed stage 3)
Stats run_fast_pipeline(size_t sensor_count) {
    auto raw = read_sensors(sensor_count);
    auto valid = validate(raw);
    fast_sort(valid);       // ← O(n log n)
    return compute_stats(valid);
}

void test() {
    Stats slow = run_slow_pipeline(1000);
    Stats fast = run_fast_pipeline(1000);

    // Both should produce identical results
    assert(std::abs(slow.mean - fast.mean) < 0.001);
    assert(std::abs(slow.median - fast.median) < 0.001);
    assert(std::abs(slow.stddev - fast.stddev) < 0.001);
}

void benchmark() {
    constexpr size_t kSmall = 5000;
    constexpr size_t kLarge = 20000;

    auto t0 = std::chrono::steady_clock::now();
    auto slow_stats = run_slow_pipeline(kSmall);
    auto t1 = std::chrono::steady_clock::now();
    auto fast_stats = run_fast_pipeline(kLarge);
    auto t2 = std::chrono::steady_clock::now();

    (void)slow_stats;
    (void)fast_stats;

    auto slow_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    auto fast_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::printf("  Slow pipeline (%zu elements): %ld us\n", kSmall, slow_us);
    std::printf("  Fast pipeline (%zu elements): %ld us\n", kLarge, fast_us);
    std::printf("  Fast processes %.0fx more data in less time!\n",
                static_cast<double>(kLarge) / static_cast<double>(kSmall));
}

}  // namespace pipeline

// ========================================================================
// Main
// ========================================================================

int main(int argc, char** argv) {
    pipeline::test();
    std::printf("ex04_callgraph_hotspot: ALL TESTS PASSED\n");

    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0)
            bench = true;
    }

    if (bench) {
        std::printf("\n=== BENCHMARK (profile with perf record -g or callgrind) ===\n");
        pipeline::benchmark();
    }

    return 0;
}
