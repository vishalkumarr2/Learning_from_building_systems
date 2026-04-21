// Puzzle 3: The Lying Benchmark
//
// This puzzle demonstrates common benchmarking mistakes that produce
// misleading results. Each "trap" shows a micro-benchmark that lies,
// and the corrected version that measures what you actually want.
//
// TRAPS:
//   1. Dead code elimination — compiler optimizes away the computation
//   2. Cold cache vs warm cache — first run is always slower
//   3. Branch predictor warming — loops become "free" after training
//   4. Measurement overhead — timing granularity hides real costs
//
// TASK: Run each trap, predict the output, then explain why it lies.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

namespace lying_bench {

// ========================================================================
// Trap 1: Dead Code Elimination
// ========================================================================

namespace trap1_dce {

// BAD benchmark: compiler sees result is unused → optimizes away entirely
void bad_benchmark() {
    constexpr size_t kN = 10'000'000;
    auto t0 = std::chrono::steady_clock::now();
    double sum = 0.0;
    for (size_t i = 0; i < kN; ++i) {
        sum += std::sqrt(static_cast<double>(i));
    }
    auto t1 = std::chrono::steady_clock::now();
    // sum is never used! Compiler may delete the entire loop.
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    std::printf("  [TRAP] sqrt loop:  %ld ns (likely 0 — DCE!)\n", ns);
}

// GOOD benchmark: prevent DCE with volatile sink
void good_benchmark() {
    constexpr size_t kN = 10'000'000;
    volatile double sink = 0.0;
    auto t0 = std::chrono::steady_clock::now();
    double sum = 0.0;
    for (size_t i = 0; i < kN; ++i) {
        sum += std::sqrt(static_cast<double>(i));
    }
    sink = sum;  // Force the compiler to keep the computation
    auto t1 = std::chrono::steady_clock::now();
    (void)sink;
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    std::printf("  [REAL] sqrt loop:  %ld ns (actual work)\n", ns);
}

}  // namespace trap1_dce

// ========================================================================
// Trap 2: Cold Cache vs Warm Cache
// ========================================================================

namespace trap2_cache {

void benchmark_traversal(const char* label, const std::vector<double>& data,
                         int runs) {
    volatile double sink = 0.0;
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < runs; ++r) {
        double sum = 0.0;
        for (double v : data) sum += v;
        sink = sum;
    }
    auto t1 = std::chrono::steady_clock::now();
    (void)sink;
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    std::printf("  %-24s %6ld us (%d runs)\n", label, us, runs);
}

void demonstrate() {
    constexpr size_t kSize = 1'000'000;
    std::vector<double> data(kSize);
    std::iota(data.begin(), data.end(), 0.0);

    std::printf("\n  Cold vs Warm cache (1M doubles):\n");
    // First run: data might not be in cache
    benchmark_traversal("[cold] single run", data, 1);
    // After first run: data is now in cache
    benchmark_traversal("[warm] single run", data, 1);
    // Proper: average over many runs
    benchmark_traversal("[proper] 100 runs avg", data, 100);
}

}  // namespace trap2_cache

// ========================================================================
// Trap 3: Branch Predictor Training
// ========================================================================

namespace trap3_branch {

// With sorted data: branch predictor trains perfectly → fast
// With random data: branch mispredictions → slow
void demonstrate() {
    constexpr size_t kSize = 100'000;
    constexpr int kRuns = 50;

    std::vector<int> data(kSize);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& v : data) v = dist(rng);

    auto count_above_threshold = [](const std::vector<int>& d, int threshold) {
        int count = 0;
        for (int v : d) {
            if (v > threshold) ++count;  // Branch
        }
        return count;
    };

    volatile int sink = 0;

    // Sorted: branch predictor learns the pattern (all-no then all-yes)
    std::vector<int> sorted = data;
    std::sort(sorted.begin(), sorted.end());

    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < kRuns; ++r) sink = count_above_threshold(sorted, 128);
    auto t1 = std::chrono::steady_clock::now();

    // Unsorted: branch predictor can't predict → pipeline stalls
    auto t2 = std::chrono::steady_clock::now();
    for (int r = 0; r < kRuns; ++r) sink = count_above_threshold(data, 128);
    auto t3 = std::chrono::steady_clock::now();

    (void)sink;

    auto sorted_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    auto random_us = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count();

    std::printf("\n  Branch prediction effect (%zu elements, threshold=128):\n", kSize);
    std::printf("    Sorted data:   %6ld us (%d runs) — predictor trained\n",
                sorted_us, kRuns);
    std::printf("    Random data:   %6ld us (%d runs) — mispredictions\n",
                random_us, kRuns);

    long sorted_safe = (sorted_us == 0) ? 1 : sorted_us;
    std::printf("    Ratio:         %.1fx slower with random\n",
                static_cast<double>(random_us) / static_cast<double>(sorted_safe));

    // Branchless version for comparison
    auto count_branchless = [](const std::vector<int>& d, int threshold) {
        int count = 0;
        for (int v : d) {
            count += (v > threshold) ? 1 : 0;  // Often compiled branchless
        }
        return count;
    };

    auto t4 = std::chrono::steady_clock::now();
    for (int r = 0; r < kRuns; ++r) sink = count_branchless(data, 128);
    auto t5 = std::chrono::steady_clock::now();

    (void)sink;

    auto branchless_us = std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count();
    std::printf("    Branchless:    %6ld us (%d runs) — no mispredictions\n",
                branchless_us, kRuns);
}

}  // namespace trap3_branch

// ========================================================================
// Trap 4: Measurement Granularity
// ========================================================================

namespace trap4_granularity {

void demonstrate() {
    // Measuring something that takes < 1 nanosecond per operation
    constexpr size_t kN = 1000;
    volatile int sink = 0;

    // BAD: timing one operation — noise dominates
    std::printf("\n  Measurement granularity:\n");
    int single_times[10];
    for (int trial = 0; trial < 10; ++trial) {
        auto t0 = std::chrono::steady_clock::now();
        sink = 42 + trial;
        auto t1 = std::chrono::steady_clock::now();
        single_times[trial] = static_cast<int>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }
    (void)sink;
    std::printf("    [TRAP] 10 single-op timings: ");
    for (int i = 0; i < 10; ++i) std::printf("%d ", single_times[i]);
    std::printf("ns\n    (Notice: noise > signal!)\n");

    // GOOD: batch N operations, divide
    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < kN; ++i) {
        sink = static_cast<int>(i) * 3 + 7;
    }
    auto t1 = std::chrono::steady_clock::now();
    (void)sink;
    auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    std::printf("    [GOOD] %zu ops in %ld ns = %.1f ns/op\n",
                kN, total_ns, static_cast<double>(total_ns) / static_cast<double>(kN));
}

}  // namespace trap4_granularity

// ========================================================================
// Self-tests (verify correctness, not performance)
// ========================================================================

void test() {
    // Trap 1: both should compute the same sum
    constexpr size_t kSmall = 100;
    double sum = 0.0;
    for (size_t i = 0; i < kSmall; ++i)
        sum += std::sqrt(static_cast<double>(i));
    assert(sum > 0.0);

    // Trap 3: sorted and unsorted should give same count
    std::vector<int> v = {1, 200, 50, 150, 100, 250, 30};
    int count = 0;
    for (int x : v) {
        if (x > 128) ++count;
    }
    assert(count == 3);  // 200, 150, 250

    std::sort(v.begin(), v.end());
    int sorted_count = 0;
    for (int x : v) {
        if (x > 128) ++sorted_count;
    }
    assert(sorted_count == count);
}

}  // namespace lying_bench

// ========================================================================
// Main
// ========================================================================

int main(int argc, char** argv) {
    lying_bench::test();
    std::printf("puzzle03_lying_benchmark: ALL TESTS PASSED\n");

    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0)
            bench = true;
    }

    if (bench) {
        std::printf("\n=== LYING BENCHMARK DEMONSTRATIONS ===\n");

        std::printf("\n[Trap 1: Dead Code Elimination]\n");
        lying_bench::trap1_dce::bad_benchmark();
        lying_bench::trap1_dce::good_benchmark();

        std::printf("\n[Trap 2: Cold vs Warm Cache]\n");
        lying_bench::trap2_cache::demonstrate();

        std::printf("\n[Trap 3: Branch Predictor Training]\n");
        lying_bench::trap3_branch::demonstrate();

        std::printf("\n[Trap 4: Measurement Granularity]\n");
        lying_bench::trap4_granularity::demonstrate();
    }

    return 0;
}
