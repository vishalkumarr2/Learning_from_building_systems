// ============================================================================
// ex04_profiling_guide.cpp — Three Intentional Performance Problems
//
// Build:
//   g++ -std=c++20 -O2 -g -o ex04_profiling ex04_profiling_guide.cpp
//
// Profile with perf:
//   perf record -g ./ex04_profiling
//   perf report
//
//   # Flame graph:
//   perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg
//
// Profile with callgrind:
//   valgrind --tool=callgrind ./ex04_profiling
//   callgrind_annotate callgrind.out.*
//   kcachegrind callgrind.out.*         # GUI
//
// Cache analysis:
//   valgrind --tool=cachegrind ./ex04_profiling
//   cg_annotate cachegrind.out.*
//
// ============================================================================

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

// Prevent compiler from optimizing away results
template <typename T>
static void do_not_optimize(T&& val) {
    asm volatile("" : "+r"(val) : : "memory");
}

// ---------------------------------------------------------------------------
// PROBLEM A: Hidden O(n²) in an innocent-looking loop
//
// This looks like it's just "inserting into a sorted position" but each
// insert into the front/middle of a vector is O(n) due to element shifting.
// Total: O(n²) for n inserts.
//
// perf will show heavy time in std::vector::insert (memmove underneath).
// ---------------------------------------------------------------------------
static std::vector<int> problem_a_quadratic_insert(std::size_t n) {
    std::vector<int> sorted;
    std::mt19937 rng(42);

    for (std::size_t i = 0; i < n; ++i) {
        int val = static_cast<int>(rng() % 1'000'000);
        // Find insertion point (binary search — O(log n))
        auto it = std::lower_bound(sorted.begin(), sorted.end(), val);
        // Insert — O(n) because elements after 'it' must shift right
        sorted.insert(it, val);
    }

    return sorted;
}

/*
 * FIXED VERSION (uncomment to compare):
 *
 * static std::vector<int> problem_a_fixed(std::size_t n) {
 *     std::vector<int> data;
 *     data.reserve(n);
 *     std::mt19937 rng(42);
 *
 *     for (std::size_t i = 0; i < n; ++i) {
 *         data.push_back(static_cast<int>(rng() % 1'000'000));
 *     }
 *
 *     // O(n log n) sort instead of O(n²) insertion
 *     std::sort(data.begin(), data.end());
 *     return data;
 * }
 */

// ---------------------------------------------------------------------------
// PROBLEM B: Cache-unfriendly random access pattern
//
// Traverses a large array in random order instead of sequential.
// Each access likely misses the L1/L2 cache.
//
// cachegrind will show very high L1 data read miss rate (~30-50%)
// compared to sequential access (~0.1%).
// ---------------------------------------------------------------------------
static int64_t problem_b_cache_unfriendly(std::size_t n) {
    // Large array that doesn't fit in L1 cache (L1 is typically 32-64 KB)
    std::vector<int64_t> data(n);
    std::iota(data.begin(), data.end(), 0);

    // Create a random permutation of indices
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0u);
    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);

    // Sum using random access — terrible for cache
    int64_t sum = 0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += data[indices[i]];
    }

    return sum;
}

/*
 * FIXED VERSION (uncomment to compare):
 *
 * static int64_t problem_b_fixed(std::size_t n) {
 *     std::vector<int64_t> data(n);
 *     std::iota(data.begin(), data.end(), 0);
 *
 *     // Sequential access — cache-friendly, hardware prefetcher helps
 *     int64_t sum = 0;
 *     for (std::size_t i = 0; i < n; ++i) {
 *         sum += data[i];
 *     }
 *     return sum;
 * }
 */

// ---------------------------------------------------------------------------
// PROBLEM C: Excessive heap allocations in a hot loop
//
// Each iteration creates and destroys a vector (heap alloc + dealloc).
// malloc/free become the bottleneck.
//
// perf will show heavy time in malloc/free/mmap.
// callgrind will show massive call counts to allocation functions.
//
// valgrind --tool=callgrind command:
//   Look for: malloc, free, operator new, operator delete
// ---------------------------------------------------------------------------
static int64_t problem_c_hot_loop_alloc(std::size_t outer, std::size_t inner) {
    int64_t total = 0;

    for (std::size_t i = 0; i < outer; ++i) {
        // PROBLEM: vector allocated and freed every iteration
        std::vector<int> temp(inner);
        for (std::size_t j = 0; j < inner; ++j) {
            temp[j] = static_cast<int>(i * inner + j);
        }
        total += std::accumulate(temp.begin(), temp.end(), int64_t{0});
    }

    return total;
}

/*
 * FIXED VERSION (uncomment to compare):
 *
 * static int64_t problem_c_fixed(std::size_t outer, std::size_t inner) {
 *     int64_t total = 0;
 *
 *     // Move allocation OUTSIDE the loop — reuse the buffer
 *     std::vector<int> temp(inner);
 *
 *     for (std::size_t i = 0; i < outer; ++i) {
 *         for (std::size_t j = 0; j < inner; ++j) {
 *             temp[j] = static_cast<int>(i * inner + j);
 *         }
 *         total += std::accumulate(temp.begin(), temp.end(), int64_t{0});
 *     }
 *
 *     return total;
 * }
 */

// ---------------------------------------------------------------------------
// Benchmark helper
// ---------------------------------------------------------------------------
template <typename Func>
static void benchmark(const char* name, Func&& func) {
    auto start = std::chrono::steady_clock::now();
    auto result = func();
    auto end = std::chrono::steady_clock::now();

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("  %-42s %6ld ms  (result: %ld)\n", name, ms.count(),
           static_cast<long>(result));
    do_not_optimize(result);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    printf("=== ex04: Profiling Guide — Programs with Intentional Performance Problems ===\n\n");

    // Sizes chosen to make each problem take ~0.5-3 seconds
    constexpr std::size_t kProblemASize = 50'000;
    constexpr std::size_t kProblemBSize = 4'000'000;   // ~32 MB
    constexpr std::size_t kProblemCOuter = 10'000;
    constexpr std::size_t kProblemCInner = 1'000;

    printf("Problem A: O(n²) sorted insert (n=%zu)\n", kProblemASize);
    benchmark("  Quadratic insert", [&]() -> int64_t {
        auto v = problem_a_quadratic_insert(kProblemASize);
        return static_cast<int64_t>(v.size());
    });

    printf("\nProblem B: Cache-unfriendly random access (n=%zu)\n", kProblemBSize);
    benchmark("  Random access sum", [&]() {
        return problem_b_cache_unfriendly(kProblemBSize);
    });

    printf("\nProblem C: Hot-loop allocation (%zux%zu)\n", kProblemCOuter, kProblemCInner);
    benchmark("  Alloc-per-iteration", [&]() {
        return problem_c_hot_loop_alloc(kProblemCOuter, kProblemCInner);
    });

    printf("\n");
    printf("=== Profiling Commands ===\n");
    printf("  perf record -g ./ex04_profiling_guide\n");
    printf("  perf report\n");
    printf("  perf script | stackcollapse-perf.pl | flamegraph.pl > flame.svg\n");
    printf("  valgrind --tool=callgrind ./ex04_profiling_guide\n");
    printf("  valgrind --tool=cachegrind ./ex04_profiling_guide\n");
    printf("\n");
    printf("After profiling, uncomment the FIXED versions and compare.\n");

    return 0;
}
