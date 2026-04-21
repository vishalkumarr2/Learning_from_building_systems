// hw05_memory_hierarchy.cpp — Memory Hierarchy
// Compile: g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -pthread hw05_memory_hierarchy.cpp -o hw05_memory_hierarchy
//
// Exercises:
//  1. Cache line size via stride benchmark
//  2. L1/L2/L3 size via random access latency inflection
//  3. TLB measurement
//  4. False sharing detailed benchmark with timing
//  5. alignas(64) fix for false sharing
//  6. Memory bandwidth measurement
//  7. Comprehensive results table

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// 1. Cache line size: stride access benchmark
//    For strides 1, 2, 4, 8, ... 4096, measure ns/access.
//    When stride >= cache line, ns/access plateaus (each access = new line).
// ---------------------------------------------------------------------------
static size_t measure_cache_line_size() {
    std::printf("── 1. Cache Line Size (stride benchmark) ──\n");

    constexpr size_t kArraySize = 8 * 1024 * 1024; // 8 MiB
    auto arr = std::make_unique<char[]>(kArraySize);
    std::memset(arr.get(), 1, kArraySize);
    char sink = 0;

    struct StrideSample { size_t stride; double ns_per_access; };
    std::vector<StrideSample> samples;

    for (size_t stride = 1; stride <= 4096; stride *= 2) {
        size_t accesses = kArraySize / stride;
        if (accesses < 1000) break;

        // Warmup
        for (size_t i = 0; i < kArraySize; i += stride) sink += arr[i];

        auto t0 = Clock::now();
        for (int rep = 0; rep < 3; ++rep) {
            for (size_t i = 0; i < kArraySize; i += stride) {
                sink += arr[i];
            }
        }
        auto t1 = Clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        double ns_per = ns / (accesses * 3.0);
        samples.push_back({stride, ns_per});
    }
    (void)sink; // prevent optimization

    std::printf("  ┌──────────┬──────────────┐\n");
    std::printf("  │  Stride  │  ns/access   │\n");
    std::printf("  ├──────────┼──────────────┤\n");

    size_t detected = 64;
    double max_jump = 0;
    for (size_t i = 0; i < samples.size(); ++i) {
        std::printf("  │  %5zu   │  %8.2f    │", samples[i].stride, samples[i].ns_per_access);
        if (i > 0) {
            double ratio = samples[i].ns_per_access / samples[i - 1].ns_per_access;
            std::printf("  (%.2fx)", ratio);
            if (ratio > max_jump && ratio > 1.2) {
                max_jump = ratio;
                detected = samples[i].stride;
            }
        }
        std::printf("\n");
    }
    std::printf("  └──────────┴──────────────┘\n");
    std::printf("  → Detected cache line size: %zu bytes\n\n", detected);
    return detected;
}

// ---------------------------------------------------------------------------
// 2. L1/L2/L3 size: random-access latency in arrays of increasing size
//    Build a pointer-chase chain, measure average chase latency.
//    Inflection points indicate cache level boundaries.
// ---------------------------------------------------------------------------
static void measure_cache_sizes() {
    std::printf("── 2. L1/L2/L3 Size (pointer chase latency) ──\n");

    // Array sizes to test: 4 KiB to 128 MiB
    std::vector<size_t> sizes;
    for (size_t s = 4 * 1024; s <= 128 * 1024 * 1024; s *= 2) {
        sizes.push_back(s);
    }

    std::printf("  ┌──────────────┬───────────────┐\n");
    std::printf("  │  Array Size  │  Latency (ns) │\n");
    std::printf("  ├──────────────┼───────────────┤\n");

    std::mt19937 rng(42);

    for (size_t sz : sizes) {
        size_t count = sz / sizeof(size_t);
        auto arr = std::make_unique<size_t[]>(count);

        // Build pointer-chase chain (random permutation)
        std::vector<size_t> indices(count);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        for (size_t i = 0; i < count - 1; ++i) {
            arr[indices[i]] = indices[i + 1];
        }
        arr[indices[count - 1]] = indices[0]; // close the loop

        // Chase pointers
        constexpr int kChases = 1'000'000;
        size_t idx = 0;

        // Warmup
        for (int i = 0; i < std::min(kChases, static_cast<int>(count)); ++i) {
            idx = arr[idx];
        }

        idx = 0;
        auto t0 = Clock::now();
        for (int i = 0; i < kChases; ++i) {
            idx = arr[idx];
        }
        auto t1 = Clock::now();
        volatile size_t vsink = idx; // prevent optimization
        (void)vsink;

        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        double ns_per_chase = ns / kChases;

        const char* unit = "KiB";
        double display_sz = sz / 1024.0;
        if (sz >= 1024 * 1024) {
            unit = "MiB";
            display_sz = sz / (1024.0 * 1024.0);
        }
        std::printf("  │ %7.0f %-3s │  %9.1f    │\n", display_sz, unit, ns_per_chase);
    }

    std::printf("  └──────────────┴───────────────┘\n");
    std::printf("  → Look for latency jumps: L1→L2 (3-4x), L2→L3 (3-5x), L3→RAM (5-10x)\n\n");
}

// ---------------------------------------------------------------------------
// 3. TLB measurement: access every 4096th byte, vary array size
//    When array exceeds TLB coverage, page walks add latency.
// ---------------------------------------------------------------------------
static void measure_tlb() {
    std::printf("── 3. TLB Measurement ──\n");

    constexpr size_t kPageSize = 4096;

    std::printf("  ┌──────────────┬───────────────┐\n");
    std::printf("  │  Array Size  │  ns/page-touch│\n");
    std::printf("  ├──────────────┼───────────────┤\n");

    for (size_t sz = 64 * 1024; sz <= 256 * 1024 * 1024; sz *= 2) {
        auto arr = std::make_unique<char[]>(sz);
        std::memset(arr.get(), 0, sz); // fault in pages
        char sink = 0;

        size_t pages = sz / kPageSize;

        // Warmup
        for (size_t i = 0; i < sz; i += kPageSize) sink += arr[i];

        auto t0 = Clock::now();
        for (int rep = 0; rep < 10; ++rep) {
            for (size_t i = 0; i < sz; i += kPageSize) {
                sink += arr[i];
            }
        }
        auto t1 = Clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        (void)sink;
        double ns_per_page = ns / (pages * 10.0);

        const char* unit = "KiB";
        double dsz = sz / 1024.0;
        if (sz >= 1024 * 1024) { unit = "MiB"; dsz = sz / (1024.0 * 1024.0); }

        std::printf("  │ %7.0f %-3s │  %9.1f    │\n", dsz, unit, ns_per_page);
    }

    std::printf("  └──────────────┴───────────────┘\n");
    std::printf("  → Jump indicates TLB capacity exceeded (typically 1-2 MiB with 4K pages)\n\n");
}

// ---------------------------------------------------------------------------
// 4 & 5. False sharing: detailed benchmark + alignas fix
// ---------------------------------------------------------------------------
struct FalseShared {
    std::atomic<uint64_t> a{0};
    std::atomic<uint64_t> b{0};  // likely same cache line as a
};

struct alignas(64) PaddedAtomic {
    std::atomic<uint64_t> val{0};
};

struct NoFalseSharing {
    PaddedAtomic a;
    PaddedAtomic b;
};

template <typename T>
static double benchmark_sharing(int64_t iterations, T& data,
                                std::atomic<uint64_t> T::* field_a,
                                std::atomic<uint64_t> T::* field_b) {
    (data.*field_a).store(0);
    (data.*field_b).store(0);

    auto t0 = Clock::now();
    std::thread ta([&]() {
        for (int64_t i = 0; i < iterations; ++i)
            (data.*field_a).fetch_add(1, std::memory_order_relaxed);
    });
    std::thread tb([&]() {
        for (int64_t i = 0; i < iterations; ++i)
            (data.*field_b).fetch_add(1, std::memory_order_relaxed);
    });
    ta.join();
    tb.join();
    auto t1 = Clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double run_false_sharing_benchmark() {
    std::printf("── 4 & 5. False Sharing (with alignas fix) ──\n");

    constexpr int64_t kIters = 20'000'000;

    FalseShared fs;
    NoFalseSharing nfs;

    // Multiple runs for stability
    double fs_best = 1e18, nfs_best = 1e18;
    for (int run = 0; run < 3; ++run) {
        double fs_ms = benchmark_sharing(kIters, fs, &FalseShared::a, &FalseShared::b);
        if (fs_ms < fs_best) fs_best = fs_ms;

        // For NoFalseSharing, we need unwrapped access
        nfs.a.val.store(0); nfs.b.val.store(0);
        auto t0 = Clock::now();
        std::thread ta([&]() {
            for (int64_t i = 0; i < kIters; ++i)
                nfs.a.val.fetch_add(1, std::memory_order_relaxed);
        });
        std::thread tb([&]() {
            for (int64_t i = 0; i < kIters; ++i)
                nfs.b.val.fetch_add(1, std::memory_order_relaxed);
        });
        ta.join(); tb.join();
        auto t1 = Clock::now();
        double nfs_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (nfs_ms < nfs_best) nfs_best = nfs_ms;
    }

    double ratio = fs_best / nfs_best;

    std::printf("  struct FalseShared   sizeof=%zu  (a,b adjacent)\n", sizeof(FalseShared));
    std::printf("  struct NoFalseSharing sizeof=%zu  (a,b on separate cache lines)\n",
                sizeof(NoFalseSharing));
    std::printf("  Iterations: %ld per thread\n\n", kIters);
    std::printf("  False sharing (adjacent):    %8.2f ms\n", fs_best);
    std::printf("  No false sharing (aligned):  %8.2f ms\n", nfs_best);
    std::printf("  Penalty ratio: %.2fx\n\n", ratio);
    return ratio;
}

// ---------------------------------------------------------------------------
// 6. Memory bandwidth: sequential read throughput
// ---------------------------------------------------------------------------
static void measure_memory_bandwidth() {
    std::printf("── 6. Memory Bandwidth (sequential read) ──\n");

    struct BWSample { size_t size; double bandwidth_gb_s; double latency_ns; };
    std::vector<BWSample> results;

    std::vector<size_t> sizes = {
        32 * 1024,        // 32 KiB  (L1)
        256 * 1024,       // 256 KiB (L2)
        4 * 1024 * 1024,  // 4 MiB   (L3)
        32 * 1024 * 1024, // 32 MiB  (L3/RAM)
        128UL * 1024 * 1024, // 128 MiB (RAM)
    };

    for (size_t sz : sizes) {
        size_t count = sz / sizeof(uint64_t);
        auto arr = std::make_unique<uint64_t[]>(count);

        // Initialize to prevent lazy allocation
        for (size_t i = 0; i < count; ++i) arr[i] = i;

        uint64_t sink = 0;
        constexpr int kReps = 10;
        double best_ns = 1e18;

        for (int rep = 0; rep < kReps; ++rep) {
            auto t0 = Clock::now();
            uint64_t sum = 0;
            for (size_t i = 0; i < count; ++i) {
                sum += arr[i];
            }
            sink += sum;
            auto t1 = Clock::now();
            double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
            if (ns < best_ns) best_ns = ns;
        }

        (void)sink;
        double bandwidth = (sz / best_ns) * 1e9 / (1024.0 * 1024.0 * 1024.0); // GB/s
        double latency_per_element = best_ns / count;

        results.push_back({sz, bandwidth, latency_per_element});
    }

    std::printf("  ┌──────────────┬──────────────┬───────────────┐\n");
    std::printf("  │  Array Size  │ Bandwidth    │  ns/element   │\n");
    std::printf("  ├──────────────┼──────────────┼───────────────┤\n");
    for (const auto& r : results) {
        const char* unit = "KiB";
        double dsz = r.size / 1024.0;
        if (r.size >= 1024 * 1024) { unit = "MiB"; dsz = r.size / (1024.0 * 1024.0); }
        std::printf("  │ %7.0f %-3s │ %7.2f GB/s │  %9.2f    │\n",
                    dsz, unit, r.bandwidth_gb_s, r.latency_ns);
    }
    std::printf("  └──────────────┴──────────────┴───────────────┘\n\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::printf("╔══════════════════════════════════════════════════════╗\n");
    std::printf("║       HW05 — Memory Hierarchy                       ║\n");
    std::printf("╚══════════════════════════════════════════════════════╝\n\n");

    size_t cache_line = measure_cache_line_size();
    measure_cache_sizes();
    measure_tlb();
    double fs_ratio = run_false_sharing_benchmark();
    measure_memory_bandwidth();

    // --- Comprehensive summary ---
    std::printf("╔══════════════════════════════════════════════════════════════════╗\n");
    std::printf("║  Comprehensive Memory Hierarchy Summary                          ║\n");
    std::printf("╠══════════════════════════════════════════════════════════════════╣\n");
    std::printf("║  Detected cache line size     : %3zu bytes                        ║\n", cache_line);
    std::printf("║  False sharing penalty ratio  : %.2fx                            ║\n", fs_ratio);
    std::printf("║                                                                  ║\n");
    std::printf("║  Typical values (your mileage varies):                           ║\n");
    std::printf("║    L1d:  32-48 KiB,   ~1 ns,   ~200 GB/s                        ║\n");
    std::printf("║    L2:  256-512 KiB,  ~3-5 ns, ~80-100 GB/s                     ║\n");
    std::printf("║    L3:   6-32 MiB,   ~10-20 ns, ~40-60 GB/s                     ║\n");
    std::printf("║    RAM:  many GB,    ~60-100 ns, ~15-40 GB/s                     ║\n");
    std::printf("╚══════════════════════════════════════════════════════════════════╝\n");

    return 0;
}
