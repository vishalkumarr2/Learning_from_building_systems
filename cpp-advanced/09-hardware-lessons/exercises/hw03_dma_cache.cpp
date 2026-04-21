// hw03_dma_cache.cpp — DMA & Cache Coherency
// Compile: g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -pthread hw03_dma_cache.cpp -o hw03_dma_cache
//
// Exercises:
//  1. Cache line effect: sequential vs strided access
//  2. Measure cache line size experimentally
//  3. False sharing benchmark
//  4. DMA double-buffer simulation
//  5. Memory barrier demonstration

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <numeric>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// Helper: benchmark a function, return nanoseconds
// ---------------------------------------------------------------------------
static double bench_ns(std::function<void()> fn, int repeats = 5) {
    double best = 1e18;
    for (int r = 0; r < repeats; ++r) {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
        if (ns < best) best = ns;
    }
    return best;
}

// ---------------------------------------------------------------------------
// 1. Sequential vs strided access
// ---------------------------------------------------------------------------
static void benchmark_access_patterns() {
    std::printf("── 1. Sequential vs Strided Access ──\n");

    constexpr size_t kSize = 16 * 1024 * 1024; // 16 MiB
    auto buf = std::make_unique<uint8_t[]>(kSize);
    std::memset(buf.get(), 1, kSize);

    volatile uint64_t sink = 0;

    // Sequential
    double seq_ns = bench_ns([&]() {
        uint64_t sum = 0;
        for (size_t i = 0; i < kSize; ++i) {
            sum += buf[i];
        }
        sink = sum;
    });

    // Strided (every 64th byte — one per cache line)
    double stride64_ns = bench_ns([&]() {
        uint64_t sum = 0;
        for (size_t i = 0; i < kSize; i += 64) {
            sum += buf[i];
        }
        sink = sum;
    });

    // Strided (every 4096th byte — one per page)
    double stride4k_ns = bench_ns([&]() {
        uint64_t sum = 0;
        for (size_t i = 0; i < kSize; i += 4096) {
            sum += buf[i];
        }
        sink = sum;
    });

    std::printf("  Array size: %zu bytes (%.1f MiB)\n", kSize, kSize / (1024.0 * 1024.0));
    std::printf("  Sequential (stride 1):     %10.1f ns  (%8.2f ns/element)\n",
                seq_ns, seq_ns / kSize);
    std::printf("  Stride 64 (1/cacheline):   %10.1f ns  (%8.2f ns/access)\n",
                stride64_ns, stride64_ns / (kSize / 64));
    std::printf("  Stride 4096 (1/page):      %10.1f ns  (%8.2f ns/access)\n",
                stride4k_ns, stride4k_ns / (kSize / 4096));
    std::printf("  Ratio (stride64/seq per byte): %.2fx\n\n",
                (stride64_ns / (kSize / 64)) / (seq_ns / kSize));
}

// ---------------------------------------------------------------------------
// 2. Measure cache line size experimentally
//    Technique: for stride S from 1..512, time summing every S-th element.
//    When stride >= cache line size, each access is a new cache line → time/access jumps.
// ---------------------------------------------------------------------------
static size_t measure_cache_line_size() {
    std::printf("── 2. Cache Line Size Measurement ──\n");

    constexpr size_t kSize = 4 * 1024 * 1024;
    auto buf = std::make_unique<uint8_t[]>(kSize);
    std::memset(buf.get(), 1, kSize);
    volatile uint64_t sink = 0;

    struct Result { size_t stride; double ns_per_access; };
    std::vector<Result> results;

    for (size_t stride = 4; stride <= 512; stride *= 2) {
        double ns = bench_ns([&]() {
            uint64_t sum = 0;
            for (size_t i = 0; i < kSize; i += stride) {
                sum += buf[i];
            }
            sink = sum;
        });
        size_t accesses = kSize / stride;
        double ns_per = ns / accesses;
        results.push_back({stride, ns_per});
    }

    std::printf("  ┌─────────┬───────────────┐\n");
    std::printf("  │ Stride  │ ns/access     │\n");
    std::printf("  ├─────────┼───────────────┤\n");

    size_t detected_line_size = 64; // fallback
    double max_jump_ratio = 0;

    for (size_t i = 0; i < results.size(); ++i) {
        std::printf("  │ %5zu   │ %9.2f     │", results[i].stride, results[i].ns_per_access);
        if (i > 0) {
            double ratio = results[i].ns_per_access / results[i - 1].ns_per_access;
            if (ratio > max_jump_ratio && ratio > 1.3) {
                max_jump_ratio = ratio;
                detected_line_size = results[i].stride;
            }
            std::printf("  (%.2fx)", ratio);
        }
        std::printf("\n");
    }
    std::printf("  └─────────┴───────────────┘\n");
    std::printf("  Detected cache line size: %zu bytes\n\n", detected_line_size);
    return detected_line_size;
}

// ---------------------------------------------------------------------------
// 3. False sharing benchmark
//    Two threads incrementing:
//    a) adjacent counters (same cache line)
//    b) separated counters (different cache lines)
// ---------------------------------------------------------------------------
struct FalseSharePair {
    alignas(64) std::atomic<uint64_t> a{0};
    std::atomic<uint64_t> b{0};  // adjacent — same or next cache line
};

struct NoFalseSharePair {
    alignas(64) std::atomic<uint64_t> a{0};
    alignas(64) std::atomic<uint64_t> b{0};  // guaranteed separate cache lines
};

template <typename Pair>
static double false_sharing_test(int64_t iterations) {
    Pair p{};
    auto t0 = Clock::now();
    std::thread t1([&]() {
        for (int64_t i = 0; i < iterations; ++i) {
            p.a.fetch_add(1, std::memory_order_relaxed);
        }
    });
    std::thread t2([&]() {
        for (int64_t i = 0; i < iterations; ++i) {
            p.b.fetch_add(1, std::memory_order_relaxed);
        }
    });
    t1.join();
    t2.join();
    auto t1_end = Clock::now();
    return std::chrono::duration<double, std::milli>(t1_end - t0).count();
}

static double run_false_sharing_benchmark() {
    std::printf("── 3. False Sharing Benchmark ──\n");

    constexpr int64_t kIters = 10'000'000;

    double shared_ms = false_sharing_test<FalseSharePair>(kIters);
    double separated_ms = false_sharing_test<NoFalseSharePair>(kIters);
    double ratio = shared_ms / separated_ms;

    std::printf("  Iterations per thread: %ld\n", kIters);
    std::printf("  Adjacent (false sharing):  %8.2f ms\n", shared_ms);
    std::printf("  Separated (no sharing):    %8.2f ms\n", separated_ms);
    std::printf("  Penalty ratio: %.2fx\n\n", ratio);
    return ratio;
}

// ---------------------------------------------------------------------------
// 4. DMA double-buffer simulation
//    Two buffers: "DMA" fills one while CPU processes the other.
// ---------------------------------------------------------------------------
static double run_double_buffer_simulation() {
    std::printf("── 4. DMA Double-Buffer Simulation ──\n");

    constexpr size_t kBufSize = 1024 * 1024;  // 1 MiB per buffer
    constexpr int kTransfers = 20;

    auto buf_a = std::make_unique<uint32_t[]>(kBufSize);
    auto buf_b = std::make_unique<uint32_t[]>(kBufSize);

    uint32_t* buffers[2] = {buf_a.get(), buf_b.get()};
    uint64_t cpu_sink = 0;

    // Simulate DMA fill (memset represents DMA writing)
    auto dma_fill = [](uint32_t* buf, size_t count, uint32_t val) {
        for (size_t i = 0; i < count; ++i) buf[i] = val;
    };

    // Simulate CPU processing (sum)
    auto cpu_process = [](const uint32_t* buf, size_t count) -> uint64_t {
        uint64_t sum = 0;
        for (size_t i = 0; i < count; ++i) sum += buf[i];
        return sum;
    };

    // --- Single buffer baseline ---
    auto t0 = Clock::now();
    for (int i = 0; i < kTransfers; ++i) {
        dma_fill(buffers[0], kBufSize, static_cast<uint32_t>(i));
        // Memory fence to simulate cache invalidation
        std::atomic_thread_fence(std::memory_order_acquire);
        cpu_sink += cpu_process(buffers[0], kBufSize);
    }
    auto t1 = Clock::now();
    double single_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // --- Double buffer (overlapped) ---
    // Prime first buffer
    dma_fill(buffers[0], kBufSize, 0);

    t0 = Clock::now();
    for (int i = 0; i < kTransfers; ++i) {
        int dma_buf = (i + 1) % 2;
        int cpu_buf = i % 2;

        // DMA fills next buffer while CPU processes current
        std::thread dma_thread([&, dma_buf, i]() {
            dma_fill(buffers[dma_buf], kBufSize, static_cast<uint32_t>(i + 1));
        });

        std::atomic_thread_fence(std::memory_order_acquire);
        cpu_sink += cpu_process(buffers[cpu_buf], kBufSize);

        dma_thread.join();
        std::atomic_thread_fence(std::memory_order_release);
    }
    t1 = Clock::now();
    double double_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double throughput_single = (kTransfers * kBufSize * 4.0) / (single_ms / 1000.0) / 1e9;
    double throughput_double = (kTransfers * kBufSize * 4.0) / (double_ms / 1000.0) / 1e9;

    std::printf("  Buffer size: %zu elements (%zu bytes)\n", kBufSize, kBufSize * 4);
    std::printf("  Transfers: %d\n", kTransfers);
    std::printf("  Single buffer:  %8.2f ms  (%.2f GB/s)\n", single_ms, throughput_single);
    std::printf("  Double buffer:  %8.2f ms  (%.2f GB/s)\n", double_ms, throughput_double);
    std::printf("  Speedup: %.2fx\n", single_ms / double_ms);
    std::printf("  (cpu_sink=%lu to prevent elision)\n\n", cpu_sink);

    return throughput_double;
}

// ---------------------------------------------------------------------------
// 5. Memory barrier demonstration
// ---------------------------------------------------------------------------
static void memory_barrier_demo() {
    std::printf("── 5. Memory Barrier Effects ──\n");

    std::atomic<int> data{0};
    std::atomic<bool> ready{false};
    volatile int observed_data = -1;

    // Producer-Consumer with acquire/release
    constexpr int kRounds = 10'000;  // Keep low — creates 2 threads per round
    int mismatches = 0;

    for (int round = 0; round < kRounds; ++round) {
        data.store(0, std::memory_order_relaxed);
        ready.store(false, std::memory_order_relaxed);

        std::thread producer([&]() {
            data.store(42, std::memory_order_relaxed);
            // Release fence ensures data=42 is visible before ready=true
            std::atomic_thread_fence(std::memory_order_release);
            ready.store(true, std::memory_order_relaxed);
        });

        std::thread consumer([&]() {
            while (!ready.load(std::memory_order_relaxed)) {
                // spin
            }
            // Acquire fence ensures we see data=42 after seeing ready=true
            std::atomic_thread_fence(std::memory_order_acquire);
            observed_data = data.load(std::memory_order_relaxed);
        });

        producer.join();
        consumer.join();

        if (observed_data != 42) ++mismatches;
    }

    std::printf("  Producer-consumer with acquire/release fences:\n");
    std::printf("  Rounds: %d, Mismatches: %d\n", kRounds, mismatches);
    std::printf("  (With proper fences, mismatches should be 0)\n");
    std::printf("  (Without fences on weakly-ordered archs like ARM, you might see > 0)\n\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::printf("╔══════════════════════════════════════════════════════╗\n");
    std::printf("║       HW03 — DMA & Cache Coherency                  ║\n");
    std::printf("╚══════════════════════════════════════════════════════╝\n\n");

    benchmark_access_patterns();
    size_t cache_line = measure_cache_line_size();
    double fs_ratio = run_false_sharing_benchmark();
    double db_throughput = run_double_buffer_simulation();
    memory_barrier_demo();

    // --- Summary ---
    std::printf("╔══════════════════════════════════════════════════════╗\n");
    std::printf("║  Summary                                             ║\n");
    std::printf("╠══════════════════════════════════════════════════════╣\n");
    std::printf("║  cache_line_size             : %3zu bytes             ║\n", cache_line);
    std::printf("║  false_sharing_penalty_ratio : %.2fx                 ║\n", fs_ratio);
    std::printf("║  double_buffer_throughput    : %.2f GB/s             ║\n", db_throughput);
    std::printf("╚══════════════════════════════════════════════════════╝\n");

    return 0;
}
