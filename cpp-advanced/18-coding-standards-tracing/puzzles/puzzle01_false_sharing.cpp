// Puzzle 1: False Sharing — The Invisible Performance Killer
//
// Two threads each increment their own counter. They don't share data...
// or DO they? When counters live on the same cache line, the CPU's
// coherency protocol (MESI) bounces the line between cores.
//
// TASK:
//   1. Run with --bench and observe the "shared" vs "padded" times
//   2. Profile with: perf stat -e L1-dcache-load-misses ./puzzle01_false_sharing --bench
//   3. Explain why padding fixes the performance gap
//
// KEY INSIGHT: sizeof(cache line) = 64 bytes on x86. Two atomics in the
// same 64 bytes = false sharing. Pad to separate cache lines.

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

namespace false_sharing {

constexpr size_t kIterations = 10'000'000;

// ── BAD: Two atomics on the same cache line ──
struct SharedCounters {
    std::atomic<int64_t> counter_a{0};
    std::atomic<int64_t> counter_b{0};
    // sizeof = 16 bytes — both fit in one 64-byte cache line
};

// ── GOOD: Padded to separate cache lines ──
struct alignas(64) PaddedCounter {
    std::atomic<int64_t> value{0};
    // Padding fills the rest of the cache line
};

struct PaddedCounters {
    PaddedCounter counter_a;
    PaddedCounter counter_b;
};

// Worker: increment one counter many times
template <typename Counter>
void worker(Counter& counter) {
    for (size_t i = 0; i < kIterations; ++i) {
        counter.fetch_add(1, std::memory_order_relaxed);
    }
}

void test() {
    // Shared version
    SharedCounters shared;
    {
        std::thread t1(worker<std::atomic<int64_t>>, std::ref(shared.counter_a));
        std::thread t2(worker<std::atomic<int64_t>>, std::ref(shared.counter_b));
        t1.join();
        t2.join();
    }
    assert(shared.counter_a.load() == static_cast<int64_t>(kIterations));
    assert(shared.counter_b.load() == static_cast<int64_t>(kIterations));

    // Padded version
    PaddedCounters padded;
    {
        std::thread t1(worker<std::atomic<int64_t>>, std::ref(padded.counter_a.value));
        std::thread t2(worker<std::atomic<int64_t>>, std::ref(padded.counter_b.value));
        t1.join();
        t2.join();
    }
    assert(padded.counter_a.value.load() == static_cast<int64_t>(kIterations));
    assert(padded.counter_b.value.load() == static_cast<int64_t>(kIterations));
}

void benchmark() {
    constexpr int kRuns = 3;

    std::printf("  Layout verification:\n");
    std::printf("    SharedCounters size:  %zu bytes (both in 1 cache line)\n",
                sizeof(SharedCounters));
    std::printf("    PaddedCounters size:  %zu bytes (each in own cache line)\n",
                sizeof(PaddedCounters));
    std::printf("    PaddedCounter align:  %zu bytes\n", alignof(PaddedCounter));

    long shared_best = 999999999;
    long padded_best = 999999999;

    for (int run = 0; run < kRuns; ++run) {
        // Shared
        SharedCounters shared;
        auto t0 = std::chrono::steady_clock::now();
        {
            std::thread t1(worker<std::atomic<int64_t>>, std::ref(shared.counter_a));
            std::thread t2(worker<std::atomic<int64_t>>, std::ref(shared.counter_b));
            t1.join();
            t2.join();
        }
        auto t1 = std::chrono::steady_clock::now();

        // Padded
        PaddedCounters padded;
        auto t2 = std::chrono::steady_clock::now();
        {
            std::thread ta(worker<std::atomic<int64_t>>, std::ref(padded.counter_a.value));
            std::thread tb(worker<std::atomic<int64_t>>, std::ref(padded.counter_b.value));
            ta.join();
            tb.join();
        }
        auto t3 = std::chrono::steady_clock::now();

        long s_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        long p_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
        if (s_ms < shared_best) shared_best = s_ms;
        if (p_ms < padded_best) padded_best = p_ms;
    }

    long padded_safe = (padded_best == 0) ? 1 : padded_best;
    std::printf("\n  Results (best of %d runs, %zu iterations each):\n", kRuns, kIterations);
    std::printf("    False sharing (bad):  %ld ms\n", shared_best);
    std::printf("    Cache-line padded:    %ld ms\n", padded_best);
    std::printf("    Ratio:                %.1fx\n",
                static_cast<double>(shared_best) / static_cast<double>(padded_safe));
}

}  // namespace false_sharing

// ========================================================================
// Main
// ========================================================================

int main(int argc, char** argv) {
    false_sharing::test();
    std::printf("puzzle01_false_sharing: ALL TESTS PASSED\n");

    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0)
            bench = true;
    }

    if (bench) {
        std::printf("\n=== FALSE SHARING BENCHMARK ===\n");
        false_sharing::benchmark();
    }

    return 0;
}
