// Week 3 — Exercise 01: RT-Forbidden Operations
//
// Prove that malloc, printf, and disk I/O are slow and non-deterministic.
// Measure latency distributions to see the long-tail outliers.
//
// Build:  cmake --build build
// Run:    ./ex01_rt_forbidden
//
// No special permissions needed.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

using Clock = std::chrono::steady_clock;
using ns    = std::chrono::nanoseconds;

// ─── Helper: print latency statistics ────────────────────────────────
struct Stats {
    double min_us, max_us, mean_us, p50_us, p99_us, p999_us;
    int outliers_above_10us;
    int outliers_above_100us;
};

Stats compute_stats(std::vector<int64_t>& latencies_ns) {
    std::sort(latencies_ns.begin(), latencies_ns.end());
    size_t n = latencies_ns.size();

    double sum = 0;
    int out10 = 0, out100 = 0;
    for (auto l : latencies_ns) {
        sum += l;
        if (l > 10'000)  out10++;
        if (l > 100'000) out100++;
    }

    return Stats{
        .min_us  = latencies_ns.front() / 1000.0,
        .max_us  = latencies_ns.back()  / 1000.0,
        .mean_us = (sum / n) / 1000.0,
        .p50_us  = latencies_ns[n * 50 / 100] / 1000.0,
        .p99_us  = latencies_ns[n * 99 / 100] / 1000.0,
        .p999_us = latencies_ns[n * 999 / 1000] / 1000.0,
        .outliers_above_10us  = out10,
        .outliers_above_100us = out100,
    };
}

void print_stats(const char* label, Stats s) {
    std::printf("\n=== %s ===\n", label);
    std::printf("  min:  %8.2f μs\n", s.min_us);
    std::printf("  p50:  %8.2f μs\n", s.p50_us);
    std::printf("  mean: %8.2f μs\n", s.mean_us);
    std::printf("  p99:  %8.2f μs\n", s.p99_us);
    std::printf("  p999: %8.2f μs\n", s.p999_us);
    std::printf("  max:  %8.2f μs\n", s.max_us);
    std::printf("  outliers >10μs:  %d\n", s.outliers_above_10us);
    std::printf("  outliers >100μs: %d\n", s.outliers_above_100us);
}

// ─── Test 1: malloc/free latency ─────────────────────────────────────
void test_malloc() {
    constexpr int N = 10'000;
    std::vector<int64_t> latencies;
    latencies.reserve(N);

    for (int i = 0; i < N; ++i) {
        // Varying sizes to stress the allocator
        size_t sz = 64 + (i % 512) * 8;

        auto t0 = Clock::now();
        void* p = std::malloc(sz);
        auto t1 = Clock::now();

        // Force the compiler not to optimize away the allocation
        if (!p) { std::abort(); }
        static_cast<volatile char*>(p)[0] = 42;

        latencies.push_back(
            std::chrono::duration_cast<ns>(t1 - t0).count());

        std::free(p);
    }

    auto s = compute_stats(latencies);
    print_stats("malloc (varying sizes, 10k calls)", s);
}

// ─── Test 2: printf latency ──────────────────────────────────────────
void test_printf() {
    constexpr int N = 10'000;
    std::vector<int64_t> latencies;
    latencies.reserve(N);

    // Redirect stdout to /dev/null so we're not I/O bound on terminal
    FILE* devnull = std::fopen("/dev/null", "w");
    if (!devnull) { std::perror("fopen /dev/null"); return; }

    for (int i = 0; i < N; ++i) {
        auto t0 = Clock::now();
        std::fprintf(devnull, "iteration %d: value=%f\n", i, 3.14159 * i);
        auto t1 = Clock::now();

        latencies.push_back(
            std::chrono::duration_cast<ns>(t1 - t0).count());
    }

    std::fclose(devnull);

    auto s = compute_stats(latencies);
    print_stats("fprintf to /dev/null (10k calls)", s);
}

// ─── Test 3: printf to actual terminal ───────────────────────────────
void test_printf_terminal() {
    constexpr int N = 1'000;  // fewer — terminal output is slow
    std::vector<int64_t> latencies;
    latencies.reserve(N);

    // Temporarily capture stdout timing
    for (int i = 0; i < N; ++i) {
        auto t0 = Clock::now();
        std::printf("terminal test %d: value=%f\n", i, 3.14159 * i);
        auto t1 = Clock::now();

        latencies.push_back(
            std::chrono::duration_cast<ns>(t1 - t0).count());
    }

    auto s = compute_stats(latencies);
    // Print stats to stderr so they're not mixed with test output
    std::fprintf(stderr, "\n=== printf to terminal (1k calls) ===\n");
    std::fprintf(stderr, "  min:  %8.2f μs\n", s.min_us);
    std::fprintf(stderr, "  p50:  %8.2f μs\n", s.p50_us);
    std::fprintf(stderr, "  mean: %8.2f μs\n", s.mean_us);
    std::fprintf(stderr, "  p99:  %8.2f μs\n", s.p99_us);
    std::fprintf(stderr, "  p999: %8.2f μs\n", s.p999_us);
    std::fprintf(stderr, "  max:  %8.2f μs\n", s.max_us);
    std::fprintf(stderr, "  outliers >10μs:  %d\n", s.outliers_above_10us);
    std::fprintf(stderr, "  outliers >100μs: %d\n", s.outliers_above_100us);
}

// ─── Test 4: Pre-allocated buffer write vs printf ────────────────────
void test_buffer_vs_printf() {
    constexpr int N = 10'000;

    // Pre-allocate a log buffer (the RT-safe way)
    constexpr size_t BUF_SIZE = 1024 * 1024;  // 1MB
    char* log_buf = static_cast<char*>(std::malloc(BUF_SIZE));
    if (!log_buf) { std::abort(); }
    size_t offset = 0;

    // Measure: snprintf into pre-allocated buffer
    std::vector<int64_t> buf_latencies;
    buf_latencies.reserve(N);

    for (int i = 0; i < N; ++i) {
        auto t0 = Clock::now();
        int written = std::snprintf(log_buf + offset,
                                    BUF_SIZE - offset,
                                    "iter=%d val=%f\n", i, 3.14159 * i);
        auto t1 = Clock::now();

        if (written > 0 && offset + written < BUF_SIZE) {
            offset += written;
        }

        buf_latencies.push_back(
            std::chrono::duration_cast<ns>(t1 - t0).count());
    }

    // Measure: fprintf to /dev/null
    std::vector<int64_t> printf_latencies;
    printf_latencies.reserve(N);
    FILE* devnull = std::fopen("/dev/null", "w");

    for (int i = 0; i < N; ++i) {
        auto t0 = Clock::now();
        std::fprintf(devnull, "iter=%d val=%f\n", i, 3.14159 * i);
        auto t1 = Clock::now();

        printf_latencies.push_back(
            std::chrono::duration_cast<ns>(t1 - t0).count());
    }

    std::fclose(devnull);
    std::free(log_buf);

    auto s1 = compute_stats(buf_latencies);
    auto s2 = compute_stats(printf_latencies);
    print_stats("snprintf to pre-allocated buffer", s1);
    print_stats("fprintf to /dev/null (comparison)", s2);
}

// ─── Test 5: Prefault stack demonstration ────────────────────────────
void test_prefault_stack() {
    std::printf("\n=== Prefault Stack Demo ===\n");

    // Measure: access large stack WITHOUT prefaulting
    {
        auto t0 = Clock::now();
        volatile char stack_array[512 * 1024];  // 512KB
        // Touch every page
        for (size_t i = 0; i < sizeof(stack_array); i += 4096) {
            stack_array[i] = static_cast<char>(i);
        }
        auto t1 = Clock::now();
        auto dur = std::chrono::duration_cast<ns>(t1 - t0).count();
        std::printf("  First touch of 512KB stack: %.2f μs\n", dur / 1000.0);
    }

    // Now the pages are faulted in, measure again
    {
        auto t0 = Clock::now();
        volatile char stack_array[512 * 1024];
        for (size_t i = 0; i < sizeof(stack_array); i += 4096) {
            stack_array[i] = static_cast<char>(i);
        }
        auto t1 = Clock::now();
        auto dur = std::chrono::duration_cast<ns>(t1 - t0).count();
        std::printf("  Second touch (pages warm):  %.2f μs\n", dur / 1000.0);
    }

    // mlockall attempt
    std::printf("\n  Attempting mlockall...\n");
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == 0) {
        std::printf("  mlockall succeeded (running as root or CAP_IPC_LOCK)\n");
        munlockall();
    } else {
        std::printf("  mlockall failed: %s\n", std::strerror(errno));
        std::printf("  (expected — run as root or with CAP_IPC_LOCK to test)\n");
    }
}

// ─── Main ────────────────────────────────────────────────────────────
int main() {
    std::printf("╔════════════════════════════════════════════════╗\n");
    std::printf("║  Exercise 01: RT-Forbidden Operation Latency  ║\n");
    std::printf("╚════════════════════════════════════════════════╝\n");

    test_malloc();
    test_printf();
    test_printf_terminal();
    test_buffer_vs_printf();
    test_prefault_stack();

    std::printf("\n─── Key Takeaway ───\n");
    std::printf("Look at the MAX and p99.9 values, not the mean.\n");
    std::printf("In real-time code, the worst case IS your performance.\n");

    return 0;
}
