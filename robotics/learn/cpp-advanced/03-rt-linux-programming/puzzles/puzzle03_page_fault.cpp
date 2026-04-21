// Week 3 — Puzzle 03: Page Fault Latency
//
// Show that accessing un-prefaulted memory causes 10-100μs latency spikes.
// Then demonstrate that mlockall + memset eliminates the spikes.
//
// Build:  cmake --build .
// Run:    ./puzzle03_page_fault
//         sudo ./puzzle03_page_fault   (for mlockall to work)

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

using Clock = std::chrono::steady_clock;
using ns    = std::chrono::nanoseconds;

constexpr size_t ALLOC_SIZE      = 64 * 1024 * 1024;  // 64MB
constexpr size_t PAGE_SIZE       = 4096;
constexpr size_t NUM_PAGES       = ALLOC_SIZE / PAGE_SIZE;
constexpr size_t SAMPLE_PAGES    = 1024;  // measure subset for clarity

struct Stats {
    double min_us, max_us, mean_us, p99_us;
    int    above_10us;
    int    above_50us;
};

Stats compute_stats(std::vector<int64_t>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    int a10 = 0, a50 = 0;
    for (auto l : v) {
        if (l > 10'000) a10++;
        if (l > 50'000) a50++;
    }
    return {
        .min_us    = v.front() / 1000.0,
        .max_us    = v.back()  / 1000.0,
        .mean_us   = (sum / n) / 1000.0,
        .p99_us    = v[n * 99 / 100] / 1000.0,
        .above_10us  = a10,
        .above_50us  = a50,
    };
}

void print_stats(const char* label, Stats s, int total) {
    std::printf("\n  === %s ===\n", label);
    std::printf("  min:  %8.2f μs\n", s.min_us);
    std::printf("  mean: %8.2f μs\n", s.mean_us);
    std::printf("  p99:  %8.2f μs\n", s.p99_us);
    std::printf("  max:  %8.2f μs\n", s.max_us);
    std::printf("  pages >10μs:  %d / %d  (%.1f%%)\n",
                s.above_10us, total, 100.0 * s.above_10us / total);
    std::printf("  pages >50μs:  %d / %d  (%.1f%%)\n",
                s.above_50us, total, 100.0 * s.above_50us / total);
}

// ═══════════════════════════════════════════════════════════════════════
// Test 1: Measure first-touch per page (cold — will page fault)
// ═══════════════════════════════════════════════════════════════════════

void test_cold_access() {
    std::printf("\n--- Test 1: Cold Access (un-prefaulted memory) ---\n");
    std::printf("  Allocating %zuMB...\n", ALLOC_SIZE / (1024*1024));

    // mmap with no population — pages are virtual-only
    char* mem = static_cast<char*>(
        mmap(nullptr, ALLOC_SIZE, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (mem == MAP_FAILED) {
        std::perror("mmap");
        return;
    }

    // Measure per-page first-touch latency
    std::vector<int64_t> latencies;
    latencies.reserve(SAMPLE_PAGES);

    // Sample every Nth page to spread across the allocation
    size_t step = NUM_PAGES / SAMPLE_PAGES;

    for (size_t i = 0; i < SAMPLE_PAGES; ++i) {
        size_t offset = i * step * PAGE_SIZE;

        auto t0 = Clock::now();
        // First write to this page → triggers a page fault
        volatile char* p = mem + offset;
        *p = 42;
        auto t1 = Clock::now();

        latencies.push_back(
            std::chrono::duration_cast<ns>(t1 - t0).count());
    }

    auto s = compute_stats(latencies);
    print_stats("First touch (page fault expected)", s, SAMPLE_PAGES);

    // Show first 10 per-page latencies
    std::printf("\n  First 10 page-touch latencies:\n  ");
    for (int i = 0; i < 10 && i < static_cast<int>(latencies.size()); ++i) {
        // Use original (unsorted) order — compute_stats sorted the vector
        std::printf(" [sorted=%5.1fμs]", latencies[i] / 1000.0);
    }
    std::printf("\n");

    munmap(mem, ALLOC_SIZE);
}

// ═══════════════════════════════════════════════════════════════════════
// Test 2: Prefaulted access (warm — no page faults)
// ═══════════════════════════════════════════════════════════════════════

void test_warm_access() {
    std::printf("\n--- Test 2: Warm Access (prefaulted with memset) ---\n");

    char* mem = static_cast<char*>(
        mmap(nullptr, ALLOC_SIZE, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    if (mem == MAP_FAILED) {
        std::perror("mmap");
        return;
    }

    // Prefault: touch every page
    std::printf("  Prefaulting %zuMB with memset...\n", ALLOC_SIZE / (1024*1024));
    auto pf_start = Clock::now();
    std::memset(mem, 0, ALLOC_SIZE);
    auto pf_end = Clock::now();
    auto pf_us = std::chrono::duration_cast<ns>(pf_end - pf_start).count() / 1000.0;
    std::printf("  Prefault took: %.2f ms\n", pf_us / 1000.0);

    // Now re-measure
    std::vector<int64_t> latencies;
    latencies.reserve(SAMPLE_PAGES);

    size_t step = NUM_PAGES / SAMPLE_PAGES;

    for (size_t i = 0; i < SAMPLE_PAGES; ++i) {
        size_t offset = i * step * PAGE_SIZE;

        auto t0 = Clock::now();
        volatile char* p = mem + offset;
        *p = 42;
        auto t1 = Clock::now();

        latencies.push_back(
            std::chrono::duration_cast<ns>(t1 - t0).count());
    }

    auto s = compute_stats(latencies);
    print_stats("After prefault (no page faults)", s, SAMPLE_PAGES);

    munmap(mem, ALLOC_SIZE);
}

// ═══════════════════════════════════════════════════════════════════════
// Test 3: mlockall + prefault
// ═══════════════════════════════════════════════════════════════════════

void test_mlock_access() {
    std::printf("\n--- Test 3: mlockall + prefault ---\n");

    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        std::printf("  mlockall failed: %s\n", std::strerror(errno));
        std::printf("  Run as root or with CAP_IPC_LOCK to test.\n");
        std::printf("  Skipping test.\n");
        return;
    }
    std::printf("  mlockall(MCL_CURRENT | MCL_FUTURE): OK\n");

    char* mem = static_cast<char*>(
        mmap(nullptr, ALLOC_SIZE, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0));
    if (mem == MAP_FAILED) {
        std::perror("mmap");
        munlockall();
        return;
    }

    // MAP_POPULATE + mlockall = pages are physical and pinned
    std::printf("  Allocated %zuMB with MAP_POPULATE (no faults needed)\n",
                ALLOC_SIZE / (1024*1024));

    std::vector<int64_t> latencies;
    latencies.reserve(SAMPLE_PAGES);

    size_t step = NUM_PAGES / SAMPLE_PAGES;

    for (size_t i = 0; i < SAMPLE_PAGES; ++i) {
        size_t offset = i * step * PAGE_SIZE;

        auto t0 = Clock::now();
        volatile char* p = mem + offset;
        *p = 42;
        auto t1 = Clock::now();

        latencies.push_back(
            std::chrono::duration_cast<ns>(t1 - t0).count());
    }

    auto s = compute_stats(latencies);
    print_stats("mlockall + MAP_POPULATE (fully pinned)", s, SAMPLE_PAGES);

    munmap(mem, ALLOC_SIZE);
    munlockall();
}

// ═══════════════════════════════════════════════════════════════════════
int main() {
    std::printf("╔════════════════════════════════════════════════════╗\n");
    std::printf("║  Puzzle 03: Page Fault Latency Spikes             ║\n");
    std::printf("╚════════════════════════════════════════════════════╝\n");
    std::printf("Page size: %zu bytes\n", PAGE_SIZE);
    std::printf("Allocation: %zuMB (%zu pages)\n",
                ALLOC_SIZE / (1024*1024), NUM_PAGES);
    std::printf("Sampling: %zu pages\n", SAMPLE_PAGES);

    test_cold_access();
    test_warm_access();
    test_mlock_access();

    std::printf("\n─── The Lesson ───\n");
    std::printf("Test 1: First touch of un-faulted pages → 10-100μs per page\n");
    std::printf("Test 2: After memset prefault → <1μs per page\n");
    std::printf("Test 3: mlockall + MAP_POPULATE → <1μs, pages can't be swapped\n");
    std::printf("\nIn RT code: ALWAYS prefault ALL memory before entering the loop.\n");
    std::printf("Use mlockall(MCL_CURRENT | MCL_FUTURE) to prevent swap-out.\n");

    return 0;
}
