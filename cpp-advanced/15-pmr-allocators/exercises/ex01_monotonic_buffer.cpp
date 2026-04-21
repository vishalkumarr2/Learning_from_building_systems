// =============================================================================
// Exercise 01: Monotonic Buffer Resource
// =============================================================================
// Topics:
//   - Pre-allocating a stack buffer and wrapping in monotonic_buffer_resource
//   - Using PMR containers (pmr::vector, pmr::string)
//   - Benchmarking vs default allocator
//   - Demonstrating that deallocation is a no-op
//   - Buffer overflow fallback to upstream
//
// Compile: g++ -std=c++2a -O2 -Wall -Wextra -o ex01 ex01_monotonic_buffer.cpp
// =============================================================================

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory_resource>
#include <string>
#include <vector>

// -----------------------------------------------------------------------------
// Helper: high-resolution timer
// -----------------------------------------------------------------------------
struct Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start_;

    Timer() : start_(Clock::now()) {}

    double elapsed_us() const {
        auto end = Clock::now();
        return std::chrono::duration<double, std::micro>(end - start_).count();
    }
};

// =============================================================================
// Part 1: Basic monotonic_buffer_resource usage
// =============================================================================
void part1_basic_usage() {
    std::cout << "=== Part 1: Basic monotonic_buffer_resource Usage ===\n\n";

    // Pre-allocate a stack buffer — this is the fast memory we'll use.
    // In a real RT system, this might be a static buffer or mmap'd memory.
    alignas(std::max_align_t) char buffer[4096];

    // Wrap in monotonic_buffer_resource.
    // null_memory_resource() as upstream → crash if we overflow (good for RT!)
    std::pmr::monotonic_buffer_resource mr(
        buffer, sizeof(buffer), std::pmr::null_memory_resource());

    // Use with pmr::vector
    std::pmr::vector<int> numbers(&mr);
    for (int i = 0; i < 100; ++i) {
        numbers.push_back(i);
    }
    std::cout << "  pmr::vector<int>: " << numbers.size() << " elements\n";

    // Use with pmr::string
    std::pmr::string greeting("Hello from monotonic buffer!", &mr);
    std::cout << "  pmr::string: \"" << greeting << "\"\n";

    // Multiple containers sharing the same resource
    std::pmr::vector<std::pmr::string> names(&mr);
    names.emplace_back("Alice");
    names.emplace_back("Bob");
    names.emplace_back("Charlie");
    std::cout << "  pmr::vector<pmr::string>: " << names.size() << " names\n";

    // Key insight: all allocations came from our 4096-byte stack buffer.
    // No heap allocation occurred!
    std::cout << "  All allocated from a 4096-byte stack buffer.\n\n";
}

// =============================================================================
// Part 2: Benchmark — monotonic vs default allocator
// =============================================================================
void part2_benchmark() {
    std::cout << "=== Part 2: Benchmark — 10K Small Object Allocations ===\n\n";

    constexpr int N = 10'000;
    constexpr int WARMUP = 3;
    constexpr int TRIALS = 5;

    // --- Default allocator (heap) ---
    double heap_best = 1e9;
    for (int trial = 0; trial < WARMUP + TRIALS; ++trial) {
        Timer t;
        {
            std::vector<std::string> v;
            v.reserve(N);
            for (int i = 0; i < N; ++i) {
                v.emplace_back("sensor_data_value_" + std::to_string(i));
            }
        }  // destructor frees all
        double us = t.elapsed_us();
        if (trial >= WARMUP && us < heap_best) heap_best = us;
    }

    // --- Monotonic buffer resource ---
    // Pre-allocate enough for all strings (~40 bytes each + vector overhead)
    constexpr size_t BUF_SIZE = 1024 * 1024;  // 1 MB
    alignas(std::max_align_t) static char buf[BUF_SIZE];

    double mono_best = 1e9;
    for (int trial = 0; trial < WARMUP + TRIALS; ++trial) {
        std::pmr::monotonic_buffer_resource mr(buf, BUF_SIZE,
                                                std::pmr::null_memory_resource());
        Timer t;
        {
            std::pmr::vector<std::pmr::string> v(&mr);
            v.reserve(N);
            for (int i = 0; i < N; ++i) {
                v.emplace_back("sensor_data_value_" + std::to_string(i));
            }
        }
        double us = t.elapsed_us();
        if (trial >= WARMUP && us < mono_best) mono_best = us;
        mr.release();
    }

    std::printf("  Default allocator (heap): %.1f μs\n", heap_best);
    std::printf("  Monotonic buffer:         %.1f μs\n", mono_best);
    std::printf("  Speedup:                  %.1fx\n\n", heap_best / mono_best);
}

// =============================================================================
// Part 3: Deallocation is a no-op — prove it with timing
// =============================================================================
void part3_dealloc_noop() {
    std::cout << "=== Part 3: Deallocation is a No-Op ===\n\n";

    constexpr int N = 50'000;
    constexpr size_t BUF_SIZE = 4 * 1024 * 1024;
    alignas(std::max_align_t) static char buf[BUF_SIZE];

    // --- Heap: deallocation takes real time ---
    std::vector<int*> heap_ptrs;
    heap_ptrs.reserve(N);
    for (int i = 0; i < N; ++i) {
        heap_ptrs.push_back(new int(i));
    }

    Timer t_heap_dealloc;
    for (auto* p : heap_ptrs) {
        delete p;
    }
    double heap_dealloc_us = t_heap_dealloc.elapsed_us();

    // --- Monotonic: deallocation is a no-op, release() frees all ---
    std::pmr::monotonic_buffer_resource mr(buf, BUF_SIZE,
                                            std::pmr::null_memory_resource());
    std::pmr::vector<int*> mono_ptrs(&mr);
    mono_ptrs.reserve(N);

    // Allocate N objects from the monotonic resource
    for (int i = 0; i < N; ++i) {
        void* p = mr.allocate(sizeof(int), alignof(int));
        new (p) int(i);  // placement new
        mono_ptrs.push_back(static_cast<int*>(p));
    }

    // "Deallocate" each one — this does NOTHING for monotonic
    Timer t_mono_dealloc;
    for (auto* p : mono_ptrs) {
        mr.deallocate(p, sizeof(int), alignof(int));
    }
    double mono_dealloc_us = t_mono_dealloc.elapsed_us();

    // Release all at once — this is the real cleanup
    Timer t_release;
    mr.release();
    double release_us = t_release.elapsed_us();

    std::printf("  Heap:  %d individual deletes:     %.1f μs\n", N, heap_dealloc_us);
    std::printf("  Mono:  %d individual deallocates:  %.1f μs (no-op!)\n", N, mono_dealloc_us);
    std::printf("  Mono:  release() all at once:      %.1f μs\n\n", release_us);
}

// =============================================================================
// Part 4: Buffer overflow — fallback to upstream
// =============================================================================
void part4_overflow() {
    std::cout << "=== Part 4: Buffer Overflow Behavior ===\n\n";

    // --- Case A: null upstream → exception on overflow ---
    {
        alignas(std::max_align_t) char tiny_buf[64];
        std::pmr::monotonic_buffer_resource mr(
            tiny_buf, sizeof(tiny_buf), std::pmr::null_memory_resource());

        std::pmr::vector<int> v(&mr);
        try {
            // 64 bytes ≈ 16 ints. push_back triggers vector growth beyond 64 bytes.
            for (int i = 0; i < 100; ++i) {
                v.push_back(i);
            }
            std::cout << "  [null upstream] ERROR: should have thrown!\n";
        } catch (const std::bad_alloc& e) {
            std::cout << "  [null upstream] Caught std::bad_alloc after "
                      << v.size() << " elements — overflow detected!\n";
            std::cout << "  This is GOOD for RT: you know immediately if you "
                         "underestimated buffer size.\n";
        }
    }

    // --- Case B: new_delete_resource upstream → silent heap fallback ---
    {
        alignas(std::max_align_t) char tiny_buf[64];
        std::pmr::monotonic_buffer_resource mr(
            tiny_buf, sizeof(tiny_buf), std::pmr::new_delete_resource());

        std::pmr::vector<int> v(&mr);
        for (int i = 0; i < 1000; ++i) {
            v.push_back(i);
        }
        std::cout << "\n  [new_delete upstream] Stored " << v.size()
                  << " elements — silently fell back to heap.\n";
        std::cout << "  WARNING: In RT, this is dangerous! You lose determinism.\n";
        std::cout << "  ALWAYS use null_memory_resource() as upstream in RT code.\n\n";
    }
}

// =============================================================================
int main() {
    std::cout << "╔══════════════════════════════════════════════════╗\n";
    std::cout << "║  Exercise 01: Monotonic Buffer Resource          ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n\n";

    part1_basic_usage();
    part2_benchmark();
    part3_dealloc_noop();
    part4_overflow();

    std::cout << "=== Summary ===\n";
    std::cout << "  - monotonic_buffer_resource = bump allocator (fastest possible)\n";
    std::cout << "  - Deallocation is a no-op; release() frees everything at once\n";
    std::cout << "  - Use null_memory_resource() upstream to catch overflow in RT\n";
    std::cout << "  - Pre-allocate buffer at startup, use in hot loop, release per cycle\n";

    return 0;
}
