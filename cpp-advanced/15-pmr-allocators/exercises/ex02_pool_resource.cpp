// =============================================================================
// Exercise 02: Pool Resource
// =============================================================================
// Topics:
//   - unsynchronized_pool_resource with custom pool_options
//   - How allocation binning by size works
//   - Memory reuse after deallocation (vs monotonic)
//   - synchronized_pool_resource for multi-threaded scenarios
//
// Compile: g++ -std=c++2a -O2 -Wall -Wextra -pthread -o ex02 ex02_pool_resource.cpp
// =============================================================================

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <memory_resource>
#include <string>
#include <thread>
#include <vector>

// -----------------------------------------------------------------------------
// TrackingResource: wraps another resource, counts allocations/deallocations
// Used here to observe pool behavior.
// -----------------------------------------------------------------------------
class TrackingResource : public std::pmr::memory_resource {
public:
    explicit TrackingResource(std::pmr::memory_resource* upstream,
                               const char* name = "tracker")
        : upstream_(upstream), name_(name) {}

    void report() const {
        std::printf("  [%s] allocs=%d, deallocs=%d, bytes_alloc=%zu, "
                    "bytes_dealloc=%zu, current=%zu\n",
                    name_, alloc_count_, dealloc_count_,
                    bytes_allocated_, bytes_deallocated_,
                    bytes_allocated_ - bytes_deallocated_);
    }

    void reset_counters() {
        alloc_count_ = dealloc_count_ = 0;
        bytes_allocated_ = bytes_deallocated_ = 0;
    }

    int alloc_count() const { return alloc_count_; }
    int dealloc_count() const { return dealloc_count_; }

private:
    void* do_allocate(size_t bytes, size_t alignment) override {
        void* p = upstream_->allocate(bytes, alignment);
        ++alloc_count_;
        bytes_allocated_ += bytes;
        return p;
    }

    void do_deallocate(void* p, size_t bytes, size_t alignment) override {
        upstream_->deallocate(p, bytes, alignment);
        ++dealloc_count_;
        bytes_deallocated_ += bytes;
    }

    bool do_is_equal(const memory_resource& other) const noexcept override {
        return this == &other;
    }

    std::pmr::memory_resource* upstream_;
    const char* name_;
    int alloc_count_ = 0;
    int dealloc_count_ = 0;
    size_t bytes_allocated_ = 0;
    size_t bytes_deallocated_ = 0;
};

struct Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start_ = Clock::now();
    double elapsed_us() const {
        return std::chrono::duration<double, std::micro>(Clock::now() - start_).count();
    }
};

// =============================================================================
// Part 1: Basic unsynchronized_pool_resource
// =============================================================================
void part1_basic_pool() {
    std::cout << "=== Part 1: Basic unsynchronized_pool_resource ===\n\n";

    // Configure pool options
    std::pmr::pool_options opts;
    opts.max_blocks_per_chunk = 256;         // Max blocks the pool grabs at once
    opts.largest_required_pool_block = 1024;  // Largest size the pool handles
    // Allocations > 1024 bytes go directly to upstream

    TrackingResource tracker(std::pmr::new_delete_resource(), "upstream");
    std::pmr::unsynchronized_pool_resource pool(opts, &tracker);

    std::cout << "  Pool options:\n";
    std::cout << "    max_blocks_per_chunk:        " << opts.max_blocks_per_chunk << "\n";
    std::cout << "    largest_required_pool_block:  " << opts.largest_required_pool_block << "\n\n";

    // Allocate various sizes — pool bins them
    std::cout << "  Allocating various sizes:\n";
    void* p8    = pool.allocate(8);
    void* p32   = pool.allocate(32);
    void* p128  = pool.allocate(128);
    void* p512  = pool.allocate(512);
    void* p2048 = pool.allocate(2048);  // > largest_required → direct upstream

    std::cout << "  After 5 allocations (8, 32, 128, 512, 2048 bytes):\n";
    tracker.report();
    std::cout << "  Note: pool may batch upstream requests (ask for a chunk of blocks).\n";
    std::cout << "  The 2048-byte alloc likely went directly to upstream.\n\n";

    pool.deallocate(p8, 8);
    pool.deallocate(p32, 32);
    pool.deallocate(p128, 128);
    pool.deallocate(p512, 512);
    pool.deallocate(p2048, 2048);
}

// =============================================================================
// Part 2: Memory reuse — pool vs monotonic
// =============================================================================
void part2_reuse_comparison() {
    std::cout << "\n=== Part 2: Memory Reuse — Pool vs Monotonic ===\n\n";

    constexpr int ROUNDS = 5;
    constexpr int N = 100;

    // --- Pool: deallocated memory is reused ---
    {
        TrackingResource tracker(std::pmr::new_delete_resource(), "pool-upstream");
        std::pmr::unsynchronized_pool_resource pool({}, &tracker);

        for (int round = 0; round < ROUNDS; ++round) {
            std::pmr::vector<int> v(&pool);
            for (int i = 0; i < N; ++i) v.push_back(i);
            // v destructor returns memory to pool → available for next round
        }

        std::cout << "  POOL after " << ROUNDS << " rounds of " << N << " ints:\n";
        tracker.report();
        std::cout << "  Pool REUSES memory — upstream allocs are minimal after round 1.\n\n";
    }

    // --- Monotonic: deallocated memory is NOT reused ---
    {
        TrackingResource tracker(std::pmr::new_delete_resource(), "mono-upstream");
        std::pmr::monotonic_buffer_resource mono(&tracker);

        for (int round = 0; round < ROUNDS; ++round) {
            std::pmr::vector<int> v(&mono);
            for (int i = 0; i < N; ++i) v.push_back(i);
            // v destructor calls deallocate → monotonic IGNORES it
            // memory is "wasted" until release()
        }

        std::cout << "  MONOTONIC after " << ROUNDS << " rounds of " << N << " ints:\n";
        tracker.report();
        std::cout << "  Monotonic NEVER reuses — every round allocates fresh from upstream.\n";
        std::cout << "  This is fine if you release() between rounds (see ex04).\n\n";
    }
}

// =============================================================================
// Part 3: Benchmark — pool vs monotonic vs heap
// =============================================================================
void part3_benchmark() {
    std::cout << "=== Part 3: Benchmark — Alloc/Dealloc Cycle ===\n\n";

    constexpr int N = 10'000;
    constexpr int TRIALS = 5;

    // Pattern: allocate N blocks of 64 bytes, deallocate all, repeat.
    // This simulates a hot path that creates and destroys temp objects.

    // --- Heap ---
    double heap_best = 1e9;
    for (int t = 0; t < TRIALS; ++t) {
        Timer timer;
        for (int round = 0; round < 10; ++round) {
            std::vector<void*> ptrs;
            ptrs.reserve(N);
            for (int i = 0; i < N; ++i)
                ptrs.push_back(::operator new(64));
            for (auto* p : ptrs)
                ::operator delete(p, 64);
        }
        double us = timer.elapsed_us();
        if (us < heap_best) heap_best = us;
    }

    // --- Pool ---
    double pool_best = 1e9;
    for (int t = 0; t < TRIALS; ++t) {
        std::pmr::unsynchronized_pool_resource pool;
        Timer timer;
        for (int round = 0; round < 10; ++round) {
            std::vector<void*> ptrs;
            ptrs.reserve(N);
            for (int i = 0; i < N; ++i)
                ptrs.push_back(pool.allocate(64));
            for (auto* p : ptrs)
                pool.deallocate(p, 64);
        }
        double us = timer.elapsed_us();
        if (us < pool_best) pool_best = us;
    }

    // --- Monotonic (with release between rounds) ---
    constexpr size_t BUF_SIZE = 2 * 1024 * 1024;
    alignas(std::max_align_t) static char buf[BUF_SIZE];
    double mono_best = 1e9;
    for (int t = 0; t < TRIALS; ++t) {
        std::pmr::monotonic_buffer_resource mono(buf, BUF_SIZE,
                                                  std::pmr::null_memory_resource());
        Timer timer;
        for (int round = 0; round < 10; ++round) {
            for (int i = 0; i < N; ++i)
                mono.allocate(64);
            // No individual dealloc — release all at once
            mono.release();
        }
        double us = timer.elapsed_us();
        if (us < mono_best) mono_best = us;
    }

    std::printf("  Heap (new/delete):   %.0f μs  (1.0x)\n", heap_best);
    std::printf("  Pool (unsync):       %.0f μs  (%.1fx)\n",
                pool_best, heap_best / pool_best);
    std::printf("  Monotonic (release): %.0f μs  (%.1fx)\n\n",
                mono_best, heap_best / mono_best);
}

// =============================================================================
// Part 4: synchronized_pool_resource — multi-threaded
// =============================================================================
void part4_synchronized() {
    std::cout << "=== Part 4: synchronized_pool_resource (Multi-threaded) ===\n\n";

    constexpr int N_THREADS = 4;
    constexpr int N_ALLOCS = 5000;

    // --- WRONG WAY: unsynchronized across threads → DATA RACE ---
    std::cout << "  (Skipping unsynchronized multi-threaded — it's undefined behavior)\n";
    std::cout << "  NEVER share unsynchronized_pool_resource across threads!\n\n";

    // --- RIGHT WAY: synchronized_pool_resource ---
    {
        std::pmr::synchronized_pool_resource sync_pool;

        Timer t;
        std::vector<std::thread> threads;
        threads.reserve(N_THREADS);

        for (int tid = 0; tid < N_THREADS; ++tid) {
            threads.emplace_back([&sync_pool, tid]() {
                // Each thread allocates and deallocates from the shared pool
                std::pmr::vector<std::pmr::string> data(&sync_pool);
                for (int i = 0; i < N_ALLOCS; ++i) {
                    data.emplace_back("thread_" + std::to_string(tid)
                                      + "_item_" + std::to_string(i));
                }
                // data destructor returns everything to sync_pool
            });
        }

        for (auto& thr : threads) thr.join();
        double us = t.elapsed_us();

        std::printf("  synchronized_pool: %d threads × %d allocs = %.0f μs\n",
                    N_THREADS, N_ALLOCS, us);
        std::cout << "  Thread-safe but slower than unsynchronized.\n";
        std::cout << "  Prefer per-thread unsynchronized pools when possible.\n\n";
    }

    // --- BEST WAY: per-thread unsynchronized pools ---
    {
        Timer t;
        std::vector<std::thread> threads;
        threads.reserve(N_THREADS);

        for (int tid = 0; tid < N_THREADS; ++tid) {
            threads.emplace_back([tid]() {
                // Each thread has its OWN pool — no locking needed
                std::pmr::unsynchronized_pool_resource local_pool;
                std::pmr::vector<std::pmr::string> data(&local_pool);
                for (int i = 0; i < N_ALLOCS; ++i) {
                    data.emplace_back("thread_" + std::to_string(tid)
                                      + "_item_" + std::to_string(i));
                }
            });
        }

        for (auto& thr : threads) thr.join();
        double us = t.elapsed_us();

        std::printf("  per-thread unsync:  %d threads × %d allocs = %.0f μs\n",
                    N_THREADS, N_ALLOCS, us);
        std::cout << "  Fastest multi-threaded approach: no lock contention.\n\n";
    }
}

// =============================================================================
int main() {
    std::cout << "╔══════════════════════════════════════════════════╗\n";
    std::cout << "║  Exercise 02: Pool Resource                      ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n\n";

    part1_basic_pool();
    part2_reuse_comparison();
    part3_benchmark();
    part4_synchronized();

    std::cout << "=== Summary ===\n";
    std::cout << "  - Pool resource bins allocations by size → fast reuse\n";
    std::cout << "  - unsynchronized = single thread (fastest)\n";
    std::cout << "  - synchronized = thread-safe (has locking overhead)\n";
    std::cout << "  - Best pattern: per-thread unsynchronized pools\n";
    std::cout << "  - Pool > heap for repeated alloc/dealloc cycles\n";
    std::cout << "  - Monotonic > pool when you can release all at once\n";

    return 0;
}
