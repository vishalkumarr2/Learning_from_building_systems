// Week 3 — Exercise 04: Pool Allocator and PMR
//
// Implement a fixed pool allocator and demonstrate PMR usage.
// Detect accidental heap allocation by replacing global operator new.
//
// Build:  cmake --build build
// Run:    ./ex04_pool_allocator

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory_resource>
#include <new>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
using ns    = std::chrono::nanoseconds;

// ═══════════════════════════════════════════════════════════════════════
// Part 1: FixedPool<T, N> — O(1) alloc/free from pre-allocated storage
// ═══════════════════════════════════════════════════════════════════════

template <typename T, std::size_t N>
class FixedPool {
public:
    FixedPool() {
        // Initialize the free list: each slot points to the next
        for (std::size_t i = 0; i < N - 1; ++i) {
            *reinterpret_cast<std::size_t*>(&storage_[i]) = i + 1;
        }
        *reinterpret_cast<std::size_t*>(&storage_[N - 1]) = SENTINEL;
        free_head_ = 0;
        allocated_ = 0;
    }

    // No copies
    FixedPool(const FixedPool&) = delete;
    FixedPool& operator=(const FixedPool&) = delete;

    // Allocate one T-sized block. Returns nullptr if pool is exhausted.
    T* allocate() {
        if (free_head_ == SENTINEL) {
            return nullptr;  // pool exhausted
        }
        std::size_t idx = free_head_;
        free_head_ = *reinterpret_cast<std::size_t*>(&storage_[idx]);
        allocated_++;
        return reinterpret_cast<T*>(&storage_[idx]);
    }

    // Return a block to the pool.
    void deallocate(T* ptr) {
        if (!ptr) return;
        auto* slot = reinterpret_cast<Storage*>(ptr);
        std::size_t idx = static_cast<std::size_t>(slot - storage_.data());
        if (idx >= N) {
            std::fprintf(stderr, "FixedPool::deallocate: pointer not from this pool!\n");
            std::abort();
        }
        *reinterpret_cast<std::size_t*>(&storage_[idx]) = free_head_;
        free_head_ = idx;
        allocated_--;
    }

    std::size_t capacity()  const { return N; }
    std::size_t allocated() const { return allocated_; }
    std::size_t available() const { return N - allocated_; }

private:
    static constexpr std::size_t SENTINEL = ~std::size_t{0};

    // Storage must be large enough for T and for a size_t (free list link)
    union Storage {
        alignas(T) std::byte data[sizeof(T)];
        std::size_t next;  // used only when the slot is free
    };

    std::array<Storage, N> storage_;
    std::size_t free_head_;
    std::size_t allocated_;
};

// ═══════════════════════════════════════════════════════════════════════
// Part 2: Heap allocation detection
// ═══════════════════════════════════════════════════════════════════════

static std::atomic<int> g_heap_alloc_count{0};
static bool g_tracking_enabled = false;

// Replace global operator new to detect unexpected heap allocations
void* operator new(std::size_t size) {
    if (g_tracking_enabled) {
        g_heap_alloc_count.fetch_add(1, std::memory_order_relaxed);
    }
    void* p = std::malloc(size);
    if (!p) throw std::bad_alloc{};
    return p;
}

void* operator new[](std::size_t size) {
    if (g_tracking_enabled) {
        g_heap_alloc_count.fetch_add(1, std::memory_order_relaxed);
    }
    void* p = std::malloc(size);
    if (!p) throw std::bad_alloc{};
    return p;
}

void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

struct HeapTracker {
    HeapTracker() {
        g_heap_alloc_count.store(0, std::memory_order_relaxed);
        g_tracking_enabled = true;
    }
    ~HeapTracker() {
        g_tracking_enabled = false;
    }
    int count() const {
        return g_heap_alloc_count.load(std::memory_order_relaxed);
    }
};

// ═══════════════════════════════════════════════════════════════════════
// Part 3: Tests and demonstrations
// ═══════════════════════════════════════════════════════════════════════

// --- Test FixedPool ---
void test_fixed_pool() {
    std::printf("\n=== FixedPool<double, 1024> ===\n");

    FixedPool<double, 1024> pool;
    std::printf("  capacity:  %zu\n", pool.capacity());
    std::printf("  available: %zu\n", pool.available());

    // Allocate some
    constexpr int ALLOCS = 100;
    double* ptrs[ALLOCS];

    auto t0 = Clock::now();
    for (int i = 0; i < ALLOCS; ++i) {
        ptrs[i] = pool.allocate();
        if (!ptrs[i]) { std::printf("  Pool exhausted at %d!\n", i); break; }
        *ptrs[i] = i * 1.5;
    }
    auto t1 = Clock::now();

    std::printf("  Allocated %d objects in %.2f μs (%.0f ns/alloc)\n",
                ALLOCS,
                std::chrono::duration_cast<ns>(t1 - t0).count() / 1000.0,
                std::chrono::duration_cast<ns>(t1 - t0).count() / (double)ALLOCS);

    // Free some
    auto t2 = Clock::now();
    for (int i = 0; i < ALLOCS; ++i) {
        pool.deallocate(ptrs[i]);
    }
    auto t3 = Clock::now();

    std::printf("  Freed %d objects in %.2f μs (%.0f ns/free)\n",
                ALLOCS,
                std::chrono::duration_cast<ns>(t3 - t2).count() / 1000.0,
                std::chrono::duration_cast<ns>(t3 - t2).count() / (double)ALLOCS);

    std::printf("  available after free: %zu\n", pool.available());
}

// --- Test PMR monotonic_buffer_resource ---
void test_pmr_monotonic() {
    std::printf("\n=== PMR monotonic_buffer_resource ===\n");

    // Stack-allocated buffer — no heap!
    alignas(16) std::byte buf[8192];
    std::pmr::monotonic_buffer_resource pool{buf, sizeof(buf)};

    {
        HeapTracker tracker;

        std::pmr::vector<int> data{&pool};
        data.reserve(100);
        for (int i = 0; i < 100; ++i) {
            data.push_back(i * i);
        }

        std::printf("  Pushed 100 ints into pmr::vector\n");
        std::printf("  Heap allocations during pmr operations: %d\n", tracker.count());
        std::printf("  (should be 0 — all from stack buffer)\n");
    }
}

// --- Test PMR pool_resource ---
void test_pmr_pool() {
    std::printf("\n=== PMR unsynchronized_pool_resource ===\n");

    alignas(16) std::byte buf[32768];
    std::pmr::monotonic_buffer_resource upstream{buf, sizeof(buf)};
    std::pmr::unsynchronized_pool_resource pool{&upstream};

    {
        HeapTracker tracker;

        // Mixed-size allocations — pool handles different size classes
        std::pmr::vector<std::pmr::string> strings{&pool};
        strings.reserve(20);
        for (int i = 0; i < 20; ++i) {
            // Short strings — should use SSO, no allocation from pool
            // Longer strings — allocated from pool
            char tmp[64];
            std::snprintf(tmp, sizeof(tmp), "sensor_reading_%04d_value=%.6f", i, i * 3.14);
            strings.emplace_back(std::pmr::string{tmp, &pool});
        }

        std::printf("  Created 20 pmr::strings in pmr::vector\n");
        std::printf("  Heap allocations: %d\n", tracker.count());
        std::printf("  (should be 0 — all from stack buffer via pool)\n");
    }
}

// --- Benchmark: PMR vs std::vector ---
void benchmark_pmr_vs_std() {
    std::printf("\n=== Benchmark: PMR pool vs std::allocator ===\n");

    constexpr int ITERS = 10'000;
    constexpr int PUSH_COUNT = 50;

    // Benchmark std::vector
    auto t0 = Clock::now();
    for (int i = 0; i < ITERS; ++i) {
        std::vector<int> vec;
        vec.reserve(PUSH_COUNT);
        for (int j = 0; j < PUSH_COUNT; ++j) {
            vec.push_back(j);
        }
    }
    auto t1 = Clock::now();
    auto std_ns = std::chrono::duration_cast<ns>(t1 - t0).count();

    // Benchmark PMR with stack buffer (reuse each iteration)
    alignas(16) std::byte buf[4096];

    auto t2 = Clock::now();
    for (int i = 0; i < ITERS; ++i) {
        std::pmr::monotonic_buffer_resource pool{buf, sizeof(buf)};
        std::pmr::vector<int> vec{&pool};
        vec.reserve(PUSH_COUNT);
        for (int j = 0; j < PUSH_COUNT; ++j) {
            vec.push_back(j);
        }
    }
    auto t3 = Clock::now();
    auto pmr_ns = std::chrono::duration_cast<ns>(t3 - t2).count();

    std::printf("  std::vector:  %8.2f μs per iteration\n",
                std_ns / (1000.0 * ITERS));
    std::printf("  pmr::vector:  %8.2f μs per iteration\n",
                pmr_ns / (1000.0 * ITERS));
    std::printf("  Speedup: %.1fx\n",
                static_cast<double>(std_ns) / pmr_ns);
}

// --- Demonstrate heap detection ---
void test_heap_detection() {
    std::printf("\n=== Heap Allocation Detection ===\n");

    // This should detect allocations
    {
        HeapTracker tracker;
        auto* p = new int[100];
        delete[] p;
        std::vector<int> v;
        v.push_back(42);
        std::printf("  new[] + vector::push_back: %d heap allocations detected\n",
                    tracker.count());
    }

    // PMR should avoid them
    {
        alignas(16) std::byte buf[4096];
        std::pmr::monotonic_buffer_resource pool{buf, sizeof(buf)};

        HeapTracker tracker;
        std::pmr::vector<int> v{&pool};
        v.reserve(100);
        for (int i = 0; i < 100; ++i) {
            v.push_back(i);
        }
        std::printf("  pmr::vector with pool: %d heap allocations detected\n",
                    tracker.count());
    }
}

// ─── Main ────────────────────────────────────────────────────────────
int main() {
    std::printf("╔════════════════════════════════════════════════════╗\n");
    std::printf("║  Exercise 04: Pool Allocator & PMR                ║\n");
    std::printf("╚════════════════════════════════════════════════════╝\n");

    test_fixed_pool();
    test_pmr_monotonic();
    test_pmr_pool();
    benchmark_pmr_vs_std();
    test_heap_detection();

    std::printf("\n─── Key Takeaways ───\n");
    std::printf("1. FixedPool: O(1) alloc/free, zero heap calls, predictable\n");
    std::printf("2. PMR monotonic: fastest for bump-allocate patterns\n");
    std::printf("3. PMR pool: handles mixed sizes without heap\n");
    std::printf("4. Replace operator new to DETECT accidental allocations\n");
    std::printf("5. In RT loops: allocate everything BEFORE the loop starts\n");

    return 0;
}
