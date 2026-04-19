// ex07_spsc_queue.cpp — Lock-Free SPSC Ring Buffer (Week 2 Mini-Project)
// Compile: g++ -std=c++20 -Wall -Wextra -Wpedantic -pthread -O2 ex07_spsc_queue.cpp -o ex07
// TSan:    g++ -std=c++20 -Wall -Wextra -Wpedantic -pthread -fsanitize=thread -O1 ex07_spsc_queue.cpp -o ex07_tsan && ./ex07_tsan
//
// ROS NOTE: This is the sensor→controller data path in your robot.
// The sensorbar SPI reader (producer) pushes readings at ~200Hz,
// the navigation estimator (consumer) pops them for state estimation.
// A lock-free SPSC queue ensures the producer (RT thread) never blocks
// on a mutex contention with the consumer (non-RT thread).

#include <atomic>
#include <thread>
#include <mutex>
#include <queue>
#include <iostream>
#include <chrono>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <new>         // hardware_destructive_interference_size
#include <optional>
#include <vector>
#include <numeric>

// ============================================================================
// Cache-line size for padding (avoid false sharing)
// ============================================================================
#ifdef __cpp_lib_hardware_interference_size
    constexpr size_t CACHE_LINE = std::hardware_destructive_interference_size;
#else
    constexpr size_t CACHE_LINE = 64;
#endif

// ============================================================================
// SPSCQueue<T, N> — Lock-Free Single-Producer Single-Consumer Ring Buffer
// ============================================================================
// - N must be a power of 2 (for fast modulo via bitmask)
// - head_: written by consumer, read by producer (to check "full")
// - tail_: written by producer, read by consumer (to check "empty")
// - ONLY acquire/release ordering — NO seq_cst
//
// Memory layout:
//   [head_ (consumer writes)]  [pad]  [tail_ (producer writes)]  [pad]  [buffer]
// The padding prevents false sharing between producer and consumer.

template <typename T, size_t N>
class SPSCQueue {
    static_assert((N & (N - 1)) == 0, "N must be a power of 2");
    static_assert(N >= 2, "N must be at least 2");

    // Consumer owns head_: the index it will read from next
    alignas(CACHE_LINE) std::atomic<size_t> head_{0};

    // Producer owns tail_: the index it will write to next
    alignas(CACHE_LINE) std::atomic<size_t> tail_{0};

    // Ring buffer storage — aligned for cache friendliness
    alignas(CACHE_LINE) T buffer_[N];

    static constexpr size_t MASK = N - 1;

public:
    SPSCQueue() = default;

    // Non-copyable, non-movable (contains atomics)
    SPSCQueue(const SPSCQueue&) = delete;
    SPSCQueue& operator=(const SPSCQueue&) = delete;

    // --- Producer interface (call from ONE thread only) ---

    bool try_push(const T& value) {
        const size_t tail = tail_.load(std::memory_order_relaxed);  // we own tail_
        const size_t next_tail = (tail + 1) & MASK;

        // Check if full: next write position == consumer's read position
        if (next_tail == head_.load(std::memory_order_acquire)) {
            return false;  // queue is full
        }

        buffer_[tail] = value;

        // Release: ensures the buffer write above is visible before
        // the consumer sees the updated tail_.
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    // Batch push: pushes up to count items, returns number actually pushed
    size_t try_push_batch(const T* items, size_t count) {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        const size_t head = head_.load(std::memory_order_acquire);

        // Available space (ring buffer: capacity is N-1 to distinguish full/empty)
        const size_t available = (head - tail - 1) & MASK;
        const size_t to_push = std::min(count, available);

        if (to_push == 0) return 0;

        for (size_t i = 0; i < to_push; ++i) {
            buffer_[(tail + i) & MASK] = items[i];
        }

        // Single release store after all writes — consumer sees everything at once
        tail_.store((tail + to_push) & MASK, std::memory_order_release);
        return to_push;
    }

    // --- Consumer interface (call from ONE thread only) ---

    bool try_pop(T& value) {
        const size_t head = head_.load(std::memory_order_relaxed);  // we own head_

        // Check if empty: our read position == producer's write position
        if (head == tail_.load(std::memory_order_acquire)) {
            return false;  // queue is empty
        }

        value = buffer_[head];

        // Release: ensures the buffer read above completes before
        // we advance head_ (so producer doesn't overwrite what we're reading).
        head_.store((head + 1) & MASK, std::memory_order_release);
        return true;
    }

    // Batch pop: pops up to count items, returns number actually popped
    size_t try_pop_batch(T* items, size_t count) {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t tail = tail_.load(std::memory_order_acquire);

        const size_t available = (tail - head) & MASK;
        const size_t to_pop = std::min(count, available);

        if (to_pop == 0) return 0;

        for (size_t i = 0; i < to_pop; ++i) {
            items[i] = buffer_[(head + i) & MASK];
        }

        head_.store((head + to_pop) & MASK, std::memory_order_release);
        return to_pop;
    }

    size_t capacity() const { return N - 1; }

    size_t size_approx() const {
        // Approximate — races are fine for diagnostics
        return (tail_.load(std::memory_order_relaxed)
              - head_.load(std::memory_order_relaxed)) & MASK;
    }
};

// ============================================================================
// Mutex-based queue wrapper for comparison
// ============================================================================

template <typename T>
class MutexQueue {
    std::queue<T> q_;
    std::mutex mtx_;
    size_t cap_;

public:
    explicit MutexQueue(size_t capacity) : cap_(capacity) {}

    bool try_push(const T& value) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (q_.size() >= cap_) return false;
        q_.push(value);
        return true;
    }

    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (q_.empty()) return false;
        value = q_.front();
        q_.pop();
        return true;
    }
};

// ============================================================================
// Latency measurement helper
// ============================================================================

struct LatencyStats {
    double avg_ns;
    double min_ns;
    double max_ns;
    uint64_t total_ops;
    double total_secs;
    uint64_t ops_per_sec;
};

// ============================================================================
// Benchmark: SPSC queue throughput and latency
// ============================================================================

template <typename Queue>
LatencyStats benchmark_spsc(Queue& q, uint64_t num_items,
                             bool measure_latency = false) {
    // Latency buffer — only used when measure_latency is true
    // We sample every 100th operation to reduce overhead
    constexpr int SAMPLE_INTERVAL = 100;
    std::vector<double> latencies;
    if (measure_latency) {
        latencies.reserve(static_cast<size_t>(num_items / SAMPLE_INTERVAL));
    }

    std::atomic<bool> producer_done{false};
    std::atomic<uint64_t> consumed{0};

    auto start = std::chrono::steady_clock::now();

    // Producer thread
    std::thread producer([&]() {
        for (uint64_t i = 0; i < num_items; ++i) {
            while (!q.try_push(static_cast<int>(i))) {
                // Busy wait — queue is full, consumer needs to catch up
                // In a real system, you'd yield or back off
            }
        }
        producer_done.store(true, std::memory_order_release);
    });

    // Consumer thread
    std::thread consumer([&]() {
        int value;
        uint64_t count = 0;
        uint64_t sample_counter = 0;

        while (count < num_items) {
            auto t0 = std::chrono::steady_clock::now();
            if (q.try_pop(value)) {
                auto t1 = std::chrono::steady_clock::now();

                if (measure_latency && (sample_counter % SAMPLE_INTERVAL == 0)) {
                    double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
                    latencies.push_back(ns);
                }
                ++sample_counter;
                ++count;
            }
        }
        consumed.store(count, std::memory_order_release);
    });

    producer.join();
    consumer.join();

    auto end = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(end - start).count();
    uint64_t ops = consumed.load(std::memory_order_relaxed);

    LatencyStats stats{};
    stats.total_ops = ops;
    stats.total_secs = secs;
    stats.ops_per_sec = static_cast<uint64_t>(static_cast<double>(ops) / secs);

    if (measure_latency && !latencies.empty()) {
        double sum = 0;
        double mn = latencies[0], mx = latencies[0];
        for (double lat : latencies) {
            sum += lat;
            mn = std::min(mn, lat);
            mx = std::max(mx, lat);
        }
        stats.avg_ns = sum / static_cast<double>(latencies.size());
        stats.min_ns = mn;
        stats.max_ns = mx;
    } else {
        // Compute avg from total time
        stats.avg_ns = (secs * 1e9) / static_cast<double>(ops);
        stats.min_ns = stats.avg_ns;  // approximate
        stats.max_ns = stats.avg_ns;
    }

    return stats;
}

void print_stats(const char* label, const LatencyStats& s) {
    std::cout << "  " << label << ":\n";
    std::cout << "    ops/sec:     " << s.ops_per_sec << "\n";
    std::cout << "    avg latency: " << s.avg_ns << " ns/op\n";
    std::cout << "    min latency: " << s.min_ns << " ns\n";
    std::cout << "    max latency: " << s.max_ns << " ns\n";
    std::cout << "    total:       " << s.total_ops << " ops in "
              << (s.total_secs * 1000.0) << " ms\n";
}

// ============================================================================
// Correctness test: verify no data loss or corruption
// ============================================================================

void test_correctness() {
    constexpr size_t CAP = 1024;  // power of 2
    constexpr uint64_t NUM = 1'000'000;

    SPSCQueue<uint64_t, CAP> q;
    std::atomic<bool> pass{true};

    std::thread producer([&]() {
        for (uint64_t i = 0; i < NUM; ++i) {
            while (!q.try_push(i)) {}
        }
    });

    std::thread consumer([&]() {
        uint64_t expected = 0;
        while (expected < NUM) {
            uint64_t val;
            if (q.try_pop(val)) {
                if (val != expected) {
                    pass.store(false, std::memory_order_relaxed);
                    std::cerr << "  MISMATCH: expected " << expected
                              << ", got " << val << "\n";
                    return;
                }
                ++expected;
            }
        }
    });

    producer.join();
    consumer.join();

    bool ok = pass.load();
    std::cout << "  FIFO ordering test (1M items): "
              << (ok ? "✓ PASS" : "✗ FAIL") << "\n";
}

void test_batch_ops() {
    constexpr size_t CAP = 256;
    SPSCQueue<int, CAP> q;

    // Push a batch
    int items[] = {10, 20, 30, 40, 50};
    size_t pushed = q.try_push_batch(items, 5);
    assert(pushed == 5);

    // Pop a batch
    int out[5] = {};
    size_t popped = q.try_pop_batch(out, 5);
    assert(popped == 5);

    bool ok = true;
    for (int i = 0; i < 5; ++i) {
        if (out[i] != items[i]) { ok = false; break; }
    }
    std::cout << "  Batch push/pop test: " << (ok ? "✓ PASS" : "✗ FAIL") << "\n";
}

void test_full_empty() {
    constexpr size_t CAP = 8;  // actual usable = 7
    SPSCQueue<int, CAP> q;

    // Fill to capacity
    int pushed = 0;
    for (int i = 0; i < 100; ++i) {
        if (q.try_push(i)) ++pushed;
        else break;
    }
    bool ok = (pushed == 7);  // N-1 = 7 usable slots
    std::cout << "  Full test (cap=8, usable=7, pushed=" << pushed << "): "
              << (ok ? "✓ PASS" : "✗ FAIL") << "\n";

    // Verify pop returns all 7
    int popped = 0;
    int val;
    while (q.try_pop(val)) ++popped;
    ok = (popped == 7);
    std::cout << "  Drain test (popped=" << popped << "): "
              << (ok ? "✓ PASS" : "✗ FAIL") << "\n";

    // Now empty — pop should fail
    ok = !q.try_pop(val);
    std::cout << "  Empty test: " << (ok ? "✓ PASS" : "✗ FAIL") << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    constexpr uint64_t NUM_ITEMS = 10'000'000;
    constexpr size_t QUEUE_CAP = 8192;  // power of 2

    std::cout << "=== Exercise 07: Lock-Free SPSC Queue (Mini-Project) ===\n";
    std::cout << "// This is the sensor→controller data path in your robot.\n\n";

    // --- Correctness ---
    std::cout << "[Correctness Tests]\n";
    test_full_empty();
    test_batch_ops();
    test_correctness();

    // --- Benchmark ---
    std::cout << "\n[Benchmark] " << NUM_ITEMS
              << " items, queue capacity " << QUEUE_CAP << "\n\n";

    // SPSC lock-free queue
    {
        SPSCQueue<int, QUEUE_CAP> q;
        auto stats = benchmark_spsc(q, NUM_ITEMS, true);
        print_stats("SPSCQueue (lock-free, acq/rel)", stats);
    }

    std::cout << "\n";

    // Mutex-based queue
    {
        MutexQueue<int> q(QUEUE_CAP - 1);
        auto stats = benchmark_spsc(q, NUM_ITEMS, true);
        print_stats("std::queue + std::mutex", stats);
    }

    // --- Analysis ---
    std::cout << "\n--- Design Notes ---\n";
    std::cout << "1. Only acquire/release — no seq_cst. This is sufficient because:\n";
    std::cout << "   - Producer only writes tail_, reads head_ (to check full)\n";
    std::cout << "   - Consumer only writes head_, reads tail_ (to check empty)\n";
    std::cout << "   - No third party needs a total order across both\n";
    std::cout << "\n2. Power-of-2 size: modulo via bitmask (& MASK) is one cycle.\n";
    std::cout << "   Division-based modulo would add ~20 cycles per operation.\n";
    std::cout << "\n3. Cache-line alignment: head_ and tail_ on separate lines\n";
    std::cout << "   prevents false sharing (producer and consumer don't bounce\n";
    std::cout << "   each other's cache lines).\n";
    std::cout << "\n4. One empty slot: we sacrifice one slot to distinguish full\n";
    std::cout << "   from empty (both would have head==tail otherwise).\n";
    std::cout << "\n5. ThreadSanitizer clean: compile with -fsanitize=thread to verify.\n";
    std::cout << "   The only shared state is two atomics with proper ordering.\n";

    return 0;
}
