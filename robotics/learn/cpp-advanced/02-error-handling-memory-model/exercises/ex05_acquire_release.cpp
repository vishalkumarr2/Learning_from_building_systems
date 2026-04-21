// ex05_acquire_release.cpp — Spinlock and Sequence Lock using only atomics
// Compile: g++ -std=c++20 -Wall -Wextra -Wpedantic -pthread -O2 ex05_acquire_release.cpp -o ex05
//
// Part A: Spinlock with atomic_flag (test_and_set / clear)
// Part B: Sequence Lock (seqlock) — readers never block, just retry
// Benchmark: both vs std::mutex, 90% read / 10% write, 4 threads, 1M ops each

#include <atomic>
#include <thread>
#include <mutex>
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <cstdint>

// ============================================================================
// Part A: Spinlock with backoff
// ============================================================================

class Spinlock {
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;

    static constexpr int SPIN_LIMIT = 64;

public:
    void lock() {
        int spins = 0;
        // test_and_set returns the OLD value. If false, we acquired the lock.
        while (flag_.test_and_set(std::memory_order_acquire)) {
            ++spins;
            if (spins >= SPIN_LIMIT) {
                // Back off: yield the CPU to reduce contention and power waste.
                // In a real system, you'd use exponential backoff or PAUSE/WFE.
                std::this_thread::yield();
                spins = 0;
            }
            // Optimization note: in C++20 (GCC 10+), you can spin on
            // flag_.test(relaxed) before retrying TAS — reduces cache-line
            // bouncing. With GCC 9, atomic_flag::test() isn't available,
            // so we just yield and retry TAS.
        }
    }

    void unlock() {
        flag_.clear(std::memory_order_release);
        // Release ensures all stores before unlock() are visible to the
        // next thread that acquires the lock.
    }

    bool try_lock() {
        return !flag_.test_and_set(std::memory_order_acquire);
    }
};

// ============================================================================
// Part B: Sequence Lock (SeqLock)
// ============================================================================
// Writers take exclusive access (only ONE writer at a time — use external mutex
// if multiple writers exist). They bump the sequence to an odd number (writing),
// do the write, then bump to even (stable).
//
// Readers NEVER block. They read the sequence, read the data, read the sequence
// again. If the sequence changed or is odd, they retry. This is wait-free for
// readers when writers are rare.

struct Point3D {
    double x, y, z;
};

class SeqLock {
    // Sequence counter: odd = write in progress, even = stable.
    // We use acquire on the first fetch_add and release on the second.
    // Acquire prevents subsequent data stores from being reordered before
    // the sequence increment (readers see odd seq before any data changes).
    // Release prevents preceding data stores from being reordered after
    // the sequence increment (readers see all data changes before even seq).
    std::atomic<uint32_t> seq_{0};

    // Data members MUST be atomic to avoid UB from concurrent read/write.
    // In C, you'd use volatile + compiler barriers, but C++ requires atomic
    // or mutex for any concurrent access (§6.9.2.1 data race = UB).
    std::atomic<double> dx_{0.0}, dy_{0.0}, dz_{0.0};

    // Writer mutual exclusion (seqlock only handles reader/writer;
    // multiple writers need their own lock)
    Spinlock writer_lock_;

public:
    void write(const Point3D& p) {
        writer_lock_.lock();

        // Odd sequence = write in progress.
        // Acquire prevents subsequent data stores from being reordered before
        // this increment — readers see odd seq before any partial data update.
        seq_.fetch_add(1, std::memory_order_acquire);  // now odd

        // Data stores use relaxed — bounded by the seq_cst fence below.
        dx_.store(p.x, std::memory_order_relaxed);
        dy_.store(p.y, std::memory_order_relaxed);
        dz_.store(p.z, std::memory_order_relaxed);

        // Release: guarantees data stores above are visible before seq becomes even.
        seq_.fetch_add(1, std::memory_order_release);  // now even

        writer_lock_.unlock();
    }

    Point3D read() const {
        Point3D result;
        // CRITICAL: do NOT use do-while + continue here!
        // In a do { ... continue; } while(cond), continue evaluates the
        // while-condition with stale variables from previous iterations.
        // Use for(;;) + break instead.
        for (;;) {
            // Acquire: pairs with writer's release of even seq, ensuring
            // data stores completed before seq went even.
            uint32_t s1 = seq_.load(std::memory_order_acquire);
            if (s1 & 1) continue;  // writer active, retry from top

            result.x = dx_.load(std::memory_order_relaxed);
            result.y = dy_.load(std::memory_order_relaxed);
            result.z = dz_.load(std::memory_order_relaxed);

            // Acquire: pairs with writer's second fetch_add (release).
            // Ensures data loads BELOW this point see the writer's stores.
            // Note: on the abstract machine, the relaxed data loads ABOVE
            // are not formally prevented from being reordered past this acquire.
            // This works correctly on x86 (TSO forbids load-load reordering)
            // and ARM (LDAR provides a full load barrier). For a fully portable
            // version, use seq_cst or an acquire fence before reading s2.
            uint32_t s2 = seq_.load(std::memory_order_acquire);
            if (s1 == s2) break;  // stable read
        }
        return result;
    }

    // Debug version for the correctness test
    // (same as read(), but exposes s1/s2 for diagnostics)
    Point3D read_debug(uint32_t& out_s1, uint32_t& out_s2) const {
        Point3D result;
        for (;;) {
            uint32_t s1 = seq_.load(std::memory_order_acquire);
            if (s1 & 1) continue;

            result.x = dx_.load(std::memory_order_relaxed);
            result.y = dy_.load(std::memory_order_relaxed);
            result.z = dz_.load(std::memory_order_relaxed);

            uint32_t s2 = seq_.load(std::memory_order_acquire);
            if (s1 == s2) {
                out_s1 = s1;
                out_s2 = s2;
                break;
            }
        }
        return result;
    }
};

// ============================================================================
// Mutex wrapper for comparison
// ============================================================================

class MutexProtected {
    mutable std::mutex mtx_;
    Point3D data_{0.0, 0.0, 0.0};

public:
    void write(const Point3D& p) {
        std::lock_guard<std::mutex> lk(mtx_);
        data_ = p;
    }

    Point3D read() const {
        std::lock_guard<std::mutex> lk(mtx_);
        return data_;
    }
};

// ============================================================================
// Spinlock wrapper for the same interface
// ============================================================================

class SpinlockProtected {
    mutable Spinlock lock_;
    Point3D data_{0.0, 0.0, 0.0};

public:
    void write(const Point3D& p) {
        lock_.lock();
        data_ = p;
        lock_.unlock();
    }

    Point3D read() const {
        lock_.lock();
        Point3D result = data_;
        lock_.unlock();
        return result;
    }
};

// ============================================================================
// Benchmark harness
// ============================================================================

template <typename Lock>
uint64_t benchmark(Lock& lock, int num_threads, int ops_per_thread, int write_pct) {
    std::atomic<uint64_t> total_ops{0};
    std::vector<std::thread> threads;

    auto start = std::chrono::steady_clock::now();

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&, t]() {
            // Per-thread RNG seeded with thread index for reproducibility
            std::mt19937 rng(static_cast<uint32_t>(t * 7919 + 42));
            std::uniform_int_distribution<int> dist(1, 100);
            uint64_t ops = 0;

            for (int i = 0; i < ops_per_thread; ++i) {
                if (dist(rng) <= write_pct) {
                    // Write (10% of the time)
                    double v = static_cast<double>(i);
                    lock.write({v, v * 2.0, v * 3.0});
                } else {
                    // Read (90% of the time)
                    Point3D p = lock.read();
                    // Use volatile to prevent optimizer from eliding the read
                    volatile double sink = p.x + p.y + p.z;
                    (void)sink;
                }
                ++ops;
            }
            total_ops.fetch_add(ops, std::memory_order_relaxed);
        });
    }

    for (auto& t : threads) t.join();

    auto elapsed = std::chrono::steady_clock::now() - start;
    double secs = std::chrono::duration<double>(elapsed).count();
    uint64_t ops = total_ops.load(std::memory_order_relaxed);
    uint64_t throughput = static_cast<uint64_t>(static_cast<double>(ops) / secs);

    return throughput;
}

// ============================================================================
// Correctness test: verify seqlock never returns torn data
// ============================================================================

void test_seqlock_consistency() {
    SeqLock sl;
    std::atomic<bool> done{false};
    std::atomic<uint64_t> torn_count{0};
    std::atomic<uint64_t> read_count{0};

    // Writer: writes consistent triples (v, 2v, 3v)
    std::thread writer([&]() {
        for (int i = 1; i <= 500'000; ++i) {
            double v = static_cast<double>(i);
            sl.write({v, v * 2.0, v * 3.0});
        }
        done.store(true, std::memory_order_release);
    });

    // Readers: verify the invariant y == 2*x && z == 3*x
    std::vector<std::thread> readers;
    for (int r = 0; r < 3; ++r) {
        readers.emplace_back([&]() {
            uint64_t local_read = 0;
            while (!done.load(std::memory_order_acquire) || local_read < 100) {
                uint32_t dbg_s1, dbg_s2;
                Point3D p = sl.read_debug(dbg_s1, dbg_s2);
                ++local_read;
                // Check consistency: y should be 2*x, z should be 3*x
                if (p.x != 0.0) {
                    double expected_y = p.x * 2.0;
                    double expected_z = p.x * 3.0;
                    if (p.y != expected_y || p.z != expected_z) {
                        uint64_t prev = torn_count.fetch_add(1, std::memory_order_relaxed);
                        if (prev < 5) {
                            // Print first few torn reads for debugging
                            fprintf(stderr, "    TORN: x=%.0f y=%.0f z=%.0f "
                                    "(expected y=%.0f z=%.0f) s1=%u s2=%u\n",
                                    p.x, p.y, p.z, expected_y, expected_z,
                                    dbg_s1, dbg_s2);
                        }
                    }
                }
            }
            read_count.fetch_add(local_read, std::memory_order_relaxed);
        });
    }

    writer.join();
    for (auto& r : readers) r.join();

    uint64_t torn = torn_count.load();
    uint64_t reads = read_count.load();
    std::cout << "  SeqLock consistency test: " << reads << " reads, "
              << torn << " torn" << (torn == 0 ? " ✓ PASS" : " ✗ FAIL") << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    constexpr int NUM_THREADS = 4;
    constexpr int OPS_PER_THREAD = 1'000'000;
    constexpr int WRITE_PCT = 10;  // 90% read, 10% write

    std::cout << "=== Exercise 05: Acquire/Release — Spinlock & SeqLock ===\n\n";

    // --- Correctness test ---
    std::cout << "[Correctness]\n";
    test_seqlock_consistency();

    // --- Benchmark ---
    std::cout << "\n[Benchmark] " << NUM_THREADS << " threads, "
              << OPS_PER_THREAD << " ops/thread, "
              << WRITE_PCT << "% write / " << (100 - WRITE_PCT) << "% read\n\n";

    {
        MutexProtected mp;
        uint64_t tp = benchmark(mp, NUM_THREADS, OPS_PER_THREAD, WRITE_PCT);
        std::cout << "  std::mutex:      " << tp << " ops/sec\n";
    }
    {
        SpinlockProtected sp;
        uint64_t tp = benchmark(sp, NUM_THREADS, OPS_PER_THREAD, WRITE_PCT);
        std::cout << "  Spinlock:        " << tp << " ops/sec\n";
    }
    {
        SeqLock sl;
        uint64_t tp = benchmark(sl, NUM_THREADS, OPS_PER_THREAD, WRITE_PCT);
        std::cout << "  SeqLock:         " << tp << " ops/sec\n";
    }

    std::cout << "\n--- Analysis ---\n";
    std::cout << "SeqLock shines at high read ratios because readers never block.\n";
    std::cout << "Spinlock is fast for low-contention short critical sections.\n";
    std::cout << "std::mutex is general-purpose — it parks threads (good for long waits).\n";
    std::cout << "\nIn robotics: seqlock is ideal for sensor data (many readers, one writer).\n";
    std::cout << "The sensorbar publisher writes at 200Hz, multiple consumers read.\n";

    return 0;
}
