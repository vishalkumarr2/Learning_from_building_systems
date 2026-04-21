// Module 16: Sanitizer Workshop
// Puzzle 01: False Positive / Suppression Challenge
//
// This code uses std::atomic with relaxed memory ordering in a pattern that
// TSan may flag as a data race. The challenge:
//
//   1. Understand WHY TSan flags this (and whether it's truly safe)
//   2. Write a suppression file to silence the TSan report
//   3. Discuss: when are suppressions justified vs hiding real bugs?
//
// Build:
//   g++ -std=c++2a -fsanitize=thread -g -O1 puzzle01_false_positive.cpp -o puzzle01_tsan -pthread
//   TSAN_OPTIONS="suppressions=puzzle01.supp" ./puzzle01_tsan
//
// Context:
//   This pattern is common in performance counters, stats gathering, and
//   approximate algorithms where exact ordering doesn't matter.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <thread>
#include <vector>

// ============================================================
// Pattern: Seqlock (simplified)
// A seqlock allows a single writer and multiple readers without blocking.
// The writer increments a sequence counter before and after writing.
// Readers retry if they observe an odd sequence (write in progress)
// or if the sequence changed during their read.
//
// This is a CORRECT lock-free pattern but TSan flags it because:
// - The data members (x_, y_) are read/written without a mutex
// - TSan doesn't understand the seqlock protocol provides safety
// - The relaxed/acquire/release ordering is correct but invisible to TSan
// ============================================================
class SeqLock {
public:
    SeqLock() : seq_(0), x_(0.0), y_(0.0) {}

    // Writer: single-threaded (only one writer allowed)
    void write(double x, double y) {
        auto s = seq_.load(std::memory_order_relaxed);
        seq_.store(s + 1, std::memory_order_release);  // Odd = write in progress

        // TSan flags these as races with the reader
        x_ = x;
        y_ = y;

        seq_.store(s + 2, std::memory_order_release);  // Even = write complete
    }

    // Reader: can be called from multiple threads
    // Returns false if read was inconsistent (caller should retry)
    bool try_read(double& x, double& y) const {
        auto s1 = seq_.load(std::memory_order_acquire);
        if (s1 & 1) return false;  // Write in progress

        // TSan flags these as races with the writer
        x = x_;
        y = y_;

        auto s2 = seq_.load(std::memory_order_acquire);
        return s1 == s2;  // Consistent if sequence didn't change
    }

    // Safe read: retries until consistent
    void read(double& x, double& y) const {
        while (!try_read(x, y)) {
            // Spin until we get a consistent snapshot
        }
    }

private:
    std::atomic<std::uint64_t> seq_;
    // These are intentionally NOT atomic — protected by the seqlock protocol
    // TSan doesn't understand this pattern and flags races on these members
    double x_;
    double y_;
};

// ============================================================
// Second pattern: approximate counter
// Multiple threads increment different cache lines, main thread
// reads the sum. No single read is "wrong" — just approximate.
// TSan flags the non-atomic counter increments.
// ============================================================
struct alignas(64) PaddedCounter {
    std::atomic<std::uint64_t> value{0};
};

class ApproximateCounter {
public:
    explicit ApproximateCounter(int num_threads)
        : counters_(static_cast<std::size_t>(num_threads)) {}

    // Each thread increments its own slot (no contention)
    void increment(int thread_id) {
        counters_[static_cast<std::size_t>(thread_id)].value.fetch_add(
            1, std::memory_order_relaxed);
    }

    // Reader gets approximate sum (may miss recent increments — that's OK)
    std::uint64_t approximate_sum() const {
        std::uint64_t sum = 0;
        for (const auto& c : counters_) {
            sum += c.value.load(std::memory_order_relaxed);
        }
        return sum;
    }

private:
    std::vector<PaddedCounter> counters_;
};

// ============================================================
// Demo: run both patterns
// ============================================================
int main() {
    std::cout << "=== Puzzle 01: SeqLock + ApproximateCounter ===\n\n";

    // --- SeqLock demo ---
    SeqLock lock;

    std::thread writer([&lock]() {
        for (int i = 0; i < 100000; ++i) {
            lock.write(static_cast<double>(i), static_cast<double>(i) * 2.0);
        }
    });

    std::atomic<int> successful_reads{0};

    std::thread reader1([&lock, &successful_reads]() {
        double x = 0, y = 0;
        for (int i = 0; i < 100000; ++i) {
            lock.read(x, y);
            // Verify consistency: y should always be 2*x
            if (static_cast<int>(y) == static_cast<int>(x) * 2) {
                successful_reads.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    std::thread reader2([&lock, &successful_reads]() {
        double x = 0, y = 0;
        for (int i = 0; i < 100000; ++i) {
            lock.read(x, y);
            if (static_cast<int>(y) == static_cast<int>(x) * 2) {
                successful_reads.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    writer.join();
    reader1.join();
    reader2.join();

    std::cout << "SeqLock: " << successful_reads.load()
              << "/200000 consistent reads\n";

    // --- ApproximateCounter demo ---
    constexpr int num_threads = 4;
    ApproximateCounter counter(num_threads);

    std::vector<std::thread> workers;
    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&counter, t]() {
            for (int i = 0; i < 100000; ++i) {
                counter.increment(t);
            }
        });
    }

    // Read approximate sum while workers are running
    std::thread summer([&counter]() {
        for (int i = 0; i < 100; ++i) {
            auto sum = counter.approximate_sum();
            (void)sum;
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    for (auto& w : workers) w.join();
    summer.join();

    std::cout << "ApproximateCounter final sum: " << counter.approximate_sum()
              << " (expected " << (num_threads * 100000) << ")\n";

    // ============================================================
    // DISCUSSION:
    //
    // The SeqLock pattern IS flagged by TSan because:
    // - x_ and y_ are plain doubles accessed from multiple threads
    // - TSan doesn't understand that the seq_ counter provides safety
    //
    // Is a suppression justified here?
    // YES — if you've verified the algorithm is correct (single writer,
    //        readers retry on inconsistency).
    // NO  — if you're not 100% sure the seqlock protocol is correctly
    //        implemented (missing memory barriers, multiple writers, etc.)
    //
    // The ApproximateCounter uses proper atomics and should NOT trigger TSan.
    // If it does, it's likely a TSan bug (very rare).
    //
    // SUPPRESSION FILE (puzzle01.supp):
    // ---
    // race:SeqLock::write
    // race:SeqLock::try_read
    // ---
    //
    // Alternative to suppression: use TSAN annotations:
    //   __tsan_acquire(&seq_) / __tsan_release(&seq_)
    //   These tell TSan about the happens-before relationship.
    //
    // Best practice: PREFER fixing the code over suppressing.
    // Only suppress when you have a PROVEN correct algorithm that TSan
    // can't model (seqlocks, custom atomics, RCU, hazard pointers).
    // ============================================================

    std::cout << "\n=== Challenge ===\n";
    std::cout << "1. Run with TSan: see the reports on SeqLock\n";
    std::cout << "2. Create puzzle01.supp with appropriate suppressions\n";
    std::cout << "3. Run with TSAN_OPTIONS=\"suppressions=puzzle01.supp\"\n";
    std::cout << "4. Verify the counter is still correct (consistent reads)\n";

    return 0;
}
