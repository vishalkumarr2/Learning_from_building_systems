// ex04_memory_model.cpp — Demonstrate memory reordering with relaxed atomics
// Compile: g++ -std=c++20 -Wall -Wextra -Wpedantic -pthread -O2 ex04_memory_model.cpp -o ex04
//
// KEY INSIGHT: CPUs have store buffers. When Thread 1 writes x=1 then y=1 with
// relaxed ordering, the stores can become visible to other cores in ANY order.
// Thread 2 might see y=1 (from Thread 1's store buffer draining y first) but
// still see x=0 (x hasn't drained from Thread 1's store buffer yet).
//
// On x86 (TSO — Total Store Order): stores are ordered relative to OTHER stores
// from the same core, so the classic x/y reordering is very hard to observe.
// However, x86 CAN reorder loads past earlier stores (StoreLoad reordering).
// We include a variant that's more likely to trigger even on x86.

#include <atomic>
#include <thread>
#include <iostream>
#include <chrono>
#include <cassert>
#include <vector>

// ============================================================================
// Part A: Classic Store-Store / Load-Load reordering (relaxed)
// ============================================================================
// On ARM/POWER this triggers readily. On x86 (TSO) it's rare for store-store
// but we try anyway. The point is: the C++ memory model ALLOWS it.

namespace relaxed_reorder {

std::atomic<int> x{0}, y{0};
// Observed holds: did Thread 2 ever see y=1 but x=0?
std::atomic<bool> observed{false};
std::atomic<uint64_t> iterations{0};

void thread1() {
    x.store(1, std::memory_order_relaxed);  // S1
    y.store(1, std::memory_order_relaxed);  // S2
    // With relaxed, S2 can become visible to Thread 2 BEFORE S1.
    // The CPU's store buffer may drain y before x, or the compiler
    // may reorder the stores entirely.
}

void thread2() {
    int ry = y.load(std::memory_order_relaxed);  // L1
    int rx = x.load(std::memory_order_relaxed);  // L2
    // If ry==1 && rx==0, we've observed reordering:
    // Thread 1's store to y was seen, but store to x was NOT yet visible.
    if (ry == 1 && rx == 0) {
        observed.store(true, std::memory_order_relaxed);
    }
}

uint64_t run(uint64_t max_attempts) {
    for (uint64_t i = 0; i < max_attempts; ++i) {
        x.store(0, std::memory_order_relaxed);
        y.store(0, std::memory_order_relaxed);
        observed.store(false, std::memory_order_relaxed);

        std::thread t1(thread1);
        std::thread t2(thread2);
        t1.join();
        t2.join();

        if (observed.load(std::memory_order_relaxed)) {
            return i + 1;
        }
    }
    return 0; // never observed
}

} // namespace relaxed_reorder

// ============================================================================
// Part B: Variant more likely to trigger on x86 — StoreLoad reordering
// ============================================================================
// x86 CAN reorder a load past an earlier store to a DIFFERENT address.
// This is Dekker's algorithm pattern: each thread stores to its own flag,
// then loads the other's flag. Both can see 0.

namespace storeload_reorder {

std::atomic<int> flag1{0}, flag2{0};
std::atomic<int> r1_val{0}, r2_val{0};
std::atomic<uint64_t> reorder_count{0};

void thread1() {
    flag1.store(1, std::memory_order_relaxed);               // Store to flag1
    r1_val.store(flag2.load(std::memory_order_relaxed),       // Load flag2
                 std::memory_order_relaxed);
}

void thread2() {
    flag2.store(1, std::memory_order_relaxed);               // Store to flag2
    r2_val.store(flag1.load(std::memory_order_relaxed),       // Load flag1
                 std::memory_order_relaxed);
}

// If both r1==0 and r2==0, StoreLoad reordering occurred:
// each thread's load executed before its own store became globally visible.

uint64_t run(uint64_t max_attempts) {
    uint64_t count = 0;
    for (uint64_t i = 0; i < max_attempts; ++i) {
        flag1.store(0, std::memory_order_relaxed);
        flag2.store(0, std::memory_order_relaxed);
        r1_val.store(-1, std::memory_order_relaxed);
        r2_val.store(-1, std::memory_order_relaxed);

        std::thread t1(thread1);
        std::thread t2(thread2);
        t1.join();
        t2.join();

        int r1 = r1_val.load(std::memory_order_relaxed);
        int r2 = r2_val.load(std::memory_order_relaxed);
        if (r1 == 0 && r2 == 0) {
            ++count;
            if (count == 1) {
                std::cout << "  [StoreLoad] First reordering observed at iteration "
                          << (i + 1) << "\n";
            }
        }
    }
    return count;
}

} // namespace storeload_reorder

// ============================================================================
// Part C: FIXED version using acquire/release — no reordering possible
// ============================================================================

namespace acquire_release_fixed {

std::atomic<int> x{0}, y{0};
std::atomic<bool> observed{false};

void thread1() {
    x.store(1, std::memory_order_relaxed);     // S1: relaxed is fine here
    y.store(1, std::memory_order_release);      // S2: release ensures S1 is visible
    // Release on y guarantees: everything before this store (including x=1)
    // is visible to any thread that does an acquire-load of y and sees 1.
}

void thread2() {
    int ry = y.load(std::memory_order_acquire); // L1: acquire pairs with release
    int rx = x.load(std::memory_order_relaxed); // L2: this sees x=1 if ry==1
    // Acquire on y guarantees: if we see y==1 (written by the release-store),
    // then ALL stores before that release (including x=1) are visible to us.
    // So if ry==1, rx MUST be 1. The "impossible" reordering cannot happen.
    if (ry == 1 && rx == 0) {
        observed.store(true, std::memory_order_relaxed);
    }
}

uint64_t run(uint64_t max_attempts) {
    for (uint64_t i = 0; i < max_attempts; ++i) {
        x.store(0, std::memory_order_relaxed);
        y.store(0, std::memory_order_relaxed);
        observed.store(false, std::memory_order_relaxed);

        std::thread t1(thread1);
        std::thread t2(thread2);
        t1.join();
        t2.join();

        if (observed.load(std::memory_order_relaxed)) {
            return i + 1; // Should NEVER happen
        }
    }
    return 0;
}

} // namespace acquire_release_fixed

// ============================================================================
// Main — run all experiments
// ============================================================================

int main() {
    constexpr uint64_t ATTEMPTS = 100'000;

    std::cout << "=== Exercise 04: C++ Memory Model & Reordering ===\n\n";

    // --- Part A: classic relaxed store-store reordering ---
    std::cout << "[Part A] Classic relaxed reordering (x/y pattern)\n";
    std::cout << "  Running " << ATTEMPTS << " attempts...\n";
    auto start = std::chrono::steady_clock::now();
    uint64_t found_at = relaxed_reorder::run(ATTEMPTS);
    auto elapsed = std::chrono::steady_clock::now() - start;
    double ms = std::chrono::duration<double, std::milli>(elapsed).count();

    if (found_at > 0) {
        std::cout << "  OBSERVED reordering after " << found_at << " iterations! ("
                  << ms << " ms)\n";
        std::cout << "  Thread 2 saw y=1 but x=0 — store-store reorder confirmed.\n";
    } else {
        std::cout << "  NOT observed in " << ATTEMPTS << " attempts (" << ms << " ms)\n";
        std::cout << "  (Expected on x86/TSO — stores from one core are ordered.\n";
        std::cout << "   On ARM/POWER this would trigger easily.)\n";
    }

    // --- Part B: StoreLoad reordering (triggers on x86) ---
    std::cout << "\n[Part B] StoreLoad reordering (Dekker pattern — triggers on x86)\n";
    std::cout << "  Running " << ATTEMPTS << " attempts...\n";
    start = std::chrono::steady_clock::now();
    uint64_t sl_count = storeload_reorder::run(ATTEMPTS);
    elapsed = std::chrono::steady_clock::now() - start;
    ms = std::chrono::duration<double, std::milli>(elapsed).count();

    std::cout << "  Observed " << sl_count << " StoreLoad reorderings out of "
              << ATTEMPTS << " attempts (" << ms << " ms)\n";
    if (sl_count > 0) {
        double pct = 100.0 * static_cast<double>(sl_count) / static_cast<double>(ATTEMPTS);
        std::cout << "  Rate: " << pct << "% — x86 store buffers in action!\n";
        std::cout << "  WHY: each core's store sits in a store buffer. The load\n";
        std::cout << "  executes speculatively, reading from cache (which doesn't\n";
        std::cout << "  yet have the other core's store). Both loads return 0.\n";
    } else {
        std::cout << "  (Unusual — try with -O2 and observe on a multi-core machine.)\n";
    }

    // --- Part C: Fixed with acquire/release ---
    std::cout << "\n[Part C] Fixed version (acquire/release)\n";
    std::cout << "  Running " << ATTEMPTS << " attempts...\n";
    start = std::chrono::steady_clock::now();
    uint64_t fixed_at = acquire_release_fixed::run(ATTEMPTS);
    elapsed = std::chrono::steady_clock::now() - start;
    ms = std::chrono::duration<double, std::milli>(elapsed).count();

    if (fixed_at > 0) {
        std::cout << "  BUG: saw reordering at iteration " << fixed_at
                  << " — acquire/release should prevent this!\n";
    } else {
        std::cout << "  No reordering observed in " << ATTEMPTS << " attempts ("
                  << ms << " ms)\n";
        std::cout << "  CORRECT: acquire/release guarantees if y==1 is seen, x==1 is too.\n";
    }

    std::cout << "\n--- Summary ---\n";
    std::cout << "relaxed:         reordering is ALLOWED by the standard\n";
    std::cout << "acquire/release: creates a happens-before edge, preventing it\n";
    std::cout << "seq_cst:         total order across ALL atomics (strongest, slowest)\n";
    std::cout << "\nOn x86 (TSO): store-store and load-load are naturally ordered,\n";
    std::cout << "only StoreLoad reordering occurs. ARM/POWER are much weaker.\n";
    std::cout << "Always program to the C++ memory model, not the hardware model.\n";

    return 0;
}
