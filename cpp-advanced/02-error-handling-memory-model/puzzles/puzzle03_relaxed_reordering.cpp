// puzzle03_relaxed_reordering.cpp
// Compile: g++ -std=c++20 -Wall -Wextra -Wpedantic -pthread -O2 puzzle03_relaxed_reordering.cpp -o puzzle03
//
// PUZZLE: This program uses 4 atomics and 2 threads with memory_order_relaxed.
//         Which combinations of (r1, r2) are possible when both threads complete?
//
//         A) r1=1, r2=1
//         B) r1=0, r2=0
//         C) r1=1, r2=0
//         D) r1=0, r2=1
//
//         Under sequential consistency, which would be IMPOSSIBLE?

#include <atomic>
#include <iostream>
#include <thread>
#include <array>
#include <map>
#include <string>

// Shared state
std::atomic<int> x{0}, y{0};
std::atomic<int> r1{0}, r2{0};

void thread1() {
    // Two stores with relaxed ordering
    x.store(1, std::memory_order_relaxed);  // S1
    r1.store(y.load(std::memory_order_relaxed), std::memory_order_relaxed);  // L1: read y into r1
}

void thread2() {
    // Two stores with relaxed ordering
    y.store(1, std::memory_order_relaxed);  // S2
    r2.store(x.load(std::memory_order_relaxed), std::memory_order_relaxed);  // L2: read x into r2
}

int main() {
    // Run many iterations to try to observe all possible outcomes
    std::map<std::string, int> outcomes;
    constexpr int ITERATIONS = 1'000'000;

    for (int i = 0; i < ITERATIONS; ++i) {
        // Reset
        x.store(0, std::memory_order_relaxed);
        y.store(0, std::memory_order_relaxed);
        r1.store(0, std::memory_order_relaxed);
        r2.store(0, std::memory_order_relaxed);

        std::thread t1(thread1);
        std::thread t2(thread2);
        t1.join();
        t2.join();

        int v1 = r1.load(std::memory_order_relaxed);
        int v2 = r2.load(std::memory_order_relaxed);
        std::string key = "r1=" + std::to_string(v1) + ", r2=" + std::to_string(v2);
        outcomes[key]++;
    }

    std::cout << "Results after " << ITERATIONS << " iterations:\n";
    std::cout << "─────────────────────────────────\n";
    for (const auto& [outcome, count] : outcomes) {
        double pct = 100.0 * count / ITERATIONS;
        std::cout << "  " << outcome << "  →  " << count
                  << " (" << pct << "%)\n";
    }

    std::cout << "\n";
    std::cout << "Under seq_cst, r1=0,r2=0 should be IMPOSSIBLE.\n";
    std::cout << "Under relaxed, ALL four combinations are allowed.\n";

    if (outcomes.count("r1=0, r2=0")) {
        std::cout << "\n★ r1=0,r2=0 WAS observed! Relaxed reordering in action.\n";
    } else {
        std::cout << "\n(r1=0,r2=0 was not observed in this run — try more iterations\n";
        std::cout << " or run on ARM/POWER where relaxed effects are more visible.)\n";
    }
}

/*
 * ═══════════════════════════════════════════════════════
 * ANSWER
 * ═══════════════════════════════════════════════════════
 *
 * The 4 atomics form the classic "store buffering" litmus test:
 *
 *   Thread 1:  x=1; r1=y;    (store x, then load y)
 *   Thread 2:  y=1; r2=x;    (store y, then load x)
 *
 * === Under Sequential Consistency ===
 * There are only 6 total interleavings of the 4 operations:
 *
 *   S1 L1 S2 L2  →  r1=0, r2=1   ✓
 *   S1 S2 L1 L2  →  r1=1, r2=1   ✓
 *   S1 S2 L2 L1  →  r1=1, r2=1   ✓
 *   S2 L2 S1 L1  →  r1=1, r2=0   ✓
 *   S2 S1 L2 L1  →  r1=1, r2=1   ✓
 *   S2 S1 L1 L2  →  r1=1, r2=1   ✓
 *
 * Result: r1=0, r2=0 is IMPOSSIBLE under seq_cst.
 * If thread 1 sees y=0, it means S2 hasn't happened yet,
 * so L2 (which comes after S2) must see x=1 (S1 already happened).
 *
 * === Under Relaxed Ordering ===
 * All four combinations are possible:
 *   r1=0, r2=0  ✓  (the "impossible" one under seq_cst)
 *   r1=0, r2=1  ✓
 *   r1=1, r2=0  ✓
 *   r1=1, r2=1  ✓
 *
 * WHY? With relaxed ordering:
 * - Each thread's store can sit in its store buffer and not be visible
 *   to the other thread yet
 * - Thread 1 stores x=1 (in its store buffer), then reads y=0
 *   (y=1 from thread 2 hasn't propagated yet)
 * - Thread 2 stores y=1 (in its store buffer), then reads x=0
 *   (x=1 from thread 1 hasn't propagated yet)
 * - Both loads "run ahead" of the other thread's store becoming visible
 *
 * This is a real hardware behavior on x86 (store-buffer forwarding)
 * and even more common on ARM/POWER (weaker memory models).
 *
 * On x86 you'll typically see r1=0,r2=0 rarely (< 0.01%) because
 * x86 has a strong memory model (TSO), but it CAN happen.
 * On ARM you'll see it much more frequently.
 *
 * NOTE: This is the exact same litmus test used in the C++ standard
 * to motivate the need for memory_order_seq_cst.
 */
