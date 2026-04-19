// puzzle04_seq_cst_vs_acq_rel.cpp
// Compile: g++ -std=c++20 -Wall -Wextra -Wpedantic -pthread -O2 puzzle04_seq_cst_vs_acq_rel.cpp -o puzzle04
//
// PUZZLE: The IRIW (Independent Reads of Independent Writes) problem.
//
//   Thread A: writes x=1
//   Thread B: writes y=1
//   Thread C: reads x, then reads y
//   Thread D: reads y, then reads x
//
//   QUESTION: Can Thread C see x=1,y=0 AND Thread D see y=1,x=0 simultaneously?
//   i.e., can they disagree about the order in which x and y were written?
//
//   Answer SEPARATELY for:
//     (a) memory_order_seq_cst
//     (b) memory_order_acq_rel (acquire loads, release stores)

#include <atomic>
#include <iostream>
#include <thread>
#include <map>
#include <string>

// ── seq_cst version ──────────────────────────────────
namespace seq_cst_test {

std::atomic<int> x{0}, y{0};

// Results from reader threads
std::atomic<int> c_saw_x{0}, c_saw_y{0};
std::atomic<int> d_saw_y{0}, d_saw_x{0};

void writer_x() { x.store(1, std::memory_order_seq_cst); }
void writer_y() { y.store(1, std::memory_order_seq_cst); }

void reader_c() {
    c_saw_x.store(x.load(std::memory_order_seq_cst), std::memory_order_relaxed);
    c_saw_y.store(y.load(std::memory_order_seq_cst), std::memory_order_relaxed);
}

void reader_d() {
    d_saw_y.store(y.load(std::memory_order_seq_cst), std::memory_order_relaxed);
    d_saw_x.store(x.load(std::memory_order_seq_cst), std::memory_order_relaxed);
}

void reset() {
    x = 0; y = 0;
    c_saw_x = 0; c_saw_y = 0;
    d_saw_y = 0; d_saw_x = 0;
}

} // namespace seq_cst_test

// ── acquire/release version ──────────────────────────
namespace acq_rel_test {

std::atomic<int> x{0}, y{0};

std::atomic<int> c_saw_x{0}, c_saw_y{0};
std::atomic<int> d_saw_y{0}, d_saw_x{0};

void writer_x() { x.store(1, std::memory_order_release); }
void writer_y() { y.store(1, std::memory_order_release); }

void reader_c() {
    c_saw_x.store(x.load(std::memory_order_acquire), std::memory_order_relaxed);
    c_saw_y.store(y.load(std::memory_order_acquire), std::memory_order_relaxed);
}

void reader_d() {
    d_saw_y.store(y.load(std::memory_order_acquire), std::memory_order_relaxed);
    d_saw_x.store(x.load(std::memory_order_acquire), std::memory_order_relaxed);
}

void reset() {
    x = 0; y = 0;
    c_saw_x = 0; c_saw_y = 0;
    d_saw_y = 0; d_saw_x = 0;
}

} // namespace acq_rel_test

// ── Test harness ─────────────────────────────────────
struct Observation {
    int cx, cy, dy, dx;
    bool disagree() const {
        // C sees x before y (cx=1, cy=0) AND D sees y before x (dy=1, dx=0)
        return (cx == 1 && cy == 0 && dy == 1 && dx == 0);
    }
    std::string label() const {
        return "C:(" + std::to_string(cx) + "," + std::to_string(cy) + ") "
             + "D:(" + std::to_string(dy) + "," + std::to_string(dx) + ")";
    }
};

int main() {
    constexpr int ITERS = 500'000;

    // ── Test 1: seq_cst ──
    {
        std::map<std::string, int> results;
        int disagree_count = 0;

        for (int i = 0; i < ITERS; ++i) {
            seq_cst_test::reset();

            std::thread ta(seq_cst_test::writer_x);
            std::thread tb(seq_cst_test::writer_y);
            std::thread tc(seq_cst_test::reader_c);
            std::thread td(seq_cst_test::reader_d);
            ta.join(); tb.join(); tc.join(); td.join();

            Observation obs{
                seq_cst_test::c_saw_x.load(),
                seq_cst_test::c_saw_y.load(),
                seq_cst_test::d_saw_y.load(),
                seq_cst_test::d_saw_x.load()
            };
            results[obs.label()]++;
            if (obs.disagree()) disagree_count++;
        }

        std::cout << "=== seq_cst (" << ITERS << " iterations) ===\n";
        for (auto& [k, v] : results) {
            std::cout << "  " << k << "  ×" << v << "\n";
        }
        std::cout << "  Disagreements (C:(1,0) D:(1,0)): " << disagree_count << "\n\n";
    }

    // ── Test 2: acquire/release ──
    {
        std::map<std::string, int> results;
        int disagree_count = 0;

        for (int i = 0; i < ITERS; ++i) {
            acq_rel_test::reset();

            std::thread ta(acq_rel_test::writer_x);
            std::thread tb(acq_rel_test::writer_y);
            std::thread tc(acq_rel_test::reader_c);
            std::thread td(acq_rel_test::reader_d);
            ta.join(); tb.join(); tc.join(); td.join();

            Observation obs{
                acq_rel_test::c_saw_x.load(),
                acq_rel_test::c_saw_y.load(),
                acq_rel_test::d_saw_y.load(),
                acq_rel_test::d_saw_x.load()
            };
            results[obs.label()]++;
            if (obs.disagree()) disagree_count++;
        }

        std::cout << "=== acquire/release (" << ITERS << " iterations) ===\n";
        for (auto& [k, v] : results) {
            std::cout << "  " << k << "  ×" << v << "\n";
        }
        std::cout << "  Disagreements (C:(1,0) D:(1,0)): " << disagree_count << "\n\n";
    }

    std::cout << "EXPECTED BEHAVIOR:\n"
              << "  seq_cst:     disagreements = 0 (global total order forbids it)\n"
              << "  acq_rel:     disagreements >= 0 (allowed by the standard)\n"
              << "  (On x86 TSO you may see 0 for both; ARM/POWER can show nonzero for acq_rel)\n";
}

/*
 * ═══════════════════════════════════════════════════════
 * ANSWER
 * ═══════════════════════════════════════════════════════
 *
 * THE IRIW PROBLEM:
 *
 * Two independent writers write to two different variables.
 * Two independent readers read both variables in opposite order.
 * Can the readers disagree about which write happened first?
 *
 *   Thread A:  x = 1
 *   Thread B:  y = 1
 *   Thread C:  reads x (gets 1), then reads y (gets 0)  → "x happened before y"
 *   Thread D:  reads y (gets 1), then reads x (gets 0)  → "y happened before x"
 *
 * (a) Under seq_cst: NO, they CANNOT disagree.
 *
 *     seq_cst establishes a SINGLE GLOBAL TOTAL ORDER of all seq_cst operations
 *     that all threads agree on. If C sees x=1 before y=1, then in the global
 *     order, x.store(1) comes before y.store(1). But then D must also agree:
 *     when D reads y=1, x=1 must already be in the global order, so D must
 *     see x=1 too.
 *
 *     The disagreement outcome C:(1,0) D:(1,0) is FORBIDDEN.
 *
 * (b) Under acquire/release: YES, they CAN disagree.
 *
 *     acquire/release only provides ordering between a specific store-release
 *     and its matching load-acquire on the SAME variable. There is no global
 *     total order. Each writer's store propagates independently through the
 *     cache coherence protocol, and different readers can see the stores
 *     arrive in different orders.
 *
 *     On architectures with weak memory models (ARM, POWER), x's store
 *     might reach Thread C's core before Thread D's core, while y's store
 *     reaches Thread D first. This is physically possible because stores
 *     propagate through an interconnect that doesn't guarantee global ordering.
 *
 *     x86 (TSO) is special: it provides a total store order, which means
 *     even acq_rel will NOT show disagreements on x86 hardware. But the
 *     C++ abstract machine still ALLOWS it, so your code can't rely on it.
 *
 * PRACTICAL RULE:
 *   If you need all threads to agree on a total order of events,
 *   you MUST use seq_cst. acquire/release is not sufficient.
 *
 *   seq_cst is the default for std::atomic operations for this reason —
 *   it's the safest option and matches programmer intuition.
 *
 *   Only downgrade to acq_rel when you have a specific producer-consumer
 *   pair and have verified the algorithm doesn't need a global order.
 */
