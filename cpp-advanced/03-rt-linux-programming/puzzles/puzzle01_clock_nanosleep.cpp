// Week 3 — Puzzle 01: Relative vs Absolute clock_nanosleep
//
// Demonstrate that relative sleep drifts due to processing time accumulation,
// while absolute sleep maintains exact periodicity.
//
// Expected: relative loses ~50μs+ per iteration, accumulating over 100 cycles.
//
// Build:  (from exercises/ build dir) cmake --build .
// Run:    ./puzzle01_clock_nanosleep

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <time.h>

// ─── Timespec helpers ────────────────────────────────────────────────
void timespec_add_ns(struct timespec& ts, int64_t ns) {
    ts.tv_nsec += ns;
    while (ts.tv_nsec >= 1'000'000'000L) {
        ts.tv_nsec -= 1'000'000'000L;
        ts.tv_sec++;
    }
}

int64_t timespec_diff_ns(const struct timespec& a, const struct timespec& b) {
    return (a.tv_sec - b.tv_sec) * 1'000'000'000L + (a.tv_nsec - b.tv_nsec);
}

// ─── Simulate some work (~200μs) ────────────────────────────────────
static volatile double work_sink;

void do_work() {
    double x = 1.0;
    for (int i = 0; i < 1000; ++i) {
        x += 0.001 * i;
    }
    work_sink = x;
}

// ═══════════════════════════════════════════════════════════════════════
// Test: Relative clock_nanosleep (WRONG for periodic tasks)
// ═══════════════════════════════════════════════════════════════════════

void test_relative() {
    constexpr int ITERS = 100;
    constexpr int64_t PERIOD_NS = 1'000'000;  // 1ms

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < ITERS; ++i) {
        do_work();  // Simulated processing

        // RELATIVE sleep: sleeps for 1ms FROM NOW
        // Problem: doesn't account for time spent in do_work()
        struct timespec req = {0, PERIOD_NS};
        clock_nanosleep(CLOCK_MONOTONIC, 0, &req, nullptr);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    int64_t elapsed_ns = timespec_diff_ns(end, start);
    int64_t expected_ns = static_cast<int64_t>(ITERS) * PERIOD_NS;
    int64_t drift_ns = elapsed_ns - expected_ns;

    std::printf("RELATIVE sleep (%d iterations, %ldμs period):\n", ITERS, PERIOD_NS / 1000);
    std::printf("  Expected total: %8.3f ms\n", expected_ns / 1'000'000.0);
    std::printf("  Actual total:   %8.3f ms\n", elapsed_ns / 1'000'000.0);
    std::printf("  Total drift:    %8.3f ms (%.1f μs/iteration)\n",
                drift_ns / 1'000'000.0, drift_ns / (1000.0 * ITERS));
    std::printf("  → Each iteration takes period + work_time. Drift accumulates!\n\n");
}

// ═══════════════════════════════════════════════════════════════════════
// Test: Absolute clock_nanosleep (CORRECT for periodic tasks)
// ═══════════════════════════════════════════════════════════════════════

void test_absolute() {
    constexpr int ITERS = 100;
    constexpr int64_t PERIOD_NS = 1'000'000;  // 1ms

    struct timespec start, end, next_wake;
    clock_gettime(CLOCK_MONOTONIC, &start);
    next_wake = start;

    for (int i = 0; i < ITERS; ++i) {
        do_work();  // Simulated processing

        // ABSOLUTE sleep: wakes at exact multiples of period
        // Accounts for processing time automatically
        timespec_add_ns(next_wake, PERIOD_NS);
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_wake, nullptr);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    int64_t elapsed_ns = timespec_diff_ns(end, start);
    int64_t expected_ns = static_cast<int64_t>(ITERS) * PERIOD_NS;
    int64_t drift_ns = elapsed_ns - expected_ns;

    std::printf("ABSOLUTE sleep (%d iterations, %ldμs period):\n", ITERS, PERIOD_NS / 1000);
    std::printf("  Expected total: %8.3f ms\n", expected_ns / 1'000'000.0);
    std::printf("  Actual total:   %8.3f ms\n", elapsed_ns / 1'000'000.0);
    std::printf("  Total drift:    %8.3f ms (%.1f μs/iteration)\n",
                drift_ns / 1'000'000.0, drift_ns / (1000.0 * ITERS));
    std::printf("  → Wakes at exact wall-clock times. Near-zero drift.\n\n");
}

// ═══════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════

int main() {
    std::printf("╔════════════════════════════════════════════════════╗\n");
    std::printf("║  Puzzle 01: Relative vs Absolute clock_nanosleep  ║\n");
    std::printf("╚════════════════════════════════════════════════════╝\n\n");

    std::printf("Each iteration does ~200μs of work, then sleeps for 1ms.\n");
    std::printf("Expected: 100 iterations × 1ms = 100ms total.\n\n");

    test_relative();
    test_absolute();

    std::printf("─── The Lesson ───\n");
    std::printf("Relative: total = N × (period + work_time) → drifts\n");
    std::printf("Absolute: total = N × period                → exact\n");
    std::printf("\nALWAYS use TIMER_ABSTIME for periodic real-time loops.\n");
    std::printf("This is Rule #1 of cyclic executive design.\n");

    return 0;
}
