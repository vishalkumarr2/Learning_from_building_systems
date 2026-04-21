// hw01_clock_drift.cpp — Clock Trees & Timing
// Compile: g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -pthread hw01_clock_drift.cpp -o hw01_clock_drift
//
// Exercises:
//  1. Measure drift between CLOCK_MONOTONIC and CLOCK_REALTIME over 5s
//  2. Calculate drift in ppm
//  3. Measure clock_gettime resolution and actual precision
//  4. Read /sys clocksource
//  5. HighResTimer class using steady_clock

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <time.h>

// ---------------------------------------------------------------------------
// Utility: read timespec as nanoseconds
// ---------------------------------------------------------------------------
static int64_t ts_to_ns(const struct timespec& ts) {
    return static_cast<int64_t>(ts.tv_sec) * 1'000'000'000LL + ts.tv_nsec;
}

// ---------------------------------------------------------------------------
// 1. Read current clocksource from sysfs
// ---------------------------------------------------------------------------
static std::string read_clocksource() {
    const char* path =
        "/sys/devices/system/clocksource/clocksource0/current_clocksource";
    std::ifstream ifs(path);
    std::string src;
    if (ifs >> src) return src;
    return "unknown";
}

// ---------------------------------------------------------------------------
// 2. Measure clock_gettime resolution (what the kernel reports)
// ---------------------------------------------------------------------------
static int64_t reported_resolution(clockid_t clk) {
    struct timespec res{};
    clock_getres(clk, &res);
    return ts_to_ns(res);
}

// ---------------------------------------------------------------------------
// 3. Measure actual precision: tight loop, find minimum non-zero delta
// ---------------------------------------------------------------------------
static int64_t measured_precision(clockid_t clk, int iterations = 100'000) {
    int64_t min_delta = INT64_MAX;
    struct timespec prev{}, cur{};
    clock_gettime(clk, &prev);

    for (int i = 0; i < iterations; ++i) {
        clock_gettime(clk, &cur);
        int64_t delta = ts_to_ns(cur) - ts_to_ns(prev);
        if (delta > 0 && delta < min_delta) {
            min_delta = delta;
        }
        prev = cur;
    }
    return min_delta;
}

// ---------------------------------------------------------------------------
// 4. Measure drift between CLOCK_MONOTONIC and CLOCK_REALTIME
//    Returns drift in parts-per-million (ppm).
// ---------------------------------------------------------------------------
static double measure_drift_ppm(int duration_sec) {
    struct timespec mono_start{}, real_start{};
    clock_gettime(CLOCK_MONOTONIC, &mono_start);
    clock_gettime(CLOCK_REALTIME, &real_start);

    int64_t diff_start = ts_to_ns(real_start) - ts_to_ns(mono_start);

    std::printf("  Measuring drift over %d seconds...\n", duration_sec);
    std::this_thread::sleep_for(std::chrono::seconds(duration_sec));

    struct timespec mono_end{}, real_end{};
    clock_gettime(CLOCK_MONOTONIC, &mono_end);
    clock_gettime(CLOCK_REALTIME, &real_end);

    int64_t diff_end = ts_to_ns(real_end) - ts_to_ns(mono_end);
    int64_t elapsed_ns =
        ts_to_ns(mono_end) - ts_to_ns(mono_start);

    // drift_ns is how much the REALTIME-MONOTONIC gap changed
    int64_t drift_ns = diff_end - diff_start;
    double ppm = (static_cast<double>(drift_ns) / static_cast<double>(elapsed_ns)) * 1'000'000.0;
    return ppm;
}

// ---------------------------------------------------------------------------
// 5. HighResTimer class using steady_clock
// ---------------------------------------------------------------------------
class HighResTimer {
public:
    void start() { start_ = std::chrono::steady_clock::now(); }

    void stop() { end_ = std::chrono::steady_clock::now(); }

    [[nodiscard]] double elapsed_us() const {
        return std::chrono::duration<double, std::micro>(end_ - start_).count();
    }

    [[nodiscard]] double elapsed_ns() const {
        return std::chrono::duration<double, std::nano>(end_ - start_).count();
    }

    [[nodiscard]] int64_t elapsed_cycles_approx() const {
        // Rough approximation assuming 3 GHz — not rdtsc
        return static_cast<int64_t>(elapsed_ns() * 3.0);
    }

private:
    std::chrono::steady_clock::time_point start_{};
    std::chrono::steady_clock::time_point end_{};
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::printf("╔══════════════════════════════════════════════════════╗\n");
    std::printf("║       HW01 — Clock Trees & Timing                   ║\n");
    std::printf("╚══════════════════════════════════════════════════════╝\n\n");

    // --- Clocksource ---
    std::string clocksrc = read_clocksource();
    std::printf("Current clocksource: %s\n\n", clocksrc.c_str());

    // --- Resolution & precision ---
    int64_t mono_res = reported_resolution(CLOCK_MONOTONIC);
    int64_t real_res = reported_resolution(CLOCK_REALTIME);
    int64_t mono_prec = measured_precision(CLOCK_MONOTONIC);
    int64_t real_prec = measured_precision(CLOCK_REALTIME);

    std::printf("┌────────────────────┬─────────────┬──────────────────────┐\n");
    std::printf("│ Clock              │  Resolution │  Measured Precision  │\n");
    std::printf("├────────────────────┼─────────────┼──────────────────────┤\n");
    std::printf("│ CLOCK_MONOTONIC    │ %7ld ns  │ %14ld ns    │\n", mono_res, mono_prec);
    std::printf("│ CLOCK_REALTIME     │ %7ld ns  │ %14ld ns    │\n", real_res, real_prec);
    std::printf("└────────────────────┴─────────────┴──────────────────────┘\n\n");

    // --- Drift measurement ---
    constexpr int kDriftSeconds = 5;
    double drift = measure_drift_ppm(kDriftSeconds);
    std::printf("\n  REALTIME vs MONOTONIC drift: %.4f ppm\n", drift);
    std::printf("  (Note: on a normal system NTP adjusts REALTIME, "
                "so non-zero drift is expected)\n\n");

    // --- HighResTimer demo ---
    HighResTimer timer;
    timer.start();
    // Something to time: tight loop
    volatile int sink = 0;
    for (int i = 0; i < 1'000'000; ++i) {
        sink += i;
    }
    timer.stop();

    std::printf("HighResTimer demo (1M additions):\n");
    std::printf("  elapsed: %.2f µs  (%.0f ns)\n",
                timer.elapsed_us(), timer.elapsed_ns());
    std::printf("  ~cycles (3 GHz est): %ld\n\n", timer.elapsed_cycles_approx());

    // --- Summary table ---
    std::printf("╔══════════════════════════════════════════════════════════════╗\n");
    std::printf("║  Summary                                                     ║\n");
    std::printf("╠══════════════════════════════════════════════════════════════╣\n");
    std::printf("║  clocksource        : %-20s                   ║\n", clocksrc.c_str());
    std::printf("║  mono resolution_ns : %-10ld                             ║\n", mono_res);
    std::printf("║  mono precision_ns  : %-10ld                             ║\n", mono_prec);
    std::printf("║  drift_ppm          : %-10.4f                             ║\n", drift);
    std::printf("╚══════════════════════════════════════════════════════════════╝\n");

    return 0;
}
