// Week 3 — Exercise 03: Cyclic Executive (1kHz Real-Time Control Loop)
//
// The core exercise: implement a 1ms-period cyclic executive with
// jitter measurement, overrun detection, and a 3-task pipeline.
//
// Build:  cmake --build build
// Run:    ./ex03_cyclic_executive               (SCHED_OTHER — still teaches)
//         sudo ./ex03_cyclic_executive          (SCHED_FIFO — see the difference)
//   or:   sudo setcap cap_sys_nice+ep ./ex03_cyclic_executive

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <csignal>
#include <thread>

#include <sched.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>

// ─── Configuration ───────────────────────────────────────────────────
constexpr int64_t PERIOD_NS      = 1'000'000;   // 1ms = 1kHz
constexpr int     DURATION_SEC   = 10;
constexpr int     TOTAL_CYCLES   = DURATION_SEC * 1000;
constexpr int     RT_PRIORITY    = 80;
constexpr int     RT_CORE        = 1;            // pin to core 1

// ─── Jitter histogram buckets ────────────────────────────────────────
struct JitterHistogram {
    int bucket_lt1us    = 0;   // < 1μs
    int bucket_1_5us    = 0;   // 1 – 5 μs
    int bucket_5_10us   = 0;   // 5 – 10 μs
    int bucket_10_50us  = 0;   // 10 – 50 μs
    int bucket_gt50us   = 0;   // > 50 μs

    int64_t min_ns = INT64_MAX;
    int64_t max_ns = 0;
    int64_t sum_ns = 0;
    int     count  = 0;
    int     overruns = 0;

    void record(int64_t jitter_ns) {
        int64_t abs_jitter = jitter_ns < 0 ? -jitter_ns : jitter_ns;
        if (abs_jitter < min_ns) min_ns = abs_jitter;
        if (abs_jitter > max_ns) max_ns = abs_jitter;
        sum_ns += abs_jitter;
        count++;

        double us = abs_jitter / 1000.0;
        if      (us < 1.0)  bucket_lt1us++;
        else if (us < 5.0)  bucket_1_5us++;
        else if (us < 10.0) bucket_5_10us++;
        else if (us < 50.0) bucket_10_50us++;
        else                bucket_gt50us++;
    }

    void print() const {
        std::printf("\n╔══════════════════════════════════════════════╗\n");
        std::printf("║         Jitter Histogram (%d cycles)       ║\n", count);
        std::printf("╠══════════════════════════════════════════════╣\n");
        std::printf("║  < 1μs:    %6d  ", bucket_lt1us);
        print_bar(bucket_lt1us); std::printf("║\n");
        std::printf("║  1-5μs:    %6d  ", bucket_1_5us);
        print_bar(bucket_1_5us); std::printf("║\n");
        std::printf("║  5-10μs:   %6d  ", bucket_5_10us);
        print_bar(bucket_5_10us); std::printf("║\n");
        std::printf("║  10-50μs:  %6d  ", bucket_10_50us);
        print_bar(bucket_10_50us); std::printf("║\n");
        std::printf("║  >50μs:    %6d  ", bucket_gt50us);
        print_bar(bucket_gt50us); std::printf("║\n");
        std::printf("╠══════════════════════════════════════════════╣\n");
        std::printf("║  min jitter:  %8.2f μs                   ║\n", min_ns / 1000.0);
        std::printf("║  max jitter:  %8.2f μs                   ║\n", max_ns / 1000.0);
        std::printf("║  mean jitter: %8.2f μs                   ║\n",
                    count > 0 ? (sum_ns / (double)count) / 1000.0 : 0.0);
        std::printf("║  overruns:    %8d                       ║\n", overruns);
        std::printf("╚══════════════════════════════════════════════╝\n");
    }

private:
    void print_bar(int val) const {
        int max_bar = 20;
        int bar = count > 0 ? (val * max_bar) / count : 0;
        for (int i = 0; i < bar; ++i) std::printf("█");
        for (int i = bar; i < max_bar; ++i) std::printf(" ");
    }
};

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

// ─── Fake RT tasks ───────────────────────────────────────────────────

// Task 1: Sensor read (simulates ~100μs)
struct SensorData {
    double acceleration;
    double angular_velocity;
};

static volatile double fake_sensor_sink;

SensorData sensor_read(int cycle) {
    // Fake computation (~100μs)
    double sum = 0;
    for (int i = 0; i < 500; ++i) {
        sum += std::sin(cycle * 0.001 + i * 0.01);
    }
    fake_sensor_sink = sum;
    return {
        .acceleration = std::sin(cycle * 0.01) * 2.0,
        .angular_velocity = std::cos(cycle * 0.01) * 0.5,
    };
}

// Task 2: PID Controller (~200μs)
struct PIDState {
    double setpoint   = 0.0;
    double integral   = 0.0;
    double prev_error = 0.0;
    double kp = 2.0, ki = 0.1, kd = 0.5;
};

double controller_compute(PIDState& pid, double measurement, double dt) {
    double error = pid.setpoint - measurement;
    pid.integral += error * dt;
    double derivative = (error - pid.prev_error) / dt;
    pid.prev_error = error;

    double output = pid.kp * error + pid.ki * pid.integral + pid.kd * derivative;

    // Add some fake computation to simulate ~200μs
    volatile double sink = 0;
    for (int i = 0; i < 1000; ++i) {
        sink += std::sin(output + i * 0.001);
    }
    (void)sink;

    return output;
}

// Task 3: Actuator write (~50μs)
static volatile double actuator_sink;

void actuator_write(double command) {
    // Fake: write command to "hardware register"
    volatile double sink = 0;
    for (int i = 0; i < 250; ++i) {
        sink += command * 0.001;
    }
    actuator_sink = sink;
}

// ─── Signal handler for clean shutdown ───────────────────────────────
static std::atomic<bool> g_running{true};

void signal_handler(int) {
    g_running.store(false, std::memory_order_relaxed);
}

// ─── Prefault stack ──────────────────────────────────────────────────
void prefault_stack() {
    volatile char dummy[512 * 1024];  // 512KB
    for (size_t i = 0; i < sizeof(dummy); i += 4096) {
        dummy[i] = 0;
    }
}

// ─── Main RT Loop ────────────────────────────────────────────────────
int main() {
    std::printf("╔════════════════════════════════════════════════════╗\n");
    std::printf("║  Exercise 03: 1kHz Cyclic Executive               ║\n");
    std::printf("╚════════════════════════════════════════════════════╝\n");

    // --- RT setup ---
    std::signal(SIGINT, signal_handler);

    // 1. mlockall
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == 0) {
        std::printf("[RT] mlockall: OK\n");
    } else {
        std::printf("[RT] mlockall: failed (%s) — page faults possible\n",
                    std::strerror(errno));
    }

    // 2. Prefault stack
    prefault_stack();
    std::printf("[RT] Stack prefaulted\n");

    // 3. Set affinity
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(RT_CORE, &cpuset);
        if (sched_setaffinity(0, sizeof(cpuset), &cpuset) == 0) {
            std::printf("[RT] Pinned to core %d\n", RT_CORE);
        } else {
            std::printf("[RT] Affinity to core %d failed: %s\n",
                        RT_CORE, std::strerror(errno));
        }
    }

    // 4. Set SCHED_FIFO
    {
        struct sched_param param{};
        param.sched_priority = RT_PRIORITY;
        if (sched_setscheduler(0, SCHED_FIFO, &param) == 0) {
            std::printf("[RT] SCHED_FIFO priority=%d: OK\n", RT_PRIORITY);
        } else {
            std::printf("[RT] SCHED_FIFO failed: %s (running as SCHED_OTHER)\n",
                        std::strerror(errno));
        }
    }

    std::printf("[RT] Starting %dHz loop for %d seconds (%d cycles)...\n\n",
                static_cast<int>(1'000'000'000L / PERIOD_NS),
                DURATION_SEC, TOTAL_CYCLES);

    // --- Initialize state ---
    JitterHistogram hist;
    PIDState pid;
    pid.setpoint = 0.0;

    // --- The loop ---
    struct timespec next_wake;
    clock_gettime(CLOCK_MONOTONIC, &next_wake);

    for (int cycle = 0; cycle < TOTAL_CYCLES && g_running; ++cycle) {
        // Advance to next period (ABSOLUTE time)
        timespec_add_ns(next_wake, PERIOD_NS);

        // ── Execute tasks ──
        SensorData sensor = sensor_read(cycle);
        double command = controller_compute(pid, sensor.acceleration, 0.001);
        actuator_write(command);

        // ── Check for overrun ──
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        int64_t remaining_ns = timespec_diff_ns(next_wake, now);

        if (remaining_ns < 0) {
            // Overrun! Processing exceeded the period.
            hist.overruns++;
            // Skip to the next aligned period
            int64_t missed = (-remaining_ns + PERIOD_NS - 1) / PERIOD_NS;
            timespec_add_ns(next_wake, missed * PERIOD_NS);
            // Don't sleep — immediately start next cycle
            continue;
        }

        // ── Sleep until next period ──
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_wake, nullptr);

        // ── Measure jitter (actual wakeup vs expected) ──
        struct timespec actual;
        clock_gettime(CLOCK_MONOTONIC, &actual);
        int64_t jitter_ns = timespec_diff_ns(actual, next_wake);
        hist.record(jitter_ns);

        // Progress indicator every second
        if (cycle > 0 && cycle % 1000 == 0) {
            std::printf("  [%ds] %d cycles complete, overruns: %d, max_jitter: %.1fμs\n",
                        cycle / 1000, cycle, hist.overruns, hist.max_ns / 1000.0);
        }
    }

    // --- Report ---
    hist.print();

    std::printf("\n─── Interpretation ───\n");
    std::printf("• With SCHED_FIFO on isolated core: expect >99%% in <5μs bucket\n");
    std::printf("• With SCHED_OTHER: expect many in 10-50μs, some >50μs\n");
    std::printf("• Overruns should be 0 with SCHED_FIFO (tasks fit in ~350μs)\n");
    std::printf("• If max_jitter > 100μs: something interrupted the RT thread\n");

    return 0;
}
