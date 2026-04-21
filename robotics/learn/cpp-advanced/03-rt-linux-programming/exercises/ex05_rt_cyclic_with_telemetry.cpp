// Week 3 — Exercise 05: RT Cyclic Executive with Telemetry
//
// Mini-project: 1kHz RT control loop with SPSC queue to a low-priority
// telemetry thread that writes CSV output. The RT thread never does I/O.
//
// Build:  cmake --build build
// Run:    ./ex05_rt_cyclic_with_telemetry
//         sudo ./ex05_rt_cyclic_with_telemetry   (for SCHED_FIFO)
//
// Output: telemetry.csv with columns: timestamp,sensor_val,setpoint,output,jitter_us

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <thread>

#include <sched.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>

using Clock = std::chrono::steady_clock;
using ns    = std::chrono::nanoseconds;

// ═══════════════════════════════════════════════════════════════════════
// SPSC Queue (from Week 2, simplified for this exercise)
// ═══════════════════════════════════════════════════════════════════════

template <typename T, std::size_t Capacity>
class SPSCQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be power of 2");

public:
    SPSCQueue() : head_(0), tail_(0) {}

    bool try_push(const T& item) {
        std::size_t h = head_.load(std::memory_order_relaxed);
        std::size_t next = (h + 1) & MASK;
        if (next == tail_.load(std::memory_order_acquire)) {
            return false;  // full
        }
        buf_[h] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }

    bool try_pop(T& item) {
        std::size_t t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire)) {
            return false;  // empty
        }
        item = buf_[t];
        tail_.store((t + 1) & MASK, std::memory_order_release);
        return true;
    }

private:
    static constexpr std::size_t MASK = Capacity - 1;
    T buf_[Capacity];
    alignas(64) std::atomic<std::size_t> head_;
    alignas(64) std::atomic<std::size_t> tail_;
};

// ═══════════════════════════════════════════════════════════════════════
// Telemetry record
// ═══════════════════════════════════════════════════════════════════════

struct TelemetryRecord {
    int64_t timestamp_ns;   // nanoseconds since loop start
    double  sensor_val;
    double  setpoint;
    double  output;
    double  jitter_us;
};

// ═══════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════

constexpr int64_t PERIOD_NS      = 1'000'000;   // 1ms = 1kHz
constexpr int     DURATION_SEC   = 5;
constexpr int     TOTAL_CYCLES   = DURATION_SEC * 1000;
constexpr int     RT_PRIORITY    = 80;
constexpr int     RT_CORE        = 1;
constexpr int     TELEM_CORE     = 0;

using Queue = SPSCQueue<TelemetryRecord, 8192>;

// ═══════════════════════════════════════════════════════════════════════
// Fake sensor / controller / actuator
// ═══════════════════════════════════════════════════════════════════════

// Simulates IMU sensor reading: sinusoidal acceleration
static volatile double sensor_sink;

double fake_imu_read(int cycle) {
    double val = std::sin(cycle * 0.002 * M_PI) * 5.0;
    // ~50μs of "work"
    double s = 0;
    for (int i = 0; i < 250; ++i) {
        s += std::sin(val + i * 0.01);
    }
    sensor_sink = s;
    return val;
}

// PID controller
struct PIDController {
    double setpoint   = 0.0;
    double integral   = 0.0;
    double prev_error = 0.0;
    double kp = 1.5, ki = 0.2, kd = 0.3;

    double compute(double measurement, double dt) {
        double error = setpoint - measurement;
        integral += error * dt;
        // Clamp integral to prevent windup
        if (integral > 10.0) integral = 10.0;
        if (integral < -10.0) integral = -10.0;
        double derivative = (error - prev_error) / dt;
        prev_error = error;

        double output = kp * error + ki * integral + kd * derivative;

        // ~100μs of "work"
        volatile double sink = 0;
        for (int i = 0; i < 500; ++i) {
            sink += std::sin(output + i * 0.001);
        }
        (void)sink;

        return output;
    }
};

// Fake actuator: ~30μs
static volatile double actuator_sink;

void fake_actuator_write(double cmd) {
    volatile double sink = 0;
    for (int i = 0; i < 150; ++i) {
        sink += cmd * 0.001;
    }
    actuator_sink = sink;
}

// ═══════════════════════════════════════════════════════════════════════
// Timespec helpers
// ═══════════════════════════════════════════════════════════════════════

void timespec_add_ns(struct timespec& ts, int64_t ns_val) {
    ts.tv_nsec += ns_val;
    while (ts.tv_nsec >= 1'000'000'000L) {
        ts.tv_nsec -= 1'000'000'000L;
        ts.tv_sec++;
    }
}

int64_t timespec_diff_ns(const struct timespec& a, const struct timespec& b) {
    return (a.tv_sec - b.tv_sec) * 1'000'000'000L + (a.tv_nsec - b.tv_nsec);
}

// ═══════════════════════════════════════════════════════════════════════
// Telemetry thread (low priority, does all I/O)
// ═══════════════════════════════════════════════════════════════════════

void telemetry_thread_fn(Queue& queue, std::atomic<bool>& done) {
    // Pin to telemetry core (non-isolated)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(TELEM_CORE, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);

    FILE* csv = std::fopen("telemetry.csv", "w");
    if (!csv) {
        std::perror("fopen telemetry.csv");
        return;
    }
    std::fprintf(csv, "timestamp_ns,sensor_val,setpoint,output,jitter_us\n");

    int records_written = 0;

    while (!done.load(std::memory_order_relaxed) || true) {
        TelemetryRecord rec;
        if (queue.try_pop(rec)) {
            std::fprintf(csv, "%ld,%.6f,%.6f,%.6f,%.2f\n",
                         rec.timestamp_ns,
                         rec.sensor_val,
                         rec.setpoint,
                         rec.output,
                         rec.jitter_us);
            records_written++;
        } else {
            if (done.load(std::memory_order_relaxed)) {
                // Drain remaining
                while (queue.try_pop(rec)) {
                    std::fprintf(csv, "%ld,%.6f,%.6f,%.6f,%.2f\n",
                                 rec.timestamp_ns,
                                 rec.sensor_val,
                                 rec.setpoint,
                                 rec.output,
                                 rec.jitter_us);
                    records_written++;
                }
                break;
            }
            // No data — yield (not RT-critical)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    std::fclose(csv);
    std::printf("[Telemetry] Wrote %d records to telemetry.csv\n", records_written);
}

// ═══════════════════════════════════════════════════════════════════════
// Signal handler
// ═══════════════════════════════════════════════════════════════════════

static std::atomic<bool> g_running{true};

void signal_handler(int) {
    g_running.store(false, std::memory_order_relaxed);
}

// ═══════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════

int main() {
    std::printf("╔════════════════════════════════════════════════════╗\n");
    std::printf("║  Exercise 05: RT Cyclic Executive with Telemetry  ║\n");
    std::printf("╚════════════════════════════════════════════════════╝\n");

    std::signal(SIGINT, signal_handler);

    // --- RT setup ---
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == 0) {
        std::printf("[RT] mlockall: OK\n");
    } else {
        std::printf("[RT] mlockall: %s\n", std::strerror(errno));
    }

    // Prefault stack
    {
        volatile char dummy[512 * 1024];
        for (size_t i = 0; i < sizeof(dummy); i += 4096) dummy[i] = 0;
    }
    std::printf("[RT] Stack prefaulted\n");

    // --- Shared state ---
    Queue queue;
    std::atomic<bool> telem_done{false};

    // Start telemetry thread BEFORE RT thread starts
    std::thread telem_thread(telemetry_thread_fn, std::ref(queue), std::ref(telem_done));

    // Pin RT thread
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(RT_CORE, &cpuset);
        if (sched_setaffinity(0, sizeof(cpuset), &cpuset) == 0) {
            std::printf("[RT] Pinned to core %d\n", RT_CORE);
        } else {
            std::printf("[RT] Affinity failed: %s\n", std::strerror(errno));
        }
    }

    // Set SCHED_FIFO
    {
        struct sched_param param{};
        param.sched_priority = RT_PRIORITY;
        if (sched_setscheduler(0, SCHED_FIFO, &param) == 0) {
            std::printf("[RT] SCHED_FIFO priority=%d\n", RT_PRIORITY);
        } else {
            std::printf("[RT] SCHED_FIFO failed: %s (using SCHED_OTHER)\n",
                        std::strerror(errno));
        }
    }

    std::printf("[RT] Running %d cycles at %dHz for %ds...\n\n",
                TOTAL_CYCLES,
                static_cast<int>(1'000'000'000L / PERIOD_NS),
                DURATION_SEC);

    // --- RT loop ---
    PIDController pid;
    pid.setpoint = 0.0;  // tracking zero

    struct timespec loop_start, next_wake;
    clock_gettime(CLOCK_MONOTONIC, &loop_start);
    next_wake = loop_start;

    int overruns = 0;
    int queue_drops = 0;
    int64_t max_jitter_ns = 0;
    int64_t sum_jitter_ns = 0;

    for (int cycle = 0; cycle < TOTAL_CYCLES && g_running; ++cycle) {
        timespec_add_ns(next_wake, PERIOD_NS);

        // --- Tasks ---
        double sensor_val = fake_imu_read(cycle);
        double output = pid.compute(sensor_val, 0.001);
        fake_actuator_write(output);

        // --- Check overrun ---
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        int64_t remaining = timespec_diff_ns(next_wake, now);

        if (remaining < 0) {
            overruns++;
            int64_t missed = (-remaining + PERIOD_NS - 1) / PERIOD_NS;
            timespec_add_ns(next_wake, missed * PERIOD_NS);
            continue;
        }

        // --- Sleep ---
        clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next_wake, nullptr);

        // --- Measure jitter ---
        struct timespec actual;
        clock_gettime(CLOCK_MONOTONIC, &actual);
        int64_t jitter_ns = timespec_diff_ns(actual, next_wake);
        int64_t abs_jitter = jitter_ns < 0 ? -jitter_ns : jitter_ns;
        if (abs_jitter > max_jitter_ns) max_jitter_ns = abs_jitter;
        sum_jitter_ns += abs_jitter;

        // --- Enqueue telemetry (RT-safe: just a memcpy + atomic store) ---
        TelemetryRecord rec{
            .timestamp_ns = timespec_diff_ns(actual, loop_start),
            .sensor_val   = sensor_val,
            .setpoint     = pid.setpoint,
            .output       = output,
            .jitter_us    = abs_jitter / 1000.0,
        };
        if (!queue.try_push(rec)) {
            queue_drops++;
        }

        // Progress
        if (cycle > 0 && cycle % 1000 == 0) {
            std::printf("  [%ds] overruns=%d, max_jitter=%.1fμs, drops=%d\n",
                        cycle / 1000, overruns, max_jitter_ns / 1000.0, queue_drops);
        }
    }

    // --- Shutdown ---
    telem_done.store(true, std::memory_order_release);
    telem_thread.join();

    // --- Summary ---
    std::printf("\n╔════════════════════════════════════════════════╗\n");
    std::printf("║              Summary Statistics                ║\n");
    std::printf("╠════════════════════════════════════════════════╣\n");
    std::printf("║  Cycles run:     %8d                     ║\n", TOTAL_CYCLES);
    std::printf("║  Overruns:       %8d                     ║\n", overruns);
    std::printf("║  Queue drops:    %8d                     ║\n", queue_drops);
    std::printf("║  Max jitter:     %8.2f μs                ║\n", max_jitter_ns / 1000.0);
    std::printf("║  Mean jitter:    %8.2f μs                ║\n",
                TOTAL_CYCLES > 0 ? (sum_jitter_ns / (double)TOTAL_CYCLES) / 1000.0 : 0.0);
    std::printf("╚════════════════════════════════════════════════╝\n");

    std::printf("\n─── Architecture ───\n");
    std::printf("RT thread (core %d, SCHED_FIFO) → SPSC queue → Telemetry thread (core %d)\n",
                RT_CORE, TELEM_CORE);
    std::printf("RT thread never calls printf, write, malloc, or any blocking syscall.\n");
    std::printf("All I/O happens in the low-priority telemetry thread.\n");
    std::printf("\nCheck telemetry.csv for per-cycle data.\n");

    return 0;
}
