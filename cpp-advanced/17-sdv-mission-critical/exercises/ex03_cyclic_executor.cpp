// =============================================================================
// Exercise 03: Deterministic Cyclic Executor
// =============================================================================
// Implement a cyclic executor that runs tasks at fixed periods with:
//   - CPU core pinning (avoids migration jitter)
//   - SCHED_FIFO real-time scheduling
//   - Jitter measurement and overrun detection
//   - Execution time statistics (min, max, average)
//
// Production relevance:
//   - Vehicle dynamics controllers run at 10ms (100Hz)
//   - Perception pipelines run at 33ms (30Hz)
//   - WCET overruns are safety violations in ISO 26262
//
// Note: SCHED_FIFO requires root/CAP_SYS_NICE. The exercise works without it
//       but logs a warning. Run with `sudo` for full real-time behavior.
// =============================================================================

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <thread>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#endif

using Clock = std::chrono::steady_clock;
using Duration = std::chrono::nanoseconds;

// ============================================================================
// Execution Statistics
// ============================================================================
struct CycleStats {
    uint64_t cycle_count = 0;
    Duration min_exec_time = Duration::max();
    Duration max_exec_time = Duration::zero();
    Duration total_exec_time = Duration::zero();
    Duration max_jitter = Duration::zero();
    uint64_t overrun_count = 0;
    uint64_t missed_deadlines = 0;

    Duration average_exec_time() const {
        if (cycle_count == 0) return Duration::zero();
        return total_exec_time / cycle_count;
    }

    void record_cycle(Duration exec_time, Duration jitter,
                       Duration budget) {
        ++cycle_count;
        total_exec_time += exec_time;

        if (exec_time < min_exec_time) min_exec_time = exec_time;
        if (exec_time > max_exec_time) max_exec_time = exec_time;
        if (jitter > max_jitter) max_jitter = jitter;
        if (exec_time > budget) ++overrun_count;
    }
};

// ============================================================================
// Cyclic Executor
// ============================================================================
class CyclicExecutor {
    using TaskFn = std::function<void()>;

    TaskFn task_;
    Duration period_;
    Duration budget_;       // WCET budget
    unsigned cpu_core_;
    int rt_priority_;
    std::atomic<bool> running_{false};
    CycleStats stats_;

    bool try_set_realtime() {
#ifdef __linux__
        // Pin to CPU core
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cpu_core_, &cpuset);
        int rc = pthread_setaffinity_np(pthread_self(),
                                         sizeof(cpuset), &cpuset);
        if (rc != 0) {
            std::cerr << "  [WARN] Could not pin to CPU " << cpu_core_
                      << " (rc=" << rc << ")\n";
        }

        // Set SCHED_FIFO
        struct sched_param param{};
        param.sched_priority = rt_priority_;
        rc = pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
        if (rc != 0) {
            std::cerr << "  [WARN] Could not set SCHED_FIFO priority "
                      << rt_priority_
                      << " (run with sudo for RT scheduling)\n";
            return false;
        }

        // Lock memory to prevent page faults
        if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
            std::cerr << "  [WARN] mlockall failed (run with sudo)\n";
        }

        return true;
#else
        return false;
#endif
    }

public:
    CyclicExecutor(TaskFn task,
                    std::chrono::milliseconds period,
                    std::chrono::microseconds budget,
                    unsigned cpu_core = 0,
                    int rt_priority = 80)
        : task_(std::move(task))
        , period_(period)
        , budget_(budget)
        , cpu_core_(cpu_core)
        , rt_priority_(rt_priority) {}

    // Run for a specified number of cycles (for testing)
    CycleStats run_cycles(uint64_t num_cycles) {
        running_ = true;
        stats_ = CycleStats{};

        try_set_realtime();

        auto next_wake = Clock::now();

        for (uint64_t i = 0; i < num_cycles && running_; ++i) {
            next_wake += period_;

            // Execute task and measure
            auto start = Clock::now();
            task_();
            auto end = Clock::now();

            Duration exec_time = end - start;

            // Sleep until next period
            std::this_thread::sleep_until(next_wake);

            // Measure jitter (how late did we wake up?)
            auto actual_wake = Clock::now();
            Duration jitter = actual_wake - next_wake;
            // Jitter can be negative (woke up early) — use absolute
            if (jitter < Duration::zero()) {
                jitter = Duration::zero();
            }

            stats_.record_cycle(exec_time, jitter, budget_);
        }

        running_ = false;
        return stats_;
    }

    void stop() { running_ = false; }
    CycleStats const& stats() const { return stats_; }
};

// ============================================================================
// Simulated workloads
// ============================================================================
namespace workloads {

// Light workload: ~100µs
void control_cycle_light() {
    volatile double sum = 0.0;
    for (int i = 0; i < 1000; ++i) {
        sum += std::sin(static_cast<double>(i) * 0.001);
    }
}

// Heavy workload: ~1ms
void control_cycle_heavy() {
    volatile double sum = 0.0;
    for (int i = 0; i < 10000; ++i) {
        sum += std::sin(static_cast<double>(i) * 0.001);
    }
}

// Variable workload: sometimes fast, sometimes slow
// Simulates real-world where some cycles have more data to process
void control_cycle_variable() {
    static int counter = 0;
    int iterations = (counter++ % 5 == 0) ? 8000 : 2000;  // spike every 5th
    volatile double sum = 0.0;
    for (int i = 0; i < iterations; ++i) {
        sum += std::sin(static_cast<double>(i) * 0.001);
    }
}

} // namespace workloads

// ============================================================================
// Helper: print statistics
// ============================================================================
void print_stats(std::string_view name, CycleStats const& stats) {
    auto us = [](Duration d) -> double {
        return static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(d).count());
    };
    std::cout << "  " << name << ":\n"
              << "    Cycles:     " << stats.cycle_count << "\n"
              << "    Exec time:  min=" << us(stats.min_exec_time)
              << "µs  max=" << us(stats.max_exec_time)
              << "µs  avg=" << us(stats.average_exec_time()) << "µs\n"
              << "    Max jitter: " << us(stats.max_jitter) << "µs\n"
              << "    Overruns:   " << stats.overrun_count << "\n";
}

// ============================================================================
// Self-Test
// ============================================================================
void test_basic_execution() {
    std::cout << "--- Test: Basic cyclic execution ---\n";

    uint64_t call_count = 0;
    CyclicExecutor exec(
        [&call_count]() { ++call_count; },
        std::chrono::milliseconds(10),
        std::chrono::microseconds(5000)
    );

    auto stats = exec.run_cycles(20);
    assert(stats.cycle_count == 20);
    assert(call_count == 20);
    std::cout << "  PASS: 20 cycles executed\n";
}

void test_overrun_detection() {
    std::cout << "--- Test: Overrun detection ---\n";

    // Budget: 200µs, but heavy workload takes ~1ms
    CyclicExecutor exec(
        workloads::control_cycle_heavy,
        std::chrono::milliseconds(10),
        std::chrono::microseconds(200)  // tight budget
    );

    auto stats = exec.run_cycles(10);
    // Heavy workload should exceed 200µs budget
    assert(stats.overrun_count > 0);
    print_stats("Heavy workload (200µs budget)", stats);
    std::cout << "  PASS: " << stats.overrun_count
              << " overruns detected\n";
}

void test_no_overruns_with_generous_budget() {
    std::cout << "--- Test: No overruns with generous budget ---\n";

    // Budget: 50ms, light workload takes ~100µs
    CyclicExecutor exec(
        workloads::control_cycle_light,
        std::chrono::milliseconds(10),
        std::chrono::microseconds(50000)  // generous budget
    );

    auto stats = exec.run_cycles(10);
    assert(stats.overrun_count == 0);
    print_stats("Light workload (50ms budget)", stats);
    std::cout << "  PASS: 0 overruns\n";
}

void test_jitter_measurement() {
    std::cout << "--- Test: Jitter measurement ---\n";

    CyclicExecutor exec(
        workloads::control_cycle_light,
        std::chrono::milliseconds(10),
        std::chrono::microseconds(5000)
    );

    auto stats = exec.run_cycles(50);
    // Max jitter should be less than one period (10ms)
    // On a loaded system without RT, this may fail — that's educational!
    auto max_jitter_us =
        std::chrono::duration_cast<std::chrono::microseconds>(
            stats.max_jitter).count();
    print_stats("Jitter test (50 cycles)", stats);

    // Lenient check: jitter should be under 5ms on any reasonable system
    assert(max_jitter_us < 5000);
    std::cout << "  PASS: Max jitter " << max_jitter_us << "µs (< 5000µs)\n";
}

void test_variable_workload_stats() {
    std::cout << "--- Test: Variable workload statistics ---\n";

    CyclicExecutor exec(
        workloads::control_cycle_variable,
        std::chrono::milliseconds(10),
        std::chrono::microseconds(5000)
    );

    auto stats = exec.run_cycles(20);
    // Min should be much smaller than max (variable workload)
    assert(stats.max_exec_time > stats.min_exec_time);
    print_stats("Variable workload", stats);
    std::cout << "  PASS: Variance detected (min < max)\n";
}

void test_stop_mechanism() {
    std::cout << "--- Test: Stop mechanism ---\n";

    std::atomic<uint64_t> count{0};
    CyclicExecutor exec(
        [&count]() { ++count; },
        std::chrono::milliseconds(5),
        std::chrono::microseconds(2000)
    );

    // Run in background thread, stop after a short delay
    std::thread runner([&exec]() {
        exec.run_cycles(1000);  // would take 5 seconds
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    exec.stop();
    runner.join();

    // Should have run far fewer than 1000 cycles
    uint64_t actual = count.load();
    assert(actual < 1000);
    assert(actual > 0);
    std::cout << "  PASS: Stopped after " << actual
              << " cycles (< 1000)\n";
}

// ============================================================================
int main() {
    std::cout << "=== Deterministic Cyclic Executor Exercise ===\n\n";

    test_basic_execution();
    test_overrun_detection();
    test_no_overruns_with_generous_budget();
    test_jitter_measurement();
    test_variable_workload_stats();
    test_stop_mechanism();

    std::cout << "\n=== ALL CYCLIC EXECUTOR TESTS PASSED ===\n";
    return 0;
}
