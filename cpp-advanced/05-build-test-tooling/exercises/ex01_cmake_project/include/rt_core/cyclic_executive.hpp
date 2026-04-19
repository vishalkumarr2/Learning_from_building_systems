#pragma once
// ============================================================================
// cyclic_executive.hpp — Fixed-Rate Cyclic Task Scheduler
//
// Runs a set of registered tasks at a fixed cycle period.
// Tracks overruns (cycle exceeded deadline) and provides timing stats.
// ============================================================================

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace rt_core {

/// A single task registered with the cyclic executive.
struct Task {
    std::string name;
    std::function<void()> func;
    std::chrono::microseconds worst_case_execution_time{0};
};

/// Timing statistics for the cyclic executive.
struct CycleStats {
    std::uint64_t total_cycles{0};
    std::uint64_t overruns{0};
    std::chrono::microseconds max_cycle_time{0};
    std::chrono::microseconds avg_cycle_time{0};
};

/// @brief A simple fixed-rate cyclic executive / rate-monotonic scheduler.
///
/// Usage:
/// @code
///   CyclicExecutive exec(std::chrono::milliseconds(10));  // 100 Hz
///   exec.add_task({"sensor_read", read_sensors, 2000us});
///   exec.add_task({"control_update", run_controller, 3000us});
///   exec.run(1000);  // run 1000 cycles
///   auto stats = exec.stats();
/// @endcode
class CyclicExecutive {
public:
    using Clock = std::chrono::steady_clock;

    /// @param cycle_period The fixed period for each cycle.
    explicit CyclicExecutive(std::chrono::microseconds cycle_period);

    /// @brief Register a task. Tasks run in the order they are added.
    void add_task(Task task);

    /// @brief Run for a fixed number of cycles. Blocks until complete or stopped.
    void run(std::size_t num_cycles);

    /// @brief Run indefinitely until stop() is called from another thread.
    void run_forever();

    /// @brief Signal the executive to stop after the current cycle.
    void stop() noexcept;

    /// @brief Get timing statistics. Safe to call after run() returns.
    [[nodiscard]] CycleStats stats() const noexcept;

    /// @brief Reset statistics counters.
    void reset_stats() noexcept;

private:
    void execute_one_cycle();

    std::chrono::microseconds cycle_period_;
    std::vector<Task> tasks_;
    std::atomic<bool> running_{false};

    // Stats (updated in execute_one_cycle)
    std::uint64_t total_cycles_{0};
    std::uint64_t overruns_{0};
    std::chrono::microseconds max_cycle_time_{0};
    std::chrono::microseconds total_time_{0};
};

}  // namespace rt_core
