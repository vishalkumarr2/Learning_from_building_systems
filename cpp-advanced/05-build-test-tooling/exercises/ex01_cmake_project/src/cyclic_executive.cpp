// ============================================================================
// cyclic_executive.cpp — Implementation of the Cyclic Executive
// ============================================================================

#include "rt_core/cyclic_executive.hpp"

#include <algorithm>
#include <thread>

namespace rt_core {

CyclicExecutive::CyclicExecutive(std::chrono::microseconds cycle_period)
    : cycle_period_(cycle_period) {}

void CyclicExecutive::add_task(Task task) {
    tasks_.push_back(std::move(task));
}

void CyclicExecutive::run(std::size_t num_cycles) {
    running_.store(true, std::memory_order_relaxed);

    for (std::size_t i = 0; i < num_cycles; ++i) {
        if (!running_.load(std::memory_order_relaxed)) {
            break;
        }
        execute_one_cycle();
    }

    running_.store(false, std::memory_order_relaxed);
}

void CyclicExecutive::run_forever() {
    running_.store(true, std::memory_order_relaxed);

    while (running_.load(std::memory_order_relaxed)) {
        execute_one_cycle();
    }
}

void CyclicExecutive::stop() noexcept {
    running_.store(false, std::memory_order_relaxed);
}

CycleStats CyclicExecutive::stats() const noexcept {
    CycleStats s;
    s.total_cycles = total_cycles_;
    s.overruns = overruns_;
    s.max_cycle_time = max_cycle_time_;
    s.avg_cycle_time = (total_cycles_ > 0)
        ? std::chrono::microseconds(total_time_.count() /
                                    static_cast<long>(total_cycles_))
        : std::chrono::microseconds{0};
    return s;
}

void CyclicExecutive::reset_stats() noexcept {
    total_cycles_ = 0;
    overruns_ = 0;
    max_cycle_time_ = std::chrono::microseconds{0};
    total_time_ = std::chrono::microseconds{0};
}

void CyclicExecutive::execute_one_cycle() {
    const auto cycle_start = Clock::now();

    // Execute all tasks in registration order
    for (const auto& task : tasks_) {
        task.func();
    }

    const auto cycle_end = Clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
        cycle_end - cycle_start);

    // Update statistics
    ++total_cycles_;
    total_time_ += elapsed;
    max_cycle_time_ = std::max(max_cycle_time_, elapsed);

    if (elapsed > cycle_period_) {
        ++overruns_;
        // Overrun: don't sleep, start next cycle immediately
        return;
    }

    // Sleep for the remainder of the cycle period
    const auto remaining = cycle_period_ - elapsed;
    std::this_thread::sleep_for(remaining);
}

}  // namespace rt_core
