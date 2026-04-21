// Placeholder test for CyclicExecutive.

#include <gtest/gtest.h>
#include "rt_core/cyclic_executive.hpp"

#include <chrono>

using namespace std::chrono_literals;

TEST(CyclicExecutive, RunsCorrectNumberOfCycles) {
    rt_core::CyclicExecutive exec(1000us);  // 1ms cycle

    int counter = 0;
    exec.add_task({"increment", [&counter]() { ++counter; }, 100us});
    exec.run(10);

    EXPECT_EQ(counter, 10);
}

TEST(CyclicExecutive, StatsTracking) {
    rt_core::CyclicExecutive exec(1000us);
    exec.add_task({"noop", []() {}, 10us});
    exec.run(5);

    auto stats = exec.stats();
    EXPECT_EQ(stats.total_cycles, 5u);
    EXPECT_EQ(stats.overruns, 0u);
    EXPECT_GT(stats.max_cycle_time.count(), 0);
}
