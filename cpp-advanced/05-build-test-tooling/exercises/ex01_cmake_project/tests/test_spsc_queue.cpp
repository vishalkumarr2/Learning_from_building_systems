// Placeholder test file for rt_core project structure.
// The full SPSC queue test suite is in ex02_gtest_suite.cpp.

#include <gtest/gtest.h>
#include "rt_core/spsc_queue.hpp"

TEST(SpscQueueBasic, PushAndPop) {
    rt_core::SpscQueue<int, 4> q;
    ASSERT_TRUE(q.try_push(42));
    auto val = q.try_pop();
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, 42);
}

TEST(SpscQueueBasic, EmptyReturnsNullopt) {
    rt_core::SpscQueue<int, 4> q;
    EXPECT_EQ(q.try_pop(), std::nullopt);
}
