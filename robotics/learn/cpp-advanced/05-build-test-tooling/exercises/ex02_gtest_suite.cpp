// ============================================================================
// ex02_gtest_suite.cpp — GoogleTest Suite for an SPSC Queue
//
// Tests: basic push/pop, full/empty behavior, concurrent stress,
//        parameterized tests, death tests.
//
// Build: Requires GoogleTest. The parent CMakeLists.txt fetches it via
//        FetchContent. If building standalone:
//          g++ -std=c++20 -I/path/to/gtest/include ex02_gtest_suite.cpp \
//              -lgtest -lgtest_main -pthread -o ex02
//
// ============================================================================

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <thread>
#include <vector>
#include <array>

// ---------------------------------------------------------------------------
// Minimal SPSC Queue (self-contained for this test file)
// ---------------------------------------------------------------------------
namespace {

template <typename T, std::size_t Cap>
class SpscQueue {
    static_assert(Cap > 0);

public:
    SpscQueue() = default;
    SpscQueue(const SpscQueue&) = delete;
    SpscQueue& operator=(const SpscQueue&) = delete;

    [[nodiscard]] bool try_push(const T& value) noexcept {
        const auto h = head_.load(std::memory_order_relaxed);
        const auto next = (h + 1) % kSlots;
        if (next == tail_.load(std::memory_order_acquire))
            return false;
        buf_[h] = value;
        head_.store(next, std::memory_order_release);
        return true;
    }

    [[nodiscard]] std::optional<T> try_pop() noexcept {
        const auto t = tail_.load(std::memory_order_relaxed);
        if (t == head_.load(std::memory_order_acquire))
            return std::nullopt;
        T val = buf_[t];
        tail_.store((t + 1) % kSlots, std::memory_order_release);
        return val;
    }

    [[nodiscard]] bool empty() const noexcept {
        return head_.load(std::memory_order_relaxed) ==
               tail_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] static constexpr std::size_t capacity() noexcept { return Cap; }

private:
    static constexpr std::size_t kSlots = Cap + 1;
    std::array<T, kSlots> buf_{};
    alignas(64) std::atomic<std::size_t> head_{0};
    alignas(64) std::atomic<std::size_t> tail_{0};
};

}  // namespace

// ===========================================================================
// Test Fixture
// ===========================================================================
class SpscQueueTest : public ::testing::Test {
protected:
    SpscQueue<int, 4> q_;  // capacity = 4
};

// ---------------------------------------------------------------------------
// Basic Push/Pop
// ---------------------------------------------------------------------------
TEST_F(SpscQueueTest, PushAndPop) {
    ASSERT_TRUE(q_.try_push(10));
    auto val = q_.try_pop();
    ASSERT_TRUE(val.has_value());
    EXPECT_EQ(*val, 10);
}

TEST_F(SpscQueueTest, FIFOOrder) {
    q_.try_push(1);
    q_.try_push(2);
    q_.try_push(3);

    EXPECT_EQ(q_.try_pop(), 1);
    EXPECT_EQ(q_.try_pop(), 2);
    EXPECT_EQ(q_.try_pop(), 3);
}

// ---------------------------------------------------------------------------
// Empty Queue Behavior
// ---------------------------------------------------------------------------
TEST_F(SpscQueueTest, PopFromEmptyReturnsNullopt) {
    EXPECT_EQ(q_.try_pop(), std::nullopt);
}

TEST_F(SpscQueueTest, EmptyAfterDrainingAll) {
    q_.try_push(1);
    q_.try_push(2);
    q_.try_pop();
    q_.try_pop();
    EXPECT_TRUE(q_.empty());
    EXPECT_EQ(q_.try_pop(), std::nullopt);
}

// ---------------------------------------------------------------------------
// Full Queue Behavior
// ---------------------------------------------------------------------------
TEST_F(SpscQueueTest, FullQueueRejectsPush) {
    // Capacity is 4
    EXPECT_TRUE(q_.try_push(1));
    EXPECT_TRUE(q_.try_push(2));
    EXPECT_TRUE(q_.try_push(3));
    EXPECT_TRUE(q_.try_push(4));
    EXPECT_FALSE(q_.try_push(5));  // Queue is full
}

TEST_F(SpscQueueTest, PushAfterPopFromFull) {
    for (int i = 0; i < 4; ++i) q_.try_push(i);
    ASSERT_FALSE(q_.try_push(99));  // full

    q_.try_pop();                   // free one slot
    EXPECT_TRUE(q_.try_push(99));   // now it fits
}

// ---------------------------------------------------------------------------
// Wraparound
// ---------------------------------------------------------------------------
TEST_F(SpscQueueTest, WraparoundManyTimes) {
    // Push and pop way more than capacity to exercise ring buffer wrapping
    for (int i = 0; i < 1000; ++i) {
        ASSERT_TRUE(q_.try_push(i));
        auto val = q_.try_pop();
        ASSERT_TRUE(val.has_value());
        EXPECT_EQ(*val, i);
    }
}

// ---------------------------------------------------------------------------
// Concurrent Stress Test: 1 producer thread, 1 consumer thread
// ---------------------------------------------------------------------------
TEST(SpscQueueConcurrent, ProducerConsumerStress) {
    constexpr std::size_t kCapacity = 256;
    constexpr int kNumItems = 100'000;

    SpscQueue<int, kCapacity> q;
    std::atomic<bool> done{false};

    // Producer thread
    std::thread producer([&]() {
        for (int i = 0; i < kNumItems; ++i) {
            while (!q.try_push(i)) {
                // spin — queue is full, consumer will catch up
            }
        }
        done.store(true, std::memory_order_release);
    });

    // Consumer thread: verify FIFO order and collect all items
    std::vector<int> received;
    received.reserve(kNumItems);

    std::thread consumer([&]() {
        int expected = 0;
        while (expected < kNumItems) {
            auto val = q.try_pop();
            if (val.has_value()) {
                received.push_back(*val);
                ++expected;
            }
        }
    });

    producer.join();
    consumer.join();

    // Verify we got every value in order
    ASSERT_EQ(received.size(), static_cast<std::size_t>(kNumItems));
    for (int i = 0; i < kNumItems; ++i) {
        EXPECT_EQ(received[static_cast<std::size_t>(i)], i)
            << "Mismatch at index " << i;
    }
}

// ---------------------------------------------------------------------------
// Parameterized Test: Different Queue Sizes
// ---------------------------------------------------------------------------
class QueueSizeTest : public ::testing::TestWithParam<std::size_t> {};

TEST_P(QueueSizeTest, FillAndDrain) {
    // We can't use GetParam() as template parameter at runtime,
    // so we test a fixed large queue and fill up to GetParam() items.
    constexpr std::size_t kMaxCap = 1024;
    SpscQueue<int, kMaxCap> q;

    const auto fill_count = std::min(GetParam(), kMaxCap);

    // Fill
    for (std::size_t i = 0; i < fill_count; ++i) {
        ASSERT_TRUE(q.try_push(static_cast<int>(i)));
    }

    // Drain and verify
    for (std::size_t i = 0; i < fill_count; ++i) {
        auto val = q.try_pop();
        ASSERT_TRUE(val.has_value());
        EXPECT_EQ(*val, static_cast<int>(i));
    }

    EXPECT_TRUE(q.empty());
}

INSTANTIATE_TEST_SUITE_P(
    Sizes,
    QueueSizeTest,
    ::testing::Values(1, 2, 4, 16, 64, 256, 1024)
);

// ---------------------------------------------------------------------------
// Capacity
// ---------------------------------------------------------------------------
TEST(SpscQueueMeta, CapacityIsCorrect) {
    SpscQueue<int, 7> q;
    EXPECT_EQ(q.capacity(), 7u);

    SpscQueue<double, 128> q2;
    EXPECT_EQ(q2.capacity(), 128u);
}
