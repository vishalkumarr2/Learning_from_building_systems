#pragma once
// ============================================================================
// spsc_queue.hpp — Lock-Free Single-Producer Single-Consumer Queue
//
// Header-only. Uses acquire/release atomics for thread safety between
// exactly ONE producer thread and ONE consumer thread.
//
// NOT safe for multiple producers or multiple consumers.
// ============================================================================

#include <atomic>
#include <array>
#include <cassert>
#include <cstddef>
#include <new>       // std::hardware_destructive_interference_size
#include <optional>
#include <type_traits>

namespace rt_core {

// Cache line size fallback for compilers that don't support the C++17 constant
#ifdef __cpp_lib_hardware_interference_size
inline constexpr std::size_t kCacheLineSize = std::hardware_destructive_interference_size;
#else
inline constexpr std::size_t kCacheLineSize = 64;
#endif

/// @brief Lock-free SPSC (Single-Producer, Single-Consumer) bounded queue.
///
/// @tparam T     Element type. Must be nothrow move-constructible.
/// @tparam Cap   Fixed capacity. Must be > 0.
///
/// Memory layout pads head_ and tail_ to separate cache lines to avoid
/// false sharing between producer and consumer.
template <typename T, std::size_t Cap>
class SpscQueue {
    static_assert(Cap > 0, "Queue capacity must be greater than zero");
    static_assert(std::is_nothrow_move_constructible_v<T>,
                  "T must be nothrow move-constructible for lock-free safety");

public:
    SpscQueue() = default;

    // Non-copyable, non-movable (contains atomics)
    SpscQueue(const SpscQueue&) = delete;
    SpscQueue& operator=(const SpscQueue&) = delete;
    SpscQueue(SpscQueue&&) = delete;
    SpscQueue& operator=(SpscQueue&&) = delete;

    /// @brief Try to push a value. Returns false if the queue is full.
    /// @note Only call from the producer thread.
    [[nodiscard]] bool try_push(const T& value) noexcept {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t next = (head + 1) % (Cap + 1);

        if (next == tail_.load(std::memory_order_acquire)) {
            return false;  // Queue is full
        }

        buffer_[head] = value;
        head_.store(next, std::memory_order_release);
        return true;
    }

    /// @brief Try to push by moving. Returns false if the queue is full.
    /// @note Only call from the producer thread.
    [[nodiscard]] bool try_push(T&& value) noexcept {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t next = (head + 1) % (Cap + 1);

        if (next == tail_.load(std::memory_order_acquire)) {
            return false;  // Queue is full
        }

        buffer_[head] = std::move(value);
        head_.store(next, std::memory_order_release);
        return true;
    }

    /// @brief Try to pop a value. Returns std::nullopt if the queue is empty.
    /// @note Only call from the consumer thread.
    [[nodiscard]] std::optional<T> try_pop() noexcept {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);

        if (tail == head_.load(std::memory_order_acquire)) {
            return std::nullopt;  // Queue is empty
        }

        T value = std::move(buffer_[tail]);
        tail_.store((tail + 1) % (Cap + 1), std::memory_order_release);
        return value;
    }

    /// @brief Check if the queue is empty.
    /// @note Relaxed — only accurate from the consumer thread.
    [[nodiscard]] bool empty() const noexcept {
        return head_.load(std::memory_order_relaxed) ==
               tail_.load(std::memory_order_relaxed);
    }

    /// @brief Approximate number of elements. May be stale.
    [[nodiscard]] std::size_t size_approx() const noexcept {
        const auto h = head_.load(std::memory_order_relaxed);
        const auto t = tail_.load(std::memory_order_relaxed);
        return (h >= t) ? (h - t) : (Cap + 1 - t + h);
    }

    /// @brief Fixed capacity of the queue.
    [[nodiscard]] static constexpr std::size_t capacity() noexcept {
        return Cap;
    }

private:
    // Ring buffer with Cap+1 slots (one slot wasted to distinguish full vs empty)
    std::array<T, Cap + 1> buffer_{};

    // Pad to separate cache lines — producer writes head_, consumer writes tail_
    alignas(kCacheLineSize) std::atomic<std::size_t> head_{0};
    alignas(kCacheLineSize) std::atomic<std::size_t> tail_{0};
};

}  // namespace rt_core
