#pragma once
// ============================================================================
// SPSCQueue — Lock-free single-producer single-consumer queue
//
// - Power-of-2 capacity (compile-time)
// - acquire/release memory ordering
// - Cache-line padding to prevent false sharing
// ============================================================================

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <optional>
#include <type_traits>

namespace flight_sw {

// Cache line size for padding
#ifdef __cpp_lib_hardware_interference_size
    inline constexpr std::size_t CACHE_LINE = std::hardware_destructive_interference_size;
#else
    inline constexpr std::size_t CACHE_LINE = 64;
#endif

template <typename T, std::size_t Capacity>
class SPSCQueue {
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "Capacity must be a power of 2");
    static_assert(Capacity >= 2, "Capacity must be >= 2");
    static_assert(std::is_trivially_copyable_v<T> || std::is_move_constructible_v<T>,
                  "T must be trivially copyable or move constructible");

    static constexpr std::size_t MASK = Capacity - 1;

public:
    SPSCQueue() = default;

    // Non-copyable, non-movable (contains atomics)
    SPSCQueue(SPSCQueue const&) = delete;
    SPSCQueue& operator=(SPSCQueue const&) = delete;

    // Producer: try to push an element
    [[nodiscard]] inline bool try_push(T const& item) noexcept {
        std::size_t cur_head = head_.load(std::memory_order_relaxed);
        std::size_t next_head = (cur_head + 1) & MASK;

        if (next_head == tail_.load(std::memory_order_acquire)) {
            return false; // Full
        }

        buffer_[cur_head] = item;
        head_.store(next_head, std::memory_order_release);
        return true;
    }

    // Producer: try to push (move)
    [[nodiscard]] inline bool try_push(T&& item) noexcept {
        std::size_t cur_head = head_.load(std::memory_order_relaxed);
        std::size_t next_head = (cur_head + 1) & MASK;

        if (next_head == tail_.load(std::memory_order_acquire)) {
            return false; // Full
        }

        buffer_[cur_head] = std::move(item);
        head_.store(next_head, std::memory_order_release);
        return true;
    }

    // Consumer: try to pop an element
    [[nodiscard]] inline std::optional<T> try_pop() noexcept {
        std::size_t cur_tail = tail_.load(std::memory_order_relaxed);

        if (cur_tail == head_.load(std::memory_order_acquire)) {
            return std::nullopt; // Empty
        }

        T item = std::move(buffer_[cur_tail]);
        tail_.store((cur_tail + 1) & MASK, std::memory_order_release);
        return item;
    }

    // Approximate size (valid from either thread but may be stale)
    [[nodiscard]] inline std::size_t size() const noexcept {
        auto h = head_.load(std::memory_order_relaxed);
        auto t = tail_.load(std::memory_order_relaxed);
        return (h - t) & MASK;
    }

    [[nodiscard]] inline bool empty() const noexcept {
        return head_.load(std::memory_order_relaxed) ==
               tail_.load(std::memory_order_relaxed);
    }

    [[nodiscard]] static constexpr std::size_t capacity() noexcept {
        return Capacity - 1; // One slot always unused
    }

private:
    // Pad head and tail to separate cache lines
    alignas(CACHE_LINE) std::atomic<std::size_t> head_{0};
    alignas(CACHE_LINE) std::atomic<std::size_t> tail_{0};
    alignas(CACHE_LINE) std::array<T, Capacity>  buffer_{};
};

} // namespace flight_sw
