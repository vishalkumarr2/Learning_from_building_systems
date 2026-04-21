#pragma once
// ============================================================================
// TraceBuffer — Fixed-size ring buffer for flight data recording
//
// - Atomic write index for RT-safe record()
// - Lock-free single-writer (main loop), reader only after stop
// - dump() for post-mortem analysis
// ============================================================================

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>

namespace flight_sw {

struct TraceEntry {
    uint64_t    timestamp_ns{0};
    char        category[16]{};   // "MODE", "SENSOR", "PID", "HEALTH", etc.
    char        message[112]{};   // Fixed-size, no heap allocation

    inline void set(std::string_view cat, std::string_view msg) noexcept {
        auto cat_len = std::min(cat.size(), sizeof(category) - 1);
        std::memcpy(category, cat.data(), cat_len);
        category[cat_len] = '\0';

        auto msg_len = std::min(msg.size(), sizeof(message) - 1);
        std::memcpy(message, msg.data(), msg_len);
        message[msg_len] = '\0';
    }
};

static_assert(sizeof(TraceEntry) <= 256,
              "TraceEntry should remain small for cache efficiency");

template <std::size_t N = 1024>
class TraceBuffer {
    static_assert((N & (N - 1)) == 0, "N must be a power of 2");

public:
    TraceBuffer() = default;

    // RT-safe: called from the cyclic executive loop
    inline void record(std::string_view category, std::string_view message) noexcept {
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        uint64_t ts = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());

        std::size_t slot = write_idx_.fetch_add(1, std::memory_order_relaxed) & MASK;
        entries_[slot].timestamp_ns = ts;
        entries_[slot].set(category, message);
        if (count_.load(std::memory_order_relaxed) < N) {
            count_.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Non-RT: dump trace to stdout (call after stopping executive)
    inline void dump(std::size_t max_entries = 0) const {
        std::size_t total = count_.load(std::memory_order_relaxed);
        std::size_t head  = write_idx_.load(std::memory_order_relaxed);

        if (total == 0) {
            std::cout << "  Trace buffer empty.\n";
            return;
        }

        std::size_t to_show = (max_entries > 0 && max_entries < total)
                                  ? max_entries : total;

        // Start from oldest entry
        std::size_t start = (total >= N) ? head : 0;

        // If limiting, skip to most recent
        if (to_show < total) {
            start = (head - to_show) & MASK;
        }

        std::cout << "  Trace buffer (" << to_show << " of " << total << " entries):\n";
        for (std::size_t i = 0; i < to_show; ++i) {
            std::size_t idx = (start + i) & MASK;
            auto const& e = entries_[idx];
            if (e.timestamp_ns == 0) continue;
            std::cout << "    [" << e.timestamp_ns / 1'000'000 << "ms] "
                      << e.category << ": " << e.message << "\n";
        }
    }

    [[nodiscard]] inline std::size_t count() const noexcept {
        return count_.load(std::memory_order_relaxed);
    }

    inline void clear() noexcept {
        write_idx_.store(0, std::memory_order_relaxed);
        count_.store(0, std::memory_order_relaxed);
    }

private:
    static constexpr std::size_t MASK = N - 1;

    std::array<TraceEntry, N>    entries_{};
    std::atomic<std::size_t>     write_idx_{0};
    std::atomic<std::size_t>     count_{0};
};

} // namespace flight_sw
