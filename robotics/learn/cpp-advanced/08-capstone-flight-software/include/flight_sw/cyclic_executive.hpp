#pragma once
// ============================================================================
// CyclicExecutive — RT cyclic executive with jitter tracking and task shedding
//
// - Configurable base period (default 1 ms)
// - Tasks registered at sub-multiples of base rate
// - clock_nanosleep for precise timing
// - SCHED_FIFO if available (graceful fallback)
// - Per-frame jitter histogram
// ============================================================================

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <sys/mman.h>
#endif

namespace flight_sw {

struct JitterStats {
    int64_t min_ns   = INT64_MAX;
    int64_t max_ns   = 0;
    int64_t sum_ns   = 0;
    uint64_t count   = 0;
    uint64_t overruns = 0;

    // Histogram: buckets [0,10), [10,20), ... [90,100), [100+) microseconds
    static constexpr std::size_t NUM_BUCKETS = 11;
    std::array<uint64_t, NUM_BUCKETS> histogram{};

    inline void record(int64_t jitter_ns) noexcept {
        if (jitter_ns < min_ns) min_ns = jitter_ns;
        if (jitter_ns > max_ns) max_ns = jitter_ns;
        sum_ns += jitter_ns;
        ++count;

        auto us = static_cast<std::size_t>(jitter_ns / 1000);
        std::size_t bucket = std::min(us / 10, NUM_BUCKETS - 1);
        ++histogram[bucket];
    }

    [[nodiscard]] inline double mean_us() const noexcept {
        return count > 0 ? static_cast<double>(sum_ns) / static_cast<double>(count) / 1000.0 : 0.0;
    }

    inline void print() const {
        std::cout << "  Jitter: min=" << min_ns/1000 << "us, max=" << max_ns/1000
                  << "us, mean=" << mean_us() << "us, overruns=" << overruns
                  << ", frames=" << count << "\n";
        std::cout << "  Histogram (us): ";
        for (std::size_t i = 0; i < NUM_BUCKETS; ++i) {
            if (i < NUM_BUCKETS - 1)
                std::cout << "[" << i*10 << "-" << (i+1)*10 << "):" << histogram[i] << " ";
            else
                std::cout << "[100+):" << histogram[i];
        }
        std::cout << "\n";
    }
};

struct TaskEntry {
    std::string name;
    uint32_t    divisor;          // run every N-th frame
    std::function<void()> callback;
    uint64_t    exec_count = 0;
    bool        shed       = false; // shed if overrunning
};

class CyclicExecutive {
public:
    explicit CyclicExecutive(std::chrono::nanoseconds period =
                                 std::chrono::microseconds{1000})
        : period_ns_{period}
    {}

    // Register a task at a given rate (rate_hz must be ≤ base_rate)
    inline void register_task(std::string name, uint32_t rate_hz,
                              std::function<void()> callback) {
        uint32_t base_hz = static_cast<uint32_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::seconds{1}).count() / period_ns_.count());
        uint32_t divisor = (rate_hz > 0 && rate_hz <= base_hz)
                               ? base_hz / rate_hz
                               : 1;
        tasks_.push_back({std::move(name), divisor, std::move(callback), 0, false});
    }

    // Run the executive for `duration`
    inline void run(std::chrono::nanoseconds duration) {
        running_.store(true, std::memory_order_release);
        try_set_realtime();

        uint64_t total_frames = static_cast<uint64_t>(duration.count() / period_ns_.count());
        if (total_frames == 0) total_frames = 1;

#ifdef __linux__
        struct timespec next;
        clock_gettime(CLOCK_MONOTONIC, &next);
#else
        auto next = std::chrono::steady_clock::now();
#endif

        for (uint64_t frame = 0; frame < total_frames; ++frame) {
            if (!running_.load(std::memory_order_acquire)) break;

            // Advance target time
#ifdef __linux__
            next.tv_nsec += static_cast<long>(period_ns_.count());
            while (next.tv_nsec >= 1'000'000'000L) {
                next.tv_nsec -= 1'000'000'000L;
                next.tv_sec += 1;
            }
#else
            next += period_ns_;
#endif

            // Execute tasks due this frame
            for (auto& task : tasks_) {
                if (task.shed) continue;
                if ((frame % task.divisor) == 0) {
                    task.callback();
                    ++task.exec_count;
                }
            }

            // Sleep until next frame
#ifdef __linux__
            clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &next, nullptr);
#else
            std::this_thread::sleep_until(next);
#endif

            // Measure jitter (how late we woke up)
#ifdef __linux__
            struct timespec now;
            clock_gettime(CLOCK_MONOTONIC, &now);
            int64_t jitter = (now.tv_sec - next.tv_sec) * 1'000'000'000LL
                           + (now.tv_nsec - next.tv_nsec);
#else
            auto now = std::chrono::steady_clock::now();
            int64_t jitter = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 now - next).count();
#endif
            if (jitter < 0) jitter = 0;
            jitter_.record(jitter);

            // Overrun detection: if jitter > period, shed lowest-priority task
            if (jitter > period_ns_.count()) {
                ++jitter_.overruns;
                if (!tasks_.empty()) {
                    tasks_.back().shed = true; // shed last registered = lowest prio
                }
            }
        }

        running_.store(false, std::memory_order_release);
    }

    inline void stop() noexcept { running_.store(false, std::memory_order_release); }

    [[nodiscard]] inline JitterStats const& jitter_stats() const noexcept { return jitter_; }
    [[nodiscard]] inline std::vector<TaskEntry> const& tasks() const noexcept { return tasks_; }

    inline void print_stats() const {
        std::cout << "=== Cyclic Executive Stats ===\n";
        jitter_.print();
        std::cout << "  Tasks:\n";
        for (auto const& t : tasks_) {
            std::cout << "    " << t.name << ": " << t.exec_count
                      << " executions (1/" << t.divisor << " frames)"
                      << (t.shed ? " [SHED]" : "") << "\n";
        }
    }

private:
    std::chrono::nanoseconds period_ns_;
    std::vector<TaskEntry>   tasks_;
    JitterStats              jitter_;
    std::atomic<bool>        running_{false};

    inline void try_set_realtime() const noexcept {
#ifdef __linux__
        // Try SCHED_FIFO — requires CAP_SYS_NICE or root
        struct sched_param sp{};
        sp.sched_priority = 80;
        if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0) {
            // Graceful fallback — run with default scheduler
        }

        // Lock memory pages to prevent page faults
        mlockall(MCL_CURRENT | MCL_FUTURE);
#endif
    }
};

} // namespace flight_sw
