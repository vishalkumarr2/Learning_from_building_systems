#pragma once
// ============================================================================
// HealthMonitor — Subsystem heartbeat rate monitoring
//
// - register_subsystem(name, expected_rate_hz, max_missed_beats)
// - heartbeat(name)  — called by each subsystem every iteration
// - check()          — returns names of failed subsystems
// ============================================================================

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace flight_sw {

struct SubsystemHealth {
    std::string name;
    double      expected_rate_hz;
    uint32_t    max_missed;
    uint64_t    beat_count    = 0;
    uint64_t    last_beat_ns  = 0;
    uint64_t    missed_beats  = 0;
    bool        failed        = false;
};

struct HealthEvent {
    uint64_t    timestamp_ns;
    std::string subsystem;
    std::string detail;
};

class HealthMonitor {
public:
    HealthMonitor() = default;

    inline void register_subsystem(std::string name, double expected_rate_hz,
                                   uint32_t max_missed = 5) {
        SubsystemHealth h;
        h.name = name;
        h.expected_rate_hz = expected_rate_hz;
        h.max_missed = max_missed;
        subsystems_[name] = h;
    }

    // Called by each subsystem to report liveness
    inline void heartbeat(std::string const& name) {
        auto it = subsystems_.find(name);
        if (it == subsystems_.end()) return;

        auto& s = it->second;
        s.last_beat_ns = now_ns();
        ++s.beat_count;
        s.missed_beats = 0;  // Reset on successful beat
        s.failed = false;
    }

    // Called periodically to check all subsystems; returns list of failed names
    [[nodiscard]] inline std::vector<std::string> check() {
        std::vector<std::string> failed;
        uint64_t current = now_ns();

        for (auto& [name, s] : subsystems_) {
            if (s.beat_count == 0) continue; // Not yet started

            double expected_period_ns = 1.0e9 / s.expected_rate_hz;
            double elapsed = static_cast<double>(current - s.last_beat_ns);

            if (elapsed > expected_period_ns * 2.0) {
                auto beats_missed = static_cast<uint64_t>(elapsed / expected_period_ns);
                s.missed_beats = beats_missed;

                if (beats_missed > s.max_missed && !s.failed) {
                    s.failed = true;
                    failed.push_back(name);
                    events_.push_back({current, name,
                        "missed " + std::to_string(beats_missed) + " beats"});
                }
            }
        }

        return failed;
    }

    [[nodiscard]] inline std::vector<HealthEvent> const& events() const noexcept {
        return events_;
    }

    [[nodiscard]] inline SubsystemHealth const* get(std::string const& name) const {
        auto it = subsystems_.find(name);
        return it != subsystems_.end() ? &it->second : nullptr;
    }

    inline void print_events() const {
        if (events_.empty()) {
            std::cout << "  No health events recorded.\n";
            return;
        }
        for (auto const& e : events_) {
            std::cout << "  [" << e.timestamp_ns / 1'000'000 << "ms] "
                      << e.subsystem << ": " << e.detail << "\n";
        }
    }

private:
    std::unordered_map<std::string, SubsystemHealth> subsystems_;
    std::vector<HealthEvent> events_;

    [[nodiscard]] static inline uint64_t now_ns() noexcept {
        auto tp = std::chrono::steady_clock::now().time_since_epoch();
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(tp).count());
    }
};

} // namespace flight_sw
