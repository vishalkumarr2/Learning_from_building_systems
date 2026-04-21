// =============================================================================
// Exercise 05: Week 4 Mini-Project — Watchdog + Health Monitor + RT Loop
// =============================================================================
// Integrates:
//   - Cyclic executive from Week 3 (100Hz main loop)
//   - 3 subsystems: sensor (50Hz), controller (100Hz), telemetry (10Hz)
//   - Health monitor at 1Hz
//   - Software watchdog: main loop must pet every cycle or enter safe mode
//   - Fault injection: sensor hangs after 5 seconds
//   - Degraded mode detection within ~1 second
//   - State transitions and timing printout
// =============================================================================

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <array>

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using Ms = std::chrono::milliseconds;
using Us = std::chrono::microseconds;

// ======================== SYSTEM STATE =======================================

enum class SystemMode {
    kNormal,
    kDegraded,
    kSafeMode,
    kShutdown
};

const char* mode_str(SystemMode m) {
    switch (m) {
        case SystemMode::kNormal:   return "NORMAL";
        case SystemMode::kDegraded: return "DEGRADED";
        case SystemMode::kSafeMode: return "SAFE_MODE";
        case SystemMode::kShutdown: return "SHUTDOWN";
    }
    return "UNKNOWN";
}

struct SystemState {
    std::atomic<SystemMode> mode{SystemMode::kNormal};
    std::atomic<bool> running{true};
    TimePoint start_time{Clock::now()};

    int64_t elapsed_ms() const {
        return std::chrono::duration_cast<Ms>(Clock::now() - start_time).count();
    }
};

// ======================== EVENT LOG ==========================================

struct LogEntry {
    int64_t time_ms;
    std::string source;
    std::string event;
};

class EventLog {
public:
    void log(const SystemState& state, const std::string& source,
             const std::string& event) {
        std::lock_guard<std::mutex> lock(mtx_);
        entries_.push_back({state.elapsed_ms(), source, event});
    }

    void print() const {
        std::lock_guard<std::mutex> lock(mtx_);
        std::cout << "\n" << std::string(70, '=') << "\n";
        std::cout << "  STATE TRANSITION LOG\n";
        std::cout << std::string(70, '=') << "\n";
        std::cout << std::setw(10) << "Time(ms)" << " | "
                  << std::setw(16) << "Source" << " | " << "Event\n";
        std::cout << std::string(70, '-') << "\n";
        for (const auto& e : entries_) {
            std::cout << std::setw(10) << e.time_ms << " | "
                      << std::setw(16) << e.source << " | "
                      << e.event << "\n";
        }
        std::cout << std::string(70, '=') << "\n";
    }

private:
    mutable std::mutex mtx_;
    std::vector<LogEntry> entries_;
};

EventLog g_log;

// ======================== SUBSYSTEM ==========================================

struct SubsystemInfo {
    std::string name;
    int target_hz;           // target frequency
    int deadline_ms;         // max time between heartbeats
    std::atomic<uint64_t> beat_count{0};
    TimePoint last_beat{Clock::now()};
    std::mutex beat_mtx;
    std::atomic<bool> healthy{true};
    int consecutive_misses{0};

    SubsystemInfo(std::string n, int hz, int dl)
        : name(std::move(n)), target_hz(hz), deadline_ms(dl) {}

    void heartbeat() {
        beat_count.fetch_add(1);
        std::lock_guard<std::mutex> lock(beat_mtx);
        last_beat = Clock::now();
    }

    TimePoint get_last_beat() {
        std::lock_guard<std::mutex> lock(beat_mtx);
        return last_beat;
    }
};

// ======================== SOFTWARE WATCHDOG ==================================
// The main loop must pet this every cycle. If it doesn't, system enters safe mode.

class SoftwareWatchdog {
public:
    SoftwareWatchdog(SystemState& state, int timeout_ms)
        : state_(state)
        , timeout_(Ms(timeout_ms))
        , last_pet_(Clock::now())
    {
        assert(timeout_ms > 0 && timeout_ms <= 5000);
    }

    void pet() {
        std::lock_guard<std::mutex> lock(pet_mtx_);
        last_pet_ = Clock::now();
    }

    void check() {
        Ms elapsed;
        {
            std::lock_guard<std::mutex> lock(pet_mtx_);
            elapsed = std::chrono::duration_cast<Ms>(Clock::now() - last_pet_);
        }
        if (elapsed > timeout_) {
            if (state_.mode.load() != SystemMode::kSafeMode) {
                g_log.log(state_, "SoftwareWDT",
                          "EXPIRED (" + std::to_string(elapsed.count()) +
                          "ms) — entering SAFE MODE");
                state_.mode.store(SystemMode::kSafeMode);
            }
        }
    }

private:
    SystemState& state_;
    Ms timeout_;
    std::mutex pet_mtx_;
    TimePoint last_pet_;
};

// ======================== HEALTH MONITOR (1Hz) ==============================

class HealthMonitor {
public:
    HealthMonitor(SystemState& state, std::vector<SubsystemInfo*> subs)
        : state_(state)
        , subsystems_(std::move(subs))
    {}

    void check() {
        if (state_.mode.load() == SystemMode::kSafeMode ||
            state_.mode.load() == SystemMode::kShutdown) {
            return;
        }

        int unhealthy_count = 0;

        for (auto* sub : subsystems_) {
            auto elapsed = std::chrono::duration_cast<Ms>(
                Clock::now() - sub->get_last_beat());

            if (elapsed.count() > sub->deadline_ms) {
                sub->consecutive_misses++;
                if (sub->consecutive_misses == 1) {
                    g_log.log(state_, "HealthMonitor",
                              sub->name + " missed deadline (" +
                              std::to_string(elapsed.count()) + "ms > " +
                              std::to_string(sub->deadline_ms) + "ms) — WARNING");
                } else if (sub->consecutive_misses == 2) {
                    sub->healthy.store(false);
                    g_log.log(state_, "HealthMonitor",
                              sub->name + " FAILED (2 consecutive misses)");
                }
                if (!sub->healthy.load()) ++unhealthy_count;
            } else {
                if (sub->consecutive_misses > 0) {
                    g_log.log(state_, "HealthMonitor",
                              sub->name + " recovered");
                }
                sub->consecutive_misses = 0;
                sub->healthy.store(true);
            }
        }

        // Transition logic
        auto current = state_.mode.load();
        if (unhealthy_count > 0 && current == SystemMode::kNormal) {
            state_.mode.store(SystemMode::kDegraded);
            g_log.log(state_, "HealthMonitor",
                      "Entering DEGRADED mode (" +
                      std::to_string(unhealthy_count) + " subsystem(s) failed)");
        } else if (unhealthy_count == 0 && current == SystemMode::kDegraded) {
            state_.mode.store(SystemMode::kNormal);
            g_log.log(state_, "HealthMonitor", "All subsystems recovered — NORMAL");
        }

        if (unhealthy_count >= 2) {
            g_log.log(state_, "HealthMonitor",
                      "Multiple failures — entering SAFE MODE");
            state_.mode.store(SystemMode::kSafeMode);
        }
    }

private:
    SystemState& state_;
    std::vector<SubsystemInfo*> subsystems_;
};

// ======================== SUBSYSTEM WORKERS ==================================

void sensor_worker(SubsystemInfo& info, SystemState& state, int hang_after_ms) {
    g_log.log(state, info.name, "Worker started @ " +
              std::to_string(info.target_hz) + "Hz");
    auto start = Clock::now();
    int period_us = 1000000 / info.target_hz;

    while (state.running.load()) {
        auto elapsed = std::chrono::duration_cast<Ms>(Clock::now() - start);
        if (hang_after_ms > 0 && elapsed.count() > hang_after_ms) {
            g_log.log(state, info.name, "*** FAULT INJECTED — HANGING ***");
            while (state.running.load()) {
                std::this_thread::sleep_for(Ms(100));
            }
            return;
        }
        info.heartbeat();
        std::this_thread::sleep_for(Us(period_us));
    }
    g_log.log(state, info.name, "Worker stopped");
}

void generic_worker(SubsystemInfo& info, SystemState& state) {
    g_log.log(state, info.name, "Worker started @ " +
              std::to_string(info.target_hz) + "Hz");
    int period_us = 1000000 / info.target_hz;

    while (state.running.load()) {
        info.heartbeat();
        std::this_thread::sleep_for(Us(period_us));
    }
    g_log.log(state, info.name, "Worker stopped");
}

// ======================== MAIN (CYCLIC EXECUTIVE) ============================

int main() {
    std::cout << "=== Week 4 Mini-Project: Watchdog + Health Monitor + RT Loop ===\n";
    std::cout << "Sensor will hang after 5 seconds. Watch for detection and degradation.\n";

    SystemState state;

    // Subsystem definitions
    SubsystemInfo sensor{"Sensor",     50,  60};   // 50Hz, 60ms deadline
    SubsystemInfo controller{"Controller", 100, 30};  // 100Hz, 30ms deadline
    SubsystemInfo telemetry{"Telemetry",  10,  200}; // 10Hz, 200ms deadline

    // Health monitor (1Hz check)
    HealthMonitor health(state, {&sensor, &controller, &telemetry});

    // Software watchdog: main loop has 50ms to pet
    SoftwareWatchdog swdt(state, 50);

    g_log.log(state, "Main", "System starting — mode=NORMAL");

    // Launch subsystem workers
    std::thread t_sensor(sensor_worker, std::ref(sensor), std::ref(state), 5000);
    std::thread t_controller(generic_worker, std::ref(controller), std::ref(state));
    std::thread t_telemetry(generic_worker, std::ref(telemetry), std::ref(state));

    g_log.log(state, "Main", "All workers launched");

    // Cyclic executive: 100Hz main loop
    constexpr int MAIN_LOOP_HZ = 100;
    constexpr int MAIN_LOOP_PERIOD_US = 1000000 / MAIN_LOOP_HZ;
    constexpr int HEALTH_CHECK_DIVISOR = MAIN_LOOP_HZ; // check at 1Hz
    constexpr int SIM_DURATION_S = 8;

    uint64_t cycle = 0;
    auto loop_start = Clock::now();

    while (state.running.load()) {
        auto cycle_begin = Clock::now();

        // 1. Pet the software watchdog (proves main loop is alive)
        swdt.pet();

        // 2. Health check at 1Hz (every 100th cycle)
        if (cycle % HEALTH_CHECK_DIVISOR == 0) {
            health.check();
            swdt.check(); // also check watchdog state

            // Log periodic status
            if (cycle % (HEALTH_CHECK_DIVISOR * 2) == 0) {
                g_log.log(state, "Main",
                          "Cycle " + std::to_string(cycle) +
                          " mode=" + mode_str(state.mode.load()) +
                          " sensor=" + std::to_string(sensor.beat_count.load()) +
                          " ctrl=" + std::to_string(controller.beat_count.load()) +
                          " telem=" + std::to_string(telemetry.beat_count.load()));
            }
        }

        // 3. Check if we should stop
        auto total_elapsed = std::chrono::duration_cast<Ms>(
            Clock::now() - loop_start);
        if (total_elapsed.count() > SIM_DURATION_S * 1000) {
            g_log.log(state, "Main", "Simulation duration reached — shutting down");
            break;
        }

        // If in SAFE MODE, we continue running for logging but would
        // normally stop actuators, set outputs to safe values etc.
        if (state.mode.load() == SystemMode::kSafeMode && cycle % 100 == 0) {
            g_log.log(state, "Main", "SAFE MODE active — actuators disabled");
        }

        cycle++;

        // 4. Sleep until next cycle
        auto cycle_end = Clock::now();
        auto cycle_duration = std::chrono::duration_cast<Us>(cycle_end - cycle_begin);
        auto sleep_time = Us(MAIN_LOOP_PERIOD_US) - cycle_duration;
        if (sleep_time.count() > 0) {
            std::this_thread::sleep_for(sleep_time);
        }
    }

    // Shutdown
    state.running.store(false);
    state.mode.store(SystemMode::kShutdown);
    g_log.log(state, "Main", "System shutdown initiated");

    t_sensor.join();
    t_controller.join();
    t_telemetry.join();

    g_log.log(state, "Main", "All workers joined — shutdown complete");

    // Print timeline
    g_log.print();

    // Summary
    std::cout << "\n=== SUMMARY ===\n";
    std::cout << "Total cycles:       " << cycle << "\n";
    std::cout << "Sensor heartbeats:  " << sensor.beat_count.load()
              << " (healthy=" << sensor.healthy.load() << ")\n";
    std::cout << "Controller beats:   " << controller.beat_count.load()
              << " (healthy=" << controller.healthy.load() << ")\n";
    std::cout << "Telemetry beats:    " << telemetry.beat_count.load()
              << " (healthy=" << telemetry.healthy.load() << ")\n";
    std::cout << "Final mode:         " << mode_str(state.mode.load()) << "\n";

    // Verify: sensor should have been detected as failed
    assert(!sensor.healthy.load() && "Sensor should be detected as failed");
    std::cout << "\n✓ Fault injection detected and handled correctly.\n";

    return 0;
}
