// =============================================================================
// Exercise 04: Watchdog + Health Monitor System
// =============================================================================
// A complete watchdog and health monitoring system:
// - Watchdog class with configurable timeout and expiry callback
// - HealthMonitor checking N subsystems for heartbeat deadlines
// - 3 subsystems at different rates, one hangs after 2 seconds
// - Timeline printing: heartbeats, deadlines, failure, shutdown
// =============================================================================

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using Duration = std::chrono::milliseconds;

// ======================== TIMELINE LOGGER ====================================

class Timeline {
public:
    void log(const std::string& subsystem, const std::string& event) {
        std::lock_guard<std::mutex> lock(mtx_);
        auto now = Clock::now();
        if (!started_) {
            start_ = now;
            started_ = true;
        }
        auto elapsed = std::chrono::duration_cast<Duration>(now - start_);
        entries_.push_back({elapsed, subsystem, event});
    }

    void print() const {
        std::lock_guard<std::mutex> lock(mtx_);
        std::cout << "\n=== EVENT TIMELINE ===\n";
        std::cout << std::setw(8) << "Time(ms)" << " | "
                  << std::setw(14) << "Subsystem" << " | " << "Event\n";
        std::cout << std::string(60, '-') << "\n";
        for (const auto& e : entries_) {
            std::cout << std::setw(8) << e.time.count() << " | "
                      << std::setw(14) << e.subsystem << " | "
                      << e.event << "\n";
        }
        std::cout << std::string(60, '=') << "\n";
    }

private:
    struct Entry {
        Duration time;
        std::string subsystem;
        std::string event;
    };

    mutable std::mutex mtx_;
    TimePoint start_{};
    bool started_ = false;
    std::vector<Entry> entries_;
};

// Global timeline for this exercise
Timeline g_timeline;

// ======================== WATCHDOG ===========================================

class Watchdog {
public:
    using Callback = std::function<void(const std::string&)>;

    Watchdog(const std::string& name, Duration timeout, Callback on_expiry)
        : name_(name)
        , timeout_(timeout)
        , on_expiry_(std::move(on_expiry))
        , running_(false)
    {
        assert(timeout.count() > 0);
        assert(on_expiry_);
    }

    ~Watchdog() {
        stop();
    }

    Watchdog(const Watchdog&) = delete;
    Watchdog& operator=(const Watchdog&) = delete;

    void start() {
        assert(!running_.load());
        running_ = true;
        last_pet_ = Clock::now();
        monitor_thread_ = std::thread([this] { monitor_loop(); });
        g_timeline.log(name_, "Watchdog STARTED (timeout=" +
                       std::to_string(timeout_.count()) + "ms)");
    }

    void stop() {
        running_ = false;
        if (monitor_thread_.joinable()) {
            monitor_thread_.join();
        }
    }

    void pet() {
        std::lock_guard<std::mutex> lock(pet_mtx_);
        last_pet_ = Clock::now();
    }

private:
    TimePoint get_last_pet() {
        std::lock_guard<std::mutex> lock(pet_mtx_);
        return last_pet_;
    }

    void monitor_loop() {
        while (running_.load()) {
            std::this_thread::sleep_for(Duration(50)); // check every 50ms
            auto elapsed = std::chrono::duration_cast<Duration>(
                Clock::now() - get_last_pet());

            if (elapsed > timeout_ && running_.load()) {
                g_timeline.log(name_, "WATCHDOG EXPIRED! (" +
                               std::to_string(elapsed.count()) + "ms since last pet)");
                on_expiry_(name_);
                running_ = false;
                return;
            }
        }
    }

    std::string name_;
    Duration timeout_;
    Callback on_expiry_;
    std::atomic<bool> running_;
    std::mutex pet_mtx_;
    TimePoint last_pet_;
    std::thread monitor_thread_;
};

// ======================== SUBSYSTEM ==========================================

enum class SubsystemStatus {
    kHealthy,
    kWarning,   // missed 1 deadline
    kFailed,    // missed 2+ deadlines
    kShutdown
};

const char* status_to_string(SubsystemStatus s) {
    switch (s) {
        case SubsystemStatus::kHealthy:  return "HEALTHY";
        case SubsystemStatus::kWarning:  return "WARNING";
        case SubsystemStatus::kFailed:   return "FAILED";
        case SubsystemStatus::kShutdown: return "SHUTDOWN";
    }
    return "UNKNOWN";
}

struct Subsystem {
    std::string name;
    Duration heartbeat_interval;       // expected interval
    Duration deadline;                 // max allowed time between heartbeats
    std::atomic<uint64_t> heartbeat_count{0};
    TimePoint last_heartbeat{Clock::now()};
    std::mutex hb_mtx;
    SubsystemStatus status{SubsystemStatus::kHealthy};
    int missed_deadlines{0};

    Subsystem(std::string n, Duration interval, Duration dl)
        : name(std::move(n)), heartbeat_interval(interval), deadline(dl) {}

    void beat() {
        heartbeat_count.fetch_add(1);
        std::lock_guard<std::mutex> lock(hb_mtx);
        last_heartbeat = Clock::now();
    }

    TimePoint get_last_heartbeat() {
        std::lock_guard<std::mutex> lock(hb_mtx);
        return last_heartbeat;
    }
};

// ======================== HEALTH MONITOR ====================================

class HealthMonitor {
public:
    explicit HealthMonitor(Duration check_interval)
        : check_interval_(check_interval)
        , running_(false)
        , system_healthy_(true)
    {}

    ~HealthMonitor() {
        stop();
    }

    HealthMonitor(const HealthMonitor&) = delete;
    HealthMonitor& operator=(const HealthMonitor&) = delete;

    void add_subsystem(Subsystem* sub) {
        assert(sub != nullptr);
        subsystems_.push_back(sub);
    }

    void start() {
        running_ = true;
        thread_ = std::thread([this] { check_loop(); });
        g_timeline.log("HealthMonitor", "STARTED (interval=" +
                       std::to_string(check_interval_.count()) + "ms)");
    }

    void stop() {
        running_ = false;
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    bool is_system_healthy() const { return system_healthy_.load(); }

private:
    void check_loop() {
        while (running_.load()) {
            std::this_thread::sleep_for(check_interval_);
            if (!running_.load()) break;
            check_all();
        }
    }

    void check_all() {
        int failed_count = 0;
        for (auto* sub : subsystems_) {
            if (sub->status == SubsystemStatus::kShutdown) {
                ++failed_count;
                continue;
            }

            auto elapsed = std::chrono::duration_cast<Duration>(
                Clock::now() - sub->get_last_heartbeat());

            if (elapsed > sub->deadline) {
                sub->missed_deadlines++;
                if (sub->missed_deadlines == 1) {
                    sub->status = SubsystemStatus::kWarning;
                    g_timeline.log("HealthMonitor",
                                   sub->name + " missed deadline (" +
                                   std::to_string(elapsed.count()) + "ms) — WARNING");
                } else {
                    sub->status = SubsystemStatus::kFailed;
                    g_timeline.log("HealthMonitor",
                                   sub->name + " FAILED (missed " +
                                   std::to_string(sub->missed_deadlines) + " deadlines)");
                    ++failed_count;
                }
            } else {
                if (sub->missed_deadlines > 0 &&
                    sub->status != SubsystemStatus::kFailed) {
                    sub->missed_deadlines = 0;
                    sub->status = SubsystemStatus::kHealthy;
                    g_timeline.log("HealthMonitor", sub->name + " recovered — HEALTHY");
                }
                g_timeline.log("HealthMonitor",
                               sub->name + " OK (beats=" +
                               std::to_string(sub->heartbeat_count.load()) + ")");
            }
        }

        if (failed_count > 0) {
            g_timeline.log("HealthMonitor",
                           "SYSTEM DEGRADED (" + std::to_string(failed_count) +
                           "/" + std::to_string(subsystems_.size()) + " failed)");
            if (failed_count >= 2) {
                g_timeline.log("HealthMonitor", "SYSTEM SHUTDOWN — too many failures");
                system_healthy_ = false;
                // Mark all as shutdown
                for (auto* sub : subsystems_) {
                    sub->status = SubsystemStatus::kShutdown;
                }
                running_ = false;
            }
        }
    }

    Duration check_interval_;
    std::atomic<bool> running_;
    std::atomic<bool> system_healthy_;
    std::vector<Subsystem*> subsystems_;
    std::thread thread_;
};

// ======================== SIMULATION ========================================

void subsystem_worker(Subsystem& sub, std::atomic<bool>& running,
                      Duration hang_after = Duration(0)) {
    auto start = Clock::now();
    while (running.load()) {
        auto elapsed = std::chrono::duration_cast<Duration>(Clock::now() - start);

        // Simulate hang after specified time
        if (hang_after.count() > 0 && elapsed > hang_after) {
            g_timeline.log(sub.name, "*** HANGING (simulated fault) ***");
            // Just stop sending heartbeats — simulates a freeze
            while (running.load()) {
                std::this_thread::sleep_for(Duration(100));
            }
            return;
        }

        sub.beat();
        std::this_thread::sleep_for(sub.heartbeat_interval);
    }
}

int main() {
    std::cout << "=== Watchdog + Health Monitor Simulation ===\n";
    std::cout << "Simulating 3 subsystems, Sensor will hang after 2s...\n\n";

    // Create subsystems
    Subsystem sensor{"Sensor",     Duration(100), Duration(250)};  // 10Hz, 250ms deadline
    Subsystem controller{"Controller", Duration(50),  Duration(150)};  // 20Hz, 150ms deadline
    Subsystem telemetry{"Telemetry",  Duration(200), Duration(500)};  // 5Hz, 500ms deadline

    // Health monitor checks every 500ms
    HealthMonitor monitor(Duration(500));
    monitor.add_subsystem(&sensor);
    monitor.add_subsystem(&controller);
    monitor.add_subsystem(&telemetry);

    // System watchdog — must be pet by main loop
    std::atomic<bool> system_running{true};
    Watchdog sys_watchdog("SystemWDT", Duration(5000), [&](const std::string&) {
        g_timeline.log("SystemWDT", "FATAL: System watchdog expired — EMERGENCY STOP");
        system_running = false;
    });

    // Start everything
    monitor.start();
    sys_watchdog.start();

    std::atomic<bool> workers_running{true};

    // Launch subsystem workers
    // Sensor hangs after 2000ms
    std::thread t_sensor(subsystem_worker, std::ref(sensor),
                         std::ref(workers_running), Duration(2000));
    std::thread t_controller(subsystem_worker, std::ref(controller),
                             std::ref(workers_running), Duration(0));
    std::thread t_telemetry(subsystem_worker, std::ref(telemetry),
                            std::ref(workers_running), Duration(0));

    g_timeline.log("Main", "All subsystems started");

    // Main loop — pet the system watchdog and monitor health
    auto sim_start = Clock::now();
    constexpr auto SIM_DURATION = Duration(5000);

    while (system_running.load()) {
        auto elapsed = std::chrono::duration_cast<Duration>(Clock::now() - sim_start);
        if (elapsed > SIM_DURATION) break;

        sys_watchdog.pet();

        if (!monitor.is_system_healthy()) {
            g_timeline.log("Main", "Health monitor reports unhealthy — shutting down");
            break;
        }

        std::this_thread::sleep_for(Duration(100));
    }

    // Shutdown
    g_timeline.log("Main", "Initiating shutdown sequence");
    workers_running = false;
    system_running = false;
    sys_watchdog.stop();
    monitor.stop();

    t_sensor.join();
    t_controller.join();
    t_telemetry.join();

    g_timeline.log("Main", "Shutdown complete");

    // Print the timeline
    g_timeline.print();

    // Summary
    std::cout << "\nFinal subsystem states:\n";
    std::cout << "  Sensor:     " << status_to_string(sensor.status)
              << " (beats=" << sensor.heartbeat_count << ")\n";
    std::cout << "  Controller: " << status_to_string(controller.status)
              << " (beats=" << controller.heartbeat_count << ")\n";
    std::cout << "  Telemetry:  " << status_to_string(telemetry.status)
              << " (beats=" << telemetry.heartbeat_count << ")\n";

    return 0;
}
