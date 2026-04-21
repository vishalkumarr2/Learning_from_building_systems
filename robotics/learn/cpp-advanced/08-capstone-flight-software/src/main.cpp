// ============================================================================
// Mini Flight Software — Capstone main.cpp
//
// Wires all components together: ModeManager, CyclicExecutive, SensorSim,
// PIDController, SPSCQueue, Watchdog, HealthMonitor, TraceBuffer.
//
// Usage:
//   ./flight_sw --duration 5 --fault-after 2 --stress
// ============================================================================

#include <flight_sw/mode_manager.hpp>
#include <flight_sw/cyclic_executive.hpp>
#include <flight_sw/sensor_sim.hpp>
#include <flight_sw/pid_controller.hpp>
#include <flight_sw/spsc_queue.hpp>
#include <flight_sw/watchdog.hpp>
#include <flight_sw/health_monitor.hpp>
#include <flight_sw/trace_buffer.hpp>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

#ifdef __linux__
#include <sched.h>
#include <sys/mman.h>
#endif

using namespace flight_sw;
using namespace std::chrono_literals;

// ── Telemetry record (pushed through SPSC queue) ─────────────────────
struct TelemetryRecord {
    uint64_t timestamp_ns{0};
    double   sensor_ax{0};
    double   pid_output{0};
    char     mode[16]{};
};

// ── Command-line config ──────────────────────────────────────────────
struct Config {
    int  duration_sec  = 5;
    int  fault_after   = -1;     // -1 = no fault injection
    bool stress        = false;
};

static Config parse_args(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--duration") == 0 && i + 1 < argc) {
            cfg.duration_sec = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--fault-after") == 0 && i + 1 < argc) {
            cfg.fault_after = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--stress") == 0) {
            cfg.stress = true;
        } else if (std::strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: flight_sw [--duration N] [--fault-after N] [--stress]\n"
                      << "  --duration N     Run for N seconds (default 5)\n"
                      << "  --fault-after N  Inject sensor STUCK fault after N seconds\n"
                      << "  --stress         Stress mode: tighter tolerances\n";
            std::exit(0);
        }
    }
    return cfg;
}

// ── Logger thread (drains SPSC queue) ────────────────────────────────
static void logger_thread_func(SPSCQueue<TelemetryRecord, 4096>& queue,
                               std::atomic<bool>& running) {
    uint64_t logged = 0;
    while (running.load(std::memory_order_relaxed) || !queue.empty()) {
        auto rec = queue.try_pop();
        if (rec.has_value()) {
            ++logged;
            // Print every 500th record to avoid flooding
            if (logged % 500 == 0) {
                std::cout << "  [TEL " << rec->timestamp_ns / 1'000'000 << "ms] "
                          << "mode=" << rec->mode
                          << " ax=" << std::fixed << std::setprecision(4) << rec->sensor_ax
                          << " pid=" << rec->pid_output << "\n";
            }
        } else {
            std::this_thread::sleep_for(1ms);
        }
    }
    std::cout << "  Logger: " << logged << " records processed.\n";
}

// ── Main ─────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    Config cfg = parse_args(argc, argv);

    std::cout << "=== Mini Flight Software — Capstone ===\n"
              << "  Duration:    " << cfg.duration_sec << "s\n"
              << "  Fault after: " << (cfg.fault_after >= 0
                                        ? std::to_string(cfg.fault_after) + "s"
                                        : "none") << "\n"
              << "  Stress mode: " << (cfg.stress ? "ON" : "OFF") << "\n\n";

    // ── RT setup ─────────────────────────────────────────────────────
#ifdef __linux__
    mlockall(MCL_CURRENT | MCL_FUTURE);

    // Try to pin to CPU 0
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
#endif

    // ── Create components ────────────────────────────────────────────
    ModeManager mode_mgr;
    TraceBuffer<1024> trace;
    HealthMonitor health;

    SensorSim sensor(1000.0, cfg.stress ? 0.001 : 0.01);
    sensor.set_signal(1.0, 2.0); // 1.0 amplitude, 2 Hz

    PIDConfig pid_cfg;
    pid_cfg.kp = 2.0;
    pid_cfg.ki = 0.5;
    pid_cfg.kd = 0.1;
    pid_cfg.dt = 0.002;  // 500 Hz
    pid_cfg.max_integral = 5.0;
    pid_cfg.output_min = -10.0;
    pid_cfg.output_max =  10.0;
    PIDController pid(pid_cfg);
    pid.set_setpoint(0.0);

    SPSCQueue<TelemetryRecord, 4096> telem_queue;

    // Shared state for tasks
    SensorReading last_reading{};
    double last_pid_output = 0.0;
    std::atomic<bool> logger_running{true};
    std::atomic<bool> fault_injected{false};

    // Fault injection time tracking
    auto start_time = std::chrono::steady_clock::now();

    // Watchdog: 100ms timeout → SafeStop
    Watchdog watchdog(100ms, [&mode_mgr, &trace]() {
        mode_mgr.transition(WatchdogTimeout{});
        trace.record("WATCHDOG", "EXPIRED → SafeStop");
    });

    // ── Register health subsystems ───────────────────────────────────
    health.register_subsystem("sensor",     1000, 10);
    health.register_subsystem("controller",  500,  5);
    health.register_subsystem("telemetry",   100,  5);

    // ── Boot sequence ────────────────────────────────────────────────
    trace.record("MODE", "Boot");
    mode_mgr.transition(InitComplete{});
    trace.record("MODE", "Boot → Nominal");
    std::cout << "  Mode: " << mode_mgr.current_state_name() << "\n";

    // ── Cyclic executive setup ───────────────────────────────────────
    CyclicExecutive exec(1ms);

    // 1 kHz — Sensor acquisition
    exec.register_task("sensor", 1000, [&]() {
        last_reading = sensor.read();
        health.heartbeat("sensor");

        // Fault injection check
        if (cfg.fault_after >= 0 && !fault_injected.load(std::memory_order_relaxed)) {
            auto elapsed = std::chrono::steady_clock::now() - start_time;
            if (elapsed >= std::chrono::seconds(cfg.fault_after)) {
                sensor.set_fault_mode(FaultMode::STUCK);
                fault_injected.store(true, std::memory_order_relaxed);
                trace.record("FAULT", "Sensor STUCK injected");
                std::cout << "  [FAULT] Sensor STUCK mode injected at "
                          << cfg.fault_after << "s\n";
            }
        }
    });

    // 500 Hz — PID controller
    exec.register_task("controller", 500, [&]() {
        if (last_reading.is_valid) {
            last_pid_output = pid.update(last_reading.values[0]);
        }
        health.heartbeat("controller");
    });

    // 100 Hz — Telemetry via SPSC queue
    exec.register_task("telemetry", 100, [&]() {
        TelemetryRecord rec;
        rec.timestamp_ns = last_reading.timestamp_ns;
        rec.sensor_ax    = last_reading.values[0];
        rec.pid_output   = last_pid_output;

        auto mode_name = mode_mgr.current_state_name();
        auto len = std::min(mode_name.size(), sizeof(rec.mode) - 1);
        std::memcpy(rec.mode, mode_name.data(), len);
        rec.mode[len] = '\0';

        (void)telem_queue.try_push(rec); // Non-blocking; drop if full
        health.heartbeat("telemetry");
    });

    // 10 Hz — Health check
    exec.register_task("health", 10, [&]() {
        auto failed = health.check();
        for (auto const& name : failed) {
            trace.record("HEALTH", std::string("FAIL: ") + name);
            if (name == "sensor") {
                bool changed = mode_mgr.transition(SensorFault{});
                if (changed) {
                    trace.record("MODE", std::string("→ ") +
                                 std::string(mode_mgr.current_state_name()));
                    std::cout << "  [HEALTH] Sensor fault → "
                              << mode_mgr.current_state_name() << "\n";
                }
            }
        }
    });

    // 1 Hz — Watchdog kick + state ID verification
    exec.register_task("heartbeat", 1, [&]() {
        watchdog.kick();
        if (!mode_mgr.verify_state_id()) {
            trace.record("SAFETY", "Hamming ID corruption detected!");
            mode_mgr.transition(ShutdownCmd{});
        }
    });

    // ── Start logger thread ──────────────────────────────────────────
    std::thread logger(logger_thread_func, std::ref(telem_queue),
                       std::ref(logger_running));

    // ── Arm watchdog and run ─────────────────────────────────────────
    watchdog.arm();
    std::cout << "\n  Starting cyclic executive for " << cfg.duration_sec << "s...\n\n";

    auto duration = std::chrono::seconds(cfg.duration_sec);
    exec.run(std::chrono::duration_cast<std::chrono::nanoseconds>(duration));

    // ── Shutdown ─────────────────────────────────────────────────────
    watchdog.disarm();
    mode_mgr.transition(ShutdownCmd{});
    trace.record("MODE", "Shutdown");

    logger_running.store(false, std::memory_order_relaxed);
    logger.join();

    // ── Final report ─────────────────────────────────────────────────
    std::cout << "\n=== Final Report ===\n";

    // Jitter stats
    exec.print_stats();

    // Mode transitions
    std::cout << "\n=== Mode Transitions ===\n";
    auto const& log = mode_mgr.transition_log();
    if (log.empty()) {
        std::cout << "  No transitions recorded.\n";
    } else {
        for (auto const& t : log) {
            std::cout << "  " << t.from << " → " << t.to
                      << " [" << t.event << "]\n";
        }
    }

    // Health events
    std::cout << "\n=== Health Events ===\n";
    health.print_events();

    // Trace dump (last 50)
    std::cout << "\n=== Trace Buffer ===\n";
    trace.dump(50);

    std::cout << "\n=== END ===\n";
    return 0;
}
