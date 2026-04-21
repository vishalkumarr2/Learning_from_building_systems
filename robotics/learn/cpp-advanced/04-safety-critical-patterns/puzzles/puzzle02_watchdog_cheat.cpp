// =============================================================================
// Puzzle 02: The Watchdog Cheat — Why Toyota's Watchdog Failed
// =============================================================================
// Toyota's unintended acceleration code had a watchdog timer. They thought
// it protected them. It didn't. Here's why:
//
// The watchdog was "petted" (fed/kicked) by a low-priority background task
// that ran independently of the throttle control task. The watchdog only
// proved the RTOS scheduler was alive — not that the throttle control
// was working correctly.
//
// When the stack overflowed and corrupted the throttle variable, the
// background task kept petting the watchdog. The watchdog never fired.
// The car kept accelerating. People died.
//
// This puzzle demonstrates:
//   Part 1: A cheatable watchdog (the Toyota pattern)
//   Part 2: A proper watchdog that verifies actual work was done
// =============================================================================

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::milliseconds;

// ======================== PART 1: THE CHEATABLE WATCHDOG ====================
// This is the Toyota anti-pattern: a background thread pets the watchdog
// regardless of whether the main task is actually running.

namespace cheatable {

class Watchdog {
public:
    Watchdog(int timeout_ms, std::function<void()> on_expire)
        : timeout_(Ms(timeout_ms))
        , on_expire_(std::move(on_expire))
        , last_pet_(Clock::now())
        , running_(true)
    {
        thread_ = std::thread([this] { monitor(); });
    }

    ~Watchdog() {
        running_ = false;
        if (thread_.joinable()) thread_.join();
    }

    void pet() {
        std::lock_guard<std::mutex> lock(mtx_);
        last_pet_ = Clock::now();
    }

    bool has_expired() const { return expired_.load(); }

private:
    void monitor() {
        while (running_.load()) {
            std::this_thread::sleep_for(Ms(50));
            Ms elapsed;
            {
                std::lock_guard<std::mutex> lock(mtx_);
                elapsed = std::chrono::duration_cast<Ms>(Clock::now() - last_pet_);
            }
            if (elapsed > timeout_) {
                expired_ = true;
                on_expire_();
                return;
            }
        }
    }

    Ms timeout_;
    std::function<void()> on_expire_;
    mutable std::mutex mtx_;
    Clock::time_point last_pet_;
    std::atomic<bool> running_{true};
    std::atomic<bool> expired_{false};
    std::thread thread_;
};

void demonstrate() {
    std::cout << "=== PART 1: Cheatable Watchdog (The Toyota Pattern) ===\n\n";

    std::atomic<bool> system_running{true};
    std::atomic<bool> main_task_alive{true};
    std::atomic<int> throttle_value{0};  // 0 = idle, 100 = full throttle

    Watchdog wdt(500, [&] {
        std::cout << "[WDT] EXPIRED! Emergency stop!\n";
        throttle_value = 0;
        system_running = false;
    });

    // THE CHEAT: A background task pets the watchdog unconditionally.
    // It knows nothing about whether the main task is working.
    std::thread background_petter([&] {
        while (system_running.load()) {
            wdt.pet();  // Just pet. Don't check anything. Toyota style.
            std::this_thread::sleep_for(Ms(100));
        }
    });

    // Main throttle control task
    std::thread throttle_task([&] {
        for (int i = 0; i < 20; ++i) {
            if (!system_running.load()) break;
            throttle_value = 50;  // normal operation
            std::this_thread::sleep_for(Ms(100));
        }
        // SIMULATE: main task hangs (stack overflow, deadlock, etc.)
        std::cout << "[Throttle] HANGING! (simulating stack corruption)\n";
        main_task_alive = false;
        throttle_value = 100;   // Worst case: stuck at full throttle!
        // This thread is now "dead" but throttle_value remains at 100
        while (system_running.load()) {
            std::this_thread::sleep_for(Ms(100));
        }
    });

    // Monitor what happens
    auto start = Clock::now();
    while (system_running.load()) {
        auto elapsed = std::chrono::duration_cast<Ms>(Clock::now() - start);
        if (elapsed > Ms(4000)) break;

        std::cout << "[T+" << elapsed.count() << "ms] "
                  << "throttle=" << throttle_value.load()
                  << " main_alive=" << main_task_alive.load()
                  << " wdt_expired=" << wdt.has_expired() << "\n";
        std::this_thread::sleep_for(Ms(500));
    }

    system_running = false;
    background_petter.join();
    throttle_task.join();

    // THE PROBLEM: The watchdog NEVER fired!
    // The background petter kept it alive even though throttle was stuck at 100.
    std::cout << "\n*** RESULT: Watchdog " << (wdt.has_expired() ? "FIRED" : "NEVER FIRED")
              << " ***\n";
    std::cout << "*** Throttle stuck at: " << throttle_value.load() << " ***\n";
    if (!wdt.has_expired()) {
        std::cout << "*** THE CAR KEPT ACCELERATING. The watchdog was useless. ***\n";
    }
    std::cout << "\n";
}

} // namespace cheatable

// ======================== PART 2: PROPER WATCHDOG ===========================
// The fix: the watchdog must verify that ACTUAL WORK was done, not just
// that someone called pet(). The monitored task must prove forward progress.

namespace proper {

class ProgressWatchdog {
public:
    // The monitored task must:
    // 1. Increment a work counter (proves it ran)
    // 2. Set a checkpoint value (proves it computed something)
    // The watchdog checks BOTH conditions.

    struct Evidence {
        std::atomic<uint64_t> work_counter{0};     // must increase each check
        std::atomic<int32_t> last_output{0};        // must be reasonable
        std::atomic<bool> task_reports_healthy{false}; // self-reported health
    };

    ProgressWatchdog(Evidence& evidence, int timeout_ms,
                     std::function<void()> on_expire)
        : evidence_(evidence)
        , timeout_(Ms(timeout_ms))
        , on_expire_(std::move(on_expire))
        , last_seen_counter_(0)
        , running_(true)
    {
        thread_ = std::thread([this] { monitor(); });
    }

    ~ProgressWatchdog() {
        running_ = false;
        if (thread_.joinable()) thread_.join();
    }

    bool has_expired() const { return expired_.load(); }

private:
    void monitor() {
        while (running_.load()) {
            std::this_thread::sleep_for(timeout_);
            if (!running_.load()) break;

            uint64_t current_counter = evidence_.work_counter.load();
            int32_t current_output = evidence_.last_output.load();
            bool self_healthy = evidence_.task_reports_healthy.load();

            // CHECK 1: Work counter must have advanced
            bool counter_advanced = (current_counter > last_seen_counter_);

            // CHECK 2: Output must be in a reasonable range
            bool output_reasonable = (current_output >= 0 && current_output <= 100);

            // CHECK 3: Task self-reports healthy
            bool all_ok = counter_advanced && output_reasonable && self_healthy;

            if (!all_ok) {
                std::cout << "[ProgressWDT] FAILURE DETECTED:\n";
                std::cout << "  counter_advanced=" << counter_advanced
                          << " (was " << last_seen_counter_
                          << ", now " << current_counter << ")\n";
                std::cout << "  output_reasonable=" << output_reasonable
                          << " (value=" << current_output << ")\n";
                std::cout << "  self_healthy=" << self_healthy << "\n";
                expired_ = true;
                on_expire_();
                return;
            }

            last_seen_counter_ = current_counter;
        }
    }

    Evidence& evidence_;
    Ms timeout_;
    std::function<void()> on_expire_;
    uint64_t last_seen_counter_;
    std::atomic<bool> running_{true};
    std::atomic<bool> expired_{false};
    std::thread thread_;
};

void demonstrate() {
    std::cout << "=== PART 2: Proper Watchdog (Verifies Progress) ===\n\n";

    std::atomic<bool> system_running{true};
    ProgressWatchdog::Evidence evidence;

    ProgressWatchdog wdt(evidence, 500, [&] {
        std::cout << "[ProgressWDT] EXPIRED! Emergency stop triggered!\n";
        system_running = false;
    });

    // Throttle control task — must prove it's working
    std::thread throttle_task([&] {
        for (int i = 0; i < 50; ++i) {
            if (!system_running.load()) break;

            // Normal operation: do real work and report it
            int throttle = 50; // computed throttle value
            evidence.work_counter.fetch_add(1);
            evidence.last_output.store(throttle);
            evidence.task_reports_healthy.store(true);

            std::this_thread::sleep_for(Ms(100));
        }

        // SIMULATE: task hangs
        std::cout << "[Throttle] HANGING! But this time watchdog will catch it.\n";
        evidence.task_reports_healthy.store(false);
        // Work counter stops incrementing — watchdog will notice!

        while (system_running.load()) {
            std::this_thread::sleep_for(Ms(100));
        }
    });

    // Note: NO background petter! The watchdog checks the evidence directly.

    auto start = Clock::now();
    while (system_running.load()) {
        auto elapsed = std::chrono::duration_cast<Ms>(Clock::now() - start);
        if (elapsed > Ms(8000)) break;

        std::cout << "[T+" << elapsed.count() << "ms] "
                  << "counter=" << evidence.work_counter.load()
                  << " output=" << evidence.last_output.load()
                  << " healthy=" << evidence.task_reports_healthy.load()
                  << " wdt_expired=" << wdt.has_expired() << "\n";
        std::this_thread::sleep_for(Ms(500));
    }

    system_running = false;
    throttle_task.join();

    std::cout << "\n*** RESULT: Watchdog " << (wdt.has_expired() ? "FIRED!" : "didn't fire")
              << " ***\n";
    if (wdt.has_expired()) {
        std::cout << "*** Emergency stop successful. No runaway throttle. ***\n";
    }
    std::cout << "\n";
}

} // namespace proper

// ======================== MAIN ===============================================

int main() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "  Puzzle 02: Why Toyota's Watchdog Failed\n";
    std::cout << std::string(70, '=') << "\n\n";

    cheatable::demonstrate();
    std::cout << std::string(70, '-') << "\n\n";
    proper::demonstrate();

    std::cout << std::string(70, '=') << "\n";
    std::cout << "LESSON:\n";
    std::cout << "  A watchdog must verify ACTUAL WORK DONE, not just a heartbeat.\n";
    std::cout << "  Toyota's watchdog only proved the scheduler ran.\n";
    std::cout << "  It never checked if the throttle task executed correctly.\n";
    std::cout << "  A background thread petting on behalf is DANGEROUS.\n";
    std::cout << std::string(70, '=') << "\n";

    return 0;
}
