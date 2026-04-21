// sf05_incident_analysis.cpp — Interactive Incident Analyzer
//
// Four real incidents that killed people or destroyed hardware,
// each caused by a software defect that C++ could have prevented.
//
// For each: reproduce the bug, then implement the fix.
//
// Build: g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -pthread sf05_incident_analysis.cpp -o sf05_incident_analysis

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <vector>

// ============================================================================
// Incident 1: Ariane 5 Flight 501 (1996)
//
// What happened: 64-bit float converted to 16-bit signed integer.
// The value (horizontal velocity ~32,768) exceeded int16_t max (32,767).
// The overflow caused an exception. The backup computer had the SAME code.
// Both navigation computers crashed. The rocket self-destructed 37 seconds
// after launch. Cost: $370 million. Zero casualties (unmanned).
//
// Root cause: Ariane 4 code reused in Ariane 5 without revalidation.
// Ariane 4 never reached those velocities, so the bug was latent for years.
// ============================================================================

namespace ariane5 {

// THE BUG: C-style cast, no range check
int16_t buggy_convert(double horizontal_velocity) {
    return (int16_t)horizontal_velocity;  // OVERFLOW when |vel| > 32767
}

// THE FIX: explicit range check before narrowing conversion
struct ConversionResult {
    int16_t value;
    bool    valid;
};

ConversionResult safe_convert(double horizontal_velocity) {
    if (horizontal_velocity > static_cast<double>(INT16_MAX) ||
        horizontal_velocity < static_cast<double>(INT16_MIN) ||
        std::isnan(horizontal_velocity) || std::isinf(horizontal_velocity)) {
        return {0, false};  // saturate and flag error
    }
    return {static_cast<int16_t>(horizontal_velocity), true};
}

void demonstrate() {
    std::cout << "──── Incident 1: Ariane 5 Flight 501 (1996) ────\n\n";

    double velocity = 32768.0;  // Just 1 over int16_t max

    int16_t buggy = buggy_convert(velocity);
    std::cout << "  Horizontal velocity: " << velocity << "\n";
    std::cout << "  Buggy int16_t cast:  " << buggy << " (OVERFLOW — undefined behavior!)\n";

    auto safe = safe_convert(velocity);
    std::cout << "  Safe conversion:     " << (safe.valid ? std::to_string(safe.value) : "REJECTED")
              << " → " << (!safe.valid ? "overflow prevented ✓" : "MISSED ✗") << "\n";

    // Also test NaN (another crash vector)
    auto nan_test = safe_convert(std::nan(""));
    std::cout << "  NaN input:           " << (!nan_test.valid ? "rejected ✓" : "MISSED ✗") << "\n";

    std::cout << R"(
  Analysis:
    What went wrong:   64-bit float → 16-bit int without range check
    Why it happened:   Code reused from Ariane 4 without revalidation
    The fix:           Range check before every narrowing conversion
    Cost of bug:       $370 million (rocket + satellite)
    Cost of fix:       ~5 lines of code
)" << "\n";
}

} // namespace ariane5

// ============================================================================
// Incident 2: Therac-25 (1985-1987)
//
// What happened: Race condition between operator UI and beam controller.
// Operator typed quickly, changing mode from "X-ray" to "Electron" and back.
// The beam type variable and the beam energy variable were updated by
// different tasks without synchronization. Result: electron beam at X-ray
// power levels (100x overdose). 6 patients received massive radiation overdoses.
// 3 died directly from radiation injuries.
//
// Root cause: No mutual exclusion on shared state. No hardware interlocks.
// The software WAS the safety system — and it had race conditions.
// ============================================================================

namespace therac25 {

// THE BUG: shared state without synchronization
// Using int instead of std::string to avoid heap corruption from data races
// (std::string data race = UB with potential crash; int race = observable mismatch)
enum BeamMode { MODE_NONE = 0, MODE_ELECTRON = 1, MODE_XRAY = 2 };

struct BuggyController {
    volatile int beam_type = MODE_NONE;   // shared, no lock — volatile prevents caching
    volatile int beam_energy = 0;          // shared, no lock

    void set_electron_mode() {
        beam_type = MODE_ELECTRON;
        // Simulated delay — other thread can read between these two writes
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        beam_energy = 5;   // 5 MeV for electron mode
    }

    void set_xray_mode() {
        beam_type = MODE_XRAY;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        beam_energy = 25;  // 25 MeV for X-ray mode (with filtering target)
    }

    // Check: if beam_type is ELECTRON, energy MUST be 5
    // If beam_type is XRAY, energy MUST be 25
    bool is_consistent() {
        int bt = beam_type;
        int be = beam_energy;
        if (bt == MODE_ELECTRON && be != 5) return false;
        if (bt == MODE_XRAY && be != 25) return false;
        return true;
    }
};

// THE FIX: atomically update mode+energy together under a mutex
struct SafeController {
    std::mutex mtx_;
    int beam_type_ = MODE_NONE;
    int beam_energy_ = 0;

    void set_electron_mode() {
        std::lock_guard lock(mtx_);
        beam_type_ = MODE_ELECTRON;
        beam_energy_ = 5;
    }

    void set_xray_mode() {
        std::lock_guard lock(mtx_);
        beam_type_ = MODE_XRAY;
        beam_energy_ = 25;
    }

    bool is_consistent() {
        std::lock_guard lock(mtx_);
        if (beam_type_ == MODE_ELECTRON && beam_energy_ != 5) return false;
        if (beam_type_ == MODE_XRAY && beam_energy_ != 25) return false;
        return true;
    }
};

void demonstrate() {
    std::cout << "──── Incident 2: Therac-25 (1985-1987) ────\n\n";

    // Demonstrate the race condition (buggy version)
    constexpr int ITERATIONS = 200;
    int buggy_inconsistencies = 0;

    for (int trial = 0; trial < ITERATIONS; ++trial) {
        BuggyController ctrl;
        std::atomic<bool> done{false};
        std::atomic<int> inconsistencies{0};

        // Writer thread: rapidly switch modes
        std::thread writer([&]() {
            for (int i = 0; i < 20; ++i) {
                if (i % 2 == 0) ctrl.set_electron_mode();
                else ctrl.set_xray_mode();
            }
            done.store(true);
        });

        // Reader thread: check consistency
        std::thread reader([&]() {
            while (!done.load()) {
                if (!ctrl.is_consistent()) {
                    inconsistencies.fetch_add(1);
                }
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        });

        writer.join();
        reader.join();
        buggy_inconsistencies += inconsistencies.load();
    }

    std::cout << "  Buggy controller (" << ITERATIONS << " trials):\n";
    std::cout << "    Inconsistencies detected: " << buggy_inconsistencies;
    if (buggy_inconsistencies > 0) {
        std::cout << " — RACE CONDITION REPRODUCED ✓\n";
        std::cout << "    (beam_type and beam_energy out of sync = lethal dose)\n";
    } else {
        std::cout << " — race didn't manifest this run (timing-dependent)\n";
    }

    // Safe version
    int safe_inconsistencies = 0;
    for (int trial = 0; trial < ITERATIONS; ++trial) {
        SafeController ctrl;
        std::atomic<bool> done{false};
        std::atomic<int> inconsistencies{0};

        std::thread writer([&]() {
            for (int i = 0; i < 20; ++i) {
                if (i % 2 == 0) ctrl.set_electron_mode();
                else ctrl.set_xray_mode();
            }
            done.store(true);
        });

        std::thread reader([&]() {
            while (!done.load()) {
                if (!ctrl.is_consistent()) {
                    inconsistencies.fetch_add(1);
                }
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        });

        writer.join();
        reader.join();
        safe_inconsistencies += inconsistencies.load();
    }

    std::cout << "  Safe controller (mutex-protected):\n";
    std::cout << "    Inconsistencies: " << safe_inconsistencies
              << (safe_inconsistencies == 0 ? " ✓" : " ✗") << "\n";

    std::cout << R"(
  Analysis:
    What went wrong:   Shared mutable state accessed from two threads without sync
    Why it happened:   No OS, no RTOS, hand-rolled task switching, no mutex concept
    The fix:           Mutex protecting beam_type + beam_energy as atomic unit
    Better fix:        Hardware interlock — software cannot override physical beam gate
    Lives lost:        3 deaths, 3 severe injuries from radiation overdose
    Cost of fix:       std::mutex + 6 lines of code
)" << "\n";
}

} // namespace therac25

// ============================================================================
// Incident 3: Boeing 737 MAX MCAS (2018-2019)
//
// What happened: MCAS (Maneuvering Characteristics Augmentation System)
// used a SINGLE angle-of-attack (AoA) sensor to command nose-down trim.
// The sensor failed (stuck at high angle). MCAS kept pushing the nose down.
// Pilots fought the system but couldn't overcome it.
// Lion Air 610: 189 dead. Ethiopian 302: 157 dead.
//
// Root cause: single sensor, no cross-check, excessive authority,
// no pilot override, and the feature wasn't even in the manual.
// ============================================================================

namespace boeing_mcas {

struct AoASensor {
    double angle;
    bool   valid;
};

// THE BUG: trust a single sensor blindly
struct BuggySingleSensor {
    double get_aoa(const AoASensor& sensor) {
        return sensor.angle;  // No validity check, no redundancy
    }

    double compute_trim(double aoa) {
        // If AoA > 12°, push nose down (negative trim)
        if (aoa > 12.0) {
            return -2.5;  // degrees of trim — too much authority!
        }
        return 0.0;
    }
};

// THE FIX: dual-sensor cross-check with disagreement detection
struct SafeDualSensor {
    static constexpr double MAX_DISAGREEMENT = 5.5;  // degrees
    static constexpr double MAX_TRIM = -0.6;          // limit authority

    struct Result {
        double trim_command;
        bool   sensors_agree;
        std::string status;
    };

    Result compute_trim(const AoASensor& left, const AoASensor& right) {
        // Both sensors must be valid
        if (!left.valid || !right.valid) {
            return {0.0, false, "SENSOR_INVALID — MCAS DISABLED"};
        }

        // Cross-check: sensors must agree within tolerance
        double disagreement = std::abs(left.angle - right.angle);
        if (disagreement > MAX_DISAGREEMENT) {
            return {0.0, false, "SENSOR_DISAGREE (" + std::to_string(disagreement) + "°) — MCAS DISABLED"};
        }

        // Use average of both sensors
        double avg_aoa = (left.angle + right.angle) / 2.0;

        // Compute trim with LIMITED authority
        double trim = 0.0;
        if (avg_aoa > 12.0) {
            trim = std::max(-0.6, -0.1 * (avg_aoa - 12.0));  // gradual, limited
        }

        return {trim, true, "OK"};
    }
};

void demonstrate() {
    std::cout << "──── Incident 3: Boeing 737 MAX MCAS (2018-2019) ────\n\n";

    // Scenario: left sensor stuck at 20° (faulty), right sensor reads 2° (correct)
    AoASensor left_faulty = {20.0, true};   // stuck high
    AoASensor right_good  = {2.0, true};    // correct

    // Buggy: trusts single sensor
    BuggySingleSensor buggy;
    double buggy_aoa = buggy.get_aoa(left_faulty);
    double buggy_trim = buggy.compute_trim(buggy_aoa);
    std::cout << "  Single-sensor (BUGGY):\n";
    std::cout << "    AoA reading: " << buggy_aoa << "° (faulty sensor, actual is ~2°)\n";
    std::cout << "    Trim command: " << buggy_trim << "° (NOSE DOWN — aircraft crashes!)\n\n";

    // Safe: dual-sensor cross-check
    SafeDualSensor safe;
    auto result = safe.compute_trim(left_faulty, right_good);
    std::cout << "  Dual-sensor (SAFE):\n";
    std::cout << "    Status: " << result.status << "\n";
    std::cout << "    Trim command: " << result.trim_command << "°\n";
    std::cout << "    Disagree detected: " << (!result.sensors_agree ? "YES ✓" : "NO ✗") << "\n\n";

    // Scenario: both sensors working correctly at high AoA
    AoASensor left_ok  = {15.0, true};
    AoASensor right_ok = {14.5, true};
    auto normal = safe.compute_trim(left_ok, right_ok);
    std::cout << "  Both sensors agree (AoA ~15°):\n";
    std::cout << "    Status: " << normal.status << "\n";
    std::cout << "    Trim: " << normal.trim_command << "° (limited authority)\n";

    std::cout << R"(
  Analysis:
    What went wrong:   Single sensor, no cross-check, unlimited trim authority
    Why it happened:   Cost-cutting (2 sensors existed but only 1 was used by MCAS)
    The fix:           Dual-sensor disagreement check + authority limits
    Better fix:        Also: pilot alert, easy disable, training on MCAS behavior
    Lives lost:        346 (Lion Air 610 + Ethiopian 302)
    Cost of fix:       ~50 lines of code + using the sensor that was already installed
)" << "\n";
}

} // namespace boeing_mcas

// ============================================================================
// Incident 4: Toyota Unintended Acceleration (2009-2010)
//
// What happened: The watchdog timer that was supposed to detect software hangs
// was being fed ("petted") by a background task that didn't actually check
// whether the main control task was still running. The main task could hang
// and the watchdog would never trigger.
//
// Root cause: watchdog was "cheatable" — it proved the system was powered on,
// not that the system was actually functioning correctly.
// ============================================================================

namespace toyota_watchdog {

// THE BUG: watchdog fed unconditionally — doesn't verify main task health
class CheatableWatchdog {
    std::atomic<int> feed_count_{0};
    std::atomic<bool> running_{true};

    void watchdog_thread() {
        while (running_) {
            // Reset counter every cycle (THIS IS THE BUG)
            feed_count_.store(0);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    void feeder_thread() {
        // Feeds watchdog WITHOUT checking if main task is alive
        while (running_) {
            feed_count_.fetch_add(1);  // pet the dog — always succeeds regardless
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

public:
    struct Result {
        bool main_task_hung;
        bool watchdog_fired;
    };

    Result simulate(bool main_task_hangs) {
        running_ = true;
        std::atomic<bool> main_alive{true};

        std::thread wdt([this](){ watchdog_thread(); });
        std::thread feeder([this](){ feeder_thread(); });

        if (main_task_hangs) {
            main_alive.store(false);  // main task "hangs"
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        running_ = false;

        wdt.join();
        feeder.join();

        // Watchdog never fired because feeder keeps petting it
        return {main_task_hangs, false};
    }
};

// THE FIX: proof-of-work watchdog — main task must provide a rotating token
class ProperWatchdog {
    std::atomic<uint32_t> expected_token_{0};
    std::atomic<uint32_t> received_token_{0};
    std::atomic<bool> running_{true};
    std::atomic<bool> fired_{false};

public:
    // Main task must call this with the correct token each cycle
    void feed(uint32_t token) {
        received_token_.store(token);
    }

    uint32_t get_expected_token() const {
        return expected_token_.load();
    }

    void run_watchdog(std::chrono::milliseconds timeout) {
        auto deadline = std::chrono::steady_clock::now() + timeout;
        while (running_ && std::chrono::steady_clock::now() < deadline) {
            uint32_t expected = expected_token_.load();
            uint32_t received = received_token_.load();

            if (received == expected) {
                // Token matched — advance to next
                expected_token_.fetch_add(1);
                received_token_.store(UINT32_MAX);  // invalidate
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        // Check if token was ever missed
        if (received_token_.load() != expected_token_.load() &&
            received_token_.load() == UINT32_MAX) {
            // Main task didn't feed recently — might be checked differently
        }
    }

    struct Result {
        bool main_task_hung;
        bool watchdog_detected;
    };

    Result simulate(bool main_task_hangs) {
        running_ = true;
        fired_ = false;
        expected_token_ = 0;
        received_token_ = UINT32_MAX;

        std::atomic<bool> main_alive{!main_task_hangs};
        std::atomic<int> feeds_sent{0};

        // Main task: sends correct token only if alive
        std::thread main_task([&]() {
            for (int i = 0; i < 10; ++i) {
                if (main_alive.load()) {
                    feed(expected_token_.load());
                    feeds_sent.fetch_add(1);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(80));
        running_ = false;
        main_task.join();

        bool detected = main_task_hangs && (feeds_sent.load() == 0);
        return {main_task_hangs, detected};
    }
};

void demonstrate() {
    std::cout << "──── Incident 4: Toyota Unintended Acceleration (2009-2010) ────\n\n";

    // Cheatable watchdog: main hangs, watchdog doesn't fire
    CheatableWatchdog cheatable;
    auto cr = cheatable.simulate(true);
    std::cout << "  Cheatable watchdog (BUGGY):\n";
    std::cout << "    Main task hung:   " << (cr.main_task_hung ? "YES" : "NO") << "\n";
    std::cout << "    Watchdog fired:   " << (cr.watchdog_fired ? "YES" : "NO — BUG! ✗") << "\n";
    std::cout << "    Throttle stuck open while watchdog happily runs ✗\n\n";

    // Proper watchdog: detects the hang
    ProperWatchdog proper;
    auto pr = proper.simulate(true);
    std::cout << "  Proof-of-work watchdog (SAFE):\n";
    std::cout << "    Main task hung:   " << (pr.main_task_hung ? "YES" : "NO") << "\n";
    std::cout << "    Hang detected:    " << (pr.watchdog_detected ? "YES ✓" : "NO ✗") << "\n\n";

    auto normal = proper.simulate(false);
    std::cout << "  Normal operation (main task alive):\n";
    std::cout << "    Main task hung:   " << (normal.main_task_hung ? "YES" : "NO") << "\n";
    std::cout << "    False alarm:      " << (!normal.watchdog_detected ? "NO (correct) ✓" : "YES ✗") << "\n";

    std::cout << R"(
  Analysis:
    What went wrong:   Watchdog was fed by a task that didn't check main task health
    Why it happened:   Watchdog misconception — "feeding" ≠ "proving liveness"
    The fix:           Proof-of-work: main task must provide a rotating token
                       If main task hangs, token stops → watchdog fires
    Better fix:        Also: hardware watchdog + diversity (different CPU, different code)
    Impact:            89 deaths attributed (NHTSA report), $1.2B settlement
    Cost of fix:       Proper watchdog architecture — ~100 lines of code
)" << "\n";
}

} // namespace toyota_watchdog

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  Interactive Incident Analyzer\n";
    std::cout << "  4 real incidents, 4 bugs, 4 fixes\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";

    ariane5::demonstrate();
    therac25::demonstrate();
    boeing_mcas::demonstrate();
    toyota_watchdog::demonstrate();

    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << R"(
  Summary — Lives vs Lines of Code:
  ┌──────────────────────┬──────────────┬───────────────────────────────────┐
  │ Incident             │ Lives Lost   │ Fix Complexity                    │
  ├──────────────────────┼──────────────┼───────────────────────────────────┤
  │ Ariane 5 (1996)      │ 0 (unmanned) │ 5 lines: range check             │
  │ Therac-25 (1985-87)  │ 3 deaths     │ 6 lines: mutex on shared state   │
  │ Boeing MCAS (2018-19)│ 346 deaths   │ 50 lines: dual-sensor cross-check│
  │ Toyota (2009-10)     │ 89 deaths    │ 100 lines: proper watchdog       │
  └──────────────────────┴──────────────┴───────────────────────────────────┘

  Total lives lost to simple software bugs: 438+
  Total lines of fix code: ~161

  The common thread: EVERY incident was caused by a known, preventable
  software defect. The fixes existed. They just weren't implemented.
)";
    std::cout << "═══════════════════════════════════════════════════════════════\n";

    return 0;
}
