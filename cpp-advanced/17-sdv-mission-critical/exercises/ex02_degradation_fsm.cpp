// =============================================================================
// Exercise 02: Graceful Degradation State Machine
// =============================================================================
// Implement a degradation manager for an autonomous vehicle that transitions
// through degradation levels based on sensor health.
//
// Requirements:
//   1. Degradation can only go DOWN (more restrictive), never up without reset
//   2. Specific sensor failures map to specific degradation levels
//   3. Multiple simultaneous failures use the most restrictive level
//   4. All transitions are logged with reason
//   5. Safe stop is reached if ANY two "critical" sensors fail
//
// Production relevance:
//   - ISO 26262 requires documented degradation strategy
//   - L3+ autonomy mandates fail-operational behavior
//   - Fleet management needs real-time degradation status
// =============================================================================

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

// ============================================================================
// Degradation Levels (from best to worst)
// ============================================================================
enum class DegLevel : uint8_t {
    FULL_AUTONOMY    = 0,  // All sensors nominal, L4
    LIMITED_AUTONOMY = 1,  // Reduced capability, lower speed
    DRIVER_ASSIST    = 2,  // L2: driver must monitor
    MANUAL_ONLY      = 3,  // All autonomy off
    MINIMAL_RISK     = 4,  // Pull over and stop
    SAFE_STOP        = 5   // Emergency stop
};

constexpr const char* to_string(DegLevel level) {
    switch (level) {
        case DegLevel::FULL_AUTONOMY:    return "FULL_AUTONOMY";
        case DegLevel::LIMITED_AUTONOMY: return "LIMITED_AUTONOMY";
        case DegLevel::DRIVER_ASSIST:    return "DRIVER_ASSIST";
        case DegLevel::MANUAL_ONLY:      return "MANUAL_ONLY";
        case DegLevel::MINIMAL_RISK:     return "MINIMAL_RISK";
        case DegLevel::SAFE_STOP:        return "SAFE_STOP";
    }
    return "UNKNOWN";
}

// Higher numeric value = more degraded (worse)
constexpr bool is_more_degraded(DegLevel a, DegLevel b) {
    return static_cast<uint8_t>(a) > static_cast<uint8_t>(b);
}

// ============================================================================
// Sensor Identifiers
// ============================================================================
enum class Sensor : uint8_t {
    LIDAR_FRONT,
    LIDAR_REAR,
    RADAR_FRONT,
    RADAR_REAR,
    CAMERA_FRONT,
    CAMERA_SURROUND,
    IMU,
    GNSS,
    WHEEL_ENCODERS,
    STEERING_ANGLE
};

constexpr const char* to_string(Sensor s) {
    switch (s) {
        case Sensor::LIDAR_FRONT:     return "LIDAR_FRONT";
        case Sensor::LIDAR_REAR:      return "LIDAR_REAR";
        case Sensor::RADAR_FRONT:     return "RADAR_FRONT";
        case Sensor::RADAR_REAR:      return "RADAR_REAR";
        case Sensor::CAMERA_FRONT:    return "CAMERA_FRONT";
        case Sensor::CAMERA_SURROUND: return "CAMERA_SURROUND";
        case Sensor::IMU:             return "IMU";
        case Sensor::GNSS:            return "GNSS";
        case Sensor::WHEEL_ENCODERS:  return "WHEEL_ENCODERS";
        case Sensor::STEERING_ANGLE:  return "STEERING_ANGLE";
    }
    return "UNKNOWN";
}

// Critical sensors: failure of any two → SAFE_STOP
constexpr bool is_critical(Sensor s) {
    return s == Sensor::IMU ||
           s == Sensor::WHEEL_ENCODERS ||
           s == Sensor::STEERING_ANGLE;
}

// ============================================================================
// Transition Log Entry
// ============================================================================
struct TransitionEntry {
    DegLevel from;
    DegLevel to;
    std::string reason;
};

// ============================================================================
// Degradation Manager
// ============================================================================
class DegradationManager {
    DegLevel current_ = DegLevel::FULL_AUTONOMY;
    std::vector<Sensor> failed_sensors_;
    std::vector<TransitionEntry> log_;

    // Single-sensor degradation mapping
    DegLevel degradation_for(Sensor s) const {
        switch (s) {
            case Sensor::LIDAR_FRONT:
                // Can fall back to radar for forward perception
                return DegLevel::LIMITED_AUTONOMY;
            case Sensor::RADAR_FRONT:
                // Can fall back to lidar for forward perception
                return DegLevel::LIMITED_AUTONOMY;
            case Sensor::LIDAR_REAR:
            case Sensor::RADAR_REAR:
                // Rear perception lost — limit reverse, lower speed
                return DegLevel::LIMITED_AUTONOMY;
            case Sensor::CAMERA_FRONT:
                // Can't read signs/lights — driver must take over
                return DegLevel::DRIVER_ASSIST;
            case Sensor::CAMERA_SURROUND:
                // Parking/lane change impaired
                return DegLevel::LIMITED_AUTONOMY;
            case Sensor::IMU:
                // Localization severely impaired — stop safely
                return DegLevel::MINIMAL_RISK;
            case Sensor::GNSS:
                // Can still localize with IMU + lidar, but degraded
                return DegLevel::LIMITED_AUTONOMY;
            case Sensor::WHEEL_ENCODERS:
                // Odometry lost — significant impact
                return DegLevel::DRIVER_ASSIST;
            case Sensor::STEERING_ANGLE:
                // Can't close steering control loop
                return DegLevel::MINIMAL_RISK;
        }
        return DegLevel::SAFE_STOP;
    }

    // Check for combined failure scenarios
    DegLevel evaluate_combined() const {
        DegLevel worst = DegLevel::FULL_AUTONOMY;

        // Rule 1: If BOTH forward perception sensors fail → SAFE_STOP
        bool lidar_front_failed = is_failed(Sensor::LIDAR_FRONT);
        bool radar_front_failed = is_failed(Sensor::RADAR_FRONT);
        if (lidar_front_failed && radar_front_failed) {
            return DegLevel::SAFE_STOP;
        }

        // Rule 2: Any two critical sensors fail → SAFE_STOP
        int critical_failures = 0;
        for (auto s : failed_sensors_) {
            if (is_critical(s)) {
                ++critical_failures;
            }
        }
        if (critical_failures >= 2) {
            return DegLevel::SAFE_STOP;
        }

        // Rule 3: Apply individual sensor degradation, take worst
        for (auto s : failed_sensors_) {
            DegLevel level = degradation_for(s);
            if (is_more_degraded(level, worst)) {
                worst = level;
            }
        }

        return worst;
    }

    bool is_failed(Sensor s) const {
        return std::find(failed_sensors_.begin(), failed_sensors_.end(), s)
               != failed_sensors_.end();
    }

    void transition_to(DegLevel new_level, std::string reason) {
        if (new_level == current_) return;

        // Key safety rule: can only degrade further, never improve
        if (!is_more_degraded(new_level, current_)) {
            // Silently ignore upgrade attempts — production would log warning
            return;
        }

        log_.push_back({current_, new_level, std::move(reason)});
        current_ = new_level;
    }

public:
    // Report a sensor failure
    void report_failure(Sensor sensor) {
        if (is_failed(sensor)) return;  // already known

        failed_sensors_.push_back(sensor);
        DegLevel required = evaluate_combined();

        std::string reason = std::string(to_string(sensor)) + " failed";
        if (failed_sensors_.size() > 1) {
            reason += " (combined: " +
                      std::to_string(failed_sensors_.size()) +
                      " failures)";
        }

        transition_to(required, reason);
    }

    // Reset to full autonomy (e.g., after sensor replacement + self-test)
    void reset() {
        failed_sensors_.clear();
        log_.push_back({current_, DegLevel::FULL_AUTONOMY,
                         "System reset after maintenance"});
        current_ = DegLevel::FULL_AUTONOMY;
    }

    DegLevel current_level() const { return current_; }
    std::vector<TransitionEntry> const& transition_log() const {
        return log_;
    }
    size_t failed_sensor_count() const { return failed_sensors_.size(); }
};

// ============================================================================
// Self-Test
// ============================================================================
void test_initial_state() {
    std::cout << "--- Test: Initial state ---\n";
    DegradationManager mgr;
    assert(mgr.current_level() == DegLevel::FULL_AUTONOMY);
    assert(mgr.failed_sensor_count() == 0);
    std::cout << "  PASS\n";
}

void test_single_non_critical_failure() {
    std::cout << "--- Test: Single non-critical failure (GNSS) ---\n";
    DegradationManager mgr;
    mgr.report_failure(Sensor::GNSS);
    assert(mgr.current_level() == DegLevel::LIMITED_AUTONOMY);
    assert(mgr.transition_log().size() == 1);
    std::cout << "  PASS: " << to_string(mgr.current_level()) << "\n";
}

void test_single_critical_failure() {
    std::cout << "--- Test: Single critical failure (IMU) ---\n";
    DegradationManager mgr;
    mgr.report_failure(Sensor::IMU);
    assert(mgr.current_level() == DegLevel::MINIMAL_RISK);
    std::cout << "  PASS: " << to_string(mgr.current_level()) << "\n";
}

void test_forward_perception_total_loss() {
    std::cout << "--- Test: Forward perception total loss ---\n";
    DegradationManager mgr;
    mgr.report_failure(Sensor::LIDAR_FRONT);
    assert(mgr.current_level() == DegLevel::LIMITED_AUTONOMY);

    mgr.report_failure(Sensor::RADAR_FRONT);
    assert(mgr.current_level() == DegLevel::SAFE_STOP);
    std::cout << "  PASS: Both forward sensors lost → SAFE_STOP\n";
}

void test_two_critical_sensors() {
    std::cout << "--- Test: Two critical sensors fail ---\n";
    DegradationManager mgr;
    mgr.report_failure(Sensor::IMU);
    assert(mgr.current_level() == DegLevel::MINIMAL_RISK);

    mgr.report_failure(Sensor::WHEEL_ENCODERS);
    assert(mgr.current_level() == DegLevel::SAFE_STOP);
    std::cout << "  PASS: Two critical failures → SAFE_STOP\n";
}

void test_no_upgrade_without_reset() {
    std::cout << "--- Test: Cannot upgrade without reset ---\n";
    DegradationManager mgr;
    mgr.report_failure(Sensor::CAMERA_FRONT);
    assert(mgr.current_level() == DegLevel::DRIVER_ASSIST);

    // Reporting a less severe failure should NOT improve the level
    mgr.report_failure(Sensor::GNSS);
    assert(mgr.current_level() == DegLevel::DRIVER_ASSIST);
    std::cout << "  PASS: Level stays at DRIVER_ASSIST (no upgrade)\n";
}

void test_reset_restores_full() {
    std::cout << "--- Test: Reset restores FULL_AUTONOMY ---\n";
    DegradationManager mgr;
    mgr.report_failure(Sensor::IMU);
    assert(mgr.current_level() == DegLevel::MINIMAL_RISK);

    mgr.reset();
    assert(mgr.current_level() == DegLevel::FULL_AUTONOMY);
    assert(mgr.failed_sensor_count() == 0);
    std::cout << "  PASS\n";
}

void test_duplicate_failure_ignored() {
    std::cout << "--- Test: Duplicate failure report ignored ---\n";
    DegradationManager mgr;
    mgr.report_failure(Sensor::LIDAR_FRONT);
    mgr.report_failure(Sensor::LIDAR_FRONT);  // duplicate
    assert(mgr.failed_sensor_count() == 1);
    assert(mgr.transition_log().size() == 1);
    std::cout << "  PASS\n";
}

void test_progressive_degradation() {
    std::cout << "--- Test: Progressive degradation ---\n";
    DegradationManager mgr;

    mgr.report_failure(Sensor::LIDAR_REAR);
    assert(mgr.current_level() == DegLevel::LIMITED_AUTONOMY);

    mgr.report_failure(Sensor::CAMERA_FRONT);
    assert(mgr.current_level() == DegLevel::DRIVER_ASSIST);

    mgr.report_failure(Sensor::IMU);
    assert(mgr.current_level() == DegLevel::MINIMAL_RISK);

    mgr.report_failure(Sensor::STEERING_ANGLE);
    assert(mgr.current_level() == DegLevel::SAFE_STOP);

    assert(mgr.transition_log().size() == 4);
    std::cout << "  PASS: 4 progressive transitions logged\n";
}

// ============================================================================
int main() {
    std::cout << "=== Graceful Degradation FSM Exercise ===\n\n";

    test_initial_state();
    test_single_non_critical_failure();
    test_single_critical_failure();
    test_forward_perception_total_loss();
    test_two_critical_sensors();
    test_no_upgrade_without_reset();
    test_reset_restores_full();
    test_duplicate_failure_ignored();
    test_progressive_degradation();

    std::cout << "\n=== ALL DEGRADATION TESTS PASSED ===\n";
    return 0;
}
