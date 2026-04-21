// =============================================================================
// Exercise 08: Watchdog Service Health Monitor
// =============================================================================
// Implement a watchdog supervision system with three monitoring modes:
//
//   1. Alive supervision — periodic heartbeat ("I'm still running")
//   2. Deadline supervision — task must complete within time budget
//   3. Logical supervision — tasks must execute in correct order
//
// This mirrors AUTOSAR Watchdog Manager (WdgM) and ISO 26262
// requirements for software monitoring at ASIL-B and above.
//
// Production relevance:
//   - Every ASIL-rated ECU has hardware watchdog + software supervision
//   - Failure to pet the watchdog triggers system-level recovery
//   - Logical monitoring catches control flow corruption (SEU, cosmic rays)
// =============================================================================

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using Duration = std::chrono::milliseconds;

// ============================================================================
// Supervision Status
// ============================================================================
enum class SupervisionStatus : uint8_t {
    OK,
    EXPIRED,       // Alive/deadline timeout
    WRONG_ORDER,   // Logical supervision violated
    DEACTIVATED
};

constexpr const char* to_string(SupervisionStatus s) {
    switch (s) {
        case SupervisionStatus::OK:          return "OK";
        case SupervisionStatus::EXPIRED:     return "EXPIRED";
        case SupervisionStatus::WRONG_ORDER: return "WRONG_ORDER";
        case SupervisionStatus::DEACTIVATED: return "DEACTIVATED";
    }
    return "UNKNOWN";
}

// ============================================================================
// Supervised Entity (one per software component)
// ============================================================================
struct SupervisedEntity {
    std::string name;
    uint16_t    id = 0;

    // Alive supervision config
    Duration    alive_period = Duration::zero();
    Duration    alive_tolerance = Duration::zero();  // margin
    TimePoint   last_alive = {};
    bool        alive_enabled = false;

    // Deadline supervision config
    Duration    deadline_max = Duration::zero();
    TimePoint   deadline_start = {};
    bool        deadline_active = false;
    bool        deadline_enabled = false;

    // Logical supervision config
    std::vector<uint8_t> expected_sequence;
    std::vector<uint8_t> actual_sequence;
    bool        logical_enabled = false;

    // Status
    SupervisionStatus status = SupervisionStatus::OK;
    uint32_t failure_count = 0;
};

// ============================================================================
// Recovery Action
// ============================================================================
enum class RecoveryAction : uint8_t {
    NONE,
    RESTART_COMPONENT,
    RESTART_PARTITION,
    RESET_ECU,
    ENTER_SAFE_STATE
};

constexpr const char* to_string(RecoveryAction a) {
    switch (a) {
        case RecoveryAction::NONE:              return "NONE";
        case RecoveryAction::RESTART_COMPONENT: return "RESTART_COMPONENT";
        case RecoveryAction::RESTART_PARTITION: return "RESTART_PARTITION";
        case RecoveryAction::RESET_ECU:         return "RESET_ECU";
        case RecoveryAction::ENTER_SAFE_STATE:  return "ENTER_SAFE_STATE";
    }
    return "UNKNOWN";
}

// ============================================================================
// Health Monitor (Watchdog Manager)
// ============================================================================
class HealthMonitor {
    std::map<uint16_t, SupervisedEntity> entities_;
    std::vector<std::pair<uint16_t, RecoveryAction>> recovery_log_;

    // Escalation thresholds
    static constexpr uint32_t RESTART_THRESHOLD = 1;
    static constexpr uint32_t PARTITION_THRESHOLD = 3;
    static constexpr uint32_t RESET_THRESHOLD = 5;

    RecoveryAction determine_recovery(uint32_t failures) const {
        if (failures >= RESET_THRESHOLD) return RecoveryAction::ENTER_SAFE_STATE;
        if (failures >= PARTITION_THRESHOLD) return RecoveryAction::RESTART_PARTITION;
        if (failures >= RESTART_THRESHOLD) return RecoveryAction::RESTART_COMPONENT;
        return RecoveryAction::NONE;
    }

    void record_failure(SupervisedEntity& ent, SupervisionStatus reason) {
        ent.status = reason;
        ++ent.failure_count;
        auto action = determine_recovery(ent.failure_count);
        recovery_log_.push_back({ent.id, action});
    }

public:
    // --- Registration ---

    void register_alive(uint16_t id, std::string name,
                         Duration period, Duration tolerance) {
        auto& ent = entities_[id];
        ent.id = id;
        ent.name = std::move(name);
        ent.alive_period = period;
        ent.alive_tolerance = tolerance;
        ent.alive_enabled = true;
        ent.last_alive = Clock::now();
        ent.status = SupervisionStatus::OK;
    }

    void register_deadline(uint16_t id, std::string name,
                            Duration max_duration) {
        auto& ent = entities_[id];
        ent.id = id;
        ent.name = std::move(name);
        ent.deadline_max = max_duration;
        ent.deadline_enabled = true;
        ent.status = SupervisionStatus::OK;
    }

    void register_logical(uint16_t id, std::string name,
                           std::vector<uint8_t> expected_seq) {
        auto& ent = entities_[id];
        ent.id = id;
        ent.name = std::move(name);
        ent.expected_sequence = std::move(expected_seq);
        ent.logical_enabled = true;
        ent.status = SupervisionStatus::OK;
    }

    // --- Runtime Operations ---

    // Alive: component reports it's still running
    void alive(uint16_t id) {
        auto it = entities_.find(id);
        if (it == entities_.end()) return;
        it->second.last_alive = Clock::now();
        it->second.status = SupervisionStatus::OK;
    }

    // Deadline: mark start of a timed operation
    void deadline_start(uint16_t id) {
        auto it = entities_.find(id);
        if (it == entities_.end()) return;
        it->second.deadline_start = Clock::now();
        it->second.deadline_active = true;
    }

    // Deadline: mark completion of a timed operation
    bool deadline_end(uint16_t id) {
        auto it = entities_.find(id);
        if (it == entities_.end()) return false;
        auto& ent = it->second;

        if (!ent.deadline_active) return false;
        ent.deadline_active = false;

        auto elapsed = std::chrono::duration_cast<Duration>(
            Clock::now() - ent.deadline_start);

        if (elapsed > ent.deadline_max) {
            record_failure(ent, SupervisionStatus::EXPIRED);
            return false;
        }

        ent.status = SupervisionStatus::OK;
        return true;
    }

    // Logical: report a checkpoint in the expected sequence
    void checkpoint(uint16_t id, uint8_t checkpoint_id) {
        auto it = entities_.find(id);
        if (it == entities_.end()) return;
        auto& ent = it->second;

        ent.actual_sequence.push_back(checkpoint_id);

        // Check prefix: every checkpoint so far must match expected
        size_t idx = ent.actual_sequence.size() - 1;
        if (idx < ent.expected_sequence.size()) {
            if (ent.actual_sequence[idx] != ent.expected_sequence[idx]) {
                record_failure(ent, SupervisionStatus::WRONG_ORDER);
            }
        }
    }

    // Logical: verify complete sequence and reset
    bool logical_verify(uint16_t id) {
        auto it = entities_.find(id);
        if (it == entities_.end()) return false;
        auto& ent = it->second;

        bool ok = (ent.actual_sequence == ent.expected_sequence);
        if (!ok) {
            record_failure(ent, SupervisionStatus::WRONG_ORDER);
        } else {
            ent.status = SupervisionStatus::OK;
        }

        ent.actual_sequence.clear();
        return ok;
    }

    // --- Periodic check (called by watchdog trigger) ---

    void check_all() {
        auto now = Clock::now();

        for (auto& [id, ent] : entities_) {
            if (ent.alive_enabled) {
                auto elapsed = std::chrono::duration_cast<Duration>(
                    now - ent.last_alive);
                Duration limit = ent.alive_period + ent.alive_tolerance;
                if (elapsed > limit) {
                    record_failure(ent, SupervisionStatus::EXPIRED);
                }
            }
        }
    }

    // --- Queries ---

    SupervisionStatus status(uint16_t id) const {
        auto it = entities_.find(id);
        if (it == entities_.end()) return SupervisionStatus::DEACTIVATED;
        return it->second.status;
    }

    uint32_t failure_count(uint16_t id) const {
        auto it = entities_.find(id);
        if (it == entities_.end()) return 0;
        return it->second.failure_count;
    }

    std::vector<std::pair<uint16_t, RecoveryAction>> const&
    recovery_log() const {
        return recovery_log_;
    }

    size_t entity_count() const { return entities_.size(); }
};

// ============================================================================
// Self-Test
// ============================================================================
void test_alive_supervision_ok() {
    std::cout << "--- Test: Alive supervision OK ---\n";
    HealthMonitor mon;

    mon.register_alive(1, "SensorFusion", Duration(100), Duration(20));
    mon.alive(1);
    mon.check_all();
    assert(mon.status(1) == SupervisionStatus::OK);
    std::cout << "  PASS\n";
}

void test_alive_supervision_expired() {
    std::cout << "--- Test: Alive supervision expired ---\n";
    HealthMonitor mon;

    // Very short period for testing
    mon.register_alive(1, "SensorFusion", Duration(10), Duration(5));

    // Wait longer than period + tolerance
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    mon.check_all();
    assert(mon.status(1) == SupervisionStatus::EXPIRED);
    assert(mon.failure_count(1) == 1);
    std::cout << "  PASS: Expired after timeout\n";
}

void test_deadline_supervision_ok() {
    std::cout << "--- Test: Deadline supervision OK ---\n";
    HealthMonitor mon;

    mon.register_deadline(2, "PathPlanner", Duration(100));
    mon.deadline_start(2);

    // Simulate fast computation
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    assert(mon.deadline_end(2));
    assert(mon.status(2) == SupervisionStatus::OK);
    std::cout << "  PASS\n";
}

void test_deadline_supervision_overrun() {
    std::cout << "--- Test: Deadline supervision overrun ---\n";
    HealthMonitor mon;

    mon.register_deadline(2, "PathPlanner", Duration(10));  // 10ms budget
    mon.deadline_start(2);

    // Simulate slow computation
    std::this_thread::sleep_for(std::chrono::milliseconds(20));

    assert(!mon.deadline_end(2));
    assert(mon.status(2) == SupervisionStatus::EXPIRED);
    std::cout << "  PASS: Deadline overrun detected\n";
}

void test_logical_supervision_correct_order() {
    std::cout << "--- Test: Logical supervision correct order ---\n";
    HealthMonitor mon;

    // Expected: INIT(1) → SENSE(2) → PLAN(3) → ACT(4)
    mon.register_logical(3, "ControlLoop", {1, 2, 3, 4});

    mon.checkpoint(3, 1);  // INIT
    mon.checkpoint(3, 2);  // SENSE
    mon.checkpoint(3, 3);  // PLAN
    mon.checkpoint(3, 4);  // ACT

    assert(mon.logical_verify(3));
    assert(mon.status(3) == SupervisionStatus::OK);
    std::cout << "  PASS\n";
}

void test_logical_supervision_wrong_order() {
    std::cout << "--- Test: Logical supervision wrong order ---\n";
    HealthMonitor mon;

    mon.register_logical(3, "ControlLoop", {1, 2, 3, 4});

    mon.checkpoint(3, 1);  // INIT
    mon.checkpoint(3, 3);  // PLAN — skipped SENSE!

    // Wrong order detected at checkpoint
    assert(mon.status(3) == SupervisionStatus::WRONG_ORDER);
    std::cout << "  PASS: Wrong order detected\n";
}

void test_logical_supervision_incomplete() {
    std::cout << "--- Test: Logical supervision incomplete ---\n";
    HealthMonitor mon;

    mon.register_logical(3, "ControlLoop", {1, 2, 3, 4});

    mon.checkpoint(3, 1);
    mon.checkpoint(3, 2);
    // Missing 3 and 4

    assert(!mon.logical_verify(3));
    assert(mon.status(3) == SupervisionStatus::WRONG_ORDER);
    std::cout << "  PASS: Incomplete sequence detected\n";
}

void test_recovery_escalation() {
    std::cout << "--- Test: Recovery escalation ---\n";
    HealthMonitor mon;

    mon.register_alive(1, "SensorFusion", Duration(5), Duration(2));

    // Generate 5 failures through repeated timeout checks
    for (int i = 0; i < 5; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        mon.check_all();
    }

    auto const& log = mon.recovery_log();
    assert(log.size() >= 5);

    // First failure → RESTART_COMPONENT
    assert(log[0].second == RecoveryAction::RESTART_COMPONENT);
    // Third failure → RESTART_PARTITION
    assert(log[2].second == RecoveryAction::RESTART_PARTITION);
    // Fifth failure → ENTER_SAFE_STATE
    assert(log[4].second == RecoveryAction::ENTER_SAFE_STATE);

    std::cout << "  PASS: Escalation: RESTART → PARTITION → SAFE_STATE\n";
}

void test_alive_recovery_after_heartbeat() {
    std::cout << "--- Test: Alive recovery after heartbeat ---\n";
    HealthMonitor mon;

    mon.register_alive(1, "SensorFusion", Duration(10), Duration(5));

    // Let it expire
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    mon.check_all();
    assert(mon.status(1) == SupervisionStatus::EXPIRED);

    // Send heartbeat
    mon.alive(1);
    assert(mon.status(1) == SupervisionStatus::OK);
    std::cout << "  PASS: Recovered after heartbeat\n";
}

void test_multiple_entities() {
    std::cout << "--- Test: Multiple entities ---\n";
    HealthMonitor mon;

    mon.register_alive(1, "SensorFusion", Duration(100), Duration(10));
    mon.register_deadline(2, "PathPlanner", Duration(50));
    mon.register_logical(3, "ControlLoop", {1, 2, 3});

    assert(mon.entity_count() == 3);

    mon.alive(1);
    mon.deadline_start(2);
    mon.deadline_end(2);
    mon.checkpoint(3, 1);
    mon.checkpoint(3, 2);
    mon.checkpoint(3, 3);
    mon.logical_verify(3);

    mon.check_all();

    assert(mon.status(1) == SupervisionStatus::OK);
    assert(mon.status(2) == SupervisionStatus::OK);
    assert(mon.status(3) == SupervisionStatus::OK);
    std::cout << "  PASS: All 3 entities monitored\n";
}

// ============================================================================
int main() {
    std::cout << "=== Watchdog Service Health Monitor Exercise ===\n\n";

    test_alive_supervision_ok();
    test_alive_supervision_expired();
    test_deadline_supervision_ok();
    test_deadline_supervision_overrun();
    test_logical_supervision_correct_order();
    test_logical_supervision_wrong_order();
    test_logical_supervision_incomplete();
    test_recovery_escalation();
    test_alive_recovery_after_heartbeat();
    test_multiple_entities();

    std::cout << "\n=== ALL HEALTH MONITOR TESTS PASSED ===\n";
    return 0;
}
