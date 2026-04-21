// =============================================================================
// Exercise 05: UDS Diagnostic Trouble Code (DTC) Manager
// =============================================================================
// Implement a DTC fault memory system following ISO 14229 (UDS) principles.
//
// Features:
//   1. Store DTCs with status byte (ISO 14229-1 Annex D)
//   2. Freeze frame data capture on first detection
//   3. Aging counter (heals after N consecutive good cycles)
//   4. Snapshot data for each DTC occurrence
//   5. Clear all DTCs (service 0x14)
//   6. Query by status mask (service 0x19 sub 0x02)
//
// DTC Status byte bits (ISO 14229-1):
//   Bit 0: testFailed                — DTC is currently failing
//   Bit 1: testFailedThisOperationCycle — Failed at least once this cycle
//   Bit 2: pendingDTC                — Waiting for confirmation
//   Bit 3: confirmedDTC              — Matured (debounce count met)
//   Bit 4: testNotCompletedSinceLastClear
//   Bit 5: testFailedSinceLastClear
//   Bit 6: testNotCompletedThisOperationCycle
//   Bit 7: warningIndicatorRequested — MIL lamp
//
// Production relevance:
//   - Every production vehicle has DTC fault memory
//   - Dealers read DTCs via OBD-II / UDS scantools
//   - OTA systems clear DTCs after software updates
// =============================================================================

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <vector>

// ============================================================================
// DTC Status Byte Flags
// ============================================================================
namespace StatusBit {
    constexpr uint8_t TEST_FAILED                        = 0x01;
    constexpr uint8_t TEST_FAILED_THIS_OP_CYCLE          = 0x02;
    constexpr uint8_t PENDING_DTC                        = 0x04;
    constexpr uint8_t CONFIRMED_DTC                      = 0x08;
    constexpr uint8_t TEST_NOT_COMPLETED_SINCE_CLEAR     = 0x10;
    constexpr uint8_t TEST_FAILED_SINCE_LAST_CLEAR       = 0x20;
    constexpr uint8_t TEST_NOT_COMPLETED_THIS_OP_CYCLE   = 0x40;
    constexpr uint8_t WARNING_INDICATOR_REQUESTED        = 0x80;
}

// Helpers for uint8_t bitwise ops (avoids int promotion warnings)
inline void set_bits(uint8_t& byte, uint8_t mask) {
    byte = static_cast<uint8_t>(byte | mask);
}
inline void clear_bits(uint8_t& byte, uint8_t mask) {
    byte = static_cast<uint8_t>(byte & static_cast<uint8_t>(~mask));
}

// ============================================================================
// Freeze Frame Data
// ============================================================================
struct FreezeFrame {
    uint64_t timestamp_ms = 0;   // capture time
    double   vehicle_speed = 0.0;
    double   engine_rpm = 0.0;
    double   battery_voltage = 0.0;
    int16_t  ambient_temp_c = 0;
    uint32_t odometer_km = 0;
};

// ============================================================================
// DTC Record
// ============================================================================
struct DtcRecord {
    uint32_t dtc_number = 0;           // 3-byte DTC (e.g. 0xC07300)
    uint8_t  status_byte = 0;
    uint16_t occurrence_count = 0;
    uint8_t  aging_counter = 0;        // counts good cycles
    FreezeFrame first_freeze;          // captured on first detection
    FreezeFrame latest_freeze;         // most recent occurrence

    bool test_failed() const {
        return status_byte & StatusBit::TEST_FAILED;
    }
    bool is_confirmed() const {
        return status_byte & StatusBit::CONFIRMED_DTC;
    }
    bool is_pending() const {
        return status_byte & StatusBit::PENDING_DTC;
    }
    bool matches_mask(uint8_t mask) const {
        return (status_byte & mask) != 0;
    }
};

// ============================================================================
// DTC Manager
// ============================================================================
class DtcManager {
    std::map<uint32_t, DtcRecord> records_;

    static constexpr uint8_t CONFIRM_THRESHOLD = 3;  // confirm after N fails
    static constexpr uint8_t AGING_THRESHOLD = 40;    // heal after N good cycles
    static constexpr uint16_t MAX_OCCURRENCE = 0xFF;

    uint16_t debounce_count(uint32_t dtc) const {
        auto it = records_.find(dtc);
        if (it == records_.end()) return 0;
        return it->second.occurrence_count;
    }

public:
    // Report a test result for a DTC
    void report_test_result(uint32_t dtc, bool failed,
                             FreezeFrame const& env = {}) {
        auto& rec = records_[dtc];

        if (rec.dtc_number == 0) {
            // New DTC
            rec.dtc_number = dtc;
            rec.status_byte = static_cast<uint8_t>(
                StatusBit::TEST_NOT_COMPLETED_SINCE_CLEAR |
                StatusBit::TEST_NOT_COMPLETED_THIS_OP_CYCLE);
        }

        // Clear "not completed" bits — we just ran the test
        clear_bits(rec.status_byte,
                   StatusBit::TEST_NOT_COMPLETED_THIS_OP_CYCLE);
        clear_bits(rec.status_byte,
                   StatusBit::TEST_NOT_COMPLETED_SINCE_CLEAR);

        if (failed) {
            set_bits(rec.status_byte, StatusBit::TEST_FAILED);
            set_bits(rec.status_byte, StatusBit::TEST_FAILED_THIS_OP_CYCLE);
            set_bits(rec.status_byte, StatusBit::TEST_FAILED_SINCE_LAST_CLEAR);
            set_bits(rec.status_byte, StatusBit::PENDING_DTC);
            rec.aging_counter = 0;

            if (rec.occurrence_count < MAX_OCCURRENCE) {
                ++rec.occurrence_count;
            }

            // Capture freeze frame
            if (rec.occurrence_count == 1) {
                rec.first_freeze = env;
            }
            rec.latest_freeze = env;

            // Confirm after threshold
            if (rec.occurrence_count >= CONFIRM_THRESHOLD) {
                set_bits(rec.status_byte, StatusBit::CONFIRMED_DTC);
                set_bits(rec.status_byte,
                         StatusBit::WARNING_INDICATOR_REQUESTED);
            }
        } else {
            // Test passed
            clear_bits(rec.status_byte, StatusBit::TEST_FAILED);

            if (rec.is_confirmed() || rec.is_pending()) {
                ++rec.aging_counter;
                if (rec.aging_counter >= AGING_THRESHOLD) {
                    // Heal: clear pending, confirmed, warning
                    clear_bits(rec.status_byte, StatusBit::PENDING_DTC);
                    clear_bits(rec.status_byte, StatusBit::CONFIRMED_DTC);
                    clear_bits(rec.status_byte,
                               StatusBit::WARNING_INDICATOR_REQUESTED);
                    // Keep TEST_FAILED_SINCE_LAST_CLEAR — only clear clears that
                }
            }
        }
    }

    // Start a new operation cycle (e.g., ignition on)
    void new_operation_cycle() {
        for (auto& [dtc, rec] : records_) {
            clear_bits(rec.status_byte,
                       StatusBit::TEST_FAILED_THIS_OP_CYCLE);
            set_bits(rec.status_byte,
                     StatusBit::TEST_NOT_COMPLETED_THIS_OP_CYCLE);
        }
    }

    // Clear all DTCs (UDS service 0x14)
    void clear_all() {
        records_.clear();
    }

    // Clear specific DTC
    void clear_dtc(uint32_t dtc) {
        records_.erase(dtc);
    }

    // Query DTCs by status mask (UDS service 0x19 sub 0x02)
    std::vector<DtcRecord> query_by_mask(uint8_t mask) const {
        std::vector<DtcRecord> result;
        for (auto const& [dtc, rec] : records_) {
            if (rec.matches_mask(mask)) {
                result.push_back(rec);
            }
        }
        return result;
    }

    // Get specific DTC record
    std::optional<DtcRecord> get(uint32_t dtc) const {
        auto it = records_.find(dtc);
        if (it == records_.end()) return std::nullopt;
        return it->second;
    }

    size_t total_stored() const { return records_.size(); }
};

// ============================================================================
// Self-Test
// ============================================================================
void test_first_failure_is_pending() {
    std::cout << "--- Test: First failure creates pending DTC ---\n";
    DtcManager mgr;

    FreezeFrame env{1000, 60.0, 3000.0, 12.5, 25, 50000};
    mgr.report_test_result(0xC07300, true, env);

    auto rec = mgr.get(0xC07300);
    assert(rec.has_value());
    assert(rec->test_failed());
    assert(rec->is_pending());
    assert(!rec->is_confirmed());  // not yet — needs 3 fails
    assert(rec->occurrence_count == 1);
    assert(rec->first_freeze.vehicle_speed == 60.0);
    std::cout << "  PASS: Pending, not confirmed, freeze captured\n";
}

void test_confirmation_after_threshold() {
    std::cout << "--- Test: Confirmed after 3 failures ---\n";
    DtcManager mgr;

    for (int i = 0; i < 3; ++i) {
        mgr.report_test_result(0xC07300, true);
    }

    auto rec = mgr.get(0xC07300);
    assert(rec->is_confirmed());
    assert(rec->status_byte & StatusBit::WARNING_INDICATOR_REQUESTED);
    assert(rec->occurrence_count == 3);
    std::cout << "  PASS: Confirmed with MIL lamp\n";
}

void test_aging_heals_dtc() {
    std::cout << "--- Test: Aging heals DTC after 40 good cycles ---\n";
    DtcManager mgr;

    // Confirm the DTC
    for (int i = 0; i < 3; ++i) {
        mgr.report_test_result(0xC07300, true);
    }
    assert(mgr.get(0xC07300)->is_confirmed());

    // 39 good cycles — still confirmed
    for (int i = 0; i < 39; ++i) {
        mgr.report_test_result(0xC07300, false);
    }
    assert(mgr.get(0xC07300)->is_confirmed());

    // 40th good cycle — healed
    mgr.report_test_result(0xC07300, false);
    auto rec = mgr.get(0xC07300);
    assert(!rec->is_confirmed());
    assert(!rec->is_pending());
    // But TEST_FAILED_SINCE_LAST_CLEAR still set
    assert(rec->status_byte & StatusBit::TEST_FAILED_SINCE_LAST_CLEAR);
    std::cout << "  PASS: Healed after 40 good cycles\n";
}

void test_clear_all() {
    std::cout << "--- Test: Clear all DTCs ---\n";
    DtcManager mgr;

    mgr.report_test_result(0xC07300, true);
    mgr.report_test_result(0x010100, true);  // Powertrain DTC (P0100)
    mgr.report_test_result(0xC00401, true);  // Network DTC (U0401)
    assert(mgr.total_stored() == 3);

    mgr.clear_all();
    assert(mgr.total_stored() == 0);
    assert(!mgr.get(0xC07300).has_value());
    std::cout << "  PASS: All DTCs cleared\n";
}

void test_query_by_status_mask() {
    std::cout << "--- Test: Query by status mask ---\n";
    DtcManager mgr;

    // DTC 1: confirmed (3 fails)
    for (int i = 0; i < 3; ++i)
        mgr.report_test_result(0xC07300, true);

    // DTC 2: pending only (1 fail)
    mgr.report_test_result(0xC07400, true);

    // DTC 3: passed test (not failing)
    mgr.report_test_result(0xC07500, false);

    // Query: confirmed only
    auto confirmed = mgr.query_by_mask(StatusBit::CONFIRMED_DTC);
    assert(confirmed.size() == 1);
    assert(confirmed[0].dtc_number == 0xC07300);

    // Query: currently failing
    auto failing = mgr.query_by_mask(StatusBit::TEST_FAILED);
    assert(failing.size() == 2);  // C07300 + C07400

    // Query: pending
    auto pending = mgr.query_by_mask(StatusBit::PENDING_DTC);
    assert(pending.size() == 2);  // C07300 + C07400

    std::cout << "  PASS: Mask queries return correct DTCs\n";
}

void test_operation_cycle_reset() {
    std::cout << "--- Test: Operation cycle resets ---\n";
    DtcManager mgr;

    mgr.report_test_result(0xC07300, true);
    assert(mgr.get(0xC07300)->status_byte &
           StatusBit::TEST_FAILED_THIS_OP_CYCLE);

    mgr.new_operation_cycle();
    auto rec = mgr.get(0xC07300);
    assert(!(rec->status_byte &
             StatusBit::TEST_FAILED_THIS_OP_CYCLE));
    assert(rec->status_byte &
           StatusBit::TEST_NOT_COMPLETED_THIS_OP_CYCLE);
    // But TEST_FAILED_SINCE_LAST_CLEAR persists
    assert(rec->status_byte &
           StatusBit::TEST_FAILED_SINCE_LAST_CLEAR);
    std::cout << "  PASS: Operation cycle flags reset correctly\n";
}

void test_freeze_frame_first_vs_latest() {
    std::cout << "--- Test: Freeze frame first vs latest ---\n";
    DtcManager mgr;

    FreezeFrame env1{1000, 60.0, 3000.0, 12.5, 25, 50000};
    FreezeFrame env2{2000, 80.0, 4000.0, 12.3, 30, 50100};
    FreezeFrame env3{3000, 100.0, 5000.0, 12.1, 35, 50200};

    mgr.report_test_result(0xC07300, true, env1);
    mgr.report_test_result(0xC07300, true, env2);
    mgr.report_test_result(0xC07300, true, env3);

    auto rec = mgr.get(0xC07300);
    // First freeze should be env1
    assert(rec->first_freeze.vehicle_speed == 60.0);
    assert(rec->first_freeze.timestamp_ms == 1000);
    // Latest freeze should be env3
    assert(rec->latest_freeze.vehicle_speed == 100.0);
    assert(rec->latest_freeze.timestamp_ms == 3000);
    std::cout << "  PASS: First and latest freeze frames preserved\n";
}

void test_aging_resets_on_new_failure() {
    std::cout << "--- Test: Aging counter resets on new failure ---\n";
    DtcManager mgr;

    // Confirm DTC
    for (int i = 0; i < 3; ++i)
        mgr.report_test_result(0xC07300, true);

    // Age for 30 cycles (not enough to heal)
    for (int i = 0; i < 30; ++i)
        mgr.report_test_result(0xC07300, false);

    // Fail again — aging resets
    mgr.report_test_result(0xC07300, true);

    // Need 40 more good cycles to heal
    for (int i = 0; i < 39; ++i)
        mgr.report_test_result(0xC07300, false);
    assert(mgr.get(0xC07300)->is_confirmed());

    mgr.report_test_result(0xC07300, false);
    assert(!mgr.get(0xC07300)->is_confirmed());
    std::cout << "  PASS: Aging reset on intermittent failure\n";
}

// ============================================================================
int main() {
    std::cout << "=== UDS DTC Manager Exercise ===\n\n";

    test_first_failure_is_pending();
    test_confirmation_after_threshold();
    test_aging_heals_dtc();
    test_clear_all();
    test_query_by_status_mask();
    test_operation_cycle_reset();
    test_freeze_frame_first_vs_latest();
    test_aging_resets_on_new_failure();

    std::cout << "\n=== ALL DTC MANAGER TESTS PASSED ===\n";
    return 0;
}
