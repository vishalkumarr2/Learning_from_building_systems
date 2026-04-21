// sf03_fault_tolerance.cpp — Fault Tolerance Patterns
//
// Hardware fails. Cosmic rays flip bits. Memory degrades.
// Safety-critical systems must DETECT and CORRECT faults at runtime.
//
// Four patterns:
//   1. TMR (Triple Modular Redundancy) — vote among 3 copies
//   2. CRC-Protected storage — detect corruption via checksum
//   3. Control flow monitoring — detect skipped/reordered functions
//   4. Hamming-distance state machine — single-bit-flip resilient states
//
// Build: g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -pthread sf03_fault_tolerance.cpp -o sf03_fault_tolerance

#include <array>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <optional>
#include <random>
#include <string>
#include <vector>

// ============================================================================
// Part 1: Triple Modular Redundancy (TMR)
//
// Used in: space systems (radiation), avionics, nuclear reactor controllers.
// Concept: store 3 copies, vote on read. Single corruption → corrected by majority.
// ============================================================================

template <typename T>
class TMR {
    std::array<T, 3> copies_{};

public:
    void store(const T& val) {
        copies_[0] = val;
        copies_[1] = val;
        copies_[2] = val;
    }

    // Majority vote. Returns nullopt if all three disagree.
    std::optional<T> load() const {
        if (copies_[0] == copies_[1]) return copies_[0];
        if (copies_[0] == copies_[2]) return copies_[0];
        if (copies_[1] == copies_[2]) return copies_[1];
        return std::nullopt;  // all three differ — unrecoverable
    }

    // Inject fault: flip a byte in copy N (simulates bit flip from radiation)
    void inject_fault(int copy_index, size_t byte_offset) {
        auto* raw = reinterpret_cast<uint8_t*>(&copies_[copy_index]);
        raw[byte_offset] ^= 0xFF;  // flip all bits in one byte
    }

    // Get raw copy for inspection
    T raw_copy(int index) const { return copies_[index]; }
};

void test_tmr() {
    std::cout << "──── TMR (Triple Modular Redundancy) ────\n\n";

    TMR<uint32_t> tmr;
    tmr.store(0xDEADBEEF);

    // Normal read
    auto val = tmr.load();
    if (val) {
        std::cout << "  Normal read: 0x" << std::hex << std::uppercase << *val << std::dec;
    } else {
        std::cout << "  Normal read: FAIL";
    }
    std::cout << " → " << (val && *val == 0xDEADBEEF ? "PASS ✓" : "FAIL ✗") << "\n";

    // Inject fault into copy 1
    tmr.inject_fault(1, 0);
    auto val2 = tmr.load();
    std::cout << "  After bit flip in copy[1]: ";
    std::cout << "copy[0]=0x" << std::hex << tmr.raw_copy(0)
              << " copy[1]=0x" << tmr.raw_copy(1)
              << " copy[2]=0x" << tmr.raw_copy(2) << std::dec << "\n";
    std::cout << "  TMR vote result: "
              << (val2 ? "corrected ✓ (majority wins)" : "UNRECOVERABLE ✗") << "\n";
    bool tmr_corrected = val2.has_value() && *val2 == 0xDEADBEEF;
    std::cout << "  Value integrity: " << (tmr_corrected ? "PASS ✓" : "FAIL ✗") << "\n";

    // Inject faults into TWO copies — should fail (no majority)
    TMR<uint32_t> tmr2;
    tmr2.store(42);
    tmr2.inject_fault(0, 0);
    tmr2.inject_fault(1, 1);  // different corruption → all three differ
    auto val3 = tmr2.load();
    std::cout << "  Two corrupted copies: " << (!val3 ? "correctly detected ✓" : "MISSED ✗") << "\n";
    std::cout << "\n";
}

// ============================================================================
// Part 2: CRC-Protected Storage
//
// Used in: all communication protocols, EEPROM/Flash storage, memory scrubbing.
// CRC32 detects >99.99% of random corruptions.
// ============================================================================

class CRC32 {
    std::array<uint32_t, 256> table_{};

    void build_table() {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t crc = i;
            for (int j = 0; j < 8; ++j) {
                crc = (crc & 1) ? (crc >> 1) ^ 0xEDB88320u : (crc >> 1);
            }
            table_[i] = crc;
        }
    }

public:
    CRC32() { build_table(); }

    uint32_t compute(const void* data, size_t len) const {
        auto* bytes = static_cast<const uint8_t*>(data);
        uint32_t crc = 0xFFFFFFFF;
        for (size_t i = 0; i < len; ++i) {
            crc = table_[(crc ^ bytes[i]) & 0xFF] ^ (crc >> 8);
        }
        return crc ^ 0xFFFFFFFF;
    }
};

static CRC32 g_crc;

template <typename T>
class CRCProtected {
    T        data_;
    uint32_t crc_;

    uint32_t compute_crc() const {
        return g_crc.compute(&data_, sizeof(T));
    }

public:
    explicit CRCProtected(const T& val) : data_(val), crc_(0) {
        crc_ = compute_crc();
    }

    bool verify() const {
        return compute_crc() == crc_;
    }

    const T& data() const { return data_; }

    // Simulate corruption
    void corrupt_byte(size_t offset) {
        auto* raw = reinterpret_cast<uint8_t*>(&data_);
        if (offset < sizeof(T)) {
            raw[offset] ^= 0x42;
        }
    }
};

void test_crc_protected() {
    std::cout << "──── CRC-Protected Storage ────\n\n";

    struct SensorReading {
        float temperature;
        float pressure;
        uint32_t timestamp;
    };

    SensorReading reading{36.6f, 1013.25f, 1000};
    CRCProtected<SensorReading> protected_reading(reading);

    std::cout << "  Initial verify: " << (protected_reading.verify() ? "PASS ✓" : "FAIL ✗") << "\n";

    // Corrupt one byte
    protected_reading.corrupt_byte(2);
    std::cout << "  After corrupting byte 2: "
              << (!protected_reading.verify() ? "corruption detected ✓" : "MISSED ✗") << "\n";
    std::cout << "\n";
}

// ============================================================================
// Part 3: Control Flow Monitoring
//
// Used in: automotive ECUs, medical devices, industrial controllers.
// Ensures functions execute in the correct sequence — skipped or reordered
// function = potential catastrophe (e.g., arm-before-safe in weapons systems).
// ============================================================================

namespace control_flow {

// Each function adds a unique prime to the flow counter.
// At the end, the counter must equal the sum of all primes.
// A skipped function → wrong sum → fault detected.

static constexpr uint32_t STEP_A_KEY = 7;
static constexpr uint32_t STEP_B_KEY = 13;
static constexpr uint32_t STEP_C_KEY = 29;
static constexpr uint32_t EXPECTED_FLOW = STEP_A_KEY + STEP_B_KEY + STEP_C_KEY;  // 49

struct FlowMonitor {
    uint32_t counter = 0;

    bool verify() const {
        return counter == EXPECTED_FLOW;
    }
};

void step_a(FlowMonitor& fm, int& data) {
    data = 10;
    fm.counter += STEP_A_KEY;
}

void step_b(FlowMonitor& fm, int& data) {
    data *= 2;
    fm.counter += STEP_B_KEY;
}

void step_c(FlowMonitor& fm, int& data) {
    data += 5;
    fm.counter += STEP_C_KEY;
}

} // namespace control_flow

void test_control_flow() {
    std::cout << "──── Control Flow Monitoring ────\n\n";

    // Normal execution: A → B → C
    {
        control_flow::FlowMonitor fm;
        int data = 0;
        control_flow::step_a(fm, data);
        control_flow::step_b(fm, data);
        control_flow::step_c(fm, data);
        std::cout << "  Normal (A→B→C): flow=" << fm.counter
                  << " expected=" << control_flow::EXPECTED_FLOW
                  << " → " << (fm.verify() ? "PASS ✓" : "FAIL ✗") << "\n";
    }

    // Fault: skip step B (simulates corrupted jump or cosmic ray on PC)
    {
        control_flow::FlowMonitor fm;
        int data = 0;
        control_flow::step_a(fm, data);
        // step_b SKIPPED — simulating fault
        control_flow::step_c(fm, data);
        std::cout << "  Skip B (A→C):   flow=" << fm.counter
                  << " expected=" << control_flow::EXPECTED_FLOW
                  << " → " << (!fm.verify() ? "fault detected ✓" : "MISSED ✗") << "\n";
    }

    // Fault: execute B twice (simulates loop corruption)
    {
        control_flow::FlowMonitor fm;
        int data = 0;
        control_flow::step_a(fm, data);
        control_flow::step_b(fm, data);
        control_flow::step_b(fm, data);  // duplicate!
        control_flow::step_c(fm, data);
        std::cout << "  Dup B (A→B→B→C): flow=" << fm.counter
                  << " expected=" << control_flow::EXPECTED_FLOW
                  << " → " << (!fm.verify() ? "fault detected ✓" : "MISSED ✗") << "\n";
    }
    std::cout << "\n";
}

// ============================================================================
// Part 4: Hamming-Distance State Machine
//
// States are encoded such that any two valid states differ by ≥ 3 bits.
// This means a single bit flip CANNOT cause a silent transition to another
// valid state — it always lands in an invalid state, which is detectable.
//
// Used in: IEC 61508 (industrial safety), DO-178C state machines.
// ============================================================================

namespace hamming_sm {

// State encoding: ≥ 3 bits apart (Hamming distance ≥ 3)
// We use 8-bit encodings chosen to be far apart.
enum class State : uint8_t {
    IDLE    = 0b00000000,  // 0x00
    ARMED   = 0b00001111,  // 0x0F  (4 bits from IDLE)
    FIRING  = 0b11110000,  // 0xF0  (4 bits from ARMED, 4 from IDLE)
    SAFE    = 0b11111111,  // 0xFF  (4 bits from FIRING, 4 from ARMED)
    INVALID = 0b10101010,  // sentinel for detection
};

// Hamming distance between two bytes
int hamming_distance(uint8_t a, uint8_t b) {
    return std::bitset<8>(a ^ b).count();
}

// Check if a state value is valid
bool is_valid_state(uint8_t raw) {
    return raw == static_cast<uint8_t>(State::IDLE) ||
           raw == static_cast<uint8_t>(State::ARMED) ||
           raw == static_cast<uint8_t>(State::FIRING) ||
           raw == static_cast<uint8_t>(State::SAFE);
}

// Inject a random single-bit flip
uint8_t flip_random_bit(uint8_t val, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, 7);
    return val ^ (1u << dist(rng));
}

} // namespace hamming_sm

void test_hamming_sm() {
    std::cout << "──── Hamming-Distance State Machine ────\n\n";

    using namespace hamming_sm;

    // Print hamming distances between all valid states
    std::array<std::pair<const char*, uint8_t>, 4> states = {{
        {"IDLE  ", static_cast<uint8_t>(State::IDLE)},
        {"ARMED ", static_cast<uint8_t>(State::ARMED)},
        {"FIRING", static_cast<uint8_t>(State::FIRING)},
        {"SAFE  ", static_cast<uint8_t>(State::SAFE)},
    }};

    std::cout << "  Hamming distances between valid states:\n";
    std::cout << "         ";
    for (const auto& [name, _] : states) std::cout << name << "  ";
    std::cout << "\n";
    for (const auto& [name_a, val_a] : states) {
        std::cout << "  " << name_a << "  ";
        for (const auto& [name_b, val_b] : states) {
            std::cout << "  " << hamming_distance(val_a, val_b) << "      ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // Verify minimum hamming distance ≥ 3
    int min_dist = 8;
    for (size_t i = 0; i < states.size(); ++i) {
        for (size_t j = i + 1; j < states.size(); ++j) {
            int d = hamming_distance(states[i].second, states[j].second);
            min_dist = std::min(min_dist, d);
        }
    }
    std::cout << "  Minimum Hamming distance: " << min_dist
              << " (need ≥ 3) → " << (min_dist >= 3 ? "PASS ✓" : "FAIL ✗") << "\n\n";

    // Inject random single-bit flips and count detection rate
    std::mt19937 rng(42);
    constexpr int TRIALS = 10000;
    int caught = 0;
    int escaped = 0;

    for (int i = 0; i < TRIALS; ++i) {
        // Pick a random valid state
        uint8_t original = states[rng() % 4].second;
        // Flip one bit
        uint8_t corrupted = flip_random_bit(original, rng);

        if (!is_valid_state(corrupted)) {
            ++caught;
        } else {
            ++escaped;  // should be 0 if hamming distance ≥ 3
        }
    }

    std::cout << "  Single-bit flip injection test (" << TRIALS << " trials):\n";
    std::cout << "    Caught:  " << caught << " (" << (100.0 * caught / TRIALS) << "%)\n";
    std::cout << "    Escaped: " << escaped << " (" << (100.0 * escaped / TRIALS) << "%)\n";
    std::cout << "    Result:  " << (escaped == 0 ? "100% detection ✓" : "SOME ESCAPED ✗") << "\n";

    // Also test 2-bit flips
    int caught_2 = 0, escaped_2 = 0;
    for (int i = 0; i < TRIALS; ++i) {
        uint8_t original = states[rng() % 4].second;
        uint8_t corrupted = flip_random_bit(original, rng);
        corrupted = flip_random_bit(corrupted, rng);  // second flip (could undo first)

        if (!is_valid_state(corrupted)) {
            ++caught_2;
        } else {
            ++escaped_2;
        }
    }
    std::cout << "\n  Double-bit flip test (" << TRIALS << " trials):\n";
    std::cout << "    Caught:  " << caught_2 << " (" << (100.0 * caught_2 / TRIALS) << "%)\n";
    std::cout << "    Escaped: " << escaped_2 << " (" << (100.0 * escaped_2 / TRIALS) << "%)\n";
    std::cout << "    Note: with Hamming dist ≥ 3, 2-bit flips are detected; only ≥ 3-bit might escape\n";
    std::cout << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  Fault Tolerance Patterns Exercise\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";

    test_tmr();
    test_crc_protected();
    test_control_flow();
    test_hamming_sm();

    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << R"(
Key Takeaways:
  1. TMR: 3 copies + majority vote. Corrects single fault. Space standard.
  2. CRC32: detects >99.99% of random data corruption. Fast, universal.
  3. Control flow monitoring: unique keys per function, verify sum at end.
     Catches skipped, reordered, or duplicated function calls.
  4. Hamming-distance encoding: valid states are ≥3 bits apart.
     Single-bit flip always lands in detectable invalid state.
  5. Rule of thumb: the cost of detecting a fault ≪ the cost of the fault.
  6. These patterns COMPOSE: TMR + CRC + flow monitoring = defense in depth.
)";
    std::cout << "═══════════════════════════════════════════════════════════════\n";

    return 0;
}
