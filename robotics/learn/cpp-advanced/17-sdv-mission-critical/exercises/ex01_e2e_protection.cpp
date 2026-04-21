// =============================================================================
// Exercise 01: AUTOSAR E2E Protection (Profile 4)
// =============================================================================
// Implement end-to-end protection for vehicle messages.
//
// In production SDV systems, every message crossing a trust boundary must be
// protected against:
//   - Bit flips (CRC check)
//   - Message loss (sequence counter with gap detection)
//   - Message repetition (counter must advance)
//   - Stale data (timestamp + max age)
//   - Plausibility faults (range checks on payload)
//
// This exercise implements a simplified E2E Profile 4 protector and checker.
// =============================================================================

#include <cassert>
#include <cstdint>
#include <cstring>
#include <array>
#include <chrono>
#include <iostream>
#include <optional>
#include <type_traits>
#include <vector>

// SFINAE helper: detect if T has is_plausible() method
template<typename T, typename = void>
struct has_is_plausible : std::false_type {};
template<typename T>
struct has_is_plausible<T, std::void_t<decltype(std::declval<T>().is_plausible())>> : std::true_type {};

// ============================================================================
// CRC-32 implementation (simplified — production uses hardware CRC)
// ============================================================================
namespace crc {

constexpr uint32_t POLYNOMIAL = 0xEDB88320u;

constexpr std::array<uint32_t, 256> generate_table() {
    std::array<uint32_t, 256> table{};
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t crc = i;
        for (int j = 0; j < 8; ++j) {
            crc = (crc >> 1) ^ (POLYNOMIAL * (crc & 1u));
        }
        table[i] = crc;
    }
    return table;
}

inline constexpr auto TABLE = generate_table();

uint32_t compute(const uint8_t* data, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) {
        crc = TABLE[(crc ^ data[i]) & 0xFFu] ^ (crc >> 8);
    }
    return crc ^ 0xFFFFFFFFu;
}

} // namespace crc

// ============================================================================
// E2E Header (Profile 4 simplified)
// ============================================================================
struct E2EHeader {
    uint32_t data_id;    // identifies the message type
    uint16_t length;     // payload length in bytes
    uint16_t counter;    // sequence number
    uint32_t crc;        // CRC-32 over header (excluding crc field) + payload
};

// ============================================================================
// E2E Check Status
// ============================================================================
enum class E2EStatus {
    OK,             // Message is valid and in sequence
    WRONG_CRC,      // Data corruption detected
    LOST,           // Counter gap detected (messages lost)
    REPEATED,       // Counter did not advance (repeated message)
    OUT_OF_RANGE,   // Payload value outside plausible range
    WRONG_DATA_ID,  // Data ID mismatch
    SIZE_ERROR      // Buffer too small
};

inline const char* to_string(E2EStatus s) {
    switch (s) {
        case E2EStatus::OK:            return "OK";
        case E2EStatus::WRONG_CRC:     return "WRONG_CRC";
        case E2EStatus::LOST:          return "LOST";
        case E2EStatus::REPEATED:      return "REPEATED";
        case E2EStatus::OUT_OF_RANGE:  return "OUT_OF_RANGE";
        case E2EStatus::WRONG_DATA_ID: return "WRONG_DATA_ID";
        case E2EStatus::SIZE_ERROR:    return "SIZE_ERROR";
    }
    return "UNKNOWN";
}

// ============================================================================
// Payload: Vehicle Speed message
// ============================================================================
struct VehicleSpeed {
    float speed_mps;       // meters per second
    float acceleration;    // m/s^2
    uint8_t quality;       // 0=invalid, 1=estimated, 2=measured

    // Plausibility check: speed must be in [-5, 90] m/s, accel in [-15, 15]
    bool is_plausible() const {
        return speed_mps >= -5.0f && speed_mps <= 90.0f &&
               acceleration >= -15.0f && acceleration <= 15.0f &&
               quality <= 2;
    }
};

// ============================================================================
// E2E Protector (sender side)
// ============================================================================
class E2EProtector {
    uint16_t tx_counter_ = 0;
    uint32_t data_id_;

public:
    explicit E2EProtector(uint32_t data_id) : data_id_(data_id) {}

    // Protect a payload: returns serialized buffer with E2E header
    template<typename T>
    std::vector<uint8_t> protect(T const& payload) {
        static_assert(std::is_trivially_copyable_v<T>,
                      "Payload must be trivially copyable for E2E");

        E2EHeader header{};
        header.data_id = data_id_;
        header.length = static_cast<uint16_t>(sizeof(T));
        header.counter = tx_counter_++;
        header.crc = 0;  // placeholder — filled after CRC computation

        // Serialize: header + payload
        constexpr size_t HDR_SIZE = sizeof(E2EHeader);
        std::vector<uint8_t> buffer(HDR_SIZE + sizeof(T));
        std::memcpy(buffer.data(), &header, HDR_SIZE);
        std::memcpy(buffer.data() + HDR_SIZE, &payload, sizeof(T));

        // Compute CRC over everything except the crc field itself
        // CRC covers: data_id(4) + length(2) + counter(2) + payload
        // Skip bytes [8..11] which are the crc field
        uint32_t crc_val = 0;
        {
            // Build CRC input: header fields before crc + payload
            std::vector<uint8_t> crc_input;
            crc_input.insert(crc_input.end(),
                              buffer.begin(),
                              buffer.begin() + offsetof(E2EHeader, crc));
            crc_input.insert(crc_input.end(),
                              buffer.begin() + HDR_SIZE,
                              buffer.end());
            crc_val = crc::compute(crc_input.data(), crc_input.size());
        }

        // Write CRC back into buffer
        std::memcpy(buffer.data() + offsetof(E2EHeader, crc),
                     &crc_val, sizeof(crc_val));

        return buffer;
    }

    uint16_t get_counter() const { return tx_counter_; }
};

// ============================================================================
// E2E Checker (receiver side)
// ============================================================================
class E2EChecker {
    uint16_t rx_expected_ = 0;
    uint32_t data_id_;
    bool first_message_ = true;
    static constexpr uint16_t MAX_DELTA = 5;  // tolerate up to 5 lost

public:
    explicit E2EChecker(uint32_t data_id) : data_id_(data_id) {}

    // Check a received buffer and extract payload
    template<typename T>
    std::pair<E2EStatus, T> check(std::vector<uint8_t> const& buffer) {
        static_assert(std::is_trivially_copyable_v<T>);
        T payload{};

        constexpr size_t HDR_SIZE = sizeof(E2EHeader);

        // Size check
        if (buffer.size() < HDR_SIZE + sizeof(T)) {
            return {E2EStatus::SIZE_ERROR, payload};
        }

        // Deserialize header
        E2EHeader header{};
        std::memcpy(&header, buffer.data(), HDR_SIZE);

        // Data ID check
        if (header.data_id != data_id_) {
            return {E2EStatus::WRONG_DATA_ID, payload};
        }

        // CRC verification
        uint32_t received_crc = header.crc;
        {
            std::vector<uint8_t> crc_input;
            crc_input.insert(crc_input.end(),
                              buffer.begin(),
                              buffer.begin() + offsetof(E2EHeader, crc));
            crc_input.insert(crc_input.end(),
                              buffer.begin() + HDR_SIZE,
                              buffer.end());
            uint32_t computed_crc =
                crc::compute(crc_input.data(), crc_input.size());
            if (computed_crc != received_crc) {
                return {E2EStatus::WRONG_CRC, payload};
            }
        }

        // Deserialize payload
        std::memcpy(&payload, buffer.data() + HDR_SIZE, sizeof(T));

        // Plausibility check (if payload supports it)
        if constexpr (has_is_plausible<T>::value) {
            if (!payload.is_plausible()) {
                return {E2EStatus::OUT_OF_RANGE, payload};
            }
        }

        // Sequence check
        E2EStatus status = E2EStatus::OK;
        if (first_message_) {
            first_message_ = false;
            rx_expected_ = static_cast<uint16_t>(header.counter + 1);
        } else {
            int32_t delta = static_cast<int32_t>(
                static_cast<int16_t>(header.counter - rx_expected_));
            if (delta == 0) {
                // Perfect
                rx_expected_ = static_cast<uint16_t>(header.counter + 1);
            } else if (delta > 0 &&
                       delta <= static_cast<int32_t>(MAX_DELTA)) {
                // Lost some, but within tolerance
                rx_expected_ = static_cast<uint16_t>(header.counter + 1);
                status = E2EStatus::LOST;
            } else if (delta < 0) {
                return {E2EStatus::REPEATED, payload};
            } else {
                // Too many lost — treat as LOST
                rx_expected_ = static_cast<uint16_t>(header.counter + 1);
                status = E2EStatus::LOST;
            }
        }

        return {status, payload};
    }
};

// ============================================================================
// Self-Test
// ============================================================================
void test_basic_protect_and_check() {
    std::cout << "--- Test: Basic protect and check ---\n";

    constexpr uint32_t DATA_ID = 0x1234;
    E2EProtector protector(DATA_ID);
    E2EChecker checker(DATA_ID);

    VehicleSpeed msg{.speed_mps = 13.5f, .acceleration = 0.2f, .quality = 2};
    auto buffer = protector.protect(msg);

    auto [status, received] = checker.check<VehicleSpeed>(buffer);
    assert(status == E2EStatus::OK);
    assert(received.speed_mps == 13.5f);
    assert(received.acceleration == 0.2f);
    assert(received.quality == 2);
    std::cout << "  PASS: Valid message accepted\n";
}

void test_crc_corruption() {
    std::cout << "--- Test: CRC corruption detection ---\n";

    constexpr uint32_t DATA_ID = 0x1234;
    E2EProtector protector(DATA_ID);
    E2EChecker checker(DATA_ID);

    VehicleSpeed msg{.speed_mps = 5.0f, .acceleration = 0.0f, .quality = 2};
    auto buffer = protector.protect(msg);

    // Corrupt one byte in the payload
    buffer[sizeof(E2EHeader) + 1] ^= 0xFF;

    auto [status, received] = checker.check<VehicleSpeed>(buffer);
    assert(status == E2EStatus::WRONG_CRC);
    std::cout << "  PASS: Corrupted message rejected (WRONG_CRC)\n";
}

void test_sequence_in_order() {
    std::cout << "--- Test: Sequential messages ---\n";

    constexpr uint32_t DATA_ID = 0xABCD;
    E2EProtector protector(DATA_ID);
    E2EChecker checker(DATA_ID);

    for (int i = 0; i < 10; ++i) {
        VehicleSpeed msg{.speed_mps = static_cast<float>(i),
                          .acceleration = 0.0f, .quality = 2};
        auto buffer = protector.protect(msg);
        auto [status, received] = checker.check<VehicleSpeed>(buffer);
        assert(status == E2EStatus::OK);
    }
    std::cout << "  PASS: 10 sequential messages accepted\n";
}

void test_message_loss() {
    std::cout << "--- Test: Message loss detection ---\n";

    constexpr uint32_t DATA_ID = 0x5678;
    E2EProtector protector(DATA_ID);
    E2EChecker checker(DATA_ID);

    // Send message 0
    VehicleSpeed msg0{.speed_mps = 0.0f, .acceleration = 0.0f, .quality = 2};
    auto buf0 = protector.protect(msg0);
    auto [s0, _r0] = checker.check<VehicleSpeed>(buf0);
    assert(s0 == E2EStatus::OK);

    // Send message 1 (skip it — don't give to checker)
    VehicleSpeed msg1{.speed_mps = 1.0f, .acceleration = 0.0f, .quality = 2};
    protector.protect(msg1);  // consumed by protector, not sent to checker

    // Send message 2
    VehicleSpeed msg2{.speed_mps = 2.0f, .acceleration = 0.0f, .quality = 2};
    auto buf2 = protector.protect(msg2);
    auto [s2, r2] = checker.check<VehicleSpeed>(buf2);
    assert(s2 == E2EStatus::LOST);
    assert(r2.speed_mps == 2.0f);
    std::cout << "  PASS: Lost message detected (LOST), data still usable\n";
}

void test_message_repetition() {
    std::cout << "--- Test: Message repetition detection ---\n";

    constexpr uint32_t DATA_ID = 0x9999;
    E2EProtector protector(DATA_ID);
    E2EChecker checker(DATA_ID);

    VehicleSpeed msg{.speed_mps = 10.0f, .acceleration = 0.0f, .quality = 2};
    auto buffer = protector.protect(msg);

    // First delivery: OK
    auto [s1, _r1] = checker.check<VehicleSpeed>(buffer);
    assert(s1 == E2EStatus::OK);

    // Replay the same buffer: REPEATED
    auto [s2, _r2] = checker.check<VehicleSpeed>(buffer);
    assert(s2 == E2EStatus::REPEATED);
    std::cout << "  PASS: Repeated message detected\n";
}

void test_plausibility_rejection() {
    std::cout << "--- Test: Plausibility rejection ---\n";

    constexpr uint32_t DATA_ID = 0x4444;
    E2EProtector protector(DATA_ID);
    E2EChecker checker(DATA_ID);

    // Speed of 200 m/s is physically impossible (720 km/h)
    VehicleSpeed msg{.speed_mps = 200.0f, .acceleration = 0.0f, .quality = 2};
    auto buffer = protector.protect(msg);

    auto [status, _r] = checker.check<VehicleSpeed>(buffer);
    assert(status == E2EStatus::OUT_OF_RANGE);
    std::cout << "  PASS: Implausible speed rejected\n";
}

void test_wrong_data_id() {
    std::cout << "--- Test: Wrong data ID ---\n";

    E2EProtector protector(0x1111);
    E2EChecker checker(0x2222);  // different data ID

    VehicleSpeed msg{.speed_mps = 5.0f, .acceleration = 0.0f, .quality = 2};
    auto buffer = protector.protect(msg);

    auto [status, _r] = checker.check<VehicleSpeed>(buffer);
    assert(status == E2EStatus::WRONG_DATA_ID);
    std::cout << "  PASS: Wrong data ID rejected\n";
}

// ============================================================================
int main() {
    std::cout << "=== E2E Protection Exercise ===\n\n";

    test_basic_protect_and_check();
    test_crc_corruption();
    test_sequence_in_order();
    test_message_loss();
    test_message_repetition();
    test_plausibility_rejection();
    test_wrong_data_id();

    std::cout << "\n=== ALL E2E TESTS PASSED ===\n";
    return 0;
}
