// =============================================================================
// Exercise 07: SecOC Authenticator
// =============================================================================
// Implement Secure Onboard Communication (SecOC) as defined in AUTOSAR.
//
// Features:
//   1. CMAC authentication using a simplified HMAC construction
//   2. Freshness counter to prevent replay attacks
//   3. Truncated MAC for bandwidth-constrained CAN/Ethernet
//   4. Verification with freshness window tolerance
//   5. Key management (per-ECU keys)
//
// SecOC protects vehicle bus messages against:
//   - Replay attacks (stale messages re-injected)
//   - Spoofing (forged messages from compromised nodes)
//   - Tampering (modified in-transit)
//
// Production relevance:
//   - UN Regulation 155 mandates cybersecurity management system
//   - AUTOSAR SecOC is the standard for in-vehicle message authentication
//   - Used on CAN, CAN-FD, FlexRay, and Automotive Ethernet
// =============================================================================

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

// ============================================================================
// Simplified CMAC (HMAC-like construction for educational purposes)
// ============================================================================
// Production would use AES-128-CMAC. Here we use a simplified hash-based MAC.
// Key insight: the STRUCTURE is what matters for learning SecOC.
namespace crypto {

// Simple hash function (SipHash-like, NOT cryptographically secure — for learning)
constexpr uint64_t mix(uint64_t h, uint8_t byte) {
    h ^= byte;
    h *= 0x9E3779B97F4A7C15ULL;  // golden ratio constant
    h ^= h >> 32;
    return h;
}

// Compute a MAC over (key + freshness + data)
std::array<uint8_t, 8> compute_mac(
        std::array<uint8_t, 16> const& key,
        uint32_t freshness_value,
        uint8_t const* data,
        size_t data_len) {
    uint64_t h = 0;

    // Mix in key
    for (auto k : key) {
        h = mix(h, k);
    }

    // Mix in freshness value (big-endian)
    h = mix(h, static_cast<uint8_t>((freshness_value >> 24) & 0xFF));
    h = mix(h, static_cast<uint8_t>((freshness_value >> 16) & 0xFF));
    h = mix(h, static_cast<uint8_t>((freshness_value >> 8) & 0xFF));
    h = mix(h, static_cast<uint8_t>(freshness_value & 0xFF));

    // Mix in data
    for (size_t i = 0; i < data_len; ++i) {
        h = mix(h, data[i]);
    }

    // Output as 8 bytes
    std::array<uint8_t, 8> mac{};
    for (size_t i = 0; i < 8; ++i) {
        mac[i] = static_cast<uint8_t>((h >> (i * 8)) & 0xFF);
    }
    return mac;
}

} // namespace crypto

// ============================================================================
// SecOC Secured Message
// ============================================================================
struct SecuredMessage {
    uint16_t pdu_id = 0;                  // Protocol Data Unit identifier
    std::vector<uint8_t> payload;          // original data
    uint32_t freshness_value = 0;          // counter/timestamp
    std::array<uint8_t, 4> truncated_mac;  // truncated to 4 bytes (32 bits)
};

// ============================================================================
// SecOC Authenticator (one per ECU / communication partner)
// ============================================================================
class SecOcAuthenticator {
    std::array<uint8_t, 16> key_{};     // shared secret key
    uint32_t tx_freshness_ = 0;         // transmit counter
    uint32_t rx_freshness_ = 0;         // last accepted receive counter
    uint16_t ecu_id_ = 0;

    static constexpr uint32_t FRESHNESS_WINDOW = 10;
    // Allow messages up to FRESHNESS_WINDOW ahead of last accepted
    // (handles out-of-order delivery within a small window)

    static constexpr size_t TRUNCATED_MAC_LEN = 4;

    std::array<uint8_t, 4> truncate_mac(
            std::array<uint8_t, 8> const& full_mac) const {
        std::array<uint8_t, 4> t{};
        std::copy_n(full_mac.begin(), 4, t.begin());
        return t;
    }

public:
    SecOcAuthenticator(uint16_t ecu_id,
                        std::array<uint8_t, 16> const& key)
        : key_(key), ecu_id_(ecu_id) {}

    // --- Transmit side ---

    SecuredMessage authenticate(uint16_t pdu_id,
                                 std::vector<uint8_t> const& payload) {
        ++tx_freshness_;

        auto full_mac = crypto::compute_mac(
            key_, tx_freshness_, payload.data(), payload.size());

        SecuredMessage msg;
        msg.pdu_id = pdu_id;
        msg.payload = payload;
        msg.freshness_value = tx_freshness_;
        msg.truncated_mac = truncate_mac(full_mac);
        return msg;
    }

    // --- Receive side ---

    enum class VerifyResult : uint8_t {
        OK,
        MAC_MISMATCH,
        REPLAY_DETECTED,
        FRESHNESS_OUT_OF_WINDOW
    };

    static constexpr const char* to_string(VerifyResult r) {
        switch (r) {
            case VerifyResult::OK: return "OK";
            case VerifyResult::MAC_MISMATCH: return "MAC_MISMATCH";
            case VerifyResult::REPLAY_DETECTED: return "REPLAY_DETECTED";
            case VerifyResult::FRESHNESS_OUT_OF_WINDOW:
                return "FRESHNESS_OUT_OF_WINDOW";
        }
        return "UNKNOWN";
    }

    VerifyResult verify(SecuredMessage const& msg) {
        // Check freshness: must be greater than last accepted
        if (msg.freshness_value <= rx_freshness_) {
            return VerifyResult::REPLAY_DETECTED;
        }

        // Check freshness window (gap too large = suspicious)
        if (msg.freshness_value > rx_freshness_ + FRESHNESS_WINDOW) {
            return VerifyResult::FRESHNESS_OUT_OF_WINDOW;
        }

        // Recompute MAC
        auto full_mac = crypto::compute_mac(
            key_, msg.freshness_value,
            msg.payload.data(), msg.payload.size());
        auto expected = truncate_mac(full_mac);

        if (msg.truncated_mac != expected) {
            return VerifyResult::MAC_MISMATCH;
        }

        // Accept: advance freshness counter
        rx_freshness_ = msg.freshness_value;
        return VerifyResult::OK;
    }

    // Sync freshness counter (e.g., after key exchange)
    void sync_freshness(uint32_t value) {
        tx_freshness_ = value;
        rx_freshness_ = value;
    }

    uint32_t tx_counter() const { return tx_freshness_; }
    uint32_t rx_counter() const { return rx_freshness_; }
};

// ============================================================================
// Self-Test
// ============================================================================
std::array<uint8_t, 16> const TEST_KEY = {
    0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
    0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10
};

void test_basic_authenticate_verify() {
    std::cout << "--- Test: Basic authenticate → verify ---\n";

    SecOcAuthenticator sender(0x01, TEST_KEY);
    SecOcAuthenticator receiver(0x02, TEST_KEY);

    auto msg = sender.authenticate(0x100, {0xDE, 0xAD, 0xBE, 0xEF});
    auto result = receiver.verify(msg);

    assert(result == SecOcAuthenticator::VerifyResult::OK);
    std::cout << "  PASS\n";
}

void test_tampered_payload() {
    std::cout << "--- Test: Tampered payload detected ---\n";

    SecOcAuthenticator sender(0x01, TEST_KEY);
    SecOcAuthenticator receiver(0x02, TEST_KEY);

    auto msg = sender.authenticate(0x100, {0xDE, 0xAD, 0xBE, 0xEF});
    msg.payload[0] = 0xFF;  // tamper!

    auto result = receiver.verify(msg);
    assert(result == SecOcAuthenticator::VerifyResult::MAC_MISMATCH);
    std::cout << "  PASS: Tamper detected\n";
}

void test_replay_attack() {
    std::cout << "--- Test: Replay attack detected ---\n";

    SecOcAuthenticator sender(0x01, TEST_KEY);
    SecOcAuthenticator receiver(0x02, TEST_KEY);

    auto msg1 = sender.authenticate(0x100, {0x01});
    receiver.verify(msg1);  // accept

    // Replay: send same message again
    auto result = receiver.verify(msg1);
    assert(result == SecOcAuthenticator::VerifyResult::REPLAY_DETECTED);
    std::cout << "  PASS: Replay rejected\n";
}

void test_sequential_messages() {
    std::cout << "--- Test: Sequential messages accepted ---\n";

    SecOcAuthenticator sender(0x01, TEST_KEY);
    SecOcAuthenticator receiver(0x02, TEST_KEY);

    for (int i = 0; i < 10; ++i) {
        std::vector<uint8_t> data = {static_cast<uint8_t>(i)};
        auto msg = sender.authenticate(0x100, data);
        auto result = receiver.verify(msg);
        assert(result == SecOcAuthenticator::VerifyResult::OK);
    }

    assert(sender.tx_counter() == 10);
    assert(receiver.rx_counter() == 10);
    std::cout << "  PASS: 10 sequential messages accepted\n";
}

void test_freshness_window_exceeded() {
    std::cout << "--- Test: Freshness window exceeded ---\n";

    SecOcAuthenticator sender(0x01, TEST_KEY);
    SecOcAuthenticator receiver(0x02, TEST_KEY);

    // Send first message
    auto msg1 = sender.authenticate(0x100, {0x01});
    receiver.verify(msg1);

    // Skip ahead: send 20 messages without receiver seeing them
    for (int i = 0; i < 20; ++i) {
        sender.authenticate(0x100, {static_cast<uint8_t>(i)});
    }

    // This message's freshness is too far ahead
    auto msg_far = sender.authenticate(0x100, {0xFF});
    auto result = receiver.verify(msg_far);
    assert(result ==
           SecOcAuthenticator::VerifyResult::FRESHNESS_OUT_OF_WINDOW);
    std::cout << "  PASS: Out-of-window rejected\n";
}

void test_wrong_key_rejected() {
    std::cout << "--- Test: Wrong key rejected ---\n";

    SecOcAuthenticator sender(0x01, TEST_KEY);

    std::array<uint8_t, 16> wrong_key{};
    wrong_key.fill(0xFF);
    SecOcAuthenticator receiver(0x02, wrong_key);

    auto msg = sender.authenticate(0x100, {0xDE, 0xAD});
    auto result = receiver.verify(msg);
    assert(result == SecOcAuthenticator::VerifyResult::MAC_MISMATCH);
    std::cout << "  PASS: Wrong key → MAC mismatch\n";
}

void test_freshness_counter_monotonicity() {
    std::cout << "--- Test: Freshness counter monotonicity ---\n";

    SecOcAuthenticator sender(0x01, TEST_KEY);
    SecOcAuthenticator receiver(0x02, TEST_KEY);

    // Accept msg 1, 2, 3
    auto m1 = sender.authenticate(0x100, {0x01});
    auto m2 = sender.authenticate(0x100, {0x02});
    auto m3 = sender.authenticate(0x100, {0x03});

    assert(receiver.verify(m1) == SecOcAuthenticator::VerifyResult::OK);
    assert(receiver.verify(m2) == SecOcAuthenticator::VerifyResult::OK);
    assert(receiver.verify(m3) == SecOcAuthenticator::VerifyResult::OK);

    // Try to accept m2 again (out of order, freshness < last accepted)
    assert(receiver.verify(m2) ==
           SecOcAuthenticator::VerifyResult::REPLAY_DETECTED);
    std::cout << "  PASS: Out-of-order message rejected\n";
}

void test_different_pdu_ids() {
    std::cout << "--- Test: Different PDU IDs ---\n";

    SecOcAuthenticator sender(0x01, TEST_KEY);
    SecOcAuthenticator receiver(0x02, TEST_KEY);

    // Different PDU IDs, same payload
    auto msg_a = sender.authenticate(0x100, {0xAA});
    auto msg_b = sender.authenticate(0x200, {0xAA});

    // Both should verify OK (freshness is per-authenticator, not per-PDU)
    assert(receiver.verify(msg_a) == SecOcAuthenticator::VerifyResult::OK);
    assert(receiver.verify(msg_b) == SecOcAuthenticator::VerifyResult::OK);
    std::cout << "  PASS\n";
}

// ============================================================================
int main() {
    std::cout << "=== SecOC Authenticator Exercise ===\n\n";

    test_basic_authenticate_verify();
    test_tampered_payload();
    test_replay_attack();
    test_sequential_messages();
    test_freshness_window_exceeded();
    test_wrong_key_rejected();
    test_freshness_counter_monotonicity();
    test_different_pdu_ids();

    std::cout << "\n=== ALL SECOC TESTS PASSED ===\n";
    return 0;
}
