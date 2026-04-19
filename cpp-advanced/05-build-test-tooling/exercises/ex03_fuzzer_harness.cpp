// ============================================================================
// ex03_fuzzer_harness.cpp — Fuzzer Harness for a Binary Message Parser
//
// Message format:
//   [0..1]  magic:     0xCA 0xFE (2 bytes)
//   [2..3]  length:    uint16_t little-endian, length of payload
//   [4]     msg_type:  uint8_t (0x01=sensor, 0x02=command, 0x03=heartbeat)
//   [5..5+length-1]  payload data
//   [5+length]  checksum: XOR of all payload bytes
//
// Build modes:
//
//   1. libFuzzer (requires clang):
//      clang++ -std=c++20 -O1 -g -fsanitize=fuzzer,address \
//              ex03_fuzzer_harness.cpp -o fuzz_parser
//      mkdir corpus && ./fuzz_parser corpus/ -max_total_time=60
//
//   2. Standalone test (any compiler):
//      g++ -std=c++20 -DSTANDALONE_TEST ex03_fuzzer_harness.cpp -o test_parser
//      ./test_parser
//
// What fuzzing typically finds in parsers like this:
//   - Out-of-bounds read when length field > actual data
//   - Integer overflow in offset calculations (length + header_size)
//   - Logic errors when payload is zero-length
//   - Checksum validation bypass with crafted inputs
//   - Crashes on truncated headers (data < 5 bytes)
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>
#include <cassert>

// ---------------------------------------------------------------------------
// Message Types
// ---------------------------------------------------------------------------
enum class MsgType : uint8_t {
    Sensor    = 0x01,
    Command   = 0x02,
    Heartbeat = 0x03,
};

struct ParsedMessage {
    MsgType type;
    std::vector<uint8_t> payload;
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr uint8_t kMagic0 = 0xCA;
static constexpr uint8_t kMagic1 = 0xFE;
static constexpr size_t kHeaderSize = 5;   // magic(2) + length(2) + type(1)
static constexpr size_t kChecksumSize = 1;
static constexpr uint16_t kMaxPayloadLen = 4096;

// ---------------------------------------------------------------------------
// parse_message — Validates and extracts a message from raw bytes.
//
// Returns std::nullopt on any validation failure.
// ---------------------------------------------------------------------------
std::optional<ParsedMessage> parse_message(const uint8_t* data, size_t size) {
    // 1. Minimum size check: header + at least checksum
    if (size < kHeaderSize + kChecksumSize) {
        return std::nullopt;
    }

    // 2. Magic number validation
    if (data[0] != kMagic0 || data[1] != kMagic1) {
        return std::nullopt;
    }

    // 3. Extract payload length (little-endian uint16_t)
    uint16_t payload_len = static_cast<uint16_t>(data[2])
                         | (static_cast<uint16_t>(data[3]) << 8);

    // 4. Sanity check on payload length
    if (payload_len > kMaxPayloadLen) {
        return std::nullopt;
    }

    // 5. Verify total message fits in buffer
    //    BUG-PRONE AREA: integer overflow if payload_len + kHeaderSize wraps.
    //    Using size_t arithmetic to prevent overflow.
    size_t total_needed = static_cast<size_t>(kHeaderSize)
                        + static_cast<size_t>(payload_len)
                        + kChecksumSize;

    if (size < total_needed) {
        return std::nullopt;
    }

    // 6. Validate message type
    uint8_t raw_type = data[4];
    if (raw_type < 0x01 || raw_type > 0x03) {
        return std::nullopt;
    }

    // 7. Extract payload
    const uint8_t* payload_start = data + kHeaderSize;
    std::vector<uint8_t> payload(payload_start, payload_start + payload_len);

    // 8. Validate checksum (XOR of all payload bytes)
    uint8_t computed_checksum = 0;
    for (uint16_t i = 0; i < payload_len; ++i) {
        computed_checksum ^= payload_start[i];
    }

    uint8_t expected_checksum = data[kHeaderSize + payload_len];
    if (computed_checksum != expected_checksum) {
        return std::nullopt;
    }

    return ParsedMessage{
        .type = static_cast<MsgType>(raw_type),
        .payload = std::move(payload),
    };
}

// ===========================================================================
// Fuzzer Entry Point
// ===========================================================================
#ifndef STANDALONE_TEST

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // The fuzzer feeds random byte sequences here.
    // parse_message should NEVER crash, regardless of input.
    // If it does, the fuzzer saves the crashing input as a reproducer.

    auto result = parse_message(data, size);

    // Optional: do something lightweight with the result to prevent
    // the compiler from optimizing the call away.
    if (result.has_value()) {
        // Verify internal consistency: payload size matches what was parsed
        assert(result->payload.size() <= kMaxPayloadLen);
    }

    return 0;  // Always return 0 — non-zero means "reject this input"
}

#else
// ===========================================================================
// Standalone Test Mode — Run a few known inputs to verify basic correctness
// ===========================================================================

#include <cstdio>
#include <cstdlib>

static void test_valid_message() {
    // Type: Sensor (0x01), Payload: {0x42, 0x43}, Checksum: 0x42^0x43 = 0x01
    uint8_t msg[] = {
        0xCA, 0xFE,         // magic
        0x02, 0x00,         // length = 2 (LE)
        0x01,               // type = Sensor
        0x42, 0x43,         // payload
        0x01                // checksum: 0x42 ^ 0x43
    };
    auto result = parse_message(msg, sizeof(msg));
    assert(result.has_value());
    assert(result->type == MsgType::Sensor);
    assert(result->payload.size() == 2);
    assert(result->payload[0] == 0x42);
    printf("[PASS] test_valid_message\n");
}

static void test_bad_magic() {
    uint8_t msg[] = {0xDE, 0xAD, 0x00, 0x00, 0x01, 0x00};
    assert(!parse_message(msg, sizeof(msg)).has_value());
    printf("[PASS] test_bad_magic\n");
}

static void test_truncated() {
    uint8_t msg[] = {0xCA, 0xFE, 0x0A};  // only 3 bytes, too short
    assert(!parse_message(msg, sizeof(msg)).has_value());
    printf("[PASS] test_truncated\n");
}

static void test_length_exceeds_buffer() {
    // Claims 100 bytes of payload but only provides 2
    uint8_t msg[] = {0xCA, 0xFE, 0x64, 0x00, 0x01, 0x42, 0x43, 0x01};
    assert(!parse_message(msg, sizeof(msg)).has_value());
    printf("[PASS] test_length_exceeds_buffer\n");
}

static void test_bad_checksum() {
    uint8_t msg[] = {0xCA, 0xFE, 0x02, 0x00, 0x01, 0x42, 0x43, 0xFF};
    assert(!parse_message(msg, sizeof(msg)).has_value());
    printf("[PASS] test_bad_checksum\n");
}

static void test_empty_payload() {
    // Length = 0, type = Heartbeat, checksum = 0 (XOR of nothing)
    uint8_t msg[] = {0xCA, 0xFE, 0x00, 0x00, 0x03, 0x00};
    auto result = parse_message(msg, sizeof(msg));
    assert(result.has_value());
    assert(result->type == MsgType::Heartbeat);
    assert(result->payload.empty());
    printf("[PASS] test_empty_payload\n");
}

static void test_invalid_type() {
    uint8_t msg[] = {0xCA, 0xFE, 0x00, 0x00, 0x99, 0x00};
    assert(!parse_message(msg, sizeof(msg)).has_value());
    printf("[PASS] test_invalid_type\n");
}

static void test_max_payload_length_exceeded() {
    // Payload length = 0xFFFF (65535) > kMaxPayloadLen (4096)
    uint8_t msg[] = {0xCA, 0xFE, 0xFF, 0xFF, 0x01, 0x00};
    assert(!parse_message(msg, sizeof(msg)).has_value());
    printf("[PASS] test_max_payload_length_exceeded\n");
}

int main() {
    printf("=== ex03: Message Parser Tests ===\n");
    test_valid_message();
    test_bad_magic();
    test_truncated();
    test_length_exceeds_buffer();
    test_bad_checksum();
    test_empty_payload();
    test_invalid_type();
    test_max_payload_length_exceeded();
    printf("\nAll %d tests passed.\n", 8);
    return 0;
}

#endif  // STANDALONE_TEST
