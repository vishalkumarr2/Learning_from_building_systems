// Module 17: SDV & Mission-Critical Systems
// Puzzle 01: CRC Endianness Trap
//
// This E2E protection implementation has a subtle bug that causes CRC
// mismatches between big-endian (PowerPC) and little-endian (ARM) ECUs.
//
// The challenge:
//   1. Find the endianness-dependent bug
//   2. Fix it so CRC is portable across architectures
//   3. Explain why AUTOSAR E2E Profile 4 specifies byte order explicitly
//
// Build:
//   g++ -std=c++2a -Wall -Wextra -o puzzle01 puzzle01_crc_endian_trap.cpp
//
// Context:
//   In real vehicles, different ECUs run different processors (Cortex-R,
//   Cortex-A, PowerPC, TriCore). Messages cross endianness boundaries
//   constantly. A CRC that depends on host byte order WILL fail silently
//   on the other end, letting corrupted data through.

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>

// "Simple" CRC-32 (Castagnoli polynomial)
uint32_t crc32c(uint8_t const* data, size_t len) {
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int j = 0; j < 8; ++j) {
            crc = (crc >> 1) ^ (0x82F63B78 * (crc & 1));
        }
    }
    return crc ^ 0xFFFFFFFF;
}

// A sensor reading message
struct SensorMessage {
    uint16_t sensor_id;   // 2 bytes
    uint32_t timestamp;   // 4 bytes
    float    value;       // 4 bytes
    uint16_t counter;     // 2 bytes
};

// ============== BUG IS SOMEWHERE IN THIS FUNCTION ==============
uint32_t compute_message_crc(SensorMessage const& msg) {
    // "Just hash the struct bytes" approach
    uint8_t const* raw = reinterpret_cast<uint8_t const*>(&msg);
    return crc32c(raw, sizeof(msg));
}
// ================================================================

// Simulate what the "other ECU" computes
// (same struct, same function — but on a different-endian machine)
uint32_t simulate_other_ecu_crc(uint16_t sensor_id, uint32_t timestamp,
                                  float value, uint16_t counter) {
    // This simulates byte-swapped fields (as if big-endian)
    // On a little-endian host, this produces different bytes
    SensorMessage msg;
    msg.sensor_id = __builtin_bswap16(sensor_id);
    msg.timestamp = __builtin_bswap32(timestamp);
    uint32_t vi;
    std::memcpy(&vi, &value, 4);
    vi = __builtin_bswap32(vi);
    std::memcpy(&msg.value, &vi, 4);
    msg.counter = __builtin_bswap16(counter);

    return compute_message_crc(msg);
}

int main() {
    std::cout << "=== Puzzle: CRC Endianness Trap ===\n\n";

    SensorMessage msg{};
    msg.sensor_id = 42;
    msg.timestamp = 1000000;
    msg.value = 3.14f;
    msg.counter = 7;

    uint32_t our_crc = compute_message_crc(msg);
    uint32_t their_crc = simulate_other_ecu_crc(
        msg.sensor_id, msg.timestamp, msg.value, msg.counter);

    std::cout << "Our CRC:   0x" << std::hex << our_crc << "\n";
    std::cout << "Their CRC: 0x" << std::hex << their_crc << "\n";

    if (our_crc == their_crc) {
        std::cout << "\nCRCs match — portable! PASS\n";
    } else {
        std::cout << "\nCRCs DON'T match — ENDIANNESS BUG!\n";
        std::cout << "\nHINT: The fix involves serializing fields in a\n"
                  << "defined byte order BEFORE computing the CRC.\n"
                  << "AUTOSAR E2E Profile 4 specifies network byte order\n"
                  << "(big-endian) for CRC input.\n";
    }

    // PUZZLE: Fix compute_message_crc() so that both sides agree.
    // The correct approach: serialize each field to a byte buffer in
    // a fixed (e.g., big-endian) order, then CRC that buffer.
    // DO NOT CRC the raw struct — padding and byte order differ.

    return 0;
}
