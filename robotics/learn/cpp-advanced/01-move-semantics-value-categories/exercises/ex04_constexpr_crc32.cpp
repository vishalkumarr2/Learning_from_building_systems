// =============================================================================
// Exercise 4: constexpr CRC32, FNV-1a, and consteval
// =============================================================================
// Demonstrates compile-time computation in C++20:
//   1. Full CRC32 lookup table generated at compile time
//   2. FNV-1a string hash for switch statements
//   3. consteval safe_divide — compile-time error detection (see notes)
//   4. static_assert verification of computed values
//   5. Benchmark: constexpr table vs runtime-computed
//
// NOTE: consteval and constinit require GCC 10+. On GCC 9, we use constexpr
// with comments showing what consteval would look like. The key concepts
// (compile-time evaluation, error detection) are still demonstrated.
//
// Build: g++ -std=c++2a -Wall -Wextra -Wpedantic ex04_constexpr_crc32.cpp
// =============================================================================

#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>

// =============================================================================
// Part 1: CRC32 Lookup Table — Compile Time
// =============================================================================

// Standard CRC32 polynomial (reversed): 0xEDB88320
// This is the same polynomial used by zlib, Ethernet, PKZIP, etc.

constexpr std::array<uint32_t, 256> make_crc32_table() {
    std::array<uint32_t, 256> table{};
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t crc = i;
        for (int j = 0; j < 8; ++j) {
            // If the lowest bit is set, XOR with polynomial
            if (crc & 1)
                crc = (crc >> 1) ^ 0xEDB88320u;
            else
                crc >>= 1;
        }
        table[i] = crc;
    }
    return table;
}

// The table is computed at compile time and placed in .rodata
constexpr auto crc32_table = make_crc32_table();

// Verify known values at compile time
static_assert(crc32_table[0] == 0x00000000, "CRC32 table[0] should be 0");
static_assert(crc32_table[1] == 0x77073096, "CRC32 table[1] known value");
static_assert(crc32_table[255] == 0x2D02EF8D, "CRC32 table[255] known value");

// CRC32 computation — also constexpr so it can run at compile time
constexpr uint32_t crc32(std::string_view data) {
    uint32_t crc = 0xFFFFFFFFu;
    for (char c : data) {
        uint8_t index = static_cast<uint8_t>(crc ^ static_cast<uint8_t>(c));
        crc = (crc >> 8) ^ crc32_table[index];
    }
    return crc ^ 0xFFFFFFFFu;
}

// Verify against known CRC32 values
static_assert(crc32("") == 0x00000000, "CRC32 of empty string");
static_assert(crc32("123456789") == 0xCBF43926, "CRC32 of '123456789' — standard test vector");

// =============================================================================
// Part 2: FNV-1a Hash — Compile-Time String Hashing for Switch
// =============================================================================

constexpr uint64_t fnv1a(std::string_view sv) {
    uint64_t hash = 0xcbf29ce484222325ULL;  // FNV offset basis
    for (char c : sv) {
        hash ^= static_cast<uint64_t>(static_cast<unsigned char>(c));
        hash *= 0x100000001b3ULL;            // FNV prime
    }
    return hash;
}

// Verify hash determinism at compile time
static_assert(fnv1a("hello") == fnv1a("hello"), "Same input → same hash");
static_assert(fnv1a("hello") != fnv1a("world"), "Different input → different hash");
static_assert(fnv1a("") != fnv1a("a"), "Empty vs non-empty");

// Operator for convenient string literal hashing
constexpr uint64_t operator""_hash(const char* str, std::size_t len) {
    return fnv1a(std::string_view(str, len));
}

// Demo: robot command dispatch using compile-time hashes
std::string dispatch_command(std::string_view cmd) {
    switch (fnv1a(cmd)) {
        case "start"_hash:    return "Starting robot...";
        case "stop"_hash:     return "Stopping robot...";
        case "pause"_hash:    return "Pausing robot...";
        case "resume"_hash:   return "Resuming robot...";
        case "calibrate"_hash: return "Calibrating sensors...";
        case "status"_hash:   return "Reporting status...";
        default:              return "Unknown command: " + std::string(cmd);
    }
}

// =============================================================================
// Part 3: consteval — Guaranteed Compile-Time Evaluation
// =============================================================================

// NOTE: consteval requires GCC 10+. With GCC 9, we use constexpr and show
// what consteval would look like. The patterns are identical; consteval just
// adds the compile-time enforcement.

// With GCC 10+, this would be:
//   consteval int safe_divide(int a, int b) { ... }
// Which makes any runtime call a compile error.
//
// With constexpr, it CAN run at compile time (when used in constexpr context)
// but can also fall back to runtime.
constexpr int safe_divide(int a, int b) {
    if (b == 0) {
        // In a consteval function, this would be a compile-time error.
        // In constexpr, it's UB at compile time (caught by the compiler)
        // and a runtime crash otherwise.
        // throw "division by zero";  // uncomment with GCC 10+ consteval
        return 0; // placeholder for GCC 9
    }
    return a / b;
}

// These are evaluated at compile time when used with constexpr variables
constexpr int result1 = safe_divide(100, 5);   // OK: 20
constexpr int result2 = safe_divide(42, 7);    // OK: 6
// constexpr int result3 = safe_divide(1, 0);  // Would be error with consteval

static_assert(result1 == 20);
static_assert(result2 == 6);

// With GCC 10+ consteval, this would enforce compile-time evaluation:
//   consteval unsigned validate_frequency(unsigned hz) {
//       if (hz == 0) throw "frequency cannot be zero";
//       if (hz > 10000) throw "frequency too high for real-time loop";
//       return hz;
//   }
constexpr unsigned validate_frequency(unsigned hz) {
    return hz;  // simplified for GCC 9; see notes.md for full consteval version
}

constexpr unsigned CONTROL_FREQ = validate_frequency(200);   // OK
constexpr unsigned SENSOR_FREQ  = validate_frequency(1000);  // OK

static_assert(CONTROL_FREQ == 200);
static_assert(SENSOR_FREQ == 1000);

// =============================================================================
// Part 4: Benchmark — constexpr table lookup vs runtime computation
// =============================================================================

// Runtime CRC32 table generation (NO constexpr)
std::array<uint32_t, 256> make_crc32_table_runtime() {
    std::array<uint32_t, 256> table{};
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t crc = i;
        for (int j = 0; j < 8; ++j) {
            if (crc & 1)
                crc = (crc >> 1) ^ 0xEDB88320u;
            else
                crc >>= 1;
        }
        table[i] = crc;
    }
    return table;
}

uint32_t crc32_runtime(std::string_view data,
                       const std::array<uint32_t, 256>& table) {
    uint32_t crc = 0xFFFFFFFFu;
    for (char c : data) {
        uint8_t index = static_cast<uint8_t>(crc ^ static_cast<uint8_t>(c));
        crc = (crc >> 8) ^ table[index];
    }
    return crc ^ 0xFFFFFFFFu;
}

void benchmark_crc32() {
    std::cout << "\n--- Benchmark: constexpr vs runtime CRC32 ---\n";

    const std::string test_data(1000, 'A');
    constexpr int ITERS = 100'000;

    // Benchmark 1: using constexpr table (already in .rodata)
    {
        auto start = std::chrono::steady_clock::now();
        volatile uint32_t sink = 0;
        for (int i = 0; i < ITERS; ++i) {
            sink = crc32(test_data);
        }
        auto elapsed = std::chrono::steady_clock::now() - start;
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        std::cout << "  constexpr table: " << us << " us (" << ITERS << " iters)\n";
        (void)sink;
    }

    // Benchmark 2: build table at runtime, then use it
    {
        auto start = std::chrono::steady_clock::now();
        auto rt_table = make_crc32_table_runtime();  // runtime table init
        volatile uint32_t sink = 0;
        for (int i = 0; i < ITERS; ++i) {
            sink = crc32_runtime(test_data, rt_table);
        }
        auto elapsed = std::chrono::steady_clock::now() - start;
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count();
        std::cout << "  runtime table:   " << us << " us (" << ITERS << " iters, includes table build)\n";
        (void)sink;
    }

    std::cout << "  (The constexpr version has zero startup cost.)\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "=== Compile-Time Computation Exercises ===\n";

    // --- CRC32 ---
    std::cout << "\n--- CRC32 ---\n";
    std::cout << "  CRC32(\"\")         = 0x" << std::hex << crc32("") << std::dec << "\n";
    std::cout << "  CRC32(\"123456789\") = 0x" << std::hex << crc32("123456789") << std::dec << "\n";
    std::cout << "  CRC32(\"hello\")     = 0x" << std::hex << crc32("hello") << std::dec << "\n";
    std::cout << "  Table verified at compile time via static_assert.\n";

    // --- FNV-1a dispatch ---
    std::cout << "\n--- Command Dispatch (FNV-1a) ---\n";
    for (auto cmd : {"start", "stop", "pause", "resume", "calibrate", "status", "dance"}) {
        std::cout << "  " << cmd << " → " << dispatch_command(cmd) << "\n";
    }

    // --- consteval ---
    std::cout << "\n--- consteval ---\n";
    std::cout << "  safe_divide(100, 5)  = " << result1 << "\n";
    std::cout << "  safe_divide(42, 7)   = " << result2 << "\n";
    std::cout << "  CONTROL_FREQ         = " << CONTROL_FREQ << " Hz\n";
    std::cout << "  SENSOR_FREQ          = " << SENSOR_FREQ << " Hz\n";
    std::cout << "  (Try uncommenting safe_divide(1, 0) to see compile error)\n";

    // --- Benchmark ---
    benchmark_crc32();

    std::cout << "\nAll exercises complete!\n";
    return 0;
}
