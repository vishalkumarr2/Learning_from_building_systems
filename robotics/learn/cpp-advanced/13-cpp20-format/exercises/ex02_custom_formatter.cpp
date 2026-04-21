// Module 13 — Exercise 02: Custom Formatters
// Compiler: GCC 13+ or Clang 17+ with -std=c++20
//
// Demonstrates:
//   - Writing std::formatter<T> specialisations
//   - Custom format specifications via parse()
//   - Vector3d formatter with 'f' (full) and 's' (short) modes
//   - Timestamp formatter with ISO 8601 output
//   - HexDump formatter for binary data (xxd-style)

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <format>
#include <iostream>
#include <string>
#include <vector>

// =============================================================================
// Type 1: Vector3d — common in robotics for positions, velocities, forces
// =============================================================================

struct Vector3d {
    double x, y, z;
};

// Custom formatter with two modes:
//   {:f} or {} — full:  "Vector3d(x=1.2345, y=2.3456, z=3.4567)"
//   {:s}       — short: "(1.2, 2.3, 3.5)"
//   {:.Nf}     — full with N decimal places
//   {:.Ns}     — short with N decimal places
template <>
struct std::formatter<Vector3d> {
    char mode = 'f';     // 'f' = full (default), 's' = short
    int precision = -1;  // -1 = default (4 for full, 1 for short)

    // parse() reads our custom format spec from the format string.
    // The parse context points to everything between ':' and '}'.
    // We must return an iterator pointing at '}' (or end).
    constexpr auto parse(std::format_parse_context& ctx) {
        auto it = ctx.begin();
        auto end = ctx.end();

        // Check for precision: ".N"
        if (it != end && *it == '.') {
            ++it;
            precision = 0;
            while (it != end && *it >= '0' && *it <= '9') {
                precision = precision * 10 + (*it - '0');
                ++it;
            }
        }

        // Check for mode: 'f' or 's'
        if (it != end && (*it == 'f' || *it == 's')) {
            mode = *it;
            ++it;
        }

        // Must end at '}'
        if (it != end && *it != '}') {
            throw std::format_error("Invalid Vector3d format spec");
        }
        return it;
    }

    auto format(const Vector3d& v, std::format_context& ctx) const {
        int prec = precision;
        if (prec < 0) {
            prec = (mode == 's') ? 1 : 4;
        }

        if (mode == 's') {
            // Short form: "(x, y, z)"
            return std::format_to(ctx.out(), "({:.{}f}, {:.{}f}, {:.{}f})",
                                  v.x, prec, v.y, prec, v.z, prec);
        }
        // Full form: "Vector3d(x=..., y=..., z=...)"
        return std::format_to(ctx.out(),
                              "Vector3d(x={:.{}f}, y={:.{}f}, z={:.{}f})",
                              v.x, prec, v.y, prec, v.z, prec);
    }
};

// =============================================================================
// Type 2: Timestamp — used for logging, RCA timelines
// =============================================================================

struct Timestamp {
    int year, month, day;
    int hour, minute, second;
    int millisecond;
};

// Formatter modes:
//   {} or {:i} — ISO 8601: "2026-04-21T10:49:49.118Z"
//   {:h}       — human:    "2026-04-21 10:49:49.118"
//   {:c}       — compact:  "20260421T104949"
template <>
struct std::formatter<Timestamp> {
    char mode = 'i';  // 'i' = ISO 8601, 'h' = human, 'c' = compact

    constexpr auto parse(std::format_parse_context& ctx) {
        auto it = ctx.begin();
        if (it != ctx.end() && (*it == 'i' || *it == 'h' || *it == 'c')) {
            mode = *it;
            ++it;
        }
        if (it != ctx.end() && *it != '}') {
            throw std::format_error("Invalid Timestamp format spec");
        }
        return it;
    }

    auto format(const Timestamp& ts, std::format_context& ctx) const {
        switch (mode) {
            case 'h':
                return std::format_to(
                    ctx.out(),
                    "{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}.{:03d}",
                    ts.year, ts.month, ts.day,
                    ts.hour, ts.minute, ts.second, ts.millisecond);
            case 'c':
                return std::format_to(
                    ctx.out(),
                    "{:04d}{:02d}{:02d}T{:02d}{:02d}{:02d}",
                    ts.year, ts.month, ts.day,
                    ts.hour, ts.minute, ts.second);
            default:  // 'i' — ISO 8601
                return std::format_to(
                    ctx.out(),
                    "{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}.{:03d}Z",
                    ts.year, ts.month, ts.day,
                    ts.hour, ts.minute, ts.second, ts.millisecond);
        }
    }
};

// =============================================================================
// Type 3: HexDump — for binary data inspection (CAN frames, SPI buffers, etc.)
// =============================================================================

struct HexDump {
    const uint8_t* data;
    size_t size;

    // Convenience constructor from vector
    HexDump(const std::vector<uint8_t>& vec)
        : data(vec.data()), size(vec.size()) {}

    // From raw pointer + size
    HexDump(const uint8_t* d, size_t s) : data(d), size(s) {}

    // From any contiguous range of bytes
    template <size_t N>
    HexDump(const std::array<uint8_t, N>& arr)
        : data(arr.data()), size(N) {}
};

// Formatter modes:
//   {} or {:x} — xxd-style:  "00000000: DE AD BE EF 01 02  ......\n"
//   {:i}       — inline:     "DE AD BE EF 01 02"
//   {:r}       — raw:        "DEADBEEF0102"
template <>
struct std::formatter<HexDump> {
    char mode = 'x';  // 'x' = xxd, 'i' = inline, 'r' = raw

    constexpr auto parse(std::format_parse_context& ctx) {
        auto it = ctx.begin();
        if (it != ctx.end() && (*it == 'x' || *it == 'i' || *it == 'r')) {
            mode = *it;
            ++it;
        }
        if (it != ctx.end() && *it != '}') {
            throw std::format_error("Invalid HexDump format spec");
        }
        return it;
    }

    auto format(const HexDump& hd, std::format_context& ctx) const {
        auto out = ctx.out();

        if (mode == 'r') {
            // Raw: "DEADBEEF"
            for (size_t i = 0; i < hd.size; ++i) {
                out = std::format_to(out, "{:02X}", hd.data[i]);
            }
            return out;
        }

        if (mode == 'i') {
            // Inline: "DE AD BE EF"
            for (size_t i = 0; i < hd.size; ++i) {
                if (i > 0) out = std::format_to(out, " ");
                out = std::format_to(out, "{:02X}", hd.data[i]);
            }
            return out;
        }

        // xxd-style: offset, hex, ASCII
        constexpr size_t BYTES_PER_LINE = 16;
        for (size_t offset = 0; offset < hd.size; offset += BYTES_PER_LINE) {
            // Offset
            out = std::format_to(out, "{:08x}: ", offset);

            // Hex bytes
            size_t line_len = std::min(BYTES_PER_LINE, hd.size - offset);
            for (size_t j = 0; j < BYTES_PER_LINE; ++j) {
                if (j < line_len) {
                    out = std::format_to(out, "{:02X} ", hd.data[offset + j]);
                } else {
                    out = std::format_to(out, "   ");  // padding
                }
                if (j == 7) out = std::format_to(out, " ");  // mid-line gap
            }

            // ASCII representation
            out = std::format_to(out, " |");
            for (size_t j = 0; j < line_len; ++j) {
                uint8_t ch = hd.data[offset + j];
                out = std::format_to(out, "{:c}",
                                     (ch >= 0x20 && ch < 0x7F) ? static_cast<char>(ch) : '.');
            }
            out = std::format_to(out, "|\n");
        }
        return out;
    }
};

// =============================================================================
// Main — demonstrate all custom formatters
// =============================================================================

int main() {
    std::cout << "=== Custom std::formatter Demonstrations ===\n\n";

    // --- Vector3d ---
    std::cout << "--- Vector3d formatter ---\n";
    Vector3d position{1.23456, -3.45678, 0.001};

    std::cout << std::format("Default (full):   {}\n", position);
    std::cout << std::format("Explicit full:    {:f}\n", position);
    std::cout << std::format("Short:            {:s}\n", position);
    std::cout << std::format("Full .2:          {:.2f}\n", position);
    std::cout << std::format("Short .3:         {:.3s}\n", position);

    // Practical: multiple vectors in a log line
    Vector3d velocity{0.5, -0.1, 0.0};
    std::cout << std::format("\nRobot state: pos={:s} vel={:s}\n",
                             position, velocity);

    // --- Timestamp ---
    std::cout << "\n--- Timestamp formatter ---\n";
    Timestamp ts{2026, 4, 21, 10, 49, 49, 118};

    std::cout << std::format("ISO 8601: {}\n", ts);
    std::cout << std::format("ISO 8601: {:i}\n", ts);
    std::cout << std::format("Human:    {:h}\n", ts);
    std::cout << std::format("Compact:  {:c}\n", ts);

    // In a log line
    std::cout << std::format("\n[{:h}] Motor fault detected at pos={:.2f}\n",
                             ts, position);

    // --- HexDump ---
    std::cout << "\n--- HexDump formatter ---\n";

    // Simulate a CAN frame + some binary data
    std::vector<uint8_t> spi_buffer = {
        0x48, 0x65, 0x6C, 0x6C, 0x6F, 0x20, 0x57, 0x6F,  // "Hello Wo"
        0x72, 0x6C, 0x64, 0x21, 0x00, 0xFF, 0xDE, 0xAD,  // "rld!...."
        0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,  // "........"
    };
    HexDump dump(spi_buffer);

    std::cout << "xxd-style (default):\n" << std::format("{:x}", dump);
    std::cout << std::format("\nInline: {:i}\n", dump);
    std::cout << std::format("Raw:    {:r}\n", dump);

    // CAN frame example
    std::array<uint8_t, 8> can_frame = {0xDE, 0xAD, 0xBE, 0xEF,
                                         0x01, 0x02, 0x03, 0x04};
    HexDump can_dump(can_frame);
    std::cout << std::format("\nCAN 0x1A0 [8]: {:i}\n", can_dump);

    std::cout << "\nAll exercises complete.\n";
    return 0;
}
