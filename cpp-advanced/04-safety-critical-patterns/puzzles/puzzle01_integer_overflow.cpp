// =============================================================================
// Puzzle 01: The Ariane 5 Bug — 64-bit to 16-bit Conversion Overflow
// =============================================================================
// On June 4, 1996, the Ariane 5 rocket exploded 37 seconds after launch.
// Cost: $370 million.
//
// Root cause: The Inertial Reference System (SRI) converted a 64-bit floating
// point horizontal velocity to a 16-bit signed integer. The value was valid
// for Ariane 4's trajectory (smaller, slower) but overflowed on Ariane 5's
// faster trajectory. The software was REUSED from Ariane 4 without re-analysis.
//
// The overflow raised an exception. The exception handler... did nothing
// useful. The SRI shut down. The backup SRI had the same bug. The rocket
// lost attitude control and self-destructed.
//
// This puzzle reproduces the exact scenario with realistic types.
// =============================================================================

#include <cassert>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>
#include <optional>

// The Ariane SRI computed "horizontal bias" (BH) from platform velocity.
// BH was a 64-bit float. It was then converted to a 16-bit signed integer
// for transmission to the flight computer.

// Simulate: convert a horizontal velocity value to 16-bit integer
// This is what the SRI code did (without range checking!)

namespace ariane {

// --- THE BUGGY CODE (what Ariane 5 actually had) ---
int16_t convert_velocity_buggy(double velocity) {
    // Direct cast: no range check, no saturation
    // On Ariane 4: velocity stayed in [-32768, 32767] range. Fine.
    // On Ariane 5: velocity exceeded 32767. BOOM.
    return static_cast<int16_t>(static_cast<int64_t>(velocity));
    // In Ada (which Ariane used), this raised Constraint_Error.
    // The handler just... shut down the SRI.
}

// --- THE FIX (what should have been done) ---
std::optional<int16_t> convert_velocity_safe(double velocity) {
    // Check range BEFORE conversion
    if (!std::isfinite(velocity)) return std::nullopt;
    if (velocity > static_cast<double>(INT16_MAX)) return std::nullopt;
    if (velocity < static_cast<double>(INT16_MIN)) return std::nullopt;
    return static_cast<int16_t>(velocity);
}

// Alternative fix: saturating conversion
int16_t convert_velocity_saturating(double velocity) {
    if (!std::isfinite(velocity)) return 0; // safe default
    if (velocity > static_cast<double>(INT16_MAX)) return INT16_MAX;
    if (velocity < static_cast<double>(INT16_MIN)) return INT16_MIN;
    return static_cast<int16_t>(velocity);
}

} // namespace ariane

// ======================== SIMULATION =========================================

struct TrajectoryPoint {
    double time_s;        // seconds after launch
    double velocity_mps;  // horizontal velocity in m/s
};

int main() {
    std::cout << "=== Puzzle 01: The Ariane 5 Bug ===\n\n";

    // Ariane 4 trajectory: gentle, stays within int16 range
    TrajectoryPoint ariane4_trajectory[] = {
        {0.0,   0.0},
        {10.0,  500.0},
        {20.0,  5000.0},
        {30.0,  15000.0},
        {37.0,  28000.0},    // still fits in int16 (max 32767)
    };

    // Ariane 5 trajectory: much more powerful, exceeds int16 range!
    TrajectoryPoint ariane5_trajectory[] = {
        {0.0,   0.0},
        {10.0,  2000.0},
        {20.0,  18000.0},
        {30.0,  33000.0},    // 33000 > 32767 — OVERFLOW!!
        {37.0,  42000.0},    // even worse
    };

    std::cout << "--- Ariane 4 Trajectory (fits in int16) ---\n";
    std::cout << std::setw(8) << "Time(s)" << std::setw(15) << "Velocity"
              << std::setw(12) << "int16" << std::setw(10) << "Status\n";

    for (const auto& pt : ariane4_trajectory) {
        auto result = ariane::convert_velocity_safe(pt.velocity_mps);
        std::cout << std::setw(8) << pt.time_s
                  << std::setw(15) << std::fixed << std::setprecision(1)
                  << pt.velocity_mps;
        if (result) {
            std::cout << std::setw(12) << *result << std::setw(10) << "OK";
        } else {
            std::cout << std::setw(12) << "OVERFLOW" << std::setw(10) << "FAIL";
        }
        std::cout << "\n";
    }

    std::cout << "\n--- Ariane 5 Trajectory (OVERFLOWS int16 at T+30s) ---\n";
    std::cout << std::setw(8) << "Time(s)" << std::setw(15) << "Velocity"
              << std::setw(12) << "int16" << std::setw(12) << "Buggy int16"
              << std::setw(10) << "Status\n";

    for (const auto& pt : ariane5_trajectory) {
        auto safe_result = ariane::convert_velocity_safe(pt.velocity_mps);
        int16_t buggy_result = ariane::convert_velocity_buggy(pt.velocity_mps);

        std::cout << std::setw(8) << pt.time_s
                  << std::setw(15) << std::fixed << std::setprecision(1)
                  << pt.velocity_mps;

        if (safe_result) {
            std::cout << std::setw(12) << *safe_result;
        } else {
            std::cout << std::setw(12) << "OVERFLOW";
        }

        std::cout << std::setw(12) << buggy_result;

        if (safe_result) {
            std::cout << std::setw(10) << "OK";
        } else {
            std::cout << std::setw(10) << "** BOOM **";
        }
        std::cout << "\n";
    }

    // Demonstrate the exact overflow point
    std::cout << "\n--- The Critical Value ---\n";
    std::cout << "INT16_MAX = " << INT16_MAX << "\n";
    std::cout << "Ariane 5 velocity at T+30s = 33000\n";
    std::cout << "33000 as int16 (buggy) = "
              << ariane::convert_velocity_buggy(33000.0) << "  <-- GARBAGE VALUE!\n";
    std::cout << "33000 as int16 (safe)  = ";
    auto r = ariane::convert_velocity_safe(33000.0);
    if (!r) std::cout << "rejected (overflow detected)\n";
    std::cout << "33000 as int16 (sat)   = "
              << ariane::convert_velocity_saturating(33000.0) << " (clamped to MAX)\n";

    // Verification
    assert(ariane::convert_velocity_safe(32767.0).has_value());
    assert(!ariane::convert_velocity_safe(32768.0).has_value());
    assert(!ariane::convert_velocity_safe(33000.0).has_value());
    assert(ariane::convert_velocity_saturating(99999.0) == INT16_MAX);
    assert(ariane::convert_velocity_saturating(-99999.0) == INT16_MIN);

    std::cout << "\n=== LESSON ===\n";
    std::cout << "1. NEVER assume data ranges from one system apply to another\n";
    std::cout << "2. ALWAYS check range before narrowing conversions\n";
    std::cout << "3. A $370M lesson in the cost of reusing code without re-analysis\n";

    return 0;
}
