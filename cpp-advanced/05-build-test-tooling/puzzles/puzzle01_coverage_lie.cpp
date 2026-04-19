// ============================================================================
// puzzle01_coverage_lie.cpp — 100% Line Coverage, Still Has a Bug
//
// This demonstrates why line coverage is insufficient. Every line of
// process_reading() is executed by the tests, but a specific INPUT
// COMBINATION triggers a crash that no test catches.
//
// YOUR TASK:
//   1. Verify that all lines of process_reading() are covered.
//   2. Find the bug.
//   3. Write the test case that exposes it.
//   4. Explain why MC/DC (Modified Condition/Decision Coverage) catches
//      it but line coverage does not.
//
// Build:
//   g++ -std=c++20 -O0 -g --coverage puzzle01_coverage_lie.cpp -o puzzle01
//   ./puzzle01
//   gcov puzzle01_coverage_lie.cpp
//   cat puzzle01_coverage_lie.cpp.gcov
//
// ============================================================================

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// The function under test
// ---------------------------------------------------------------------------
struct SensorReading {
    double value;
    bool is_valid;
    bool needs_calibration;
};

/// Processes a sensor reading. Returns a normalized value [0, 100].
///
/// Rules:
///   - If invalid, return -1.0
///   - If needs calibration, apply offset of +5.0
///   - Clamp to [0, 100]
///   - If value is NaN after processing, return -1.0
///
double process_reading(SensorReading r) {
    if (!r.is_valid) {
        return -1.0;                          // Line A
    }

    double result = r.value;

    if (r.needs_calibration) {
        result += 5.0;                        // Line B
    }

    // Clamp to [0, 100]
    if (result < 0.0) {
        result = 0.0;                         // Line C
    }
    if (result > 100.0) {
        result = 100.0;                       // Line D
    }

    // Safety check: NaN guard
    // BUG: This condition uses || instead of &&
    // When is_valid=true AND needs_calibration=true, this guard is
    // evaluated, but the NaN check itself is correct for normal cases.
    // The REAL bug: when value=NaN AND is_valid=true, the NaN doesn't
    // trigger the clamp conditions (NaN comparisons always return false),
    // so result stays NaN, but then:
    if (!r.is_valid || !r.needs_calibration) {
        // This "optimization" path skips the NaN check below
        // for readings that are valid AND need calibration
        return result;                        // Line E
    }

    // NaN check only runs for (is_valid=true, needs_calibration=true)
    if (std::isnan(result)) {
        return -1.0;                          // Line F
    }

    return result;                            // Line G
}

// ---------------------------------------------------------------------------
// Tests that achieve 100% LINE coverage
// ---------------------------------------------------------------------------
static void test_invalid_reading() {
    // Hits Line A
    double r = process_reading({42.0, false, false});
    assert(r == -1.0);
    printf("[PASS] test_invalid_reading\n");
}

static void test_normal_reading() {
    // Hits Line E (is_valid=true, needs_calibration=false → !needs_cal = true)
    double r = process_reading({50.0, true, false});
    assert(r == 50.0);
    printf("[PASS] test_normal_reading\n");
}

static void test_needs_calibration() {
    // Hits Line B (calibration offset) + Line G
    double r = process_reading({45.0, true, true});
    assert(r == 50.0);
    printf("[PASS] test_needs_calibration\n");
}

static void test_clamp_low() {
    // Hits Line C (clamp to 0)
    double r = process_reading({-10.0, true, false});
    assert(r == 0.0);
    printf("[PASS] test_clamp_low\n");
}

static void test_clamp_high() {
    // Hits Line D (clamp to 100)
    double r = process_reading({200.0, true, false});
    assert(r == 100.0);
    printf("[PASS] test_clamp_high\n");
}

static void test_nan_with_calibration() {
    // Hits Line F (NaN detected)
    double r = process_reading({NAN, true, true});
    // NaN + 5.0 = NaN, then NaN check catches it
    assert(r == -1.0);
    printf("[PASS] test_nan_with_calibration\n");
}

// ---------------------------------------------------------------------------
// Main — Run all tests
// ---------------------------------------------------------------------------
int main() {
    printf("=== puzzle01: Coverage Lie ===\n");
    printf("Every line of process_reading() will be covered.\n");
    printf("But there IS a bug. Can you find it?\n\n");

    test_invalid_reading();
    test_normal_reading();
    test_needs_calibration();
    test_clamp_low();
    test_clamp_high();
    test_nan_with_calibration();

    printf("\nAll tests passed. 100%% line coverage achieved.\n");
    printf("\n");
    printf("HINT: What happens with {NAN, true, false}?\n");
    printf("      The NaN guard on Line F only runs when\n");
    printf("      is_valid=true AND needs_calibration=true.\n");
    printf("      For {NAN, true, false}, we take Line E.\n");
    printf("      NaN survives clamping (NaN < 0.0 is false!).\n");
    printf("      Result: NaN returned as a 'valid' reading.\n");
    printf("\n");
    printf("MC/DC would require testing each condition in the\n");
    printf("compound decision independently, forcing the test\n");
    printf("  {NAN, true, false} — which exposes the bug.\n");

    return 0;
}
