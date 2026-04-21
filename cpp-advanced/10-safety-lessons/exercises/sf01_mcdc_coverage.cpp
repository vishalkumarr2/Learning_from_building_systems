// sf01_mcdc_coverage.cpp — DO-178C MC/DC Coverage Exercise
//
// MC/DC (Modified Condition/Decision Coverage) requires that each condition
// in a decision is shown to independently affect the decision outcome.
// For each condition C in decision D:
//   - Find two test cases where C differs, all other conditions are the same,
//     and the decision outcome differs.
//
// This is the gold standard for safety-critical avionics software (DAL A).
//
// Exercise: implement compute_brake and systematically derive MC/DC test cases.
//
// Build: g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -pthread sf01_mcdc_coverage.cpp -o sf01_mcdc_coverage

#include <cstdint>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cassert>

// ============================================================================
// Part 1: The safety-critical function under test
// ============================================================================

enum class BrakeCommand : uint8_t {
    NO_BRAKE   = 0,
    FULL_BRAKE = 1,
};

// Decision: brake = obstacle_near OR speed_over_limit OR driver_brake
// This is a 3-condition OR decision.
//
// In real automotive software (ISO 26262), every condition that can trigger
// emergency braking must be independently validated.
BrakeCommand compute_brake(bool obstacle_near, bool speed_over_limit, bool driver_brake) {
    if (obstacle_near || speed_over_limit || driver_brake) {
        return BrakeCommand::FULL_BRAKE;
    }
    return BrakeCommand::NO_BRAKE;
}

// ============================================================================
// Part 2: MC/DC test case derivation
// ============================================================================

struct TestCase {
    int         id;
    bool        obstacle_near;
    bool        speed_over_limit;
    bool        driver_brake;
    BrakeCommand expected;
    std::string  mcdc_note;  // which condition independence this demonstrates
};

// For a 3-condition OR (A || B || C):
//
// MC/DC requires pairs where ONE condition flips and the outcome flips,
// while the other conditions are held constant.
//
// Key insight for OR: a condition can only independently affect the outcome
// when the OTHER conditions are FALSE (otherwise the OR is already true).
//
// Independence pairs:
//   Condition A (obstacle_near):    T1(F,F,F)→NO  vs T2(T,F,F)→FULL  (A flips, B=F, C=F)
//   Condition B (speed_over_limit): T1(F,F,F)→NO  vs T3(F,T,F)→FULL  (B flips, A=F, C=F)
//   Condition C (driver_brake):     T1(F,F,F)→NO  vs T4(F,F,T)→FULL  (C flips, A=F, B=F)
//
// Minimum MC/DC test set for 3-condition OR: 4 tests (N+1 where N=conditions)
// Compare: simple branch coverage needs only 2 tests (one true, one false path)
// Compare: full truth table has 2^3 = 8 rows

std::vector<TestCase> build_mcdc_tests() {
    return {
        // T1: baseline — all false → NO_BRAKE
        {1, false, false, false, BrakeCommand::NO_BRAKE,
         "Baseline: all conditions false"},

        // T2: obstacle_near flips — pair with T1 shows A's independence
        {2, true,  false, false, BrakeCommand::FULL_BRAKE,
         "Pair(T1,T2): obstacle_near independently causes FULL_BRAKE"},

        // T3: speed_over_limit flips — pair with T1 shows B's independence
        {3, false, true,  false, BrakeCommand::FULL_BRAKE,
         "Pair(T1,T3): speed_over_limit independently causes FULL_BRAKE"},

        // T4: driver_brake flips — pair with T1 shows C's independence
        {4, false, false, true,  BrakeCommand::FULL_BRAKE,
         "Pair(T1,T4): driver_brake independently causes FULL_BRAKE"},
    };
}

// ============================================================================
// Part 3: Full truth table for comparison
// ============================================================================

struct TruthTableRow {
    bool a, b, c;
    BrakeCommand expected;
};

std::vector<TruthTableRow> full_truth_table() {
    return {
        {false, false, false, BrakeCommand::NO_BRAKE},
        {false, false, true,  BrakeCommand::FULL_BRAKE},
        {false, true,  false, BrakeCommand::FULL_BRAKE},
        {false, true,  true,  BrakeCommand::FULL_BRAKE},
        {true,  false, false, BrakeCommand::FULL_BRAKE},
        {true,  false, true,  BrakeCommand::FULL_BRAKE},
        {true,  true,  false, BrakeCommand::FULL_BRAKE},
        {true,  true,  true,  BrakeCommand::FULL_BRAKE},
    };
}

// ============================================================================
// Part 4: Run and report
// ============================================================================

std::string bool_str(bool b) { return b ? "T" : "F"; }
std::string brake_str(BrakeCommand bc) {
    return bc == BrakeCommand::FULL_BRAKE ? "FULL_BRAKE" : "NO_BRAKE ";
}

int main() {
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  DO-178C MC/DC Coverage Exercise\n";
    std::cout << "  Decision: brake = obstacle_near OR speed_over_limit OR driver_brake\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";

    // --- MC/DC tests ---
    auto tests = build_mcdc_tests();
    int pass_count = 0;
    int fail_count = 0;

    std::cout << "MC/DC Test Results:\n";
    std::cout << "┌────┬───────────┬─────────────┬──────────┬─────────────┬─────────────┬────────┐\n";
    std::cout << "│ ID │ obstacle  │ speed_over  │ driver   │ expected    │ actual      │ result │\n";
    std::cout << "├────┼───────────┼─────────────┼──────────┼─────────────┼─────────────┼────────┤\n";

    for (const auto& tc : tests) {
        BrakeCommand actual = compute_brake(tc.obstacle_near, tc.speed_over_limit, tc.driver_brake);
        bool passed = (actual == tc.expected);
        if (passed) ++pass_count; else ++fail_count;

        std::cout << "│ " << std::setw(2) << tc.id << " │"
                  << "    " << bool_str(tc.obstacle_near) << "      │"
                  << "      " << bool_str(tc.speed_over_limit) << "      │"
                  << "    " << bool_str(tc.driver_brake) << "    │"
                  << " " << brake_str(tc.expected) << "  │"
                  << " " << brake_str(actual) << "  │"
                  << " " << (passed ? "PASS " : "FAIL ") << "  │\n";
    }
    std::cout << "└────┴───────────┴─────────────┴──────────┴─────────────┴─────────────┴────────┘\n\n";

    // --- Independence pairs ---
    std::cout << "Independence Pairs (the core of MC/DC):\n";
    std::cout << "┌─────────────────────┬────────────┬─────────────────────────────────────────┐\n";
    std::cout << "│ Condition           │ Test Pair  │ Demonstration                           │\n";
    std::cout << "├─────────────────────┼────────────┼─────────────────────────────────────────┤\n";
    std::cout << "│ obstacle_near       │ T1 vs T2   │ A flips F→T, B=F, C=F → outcome flips  │\n";
    std::cout << "│ speed_over_limit    │ T1 vs T3   │ B flips F→T, A=F, C=F → outcome flips  │\n";
    std::cout << "│ driver_brake        │ T1 vs T4   │ C flips F→T, A=F, B=F → outcome flips  │\n";
    std::cout << "└─────────────────────┴────────────┴─────────────────────────────────────────┘\n\n";

    // --- Coverage comparison ---
    std::cout << "Coverage Comparison:\n";
    std::cout << "  Simple branch coverage:  2 tests (one T path, one F path)\n";
    std::cout << "  MC/DC coverage:          " << tests.size() << " tests (N+1 = 3+1)\n";
    std::cout << "  Full truth table:        8 tests (2^N = 2^3)\n\n";

    // --- Full truth table verification ---
    std::cout << "Full Truth Table Verification:\n";
    auto table = full_truth_table();
    bool all_ok = true;
    for (const auto& row : table) {
        BrakeCommand actual = compute_brake(row.a, row.b, row.c);
        if (actual != row.expected) {
            std::cout << "  FAIL: (" << bool_str(row.a) << "," << bool_str(row.b)
                      << "," << bool_str(row.c) << ") expected " << brake_str(row.expected)
                      << " got " << brake_str(actual) << "\n";
            all_ok = false;
        }
    }
    if (all_ok) {
        std::cout << "  All 8 truth table rows PASS ✓\n\n";
    }

    // --- MC/DC note for each test ---
    std::cout << "Per-test MC/DC notes:\n";
    for (const auto& tc : tests) {
        std::cout << "  T" << tc.id << ": " << tc.mcdc_note << "\n";
    }

    std::cout << "\n══════════════════════════════════════════════════════════════\n";
    std::cout << "  Summary: " << pass_count << " passed, " << fail_count << " failed\n";
    std::cout << "══════════════════════════════════════════════════════════════\n";

    // --- Why MC/DC matters ---
    std::cout << R"(
Key Takeaways:
  1. MC/DC requires N+1 tests for N conditions (vs 2 for branch, 2^N for exhaustive)
  2. Each condition must be shown to INDEPENDENTLY affect the outcome
  3. For OR: independence requires other conditions to be FALSE
  4. For AND: independence requires other conditions to be TRUE
  5. DO-178C Level A (flight-critical) requires MC/DC — branch coverage is insufficient
  6. ISO 26262 ASIL D (automotive) also references MC/DC
  7. The technique scales: 20 conditions → 21 MC/DC tests vs 1,048,576 exhaustive tests
)";

    return (fail_count == 0) ? 0 : 1;
}
