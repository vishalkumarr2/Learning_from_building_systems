// Module 17: SDV & Mission-Critical Systems
// Puzzle 03: OTA Version Rollback Trap
//
// This OTA update system has a subtle logic bug in version comparison
// that allows DOWNGRADES to be treated as upgrades in certain cases.
// An attacker (or buggy cloud backend) could exploit this to roll back
// a vehicle to a vulnerable firmware version.
//
// The challenge:
//   1. Find the version comparison bug
//   2. Construct a version pair that triggers the wrong comparison
//   3. Fix the comparison to be correct for semantic versioning
//   4. Bonus: explain why OTA systems also need rollback counters
//      (even with correct version comparison)
//
// Build:
//   g++ -std=c++2a -Wall -Wextra -o puzzle03 puzzle03_ota_rollback_trap.cpp
//
// Context:
//   ISO/SAE 21434 (Cybersecurity) and UN R156 (Software Updates) require
//   that OTA systems prevent unauthorized downgrades. A version comparison
//   bug is a real CVE category in automotive and IoT.

#include <cassert>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>

struct Version {
    uint8_t major = 0;
    uint8_t minor = 0;
    uint8_t patch = 0;

    std::string to_string() const {
        return std::to_string(major) + "." +
               std::to_string(minor) + "." +
               std::to_string(patch);
    }

    static Version parse(std::string const& s) {
        Version v;
        char dot1, dot2;
        unsigned M, m, p;
        std::istringstream iss(s);
        iss >> M >> dot1 >> m >> dot2 >> p;
        v.major = static_cast<uint8_t>(M);
        v.minor = static_cast<uint8_t>(m);
        v.patch = static_cast<uint8_t>(p);
        return v;
    }
};

// ============== BUG IS IN THIS FUNCTION ==============
bool is_upgrade(Version const& current, Version const& candidate) {
    // "Simple" version comparison
    // This looks correct... but has a subtle overflow / promotion issue
    // when encoding version as a single integer for comparison.

    // Encode as single number for easy comparison
    int current_code  = current.major  * 1000 + current.minor  * 10 + current.patch;
    int candidate_code = candidate.major * 1000 + candidate.minor * 10 + candidate.patch;

    return candidate_code > current_code;
}
// =====================================================

void test(std::string const& current_str, std::string const& candidate_str,
          bool expected) {
    auto current = Version::parse(current_str);
    auto candidate = Version::parse(candidate_str);
    bool result = is_upgrade(current, candidate);

    std::cout << "  " << current_str << " → " << candidate_str
              << " : is_upgrade=" << (result ? "true" : "false");

    if (result != expected) {
        std::cout << "  *** WRONG! Expected " << (expected ? "true" : "false")
                  << " ***\n";
    } else {
        std::cout << "  OK\n";
    }
}

int main() {
    std::cout << "=== Puzzle: OTA Version Rollback Trap ===\n\n";

    std::cout << "Obvious cases (work fine):\n";
    test("1.0.0", "2.0.0", true);    // major upgrade
    test("1.0.0", "1.1.0", true);    // minor upgrade
    test("1.0.0", "1.0.1", true);    // patch upgrade
    test("2.0.0", "1.0.0", false);   // major downgrade

    std::cout << "\nTricky cases (where the bug hides):\n";
    test("1.9.0", "1.10.0", true);   // minor 9 → 10 : should be upgrade
    test("1.99.0", "1.100.0", true); // minor overflow: > uint8_t but shows bug
    test("1.0.9", "1.0.10", true);   // patch 9 → 10
    test("2.0.0", "1.99.9", false);  // downgrade despite high minor/patch

    std::cout << "\n"
              << "HINT: The encoding `major*1000 + minor*10 + patch` only works\n"
              << "when minor and patch are single digits (0-9).\n"
              << "When minor=10, major*1000 + 10*10 = major*1000 + 100,\n"
              << "which overlaps with the next major version's range!\n\n"
              << "Example: 1.10.0 encodes as 1*1000 + 10*10 + 0 = 1100\n"
              << "         But 1.9.0  encodes as 1*1000 + 9*10 + 0 = 1090\n"
              << "         And 2.0.0  encodes as 2*1000 + 0     + 0 = 2000\n"
              << "So 1.10.0 (1100) < 2.0.0 (2000) — this one works by luck.\n\n"
              << "But consider: 1.0.11 = 1000 + 0 + 11 = 1011\n"
              << "              1.1.0  = 1000 + 10 + 0  = 1010\n"
              << "So is_upgrade(1.1.0, 1.0.11) = true — WRONG! (downgrade)\n\n"
              << "FIX: Compare lexicographically: major first, then minor, then patch.\n"
              << "     Or use a wider encoding: major*10000 + minor*100 + patch\n"
              << "     (only works if each field < 100).\n"
              << "     Best: direct tuple comparison.\n";

    // Demonstrate the actual exploit
    std::cout << "\n=== Exploit Demonstration ===\n";
    test("1.1.0", "1.0.11", false);  // should be downgrade, is_upgrade says true!

    return 0;
}
