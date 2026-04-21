// Module 12, Exercise 2: Projections — The Killer Feature
// Compiler: GCC 12+ or Clang 14+ with -std=c++20
//
// Projections let you say "compare by THIS member/function" without
// writing a full comparator lambda. They work with all range algorithms.
//
// Demonstrates:
//   - ranges::sort with member pointer projections
//   - ranges::max_element / min_element with projections
//   - Projections vs verbose lambda comparators
//   - Real-world: robot waypoints sorted by distance, then priority

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <ranges>
#include <string>
#include <vector>

namespace ranges = std::ranges;

// ============================================================
// Data types
// ============================================================

struct Waypoint {
    std::string name;
    double x, y;          // position in meters
    int    priority;      // 1 = highest, 10 = lowest

    double distance_from_origin() const {
        return std::sqrt(x * x + y * y);
    }
};

struct RobotTelemetry {
    int    robot_id;
    double battery_pct;
    double speed_mps;     // meters per second
    int    tasks_done;
};

void print_waypoints(const std::vector<Waypoint>& wps, const std::string& label) {
    std::cout << label << ":\n";
    for (const auto& w : wps) {
        std::cout << "  " << std::setw(12) << w.name
                  << "  pos=(" << std::setw(5) << w.x << ", " << std::setw(5) << w.y << ")"
                  << "  dist=" << std::fixed << std::setprecision(1) << w.distance_from_origin()
                  << "  pri=" << w.priority << "\n";
    }
    std::cout << "\n";
}

// ============================================================
// Part 1: Basic projections — sort by member
// ============================================================
void part1_sort_by_member() {
    std::cout << "=== Part 1: Sort by Member Projection ===\n";

    std::vector<Waypoint> wps = {
        {"charging",  1.0, 2.0, 3},
        {"pickup_A",  8.0, 1.0, 1},
        {"dropoff_B", 3.0, 7.0, 5},
        {"staging",   0.5, 0.5, 2},
        {"exit",     10.0, 10.0, 8},
    };

    // --- WITHOUT projections (verbose) ---
    auto wps_old = wps;  // copy for comparison
    std::sort(wps_old.begin(), wps_old.end(),
        [](const Waypoint& a, const Waypoint& b) {
            return a.priority < b.priority;
        });
    print_waypoints(wps_old, "Sorted by priority (OLD: lambda comparator)");

    // --- WITH projections (clean) ---
    // The third argument is the projection: "extract this before comparing"
    // {} means default comparator (std::less)
    ranges::sort(wps, {}, &Waypoint::priority);
    print_waypoints(wps, "Sorted by priority (NEW: projection)");

    // Sort by name (alphabetical)
    ranges::sort(wps, {}, &Waypoint::name);
    print_waypoints(wps, "Sorted by name");
}

// ============================================================
// Part 2: Projections with callable (not just member pointers)
// ============================================================
void part2_callable_projections() {
    std::cout << "=== Part 2: Callable Projections ===\n";

    std::vector<Waypoint> wps = {
        {"charging",  1.0, 2.0, 3},
        {"pickup_A",  8.0, 1.0, 1},
        {"dropoff_B", 3.0, 7.0, 5},
        {"staging",   0.5, 0.5, 2},
        {"exit",     10.0, 10.0, 8},
    };

    // Sort by distance from origin — projection is a member function
    ranges::sort(wps, {}, &Waypoint::distance_from_origin);
    print_waypoints(wps, "Sorted by distance from origin");

    // Sort by distance, descending — change comparator, keep projection
    ranges::sort(wps, std::greater{}, &Waypoint::distance_from_origin);
    print_waypoints(wps, "Sorted by distance (descending)");

    // Projection can be any lambda
    ranges::sort(wps, {}, [](const Waypoint& w) {
        return std::abs(w.x - w.y);  // sort by |x - y|
    });
    print_waypoints(wps, "Sorted by |x - y|");
}

// ============================================================
// Part 3: Projections with find/min/max
// ============================================================
void part3_find_min_max() {
    std::cout << "=== Part 3: Projections with find/min/max ===\n";

    std::vector<RobotTelemetry> fleet = {
        {1, 85.0, 1.2, 15},
        {2, 42.0, 0.8, 22},
        {3, 97.0, 1.5, 8},
        {4, 15.0, 0.0, 30},  // battery critical, stopped
        {5, 63.0, 1.1, 18},
    };

    // Find robot with lowest battery
    auto lowest_bat = ranges::min_element(fleet, {}, &RobotTelemetry::battery_pct);
    std::cout << "Lowest battery: robot " << lowest_bat->robot_id
              << " at " << lowest_bat->battery_pct << "%\n";

    // Find robot with most tasks done
    auto most_productive = ranges::max_element(fleet, {}, &RobotTelemetry::tasks_done);
    std::cout << "Most productive: robot " << most_productive->robot_id
              << " with " << most_productive->tasks_done << " tasks\n";

    // Find first robot that's stopped (speed == 0)
    auto stopped = ranges::find(fleet, 0.0, &RobotTelemetry::speed_mps);
    if (stopped != fleet.end()) {
        std::cout << "Stopped robot: " << stopped->robot_id
                  << " (battery: " << stopped->battery_pct << "%)\n";
    }

    // All robots with battery > 50%
    auto healthy_count = ranges::count_if(fleet,
        [](double b) { return b > 50.0; },
        &RobotTelemetry::battery_pct);
    std::cout << "Robots with battery > 50%: " << healthy_count << "\n\n";
}

// ============================================================
// Part 4: Compound sort — primary + secondary key
// ============================================================
void part4_compound_sort() {
    std::cout << "=== Part 4: Multi-Key Sort ===\n";

    std::vector<Waypoint> wps = {
        {"A_close_hi",  1.0, 1.0, 1},
        {"B_close_lo",  1.5, 0.5, 5},
        {"C_far_hi",    9.0, 9.0, 1},
        {"D_far_lo",    8.0, 8.0, 5},
        {"E_mid_hi",    4.0, 3.0, 1},
        {"F_mid_lo",    5.0, 4.0, 5},
    };

    // Two-pass sort: first by priority (primary), then by distance (secondary)
    // Stable sort preserves relative order within same priority
    ranges::stable_sort(wps, {}, &Waypoint::distance_from_origin);  // secondary first
    ranges::stable_sort(wps, {}, &Waypoint::priority);               // primary last

    print_waypoints(wps, "Sorted by priority, then distance (stable)");

    // Or use a single comparator with projection-style logic:
    ranges::sort(wps, [](const Waypoint& a, const Waypoint& b) {
        if (a.priority != b.priority) return a.priority < b.priority;
        return a.distance_from_origin() < b.distance_from_origin();
    });
    print_waypoints(wps, "Same result with single comparator");
}

int main() {
    part1_sort_by_member();
    part2_callable_projections();
    part3_find_min_max();
    part4_compound_sort();
    return 0;
}
