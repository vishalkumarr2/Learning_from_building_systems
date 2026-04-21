// Module 12, Exercise 1: Range Basics
// Compiler: GCC 12+ or Clang 14+ with -std=c++20
//
// Demonstrates:
//   - views::filter, views::transform, views::take with pipe syntax
//   - Lazy evaluation proof via side effects
//   - Comparison with traditional for-loop approach

#include <algorithm>
#include <iostream>
#include <ranges>
#include <vector>

namespace views = std::views;
namespace ranges = std::ranges;

// Simulated sensor reading (e.g., lidar range in meters)
struct SensorReading {
    int    sensor_id;
    double range_m;      // measured distance
    bool   is_valid;     // hardware validity flag
};

// ============================================================
// Part 1: Basic filter + transform + take pipeline
// ============================================================
void part1_basic_pipeline() {
    std::cout << "=== Part 1: Filter → Transform → Take ===\n";

    std::vector<SensorReading> scan = {
        {0, 1.23, true},  {1, -0.01, false}, {2, 5.67, true},
        {3, 0.00, false},  {4, 3.45, true},  {5, 12.8, true},
        {6, 0.50, true},  {7, -1.0, false},  {8, 7.77, true},
        {9, 25.0, true},  {10, 0.02, true},  {11, 4.56, true},
    };

    // Pipeline: keep valid readings → convert m → mm → take first 5
    auto pipeline = scan
        | views::filter([](const SensorReading& r) {
              return r.is_valid && r.range_m > 0.05;
          })
        | views::transform([](const SensorReading& r) {
              return r.range_m * 1000.0;  // meters → millimeters
          })
        | views::take(5);

    // Nothing has been computed yet! The pipeline is just a description.
    // Work happens here, one element at a time:
    std::cout << "Valid ranges (mm), first 5: ";
    for (double mm : pipeline) {
        std::cout << mm << " ";
    }
    std::cout << "\n\n";
}

// ============================================================
// Part 2: Prove laziness with side effects
// ============================================================
void part2_laziness_proof() {
    std::cout << "=== Part 2: Proving Laziness ===\n";

    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    int transform_calls = 0;

    // Create the view — this does ZERO work
    auto lazy = data
        | views::filter([](int x) {
              std::cout << "  filter(" << x << ")\n";
              return x % 2 == 0;  // keep evens
          })
        | views::transform([&transform_calls](int x) {
              ++transform_calls;
              std::cout << "  transform(" << x << ") → " << x * x << "\n";
              return x * x;
          })
        | views::take(3);  // we only want 3 results

    std::cout << "Pipeline created. transform_calls so far: "
              << transform_calls << " (should be 0)\n\n";

    std::cout << "Now iterating:\n";
    for (int val : lazy) {
        std::cout << "  → got: " << val << "\n";
    }

    // Only 3 transforms happened — not 10, not 5.
    // Filter ran on 1,2,3,4,5,6 (stopped after finding 3 evens: 2,4,6)
    // Transform ran on 2, 4, 6 only.
    std::cout << "\nTotal transform calls: " << transform_calls
              << " (should be 3, not 10)\n\n";
}

// ============================================================
// Part 3: Traditional for-loop equivalent (for comparison)
// ============================================================
void part3_traditional_approach() {
    std::cout << "=== Part 3: Traditional For-Loop (contrast) ===\n";

    std::vector<SensorReading> scan = {
        {0, 1.23, true},  {1, -0.01, false}, {2, 5.67, true},
        {3, 0.00, false},  {4, 3.45, true},  {5, 12.8, true},
        {6, 0.50, true},  {7, -1.0, false},  {8, 7.77, true},
        {9, 25.0, true},  {10, 0.02, true},  {11, 4.56, true},
    };

    // Same logic as Part 1, but with a manual loop
    std::vector<double> result;
    int count = 0;
    for (const auto& r : scan) {
        if (!r.is_valid || r.range_m <= 0.05) continue;
        result.push_back(r.range_m * 1000.0);
        if (++count == 5) break;
    }

    std::cout << "Valid ranges (mm), first 5: ";
    for (double mm : result) {
        std::cout << mm << " ";
    }
    std::cout << "\n";

    // The ranges version:
    //   - No mutable counter
    //   - No manual break
    //   - No intermediate vector (lazy!)
    //   - Composable — you can store/reuse the pipeline
    std::cout << "(Same result as Part 1, but more boilerplate)\n\n";
}

// ============================================================
// Part 4: Composing reusable pipelines
// ============================================================
void part4_composable_pipelines() {
    std::cout << "=== Part 4: Storing & Reusing Pipelines ===\n";

    // You can store a partial pipeline as a variable
    auto positive_evens = views::filter([](int x) { return x > 0 && x % 2 == 0; });
    auto squared = views::transform([](int x) { return x * x; });

    // Compose once, apply to different data
    auto pipeline = positive_evens | squared | views::take(3);

    std::vector<int> data_a = {-3, 2, 5, 8, -1, 4, 6, 10};
    std::vector<int> data_b = {100, 200, 7, 300, 9, 400};

    std::cout << "data_a | pipeline: ";
    for (int x : data_a | pipeline) std::cout << x << " ";
    std::cout << "\n";

    std::cout << "data_b | pipeline: ";
    for (int x : data_b | pipeline) std::cout << x << " ";
    std::cout << "\n\n";
}

int main() {
    part1_basic_pipeline();
    part2_laziness_proof();
    part3_traditional_approach();
    part4_composable_pipelines();
    return 0;
}
