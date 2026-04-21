// Module 12, Puzzle 2: The Single-Pass View Trap
// Compiler: GCC 12+ or Clang 14+ with -std=c++20
//
// CHALLENGE: Understand why some views can only be iterated once,
// and learn strategies to work around this limitation.
//
// Key concepts:
//   - input_range vs forward_range
//   - istream_view is single-pass (consumes the stream)
//   - Materializing a view to enable re-iteration

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <ranges>
#include <sstream>
#include <string>
#include <vector>

namespace views = std::views;
namespace ranges = std::ranges;

// ============================================================
// Case 1: istream_view — iterate once, then it's gone
// ============================================================
void case1_single_pass() {
    std::cout << "=== Case 1: Single-Pass istream_view ===\n";

    std::istringstream input("10 20 30 40 50");
    auto stream_view = ranges::istream_view<int>(input);

    // First iteration: works
    std::cout << "First iteration: ";
    for (int x : stream_view) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    // Second iteration: EMPTY — the stream is consumed
    std::cout << "Second iteration: ";
    for (int x : stream_view) {
        std::cout << x << " ";  // never executes
    }
    std::cout << "(empty!)\n\n";

    // WHY: istream_view models input_range, not forward_range.
    // input_range = single-pass. Once you advance past an element, it's gone.
    // The underlying stream has been fully read — there's nothing left.
}

// ============================================================
// Case 2: The fix — materialize before re-use
// ============================================================
void case2_materialize_fix() {
    std::cout << "=== Case 2: Materialize to Re-Iterate ===\n";

    std::istringstream input("10 20 30 40 50");
    auto stream_view = ranges::istream_view<int>(input);

    // Materialize into a vector — now we own the data
    std::vector<int> data(stream_view.begin(), stream_view.end());

    // Can iterate as many times as we want
    std::cout << "First:  ";
    for (int x : data) std::cout << x << " ";
    std::cout << "\n";

    std::cout << "Second: ";
    for (int x : data) std::cout << x << " ";
    std::cout << "\n";

    std::cout << "Third (reversed): ";
    for (int x : data | views::reverse) std::cout << x << " ";
    std::cout << "\n\n";
}

// ============================================================
// Case 3: Which views are single-pass vs multi-pass?
// ============================================================
void case3_pass_categories() {
    std::cout << "=== Case 3: Single-Pass vs Multi-Pass Views ===\n\n";

    // MULTI-PASS (forward_range or better) — safe to iterate multiple times:
    //   - views::filter(container, pred)   — container is multi-pass
    //   - views::transform(container, fn)  — container is multi-pass
    //   - views::take(container, n)        — container is multi-pass
    //   - views::reverse(container)        — container is bidirectional
    //   - views::iota(start, end)          — generates values, always multi-pass

    std::vector<int> v = {1, 2, 3, 4, 5};
    auto filtered = v | views::filter([](int x) { return x > 2; });

    std::cout << "filter over vector (multi-pass):\n";
    std::cout << "  Pass 1: ";
    for (int x : filtered) std::cout << x << " ";
    std::cout << "\n  Pass 2: ";
    for (int x : filtered) std::cout << x << " ";
    std::cout << "\n\n";

    // iota is always multi-pass
    auto five = views::iota(1, 6);
    std::cout << "iota(1, 6) (multi-pass):\n";
    std::cout << "  Pass 1: ";
    for (int x : five) std::cout << x << " ";
    std::cout << "\n  Pass 2: ";
    for (int x : five) std::cout << x << " ";
    std::cout << "\n\n";

    // SINGLE-PASS (input_range only):
    //   - ranges::istream_view<T>(stream)  — stream is consumed
    //   - Any view over a single-pass source
    //
    // The view inherits the "pass capability" of its source.
    // If the source is single-pass, the view is single-pass.

    std::cout << "Rule: A view is at most as capable as its source.\n";
    std::cout << "  vector → filter → transform → still forward_range (multi-pass)\n";
    std::cout << "  istream → filter → transform → still input_range (single-pass)\n\n";
}

// ============================================================
// Case 4: views::common — adaptor for legacy algorithms
// ============================================================
void case4_common_view() {
    std::cout << "=== Case 4: views::common for Legacy Compatibility ===\n";

    // Some views have different begin/end types (sentinel pattern).
    // Legacy algorithms (pre-C++20) require begin/end to be the SAME type.
    // views::common forces them to match.

    auto bounded_iota = views::iota(1, 10);
    // iota_view<int, int> has matching begin/end — already common

    // Example where common is needed: take on an unbounded range
    auto first_five = views::iota(1) | views::take(5);
    // take_view over unbounded iota has a sentinel end — not an iterator.

    // views::common wraps it so begin/end are the same type
    auto common_five = first_five | views::common;

    // Now we can use it with legacy algorithms that need matching iterators
    int sum = std::accumulate(common_five.begin(), common_five.end(), 0);
    std::cout << "Sum of first 5 natural numbers: " << sum << "\n";

    // With C++20 ranges algorithms, you don't need common:
    // ranges::for_each handles sentinels natively
    int sum2 = 0;
    ranges::for_each(first_five, [&sum2](int x) { sum2 += x; });
    std::cout << "Same with ranges::for_each:     " << sum2 << "\n\n";
}

// ============================================================
// Case 5: The real-world trap — processing a data file twice
// ============================================================
void case5_real_world_trap() {
    std::cout << "=== Case 5: Real-World — Processing Data Twice ===\n";

    // Scenario: you read sensor calibration data from a stream.
    // You want to: (1) compute the mean, (2) compute std deviation.
    // Both need a full pass over the data.

    std::istringstream calibration_data("1.0 2.5 3.0 1.5 4.0 2.0 3.5");

    // WRONG approach: try to iterate the stream view twice
    // auto readings = ranges::istream_view<double>(calibration_data);
    // double sum = 0; int n = 0;
    // for (double x : readings) { sum += x; ++n; }
    // double mean = sum / n;
    // // Second pass for std dev — OOPS, readings is empty!

    // CORRECT: materialize first
    auto raw = ranges::istream_view<double>(calibration_data);
    std::vector<double> readings(raw.begin(), raw.end());

    // Now we can do multiple passes
    double sum = 0.0;
    for (double x : readings) sum += x;
    double mean = sum / static_cast<double>(readings.size());

    double sq_diff_sum = 0.0;
    for (double x : readings) {
        double diff = x - mean;
        sq_diff_sum += diff * diff;
    }
    double stddev = std::sqrt(sq_diff_sum / static_cast<double>(readings.size()));

    std::cout << "Readings: ";
    for (double x : readings) std::cout << x << " ";
    std::cout << "\nMean:   " << mean;
    std::cout << "\nStddev: " << stddev << "\n\n";

    std::cout << "Lesson: If you need multiple passes, materialize the view first.\n";
    std::cout << "Cost: one allocation. Benefit: correctness.\n";
}

int main() {
    case1_single_pass();
    case2_materialize_fix();
    case3_pass_categories();
    case4_common_view();
    case5_real_world_trap();
    return 0;
}
