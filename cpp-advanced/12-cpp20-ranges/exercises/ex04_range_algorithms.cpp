// Module 12, Exercise 4: Range Algorithms vs Classic Algorithms
// Compiler: GCC 12+ or Clang 14+ with -std=c++20
//
// Demonstrates:
//   - Side-by-side: std::sort vs ranges::sort
//   - ranges::find_if with projection
//   - Materializing views into containers (C++20 manual, C++23 ranges::to)
//   - partition, rotate, unique with ranges
//   - How ranges prevent iterator mismatch bugs

#include <algorithm>
#include <iostream>
#include <numeric>
#include <ranges>
#include <string>
#include <vector>

namespace ranges = std::ranges;
namespace views = std::views;

struct Task {
    int         id;
    std::string name;
    int         priority;   // 1 = critical, 5 = low
    bool        completed;
};

void print_tasks(const std::vector<Task>& tasks, const std::string& label) {
    std::cout << label << ":\n";
    for (const auto& t : tasks) {
        std::cout << "  [" << t.id << "] " << t.name
                  << " (pri=" << t.priority
                  << ", " << (t.completed ? "done" : "TODO") << ")\n";
    }
    std::cout << "\n";
}

// ============================================================
// Part 1: Classic vs Ranges sort
// ============================================================
void part1_sort_comparison() {
    std::cout << "=== Part 1: Classic sort vs ranges::sort ===\n";

    std::vector<int> v1 = {5, 3, 8, 1, 9, 2, 7};
    std::vector<int> v2 = v1;

    // Classic — must pass begin/end
    std::sort(v1.begin(), v1.end());

    // Ranges — pass the container directly. Safer, cleaner.
    ranges::sort(v2);

    // Both produce the same result
    std::cout << "Classic sort: ";
    for (int x : v1) std::cout << x << " ";
    std::cout << "\nRanges sort:  ";
    for (int x : v2) std::cout << x << " ";
    std::cout << "\n\n";

    // The bug ranges prevents:
    // std::vector<int> a = {3, 1, 2};
    // std::vector<int> b = {9, 8, 7};
    // std::sort(a.begin(), b.end());   // COMPILES! UB!
    // ranges::sort takes ONE range — can't mismatch.
}

// ============================================================
// Part 2: find_if with projection
// ============================================================
void part2_find_with_projection() {
    std::cout << "=== Part 2: find_if with Projection ===\n";

    std::vector<Task> tasks = {
        {1, "Navigate to A",    2, false},
        {2, "Pick up pallet",   1, false},
        {3, "Charge battery",   3, true},
        {4, "Deliver to dock",  1, false},
        {5, "Report status",    5, true},
    };

    // Find first incomplete critical task (priority == 1)
    auto it = ranges::find_if(tasks,
        [](int pri) { return pri == 1; },
        &Task::priority);
    // Note: projection extracts priority, predicate tests it.

    if (it != tasks.end()) {
        std::cout << "First critical task: [" << it->id << "] " << it->name << "\n";
    }

    // Classic equivalent (more verbose):
    auto it2 = std::find_if(tasks.begin(), tasks.end(),
        [](const Task& t) { return t.priority == 1; });
    if (it2 != tasks.end()) {
        std::cout << "Classic find:        [" << it2->id << "] " << it2->name << "\n";
    }

    // Find by name
    auto by_name = ranges::find(tasks, "Charge battery", &Task::name);
    if (by_name != tasks.end()) {
        std::cout << "Found by name:       [" << by_name->id << "] " << by_name->name << "\n";
    }

    std::cout << "\n";
}

// ============================================================
// Part 3: Materializing views into containers
// ============================================================
void part3_materialize() {
    std::cout << "=== Part 3: View → Container ===\n";

    std::vector<int> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Create a view
    auto evens_squared = data
        | views::filter([](int x) { return x % 2 == 0; })
        | views::transform([](int x) { return x * x; });

    // C++20: materialize with iterator constructor
    std::vector<int> result(evens_squared.begin(), evens_squared.end());

    // C++23 would be: auto result = evens_squared | ranges::to<std::vector>();

    std::cout << "Evens squared: ";
    for (int x : result) std::cout << x << " ";
    std::cout << "\n";

    // Helper pattern for C++20 (poor man's ranges::to):
    auto to_vector = [](auto&& rng) {
        using T = ranges::range_value_t<decltype(rng)>;
        return std::vector<T>(ranges::begin(rng), ranges::end(rng));
    };

    auto result2 = to_vector(data | views::filter([](int x) { return x > 5; }));
    std::cout << "Greater than 5: ";
    for (int x : result2) std::cout << x << " ";
    std::cout << "\n\n";
}

// ============================================================
// Part 4: partition, rotate, unique
// ============================================================
void part4_more_algorithms() {
    std::cout << "=== Part 4: Partition, Rotate, Unique ===\n";

    // --- Partition: move completed tasks to end ---
    std::vector<Task> tasks = {
        {1, "Navigate",   2, false},
        {2, "Pick up",    1, true},
        {3, "Charge",     3, true},
        {4, "Deliver",    1, false},
        {5, "Report",     5, false},
    };

    // Partition: incomplete first, completed last
    auto [mid] = ranges::partition(tasks, [](bool c) { return !c; }, &Task::completed);

    std::cout << "After partition (incomplete first):\n";
    std::cout << "  Incomplete:\n";
    for (auto it = tasks.begin(); it != mid; ++it) {
        std::cout << "    [" << it->id << "] " << it->name << "\n";
    }
    std::cout << "  Completed:\n";
    for (auto it = mid; it != tasks.end(); ++it) {
        std::cout << "    [" << it->id << "] " << it->name << "\n";
    }

    // --- Unique: remove consecutive duplicates ---
    std::vector<int> readings = {1, 1, 2, 2, 2, 3, 1, 1, 4, 4};
    std::cout << "\nBefore unique: ";
    for (int x : readings) std::cout << x << " ";

    auto [new_end] = ranges::unique(readings);
    readings.erase(new_end, readings.end());

    std::cout << "\nAfter unique:  ";
    for (int x : readings) std::cout << x << " ";
    std::cout << "\n";

    // --- Rotate: bring element to front ---
    std::vector<int> queue = {1, 2, 3, 4, 5};
    std::cout << "\nBefore rotate: ";
    for (int x : queue) std::cout << x << " ";

    // Rotate so that '3' is at the front
    auto pos = ranges::find(queue, 3);
    ranges::rotate(queue, pos);

    std::cout << "\nAfter rotate:  ";
    for (int x : queue) std::cout << x << " ";
    std::cout << "\n\n";
}

// ============================================================
// Part 5: Ranges prevent iterator mismatches
// ============================================================
void part5_safety() {
    std::cout << "=== Part 5: Safety — No Iterator Mismatches ===\n";

    std::vector<int> a = {5, 3, 1};
    std::vector<int> b = {9, 7, 6};

    // Classic: this would COMPILE but crash
    // std::sort(a.begin(), b.end());  // UB!

    // Ranges: you pass the WHOLE range. Can't mix up.
    ranges::sort(a);
    ranges::sort(b);

    std::cout << "a sorted: ";
    for (int x : a) std::cout << x << " ";
    std::cout << "\nb sorted: ";
    for (int x : b) std::cout << x << " ";
    std::cout << "\n";

    // ranges::copy also takes a range + output iterator
    std::vector<int> merged;
    merged.reserve(a.size() + b.size());
    ranges::copy(a, std::back_inserter(merged));
    ranges::copy(b, std::back_inserter(merged));
    ranges::sort(merged);

    std::cout << "merged:   ";
    for (int x : merged) std::cout << x << " ";
    std::cout << "\n\n";
}

// ============================================================
// Part 6: ranges::for_each with projection
// ============================================================
void part6_for_each() {
    std::cout << "=== Part 6: for_each with Projection ===\n";

    std::vector<Task> tasks = {
        {1, "Navigate",   2, false},
        {2, "Pick up",    1, true},
        {3, "Charge",     3, false},
    };

    // Print just the names using projection
    std::cout << "Task names: ";
    ranges::for_each(tasks, [](const std::string& name) {
        std::cout << name << ", ";
    }, &Task::name);
    std::cout << "\n\n";
}

int main() {
    part1_sort_comparison();
    part2_find_with_projection();
    part3_materialize();
    part4_more_algorithms();
    part5_safety();
    part6_for_each();
    return 0;
}
