// Module 12, Puzzle 1: Dangling View References
// Compiler: GCC 12+ or Clang 14+ with -std=c++20
//
// CHALLENGE: Predict which operations are safe and which create dangling
// references. Read each case, predict the outcome, then run to check.
//
// Key concept: views don't own data. If the source dies, the view dangles.
// Range algorithms on temporaries return ranges::dangling instead of iterators.

#include <algorithm>
#include <iostream>
#include <ranges>
#include <string>
#include <vector>

namespace ranges = std::ranges;
namespace views = std::views;

// Helper: returns a temporary vector
std::vector<int> make_data() {
    return {5, 3, 8, 1, 9, 2, 7, 4, 6};
}

// ============================================================
// Case 1: ranges::find on a temporary
// ============================================================
void case1_find_on_temporary() {
    std::cout << "=== Case 1: ranges::find on temporary ===\n";

    // What does this return?
    auto result = ranges::find(make_data(), 8);

    // ANSWER: result is std::ranges::dangling — NOT an iterator.
    // The vector returned by make_data() is destroyed after the call.
    //
    // If you try: *result → COMPILE ERROR
    // The type system PREVENTS you from dereferencing dangling iterators.

    // Uncomment to see the compile error:
    // std::cout << *result << "\n";  // error: no match for 'operator*'

    std::cout << "ranges::find(make_data(), 8) returns ranges::dangling\n";
    std::cout << "Cannot dereference — compile error if you try.\n";

    // SAFE version: store the vector first
    auto data = make_data();
    auto safe_it = ranges::find(data, 8);
    if (safe_it != data.end()) {
        std::cout << "Safe: found " << *safe_it << "\n";
    }
    std::cout << "\n";
}

// ============================================================
// Case 2: View over a local that goes out of scope
// ============================================================
void case2_dangling_view() {
    std::cout << "=== Case 2: View over destroyed source ===\n";

    // This lambda creates a view over a local vector — DANGEROUS
    auto make_view = []() {
        std::vector<int> local = {1, 2, 3, 4, 5};
        return local | views::filter([](int x) { return x > 2; });
        // local is destroyed here. The returned view holds a dangling reference.
    };

    // The view exists, but iterating it is UB
    // auto v = make_view();
    // for (int x : v) { ... }  // UB! Reading freed memory.

    std::cout << "Creating a view over a local vector and returning it\n";
    std::cout << "is UNDEFINED BEHAVIOR. The vector is destroyed.\n";
    std::cout << "(We don't iterate to avoid actual UB in this demo.)\n\n";

    // SAFE pattern: return the container, then create the view
    auto make_data_safe = []() {
        return std::vector<int>{1, 2, 3, 4, 5};
    };
    auto data = make_data_safe();
    auto safe_view = data | views::filter([](int x) { return x > 2; });

    std::cout << "Safe: ";
    for (int x : safe_view) {
        std::cout << x << " ";
    }
    std::cout << "\n\n";
}

// ============================================================
// Case 3: Borrowed ranges — the safe exception
// ============================================================
void case3_borrowed_ranges() {
    std::cout << "=== Case 3: Borrowed Ranges (safe temporaries) ===\n";

    // Some types are "borrowed ranges" — safe to return iterators from temporaries.
    // std::span and std::string_view are borrowed ranges because they don't own data.

    // This is SAFE — string_view is a borrowed range
    std::string s = "hello world";
    auto it = ranges::find(std::string_view(s), 'w');
    // it is a real iterator, not dangling!
    if (it != std::string_view(s).end()) {
        std::cout << "Found '" << *it << "' in string_view (safe — borrowed range)\n";
    }

    // This would be DANGLING:
    // auto it2 = ranges::find(std::string("hello"), 'h');
    // it2 would be ranges::dangling because std::string is NOT a borrowed range.

    std::cout << "\nborrowed_range types return real iterators from temporaries.\n";
    std::cout << "Non-borrowed types return ranges::dangling.\n\n";
}

// ============================================================
// Case 4: Predict the type! (mental exercise)
// ============================================================
void case4_predict_types() {
    std::cout << "=== Case 4: Predict the Type ===\n\n";

    std::vector<int> v = {1, 2, 3};

    // Q1: What's the type of it1?
    auto it1 = ranges::find(v, 2);
    // A: std::vector<int>::iterator — v is an lvalue, so iterator is valid
    std::cout << "Q1: ranges::find(lvalue_vec, 2) → real iterator\n";
    std::cout << "    Found: " << *it1 << "\n";

    // Q2: What's the type of it2?
    auto it2 = ranges::find(make_data(), 2);
    // A: ranges::dangling — make_data() returns a temporary
    std::cout << "Q2: ranges::find(temporary_vec, 2) → ranges::dangling\n";
    (void)it2;  // suppress unused warning

    // Q3: What about sort?
    // ranges::sort(make_data());  // OK! sort returns a subrange,
    // but the temporary is dead after the statement. The SORT itself is fine
    // because it operates during the full-expression. The RESULT is useless.
    std::cout << "Q3: ranges::sort(temporary) → sorts then discards (legal but useless)\n";

    // Q4: min_element on temporary?
    auto it4 = ranges::min_element(make_data());
    // A: ranges::dangling
    std::cout << "Q4: ranges::min_element(temporary) → ranges::dangling\n";
    (void)it4;

    std::cout << "\nRule: If the source is a temporary (rvalue) and NOT a borrowed_range,\n";
    std::cout << "      any algorithm returning an iterator returns ranges::dangling instead.\n";
    std::cout << "      This is a COMPILE-TIME safety net, not a runtime check.\n";
}

int main() {
    case1_find_on_temporary();
    case2_dangling_view();
    case3_borrowed_ranges();
    case4_predict_types();
    return 0;
}
