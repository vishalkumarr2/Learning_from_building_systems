// =============================================================================
// 💀 Puzzle 2: The vector<unique_ptr> Copy Puzzle
// =============================================================================
// QUESTION: Does this compile? If not, which line fails? What happens?
//
// Build: g++ -std=c++2a -Wall -Wextra -Wpedantic puzzle02_moved_from_vector.cpp
// =============================================================================

#include <iostream>
#include <memory>
#include <vector>

int main() {
    // --- Setup ---
    std::vector<std::unique_ptr<int>> v;
    v.push_back(std::make_unique<int>(10));
    v.push_back(std::make_unique<int>(20));
    v.push_back(std::make_unique<int>(30));

    std::cout << "Original vector (size=" << v.size() << "):\n";
    for (const auto& ptr : v) {
        std::cout << "  " << (ptr ? std::to_string(*ptr) : "null") << "\n";
    }

    // --- QUESTION 1: Does this compile? ---
    // auto v2 = v;  // UNCOMMENT TO TEST — this is a COPY

    // --- What DOES work: MOVE ---
    auto v2 = std::move(v);

    std::cout << "\nAfter move:\n";
    std::cout << "v2 (moved-to, size=" << v2.size() << "):\n";
    for (const auto& ptr : v2) {
        std::cout << "  " << (ptr ? std::to_string(*ptr) : "null") << "\n";
    }

    // --- QUESTION 2: What does v look like now? ---
    std::cout << "\nv (moved-from, size=" << v.size() << "):\n";
    for (const auto& ptr : v) {
        std::cout << "  " << (ptr ? std::to_string(*ptr) : "null") << "\n";
    }

    // --- QUESTION 3: Is v usable? ---
    std::cout << "\nIs moved-from vector usable?\n";
    std::cout << "  v.empty() = " << (v.empty() ? "true" : "false") << "\n";
    std::cout << "  v.size()  = " << v.size() << "\n";

    // Can we push to it?
    v.push_back(std::make_unique<int>(99));
    std::cout << "  After push_back(99): size=" << v.size() << "\n";
    std::cout << "  v[0] = " << *v[0] << "\n";

    return 0;
}

// =============================================================================
// ANSWERS
// =============================================================================
//
// QUESTION 1: Does `auto v2 = v;` compile?
//   NO. It's a compile error.
//   std::unique_ptr is NOT copyable (copy constructor is deleted).
//   std::vector<T>'s copy constructor requires T to be copyable.
//   Error: "use of deleted function 'unique_ptr(const unique_ptr&)'"
//
// QUESTION 2: What does v look like after `auto v2 = std::move(v)`?
//   v is in a valid-but-unspecified state.
//   In practice (on all major STL implementations), v is EMPTY:
//     v.size() == 0
//     v.empty() == true
//   This is because vector's move constructor transfers ownership of the
//   internal heap buffer, leaving the source with a null pointer.
//
//   But the STANDARD only guarantees "valid but unspecified." You should
//   not write code that depends on v being empty after a move.
//
// QUESTION 3: Is v usable after the move?
//   YES. You can:
//   - Destroy it (dtor works)
//   - Assign to it
//   - Call size(), empty()
//   - Push new elements to it
//   You CANNOT:
//   - Assume it's empty (even though it is in practice)
//   - Read elements that "should" still be there
//
// KEY TAKEAWAY:
//   unique_ptr enforces single ownership at the type system level.
//   You can't accidentally copy a vector of unique_ptrs — the compiler
//   stops you. This forces you to think about ownership transfer (move)
//   vs shared access (shared_ptr, raw pointer, reference).
// =============================================================================
