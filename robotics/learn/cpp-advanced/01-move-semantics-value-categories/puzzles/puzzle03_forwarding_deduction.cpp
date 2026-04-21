// =============================================================================
// 💀 Puzzle 3: Forwarding Reference Deduction
// =============================================================================
// QUESTION: For each call to f(T&& x), what is T deduced as?
//           And what is the type of x after reference collapsing?
//
// Fill in your predictions before running.
//
// Build: g++ -std=c++2a -Wall -Wextra -Wpedantic puzzle03_forwarding_deduction.cpp
// =============================================================================

#include <iostream>
#include <string>
#include <type_traits>

// This helper prints what T was deduced as and what x's type is.
template<typename T>
void f(T&& x) {
    // What is T?
    if constexpr (std::is_lvalue_reference_v<T>) {
        using inner = std::remove_reference_t<T>;
        if constexpr (std::is_const_v<inner>)
            std::cout << "T = const int&";
        else
            std::cout << "T = int&      ";
    } else {
        std::cout << "T = int       ";
    }

    // What is x's type? (T&& after reference collapsing)
    using x_type = decltype(x);
    if constexpr (std::is_lvalue_reference_v<x_type>) {
        using inner = std::remove_reference_t<x_type>;
        if constexpr (std::is_const_v<inner>)
            std::cout << "  x type: const int&";
        else
            std::cout << "  x type: int&      ";
    } else if constexpr (std::is_rvalue_reference_v<x_type>) {
        std::cout << "  x type: int&&     ";
    }

    // What is x's VALUE CATEGORY? (always lvalue — it has a name!)
    static_assert(std::is_lvalue_reference_v<decltype((x))>,
        "x is always an lvalue, regardless of its declared type");
    std::cout << "  x category: lvalue (always)\n";
}

int main() {
    int a = 1;
    const int b = 2;
    int& c = a;

    // =========================================================================
    // CASE 1: f(a) — passing an lvalue (int)
    // Predict: T = ???    x type = ???
    // =========================================================================
    std::cout << "f(a):            ";
    f(a);

    // =========================================================================
    // CASE 2: f(b) — passing a const lvalue (const int)
    // Predict: T = ???    x type = ???
    // =========================================================================
    std::cout << "f(b):            ";
    f(b);

    // =========================================================================
    // CASE 3: f(c) — passing an lvalue reference (int&)
    // Predict: T = ???    x type = ???
    // =========================================================================
    std::cout << "f(c):            ";
    f(c);

    // =========================================================================
    // CASE 4: f(42) — passing a prvalue (int literal)
    // Predict: T = ???    x type = ???
    // =========================================================================
    std::cout << "f(42):           ";
    f(42);

    // =========================================================================
    // CASE 5: f(std::move(a)) — passing an xvalue
    // Predict: T = ???    x type = ???
    // =========================================================================
    std::cout << "f(std::move(a)): ";
    f(std::move(a));

    return 0;
}

// =============================================================================
// ANSWERS
// =============================================================================
//
// CASE 1: f(a)
//   a is lvalue (int)
//   T deduced as: int&
//   x type: int& && → int&  (reference collapsing: & + && = &)
//   x value category: lvalue
//
// CASE 2: f(b)
//   b is lvalue (const int)
//   T deduced as: const int&
//   x type: const int& && → const int&  (const preserved through collapsing)
//   x value category: lvalue
//
// CASE 3: f(c)
//   c is lvalue (int& — but that's just its declared type)
//   T deduced as: int&  (same as case 1, references are transparent to deduction)
//   x type: int& && → int&
//   x value category: lvalue
//
// CASE 4: f(42)
//   42 is prvalue (int)
//   T deduced as: int  (no reference — it's an rvalue)
//   x type: int&& (T&& where T=int → int&&)
//   x value category: lvalue  ← THIS is the key insight from puzzle 1
//
// CASE 5: f(std::move(a))
//   std::move(a) is xvalue (int&&)
//   T deduced as: int  (same as prvalue — both are rvalues)
//   x type: int&&
//   x value category: lvalue  ← same as case 4
//
// =============================================================================
// SUMMARY TABLE
// =============================================================================
//
//   Argument        | T deduced as | x type    | x category
//   ────────────────┼──────────────┼───────────┼──────────
//   a  (lvalue)     | int&         | int&      | lvalue
//   b  (const lval) | const int&   | const int&| lvalue
//   c  (int& lval)  | int&         | int&      | lvalue
//   42 (prvalue)    | int          | int&&     | lvalue
//   move(a) (xval)  | int          | int&&     | lvalue
//
// KEY RULE: x's value category is ALWAYS lvalue because it has a name.
//           Only std::forward<T>(x) can recover the original category.
// =============================================================================
