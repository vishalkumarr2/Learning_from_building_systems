// =============================================================================
// 💀 Puzzle 1: The Named Rvalue Reference is an Lvalue
// =============================================================================
// 99% of C++ developers get this wrong on first encounter.
//
// QUESTION: What does this program print?
// Think carefully before scrolling to the answer.
//
// Build: g++ -std=c++2a -Wall -Wextra -Wpedantic puzzle01_named_rref.cpp
// =============================================================================

#include <iostream>

void f(int& /*x*/)  { std::cout << "lvalue overload\n"; }
void f(int&& /*x*/) { std::cout << "rvalue overload\n"; }

template<typename T>
void g(T&& x) {
    // x is declared as T&& — a forwarding reference.
    // But what is the value category of the EXPRESSION "x" here?
    f(x);
}

int main() {
    int a = 1;

    std::cout << "g(a):   ";
    g(a);    // What gets printed?

    std::cout << "g(1):   ";
    g(1);    // What gets printed? THIS ONE surprises people.

    std::cout << "g(std::move(a)): ";
    g(std::move(a));  // And this one?
}

// =============================================================================
// ANSWER — Don't peek until you've predicted all three!
// =============================================================================
//
// g(a):              prints "lvalue overload"
// g(1):              prints "lvalue overload"  ← THE SURPRISE!
// g(std::move(a)):   prints "lvalue overload"  ← Also lvalue!
//
// =============================================================================
// EXPLANATION
// =============================================================================
//
// Case 1: g(a)
//   - a is an lvalue (int)
//   - T is deduced as int& (forwarding reference rule for lvalues)
//   - x has type int& && → int& (reference collapsing)
//   - f(x) calls f(int&) — x is an lvalue. ✓ Expected.
//
// Case 2: g(1)
//   - 1 is a prvalue (int)
//   - T is deduced as int
//   - x has type int&&
//   - BUT: x has a NAME → it's an LVALUE expression
//   - f(x) calls f(int&) — x is an lvalue even though its TYPE is int&&
//
//   The key insight: the TYPE of x (int&&) and the VALUE CATEGORY of the
//   expression "x" are DIFFERENT things.
//   - Type: int&& (rvalue reference)
//   - Value category: lvalue (it has a name, you can take &x)
//
// Case 3: g(std::move(a))
//   - std::move(a) is an xvalue
//   - T is deduced as int
//   - x has type int&&
//   - Same as case 2: x has a name → lvalue
//   - f(x) calls f(int&)
//
// =============================================================================
// THE FIX: Use std::forward to restore the value category
// =============================================================================
//
// template<typename T>
// void g_correct(T&& x) {
//     f(std::forward<T>(x));  // preserves the original value category
// }
//
// g_correct(a):            → f(int&)   — lvalue preserved
// g_correct(1):            → f(int&&)  — rvalue preserved!
// g_correct(std::move(a)): → f(int&&)  — rvalue preserved!
//
// This is WHY std::forward exists. Without it, perfect forwarding is broken.
// =============================================================================
