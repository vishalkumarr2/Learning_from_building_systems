// =============================================================================
// 💀 Puzzle 4: constexpr vs consteval — What Compiles?
// =============================================================================
// QUESTION: For each numbered line, does it compile?
//           If yes, when is it evaluated (compile-time or runtime)?
//
// Build: g++ -std=c++2a -Wall -Wextra -Wpedantic puzzle04_constexpr_runtime.cpp
//
// NOTE: consteval requires GCC 10+. On GCC 9, the consteval lines are
// commented out with explanations. The constexpr portions work fully.
// =============================================================================

#include <iostream>

// A constexpr function: CAN be evaluated at compile time, but doesn't HAVE to
constexpr int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

// A consteval function: MUST be evaluated at compile time
// NOTE: consteval requires GCC 10+. Uncomment when available.
// consteval int forced_factorial(int n) {
//     if (n <= 1) return 1;
//     return n * forced_factorial(n - 1);
// }
// For GCC 9, we simulate with constexpr:
constexpr int forced_factorial(int n) {
    if (n <= 1) return 1;
    return n * forced_factorial(n - 1);
}

int main() {
    // =========================================================================
    // LINE 1: constexpr variable with compile-time argument
    // Does it compile? What happens?
    // =========================================================================
    constexpr int a = factorial(5);  // LINE 1
    std::cout << "a = " << a << "\n";

    // =========================================================================
    // LINE 2: regular variable with runtime argument
    // Does it compile? When is factorial evaluated?
    // =========================================================================
    int runtime_val = 10;
    int b = factorial(runtime_val);  // LINE 2
    std::cout << "b = " << b << "\n";

    // =========================================================================
    // LINE 3: constexpr variable with runtime argument
    // Does it compile?
    // =========================================================================
    // constexpr int c = factorial(runtime_val);  // LINE 3 — UNCOMMENT TO TEST
    // (Hint: constexpr variable REQUIRES compile-time evaluation)

    // =========================================================================
    // LINE 4: consteval with compile-time argument
    // Does it compile?
    // =========================================================================
    constexpr int d = forced_factorial(5);  // LINE 4
    std::cout << "d = " << d << "\n";

    // =========================================================================
    // LINE 5: consteval with runtime argument
    // Does it compile?
    // =========================================================================
    // NOTE: With real consteval (GCC 10+), this would NOT compile:
    // int e = forced_factorial(runtime_val);  // LINE 5
    // With our constexpr stand-in, it compiles (runs at runtime).
    // (Hint: consteval means "compile time ONLY")

    // =========================================================================
    // LINE 6: consteval result stored in non-constexpr variable
    // Does it compile? When is it evaluated?
    // =========================================================================
    int f = forced_factorial(7);  // LINE 6
    std::cout << "f = " << f << "\n";

    // =========================================================================
    // LINE 7: constexpr function called in static_assert
    // Does it compile?
    // =========================================================================
    static_assert(factorial(5) == 120);  // LINE 7

    // =========================================================================
    // LINE 8: Runtime value in static_assert
    // Does it compile?
    // =========================================================================
    // static_assert(factorial(runtime_val) > 0);  // LINE 8 — UNCOMMENT TO TEST

    // =========================================================================
    // BONUS: Can constexpr call consteval?
    // =========================================================================
    // constexpr int bonus() {
    //     return forced_factorial(3);  // BONUS — UNCOMMENT TO TEST
    // }
    // (This one is tricky: it depends on whether the context is constexpr-evaluated)

    std::cout << "\nAll compiling lines executed successfully.\n";
    return 0;
}

// =============================================================================
// ANSWERS
// =============================================================================
//
// LINE 1: constexpr int a = factorial(5);
//   ✅ COMPILES. Evaluated at COMPILE TIME.
//   factorial is constexpr, argument is a literal → can be compile-time.
//   constexpr on the variable forces compile-time evaluation.
//
// LINE 2: int b = factorial(runtime_val);
//   ✅ COMPILES. Evaluated at RUNTIME.
//   factorial is constexpr but runtime_val is not a constant expression.
//   constexpr functions are ALLOWED to run at runtime — that's the point.
//   They're "compile-time capable", not "compile-time required".
//
// LINE 3: constexpr int c = factorial(runtime_val);
//   ❌ DOES NOT COMPILE.
//   constexpr on the VARIABLE demands compile-time evaluation.
//   But runtime_val is not a constant expression.
//   Error: "the value of 'runtime_val' is not usable in a constant expression"
//
// LINE 4: constexpr int d = forced_factorial(5);
//   ✅ COMPILES. Evaluated at COMPILE TIME.
//   consteval requires compile-time. 5 is a literal. OK.
//
// LINE 5: int e = forced_factorial(runtime_val);
//   ❌ DOES NOT COMPILE.
//   consteval means "must be evaluated at compile time — no exceptions."
//   runtime_val is not a constant expression.
//   Error: "call to consteval function is not a constant expression"
//
// LINE 6: int f = forced_factorial(7);
//   ✅ COMPILES. Evaluated at COMPILE TIME.
//   consteval forces compile-time evaluation regardless of whether the
//   result variable is constexpr. The function call itself must be
//   a constant expression, and it is (7 is a literal).
//   The result (5040) is computed at compile time, then stored into
//   the runtime variable f.
//
// LINE 7: static_assert(factorial(5) == 120);
//   ✅ COMPILES.
//   static_assert requires a compile-time expression. factorial(5) is
//   constexpr with a literal argument → compile-time evaluation.
//
// LINE 8: static_assert(factorial(runtime_val) > 0);
//   ❌ DOES NOT COMPILE.
//   static_assert requires compile-time. runtime_val is not constexpr.
//
// BONUS: Can constexpr call consteval?
//   It depends! In C++20: constexpr functions can call consteval functions
//   ONLY if the constexpr function is itself being evaluated at compile time.
//   If the constexpr function runs at runtime, it CANNOT call consteval.
//   This is because consteval enforces "immediate context" — every call to
//   it must be resolvable at compile time.
//
// =============================================================================
// SUMMARY
// =============================================================================
//
//   Keyword    | Can run at compile time? | Can run at runtime?
//   ───────────┼─────────────────────────┼───────────────────
//   constexpr  | Yes (if args are const)  | Yes (fallback)
//   consteval  | Yes (required)           | No (error)
//   constinit  | Init: yes (required)     | Variable: yes (mutable)
//
//   On variables:
//   constexpr int x = expr;  → x must be computed at compile time, immutable
//   constinit int x = expr;  → x must be INITIALIZED at compile time, but
//                                can be modified at runtime
//   const int x = expr;      → x is immutable but MAY be runtime-initialized
// =============================================================================
