// =============================================================================
// 💀 Puzzle 5: Concept Subsumption and Overload Resolution
// =============================================================================
// QUESTION: For each call, which overload is selected? Why?
//
// Build: g++ -std=c++2a -fconcepts -Wall -Wextra -Wpedantic puzzle05_concept_overload.cpp
// =============================================================================

#include <iostream>
#include <type_traits>

// --- Concept hierarchy ---
// Arithmetic ← Integral ← SignedIntegral
//                       ← UnsignedIntegral

template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template<typename T>
concept Integral = Arithmetic<T> && std::is_integral_v<T>;

template<typename T>
concept SignedIntegral = Integral<T> && std::is_signed_v<T>;

template<typename T>
concept UnsignedIntegral = Integral<T> && std::is_unsigned_v<T>;

template<typename T>
concept FloatingPoint = Arithmetic<T> && std::is_floating_point_v<T>;

// --- Overload set ---
// All five overloads are in scope. Which one wins for each call?
// NOTE: GCC 9 lacks abbreviated function templates (Concept auto param).
// Using explicit template<> with requires clause instead.

template<typename T>
std::string describe(T x) {
    (void)x;
    return "1: unconstrained";
}

template<typename T> requires Arithmetic<T>
std::string describe(T x) {
    (void)x;
    return "2: Arithmetic";
}

template<typename T> requires Integral<T>
std::string describe(T x) {
    (void)x;
    return "3: Integral";
}

template<typename T> requires SignedIntegral<T>
std::string describe(T x) {
    (void)x;
    return "4: SignedIntegral";
}

template<typename T> requires FloatingPoint<T>
std::string describe(T x) {
    (void)x;
    return "5: FloatingPoint";
}

int main() {
    // =========================================================================
    // CALL 1: signed integer (int)
    // Satisfies: unconstrained, Arithmetic, Integral, SignedIntegral
    // Which wins?
    // =========================================================================
    std::cout << "describe(42)        → " << describe(42) << "\n";

    // =========================================================================
    // CALL 2: unsigned integer (unsigned int)
    // Satisfies: unconstrained, Arithmetic, Integral
    // (NOT SignedIntegral, NOT FloatingPoint)
    // Which wins?
    // =========================================================================
    std::cout << "describe(42u)       → " << describe(42u) << "\n";

    // =========================================================================
    // CALL 3: double
    // Satisfies: unconstrained, Arithmetic, FloatingPoint
    // (NOT Integral, NOT SignedIntegral)
    // Which wins?
    // =========================================================================
    std::cout << "describe(3.14)      → " << describe(3.14) << "\n";

    // =========================================================================
    // CALL 4: bool
    // Satisfies: unconstrained, Arithmetic, Integral, UnsignedIntegral
    // (bool is integral and unsigned in C++!)
    // Which wins?
    // =========================================================================
    std::cout << "describe(true)      → " << describe(true) << "\n";

    // =========================================================================
    // CALL 5: char (signed on most platforms)
    // Satisfies: unconstrained, Arithmetic, Integral, SignedIntegral
    // Which wins?
    // =========================================================================
    std::cout << "describe('A')       → " << describe('A') << "\n";

    // =========================================================================
    // CALL 6: string literal (const char*)
    // Satisfies: only unconstrained
    // Which wins?
    // =========================================================================
    std::cout << "describe(\"hello\")   → " << describe("hello") << "\n";

    // =========================================================================
    // CALL 7: float
    // =========================================================================
    std::cout << "describe(1.0f)      → " << describe(1.0f) << "\n";

    // =========================================================================
    // CALL 8: long long (signed)
    // =========================================================================
    std::cout << "describe(100LL)     → " << describe(100LL) << "\n";

    return 0;
}

// =============================================================================
// ANSWERS
// =============================================================================
//
// CALL 1: describe(42) → "4: SignedIntegral"
//   42 is int (signed integral).
//   Matches: unconstrained, Arithmetic, Integral, SignedIntegral.
//   SignedIntegral subsumes Integral (it requires everything Integral does,
//   plus is_signed). Integral subsumes Arithmetic. So SignedIntegral is the
//   MOST CONSTRAINED — it wins.
//
// CALL 2: describe(42u) → "3: Integral"
//   42u is unsigned int.
//   Matches: unconstrained, Arithmetic, Integral.
//   Does NOT match SignedIntegral (not signed) or FloatingPoint.
//   Integral subsumes Arithmetic. So Integral wins.
//   (Note: UnsignedIntegral is defined as a concept but has no overload.)
//
// CALL 3: describe(3.14) → "5: FloatingPoint"
//   3.14 is double.
//   Matches: unconstrained, Arithmetic, FloatingPoint.
//   FloatingPoint subsumes Arithmetic (requires Arithmetic + is_floating_point).
//   FloatingPoint wins.
//
// CALL 4: describe(true) → "3: Integral"
//   bool is integral and unsigned in C++.
//   Matches: unconstrained, Arithmetic, Integral.
//   Does NOT match SignedIntegral (bool is unsigned).
//   Integral subsumes Arithmetic. Integral wins.
//
// CALL 5: describe('A') → "4: SignedIntegral"
//   char is typically signed on x86 Linux (implementation-defined).
//   If signed: matches SignedIntegral (most constrained) → "4".
//   If unsigned (some ARM platforms): would match only Integral → "3".
//   On x86-64 Linux with GCC/Clang: "4: SignedIntegral".
//
// CALL 6: describe("hello") → "1: unconstrained"
//   "hello" is const char[6] → decays to const char*.
//   is_arithmetic_v<const char*> is false.
//   Only the unconstrained overload matches.
//
// CALL 7: describe(1.0f) → "5: FloatingPoint"
//   float is floating point. Same logic as CALL 3.
//
// CALL 8: describe(100LL) → "4: SignedIntegral"
//   long long is signed integral. Same logic as CALL 1.
//
// =============================================================================
// HOW SUBSUMPTION WORKS
// =============================================================================
//
// The compiler decomposes concepts into "atomic constraints" and checks if
// one set of constraints is a SUBSET of another.
//
// SignedIntegral = Arithmetic<T> && is_integral_v<T> && is_signed_v<T>
// Integral       = Arithmetic<T> && is_integral_v<T>
// Arithmetic     = is_arithmetic_v<T>
//
// SignedIntegral ⊃ Integral ⊃ Arithmetic ⊃ unconstrained
//
// The "most constrained" viable overload wins. If there's no single most
// constrained candidate, it's ambiguous (compile error).
//
// IMPORTANT: Subsumption only works with NAMED CONCEPTS, not arbitrary
// requires clauses. Two inline requires expressions are never considered
// to subsume each other, even if one is logically stronger.
// =============================================================================
