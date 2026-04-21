// puzzle02_optional_gotcha.cpp
// Compile: g++ -std=c++20 -Wall -Wextra -Wpedantic -pthread -O2 puzzle02_optional_gotcha.cpp -o puzzle02
//
// PUZZLE: For each scenario below, predict what is printed.
//         optional<bool> is a well-known footgun.

#include <iostream>
#include <optional>
#include <functional>  // reference_wrapper
#include <string>

void scenario_1() {
    std::cout << "--- Scenario 1: optional<bool> with false ---\n";
    std::optional<bool> ob = false;

    if (ob)
        std::cout << "A: ob is truthy\n";
    else
        std::cout << "A: ob is falsy\n";

    if (*ob)
        std::cout << "B: *ob is true\n";
    else
        std::cout << "B: *ob is false\n";

    if (ob.value())
        std::cout << "C: ob.value() is true\n";
    else
        std::cout << "C: ob.value() is false\n";
}

void scenario_2() {
    std::cout << "\n--- Scenario 2: optional<bool> default vs nullopt ---\n";
    std::optional<bool> a;             // no value
    std::optional<bool> b = false;     // has value: false
    std::optional<bool> c = true;      // has value: true
    std::optional<bool> d = std::nullopt;  // no value

    std::cout << "a has_value: " << a.has_value() << ", bool(a): " << static_cast<bool>(a) << "\n";
    std::cout << "b has_value: " << b.has_value() << ", bool(b): " << static_cast<bool>(b) << "\n";
    std::cout << "c has_value: " << c.has_value() << ", bool(c): " << static_cast<bool>(c) << "\n";
    std::cout << "d has_value: " << d.has_value() << ", bool(d): " << static_cast<bool>(d) << "\n";
}

void scenario_3() {
    std::cout << "\n--- Scenario 3: value_or with optional<bool> ---\n";
    std::optional<bool> empty;
    std::optional<bool> has_false = false;

    // What does value_or return for each?
    bool r1 = empty.value_or(true);
    bool r2 = has_false.value_or(true);

    std::cout << "empty.value_or(true)     = " << std::boolalpha << r1 << "\n";
    std::cout << "has_false.value_or(true)  = " << std::boolalpha << r2 << "\n";
}

void scenario_4() {
    std::cout << "\n--- Scenario 4: optional<reference_wrapper<int>> ---\n";
    // std::optional<int&> doesn't compile before C++26.
    // Workaround: optional<reference_wrapper<int>>

    int x = 42;
    std::optional<std::reference_wrapper<int>> opt_ref = std::ref(x);

    // Modify through the optional
    opt_ref->get() = 99;
    std::cout << "x after modification through opt_ref: " << x << "\n";

    // Reset the optional — does NOT destroy x
    opt_ref = std::nullopt;
    std::cout << "x after opt_ref reset: " << x << "\n";
    std::cout << "opt_ref has_value: " << opt_ref.has_value() << "\n";
}

void scenario_5() {
    std::cout << "\n--- Scenario 5: comparison traps ---\n";
    std::optional<int> a = 0;
    std::optional<int> b;       // nullopt
    std::optional<int> c = 0;

    // What do these comparisons yield?
    std::cout << "(a == 0):        " << (a == 0) << "\n";
    std::cout << "(b == 0):        " << (b == 0) << "\n";
    std::cout << "(a == b):        " << (a == b) << "\n";
    std::cout << "(a == c):        " << (a == c) << "\n";
    std::cout << "(b < a):         " << (b < a) << "\n";   // nullopt < any value
    std::cout << "(b == std::nullopt): " << (b == std::nullopt) << "\n";
    std::cout << "(a == std::nullopt): " << (a == std::nullopt) << "\n";

    // The subtle one:
    std::optional<int> zero = 0;
    if (zero)
        std::cout << "optional<int>{0} is truthy (has value)\n";
    else
        std::cout << "optional<int>{0} is falsy\n";
}

int main() {
    scenario_1();
    scenario_2();
    scenario_3();
    scenario_4();
    scenario_5();
}

/*
 * ═══════════════════════════════════════════════════════
 * ANSWER
 * ═══════════════════════════════════════════════════════
 *
 * --- Scenario 1 ---
 * A: ob is truthy          <-- SURPRISE! optional's bool conversion checks has_value(), NOT the contained bool!
 * B: *ob is false           <-- dereferencing gives the actual bool value
 * C: ob.value() is false    <-- same as *ob
 *
 * KEY INSIGHT: `if (ob)` asks "does ob contain a value?" — answer is YES (it contains false).
 *             `if (*ob)` asks "what is the contained value?" — answer is false.
 *
 * --- Scenario 2 ---
 * a has_value: 0, bool(a): 0    <-- default constructed = empty
 * b has_value: 1, bool(b): 1    <-- contains false, but HAS a value → truthy!
 * c has_value: 1, bool(c): 1    <-- contains true, has a value → truthy
 * d has_value: 0, bool(d): 0    <-- explicit nullopt = empty
 *
 * --- Scenario 3 ---
 * empty.value_or(true)     = true     <-- no value, returns default
 * has_false.value_or(true)  = false    <-- HAS a value (false), returns it, ignores default
 *
 * --- Scenario 4 ---
 * x after modification through opt_ref: 99   <-- reference_wrapper acts as a reference
 * x after opt_ref reset: 99                    <-- x survives; optional doesn't own it
 * opt_ref has_value: 0
 *
 * Note: optional<int&> is coming in C++26 (P2988). Until then, use reference_wrapper.
 *
 * --- Scenario 5 ---
 * (a == 0):        1     <-- optional<int>{0} == 0: compares contained value
 * (b == 0):        0     <-- nullopt != any value
 * (a == b):        0     <-- engaged != disengaged
 * (a == c):        1     <-- both engaged, same value
 * (b < a):         1     <-- nullopt is "less than" any engaged optional
 * (b == nullopt):  1
 * (a == nullopt):  0     <-- a has a value
 * optional<int>{0} is truthy (has value)   <-- same trap: conversion checks has_value, not contained value
 *
 * RULE: Never use `if (opt)` when T is bool or int and you care about
 *       the contained value. Use `if (opt.has_value() && *opt)` or
 *       `if (opt.value_or(false))` for optional<bool>.
 */
