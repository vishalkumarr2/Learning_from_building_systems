// Week 7 — Puzzle 2: Temporary Lifetime Extension Scenarios
// ==========================================================
// 5 scenarios — which are safe, which are dangling?
// Think about each one before reading the answer!
//
// Compile: g++ -std=c++20 -Wall -Wextra -O0 -g puzzle02_temporary_lifetime.cpp -o puzzle02
// With ASan: g++ -std=c++20 -fsanitize=address -O0 -g puzzle02_temporary_lifetime.cpp -o puzzle02_asan

#include <iostream>
#include <string>
#include <vector>

using namespace std::string_literals; // for "..."s

// Helper: returns string BY VALUE
std::string make_string() {
    return "from make_string";
}

// Helper: returns string BY CONST REFERENCE to static
const std::string& static_ref() {
    static std::string s = "static string";
    return s;
}

// Helper: returns string BY CONST REFERENCE to parameter
const std::string& identity_ref(const std::string& s) {
    return s; // Returns reference to whatever was passed in
}

// ============================================================
// Scenario 1: const ref to temporary from function returning by value
// ============================================================

void scenario_01() {
    std::cout << "Scenario 1: const string& r = make_string();\n";

    const std::string& r = make_string();
    // make_string() returns by VALUE → creates a temporary
    // Binding temporary to const lvalue reference EXTENDS its lifetime
    // to match the reference's scope.

    std::cout << "  r = \"" << r << "\"\n";
    std::cout << "  VERDICT: ✅ SAFE — lifetime extended to scope of r\n";
    std::cout << "  RULE: Binding a prvalue to a const& extends the temporary's lifetime.\n\n";
}

// ============================================================
// Scenario 2: auto& bound to string literal operator result
// ============================================================

void scenario_02() {
    std::cout << "Scenario 2: const auto& r = \"hello\"s;\n";

    const auto& r = "hello"s; // "hello"s creates a temporary std::string
    // Same as scenario 1: temporary bound to const ref → lifetime extended

    std::cout << "  r = \"" << r << "\"\n";
    std::cout << "  VERDICT: ✅ SAFE — temporary std::string lifetime extended\n";
    std::cout << "  NOTE: \"hello\"s is operator\"\"s() which returns std::string by value.\n\n";
}

// ============================================================
// Scenario 3: const ref to element of vector temporary
// ============================================================

void scenario_03() {
    std::cout << "Scenario 3: const auto& r = std::vector<int>{1,2,3}[0];\n";

    // This one is TRICKY!
    // std::vector<int>{1,2,3} creates a temporary vector.
    // operator[] returns a REFERENCE to an element inside that temporary.
    // Does the temporary vector's lifetime get extended?

    // In C++, lifetime extension only applies when binding DIRECTLY to a temporary.
    // Here, we're binding to a reference RETURNED BY a member function of a temporary.
    // The temporary vector is destroyed at the end of the full-expression.

    // HOWEVER: since C++23 (P2718R0), this is being addressed. Pre-C++23:
    const auto& r = std::vector<int>{1, 2, 3}[0];
    // The vector temporary MAY be destroyed before r is used.
    // In practice, many compilers extend the lifetime, but it's NOT guaranteed pre-C++23.

    std::cout << "  r = " << r << "\n";
    std::cout << "  VERDICT: ⚠️  DANGEROUS pre-C++23\n";
    std::cout << "  The vector temporary is destroyed at the semicolon.\n";
    std::cout << "  The reference to element [0] may dangle.\n";
    std::cout << "  FIX: auto v = std::vector<int>{1,2,3}; const auto& r = v[0];\n\n";

    // Note: GCC and Clang may NOT warn about this! ASan might not catch it either
    // because the vector's internal buffer may still be accessible (not yet reused).
}

// ============================================================
// Scenario 4: const ref from function returning const ref
// ============================================================

void scenario_04() {
    std::cout << "Scenario 4: const string& r = identity_ref(make_string());\n";

    // make_string() returns a temporary string.
    // identity_ref() takes it by const ref and returns that same const ref.
    // The temporary's lifetime:
    //   - Is it extended because we bind to const ref?
    //   - NO! Lifetime extension does NOT propagate through function calls.
    //   - The temporary dies at the end of the full-expression (the semicolon).

    const std::string& r = identity_ref(make_string());
    // r is now a DANGLING REFERENCE!
    // The temporary from make_string() was destroyed after the full-expression.

    // Reading r is UB. It might "work" because the memory hasn't been reused yet.
    // std::cout << "  r = \"" << r << "\"\n"; // UB! Don't do this!

    std::cout << "  VERDICT: ❌ DANGLING — temporary destroyed at semicolon\n";
    std::cout << "  RULE: Lifetime extension does NOT propagate through function returns.\n";
    std::cout << "  FIX: std::string r = identity_ref(make_string()); // copy\n\n";
}

// ============================================================
// Scenario 5: const ref to subobject of temporary
// ============================================================

struct Wrapper {
    std::string name;
    int value;

    const std::string& get_name() const { return name; }
};

Wrapper make_wrapper() {
    return Wrapper{"hello", 42};
}

void scenario_05() {
    std::cout << "Scenario 5: const string& r = make_wrapper().get_name();\n";

    // make_wrapper() returns a temporary Wrapper.
    // .get_name() returns a reference to its internal string member.
    // The temporary Wrapper (and its members) would normally die at the semicolon.

    // Does lifetime extension kick in?
    // NO — same rule as scenario 4. Lifetime extension doesn't propagate
    // through member function return values.

    const std::string& r = make_wrapper().get_name();
    // r is DANGLING — the Wrapper temporary (and its .name member) are destroyed.

    // Contrast with direct member access (C++20):
    // const auto& r2 = make_wrapper().name;  // This IS lifetime-extended!
    // Because .name is a direct member access, not a function call.

    // std::cout << "  r = \"" << r << "\"\n"; // UB!

    std::cout << "  VERDICT: ❌ DANGLING — member function return, no extension\n";
    std::cout << "  CONTRAST: const auto& r = make_wrapper().name; // ✅ SAFE\n";
    std::cout << "  Direct member access on temporary DOES extend lifetime.\n";
    std::cout << "  Member function returning ref does NOT.\n\n";

    // Safe version: direct member access
    const auto& safe_name = make_wrapper().name;
    std::cout << "  Direct member access: \"" << safe_name << "\" ✅\n\n";
}

// ============================================================
// Summary
// ============================================================

void summary() {
    std::cout << "=== LIFETIME EXTENSION RULES SUMMARY ===\n\n";
    std::cout << "✅ EXTENDS lifetime:\n";
    std::cout << "   const T& r = <temporary>;          // Direct binding\n";
    std::cout << "   const auto& r = \"hello\"s;           // Direct binding to prvalue\n";
    std::cout << "   const auto& r = Wrapper{}.name;     // Direct member access (C++20)\n";
    std::cout << "   T&& r = <temporary>;                // Rvalue reference binding\n\n";

    std::cout << "❌ Does NOT extend lifetime:\n";
    std::cout << "   const T& r = func_returning_ref();  // Through function return\n";
    std::cout << "   const T& r = obj.method();          // Through member function return\n";
    std::cout << "   const T& r = temp_vec[0];           // Through operator[] (pre-C++23)\n";
    std::cout << "   const T& r = static_cast<const T&>(temp);  // Through cast\n\n";

    std::cout << "RULE OF THUMB:\n";
    std::cout << "  If in doubt, COPY the value (auto r = ...) instead of using const&.\n";
    std::cout << "  The cost of a copy is almost always less than the cost of a dangling ref bug.\n";
}

// ============================================================

int main() {
    std::cout << "=== PUZZLE: TEMPORARY LIFETIME SCENARIOS ===\n";
    std::cout << "Think about each scenario before reading the verdict!\n";
    std::cout << "==============================================\n\n";

    scenario_01();
    scenario_02();
    scenario_03();
    scenario_04();
    scenario_05();

    summary();

    return 0;
}
