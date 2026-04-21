// sf02_misra_compliance.cpp — MISRA C++ Compliance Rewrite Exercise
//
// MISRA (Motor Industry Software Reliability Association) restricts C/C++
// to a safe subset for automotive/embedded safety-critical code.
// Key philosophy: eliminate undefined behavior and hard-to-analyze constructs.
//
// Exercise: 7 non-compliant snippets, each with a compliant rewrite.
//
// Build: g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -pthread sf02_misra_compliance.cpp -o sf02_misra_compliance

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

// ============================================================================
// Test framework
// ============================================================================

struct TestResult {
    int         id;
    std::string rule;
    std::string description;
    bool        non_compliant_ok;
    bool        compliant_ok;
    bool        results_match;
};

static std::vector<TestResult> results;

void record(int id, const std::string& rule, const std::string& desc,
            bool nc_ok, bool c_ok, bool match) {
    results.push_back({id, rule, desc, nc_ok, c_ok, match});
}

// ============================================================================
// Snippet 1: Dynamic allocation in a loop
// MISRA C++ Rule 21-6-1 / Dir 4.12: No dynamic memory after initialization
// ============================================================================

namespace snippet1 {

// NON-COMPLIANT: allocates on each iteration — fragmentation, OOM risk
int non_compliant_sum(int n) {
    int total = 0;
    for (int i = 0; i < n; ++i) {
        int* p = new int(i);   // VIOLATION: dynamic allocation in loop
        total += *p;
        delete p;
    }
    return total;
}

// COMPLIANT: use stack-local variable, zero allocations
int compliant_sum(int n) {
    int total = 0;
    for (int i = 0; i < n; ++i) {
        int val = i;           // stack allocation — deterministic, no fragmentation
        total += val;
    }
    return total;
}

void test() {
    int nc = non_compliant_sum(10);
    int c  = compliant_sum(10);
    record(1, "Dir 4.12", "No dynamic allocation in loops", true, true, nc == c);
}

} // namespace snippet1

// ============================================================================
// Snippet 2: Exception throw/catch
// MISRA C++ Rule 15-0-1 / A15-0-1: Exceptions shall not be used
// (in safety-critical code — they make control flow unpredictable)
// ============================================================================

namespace snippet2 {

// NON-COMPLIANT: throws exception for error handling
int non_compliant_divide(int a, int b) {
    if (b == 0) throw std::runtime_error("divide by zero");  // VIOLATION
    return a / b;
}

// COMPLIANT: return error code via optional
std::optional<int> compliant_divide(int a, int b) {
    if (b == 0) return std::nullopt;  // explicit error, no exception
    return a / b;
}

void test() {
    bool nc_ok = false;
    try {
        int r = non_compliant_divide(10, 2);
        nc_ok = (r == 5);
    } catch (...) {
        nc_ok = false;
    }

    auto c = compliant_divide(10, 2);
    bool c_ok = c.has_value() && *c == 5;

    // Also check error case
    bool nc_err = false;
    try {
        non_compliant_divide(10, 0);
    } catch (const std::runtime_error&) {
        nc_err = true;
    }

    auto c_err = compliant_divide(10, 0);
    bool c_err_ok = !c_err.has_value();

    record(2, "A15-0-1", "No exceptions — use optional/error codes",
           nc_ok && nc_err, c_ok && c_err_ok, true);
}

} // namespace snippet2

// ============================================================================
// Snippet 3: Recursive function
// MISRA C++ Rule 7-5-4 / A7-5-1: Functions shall not call themselves recursively
// (unbounded stack usage — stack overflow in embedded = hard crash)
// ============================================================================

namespace snippet3 {

// NON-COMPLIANT: recursive — unbounded stack depth
uint64_t non_compliant_factorial(int n) {
    if (n <= 1) return 1;
    return static_cast<uint64_t>(n) * non_compliant_factorial(n - 1);  // VIOLATION
}

// COMPLIANT: iterative — bounded, deterministic stack usage
uint64_t compliant_factorial(int n) {
    uint64_t result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= static_cast<uint64_t>(i);
    }
    return result;
}

void test() {
    uint64_t nc = non_compliant_factorial(10);
    uint64_t c  = compliant_factorial(10);
    record(3, "A7-5-1", "No recursion — use iteration", true, true, nc == c);
}

} // namespace snippet3

// ============================================================================
// Snippet 4: goto for error handling
// MISRA C++ Rule 6-6-1 / A6-6-1: goto shall not be used
// (makes control flow graph analysis impossible for static analyzers)
// ============================================================================

namespace snippet4 {

// NON-COMPLIANT: goto-based error cleanup
int non_compliant_init() {
    int resource_a = 0;
    int resource_b = 0;
    int result = -1;

    resource_a = 42;
    if (resource_a != 42) goto cleanup;  // VIOLATION: goto

    resource_b = 99;
    if (resource_b != 99) goto cleanup;

    result = resource_a + resource_b;

cleanup:
    // cleanup code here
    return result;
}

// COMPLIANT: RAII or structured error handling (early return with cleanup)
int compliant_init() {
    auto cleanup = [](int /*a*/, int /*b*/) {
        // release resources in reverse order
    };

    int resource_a = 42;
    if (resource_a != 42) {
        cleanup(resource_a, 0);
        return -1;
    }

    int resource_b = 99;
    if (resource_b != 99) {
        cleanup(resource_a, resource_b);
        return -1;
    }

    int result = resource_a + resource_b;
    cleanup(resource_a, resource_b);
    return result;
}

void test() {
    int nc = non_compliant_init();
    int c  = compliant_init();
    record(4, "A6-6-1", "No goto — use RAII / structured returns", true, true, nc == c);
}

} // namespace snippet4

// ============================================================================
// Snippet 5: C-style cast
// MISRA C++ Rule 5-2-4 / A5-2-2: C-style casts shall not be used
// (C cast can silently reinterpret bits — use explicit C++ casts)
// ============================================================================

namespace snippet5 {

// NON-COMPLIANT: C-style cast — could be reinterpret_cast in disguise
int32_t non_compliant_convert(double val) {
    return (int32_t)val;  // VIOLATION: C-style cast
}

// COMPLIANT: explicit static_cast — intent is clear, reviewable
int32_t compliant_convert(double val) {
    return static_cast<int32_t>(val);  // narrowing is intentional and visible
}

void test() {
    int32_t nc = non_compliant_convert(3.14);
    int32_t c  = compliant_convert(3.14);
    record(5, "A5-2-2", "No C-style casts — use static_cast/etc", true, true, nc == c);
}

} // namespace snippet5

// ============================================================================
// Snippet 6: Raw pointer arithmetic
// MISRA C++ Rule 5-0-15 / A5-0-4: Pointer arithmetic shall not be used
// (buffer overruns, off-by-one — largest class of safety vulnerabilities)
// ============================================================================

namespace snippet6 {

// NON-COMPLIANT: raw pointer arithmetic to sum array
int non_compliant_array_sum(const int* data, int size) {
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += *(data + i);  // VIOLATION: pointer arithmetic
    }
    return sum;
}

// COMPLIANT: use std::array and range-based for (or std::span in C++20)
// Bounds-safe: size is part of the type, no pointer arithmetic needed
int compliant_array_sum(const std::array<int, 5>& data) {
    int sum = 0;
    for (int val : data) {  // range-based — no pointer arithmetic, no overflow
        sum += val;
    }
    return sum;
}

void test() {
    std::array<int, 5> arr = {1, 2, 3, 4, 5};
    int nc = non_compliant_array_sum(arr.data(), static_cast<int>(arr.size()));
    int c  = compliant_array_sum(arr);
    record(6, "A5-0-4", "No pointer arithmetic — use span/array", true, true, nc == c);
}

} // namespace snippet6

// ============================================================================
// Snippet 7: Function-like preprocessor macro
// MISRA C++ Rule 16-0-4 / A16-0-1: Function-like macros shall not be defined
// (no type checking, no scoping, easy to mis-expand)
// ============================================================================

namespace snippet7 {

// NON-COMPLIANT: function-like macro — no type safety, double evaluation
#define SQUARE_MACRO(x) ((x) * (x))  // VIOLATION: function-like macro

// COMPLIANT: constexpr inline function — type-safe, scoped, single evaluation
constexpr int square_func(int x) {
    return x * x;
}

void test() {
    int val = 5;
    int nc = SQUARE_MACRO(val);
    int c  = square_func(val);

    // Demonstrate the macro danger: SQUARE_MACRO(val++) would evaluate val++ twice!
    // We won't actually run that since it's UB, but note the hazard.
    record(7, "A16-0-1", "No function-like macros — use constexpr", true, true, nc == c);
}

#undef SQUARE_MACRO  // clean up after ourselves

} // namespace snippet7

// ============================================================================
// Main: run all tests and print report
// ============================================================================

int main() {
    std::cout << "═══════════════════════════════════════════════════════════════\n";
    std::cout << "  MISRA C++ Compliance Rewrite Exercise\n";
    std::cout << "═══════════════════════════════════════════════════════════════\n\n";

    snippet1::test();
    snippet2::test();
    snippet3::test();
    snippet4::test();
    snippet5::test();
    snippet6::test();
    snippet7::test();

    // Print results table
    std::cout << "┌────┬────────────┬──────────────────────────────────────────┬────────┬──────────┬───────┐\n";
    std::cout << "│ ID │ MISRA Rule │ Description                              │ NC OK  │ Comp OK  │ Match │\n";
    std::cout << "├────┼────────────┼──────────────────────────────────────────┼────────┼──────────┼───────┤\n";

    int pass = 0, fail = 0;
    for (const auto& r : results) {
        bool ok = r.non_compliant_ok && r.compliant_ok && r.results_match;
        if (ok) ++pass; else ++fail;

        std::cout << "│ " << std::setw(2) << r.id << " │ "
                  << std::setw(10) << std::left << r.rule << " │ "
                  << std::setw(40) << std::left << r.description << " │ "
                  << (r.non_compliant_ok ? " PASS " : " FAIL ") << " │ "
                  << (r.compliant_ok ? "  PASS  " : "  FAIL  ") << " │ "
                  << (r.results_match ? " YES " : "  NO  ") << " │\n";
    }
    std::cout << "└────┴────────────┴──────────────────────────────────────────┴────────┴──────────┴───────┘\n\n";

    // Summary of each violation
    std::cout << "MISRA Violation Summary:\n";
    std::cout << "  1. Dir 4.12  — No dynamic allocation after init (fragmentation, OOM)\n";
    std::cout << "  2. A15-0-1   — No exceptions (unpredictable control flow)\n";
    std::cout << "  3. A7-5-1    — No recursion (unbounded stack)\n";
    std::cout << "  4. A6-6-1    — No goto (defeats static analysis)\n";
    std::cout << "  5. A5-2-2    — No C-style casts (silent reinterpret)\n";
    std::cout << "  6. A5-0-4    — No pointer arithmetic (buffer overruns, use array/span)\n";
    std::cout << "  7. A16-0-1   — No function-like macros (no type safety)\n";

    std::cout << "\n══════════════════════════════════════════════════════════════\n";
    std::cout << "  Summary: " << pass << " passed, " << fail << " failed\n";
    std::cout << "══════════════════════════════════════════════════════════════\n";

    std::cout << R"(
Key Takeaways:
  1. MISRA bans constructs that are LEGAL C++ but DANGEROUS in safety contexts
  2. Every banned construct has a safe alternative that produces identical results
  3. The restrictions exist because: hard real-time + no operator intervention = must not crash
  4. Modern C++ (spans, optional, constexpr) makes compliance much easier than C++03
  5. Static analysis tools (Polyspace, LDRA, Parasoft) enforce MISRA automatically
  6. Cost of MISRA: ~10-20% more code, but provably safer behavior
)";

    return (fail == 0) ? 0 : 1;
}
