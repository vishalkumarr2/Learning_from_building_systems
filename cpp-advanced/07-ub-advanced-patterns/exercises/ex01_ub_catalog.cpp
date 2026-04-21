// Week 7 — Exercise 1: The 20 UB Forms Catalog
// ================================================
// Compile: g++ -std=c++20 -Wall -Wextra -O0 -g ex01_ub_catalog.cpp -o ex01
// With UBSan: g++ -std=c++20 -fsanitize=undefined,address -O0 -g ex01_ub_catalog.cpp -o ex01_san
//
// By default, FIXED is defined — all functions run the safe version.
// Comment out #define FIXED to see the UB versions (run under sanitizers!).

#define FIXED

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <vector>

// Helper to prevent compiler from optimizing away results
template <typename T>
void do_not_optimize(T const& val) {
    asm volatile("" : : "r,m"(val) : "memory");
}

// ============================================================
// CATEGORY A: MEMORY UB
// ============================================================

// UB-01: Use-After-Free [CWE-416]
// Standard: [basic.stc.dynamic.deallocation] — accessing freed memory is UB
void ub_01_use_after_free() {
    std::cout << "UB-01: Use-After-Free\n";
#ifdef FIXED
    auto p = std::make_unique<int>(42);
    std::cout << "  FIXED: value = " << *p << " (accessed before release)\n";
    // p released automatically at scope end
#else
    int* p = new int(42);
    delete p;
    std::cout << "  UB: value = " << *p << " (use-after-free!)\n";
#endif
}

// UB-02: Buffer Overflow [CWE-787]
// Standard: [expr.add] — pointer arithmetic past array bounds is UB
void ub_02_buffer_overflow() {
    std::cout << "UB-02: Buffer Overflow\n";
    int arr[5] = {10, 20, 30, 40, 50};
#ifdef FIXED
    // Use .at() or explicit bounds check
    for (int i = 0; i < 5; ++i) {
        do_not_optimize(arr[i]);
    }
    std::cout << "  FIXED: accessed only indices 0..4\n";
#else
    // Reads past end of array
    for (int i = 0; i <= 5; ++i) {
        do_not_optimize(arr[i]); // arr[5] is out-of-bounds
    }
    std::cout << "  UB: accessed arr[5] (out-of-bounds)\n";
#endif
}

// UB-03: Null Pointer Dereference [CWE-476]
// Standard: [expr.unary.op] — dereferencing null is UB
void ub_03_null_deref() {
    std::cout << "UB-03: Null Pointer Dereference\n";
#ifdef FIXED
    int* p = nullptr;
    if (p != nullptr) {
        std::cout << "  value = " << *p << "\n";
    } else {
        std::cout << "  FIXED: null check prevented dereference\n";
    }
#else
    int* p = nullptr;
    std::cout << "  UB: value = " << *p << " (null deref!)\n";
#endif
}

// UB-04: Dangling Reference
// Standard: [basic.stc.auto] — reference to expired automatic variable is UB
static const std::string& bad_dangling_ref() {
    std::string local = "dangling";
    return local; // WARNING: returning reference to local
}
void ub_04_dangling_reference() {
    std::cout << "UB-04: Dangling Reference\n";
#ifdef FIXED
    // Return by value, not reference
    auto make_str = []() -> std::string { return "safe"; };
    std::string s = make_str();
    std::cout << "  FIXED: value = " << s << "\n";
#else
    const std::string& r = bad_dangling_ref();
    std::cout << "  UB: value = " << r << " (dangling reference!)\n";
#endif
}

// UB-05: Uninitialized Read
// Standard: [dcl.init] — reading indeterminate value is UB
void ub_05_uninitialized_read() {
    std::cout << "UB-05: Uninitialized Read\n";
#ifdef FIXED
    int x = 0; // Always initialize
    std::cout << "  FIXED: x = " << x << " (initialized to 0)\n";
#else
    int x; // Uninitialized
    std::cout << "  UB: x = " << x << " (uninitialized read!)\n";
#endif
}

// ============================================================
// CATEGORY B: ARITHMETIC UB
// ============================================================

// UB-06: Signed Integer Overflow
// Standard: [expr] — signed overflow is UB
void ub_06_signed_overflow() {
    std::cout << "UB-06: Signed Integer Overflow\n";
#ifdef FIXED
    int a = std::numeric_limits<int>::max();
    long long result = static_cast<long long>(a) + 1; // Widen first
    std::cout << "  FIXED: INT_MAX + 1 (as long long) = " << result << "\n";
#else
    int a = std::numeric_limits<int>::max();
    int result = a + 1; // Signed overflow!
    std::cout << "  UB: INT_MAX + 1 = " << result << "\n";
#endif
}

// UB-07: Division by Zero
// Standard: [expr.mul] — integer division/modulo by zero is UB
void ub_07_division_by_zero() {
    std::cout << "UB-07: Division by Zero\n";
#ifdef FIXED
    int a = 42, b = 0;
    if (b != 0) {
        std::cout << "  result = " << (a / b) << "\n";
    } else {
        std::cout << "  FIXED: division by zero prevented\n";
    }
#else
    volatile int a = 42, b = 0;
    std::cout << "  UB: 42 / 0 = " << (a / b) << "\n";
#endif
}

// UB-08: Shift UB
// Standard: [expr.shift] — shift by >= bitwidth or negative amount is UB
void ub_08_shift_ub() {
    std::cout << "UB-08: Shift UB\n";
#ifdef FIXED
    uint32_t x = 1;
    int shift = 31; // Max safe shift for 32-bit
    uint32_t result = x << shift;
    std::cout << "  FIXED: 1u << 31 = " << result << "\n";
#else
    int x = 1;
    int result = x << 32; // Shift by bitwidth!
    std::cout << "  UB: 1 << 32 = " << result << "\n";
#endif
}

// ============================================================
// CATEGORY C: TYPE SYSTEM UB
// ============================================================

// UB-09: Strict Aliasing Violation
// Standard: [basic.lval] — accessing object through wrong type pointer is UB
void ub_09_strict_aliasing() {
    std::cout << "UB-09: Strict Aliasing Violation\n";
#ifdef FIXED
    float f = 3.14f;
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits)); // Safe type punning
    std::cout << "  FIXED: float 3.14 bits = 0x" << std::hex << bits << std::dec << "\n";
#else
    float f = 3.14f;
    // Violates strict aliasing — int* and float* can't alias
    uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
    std::cout << "  UB: float 3.14 bits = 0x" << std::hex << bits << std::dec << "\n";
#endif
}

// UB-10: Type Punning via Union (C++ specific)
// Standard: [class.union] — reading inactive union member is UB in C++
void ub_10_union_type_pun() {
    std::cout << "UB-10: Type Punning via Union\n";
    union Pun {
        float f;
        uint32_t u;
    };
#ifdef FIXED
    float f = 2.718f;
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    std::cout << "  FIXED: float 2.718 as uint32 = " << u << " (via memcpy)\n";
#else
    Pun p;
    p.f = 2.718f;
    std::cout << "  UB: float 2.718 as uint32 = " << p.u << " (union pun)\n";
#endif
}

// UB-11: Alignment Violation
// Standard: [basic.align] — misaligned access is UB
void ub_11_alignment() {
    std::cout << "UB-11: Alignment Violation\n";
#ifdef FIXED
    alignas(alignof(int)) char buf[sizeof(int) * 2] = {};
    int* p = reinterpret_cast<int*>(buf); // Properly aligned
    *p = 42;
    std::cout << "  FIXED: aligned access = " << *p << "\n";
#else
    char buf[16] = {};
    // Misaligned: int requires 4-byte alignment, buf+1 is 1-byte aligned
    int* p = reinterpret_cast<int*>(buf + 1);
    *p = 42;
    std::cout << "  UB: misaligned access = " << *p << "\n";
#endif
}

// ============================================================
// CATEGORY D: SEQUENCE & EVALUATION UB
// ============================================================

// UB-12: Unsequenced Modification
// Standard: [intro.execution] — two unsequenced modifications to same scalar is UB
void ub_12_unsequenced() {
    std::cout << "UB-12: Unsequenced Modification\n";
#ifdef FIXED
    int i = 0;
    i++;           // Step 1
    int j = i + 1; // Step 2 (sequenced)
    std::cout << "  FIXED: i=" << i << ", j=" << j << "\n";
#else
    int i = 0;
    // Note: `arr[idx] = idx++` was UB in C++14 and earlier, but C++17 sequences
    // the right operand of = before the left, making it well-defined.
    // The classic UB example that's still UB even in C++17:
    i = i++ + ++i; // UB: two unsequenced modifications of i in one expression
    std::cout << "  UB: i = i++ + ++i → i=" << i << " (could be anything)\n";
#endif
}

// UB-13: Order of Evaluation
// Not technically UB but unspecified; included for completeness
void ub_13_evaluation_order() {
    std::cout << "UB-13: Evaluation Order (unspecified, not UB)\n";
    int counter = 0;
    auto inc = [&]() { return ++counter; };
#ifdef FIXED
    int a = inc();
    int b = inc();
    std::cout << "  FIXED: a=" << a << ", b=" << b << " (sequenced)\n";
#else
    // The order of inc() calls is unspecified
    auto add = [](int x, int y) { return x + y; };
    int result = add(inc(), inc());
    std::cout << "  UNSPECIFIED: add(inc(), inc()) = " << result << "\n";
#endif
}

// ============================================================
// CATEGORY E: LIFETIME UB
// ============================================================

// UB-14: Returning Reference to Local (see UB-04, but explicit function case)
void ub_14_return_local_ref() {
    std::cout << "UB-14: Returning Reference to Local\n";
    std::cout << "  (Same as UB-04, see dangling reference example)\n";
    std::cout << "  FIXED: return by value, not by reference\n";
}

// UB-15: Temporary Lifetime Extension Failure
void ub_15_temporary_lifetime() {
    std::cout << "UB-15: Temporary Lifetime Extension\n";
#ifdef FIXED
    // Safe: binding rvalue ref to const lvalue ref extends lifetime
    const std::string& r = std::string("hello");
    std::cout << "  FIXED: temporary extended, r = " << r << "\n";
#else
    // Dangerous pattern: function returns const ref, no extension
    auto get_ref = []() -> const std::string& {
        static std::string s;
        s = "temp";
        return s; // OK here because static, but teaching the pattern
    };
    const std::string& r = get_ref();
    std::cout << "  DEMO: r = " << r << " (safe only because static)\n";
#endif
}

// UB-16: Use After std::move
void ub_16_use_after_move() {
    std::cout << "UB-16: Use After std::move\n";
#ifdef FIXED
    std::string s = "hello";
    std::string t = std::move(s);
    // After move, only use s for assignment or destruction
    s = "reassigned"; // This is fine
    std::cout << "  FIXED: s after reassign = " << s << ", t = " << t << "\n";
#else
    std::string s = "hello";
    std::string t = std::move(s);
    // s is in "valid but unspecified" state — technically not UB but a bug
    std::cout << "  RISKY: s after move = '" << s << "' (unspecified state)\n";
#endif
}

// ============================================================
// CATEGORY F: MISCELLANEOUS LETHAL UB
// ============================================================

// UB-17: Infinite Loop Without Side Effects
// Standard: [intro.progress] — forward progress guarantee
void ub_17_infinite_loop() {
    std::cout << "UB-17: Infinite Loop Without Side Effects\n";
#ifdef FIXED
    // Loop with side effect (volatile) — well-defined
    volatile bool keep_going = true;
    int iterations = 0;
    while (keep_going) {
        if (++iterations >= 5) keep_going = false;
    }
    std::cout << "  FIXED: loop with volatile flag, " << iterations << " iterations\n";
#else
    std::cout << "  UB: infinite loop without side effects would be compiled away\n";
    // while(true) {} // <- DON'T: compiler may delete this at -O2
#endif
}

// UB-18: [[noreturn]] Function That Returns
[[noreturn]] static void supposed_to_not_return(bool do_abort) {
    if (do_abort) {
        std::abort();
    }
    // UB if we reach here! (commented out so we don't crash)
}
void ub_18_noreturn_returns() {
    std::cout << "UB-18: [[noreturn]] Function That Returns\n";
#ifdef FIXED
    std::cout << "  FIXED: calling [[noreturn]] function that actually aborts\n";
    // supposed_to_not_return(true); // Would abort — don't call in demo
    std::cout << "  (skipping actual call to avoid abort)\n";
#else
    std::cout << "  UB: if [[noreturn]] function returns, behavior is undefined\n";
#endif
}

// UB-19: Double Delete
// Standard: [expr.delete] — deleting already-freed memory is UB
void ub_19_double_delete() {
    std::cout << "UB-19: Double Delete\n";
#ifdef FIXED
    auto p = std::make_unique<int>(42);
    std::cout << "  FIXED: using unique_ptr, no double delete possible\n";
    // unique_ptr handles deletion automatically
#else
    int* p = new int(42);
    delete p;
    // delete p; // DON'T: double delete — heap corruption
    std::cout << "  UB: double delete causes heap corruption (not executed)\n";
#endif
}

// UB-20: ODR Violation
// Standard: [basic.def.odr] — one definition rule
void ub_20_odr_violation() {
    std::cout << "UB-20: ODR Violation\n";
    std::cout << "  Cannot demo in single TU — requires separate compilation units\n";
    std::cout << "  Example: two .cpp files define 'struct Config { int x; }'\n";
    std::cout << "  with different member types. Linker picks one silently.\n";
    std::cout << "  FIXED: use header-only definitions, inline variables, or LTO\n";
}

// ============================================================

int main() {
    std::cout << "=== UB CATALOG: 20 MOST DANGEROUS FORMS ===\n";
#ifdef FIXED
    std::cout << "MODE: FIXED (safe versions)\n";
#else
    std::cout << "MODE: UB (run under -fsanitize=undefined,address!)\n";
#endif
    std::cout << "=============================================\n\n";

    ub_01_use_after_free();
    ub_02_buffer_overflow();
    ub_03_null_deref();
    ub_04_dangling_reference();
    ub_05_uninitialized_read();

    std::cout << "\n";
    ub_06_signed_overflow();
    ub_07_division_by_zero();
    ub_08_shift_ub();

    std::cout << "\n";
    ub_09_strict_aliasing();
    ub_10_union_type_pun();
    ub_11_alignment();

    std::cout << "\n";
    ub_12_unsequenced();
    ub_13_evaluation_order();

    std::cout << "\n";
    ub_14_return_local_ref();
    ub_15_temporary_lifetime();
    ub_16_use_after_move();

    std::cout << "\n";
    ub_17_infinite_loop();
    ub_18_noreturn_returns();
    ub_19_double_delete();
    ub_20_odr_violation();

    std::cout << "\n=== ALL DONE ===\n";
    return 0;
}
