// Module 16: Sanitizer Workshop
// Exercise 02: UndefinedBehaviorSanitizer Bug Hunt
//
// This file contains 8 intentional undefined behavior patterns.
// Compile with UBSan and run each one:
//
//   g++ -std=c++2a -fsanitize=undefined -g ex02_ubsan_bugs.cpp -o ex02_ubsan
//   UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1" ./ex02_ubsan <bug_number>
//
// Each function triggers a specific category of UB that UBSan detects.

#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

// ============================================================
// Bug 1: Signed integer overflow
// UBSan reports: "signed integer overflow: 2147483647 + 1 cannot be
//                 represented in type 'int'"
// ============================================================
__attribute__((noinline))
void bug_signed_overflow() {
    volatile int a = INT_MAX;
    // BUG: INT_MAX + 1 is undefined behavior for signed integers
    volatile int b = a + 1;
    std::cout << "bug_signed_overflow: INT_MAX + 1 = " << b << "\n";
}

// ============================================================
// Bug 2: Shift exponent too large
// UBSan reports: "shift exponent 33 is too large for 32-bit type 'int'"
// Shifting by >= the bit width is UB.
// ============================================================
__attribute__((noinline))
void bug_shift_exponent() {
    volatile int val = 1;
    volatile int shift = 33;  // >= 32 bits
    // BUG: shift by 33 on a 32-bit int
    volatile int result = val << shift;
    std::cout << "bug_shift_exponent: 1 << 33 = " << result << "\n";
}

// ============================================================
// Bug 3: Null pointer dereference
// UBSan reports: "load of null pointer of type 'int'"
// Dereferencing a null pointer is UB (even if hardware may trap).
// ============================================================
__attribute__((noinline))
void bug_null_deref() {
    volatile int* p = nullptr;
    // BUG: dereferencing nullptr
    volatile int val = *p;
    std::cout << "bug_null_deref: read " << val << " from nullptr\n";
}

// ============================================================
// Bug 4: Integer division by zero
// UBSan reports: "division by zero"
// Integer division by zero is undefined behavior in C++.
// ============================================================
__attribute__((noinline))
void bug_division_by_zero() {
    volatile int numerator = 42;
    volatile int denominator = 0;
    // BUG: integer division by zero
    volatile int result = numerator / denominator;
    std::cout << "bug_division_by_zero: 42 / 0 = " << result << "\n";
}

// ============================================================
// Bug 5: Alignment violation
// UBSan reports: "load of misaligned address ... for type 'int',
//                 which requires 4 byte alignment"
// Accessing data through an improperly aligned pointer.
// ============================================================
__attribute__((noinline))
void bug_alignment_violation() {
    // Create a buffer with known misalignment
    alignas(8) char buffer[16] = {};
    buffer[1] = 0x42;

    // BUG: casting char* at offset 1 to int* creates misaligned pointer
    volatile int* misaligned = reinterpret_cast<int*>(&buffer[1]);
    volatile int val = *misaligned;
    std::cout << "bug_alignment_violation: read " << val << " from misaligned address\n";
}

// ============================================================
// Bug 6: Float-cast overflow
// UBSan reports: "value ... is outside the range of representable values
//                 of type 'int'"
// A floating point value too large to fit in the target integer type.
// ============================================================
__attribute__((noinline))
void bug_float_cast_overflow() {
    volatile double huge = 1.0e18;  // Larger than INT_MAX (2^31 - 1)
    // BUG: casting double to int when value exceeds int range
    volatile int val = static_cast<int>(huge);
    std::cout << "bug_float_cast_overflow: (int)1e18 = " << val << "\n";
}

// ============================================================
// Bug 7: Variable-length array (VLA) with non-positive bound
// UBSan reports: "variable length array bound evaluates to non-positive value"
// VLAs are a GCC extension in C++; a negative size is UB.
// ============================================================
__attribute__((noinline))
void bug_vla_bound() {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"
    volatile int size = -1;
    // BUG: VLA with negative size — undefined behavior
    // GCC allows VLAs as extension; UBSan catches the invalid bound
    int vla[size];
    vla[0] = 42;
    volatile int val = vla[0];
    std::cout << "bug_vla_bound: VLA[-1], read " << val << "\n";
#pragma GCC diagnostic pop
}

// ============================================================
// Bug 8: Left-shift of negative value
// UBSan reports: "left shift of negative value -1"
// Left-shifting a negative signed integer is UB in C++ (until C++20
// where it's implementation-defined, but UBSan still catches it).
// ============================================================
__attribute__((noinline))
void bug_signed_shift_negative() {
    volatile int neg = -1;
    // BUG: left-shifting a negative value
    volatile int result = neg << 1;
    std::cout << "bug_signed_shift_negative: (-1) << 1 = " << result << "\n";
}

// ============================================================
// Main: select bug by command-line argument
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <bug_number 1-8>\n";
        std::cerr << "  1: signed-integer-overflow (INT_MAX + 1)\n";
        std::cerr << "  2: shift-exponent too large (1 << 33)\n";
        std::cerr << "  3: null-pointer dereference\n";
        std::cerr << "  4: integer division by zero\n";
        std::cerr << "  5: alignment violation (misaligned int*)\n";
        std::cerr << "  6: float-cast overflow (1e18 -> int)\n";
        std::cerr << "  7: VLA with negative bound\n";
        std::cerr << "  8: left-shift of negative value\n";
        return 1;
    }

    int bug = std::atoi(argv[1]);

    switch (bug) {
        case 1: bug_signed_overflow(); break;
        case 2: bug_shift_exponent(); break;
        case 3: bug_null_deref(); break;
        case 4: bug_division_by_zero(); break;
        case 5: bug_alignment_violation(); break;
        case 6: bug_float_cast_overflow(); break;
        case 7: bug_vla_bound(); break;
        case 8: bug_signed_shift_negative(); break;
        default:
            std::cerr << "Invalid bug number: " << bug << " (valid: 1-8)\n";
            return 1;
    }

    return 0;
}
