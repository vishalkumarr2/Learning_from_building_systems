// Week 7 — Exercise 5: UB Detector Challenge
// =============================================
// 10 code snippets with hidden UB. Find the bugs!
//
// Compile: g++ -std=c++20 -O0 -g ex05_ub_detector_challenge.cpp -o ex05
// With sanitizers: g++ -std=c++20 -fsanitize=undefined,address -O0 -g \
//                  ex05_ub_detector_challenge.cpp -o ex05_san
//
// By default, FIXED is defined — safe versions run.
// Comment out #define FIXED to see the UB versions.
// Run under sanitizers to verify each bug.

#define FIXED

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

template <typename T>
void do_not_optimize(T const& val) {
    asm volatile("" : : "r,m"(val) : "memory");
}

// ============================================================
// Challenge 1: Strict Aliasing
// ============================================================

void challenge_01() {
    std::cout << "Challenge 01: Strict Aliasing\n";
    float values[] = {1.0f, 2.0f, 3.0f, 4.0f};

#ifdef FIXED
    // Safe: use memcpy for type punning
    for (int i = 0; i < 4; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &values[i], sizeof(bits));
        std::cout << "  float " << values[i] << " = 0x" << std::hex << bits << std::dec << "\n";
    }
#else
    // Bug: accessing float array through int pointer violates strict aliasing
    uint32_t* int_view = reinterpret_cast<uint32_t*>(values);
    for (int i = 0; i < 4; ++i) {
        std::cout << "  float " << values[i] << " = 0x" << std::hex << int_view[i] << std::dec << "\n";
    }
#endif
}
// BUG: reinterpret_cast<uint32_t*>(float_array) violates strict aliasing [basic.lval].
// Compiler may reorder or eliminate loads. Use std::memcpy or std::bit_cast.

// ============================================================
// Challenge 2: Signed Integer Overflow in Accumulation
// ============================================================

void challenge_02() {
    std::cout << "Challenge 02: Signed Overflow in Accumulation\n";

#ifdef FIXED
    // Safe: use int64_t for accumulation
    int values[] = {1'000'000'000, 1'000'000'000, 1'000'000'000};
    int64_t sum = 0;
    for (int v : values) {
        sum += v;
    }
    std::cout << "  Sum: " << sum << "\n";
#else
    // Bug: sum overflows int (3 billion > INT_MAX ≈ 2.1 billion)
    int values[] = {1'000'000'000, 1'000'000'000, 1'000'000'000};
    int sum = 0;
    for (int v : values) {
        sum += v; // Overflow on third addition!
    }
    std::cout << "  Sum: " << sum << "\n";
#endif
}
// BUG: int sum overflows when accumulating 3 × 10^9 > INT_MAX.
// UBSan catches: "signed integer overflow: 2000000000 + 1000000000".

// ============================================================
// Challenge 3: Null Through Reference
// ============================================================

void challenge_03() {
    std::cout << "Challenge 03: Null Through Reference\n";

#ifdef FIXED
    int x = 42;
    int* p = &x;
    if (p) {
        int& r = *p;
        std::cout << "  Value: " << r << "\n";
    }
#else
    // Bug: creating a reference from a null pointer is UB, even if not accessed
    int* p = nullptr;
    int& r = *p; // UB right here, even without reading r
    do_not_optimize(r);
    std::cout << "  (reference created from null — UB even if unused)\n";
#endif
}
// BUG: Dereferencing nullptr to create a reference is UB immediately [expr.unary.op].
// The reference binding itself is UB, not just reading through it.

// ============================================================
// Challenge 4: Dangling Reference from Ternary
// ============================================================

void challenge_04() {
    std::cout << "Challenge 04: Dangling Reference from Ternary\n";

#ifdef FIXED
    std::string a = "alpha";
    std::string b = "beta";
    bool cond = true;
    // Return by value, not reference
    std::string result = cond ? a : b;
    std::cout << "  Value: " << result << "\n";
#else
    // Bug: ternary creates a temporary when types differ or conversions happen
    std::string a = "alpha";
    bool cond = true;
    // "beta" is const char*, so the ternary creates a temporary std::string
    // Binding to const ref extends lifetime here, but be careful with subtle variants
    const std::string& result = cond ? a : std::string("beta");
    // If cond is false, result refers to the temporary — which IS lifetime-extended
    // But if this were returned from a function, it wouldn't be!
    std::cout << "  Value: " << result << "\n";
    // This specific case is actually OK due to lifetime extension,
    // but the pattern is treacherous in functions. Demo the idea:
    auto get = [&]() -> const std::string& {
        return cond ? a : std::string("beta"); // Dangling if cond is false!
    };
    // const std::string& r = get(); // Would dangle — don't call
    std::cout << "  (lambda version would dangle — not called)\n";
#endif
}
// BUG: Returning const ref to a temporary created inside a ternary in a function.
// Lifetime extension does NOT apply through function return.

// ============================================================
// Challenge 5: Double Delete via Raw Pointer Alias
// ============================================================

void challenge_05() {
    std::cout << "Challenge 05: Double Delete via Aliasing\n";

#ifdef FIXED
    auto p = std::make_shared<int>(42);
    auto q = p; // shared_ptr — reference counted, no double delete
    std::cout << "  *p=" << *p << ", *q=" << *q << ", count=" << p.use_count() << "\n";
#else
    // Bug: two raw pointers to same allocation, both try to delete
    int* p = new int(42);
    int* q = p; // Alias!
    std::cout << "  *p=" << *p << ", *q=" << *q << "\n";
    delete p;
    // delete q; // Double delete — heap corruption (not executed to avoid crash)
    std::cout << "  (double delete would crash — not executed)\n";
#endif
}
// BUG: Two raw pointers to the same allocation. delete on both = heap corruption.
// ASan catches immediately. Fix: use shared_ptr or unique_ptr with clear ownership.

// ============================================================
// Challenge 6: Buffer Overrun in String Processing
// ============================================================

void challenge_06() {
    std::cout << "Challenge 06: Buffer Overrun in String Processing\n";

#ifdef FIXED
    char src[] = "Hello, World! This is a long string for testing.";
    constexpr size_t BUFSIZE = 16;
    char dst[BUFSIZE];
    // Safe: use strncat with proper size accounting
    dst[0] = '\0';
    strncat(dst, src, BUFSIZE - 1);
    dst[BUFSIZE - 1] = '\0'; // Ensure null termination
    std::cout << "  Safe copy: \"" << dst << "\"\n";
#else
    // Bug: strcpy with source longer than destination
    char src[] = "Hello, World! This is a long string for testing.";
    char dst[16];
    strcpy(dst, src); // Buffer overrun!
    std::cout << "  Overflowed: \"" << dst << "\"\n";
#endif
}
// BUG: strcpy doesn't check bounds. Source (49 chars) overflows dst (16 chars).
// ASan catches: "stack-buffer-overflow".

// ============================================================
// Challenge 7: Uninitialized Read Affecting Control Flow
// ============================================================

void challenge_07() {
    std::cout << "Challenge 07: Uninitialized Read in Control Flow\n";

#ifdef FIXED
    bool flags[8] = {}; // Zero-initialized
    // Set some flags
    flags[0] = true;
    flags[3] = true;
    int count = 0;
    for (int i = 0; i < 8; ++i) {
        if (flags[i]) ++count;
    }
    std::cout << "  Flags set: " << count << "\n";
#else
    // Bug: array not initialized, control flow depends on garbage values
    bool flags[8]; // Uninitialized!
    flags[0] = true;
    flags[3] = true;
    // Only [0] and [3] are set; [1],[2],[4],[5],[6],[7] are indeterminate
    int count = 0;
    for (int i = 0; i < 8; ++i) {
        if (flags[i]) ++count; // UB: reading uninitialized bool
    }
    std::cout << "  Flags set: " << count << " (expected 2, got garbage)\n";
#endif
}
// BUG: Reading uninitialized bools is UB. Even bool can have trap representations.
// MSan catches this. Compiler may assume any uninitialized bool is either true or false.

// ============================================================
// Challenge 8: Alignment Violation via Packed Buffer
// ============================================================

void challenge_08() {
    std::cout << "Challenge 08: Alignment Violation\n";

#ifdef FIXED
    // Safe: use memcpy to deserialize from byte buffer
    uint8_t packet[] = {0x01, 0x00, 0x00, 0x00,
                        0x00, 0x00, 0x20, 0x41,  // 10.0f in IEEE 754 (little-endian)
                        0xFF};
    uint32_t header;
    float value;
    std::memcpy(&header, packet, sizeof(header));
    std::memcpy(&value, packet + 4, sizeof(value));
    std::cout << "  Header: " << header << ", Value: " << value << "\n";
#else
    // Bug: casting misaligned byte pointer to int*/float*
    uint8_t packet[] = {0x01, 0x00, 0x00, 0x00,
                        0x00, 0x00, 0x20, 0x41,
                        0xFF};
    // packet+1 is NOT aligned for uint32_t or float
    uint32_t* header = reinterpret_cast<uint32_t*>(packet + 1); // Misaligned!
    float* value = reinterpret_cast<float*>(packet + 3);        // Misaligned!
    std::cout << "  Header: " << *header << ", Value: " << *value << "\n";
#endif
}
// BUG: reinterpret_cast to types with alignment > 1 from arbitrary byte offsets.
// SIGBUS on ARM/RISC-V, silent perf penalty on x86. UBSan catches with -fsanitize=alignment.

// ============================================================
// Challenge 9: Infinite Loop UB (Forward Progress)
// ============================================================

void challenge_09() {
    std::cout << "Challenge 09: Infinite Loop UB\n";

#ifdef FIXED
    // Safe: loop has observable side effect (volatile)
    volatile int x = 0;
    int iterations = 0;
    while (x == 0) {
        ++iterations;
        if (iterations >= 100) x = 1; // Will eventually terminate
    }
    std::cout << "  Loop terminated after " << iterations << " iterations\n";
#else
    // Bug: loop without side effects — compiler may delete it or assume
    // it terminates (C++11 forward progress guarantee)
    int x = 0;
    auto trivial_check = [](int v) { return v == 0; };
    // This loop has no observable side effects and no volatile/atomic
    while (trivial_check(x)) {
        // Nothing happens — UB per [intro.progress]
        // Compiler may optimize to: just skip the entire loop
        break; // Added break so we don't actually hang — but without it, UB
    }
    std::cout << "  (loop with break — real UB version would hang or be deleted)\n";
#endif
}
// BUG: A loop with no side effects (no I/O, no volatile, no atomic) that doesn't
// terminate is UB. The compiler may assume it terminates and optimize accordingly.

// ============================================================
// Challenge 10: Type Punning via Union
// ============================================================

void challenge_10() {
    std::cout << "Challenge 10: Type Punning via Union\n";

#ifdef FIXED
    // Safe: use memcpy / bit_cast
    float f = 1.0f;
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    std::cout << "  1.0f as uint32: 0x" << std::hex << u << std::dec << "\n";

    // C++20: std::bit_cast
    // auto u2 = std::bit_cast<uint32_t>(f);
#else
    // Bug: in C++, reading inactive union member is UB
    union {
        float f;
        uint32_t u;
    } pun;
    pun.f = 1.0f;
    std::cout << "  1.0f as uint32: 0x" << std::hex << pun.u << std::dec << "\n";
    // Works on GCC due to extension, but is UB per C++ standard
#endif
}
// BUG: Reading a union member that was not the last one written is UB in C++.
// C allows it (implementation-defined), C++ does not. Use std::memcpy or std::bit_cast.

// ============================================================

int main() {
    std::cout << "=== UB DETECTOR CHALLENGE ===\n";
#ifdef FIXED
    std::cout << "MODE: FIXED (safe versions)\n";
#else
    std::cout << "MODE: BUGGY (run under -fsanitize=undefined,address!)\n";
#endif
    std::cout << "=============================\n\n";

    challenge_01();
    challenge_02();
    challenge_03();
    challenge_04();
    challenge_05();
    challenge_06();
    challenge_07();
    challenge_08();
    challenge_09();
    challenge_10();

    std::cout << "\n=== ALL CHALLENGES COMPLETE ===\n";
    std::cout << "Score: find all 10 bugs before checking the // BUG comments!\n";
    return 0;
}
