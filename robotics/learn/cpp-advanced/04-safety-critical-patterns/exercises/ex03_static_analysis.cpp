// =============================================================================
// Exercise 03: Static Analysis — Code That Triggers Specific Warnings
// =============================================================================
// Each section shows a BUGGY version (the kind of code static analysis catches)
// and a FIXED version. Compile with -Wall -Wextra -Wshadow -Wconversion to see.
//
// Recommended .clang-tidy config:
// ---
// Checks: >
//   bugprone-*,
//   bugprone-use-after-move,
//   bugprone-dangling-handle,
//   bugprone-signed-char-misuse,
//   cert-*,
//   cppcoreguidelines-pro-bounds-*,
//   misc-redundant-expression,
//   misc-unused-using-decls,
//   modernize-use-nullptr,
//   performance-move-const-arg,
//   readability-*,
//   -readability-magic-numbers
// WarningsAsErrors: 'bugprone-use-after-move,bugprone-dangling-handle'
// ---
// =============================================================================

#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <optional>
#include <memory>

// ======================== BUG 1: SIGNED/UNSIGNED COMPARISON =================
// -Wsign-compare catches this, but without the flag it compiles silently.

namespace bug1 {

namespace buggy {
// This function checks if an index is "in range" — but gets it wrong
// when idx is negative because the signed→unsigned promotion makes it huge.
bool is_in_range(int idx, unsigned int size) {
    // BUG: if idx == -1, it becomes 4294967295 as unsigned, which is > size
    // But the comparison idx < size promotes idx to unsigned first!
    // So -1 < 10u evaluates to FALSE on most compilers (wraps to huge uint).
    return idx < static_cast<int>(size) && idx >= 0; // sneaky: casting wrong way
    // Actually the REAL buggy version people write:
    // return idx < size;  // -1 becomes 4294967295, comparison is false
}
} // namespace buggy

namespace fixed {
bool is_in_range(int idx, std::size_t size) {
    // Fix: check negative first, then compare as size_t
    if (idx < 0) return false;
    return static_cast<std::size_t>(idx) < size;
}
} // namespace fixed

void test() {
    // The buggy version would say -1 is NOT in range (correct by accident here)
    // but consider: the buggy code `idx < size` for idx=-1, size=10 is FALSE
    // This is the "right answer for the wrong reason"

    assert(fixed::is_in_range(5, 10) == true);
    assert(fixed::is_in_range(-1, 10) == false);
    assert(fixed::is_in_range(10, 10) == false);
    assert(fixed::is_in_range(0, 0) == false);

    std::cout << "  Bug 1 (signed/unsigned): PASS\n";
}
} // namespace bug1

// ======================== BUG 2: UNINITIALIZED VARIABLE =====================
// -Wmaybe-uninitialized catches this.

namespace bug2 {

namespace buggy {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
int classify_temperature(int temp_c) {
    int category; // BUG: uninitialized
    if (temp_c > 100) {
        category = 3; // boiling
    } else if (temp_c > 50) {
        category = 2; // hot
    } else if (temp_c > 0) {
        category = 1; // warm
    }
    // BUG: if temp_c <= 0, category is uninitialized!
    return category;
}
#pragma GCC diagnostic pop
} // namespace buggy

namespace fixed {
int classify_temperature(int temp_c) {
    if (temp_c > 100) return 3;
    if (temp_c > 50)  return 2;
    if (temp_c > 0)   return 1;
    return 0; // freezing — all paths return a value
}
} // namespace fixed

void test() {
    assert(fixed::classify_temperature(150) == 3);
    assert(fixed::classify_temperature(75) == 2);
    assert(fixed::classify_temperature(25) == 1);
    assert(fixed::classify_temperature(-10) == 0);

    std::cout << "  Bug 2 (uninitialized variable): PASS\n";
}
} // namespace bug2

// ======================== BUG 3: DANGLING REFERENCE =========================
// -Wreturn-local-addr catches returning address of local.
// bugprone-dangling-handle catches string_view cases.

namespace bug3 {

namespace buggy {
// BUG: returns reference to a local variable — instant UB when used
// const std::string& make_greeting(const std::string& name) {
//     std::string result = "Hello, " + name + "!";
//     return result; // DANGLING: result destroyed at }
// }

// Slightly more subtle: string_view to a temporary
// std::string_view get_prefix(const std::string& s) {
//     std::string trimmed = s.substr(0, 5);
//     return trimmed; // DANGLING: trimmed destroyed, string_view points to freed memory
// }
} // namespace buggy

namespace fixed {
// Fix 1: Return by value (compiler applies RVO/NRVO)
std::string make_greeting(const std::string& name) {
    std::string result = "Hello, " + name + "!";
    return result; // returned by value, no dangling
}

// Fix 2: Return a string, not a view, when the source is temporary
std::string get_prefix(const std::string& s) {
    if (s.size() <= 5) return s;
    return s.substr(0, 5); // return by value
}
} // namespace fixed

void test() {
    assert(fixed::make_greeting("World") == "Hello, World!");
    assert(fixed::get_prefix("HelloWorld") == "Hello");
    assert(fixed::get_prefix("Hi") == "Hi");

    std::cout << "  Bug 3 (dangling reference): PASS\n";
}
} // namespace bug3

// ======================== BUG 4: USE AFTER MOVE =============================
// bugprone-use-after-move catches this.

namespace bug4 {

namespace buggy {
void process_data() {
    std::vector<int> data = {1, 2, 3, 4, 5};
    std::vector<int> backup = std::move(data);

    // BUG: data is in a moved-from state. size() is valid (returns 0),
    // but accessing elements is UB — and the programmer probably intended
    // to use data, not realizing it's been emptied.
    // int first = data[0]; // UB: data is empty after move
    // std::cout << "Data size: " << data.size() << "\n"; // prints 0, not 5
}
} // namespace buggy

namespace fixed {
void process_data() {
    std::vector<int> data = {1, 2, 3, 4, 5};
    std::vector<int> backup = data; // COPY, not move — if you need both

    assert(data.size() == 5);    // data is still valid
    assert(backup.size() == 5);  // backup has independent copy

    // Or if you truly want to transfer ownership, never use data again:
    std::vector<int> transferred;
    transferred = std::move(data);
    // data is moved-from here — do NOT use data below this line
    // (reassign or let it go out of scope)
    data.clear(); // optional: reset to known empty state
}

// Better pattern: use optional to make the "consumed" state explicit
std::optional<std::vector<int>> create_data() {
    return std::vector<int>{1, 2, 3, 4, 5};
}

void consume_data(std::vector<int>&& data [[maybe_unused]]) {
    assert(!data.empty());
    // ... process ...
}
} // namespace fixed

void test() {
    fixed::process_data();
    auto data = fixed::create_data();
    assert(data.has_value());
    assert(data->size() == 5);
    fixed::consume_data(std::move(*data));

    std::cout << "  Bug 4 (use-after-move): PASS\n";
}
} // namespace bug4

// ======================== BUG 5: IMPLICIT NARROWING CONVERSION ==============
// -Wconversion catches this.

namespace bug5 {

namespace buggy {
struct Pixel {
    uint8_t r, g, b;
};

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
Pixel blend(Pixel a, Pixel b) {
    // BUG: integer promotion to int, then narrowing back to uint8_t
    // If a.r + b.r > 255, it wraps. The programmer intended averaging but
    // the intermediate result can overflow uint8_t on assignment.
    Pixel result;
    result.r = (a.r + b.r) / 2; // promotion to int, then truncation
    result.g = (a.g + b.g) / 2; // -Wconversion warns about this
    result.b = (a.b + b.b) / 2;
    return result;
}
#pragma GCC diagnostic pop
} // namespace buggy

namespace fixed {
struct Pixel {
    uint8_t r, g, b;
};

Pixel blend(Pixel a, Pixel b) {
    // Fix: explicit casts documenting the conversion
    Pixel result;
    result.r = static_cast<uint8_t>((static_cast<unsigned>(a.r) + b.r) / 2u);
    result.g = static_cast<uint8_t>((static_cast<unsigned>(a.g) + b.g) / 2u);
    result.b = static_cast<uint8_t>((static_cast<unsigned>(a.b) + b.b) / 2u);
    return result;
}
} // namespace fixed

void test() {
    fixed::Pixel a{200, 100, 50};
    fixed::Pixel b{100, 200, 150};
    fixed::Pixel result = fixed::blend(a, b);
    assert(result.r == 150);
    assert(result.g == 150);
    assert(result.b == 100);

    // Edge case: max values
    fixed::Pixel white{255, 255, 255};
    fixed::Pixel black{0, 0, 0};
    result = fixed::blend(white, black);
    assert(result.r == 127); // (255+0)/2 = 127
    assert(result.g == 127);

    std::cout << "  Bug 5 (narrowing conversion): PASS\n";
}
} // namespace bug5

// ======================== TESTS =============================================

int main() {
    std::cout << "=== Static Analysis Exercise ===\n";
    bug1::test();
    bug2::test();
    bug3::test();
    bug4::test();
    bug5::test();
    std::cout << "=== ALL TESTS PASSED ===\n";
    return 0;
}
