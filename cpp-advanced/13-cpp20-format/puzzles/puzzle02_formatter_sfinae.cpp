// Module 13 — Puzzle 02: Formatter SFINAE / Concepts
// Compiler: GCC 13+ or Clang 17+ with -std=c++20
//
// Explores what happens when you try to format a type without a formatter,
// and how to build compile-time checks using C++20 concepts.
//
// Puzzles:
//   1. What error do you get formatting a type with no formatter?
//   2. Write a Formattable concept
//   3. Use it for better error messages in a logging function
//   4. Conditional formatting with if-constexpr

#include <concepts>
#include <format>
#include <iostream>
#include <string>
#include <type_traits>

// =============================================================================
// Puzzle 1: What happens with no formatter?
// =============================================================================

// A type with NO std::formatter specialisation
struct Opaque {
    int internal_id;
    double secret_value;
};

// Uncomment this to see the compile error:
//
// void puzzle1_no_formatter() {
//     Opaque obj{42, 3.14};
//     std::string s = std::format("obj={}", obj);
//     // ERROR: "no matching function for call to 'format'"
//     // or: "cannot format argument... formatter<Opaque> is disabled"
//     //
//     // The error is correct but can be confusing in a deep template stack.
//     // That's what puzzle 2 solves.
// }

// =============================================================================
// Puzzle 2: The Formattable concept
// =============================================================================

// This concept checks whether std::formatter<T> is valid and usable.
// It verifies that std::format("{}", val) would compile.
//
// How it works:
//   std::formattable<T, char> is a standard C++23 concept, but we can
//   build our own for C++20 using SFINAE / requires expressions.

template <typename T>
concept Formattable = requires(T val, std::format_context ctx) {
    // Check that std::formatter<std::remove_cvref_t<T>> can be instantiated
    // and has a valid format() method
    {
        std::formatter<std::remove_cvref_t<T>>{}.format(val, ctx)
    } -> std::convertible_to<std::format_context::iterator>;
};

// Verify our concept works at compile time:
static_assert(Formattable<int>,         "int should be formattable");
static_assert(Formattable<double>,      "double should be formattable");
static_assert(Formattable<std::string>, "string should be formattable");
static_assert(Formattable<const char*>, "const char* should be formattable");
static_assert(Formattable<bool>,        "bool should be formattable");

// Opaque has no formatter, so this should be false:
static_assert(!Formattable<Opaque>,     "Opaque should NOT be formattable");

// =============================================================================
// Puzzle 3: Better error messages with constrained logging
// =============================================================================

// A logging function that gives a clear error when you pass an unformattable type.
// Without the concept, you get pages of template errors.
// With it, you get: "T does not satisfy Formattable"

// Constrained: each argument must be Formattable
template <Formattable... Args>
void safe_log(std::format_string<Args...> fmt, Args&&... args) {
    std::cout << "[LOG] " << std::format(fmt, std::forward<Args>(args)...) << '\n';
}

// Uncomment to see the improved error message:
//
// void puzzle3_better_error() {
//     Opaque obj{42, 3.14};
//     safe_log("obj={}", obj);
//     // ERROR: "constraints not satisfied" / "Opaque does not satisfy Formattable"
//     // Much clearer than the raw template error!
// }

// =============================================================================
// Puzzle 4: Conditional formatting with if-constexpr
// =============================================================================

// Sometimes you want a function that handles both formattable and
// non-formattable types gracefully at runtime, using a fallback.

template <typename T>
std::string to_debug_string(const T& val) {
    if constexpr (Formattable<T>) {
        // Type has a formatter — use std::format
        return std::format("{}", val);
    } else {
        // No formatter — produce a fallback with type name and address
        return std::format("<{}@{}>",
                           typeid(T).name(),
                           static_cast<const void*>(&val));
    }
}

// =============================================================================
// Puzzle 5: Now make Opaque formattable and re-check
// =============================================================================

// Let's add a formatter for Opaque and verify the concept updates correctly.

struct FormattedOpaque {
    int id;
    double value;
};

template <>
struct std::formatter<FormattedOpaque> {
    constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }
    auto format(const FormattedOpaque& obj, std::format_context& ctx) const {
        return std::format_to(ctx.out(), "FormattedOpaque(id={}, value={:.2f})",
                              obj.id, obj.value);
    }
};

static_assert(Formattable<FormattedOpaque>, "FormattedOpaque should now be formattable");

// =============================================================================
// Puzzle 6: Concept composition — Loggable types
// =============================================================================

// A stronger concept: Loggable means formattable AND has a name() method
// (common pattern for robot components)

template <typename T>
concept Named = requires(const T& val) {
    { val.name() } -> std::convertible_to<std::string_view>;
};

template <typename T>
concept Loggable = Formattable<T> && Named<T>;

struct Motor {
    std::string id_;
    double current_;
    double temperature_;

    std::string_view name() const { return id_; }
};

template <>
struct std::formatter<Motor> {
    constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }
    auto format(const Motor& m, std::format_context& ctx) const {
        return std::format_to(ctx.out(), "Motor({}: {:.2f}A, {:.1f}°C)",
                              m.id_, m.current_, m.temperature_);
    }
};

static_assert(Loggable<Motor>, "Motor should be Loggable");
static_assert(!Loggable<int>,  "int is Formattable but not Named → not Loggable");

// Log function that requires Loggable — adds the component name automatically
template <Loggable T>
void component_log(const T& component, std::string_view message) {
    std::cout << std::format("[{}] {}: {}\n",
                             "INFO", component.name(), message);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "=== Formatter SFINAE / Concepts Puzzle ===\n\n";

    // --- Concept verification ---
    std::cout << "--- Concept checks (compile-time, all passed) ---\n";
    std::cout << "  Formattable<int>:            true\n";
    std::cout << "  Formattable<double>:         true\n";
    std::cout << "  Formattable<std::string>:    true\n";
    std::cout << "  Formattable<Opaque>:         false (no formatter)\n";
    std::cout << "  Formattable<FormattedOpaque>: true (has formatter)\n";
    std::cout << "  Loggable<Motor>:             true (Formattable + Named)\n";
    std::cout << "  Loggable<int>:               false (Formattable but not Named)\n";

    // --- safe_log works for formattable types ---
    std::cout << "\n--- safe_log (concept-constrained) ---\n";
    safe_log("int: {}", 42);
    safe_log("float: {:.3f}", 3.14159);
    safe_log("string: {}", "hello");

    FormattedOpaque fo{99, 2.718};
    safe_log("object: {}", fo);

    // --- to_debug_string: handles both cases ---
    std::cout << "\n--- to_debug_string (if-constexpr fallback) ---\n";
    std::cout << "  int:            " << to_debug_string(42) << '\n';
    std::cout << "  double:         " << to_debug_string(3.14) << '\n';
    std::cout << "  string:         " << to_debug_string(std::string("hello")) << '\n';

    Opaque opaque{7, 1.41};
    std::cout << "  Opaque (no fmt): " << to_debug_string(opaque) << '\n';
    std::cout << "  FormattedOpaque: " << to_debug_string(fo) << '\n';

    // --- component_log: requires Loggable ---
    std::cout << "\n--- component_log (requires Loggable) ---\n";
    Motor left{"left_wheel", 2.34, 45.2};
    Motor right{"right_wheel", 2.18, 43.8};

    component_log(left, "Current nominal");
    component_log(right, "Temperature rising");

    // This won't compile — int is not Loggable:
    // component_log(42, "nope");

    std::cout << "\n--- Exercises for the reader ---\n";
    std::cout << "1. Uncomment puzzle1_no_formatter() — observe the raw error\n";
    std::cout << "2. Uncomment puzzle3_better_error() — compare the error message\n";
    std::cout << "3. Add a formatter for Opaque and verify the concept flips to true\n";
    std::cout << "4. Write a concept 'Serialisable' = Formattable + has serialize()\n";
    std::cout << "5. Combine Loggable with variadic args for a full component logger\n";

    std::cout << "\nAll puzzles complete.\n";
    return 0;
}
