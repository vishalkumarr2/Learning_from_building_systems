// =============================================================================
// Exercise 1: Value Category Identification and Proof
// =============================================================================
// Instructions:
//   1. For each expression below, predict its value category BEFORE reading
//      the static_asserts.
//   2. Understand WHY each expression has that category.
//   3. Compile to verify: g++ -std=c++2a -Wall -Wextra -Wpedantic ex01_value_categories.cpp
//   4. If any static_assert fails, your compiler disagrees with the standard.
//
// Key rules:
//   - decltype((expr)) for EXPRESSIONS (note double parens):
//       lvalue  → T&
//       xvalue  → T&&
//       prvalue → T  (no reference)
//   - decltype(name) WITHOUT extra parens gives the DECLARED TYPE, not the category.
// =============================================================================

#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

// --- Helper traits for clarity ---
template<typename T>
constexpr bool is_lvalue = std::is_lvalue_reference_v<T>;

template<typename T>
constexpr bool is_xvalue = std::is_rvalue_reference_v<T>;

template<typename T>
constexpr bool is_prvalue = !std::is_reference_v<T>;

// --- Category tag printer ---
template<typename T>
const char* category_name() {
    if constexpr (std::is_lvalue_reference_v<T>) return "lvalue";
    else if constexpr (std::is_rvalue_reference_v<T>) return "xvalue";
    else return "prvalue";
}

// --- Some helper types and functions ---
struct Widget {
    int val = 0;
    int& ref_member() & { return val; }
    int&& rval_member() && { return std::move(val); }
};

Widget make_widget() { return Widget{42}; }
int global_var = 100;
int& get_ref() { return global_var; }
int&& get_rref() { return std::move(global_var); }

int main() {
    int x = 42;
    int& ref = x;
    int&& rref = std::move(x);
    const int cx = 10;
    int arr[3] = {1, 2, 3};
    (void)cx;   // used in static_assert expressions below
    (void)arr;  // used in static_assert expressions below
    Widget w;
    std::string s = "hello";

    // =========================================================================
    // EXPRESSION 1: x  (a named variable)
    // Prediction: ______ 
    // =========================================================================
    static_assert(is_lvalue<decltype((x))>, "x is an lvalue");

    // =========================================================================
    // EXPRESSION 2: 42  (an integer literal)
    // Prediction: ______
    // =========================================================================
    static_assert(is_prvalue<decltype((42))>, "42 is a prvalue");

    // =========================================================================
    // EXPRESSION 3: std::move(x)  (explicit cast to rvalue)
    // Prediction: ______
    // =========================================================================
    static_assert(is_xvalue<decltype((std::move(x)))>, "std::move(x) is an xvalue");

    // =========================================================================
    // EXPRESSION 4: ref  (a named lvalue reference)
    // Prediction: ______
    // =========================================================================
    static_assert(is_lvalue<decltype((ref))>, "named reference is an lvalue");

    // =========================================================================
    // EXPRESSION 5: rref  (a NAMED rvalue reference — the big gotcha!)
    // Prediction: ______
    // =========================================================================
    static_assert(is_lvalue<decltype((rref))>, 
        "Named rvalue reference is an LVALUE! It has a name.");

    // =========================================================================
    // EXPRESSION 6: x + 1  (arithmetic on lvalue)
    // Prediction: ______
    // =========================================================================
    static_assert(is_prvalue<decltype((x + 1))>, "x + 1 is a prvalue");

    // =========================================================================
    // EXPRESSION 7: "hello"  (string literal)
    // Prediction: ______
    // =========================================================================
    static_assert(is_lvalue<decltype(("hello"))>, 
        "String literal is an lvalue (it has an address in .rodata)");

    // =========================================================================
    // EXPRESSION 8: arr[1]  (array subscript)
    // Prediction: ______
    // =========================================================================
    static_assert(is_lvalue<decltype((arr[1]))>, "subscript operator yields lvalue");

    // =========================================================================
    // EXPRESSION 9: cx  (const variable)
    // Prediction: ______
    // =========================================================================
    static_assert(is_lvalue<decltype((cx))>, "const variable is still an lvalue");

    // =========================================================================
    // EXPRESSION 10: make_widget()  (function returning by value)
    // Prediction: ______
    // =========================================================================
    static_assert(is_prvalue<decltype((make_widget()))>,
        "Function returning T by value yields a prvalue");

    // =========================================================================
    // EXPRESSION 11: get_ref()  (function returning T&)
    // Prediction: ______
    // =========================================================================
    static_assert(is_lvalue<decltype((get_ref()))>,
        "Function returning T& yields an lvalue");

    // =========================================================================
    // EXPRESSION 12: get_rref()  (function returning T&&)
    // Prediction: ______
    // =========================================================================
    static_assert(is_xvalue<decltype((get_rref()))>,
        "Function returning T&& yields an xvalue");

    // =========================================================================
    // EXPRESSION 13: static_cast<int&&>(x)  (explicit xvalue cast)
    // Prediction: ______
    // =========================================================================
    static_assert(is_xvalue<decltype((static_cast<int&&>(x)))>,
        "Casting to T&& produces an xvalue — this is what std::move does");

    // =========================================================================
    // EXPRESSION 14: std::move(w).val  (member of rvalue)
    // Prediction: ______
    // =========================================================================
    static_assert(is_xvalue<decltype((std::move(w).val))>,
        "Member access on xvalue yields xvalue");

    // =========================================================================
    // EXPRESSION 15: w.val  (member of lvalue)
    // Prediction: ______
    // =========================================================================
    static_assert(is_lvalue<decltype((w.val))>, "Member of lvalue is lvalue");

    // --- Print results at runtime for visual confirmation ---
    std::cout << "=== Value Category Proof ===\n\n";
    std::cout << " 1. x                    : " << category_name<decltype((x))>() << "\n";
    std::cout << " 2. 42                   : " << category_name<decltype((42))>() << "\n";
    std::cout << " 3. std::move(x)         : " << category_name<decltype((std::move(x)))>() << "\n";
    std::cout << " 4. ref                  : " << category_name<decltype((ref))>() << "\n";
    std::cout << " 5. rref (NAMED!)        : " << category_name<decltype((rref))>() << "\n";
    std::cout << " 6. x + 1               : " << category_name<decltype((x + 1))>() << "\n";
    std::cout << " 7. \"hello\"              : " << category_name<decltype(("hello"))>() << "\n";
    std::cout << " 8. arr[1]              : " << category_name<decltype((arr[1]))>() << "\n";
    std::cout << " 9. cx                   : " << category_name<decltype((cx))>() << "\n";
    std::cout << "10. make_widget()        : " << category_name<decltype((make_widget()))>() << "\n";
    std::cout << "11. get_ref()            : " << category_name<decltype((get_ref()))>() << "\n";
    std::cout << "12. get_rref()           : " << category_name<decltype((get_rref()))>() << "\n";
    std::cout << "13. static_cast<int&&>(x): " << category_name<decltype((static_cast<int&&>(x)))>() << "\n";
    std::cout << "14. std::move(w).val     : " << category_name<decltype((std::move(w).val))>() << "\n";
    std::cout << "15. w.val               : " << category_name<decltype((w.val))>() << "\n";

    std::cout << "\nAll 15 static_asserts passed — your predictions are correct!\n";
    return 0;
}
