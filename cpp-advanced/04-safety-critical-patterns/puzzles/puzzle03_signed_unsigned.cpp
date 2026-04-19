// =============================================================================
// Puzzle 03: Signed/Unsigned Comparison Traps
// =============================================================================
// 5 traps that produce wrong results due to signed/unsigned comparison.
// For each: predict the output, then read the explanation.
//
// Compile with: g++ -std=c++20 -o puzzle03 puzzle03_signed_unsigned.cpp
// NOTE: -Wsign-compare would warn on ALL of these. That's the point.
// =============================================================================

#include <cstdint>
#include <cstddef>
#include <iostream>
#include <vector>
#include <string>

// Helper to print trap results
void trap(int num, const std::string& desc, bool result, bool expected,
          const std::string& explanation) {
    std::cout << "\n--- Trap " << num << ": " << desc << " ---\n";
    std::cout << "  Result:   " << (result ? "true" : "false") << "\n";
    std::cout << "  Expected: " << (expected ? "true" : "false") << "\n";
    std::cout << "  Correct:  " << (result == expected ? "YES" : "*** SURPRISE! ***") << "\n";
    std::cout << "  Why: " << explanation << "\n";
}

int main() {
    std::cout << "=== Puzzle 03: Signed/Unsigned Comparison Traps ===\n";
    std::cout << "Each trap shows a comparison that might not do what you think.\n";

    // ======================== TRAP 1 ========================================
    // "Is -1 less than 1?"
    {
        int a = -1;
        unsigned int b = 1;
        bool result = (a < b);
        // You'd think: -1 < 1 is obviously true. Right?
        // WRONG! When comparing int to unsigned int, the int is converted
        // to unsigned. -1 becomes 4294967295 (UINT_MAX). 4294967295 < 1 is FALSE.
        trap(1, "Is -1 < 1u ?",
             result, false,
             "-1 is promoted to unsigned (4294967295). 4294967295 < 1 is false!");
    }

    // ======================== TRAP 2 ========================================
    // "Is this vector index valid?"
    {
        std::vector<int> v = {10, 20, 30};
        int index = -1;
        // Common pattern: check if index is in bounds
        bool result = (index < v.size()); // v.size() returns size_t (unsigned)
        // index (-1) is promoted to size_t: becomes 18446744073709551615 on 64-bit
        // That's NOT less than 3.
        trap(2, "Is index=-1 < v.size()=3 ?",
             result, false,
             "index=-1 promoted to size_t becomes huge (18446744073709551615). Not < 3!");
    }

    // ======================== TRAP 3 ========================================
    // "Loop that never runs"
    {
        unsigned int limit = 0;
        int count = 0;
        for (unsigned int i = 0; i < limit - 1; ++i) {
            // You might think: limit is 0, so limit-1 is -1, loop doesn't run.
            // WRONG! limit is unsigned. 0u - 1 = UINT_MAX (4294967295).
            // The loop runs ~4 billion times!
            count++;
            if (count > 5) break; // safety: don't actually run billions of times
        }
        bool ran = (count > 0);
        trap(3, "Does 'for(i=0; i < 0u-1; i++)' run?",
             ran, true,
             "0u - 1 wraps to UINT_MAX (4294967295). The loop runs ~4 billion times!");
    }

    // ======================== TRAP 4 ========================================
    // "Which is bigger: -1 or 0?"
    {
        int32_t a = -1;
        uint32_t b = 0;
        bool result = (a > b);
        // -1 is converted to uint32_t: becomes 4294967295.
        // 4294967295 > 0? YES! So -1 appears "greater than" 0.
        trap(4, "Is -1 > 0u ?",
             result, true,
             "-1 as uint32_t = 4294967295. So 4294967295 > 0 is true. "
             "Negative numbers appear HUGE!");
    }

    // ======================== TRAP 5 ========================================
    // "Subtracting sizes"
    {
        std::vector<int> shorter = {1, 2, 3};
        std::vector<int> longer = {1, 2, 3, 4, 5};
        // Calculate difference in sizes
        auto diff = shorter.size() - longer.size();
        // You'd think: 3 - 5 = -2
        // But size_t is unsigned! 3 - 5 wraps to a huge positive number.
        bool is_negative = (diff < 0); // Always false because diff is unsigned!
        bool is_huge = (diff > 1000000);

        trap(5, "Is (3-element.size() - 5-element.size()) negative?",
             is_negative, false,
             "size_t is unsigned. 3 - 5 wraps to 18446744073709551614. "
             "The 'difference' is a huge positive number, not -2!");

        std::cout << "  Actual diff value: " << diff << "\n";
        std::cout << "  Is diff > 1000000? " << (is_huge ? "YES!" : "no") << "\n";
    }

    // ======================== FIXES =========================================
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "FIXES:\n";
    std::cout << "  1. ALWAYS check negative values BEFORE comparing with unsigned:\n";
    std::cout << "     if (index >= 0 && static_cast<size_t>(index) < v.size())\n";
    std::cout << "  2. Use std::ssize() (C++20) to get signed sizes:\n";
    std::cout << "     if (index < std::ssize(v))\n";
    std::cout << "  3. Use std::cmp_less (C++20) for safe mixed comparisons:\n";
    std::cout << "     if (std::cmp_less(a, b)) // handles signed/unsigned correctly\n";
    std::cout << "  4. Compile with -Wsign-compare to catch ALL of these at compile time\n";
    std::cout << "  5. For size differences, cast to ptrdiff_t first:\n";
    std::cout << "     ptrdiff_t diff = static_cast<ptrdiff_t>(a.size()) - b.size();\n";
    std::cout << std::string(60, '=') << "\n";

    // Demonstrate the C++20 fix
    std::cout << "\n--- C++20 Fix: std::cmp_less ---\n";
    int neg = -1;
    unsigned int pos = 1;
#if __cplusplus >= 202002L && __has_include(<utility>) && defined(__cpp_lib_integer_comparison_functions)
    std::cout << "  std::cmp_less(-1, 1u) = "
              << std::cmp_less(neg, pos) << " (correct: true)\n";
#else
    // Manual safe comparison for compilers without std::cmp_less
    auto safe_cmp_less = [](int a, unsigned int b) -> bool {
        if (a < 0) return true;
        return static_cast<unsigned int>(a) < b;
    };
    std::cout << "  safe_cmp_less(-1, 1u) = "
              << safe_cmp_less(neg, pos) << " (correct: true)\n";
    std::cout << "  (std::cmp_less not available on this compiler)\n";
#endif
    std::cout << "  (-1 < 1u)             = "
              << (neg < pos) << " (WRONG: false)\n";

    return 0;
}
