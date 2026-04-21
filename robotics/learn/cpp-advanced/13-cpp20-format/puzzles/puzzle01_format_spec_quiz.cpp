// Module 13 — Puzzle 01: Format Spec Quiz
// Compiler: GCC 13+ or Clang 17+ with -std=c++20
//
// For each format call below, predict the output BEFORE running.
// Then compile and run to check your answers.
//
// This tests your understanding of the format specification mini-language:
//   [[fill]align][sign][#][0][width][.precision][L][type]

#include <cmath>
#include <format>
#include <iostream>
#include <limits>
#include <string>

int main() {
    std::cout << "=== Format Specification Quiz ===\n";
    std::cout << "Predict each output, then check against the actual result.\n";
    std::cout << "Outputs are shown between | pipes | for clarity.\n\n";

    int q = 0;

    // Q1: Basic right-align with zero-pad
    // What does {:08d} do to a negative number?
    std::cout << std::format("Q{:02d}: |{:08d}|\n", ++q, -42);
    // YOUR PREDICTION: _______________

    // Q2: Left-align with fill character
    // Does the fill go on the right for left-align?
    std::cout << std::format("Q{:02d}: |{:~<10}|\n", ++q, "hello");
    // YOUR PREDICTION: _______________

    // Q3: Centre-align — odd-length padding
    // When padding is odd, where does the extra character go?
    std::cout << std::format("Q{:02d}: |{:^9}|\n", ++q, "AB");
    // YOUR PREDICTION: _______________

    // Q4: Precision on a string — truncation!
    // .5 on a string means "at most 5 characters"
    std::cout << std::format("Q{:02d}: |{:.5}|\n", ++q, "Hello, World!");
    // YOUR PREDICTION: _______________

    // Q5: Width + precision on string — truncate then pad
    std::cout << std::format("Q{:02d}: |{:>10.5}|\n", ++q, "Hello, World!");
    // YOUR PREDICTION: _______________

    // Q6: Hex with alternate form and uppercase
    std::cout << std::format("Q{:02d}: |{:#010X}|\n", ++q, 255);
    // YOUR PREDICTION: _______________

    // Q7: Binary with alternate form
    std::cout << std::format("Q{:02d}: |{:#010b}|\n", ++q, 42);
    // YOUR PREDICTION: _______________

    // Q8: Float with sign — positive number with space
    std::cout << std::format("Q{:02d}: |{: .2f}|\n", ++q, 3.14);
    // YOUR PREDICTION: _______________

    // Q9: Float with sign — negative number with space
    std::cout << std::format("Q{:02d}: |{: .2f}|\n", ++q, -3.14);
    // YOUR PREDICTION: _______________

    // Q10: NaN formatting
    double nan_val = std::numeric_limits<double>::quiet_NaN();
    std::cout << std::format("Q{:02d}: |{:>10}|\n", ++q, nan_val);
    // YOUR PREDICTION: _______________

    // Q11: Infinity formatting
    double inf_val = std::numeric_limits<double>::infinity();
    std::cout << std::format("Q{:02d}: |{:+>10}|\n", ++q, inf_val);
    // YOUR PREDICTION: _______________

    // Q12: Negative infinity
    std::cout << std::format("Q{:02d}: |{:>10}|\n", ++q, -inf_val);
    // YOUR PREDICTION: _______________

    // Q13: Empty string with width
    std::cout << std::format("Q{:02d}: |{:>10}|\n", ++q, "");
    // YOUR PREDICTION: _______________

    // Q14: Bool as text vs integer
    std::cout << std::format("Q{:02d}: |{}| vs |{:d}|\n", ++q, true, true);
    // YOUR PREDICTION: _______________

    // Q15: Char type specifier
    std::cout << std::format("Q{:02d}: |{:c}| |{:c}| |{:c}|\n", ++q, 72, 101, 121);
    // YOUR PREDICTION: _______________

    // Q16: Positional arguments — tricky reuse
    std::cout << std::format("Q{:02d}: |{1}{0}{1}|\n", ++q, "B", "A");
    // YOUR PREDICTION: _______________

    // Q17: Very small float — default vs fixed
    double tiny = 0.000001234;
    std::cout << std::format("Q{:02d}: default=|{}| fixed=|{:.10f}| sci=|{:.3e}|\n",
                             ++q, tiny, tiny, tiny);
    // YOUR PREDICTION: _______________

    // Q18: Large integer with zero-pad — does sign eat a position?
    std::cout << std::format("Q{:02d}: |{:+010d}|\n", ++q, 42);
    // YOUR PREDICTION: _______________

    // Q19: Hex of zero
    std::cout << std::format("Q{:02d}: |{:#x}|\n", ++q, 0);
    // YOUR PREDICTION: _______________

    // Q20: Precision 0 on a float — no decimal point
    std::cout << std::format("Q{:02d}: |{:.0f}|\n", ++q, 3.7);
    // YOUR PREDICTION: _______________

    std::cout << "\n=== Answers ===\n";
    std::cout << "Q01: |-0000042|  — zero-pad preserves sign, pads between sign and digits\n";
    std::cout << "Q02: |hello~~~~~|  — left-align, fill ~ on the right\n";
    std::cout << "Q03: |   AB    | or similar  — extra space goes to the right\n";
    std::cout << "Q04: |Hello|  — precision truncates strings\n";
    std::cout << "Q05: |     Hello|  — truncate to 5, then right-align in 10\n";
    std::cout << "Q06: |0X000000FF|  — alternate form includes 0X, zero-pad to width 10\n";
    std::cout << "Q07: |0b00101010|  — binary with 0b prefix, zero-pad to width 10\n";
    std::cout << "Q08: | 3.14|  — space sign: positive gets leading space\n";
    std::cout << "Q09: |-3.14|  — space sign: negative gets minus\n";
    std::cout << "Q10: |       nan|  — NaN right-aligned in width 10\n";
    std::cout << "Q11: |      +inf|  — +inf with explicit + sign\n";
    std::cout << "Q12: |      -inf|  — -inf right-aligned\n";
    std::cout << "Q13: |          |  — empty string, 10 spaces\n";
    std::cout << "Q14: |true| vs |1|  — default=text, :d=integer\n";
    std::cout << "Q15: |H| |e| |y|  — ASCII 72=H, 101=e, 121=y\n";
    std::cout << "Q16: |ABA|  — wait, arg 0 is the quiz number! check this one\n";
    std::cout << "Q17: check scientific notation rules\n";
    std::cout << "Q18: |+000000042|  — sign + 9 digits = width 10\n";
    std::cout << "Q19: |0|  — alternate form for 0 is just '0' (no 0x prefix)\n";
    std::cout << "Q20: |4|  — .0f rounds 3.7 to 4, no decimal point\n";

    std::cout << "\nRun the program and compare line-by-line!\n";

    return 0;
}
