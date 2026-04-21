// ex01_exception_cost.cpp — Benchmark: exceptions vs error codes vs std::optional
// Compile: g++ -std=c++20 -Wall -Wextra -Wpedantic -O2 -pthread ex01_exception_cost.cpp -o ex01
//
// KEY INSIGHT: Exceptions follow the "zero-cost abstraction" model on the HAPPY path.
// Modern compilers (GCC/Clang) use table-based exception handling (Itanium ABI):
//   - Happy path: NO runtime overhead — no bookkeeping, no checks, no branches.
//     The compiler emits "unwind tables" in a separate .eh_frame section that are
//     only consulted when an exception is actually thrown.
//   - Error path: EXTREMELY expensive (1000x–10000x slower than error codes) because:
//     1. _Unwind_RaiseException walks the call stack using DWARF unwind info
//     2. Each frame's personality routine is called to find matching catch clauses
//     3. Stack unwinding destroys locals (calling destructors) frame by frame
//     4. RTTI (dynamic_cast-like) matching compares exception type against catch types
//     5. Memory allocation for the exception object itself (via __cxa_allocate_exception)
//
// Error codes: tiny constant cost on both paths (just a branch).
// std::optional: similar to error codes but carries no error info — just "empty".

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

static constexpr int ITERATIONS = 1'000'000;

// ── Validation functions ─────────────────────────────────────────────

// 1. Exception-based
inline int validate_exception(int x) {
    if (x < 0) {
        throw std::invalid_argument("negative value");
    }
    return x * 2;
}

// 2. Error-code-based (returns 0 on success, -1 on error; result via out-param)
inline int validate_errcode(int x, int& result) {
    if (x < 0) {
        return -1;
    }
    result = x * 2;
    return 0;
}

// 3. std::optional-based
inline std::optional<int> validate_optional(int x) {
    if (x < 0) {
        return std::nullopt;
    }
    return x * 2;
}

// ── Benchmark harness ────────────────────────────────────────────────

struct BenchResult {
    std::string method;
    int64_t happy_ns;
    int64_t error_ns;
};

template <typename F>
int64_t measure_ns(F&& func) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < ITERATIONS; ++i) {
        func(i);
    }
    auto end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
}

int main() {
    // Volatile sink to prevent the compiler from optimizing away results
    volatile int sink = 0;

    // ── Exception benchmark ──────────────────────────────────────────
    int64_t exc_happy = measure_ns([&](int i) {
        sink = validate_exception(i);  // i >= 0 → always succeeds
    });

    int64_t exc_error = measure_ns([&](int) {
        try {
            sink = validate_exception(-1);
        } catch (const std::invalid_argument&) {
            sink = -1;
        }
    });

    // ── Error code benchmark ─────────────────────────────────────────
    int result = 0;
    int64_t ec_happy = measure_ns([&](int i) {
        validate_errcode(i, result);
        sink = result;
    });

    int64_t ec_error = measure_ns([&](int) {
        int r = 0;
        if (validate_errcode(-1, r) != 0) {
            sink = -1;
        } else {
            sink = r;
        }
    });

    // ── std::optional benchmark ──────────────────────────────────────
    int64_t opt_happy = measure_ns([&](int i) {
        auto v = validate_optional(i);
        sink = v.value_or(-1);
    });

    int64_t opt_error = measure_ns([&](int) {
        auto v = validate_optional(-1);
        sink = v.value_or(-1);
    });

    // ── Print results ────────────────────────────────────────────────
    auto results = {
        BenchResult{"exception",      exc_happy, exc_error},
        BenchResult{"error_code",     ec_happy,  ec_error},
        BenchResult{"std::optional",  opt_happy, opt_error},
    };

    std::cout << "\n";
    std::cout << std::left << std::setw(16) << "Method"
              << std::right << std::setw(14) << "Happy (ns)"
              << std::setw(14) << "Error (ns)"
              << std::setw(14) << "Ratio E/H"
              << "\n";
    std::cout << std::string(58, '-') << "\n";

    for (const auto& r : results) {
        double ratio = (r.happy_ns > 0)
            ? static_cast<double>(r.error_ns) / static_cast<double>(r.happy_ns)
            : 0.0;
        std::cout << std::left << std::setw(16) << r.method
                  << std::right << std::setw(14) << r.happy_ns
                  << std::setw(14) << r.error_ns
                  << std::setw(13) << std::fixed << std::setprecision(1) << ratio << "x"
                  << "\n";
    }

    std::cout << "\n(Each method called " << ITERATIONS << " times. "
              << "Times are total, not per-call.)\n";
    std::cout << "\nExpected: exception happy ≈ error_code happy (zero-cost on happy path),\n"
              << "          exception error >> error_code error   (stack unwinding + RTTI).\n\n";

    return 0;
}
