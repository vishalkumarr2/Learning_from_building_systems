// Module 13 — Exercise 03: Format Performance Benchmark
// Compiler: GCC 13+ or Clang 17+ with -std=c++20
//
// Benchmarks:
//   1. snprintf     — C-style, fast but unsafe
//   2. ostringstream — C++ streams, safe but slow
//   3. std::format   — C++20, safe and fast
//   4. std::format_to — C++20 with pre-allocated buffer
//   5. Manual string concatenation (operator+)
//
// Build with optimisations for meaningful results:
//   g++ -std=c++20 -O2 -Wall -Wextra ex03_format_benchmark.cpp -o ex03_format_benchmark

#include <chrono>
#include <cstdio>
#include <format>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Use volatile sink to prevent the optimiser from eliminating our work.
// We accumulate a checksum from the output to ensure side effects.
static volatile size_t g_sink = 0;

// Helper: time a callable, return elapsed milliseconds
template <typename Func>
double bench(const char* label, int iterations, Func&& fn) {
    // Warm-up: 1000 iterations to fill caches
    for (int i = 0; i < 1000; ++i) {
        fn(i);
    }

    auto start = std::chrono::high_resolution_clock::now();
    size_t checksum = 0;
    for (int i = 0; i < iterations; ++i) {
        checksum += fn(i);
    }
    auto end = std::chrono::high_resolution_clock::now();
    g_sink = checksum;  // prevent dead-code elimination

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << std::format("  {:<25} {:>8.2f} ms  ({} iters)\n",
                             label, ms, iterations);
    return ms;
}

int main() {
    std::cout << "=== std::format Performance Benchmark ===\n";
    std::cout << "Formatting: sensor_name + float + int + hex, repeated N times\n";
    std::cout << "Build with -O2 for meaningful results!\n\n";

    constexpr int N = 100'000;

    // Test data — simulates formatting a sensor log line
    const char* sensor_name = "sensorbar_front";
    double value = 72.345678;
    int seq_num = 42;
    unsigned int reg = 0xDEADBEEF;

    // -----------------------------------------------------------------
    // Benchmark 1: snprintf
    // -----------------------------------------------------------------
    double t_snprintf = bench("snprintf", N, [&](int i) -> size_t {
        char buf[256];
        int n = snprintf(buf, sizeof(buf),
                         "[%06d] %s: value=%.3f reg=0x%08X",
                         seq_num + i, sensor_name, value + i * 0.001, reg);
        return static_cast<size_t>(n);
    });

    // -----------------------------------------------------------------
    // Benchmark 2: ostringstream
    // -----------------------------------------------------------------
    double t_oss = bench("ostringstream", N, [&](int i) -> size_t {
        std::ostringstream oss;
        oss << "[" << std::setw(6) << std::setfill('0') << (seq_num + i) << "] "
            << sensor_name << ": value="
            << std::fixed << std::setprecision(3) << (value + i * 0.001)
            << " reg=0x" << std::hex << std::setw(8) << std::setfill('0')
            << std::uppercase << reg;
        return oss.str().size();
    });

    // -----------------------------------------------------------------
    // Benchmark 3: std::format (returns std::string)
    // -----------------------------------------------------------------
    double t_format = bench("std::format", N, [&](int i) -> size_t {
        std::string s = std::format("[{:06d}] {}: value={:.3f} reg=0x{:08X}",
                                    seq_num + i, sensor_name,
                                    value + i * 0.001, reg);
        return s.size();
    });

    // -----------------------------------------------------------------
    // Benchmark 4: std::format_to (pre-allocated buffer)
    // -----------------------------------------------------------------
    double t_format_to = bench("std::format_to (prealloc)", N, [&](int i) -> size_t {
        std::string buf;
        buf.reserve(128);  // pre-allocate
        std::format_to(std::back_inserter(buf),
                       "[{:06d}] {}: value={:.3f} reg=0x{:08X}",
                       seq_num + i, sensor_name,
                       value + i * 0.001, reg);
        return buf.size();
    });

    // -----------------------------------------------------------------
    // Benchmark 5: std::string concatenation (operator+)
    // -----------------------------------------------------------------
    double t_concat = bench("string concat (+)", N, [&](int i) -> size_t {
        // Intentionally clunky — this is what people do without std::format
        std::string s = "[";
        // Manual zero-pad for seq
        std::string seq_str = std::to_string(seq_num + i);
        while (seq_str.size() < 6) seq_str = "0" + seq_str;
        s += seq_str + "] " + sensor_name + ": value=";

        // Manual precision control is painful
        char vbuf[32];
        snprintf(vbuf, sizeof(vbuf), "%.3f", value + i * 0.001);
        s += vbuf;

        char rbuf[32];
        snprintf(rbuf, sizeof(rbuf), "0x%08X", reg);
        s += " reg=" + std::string(rbuf);

        return s.size();
    });

    // -----------------------------------------------------------------
    // Results summary
    // -----------------------------------------------------------------
    std::cout << "\n=== Results Summary ===\n";
    std::cout << std::format("{:<28} {:>8} {:>10}\n", "Method", "Time(ms)", "Relative");
    std::cout << std::string(48, '-') << '\n';

    struct Result {
        const char* name;
        double ms;
    };
    std::vector<Result> results = {
        {"snprintf",               t_snprintf},
        {"ostringstream",          t_oss},
        {"std::format",            t_format},
        {"std::format_to",         t_format_to},
        {"string concat (+)",      t_concat},
    };

    for (const auto& r : results) {
        double relative = r.ms / t_snprintf;
        std::cout << std::format("{:<28} {:>7.2f}  {:>9.2f}×\n",
                                 r.name, r.ms, relative);
    }

    std::cout << "\n--- Key Takeaways ---\n";
    std::cout << "• std::format should be within ~1.5× of snprintf\n";
    std::cout << "• ostringstream is typically 3-5× slower than snprintf\n";
    std::cout << "• std::format_to with pre-allocated buffer is fastest C++20 option\n";
    std::cout << "• string concatenation is both slow AND unreadable\n";
    std::cout << "\nNote: Results vary by compiler, flags, and hardware.\n";
    std::cout << "Build with -O2 for representative numbers.\n";

    return 0;
}
