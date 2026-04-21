// Exercise 5: Syscall Reduction — Minimizing I/O Overhead
//
// Demonstrates how system call frequency affects performance.
// Compare: many small writes vs buffered writes vs mmap.
//
// Self-test: Verifies all methods produce the same output.
// Performance: Run with `strace -c` to count syscalls.
//
// TOOL PRACTICE:
//   strace -c ./ex05_syscall_reduction
//   strace -c ./ex05_syscall_reduction --bench
//   strace -e trace=write ./ex05_syscall_reduction --bench

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace syscall_demo {

constexpr size_t kLineCount = 10000;

// Generate sample data
std::vector<std::string> generate_data(size_t count) {
    std::vector<std::string> lines;
    lines.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        char buf[128];
        std::snprintf(buf, sizeof(buf),
            "sensor_%04zu: value=%.6f status=OK timestamp=%zu",
            i, static_cast<double>(i) * 0.001, i * 100);
        lines.emplace_back(buf);
    }
    return lines;
}

// Method 1: Write each line separately (many syscalls)
// Each fprintf/fwrite is a separate write() syscall if unbuffered
void write_unbuffered(const std::vector<std::string>& lines, const char* path) {
    FILE* f = std::fopen(path, "w");
    assert(f != nullptr);
    // Disable stdio buffering to force one syscall per write
    std::setvbuf(f, nullptr, _IONBF, 0);
    for (const auto& line : lines) {
        std::fwrite(line.c_str(), 1, line.size(), f);
        std::fwrite("\n", 1, 1, f);
    }
    std::fclose(f);
}

// Method 2: Use default stdio buffering (few syscalls)
void write_buffered(const std::vector<std::string>& lines, const char* path) {
    FILE* f = std::fopen(path, "w");
    assert(f != nullptr);
    // Default buffering: _IOFBF with 8KB buffer
    for (const auto& line : lines) {
        std::fprintf(f, "%s\n", line.c_str());
    }
    std::fclose(f);
}

// Method 3: Build in memory, single write
void write_bulk(const std::vector<std::string>& lines, const char* path) {
    // Pre-calculate total size
    size_t total = 0;
    for (const auto& line : lines)
        total += line.size() + 1;  // +1 for newline

    // Build single buffer
    std::string buffer;
    buffer.reserve(total);
    for (const auto& line : lines) {
        buffer += line;
        buffer += '\n';
    }

    // Single write
    FILE* f = std::fopen(path, "w");
    assert(f != nullptr);
    std::fwrite(buffer.c_str(), 1, buffer.size(), f);
    std::fclose(f);
}

// Method 4: C++ ofstream (for comparison)
void write_stream(const std::vector<std::string>& lines, const char* path) {
    std::ofstream ofs(path);
    assert(ofs.good());
    for (const auto& line : lines)
        ofs << line << '\n';
}

// Read file into string for comparison
std::string read_file(const char* path) {
    std::ifstream ifs(path);
    std::string content((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());
    return content;
}

void test() {
    auto lines = generate_data(100);  // small for test

    write_unbuffered(lines, "/tmp/ex05_unbuf.txt");
    write_buffered(lines, "/tmp/ex05_buf.txt");
    write_bulk(lines, "/tmp/ex05_bulk.txt");
    write_stream(lines, "/tmp/ex05_stream.txt");

    std::string ref = read_file("/tmp/ex05_unbuf.txt");
    assert(!ref.empty());
    assert(read_file("/tmp/ex05_buf.txt") == ref);
    assert(read_file("/tmp/ex05_bulk.txt") == ref);
    assert(read_file("/tmp/ex05_stream.txt") == ref);

    // Cleanup
    std::remove("/tmp/ex05_unbuf.txt");
    std::remove("/tmp/ex05_buf.txt");
    std::remove("/tmp/ex05_bulk.txt");
    std::remove("/tmp/ex05_stream.txt");
}

void benchmark() {
    auto lines = generate_data(kLineCount);

    struct Result {
        const char* name;
        long us;
    };
    std::vector<Result> results;

    auto time_it = [&](const char* name, auto fn) {
        auto t0 = std::chrono::steady_clock::now();
        fn();
        auto t1 = std::chrono::steady_clock::now();
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        results.push_back({name, us});
    };

    time_it("unbuffered", [&]{ write_unbuffered(lines, "/tmp/ex05_perf_unbuf.txt"); });
    time_it("buffered",   [&]{ write_buffered(lines, "/tmp/ex05_perf_buf.txt"); });
    time_it("bulk",       [&]{ write_bulk(lines, "/tmp/ex05_perf_bulk.txt"); });
    time_it("ofstream",   [&]{ write_stream(lines, "/tmp/ex05_perf_stream.txt"); });

    std::printf("  %-12s %8s  %s\n", "Method", "Time(us)", "Relative");
    long baseline = results[0].us;
    if (baseline == 0) baseline = 1;
    for (const auto& r : results) {
        std::printf("  %-12s %8ld  %.1fx\n", r.name, r.us,
                    static_cast<double>(r.us) / static_cast<double>(baseline));
    }

    // Cleanup
    std::remove("/tmp/ex05_perf_unbuf.txt");
    std::remove("/tmp/ex05_perf_buf.txt");
    std::remove("/tmp/ex05_perf_bulk.txt");
    std::remove("/tmp/ex05_perf_stream.txt");
}

}  // namespace syscall_demo

// ========================================================================
// Main
// ========================================================================

int main(int argc, char** argv) {
    syscall_demo::test();
    std::printf("ex05_syscall_reduction: ALL TESTS PASSED\n");

    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0)
            bench = true;
    }

    if (bench) {
        std::printf("\n=== BENCHMARK (run with 'strace -c' to see syscall counts) ===\n");
        std::printf("  Writing %zu lines per method:\n", syscall_demo::kLineCount);
        syscall_demo::benchmark();
    }

    return 0;
}
