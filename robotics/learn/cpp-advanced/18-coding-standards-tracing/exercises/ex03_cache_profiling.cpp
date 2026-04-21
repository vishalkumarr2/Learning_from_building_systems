// Exercise 3: Cache-Friendly Data Access Patterns
//
// Demonstrates the performance impact of data layout on CPU cache:
//   Part A: Row-major vs column-major matrix traversal
//   Part B: Array of Structs (AoS) vs Struct of Arrays (SoA)
//   Part C: Sequential vs strided access
//
// Self-test: Verifies correctness of all implementations.
// Performance: Run with `perf stat` to see cache miss differences.
//
// TOOL PRACTICE:
//   perf stat -e cache-misses,cache-references,L1-dcache-load-misses ./ex03_cache_profiling
//   valgrind --tool=cachegrind ./ex03_cache_profiling

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

// ========================================================================
// Part A: Row-major vs Column-major Matrix Traversal
// ========================================================================

namespace matrix_access {

constexpr size_t kSize = 512;

// Row-major access: sequential in memory (cache-friendly)
double sum_row_major(const std::vector<std::vector<double>>& mat) {
    double sum = 0.0;
    for (size_t i = 0; i < kSize; ++i)
        for (size_t j = 0; j < kSize; ++j)
            sum += mat[i][j];
    return sum;
}

// Column-major access: stride = kSize * sizeof(double) (cache-hostile)
double sum_col_major(const std::vector<std::vector<double>>& mat) {
    double sum = 0.0;
    for (size_t j = 0; j < kSize; ++j)
        for (size_t i = 0; i < kSize; ++i)
            sum += mat[i][j];
    return sum;
}

void test() {
    std::vector<std::vector<double>> mat(kSize, std::vector<double>(kSize));
    for (size_t i = 0; i < kSize; ++i)
        for (size_t j = 0; j < kSize; ++j)
            mat[i][j] = 1.0;

    double row_sum = sum_row_major(mat);
    double col_sum = sum_col_major(mat);
    double expected = static_cast<double>(kSize * kSize);
    assert(std::abs(row_sum - expected) < 0.01);
    assert(std::abs(col_sum - expected) < 0.01);
}

void benchmark() {
    std::vector<std::vector<double>> mat(kSize, std::vector<double>(kSize));
    for (size_t i = 0; i < kSize; ++i)
        for (size_t j = 0; j < kSize; ++j)
            mat[i][j] = static_cast<double>(i * kSize + j);

    constexpr int kRuns = 20;
    volatile double sink;

    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < kRuns; ++r) sink = sum_row_major(mat);
    auto t1 = std::chrono::steady_clock::now();
    for (int r = 0; r < kRuns; ++r) sink = sum_col_major(mat);
    auto t2 = std::chrono::steady_clock::now();

    (void)sink;

    auto row_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    auto col_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::printf("  Row-major: %6ld us (%d runs)\n", row_us, kRuns);
    std::printf("  Col-major: %6ld us (%d runs)\n", col_us, kRuns);
    std::printf("  Ratio:     %.1fx slower\n",
                static_cast<double>(col_us) / static_cast<double>(row_us));
}

}  // namespace matrix_access

// ========================================================================
// Part B: Array of Structs (AoS) vs Struct of Arrays (SoA)
// ========================================================================

namespace aos_vs_soa {

constexpr size_t kCount = 100000;

// AoS: all fields of one element are contiguous
struct ParticleAoS {
    float x, y, z;
    float vx, vy, vz;
    float mass;
    float charge;
};

// SoA: each field is a separate contiguous array
struct ParticlesSoA {
    std::vector<float> x, y, z;
    std::vector<float> vx, vy, vz;
    std::vector<float> mass;
    std::vector<float> charge;

    void resize(size_t n) {
        x.resize(n); y.resize(n); z.resize(n);
        vx.resize(n); vy.resize(n); vz.resize(n);
        mass.resize(n); charge.resize(n);
    }
};

// Update only positions — AoS loads mass + charge into cache too (waste)
double update_positions_aos(std::vector<ParticleAoS>& particles, float dt) {
    double total_x = 0.0;
    for (auto& p : particles) {
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.z += p.vz * dt;
        total_x += static_cast<double>(p.x);
    }
    return total_x;
}

// Update only positions — SoA only touches x,y,z,vx,vy,vz arrays
double update_positions_soa(ParticlesSoA& ps, float dt) {
    double total_x = 0.0;
    for (size_t i = 0; i < ps.x.size(); ++i) {
        ps.x[i] += ps.vx[i] * dt;
        ps.y[i] += ps.vy[i] * dt;
        ps.z[i] += ps.vz[i] * dt;
        total_x += static_cast<double>(ps.x[i]);
    }
    return total_x;
}

void test() {
    // AoS test
    std::vector<ParticleAoS> aos(100);
    for (size_t i = 0; i < 100; ++i) {
        aos[i].x = static_cast<float>(i);
        aos[i].vx = 1.0f;
        aos[i].y = 0.0f; aos[i].vy = 0.0f;
        aos[i].z = 0.0f; aos[i].vz = 0.0f;
        aos[i].mass = 1.0f; aos[i].charge = 0.0f;
    }
    update_positions_aos(aos, 0.1f);
    assert(std::abs(aos[0].x - 0.1f) < 0.001f);  // 0 + 1*0.1
    assert(std::abs(aos[50].x - 50.1f) < 0.001f); // 50 + 1*0.1

    // SoA test
    ParticlesSoA soa;
    soa.resize(100);
    for (size_t i = 0; i < 100; ++i) {
        soa.x[i] = static_cast<float>(i);
        soa.vx[i] = 1.0f;
        soa.y[i] = 0.0f; soa.vy[i] = 0.0f;
        soa.z[i] = 0.0f; soa.vz[i] = 0.0f;
        soa.mass[i] = 1.0f; soa.charge[i] = 0.0f;
    }
    update_positions_soa(soa, 0.1f);
    assert(std::abs(soa.x[0] - 0.1f) < 0.001f);
    assert(std::abs(soa.x[50] - 50.1f) < 0.001f);
}

void benchmark() {
    // AoS setup
    std::vector<ParticleAoS> aos(kCount);
    for (size_t i = 0; i < kCount; ++i) {
        float fi = static_cast<float>(i);
        aos[i] = {fi, fi, fi, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f};
    }

    // SoA setup
    ParticlesSoA soa;
    soa.resize(kCount);
    for (size_t i = 0; i < kCount; ++i) {
        float fi = static_cast<float>(i);
        soa.x[i] = fi; soa.y[i] = fi; soa.z[i] = fi;
        soa.vx[i] = 1.0f; soa.vy[i] = 1.0f; soa.vz[i] = 1.0f;
        soa.mass[i] = 1.0f; soa.charge[i] = 0.0f;
    }

    constexpr int kRuns = 200;
    volatile double sink;

    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < kRuns; ++r) sink = update_positions_aos(aos, 0.001f);
    auto t1 = std::chrono::steady_clock::now();
    for (int r = 0; r < kRuns; ++r) sink = update_positions_soa(soa, 0.001f);
    auto t2 = std::chrono::steady_clock::now();

    (void)sink;

    auto aos_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    auto soa_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::printf("  AoS: %6ld us (%d runs)\n", aos_us, kRuns);
    std::printf("  SoA: %6ld us (%d runs)\n", soa_us, kRuns);
    std::printf("  Ratio: AoS is %.1fx of SoA\n",
                static_cast<double>(aos_us) / static_cast<double>(soa_us));
}

}  // namespace aos_vs_soa

// ========================================================================
// Part C: Sequential vs Strided Access
// ========================================================================

namespace stride_access {

constexpr size_t kElements = 1 << 20;  // 1M elements

double sum_sequential(const std::vector<int>& data) {
    double sum = 0.0;
    for (size_t i = 0; i < data.size(); ++i)
        sum += static_cast<double>(data[i]);
    return sum;
}

double sum_stride_16(const std::vector<int>& data) {
    double sum = 0.0;
    // Stride 16: access every 16th element (64 bytes apart = cache line size)
    for (size_t i = 0; i < data.size(); i += 16)
        sum += static_cast<double>(data[i]);
    return sum;
}

void test() {
    std::vector<int> data(256);
    std::iota(data.begin(), data.end(), 0);

    double seq_sum = sum_sequential(data);
    assert(std::abs(seq_sum - 32640.0) < 0.01);  // sum(0..255) = 255*256/2

    double stride_sum = sum_stride_16(data);
    // Elements 0, 16, 32, ..., 240 = 16 elements
    // Sum = 16 * (0 + 240) / 2 = 1920
    assert(std::abs(stride_sum - 1920.0) < 0.01);
}

void benchmark() {
    std::vector<int> data(kElements);
    std::iota(data.begin(), data.end(), 0);

    constexpr int kRuns = 50;
    volatile double sink;

    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < kRuns; ++r) sink = sum_sequential(data);
    auto t1 = std::chrono::steady_clock::now();
    for (int r = 0; r < kRuns; ++r) sink = sum_stride_16(data);
    auto t2 = std::chrono::steady_clock::now();

    (void)sink;

    auto seq_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    auto stride_us = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::printf("  Sequential:  %6ld us (%d runs, %zu elements)\n",
                seq_us, kRuns, kElements);
    std::printf("  Stride-16:   %6ld us (%d runs, %zu elements/16)\n",
                stride_us, kRuns, kElements / 16);
    // Note: stride accesses 16x fewer elements, but cache miss rate is much higher
    // per-element cost = time / (runs * elements_accessed)
    auto seq_ns_per = (seq_us * 1000) / (kRuns * static_cast<long>(kElements));
    auto stride_ns_per = (stride_us * 1000) / (kRuns * static_cast<long>(kElements / 16));
    std::printf("  Per-element: seq=%ld ns, stride=%ld ns\n", seq_ns_per, stride_ns_per);
}

}  // namespace stride_access

// ========================================================================
// Main
// ========================================================================

int main(int argc, char** argv) {
    // Run correctness tests
    matrix_access::test();
    aos_vs_soa::test();
    stride_access::test();

    std::printf("ex03_cache_profiling: ALL TESTS PASSED\n");

    // Run benchmarks if --bench flag passed
    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--bench") == 0)
            bench = true;
    }

    if (bench) {
        std::printf("\n=== BENCHMARKS (use with 'perf stat' for cache metrics) ===\n");
        std::printf("\n[Matrix Row vs Column]\n");
        matrix_access::benchmark();
        std::printf("\n[AoS vs SoA Particle Update]\n");
        aos_vs_soa::benchmark();
        std::printf("\n[Sequential vs Strided Access]\n");
        stride_access::benchmark();
    }

    return 0;
}
