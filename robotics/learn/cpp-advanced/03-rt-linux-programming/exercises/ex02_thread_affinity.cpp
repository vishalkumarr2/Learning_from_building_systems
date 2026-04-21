// Week 3 — Exercise 02: Thread Affinity and RT Scheduling
//
// Set CPU affinity and RT scheduling policy. Measure the effect on
// timing determinism. Compare isolated vs non-isolated cores.
//
// Build:  cmake --build build
// Run:    ./ex02_thread_affinity            (normal user — SCHED_OTHER)
//         sudo ./ex02_thread_affinity       (root — SCHED_FIFO)
//   or:   sudo setcap cap_sys_nice+ep ./ex02_thread_affinity && ./ex02_thread_affinity

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sched.h>
#include <unistd.h>

using Clock = std::chrono::steady_clock;
using ns    = std::chrono::nanoseconds;

// ─── Stats ───────────────────────────────────────────────────────────
struct TimingStats {
    double min_us, max_us, mean_us, p99_us, p999_us;
};

TimingStats compute(std::vector<int64_t>& v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    return {
        .min_us  = v.front() / 1000.0,
        .max_us  = v.back()  / 1000.0,
        .mean_us = (sum / n) / 1000.0,
        .p99_us  = v[n * 99 / 100] / 1000.0,
        .p999_us = v[n * 999 / 1000] / 1000.0,
    };
}

void print_timing(const char* label, TimingStats s) {
    std::printf("  %-35s  min=%6.2f  mean=%6.2f  p99=%7.2f  p999=%8.2f  max=%8.2f μs\n",
                label, s.min_us, s.mean_us, s.p99_us, s.p999_us, s.max_us);
}

// ─── Tight loop workload ─────────────────────────────────────────────
// Simulates ~10μs of computation (depends on CPU speed).
static volatile double sink;

void do_computation() {
    double x = 1.0;
    for (int i = 0; i < 200; ++i) {
        x = std::sin(x + 0.001);
    }
    sink = x;
}

// ─── Measure iteration timing on current core ────────────────────────
std::vector<int64_t> measure_iterations(int iterations) {
    std::vector<int64_t> latencies;
    latencies.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto t0 = Clock::now();
        do_computation();
        auto t1 = Clock::now();
        latencies.push_back(
            std::chrono::duration_cast<ns>(t1 - t0).count());
    }
    return latencies;
}

// ─── Set CPU affinity ────────────────────────────────────────────────
bool set_affinity(int core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    int rc = sched_setaffinity(0, sizeof(cpuset), &cpuset);
    if (rc != 0) {
        std::fprintf(stderr, "  sched_setaffinity(core %d) failed: %s\n",
                     core, std::strerror(errno));
        return false;
    }
    return true;
}

// ─── Get current core ────────────────────────────────────────────────
int get_current_core() {
    return sched_getcpu();
}

// ─── Set SCHED_FIFO ──────────────────────────────────────────────────
bool set_sched_fifo(int priority) {
    struct sched_param param{};
    param.sched_priority = priority;
    int rc = sched_setscheduler(0, SCHED_FIFO, &param);
    if (rc != 0) {
        std::fprintf(stderr, "  sched_setscheduler(SCHED_FIFO, %d) failed: %s\n",
                     priority, std::strerror(errno));
        std::fprintf(stderr, "  (need root or CAP_SYS_NICE)\n");
        return false;
    }
    return true;
}

// ─── Set SCHED_RR ────────────────────────────────────────────────────
bool set_sched_rr(int priority) {
    struct sched_param param{};
    param.sched_priority = priority;
    int rc = sched_setscheduler(0, SCHED_RR, &param);
    if (rc != 0) {
        std::fprintf(stderr, "  sched_setscheduler(SCHED_RR, %d) failed: %s\n",
                     priority, std::strerror(errno));
        return false;
    }
    return true;
}

// ─── Reset to SCHED_OTHER ────────────────────────────────────────────
void reset_sched() {
    struct sched_param param{};
    param.sched_priority = 0;
    sched_setscheduler(0, SCHED_OTHER, &param);
}

// ─── Test: run on each available core ────────────────────────────────
void test_per_core() {
    int num_cpus = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
    std::printf("\nSystem has %d online CPUs\n\n", num_cpus);

    constexpr int ITERS = 50'000;

    for (int core = 0; core < std::min(num_cpus, 8); ++core) {
        if (!set_affinity(core)) continue;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        auto lats = measure_iterations(ITERS);
        auto s = compute(lats);
        char label[64];
        std::snprintf(label, sizeof(label), "Core %d (SCHED_OTHER)", core);
        print_timing(label, s);
    }
}

// ─── Test: cache migration penalty ───────────────────────────────────
void test_cache_migration() {
    int num_cpus = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
    if (num_cpus < 2) {
        std::printf("\n  Skipping cache migration test (need ≥2 cores)\n");
        return;
    }

    std::printf("\n--- Cache Migration Penalty ---\n");
    constexpr int ITERS = 10'000;

    // Warm up on core 0
    set_affinity(0);
    auto warm = measure_iterations(ITERS);
    auto s_warm = compute(warm);
    print_timing("After warmup on core 0", s_warm);

    // Migrate to core 1 — first iterations will be cache-cold
    set_affinity(1);
    auto cold = measure_iterations(ITERS);
    auto s_cold = compute(cold);
    print_timing("After migration to core 1", s_cold);

    std::printf("  (First few iterations on new core have cache-miss penalty)\n");

    // Show first 5 iterations explicitly
    std::printf("  First 5 latencies after migration:");
    for (int i = 0; i < 5 && i < static_cast<int>(cold.size()); ++i) {
        std::printf(" %.1fμs", cold[i] / 1000.0);
    }
    std::printf("\n");
}

// ─── Test: SCHED_FIFO vs SCHED_OTHER ─────────────────────────────────
void test_sched_policies() {
    std::printf("\n--- Scheduling Policy Comparison ---\n");
    constexpr int ITERS = 50'000;

    // SCHED_OTHER (default)
    reset_sched();
    set_affinity(0);
    auto other = measure_iterations(ITERS);
    auto s_other = compute(other);
    print_timing("SCHED_OTHER (normal)", s_other);

    // SCHED_FIFO
    if (set_sched_fifo(80)) {
        auto fifo = measure_iterations(ITERS);
        auto s_fifo = compute(fifo);
        print_timing("SCHED_FIFO priority=80", s_fifo);
        reset_sched();
    }

    // SCHED_RR
    if (set_sched_rr(80)) {
        auto rr = measure_iterations(ITERS);
        auto s_rr = compute(rr);
        print_timing("SCHED_RR priority=80", s_rr);
        reset_sched();
    }
}

// ─── Main ────────────────────────────────────────────────────────────
int main() {
    std::printf("╔════════════════════════════════════════════════════╗\n");
    std::printf("║  Exercise 02: Thread Affinity & RT Scheduling     ║\n");
    std::printf("╚════════════════════════════════════════════════════╝\n");

    std::printf("Current core: %d\n", get_current_core());
    std::printf("Current PID:  %d\n", getpid());

    test_per_core();
    test_cache_migration();
    test_sched_policies();

    std::printf("\n─── Key Takeaways ───\n");
    std::printf("1. Core 0 typically has worse jitter (kernel housekeeping)\n");
    std::printf("2. Migrating between cores causes a cache-cold spike\n");
    std::printf("3. SCHED_FIFO dramatically reduces p99/max latency\n");
    std::printf("4. Pin RT threads to isolated cores for best determinism\n");

    return 0;
}
