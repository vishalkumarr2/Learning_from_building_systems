// =============================================================================
// Exercise 04: Real-Time Arena Pattern
// =============================================================================
// Topics:
//   - Simulating a 1kHz control loop with pre-allocated arena
//   - Zero-allocation hot path using monotonic_buffer_resource
//   - Per-cycle allocate → work → release pattern
//   - Benchmark: arena vs heap allocation per cycle
//   - This is THE pattern for game engines and RT robotics
//
// Compile: g++ -std=c++2a -O2 -Wall -Wextra -pthread -o ex04 ex04_realtime_arena.cpp
// =============================================================================

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory_resource>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

// =============================================================================
// Simulated sensor/control data structures
// =============================================================================
struct SensorReading {
    double x, y, z;
    double timestamp;
    uint32_t sensor_id;
    uint8_t  status;
};

struct ControlOutput {
    double linear_vel;
    double angular_vel;
    double timestamp;
};

struct Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start_ = Clock::now();
    double elapsed_us() const {
        return std::chrono::duration<double, std::micro>(Clock::now() - start_).count();
    }
    double elapsed_ns() const {
        return std::chrono::duration<double, std::nano>(Clock::now() - start_).count();
    }
};

// =============================================================================
// The "work" simulating one control cycle
// Creates temporary containers, processes data, produces output.
// In a real system: read sensors → filter → compute control → publish.
// =============================================================================

// --- WRONG WAY: Heap allocation every cycle ---
ControlOutput do_work_heap(int cycle) {
    // These containers heap-allocate every single cycle
    std::vector<SensorReading> readings;
    readings.reserve(32);
    for (int i = 0; i < 32; ++i) {
        readings.push_back({
            std::sin(cycle * 0.01 + i), std::cos(cycle * 0.01 + i),
            static_cast<double>(i) * 0.1, cycle * 0.001,
            static_cast<uint32_t>(i), 0
        });
    }

    // Filter valid readings
    std::vector<SensorReading> valid;
    valid.reserve(32);
    std::copy_if(readings.begin(), readings.end(), std::back_inserter(valid),
                 [](const SensorReading& r) { return r.status == 0; });

    // Compute average position
    double avg_x = 0, avg_y = 0;
    for (const auto& r : valid) {
        avg_x += r.x;
        avg_y += r.y;
    }
    if (!valid.empty()) {
        avg_x /= static_cast<double>(valid.size());
        avg_y /= static_cast<double>(valid.size());
    }

    // Compute control output
    return {avg_x * 0.5, avg_y * 0.3, cycle * 0.001};
}

// --- RIGHT WAY: Arena allocation, zero heap usage ---
ControlOutput do_work_arena(int cycle, std::pmr::memory_resource* arena) {
    // These containers allocate from the arena — no heap!
    std::pmr::vector<SensorReading> readings(arena);
    readings.reserve(32);
    for (int i = 0; i < 32; ++i) {
        readings.push_back({
            std::sin(cycle * 0.01 + i), std::cos(cycle * 0.01 + i),
            static_cast<double>(i) * 0.1, cycle * 0.001,
            static_cast<uint32_t>(i), 0
        });
    }

    std::pmr::vector<SensorReading> valid(arena);
    valid.reserve(32);
    std::copy_if(readings.begin(), readings.end(), std::back_inserter(valid),
                 [](const SensorReading& r) { return r.status == 0; });

    double avg_x = 0, avg_y = 0;
    for (const auto& r : valid) {
        avg_x += r.x;
        avg_y += r.y;
    }
    if (!valid.empty()) {
        avg_x /= static_cast<double>(valid.size());
        avg_y /= static_cast<double>(valid.size());
    }

    return {avg_x * 0.5, avg_y * 0.3, cycle * 0.001};
}

// =============================================================================
// Part 1: Basic arena pattern — pre-allocate, use, release
// =============================================================================
void part1_basic_arena() {
    std::cout << "=== Part 1: Basic Arena Pattern ===\n\n";

    // STARTUP: Pre-allocate arena buffer
    constexpr size_t ARENA_SIZE = 64 * 1024;  // 64 KB — enough for our control loop
    alignas(std::max_align_t) static char arena_buf[ARENA_SIZE];

    // Use null_memory_resource as upstream — crash if we overflow!
    // This is intentional: in RT, silent heap fallback = bug.
    std::pmr::monotonic_buffer_resource arena(
        arena_buf, ARENA_SIZE, std::pmr::null_memory_resource());

    std::cout << "  Arena: " << ARENA_SIZE << " bytes pre-allocated on stack\n";
    std::cout << "  Upstream: null_memory_resource (crash on overflow)\n\n";

    // HOT LOOP: 10 simulated cycles
    for (int cycle = 0; cycle < 10; ++cycle) {
        // All allocations go to arena_buf
        ControlOutput out = do_work_arena(cycle, &arena);

        if (cycle < 3 || cycle == 9) {
            std::printf("  Cycle %d: vel=(%.3f, %.3f)\n",
                        cycle, out.linear_vel, out.angular_vel);
        } else if (cycle == 3) {
            std::cout << "  ...\n";
        }

        // Reset arena for next cycle — O(1), pointer back to start
        arena.release();
    }

    std::cout << "  All 10 cycles completed with ZERO heap allocations.\n\n";
}

// =============================================================================
// Part 2: Benchmark — arena vs heap per cycle
// =============================================================================
void part2_benchmark() {
    std::cout << "=== Part 2: Benchmark — 1000 Control Cycles ===\n\n";

    constexpr int CYCLES = 1000;
    constexpr int TRIALS = 5;

    // --- Heap path ---
    double heap_best = 1e9;
    for (int trial = 0; trial < TRIALS; ++trial) {
        Timer t;
        volatile double sink = 0;
        for (int c = 0; c < CYCLES; ++c) {
            auto out = do_work_heap(c);
            sink += out.linear_vel;
        }
        double us = t.elapsed_us();
        if (us < heap_best) heap_best = us;
        (void)sink;
    }

    // --- Arena path ---
    constexpr size_t ARENA_SIZE = 64 * 1024;
    alignas(std::max_align_t) static char arena_buf[ARENA_SIZE];

    double arena_best = 1e9;
    for (int trial = 0; trial < TRIALS; ++trial) {
        std::pmr::monotonic_buffer_resource arena(
            arena_buf, ARENA_SIZE, std::pmr::null_memory_resource());

        Timer t;
        volatile double sink = 0;
        for (int c = 0; c < CYCLES; ++c) {
            auto out = do_work_arena(c, &arena);
            sink += out.linear_vel;
            arena.release();
        }
        double us = t.elapsed_us();
        if (us < arena_best) arena_best = us;
        (void)sink;
    }

    double per_cycle_heap_ns  = (heap_best / CYCLES) * 1000.0;
    double per_cycle_arena_ns = (arena_best / CYCLES) * 1000.0;

    std::printf("  %-25s %8.0f μs total  (%6.0f ns/cycle)\n",
                "Heap (new/delete):", heap_best, per_cycle_heap_ns);
    std::printf("  %-25s %8.0f μs total  (%6.0f ns/cycle)\n",
                "Arena (monotonic):", arena_best, per_cycle_arena_ns);
    std::printf("  Speedup: %.1fx\n\n", heap_best / arena_best);

    // Context: at 1kHz, each cycle has 1,000,000 ns budget.
    // Heap allocation might use 1-10% of that budget. Arena uses ~0.01%.
    std::printf("  At 1kHz (1ms budget):\n");
    std::printf("    Heap alloc overhead: %.1f%% of cycle budget\n",
                per_cycle_heap_ns / 10000.0);
    std::printf("    Arena alloc overhead: %.2f%% of cycle budget\n\n",
                per_cycle_arena_ns / 10000.0);
}

// =============================================================================
// Part 3: Latency jitter analysis
// =============================================================================
void part3_jitter() {
    std::cout << "=== Part 3: Latency Jitter (Worst-Case Matters in RT!) ===\n\n";

    constexpr int CYCLES = 5000;
    constexpr size_t ARENA_SIZE = 64 * 1024;
    alignas(std::max_align_t) static char arena_buf[ARENA_SIZE];

    std::vector<double> heap_times, arena_times;
    heap_times.reserve(CYCLES);
    arena_times.reserve(CYCLES);

    // Measure each cycle individually
    for (int c = 0; c < CYCLES; ++c) {
        {
            Timer t;
            auto out = do_work_heap(c);
            (void)out;
            heap_times.push_back(t.elapsed_ns());
        }
        {
            std::pmr::monotonic_buffer_resource arena(
                arena_buf, ARENA_SIZE, std::pmr::null_memory_resource());
            Timer t;
            auto out = do_work_arena(c, &arena);
            (void)out;
            arena_times.push_back(t.elapsed_ns());
            // arena goes out of scope — releases automatically
        }
    }

    // Sort for percentile analysis
    std::sort(heap_times.begin(), heap_times.end());
    std::sort(arena_times.begin(), arena_times.end());

    auto percentile = [](const std::vector<double>& v, double p) {
        size_t idx = static_cast<size_t>(v.size() * p);
        if (idx >= v.size()) idx = v.size() - 1;
        return v[idx];
    };

    std::printf("  %-12s %8s %8s %8s %8s %8s\n",
                "", "median", "p90", "p99", "p99.9", "max");
    std::printf("  %-12s %7.0fns %7.0fns %7.0fns %7.0fns %7.0fns\n", "Heap:",
                percentile(heap_times, 0.5), percentile(heap_times, 0.9),
                percentile(heap_times, 0.99), percentile(heap_times, 0.999),
                heap_times.back());
    std::printf("  %-12s %7.0fns %7.0fns %7.0fns %7.0fns %7.0fns\n", "Arena:",
                percentile(arena_times, 0.5), percentile(arena_times, 0.9),
                percentile(arena_times, 0.99), percentile(arena_times, 0.999),
                arena_times.back());

    std::printf("\n  Jitter (max/median):\n");
    std::printf("    Heap:  %.1fx\n",
                heap_times.back() / percentile(heap_times, 0.5));
    std::printf("    Arena: %.1fx\n",
                arena_times.back() / percentile(arena_times, 0.5));
    std::cout << "  Lower jitter = more deterministic = better for RT.\n\n";
}

// =============================================================================
// Part 4: Zero-allocation proof with null upstream
// =============================================================================
void part4_zero_alloc_proof() {
    std::cout << "=== Part 4: Zero-Allocation Proof ===\n\n";

    constexpr size_t ARENA_SIZE = 64 * 1024;
    alignas(std::max_align_t) static char arena_buf[ARENA_SIZE];

    // null_memory_resource: throws bad_alloc on ANY allocation.
    // If our arena is big enough, the null upstream is never hit.
    // If it IS hit → we get an immediate crash (good!) instead of silent heap use.
    std::pmr::monotonic_buffer_resource arena(
        arena_buf, ARENA_SIZE, std::pmr::null_memory_resource());

    bool all_ok = true;
    for (int c = 0; c < 100; ++c) {
        try {
            auto out = do_work_arena(c, &arena);
            (void)out;
            arena.release();
        } catch (const std::bad_alloc&) {
            std::printf("  OVERFLOW at cycle %d! Arena too small.\n", c);
            all_ok = false;
            break;
        }
    }

    if (all_ok) {
        std::cout << "  100 cycles completed with null upstream — "
                     "ZERO heap allocations guaranteed!\n";
        std::cout << "  This is the strongest guarantee: if it ran, it didn't heap-allocate.\n";
    }
    std::cout << "\n";
}

// =============================================================================
int main() {
    std::cout << "╔══════════════════════════════════════════════════╗\n";
    std::cout << "║  Exercise 04: Real-Time Arena Pattern            ║\n";
    std::cout << "╚══════════════════════════════════════════════════╝\n\n";

    part1_basic_arena();
    part2_benchmark();
    part3_jitter();
    part4_zero_alloc_proof();

    std::cout << "=== Summary ===\n";
    std::cout << "  - Pre-allocate arena at startup (non-RT context)\n";
    std::cout << "  - Hot loop: allocate from arena → work → release()\n";
    std::cout << "  - null_memory_resource upstream = crash on overflow (intentional)\n";
    std::cout << "  - Arena gives deterministic timing with near-zero jitter\n";
    std::cout << "  - This is THE pattern for game engines, RT control, sensor processing\n";

    return 0;
}
