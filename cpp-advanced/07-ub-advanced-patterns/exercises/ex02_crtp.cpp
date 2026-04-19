// Week 7 — Exercise 2: CRTP — Static Polymorphism
// =================================================
// Compile: g++ -std=c++20 -O2 -Wall -Wextra ex02_crtp.cpp -o ex02 -pthread
//
// Topics:
// 1. SensorBase<Derived> — CRTP dispatch
// 2. Three concrete sensors: IMU, Lidar, Encoder
// 3. process_sensor() template function
// 4. Benchmark: CRTP vs virtual (10M iterations)
// 5. CRTP counter mixin

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// ============================================================
// Part 1: CRTP Sensor Base
// ============================================================

// The CRTP base: Derived passes itself as template arg
template <typename Derived>
struct SensorBase {
    // Static dispatch — resolved at compile time, inlined by optimizer
    double read() {
        return static_cast<Derived*>(this)->read_impl();
    }

    bool is_valid() {
        return static_cast<Derived*>(this)->is_valid_impl();
    }

    // Default implementation for name
    std::string name() const {
        return static_cast<const Derived*>(this)->name_impl();
    }
};

// Concrete sensor: IMU (accelerometer + gyroscope)
struct IMU : SensorBase<IMU> {
    double accel_x_ = 0.0;
    double accel_y_ = 0.0;
    double accel_z_ = 9.81;

    double read_impl() {
        // Return magnitude of acceleration
        return accel_z_; // Simplified
    }

    bool is_valid_impl() {
        return accel_z_ > 0.0 && accel_z_ < 20.0;
    }

    std::string name_impl() const { return "IMU"; }
};

// Concrete sensor: Lidar (2D range scanner)
struct Lidar : SensorBase<Lidar> {
    double min_range_ = 0.05;
    double max_range_ = 30.0;
    double current_range_ = 5.2;

    double read_impl() {
        return current_range_;
    }

    bool is_valid_impl() {
        return current_range_ >= min_range_ && current_range_ <= max_range_;
    }

    std::string name_impl() const { return "Lidar"; }
};

// Concrete sensor: Wheel Encoder
struct Encoder : SensorBase<Encoder> {
    int64_t ticks_ = 0;
    double ticks_per_meter_ = 1000.0;

    double read_impl() {
        return static_cast<double>(ticks_) / ticks_per_meter_;
    }

    bool is_valid_impl() {
        return ticks_per_meter_ > 0.0;
    }

    std::string name_impl() const { return "Encoder"; }
};

// Template function that works with any CRTP sensor
// This is a compile-time-resolved function — no virtual dispatch
template <typename Derived>
void process_sensor(SensorBase<Derived>& sensor) {
    if (sensor.is_valid()) {
        double val = sensor.read();
        std::cout << "  " << sensor.name() << ": value=" << val << " (valid)\n";
    } else {
        std::cout << "  " << sensor.name() << ": INVALID\n";
    }
}

// ============================================================
// Part 2: Virtual Interface for Comparison
// ============================================================

struct VirtualSensorBase {
    virtual ~VirtualSensorBase() = default;
    virtual double read() = 0;
    virtual bool is_valid() = 0;
};

struct VirtualIMU : VirtualSensorBase {
    double accel_z_ = 9.81;
    double read() override { return accel_z_; }
    bool is_valid() override { return accel_z_ > 0.0 && accel_z_ < 20.0; }
};

struct VirtualLidar : VirtualSensorBase {
    double range_ = 5.2;
    double read() override { return range_; }
    bool is_valid() override { return range_ >= 0.05 && range_ <= 30.0; }
};

// ============================================================
// Part 3: CRTP Counter Mixin
// ============================================================

// Counts live instances of each derived type — a common CRTP mixin
template <typename Derived>
struct InstanceCounted {
    static inline int count_ = 0;

    InstanceCounted() { ++count_; }
    ~InstanceCounted() { --count_; }

    // Non-copyable by default (avoid double-counting)
    InstanceCounted(const InstanceCounted&) { ++count_; }
    InstanceCounted& operator=(const InstanceCounted&) = default;
    InstanceCounted(InstanceCounted&&) noexcept { ++count_; }
    InstanceCounted& operator=(InstanceCounted&&) noexcept = default;

    static int instance_count() { return count_; }
};

// Widgets that count themselves
struct WidgetA : InstanceCounted<WidgetA> {
    int data_ = 0;
};

struct WidgetB : InstanceCounted<WidgetB> {
    std::string label_ = "default";
};

// ============================================================
// Part 4: Benchmark
// ============================================================

static constexpr int BENCH_ITERATIONS = 10'000'000;

template <typename T>
void do_not_optimize(T const& val) {
    asm volatile("" : : "r,m"(val) : "memory");
}

void benchmark_crtp() {
    IMU imu;
    auto start = std::chrono::high_resolution_clock::now();

    double sum = 0.0;
    for (int i = 0; i < BENCH_ITERATIONS; ++i) {
        sum += imu.read(); // Statically dispatched, likely inlined
        do_not_optimize(sum);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "  CRTP:    " << ns / 1'000'000 << " ms ("
              << BENCH_ITERATIONS << " calls, sum=" << sum << ")\n";
    // On Godbolt with -O2, read() inlines to a single `movsd` — no call at all.
    // The entire loop may unroll + vectorize.
}

void benchmark_virtual() {
    auto imu = std::make_unique<VirtualIMU>();
    VirtualSensorBase* base = imu.get();
    auto start = std::chrono::high_resolution_clock::now();

    double sum = 0.0;
    for (int i = 0; i < BENCH_ITERATIONS; ++i) {
        sum += base->read(); // vtable lookup + indirect call
        do_not_optimize(sum);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "  Virtual: " << ns / 1'000'000 << " ms ("
              << BENCH_ITERATIONS << " calls, sum=" << sum << ")\n";
    // On Godbolt: call through vtable, `call qword ptr [rax]`.
    // GCC can sometimes de-virtualize if the concrete type is provable.
}

/*
 * Assembly comparison (what Godbolt shows at -O2):
 *
 * CRTP read():
 *   The call to imu.read() becomes:
 *     movsd xmm0, QWORD PTR [rdi+16]   ; direct load of accel_z_
 *     ret
 *   And the loop body is just addsd in a tight loop (or vectorized).
 *
 * Virtual read():
 *   mov rax, QWORD PTR [rdi]            ; load vtable pointer
 *   call QWORD PTR [rax+16]             ; indirect call through vtable
 *   This prevents inlining and vectorization.
 *
 * Typical result: CRTP is 3-10x faster for trivial methods.
 * For complex methods, the difference shrinks because call overhead
 * becomes negligible relative to method body.
 */

// ============================================================

int main() {
    std::cout << "=== CRTP: STATIC POLYMORPHISM ===\n\n";

    // Part 1: Process sensors
    std::cout << "Part 1: CRTP Sensor Dispatch\n";
    IMU imu;
    Lidar lidar;
    Encoder encoder;
    encoder.ticks_ = 5000; // 5 meters

    process_sensor(imu);
    process_sensor(lidar);
    process_sensor(encoder);

    // Part 2: Benchmark
    std::cout << "\nPart 2: CRTP vs Virtual Benchmark (" << BENCH_ITERATIONS << " iterations)\n";
    benchmark_crtp();
    benchmark_virtual();

    // Part 3: Counter mixin
    std::cout << "\nPart 3: CRTP Counter Mixin\n";
    {
        WidgetA a1, a2, a3;
        WidgetB b1, b2;
        std::cout << "  WidgetA instances: " << WidgetA::instance_count() << "\n";
        std::cout << "  WidgetB instances: " << WidgetB::instance_count() << "\n";

        {
            WidgetA a4;
            std::cout << "  After creating a4: WidgetA=" << WidgetA::instance_count() << "\n";
        }
        std::cout << "  After a4 destroyed: WidgetA=" << WidgetA::instance_count() << "\n";
    }
    std::cout << "  After all destroyed: WidgetA=" << WidgetA::instance_count()
              << ", WidgetB=" << WidgetB::instance_count() << "\n";

    std::cout << "\n=== DONE ===\n";
    return 0;
}
