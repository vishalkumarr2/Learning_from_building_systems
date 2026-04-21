// =============================================================================
// Exercise 5: C++20 Concepts (GCC 9 Concepts TS compatible)
// =============================================================================
// Demonstrates:
//   1. Sensor concept for robot hardware abstraction
//   2. DmaCompatible concept for shared memory / DMA buffers
//   3. SFINAE → Concepts rewrite (before/after)
//   4. Concept subsumption with Integral / SignedIntegral
//   5. Exercise: write a Controller concept
//
// NOTE: GCC 9 supports Concepts TS via -fconcepts but lacks the <concepts>
// header and abbreviated function templates (Concept auto param). We use
// requires clauses and template<Concept T> form instead. With GCC 10+,
// see comments for the terse abbreviated syntax.
//
// Build: g++ -std=c++2a -fconcepts -Wall -Wextra -Wpedantic ex05_concepts.cpp
// =============================================================================

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>

// =============================================================================
// Part 1: Sensor Concept
// =============================================================================

// With GCC 10+ <concepts>:
//   { t.read() } -> std::convertible_to<double>;
// With GCC 9, we use plain requires + is_convertible:
template<typename T>
concept Sensor = requires(T t) {
    { t.read() };
    requires std::is_convertible_v<decltype(t.read()), double>;
    { t.is_valid() };
    requires std::is_same_v<decltype(t.is_valid()), bool>;
    requires std::is_convertible_v<decltype(T::sample_rate_hz), unsigned>;
};

// A compliant sensor
struct ImuSensor {
    static constexpr unsigned sample_rate_hz = 200;
    double last_reading_ = 0.0;
    bool valid_ = true;

    double read() { return last_reading_; }
    bool is_valid() const { return valid_; }
};

// Another compliant sensor
struct EncoderSensor {
    static constexpr unsigned sample_rate_hz = 1000;
    int32_t ticks_ = 0;
    bool connected_ = true;

    int32_t read() { return ticks_; }
    bool is_valid() const { return connected_; }
};

// This does NOT satisfy the Sensor concept — missing sample_rate_hz
struct BrokenSensor {
    double read() { return 0.0; }
    bool is_valid() const { return true; }
};

// Verify concepts at compile time
static_assert(Sensor<ImuSensor>, "ImuSensor satisfies Sensor");
static_assert(Sensor<EncoderSensor>, "EncoderSensor satisfies Sensor");
static_assert(!Sensor<BrokenSensor>, "BrokenSensor does NOT satisfy Sensor");
static_assert(!Sensor<int>, "int is not a Sensor");

// Generic function constrained by Sensor concept
// GCC 10+ terse: void log_sensor(Sensor auto& sensor, ...)
template<Sensor S>
void log_sensor(S& sensor, const std::string& name) {
    if (sensor.is_valid()) {
        std::cout << "  [" << name << " @ " << S::sample_rate_hz << "Hz] "
                  << "reading = " << sensor.read() << "\n";
    } else {
        std::cout << "  [" << name << "] INVALID\n";
    }
}

// =============================================================================
// Part 2: DmaCompatible Concept
// =============================================================================

template<typename T>
concept DmaCompatible = std::is_trivially_copyable_v<T>
                     && (alignof(T) >= 4)
                     && (sizeof(T) % 4 == 0);

struct alignas(4) ImuData {
    float accel[3];
    float gyro[3];
    uint64_t timestamp_ns;
};

struct alignas(4) OdomData {
    float x, y, theta;
    float vx, vy, omega;
    uint64_t timestamp_ns;
};

static_assert(DmaCompatible<ImuData>, "ImuData is DMA-compatible");
static_assert(DmaCompatible<OdomData>, "OdomData is DMA-compatible");
static_assert(!DmaCompatible<std::string>, "string is NOT DMA-compatible");

template<DmaCompatible T>
void dma_write(const T& data, void* /* dest_addr */) {
    std::cout << "  DMA write: " << sizeof(T) << " bytes, align="
              << alignof(T) << "\n";
    (void)data;
}

// =============================================================================
// Part 3: SFINAE → Concepts Rewrite
// =============================================================================

// BEFORE (C++17 SFINAE horror):
template<typename T,
         typename = std::enable_if_t<
             std::is_arithmetic_v<T> && !std::is_same_v<T, bool>>>
T clamp_sfinae(T val, T lo, T hi) {
    return (val < lo) ? lo : (val > hi) ? hi : val;
}

// AFTER (C++20 Concepts — clean and readable):
template<typename T>
concept Numeric = std::is_arithmetic_v<T> && !std::is_same_v<T, bool>;

// GCC 10+ terse: Numeric auto clamp_concept(Numeric auto val, ...)
// GCC 9: use template with requires
template<Numeric T>
T clamp_concept(T val, T lo, T hi) {
    return (val < lo) ? lo : (val > hi) ? hi : val;
}

// =============================================================================
// Part 4: Concept Subsumption — Which Overload Wins?
// =============================================================================

template<typename T>
concept IsIntegral = std::is_integral_v<T>;

template<typename T>
concept IsSignedIntegral = IsIntegral<T> && std::is_signed_v<T>;

// GCC 9: use requires clause form for overloading
// (abbreviated function template `Concept auto` doesn't participate in
// subsumption on GCC 9; use explicit template declarations)

template<typename T>
std::string categorize(T x) {
    (void)x;
    return "unconstrained (any type)";
}

template<typename T> requires IsIntegral<T>
std::string categorize(T x) {
    (void)x;
    return "Integral";
}

template<typename T> requires IsSignedIntegral<T>
std::string categorize(T x) {
    (void)x;
    return "SignedIntegral (subsumes Integral)";
}

// =============================================================================
// Part 5: Exercise — Write a Controller Concept
// =============================================================================

template<typename T>
concept Controller = requires(T t, double sp) {
    { t.set_setpoint(sp) };
    requires std::is_convertible_v<decltype(t.compute()), double>;
    requires std::is_convertible_v<decltype(t.output()), double>;
    { t.reset() };
};

struct PidController {
    double kp_ = 1.0, ki_ = 0.1, kd_ = 0.01;
    double setpoint_ = 0.0;
    double integral_ = 0.0;
    double prev_error_ = 0.0;
    double output_ = 0.0;

    void set_setpoint(double sp) { setpoint_ = sp; }

    double compute() {
        double error = setpoint_;
        integral_ += error;
        double derivative = error - prev_error_;
        prev_error_ = error;
        output_ = kp_ * error + ki_ * integral_ + kd_ * derivative;
        return output_;
    }

    double output() const { return output_; }
    void reset() { integral_ = 0; prev_error_ = 0; output_ = 0; }
};

struct BangBangController {
    double setpoint_ = 0.0;
    double out_ = 0.0;
    double threshold_ = 0.1;

    void set_setpoint(double sp) { setpoint_ = sp; }
    double compute() {
        out_ = (setpoint_ > threshold_) ? 1.0 : 0.0;
        return out_;
    }
    double output() const { return out_; }
    void reset() { out_ = 0.0; }
};

static_assert(Controller<PidController>, "PID satisfies Controller");
static_assert(Controller<BangBangController>, "BangBang satisfies Controller");
static_assert(!Controller<int>, "int is not a Controller");
static_assert(!Controller<ImuSensor>, "Sensor is not a Controller");

template<Controller C>
void run_control_step(C& ctrl, double setpoint) {
    ctrl.set_setpoint(setpoint);
    double u = ctrl.compute();
    std::cout << "  setpoint=" << setpoint << " → output=" << u << "\n";
}

// =============================================================================
// Main
// =============================================================================

int main() {
    std::cout << "=== C++20 Concepts Exercises ===\n";

    // --- Part 1: Sensor Concept ---
    std::cout << "\n--- Sensor Concept ---\n";
    ImuSensor imu;
    imu.last_reading_ = 9.81;
    EncoderSensor enc;
    enc.ticks_ = 42000;

    log_sensor(imu, "IMU");
    log_sensor(enc, "Encoder");

    // --- Part 2: DmaCompatible ---
    std::cout << "\n--- DmaCompatible Concept ---\n";
    ImuData imu_data{};
    OdomData odom_data{};
    dma_write(imu_data, nullptr);
    dma_write(odom_data, nullptr);

    // --- Part 3: SFINAE vs Concepts ---
    std::cout << "\n--- SFINAE → Concepts ---\n";
    std::cout << "  clamp_sfinae(15, 0, 10)      = " << clamp_sfinae(15, 0, 10) << "\n";
    std::cout << "  clamp_concept(15.5, 0.0, 10.0) = " << clamp_concept(15.5, 0.0, 10.0) << "\n";

    // --- Part 4: Subsumption ---
    std::cout << "\n--- Concept Subsumption ---\n";
    std::cout << "  42       → " << categorize(42) << "\n";
    std::cout << "  42u      → " << categorize(42u) << "\n";
    std::cout << "  3.14     → " << categorize(3.14) << "\n";
    std::cout << "  \"hello\"  → " << categorize("hello") << "\n";

    // --- Part 5: Controller Concept ---
    std::cout << "\n--- Controller Concept ---\n";
    PidController pid;
    BangBangController bang;

    std::cout << "  PID controller:\n";
    run_control_step(pid, 1.0);
    run_control_step(pid, 0.5);

    std::cout << "  BangBang controller:\n";
    run_control_step(bang, 1.0);
    run_control_step(bang, 0.05);

    std::cout << "\nAll concept exercises passed!\n";
    return 0;
}
