// Week 7 — Exercise 3: Policy-Based PID Controller
// ==================================================
// Compile: g++ -std=c++20 -O2 -Wall -Wextra ex03_policy_design.cpp -o ex03
//
// Topics:
// 1. Policy-based design — composition via template parameters
// 2. Three policy axes: Integration, Saturation, Derivative
// 3. Multiple combinations, all zero-overhead
// 4. Benchmark: policy-based vs virtual strategy pattern

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>

template <typename T>
void do_not_optimize(T const& val) {
    asm volatile("" : : "r,m"(val) : "memory");
}

// ============================================================
// Integration Policies
// ============================================================

struct TrapezoidalIntegration {
    static double integrate(double integral, double error, double prev_error, double dt) {
        // Trapezoidal rule: average of current and previous error
        return integral + 0.5 * (error + prev_error) * dt;
    }
};

struct EulerIntegration {
    static double integrate(double integral, double error, double /*prev_error*/, double dt) {
        // Simple forward Euler
        return integral + error * dt;
    }
};

struct SimpsonIntegration {
    // Simpson's 1/3 rule (using current, previous, and a midpoint estimate)
    static double integrate(double integral, double error, double prev_error, double dt) {
        double mid = 0.5 * (error + prev_error);
        return integral + (dt / 6.0) * (prev_error + 4.0 * mid + error);
    }
};

// ============================================================
// Saturation Policies
// ============================================================

struct ClampSaturation {
    static double clamp(double value, double min_val, double max_val) {
        return std::clamp(value, min_val, max_val);
    }
};

struct NoneSaturation {
    static double clamp(double value, double /*min_val*/, double /*max_val*/) {
        return value; // No saturation — let it wind up
    }
};

struct TanhSaturation {
    // Smooth saturation using tanh — no hard clipping
    static double clamp(double value, double /*min_val*/, double max_val) {
        return max_val * std::tanh(value / max_val);
    }
};

// ============================================================
// Derivative Policies
// ============================================================

struct RawDerivative {
    static double compute(double error, double prev_error, double /*prev_deriv*/, double dt) {
        if (dt <= 0.0) return 0.0;
        return (error - prev_error) / dt;
    }
};

struct FilteredDerivative {
    // First-order low-pass filter on derivative (alpha = 0.1)
    static constexpr double kAlpha = 0.1;

    static double compute(double error, double prev_error, double prev_deriv, double dt) {
        if (dt <= 0.0) return prev_deriv;
        double raw = (error - prev_error) / dt;
        return prev_deriv + kAlpha * (raw - prev_deriv);
    }
};

struct BackwardDifferenceDerivative {
    // Derivative on measurement, not error (reduces derivative kick)
    // For simplicity, same formula but conceptually applied to -measurement
    static double compute(double error, double prev_error, double /*prev_deriv*/, double dt) {
        if (dt <= 0.0) return 0.0;
        return (error - prev_error) / dt;
    }
};

// ============================================================
// Policy-Based PID Controller
// ============================================================

template <typename IntegrationPolicy,
          typename SaturationPolicy,
          typename DerivativePolicy>
class PIDController {
public:
    PIDController(double kp, double ki, double kd,
                  double integral_min, double integral_max)
        : kp_(kp), ki_(ki), kd_(kd),
          integral_min_(integral_min), integral_max_(integral_max) {}

    double compute(double setpoint, double measurement, double dt) {
        double error = setpoint - measurement;

        // Integration with policy
        integral_ = IntegrationPolicy::integrate(integral_, error, prev_error_, dt);

        // Anti-windup with saturation policy
        integral_ = SaturationPolicy::clamp(integral_, integral_min_, integral_max_);

        // Derivative with policy
        derivative_ = DerivativePolicy::compute(error, prev_error_, derivative_, dt);

        // PID output
        double output = kp_ * error + ki_ * integral_ + kd_ * derivative_;

        prev_error_ = error;
        return output;
    }

    void reset() {
        integral_ = 0.0;
        derivative_ = 0.0;
        prev_error_ = 0.0;
    }

    double integral() const { return integral_; }
    double derivative() const { return derivative_; }

private:
    double kp_, ki_, kd_;
    double integral_min_, integral_max_;
    double integral_ = 0.0;
    double derivative_ = 0.0;
    double prev_error_ = 0.0;
};

// ============================================================
// Virtual Strategy PID (for benchmark comparison)
// ============================================================

struct IIntegration {
    virtual ~IIntegration() = default;
    virtual double integrate(double integral, double error, double prev_error, double dt) = 0;
};

struct ISaturation {
    virtual ~ISaturation() = default;
    virtual double clamp(double value, double min_val, double max_val) = 0;
};

struct IDerivative {
    virtual ~IDerivative() = default;
    virtual double compute(double error, double prev_error, double prev_deriv, double dt) = 0;
};

struct VirtualEuler : IIntegration {
    double integrate(double integral, double error, double /*prev*/, double dt) override {
        return integral + error * dt;
    }
};

struct VirtualClamp : ISaturation {
    double clamp(double value, double min_val, double max_val) override {
        return std::clamp(value, min_val, max_val);
    }
};

struct VirtualRaw : IDerivative {
    double compute(double error, double prev_error, double /*prev_deriv*/, double dt) override {
        return dt > 0.0 ? (error - prev_error) / dt : 0.0;
    }
};

class VirtualPIDController {
public:
    VirtualPIDController(double kp, double ki, double kd,
                         double imin, double imax,
                         std::unique_ptr<IIntegration> integ,
                         std::unique_ptr<ISaturation> sat,
                         std::unique_ptr<IDerivative> deriv)
        : kp_(kp), ki_(ki), kd_(kd), imin_(imin), imax_(imax),
          integ_(std::move(integ)), sat_(std::move(sat)), deriv_(std::move(deriv)) {}

    double compute(double setpoint, double measurement, double dt) {
        double error = setpoint - measurement;
        integral_ = integ_->integrate(integral_, error, prev_error_, dt);
        integral_ = sat_->clamp(integral_, imin_, imax_);
        derivative_ = deriv_->compute(error, prev_error_, derivative_, dt);
        double output = kp_ * error + ki_ * integral_ + kd_ * derivative_;
        prev_error_ = error;
        return output;
    }

private:
    double kp_, ki_, kd_, imin_, imax_;
    double integral_ = 0.0, derivative_ = 0.0, prev_error_ = 0.0;
    std::unique_ptr<IIntegration> integ_;
    std::unique_ptr<ISaturation> sat_;
    std::unique_ptr<IDerivative> deriv_;
};

// ============================================================
// Type aliases for convenient PID configurations
// ============================================================

// Standard industrial PID: Euler + Clamp + Filtered
using StandardPID = PIDController<EulerIntegration, ClampSaturation, FilteredDerivative>;

// High-precision PID: Trapezoidal + Clamp + Filtered
using PrecisionPID = PIDController<TrapezoidalIntegration, ClampSaturation, FilteredDerivative>;

// Aggressive PID: Euler + None + Raw (no filtering, no anti-windup)
using AggressivePID = PIDController<EulerIntegration, NoneSaturation, RawDerivative>;

// Smooth PID: Simpson + Tanh + Filtered
using SmoothPID = PIDController<SimpsonIntegration, TanhSaturation, FilteredDerivative>;

// ============================================================
// Simulation Helper
// ============================================================

template <typename PID>
void simulate(PID& pid, const std::string& name, double setpoint, int steps) {
    double measurement = 0.0;
    double dt = 0.01; // 100 Hz
    double plant_gain = 0.1; // Simple first-order plant

    std::cout << "  " << name << " → setpoint=" << setpoint << "\n";
    for (int i = 0; i < steps; ++i) {
        double output = pid.compute(setpoint, measurement, dt);
        measurement += plant_gain * output * dt; // Simple plant model

        if (i % (steps / 5) == 0) {
            std::cout << "    step " << i << ": meas=" << measurement
                      << ", out=" << output << "\n";
        }
    }
    std::cout << "    final: measurement=" << measurement << "\n";
    pid.reset();
}

// ============================================================
// Benchmark
// ============================================================

static constexpr int BENCH_ITERS = 10'000'000;

void benchmark() {
    // Policy-based
    StandardPID policy_pid(2.0, 0.5, 0.1, -100.0, 100.0);
    double measurement = 0.0;
    double setpoint = 1.0;
    double dt = 0.001;

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        double out = policy_pid.compute(setpoint, measurement, dt);
        measurement += 0.1 * out * dt;
        do_not_optimize(measurement);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto ns1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // Virtual-based
    VirtualPIDController virt_pid(
        2.0, 0.5, 0.1, -100.0, 100.0,
        std::make_unique<VirtualEuler>(),
        std::make_unique<VirtualClamp>(),
        std::make_unique<VirtualRaw>()
    );
    measurement = 0.0;

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < BENCH_ITERS; ++i) {
        double out = virt_pid.compute(setpoint, measurement, dt);
        measurement += 0.1 * out * dt;
        do_not_optimize(measurement);
    }
    end = std::chrono::high_resolution_clock::now();
    auto ns2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    std::cout << "  Policy-based: " << ns1 / 1'000'000 << " ms\n";
    std::cout << "  Virtual-based: " << ns2 / 1'000'000 << " ms\n";
    if (ns2 > ns1 && ns1 > 0) {
        std::cout << "  Speedup: " << static_cast<double>(ns2) / static_cast<double>(ns1)
                  << "x\n";
    }
}

// ============================================================

int main() {
    std::cout << "=== POLICY-BASED PID CONTROLLER ===\n\n";

    // Part 1: Different PID configurations
    std::cout << "Part 1: PID Configurations\n";

    StandardPID standard(2.0, 0.5, 0.1, -10.0, 10.0);
    simulate(standard, "StandardPID (Euler+Clamp+Filtered)", 1.0, 500);

    PrecisionPID precision(2.0, 0.5, 0.1, -10.0, 10.0);
    simulate(precision, "PrecisionPID (Trapezoidal+Clamp+Filtered)", 1.0, 500);

    AggressivePID aggressive(2.0, 0.5, 0.1, -10.0, 10.0);
    simulate(aggressive, "AggressivePID (Euler+None+Raw)", 1.0, 500);

    SmoothPID smooth(2.0, 0.5, 0.1, -10.0, 10.0);
    simulate(smooth, "SmoothPID (Simpson+Tanh+Filtered)", 1.0, 500);

    // Part 2: Benchmark
    std::cout << "\nPart 2: Benchmark (" << BENCH_ITERS << " iterations)\n";
    benchmark();

    std::cout << "\n=== DONE ===\n";
    return 0;
}
