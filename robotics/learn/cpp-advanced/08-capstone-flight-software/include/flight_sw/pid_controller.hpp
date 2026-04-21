#pragma once
// ============================================================================
// PIDController — CRTP-based PID with anti-windup and output saturation
//
// Uses the Curiously Recurring Template Pattern so derived controllers can
// customise compute_error() or filtering without virtual dispatch.
// ============================================================================

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace flight_sw {

inline constexpr double kPiAngle = 3.14159265358979323846;

struct PIDConfig {
    double kp           = 1.0;
    double ki           = 0.0;
    double kd           = 0.0;
    double max_integral = 10.0;    // anti-windup clamp
    double output_min   = -1.0;
    double output_max   =  1.0;
    double dt           = 0.002;   // default 500 Hz
    double deriv_filter  = 0.1;    // low-pass coefficient for derivative (0 = no filter)
};

template <typename Derived>
class PIDBase {
public:
    explicit PIDBase(PIDConfig config) : cfg_{config} {}

    // Main update: returns control output
    [[nodiscard]] inline double update(double measurement) noexcept {
        double error = static_cast<Derived*>(this)->compute_error(setpoint_, measurement);

        // Proportional
        double p_term = cfg_.kp * error;

        // Integral with anti-windup
        integral_ += error * cfg_.dt;
        integral_ = std::clamp(integral_, -cfg_.max_integral, cfg_.max_integral);
        double i_term = cfg_.ki * integral_;

        // Derivative with low-pass filter
        double raw_deriv = (error - prev_error_) / cfg_.dt;
        if (cfg_.deriv_filter > 0.0) {
            filtered_deriv_ = cfg_.deriv_filter * raw_deriv +
                              (1.0 - cfg_.deriv_filter) * filtered_deriv_;
        } else {
            filtered_deriv_ = raw_deriv;
        }
        double d_term = cfg_.kd * filtered_deriv_;

        prev_error_ = error;

        // Output saturation
        double output = p_term + i_term + d_term;
        return std::clamp(output, cfg_.output_min, cfg_.output_max);
    }

    inline void set_setpoint(double sp) noexcept { setpoint_ = sp; }
    [[nodiscard]] inline double setpoint() const noexcept { return setpoint_; }

    inline void reset() noexcept {
        integral_ = 0.0;
        prev_error_ = 0.0;
        filtered_deriv_ = 0.0;
    }

    [[nodiscard]] inline PIDConfig const& config() const noexcept { return cfg_; }

    // Allow runtime retuning
    inline void set_config(PIDConfig const& c) noexcept { cfg_ = c; }

protected:
    PIDConfig cfg_;
    double setpoint_      = 0.0;
    double integral_      = 0.0;
    double prev_error_    = 0.0;
    double filtered_deriv_= 0.0;
};

// ── Concrete PID controller (simple error = setpoint - measurement) ──
class PIDController : public PIDBase<PIDController> {
public:
    using PIDBase::PIDBase;

    [[nodiscard]] inline double compute_error(double sp, double meas) const noexcept {
        return sp - meas;
    }
};

// ── Angle-aware PID (wraps error to [-π, π]) ────────────────────────
class AnglePID : public PIDBase<AnglePID> {
public:
    using PIDBase::PIDBase;

    [[nodiscard]] inline double compute_error(double sp, double meas) const noexcept {
        double err = sp - meas;
        // Wrap to [-π, π]
        while (err >  kPiAngle) err -= 2.0 * kPiAngle;
        while (err < -kPiAngle) err += 2.0 * kPiAngle;
        return err;
    }
};

} // namespace flight_sw
