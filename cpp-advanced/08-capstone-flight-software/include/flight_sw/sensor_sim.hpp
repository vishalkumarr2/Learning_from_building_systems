#pragma once
// ============================================================================
// SensorSim — Simulated 6-axis IMU with fault injection
//
// Generates sin-wave base signal + configurable Gaussian noise.
// Fault modes: NONE, STUCK, NOISE, DROPOUT
// ============================================================================

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <random>

namespace flight_sw {

inline constexpr double kPi = 3.14159265358979323846;

enum class FaultMode : uint8_t {
    NONE    = 0,
    STUCK   = 1,   // Value frozen at last reading
    NOISE   = 2,   // 10× noise amplification
    DROPOUT = 3    // Intermittent invalid readings
};

struct SensorReading {
    uint64_t                timestamp_ns{0};
    std::array<double, 6>   values{};         // ax, ay, az, gx, gy, gz
    bool                    is_valid{true};
};

class SensorSim {
public:
    explicit SensorSim(double sample_rate_hz = 1000.0,
                       double noise_stddev   = 0.01,
                       unsigned seed         = 42)
        : sample_dt_ns_{static_cast<uint64_t>(1.0e9 / sample_rate_hz)}
        , noise_stddev_{noise_stddev}
        , rng_{seed}
        , noise_dist_{0.0, noise_stddev}
        , dropout_dist_{0.0, 1.0}
    {}

    // Set base signal amplitude and frequency for demo purposes
    inline void set_signal(double amplitude, double frequency_hz) noexcept {
        amplitude_ = amplitude;
        frequency_ = frequency_hz;
    }

    inline void set_fault_mode(FaultMode mode) noexcept {
        fault_ = mode;
    }

    [[nodiscard]] inline FaultMode fault_mode() const noexcept { return fault_; }

    // Read one sample (advances internal time)
    [[nodiscard]] inline SensorReading read() noexcept {
        double t = static_cast<double>(time_ns_) * 1.0e-9;
        time_ns_ += sample_dt_ns_;
        ++sample_count_;

        SensorReading r;
        r.timestamp_ns = time_ns_;

        switch (fault_) {
        case FaultMode::NONE:
            generate_nominal(r, t);
            break;

        case FaultMode::STUCK:
            if (!stuck_captured_) {
                generate_nominal(last_reading_, t);
                stuck_captured_ = true;
            }
            r = last_reading_;
            r.timestamp_ns = time_ns_;
            break;

        case FaultMode::NOISE:
            generate_nominal(r, t);
            for (auto& v : r.values) {
                v += noise_dist_(rng_) * 10.0;  // 10× noise
            }
            break;

        case FaultMode::DROPOUT:
            if (dropout_dist_(rng_) < 0.3) {  // 30% dropout rate
                r.is_valid = false;
                r.values.fill(0.0);
            } else {
                generate_nominal(r, t);
            }
            break;
        }

        if (r.is_valid) last_reading_ = r;
        return r;
    }

    [[nodiscard]] inline uint64_t sample_count() const noexcept { return sample_count_; }

    // Reset fault mode and stuck state
    inline void reset() noexcept {
        fault_ = FaultMode::NONE;
        stuck_captured_ = false;
    }

private:
    uint64_t sample_dt_ns_;
    uint64_t time_ns_       = 0;
    uint64_t sample_count_  = 0;
    double   noise_stddev_;
    double   amplitude_     = 1.0;
    double   frequency_     = 1.0;    // Hz

    FaultMode fault_        = FaultMode::NONE;
    bool stuck_captured_    = false;
    SensorReading last_reading_{};

    std::mt19937                     rng_;
    std::normal_distribution<double> noise_dist_;
    std::uniform_real_distribution<double> dropout_dist_;

    inline void generate_nominal(SensorReading& r, double t) noexcept {
        double base = amplitude_ * std::sin(2.0 * kPi * frequency_ * t);
        // Accelerometer: base signal on X, small cross-coupling on Y/Z
        r.values[0] = base + noise_dist_(rng_);
        r.values[1] = 0.1 * base + noise_dist_(rng_);
        r.values[2] = 9.81 + noise_dist_(rng_);
        // Gyroscope: derivative-ish signal
        double gyro_base = amplitude_ * 2.0 * kPi * frequency_ *
                           std::cos(2.0 * kPi * frequency_ * t);
        r.values[3] = gyro_base + noise_dist_(rng_);
        r.values[4] = 0.05 * gyro_base + noise_dist_(rng_);
        r.values[5] = noise_dist_(rng_);
        r.is_valid = true;
    }
};

} // namespace flight_sw
