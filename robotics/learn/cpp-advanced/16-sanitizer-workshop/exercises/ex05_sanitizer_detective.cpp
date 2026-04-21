// Module 16: Sanitizer Workshop
// Exercise 05: Sanitizer Detective — Real-World-Like Subtle Bugs
//
// A RobotController class with sensor fusion and command output.
// Contains 3 hidden bugs that only manifest under specific conditions:
//
//   Bug A: A use-after-move that works "by luck" most of the time
//   Bug B: An unsigned underflow in array indexing
//   Bug C: A race between the control thread and the telemetry thread
//
// Your task: figure out WHICH sanitizer to use for each bug, then find & fix them.
//
// Build commands:
//   g++ -std=c++2a -g -O1 ex05_sanitizer_detective.cpp -o ex05 -pthread
//   g++ -std=c++2a -fsanitize=address -fno-omit-frame-pointer -g ex05_sanitizer_detective.cpp -o ex05_asan -pthread
//   g++ -std=c++2a -fsanitize=undefined -g ex05_sanitizer_detective.cpp -o ex05_ubsan -pthread
//   g++ -std=c++2a -fsanitize=thread -g ex05_sanitizer_detective.cpp -o ex05_tsan -pthread
//
// Hint: Bug A triggers only when sensor_count > 4
//       Bug B triggers only when the first sensor reading is 0
//       Bug C triggers when telemetry is read while control loop runs

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

// ============================================================
// Sensor data structure
// ============================================================
struct SensorReading {
    std::string sensor_name;
    std::vector<double> values;
    unsigned int sequence_number;

    SensorReading() : sequence_number(0) {}
    SensorReading(std::string name, std::vector<double> vals, unsigned int seq)
        : sensor_name(std::move(name)), values(std::move(vals)), sequence_number(seq) {}
};

// ============================================================
// Telemetry data (shared between threads)
// ============================================================
struct Telemetry {
    double commanded_velocity;
    double estimated_position;
    int error_count;
    std::string last_status;
};

// ============================================================
// RobotController class
// ============================================================
class RobotController {
public:
    explicit RobotController(int sensor_count)
        : sensor_count_(sensor_count),
          running_(false),
          telemetry_{0.0, 0.0, 0, "INIT"} {
        sensor_history_.resize(static_cast<std::size_t>(sensor_count));
    }

    // Run the control loop for a given number of iterations
    void run(int iterations) {
        running_.store(true);

        // Start telemetry reporting thread
        std::thread telemetry_thread([this]() {
            while (running_.load()) {
                report_telemetry();
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        });

        // Main control loop
        for (int i = 0; i < iterations && running_.load(); ++i) {
            auto readings = acquire_sensors(i);
            auto fused = fuse_sensors(std::move(readings));
            compute_command(fused);
        }

        running_.store(false);
        telemetry_thread.join();
    }

private:
    // Simulate sensor acquisition
    std::vector<SensorReading> acquire_sensors(int step) {
        std::vector<SensorReading> readings;
        readings.reserve(static_cast<std::size_t>(sensor_count_));

        for (int s = 0; s < sensor_count_; ++s) {
            std::vector<double> values;
            values.reserve(10);
            for (int v = 0; v < 10; ++v) {
                // Generate synthetic sensor data
                double val = static_cast<double>(step * sensor_count_ + s + v);
                values.push_back(val * 0.01);
            }
            readings.emplace_back(
                "sensor_" + std::to_string(s),
                std::move(values),
                static_cast<unsigned int>(step)
            );
        }
        return readings;
    }

    // BUG A: Use-after-move
    // The readings vector is moved into process_batch(), then we access
    // it again to extract the "best" reading. This works when sensor_count <= 4
    // because small-vector optimization or luck keeps the data accessible.
    // With sensor_count > 4, the moved-from vector is truly empty.
    __attribute__((noinline))
    double fuse_sensors(std::vector<SensorReading> readings) {
        if (readings.empty()) return 0.0;

        // Process the batch (consumes readings via move)
        double batch_result = process_batch(std::move(readings));

        // BUG A: accessing 'readings' AFTER it was moved from!
        // The vector is in a valid-but-unspecified state (likely empty).
        // When sensor_count > 4, readings.size() is 0 and readings[0] is UB.
        // With sensor_count <= 4, SSO or other factors may keep data around.
        double best = 0.0;
        if (!readings.empty()) {
            // This branch rarely executes after a move (vector is usually empty)
            best = readings[0].values.empty() ? 0.0 : readings[0].values[0];
        } else {
            // Fallback — but this hides the bug in most runs
            best = batch_result;
        }

        return best + batch_result;
    }

    __attribute__((noinline))
    double process_batch(std::vector<SensorReading> batch) {
        double sum = 0.0;
        for (const auto& reading : batch) {
            for (double v : reading.values) {
                sum += v;
            }
        }
        return sum / static_cast<double>(std::max(batch.size(), std::size_t{1}));
    }

    // BUG B: Unsigned underflow in array indexing
    // When the sensor reading at index 0 is exactly 0.0, the computed index
    // underflows because we subtract 1 from a size_t that's 0.
    __attribute__((noinline))
    void compute_command(double fused_value) {
        // Determine which history slot to use based on fused value
        std::size_t slot_count = sensor_history_.size();

        // Convert fused_value to a history index
        auto raw_idx = static_cast<std::size_t>(fused_value * 100.0);
        std::size_t idx = raw_idx % slot_count;

        // BUG B: If fused_value is exactly 0.0 on the first iteration,
        // raw_idx is 0. Then (idx - 1) underflows to SIZE_MAX.
        // This causes a massive out-of-bounds access.
        std::size_t prev_idx = idx - 1;  // BUG: underflows when idx == 0

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
        if (prev_idx < slot_count) {
            sensor_history_[prev_idx] = fused_value;
        }
#pragma GCC diagnostic pop

        // BUG C: Update telemetry without lock — races with report_telemetry()
        telemetry_.commanded_velocity = fused_value * 0.5;
        telemetry_.estimated_position += fused_value * 0.001;
        telemetry_.last_status = "RUNNING_" + std::to_string(static_cast<int>(fused_value));
    }

    // BUG C: Read telemetry from a different thread without synchronization
    __attribute__((noinline))
    void report_telemetry() {
        // BUG C: reading telemetry_ members while compute_command() writes them
        // This is a data race on commanded_velocity, estimated_position, last_status
        double vel = telemetry_.commanded_velocity;
        double pos = telemetry_.estimated_position;
        // Accessing std::string from two threads without sync is especially dangerous
        // (can cause use-after-free on string's internal buffer)
        std::string status = telemetry_.last_status;

        // Suppress output to avoid flooding
        if (vel > 100.0) {
            std::cout << "Telemetry: vel=" << vel << " pos=" << pos
                      << " status=" << status << "\n";
        }
    }

    int sensor_count_;
    std::atomic<bool> running_;
    std::vector<double> sensor_history_;
    Telemetry telemetry_;
};

// ============================================================
// Main: different scenarios trigger different bugs
// ============================================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <scenario>\n";
        std::cerr << "  1: Normal run (2 sensors, 50 iterations) — may appear clean\n";
        std::cerr << "  2: More sensors (6 sensors) — triggers Bug A (use-after-move)\n";
        std::cerr << "  3: Zero initial reading — triggers Bug B (unsigned underflow)\n";
        std::cerr << "  4: Full stress test — triggers Bug C (data race)\n";
        std::cerr << "  5: All bugs (6 sensors, 200 iterations)\n";
        return 1;
    }

    int scenario = std::atoi(argv[1]);

    switch (scenario) {
        case 1: {
            // Likely appears clean (bugs hidden)
            RobotController ctrl(2);
            ctrl.run(50);
            break;
        }
        case 2: {
            // Bug A: use-after-move with many sensors
            RobotController ctrl(6);
            ctrl.run(100);
            break;
        }
        case 3: {
            // Bug B: unsigned underflow (first iteration fused_value ≈ 0)
            RobotController ctrl(1);
            ctrl.run(10);
            break;
        }
        case 4: {
            // Bug C: data race between control and telemetry threads
            RobotController ctrl(3);
            ctrl.run(500);
            break;
        }
        case 5: {
            // All bugs: stress test
            RobotController ctrl(6);
            ctrl.run(200);
            break;
        }
        default:
            std::cerr << "Invalid scenario: " << scenario << "\n";
            return 1;
    }

    std::cout << "Scenario " << scenario << " completed.\n";
    return 0;
}
