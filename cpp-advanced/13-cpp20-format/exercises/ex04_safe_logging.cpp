// Module 13 — Exercise 04: Type-Safe Logging with std::format
// Compiler: GCC 13+ or Clang 17+ with -std=c++20
//
// Builds a production-quality logging system that:
//   - Uses std::format for type-safe message formatting
//   - Provides compile-time format string validation
//   - Is thread-safe with mutex-protected output
//   - Includes timestamps and log levels
//   - Demonstrates how this replaces ROS_INFO() / RCLCPP_INFO()
//   - Prevents format string vulnerabilities (no runtime format strings)

#include <chrono>
#include <cstdint>
#include <format>
#include <iostream>
#include <mutex>
#include <source_location>
#include <string_view>
#include <thread>

// =============================================================================
// Log levels — ordered by severity
// =============================================================================
enum class LogLevel : uint8_t {
    TRACE = 0,
    DEBUG = 1,
    INFO  = 2,
    WARN  = 3,
    ERROR = 4,
    FATAL = 5,
};

// Formatter for LogLevel so we can use it directly in std::format
template <>
struct std::formatter<LogLevel> : std::formatter<std::string_view> {
    auto format(LogLevel level, std::format_context& ctx) const {
        std::string_view name;
        switch (level) {
            case LogLevel::TRACE: name = "TRACE"; break;
            case LogLevel::DEBUG: name = "DEBUG"; break;
            case LogLevel::INFO:  name = "INFO "; break;
            case LogLevel::WARN:  name = "WARN "; break;
            case LogLevel::ERROR: name = "ERROR"; break;
            case LogLevel::FATAL: name = "FATAL"; break;
        }
        return std::formatter<std::string_view>::format(name, ctx);
    }
};

// =============================================================================
// Logger — thread-safe, type-safe logging with compile-time format checking
// =============================================================================
class Logger {
public:
    // Set the minimum log level (messages below this are silently dropped)
    static void set_level(LogLevel level) {
        min_level_ = level;
    }

    static LogLevel level() { return min_level_; }

    // Core log function — uses std::format_string for compile-time validation.
    //
    // std::format_string<Args...> is the key: it validates the format string
    // at compile time, catching type mismatches and missing arguments.
    // This is what makes std::format immune to format string attacks.
    template <typename... Args>
    static void log(LogLevel level,
                    std::format_string<Args...> fmt,
                    Args&&... args) {
        if (level < min_level_) return;

        // Format the user message (compile-time validated)
        std::string user_msg = std::format(fmt, std::forward<Args>(args)...);

        // Get current timestamp
        auto now = std::chrono::system_clock::now();
        auto epoch = now.time_since_epoch();
        auto secs = std::chrono::duration_cast<std::chrono::seconds>(epoch);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(epoch) -
                  std::chrono::duration_cast<std::chrono::milliseconds>(secs);

        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::tm tm_buf{};
        gmtime_r(&time_t_now, &tm_buf);

        // Build full log line with timestamp, level, thread ID
        std::string line = std::format(
            "[{:04d}-{:02d}-{:02d}T{:02d}:{:02d}:{:02d}.{:03d}Z] "
            "[{}] [tid:{:>5}] {}",
            tm_buf.tm_year + 1900, tm_buf.tm_mon + 1, tm_buf.tm_mday,
            tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec,
            static_cast<int>(ms.count()),
            level,
            std::hash<std::thread::id>{}(std::this_thread::get_id()) % 100000,
            user_msg);

        // Thread-safe output
        std::lock_guard<std::mutex> lock(mutex_);
        if (level >= LogLevel::WARN) {
            std::cerr << line << '\n';
        } else {
            std::cout << line << '\n';
        }
    }

    // Convenience methods — each wraps log() with a fixed level
    template <typename... Args>
    static void trace(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::TRACE, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static void debug(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::DEBUG, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static void info(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::INFO, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static void warn(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::WARN, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static void error(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::ERROR, fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static void fatal(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::FATAL, fmt, std::forward<Args>(args)...);
    }

private:
    static inline LogLevel min_level_ = LogLevel::TRACE;
    static inline std::mutex mutex_;
};

// =============================================================================
// Simulated robot types (for demonstration)
// =============================================================================

struct Pose2D {
    double x, y, theta;
};

template <>
struct std::formatter<Pose2D> {
    constexpr auto parse(std::format_parse_context& ctx) { return ctx.begin(); }
    auto format(const Pose2D& p, std::format_context& ctx) const {
        return std::format_to(ctx.out(), "(x={:.3f}, y={:.3f}, θ={:.3f})",
                              p.x, p.y, p.theta);
    }
};

enum class MotorState : uint8_t { OK, WARNING, FAULT, DISCONNECTED };

template <>
struct std::formatter<MotorState> : std::formatter<std::string_view> {
    auto format(MotorState s, std::format_context& ctx) const {
        std::string_view name;
        switch (s) {
            case MotorState::OK:           name = "OK"; break;
            case MotorState::WARNING:      name = "WARNING"; break;
            case MotorState::FAULT:        name = "FAULT"; break;
            case MotorState::DISCONNECTED: name = "DISCONNECTED"; break;
        }
        return std::formatter<std::string_view>::format(name, ctx);
    }
};

// =============================================================================
// Main — demonstrate the logging system
// =============================================================================

int main() {
    std::cout << "=== Type-Safe Logging with std::format ===\n\n";

    // --- Basic logging at all levels ---
    Logger::set_level(LogLevel::TRACE);

    Logger::trace("System initialising, PID={}", getpid());
    Logger::debug("Loading config from {}", "/etc/robot/config.yaml");
    Logger::info("Robot started, firmware v{}.{}.{}", 1, 2, 4);
    Logger::warn("Battery at {}%, consider charging", 15);
    Logger::error("Motor fault: current={:.2f}A exceeds limit={:.2f}A", 5.8, 5.0);
    Logger::fatal("Watchdog timeout after {}ms, initiating E-stop", 500);

    std::cout << "\n--- Filtering: only WARN and above ---\n\n";
    Logger::set_level(LogLevel::WARN);

    Logger::debug("This won't appear");
    Logger::info("Neither will this");
    Logger::warn("Low battery: {}%", 10);
    Logger::error("Sensorbar dropout on {}", "sensorbar_front");

    // --- Practical robotics examples ---
    std::cout << "\n--- Robotics examples (all levels) ---\n\n";
    Logger::set_level(LogLevel::TRACE);

    // Navigation
    Pose2D pose{1.234, -0.567, 1.571};
    Logger::info("Navigation: pose={}", pose);
    Logger::debug("Path planner: {} waypoints, ETA={:.1f}s", 12, 8.5);

    // Motor diagnostics
    MotorState left_motor = MotorState::OK;
    MotorState right_motor = MotorState::FAULT;
    Logger::info("Motor status: left={} right={}", left_motor, right_motor);

    if (right_motor == MotorState::FAULT) {
        Logger::error("Motor {} in state {}, current={:.2f}A temp={:.1f}°C",
                      "right_wheel", right_motor, 6.2, 78.5);
    }

    // Sensor hex data
    uint8_t spi_resp[] = {0xFF, 0x00, 0xA5, 0x5A};
    std::string hex;
    for (auto b : spi_resp) {
        std::format_to(std::back_inserter(hex), "{:02X} ", b);
    }
    Logger::debug("SPI response [{}B]: {}", sizeof(spi_resp), hex);

    // Register dump
    uint32_t status_reg = 0xDEAD0042;
    Logger::debug("Status register: {:#010x} (bits: {:032b})", status_reg, status_reg);

    // --- Thread safety demonstration ---
    std::cout << "\n--- Multi-threaded logging ---\n\n";

    auto worker = [](int id, int count) {
        for (int i = 0; i < count; ++i) {
            Logger::info("Worker {}: iteration {}/{}", id, i + 1, count);
        }
    };

    std::thread t1(worker, 1, 3);
    std::thread t2(worker, 2, 3);
    t1.join();
    t2.join();

    // --- Compile-time safety ---
    std::cout << "\n--- Compile-time safety ---\n";
    std::cout << "Uncomment the lines below to see compile errors:\n\n";

    // COMPILE ERROR: wrong number of arguments
    // Logger::info("{} {} {}", 1, 2);

    // COMPILE ERROR: format spec 'd' invalid for string
    // Logger::info("{:d}", "not a number");

    // FORMAT STRING ATTACK PREVENTION:
    // In printf-world, user input as format string → stack read/write.
    // With std::format, the format string MUST be a compile-time constant.
    //
    // std::string user_input = get_user_input();
    // Logger::info(user_input, args...);  // WON'T COMPILE — user_input isn't constexpr
    //
    // This is a massive security win for any system processing external input.

    std::cout << "Compare with old ROS-style logging:\n";
    std::cout << "  ROS1:  ROS_INFO(\"Pose: x=%.3f y=%.3f\", x, y); // printf, unsafe\n";
    std::cout << "  ROS2:  RCLCPP_INFO(logger, \"Pose: x=%.3f\", x); // still printf\n";
    std::cout << "  Ours:  Logger::info(\"Pose: x={:.3f} y={:.3f}\", x, y); // type-safe!\n";

    std::cout << "\nAll exercises complete.\n";
    return 0;
}
