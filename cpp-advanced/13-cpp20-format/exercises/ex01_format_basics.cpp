// Module 13 — Exercise 01: std::format Basics
// Compiler: GCC 13+ or Clang 17+ with -std=c++20
//
// This exercise covers fundamental std::format usage:
//   - Basic type formatting (int, float, string, bool)
//   - Positional arguments
//   - Fill, align, width, precision
//   - Integer representations: decimal, hex, octal, binary
//   - Compile-time format string validation

#include <cstdint>
#include <format>
#include <iostream>
#include <string>

int main() {
    std::cout << "=== C++20 std::format Basics ===\n\n";

    // ---------------------------------------------------------------
    // 1. Basic types — auto-numbered arguments
    // ---------------------------------------------------------------
    std::cout << "--- 1. Basic types ---\n";

    int rpm = 3200;
    double temperature = 42.567;
    std::string sensor_name = "imu_front";
    bool is_active = true;

    // {} picks arguments left-to-right, formats with default presentation
    std::cout << std::format("RPM: {}\n", rpm);
    std::cout << std::format("Temp: {}\n", temperature);
    std::cout << std::format("Sensor: {}\n", sensor_name);
    std::cout << std::format("Active: {}\n", is_active);

    // Multiple arguments in one call
    std::cout << std::format("Sensor '{}' reads {:.1f}°C at {} RPM (active={})\n",
                             sensor_name, temperature, rpm, is_active);

    // ---------------------------------------------------------------
    // 2. Positional arguments — reuse and reorder
    // ---------------------------------------------------------------
    std::cout << "\n--- 2. Positional arguments ---\n";

    // {0}, {1}, {2} refer to arguments by index (0-based)
    std::cout << std::format("{0} is {1}, {0} is always {1}\n", "pi", 3.14159);

    // Useful for internationalisation or repeated values
    std::cout << std::format("Motor {0}: current={1:.2f}A, max={2:.2f}A, "
                             "utilisation={3:.1f}% (motor {0})\n",
                             "left_wheel", 2.34, 5.0, 46.8);

    // ---------------------------------------------------------------
    // 3. Fill, align, width
    // ---------------------------------------------------------------
    std::cout << "\n--- 3. Fill, align, width ---\n";

    // Right-align (default for numbers) in 15-char field
    std::cout << std::format("|{:>15}|\n", "right");
    // Left-align in 15-char field
    std::cout << std::format("|{:<15}|\n", "left");
    // Centre-align
    std::cout << std::format("|{:^15}|\n", "centre");

    // Custom fill character
    std::cout << std::format("|{:*>15}|\n", "stars");
    std::cout << std::format("|{:->15}|\n", "dashes");
    std::cout << std::format("|{:.^15}|\n", "dots");

    // Zero-padding for numbers (the '0' flag)
    std::cout << std::format("Frame: {:06d}\n", 42);
    std::cout << std::format("Addr:  0x{:08X}\n", 0xDEAD);

    // ---------------------------------------------------------------
    // 4. Precision (floating-point)
    // ---------------------------------------------------------------
    std::cout << "\n--- 4. Precision ---\n";

    double pi = 3.141592653589793;

    // Default (shortest representation that round-trips)
    std::cout << std::format("default:  {}\n", pi);
    // Fixed notation with N decimal places
    std::cout << std::format("fixed .2: {:.2f}\n", pi);
    std::cout << std::format("fixed .6: {:.6f}\n", pi);
    // Scientific notation
    std::cout << std::format("sci   .4: {:.4e}\n", pi);
    // General (picks shorter of fixed/scientific)
    std::cout << std::format("general:  {:.4g}\n", pi);

    // Width + precision combined — common for sensor data tables
    std::cout << std::format("Table cell: [{:>12.4f}]\n", pi);

    // ---------------------------------------------------------------
    // 5. Integer representations
    // ---------------------------------------------------------------
    std::cout << "\n--- 5. Integer representations ---\n";

    uint32_t reg_value = 0xCAFEBABE;

    std::cout << std::format("decimal:     {:d}\n", reg_value);
    std::cout << std::format("hex lower:   {:x}\n", reg_value);
    std::cout << std::format("hex upper:   {:X}\n", reg_value);
    std::cout << std::format("hex prefix:  {:#x}\n", reg_value);
    std::cout << std::format("octal:       {:o}\n", reg_value);
    std::cout << std::format("octal #:     {:#o}\n", reg_value);
    std::cout << std::format("binary:      {:b}\n", static_cast<uint8_t>(0b10110011));
    std::cout << std::format("binary #:    {:#b}\n", static_cast<uint8_t>(0b10110011));
    std::cout << std::format("char:        {:c}\n", 65);  // 'A'

    // Robotics use: formatting CAN frame data
    std::cout << "\n--- CAN frame hex dump ---\n";
    uint8_t can_data[] = {0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04};
    std::string hex_dump;
    for (auto byte : can_data) {
        std::format_to(std::back_inserter(hex_dump), "{:02X} ", byte);
    }
    std::cout << std::format("CAN ID 0x{:03X}: [{}]\n", 0x1A0, hex_dump);

    // ---------------------------------------------------------------
    // 6. Sign control
    // ---------------------------------------------------------------
    std::cout << "\n--- 6. Sign control ---\n";

    double accel_x = 1.5;
    double accel_y = -0.3;
    double accel_z = 0.0;

    // '+' always shows sign — great for vector components
    std::cout << std::format("accel: x={:+.2f} y={:+.2f} z={:+.2f}\n",
                             accel_x, accel_y, accel_z);
    // ' ' (space) for positive alignment without '+' clutter
    std::cout << std::format("accel: x={: .2f} y={: .2f} z={: .2f}\n",
                             accel_x, accel_y, accel_z);

    // ---------------------------------------------------------------
    // 7. Boolean formatting
    // ---------------------------------------------------------------
    std::cout << "\n--- 7. Boolean formatting ---\n";

    std::cout << std::format("as text:    {}\n", true);     // "true"
    std::cout << std::format("as integer: {:d}\n", true);   // "1"

    // ---------------------------------------------------------------
    // 8. Practical: sensor data table
    // ---------------------------------------------------------------
    std::cout << "\n--- 8. Sensor data table ---\n";

    // Header
    std::cout << std::format("{:<18} {:>8} {:>8} {:>8} {:>6}\n",
                             "Sensor", "Value", "Min", "Max", "Unit");
    std::cout << std::string(52, '-') << '\n';

    // Data rows — aligned columns for readability
    std::cout << std::format("{:<18} {:>8.2f} {:>8.2f} {:>8.2f} {:>6}\n",
                             "sensorbar_front", 72.5, 60.0, 85.0, "raw");
    std::cout << std::format("{:<18} {:>8.2f} {:>8.2f} {:>8.2f} {:>6}\n",
                             "sensorbar_rear", 68.3, 55.0, 80.0, "raw");
    std::cout << std::format("{:<18} {:>8.2f} {:>8.2f} {:>8.2f} {:>6}\n",
                             "motor_temp_L", 45.2, 20.0, 80.0, "°C");
    std::cout << std::format("{:<18} {:>8.2f} {:>8.2f} {:>8.2f} {:>6}\n",
                             "motor_temp_R", 43.8, 20.0, 80.0, "°C");
    std::cout << std::format("{:<18} {:>8.2f} {:>8.2f} {:>8.2f} {:>6}\n",
                             "battery_voltage", 25.4, 22.0, 28.0, "V");

    // ---------------------------------------------------------------
    // 9. Compile-time format string validation
    // ---------------------------------------------------------------
    std::cout << "\n--- 9. Compile-time safety ---\n";
    std::cout << "std::format validates format strings at compile time.\n";
    std::cout << "Uncomment the lines below to see compile errors:\n\n";

    // COMPILE ERROR: too few arguments
    // std::format("{} {} {}", 1, 2);  // expected 3 args, got 2

    // COMPILE ERROR: invalid spec for string type
    // std::format("{:d}", "hello");   // 'd' is for integers, not strings

    // COMPILE ERROR: argument index out of range
    // std::format("{3}", 1, 2);       // only indices 0,1 available

    std::cout << "All exercises complete.\n";
    return 0;
}
