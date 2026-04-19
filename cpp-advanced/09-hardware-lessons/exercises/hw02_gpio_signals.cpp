// hw02_gpio_signals.cpp — Resistors & Signal Integrity (simulation)
// Compile: g++ -std=c++20 -O2 -Wall -Wextra -Wpedantic -pthread hw02_gpio_signals.cpp -o hw02_gpio_signals
//
// Exercises:
//  1. Voltage divider calculation
//  2. ADC to voltage conversion, battery voltage reconstruction
//  3. I2C pull-up resistance calculator
//  4. Rise time calculator
//  5. I2C speed table
//  6. Interactive R1/R2 calculator

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>

// ---------------------------------------------------------------------------
// 1. Voltage divider: V_out = V_in * R2 / (R1 + R2)
// ---------------------------------------------------------------------------
constexpr double voltage_divider(double v_in, double r1, double r2) {
    return v_in * r2 / (r1 + r2);
}

// ---------------------------------------------------------------------------
// 2a. ADC raw reading → voltage
//     V = raw * V_ref / (2^bits - 1)
// ---------------------------------------------------------------------------
constexpr double adc_to_voltage(uint32_t raw, uint8_t bits, double v_ref) {
    return static_cast<double>(raw) * v_ref / static_cast<double>((1u << bits) - 1u);
}

// ---------------------------------------------------------------------------
// 2b. Reconstruct battery voltage from ADC voltage assuming resistor divider
//     V_batt = V_adc * (R1 + R2) / R2
// ---------------------------------------------------------------------------
constexpr double voltage_to_battery(double v_adc, double r1, double r2) {
    return v_adc * (r1 + r2) / r2;
}

// ---------------------------------------------------------------------------
// 3. I2C pull-up: R_max = rise_time / (0.8473 * C_bus)
//    The 0.8473 factor comes from RC charging to 0.7*Vcc threshold
//    (solving V(t) = Vcc*(1 - e^(-t/RC)) for V = 0.7*Vcc gives t ≈ 0.8473*RC)
// ---------------------------------------------------------------------------
constexpr double i2c_max_pullup(double rise_time_s, double c_bus_f) {
    return rise_time_s / (0.8473 * c_bus_f);
}

// ---------------------------------------------------------------------------
// 4. Rise time: t_rise ≈ 0.8473 * R_pullup * C_bus  (to 70% Vcc)
//    Returns nanoseconds
// ---------------------------------------------------------------------------
constexpr double rise_time_ns(double r_pullup, double c_bus_f) {
    return 0.8473 * r_pullup * c_bus_f * 1e9;
}

// ---------------------------------------------------------------------------
// 5. I2C speed specifications
// ---------------------------------------------------------------------------
struct I2CMode {
    const char* name;
    double freq_hz;
    double max_rise_time_ns;  // from I2C spec
};

static constexpr I2CMode i2c_modes[] = {
    {"Standard (100 kHz)", 100'000, 1000.0},
    {"Fast (400 kHz)",     400'000,  300.0},
    {"Fast+ (1 MHz)",    1'000'000,  120.0},
};

// ---------------------------------------------------------------------------
// 6. Calculate R1, R2 for a divider that maps V_batt to V_adc_max
//    Given a preferred R2, compute R1.
//    R1 = R2 * (V_batt / V_adc_max - 1)
// ---------------------------------------------------------------------------
static void interactive_divider_calculator() {
    std::printf("\n═══ Interactive Battery Divider Calculator ═══\n");
    std::printf("Given: battery voltage, ADC reference, ADC bits, preferred R2\n");
    std::printf("Outputs: R1 to produce full-scale reading at V_batt\n\n");

    double v_batt{}, v_ref{}, r2{};
    int bits{};

    std::printf("  Battery voltage (V): ");
    if (!(std::cin >> v_batt) || v_batt <= 0) {
        std::printf("  Invalid input, using defaults: 12.6V, 3.3V ref, 12-bit, R2=10kΩ\n");
        v_batt = 12.6; v_ref = 3.3; bits = 12; r2 = 10'000;
    } else {
        std::printf("  ADC reference voltage (V): ");
        std::cin >> v_ref;
        std::printf("  ADC bits: ");
        std::cin >> bits;
        std::printf("  Preferred R2 (Ohms): ");
        std::cin >> r2;
    }

    double v_adc_max = v_ref; // full-scale = V_ref
    double r1 = r2 * (v_batt / v_adc_max - 1.0);

    double v_out_check = voltage_divider(v_batt, r1, r2);
    uint32_t max_raw = (1u << bits) - 1u;

    std::printf("\n  ┌───────────────────────────────────────┐\n");
    std::printf("  │ Battery Divider Design                │\n");
    std::printf("  ├───────────────────────────────────────┤\n");
    std::printf("  │ V_batt       = %8.2f V             │\n", v_batt);
    std::printf("  │ V_ref (ADC)  = %8.2f V             │\n", v_ref);
    std::printf("  │ ADC bits     = %8d               │\n", bits);
    std::printf("  │ R1 (top)     = %8.0f Ω             │\n", r1);
    std::printf("  │ R2 (bottom)  = %8.0f Ω             │\n", r2);
    std::printf("  │ V_out @ Vbat = %8.4f V  (≈ Vref)  │\n", v_out_check);
    std::printf("  │ ADC max raw  = %8u               │\n", max_raw);
    std::printf("  └───────────────────────────────────────┘\n");

    // Verification: reconstruct from mid-scale ADC reading
    uint32_t mid_raw = max_raw / 2;
    double v_measured = adc_to_voltage(mid_raw, static_cast<uint8_t>(bits), v_ref);
    double v_batt_reconstructed = voltage_to_battery(v_measured, r1, r2);
    std::printf("\n  Verification: ADC reads %u → %.3f V → battery = %.2f V "
                "(expect ≈ %.2f V)\n",
                mid_raw, v_measured, v_batt_reconstructed, v_batt / 2.0);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::printf("╔══════════════════════════════════════════════════════╗\n");
    std::printf("║       HW02 — Resistors & Signal Integrity           ║\n");
    std::printf("╚══════════════════════════════════════════════════════╝\n\n");

    // --- Voltage divider demo ---
    std::printf("── 1. Voltage Divider ──\n");
    double v_in = 12.0, r1 = 10'000, r2 = 4'700;
    double v_out = voltage_divider(v_in, r1, r2);
    std::printf("  V_in=%.1fV, R1=%.0fΩ, R2=%.0fΩ → V_out = %.3f V\n\n",
                v_in, r1, r2, v_out);

    // --- ADC conversion demo ---
    std::printf("── 2. ADC Conversion ──\n");
    uint32_t raw = 2048;
    uint8_t adc_bits = 12;
    double v_ref = 3.3;
    double v_adc = adc_to_voltage(raw, adc_bits, v_ref);
    double v_batt = voltage_to_battery(v_adc, 27'000, 10'000);
    std::printf("  ADC raw=%u, bits=%u, Vref=%.1fV → V_adc=%.3fV\n",
                raw, adc_bits, v_ref, v_adc);
    std::printf("  With R1=27kΩ, R2=10kΩ divider → V_batt=%.2fV\n\n",
                v_batt);

    // --- I2C pull-up table ---
    std::printf("── 3. I2C Pull-up Resistance Table ──\n");
    double c_bus_pF[] = {50, 100, 200, 400};

    std::printf("  ┌──────────────────┬──────────┬");
    for (size_t i = 0; i < 4; ++i) std::printf("──────────┬");
    std::printf("\n");

    std::printf("  │ Mode             │ Max Rise │");
    for (auto c : c_bus_pF) std::printf(" %4.0f pF  │", c);
    std::printf("\n");

    std::printf("  ├──────────────────┼──────────┼");
    for (size_t i = 0; i < 4; ++i) std::printf("──────────┼");
    std::printf("\n");

    for (const auto& mode : i2c_modes) {
        std::printf("  │ %-16s │ %4.0f ns  │", mode.name, mode.max_rise_time_ns);
        for (auto c : c_bus_pF) {
            double r_max = i2c_max_pullup(mode.max_rise_time_ns * 1e-9, c * 1e-12);
            if (r_max >= 1000)
                std::printf(" %5.1fkΩ  │", r_max / 1000.0);
            else
                std::printf(" %5.0f Ω  │", r_max);
        }
        std::printf("\n");
    }

    std::printf("  └──────────────────┴──────────┴");
    for (size_t i = 0; i < 4; ++i) std::printf("──────────┴");
    std::printf("\n\n");

    // --- Rise time examples ---
    std::printf("── 4. Rise Time Calculator ──\n");
    double pullups[] = {1'000, 2'200, 4'700, 10'000};
    double c_test = 200e-12; // 200 pF

    std::printf("  C_bus = 200 pF\n");
    std::printf("  ┌────────────┬───────────────┐\n");
    std::printf("  │ R_pullup   │ Rise time     │\n");
    std::printf("  ├────────────┼───────────────┤\n");
    for (double rp : pullups) {
        double rt = rise_time_ns(rp, c_test);
        std::printf("  │ %6.0f Ω   │ %8.1f ns   │\n", rp, rt);
    }
    std::printf("  └────────────┴───────────────┘\n");

    // --- Interactive section ---
    interactive_divider_calculator();

    std::printf("\nDone.\n");
    return 0;
}
