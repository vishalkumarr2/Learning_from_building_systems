# 05 — MCU Motor Control Hardware
### PWM, H-bridges, current sensing, and encoders — the silicon between your PID and the motor

**Prerequisite:** `04-discrete-time-control.md` (discrete PID, sampling)
**Unlocks:** `06-fixed-point-pid.md` (software that runs on this hardware)

---

## Why Should I Care? (OKS Context)

The PID controller computes a number. But the motor needs *current* flowing through *copper*. Between the PID output and actual motor torque, there's an H-bridge, a current sensor, a PWM generator, and an encoder. When any of these misbehaves, the PID is flying blind.

**Real OKS failures traced to hardware:**
- Encoder connector vibration → intermittent count loss → speed estimate spikes → PID overreacts → oscillation
- Current sensor offset drift with temperature → steady-state current error → motor runs hot
- PWM dead-time too short → H-bridge shoot-through → FET damage

---

# PART 1 — PWM GENERATION

## 1.1 Why PWM?

A DC motor needs variable voltage. You could use a linear regulator, but it wastes power as heat. **PWM** switches the full supply voltage on and off rapidly. The motor's inductance acts as a low-pass filter, averaging the pulses into effective DC.

**Effective voltage:** $V_{eff} = V_{supply} \times D$

where $D$ = duty cycle (0.0 to 1.0). At $D = 0.5$, the motor sees half the supply voltage.

```
PWM at 50% duty cycle:
  V ───┐     ┌───┐     ┌───┐     ┌───
  │    │     │   │     │   │     │
  │    │     │   │     │   │     │
  0 ───┘     └───┘     └───┘     └───
       ├─Ton─┤Toff├─Ton─┤

  D = Ton / (Ton + Toff) = 0.5
  f_pwm = 1 / (Ton + Toff)
```

## 1.2 PWM Frequency Selection

| Frequency | Motor effect | Audio | OKS choice |
|-----------|-------------|-------|------------|
| 1–4 kHz | High current ripple, inefficient | Audible whine | ❌ |
| 10–20 kHz | Moderate ripple, good efficiency | Barely audible | ✅ 20 kHz |
| 50–100 kHz | Low ripple, switching losses increase | Silent | ❌ (FET losses) |

**OKS uses 20 kHz** — just above human hearing, low enough for efficient FET switching.

**Current ripple** at PWM frequency $f_{pwm}$:

$$\Delta I = \frac{V_{supply} \cdot D \cdot (1-D)}{L \cdot f_{pwm}}$$

At 20 kHz, $V_{supply} = 12$ V, $L = 1$ mH, $D = 0.5$:
$\Delta I = \frac{12 \times 0.25}{0.001 \times 20000} = 0.15$ A peak-to-peak.

At 50 kHz, the ripple drops to 0.06 A — but the FET switching losses triple.

## 1.3 Timer Configuration (STM32)

The STM32's hardware timer generates PWM without CPU intervention:

```c
// STM32 HAL - TIM1 configured for center-aligned PWM at 20 kHz
// System clock = 168 MHz, prescaler = 0, ARR = 4199
// Center-aligned → effective PWM freq = 168e6 / (2 × 4200) = 20 kHz

void motor_pwm_init(void) {
    TIM_HandleTypeDef htim1;
    htim1.Instance = TIM1;
    htim1.Init.Prescaler = 0;
    htim1.Init.CounterMode = TIM_COUNTERMODE_CENTERALIGNED1;
    htim1.Init.Period = 4199;  // ARR value
    htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
    HAL_TIM_PWM_Init(&htim1);
    
    // Channel 1: motor PWM
    TIM_OC_InitTypeDef oc_config;
    oc_config.OCMode = TIM_OCMODE_PWM1;
    oc_config.Pulse = 0;        // Start at 0% duty
    oc_config.OCPolarity = TIM_OCPOLARITY_HIGH;
    HAL_TIM_PWM_ConfigChannel(&htim1, &oc_config, TIM_CHANNEL_1);
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);
}

void motor_set_duty(float duty) {
    // duty: -1.0 to +1.0
    uint16_t ccr = (uint16_t)(fabsf(duty) * 4199.0f);
    __HAL_TIM_SET_COMPARE(&htim1, TIM_CHANNEL_1, ccr);
    // Set direction pin based on sign
    HAL_GPIO_WritePin(DIR_PORT, DIR_PIN, duty >= 0 ? GPIO_PIN_SET : GPIO_PIN_RESET);
}
```

**Center-aligned PWM** is preferred for motor control because the ADC can sample current at the center of the PWM period (when all switches are in a known state), giving a clean current measurement.

---

# PART 2 — H-BRIDGE

## 2.1 How an H-Bridge Works

An H-bridge uses 4 switches (MOSFETs) to control current direction through the motor:

```
        V+
        │
    ┌───┴───┐
    │       │
   Q1      Q3
    │       │
    ├── M ──┤     M = Motor
    │       │
   Q2      Q4
    │       │
    └───┬───┘
        │
       GND

Forward:  Q1 ON, Q4 ON, Q2 OFF, Q3 OFF → current flows left to right
Reverse:  Q3 ON, Q2 ON, Q1 OFF, Q4 OFF → current flows right to left
Brake:    Q1 ON, Q3 ON (or Q2 ON, Q4 ON) → motor shorted → dynamic braking
Coast:    All OFF → motor coasts (back-EMF drives current through body diodes)
```

## 2.2 Dead Time (Critical Safety Feature)

If Q1 and Q2 are BOTH on simultaneously → short circuit from V+ to GND → **shoot-through** → massive current → FET destruction.

**Dead time** is a brief period (0.5–2 µs) when BOTH switches in a leg are OFF during transitions:

```
Q1:  ────┐       ┌────
         │       │
Q2:      └─┐   ┌─┘
            │   │
            ├───┤ ← dead time (both OFF)
```

**OKS STM32 dead time:** 1 µs, configured in TIM1's BDTR register:

```c
TIM_BreakDeadTimeConfigTypeDef sBreakDeadTimeConfig;
sBreakDeadTimeConfig.DeadTime = 168;  // 1 µs at 168 MHz
HAL_TIMEx_ConfigBreakDeadTime(&htim1, &sBreakDeadTimeConfig);
```

**Dead-time distortion:** During dead time, the output voltage is neither high nor low — it depends on current direction. This creates a nonlinearity, especially visible at low duty cycles. The PID controller sees this as a dead zone near zero speed. Compensation:

```c
// Dead-time compensation
float compensated_duty = duty;
if (fabsf(duty) > 0.01f) {
    float dead_time_comp = DEAD_TIME_US * PWM_FREQ * 2.0f; // normalized
    compensated_duty += (duty > 0) ? dead_time_comp : -dead_time_comp;
}
```

---

# PART 3 — CURRENT SENSING

## 3.1 Why Measure Current?

1. **Current limiting:** Prevent motor/driver damage (stall current can be 3–5× running current)
2. **Inner control loop:** Current PID provides fastest torque control
3. **Diagnostics:** Abnormal current = motor fault, binding, overload

## 3.2 Sensing Methods

| Method | Principle | OKS usage | Pros/Cons |
|--------|-----------|-----------|-----------|
| **Shunt resistor** | Low-R in series, measure voltage | ✅ Low-side | Cheap, fast; power loss = $I^2 R$ |
| **Hall effect** | Magnetic field sensing | ❌ | No power loss; expensive, slow |
| **Inline (high-side)** | Shunt on V+ side | Some models | Sees all current; needs differential amp |

**OKS low-side shunt sensing:**

```
        V+
        │
       Q1, Q3 (H-bridge)
        │
       MOTOR
        │
       Q2, Q4 (H-bridge)
        │
       ┤R_shunt├──→ to ADC (via op-amp)
        │
       GND
```

$R_{shunt}$ = 10–50 mΩ. At 3A: $V_{sense} = 3 \times 0.02 = 60$ mV. This needs amplification.

**Current sense amplifier:**

```c
// OKS current sense: 20 mΩ shunt, 50× gain amplifier
// ADC: 12-bit, 3.3V reference
// I = (ADC_value / 4096 * 3.3 - offset) / (R_shunt * gain)

#define R_SHUNT     0.020f    // 20 mΩ
#define AMP_GAIN    50.0f     // INA180 or similar
#define ADC_VREF    3.3f
#define ADC_RES     4096.0f
#define V_OFFSET    1.65f     // Midpoint for bidirectional sensing

float read_motor_current(uint16_t adc_raw) {
    float voltage = (float)adc_raw / ADC_RES * ADC_VREF;
    float current = (voltage - V_OFFSET) / (R_SHUNT * AMP_GAIN);
    return current;  // Amps, signed
}
```

## 3.3 ADC Timing with PWM

Sample the ADC at the **center of the PWM period** (when the low-side FETs are on and current is flowing through the shunt):

```
PWM:    ──┐    ┌──────────┐    ┌──
          │    │          │    │
          └────┘          └────┘
              ▲               ▲
              │               │
           ADC sample      ADC sample
           (center)        (center)
```

STM32 can trigger ADC from the timer's center event automatically — no CPU involvement.

---

# PART 4 — QUADRATURE ENCODERS

## 4.1 How Encoders Work

A quadrature encoder outputs two square waves (A and B) shifted by 90°:

```
Forward rotation:
  A: ──┐  ┌──┐  ┌──┐  ┌──
       │  │  │  │  │  │
       └──┘  └──┘  └──┘

  B:    ──┐  ┌──┐  ┌──┐  ┌──
          │  │  │  │  │  │
          └──┘  └──┘  └──┘
       ↑
       B leads A by 90°

Reverse rotation:
  A:    ──┐  ┌──┐  ┌──┐  ┌──
          │  │  │  │  │  │
          └──┘  └──┘  └──┘

  B: ──┐  ┌──┐  ┌──┐  ┌──
       │  │  │  │  │  │
       └──┘  └──┘  └──┘
       ↑
       A leads B by 90°
```

**4× decoding** counts all edges of both A and B channels:

| A | B | Event | Direction |
|---|---|-------|-----------|
| ↑ | 0 | Count +1 | Forward |
| ↑ | 1 | Count -1 | Reverse |
| ↓ | 0 | Count -1 | Reverse |
| ↓ | 1 | Count +1 | Forward |

**STM32 encoder mode** does this in hardware:

```c
void encoder_init(void) {
    TIM_HandleTypeDef htim3;
    htim3.Instance = TIM3;
    
    TIM_Encoder_InitTypeDef encoder_config;
    encoder_config.EncoderMode = TIM_ENCODERMODE_TI12;  // 4× mode
    encoder_config.IC1Polarity = TIM_ICPOLARITY_RISING;
    encoder_config.IC2Polarity = TIM_ICPOLARITY_RISING;
    encoder_config.IC1Filter = 0x05;  // Noise filter: 8 clock cycles
    
    HAL_TIM_Encoder_Init(&htim3, &encoder_config);
    HAL_TIM_Encoder_Start(&htim3, TIM_CHANNEL_ALL);
}

int32_t encoder_get_count(void) {
    return (int16_t)__HAL_TIM_GET_COUNTER(&htim3);
}
```

## 4.2 Speed Computation From Encoder

```c
static int32_t prev_count = 0;

float compute_speed_rad_per_sec(float dt) {
    int32_t count = encoder_get_count();
    int32_t diff = count - prev_count;
    prev_count = count;
    
    // Handle 16-bit timer overflow
    if (diff > 32767) diff -= 65536;
    if (diff < -32767) diff += 65536;
    
    float rad_per_count = 2.0f * M_PI / (float)(ENCODER_CPR * 4);
    return (float)diff * rad_per_count / dt;
}
```

## 4.3 Common Encoder Failure Modes

| Failure | Symptom | OKS ticket |
|---------|---------|------------|
| **Loose connector** | Intermittent zero-speed spikes | #98367 |
| **Electrical noise** | Random count jumps (1000+ counts in one sample) | #99835 |
| **Index pulse loss** | Cumulative position drift | Commissioning issue |
| **One channel dead** | Speed always positive (can't detect direction) | #97229 |

**Noise filter:** The STM32 encoder mode has a configurable digital filter (IC1Filter, IC2Filter). Set to 0x05 (8 clock cycles) → ignores glitches shorter than 48 ns at 168 MHz.

---

# PART 5 — PUTTING IT TOGETHER

## 5.1 The Complete Motor Control Signal Path

```
                    ┌─────────────────────────────────────────────┐
                    │              STM32 MCU                       │
                    │                                             │
  speed_cmd ──SPI──→│ Speed PID ──→ Current PID ──→ PWM Register │
                    │    ↑              ↑             │           │
                    │    │              │             ▼           │
                    │ Encoder ◄──   ADC ◄──     TIM1 PWM out     │
                    │ (TIM3)      (DMA)           │             │
                    └─────────────────────────────│─────────────┘
                                                  │
                    ┌─────────────────────────────│─────────────┐
                    │          POWER STAGE         │             │
                    │                              ▼             │
                    │  V_bat ──→ H-Bridge ──→ Motor ──→ Encoder │
                    │               │                     │      │
                    │           Shunt R ──→ Amp           │      │
                    │               │         │           │      │
                    │               ▼         ▼           ▼      │
                    │            to ADC    to ADC     to TIM3    │
                    └────────────────────────────────────────────┘
```

**Timing budget (per 10 kHz cycle = 100 µs):**

| Step | Time | How |
|------|------|-----|
| ADC current sample | 1 µs | DMA-triggered by timer center event |
| Encoder read | 0.1 µs | Hardware register read |
| Speed computation | 0.5 µs | Integer subtract + multiply |
| Speed PID | 0.5 µs | 3 multiply-adds |
| Current PID | 0.3 µs | 2 multiply-adds |
| PWM update | 0.1 µs | Register write |
| **Total** | **~2.5 µs** | **2.5% of 100 µs budget** |

The remaining 97.5% is available for SPI communication, diagnostics, watchdog, and other tasks.

---

## Checkpoint Questions

1. Why does OKS use 20 kHz PWM instead of 5 kHz or 100 kHz?
2. What is dead time and why is it necessary? What happens without it?
3. Why sample the ADC at the center of the PWM period instead of at the edge?
4. An encoder has 512 CPR. With 4× decoding, how many counts per revolution?
5. The encoder count jumps by 5000 in one 100 µs sample. Is the motor spinning that fast, or is something wrong?
6. Draw the signal path from speed command to motor torque, identifying every A/D and D/A conversion.

---

## Key Takeaways

- **PWM frequency = 20 kHz**: above hearing, efficient FET switching, acceptable current ripple
- **Dead time prevents shoot-through** but creates a nonlinearity that needs compensation
- **Current sensing** via low-side shunt + amplifier, ADC triggered at PWM center
- **Quadrature encoder** decoded in hardware (STM32 timer encoder mode), with noise filtering
- **The complete signal path** runs in ~2.5 µs of a 100 µs budget — hardware offloading (DMA, timer, encoder mode) is essential
