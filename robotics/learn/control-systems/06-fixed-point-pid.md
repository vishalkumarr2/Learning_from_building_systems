# 06 — Fixed-Point PID on a Microcontroller
### No floats, no malloc, no mercy — integer-only control at 10 kHz

**Prerequisite:** `05-mcu-motor-control.md` (hardware), `04-discrete-time-control.md` (discrete PID)
**Unlocks:** `07-cascaded-loops.md` (stacking multiple PIDs)

---

## Why Should I Care? (Practical Context)

The STM32F4 in our robot *has* a floating-point unit (FPU), so you might think: "Why bother with fixed-point?" Three reasons:

1. **Deterministic timing:** Fixed-point operations take exactly the same time every call. Floating-point can vary with denormalized numbers, NaN propagation, and FPU pipeline stalls.
2. **Smaller MCUs:** The line-sensor controller (STM32G0 series) has NO FPU. All math must be integer.
3. **FPGA/DSP targets:** When porting PID to an FPGA (for current loop offloading), you need integer math.

**Real AMR failure:** A firmware developer used `float` for the current PID on an STM32G0-based line-sensor board. Compilation succeeded (soft-float), but execution was 20× slower (software float emulation). The 10 kHz ISR took 150 µs instead of 7 µs, causing timer overruns and erratic sensor readings.

---

# PART 1 — FIXED-POINT ARITHMETIC

## 1.1 What Is Fixed-Point?

Fixed-point represents fractional numbers using integers by implicitly dividing by a power of 2:

$$\text{real value} = \frac{\text{integer value}}{2^Q}$$

**Q16.16 format** (32-bit total, 16 integer bits, 16 fractional bits):
- Integer range: $-32768$ to $+32767$
- Fractional resolution: $1/2^{16} = 0.0000153$
- This gives ~4.5 decimal digits of precision

```
  32-bit integer: [SIIIIIIIIIIIIIII.FFFFFFFFFFFFFFFF]
                   ↑               ↑
                   sign bit        implied decimal point
                   
  Example: 3.14159 in Q16.16:
  3.14159 × 65536 = 205887.3 → store as integer 205887
  
  Verify: 205887 / 65536 = 3.14158... ✓ (error = 0.00001)
```

## 1.2 Fixed-Point Operations

```c
typedef int32_t q16_t;  // Q16.16 fixed-point

#define Q16_SHIFT  16
#define Q16_ONE    (1 << Q16_SHIFT)  // = 65536

// Convert float to Q16.16
static inline q16_t q16_from_float(float f) {
    return (q16_t)(f * Q16_ONE);
}

// Convert Q16.16 to float (for debugging/logging only)
static inline float q16_to_float(q16_t q) {
    return (float)q / Q16_ONE;
}

// Addition: same as integer addition (trivial)
static inline q16_t q16_add(q16_t a, q16_t b) {
    return a + b;
}

// Multiplication: must shift right by Q to correct the scale
static inline q16_t q16_mul(q16_t a, q16_t b) {
    // a × b gives Q32.32 (64-bit), shift right by 16 to get Q16.16
    return (q16_t)(((int64_t)a * (int64_t)b) >> Q16_SHIFT);
}

// Division: shift left first to preserve precision
static inline q16_t q16_div(q16_t a, q16_t b) {
    return (q16_t)(((int64_t)a << Q16_SHIFT) / b);
}
```

## 1.3 Why 64-bit Intermediates Are Essential

When you multiply two Q16.16 numbers:
- $a = 3.5$ → stored as $3.5 \times 65536 = 229376$
- $b = 2.0$ → stored as $2.0 \times 65536 = 131072$
- $a \times b = 229376 \times 131072 = 30064771072$ (exceeds 32-bit!)

**Without 64-bit intermediate:** overflow → garbage result → PID goes wild.
**With 64-bit intermediate:** $30064771072 >> 16 = 458752 = 7.0 \times 65536$ ✓

The ARM Cortex-M4 has a single-cycle 32×32→64 multiply instruction (`SMULL`), so this is essentially free.

## 1.4 Overflow Protection

```c
// Saturating addition (prevents wrap-around)
static inline q16_t q16_add_sat(q16_t a, q16_t b) {
    int64_t sum = (int64_t)a + (int64_t)b;
    if (sum > INT32_MAX) return INT32_MAX;
    if (sum < INT32_MIN) return INT32_MIN;
    return (q16_t)sum;
}
```

**Rule of thumb:** Use saturating arithmetic for the integrator (accumulated value grows unbounded). Use regular arithmetic for P and D terms (bounded by input range).

---

# PART 2 — Q-FORMAT SELECTION

## 2.1 Choosing the Right Q Format

The format determines your **range vs precision** trade-off:

| Format | Total bits | Integer bits | Frac bits | Range | Resolution |
|--------|-----------|-------------|-----------|-------|------------|
| Q8.8 | 16 | 8 | 8 | ±127 | 0.0039 |
| Q16.16 | 32 | 16 | 16 | ±32767 | 0.0000153 |
| Q8.24 | 32 | 8 | 24 | ±127 | 0.0000000596 |
| Q24.8 | 32 | 24 | 8 | ±8M | 0.0039 |
| Q1.31 | 32 | 1 | 31 | ±1.0 | 0.00000000047 |

## 2.2 AMR Motor Control Q-Format Choices

| Signal | Physical range | Needed resolution | Q format | Reasoning |
|--------|---------------|-------------------|----------|-----------|
| Motor current | ±10 A | 0.01 A | Q16.16 | 10 A fits in 16 int bits; 0.001 A resolution |
| Speed (rad/s) | ±200 rad/s | 0.01 rad/s | Q16.16 | 200 fits comfortably |
| Position (rad) | 0–2π × 10⁶ revs | 0.001 rad | Q24.8 | Large range for cumulative position |
| PID gains | 0.001–100 | 0.0001 | Q16.16 | Good balance |
| PWM duty | 0.0–1.0 | 0.0001 | Q1.15 (16-bit) | Exact mapping to timer compare register |

**Key insight:** You can use different Q formats for different signals, but you must **align** them before arithmetic (shift to match Q values).

## 2.3 Mixed Q-Format Arithmetic

When multiplying Q16.16 gain × Q16.16 error → result is Q32.32 (stored in 64-bit), then truncate to Q16.16 by shifting right 16.

When multiplying Q16.16 gain × Q24.8 position → result is Q40.24, shift right 8 to get Q16.16.

**General rule:** $Q_{result} = Q_a + Q_b$, then shift right by $(Q_a + Q_b - Q_{desired})$.

---

# PART 3 — FIXED-POINT PID IMPLEMENTATION

## 3.1 Velocity Form (Production-Ready)

```c
typedef struct {
    q16_t Kp;           // Q16.16
    q16_t Ki;           // Q16.16 (already includes Ts)
    q16_t Kd;           // Q16.16 (already includes 1/Ts)
    q16_t prev_error;   // Q16.16
    q16_t prev2_error;  // Q16.16
    q16_t output;       // Q16.16
    q16_t output_min;   // Q16.16
    q16_t output_max;   // Q16.16
} FixedPID;

void pid_init(FixedPID *pid, float kp, float ki, float kd, float ts,
              float out_min, float out_max) {
    pid->Kp = q16_from_float(kp);
    pid->Ki = q16_from_float(ki * ts);     // Pre-multiply by Ts
    pid->Kd = q16_from_float(kd / ts);     // Pre-multiply by 1/Ts
    pid->prev_error = 0;
    pid->prev2_error = 0;
    pid->output = 0;
    pid->output_min = q16_from_float(out_min);
    pid->output_max = q16_from_float(out_max);
}

q16_t pid_compute(FixedPID *pid, q16_t error) {
    // Velocity form: Δu = Kp*(e - e_prev) + Ki*e + Kd*(e - 2*e_prev + e_prev2)
    q16_t de = error - pid->prev_error;
    q16_t d2e = error - 2 * pid->prev_error + pid->prev2_error;
    
    q16_t delta_u = q16_mul(pid->Kp, de)
                  + q16_mul(pid->Ki, error)
                  + q16_mul(pid->Kd, d2e);
    
    pid->output = q16_add_sat(pid->output, delta_u);
    
    // Clamp (inherent anti-windup)
    if (pid->output > pid->output_max) pid->output = pid->output_max;
    if (pid->output < pid->output_min) pid->output = pid->output_min;
    
    // Shift history
    pid->prev2_error = pid->prev_error;
    pid->prev_error = error;
    
    return pid->output;
}
```

## 3.2 Pre-Multiplying Gains

Notice: `Ki = ki * ts` and `Kd = kd / ts` are computed once at initialization. This eliminates two multiplies per loop iteration. At 10 kHz on a Cortex-M0+ (no hardware multiplier), saving 2 multiplies = saving 40 µs.

**But:** If $T_s$ changes (e.g., adaptive rate), you must recalculate the gains. In AMR, $T_s$ is fixed, so pre-multiplication is safe.

## 3.3 Why Velocity Form Avoids Integrator Overflow

In positional form, the integral term accumulates forever:

$$I[n] = \sum_{k=0}^{n} e[k] \cdot T_s$$

If the error is 0.1 and the rate is 10 kHz, after just 1 minute:
$I = 0.1 \times 10000 \times 60 = 60000$ — this approaches Q16.16 overflow (32767 in integer part)!

In velocity form, the "integral" is implicit in the accumulated output, which is clamped every cycle. No unbounded accumulation.

---

# PART 4 — ISR STRUCTURE AND TIMING

## 4.1 The 10 kHz Timer ISR

```c
// Called by TIM1 update interrupt at 10 kHz
void TIM1_UP_IRQHandler(void) {
    // 1. Clear interrupt flag (MUST be first)
    __HAL_TIM_CLEAR_FLAG(&htim1, TIM_FLAG_UPDATE);
    
    // 2. Read current (ADC was triggered by timer, result ready via DMA)
    q16_t current_measured = adc_to_q16(adc_dma_buffer[0]);
    
    // 3. Read encoder
    int32_t enc_count = (int16_t)TIM3->CNT;
    static int32_t prev_enc = 0;
    int32_t enc_diff = enc_count - prev_enc;
    prev_enc = enc_count;
    
    // Handle 16-bit overflow
    if (enc_diff > 32767) enc_diff -= 65536;
    if (enc_diff < -32767) enc_diff += 65536;
    
    // 4. Compute speed (Q16.16, rad/s)
    q16_t speed = q16_mul(q16_from_float(RAD_PER_COUNT), 
                          (q16_t)(enc_diff << Q16_SHIFT));
    // Divide by Ts already baked into the constant
    
    // 5. Speed PID → current command
    q16_t speed_error = speed_setpoint - speed;
    q16_t current_cmd = pid_compute(&speed_pid, speed_error);
    
    // 6. Current PID → PWM duty
    q16_t current_error = current_cmd - current_measured;
    q16_t duty = pid_compute(&current_pid, current_error);
    
    // 7. Update PWM (convert Q16.16 duty to timer compare value)
    int32_t ccr = (int32_t)(((int64_t)duty * PWM_PERIOD) >> Q16_SHIFT);
    if (ccr < 0) ccr = 0;
    if (ccr > PWM_PERIOD) ccr = PWM_PERIOD;
    TIM1->CCR1 = (uint16_t)ccr;
    
    // 8. Update direction
    if (duty < 0) {
        GPIOA->BSRR = DIR_PIN;  // Set direction reverse
    } else {
        GPIOA->BSRR = DIR_PIN << 16;  // Reset direction
    }
    
    // 9. Feed watchdog
    IWDG->KR = 0xAAAA;
}
```

## 4.2 ISR Timing Rules

1. **No blocking:** Never wait for anything. No `while` loops, no `HAL_Delay`, no mutex locks.
2. **No dynamic allocation:** `malloc` in an ISR is a fatal error. Everything is pre-allocated.
3. **No floating point** (on MCUs without FPU, or when determinism is required).
4. **Minimize branches:** Branch misprediction on Cortex-M = 3-cycle penalty. Use arithmetic where possible.
5. **Constant execution time:** The ISR should take the same time whether error is 0 or 1000.

**Measuring ISR duration (toggle a GPIO):**

```c
void TIM1_UP_IRQHandler(void) {
    GPIOB->BSRR = DEBUG_PIN;   // Pin HIGH at ISR start
    
    // ... all PID computation ...
    
    GPIOB->BSRR = DEBUG_PIN << 16;  // Pin LOW at ISR end
}
// Measure the pulse width on an oscilloscope → ISR duration
```

---

# PART 5 — COMMON FIXED-POINT PITFALLS

## 5.1 The Deadly Pitfalls

### Pitfall 1: Forgetting 64-bit Intermediate

```c
// WRONG: 32-bit overflow
q16_t result = (a * b) >> 16;  // a*b overflows int32_t!

// RIGHT: 64-bit intermediate
q16_t result = (q16_t)(((int64_t)a * b) >> 16);
```

### Pitfall 2: Signed vs Unsigned Shift

```c
// WRONG: Arithmetic shift on unsigned type
uint32_t x = 0x80000000;
int32_t result = x >> 16;  // Implementation-defined! May not sign-extend

// RIGHT: Cast to signed first
int32_t result = (int32_t)x >> 16;
```

### Pitfall 3: Loss of Precision in Gain Pre-multiplication

If $K_i = 0.01$ and $T_s = 0.0001$:
$K_i \times T_s = 0.000001$

In Q16.16: $0.000001 \times 65536 = 0.065$ → stored as **0** (integer truncation!)

**Fix:** Use a higher Q format for gains (Q8.24) or scale the error instead.

### Pitfall 4: Integer Division Truncation

```c
// WRONG: loses fractional part
q16_t half = Q16_ONE / 3;  // = 65536 / 3 = 21845 (0.3333... truncated)

// This is actually fine for most cases (error < 0.00002)
// But beware cumulative truncation in filters
```

### Pitfall 5: Comparing Fixed-Point Across Different Q Formats

```c
// WRONG: comparing Q16.16 with Q8.24 directly
if (speed_q16 > threshold_q24)  // Meaningless comparison!

// RIGHT: convert to same Q format first
if (speed_q16 > (threshold_q24 >> 8))
```

## 5.2 Testing Fixed-Point: Back-to-Back with Float

The gold standard: run the float and fixed-point PID in parallel and compare outputs.

```python
# Python verification script
import struct

def q16_from_float(f):
    return int(f * 65536) & 0xFFFFFFFF  # Simulate 32-bit

def q16_mul(a, b):
    # Simulate 64-bit intermediate
    result = (a * b) >> 16
    # Clamp to 32-bit signed
    if result > 0x7FFFFFFF:
        result -= 0x100000000
    if result < -0x80000000:
        result += 0x100000000
    return result

def compare_pid(kp, ki, kd, ts, errors):
    """Run float and fixed-point PID, report max deviation."""
    # Float reference
    float_integral = 0.0
    float_prev_err = 0.0
    
    # Fixed-point
    fp_integral = 0
    fp_prev_err = 0
    fp_kp = q16_from_float(kp)
    fp_ki_ts = q16_from_float(ki * ts)
    fp_kd_ts = q16_from_float(kd / ts)
    
    max_deviation = 0.0
    
    for e in errors:
        # Float PID
        float_integral += e * ts
        float_deriv = (e - float_prev_err) / ts
        float_out = kp * e + ki * float_integral + kd * float_deriv
        float_prev_err = e
        
        # Fixed-point PID (positional form for comparison)
        fp_e = q16_from_float(e)
        fp_integral += q16_mul(fp_ki_ts, fp_e)
        fp_deriv = q16_mul(fp_kd_ts, fp_e - fp_prev_err)
        fp_out = q16_mul(fp_kp, fp_e) + fp_integral + fp_deriv
        fp_prev_err = fp_e
        
        # Compare
        fp_out_float = fp_out / 65536.0
        deviation = abs(float_out - fp_out_float)
        max_deviation = max(max_deviation, deviation)
    
    return max_deviation
```

---

# PART 6 — WHEN TO USE FLOAT ANYWAY

The Cortex-M4 FPU does single-precision float in **1 cycle** (multiply) or **14 cycles** (divide). If your MCU has an FPU:

| Criterion | Fixed-point | Float |
|-----------|-------------|-------|
| MCU has FPU | Use float | ✅ Use float |
| MCU has no FPU | ✅ Must use fixed | Can't (too slow) |
| Deterministic timing required | ✅ Fixed | Float (mostly deterministic on M4) |
| Porting to FPGA later | ✅ Fixed | Must convert |
| Team familiarity | Lower | ✅ Higher |

**AMR decision:** Float for speed/position PID on STM32F4 (has FPU). Fixed-point for line-sensor controller on STM32G0 (no FPU) and for any future FPGA offloading.

---

## Checkpoint Questions

1. What is Q16.16 format? What's the range and resolution?
2. Why does `q16_mul` need a 64-bit intermediate? Show an example that overflows without it.
3. Write a Q16.16 `q16_add_sat()` function with overflow protection.
4. $K_i = 0.001$, $T_s = 0.0001$. Can you pre-multiply $K_i \times T_s$ in Q16.16? What goes wrong?
5. Why does the velocity form avoid integrator overflow?
6. List three things you must NEVER do inside a 10 kHz timer ISR.
7. How would you verify that your fixed-point PID matches the float reference?

---

## Key Takeaways

- **Fixed-point** = integer arithmetic with an implied binary point. Use Q16.16 for most motor control.
- **64-bit intermediates** in multiplication are non-negotiable — overflow is silent and catastrophic
- **Velocity form PID** is preferred: inherent anti-windup, no unbounded accumulation
- **Pre-multiply gains by $T_s$** to save cycles, but check for precision loss
- **ISR rules:** No blocking, no malloc, no floats (without FPU), constant execution time
- **Test by comparing** fixed-point output to float reference — max deviation should be < 0.1%
