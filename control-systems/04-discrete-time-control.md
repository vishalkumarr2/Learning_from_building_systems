# 04 — Discrete-Time Control
### Because microcontrollers don't do calculus — they do loops

**Prerequisite:** `03-pid-controller.md` (continuous PID, tuning)
**Unlocks:** `05-mcu-motor-control.md` (hardware that executes the discrete PID)

---

## Why Should I Care? (OKS Context)

The PID equations in Lesson 03 are *continuous-time* — they assume infinitely fast computation with infinitely precise math. Real controllers run on a microcontroller that:

1. Samples the encoder at **discrete intervals** (every 100 µs = 10 kHz)
2. Computes using **finite-precision arithmetic** (32-bit float or fixed-point)
3. Updates the PWM at the **same discrete rate**

The gap between continuous theory and discrete practice causes real bugs:
- **Aliasing:** Sample too slowly and fast disturbances appear as slow ones
- **Quantization:** Encoder resolution limits the minimum detectable speed
- **Integration method:** Forward Euler vs. Tustin bilinear transform changes the stability boundary

**OKS example:** The STM32 motor control runs at 10 kHz. At this rate, the electrical time constant (0.5 ms = 2 kHz bandwidth) is adequately sampled. But if someone reduces the rate to 1 kHz (100 µs → 1 ms) "to save CPU," the current loop becomes unstable because the electrical dynamics aren't sampled fast enough.

---

# PART 1 — SAMPLING AND ALIASING

## 1.1 The Sampling Theorem (Shannon/Nyquist)

To faithfully capture a signal of frequency $f$, you must sample at rate $f_s \geq 2f$.

**For control systems, the rule is stricter:**

$$f_s \geq 10 \times f_{BW}$$

where $f_{BW}$ is the closed-loop bandwidth. This is because:
- Shannon says 2× is the *minimum* to avoid aliasing
- But a controller needs to *react* to the signal, not just record it
- 10× gives enough samples per cycle for the controller to effectively act

**OKS motor control sampling rates:**

| Loop | Bandwidth | Min sample rate (10×) | Actual OKS rate | Margin |
|------|-----------|----------------------|-----------------|--------|
| Current | ~2 kHz | 20 kHz | 20 kHz | 1× (tight!) |
| Speed | ~50 Hz | 500 Hz | 10 kHz | 20× (comfortable) |
| Position | ~5 Hz | 50 Hz | 1 kHz | 20× (comfortable) |
| Nav2 cmd_vel | ~2 Hz | 20 Hz | 50 Hz | 2.5× (adequate) |

## 1.2 What Aliasing Looks Like in Control

If you sample a 300 Hz vibration at 500 Hz, it appears as a 200 Hz signal (500 - 300 = 200). The controller "sees" a phantom 200 Hz disturbance and tries to reject it — making things worse.

**OKS example:** Motor cogging at 200 Hz + speed loop at 10 kHz = no problem (10000 >> 2×200). But if the SPI bus delays speed feedback to an effective 300 Hz rate, the cogging can alias into the controller's useful bandwidth.

**Solution:** Anti-alias filter before sampling. A simple first-order low-pass filter:

$$H(s) = \frac{1}{\tau_f s + 1}, \quad \tau_f = \frac{1}{2\pi f_{cutoff}}$$

Set $f_{cutoff} = f_s / 4$ to prevent aliasing while preserving useful bandwidth.

---

# PART 2 — DISCRETIZATION METHODS

## 2.1 The Problem

The continuous PID controller:

$$C(s) = K_p + \frac{K_i}{s} + K_d s$$

Must be converted to a discrete-time difference equation that a microcontroller can execute every $T_s$ seconds.

The question: **How do you approximate the continuous integral and derivative?**

## 2.2 Forward Euler (Simplest, Least Accurate)

Replace: $s \approx \frac{z - 1}{T_s}$

**Integral:**

$$\int e \,dt \approx \sum_{k=0}^{n} e[k] \cdot T_s$$

or equivalently: $I[n] = I[n-1] + T_s \cdot e[n]$

**Derivative:**

$$\frac{de}{dt} \approx \frac{e[n] - e[n-1]}{T_s}$$

**Full discrete PID (Forward Euler):**

```c
// Positional form
float pid_compute(float error, float dt) {
    integral += error * dt;                          // Forward Euler integration
    float derivative = (error - prev_error) / dt;    // Forward Euler derivative
    prev_error = error;
    
    return Kp * error + Ki * integral + Kd * derivative;
}
```

**Pros:** Dead simple, one multiply-add per term.
**Cons:** Can be unstable for fast dynamics. The stability region of Forward Euler is a circle of radius 1 centered at (-1, 0) in the z-plane. If the motor's poles lie outside this circle at your sample rate, the discrete controller is unstable even though the continuous one is stable.

## 2.3 Backward Euler (More Stable)

Replace: $s \approx \frac{z - 1}{T_s \cdot z}$

**Integral:**

$$I[n] = I[n-1] + T_s \cdot e[n]$$

(Same equation, but it's derived differently and has better stability properties.)

**Derivative:**

$$D[n] = \frac{e[n] - e[n-1]}{T_s}$$

Backward Euler is **unconditionally stable** for stable continuous systems — it never introduces instability through discretization. But it's more damped than the continuous system (slightly sluggish).

## 2.4 Bilinear Transform / Tustin's Method (Best for Most Cases)

Replace: $s \approx \frac{2}{T_s} \cdot \frac{z - 1}{z + 1}$

This maps the left half of the s-plane exactly to the interior of the unit circle in the z-plane. **Stability is preserved perfectly.**

**Tustin integral:**

$$I[n] = I[n-1] + \frac{T_s}{2}(e[n] + e[n-1])$$

This is the **trapezoidal rule** — average of current and previous error × time step. Much more accurate than Forward Euler for the same sample rate.

**Tustin derivative (with filter):**

$$D[n] = \frac{2K_d}{2\tau_f + T_s} (e[n] - e[n-1]) + \frac{2\tau_f - T_s}{2\tau_f + T_s} D[n-1]$$

where $\tau_f = K_d / (N \cdot K_p)$ is the derivative filter time constant.

**Full Tustin PID:**

```c
float pid_compute_tustin(float error) {
    // Proportional
    float P = Kp * error;
    
    // Integral (trapezoidal)
    integral += 0.5f * Ki * Ts * (error + prev_error);
    
    // Derivative (filtered, Tustin)
    float c1 = 2.0f * Kd / (2.0f * tau_f + Ts);
    float c2 = (2.0f * tau_f - Ts) / (2.0f * tau_f + Ts);
    deriv = c1 * (error - prev_error) + c2 * prev_deriv;
    
    prev_error = error;
    prev_deriv = deriv;
    
    return P + integral + deriv;
}
```

**OKS firmware uses Tustin for the speed loop** because it's the best accuracy-to-cost ratio. The current loop uses Forward Euler because at 20 kHz, even Forward Euler is accurate enough (the electrical time constant is only ~1 ms = 20 samples).

## 2.5 Comparison of Methods

| Method | Stability | Accuracy | Computation | When to use |
|--------|-----------|----------|-------------|-------------|
| Forward Euler | Can destabilize | Low | 1 multiply + 1 add | Very fast sample rate (>20× bandwidth) |
| Backward Euler | Always stable | Low-Medium | 1 multiply + 1 add | Safety-critical, slow sample rate |
| Tustin (Bilinear) | Preserves stability | High | 2 multiplies + 2 adds | Default choice for motor control |
| ZOH (exact) | Preserves stability | Exact | Matrix exponential | Simulation only (too expensive for MCU) |

---

# PART 3 — VELOCITY FORM PID

## 3.1 Positional vs Velocity Form

The **positional form** computes the absolute output:

$$u[n] = K_p e[n] + K_i \sum e[k] T_s + K_d \frac{e[n] - e[n-1]}{T_s}$$

The **velocity form** computes the *change* in output:

$$\Delta u[n] = u[n] - u[n-1] = K_p(e[n] - e[n-1]) + K_i T_s \cdot e[n] + K_d \frac{e[n] - 2e[n-1] + e[n-2]}{T_s}$$

Then: $u[n] = u[n-1] + \Delta u[n]$

## 3.2 Why Velocity Form Is Better for Embedded

1. **No integral windup by construction:** The integral is implicit in $u[n-1]$. Clamping $u[n]$ automatically limits the integral.
2. **Bumpless transfer:** Switching to manual mode just means stopping the $\Delta u$ updates. No accumulated integral to manage.
3. **Simpler anti-windup:** Just clamp $u[n]$ to $[u_{min}, u_{max}]$ after each step.

**OKS firmware velocity-form PID:**

```c
typedef struct {
    float Kp, Ki, Kd;
    float Ts;              // Sample period
    float prev_error;
    float prev_prev_error;
    float output;          // Accumulated output
    float output_min, output_max;
} PID_VelocityForm;

float pid_velocity_compute(PID_VelocityForm *pid, float error) {
    float delta_u = pid->Kp * (error - pid->prev_error)
                  + pid->Ki * pid->Ts * error
                  + pid->Kd / pid->Ts * (error - 2.0f * pid->prev_error + pid->prev_prev_error);
    
    pid->output += delta_u;
    
    // Clamp (inherent anti-windup)
    if (pid->output > pid->output_max) pid->output = pid->output_max;
    if (pid->output < pid->output_min) pid->output = pid->output_min;
    
    pid->prev_prev_error = pid->prev_error;
    pid->prev_error = error;
    
    return pid->output;
}
```

---

# PART 4 — QUANTIZATION AND RESOLUTION

## 4.1 Encoder Quantization

A quadrature encoder with $N$ counts per revolution gives:

$$\Delta\theta_{min} = \frac{2\pi}{4N} \text{ rad (with 4× decoding)}$$

**Speed from encoder** at sample rate $f_s$:

$$\omega[n] = \frac{\theta[n] - \theta[n-1]}{T_s} = \frac{\Delta\theta \cdot \text{count\_diff}}{T_s}$$

**Minimum detectable speed:**

$$\omega_{min} = \frac{2\pi}{4N \cdot T_s}$$

**OKS encoder:** $N = 512$ counts/rev, 4× decoding = 2048 edges/rev, $f_s = 10$ kHz:

$$\omega_{min} = \frac{2\pi}{2048 \times 0.0001} = 30.7 \text{ rad/s} = 293 \text{ RPM}$$

That's a terrible resolution for low-speed control! At the wheel (after 50:1 gearbox), the minimum detectable speed is 30.7/50 = 0.61 rad/s. For the OKS wheel (radius ~0.065 m), that's 0.04 m/s.

## 4.2 Improving Speed Resolution

### Method 1: Reduce sample rate for speed calculation

Instead of computing speed every 100 µs (10 kHz), compute it every 1 ms (1 kHz) but still run the PID at 10 kHz with the most recent speed estimate.

$$\omega_{min} \text{@ 1 kHz} = \frac{2\pi}{2048 \times 0.001} = 3.07 \text{ rad/s}$$

10× better resolution, but 10× more delay in the speed estimate.

### Method 2: Moving average filter

Average the count differences over $M$ samples:

$$\omega[n] = \frac{1}{M \cdot T_s} \sum_{k=0}^{M-1} (\theta[n-k] - \theta[n-k-1])$$

Equivalently: $\omega[n] = \frac{\theta[n] - \theta[n-M]}{M \cdot T_s}$

This is a FIR low-pass filter. $M = 10$ gives 10× better resolution with only $M \cdot T_s / 2 = 0.5$ ms group delay.

### Method 3: Period measurement

Instead of counting edges per time, measure time per edge. At low speed, this gives much better resolution. At high speed, revert to counting.

**OKS firmware uses a hybrid:** Edge counting at high speed (>100 RPM) and period measurement at low speed (<100 RPM), with crossover blending.

## 4.3 ADC Quantization (Current Sensing)

The current sensor's ADC has $n$-bit resolution:

$$\Delta I_{min} = \frac{I_{range}}{2^n}$$

For OKS: 12-bit ADC, ±10A range → $\Delta I = 20/4096 = 4.9$ mA per LSB. This is adequate for current control.

---

# PART 5 — SAMPLE RATE SELECTION GUIDE

## 5.1 The Trade-off

| Higher sample rate | Lower sample rate |
|---|---|
| ✅ Better accuracy | ✅ Less CPU load |
| ✅ More stability margin | ✅ More time for computation |
| ✅ Less aliasing | ✅ Better encoder resolution |
| ❌ More CPU load | ❌ Risk of aliasing |
| ❌ Worse encoder resolution (count diff = 0 or 1) | ❌ Risk of instability |

## 5.2 Decision Framework

```
1. Determine the fastest dynamic you need to control
   → OKS: electrical time constant = 0.5 ms → f_elec = 2 kHz

2. Multiply by 10-20× for the minimum sample rate
   → f_s_min = 20 kHz for current loop
   → f_s_min = 500 Hz for speed loop (bandwidth ~50 Hz)

3. Check encoder resolution at that rate
   → 20 kHz with 2048 CPR = ω_min = 61 rad/s (too coarse for speed)
   → Use period measurement or lower-rate speed computation

4. Check CPU budget
   → STM32F4 @ 168 MHz: ~8400 cycles per 20 kHz interrupt
   → PID computation: ~50-100 cycles
   → Plenty of headroom

5. Choose the highest rate your CPU can sustain
   → Current loop: 20 kHz
   → Speed loop: 10 kHz (or 1 kHz speed estimate feeding 10 kHz PID)
```

---

## Checkpoint Questions

1. Why is the control engineering rule $f_s \geq 10 \times f_{BW}$ stricter than Shannon's $f_s \geq 2f$?
2. Forward Euler discretization of the integral term gives `I[n] = I[n-1] + Ts * e[n]`. Write the Tustin version.
3. A motor's electrical time constant is 0.5 ms. What is the minimum acceptable sample rate for the current loop?
4. Explain why the velocity form of PID has inherent anti-windup.
5. An encoder has 1024 CPR with 4× decoding. At 10 kHz sampling, what is the minimum detectable speed in RPM?
6. Why would reducing the PID loop from 10 kHz to 1 kHz improve encoder-based speed resolution?
7. The current loop is stable at 20 kHz but oscillates when reduced to 5 kHz. Where in the z-plane did the poles go?

---

## Key Takeaways

- **Discretization method matters:** Tustin (bilinear) preserves stability; Forward Euler can destabilize
- **Velocity form PID** is preferred in embedded systems for inherent anti-windup and bumpless transfer
- **Sample rate must be ≥ 10× bandwidth** — not just 2× (Shannon is for recording, not controlling)
- **Encoder quantization** limits low-speed resolution; use period measurement or averaging to compensate
- **The choice of sample rate is a 4-way trade-off:** accuracy vs CPU vs encoder resolution vs computation time
