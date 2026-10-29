# 03 — PID Controller
### The workhorse of industrial control — and why 95% of all control loops use it

**Prerequisite:** `02-modeling-dc-motors.md` (motor transfer function, time constants)
**Unlocks:** `04-discrete-time-control.md` (you design in continuous, then discretize)

---

## Why Should I Care? (Practical Context)

Every motor on every warehouse robot uses PID. The line-sensor controller uses PID. The lifter position controller uses PID. Nav2's DWB uses a PID-like feedback law. When firmware tunes a new robot model, they're tuning PID gains.

When you see a robot oscillating, overshooting, or sluggishly following commands — it's a PID problem. This lesson teaches you to *predict* the behavior from the gains, *tune* the gains systematically, and *recognize* when PID isn't enough.

**Real AMR failures caused by PID issues:**
- Ticket Incident-B: Motor oscillation at 180 Hz — Kp set too high after firmware update
- Ticket Incident-C: Robot overshoot into rack after e-stop release — integral windup
- Ticket Incident-D: Sluggish trajectory following — Kp reduced to 40% of original during "safety fix"

---

# PART 1 — THE PID EQUATION

## 1.1 Continuous-Time PID

$$u(t) = K_p \cdot e(t) + K_i \int_0^t e(\tau)\,d\tau + K_d \cdot \frac{de(t)}{dt}$$

where:
- $e(t) = r(t) - y(t)$ = error (reference minus measurement)
- $u(t)$ = control output (PWM duty, voltage command)
- $K_p$ = proportional gain
- $K_i$ = integral gain
- $K_d$ = derivative gain

**Laplace form:**

$$C(s) = K_p + \frac{K_i}{s} + K_d s = K_p \left(1 + \frac{1}{T_i s} + T_d s\right)$$

where $T_i = K_p / K_i$ (integral time) and $T_d = K_d / K_p$ (derivative time).

## 1.2 What Each Term Does

### Proportional (P): The Spring

$$u_P = K_p \cdot e(t)$$

The output is proportional to the current error. Like a spring: the further from target, the harder you push.

**What it gives you:** Immediate response to error. Higher $K_p$ → faster response.
**What it costs you:** Higher $K_p$ → more overshoot, eventually oscillation. **Never** zero steady-state error alone (unless the plant already has an integrator).

**AMR intuition:** $K_p$ = "how aggressively do we correct speed error?" High $K_p$ → motor snaps to target speed quickly but may overshoot and ring. Low $K_p$ → motor slowly drifts toward target.

```
Response with increasing Kp:

  ω(t)
  │     Kp = 3.0 (oscillating)
  │    /╲  /╲
  │   /  ╲/  ╲───  Kp = 1.5 (overshoot + ring)
  │  /        ╱──── Kp = 0.8 (good)
  │ /      ╱─────── Kp = 0.3 (sluggish)
  │/    ╱
  └──────────────── t
  ω_ref ─ ─ ─ ─ ─ ─ ─ ─
```

### Integral (I): The Memory

$$u_I = K_i \int_0^t e(\tau)\,d\tau$$

Accumulates past error. Even a tiny persistent error eventually builds up a large control signal.

**What it gives you:** Eliminates steady-state error. If the motor runs 0.1 rad/s too slow, the integral keeps growing until the PWM increases enough to close the gap.
**What it costs you:** Slows down the response (adds phase lag). Can cause **integral windup** (see Part 3).

**AMR intuition:** $K_i$ = "how impatient are we about persistent error?" High $K_i$ → drives error to zero quickly but makes the system more oscillatory. Low $K_i$ → takes a long time to eliminate offset.

### Derivative (D): The Prediction

$$u_D = K_d \cdot \frac{de(t)}{dt}$$

Responds to the *rate of change* of error. If the error is decreasing fast (we're approaching the target quickly), it *reduces* the control output to prevent overshoot.

**What it gives you:** Reduced overshoot, improved stability. Acts as a damper.
**What it costs you:** Amplifies high-frequency noise. In practice, ALWAYS used with a low-pass filter (see 1.4).

**AMR intuition:** $K_d$ = "how much do we brake as we approach the target?" High $K_d$ → smooth approach, no overshoot, but slow. Low $K_d$ → less damping, may overshoot.

## 1.3 Combining the Terms

| Controller | Steady-state error | Overshoot | Oscillation risk | Use case |
|-----------|-------------------|-----------|-----------------|----------|
| P only | Yes (offset) | Moderate | Low | Quick regulation, don't need zero error |
| PI | Zero | Moderate→High | Medium | Most robot motor loops |
| PD | Yes (reduced) | Low | Low | Position loops where speed has integral |
| PID | Zero | Tunable | Medium→High | General purpose, need zero error + damping |

**robot motor speed loop:** Uses PI (no derivative — encoder noise makes D noisy at 10 kHz).
**robot motor current loop:** Uses P-only (fast response, steady-state error is absorbed by outer speed loop).

## 1.4 Derivative Filter (Practical D-Term)

Pure derivative amplifies noise. A high-frequency encoder glitch that lasts 1 µs produces a derivative spike of enormous magnitude. Solution: filter the derivative:

$$D_{filtered}(s) = \frac{K_d s}{1 + \frac{K_d}{N \cdot K_p} s} = \frac{K_d s}{\frac{\tau_f}{1} s + 1}$$

where $N$ = filter coefficient (typically 8–20). This limits the derivative gain at high frequencies to $N \cdot K_p$.

**Practical rule:** Use $N = 10$ unless you have a reason not to. This means the D-term acts as pure derivative up to $f = N \cdot K_p / (2\pi K_d)$, then rolls off.

## 1.5 Derivative Kick

When the reference $r(t)$ changes suddenly (step command), the error changes instantly, and $de/dt$ → ∞. The derivative term produces a huge spike ("derivative kick") that can damage actuators.

**Solution:** Differentiate the *measurement* instead of the *error*:

$$u_D = -K_d \cdot \frac{dy(t)}{dt} \quad \text{(derivative on measurement)}$$

The measurement $y(t)$ changes smoothly (motor can't teleport), so no spike. Many textbooks use $\frac{de}{dt}$, but production code uses $-\frac{dy}{dt}$.

**robot firmware uses derivative-on-measurement.** If you ever see derivative-on-error in embedded code, it's a bug.

---

# PART 2 — PID TUNING METHODS

## 2.1 Manual Tuning (The Practical Way)

1. **Set $K_i = 0$, $K_d = 0$**
2. **Increase $K_p$** until the output oscillates at a steady frequency. This is the **ultimate gain** $K_u$. The oscillation period is $T_u$.
3. **Back off $K_p$** to ~60% of $K_u$.
4. **Add $K_i$** starting small. Increase until steady-state error disappears without excessive overshoot.
5. **Add $K_d$** (if needed) to reduce overshoot. Start small.

**The feel-based approach used by robot firmware engineers:**

```
Start with Kp only:
  - "Motor barely moves" → Kp too low, 2x it
  - "Motor gets to speed but settles 5% below" → need Ki
  - "Motor overshoots and rings 3 times before settling" → Kp too high, reduce by 30%
  - "Motor buzzes at high frequency" → WAY too much gain, reduce by 10x

Add Ki:
  - "Offset gone but overshoot increased" → reduce Ki
  - "Still has offset after 1 second" → increase Ki
  - "Oscillation at low frequency (1-5 Hz)" → Ki too high

Add Kd (rarely needed for speed loops):
  - "Overshoot reduced but response is sluggish" → Kd too high
  - "Noise appears on the control signal" → Kd amplifying encoder noise, add filter
```

## 2.2 Ziegler-Nichols Method (The Textbook Way)

### Method 1: Ultimate Gain Method

1. Set $K_i = 0$, $K_d = 0$
2. Increase $K_p$ until sustained oscillation → $K_u$ (ultimate gain), $T_u$ (ultimate period)
3. Set gains from table:

| Controller | $K_p$ | $K_i$ | $K_d$ |
|-----------|-------|-------|-------|
| P | $0.50 \cdot K_u$ | — | — |
| PI | $0.45 \cdot K_u$ | $0.54 \cdot K_u / T_u$ | — |
| PID | $0.60 \cdot K_u$ | $1.2 \cdot K_u / T_u$ | $0.075 \cdot K_u \cdot T_u$ |

**Warning:** Ziegler-Nichols gives aggressive tuning (quarter-decay ratio — 25% overshoot). For AMR, you usually want less overshoot. Use the values as a starting point and reduce $K_p$ by 30–50%.

### Method 2: Step Response Method

Apply a step input to the *open-loop* plant. Measure:
- $L$ = time delay (from step to first response)
- $T$ = time constant (time from first response to 63% of final value)
- $K$ = steady-state gain

Then:

| Controller | $K_p$ | $K_i$ | $K_d$ |
|-----------|-------|-------|-------|
| P | $\frac{T}{K \cdot L}$ | — | — |
| PI | $\frac{0.9T}{K \cdot L}$ | $\frac{0.27T}{K \cdot L^2}$ | — |
| PID | $\frac{1.2T}{K \cdot L}$ | $\frac{0.6T}{K \cdot L^2}$ | $\frac{0.6T \cdot L}{K}$ |

## 2.3 Cohen-Coon Method (Better for Slow Processes)

If $L/T > 0.3$ (significant delay relative to time constant), Cohen-Coon gives better results than Ziegler-Nichols. Tables omitted for brevity — the point is: **no single tuning formula works for all systems**. Use them as starting points, then iterate.

## 2.4 Rule-of-Thumb Gains for Robot Motors

From firmware team's experience (these are starting points, not final values):

| Parameter | Speed loop (PI) | Current loop (P) | Position loop (PID) |
|-----------|----------------|------------------|-------------------|
| $K_p$ | 0.5 – 2.0 | 0.1 – 0.5 | 5.0 – 20.0 |
| $K_i$ | 10 – 100 | 0 (or very small) | 0.5 – 5.0 |
| $K_d$ | 0 | 0 | 0.01 – 0.1 |
| Sample rate | 10 kHz | 20 kHz | 1 kHz |

---

# PART 3 — ANTI-WINDUP (THE #1 PID PITFALL)

## 3.1 What Is Integral Windup?

When the control output **saturates** (PWM at 100%, current at maximum), the error persists but the controller can't increase the output further. However, the integral term *keeps accumulating*. When the saturation condition ends, the accumulated integral causes a massive overshoot.

**The scenario (AMR ticket Incident-C):**

```
Timeline:
  t=0:    Robot moving at 0.3 m/s
  t=1:    Robot hits obstacle, e-stop activates, PWM → 0
  t=1-5:  E-stop active. Speed command still 0.3 m/s. Error = 0.3 - 0 = 0.3 m/s.
          Integral accumulates: 0.3 × 4 seconds = 1.2 m/s·s
  t=5:    E-stop released. Integral is huge.
          Controller output = Kp × 0.3 + Ki × 1.2 = massive
          PWM goes to 100%.
          Motor surges forward → robot overshoots into rack.
```

## 3.2 Anti-Windup Strategies

### Strategy 1: Output Clamping (Simple but Incomplete)

```c
float integral = 0;
float pid_compute(float error, float dt) {
    integral += error * dt;
    
    // Clamp integral
    if (integral > INTEGRAL_MAX) integral = INTEGRAL_MAX;
    if (integral < INTEGRAL_MIN) integral = INTEGRAL_MIN;
    
    float output = Kp * error + Ki * integral + Kd * deriv;
    
    // Clamp output
    if (output > OUTPUT_MAX) output = OUTPUT_MAX;
    if (output < OUTPUT_MIN) output = OUTPUT_MIN;
    
    return output;
}
```

**Problem:** The clamp values are arbitrary. Too tight → integral doesn't help. Too loose → windup still happens.

### Strategy 2: Conditional Integration (Better)

Only integrate when the output is NOT saturated:

```c
float pid_compute(float error, float dt) {
    float p_term = Kp * error;
    float d_term = Kd * (error - prev_error) / dt;
    
    float output_before_sat = p_term + Ki * integral + d_term;
    
    // Only integrate if not saturated OR error is reducing saturation
    bool saturated_high = output_before_sat > OUTPUT_MAX;
    bool saturated_low  = output_before_sat < OUTPUT_MIN;
    bool error_reducing = (saturated_high && error < 0) || 
                          (saturated_low && error > 0);
    
    if (!saturated_high && !saturated_low || error_reducing) {
        integral += error * dt;
    }
    
    float output = p_term + Ki * integral + d_term;
    return CLAMP(output, OUTPUT_MIN, OUTPUT_MAX);
}
```

### Strategy 3: Back-Calculation (The Professional Way)

Feed the saturation error back into the integrator:

$$\frac{d}{dt} \int e = e(t) + \frac{1}{T_t} (u_{saturated} - u_{unsaturated})$$

where $T_t$ is the tracking time constant (often set to $T_t = \sqrt{T_i \cdot T_d}$ or $T_t = T_d$).

```c
float pid_compute(float error, float dt) {
    float p_term = Kp * error;
    float d_term = Kd * derivative;
    float output_unsat = p_term + Ki * integral + d_term;
    float output_sat = CLAMP(output_unsat, OUTPUT_MIN, OUTPUT_MAX);
    
    // Back-calculation: reduce integral by saturation amount
    float sat_error = output_sat - output_unsat;
    integral += (error + sat_error / Tt) * dt;
    
    return output_sat;
}
```

**robot firmware uses back-calculation** for the speed PID loop. The current loop uses simple clamping since it rarely saturates (the speed loop limits the current command).

### Strategy 4: Reset on E-Stop

For AMR specifically, the simplest additional protection:

```c
void on_estop_release(void) {
    pid.integral = 0;  // Reset integral to prevent windup surge
    // Optionally: ramp up the setpoint instead of step
}
```

---

# PART 4 — PID ANALYSIS WITH MATH

## 4.1 Closed-Loop with PI Controller

Plant: $G(s) = \frac{K_m}{\tau_m s + 1}$ (motor speed)

Controller: $C(s) = K_p + \frac{K_i}{s} = \frac{K_p s + K_i}{s}$

Open-loop: $L(s) = C(s) \cdot G(s) = \frac{K_m(K_p s + K_i)}{s(\tau_m s + 1)}$

Closed-loop:

$$T(s) = \frac{L(s)}{1 + L(s)} = \frac{K_m(K_p s + K_i)}{s(\tau_m s + 1) + K_m(K_p s + K_i)}$$

$$= \frac{K_m(K_p s + K_i)}{\tau_m s^2 + (1 + K_m K_p) s + K_m K_i}$$

**Matching to standard second-order form** $\frac{\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2}$:

$$\omega_n = \sqrt{\frac{K_m K_i}{\tau_m}}, \quad \zeta = \frac{1 + K_m K_p}{2\sqrt{\tau_m K_m K_i}}$$

**This is powerful!** You can directly compute the natural frequency and damping ratio from the PID gains and motor parameters:

- Increasing $K_i$ → increases $\omega_n$ (faster) but decreases $\zeta$ (more oscillatory)
- Increasing $K_p$ → increases $\zeta$ (more damped) but also changes the zero location

**Design target:** $\zeta = 0.7$ (5% overshoot, fast settling). Solve for $K_p$ and $K_i$:

$$K_i = \frac{\omega_n^2 \tau_m}{K_m}, \quad K_p = \frac{2\zeta\omega_n\tau_m - 1}{K_m}$$

## 4.2 Numerical Example (AMR Motor)

Motor: $K_m = 10$ rad/s/V, $\tau_m = 0.1$ s

Target: $\omega_n = 50$ rad/s, $\zeta = 0.7$

$$K_i = \frac{50^2 \times 0.1}{10} = 25 \quad [\text{V/rad}]$$

$$K_p = \frac{2 \times 0.7 \times 50 \times 0.1 - 1}{10} = \frac{7 - 1}{10} = 0.6 \quad [\text{V·s/rad}]$$

**Verification in Python:**

```python
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

Km, tau_m = 10.0, 0.1
Kp, Ki = 0.6, 25.0

# Closed-loop TF
num_cl = [Km * Kp, Km * Ki]
den_cl = [tau_m, 1 + Km * Kp, Km * Ki]

sys_cl = signal.TransferFunction(num_cl, den_cl)
t = np.linspace(0, 0.2, 1000)
t_out, y_out = signal.step(sys_cl, T=t)

plt.plot(t_out * 1000, y_out)
plt.xlabel('Time (ms)')
plt.ylabel('Speed (normalized)')
plt.title(f'Closed-loop step response (Kp={Kp}, Ki={Ki})')
plt.axhline(y=1.0, color='r', linestyle='--', label='Reference')
plt.grid(True)
plt.legend()
plt.show()
```

Expected: ~5% overshoot, settling in ~60 ms. If you see more overshoot → reduce $K_i$. If too slow → increase $\omega_n$.

---

# PART 5 — PRACTICAL PID CONSIDERATIONS

## 5.1 Bumpless Transfer

When switching between manual and automatic mode (or between different PID parameter sets), the integral term must be initialized to the current control output. Otherwise, there's a sudden jump in output ("bump"):

```c
void switch_to_auto(float current_output) {
    pid.integral = (current_output - Kp * error) / Ki;
}
```

## 5.2 Output Rate Limiting

Even with good tuning, you may want to limit how fast the output changes to prevent mechanical stress:

```c
float rate_limited_output(float new_output, float prev_output, float max_rate, float dt) {
    float delta = new_output - prev_output;
    float max_delta = max_rate * dt;
    if (delta > max_delta) delta = max_delta;
    if (delta < -max_delta) delta = -max_delta;
    return prev_output + delta;
}
```

## 5.3 Setpoint Weighting

Sometimes you want the P and D terms to respond differently to setpoint changes vs disturbances. **Setpoint weighting** applies factors $b$ and $c$:

$$u = K_p(b \cdot r - y) + K_i \int(r - y)dt + K_d \frac{d(c \cdot r - y)}{dt}$$

- $b = 0$: P-term only responds to measurement (no setpoint overshoot). **Common in our robot.**
- $b = 1$: Normal PID (responds to setpoint changes)
- $c = 0$: D-term on measurement only (eliminates derivative kick)

---

## Checkpoint Questions

1. Write the PID equation in both time domain and Laplace domain.
2. A PI controller eliminates steady-state error. Where does the "extra" torque come from physically?
3. You increase $K_p$ and the motor starts oscillating at 200 Hz. What happened in terms of poles?
4. Explain integral windup in 3 sentences. Why does back-calculation fix it?
5. Ziegler-Nichols ultimate gain method: you find $K_u = 2.0$, $T_u = 0.05$ s. What are the ZN-recommended PID gains?
6. Why does robot firmware use derivative-on-measurement instead of derivative-on-error?
7. Design a PI controller for a motor with $K_m = 8$ rad/s/V, $\tau_m = 0.15$ s, targeting $\zeta = 0.7$, $\omega_n = 40$ rad/s.

---

## Key Takeaways

- PID = Proportional (spring) + Integral (memory) + Derivative (prediction)
- **Integral** eliminates steady-state error but risks windup; **derivative** reduces overshoot but amplifies noise
- **Anti-windup is non-optional** in any real system with actuator limits
- Use **derivative-on-measurement** (not error) to avoid derivative kick
- Ziegler-Nichols is a starting point, not a final answer — always verify on the real system
- For a first-order plant + PI controller, you can **analytically compute** $K_p$ and $K_i$ from desired $\omega_n$ and $\zeta$
