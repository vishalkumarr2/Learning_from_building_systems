# 07 — Cascaded Control Loops
### Current inside speed inside position — three PIDs, one motor

**Prerequisite:** `06-fixed-point-pid.md` (fixed-point PID), `05-mcu-motor-control.md` (hardware)
**Unlocks:** `08-trajectory-tracking.md` (what feeds the position/speed commands)

---

## Why Should I Care? (OKS Context)

A single PID controlling motor speed from voltage is *okay* for a toy. For an OKS warehouse robot carrying 200 kg at 1 m/s, you need **cascaded loops**: an inner current (torque) loop, a middle speed loop, and an outer position loop. Each loop handles a different timescale and gives you independent control over torque, speed, and position.

**Why cascade matters in practice:**
- Without current limiting → motor stall at startup = 15A peak → trips the battery BMS → e-stop
- Without speed limiting → acceleration spike → load slides off the shelf
- Without position loop → can't hold position at a station (drift on slopes)

**OKS cascade architecture:**

```
Nav2 → position_cmd (50 Hz)
          ↓
     Position PID (1 kHz)
          ↓ speed_cmd
     Speed PID (10 kHz)
          ↓ current_cmd
     Current PID (20 kHz)
          ↓ duty
     PWM → H-Bridge → Motor
```

---

# PART 1 — WHY CASCADE?

## 1.1 The Problem with a Single Loop

A single PID from position to PWM duty:

$$u = K_p(r - \theta) + K_i \int(r - \theta)dt + K_d \frac{d(r - \theta)}{dt}$$

**Problems:**
1. **Can't limit current:** The PID output is a duty cycle. If the position error is large, the duty goes to 100%, drawing unlimited current.
2. **Slow response to disturbances:** If the load suddenly increases, the motor slows. The position PID notices... eventually... after the position has drifted. By then, the error is large and the response is aggressive.
3. **Can't independently tune torque, speed, and position:** You have 3 gains but 3 *different* dynamic behaviors to control.

## 1.2 The Cascade Solution

Each loop controls one physical quantity and limits it:

| Loop | Controls | Limits | Rate | Bandwidth |
|------|----------|--------|------|-----------|
| **Current (inner)** | Motor current → torque | Max current (e.g., 5A) | 20 kHz | ~2 kHz |
| **Speed (middle)** | Motor speed | Max speed (e.g., 100 RPM at wheel) | 10 kHz | ~50 Hz |
| **Position (outer)** | Wheel/robot position | Max speed command | 1 kHz | ~5 Hz |

**The 10:1 bandwidth rule:** Each outer loop should have bandwidth ≤ 1/10 of the inner loop. This ensures the inner loop "looks like a gain of 1" to the outer loop (it's so fast that it perfectly tracks its setpoint).

$$f_{BW,outer} \leq \frac{1}{10} f_{BW,inner}$$

---

# PART 2 — THE CURRENT (TORQUE) LOOP

## 2.1 Purpose

The current loop:
1. Makes the motor **behave like a torque source** (command current → get torque)
2. Protects against **overcurrent** (hard-limits current even if speed PID demands more)
3. Rejects electrical disturbances (back-EMF changes) **fast**

## 2.2 Plant Model

The motor's electrical equation:

$$V = Ri + L\frac{di}{dt} + K_e \omega$$

From the current loop's perspective, $K_e \omega$ is a disturbance (back-EMF). The "plant" is:

$$G_i(s) = \frac{I(s)}{V(s)} = \frac{1/R}{\frac{L}{R}s + 1} = \frac{1/R}{\tau_e s + 1}$$

where $\tau_e = L/R$ is the electrical time constant (~0.5 ms for OKS motor).

## 2.3 PI Controller (No D)

The current loop uses **PI only** (no derivative) because:
- The electrical dynamics are fast and noisy
- Derivative amplifies current sensor noise
- PI is sufficient for first-order plant + disturbance rejection

$$C_i(s) = K_{p,i} + \frac{K_{i,i}}{s}$$

**Tuning by pole placement:** Place the closed-loop pole at $f_{cl} = 2$ kHz ($\omega_{cl} = 4000\pi$ rad/s):

$$K_{p,i} = L \cdot \omega_{cl} = 0.001 \times 4000\pi = 12.57$$

$$K_{i,i} = R \cdot \omega_{cl} = 1.0 \times 4000\pi = 12566$$

**This cancels the plant pole** (zero of PI at $R/L$ cancels pole at $R/L$), leaving a first-order closed-loop with bandwidth $\omega_{cl}$.

## 2.4 Implementation

```c
// Current PID — runs at 20 kHz
// PI only, velocity form
typedef struct {
    q16_t Kp;
    q16_t Ki_Ts;        // Ki × Ts pre-multiplied
    q16_t prev_error;
    q16_t output;
    q16_t output_min;   // = -MAX_DUTY (Q16.16)
    q16_t output_max;   // = +MAX_DUTY (Q16.16)
} CurrentPI;

q16_t current_pi_compute(CurrentPI *pi, q16_t current_cmd, q16_t current_meas) {
    q16_t error = current_cmd - current_meas;
    
    // Velocity form: Δu = Kp*(e - e_prev) + Ki*Ts*e
    q16_t delta_u = q16_mul(pi->Kp, error - pi->prev_error)
                  + q16_mul(pi->Ki_Ts, error);
    
    pi->output = q16_add_sat(pi->output, delta_u);
    
    // Clamp to valid duty cycle
    if (pi->output > pi->output_max) pi->output = pi->output_max;
    if (pi->output < pi->output_min) pi->output = pi->output_min;
    
    pi->prev_error = error;
    return pi->output;  // This is the PWM duty cycle
}
```

---

# PART 3 — THE SPEED LOOP

## 3.1 Purpose

The speed loop:
1. Makes the motor **behave like a speed source** (command speed → get speed)
2. Limits acceleration (rate-limits the current command)
3. Compensates for load torque disturbances

## 3.2 Plant Model

From the speed loop's perspective, the current loop is fast enough to be "transparent" (gain ≈ 1). The "plant" is the motor's mechanical dynamics:

$$G_\omega(s) = \frac{\Omega(s)}{T(s)} = \frac{1/B}{Js/B + 1} = \frac{1/B}{\tau_m s + 1}$$

where $\tau_m = J/B$ is the mechanical time constant (~50 ms for OKS with gearbox).

But since the current loop output → torque: $T = K_t \times i$, the combined plant is:

$$G_\omega(s) = \frac{K_t}{Js + B}$$

## 3.3 PI Controller

Again PI only, because:
- Speed from encoder is noisy (quantization)
- D term amplifies encoder noise → duty cycle jitter → audible motor whine

**Tuning:** Bandwidth ≤ 1/10 of current loop = 200 Hz max. Target: ~50 Hz.

$$K_{p,\omega} = \frac{J \cdot \omega_{BW}}{K_t} = \frac{0.001 \times 100\pi}{0.05} = 6.28$$

$$K_{i,\omega} = \frac{B \cdot \omega_{BW}}{K_t} = \frac{0.002 \times 100\pi}{0.05} = 12.57$$

## 3.4 Output of Speed PID = Current Setpoint

```c
// Speed PID — conceptually at 10 kHz (or 1 kHz with feedforward)
q16_t speed_pid_compute(SpeedPID *pid, q16_t speed_cmd, q16_t speed_meas) {
    q16_t error = speed_cmd - speed_meas;
    
    // Standard velocity-form PI
    q16_t delta_u = q16_mul(pid->Kp, error - pid->prev_error)
                  + q16_mul(pid->Ki_Ts, error);
    
    pid->output = q16_add_sat(pid->output, delta_u);
    
    // Clamp to MAX CURRENT (this is the safety limit!)
    if (pid->output > pid->current_limit_pos) pid->output = pid->current_limit_pos;
    if (pid->output < pid->current_limit_neg) pid->output = pid->current_limit_neg;
    
    pid->prev_error = error;
    return pid->output;  // This is the current setpoint for the inner loop
}
```

**Critical point:** The speed PID output is clamped to the **maximum allowed current**, not the maximum PWM duty. This is how cascade provides current limiting — the speed loop can never ask for more current than the motor/driver can safely handle.

---

# PART 4 — THE POSITION LOOP

## 4.1 Purpose

The position loop:
1. Tracks a position trajectory (for docking, station alignment)
2. Holds position against external disturbances (slopes, impacts)
3. Limits speed (the position PID output = speed command)

## 4.2 Plant Model

From the position loop's perspective, the speed loop is a perfect speed source (gain ≈ 1). The "plant" is just integration:

$$\frac{\Theta(s)}{\Omega(s)} = \frac{1}{s}$$

An integrator — infinite DC gain already, so:

## 4.3 P Controller (Often Sufficient!)

For a pure integrator plant, a simple proportional controller gives first-order closed-loop response:

$$C_{pos}(s) = K_{p,pos}$$

$$G_{cl}(s) = \frac{K_{p,pos}/s}{1 + K_{p,pos}/s} = \frac{K_{p,pos}}{s + K_{p,pos}}$$

Bandwidth = $K_{p,pos}$ rad/s. For 5 Hz bandwidth: $K_{p,pos} = 10\pi \approx 31$.

**Why no I term?** The plant itself is an integrator (speed → position). Adding I to the controller means **double integration** → 180° phase lag → stability problems. Use I only if there's a steady-state position error (e.g., friction preventing final positioning).

**Why no D term?** D on position = a speed feedforward, which can be useful but makes tuning harder. Usually handled separately as explicit feedforward (see Lesson 08).

## 4.4 Implementation

```c
// Position PID — runs at 1 kHz
// P-only with optional feedforward
q16_t position_pid_compute(PositionPID *pid, q16_t pos_cmd, q16_t pos_meas,
                           q16_t feedforward_speed) {
    q16_t error = pos_cmd - pos_meas;
    
    // P-only
    q16_t speed_cmd = q16_mul(pid->Kp, error) + feedforward_speed;
    
    // Clamp to MAX SPEED
    if (speed_cmd > pid->speed_limit_pos) speed_cmd = pid->speed_limit_pos;
    if (speed_cmd < pid->speed_limit_neg) speed_cmd = pid->speed_limit_neg;
    
    return speed_cmd;  // This is the speed setpoint for the middle loop
}
```

---

# PART 5 — BANDWIDTH SEPARATION AND TUNING ORDER

## 5.1 Bandwidth Summary

```
Position loop: 5 Hz bandwidth → responds in ~50 ms
    │
    │ 10× separation
    ▼
Speed loop: 50 Hz bandwidth → responds in ~5 ms
    │
    │ 10× separation (at least)
    ▼
Current loop: 2 kHz bandwidth → responds in ~0.15 ms
```

## 5.2 Tuning Order (CRITICAL)

**Always tune from the inside out:**

1. **Current loop first** — disconnect speed loop, command current directly, verify step response
2. **Speed loop second** — keep current loop tuned, disconnect position loop, command speed directly
3. **Position loop last** — keep everything tuned, command position

**Why inside-out?** Each outer loop assumes the inner loop is already working. If you tune the speed loop with a badly-tuned current loop, you're compensating for the current loop's errors — and everything breaks when you fix the current loop later.

## 5.3 Practical Tuning Procedure

```
STEP 1: Current loop tuning
  a. Set speed PID output to 0
  b. Inject step current commands (e.g., 0 → 1A → 0A → -1A)
  c. Observe actual current on oscilloscope
  d. Adjust Kp_i and Ki_i for:
     - Rise time < 0.5 ms (for 2 kHz bandwidth)
     - No overshoot (or < 5%)
     - No ringing
     - No audible motor whine

STEP 2: Speed loop tuning
  a. Set position PID output to 0
  b. Inject step speed commands (e.g., 0 → 50 RPM → 0 → -50 RPM)
  c. Observe actual speed (encoder feedback)
  d. Adjust Kp_w and Ki_w for:
     - Rise time < 20 ms (for 50 Hz bandwidth)
     - Overshoot < 10%
     - Steady-state error = 0 (guaranteed by I term)
     - No oscillation under load changes

STEP 3: Position loop tuning
  a. Inject step position commands (e.g., rotate 1 revolution)
  b. Observe actual position
  c. Adjust Kp_pos for:
     - Smooth approach, no overshoot
     - Position held within ±1 encoder count
     - Speed limit not exceeded during slew
```

---

# PART 6 — INTER-LOOP COORDINATION

## 6.1 Rate Differences

The loops run at different rates. How do they communicate?

```
20 kHz (current)     : ─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─
10 kHz (speed)       : ─┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───
1 kHz (position)     : ─┬───────────┬───────────┬───────────┬─────
```

**Implementation:** The speed PID runs in the 10 kHz ISR. The current PID runs in the 20 kHz ISR. The position PID runs in a separate 1 kHz timer ISR (or in the main loop at 1 kHz).

**Data flow:** Each loop reads the *most recent* setpoint from the loop above it. No synchronization needed because the inner loop always runs faster.

```c
// Shared between ISRs (volatile for compiler)
volatile q16_t speed_setpoint;     // Written by position PID, read by speed PID
volatile q16_t current_setpoint;   // Written by speed PID, read by current PID

// 1 kHz position timer ISR
void position_timer_handler(void) {
    q16_t pos = read_position();
    speed_setpoint = position_pid_compute(&pos_pid, pos_cmd, pos);
}

// 10 kHz speed timer ISR
void speed_timer_handler(void) {
    q16_t spd = compute_speed();
    current_setpoint = speed_pid_compute(&spd_pid, speed_setpoint, spd);
}

// 20 kHz current timer ISR
void current_timer_handler(void) {
    q16_t cur = read_current();
    q16_t duty = current_pi_compute(&cur_pid, current_setpoint, cur);
    set_pwm_duty(duty);
}
```

## 6.2 Anti-Windup Across Loops

When the current is saturated (motor at max torque), the speed PID's integral winds up because speed can't track the command. Solution:

**Back-calculation anti-windup:**

```c
// In speed PID, after computing output
q16_t current_cmd_raw = pid->output;  // Before clamping
q16_t current_cmd_clamped = clamp(current_cmd_raw, -I_MAX, I_MAX);

// Back-calculate: reduce integral by the amount that was clamped
q16_t saturation = current_cmd_clamped - current_cmd_raw;
pid->output += q16_mul(pid->back_calc_gain, saturation);
```

The same applies to the position loop when speed is saturated.

---

## Checkpoint Questions

1. Why must the current loop be at least 10× faster than the speed loop?
2. If you only had one PID (position → duty), how would you limit motor current? (Hint: you can't, easily.)
3. The speed PID output is clamped to ±3A. What does this mean physically?
4. Why is the position controller often P-only? What happens if you add an I term?
5. You tune the speed loop perfectly, then re-tune the current loop. Does the speed loop still work? Why or why not?
6. Sketch the step response of the entire cascade to a 1-revolution position command. Label the current, speed, and position trajectories.
7. The motor hits a wall (stall). Describe what each loop does in the next 100 ms.

---

## Key Takeaways

- **Cascade = nested loops:** current (fastest) → speed → position (slowest)
- **Each loop limits its output:** current PID limits duty, speed PID limits current, position PID limits speed
- **10:1 bandwidth separation** ensures each loop is "transparent" to the one above
- **Tune from inside out:** current first, speed second, position last
- **Position loop often needs only P** because the plant (speed → position) is already an integrator
- **Anti-windup propagates outward:** when an inner loop saturates, outer loop integrals must be managed
