# Debug Session 2 — Integral Windup: "The Robot Lurches After E-Stop Release"

**Skills tested:** Anti-windup, integrator dynamics, e-stop recovery
**Difficulty:** Intermediate–Advanced
**Time estimate:** 30–45 minutes

---

## The Scenario

**Operator report from warehouse:**

> After releasing the e-stop button, the robot suddenly lurches forward 20–30 cm before stopping. This happens every time. The e-stop is pressed when the robot is driving forward at 0.3 m/s to stop it near a rack. After clearing the e-stop, the robot jumps forward dangerously close to the rack.

**What you have:**
- Speed PID with anti-windup (supposedly implemented)
- E-stop cuts power to motor H-bridges via hardware relay
- PID loop continues running during e-stop (software doesn't know about hardware e-stop immediately)
- Anti-windup strategy: integral clamping (output limits ±1.0 duty)

---

## Step 1 — Reproduce and Observe

### 1.1 Timeline of Events

```
Time    Event                          Speed_sp  Speed_meas  Integral  PWM_duty
─────────────────────────────────────────────────────────────────────────────
t=0.0   Driving normally               5.0       4.95        0.02      0.38
t=0.5   E-stop pressed (hardware)      5.0       5.0→0       0.02      0.38→0*
        * H-bridge disabled by relay, actual duty = 0 regardless of PWM

t=0.5   PID still running:
        error = 5.0 - 0.0 = 5.0
        integral += 5.0 * 0.001 = 0.005 per ms

t=0.6   integral = 0.02 + 5.0*100ms/1000 = 0.52
t=0.7   integral = 0.52 + 5.0*100ms/1000 = 1.02  → clamped to 1.0
t=0.8   integral = 1.0 (clamped at max)
...
t=3.0   Still clamped at 1.0 (2.5 seconds of e-stop)

t=3.0   E-stop released
        H-bridge re-enabled
        PID output = Kp*error + Ki*integral + Kd*de
                   = 5.0*5.0 + 25.0*1.0 + ...
                   = 25.0 + 25.0 = 50.0  → clamped to 1.0
        PWM duty = 100%  ← FULL POWER LURCH!
```

### 1.2 The Problem

The anti-windup clamp limits the integral to keep the output within ±1.0. But **1.0 is still 100% duty cycle!** When the e-stop is released, the motor receives full power because:
1. The speed is still 0 (stopped) → large error
2. The integral is at maximum (wound up during e-stop)
3. Both proportional and integral terms point the same direction

The robot accelerates at maximum torque until the speed catches up to the setpoint, overshoots, and eventually settles — but not before lurching forward 20–30 cm.

---

## Step 2 — Why the Anti-Windup Failed

### 2.1 The Clamp Is Not Anti-Windup

Look at the "anti-windup" code:

```c
// BAD: This is NOT proper anti-windup
float pid_compute(PIDController *pid, float error, float dt) {
    pid->integral += error * dt;
    
    // Clamp integral
    if (pid->integral > INTEGRAL_MAX) pid->integral = INTEGRAL_MAX;
    if (pid->integral < -INTEGRAL_MAX) pid->integral = -INTEGRAL_MAX;
    
    float output = pid->Kp * error + pid->Ki * pid->integral + pid->Kd * derivative;
    
    // Clamp output
    if (output > OUTPUT_MAX) output = OUTPUT_MAX;
    if (output < -OUTPUT_MAX) output = -OUTPUT_MAX;
    
    return output;
}
```

**The problem:** The integral is clamped based on its own range, but it doesn't know whether the output is **actually being applied.** During e-stop, the H-bridge is disabled — the output is effectively zero regardless of what PID computes. But the integral keeps accumulating (up to its clamp limit).

**True anti-windup requires knowing when the actuator is saturated** (or disconnected).

### 2.2 The Correct Mental Model

```
During normal operation:        During e-stop:
  Error → PID → Motor            Error → PID → [disconnected]
  Motor → Speed                   Motor → 0
  Speed → Error (closed loop)     0 → Error = LARGE (open loop!)
  
  Integral grows slowly           Integral grows rapidly
  because error is small          because error is huge and
                                  there's no feedback to reduce it
```

---

## Step 3 — Fix It

### 3.1 Fix 1: Back-Calculation Anti-Windup

The integral should track the **actual applied output**, not the desired output:

```c
float pid_compute_with_back_calc(PIDController *pid, float error, float dt,
                                  bool actuator_enabled) {
    float p_term = pid->Kp * error;
    float d_term = pid->Kd * (error - pid->prev_error) / dt;
    float i_term = pid->Ki * pid->integral;
    
    float desired_output = p_term + i_term + d_term;
    
    // Clamp to actuator limits
    float actual_output = clamp(desired_output, -OUTPUT_MAX, OUTPUT_MAX);
    
    // If actuator is disabled (e-stop), effective output is 0
    if (!actuator_enabled) {
        actual_output = 0.0f;
    }
    
    // Back-calculation: adjust integral based on what was actually applied
    float saturation_error = actual_output - desired_output;
    pid->integral += error * dt + (saturation_error / pid->Ki) * pid->Kb;
    
    pid->prev_error = error;
    return actual_output;
}
```

$K_b$ (back-calculation gain) determines how fast the integral "unwinds." Typical: $K_b = 1/T_i$ where $T_i = K_p/K_i$.

### 3.2 Fix 2: Reset Integral on E-Stop

Simpler and more robust for the robot case:

```c
// In the main control loop
void control_loop(void) {
    if (estop_active()) {
        // Reset integral to zero during e-stop
        pid_reset_integral(&speed_pid_left);
        pid_reset_integral(&speed_pid_right);
        
        // Also reset setpoint to zero
        speed_setpoint_left = 0.0f;
        speed_setpoint_right = 0.0f;
        return;  // Don't compute PID output
    }
    
    // Normal PID operation
    // ...
}
```

### 3.3 Fix 3: Ramp Setpoint After E-Stop Release

Even with integral reset, the proportional term will cause a jump if the setpoint is still non-zero:

```c
void post_estop_recovery(void) {
    static float ramp_factor = 0.0f;
    
    if (estop_just_released()) {
        ramp_factor = 0.0f;  // Start from zero
    }
    
    if (ramp_factor < 1.0f) {
        ramp_factor += RAMP_RATE * dt;  // e.g., RAMP_RATE = 2.0 → 500ms to full speed
        if (ramp_factor > 1.0f) ramp_factor = 1.0f;
    }
    
    // Apply ramp to setpoint
    float effective_setpoint = cmd_setpoint * ramp_factor;
    
    float output = pid_compute(&speed_pid, effective_setpoint, speed_measured);
}
```

**AMR implements all three:** Reset integral on e-stop, ramp setpoint on release, and back-calculation anti-windup during normal operation.

---

## Step 4 — Verify

### 4.1 Before Fix

```
t=3.0 (e-stop release):
  Speed: 0 → 7.2 rad/s in 50ms (overshoot 44%)
  Position: +22 cm lurch
  Current: peaks at 15A (overcurrent warning)
```

### 4.2 After Fix (all three measures)

```
t=3.0 (e-stop release):
  Speed: 0 → ramps to 5.0 over 500ms (no overshoot)
  Position: smooth acceleration, no lurch
  Current: peaks at 3A (normal)
```

---

## Step 5 — The Deeper Lesson

### 5.1 Where Else Does This Happen?

Integral windup occurs whenever the **control loop is broken**:

| Situation | Loop broken because | Effect |
|-----------|-------------------|--------|
| E-stop | Actuator disconnected | Lurch on release |
| Motor stall | Torque limit reached, speed=0 | Lurch when unstalled |
| Wheel in the air | No load → speed maxes instantly | Slam when wheel touches ground |
| Mode switch | Controller changes from position to speed mode | Spike at mode transition |
| Sensor failure | Measured value freezes or jumps | Integral diverges |

### 5.2 The Golden Rule

> **If the actuator can't do what the controller wants, the integral must know about it.**

This is the fundamental principle of anti-windup. The integral should only accumulate error when the system is actually responding to the control effort.

---

## Checkpoint Questions

1. Why doesn't clamping the integral term prevent windup during e-stop?
2. The integral accumulated to 1.0 during 2.5 seconds of e-stop with error = 5.0 and Ki = 25.0. Show the math: $1.0 = 25.0 \times 5.0 \times t_{windup}$. Solve for $t_{windup}$.
3. Back-calculation anti-windup uses a gain $K_b$. If $K_b$ is too large, what happens? Too small?
4. Why does AMR reset the setpoint to zero during e-stop (not just the integral)?
5. A robot wheel lifts off the ground during a ramp ascent. The speed PID winds up. What happens when the wheel touches down again?
