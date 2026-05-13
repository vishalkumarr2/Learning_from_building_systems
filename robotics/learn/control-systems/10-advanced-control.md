# 10 — Advanced Control: Beyond PID
### Feedforward, gain scheduling, MPC, and knowing when PID isn't enough

**Prerequisite:** All previous lessons (01–09)
**Unlocks:** Debugging sessions, exercises, senior interview questions

---

## Why Should I Care? (OKS Context)

PID handles 90% of OKS motor control. But the remaining 10% causes 50% of the field incidents:

- **Robot loaded vs. unloaded:** A robot carrying 200 kg payload has 3× the inertia. PID gains tuned for empty robot oscillate with load. → **Gain scheduling**
- **Predictable motion:** When the robot follows a known trajectory, waiting for error to build before reacting is wasteful. → **Feedforward**
- **Tight constraints:** Navigating through narrow passages while respecting speed, acceleration, and obstacle distance simultaneously. → **MPC**

This lesson covers the three techniques that OKS engineers reach for when PID alone fails.

---

# PART 1 — FEEDFORWARD CONTROL

## 1.1 The Fundamental Insight

PID is **reactive** — it waits for error, then corrects. Feedforward is **proactive** — if you know the desired trajectory, compute the control effort needed **before** error appears.

```
                   Feedforward
Desired ───┬──────────────────────────────────┐
trajectory  │                                  ↓ (+)
            │     ┌──────┐   Error   ┌─────┐  ∑ ──→ Plant
            └────→│ - Σ  │─────────→│ PID │──↑
                  └──────┘          └─────┘
                    ↑
              Measured output
```

**PID alone:** Motor must first fall behind → error accumulates → PID reacts → always lagging.

**PID + Feedforward:** Feedforward provides ~90% of control effort. PID handles the remaining ~10% (model errors, disturbances).

## 1.2 Velocity Feedforward

For a DC motor with transfer function $G(s) = \frac{K}{Js + B}$:

At steady state, the torque needed to maintain speed $\omega$ is:

$$\tau_{ss} = B \cdot \omega$$

The voltage needed:

$$V_{ff} = \frac{B \cdot \omega}{K_t} \cdot R_a + K_e \cdot \omega$$

In practice, we use a simpler model — a **velocity feedforward gain** $K_{ff,v}$:

$$u_{ff} = K_{ff,v} \cdot \omega_{desired}$$

```c
// Velocity feedforward in speed PID loop
float speed_pid_with_ff(float speed_setpoint, float speed_measured) {
    // Feedforward: voltage proportional to desired speed
    float ff = KFF_VEL * speed_setpoint;
    
    // PID on the error (handles remaining mismatch)
    float error = speed_setpoint - speed_measured;
    float pid_out = pid_compute(&speed_pid, error);
    
    // Sum feedforward + PID
    float output = ff + pid_out;
    
    return clamp(output, -MAX_DUTY, MAX_DUTY);
}
```

**Calibrating $K_{ff,v}$:** Run the motor at several constant speeds, record the steady-state duty cycle, fit a line. The slope is $K_{ff,v}$.

```python
# Calibration: constant speed → steady-state duty
import numpy as np

speeds = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]  # rad/s
duties = [0.08, 0.15, 0.22, 0.37, 0.52, 0.74]  # measured steady-state duty

# Linear fit: duty = K_ff_v * speed + offset
K_ff_v, offset = np.polyfit(speeds, duties, 1)
print(f"K_ff_v = {K_ff_v:.4f}, offset = {offset:.4f}")
# Typical OKS result: K_ff_v ≈ 0.074, offset ≈ 0.01 (friction)
```

## 1.3 Acceleration Feedforward

Newton's second law: $F = ma$, or for rotation: $\tau = J \cdot \alpha$.

When accelerating, the motor needs extra torque beyond what's needed for steady speed:

$$u_{ff,accel} = K_{ff,a} \cdot \dot{\omega}_{desired}$$

```c
float speed_pid_with_full_ff(float sp, float sp_prev, float meas, float dt) {
    // Velocity feedforward
    float ff_vel = KFF_VEL * sp;
    
    // Acceleration feedforward
    float accel = (sp - sp_prev) / dt;
    float ff_accel = KFF_ACCEL * accel;
    
    // PID for residual error
    float error = sp - meas;
    float pid_out = pid_compute(&speed_pid, error);
    
    return clamp(ff_vel + ff_accel + pid_out, -MAX_DUTY, MAX_DUTY);
}
```

**Impact of feedforward on OKS:**
- Without FF: Tracking error during acceleration is 50–100 rpm
- With velocity FF: Tracking error drops to 10–20 rpm
- With velocity + acceleration FF: Tracking error drops to 2–5 rpm

---

# PART 2 — GAIN SCHEDULING

## 2.1 The Problem: One PID Doesn't Fit All

The OKS robot motor has different dynamics depending on operating conditions:

| Condition | Inertia $J$ | Friction $B$ | Optimal $K_p$ |
|-----------|-------------|-------------|---------------|
| Unloaded | 0.005 kg·m² | 0.01 N·m·s | 2.5 |
| 50 kg payload | 0.010 kg·m² | 0.015 N·m·s | 4.0 |
| 200 kg payload | 0.020 kg·m² | 0.025 N·m·s | 6.5 |
| Low speed (< 1 rad/s) | Same | Higher (stiction) | +30% |
| High speed (> 8 rad/s) | Same | Lower (viscous dominant) | -10% |

**A single PID tuning is always a compromise.** Gain scheduling uses different gains for different operating conditions.

## 2.2 Lookup-Table Gain Scheduling

The simplest approach: a table of PID gains indexed by operating condition.

```c
typedef struct {
    float Kp, Ki, Kd;
} PIDGains;

// Gain table indexed by estimated load (from current draw)
static const PIDGains gain_table[] = {
    // load_index=0: unloaded
    { .Kp = 2.5, .Ki = 15.0, .Kd = 0.01 },
    // load_index=1: light load (~50 kg)
    { .Kp = 4.0, .Ki = 20.0, .Kd = 0.015 },
    // load_index=2: heavy load (~200 kg)
    { .Kp = 6.5, .Ki = 30.0, .Kd = 0.025 },
};

int estimate_load_index(float avg_current) {
    if (avg_current < 1.0f) return 0;       // Unloaded
    if (avg_current < 3.0f) return 1;       // Light
    return 2;                                 // Heavy
}

void update_pid_gains(PIDController *pid, float avg_current) {
    int idx = estimate_load_index(avg_current);
    pid->Kp = gain_table[idx].Kp;
    pid->Ki = gain_table[idx].Ki;
    pid->Kd = gain_table[idx].Kd;
}
```

## 2.3 Interpolated Gain Scheduling

Discrete jumps between gain sets cause transients. Linear interpolation is smoother:

```c
PIDGains interpolate_gains(float load_factor) {
    // load_factor: 0.0 (unloaded) to 1.0 (max load)
    // Linearly interpolate between min and max gains
    PIDGains result;
    result.Kp = KP_MIN + load_factor * (KP_MAX - KP_MIN);
    result.Ki = KI_MIN + load_factor * (KI_MAX - KI_MIN);
    result.Kd = KD_MIN + load_factor * (KD_MAX - KD_MIN);
    return result;
}
```

## 2.4 Speed-Dependent Scheduling

At very low speed, stiction dominates. At high speed, back-EMF matters more. OKS uses different integral gains at low vs. high speed:

```c
void speed_dependent_gains(PIDController *pid, float speed_abs) {
    if (speed_abs < SPEED_LOW_THRESH) {
        // Low speed: higher Ki to overcome stiction
        // Lower Kp to prevent oscillation around zero
        pid->Ki = KI_LOW_SPEED;
        pid->Kp = KP_LOW_SPEED;
    } else {
        pid->Ki = KI_NORMAL;
        pid->Kp = KP_NORMAL;
    }
}
```

**Bumpless gain transfer:** When switching gains, the integral term must be adjusted to prevent output jumps:

```c
void bumpless_gain_switch(PIDController *pid, float new_Ki) {
    // Adjust integral accumulator so output doesn't change
    // old_output = Kp*e + Ki_old * integral + Kd*de
    // new_output = Kp*e + Ki_new * integral_new + Kd*de
    // Set integral_new = Ki_old * integral / Ki_new
    if (new_Ki > 0.001f) {
        pid->integral = pid->Ki * pid->integral / new_Ki;
    }
    pid->Ki = new_Ki;
}
```

---

# PART 3 — MODEL PREDICTIVE CONTROL (MPC)

## 3.1 What Is MPC?

MPC is an optimization-based controller. At every timestep:
1. **Predict** the system state over a horizon of $N$ steps using a model
2. **Optimize** a control sequence that minimizes a cost function (tracking error + control effort)
3. **Apply** only the **first** control action
4. **Repeat** at the next timestep (receding horizon)

```
Past              Now              Future (predicted)
  │                │                │
  ├─ measured ─────●── predicted ──►│
  │                │  ╱             │
  │                │ ╱  Optimized   │
  │                │╱   trajectory  │
  │                ●────────────────│
  │                │  Only first    │
  │                │  step applied  │
```

## 3.2 MPC Formulation

**State:** $\mathbf{x} = [x, y, \theta, v, \omega]^T$ (robot pose + velocities)

**Input:** $\mathbf{u} = [a, \dot{\omega}]^T$ (linear acceleration, angular acceleration)

**Model (discrete):**

$$\mathbf{x}_{k+1} = f(\mathbf{x}_k, \mathbf{u}_k)$$

For differential drive:
$$x_{k+1} = x_k + v_k \cos(\theta_k) \cdot \Delta t$$
$$y_{k+1} = y_k + v_k \sin(\theta_k) \cdot \Delta t$$
$$\theta_{k+1} = \theta_k + \omega_k \cdot \Delta t$$
$$v_{k+1} = v_k + a_k \cdot \Delta t$$
$$\omega_{k+1} = \omega_k + \dot{\omega}_k \cdot \Delta t$$

**Cost function:**

$$J = \sum_{k=1}^{N} \left[ q_1(x_k - x_k^{ref})^2 + q_2(y_k - y_k^{ref})^2 + q_3(\theta_k - \theta_k^{ref})^2 \right] + \sum_{k=0}^{N-1} \left[ r_1 a_k^2 + r_2 \dot{\omega}_k^2 \right]$$

**Subject to constraints:**
- $|v| \leq v_{max}$
- $|\omega| \leq \omega_{max}$
- $|a| \leq a_{max}$
- Obstacle avoidance: distance to obstacles $\geq d_{safe}$

## 3.3 Simplified Python MPC (Educational)

```python
import numpy as np
from scipy.optimize import minimize

def robot_dynamics(state, control, dt):
    """Predict next state."""
    x, y, theta, v, omega = state
    a, alpha = control
    
    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + omega * dt
    v_new = v + a * dt
    omega_new = omega + alpha * dt
    
    return np.array([x_new, y_new, theta_new, v_new, omega_new])

def mpc_cost(control_sequence, current_state, reference_path, N, dt,
             Q, R, v_max, omega_max, a_max):
    """MPC cost function to minimize."""
    controls = control_sequence.reshape(N, 2)
    state = current_state.copy()
    cost = 0.0
    
    for k in range(N):
        # Apply control
        state = robot_dynamics(state, controls[k], dt)
        
        # Tracking cost
        ref = reference_path[k]
        cost += Q[0] * (state[0] - ref[0])**2  # x error
        cost += Q[1] * (state[1] - ref[1])**2  # y error
        cost += Q[2] * (state[2] - ref[2])**2  # theta error
        
        # Control effort cost
        cost += R[0] * controls[k, 0]**2  # acceleration penalty
        cost += R[1] * controls[k, 1]**2  # angular accel penalty
        
        # Soft velocity constraints (penalty method)
        if abs(state[3]) > v_max:
            cost += 1000 * (abs(state[3]) - v_max)**2
        if abs(state[4]) > omega_max:
            cost += 1000 * (abs(state[4]) - omega_max)**2
    
    return cost

def run_mpc(current_state, reference_path, N=10, dt=0.1):
    """Run one MPC step."""
    Q = [10.0, 10.0, 5.0]   # State tracking weights
    R = [1.0, 1.0]           # Control effort weights
    
    # Initial guess: zero controls
    u0 = np.zeros(N * 2)
    
    # Optimize
    result = minimize(
        mpc_cost, u0,
        args=(current_state, reference_path, N, dt, Q, R, 0.5, 0.5, 0.3),
        method='SLSQP'
    )
    
    # Return first control action only
    optimal_controls = result.x.reshape(N, 2)
    return optimal_controls[0]  # [acceleration, angular_acceleration]
```

## 3.4 MPC: Pros and Cons

| Aspect | MPC | PID + Feedforward |
|--------|-----|------------------|
| Constraint handling | Explicit (in optimization) | Implicit (clamping) |
| Multi-variable | Natural | Separate loops |
| Preview (look-ahead) | Yes (horizon N) | No |
| Computation | Heavy (optimization every step) | Light (arithmetic) |
| Tuning | Cost weights Q, R | Kp, Ki, Kd |
| Guarantee | Optimal within model | Heuristic |
| Model requirement | Good model needed | Model-free |
| Real-time feasibility | Hard on MCU | Easy on MCU |

**OKS usage:** MPC runs on the **Jetson** for path tracking (replacing or supplementing DWB). The MCU still runs PID for low-level motor control. This is the standard architecture for mobile robots:
- **Jetson (20-50 Hz):** MPC or DWB → cmd_vel (v, ω)
- **MCU (10 kHz):** PID cascade → motor PWM

## 3.5 When to Use MPC vs. PID

```
Decision tree:
                Is it a SISO loop?
                 ╱          ╲
               Yes            No
               │              │
         Is bandwidth     → Consider MPC
         > 100 Hz?         or MIMO controller
          ╱      ╲
        Yes       No
        │         │
    PID on MCU   Can you formulate
    (only option)  constraints explicitly?
                    ╱          ╲
                  Yes            No
                  │              │
              MPC on Jetson   PID + feedforward
              (worth the       + gain scheduling
               computation?)
```

---

# PART 4 — OTHER ADVANCED TECHNIQUES (OVERVIEW)

## 4.1 LQR (Linear Quadratic Regulator)

LQR finds the **optimal** state-feedback gains for a **linear** system by minimizing:

$$J = \int_0^\infty (\mathbf{x}^T Q \mathbf{x} + \mathbf{u}^T R \mathbf{u}) dt$$

Result: $\mathbf{u} = -K \mathbf{x}$, where $K$ is computed by solving the Riccati equation.

**vs. PID:** LQR is optimal for linear systems; PID is a special case. LQR naturally handles MIMO (multi-input, multi-output) systems.

**Limitation:** Requires a linear model. Real motors are nonlinear (saturation, stiction). Works well near an operating point.

```python
from scipy.linalg import solve_continuous_are

def compute_lqr_gain(A, B, Q, R):
    """Compute LQR gain matrix K."""
    # Solve Algebraic Riccati Equation
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K
```

## 4.2 Adaptive Control

**Idea:** Estimate the plant parameters online and adjust the controller in real-time.

For OKS, this means:
- Measure motor response to known inputs
- Estimate inertia $J$ and friction $B$
- Update PID gains or feedforward parameters

**Model Reference Adaptive Control (MRAC):**

```python
class SimpleAdaptive:
    """Simplified MIT rule adaptive controller."""
    def __init__(self, gamma=0.01):
        self.gamma = gamma  # Adaptation rate
        self.theta = 1.0    # Estimated plant gain
    
    def update(self, reference_output, actual_output, input_signal):
        # Error between reference model and actual
        error = actual_output - reference_output
        
        # MIT rule: adjust parameter to reduce error
        self.theta -= self.gamma * error * input_signal
        
        return self.theta
```

**OKS doesn't use full adaptive control** (too risky for production — stability isn't guaranteed during adaptation). Instead, OKS uses:
- **Offline calibration** at deployment
- **Gain scheduling** (Part 2) based on load estimation
- **Feedforward calibration** during commissioning

## 4.3 Disturbance Observer (DOB)

A DOB estimates and compensates for unmodeled disturbances:

```
             ┌──────────┐
  u ────────►│  Plant    ├──── y
             └──────────┘
                  ↑
             ┌──────────┐
  d_hat ←────┤   DOB    │◄── y
  (estimated)└──────────┘
```

$$\hat{d} = \text{LPF}\{y - \hat{G}(s) \cdot u\}$$

The estimated disturbance $\hat{d}$ is subtracted from the control signal, effectively canceling the disturbance.

**OKS application:** The robot drives over bumps and uneven floor. The DOB estimates the disturbance torque from floor irregularities and compensates, keeping speed constant.

---

# PART 5 — MCU vs JETSON: CHOOSING WHERE TO RUN

## 5.1 Decision Matrix

| Controller | Where | Why |
|-----------|-------|-----|
| Current PID | MCU (10 kHz) | Electrical dynamics, µs response |
| Speed PID + FF | MCU (1–10 kHz) | Mechanical dynamics, determinism |
| Position PID | MCU (100–1000 Hz) | Encoder counting, tight loop |
| Gain scheduling | MCU | Simple lookup, fast |
| Velocity smoother | Jetson (50 Hz) | Part of Nav2 pipeline |
| Pure pursuit / DWB | Jetson (20 Hz) | Needs map, costmap, TF |
| MPC (path tracking) | Jetson (10–50 Hz) | Heavy computation, constraints |
| LQR | Either | Light computation once K is computed |

## 5.2 Why Not Run Everything on the Jetson?

**Latency:** The Jetson runs Linux. Even with `preempt_RT`, worst-case latency is ~1 ms. For current control at 10 kHz (100 µs period), 1 ms jitter is 10× the loop period. The motor would oscillate or blow a fuse.

**Reliability:** Linux can OOM-kill, GC-pause, or kernel panic. The MCU's bare-metal or RTOS code has none of these failure modes. Motor safety must survive a Jetson crash.

## 5.3 Why Not Run Everything on the MCU?

**Computation:** MPC requires solving an optimization problem every timestep. On a 168 MHz STM32, this would take ~50–100 ms — far too slow for 20 Hz updates.

**Memory:** Path planning needs the full occupancy grid (e.g., 200×200 cells × 4 bytes = 160 KB). The STM32F4 has 192 KB SRAM total.

**Connectivity:** The MCU has no LiDAR driver, no camera interface, no ROS2 DDS stack.

---

## Checkpoint Questions

1. Draw the block diagram for PID + velocity feedforward. Where does the feedforward signal enter?
2. You've tuned PID for an empty robot. The robot picks up a 200 kg payload. What changes in the system dynamics, and how should the gains respond?
3. MPC runs at 20 Hz on the Jetson. The horizon is N=10 steps with dt=0.1s. How far ahead does MPC "see"?
4. Why can't you run MPC on the STM32 MCU? Give two reasons.
5. A gain scheduling system switches from low-speed gains to normal-speed gains at exactly $\omega = 1.0$ rad/s. What problem occurs when the robot speed oscillates around 1.0 rad/s?
6. The velocity feedforward gain is calibrated at 25°C. At 60°C (motor hot), the winding resistance increases by 30%. What happens to the feedforward accuracy?
7. When would you choose LQR over PID? When would LQR be a bad choice?

---

## Key Takeaways

- **Feedforward** is the single most impactful addition to PID — eliminates most tracking error during known trajectories
- **Gain scheduling** handles varying operating conditions (load, speed) with a lookup table or interpolation
- **MPC** handles constraints explicitly and optimizes over a horizon, but requires a model and heavy computation
- **The two-layer split persists:** MCU runs PID + FF + gain scheduling (fast, deterministic). Jetson runs path following, MPC, planning (slow, flexible).
- **Don't over-engineer:** 90% of OKS motor control is PID + velocity feedforward. Add complexity only when evidence shows PID isn't sufficient.
- **Calibration > algorithms:** A well-calibrated PID + feedforward usually outperforms a poorly-calibrated MPC.
