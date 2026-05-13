# 01 — What Is Control?
### From open-loop guessing to closed-loop precision

**Prerequisite:** None — this is the entry point.
**Unlocks:** `02-modeling-dc-motors.md` (you need to understand "plant" and "transfer function" before modeling a motor)

---

## Why Should I Care? (OKS Context)

Every OKS robot has two nested control problems running simultaneously:

1. **Inner loop (STM32):** "Make the left wheel spin at exactly 1.5 rad/s" — a speed control problem
2. **Outer loop (Jetson):** "Follow this trajectory from tile A to tile B" — a trajectory tracking problem

Both are *feedback control systems*. If you don't understand what that means — what feedback buys you, what happens when it's removed, and how the math describes it — you'll keep seeing symptoms without understanding causes.

**Real OKS failure:** In ticket #98367, a robot's left motor controller lost its encoder feedback (connector vibrated loose). The motor was running **open-loop** — just applying a fixed voltage. Without feedback, the wheel speed drifted with load (turning corners = more friction = slower). The robot veered left on every turn until the estimator delocalized. This note explains *why* that happened and *why* feedback prevents it.

---

# PART 1 — OPEN LOOP vs CLOSED LOOP

## 1.1 The Shower Problem

**ELI15:** You're in a shower. You want the water at 40°C.

**Open-loop approach:** Calculate exactly how far to turn the hot and cold knobs based on water pressure, pipe temperature, and ambient conditions. Set them. Walk in. Hope for the best.

**Closed-loop approach:** Turn the knobs. Feel the water. Too cold? Turn hot knob more. Too hot? Back off. Repeat until comfortable.

The open-loop approach requires a *perfect model* of the system. The closed-loop approach only requires that you can *measure the output* and *adjust the input*. In the real world, models are never perfect — pipes have different pressure at different times of day, the water heater's temperature drifts. **Feedback makes the system robust to things you didn't model.**

```
OPEN LOOP:
                    ┌──────────┐
  desired ────────→ │ Controller │────→ Plant ────→ actual output
  output            └──────────┘         │
                                         │
                    (no measurement,      │
                     no correction)       ▼
                                      who knows?


CLOSED LOOP:
                         error    ┌──────────┐              ┌───────┐
  desired ──→ ⊕ ──────────────→  │ Controller │─── u(t) ──→ │ Plant │──→ actual output
  (reference) │ -                 └──────────┘              └───────┘      │
              │                                                            │
              └──────────────────── measurement ◄──────────────────────────┘
                                     (sensor)
```

**Key terms:**
- **Reference (r):** What you *want* (40°C, 1.5 rad/s, position on tile)
- **Plant:** The thing you're controlling (shower, motor, robot)
- **Error (e):** Reference minus measurement: `e(t) = r(t) - y(t)`
- **Controller:** Computes the control input from the error
- **Control input (u):** The actuator command (valve position, PWM duty, cmd_vel)
- **Output (y):** The measured result (water temperature, wheel speed, robot position)
- **Disturbance (d):** Anything that pushes the output away from the reference without your control (friction, slope, wind)

## 1.2 Why Open Loop Fails

Open loop works when:
1. The plant model is *exactly known*
2. There are *no disturbances*
3. The plant doesn't *change over time*

None of these hold for a motor on an AMR:
- Motor constants drift with temperature (winding resistance increases ~0.4%/°C)
- Floor friction changes (tile joints, debris, wet patches)
- Battery voltage drops as charge depletes (12.6V → 10.5V over a shift)
- Wheel diameter changes with tire wear

**Quantified OKS example:** An OKS motor at fresh battery (12.4V) vs depleted battery (10.8V) produces ~13% less torque at the same PWM duty cycle. On a 3-meter tile traverse at 0.3 m/s, that's 0.04 m/s speed error × 10 seconds = **40 cm position error** by the end of the tile. With feedback, the speed PID compensates within 2 control cycles (~0.2 ms).

## 1.3 What Feedback Buys You (The Three Gifts)

1. **Disturbance rejection:** Floor friction changed? The controller compensates automatically.
2. **Robustness to model error:** Motor constant wrong by 20%? Feedback corrects the steady-state error.
3. **Stability of an unstable plant:** Some systems (inverted pendulum, balancing robot) are physically unstable — they fall over without active control. Feedback can stabilize them.

**What feedback costs you:**
- You need a *sensor* (encoder, current sensor, IMU)
- The system can *oscillate* if the controller is tuned wrong
- The sensor introduces *delay*, which limits how fast you can react

---

# PART 2 — BLOCK DIAGRAMS AND TRANSFER FUNCTIONS

## 2.1 Block Diagram Algebra

Every control system can be drawn as blocks connected by signals. The rules for simplifying them are mechanical — you can always reduce any block diagram to a single transfer function.

**The three fundamental connections:**

```
SERIES:
  X ──→ [G₁] ──→ [G₂] ──→ Y     ⟹    X ──→ [G₁ · G₂] ──→ Y


PARALLEL:
          ┌──→ [G₁] ──┐
  X ──→ ⊕ ┤            ├──→ Y    ⟹    X ──→ [G₁ + G₂] ──→ Y
          └──→ [G₂] ──┘


NEGATIVE FEEDBACK:
                    ┌──────┐
  R ──→ ⊕ ──→ E ──→│  G   │──→ Y      ⟹    Y/R = G / (1 + G·H)
        │ -         └──────┘  │
        │    ┌──────┐         │
        └────│  H   │◄────────┘
             └──────┘
```

The **closed-loop transfer function** `T(s) = G(s) / (1 + G(s)·H(s))` is the single most important equation in control theory. Everything else is a variation of it.

**OKS motor example:**
- G(s) = PID controller × motor transfer function
- H(s) = encoder feedback (usually = 1 if measuring speed directly)
- T(s) = closed-loop speed response

## 2.2 Transfer Functions (Laplace Domain)

A transfer function describes the input→output relationship of a linear time-invariant (LTI) system in the Laplace domain.

**Why Laplace?** Because differential equations become algebraic equations. Instead of solving:

$$m\ddot{x} + b\dot{x} + kx = F(t)$$

You solve:

$$X(s) = \frac{F(s)}{ms^2 + bs + k}$$

**The transfer function** is:

$$G(s) = \frac{Y(s)}{U(s)} = \frac{\text{output}}{\text{input}}$$

**First-order system** (one energy storage element — e.g., RC circuit, motor speed):

$$G(s) = \frac{K}{\tau s + 1}$$

- $K$ = steady-state gain (how much output per unit input)
- $\tau$ = time constant (63% of final value reached at $t = \tau$)

**Step response of a first-order system:**

$$y(t) = K \cdot (1 - e^{-t/\tau})$$

```
  y(t)
  K ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
  │                    _______________
  │                 /
  │              /
  │           /          63% at t = τ
  │        /               95% at t = 3τ
  │     /                    99% at t = 5τ
  │  /
  │/
  └─────────────────────────────────── t
  0    τ    2τ    3τ    4τ    5τ
```

**Second-order system** (two energy storage elements — e.g., motor position, spring-mass-damper):

$$G(s) = \frac{\omega_n^2}{s^2 + 2\zeta\omega_n s + \omega_n^2}$$

- $\omega_n$ = natural frequency (how fast the system wants to oscillate)
- $\zeta$ = damping ratio:
  - $\zeta = 0$: pure oscillation (undamped)
  - $0 < \zeta < 1$: oscillation that decays (underdamped) — most real motors
  - $\zeta = 1$: fastest non-oscillating response (critically damped)
  - $\zeta > 1$: slow, no oscillation (overdamped)

**OKS motor speed loop:** Typically underdamped ($\zeta \approx 0.5–0.8$) — you *want* a bit of overshoot for fast response, but not so much that the wheel surges.

## 2.3 Poles and Zeros

The **poles** of a transfer function are the roots of the denominator. They determine:
- **Stability:** All poles must have negative real parts (left half of the s-plane) for the system to be stable
- **Speed of response:** Poles further from the origin → faster response
- **Oscillation:** Complex poles → oscillation; real poles → exponential decay

```
           Im(s)
            │
            │     × ←── complex pole pair
            │    /       (oscillates)
            │   /
  ──────────┼──/────────── Re(s)
     stable │ /  unstable
     region │/   region
            │
            │     ×
            │
```

**Critical insight for senior engineers:** A system is stable if and only if all closed-loop poles are in the left half-plane. When you increase PID gains, you move the poles. Too much gain → poles cross into the right half-plane → instability → the motor oscillates, the robot shakes.

## 2.4 Steady-State Error

The error that remains after the system has settled. Depends on the **system type** (number of integrators in the open-loop transfer function):

| System Type | Step input error | Ramp input error | Note |
|-------------|-----------------|-------------------|------|
| Type 0 (no integrator) | $\frac{1}{1+K_p}$ | ∞ | P-only controller on a motor *speed* loop |
| Type 1 (one integrator) | 0 | $\frac{1}{K_v}$ | PI controller or position loop with speed controller |
| Type 2 (two integrators) | 0 | 0 | Rare — double integrator plant |

**OKS relevance:** The STM32 speed PID uses PI control (proportional + integral). The integral term adds a "free" integrator, making the system Type 1 → **zero steady-state error for constant speed commands**. Without the I term (P-only), the wheel speed would always be slightly wrong.

---

# PART 3 — STABILITY ANALYSIS (INTUITION, NOT PROOFS)

## 3.1 When Does a Feedback System Go Unstable?

The closed-loop system $T(s) = \frac{G(s)}{1 + G(s)H(s)}$ goes unstable when $1 + G(s)H(s) = 0$, i.e., when the loop gain $G(s)H(s) = -1$.

**Physical intuition:** Imagine you're in a room with a microphone and speaker. If the speaker volume (gain) is high enough AND the sound reaches the microphone with the right delay (phase = -180°), the system self-oscillates — that's microphone feedback squeal. Same physics.

For a control loop:
1. **Gain condition:** The loop gain magnitude equals 1 (0 dB)
2. **Phase condition:** The loop phase equals -180° (signal inverts)

If both happen at the same frequency → sustained oscillation. If the gain is *greater* than 1 at -180° → growing oscillation → instability.

## 3.2 Gain Margin and Phase Margin

- **Gain margin:** How much you can increase the gain before instability. Measured at the frequency where phase = -180°.
- **Phase margin:** How much more phase lag the system can tolerate before instability. Measured at the frequency where gain = 0 dB (unity).

**Rule of thumb for robotics:**
- Phase margin ≥ 45° → safe, well-damped
- Phase margin 30–45° → works but oscillates on disturbances
- Phase margin < 30° → dangerous, may oscillate

**OKS motor tuning:** When the firmware team tunes PID gains on a new motor, they increase Kp until the motor starts to buzz (oscillation). That buzz frequency is the **gain crossover frequency**. They then back off by ~6 dB. This is empirical gain margin tuning.

## 3.3 The Deadly Trio: Gain, Delay, and Resonance

Three things kill control loops in practice:

1. **Too much gain:** Amplifies noise, causes oscillation
2. **Too much delay:** By the time the controller reacts, the error has changed — overcorrection → oscillation
3. **Resonance:** The mechanical structure has a natural frequency. If the control bandwidth approaches it, the system self-excites.

**OKS example of each:**
- **Gain:** Kp set to 5.0 instead of 0.5 on the speed loop → motor buzzes at 200 Hz
- **Delay:** SPI bus latency increases from 1 ms to 15 ms after a firmware bug → cmd_vel arrives too late → heading oscillation
- **Resonance:** The robot's chassis has a mechanical resonance at ~35 Hz. If the motor current loop bandwidth exceeds ~20 Hz, it excites the chassis → vibration → sensorbar readings become noisy

---

# PART 4 — THE LANGUAGE OF CONTROL SPECIFICATIONS

## 4.1 Time-Domain Specifications

When you tune a controller, you're trading off these competing metrics:

```
  y(t)
  │        overshoot (Mp)
  │       ╱
  │  ────╳─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ final value
  │ /    │ ╲
  │/     │  ╲___________
  │      │       ╲
  │      │        settling time (ts)
  │      │
  │   rise time (tr)
  │
  └──────────────────────────────── t
```

| Metric | Definition | Typical OKS motor spec |
|--------|-----------|----------------------|
| **Rise time** ($t_r$) | Time from 10% to 90% of final value | < 5 ms (speed loop) |
| **Overshoot** ($M_p$) | Peak value above final value, as % | < 15% |
| **Settling time** ($t_s$) | Time to stay within ±2% of final value | < 20 ms |
| **Steady-state error** ($e_{ss}$) | Error after settling | 0 (with integral term) |

**The fundamental trade-off:** Faster rise time ↔ more overshoot. You can't have both without a more sophisticated controller (feedforward, MPC).

## 4.2 Why These Numbers Matter for OKS

The robot receives a new `cmd_vel` every 20 ms (50 Hz). The motor speed PID must settle *within one cmd_vel period* — otherwise the wheel is still responding to the *previous* command when the *next* one arrives. If settling time > 20 ms, the commands pile up and the system becomes effectively open-loop.

This is why the **inner loop must be 10–100x faster than the outer loop**. It's not a rule of thumb — it's a mathematical necessity. The inner loop is the "actuator" from the outer loop's perspective, and that actuator must have settled before the next outer-loop command arrives.

---

# PART 5 — CONTROL SYSTEM TYPES IN ROBOTICS

## 5.1 The Control Zoo

| Type | What it controls | Example | OKS equivalent |
|------|-----------------|---------|----------------|
| **Regulator** | Keep output constant despite disturbances | Thermostat | Motor speed hold on a slope |
| **Servo** | Track a changing reference | Robot arm following a trajectory | Wheel speed tracking cmd_vel profile |
| **State feedback** | Uses full state vector, not just output | LQR | Not used in OKS (estimator is separate) |
| **Optimal** | Minimizes a cost function | MPC | Used in some research AMRs, not OKS |
| **Adaptive** | Adjusts controller gains online | MRAC | Not in OKS (gains are fixed at commissioning) |

OKS uses **PI servo control** for motor speed and **geometric tracking** (pure pursuit variant) for trajectory following. Simple, robust, well-understood.

## 5.2 Single-Input Single-Output (SISO) vs MIMO

A differential-drive robot has two motors. You could control them independently (two SISO loops) or as a coupled system (one MIMO loop that accounts for how left wheel affects right wheel heading).

OKS uses **decoupled SISO**: each motor has its own independent speed PID. The coupling between wheels is handled at the *kinematic* level (cmd_vel decomposition into left/right wheel speeds), not at the *dynamic* level.

**When does MIMO matter?** When the cross-coupling is strong — e.g., a quadrotor where changing one motor's speed tilts the whole frame, affecting all other motors. For a heavy ground robot on flat floor, SISO is sufficient.

---

## Checkpoint Questions

1. Draw the block diagram of a closed-loop speed control system for one motor. Label: reference, error, controller, plant, sensor, output.
2. What is the closed-loop transfer function if G(s) = 10/(s+5) and H(s) = 1?
3. A P-only controller on a speed loop has steady-state error of 5%. Why? What do you add to eliminate it?
4. The motor starts oscillating when you increase Kp to 3.0. At Kp = 2.0, it doesn't oscillate. Estimate the gain margin.
5. Why must the inner speed loop settle in < 20 ms for an OKS robot receiving cmd_vel at 50 Hz?

---

## Key Takeaways

- **Feedback** makes systems robust to model error and disturbances, at the cost of potential instability
- **Transfer functions** convert differential equations to algebra — poles determine stability and response speed
- **Stability** requires all closed-loop poles in the left half-plane; gain margin and phase margin quantify how close you are to instability
- **The inner loop must be 10–100x faster than the outer loop** — this is a mathematical constraint, not a preference
- **PID with integral term** eliminates steady-state error for step commands (Type 1 system)
