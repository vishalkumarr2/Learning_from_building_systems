# 02 — Modeling DC Motors
### The plant you're actually controlling

**Prerequisite:** `01-what-is-control.md` (block diagrams, transfer functions, poles)
**Unlocks:** `03-pid-controller.md` — you need the motor transfer function to design a controller for it

---

## Why Should I Care? (OKS Context)

Before you can design a PID controller, you need to know *what you're controlling*. On an OKS robot, that's a **brushed DC motor** (earlier models) or **brushless DC motor** (BLDC, newer models) driving each wheel through a gearbox.

When you see a motor oscillating, or the robot can't maintain speed on a ramp, or the current draw spikes unexpectedly — the explanation is in the motor's physics. This lesson derives the motor model from first principles so you can predict these behaviors before they happen.

**Real OKS scenario:** A firmware update changed the motor PWM frequency from 20 kHz to 50 kHz. The motors ran hotter and drew more current at the same speed. Why? Because the motor's electrical time constant ($L/R \approx 0.5$ ms) is comparable to the 20 µs PWM period at 50 kHz — the current ripple increased. You can't predict this without understanding the electrical model.

---

# PART 1 — THE DC MOTOR AS TWO COUPLED SYSTEMS

## 1.1 Physical Intuition

A DC motor converts electrical energy to mechanical energy. It's two systems coupled through electromagnetism:

1. **Electrical circuit:** Voltage in → current through winding → magnetic field
2. **Mechanical system:** Magnetic field × current → torque → rotation

The coupling goes *both ways*:
- Current → torque (motor action)
- Rotation → back-EMF voltage (generator action)

This bidirectional coupling is what makes the motor self-regulating: as it speeds up, back-EMF increases, which reduces current, which reduces torque, until equilibrium.

```
    ELECTRICAL SIDE              COUPLING              MECHANICAL SIDE
   ┌─────────────────┐     ┌──────────────┐     ┌──────────────────┐
   │                  │     │              │     │                  │
   │  V_in ──→ R ──→ L ──→│ τ = Kt · i   │──→  │  J·(dω/dt) + B·ω │
   │       ↑         │     │              │     │         = τ       │
   │       │         │     │ e = Ke · ω   │◄── │                  │
   │       │         │     │              │     │                  │
   │  V_in = R·i +   │     └──────────────┘     └──────────────────┘
   │  L·(di/dt) + e  │
   │                  │
   └─────────────────┘
```

## 1.2 The Electrical Equation

Apply Kirchhoff's voltage law around the motor circuit:

$$V(t) = R \cdot i(t) + L \cdot \frac{di(t)}{dt} + e(t)$$

where:
- $V(t)$ = applied voltage (from H-bridge PWM)
- $R$ = winding resistance (Ω) — typically 0.5–5 Ω for small robot motors
- $L$ = winding inductance (H) — typically 0.1–5 mH
- $i(t)$ = motor current (A)
- $e(t) = K_e \cdot \omega(t)$ = back-EMF voltage
- $K_e$ = back-EMF constant (V·s/rad)

**Electrical time constant:** $\tau_e = L/R$

For a typical OKS motor: $L \approx 1$ mH, $R \approx 2$ Ω → $\tau_e = 0.5$ ms.
This means the current responds to voltage changes in ~0.5 ms — fast enough that for the *speed* loop (settling in ~10 ms), we can sometimes approximate the electrical dynamics as instantaneous.

## 1.3 The Mechanical Equation

Newton's second law for rotation:

$$J \cdot \frac{d\omega(t)}{dt} = \tau_{motor}(t) - B \cdot \omega(t) - \tau_{load}(t)$$

where:
- $J$ = total moment of inertia (kg·m²) — motor rotor + gearbox + wheel
- $\omega(t)$ = angular velocity (rad/s)
- $\tau_{motor} = K_t \cdot i(t)$ = motor torque
- $K_t$ = torque constant (N·m/A) — numerically equal to $K_e$ in SI units
- $B$ = viscous friction coefficient (N·m·s/rad)
- $\tau_{load}$ = external load torque (friction, slope, payload)

**Mechanical time constant:** $\tau_m = J \cdot R / (K_t \cdot K_e)$

For a typical OKS motor+gearbox: $\tau_m \approx 50–200$ ms. This is the dominant dynamic — mechanical response is 100–400× slower than electrical response.

## 1.4 The Key Insight: $K_t = K_e$

In SI units, the torque constant and back-EMF constant are **exactly equal**. This isn't a coincidence — it's conservation of energy:

$$P_{electrical} = e \cdot i = K_e \omega \cdot i$$
$$P_{mechanical} = \tau \cdot \omega = K_t i \cdot \omega$$

For power to be conserved: $K_e = K_t$. This constant is often called simply $K$ or $K_m$ (motor constant).

**Practical consequence:** If the datasheet gives you $K_e$ = 0.01 V·s/rad, you immediately know $K_t$ = 0.01 N·m/A. One less parameter to measure.

---

# PART 2 — TRANSFER FUNCTIONS

## 2.1 Voltage → Speed Transfer Function

Taking the Laplace transform of both equations and eliminating current:

$$\frac{\Omega(s)}{V(s)} = \frac{K_t}{(Ls + R)(Js + B) + K_t K_e}$$

This is a **second-order system** with two time constants (electrical and mechanical).

**Simplified form** (when $L \ll R$, i.e., electrical dynamics are much faster than mechanical — usually true for small motors):

$$\frac{\Omega(s)}{V(s)} \approx \frac{K_t / (RB + K_t K_e)}{\frac{RJ}{RB + K_t K_e} s + 1} = \frac{K_{motor}}{\tau_m s + 1}$$

where:
- $K_{motor} = \frac{K_t}{RB + K_t K_e}$ = DC gain (rad/s per volt)
- $\tau_m = \frac{RJ}{RB + K_t K_e}$ = electromechanical time constant

**This is a first-order system!** For most robot motor applications, you can model the motor as a simple first-order lag from voltage to speed.

## 2.2 Voltage → Position Transfer Function

Position is the integral of speed: $\Theta(s) = \Omega(s) / s$

$$\frac{\Theta(s)}{V(s)} = \frac{K_{motor}}{s(\tau_m s + 1)}$$

This has a **pole at s = 0** (the integrator from speed to position), making it a Type 1 system. This means:
- Position eventually tracks a step input with zero error (it'll get there)
- But it's **marginally stable** — without damping, a position controller can oscillate

## 2.3 Voltage → Current Transfer Function

$$\frac{I(s)}{V(s)} = \frac{Js + B}{(Ls + R)(Js + B) + K_t K_e}$$

The current has a *zero* in the numerator — it initially spikes (when you apply voltage, back-EMF is zero, so all voltage drives current) then settles as the motor spins up and back-EMF reduces the current.

```
  Current
  │
  │  peak (stall current = V/R)
  │╲
  │ ╲
  │  ╲
  │   ╲___________  steady-state (running current)
  │
  └──────────────────── t
  0   τ_e        τ_m
```

**OKS relevance:** This initial current spike is why motor drivers have **current limiting**. At stall (wheel blocked), current = V/R = 12V / 2Ω = 6A. The motor driver may only handle 3A continuous. The current PID loop (innermost loop) prevents this spike from damaging the driver.

---

# PART 3 — PRACTICAL MOTOR PARAMETERS

## 3.1 Reading a Motor Datasheet

A typical OKS motor datasheet provides:

| Parameter | Symbol | Typical value | Unit |
|-----------|--------|---------------|------|
| Nominal voltage | $V_{nom}$ | 12 | V |
| No-load speed | $\omega_{nl}$ | 6000 | RPM (= 628 rad/s) |
| No-load current | $I_{nl}$ | 0.2 | A |
| Stall torque | $\tau_{stall}$ | 0.5 | N·m |
| Stall current | $I_{stall}$ | 6.0 | A |
| Terminal resistance | $R$ | 2.0 | Ω |
| Rotor inertia | $J_{rotor}$ | 5.0 × 10⁻⁶ | kg·m² |

**Derived parameters:**

$$K_t = \frac{\tau_{stall}}{I_{stall}} = \frac{0.5}{6.0} = 0.083 \text{ N·m/A}$$

$$K_e = \frac{V_{nom} - R \cdot I_{nl}}{\omega_{nl}} = \frac{12 - 2 \times 0.2}{628} = 0.0185 \text{ V·s/rad}$$

Wait — $K_t \neq K_e$? That's because the datasheet uses RPM and we need rad/s, and there's friction loss at no-load.

**More accurate:** $K_e = K_t = 0.083$ V·s/rad. The discrepancy comes from friction ($B$) absorbing torque at no-load. Use the stall torque / stall current ratio as the most reliable estimate.

$$B = \frac{\tau_{stall} - K_t \cdot I_{stall}}{no\_load\_speed}$$

...but since $\tau_{stall} = K_t \cdot I_{stall}$ by definition, we get $B$ from the no-load torque:

$$B = \frac{K_t \cdot I_{nl}}{\omega_{nl}} = \frac{0.083 \times 0.2}{628} = 2.6 \times 10^{-5} \text{ N·m·s/rad}$$

## 3.2 The Gearbox Changes Everything

OKS motors have a **gearbox** (typical ratio $N$ = 50:1 to 100:1). The gearbox transforms:

| Parameter | Before gearbox | After gearbox |
|-----------|----------------|---------------|
| Speed | $\omega_{motor}$ | $\omega_{wheel} = \omega_{motor} / N$ |
| Torque | $\tau_{motor}$ | $\tau_{wheel} = \tau_{motor} \times N \times \eta$ |
| Inertia (reflected) | $J_{motor}$ | $J_{total} = J_{motor} \times N^2 + J_{wheel}$ |

**Critical:** Inertia reflects as $N^2$, not $N$. A 100:1 gearbox makes the motor see the wheel as 10,000× lighter. This is why geared motors can accelerate heavy robots quickly.

**The effective mechanical time constant after gearing:**

$$\tau_m = \frac{(J_{motor} N^2 + J_{wheel}) \cdot R}{K_t^2 + R \cdot B_{total}}$$

For OKS with $N = 50$, $J_{motor} = 5 \times 10^{-6}$:
- $J_{reflected} = 5 \times 10^{-6} \times 2500 = 0.0125$ kg·m²
- $J_{wheel} \approx 0.005$ kg·m²
- $J_{total} \approx 0.0175$ kg·m²
- $\tau_m \approx \frac{0.0175 \times 2}{0.083^2} \approx 5.1$ s

That's very slow! This means the motor (after gearing) takes ~5 seconds to reach full speed from a voltage step. **This is why you need a PID controller** — proportional-only control with this time constant would be painfully slow. The derivative term predicts the trajectory; the integral term eliminates steady-state error.

## 3.3 The Speed-Torque Curve

Every DC motor has a linear speed-torque relationship:

```
  Speed (ω)
  ω_nl ──────╲
  │           ╲
  │            ╲
  │             ╲
  │              ╲
  │               ╲
  │                ╲
  │                 ╲
  0 ─────────────────╲──── Torque (τ)
  0                   τ_stall
```

$$\omega = \omega_{nl} \left(1 - \frac{\tau}{\tau_{stall}}\right)$$

or equivalently: $\omega = \frac{V}{K_e} - \frac{R}{K_e K_t} \tau$

**Operating point:** The motor settles where the speed-torque line intersects the load curve. If the load increases (slope, heavier payload), the operating point moves left (lower speed, higher torque, higher current).

**OKS relevance:** When a robot carries a heavier bin, the load torque increases. Without feedback, the motor slows down. With speed PID, the controller increases PWM to maintain the setpoint — but current increases. If current exceeds the driver's limit, it saturates. This is **torque saturation** — the controller wants more torque than the motor can provide. The robot slows down despite the controller's best effort.

---

# PART 4 — SIMULATION IN PYTHON

## 4.1 Motor Step Response Simulation

```python
import numpy as np
import matplotlib.pyplot as plt

# Motor parameters (OKS-like geared motor)
R  = 2.0        # Ω
L  = 0.001      # H (1 mH — we'll include it for accuracy)
Kt = 0.083      # N·m/A = V·s/rad
Ke = Kt         # SI units
J  = 0.0175     # kg·m² (reflected through gearbox)
B  = 0.001      # N·m·s/rad (friction)

# Transfer function: V(s) → ω(s)
# G(s) = Kt / [(Ls + R)(Js + B) + KtKe]
#       = Kt / [LJs² + (LB + RJ)s + (RB + KtKe)]

a2 = L * J                    # s² coefficient
a1 = L * B + R * J            # s¹ coefficient
a0 = R * B + Kt * Ke          # s⁰ coefficient
K_dc = Kt / a0                # DC gain

print(f"DC gain: {K_dc:.2f} rad/s per volt")
print(f"Electrical τ: {L/R*1000:.2f} ms")
print(f"Mechanical τ: {R*J/(Kt*Ke)*1000:.1f} ms")

# Simulate step response using scipy
from scipy import signal

num = [Kt]
den = [a2, a1, a0]
sys = signal.TransferFunction(num, den)

t = np.linspace(0, 0.5, 5000)
t_out, y_out = signal.step(sys, T=t)

# Apply a 12V step
V_step = 12.0
speed_rpm = y_out * V_step * 60 / (2 * np.pi)  # convert to RPM

plt.figure(figsize=(10, 6))
plt.plot(t_out * 1000, speed_rpm)
plt.xlabel('Time (ms)')
plt.ylabel('Speed (RPM)')
plt.title('Motor Step Response (12V applied at t=0)')
plt.grid(True)
plt.axhline(y=speed_rpm[-1], color='r', linestyle='--',
            label=f'Steady state: {speed_rpm[-1]:.0f} RPM')
plt.legend()
plt.savefig('motor_step_response.png', dpi=100)
plt.show()
```

## 4.2 Current During Startup

```python
# Current transfer function: V(s) → I(s)
# I(s) = (Js + B) / [(Ls + R)(Js + B) + KtKe] * V(s)
num_i = [J, B]
den_i = [a2, a1, a0]
sys_i = signal.TransferFunction(num_i, den_i)

t_out, i_out = signal.step(sys_i, T=t)
current = i_out * V_step

plt.figure(figsize=(10, 6))
plt.plot(t_out * 1000, current)
plt.xlabel('Time (ms)')
plt.ylabel('Current (A)')
plt.title('Motor Current During Startup')
plt.axhline(y=V_step/R, color='r', linestyle='--',
            label=f'Stall current: {V_step/R:.1f} A')
plt.axhline(y=current[-1], color='g', linestyle='--',
            label=f'Running current: {current[-1]:.2f} A')
plt.legend()
plt.grid(True)
plt.savefig('motor_current_startup.png', dpi=100)
plt.show()
```

---

# PART 5 — NONLINEARITIES (THE REAL WORLD)

## 5.1 Things the Linear Model Ignores

The transfer function model assumes linearity. Real motors have:

| Nonlinearity | Effect | When it matters |
|-------------|--------|-----------------|
| **Stiction** | Motor doesn't move until torque exceeds static friction | Low-speed precision moves |
| **Backlash** | Gearbox has play; reversing direction → dead zone | Position control, direction changes |
| **Saturation** | PWM can't exceed 100%; current can't exceed driver limit | High-load situations |
| **Cogging** | Permanent magnets create torque ripple at certain positions | Low-speed smoothness |
| **Temperature** | R increases ~0.4%/°C; Kt changes with magnet temperature | Long duty cycles |
| **Dead zone** | Motor doesn't respond below a minimum voltage (due to friction) | Very low speed commands |

**OKS motors exhibit ALL of these.** The PID controller partially compensates, but extreme cases (stiction at very low speed, saturation on ramps) require explicit handling:
- **Dead zone compensation:** Add a minimum PWM offset when speed command > 0
- **Anti-backlash:** In position mode, always approach from the same direction
- **Current limiting:** Clamp the PID output to prevent driver damage

## 5.2 Coulomb Friction Model

Real friction isn't purely viscous ($B \cdot \omega$). It includes:

$$\tau_{friction} = B \cdot \omega + \tau_{coulomb} \cdot \text{sign}(\omega) + \tau_{stiction} \cdot \delta(\omega = 0)$$

```
  Friction torque
  │
  │  ────────────── τ_stiction (breakaway)
  │ /
  │/ τ_coulomb
  ├─────────────────────── ω
  │╲ -τ_coulomb
  │ ╲
  │  ────────────── -τ_stiction
  │
```

**The stiction problem:** At zero speed, static friction is *higher* than kinetic friction. The motor must produce more torque to *start* moving than to *keep* moving. This creates a **dead zone** around zero speed where the PID controller requests torque but the motor doesn't move. The integral term winds up. When the motor finally breaks free, the accumulated integral causes an overshoot.

---

## Checkpoint Questions

1. Write the voltage equation and torque equation for a DC motor. Identify each term.
2. What is the back-EMF and why does it make the motor self-regulating?
3. Derive the simplified (first-order) voltage→speed transfer function. What assumption allows the simplification?
4. A motor with $K_t = 0.05$ N·m/A, $R = 1.5$ Ω, $J = 0.01$ kg·m² has mechanical time constant ___?
5. Why does reflected inertia scale as $N^2$ through a gearbox?
6. The motor draws 3A at stall and 0.5A when running at full speed. Where is the missing power going?
7. Why does changing PWM frequency from 20 kHz to 50 kHz increase motor heating?

---

## Key Takeaways

- A DC motor is two coupled first-order systems: electrical ($\tau_e = L/R$) and mechanical ($\tau_m = RJ/K_t^2$)
- The simplified voltage→speed transfer function is first-order: $G(s) = K/(\tau_m s + 1)$
- $K_t = K_e$ in SI units (conservation of energy)
- Gearbox reflects inertia as $N^2$ — dominant term in the mechanical time constant
- Real motors have stiction, backlash, saturation, and cogging that the linear model doesn't capture
- The current spike at startup can be $V/R$ (6A at 12V/2Ω) — **current limiting is essential**
