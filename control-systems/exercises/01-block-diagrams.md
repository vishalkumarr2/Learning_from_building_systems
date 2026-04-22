# Exercise 1 — Block Diagrams and Transfer Functions

**Covers:** Lessons 01–02
**Difficulty:** Foundation

---

## Problem 1: Identify the Elements

Draw the block diagram for the following verbal description:

> "A thermostat measures room temperature, compares it to the setpoint (22°C), and adjusts the heater power. The room's temperature changes based on heater power minus heat loss through the walls."

Identify: sensor, controller, plant, reference, error, feedback path.

---

## Problem 2: Transfer Function from Block Diagram

Given:

```
R(s) ──→(+)──→ [ C(s) = 5 ] ──→ [ G(s) = 1/(s+2) ] ──→ Y(s)
          ↑ (-)                                           │
          └───────────────────────────────────────────────┘
```

a) Write the closed-loop transfer function $T(s) = Y(s)/R(s)$.

b) Find the poles. Is the system stable?

c) If $R(s) = 1/s$ (unit step), what is the steady-state value of $y(t)$?

---

## Problem 3: DC Motor Transfer Function

A DC motor has: $R_a = 1.5 \Omega$, $L_a = 0.002$ H, $K_t = K_e = 0.05$ N·m/A, $J = 0.001$ kg·m², $B = 0.005$ N·m·s/rad.

a) Write the transfer function $G(s) = \Omega(s) / V(s)$.

b) What are the poles? Classify as overdamped, critically damped, or underdamped.

c) Can you neglect $L_a$? (Hint: compare electrical and mechanical time constants.)

---

## Problem 4: Cascaded Blocks

Simplify:

```
R(s) → [G1 = 2/(s+1)] → [G2 = 3/(s+5)] → Y(s)
```

What is the overall transfer function? Find the DC gain (value at $s = 0$).

---

## Problem 5: Feedback with Sensor Dynamics

```
R(s) ──→(+)──→ [ C(s) ] ──→ [ G(s) = 10/(s+1) ] ──→ Y(s)
          ↑ (-)                                        │
          └──────── [ H(s) = 1/(0.01s+1) ] ←──────────┘
```

The sensor $H(s)$ has a 10 ms time constant. At what frequency does the sensor start introducing significant phase lag (> 5°)?

---

## Solutions

<details>
<summary>Click to reveal solutions</summary>

**Problem 2:**

a) $T(s) = \frac{C \cdot G}{1 + C \cdot G} = \frac{5/(s+2)}{1 + 5/(s+2)} = \frac{5}{s+7}$

b) Pole at $s = -7$. Stable (negative real part).

c) $y_{ss} = \lim_{s \to 0} s \cdot T(s) \cdot \frac{1}{s} = T(0) = 5/7 \approx 0.714$

**Problem 3:**

a) $G(s) = \frac{K_t}{(L_a s + R_a)(Js + B) + K_t K_e}$

Numerator: $0.05$
Denominator: $(0.002s + 1.5)(0.001s + 0.005) + 0.0025$

$= 2 \times 10^{-6}s^2 + 0.0115s + 0.01$

b) $\tau_e = L_a/R_a = 1.33$ ms, $\tau_m = J/B = 200$ ms. $\tau_m \gg \tau_e$, so the system has two real poles (overdamped). The fast pole is at $\approx -R_a/L_a = -750$, the slow pole at $\approx -B/J \approx -5$.

c) Yes, $L_a$ can be neglected for speed control since $\tau_e \ll \tau_m$.

**Problem 5:**

Phase lag of 5° occurs when $\omega \tau = \tan(5°) = 0.0875$. With $\tau = 0.01$ s: $\omega = 8.75$ rad/s = **1.39 Hz**.

</details>
