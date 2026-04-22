# Exercise 3 — Discrete-Time Control

**Covers:** Lesson 04
**Difficulty:** Intermediate

---

## Problem 1: Sampling Theorem

A DC motor's mechanical bandwidth is 50 Hz (speed loop). The encoder provides position data.

a) What is the minimum sampling frequency? (Nyquist criterion)

b) What sampling frequency would you actually use? Why?

c) At your chosen sample rate, how many encoder counts does a 2048 CPR encoder produce per sample at 10 rad/s?

---

## Problem 2: Discretization Methods

The continuous PI controller is $C(s) = 2 + \frac{10}{s}$.

Discretize using:
a) Forward Euler with $T_s = 1$ ms
b) Backward Euler with $T_s = 1$ ms
c) Tustin (bilinear) with $T_s = 1$ ms

For each, write the difference equation: $u[n] = f(u[n-1], e[n], e[n-1])$.

---

## Problem 3: Velocity Form

Convert the standard PID:

$$u[n] = K_p e[n] + K_i T_s \sum e[k] + K_d \frac{e[n] - e[n-1]}{T_s}$$

to velocity (incremental) form: $\Delta u[n] = u[n] - u[n-1]$.

Show all derivation steps.

---

## Problem 4: Aliasing

An encoder at 2048 CPR is sampled at 100 Hz. The motor shaft rotates at 60 rad/s.

a) How many encoder edges occur per sample period?

b) Is the speed measurement aliased? (Apply Nyquist to the encoder signal frequency.)

c) What is the actual speed resolution at this sample rate?

---

## Problem 5: Sample Rate Selection

You're designing a cascaded control system:
- Current loop bandwidth: 2 kHz
- Speed loop bandwidth: 200 Hz  
- Position loop bandwidth: 20 Hz

What sample rates would you choose for each loop? Justify with the 10× rule.

---

## Solutions

<details>
<summary>Click to reveal solutions</summary>

**Problem 1:**

a) $f_{min} = 2 \times 50 = 100$ Hz (Nyquist)

b) $f_s = 10 \times 50 = 500$ Hz minimum. Practical choice: 1 kHz (standard for motor speed control, allows 10× bandwidth margin).

c) At 10 rad/s: counts per second = $10 \times 2048 / (2\pi) = 3260$ counts/s. At $f_s = 1$ kHz: **3.26 counts per sample** — this is low! Speed resolution is limited. Consider lower sample rate or higher CPR encoder.

**Problem 2a: Forward Euler**

$C(z) = 2 + 10 \cdot \frac{T_s}{z-1}$

$u[n] = u[n-1] + 2(e[n] - e[n-1]) + 0.01 \cdot e[n-1]$

**Problem 2c: Tustin**

$C(z) = 2 + 10 \cdot \frac{T_s(z+1)}{2(z-1)}$

$u[n] = u[n-1] + 2(e[n] - e[n-1]) + 0.005(e[n] + e[n-1])$

**Problem 5:**

Current loop: $f_s = 10 \times 2000 = 20$ kHz (typical for FOC)
Speed loop: $f_s = 10 \times 200 = 2$ kHz (or 1 kHz minimum)
Position loop: $f_s = 10 \times 20 = 200$ Hz (or up to 1 kHz)

</details>
