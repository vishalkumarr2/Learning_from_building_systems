# Exercise 2 — PID Tuning

**Covers:** Lesson 03
**Difficulty:** Intermediate

---

## Problem 1: Manual Tuning Sequence

You have a DC motor with unknown parameters. You start with $K_p = 0$, $K_i = 0$, $K_d = 0$.

Describe the **exact sequence** of steps you'd follow to tune the PID controller manually. For each step, state:
- Which gain you adjust
- What you observe
- How you know when to stop

---

## Problem 2: Ziegler-Nichols

You perform the Z-N oscillation test:
- With P-only control, you increase $K_p$ until sustained oscillation at $K_u = 8.0$
- The oscillation period is $T_u = 0.04$ s (25 Hz)

a) Using Z-N tuning rules, compute $K_p$, $K_i$, $K_d$ for a PID controller.

b) The Z-N method gives aggressive tuning (~25% overshoot). How would you modify the gains for a warehouse robot where overshoot must be < 5%?

---

## Problem 3: Anti-Windup Comparison

A motor runs into a wall. The speed setpoint is 5 rad/s but actual speed is 0. The PID has $K_i = 20$. The wall blocks the motor for 3 seconds, then is removed.

Compare the response (integral value and output after wall removal) for:
a) No anti-windup
b) Integral clamping ($|integral| \leq 2.0$)
c) Back-calculation ($K_b = 10$)
d) Conditional integration (stop integrating when output is saturated)

---

## Problem 4: Derivative Kick

The setpoint changes from 0 to 5 rad/s as a step. The PID has $K_d = 0.02$ and the loop runs at 1 kHz ($dt = 1$ ms).

a) What is the derivative term at the instant of the step change?

b) Why is this a problem?

c) Rewrite the derivative term to eliminate derivative kick.

---

## Problem 5: Python Simulation

Write a Python script that:
1. Simulates a first-order plant: $G(s) = \frac{10}{0.1s + 1}$
2. Implements a discrete PID with anti-windup (your choice of method)
3. Applies a step setpoint from 0 to 1.0 at t=1.0s
4. Applies a disturbance of -0.5 at t=3.0s
5. Plots setpoint, output, and control effort on the same time axis

Target: settling time < 0.5s, overshoot < 10%, zero steady-state error.

---

## Solutions

<details>
<summary>Click to reveal solutions</summary>

**Problem 2a:**

Z-N PID: $K_p = 0.6 \times K_u = 4.8$, $T_i = 0.5 \times T_u = 0.02$ s, $T_d = 0.125 \times T_u = 0.005$ s.

$K_i = K_p / T_i = 240$, $K_d = K_p \times T_d = 0.024$.

**Problem 2b:** Use the Pessen Integral Rule or multiply $K_p$ by 0.5, $K_i$ by 0.5, $K_d$ by 1.5.

**Problem 4a:**

$D = K_d \times \frac{\Delta e}{\Delta t} = 0.02 \times \frac{5.0}{0.001} = 100.0$

This is a spike of 100 (while normal output is ~0.5). It will saturate the actuator and cause a mechanical shock.

**Problem 4c:** Differentiate measurement only:

$D = -K_d \times \frac{y[n] - y[n-1]}{dt}$

</details>
