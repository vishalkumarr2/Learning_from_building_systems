# Exercise 6 — Cascade Controller Design

**Covers:** Lesson 07
**Difficulty:** Hard

---

## Problem 1: Bandwidth Planning

You're designing a cascade controller for a robot wheel drive:
- Motor electrical time constant: $\tau_e = 1.5$ ms
- Motor mechanical time constant: $\tau_m = 40$ ms
- Position loop required bandwidth: 5 Hz

a) What bandwidth should the current loop have? (Use the 10:1 rule.)

b) What bandwidth should the speed loop have?

c) What sample rates would you choose for each loop?

d) If the current loop bandwidth is limited to 2 kHz by the PWM frequency, what is the maximum achievable position loop bandwidth?

---

## Problem 2: Current Loop Design

The motor's electrical model is: $V = L_a \frac{di}{dt} + R_a i + V_{emf}$

With $L_a = 0.5$ mH, $R_a = 0.5 \Omega$, PWM voltage range: 0–24V, max current: 10A.

a) Design a PI current controller using the pole cancellation method:
   - Set $K_p / K_i = L_a / R_a$ (cancels the electrical pole)
   - Choose $K_p$ for a bandwidth of 2 kHz

b) What is the closed-loop current step response?

c) Add current saturation at ±10A. What anti-windup method would you use?

---

## Problem 3: Speed Loop on Top

The current loop from Problem 2 is now your "inner actuator" for the speed loop.

Model it as: $G_{current}(s) \approx \frac{1}{\tau_{cl}s + 1}$ where $\tau_{cl}$ is the closed-loop current bandwidth.

Plant: $\frac{\omega(s)}{T(s)} = \frac{1}{Js + B}$ with $J = 0.001$ kg·m², $B = 0.005$ N·m·s/rad.

a) What is the combined plant from current setpoint to speed?

b) Design a PI speed controller for 200 Hz bandwidth.

c) Simulate a speed step from 0 to 10 rad/s. What does the current setpoint look like during the transient?

---

## Problem 4: Inside-Out Tuning

You have a cascade system with current, speed, and position loops. The plant parameters are uncertain — you only know rough estimates.

Describe the step-by-step tuning procedure:
a) Which loop do you tune first?
b) How do you validate each loop before moving to the next?
c) What tests do you perform?
d) What should you look for in the frequency response?

---

## Problem 5: Cross-Loop Interaction

The speed loop commands 3A from the current loop. The current loop is saturated at 10A due to a load disturbance.

a) What happens to the speed loop's integral term?

b) How does cross-loop anti-windup work? Write pseudocode.

c) Without cross-loop anti-windup, the speed overshoots by 40% when the disturbance clears. With it, overshoot drops to 8%. Explain why.

---

## Solutions

<details>
<summary>Click to reveal solutions</summary>

**Problem 1:**

a) Current loop bandwidth = 10 × speed loop BW. Speed loop BW = 10 × position BW = 50 Hz. So current BW = 500 Hz minimum. But the electrical pole is at $1/(2\pi\tau_e) = 106$ Hz, and we want headroom. Target: 2 kHz.

b) Speed BW = 50 Hz (10× position BW of 5 Hz).

c) Current: 20 kHz (10× BW). Speed: 1–2 kHz (10× BW). Position: 200–500 Hz (10× BW).

d) Max speed BW = 2000/10 = 200 Hz. Max position BW = 200/10 = 20 Hz. So your design is feasible with headroom.

**Problem 2a:**

Pole cancellation: $T_i = L_a/R_a = 0.001$ s. 

$K_p$ for 2 kHz bandwidth: The closed-loop is approximately $\frac{1}{\tau_{cl}s + 1}$ where $\tau_{cl} = L_a / K_p$.

For 2 kHz: $\tau_{cl} = 1/(2\pi \times 2000) = 0.0000796$ s.

$K_p = L_a / \tau_{cl} = 0.0005 / 0.0000796 = 6.28$

$K_i = K_p / T_i = 6.28 / 0.001 = 6280$

**Problem 5b:**

```python
# In speed loop:
speed_error = speed_ref - speed_actual
speed_integral += speed_error * dt

# Compute ideal current command
i_cmd = Kp_speed * speed_error + Ki_speed * speed_integral

# Apply current limit
i_cmd_sat = clamp(i_cmd, -I_MAX, I_MAX)

# Cross-loop anti-windup: if current loop is saturated,
# back-calculate the speed integral
if i_cmd != i_cmd_sat:
    speed_integral -= (i_cmd - i_cmd_sat) / Ki_speed
```

</details>
