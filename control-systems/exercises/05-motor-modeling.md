# Exercise 5 — Motor Modeling and Identification

**Covers:** Lesson 02
**Difficulty:** Intermediate

---

## Problem 1: Step Response Identification

You apply a step voltage of 12V to a DC motor and record the speed:

| Time (ms) | Speed (rad/s) |
|-----------|---------------|
| 0         | 0.0           |
| 10        | 2.1           |
| 20        | 4.0           |
| 30        | 5.6           |
| 40        | 6.9           |
| 50        | 8.0           |
| 75        | 9.5           |
| 100       | 10.2          |
| 150       | 10.8          |
| 200       | 10.9          |
| 300       | 11.0          |
| 500       | 11.0          |

a) What is the steady-state speed $\omega_{ss}$?

b) Find the DC gain $K = \omega_{ss} / V_{in}$.

c) The mechanical time constant $\tau_m$ is the time to reach 63.2% of steady state. Estimate $\tau_m$ from the data.

d) Write the first-order transfer function $G(s) = \frac{K}{(\tau_m s + 1)}$.

---

## Problem 2: Python System Identification

Given the same data, write a Python script that:

1. Fits a first-order model $\omega(t) = K_{dc} V_{in} (1 - e^{-t/\tau})$ using `scipy.optimize.curve_fit`
2. Fits a second-order model and compares
3. Plots measured data, first-order fit, and second-order fit
4. Computes $R^2$ for both fits
5. Recommends which model to use

---

## Problem 3: Back-EMF Measurement

With the motor spinning at $\omega = 11.0$ rad/s under $V = 12$ V and drawing $I = 0.8$ A:

a) Calculate the back-EMF: $V_{emf} = K_e \times \omega$

b) Given $V = IR_a + V_{emf}$, find $R_a$ and $K_e$.

c) What current does the motor draw at stall ($\omega = 0$)?

d) What is the stall torque if $K_t = K_e$?

---

## Problem 4: Gearbox Modeling

The motor from Problem 1 is connected to a 10:1 gearbox (gear ratio $N = 10$).

a) What is the output shaft speed at steady state?

b) What is the output torque at stall? (Assume 85% gearbox efficiency.)

c) Write the transfer function from voltage to output shaft speed.

d) Does the gearbox change the time constant? Why or why not?

---

## Problem 5: Nonlinear Effects

Repeat the step test, but this time at $V = 2$ V (low voltage). The motor doesn't start until $V > 1.5$ V.

a) What physical phenomenon causes this?

b) How would you model this in simulation?

c) Does this affect your PID controller? If so, how?

d) What is a common workaround in motor control firmware?

---

## Solutions

<details>
<summary>Click to reveal solutions</summary>

**Problem 1:**

a) $\omega_{ss} = 11.0$ rad/s (from data at t=300ms+)

b) $K = 11.0 / 12.0 = 0.917$ rad/s/V

c) 63.2% of 11.0 = 6.95 rad/s. From the table, this occurs at $t \approx 40$ ms. So $\tau_m \approx 0.04$ s = 40 ms.

d) $G(s) = \frac{0.917}{0.04s + 1}$

**Problem 3:**

b) At steady state: $V_{emf} = K_e \times 11.0$. From $V = IR_a + V_{emf}$: $12 = 0.8 R_a + 11.0 K_e$. 

The electrical equation gives us: $K_e = V_{emf}/\omega$. We need another equation. From the DC motor model at steady state: $T_{load} = K_t I = B\omega$. If we assume small friction, most of the voltage drop is across back-EMF: $V_{emf} \approx 12 - 0.8 \times R_a$.

With typical $R_a$ values (try $R_a = 1.0\Omega$): $V_{emf} = 12 - 0.8 = 11.2$ V, $K_e = 11.2 / 11.0 = 1.018$ V·s/rad.

c) At stall: $I_{stall} = V/R_a = 12/1.0 = 12$ A

d) $T_{stall} = K_t \times I_{stall} = 1.018 \times 12 = 12.2$ N·m

**Problem 5:**

a) Static friction (stiction/Coulomb friction). The motor must overcome breakaway torque.

b) Add a dead-zone: if $|V| < V_{threshold}$, $\omega = 0$.

c) Yes — the integral term winds up trying to overcome stiction, then causes overshoot when the motor finally moves.

d) Feed-forward dead-zone compensation: add a bias voltage ($\pm V_{threshold}$) to the PID output whenever a nonzero velocity is commanded.

</details>
