# Senior Interview Questions — Control Theory Deep Dive

**Purpose:** Questions that trip up candidates who memorized textbooks but don't understand the physics.
**Level:** Senior firmware/controls engineer (5+ years)

---

## Question 1: The Unstable Stable System

> "I have a system with all poles in the left half-plane. Is it stable?"

**Expected weak answer:** "Yes, left-half-plane poles mean stable."

**Tricky follow-up:** "What if the system is nonlinear? What if there's a limit cycle? What if the poles are at $s = -0.001 \pm j1000$?"

**Deep answer:**
- LHP poles guarantee BIBO stability for **linear, time-invariant** systems only
- Nonlinear systems can have LHP linearized poles but still exhibit limit cycles, bifurcations, or chaos
- Poles at $s = -0.001 \pm j1000$ are technically stable but have a damping ratio $\zeta = 0.000001$ — practically, any noise excites the oscillation and the system "rings" indefinitely
- In the real world: component tolerances, temperature drift, and aging can shift poles. A system designed with poles barely in the LHP is one tolerance away from instability

**What you're testing:** Does the candidate understand the gap between mathematical stability and practical stability?

---

## Question 2: Phase Margin Paradox

> "My system has 60° phase margin. Is it robust?"

**Expected weak answer:** "Yes, 60° is considered good."

**Tricky follow-up:** "What if the gain margin is 1.5 dB?"

**Deep answer:**
- Phase margin and gain margin are **complementary**, not redundant
- 60° phase margin with 1.5 dB gain margin means the system is robust to phase perturbations but **fragile to gain changes** — a 20% increase in motor $K_t$ (e.g., from temperature change) could destabilize it
- The correct robustness metric is the **sensitivity peak** $M_s = \max|S(j\omega)|$ where $S = 1/(1+GC)$
- $M_s < 2.0$ (6 dB) is a much better robustness criterion than PM or GM alone
- Classic trap: a system can have infinite gain margin but poor phase margin, or vice versa

---

## Question 3: The Derivative Paradox

> "Adding derivative action always improves a PID controller's transient response. True or false?"

**Expected weak answer:** "True, derivative adds phase lead and anticipates the error."

**Deep answer:** **False** in several practical cases:

1. **Noisy measurements:** Derivative amplifies high-frequency noise. If SNR is poor, $K_d > 0$ makes things worse. The practical $K_d$ limit is set by sensor noise, not by control theory.

2. **Quantized feedback:** Encoder-based speed measurement at low speeds gives discrete jumps. Differentiating this produces large spikes.

3. **Non-minimum-phase plants:** Systems with right-half-plane zeros (e.g., unstable bicycle at low speed) have an initial response in the **wrong direction**. Derivative action amplifies this wrong-direction transient.

4. **Integrating plants:** For a motor position loop (plant is $1/s$), PI is often sufficient. Adding D can cause high-frequency oscillation without improving settling time.

**What you're testing:** Does the candidate know when to **not** use a tool?

---

## Question 4: Ziegler-Nichols in 2025

> "Would you use Ziegler-Nichols tuning for a production robot motor controller?"

**Expected weak answer:** "Yes, it's a standard method" or "No, it's outdated."

**Deep answer:** "It depends, but probably not as the final tuning."

- Z-N was designed in 1942 for **chemical process control** (slow plants, minutes-scale time constants)
- It gives ~25% overshoot by design — unacceptable for precise motor control
- It requires oscillation testing which can be **destructive** for mechanical systems
- **But:** Z-N is valuable as a **starting point** for manual tuning or as a sanity check
- Better alternatives for motor control:
  - **Relay feedback** (modified Z-N without sustained oscillation)
  - **Frequency response identification** + loop-shaping
  - **Model-based tuning** (identify plant, compute optimal PID)
  - **Auto-tuning** (embedded identification + optimization)
- In production OKS robots: we use model-based tuning with per-robot calibration, not Z-N

---

## Question 5: The Sampling Trap

> "I'm running my PID at 10 kHz. My plant bandwidth is 50 Hz. Is 10 kHz overkill?"

**Expected weak answer:** "10 kHz is 200× the bandwidth, so yes, it's overkill."

**Deep answer:** It depends on **which loop**:

- For a **speed loop** (plant BW 50 Hz): 10 kHz is 200× Nyquist, which is more than needed. 1 kHz would be fine.
- For a **current loop** in FOC: electrical time constants are 0.1–1 ms, so the plant bandwidth could be 1–10 kHz. Here 10 kHz is barely enough, and 20 kHz is better.
- The 10 kHz is also needed for **PWM synchronization**: the PID should run synchronously with the PWM carrier to avoid subharmonic injection and current ripple.
- Even for a 50 Hz speed loop, 10 kHz isn't wasted if you're doing **multi-rate control**: the 10 kHz ISR handles current control and provides a clean current measurement to the 1 kHz speed loop.

**Trap:** The candidate might say "just use Nyquist: 2 × 50 = 100 Hz is enough." This is the sampling theorem applied incorrectly — you need 10× bandwidth for control, not 2×.

---

## Question 6: Right-Half-Plane Zero

> "The closed-loop step response initially goes in the wrong direction before correcting. What's happening?"

**Expected weak answer:** "Maybe the gains are wrong" or "It's overshoot."

**Deep answer:** This is the **inverse response** caused by a right-half-plane (RHP) zero:

- If the plant transfer function has a zero at $s = +a$ (positive real), the step response includes a term $-e^{-at}$ that initially drives the output backward
- Physical example: a boiler — increasing fuel first cools the drum briefly (expansion absorbs heat) before temperature rises
- Robotics example: flexible drive shaft — applying torque first compresses the shaft (negative displacement) before the load starts moving
- **Critical implication:** You **cannot** design an arbitrarily fast controller for a plant with RHP zeros. The maximum bandwidth is limited to roughly $\omega_{BW} < a/2$ where $a$ is the RHP zero location

**What you're testing:** Can the candidate diagnose a real physical phenomenon from a step response without a Bode plot?

---

## Question 7: Pole-Zero Cancellation Danger

> "I have a plant with a slow pole at $s = -0.1$. I'll add a zero at $s = -0.1$ in my controller to cancel it. Good idea?"

**Expected weak answer:** "Yes, pole-zero cancellation simplifies the system."

**Deep answer:** **Dangerous if the pole is unstable or poorly known:**

1. **Exact cancellation is impossible** in practice. You can never match the pole exactly due to parameter uncertainty. With $±5%$ tolerance on the motor constant, your zero might be at $s = -0.095$ while the pole is at $s = -0.105$.

2. **If the plant pole were unstable** (RHP), cancellation would hide an unstable mode. It would disappear from the transfer function but would still be **internally excited** by noise, growing exponentially.

3. **Even for stable poles:** imperfect cancellation leaves a slow "tail" in the transient response that can take 10× longer than expected to settle.

4. **When it IS acceptable:** Cancelling a well-known, stable pole (like the electrical pole in a current loop where $L/R$ is precisely measured) is standard practice.

---

## Question 8: The Waterbed Effect

> "I've reduced the sensitivity function $|S(j\omega)|$ below 0 dB at low frequencies, making the system very good at rejecting low-frequency disturbances. Any downside?"

**Expected weak answer:** "Sounds good, no downside."

**Deep answer:** The **Bode Sensitivity Integral** (waterbed effect):

$$\int_0^{\infty} \ln|S(j\omega)| \, d\omega = \pi \sum_k p_k$$

For a stable plant (no RHP poles), the integral equals zero. This means: **if you push sensitivity down at one frequency, it must come up at another.**

- Reducing low-frequency sensitivity (good disturbance rejection) forces **increased sensitivity at higher frequencies** (noise amplification)
- With RHP poles, it's even worse — the integral is positive, meaning you have a net penalty
- Practical implication: you can't have a controller that's good at everything. There's always a tradeoff.

---

## Question 9: Digital Control Delay

> "I switch from an analog PID to a digital PID running at 1 kHz. Everything else is the same. Why did my phase margin drop?"

**Expected weak answer:** "Maybe a software bug" or "Sampling effects."

**Deep answer:** **Computation delay + ZOH delay.**

The digital controller introduces at least 1.5 sample periods of delay:
- 0.5 $T_s$ from the zero-order hold (average delay of sample-and-hold)
- 1.0 $T_s$ from computation delay (sample → compute → output happens one period later)

At $T_s = 1$ ms: total delay ≈ 1.5 ms.

Phase lag from delay: $\Delta\phi = -\omega \times \tau_{delay}$ (in radians)

At the gain crossover frequency (say 200 Hz = 1257 rad/s):

$\Delta\phi = -1257 \times 0.0015 = -1.89$ rad $= -108°$

That's a **catastrophic** loss of phase margin. Even at 50 Hz crossover:

$\Delta\phi = -314 \times 0.0015 = -0.47$ rad $= -27°$

This is why digital controllers must run much faster than the bandwidth, and why the "10× rule" exists.

---

## Question 10: Bumpless Transfer

> "The robot switches from manual joystick control to autonomous mode. The motors jerk violently on switchover. Why?"

**Expected weak answer:** "The PID starts from zero" or "Gains are too high."

**Deep answer:** **Integral state discontinuity.**

When switching to auto:
1. The PID integral is zero (just initialized)
2. But the motor is already running at some speed, requiring a nonzero steady-state output
3. The PID immediately outputs a large error × $K_p$ (since it doesn't know the motor is already running)
4. The integral starts winding up from zero

**Fix — Bumpless transfer:**
```
On switch to auto:
  pid.output = current_actuator_value  // Match current output
  pid.integral = current_actuator_value - Kp * current_error  // Back-calculate
  pid.setpoint = current_measurement   // Start from where you are
  // Then ramp setpoint to desired value
```

This ensures the PID output is continuous across the mode switch.
