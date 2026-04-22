# Debug Session 1 — Motor Oscillation: "The Wheel Won't Stop Shaking"

**Skills tested:** PID tuning, stability analysis, frequency-domain thinking
**Difficulty:** Intermediate
**Time estimate:** 45–60 minutes

---

## The Scenario

**Field report from warehouse (Site-A site):**

> Robot amr-warehouse-08 exhibits a loud buzzing sound from the right wheel at constant speed. The wheel visibly oscillates at ~30 Hz. The left wheel is fine. The robot was working normally yesterday. No firmware or software changes were deployed.

**What you have:**
- Speed PID runs at 1 kHz on the MCU
- PID gains: $K_p = 5.0$, $K_i = 25.0$, $K_d = 0.02$
- Encoder: 2048 CPR × 4 (quadrature) = 8192 counts/rev
- Motor: 24V DC, 12:1 gearbox
- Right motor was replaced last week (maintenance record)

---

## Step 1 — Observe Before Touching

### 1.1 Data Collection

You SSH into the robot and log speed data for 5 seconds at constant setpoint = 5.0 rad/s:

```
Time(ms)   Setpoint   Measured    Error    PWM_duty
0          5.000      4.82       0.180    0.412
1          5.000      5.31      -0.310    0.378
2          5.000      4.65       0.350    0.445
3          5.000      5.42      -0.420    0.361
4          5.000      4.53       0.470    0.462
5          5.000      5.55      -0.550    0.340
...
```

**Observation:** The measured speed oscillates around the setpoint with **growing** amplitude. This is an **unstable oscillation**, not just noise.

### 1.2 Frequency Analysis

```python
import numpy as np
from scipy.fft import fft, fftfreq

# Load speed data (1000 samples at 1 kHz)
speed_data = np.loadtxt('right_motor_speed.csv')
dt = 0.001
N = len(speed_data)

# Remove mean (DC component)
speed_ac = speed_data - np.mean(speed_data)

# FFT
freqs = fftfreq(N, dt)[:N//2]
spectrum = np.abs(fft(speed_ac))[:N//2] * 2/N

# Find peak frequency
peak_idx = np.argmax(spectrum[1:]) + 1  # Skip DC
peak_freq = freqs[peak_idx]
print(f"Oscillation frequency: {peak_freq:.1f} Hz")
# Result: 33.2 Hz
```

**Key finding:** Oscillation at 33.2 Hz. The PID loop runs at 1 kHz, so 33.2 Hz is well within control bandwidth.

---

## Step 2 — Form Hypotheses

| # | Hypothesis | Evidence needed |
|---|-----------|----------------|
| H1 | $K_p$ too high → marginal stability | Reduce $K_p$ → does oscillation stop? |
| H2 | Encoder noise → derivative kick | Check raw encoder counts, $K_d$ contribution |
| H3 | Mechanical resonance (new motor) | Measure with motor disabled (coast down) |
| H4 | Phase lag from computation | Measure actual loop timing |

---

## Step 3 — Investigate H1: Gain Too High

### 3.1 The Stability Theory

For a first-order plant (motor + load):

$$G(s) = \frac{K_m}{Js + B}$$

With proportional gain $K_p$, the closed-loop is:

$$T(s) = \frac{K_p K_m}{Js + B + K_p K_m}$$

This is **always stable** for first-order systems. So why is it oscillating?

**Answer:** The real system isn't first order. The encoder measurement, PWM, and motor electrical dynamics add delays. The plant is more like:

$$G(s) = \frac{K_m}{(Js + B)(\tau_e s + 1)} \cdot e^{-s T_{delay}}$$

where $\tau_e$ is the electrical time constant and $T_{delay}$ is the computational + PWM delay.

### 3.2 Why the Replacement Motor Changed Things

The maintenance record shows the right motor was replaced. The replacement motor has:
- **Different winding resistance** → different electrical time constant $\tau_e$
- **Different rotor inertia** → different mechanical time constant $J/B$

**The old motor** (lower inductance, faster electrical response) was stable with these gains. **The new motor** (higher inductance, slower electrical response) introduces more phase lag → the same gains are now marginally unstable.

### 3.3 Gain Margin Analysis

```python
import control

# Old motor parameters
Km_old = 0.05   # motor constant
J = 0.005        # inertia
B = 0.01         # friction
tau_e_old = 0.001  # 1 ms electrical time constant

# New motor parameters (higher inductance)
tau_e_new = 0.003  # 3 ms electrical time constant

# Plant transfer functions
G_old = control.tf([Km_old], [J*tau_e_old, J+B*tau_e_old, B])
G_new = control.tf([Km_old], [J*tau_e_new, J+B*tau_e_new, B])

# PID controller (continuous approximation)
Kp, Ki, Kd = 5.0, 25.0, 0.02
C = control.tf([Kd, Kp, Ki], [1, 0])

# Gain and phase margins
gm_old, pm_old, _, _ = control.margin(C * G_old)
gm_new, pm_new, _, _ = control.margin(C * G_new)

print(f"Old motor: Gain margin = {20*np.log10(gm_old):.1f} dB, Phase margin = {pm_old:.1f}°")
print(f"New motor: Gain margin = {20*np.log10(gm_new):.1f} dB, Phase margin = {pm_new:.1f}°")

# Typical result:
# Old motor: Gain margin = 12.3 dB, Phase margin = 45.2°  ← Stable
# New motor: Gain margin = 3.1 dB, Phase margin = 8.7°    ← Barely stable!
```

**Root cause:** The new motor's higher electrical time constant reduced the phase margin from 45° to ~9°, causing marginal instability. The 33 Hz oscillation is the **limit cycle** at the gain crossover frequency.

---

## Step 4 — Fix It

### 4.1 Quick Fix: Reduce Kp

Reduce $K_p$ from 5.0 to 3.0. The oscillation stops immediately. But tracking performance degrades.

### 4.2 Proper Fix: Re-tune for the New Motor

Identify the new motor's electrical time constant:

```python
# Step response test: apply constant duty, measure current rise
# Current: I(t) = I_ss * (1 - exp(-t/tau_e))
# Fit exponential to find tau_e

from scipy.optimize import curve_fit

def current_model(t, I_ss, tau_e):
    return I_ss * (1 - np.exp(-t / tau_e))

popt, _ = curve_fit(current_model, time_data, current_data, p0=[5.0, 0.002])
tau_e_measured = popt[1]
print(f"Measured tau_e = {tau_e_measured*1000:.2f} ms")
# Result: tau_e = 2.8 ms (confirms the hypothesis)
```

Re-tune PID for the new motor:
- Reduce $K_p$ to 3.5 (restore 35°+ phase margin)
- Reduce $K_d$ to 0.008 (less amplification of high-freq noise)
- Keep $K_i$ at 25.0 (steady-state accuracy preserved)

### 4.3 Root Cause Prevention

**The real fix:** Don't use fixed PID gains for different motors. Either:
1. Add motor identification to the commissioning procedure
2. Use gain scheduling based on measured motor parameters
3. Store motor-specific PID gains in EEPROM per motor slot

---

## Step 5 — Verify

After re-tuning:
```
Oscillation amplitude: 0.82 rad/s → 0.05 rad/s (noise floor)
Phase margin: 8.7° → 38.2°
Step response overshoot: 45% → 12%
Settling time: ∞ (oscillating) → 120 ms
```

---

## Lessons Learned

1. **Motor replacement changes the plant** — never assume PID gains transfer between motors.
2. **Oscillation frequency tells you where the instability is.** 33 Hz → likely related to electrical time constant or sensor delay, not mechanical resonance (which would be < 10 Hz).
3. **Phase margin < 15° is asking for trouble.** Target 30–45° for industrial motors.
4. **Limit cycles** (constant-amplitude oscillation) indicate the system is right at the stability boundary. The nonlinearity (PWM saturation) prevents the oscillation from growing to infinity.
5. **Always measure before tuning.** The FFT told us the oscillation frequency in 1 second. Without it, we'd be guessing.
