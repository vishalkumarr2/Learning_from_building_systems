# 02 — The Kalman Filter & EKF
### Why OKS covariance grows during motion and shrinks on measurement

**Prerequisite:** `01-dead-reckoning.md` — unicycle motion model, arc integration, dead-reckoning drift
**Unlocks:** `03-measurement-models.md` — sensorbar line constraints, Mahalanobis gating, innovation residuals

---

## Why Should I Care? (OKS Project Context)

Every `ERROR_INVALID_STATE` ticket you investigate is, at its core, a Kalman filter
story: something caused `P` (the covariance matrix) to blow up to ∞. Before you can read a
covariance trace from a bag file and say *"this was a slip event, not a delocalization"* you
need to understand **what P represents**, **how it grows**, and **under what conditions it is
forced to ∞**.

Concretely, the the navigation estimator is an Extended Kalman Filter running at the odometry rate (~25 Hz).
The pattern you will see in every failing bag:

```
[OK]   P(x,x) ~0.01   →  robot moving, P grows slowly to ~0.05
[OK]   P(x,x) ~0.05   →  sensorbar update, P drops back to ~0.01
[FAIL] P(x,x) = inf   →  slip_detection OR collision_detection forced P→∞
[FAIL] isFinite(P) = false →  fires ERROR_INVALID_STATE
```

After this module you will be able to:
1. Predict, by inspection, whether a covariance trace is "normal dead-reckoning drift" or "anomaly"
2. Identify which code path set P to ∞ from the covariance value alone
3. Explain why the OKS shortcut `min(cov_theta, 1e5) * delta_trans²` is a valid approximation
   to the full EKF Jacobian term

---

# PART 1 — THE KALMAN FILTER INTUITION

## 1.1 A Robot's Pose Is a Probability Distribution, Not a Point

The dead-reckoning integrator from Module 01 gives you a single pose estimate:
`(x, y, θ) = (2.31, 0.87, 0.14 rad)`. That looks precise. But it carries error.

The Kalman filter forces you to be honest: **the robot's true pose is unknown**. What you have
is a *belief* — a probability distribution over all possible poses. The filter tracks:

- **The mean** `x̂` — your best single-point estimate
- **The covariance** `P` — how uncertain you are, and in which directions

For a 1D example (robot on a line, only tracking position x):

```
Uncertainty σ = 0.3 m  →  P(x,x) = σ² = 0.09 m²

         Belief: x ~ N(5.0, 0.09)

    Probability
    density
         │
   0.13  │              ██
         │            ██████
   0.09  │          ██████████
         │        ██████████████
   0.05  │      ██████████████████
         │    ████████████████████████
   0.01  │ ██████████████████████████████
         └──────────────────────────────── x (m)
              4.1  4.4  4.7  5.0  5.3  5.6  5.9
                               ↑
                           Best estimate
                          (mean = 5.0 m)
```

**The shading represents probability mass.** 68% of the area falls within ±σ = ±0.3 m of the
mean. In OKS terms: `P(x,x) = 0.09` means `σ = 0.30 m`, so the robot is within ±0.30 m of
the estimated position with 68% probability.

## 1.2 Two Sources of Information

The filter combines two independent sources of information about the robot's pose:

| Source | How it works | Error grows over time? |
|--------|-------------|----------------------|
| **Motion model** (prediction) | Integrate wheel odometry | YES — dead-reckoning drift accumulates |
| **Sensor measurement** (update) | Compare prediction to sensorbar line | NO — resets uncertainty each time |

The key insight that makes Kalman filtering work:

> **Two uncertain estimates, when independent, can be combined into one estimate that is MORE
> certain than either alone.**

## 1.3 Fusion of Two Gaussians

Suppose you have two estimates of the same quantity x:

- Estimate A: mean = 4.0, variance = 1.0  (dead-reckoning, vague)
- Estimate B: mean = 5.0, variance = 0.25 (sensorbar, sharper)

```
           A: N(4.0, 1.0)           B: N(5.0, 0.25)
         (wide, from odometry)    (narrow, from sensor)

Prob  ↑                              ↑
      │     ▓▓▓                      │      ▓
      │   ▓▓▓▓▓▓▓                    │    ▓▓▓▓▓
      │  ▓▓▓▓▓▓▓▓▓▓                  │   ▓▓▓▓▓▓▓
      │ ▓▓▓▓▓▓▓▓▓▓▓▓▓                │  ▓▓▓▓▓▓▓▓▓
      └──────────────── x            └──────────────── x
          2   3   4   5                  4   5   6

          Fused: N(4.8, 0.2)
          ↑ Sharper than either source!

Prob  ↑
      │         ▓▓▓
      │       ▓▓▓▓▓▓▓
      │      ▓▓▓▓▓▓▓▓▓▓
      └──────────────────── x
              4.3 4.8 5.3
```

The fused result (mean = 4.8, variance = 0.2) is:
- Pulled toward B (the more certain estimate)
- Narrower than either source alone

## 1.4 The Weighting Formula

The fusion weights are the **Kalman gain** K:

```
          P_A
K = ───────────────    →   x_fused = x_A + K × (x_B - x_A)
       P_A + P_B

With numbers:
K = 1.0 / (1.0 + 0.25) = 0.80

x_fused = 4.0 + 0.80 × (5.0 - 4.0) = 4.0 + 0.80 = 4.80  ✓

P_fused = (1 - K) × P_A = (1 - 0.80) × 1.0 = 0.20        ✓
```

**Key insight:** When the measurement is perfect (P_B → 0), K → 1 and we trust the sensor
completely. When the measurement is terrible (P_B → ∞), K → 0 and we trust only odometry.

---

# PART 2 — KALMAN FILTER EQUATIONS (LINEAR CASE)

## 2.1 State and Notation

State vector for a 1D robot tracking position and velocity:

```
        ┌   ┐
x̂  =   │ x │   (position estimate, m)
        │ ẋ │   (velocity estimate, m/s)
        └   ┘

P  =   ┌  P_xx   P_xẋ  ┐   (2×2 covariance matrix)
       │  P_ẋx   P_ẋẋ  │
       └                ┘

P_xx  = variance of position
P_ẋẋ  = variance of velocity
P_xẋ  = cross-covariance (how position error correlates with velocity error)
```

## 2.2 Prediction Step

Apply the motion model to propagate both the mean and the covariance forward by one time step Δt:

```
State prediction:
    x̂⁻  =  F x̂                             ... (1)

Covariance prediction:
    P⁻   =  F P Fᵀ + Q                     ... (2)
```

**F** — State transition matrix (encodes the motion model):

```
For constant-velocity model:

F = ┌ 1   Δt ┐     x_new = x_old + ẋ·Δt
    │ 0    1 ┘     ẋ_new = ẋ_old  (velocity unchanged)
```

**Q** — Process noise covariance (how much uncertainty to ADD each step from model errors):

```
Q = ┌ σ²_pos    0    ┐     (position noise, e.g. from wheel slip)
    │   0    σ²_vel  ┘     (velocity noise, e.g. from acceleration)
```

**Physical meaning of Eq. (2):**
- `F P Fᵀ` — transforms the existing uncertainty through the motion model
- `+ Q` — ADDS uncertainty because the motion model is imperfect

```
BEFORE prediction:                    AFTER prediction:
uncertainty ellipse (small)          uncertainty ellipse (larger)

     θ                                     θ
     │     ●                               │        ●
     │    ███                              │     ████████
     │     ●                               │        ●
     └──────── x                           └──────────── x
               σ_x small                           σ_x grew by √Q_xx
```

> OKS note: `predict()` computes `P⁻ = F P Fᵀ + Q` where Q is built from
> `k_trans_noise`, `k_trans_rot_noise`, `k_rot_trans_noise`,
> `k_rot_rot_noise`, and a time-based noise `k_time_trans_noise`.
> The staleness scaling: if odom is late by >2× its period, all Q terms are multiplied by
> `scaling_factor` to inflate variance proportionally.

## 2.3 Update Step

When a measurement z arrives, use it to correct the prediction:

```
Innovation (prediction error):
    ν  =  z - H x̂⁻                          ... (3)

Kalman gain:
    K  =  P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹             ... (4)

State update:
    x̂  =  x̂⁻ + K ν                          ... (5)

Covariance update:
    P  =  (I - KH) P⁻                        ... (6)
```

**H** — Measurement matrix: maps the state to what the sensor measures.

```
If we only measure position (not velocity) via GPS:

H = [ 1   0 ]     z = x_true + noise ~ N(0, R)
```

**R** — Measurement noise covariance: how noisy is the sensor?

```
R = [σ²_gps]   (e.g., 0.25 m²  →  σ_gps = 0.5 m)
```

**ν = z − H x̂⁻** — The *innovation*: how far off was our prediction from what we measured?

```
Prediction said:  x̂⁻ = 5.0 m
GPS measured:     z   = 5.3 m
Innovation:       ν   = 5.3 - 5.0 = 0.3 m   ← "we were 0.3 m off"
```

**Physical meaning of Eq. (6):**
After absorbing the measurement, covariance SHRINKS:
```
    P = P⁻ - K H P⁻          (the "K H P⁻" term is always subtracted)

    More precise measurement (small R)  →  large K  →  large reduction
    Worse measurement (large R)         →  small K  →  small reduction
```

```
BEFORE update:                         AFTER update:
(prediction, uncertain)              (measurement absorbed, confident)

     θ                                     θ
     │        ●                            │    ●
     │     ████████                        │   ███
     │        ●                            │    ●
     └──────────── x                       └──────── x
           σ_x large                           σ_x small
```

## 2.4 Worked Example — 1D Robot: One Full Predict/Update Cycle

**Setup:** Robot moving at ~1 m/s. Δt = 0.1 s.

```
Initial state:
    x̂ = [2.0 m,  1.0 m/s]ᵀ

Initial covariance:
    P = ┌ 0.04   0.0 ┐   (σ_x = 0.2 m,  σ_ẋ = 0.1 m/s)
        │ 0.0   0.01 ┘

Parameters:
    F = ┌ 1   0.1 ┐     Q = ┌ 0.001   0.0 ┐     R = [0.09]  (σ_gps = 0.3 m)
        │ 0   1.0 ┘         │  0.0   0.001┘

GPS measurement arrives: z = 2.12 m
```

**STEP 1: Predict**

```
x̂⁻ = F x̂ = ┌ 1   0.1 ┐ ┌ 2.0 ┐ = ┌ 2.0 + 0.1×1.0 ┐ = ┌ 2.10 ┐
             │ 0   1.0 ┘ │ 1.0 ┘   │ 0.0 + 1.0×1.0 ┘   │ 1.00 ┘

P⁻ = F P Fᵀ + Q

F P = ┌ 1  0.1 ┐ ┌ 0.04  0.0 ┐ = ┌ 0.04+0.1×0.0   0.0+0.1×0.01 ┐
      │ 0  1.0 ┘ │ 0.0  0.01 ┘   │ 0.0             0.01          ┘
    = ┌ 0.040   0.001 ┐
      │ 0.000   0.010 ┘

F P Fᵀ = ┌ 0.040  0.001 ┐ ┌ 1   0 ┐ = ┌ 0.040+0.001×0.1  0.001 ┐
          │ 0.000  0.010 ┘ │ 0.1 1 ┘   │ 0.000+0.010×0.1  0.010 ┘
        = ┌ 0.0401   0.001 ┐
          │ 0.001    0.010 ┘

P⁻ = F P Fᵀ + Q = ┌ 0.0401+0.001  0.001 ┐ = ┌ 0.0411   0.001 ┐
                   │ 0.001         0.011 ┘   │ 0.001    0.011 ┘
```

After prediction: σ_x grew from 0.200 m to √0.0411 = **0.203 m** (tiny — Δt was short)

**STEP 2: Update**

```
H = [1  0]   (we only measure position)

Innovation:
    ν = z - H x̂⁻ = 2.12 - [1 0]×[2.10, 1.00]ᵀ = 2.12 - 2.10 = 0.02 m

Innovation covariance:
    S = H P⁻ Hᵀ + R = [1 0] ┌0.0411  0.001┐ [1]ᵀ + [0.09]
                             │0.001   0.011┘ [0]
      = 0.0411 + 0.09 = 0.1311

Kalman gain:
    K = P⁻ Hᵀ S⁻¹ = ┌0.0411┐ × (1/0.1311) = ┌ 0.3135 ┐
                     │0.001 ┘                 │ 0.0076 ┘

State update:
    x̂ = x̂⁻ + K ν = ┌2.10┐ + ┌0.3135┐ × 0.02 = ┌ 2.1063 ┐
                     │1.00┘   │0.0076┘             │ 1.0002 ┘

Covariance update:
    I - KH = ┌1 0┐ - ┌0.3135┐[1 0] = ┌1-0.3135  0┐ = ┌0.6865   0┐
             │0 1┘   │0.0076┘         │0-0.0076  1┘   │-0.0076  1┘

    P = (I-KH) P⁻ = ┌0.6865   0 ┐ ┌0.0411  0.001┐
                     │-0.0076  1 ┘ │0.001   0.011┘
      ≈ ┌ 0.0282   0.000687 ┐
        │ -0.000313  0.01089 ┘
```

After update: σ_x dropped from **0.203 m** to **√0.0282 = 0.168 m** — the GPS helped.

---

# PART 3 — EXTENDED KALMAN FILTER (EKF) FOR NONLINEAR SYSTEMS

## 3.1 Why Linear KF Fails for Robots

The linear KF assumes the motion model is a matrix multiplication: `x̂⁻ = F x̂`. But the
unicycle motion model from Module 01 is nonlinear:

```
  x_new = x + s × cos(θ + Δθ/2)
  y_new = y + s × sin(θ + Δθ/2)
  θ_new = θ + Δθ

where s = arc length, Δθ = heading change
```

The sin/cos of the current heading angle θ means we **cannot** write this as a matrix times the
state vector. If we pretend it's linear and use `F × [x, y, θ]ᵀ`, we will get the wrong answer,
especially when uncertainty in θ is large.

## 3.2 EKF Solution: Linearize Around the Current Estimate

The EKF solution: **linearize** the nonlinear function `f(x)` around the current estimate `x̂`
using the first-order Taylor expansion:

```
f(x) ≈ f(x̂) + F × (x - x̂)

where F = Jacobian of f evaluated at x̂:

    ∂f₁/∂x₁   ∂f₁/∂x₂   ∂f₁/∂x₃
F = ∂f₂/∂x₁   ∂f₂/∂x₂   ∂f₂/∂x₃
    ∂f₃/∂x₁   ∂f₃/∂x₂   ∂f₃/∂x₃
```

The EKF prediction equations become:
```
    x̂⁻  =  f(x̂)           ← use the FULL nonlinear model for the mean
    P⁻   =  F P Fᵀ + Q     ← use the LINEARIZED model (Jacobian) for the covariance
```

## 3.3 Jacobian of the Unicycle Model

From the unicycle equations above, the partial derivatives with respect to state `[x, y, θ]`:

```
∂x_new/∂x = 1,   ∂x_new/∂y = 0,   ∂x_new/∂θ = -s × sin(θ + Δθ/2)
∂y_new/∂x = 0,   ∂y_new/∂y = 1,   ∂y_new/∂θ =  s × cos(θ + Δθ/2)
∂θ_new/∂x = 0,   ∂θ_new/∂y = 0,   ∂θ_new/∂θ = 1

    ┌  1    0    -s·sin(θ+Δθ/2) ┐
F = │  0    1     s·cos(θ+Δθ/2) │
    │  0    0          1        ┘
```

The critical off-diagonal term is in row 0 or row 1, column 2 (θ):

```
∂x_new/∂θ = -s·sin(θ + Δθ/2)   ≈  -s·sin(θ)   for small Δθ
∂y_new/∂θ =  s·cos(θ + Δθ/2)   ≈   s·cos(θ)   for small Δθ
```

**Physical meaning:** If I move forward a distance `s` and I'm uncertain about my heading by
δθ, my lateral position uncertainty grows by `s × δθ`. This is the source of lateral drift in
dead-reckoning.

## 3.4 EKF Covariance Propagation in Full

After applying `P⁻ = F P Fᵀ + Q`, the new lateral covariance (for a robot driving roughly
along the x-axis) expands approximately as:

```
P⁻_yy  ≈  P_yy  +  (∂y/∂θ)² × P_θθ  +  Q_yy

         = P_yy  +  s²·cos²(θ) × P_θθ  +  Q_yy

         ≈ P_yy  +  s² × P_θθ  +  Q_yy    (for θ ≈ 0)
                   ─────────────
                   This is the dominant cross-term:
                   lateral uncertainty grows quadratically with distance
                   when heading uncertainty is large
```

## 3.5 The OKS Shortcut

**OKS does NOT compute the Jacobian.** Instead, `predict()` uses a fixed noise model:

```cpp
// From OKS  (simplified)
lateral_var = k_trans_lat_noise * delta_trans
            + k_rot_lat_noise   * delta_rot
            + min(cov_theta, 1e5)        * delta_trans * delta_trans
                  ────────────────────────────────────────────────────
                  ↑ This IS the EKF Jacobian term (∂y/∂θ)² × P_θθ ≈ s² × P_θθ
```

The term `min(cov_theta, 1e5) * delta_trans²` is mathematically equivalent to the EKF
cross-covariance contribution `(∂y_new/∂θ)² × P_θθ = s² × P_θθ`, with the simplification
that it uses the current heading covariance `P_θθ` directly (no sin/cos approximation needed
when the robot drives mostly straight).

**The `min(..., 1e5)` cap** prevents runaway variance growth during a slip event: if P_θθ
was already set to ∞ by slip detection, this cap keeps the lateral variance growth bounded
at `1e5 × delta_trans²` per step.

## 3.6 When the EKF Approximation Breaks

The linearization `F P Fᵀ` is only valid when the uncertainty `P` is **small relative to
the curvature of f**. In practice, for robots:

```
✓ Safe: σ_θ < 5°  (P_θθ < 0.008 rad²)  — linearization error is negligible
⚠ Caution: σ_θ ≈ 10-20°  — lateral covariance underestimated
✗ Broken: σ_θ > 45°  — the Gaussian is no longer a good representation of the belief

Visualization:
                   True belief                    Gaussian approximation

    y   (bimodal after severe                  y  (single mode, too narrow)
         heading ambiguity)
    │      ●●●        ●●●                      │
    │     ●●●●●●    ●●●●●●                     │        ●●●●●
    │         ●●●●●●●                          │      ●●●●●●●●●
    └──────────────────── x                    └──────────────── x
```

In OKS, this situation is pre-empted: if `P_θθ` grows large enough to be capped by the
`min(..., 1e5)` guard, or slip detection fires, the state is flagged as unreliable before
the approximation breaks catastrophically.

---

# PART 4 — READING COVARIANCE TRACES

## 4.1 Interpreting Covariance Values

```
P(x,x) value    σ = √P(x,x)    Meaning
─────────────   ───────────    ──────────────────────────────────────────────
0.0001          0.01 m         After sensorbar update — very tight
0.0025          0.05 m         Typical after recent sensorbar update
0.01            0.10 m         Normal during short motion segment
0.04            0.20 m         Moderate dead-reckoning accumulation (caution)
0.25            0.50 m         Long travel without sensorbar — borderline
1.0             1.00 m         Bad — need localization ASAP
∞ (inf)         ∞              Slip/collision detection fired
```

**Key insight:** In the OKS planner, obstacles are avoided using the estimated pose. If
`P(x,x) = 0.25` and the robot is nominally 0.4 m from a shelf, the 2-sigma envelope extends
to `0.4 - 2×0.5 = -0.6 m` — inside the shelf. This is why large covariance causes planning
failures before it causes a physical collision.

## 4.2 What Normal Growth Looks Like

Between sensorbar updates, covariance grows approximately linearly with distance traveled:

```
P(x,x)_new ≈ P(x,x)_old + k_trans_noise × delta_trans
                         + min(P_θθ, 1e5) × delta_trans²
```

For typical OKS parameters (`k_trans_noise = 0.01`) and short moves:
- Moving 0.1 m: ΔP_xx ≈ 0.01 × 0.1 = 0.001 → negligible
- Moving 1.0 m: ΔP_xx ≈ 0.01 × 1.0 = 0.010 → noticeable
- Moving 5.0 m: ΔP_xx ≈ 0.01 × 5.0 = 0.050 → significant

## 4.3 ASCII Plot: Covariance Time Trace

```
P(x,x)
  ↑
  │
0.30 ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  (planning threshold)
  │
0.20 ─                                          ╭────── slip detection fires
  │                                         ╭───╯
  │                                     ╭───╯   ← covariance growing fast
0.10 ─          ╭────╮          ╭───────╯         (no sensorbar on this aisle)
  │             │    │          │
  │      ╭──────╯    ╰──────────╯
0.02 ─ ──╯                                       ← after sensorbar update
  │
0.00 ─────┬──────────┬──────────┬────────────────── time (s)
         t=0        t=5        t=10
          ↑          ↑          ↑
      robot starts   sensorbar  sensorbar
      moving         update     missed
                  (P drops)    (P keeps growing)

KEY: ────── = normal    ────↑─ = measurement update   ──X─ = slip event
```

## 4.4 Sudden Jumps to INF

When you see `P(x,x) = inf` in a bag, there are exactly two OKS code paths that write ∞:

**Path A — Slip detection (velocity window check in `odometryCallback()`):**
```
Trigger: |v_window_mean - v_current| > slip_threshold
Effect:  P(x,x) = INF, P(y,y) = INF, P(θ,θ) = INF
Why:     Wheel slipped — odometry is unreliable, position completely unknown
```

**Path B — Collision detection (gyro discontinuity in `imuCallback()`):**
```
Trigger: |ω_imu - ω_predicted| > collision_threshold
Effect:  transitions estimator to terminal/collision state, P → ∞
Why:     Unexpected rotation = likely collision = pose is unknown
```

**Path C — Staleness (odom not arriving):**
```
Trigger: odom message period > 2× expected period
Effect:  scaling_factor × Q applied — P grows, but not necessarily to INF
         (unless this persists for many cycles)
Why:     Cannot dead-reckon without odometry
```

To distinguish A from B in a bag:
1. Check if `/imu` topic shows a rotation spike (Path B)
2. Check if `/odom` shows velocity discontinuity with no IMU anomaly (Path A)
3. Check if `/odom` has gaps > 2× expected period (Path C)

## 4.5 The `isFinite()` Guard

In ``, **before every motion command** (and often in the control loop),
the estimator checks:

```cpp
if (!nav_state.isFinite()) {
    // fires ERROR_INVALID_STATE
    // robot stops, recovery behavior triggered
}
```

`isFinite()` returns false if ANY of the following are NaN or ±Inf:
- Position: x, y, θ
- Velocity: v_x, v_y, ω
- Any diagonal element of P: P(x,x), P(y,y), P(θ,θ)

**This means:** Even if only `P(θ,θ)` is ∞ (e.g., from a small heading slip), the ENTIRE
estimated state is considered invalid and navigation halts. This is conservative by design.

---

# PART 5 — OKS CODE CONNECTION

## 5.1 KF Concept → OKS Code Mapping

| KF Concept | OKS Code | Location | Notes |
|-----------|----------|----------|-------|
| State vector `x̂` | `NavigatorState.pose` (x, y, θ, v, ω) | `` | 5-element state |
| State transition `f(x̂)` | Unicycle arc integration | `predict()` | Nonlinear — applied to mean directly |
| Jacobian `F = ∂f/∂x` | **NOT computed** | — | OKS uses fixed noise model instead |
| Process noise `Q` | Fixed noise params: `k_trans_noise`, etc. | `predict()` | Parameters from config YAML |
| Covariance prediction `P⁻ = FPFᵀ + Q` | `noise_model.apply(delta_trans, delta_rot, delta_t)` | `predict()` | Q dominant; FPFᵀ approximated via OKS shortcut |
| Measurement matrix `H` | Sensorbar line constraint | `update()` | Projects state to line offset |
| Measurement noise `R` | Sensorbar uncertainty params | `update()` | From sensor calibration |
| Innovation `ν = z - Hx̂⁻` | Sensorbar offset residual | `update()` | Gated by Mahalanobis distance |
| Kalman gain `K` | Computed in `update()` | `update()` | Standard formula |
| State update `x̂ = x̂⁻ + Kν` | Pose correction from sensorbar | `update()` | Applied to mean |
| Covariance update `P = (I-KH)P⁻` | Covariance shrink after sensorbar | `update()` | Standard formula |
| `P → ∞` | Slip detection or collision state | `odometryCallback()` / `imuCallback()` | Writes literal `std::numeric_limits<double>::infinity()` |
| `isFinite(P)` check | `nav_state.isFinite()` | `` | Fires `ERROR_INVALID_STATE` |
| Odom staleness | Scaling factor on Q | `predict()` | If dt > 2× period |

## 5.2 The OKS Noise Model vs. the Textbook EKF

Textbook EKF:
```
P⁻ = F P Fᵀ + Q_odometry
       ↑
   F = Jacobian of unicycle model (computed at runtime from θ, s)
   Q_odometry = noise proportional to wheel encoder resolution
```

OKS implementation:
```
P⁻_xx += k_trans_noise * delta_trans
       +  min(P_θθ, 1e5) * delta_trans²         ← ≈ F P Fᵀ cross-term
       +  k_time_trans_noise * delta_t          ← time-based noise

P⁻_θθ += k_rot_rot_noise * |delta_rot|
       +  k_trans_rot_noise * delta_trans
```

**Why this approximation is valid:** For an OKS robot traveling mostly in straight lines along
aisle corridors, θ ≈ 0 for the majority of motion. Under this condition:
- `cos²(θ) ≈ 1` so lateral ≈ y
- `sin²(θ) ≈ 0` so longitudinal ≈ x
- The OKS `delta_trans²` term correctly captures lateral growth proportional to `P_θθ`

The approximation degrades during sharp turns, but sensorbar updates on line crossings
typically reset the covariance before the approximation error accumulates significantly.

## 5.3 OKS Does NOT Use odom Covariance

The `/odom` ROS message has a `pose.covariance` field (6×6 matrix). OKS **ignores this**.
Instead, noise parameters are loaded from the configuration file and treated as constants.

```
odom.pose.covariance  ──IGNORED──→  OKS uses fixed yaml params instead:
                                        k_trans_noise: 0.01
                                        k_rot_rot_noise:     0.05
                                        ...
```

This is a deliberate design choice: the wheel encoder publishers on OKS robots do not
provide reliable covariance estimates, so a calibrated fixed model is more trustworthy.

---

## Implementation Connection

Consolidated reference for connecting this module's theory to the codebase:

```
File: 
├── predict()
│   ├── Applies unicycle f(x̂) to propagate mean state
│   ├── Builds Q from: k_trans_noise, k_rot_rot_noise,
│   │                  k_trans_lat_noise, k_time_trans_noise
│   ├── Key term: min(cov_theta, 1e5) * delta_trans² (EKF Jacobian approximation)
│   └── Staleness: if dt > 2×period → Q *= scaling_factor
│
├── update()
│   ├── Takes sensorbar measurement z (line offset + orientation)
│   ├── Computes innovation ν = z - H x̂⁻
│   ├── Computes Kalman gain K = P⁻Hᵀ(HP⁻Hᵀ + R)⁻¹
│   ├── Applies state update x̂ = x̂⁻ + Kν
│   └── Applies covariance update P = (I-KH)P⁻
│
├── odometryCallback() slip detection
│   ├── Velocity window comparison: |mean(v_window) - v_current| > threshold
│   └── If triggered: P(x,x) = P(y,y) = P(θ,θ) = ∞  ← fires ERROR_INVALID_STATE
│
└── imuCallback() collision detection
    ├── Gyro discontinuity check: |ω_imu - ω_predicted| > threshold
    └── If triggered: transition to collision state, P → ∞

File: 
└── isFinite()
    ├── Checks all pose elements (x, y, θ, v, ω)
    ├── Checks all diagonal covariance elements (P_xx, P_yy, P_θθ)
    └── Returns false if ANY is NaN or ±Inf → triggers ERROR_INVALID_STATE
```

---

## Summary — What to Remember

| Concept | Formula | OKS Meaning |
|---------|---------|-------------|
| Belief | Gaussian N(x̂, P) | Robot pose is uncertain, tracked by mean + covariance |
| Prediction grows P | P⁻ = FPFᵀ + Q | Every `predict()` call increases covariance |
| Update shrinks P | P = (I-KH)P⁻ | Every sensorbar `update()` decreases covariance |
| Kalman gain | K = P⁻Hᵀ(HP⁻Hᵀ+R)⁻¹ | How much to trust sensor vs. prediction |
| Innovation | ν = z - Hx̂⁻ | How far off prediction was from measurement |
| EKF linearization | F = ∂f/∂x | OKS approximates this with the `min(P_θθ,1e5)×s²` term |
| Lateral variance growth | ΔP_yy ≈ s² × P_θθ | Robot drifts laterally when heading uncertain |
| P → ∞ | slip or collision | `odometryCallback()`/`imuCallback()` write literal infinity |
| isFinite check | ANY infinite/NaN → stop | `` guards all motion commands |
| σ = √P | P(x,x)=0.0025 → σ=5cm | 68% of mass within ±σ of the estimated position |

### Quick Diagnostic Checklist

When you see `ERROR_INVALID_STATE` in a bag:

```
1. Find the timestamp where P(x,x) first becomes inf
2. Look back 50-100 ms:
   a. IMU gyro spike?          → collision_detection fired (Path B)
   b. Odom velocity drop?      → slip_detection fired (Path A)
   c. Odom topic gap?          → staleness path → gradual P growth → eventual inf
3. Check prior covariance level: was P already high before the event?
   High P before event  → prior drift, sensor updates missed (aisle with no lines)
   Low P before event   → sudden mechanical event (actual slip/bump)
4. Confirm with isFinite() log timestamp matching the P→inf timestamp
```

---

*Next: `03-measurement-models.md` — Sensorbar line constraints, Mahalanobis gating,
and how the update() function decides whether to accept or reject a measurement.*
