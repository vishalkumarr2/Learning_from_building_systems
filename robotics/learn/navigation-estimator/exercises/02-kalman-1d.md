# Exercises 02 — Kalman Filter: 1D Problems
### Companion exercises for `02-kalman-filter.md`

**Work through these sequentially.** Section A tests conceptual understanding, Section B builds
a complete 1D filter from scratch, and Section C ties everything back to robot code parameters.

---

## Section A — Conceptual Questions (5 questions)

**A1.** The Kalman gain K is computed as `K = P⁻Hᵀ(HP⁻Hᵀ + R)⁻¹`.
What happens to K (and to the resulting state estimate) when:
- (a) The measurement is perfect: R → 0
- (b) The measurement is useless: R → ∞
- (c) We have no prior information at all: P⁻ → ∞

<details>
<summary>Answer A1</summary>

**(a) R → 0 (perfect measurement):**
```
K = P⁻Hᵀ (HP⁻Hᵀ + 0)⁻¹ = P⁻Hᵀ (HP⁻Hᵀ)⁻¹

For 1D with H=1:  K = P⁻ / P⁻ = 1

x̂ = x̂⁻ + 1 × (z - x̂⁻) = z
```
We fully trust the measurement and ignore the prediction.

**(b) R → ∞ (useless measurement):**
```
K = P⁻Hᵀ (HP⁻Hᵀ + ∞)⁻¹ → 0

x̂ = x̂⁻ + 0 × (z - x̂⁻) = x̂⁻
```
We ignore the measurement completely and keep the prediction.

**(c) P⁻ → ∞ (no prior information):**
```
K = ∞ × (∞ + R)⁻¹ → 1  (∞ dominates the denominator)

x̂ = x̂⁻ + 1 × (z - x̂⁻) = z
```
Again we fully trust the measurement. This makes sense — if we know nothing
about position, the first measurement should be taken at face value.

**AMR relevance:** When slip detection sets P → ∞, the next line-sensor `update()` will have K → 1,
meaning the line-sensor measurement fully overrides the (now-unknown) predicted pose. This is the
recovery mechanism: one good line-sensor hit re-localizes the robot.
</details>

---

**A2.** After a predict step, the covariance is `P⁻ = 0.09 m²`. After an update step using
a measurement with noise `R = 0.04 m²`, what is the posterior covariance P?
*(Assume 1D, H = 1)*

<details>
<summary>Answer A2</summary>

```
K = P⁻ / (P⁻ + R) = 0.09 / (0.09 + 0.04) = 0.09 / 0.13 ≈ 0.692

P = (1 - K) × P⁻ = (1 - 0.692) × 0.09 = 0.308 × 0.09 ≈ 0.0277 m²

σ_posterior = √0.0277 ≈ 0.166 m
```

Compare to the pre-update uncertainty: σ_prior = √0.09 = 0.30 m.
The measurement (σ_R = √0.04 = 0.20 m) is more precise than the prediction (0.30 m),
so the posterior (0.166 m) is tighter than either source. This is the "sharper than either
alone" property from the main notes.

**Alternative formula** (equivalent, often faster):
```
1/P = 1/P⁻ + 1/R = 1/0.09 + 1/0.04 = 11.11 + 25.00 = 36.11
P = 1/36.11 ≈ 0.0277 m²  ✓
```
</details>

---

**A3.** Explain in one sentence each the physical meaning of: (a) the innovation, (b) the
innovation covariance S = HP⁻Hᵀ + R, (c) the condition when Mahalanobis gating rejects
an update.

<details>
<summary>Answer A3</summary>

**(a) Innovation ν = z − Hx̂⁻:**
The innovation is the discrepancy between what the sensor actually measured and what the
filter predicted the sensor would measure, before applying any correction.

**(b) Innovation covariance S = HP⁻Hᵀ + R:**
S is the *expected* spread of the innovation — it combines prediction uncertainty (HP⁻Hᵀ)
with sensor noise (R) to quantify how surprising a given measurement discrepancy would be.

**(c) Mahalanobis gating:**
A measurement is rejected when its normalized distance from the prediction exceeds a
threshold: `νᵀ S⁻¹ ν > χ²_threshold`.
This means: "the measurement is so far from our prediction that, given our known uncertainty,
the probability of this occurring by random noise alone is below the acceptance threshold."

**AMR relevance:** `update()` in `` computes the Mahalanobis distance for the
line-sensor line offset. If the robot is far off a line (perhaps because it drove into the
wrong aisle), the line-sensor reading will be rejected as an outlier rather than
catastrophically corrupting the pose estimate.
</details>

---

**A4.** The AMR shortcut adds the term `min(cov_theta, 1e5) * delta_trans²` to the lateral
variance at each prediction step. Under what robot motion scenario does this term become the
dominant contributor to lateral uncertainty growth?

<details>
<summary>Answer A4</summary>

This term dominates when:
1. `cov_theta` (heading variance) is large — e.g., after several missed line-sensor updates
   that would normally constrain heading, or after a partial slip event that inflated θ
   uncertainty without hitting the full-slip threshold.
2. `delta_trans` is large — the robot is traveling a long distance in a straight line.

The combination is exactly: **long aisle travel without line-sensor hits, after heading has
already drifted.** This is the classic AMR failure mode where the robot "wanders" laterally
in an aisle with sparse painted lines, until covariance grows large enough to trigger a
planning failure or a `ERROR_STATE_INVALID`.

Numerically: if `cov_theta = 0.01 rad²` (σ_θ ≈ 5.7°) and the robot travels 3 m without a
line-sensor update, the additional lateral variance is `0.01 × 9 = 0.09 m²` (σ ≈ 0.30 m).
The fixed `k_trans_lat_noise × delta_trans` term contributes only `0.001 × 3 = 0.003 m²`
by comparison — 30× smaller.
</details>

---

**A5.** AMR does not read the `pose.covariance` field from the `/odom` ROS message. Why is
this a deliberate and reasonable engineering choice, rather than a bug?

<details>
<summary>Answer A5</summary>

The `pose.covariance` field in a ROS odometry message is populated by the odometry publisher
(typically the motor controller firmware or a ROS driver node). In practice:

1. **Publishers often lie:** Many wheel encoder drivers either publish zeros, identity matrices,
   or canned diagonal values that do not reflect actual measurement quality.

2. **Calibrated fixed parameters are more reliable:** A noise model tuned against real robot
   motion data (e.g., by driving a robot over a known path and measuring actual position error)
   gives more accurate covariance than trusting the publisher.

3. **Consistency:** AMR operates across multiple robot generations. Using a fixed noise model
   in the config YAML means the estimator behavior is predictable and version-controlled,
   rather than dependent on whatever the firmware team decided to publish.

**The downside:** If the actual wheel slip characteristics change (e.g., floor material changes,
wheel wear), the fixed model will be wrong until someone re-tunes the parameters.
This is why floor-specific tuning files exist in the robot site configuration.
</details>

---

## Section B — Build a 1D Kalman Filter by Hand

### Problem Setup

A robot moves along a 1D corridor at approximately constant velocity. It carries a GPS receiver
(noisy position measurements only — no velocity from GPS). You will run 3 complete predict/update
cycles and track how the estimate evolves.

**All matrices and initial conditions:**

```
State vector:    x̂ = [position (m), velocity (m/s)]ᵀ

Initial state:   x̂₀ = [0.0,  0.5]ᵀ          (start at 0 m, going 0.5 m/s)

Initial covariance:
    P₀ = ┌ 1.0    0.0 ┐         (σ_x = 1.0 m, σ_v = 0.2 m/s, uncorrelated)
         │ 0.0    0.04┘

Time step:       Δt = 1.0 s

State transition:
    F = ┌ 1   1.0 ┐             (position += velocity × Δt)
        │ 0   1.0 ┘             (velocity unchanged)

Process noise:
    Q = ┌ 0.1    0.0 ┐         (small model uncertainty)
        │ 0.0    0.01┘

Measurement matrix:  H = [1  0]    (GPS measures position only)
Measurement noise:   R = [0.5]     (σ_GPS = √0.5 ≈ 0.707 m)

Measurements:
    z₁ = 0.4 m    (at t = 1s)
    z₂ = 1.1 m    (at t = 2s)
    z₃ = 1.6 m    (at t = 3s)
```

### Your Task

For each cycle k = 1, 2, 3, compute:

1. **Predict:** `x̂⁻ₖ = F x̂ₖ₋₁` and `P⁻ₖ = F Pₖ₋₁ Fᵀ + Q`
2. **Innovation:** `νₖ = zₖ - H x̂⁻ₖ`
3. **Innovation covariance:** `Sₖ = H P⁻ₖ Hᵀ + R`
4. **Kalman gain:** `Kₖ = P⁻ₖ Hᵀ Sₖ⁻¹`
5. **Update state:** `x̂ₖ = x̂⁻ₖ + Kₖ νₖ`
6. **Update covariance:** `Pₖ = (I - Kₖ H) P⁻ₖ`

Fill in the table:

| Step | x̂_pos | x̂_vel | P_xx | P_vv | ν | K_pos |
|------|--------|--------|------|------|---|-------|
| 0 (init) | 0.0 | 0.5 | 1.00 | 0.04 | — | — |
| 1 predict | ? | ? | ? | ? | — | — |
| 1 update | ? | ? | ? | ? | ? | ? |
| 2 predict | ? | ? | ? | ? | — | — |
| 2 update | ? | ? | ? | ? | ? | ? |
| 3 predict | ? | ? | ? | ? | — | — |
| 3 update | ? | ? | ? | ? | ? | ? |

<details>
<summary>Full solution — Cycle 1</summary>

**CYCLE 1: Predict from k=0 to predicted k=1**

```
x̂⁻₁ = F x̂₀ = ┌1  1┐ ┌0.0┐ = ┌0.0 + 1×0.5┐ = ┌0.5┐
               │0  1┘ │0.5┘   │0.0 + 1×0.5┘   │0.5┘

F P₀ = ┌1  1┐ ┌1.00  0.0 ┐ = ┌1.00+0.04   0.04 ┐ = ┌1.04   0.04┐
        │0  1┘ │0.0   0.04┘   │0.00+0.04   0.04 ┘   │0.04   0.04┘

F P₀ Fᵀ = ┌1.04  0.04┐ ┌1  0┐ = ┌1.04+0.04   0.04┐ = ┌1.08   0.04┐
           │0.04  0.04┘ │1  1┘   │0.04+0.04   0.04┘   │0.04   0.04┘

P⁻₁ = F P₀ Fᵀ + Q = ┌1.08+0.1   0.04 ┐ = ┌1.18   0.04┐
                      │0.04       0.04+0.01┘   │0.04   0.05┘
```

**CYCLE 1: Update with z₁ = 0.4**

```
ν₁ = z₁ - H x̂⁻₁ = 0.4 - [1 0]×[0.5, 0.5]ᵀ = 0.4 - 0.5 = -0.1 m

S₁ = H P⁻₁ Hᵀ + R = [1 0]┌1.18  0.04┐[1]ᵀ + [0.5] = 1.18 + 0.5 = 1.68
                          │0.04  0.05┘[0]

K₁ = P⁻₁ Hᵀ S₁⁻¹ = ┌1.18┐ × (1/1.68) = ┌0.702┐
                     │0.04┘               │0.024┘

x̂₁ = x̂⁻₁ + K₁ ν₁ = ┌0.5┐ + ┌0.702┐ × (-0.1) = ┌0.5 - 0.0702┐ = ┌0.430┐
                      │0.5┘   │0.024┘              │0.5 - 0.0024┘   │0.498┘

I - K₁H = ┌1  0┐ - ┌0.702┐[1  0] = ┌1-0.702   0┐ = ┌0.298  0┐
           │0  1┘   │0.024┘          │0-0.024   1┘   │-0.024 1┘

P₁ = (I - K₁H) P⁻₁ = ┌0.298   0  ┐ ┌1.18  0.04┐
                        │-0.024  1  ┘ │0.04  0.05┘
   = ┌0.298×1.18   0.298×0.04 ┐ = ┌0.352   0.012┐
     │-0.024×1.18+0.04  -0.024×0.04+0.05┘   │0.012   0.049┘
```

**Cycle 1 result:** x̂ = [0.430 m, 0.498 m/s], P_xx = 0.352 m², P_vv = 0.049 (m/s)²

Observation: GPS pulled the position estimate from 0.5 → 0.430 (toward z₁=0.4), but not
all the way to 0.4 because K < 1 (prediction still contributes). Velocity barely moved
(K_vel = 0.024 is tiny — GPS doesn't measure velocity directly).
</details>

<details>
<summary>Full solution — Cycle 2</summary>

**CYCLE 2: Predict from k=1 to predicted k=2**

```
x̂⁻₂ = F x̂₁ = ┌1  1┐ ┌0.430┐ = ┌0.430 + 0.498┐ = ┌0.928┐
               │0  1┘ │0.498┘   │0.498         ┘   │0.498┘

F P₁ = ┌1  1┐ ┌0.352  0.012┐ = ┌0.352+0.012   0.012+0.049┐ = ┌0.364  0.061┐
        │0  1┘ │0.012  0.049┘   │0.012         0.049       ┘   │0.012  0.049┘

F P₁ Fᵀ = ┌0.364  0.061┐ ┌1  0┐ = ┌0.364+0.061   0.061┐ = ┌0.425  0.061┐
           │0.012  0.049┘ │1  1┘   │0.012+0.049   0.049┘   │0.061  0.049┘

P⁻₂ = ┌0.425+0.1   0.061 ┐ = ┌0.525  0.061┐
       │0.061       0.049+0.01┘   │0.061  0.059┘
```

**CYCLE 2: Update with z₂ = 1.1**

```
ν₂ = 1.1 - [1 0]×[0.928, 0.498]ᵀ = 1.1 - 0.928 = 0.172 m

S₂ = 0.525 + 0.5 = 1.025

K₂ = ┌0.525┐ / 1.025 = ┌0.512┐
     │0.061┘            │0.060┘

x̂₂ = ┌0.928┐ + ┌0.512┐ × 0.172 = ┌0.928 + 0.088┐ = ┌1.016┐
      │0.498┘   │0.060┘            │0.498 + 0.010┘   │0.508┘

P₂ ≈ (1 - 0.512) × 0.525 = 0.488 × 0.525 ≈ 0.256 m²  (P_xx approx)
P_vv ≈ 0.059 - 0.060×0.061 ≈ 0.055 (m/s)²
```

**Cycle 2 result:** x̂ = [1.016 m, 0.508 m/s], P_xx ≈ 0.256 m²

Observation: P_xx dropped from 0.352 → 0.256 after the update. The filter is converging.
Velocity estimate drifted slightly upward (0.498 → 0.508) because measurements are slightly
ahead of the dead-reckoning prediction, suggesting the robot is moving slightly faster than
initially estimated.
</details>

<details>
<summary>Full solution — Cycle 3</summary>

**CYCLE 3: Predict from k=2 to predicted k=3**

```
x̂⁻₃ = ┌1.016 + 0.508┐ = ┌1.524┐
        │0.508         ┘   │0.508┘

P⁻₃ ≈ ┌0.256 + 0.055 + 0.1    ...┐ ≈ ┌0.411  ...┐
       │...                0.065  ┘   │...   0.065┘

(Using simplified scalar update for P_xx: new_xx ≈ old_xx + old_vv + Q_xx)
```

**CYCLE 3: Update with z₃ = 1.6**

```
ν₃ = 1.6 - 1.524 = 0.076 m

S₃ = 0.411 + 0.5 = 0.911

K₃ = 0.411 / 0.911 = 0.451

x̂₃_pos = 1.524 + 0.451 × 0.076 = 1.524 + 0.034 = 1.558 m
x̂₃_vel ≈ 0.508 (very little update from GPS, which only measures position)

P₃_xx = (1 - 0.451) × 0.411 = 0.549 × 0.411 ≈ 0.226 m²
```

**Cycle 3 result:** x̂ = [1.558 m, 0.508 m/s], P_xx ≈ 0.226 m²

**Pattern across all 3 cycles:**
```
Cycle   P_xx (after update)   σ_x
  1         0.352              0.593 m
  2         0.256              0.506 m
  3         0.226              0.476 m
```

P_xx is slowly decreasing — the filter is converging to a tighter estimate.
It would continue to decrease until it reaches a steady-state value determined by the
balance between Q (process noise inflating P) and R/P ratio (measurement shrinking P).
</details>

---

## Section C — AMR-Specific Calculation Questions

### C1 — Prediction step with the estimator noise model

**Given:**
- Current `P(x,x) = 0.0025 m²`  (σ_x = 0.05 m, a tight estimate after a line-sensor hit)
- `P(θ,θ) = 0.0010 rad²`  (σ_θ ≈ 1.8°, well-constrained heading)
- Robot moves forward `delta_trans = 0.5 m` in this time step
- `delta_rot = 0.0 rad`  (straight-line motion)
- AMR noise parameters:
  - `k_trans_noise = 0.01`
  - `k_trans_lat_noise = 0.005`
  - `k_time_trans_noise = 0.0`  (ignore time component)
- No measurement update this step

**Question:** What is `P(x,x)` after the prediction step?

*(Assume the robot is driving along the x-axis so "longitudinal" ≈ x and "lateral" ≈ y.)*

<details>
<summary>Answer C1</summary>

Using the the estimator noise model from `predict()`:

```
ΔP(x,x) = k_trans_noise × delta_trans
         + min(P_θθ, 1e5) × delta_trans²

         = 0.01 × 0.5
         + min(0.0010, 1e5) × 0.5²

         = 0.005
         + 0.0010 × 0.25

         = 0.005 + 0.00025

         = 0.00525 m²

P(x,x)_new = P(x,x)_old + ΔP(x,x)
           = 0.0025 + 0.00525
           = 0.00775 m²

σ_x_new = √0.00775 ≈ 0.088 m
```

**Breakdown of contributions:**
- Fixed translational noise (`0.01 × 0.5 = 0.005`): contributes **95.2%** of new variance
- Heading-coupled term (`0.0010 × 0.25 = 0.00025`): contributes **4.8%**

When P_θθ is small (well-constrained heading), the fixed noise dominates.
The heading-coupled term only becomes dominant when `P_θθ > k_trans_noise = 0.01 rad²`
(σ_θ > 5.7°). In this example, `P_θθ = 0.001 << 0.01`, so heading uncertainty is not yet the
dominant source of error.

**σ_x grew from 0.050 m to 0.088 m** for a 0.5 m move — about a 76% increase in standard deviation.
</details>

---

### C2 — Minimum covariance ratio for K > 0.5

**Question:** For a 1D filter with measurement matrix `H = [1]` (direct position measurement),
what is the minimum ratio `P⁻ / R` such that the Kalman gain `K > 0.5`?

Derive algebraically and verify with a numerical example using `R = 0.04 m²` (σ = 0.2 m,
typical line-sensor uncertainty).

<details>
<summary>Answer C2</summary>

**Algebraic derivation:**

```
K = P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹

For H = 1 (1D):
K = P⁻ / (P⁻ + R)

Set K > 0.5:
P⁻ / (P⁻ + R) > 0.5

P⁻ > 0.5 × (P⁻ + R)
P⁻ > 0.5 P⁻ + 0.5 R
0.5 P⁻ > 0.5 R
P⁻ > R
```

**Result: K > 0.5 if and only if P⁻ > R**

Equivalently: the Kalman gain exceeds 0.5 (meaning we trust the sensor more than the
prediction) when the prediction uncertainty is larger than the measurement noise.

**Numerical verification with R = 0.04 m² (σ_sensor = 0.2 m):**

```
P⁻ = 0.03 m²  (< R):  K = 0.03 / (0.03 + 0.04) = 0.03/0.07 = 0.429 < 0.5  ✓
P⁻ = 0.04 m²  (= R):  K = 0.04 / 0.08 = 0.500 = 0.5 (boundary)             ✓
P⁻ = 0.05 m²  (> R):  K = 0.05 / 0.09 = 0.556 > 0.5                        ✓
P⁻ = 0.10 m²  (> R):  K = 0.10 / 0.14 = 0.714 > 0.5                        ✓
```

**AMR interpretation:**
- `R = 0.04 m²` is typical for a well-calibrated line-sensor (σ ≈ 0.2 m lateral accuracy)
- `P(x,x) = 0.04 m²` means the robot has accumulated uncertainty equivalent to the sensor noise
- For `P(x,x) > 0.04 m²`, the filter trusts the line-sensor more than its dead-reckoning prediction

In practice, after a robot travels ~4 m without a line-sensor update
(ΔP_xx ≈ 0.01 × 4 = 0.04 m² from `k_trans_noise`), the transition point K = 0.5
is reached. This is consistent with robot site layout requirements: line-sensor lines should be
placed no more than ~4 m apart on heavily-traveled aisles for reliable localization.
</details>

---

*Back to main notes: `02-kalman-filter.md`*
*Next exercises: `03-measurement-models-exercises.md`*
