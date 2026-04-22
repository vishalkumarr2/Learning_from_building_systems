# 01 — Dead-Reckoning & Wheel Odometry
### The math underneath AMR `predict()`

**Prerequisite:** None — this is the entry point for the navigation-estimator track.
**Unlocks:** `02-kalman-filter.md` — you need to know what `delta_trans` and `delta_theta` *mean* before you can understand how their variances propagate.

---

## Why Should I Care? (AMR Project Context)

Every warehouse robot navigation failure starts with `predict()`. When you open a bag from an incident like a bag from a recent incident and see the robot's estimated position drifting off a tile before the line-sensor could correct it, you're watching dead-reckoning accumulate error in real time.

Understanding this material lets you answer questions you currently have to guess at:

- **"Why did the robot overshoot the target by 12 cm on a 3 m move?"** → Systematic heading bias from wheel diameter mismatch; you can *calculate* the drift before even opening the bag.
- **"Why does `lateral_var` blow up even when the robot moves in a straight line?"** → Because lateral error grows as `cov_theta × delta_trans²`. The higher the angular uncertainty, the more uncertain your sideways position gets just by driving forward.
- **"What does `k_trans_noise` actually model?"** → Wheel slip per unit distance: the further you drive, the more uncertain your along-track position. This note gives you the physical derivation of that coefficient.
- **"Why is the staleness scaling needed?"** → If odom goes stale, you have no dead-reckoning input at all — the robot is *guessing* its position. The scaling inflates variance to reflect that.

The the navigation estimator is deliberately simple: no full Jacobian, no odom message covariance, a fixed linear noise model. That simplicity is a feature, but it means the noise parameters (`k_trans_noise`, etc.) are the *entire model of wheel reliability*. If those are wrong, or if wheel slip is outside their range, every bag you analyse will show systematic unexplained covariance growth.

---

# PART 1 — THE DIFFERENTIAL DRIVE ROBOT

## 1.1 Physical Setup

An warehouse AMR (and most small robots) uses **differential drive**: two driven wheels on a common axle, no steering mechanism. Turning is achieved by driving the wheels at different speeds. A castor wheel provides balance but contributes no drive.

**ELI15 analogy:** Imagine two people holding opposite ends of a stick. If both walk forward equally fast, the stick moves straight. If the left person walks faster, the stick rotates clockwise (from above). The stick's centre traces an arc. Differential drive is exactly this.

```
        TOP VIEW OF ROBOT
        ==================

              FRONT
                ^
                |
    +-----------+----------+
    |                      |
    |   Left       Right   |
 L --[●]--+---B---+--[●]-- R      ● = driven wheel
    |      |       |      |       B = wheel baseline (axle width)
    |      +---+---+      |
    |          |          |
    |       castor        |
    +-----------+----------+
                |
               REAR

    Coordinate frame:
      X  →  forward
      Y  →  left
      θ  = heading angle (CCW positive, 0 = facing +X)

    B = distance between wheel contact patches [metres]
    r = wheel radius [metres]
    Encoder on each wheel counts ticks per revolution
```

## 1.2 Wheel Encoders: Ticks to Arc Length

Each wheel has a quadrature encoder measuring how far the wheel has rotated. The encoder produces `ticks` — an integer count that increments (or decrements) for each small angular step.

**The formula:**

```
                  ticks
arc_length = ─────────────── × 2π × r
              ticks_per_rev
```

Where:
- `ticks` = encoder count change since last measurement
- `ticks_per_rev` = encoder resolution (counts per full wheel revolution)
- `r` = wheel radius in metres
- `2π × r` = circumference of the wheel

Dimensionally: `[ticks / (ticks/rev)] × [m/rev] = [m]` ✓

**ASCII derivation:**

```
    Full revolution = 2π × r metres of ground contact
    Encoder has N = ticks_per_rev counts per revolution

    One tick = (2π × r) / N metres

                     n ticks
    arc_length = n × ──────── = (n / N) × 2π × r
                      N ticks
```

## 1.3 Worked Example — Encoder Ticks to Arc Length

**Given:**
- Left wheel: `n_L = 100` ticks
- Right wheel: `n_R = 98` ticks
- Wheel radius: `r = 0.05 m`
- Encoder resolution: `N = 1000 ticks/rev`

**Step 1:** Compute left arc length:
```
s_L = (100 / 1000) × 2π × 0.05
    = 0.1 × 0.3142
    = 0.03142 m  ≈ 31.4 mm
```

**Step 2:** Compute right arc length:
```
s_R = (98 / 1000) × 2π × 0.05
    = 0.098 × 0.3142
    = 0.03079 m  ≈ 30.8 mm
```

**Step 3:** Difference:
```
s_R - s_L = 30.8 - 31.4 = -0.6 mm
```

The right wheel travelled 0.6 mm *less* than the left → the robot turned slightly right.

> **Key insight:** Even two encoder counts of difference (0.2% asymmetry) causes rotation. Over 100 odom cycles this accumulates. This is why wheel calibration matters so much in warehouses.

---

# PART 2 — ARC INTEGRATION: THE UNICYCLE MODEL

## 2.1 The Geometry of Unequal Wheel Motion

When `s_L ≠ s_R`, the robot traces an **arc**, not a straight line. Both wheels travel on concentric arcs centred at the same Instantaneous Centre of Curvature (ICC).

```
    ARC GEOMETRY (TOP VIEW)
    =======================

                     ICC
                      *
                     /|
                    / |
                   /  |
                  /   |
    r_L + B      /    | r_R
                /     |
               /      |
    LEFT -----/-------/------ RIGHT
    WHEEL    /arc_L  /arc_R   WHEEL
            /       /
           /       /
          +-------+
          robot
          (start)

    r_L = radius of left wheel arc
    r_R = radius of right wheel arc
    B   = axle baseline = r_R - r_L

    Since both wheels rotate over the SAME angle Δθ:
        s_L = r_L × Δθ
        s_R = r_R × Δθ
    Subtracting:
        s_R - s_L = (r_R - r_L) × Δθ = B × Δθ
    Therefore:
        Δθ = (s_R - s_L) / B                  ← HEADING CHANGE
```

## 2.2 Full Derivation of the Unicycle Kinematics

**Step 1 — Heading change:**

From the arc geometry above:

$$\Delta\theta = \frac{s_R - s_L}{B}$$

**Step 2 — Forward distance (arc midpoint):**

The robot's centre travels the average arc:

$$s = \frac{s_R + s_L}{2}$$

**Step 3 — Why use the midpoint heading `θ + Δθ/2`?**

```
    POSITION UPDATE ERROR ILLUSTRATION
    ====================================

    If we use old heading θ:
        x' = x + s·cos(θ)    ← wrong: ignores the turn
                               undershoots curve

    If we use new heading θ':
        x' = x + s·cos(θ')   ← wrong: overcorrects
                               overshoots curve

    Correct: use MIDPOINT heading (θ + Δθ/2):
        The robot was pointing BETWEEN old and new headings
        during the arc. The midpoint is the best single
        direction to approximate the arc as a chord.

        For Δθ small, this is exact. For large Δθ it's
        still better than either endpoint.

    VISUAL:
          θ                θ + Δθ/2               θ + Δθ
          |                    |                       |
    ─────●──────────────────────────────────────────●─────
          start heading                        end heading
                              ↑
                          use THIS for position update
```

**Step 4 — Position update equations:**

$$x' = x + s \cdot \cos\!\left(\theta + \frac{\Delta\theta}{2}\right)$$

$$y' = y + s \cdot \sin\!\left(\theta + \frac{\Delta\theta}{2}\right)$$

$$\theta' = \theta + \Delta\theta$$

These five equations *are* wheel odometry (the unicycle model).

## 2.3 Worked Example — Full Pose Update

**Given:** Robot starts at `(x=0, y=0, θ=0°)`. Encoder readings from Section 1.3:
- `s_L = 0.03142 m`, `s_R = 0.03079 m`, `B = 0.30 m`

**Step 1 — Heading change:**
```
Δθ = (s_R - s_L) / B
   = (0.03079 - 0.03142) / 0.30
   = -0.00063 / 0.30
   = -0.0021 rad   (≈ -0.12°, turning right)
```

**Step 2 — Forward distance:**
```
s = (s_R + s_L) / 2
  = (0.03079 + 0.03142) / 2
  = 0.03111 m   (≈ 31.1 mm)
```

**Step 3 — Midpoint heading:**
```
θ_mid = θ + Δθ/2 = 0 + (-0.0021)/2 = -0.00105 rad
```

**Step 4 — Position update:**
```
x' = 0 + 0.03111 × cos(-0.00105)
   = 0 + 0.03111 × 0.99999945
   ≈ 0.03111 m

y' = 0 + 0.03111 × sin(-0.00105)
   = 0 + 0.03111 × (-0.00105)
   ≈ -0.0000327 m   (≈ -0.03 mm lateral shift)

θ' = 0 + (-0.0021) = -0.0021 rad
```

**Result:** `(x'=31.1 mm, y'=-0.033 mm, θ'=-0.12°)` — nearly straight, with a tiny clockwise drift.

> **Key insight:** The lateral displacement (`y'`) from 2 missing encoder ticks is only 0.033 mm per step. But `Δθ` is -0.0021 rad per step. After 1000 steps (31 metres of travel), heading error is `1000 × 0.0021 = 2.1 rad` — the robot has spun 120°! This is why **heading error is the primary concern**, not per-step position error.

---

# PART 3 — ERROR ACCUMULATION

## 3.1 Why Heading Error Is Catastrophic

The unicycle equations show that `x'` and `y'` depend on `cos(θ)` and `sin(θ)`. A heading error of `ε` causes a *lateral* position error that grows proportionally to distance:

$$\text{lateral error} \approx s_{\text{total}} \times \sin(\varepsilon) \approx s_{\text{total}} \times \varepsilon \quad \text{(for small } \varepsilon \text{)}$$

**Quantified example: 1° heading error after 10 m of travel:**
```
ε = 1° = 0.01745 rad
lateral_error = 10 m × sin(0.01745)
              = 10 × 0.01745
              = 0.1745 m  ≈ 17.5 cm
```

A 1° heading miscalibration produces **17 cm of sideways drift per 10 metres**. In a warehouse with 50 cm aisle margins, that is a navigation failure after ~30 m of travel.

## 3.2 Types of Error

```
    ERROR TAXONOMY
    ===============

    SYSTEMATIC (deterministic, repeatable)
    ├── Wheel diameter mismatch
    │     Left wheel 0.1% larger than right
    │     → constant rotation at constant speed
    │     → grows linearly with distance
    │
    ├── Baseline (axle width) calibration error
    │     B is 1 mm off → all Δθ computations are wrong by B_err/B²
    │
    └── Encoder resolution (if mis-set)
          Wrong ticks_per_rev → wrong arc length every tick

    RANDOM (stochastic, unpredictable)
    ├── Wheel slip (on smooth floors, sharp turns)
    │     Encoder counts, but wheel doesn't move ground contact
    │
    ├── Ground height variation (floor bumps, ramps)
    │     r_effective changes if floor isn't flat
    │
    └── Encoder quantisation noise
          ±1 tick uncertainty per reading
```

**Why systematic errors are worse:** They accumulate *monotonically*. Random errors partially cancel over time. Systematic errors never cancel — they compound.

## 3.3 Why AMR Uses Line-Sensor Corrections

Without external corrections, the table below shows what happens:

```
    DEAD-RECKONING DRIFT IN AN AMR WAREHOUSE
    ==========================================

    Assumptions:
      • Heading bias: 0.05°/m (typical without calibration)
      • Straight path, 100 m total travel

    After  10 m: heading error = 0.5°,  lateral drift ≈  87 mm
    After  20 m: heading error = 1.0°,  lateral drift ≈ 349 mm  ← off tile
    After  50 m: heading error = 2.5°,  lateral drift ≈ 2.18 m  ← wrong aisle
    After 100 m: heading error = 5.0°,  lateral drift ≈ 8.72 m  ← wall collision

    ────────────────────────────────────────────────────────────
    VISUAL DRIFT:

    ──────────────────────────────────────── true path (straight)
     ╲
      ╲ 0.5° drift
       ╲
        ──────────────────────────────── 10 m
         ╲
          ╲ 1.0° total
           ╲
            ──────────────────── 20 m
             ╲
              ╲
               ╲
                ─────── 50 m (off path by 2 m)
```

The floor sensor provides a measurement every time the robot crosses a floor line. This corrects the accumulated heading and position error back toward ground truth. Without line-sensor updates, a 100 m warehouse run is not navigable with wheel odometry alone.

## 3.4 The Encoder Wrap-Around Trap

Most embedded systems use `uint16_t` or `int16_t` for encoder counts. When the counter overflows:

```
    uint16 wraps:  65535 → 0     (appears as -65535 ticks)
    int16 wraps:   32767 → -32768 (appears as ±65535 ticks)

    Correct handling:
        delta = (int16_t)(new_count - old_count)
                 ↑ explicit int16 cast handles wrap-around

    Wrong handling:
        delta = new_count - old_count
                 ↑ if both are uint32, no wrap-around occurs
                   but the delta is 65535 ticks instead of 1
```

A single wrap-around event produces a `delta_trans` spike of ~(65535/N) × 2πr metres. With N=1000 and r=0.05, that is **20.6 m** of phantom movement in one tick. This causes a covariance spike of `k_trans_noise × 20.6`, potentially setting variance to INF.

---

# PART 4 — AMR CODE CONNECTION

## 4.1 How AMR `predict()` Uses This Math

AMR's `predict()` function (, ) receives the *already-computed* `delta_trans` and `delta_theta` from the odom message comparison. It does **not** re-derive them from encoder ticks — the ROS nav_msgs/Odometry message already integrates wheel odometry (via the robot's base driver). The estimator receives the *state change* directly.

The prediction step does two things:

1. **State propagation** — applies the unicycle equations to advance `(x, y, θ)`.
2. **Covariance propagation** — grows the covariance matrix using a linear noise model.

## 4.2 The AMR Linear Noise Model

AMR uses a **simplified** noise model that is NOT the standard EKF Jacobian-based propagation. Instead:

```cpp
// From  predict() ~

trans_var   = k_trans_noise   * delta_trans
            + k_rot_pos_noise     * delta_theta
            + k_time_pos_noise      * delta_t

theta_var   = k_trans_rot_noise  * delta_trans
            + k_rot_noise  * delta_theta
            + k_time_rot_noise   * delta_t

lateral_var = k_trans_lat_noise * delta_trans
            + ...
            + min(cov_theta, 1e5)        * delta_trans^2
```

Each line is a **linear combination of the motion inputs** with fixed scaling coefficients. This is a deliberate simplification over the full EKF Jacobian propagation:

```
    STANDARD EKF COVARIANCE UPDATE:
        P' = F × P × Fᵀ + Q

    AMR SIMPLIFICATION:
        ΔP = k₁ × |delta_trans| + k₂ × |delta_theta| + k₃ × delta_t
```

Why? The full Jacobian requires computing partial derivatives of the unicycle model and multiplying two 3×3 matrices per prediction step. For AMR's use case (line-sensor corrections every ~10 cm), the simplified model is accurate enough — the measurement update dominates.

## 4.3 Why `min(cov_theta, 1e5) × delta_trans²` in `lateral_var`

This term is the most subtle in the noise model. Here's the physical intuition:

```
    IF YOU DON'T KNOW YOUR HEADING, MOVING FORWARD MAKES X AND Y BOTH UNCERTAIN

    Scenario: cov_theta = 0.01 rad² (small, well-known heading)
               delta_trans = 0.1 m
               lateral contribution = 0.01 × 0.1² = 0.0001 m²  (very small)

    Scenario: cov_theta = 1.0 rad² (heading very uncertain after a slip)
               delta_trans = 0.1 m
               lateral contribution = 1.0 × 0.1² = 0.01 m²  (100× larger!)

                HEADING UNCERTAINTY → LATERAL POSITION UNCERTAINTY

                     θ + σ_θ
                    /
                   /     ← uncertain heading: position could be ANYWHERE
                  /         along this fan after driving delta_trans
    ─────────●──────────────  θ
              \
               \     θ - σ_θ

    Width of the fan at distance d = d × σ_θ
    Variance of fan width = d² × var(θ)  ← This is exactly cov_theta × delta_trans²
```

The `min(cov_theta, 1e5)` cap prevents numerical overflow when heading variance is INF (after slip detection). Without the cap, a single slip would propagate infinite lateral variance immediately — the cap limits it to a finite but very large value.

## 4.4 Complete Concept-to-Code Mapping

| Concept (this document) | AMR Code Location | Variable Name |
|-------------------------|-------------------|---------------|
| Arc length from encoders | Robot base driver (not estimator) | Published in `odom.pose.pose` |
| `s = (s_R + s_L) / 2` | `predict()` input | `delta_trans` |
| `Δθ = (s_R - s_L) / B` | `predict()` input | `delta_theta` |
| Along-track noise (wheel slip per metre) | `predict()` noise formula | `k_trans_noise` |
| Rotation noise per rad of turn | `predict()` noise formula | `k_rot_noise` |
| Temporal drift (jitter, latency) | `predict()` noise formula | `k_time_pos_noise` |
| Heading uncertainty × travel → lateral error | `predict()` `lateral_var` | `min(cov_theta, 1e5) * delta_trans²` |
| Odom message staleness | `predict()` scaling | `scaling_factor` (if `Δt > 2×expected_period`) |
| Midpoint heading `θ + Δθ/2` | State propagation in `predict()` | Implicit in `cos/sin` decomposition |

## 4.5 What AMR Does NOT Do

Understanding the omissions prevents wrong mental models:

```
    AMR DOES NOT:
    ─────────────
    ✗ Read odom.twist.covariance or odom.pose.covariance
      (these fields are set by the base driver but estimator ignores them)

    ✗ Compute the full EKF Jacobian F for covariance propagation
      (the Jacobian-based P' = FPFᵀ + Q is replaced by the linear noise model)

    ✗ Distinguish between different surfaces (tiles vs. concrete)
      (noise parameters are fixed; slip on shiny tile is not modelled differently)

    ✗ Handle encoder wrap-around itself
      (relies on the base driver to provide correct delta_trans, delta_theta)
```

---

# PART 5 — COMMON MISTAKES & DEBUG TIPS

## 5.1 Wrong Wheel Radius Calibration

**Symptom in bags:** Systematic drift in a fixed direction on every straight run. The drift is proportional to distance. Plot `estimated_x` vs `ground_truth_x` — if the slope is not 1.0, wheel radius is wrong.

**Formula for diagnosis:**
```
    Measured radius r_m, true radius r_t:
    Odometry scale error = r_m / r_t

    If r_m = 0.0505 m, r_t = 0.0500 m (1% error):
    - Every 10 m of odom reads as 10.1 m of true distance
    - On a 100 m path, robot is 1 m ahead of where it thinks
    - Results in systematic undershoot of final position
```

**Important:** Asymmetric wheel wear causes the same effect, but only on the worn wheel. This appears as heading drift, not pure scale error.

## 5.2 Encoder Wrap-Around (uint16 → int16 Conversion)

**When it happens:** Robot runs continuously for a long time; encoders count past 65535. The base driver wraps the counter but does not sign-extend correctly.

**How to spot in a bag:** Single-tick `delta_trans` spike of ~20 m at a precise time in the `/odom` topic. This causes a `trans_var` spike that may or may not clear depending on the next line-sensor update.

**The correct C++ cast:**
```cpp
int16_t delta = static_cast<int16_t>(current_encoder - prev_encoder);
//              ↑ explicit int16 cast: wraps correctly at ±32768
```

## 5.3 Why AMR Computes Variance from Delta, Not Cumulative State

In a naive dead-reckoning system you might track total path length and compute total variance. AMR does NOT do this — it adds variance *incrementally* based on each new `delta_trans` and `delta_theta`.

**Why this is correct:** Variance is *additive* for independent increments. Each odom step is assumed independent (no correlation between wheel slips at step 5 and step 6). This is the Markov property: the state covariance at time `t+1` only depends on the state covariance at time `t` and the new measurement, not on all previous measurements.

**What would go wrong with cumulative:** Cumulative variance would be wrong after a line-sensor measurement *decreases* covariance. You'd be summing over the whole history rather than propagating from the post-measurement state.

## 5.4 How to Spot Dead-Reckoning Failure in a Bag

```
    DIAGNOSIS CHECKLIST FOR BAG ANALYSIS
    ======================================

    1. Compare /odom and /estimated_state position tracks
       │ If they diverge SLOWLY over many seconds → dead-reckoning drift
       │ If they jump SUDDENLY → line-sensor rejected update or slip event

    2. Check delta_trans magnitude in odom deltas
       │ Normal: ≤ 0.01 m per 50 Hz odom step (≤ 0.5 m/s)
       │ Anomaly: delta_trans > 0.1 m in one step → wrap-around or encoder fault

    3. Correlate position jump with velocity
       │ A position jump WITHOUT matching velocity in cmd_vel or odom.twist
       │ means the estimator moved without the robot moving → EKF divergence
       │ (not dead-reckoning failure, but a different kind)

    4. Plot cov_xx and cov_yy over time:
       │ Gradual growth between line-sensor updates → normal prediction
       │ Growth without any measurement shrinks → line-sensor not triggering
       │ Sudden jump to 1e5 → hit cov_theta cap in lateral_var formula
       │ Jump to INF → slip detection triggered

    5. Check staleness scaling events:
         Log line: "odom stale, applying scaling_factor"
         Covariance will grow faster than normal for that window
```

## 5.5 The Staleness Scaling Mechanism

AMR's `predict()` checks how long ago odom was received. If the gap exceeds `2 × expected_period`:

```
    scaling_factor = (actual_gap / expected_period)²

    Example: expected period = 20 ms, odom missed for 100 ms
    scaling_factor = (100/20)² = 25×

    trans_var (applied) = 25 × (normal trans_var)
```

This inflates variance to reflect that the robot's true position is much less certain when odom is stale. Common cause: CPU overload on the Jetson, ROS callback queue backup, network drop on WiFi-tethered odom.

---

## Implementation Connection

Full cross-reference of the concepts in this document to ``:

```
01-dead-reckoning.md concept               reference
─────────────────────────────────────────────────────────────────
Wheel arc length formula                  Not in estimator; in base driver
                                          Estimator receives odom.pose delta

Unicycle model (x', y', θ')               predict() ~+
                                          State propagation using delta_trans,
                                          delta_theta, and heading decomposition

Along-track variance growth               k_trans_noise * delta_trans
Heading variance growth                   k_rot_noise * delta_theta
Temporal variance growth                  k_time_pos_noise * delta_t

Lateral variance from heading uncertainty min(cov_theta, 1e5) * delta_trans^2
  (Section 2.2 fan geometry)

Staleness scaling                         _last_odom_received_time comparison
                                          scaling_factor inflation

Encoder wrap-around (Section 5.2)        NOT handled by estimator;
                                         must be handled by base driver

INF covariance → ESTIMATED_STATE_NOT_FINITE
                                          isFinite() check
                                         (triggered by slip or collision,
                                          NOT by prediction step alone)
```

---

## Summary — What to Remember

| Concept | Formula | AMR Parameter | What Goes Wrong If Wrong |
|---------|---------|---------------|--------------------------|
| Arc length from encoder | `s = (n/N) × 2π × r` | Inputs to base driver | Wrong `r` → systematic scale drift |
| Heading change | `Δθ = (s_R − s_L) / B` | `delta_theta` input | Wrong `B` → systematic heading drift |
| Forward distance | `s = (s_R + s_L) / 2` | `delta_trans` input | Wrong `r` asymmetry → scale error |
| Position update | `x' = x + s·cos(θ + Δθ/2)` | State propagation | — (derived from above) |
| Midpoint heading | `θ_mid = θ + Δθ/2` | Implicit in cos/sin | Using old θ → systematic undershoot on arcs |
| Along-track variance | `k₁ × delta_trans` | `k_trans_noise` | Too low → overconfident → Mahalanobis rejection of valid line-sensor |
| Heading variance | `k₂ × delta_theta` | `k_rot_noise` | Too high → line-sensor rejected prematurely |
| Lateral variance | `min(cov_θ, 1e5) × delta_trans²` | `cov_theta` at time of prediction | Uncapped → overflow after slip; over-capped → underestimates lateral uncertainty |
| Staleness scaling | `(gap / period)²` × variance | `scaling_factor` | Missing → overconfident during odom dropout |
| Encoder wrap | `(int16_t)(new − old)` | Base driver, not estimator | Phantom 20 m spike → covariance spike → possible ESTIMATED_STATE_NOT_FINITE |
| Heading error propagation | lateral_err ≈ `d × ε` | All of the above | 1° bias → 17 cm per 10 m → navigation failure |

**The one equation to burn into memory:**

$$\boxed{\Delta\theta = \frac{s_R - s_L}{B}}$$

This drives everything else. Heading error accumulates; position error follows from heading error. Every AMR navigation failure investigation should start by asking: *is the heading estimate correct?*
