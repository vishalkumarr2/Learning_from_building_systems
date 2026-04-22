# 03 — Measurement Models & Mahalanobis Gating
### Why AMR sometimes rejects line-sensor updates

**Prerequisite:** `02-kalman-filter.md` (EKF predict/update equations, covariance growth)
**Unlocks:** `04-imu-fusion.md` (theta update, gyro bias), `05-failure-modes.md` (diagnosing silent rejections)

---

## Why Should I Care? (AMR Project Context)

You are staring at a bag where the robot drifts 0.3m off a lane line over 8 seconds, yet the
line-sensor is firing. The estimator logs show nothing — no explicit rejection messages, no errors.
The line-sensor is silently ignored. This happens every day on the SCL floor.

Understanding measurement models tells you exactly why:
- **Mahalanobis gating** can reject a perfectly good line-sensor reading if the state covariance
  is too tight (P too small) — the measurement looks like an outlier even when the robot is
  close to the line.
- **AMR does not use a standard Kalman gain** for XY updates. It clamps the state to a range
  and sets covariance to the variance of a uniform distribution. Knowing this lets you compute
  the exact threshold at which a line-sensor range "wins" over the current covariance.
- **`is_reliable=True` does not mean the reading is accurate.** Line-Sensor duplicates (11–30%
  of frames in SCL bags) pass the reliability gate but carry identical stale offsets. They look
  like valid measurements to the estimator.

By the end of this chapter you will be able to:
1. Compute whether a given line-sensor reading will be accepted or rejected without running code.
2. Explain why the clamp update is mathematically equivalent to a uniform-likelihood measurement.
3. Identify in a bag trace the difference between "measurement rejected by Mahalanobis gate"
   and "measurement accepted but wrong (duplicate)".

---

# PART 1 — WHAT IS A MEASUREMENT MODEL

---

## 1.1 The Sensor-to-State Relationship

The **state vector** x holds what we believe about the robot:

```
        ┌ x  ┐    ← world-frame X position (metres)
    x = │ y  │    ← world-frame Y position (metres)
        └ θ  ┘    ← heading angle (radians, CCW from +X)
```

The **measurement** z is what a sensor actually reports. A measurement model is the function
that predicts what the sensor *should* report if the state were exactly x:

```
    z = h(x) + noise
```

For a **linear** relationship we write `h(x) = H x` where H is the measurement matrix.

**Example — GPS measuring position directly:**

```
         ┌ 1  0  0 ┐   ┌ x ┐   ┌ x ┐
    z = H x =       │   │ y │ = │ y │
         └ 0  1  0 ┘   └ θ ┘   └ y ┘

    H = [1 0 0]   (measuring x only)
    H = [0 1 0]   (measuring y only)
```

GPS gives us two independent scalar measurements, each with its own H row.

**Key insight:** H tells the filter which component of the state the sensor is observing. A row
of H like `[0, 1, 0]` means "the sensor only sees y; x and θ are invisible to it."

---

## 1.2 The Line-Sensor as a Line Constraint

The floor sensor is an array of IR emitter-detector pairs mounted under the robot chassis.
When the robot passes over a floor line, the detectors that cross the line return a different
reflectance signal. By finding the centroid of activated sensors, the firmware computes the
**lateral offset** of the robot from the line.

```
    Floor view (top-down):

         Robot chassis
         ┌─────────────────────┐
         │                     │
         │  ○ ○ ○ ● ● ○ ○ ○   │  ← IR array (● = detecting line)
         │                     │
         └─────────────────────┘
                  │
                  │  centroid of ● sensors
                  ▼
         ├────────┼────────────┤
                  └── offset from line centreline

    Floor:
    ══════════════════════════════  ← lane line (e.g. painted stripe)
```

The offset is a **1D measurement** — it tells the robot how far it is perpendicularly from
the line. It does NOT tell us how far along the line the robot is. This distinction creates
the two measurement types.

---

## 1.3 Orthogonal vs Parallel Measurements

```
    ORTHOGONAL MEASUREMENT                PARALLEL MEASUREMENT
    (robot crosses line ~perpendicularly)  (robot travels along a line)

          ↑ direction of travel                  → direction of travel
          │
          │         Floor line                   ════════════════════
    ══════╪════════════════════                        │
          │ ←── line-sensor fires here              line-sensor fires
          │      measures LATERAL                 measures AHEAD/BEHIND
          │      offset from line                 offset from line

    Updates: X or Y (one axis only,          Updates: the OTHER lateral axis
             perpendicular to line)          (perpendicular to travel direction)

    H = [1 0 0]  (if line is N-S,            H = [0 1 0]  (if line is E-W,
                  robot goes E-W,                          robot goes N-S,
                  measures X)                              measures Y)
```

**Why 1D?** One line crossing gives ONE perpendicular offset measurement. The parallel component
(how far along the line the robot is) is unobservable from a single crossing — you would need
two intersecting lines or a landmark for that.

**ELI15 — the train on tracks:** A train on a straight track knows exactly how far it is from
the track (0m — it's constrained). But the track itself tells nothing about how far along the
route the train has travelled. The line-sensor is the track: it constrains lateral position perfectly
but reveals nothing about longitudinal progress.

---

# PART 2 — THE LINE-SENSOR AS A LINE CONSTRAINT

---

## 2.1 The AMR Floor Map

The SCL warehouse floor has a grid of guidance lines:

```
    Y
    │
    │  ─────────────────────────────────────────  Y = 6.0m
    │
    │  ─────────────────────────────────────────  Y = 4.5m
    │
    │  ─────────────────────────────────────────  Y = 3.0m
    │
    │  ─────────────────────────────────────────  Y = 1.5m
    │
    └──────────────────────────────────────────── X

    Vertical lines (N-S) constrain X:
    │        │        │        │
    X=1.0  X=2.5  X=4.0  X=5.5
```

When the robot crosses a horizontal line (E-W), the line-sensor fires a measurement that
constrains **Y**. When it crosses a vertical line (N-S), the measurement constrains **X**.
This is why in ``, the line type (horizontal/vertical) determines which state
component gets updated.

---

## 2.2 The Measurement Range (Not a Point!)

Unlike a GPS fix that gives a precise coordinate, the line-sensor gives a **range**:

```
    Line-Sensor has N sensors, each 6mm apart.
    If 3 sensors are activated:

    ─── Sensor 1 (6mm spacing) ─── Sensor 2 ─── Sensor 3 ───

    The line is somewhere within the activated span.
    Centroid computation gives ±3mm resolution.

    Typical reported range:    [meas.min, meas.max]
                                e.g. [1.985m, 2.015m]
                                    ↑ 3cm total width ↑
```

This is why AMR uses a **clamping update** rather than a Kalman gain — the measurement is
honestly a range, not a point. The uniform distribution is the right model for "the robot is
somewhere in this 3cm band, with equal probability anywhere in the band."

---

# PART 3 — INNOVATION AND MAHALANOBIS DISTANCE

---

## 3.1 What Is Innovation?

Before the filter accepts a measurement, it checks consistency. The **innovation** is:

```
    ν = z - h(x̂⁻)

    where:
      z      = actual measurement from sensor
      x̂⁻    = prior state estimate (before this measurement)
      h(x̂⁻) = predicted measurement (what sensor SHOULD read if x̂⁻ is correct)
```

For the linear case: `ν = z - H x̂⁻`

- Small ν → measurement consistent with prediction → probably accept
- Large ν → measurement far from prediction → possibly an outlier or sensor fault

**But what is "large"?** 0.1m is large if the state uncertainty is 0.001m, but tiny if
uncertainty is 1.0m. The filter needs to normalize by uncertainty.

---

## 3.2 Mahalanobis Distance

The **Mahalanobis distance** is the innovation normalized by the total uncertainty in the
measurement space:

```
                                              ⁻¹
    d² = νᵀ  (H P⁻ Hᵀ + R)   ν

    where:
      P⁻  = prior state covariance (uncertainty before this measurement)
      R   = measurement noise covariance (sensor uncertainty)
      H   = measurement matrix

    S = H P⁻ Hᵀ + R    ← "innovation covariance"
                           (total uncertainty in measurement space)

    d² = νᵀ S⁻¹ ν
```

**Scalar case** (one measurement, one uncertain state dimension):

```
    d = |z - predicted| / sqrt(P_component + R)
```

`P_component + R` is the sum of "how uncertain we are about the state" plus "how uncertain
the sensor is." Together they define the expected spread of the innovation.

---

## 3.3 The 2-Sigma Gate (robot)

```
    Innovation space (2D example):

    ┌────────────────────────────────────────┐
    │                                        │
    │        REJECT         •                │   ← outlier
    │                                        │
    │              ┌─────────┐               │
    │         •   /           \    •         │
    │            │  ACCEPT     │             │
    │            │   (d < 2)   │             │
    │         •   \           /    •         │
    │              └─────────┘               │
    │                                        │
    │    The ellipse shape comes from S⁻¹:   │
    │    correlated axes → tilted ellipse    │
    │    uncorrelated → axis-aligned ellipse │
    └────────────────────────────────────────┘

    The boundary is at d = 2 (2 sigma).
    Area inside ellipse ≈ 95% of expected innovations.
    Only 5% of valid measurements fall outside — if d > 2, reject.
```

**Implementation:** In ` update()`, the rejection check is:

```cpp
    // Pseudocode from update() ~
    double mahal_dist = innovation / sqrt(P_xx + R);
    if (mahal_dist > 2.0) {
        // reject this measurement
        continue;
    }
```

---

## 3.4 Worked Example — Mahalanobis Distance

**Scenario:** Robot moving east, approaches a N-S line at X = 2.00m.

**Given:**
```
    State estimate:      x̂⁻(x) = 2.05m   (dead-reckoning puts us 5cm east of line)
    State covariance:    P⁻(x,x) = 0.010 m²
    Measurement:         z = 2.00m         (line-sensor reports line at exactly this X)
    Measurement noise:   R = 0.005 m²      (line-sensor has ~7cm std dev)
```

**Step 1: Compute innovation**
```
    ν = z - H x̂⁻ = 2.00 - 2.05 = -0.05m
```

**Step 2: Compute innovation standard deviation**
```
    S = H P⁻ Hᵀ + R
      = P⁻(x,x) + R          (H = [1,0,0] so H P⁻ Hᵀ = P⁻(x,x))
      = 0.010 + 0.005
      = 0.015 m²

    sqrt(S) = 0.122m
```

**Step 3: Mahalanobis distance**
```
    d = |ν| / sqrt(S) = 0.05 / 0.122 = 0.41
```

**Decision: ACCEPTED** (d = 0.41 < 2.0) ✓

---

**Now try with very tight covariance:**
```
    P⁻(x,x) = 0.001 m²   (state is very confident — tight covariance)

    S = 0.001 + 0.005 = 0.006 m²
    d = 0.05 / sqrt(0.006) = 0.05 / 0.077 = 0.65
```
**Decision: ACCEPTED** (d = 0.65 < 2.0) ✓ — still passes because 5cm is within 2σ.

---

**Now try with a large drift:**
```
    Innovation = 0.4m  (robot is 40cm from the line — something went wrong)
    P⁻(x,x) = 0.010 m²

    S = 0.015 m²
    d = 0.4 / 0.122 = 3.28
```
**Decision: REJECTED** (d = 3.28 > 2.0) ✗ — the measurement is 3.3σ away from prediction.

This is the "silent rejection" scenario. The line-sensor fired, the measurement was valid,
but the filter rejected it because dead-reckoning had drifted so far that the correct
position appeared to be an outlier.

---

## 3.5 The Danger Zone: When Tight P Causes Rejections

**Counter-intuitive:** A tighter covariance P can cause MORE rejections, not fewer.

```
    Scenario: Robot has been on a long straight with many line-sensor updates.
    P⁻(x,x) is driven very low, say 0.0001 m².

    Next measurement has a genuine 8cm offset (real drift, not outlier):
    S = 0.0001 + 0.005 = 0.0051 m²
    d = 0.08 / sqrt(0.0051) = 0.08 / 0.071 = 1.12    ← still accepted

    But at 12cm offset:
    d = 0.12 / 0.071 = 1.68    ← still accepted

    At 15cm offset:
    d = 0.15 / 0.071 = 2.11    ← REJECTED

    With looser covariance P = 0.010:
    d = 0.15 / 0.122 = 1.23    ← accepted!
```

**Key insight:** After a slip event, P gets inflated to infinity (estimator detects slip and
sets P → ∞). This WIDENS the acceptance gate, allowing the next line-sensor update to be
accepted even with a large innovation — which is exactly correct behavior. The filter recovers
by accepting a measurement it would have rejected pre-slip.

---

# PART 4 — AMR CLAMP UPDATE (NON-STANDARD KALMAN)

---

## 4.1 Standard Kalman Update vs AMR Clamp Update

**Standard Kalman filter** computes a gain K and updates the state:

```
    Standard KF update:
    K = P⁻ Hᵀ (H P⁻ Hᵀ + R)⁻¹       ← Kalman gain
    x̂⁺ = x̂⁻ + K (z - H x̂⁻)          ← state update (weighted blend)
    P⁺ = (I - K H) P⁻                 ← covariance update

    K is between 0 and 1:
      K → 0: sensor has much higher noise than state uncertainty → barely moves
      K → 1: sensor is much more accurate than state → jump to sensor reading
```

**AMR XY update** (from ``):

```
    AMR clamp update:
    x̂⁺(x) = clamp(x̂⁻(x), meas.x_min, meas.x_max)
    P⁺(x,x) = min(P⁻(x,x), (meas.x_range)² / 12)

    where meas.x_range = meas.x_max - meas.x_min
```

These are NOT the same operation. Why does AMR use clamp instead of Kalman gain?

---

## 4.2 Why Clamp Is Correct for Line-Sensor

The line-sensor gives a **range** measurement: "the line is somewhere between X=1.985m and
X=2.015m." A Gaussian model is wrong here — there is no single "best guess" within the range;
the true position is equally likely anywhere in the 3cm band. The right model is **uniform**.

```
    Probability density over X:

    Gaussian state estimate:                Uniform line-sensor measurement:
         ▲ p(x)                                  ▲ p(z|x)
         │    ╭───╮                               │
         │   ╱     ╲                             ─┤─────────────────┤─
         │  ╱       ╲                             │
         │╱           ╲                           └──────────────────── X
         └──────────── X                          1.985   2.015
              2.05

    After combining:                        Clamp result:
         ▲ p(x|z)                                ▲ p(x|z)
         │ ┌─────┐                               │ ┌─────────┐
         │ │     │ ← truncated Gaussian          │ │ uniform │
         │ │     │   (state within range)        │ │  band   │
         └─┴─────┴──── X                        └─┴─────────┴─── X
           1.985 2.015                            1.985     2.015
```

The **clamp** operation is geometrically equivalent to multiplying the Gaussian state by the
uniform measurement likelihood and renormalizing — the result is the state truncated to the
measurement range.

---

## 4.3 The Variance-of-Uniform Formula

For a uniform distribution over `[a, b]`:

```
         (b - a)²
    σ² = ────────
             12

    Derivation:
      E[X] = (a + b) / 2
      Var[X] = E[(X - μ)²]
             = ∫_a^b (x - (a+b)/2)² · (1/(b-a)) dx
             = (b-a)² / 12    ✓
```

**floor sensor range example:**
```
    Range = 0.06m (6cm span, ±3cm from line)
    a = 1.970m, b = 2.030m

    σ²_uniform = (0.06)² / 12 = 0.0036 / 12 = 0.0003 m²
    σ_uniform  = 0.0173m = 1.73cm
```

The covariance update rule `P⁺ = min(P⁻, σ²_uniform)` says:
- If the state was already more certain than the line-sensor → keep the current covariance.
- If the state was less certain (larger P) → tighten it to the line-sensor's uniform variance.

---

## 4.4 Clamp Update Decision Tree

```
                  Is state already inside measurement range?
                          ┌──────────┴──────────┐
                         YES                    NO
                          │                     │
              No position change          Snap to nearest edge
              x̂⁺ = x̂⁻                  (clamp to [min, max])
                          │                     │
                          └──────────┬──────────┘
                                     │
                     Is P⁻ < σ²_uniform?
                          ┌──────────┴──────────┐
                         YES                    NO
                          │                     │
                Keep P (already tight)    Set P = σ²_uniform
                P⁺ = P⁻                  (loosen to match range)
```

**Key insight:** The covariance update is asymmetric — it can only DECREASE (tighten).
It never increases P based on a line-sensor measurement. Only `predict()` increases P
(via process noise Q). This means a single bad line-sensor reading that passes the Mahalanobis
gate can drive P very low, making subsequent measurements harder to accept.

---

# PART 5 — OFFSET AMBIGUITY CHECK

---

## 5.1 The Duplicate Line Problem

```
    Floor view (Y axis):

    ─────────────────────────────────────────────  Y = 4.500m (line A)
    ─────────────────────────────────────────────  Y = 3.000m (line B)
    ─────────────────────────────────────────────  Y = 1.500m (line C)

    Line spacing = 1.5m

    Robot is at Y = 2.980m (near line B, small offset = 0.020m from B)

    But ALSO: Y = 1.520m would give the SAME line-sensor offset from line C!
    (robot at +0.020m above line C looks identical to robot at -0.020m below line B)
```

A single line-sensor crossing is ambiguous between:
- The robot is near the **intended** line at the predicted position, or
- The robot is near the **adjacent** line (robot has drifted by ±1 line_spacing)

---

## 5.2 Probability Ratio Gate

AMR resolves the ambiguity with a likelihood ratio test:

```
    Hypothesis H0: robot is at the primary line (consistent with state estimate)
    Hypothesis H1: robot is at the secondary line (N lines away)

    Compute p(z | H0) and p(z | H1) using the current state distribution.

    If p(z | H1) / p(z | H0) > threshold:
        → ambiguous → REJECT measurement
    Else:
        → H0 is sufficiently more likely → ACCEPT
```

In practice, if the robot state estimate is well within a lane (P is small), H1 requires the
robot to be ~1.5m away from its estimated position — extremely unlikely — so the ratio is tiny
and the measurement is accepted. But if P is large (after a slip, or after long dead-reckoning),
H1 becomes plausible, and the measurement may be rejected.

---

## 5.3 Full Update Decision Sequence

```
    Line-Sensor fires a measurement z at time t
                    │
                    ▼
    ┌───────────────────────────────┐
    │  1. Advance state to time t   │  ← predict() to sync timestamps
    │     (grow P via process noise)│
    └───────────────┬───────────────┘
                    │
                    ▼
    ┌───────────────────────────────┐
    │  2. is_reliable check         │  ← if False: EXIT (SPI frame miss, all-zero)
    └───────────────┬───────────────┘
                    │ (is_reliable = True)
                    ▼
    ┌───────────────────────────────┐
    │  3. Mahalanobis distance      │  ← if d > 2: EXIT (silent rejection)
    │     d = |ν| / sqrt(P + R)    │
    └───────────────┬───────────────┘
                    │ (d ≤ 2)
                    ▼
    ┌───────────────────────────────┐
    │  4. Offset ambiguity check    │  ← if ratio > threshold: EXIT (wrong lane)
    │     probability ratio test    │
    └───────────────┬───────────────┘
                    │ (unambiguous)
                    ▼
    ┌───────────────────────────────┐
    │  5. Clamp state to range      │  ← x̂⁺ = clamp(x̂⁻, meas.min, meas.max)
    │  6. Update covariance         │  ← P⁺ = min(P⁻, (range)²/12)
    └───────────────────────────────┘
```

---

# PART 6 — AMR CODE CONNECTION

---

## 6.1 Full Math-to-Code Mapping

| Math concept | AMR source location | Notes |
|---|---|---|
| Innovation `ν = z - H x̂⁻` | ` update()` ~L1950 | Computed as `(meas.value - state_at_t.x)` |
| Innovation covariance S | `update()` ~L1960 | `P_xx + R` for scalar case |
| Mahalanobis distance d | `update()` ~L1962 | `innovation / sqrt(S)` |
| 2σ gate | `update()` ~L1964 | `if (d > 2.0) continue;` |
| Ambiguity ratio | `update()` ~L1970 | Computed against secondary line hypothesis |
| Clamp state | `update()` ~L1985 | `clamp(state, meas.min, meas.max)` |
| Uniform variance | `update()` ~L1990 | `(meas.range)² / 12` |
| Covariance min | `update()` ~L1992 | `cov = min(cov, uniform_var)` |
| Theta Kalman update | `update()` ~L2050 | Standard K gain used (not clamp) |
| Reliability gate | `processOrthogonalCrossing()` L876 | `if (!is_reliable) return;` |
| Reliability gate | `processParallelCrossing()` L1027 | Same pattern |
| All-black surface | `detectAllBlack()` L1412 | Blanket rejection |
| Pose measurement gen | `computeMeasurements()` | Converts IR pairs to x/y/theta bounds |

---

## 6.2 The `is_reliable` Trap — Why Duplicates Are Invisible

```
    Timeline of line-sensor messages at 50Hz (20ms intervals):

    t=0ms    msg: {offset=-0.015m, is_reliable=True}   ← fresh reading
    t=20ms   msg: {offset=-0.015m, is_reliable=True}   ← DUPLICATE (SPI retry)
    t=40ms   msg: {offset=-0.015m, is_reliable=True}   ← DUPLICATE
    t=60ms   msg: {offset=-0.018m, is_reliable=True}   ← fresh reading (robot moved)
```

All four messages have `is_reliable=True`. The estimator's `is_reliable` gate at 
only catches SPI frame misses and all-zero readings (hardware-level failures). It has no
concept of "stale data from a previous read cycle."

**The filter's perspective:**
- It sees three measurements at t=0ms, t=20ms, t=40ms that all say -0.015m
- Each one passes the Mahalanobis gate (innovation stays near zero since state isn't changing
  between them — or covariance is tight from the first update)
- Each one executes the clamp update, re-confirming the same position
- P is driven even lower each time (the filter becomes MORE confident)
- When the fresh reading at t=60ms arrives, P is very tight
- If the robot has actually moved 5cm and the fresh reading reflects this, the innovation
  may now EXCEED the 2σ gate because P is so tight

```
    Result: Duplicates → over-confident state → genuine updates rejected

    This is the core mechanism behind "silent rejection" in SCL bags.
```

---

## 6.3 Theta Update Is Different (Standard Kalman)

For heading θ, AMR uses the standard Kalman update with gyro bias correction:

```
    z_θ = measured heading from line-sensor (line direction gives θ constraint)
    bias = estimated gyro bias (states b_x, b_y, b_z)

    H_θ = [0, 0, 1, 0, 0, 0, ...]   ← only sees θ, not XY

    K = P⁻ Hᵀ / (H P⁻ Hᵀ + R_θ)   ← standard Kalman gain
    θ̂⁺ = θ̂⁻ + K (z_θ - θ̂⁻)        ← weighted update
    P⁺ = (1 - K H) P⁻               ← covariance shrinks
```

The reason theta uses a proper Kalman gain while XY uses clamp:
- θ measurement from the line-sensor line direction IS a precise point measurement (the line
  has a known orientation, and the robot either aligns or doesn't) — Gaussian noise is honest.
- XY measurement is a range (the robot is somewhere within the IR span) — uniform is honest.

---

# PART 7 — PUTTING IT ALL TOGETHER: BAG DIAGNOSIS

---

## 7.1 Reading Covariance Traces to Diagnose Rejection Causes

When investigating a bag where the robot drifts off path despite line-sensor activity:

```
    Covariance P_xx trace:

    P_xx  │
    0.010 │──╮
          │  │  (growing during prediction)
    0.001 │  ╰──┐  (line-sensor update — P drops)
          │     │
    0.0003│     ╰────┐  (multiple updates — very tight)
          │          │
    0.0003│          ════════════  ← stuck at minimum (duplicates!)
          │
          └──────────────────────────────────── time

    When P stays flat at ~0.0003 for many seconds, the robot is receiving
    line-sensor updates (P is being "refreshed") but not fresh independent ones.
    Duplicates keep confirming the same position → P never grows.

    A genuine fresh update that differs even slightly now has d > 2:
    d = 0.05 / sqrt(0.0003 + 0.005) = 0.05 / 0.072 = 0.69  ← still accepted
    (in this case OK — but see the danger zone example in 3.5 above)
```

---

## 7.2 Summary — What to Remember

| Concept | Formula / Rule | AMR Code | Diagnostic signal in bag |
|---|---|---|---|
| Innovation | `ν = z - H x̂⁻` | `update()` L1950 | Large ν with small P → silent rejection |
| Mahalanobis gate | `d = ν / sqrt(P + R) < 2` | `update()` L1964 | P very tight after duplicate flood → gate tightens |
| Clamp update | `x̂⁺ = clamp(x̂⁻, min, max)` | `update()` L1985 | State jumps to range boundary if outside |
| Uniform variance | `σ² = (range)² / 12` | `update()` L1990 | Sets floor on P after update |
| P update | `P⁺ = min(P⁻, σ²_unif)` | `update()` L1992 | P can only decrease from line-sensor |
| Ambiguity check | Likelihood ratio < threshold | `update()` L1970 | Triggered when P is large and two lanes plausible |
| Reliability gate | `if (!is_reliable) return` | L876, L1027 | Only catches hardware-level SPI fails |
| Duplicate blind spot | `is_reliable=True` on stale data | L876 (no timestamp check) | 11–30% of line-sensor messages; over-tightens P |
| Theta uses KF gain | `K = P Hᵀ / (H P Hᵀ + R)` | `update()` L2050 | Theta covariance decreases smoothly |

---

## Further Reading

- `02-kalman-filter.md` — derivation of the Kalman gain K and why it minimises posterior variance
- `04-imu-fusion.md` — how gyro bias interacts with the theta update and why drift accumulates
- `05-failure-modes.md` — full classification of bag-observable failure signatures with timestamps
- `` lines 876, 1027, 1412, 1900–2100 — the exact gates and update logic
- `computeMeasurements()` — how raw IR sensor pairs become `[x_min, x_max, y_min, y_max, θ_min, θ_max]`
