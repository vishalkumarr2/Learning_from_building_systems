# Exercises: Measurement Models & Mahalanobis Gating
### Chapter 03: Line constraints, innovation gating, and the robot clamp update

**Self-assessment guide:** Work each section in order. Write your answer before expanding the
details block. This chapter is the core of daily bag diagnosis — spend extra time on Section B
(numerical). If you can compute d by hand in under 2 minutes, you can diagnose silent rejections
in the field without a script.

**AMR context:** Every time you see a robot drift off a lane line while the line-sensor is active,
one of the gates in ` update()` is silently dropping measurements. This exercise set
builds the muscle to compute which gate, and why.

---

## Section A — Conceptual Questions

**A1.** Explain the difference between an *orthogonal* and a *parallel* line-sensor measurement
in the the navigation estimator. Which component of the state vector does each update? Why can a single
line crossing only update one state component, not two?

<details><summary>Answer</summary>

**Orthogonal measurement:** The robot crosses a floor line approximately perpendicularly.
The line-sensor fires across the line, measuring the lateral offset. If the line runs N-S
(vertical), the perpendicular direction is E-W, so the measurement updates **X**. If the
line runs E-W (horizontal), it updates **Y**.

**Parallel measurement:** The robot travels along a line. The line-sensor detects the line
running parallel to travel. The perpendicular offset from the robot's path to the line updates
the **opposite lateral axis** from travel direction.

**Why only one component?** A line is a 1D constraint — it eliminates one degree of freedom.
Knowing "the robot is on this line" tells us the perpendicular position, but nothing about where
along the line the robot is. The measurement matrix H for a scalar measurement has only one
non-zero entry: `H = [1, 0, 0]` (for X) or `H = [0, 1, 0]` (for Y). The heading θ remains
unobservable from this single crossing.

**ELI15:** A train on a track knows it's on the track (lateral = 0) but the track itself says
nothing about how far along the route it has travelled.
</details>

---

**A2.** The Mahalanobis distance gate rejects a measurement when `d > 2`. Explain why a
*tighter* state covariance P can paradoxically cause MORE measurements to be rejected.
Give a concrete scenario where this leads to navigation failure.

<details><summary>Answer</summary>

**Mahalanobis distance:** `d = |ν| / sqrt(P_component + R)`

The denominator is `sqrt(P + R)`. When P is small (tight covariance), the denominator shrinks,
so the same innovation ν produces a larger d.

**Concrete scenario:**
1. Robot travels a long straight corridor, receiving many line-sensor updates.
2. Due to duplicate line-sensor messages (11–30% of SCL frames), P is repeatedly confirmed
   without genuine new information. P shrinks to near the uniform-variance floor (~0.0003 m²).
3. A wheelspin event causes the robot to drift 12cm laterally. The estimator does NOT detect
   this (P is tight, so slip detection threshold may not trigger).
4. The next genuine line-sensor update has innovation ν = 0.12m.
   `d = 0.12 / sqrt(0.0003 + 0.005) = 0.12 / 0.072 = 1.67` ← still accepted (barely)
5. But if ν = 0.15m: `d = 0.15 / 0.072 = 2.08` ← REJECTED.
6. The state is now stuck at the wrong position, and every subsequent line-sensor update
   (which correctly reports the line location) appears as an outlier to the over-confident filter.

**Root cause:** Duplicate measurements made P artificially tight. A genuine recovery measurement
was rejected because the filter "knew" too confidently where it was.
</details>

---

**A3.** AMR uses a clamp update (`x̂⁺ = clamp(x̂⁻, min, max)`) instead of the standard
Kalman gain update (`x̂⁺ = x̂⁻ + K ν`). For what physical reason is the clamp model more
honest for line-sensor XY measurements? What probability distribution does the clamp represent?

<details><summary>Answer</summary>

**Physical reason:** The line-sensor reports a *range*, not a point. The robot is somewhere
within the span of activated IR sensors — approximately ±3cm around the line centroid. There
is no "most likely position" within this band; the robot could genuinely be anywhere in it.

**Probability distribution:** The clamp update corresponds to multiplying the Gaussian state
prior by a **uniform likelihood** over `[meas.min, meas.max]` and renormalising. The result is
the state truncated to the measurement range — equivalent to a truncated Gaussian.

**Contrast with Kalman gain:** The standard update `x̂⁺ = x̂⁻ + K ν` assumes the measurement
is a point estimate corrupted by Gaussian noise. Using this for a range measurement would
place the updated state at a weighted average between x̂⁻ and the band centre — which could
be *outside* the measurement range if x̂⁻ is close to an edge. The clamp prevents this
physically impossible outcome.

**Covariance update `P⁺ = min(P⁻, (range)²/12)`:** The variance of a uniform distribution
over width w is `w²/12`. Setting P to this value means the filter honestly represents
"after this measurement, we know the robot is somewhere in the band, with uniform confidence."
</details>

---

**A4.** What is the purpose of the offset ambiguity check in ` update()`?
Describe the scenario where it would reject an otherwise valid line-sensor measurement.
Under what state-covariance condition is the ambiguity check most likely to trigger?

<details><summary>Answer</summary>

**Purpose:** The floor has many parallel lines with known spacing (e.g. 1.5m apart). A single
line-sensor crossing is geometrically identical whether the robot is near line A or near the
adjacent line B (the offset from the line looks the same; only the absolute position differs).
The ambiguity check uses a likelihood ratio to determine which hypothesis (near A or near B)
is better supported by the current state estimate.

**Rejection scenario:** The robot is at Y = 3.02m (near line B at Y = 3.0m). But after a
slip event, the state estimates Y = 4.48m (near line A at Y = 4.5m). The line-sensor fires
and reports offset = +0.02m. This is consistent with both:
- Robot near line B: predicted Y = 3.0 + 0.02 = 3.02m (correct)
- Robot near line A: predicted Y = 4.5 - 0.02 = 4.48m (the wrong hypothesis that matches
  the drifted state estimate)

The probability ratio `p(H1) / p(H0)` is high → measurement rejected as ambiguous → the
robot cannot self-correct with this line-sensor reading.

**When most likely to trigger:** When P is **large** (after a slip, after long dead-reckoning
without updates). Large P means the state distribution is spread over multiple line spacings,
making both hypotheses plausible. With tight P, the state is localised within a single lane
and the secondary hypothesis is strongly penalised.
</details>

---

**A5.** The `is_reliable` flag in line-sensor messages is checked at lines 876 and 1027 in
``. Explain what `is_reliable=False` indicates and why it only catches ~1% of
problematic messages. What type of problem does it miss entirely?

<details><summary>Answer</summary>

**`is_reliable=False` indicates:**
- An SPI frame miss (the STM32 firmware did not deliver a fresh sensor reading to the Jetson
  in time — the ring buffer contained stale data)
- All-zero sensor readings (the line-sensor reported all sensors as identical, which indicates
  a hardware-level fault — broken sensor, blocked IR emitters, or power issue)

**Why only ~1% of messages:** These are hardware-level failures. The line-sensor firmware sets
`is_reliable=False` only when it *knows* the reading is invalid (CRC error, timeout, all-zero).
In normal operation, 99% of messages have well-formed data that passes hardware checks.

**What it misses entirely:** SPI *retry duplicates* — when the Jetson re-requests the last
line-sensor frame and receives an identical copy of a previous valid reading. This data is
hardware-valid (correct CRC, non-zero, properly formatted) so `is_reliable=True` is set.
But the *timestamp* is stale and the *content* is identical to the previous message.

In SCL bags, 11–30% of line-sensor messages are such duplicates. All of them have `is_reliable=True`.
The filter has no way to distinguish them from genuine updates without explicit duplicate
detection (checking if the content and/or timestamp matches the previous message).

**Fix direction:** The estimator would need a "content change" or "message sequence number"
check in addition to the `is_reliable` gate to catch duplicates.
</details>

---

## Section B — Numerical Exercises

For each scenario below, compute the Mahalanobis distance d and state whether the measurement
is **ACCEPTED** (d ≤ 2) or **REJECTED** (d > 2).

**Given constants for all scenarios:**
- Measurement noise: `R = 0.005 m²`
- Mahalanobis threshold: `d_max = 2.0`

---

**B1. Scenario: Normal operation (should be accepted)**

```
    State estimate:      x̂⁻(x) = 1.520m
    State covariance:    P⁻(x,x) = 0.008 m²
    Line-Sensor reading:   z = 1.500m   (line at X = 1.5m)
```

<details><summary>Solution</summary>

```
    Innovation:        ν = z - x̂⁻ = 1.500 - 1.520 = -0.020m

    Innovation var:    S = P⁻(x,x) + R = 0.008 + 0.005 = 0.013 m²
    Innovation std:    √S = 0.1140m

    Mahalanobis d:     d = |ν| / √S = 0.020 / 0.1140 = 0.175

    Decision: ACCEPTED ✓  (d = 0.18 << 2.0)
```

**Interpretation:** The robot is 2cm from its predicted line crossing. With moderate covariance,
this is well within the acceptance ellipse. The clamp update will snap the state to within
`[meas.min, meas.max]` (likely already inside).
</details>

---

**B2. Scenario: Borderline case — robot drifted after long corridor (accepted, just)**

```
    State estimate:      x̂⁻(x) = 1.270m
    State covariance:    P⁻(x,x) = 0.025 m²   (large — long dead-reckoning stretch)
    Line-Sensor reading:   z = 1.500m   (line at X = 1.5m, robot drifted ~23cm)
```

<details><summary>Solution</summary>

```
    Innovation:        ν = 1.500 - 1.270 = +0.230m

    Innovation var:    S = 0.025 + 0.005 = 0.030 m²
    Innovation std:    √S = 0.1732m

    Mahalanobis d:     d = 0.230 / 0.1732 = 1.328

    Decision: ACCEPTED ✓  (d = 1.33 < 2.0)
```

**Interpretation:** The robot has drifted 23cm but the large P (built up over the long
dead-reckoning stretch) widens the acceptance ellipse enough to accept the measurement.
This is correct and desired behaviour — large P allows recovery from genuine drift.

Note: if P had been tight (e.g. 0.001 m²) due to duplicate flooding:
`d = 0.230 / sqrt(0.001 + 0.005) = 0.230 / 0.0775 = 2.97` → **REJECTED** — the recovery
measurement would have been silently dropped.
</details>

---

**B3. Scenario: Rejection — genuine outlier (wrong lane or sensor fault)**

```
    State estimate:      x̂⁻(x) = 1.490m
    State covariance:    P⁻(x,x) = 0.010 m²
    Line-Sensor reading:   z = 3.000m   (robot at X=1.5m seeing X=3.0m line — wrong lane)
```

<details><summary>Solution</summary>

```
    Innovation:        ν = 3.000 - 1.490 = +1.510m

    Innovation var:    S = 0.010 + 0.005 = 0.015 m²
    Innovation std:    √S = 0.1225m

    Mahalanobis d:     d = 1.510 / 0.1225 = 12.33

    Decision: REJECTED ✗  (d = 12.3 >> 2.0)
```

**Interpretation:** A measurement 1.5m away from the predicted position is a clear outlier —
12σ is astronomically unlikely under a Gaussian model. The Mahalanobis gate correctly rejects
this as either a wrong-lane reading, a spurious reflection, or a sensor fault.

Note: this would also be caught by the offset ambiguity check at step 4 — the probability
ratio would flag this as belonging to a different line entirely.
</details>

---

## Section C — Python Implementation

**Task:** Implement the standard Kalman filter update step as a Python function.
This is the *standard* update (not the robot clamp variant) — it forms the mathematical
foundation for understanding why AMR deviated from it.

**Function signature:**

```python
import numpy as np

def ekf_update(
    x: np.ndarray,    # Prior state estimate, shape (n,)
    P: np.ndarray,    # Prior covariance, shape (n, n)
    z: np.ndarray,    # Measurement, shape (m,)
    H: np.ndarray,    # Measurement matrix, shape (m, n)
    R: np.ndarray,    # Measurement noise covariance, shape (m, m)
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Standard Kalman filter update step.

    Returns:
        x_post: Posterior state estimate, shape (n,)
        P_post: Posterior covariance, shape (n, n)
        mahal_d: Mahalanobis distance (scalar) — for gating decisions
    """
```

**Expected behaviour:**
- Innovation: `nu = z - H @ x`
- Innovation covariance: `S = H @ P @ H.T + R`
- Kalman gain: `K = P @ H.T @ np.linalg.inv(S)`
- State update: `x_post = x + K @ nu`
- Covariance update: `P_post = (I - K @ H) @ P`
- Mahalanobis distance: `sqrt(nu.T @ inv(S) @ nu)`

<details><summary>Reference implementation</summary>

```python
import numpy as np

def ekf_update(
    x: np.ndarray,
    P: np.ndarray,
    z: np.ndarray,
    H: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = x.shape[0]
    nu = z - H @ x                          # innovation
    S = H @ P @ H.T + R                     # innovation covariance
    K = P @ H.T @ np.linalg.inv(S)          # Kalman gain
    x_post = x + K @ nu                     # state update
    P_post = (np.eye(n) - K @ H) @ P        # covariance update (Joseph form safer)
    mahal_sq = float(nu.T @ np.linalg.inv(S) @ nu)
    mahal_d = np.sqrt(max(mahal_sq, 0.0))   # numerical guard
    return x_post, P_post, mahal_d


# --- Test cases ---

def test_identity_measurement():
    """H=I, R small: posterior should be close to measurement."""
    x = np.array([2.05, 1.00, 0.1])
    P = np.diag([0.01, 0.01, 0.01])
    z = np.array([2.00, 1.00])
    H = np.array([[1, 0, 0],
                  [0, 1, 0]])
    R = np.diag([0.005, 0.005])
    x_post, P_post, d = ekf_update(x, P, z, H, R)
    assert x_post[0] < x[0], "X should move toward measurement"
    assert P_post[0, 0] < P[0, 0], "Covariance should shrink"
    assert d < 2.0, f"Expected accepted measurement, got d={d:.3f}"
    print(f"x_post[0] = {x_post[0]:.4f}  (expected ~2.017)")
    print(f"P_post[0,0] = {P_post[0,0]:.5f}  (expected ~0.00333)")
    print(f"Mahal d = {d:.3f}  (expected 0.41)")

def test_large_innovation_rejected():
    """Innovation far from prediction: d should exceed 2."""
    x = np.array([1.49])
    P = np.array([[0.010]])
    z = np.array([3.00])
    H = np.array([[1.0]])
    R = np.array([[0.005]])
    _, _, d = ekf_update(x, P, z, H, R)
    assert d > 2.0, f"Expected rejection, got d={d:.3f}"
    print(f"Mahal d = {d:.3f}  (expected ~12.3 → REJECTED)")

def test_covariance_shrinks():
    """After update, covariance must be smaller than prior."""
    x = np.array([0.0, 0.0])
    P = np.diag([1.0, 1.0])
    z = np.array([0.1])
    H = np.array([[1.0, 0.0]])
    R = np.array([[0.1]])
    _, P_post, _ = ekf_update(x, P, z, H, R)
    assert P_post[0, 0] < P[0, 0], "X covariance must shrink after measurement"
    assert abs(P_post[1, 1] - P[1, 1]) < 1e-10, "Y covariance unchanged (H only observes X)"
    print(f"P_post = {np.diag(P_post)}")

if __name__ == "__main__":
    test_identity_measurement()
    test_large_innovation_rejected()
    test_covariance_shrinks()
    print("All tests passed.")
```

**Expected output:**
```
x_post[0] = 2.0167  (weighted average: x moved 2/3 toward measurement)
P_post[0,0] = 0.00333  (1/3 of original P — sensor reduces uncertainty)
Mahal d = 0.409  (well within 2σ gate)
Mahal d = 12.327  (REJECTED)
P_post = [0.09091  1.0]   (X tightened, Y unchanged)
All tests passed.
```

**Key observation for the robot:** Notice how `x_post[0] = 2.0167`, *not* 2.0 (the measurement).
The Kalman gain K = 0.667 here, so the state moves 2/3 of the way toward the measurement.
In AMR's clamp update, the state would instead be forced inside `[meas.min, meas.max]`
regardless of K — a fundamentally different (and for range measurements, more honest) update.
</details>

---

## Section D — AMR-Specific: Uniform Variance Gate Analysis

**Problem:**

A line-sensor measurement has a range of **0.06m** (the IR array span covers 6cm — 3cm either
side of the detected line centroid):
- `meas.min = 2.470m`
- `meas.max = 2.530m`
- `meas.range = 0.060m`

The current state estimate and covariance are:
- `x̂⁻(x) = 2.510m`  (state estimate is inside the range)
- Two cases: `P⁻(x,x) = 0.010 m²` and `P⁻(x,x) = 0.0001 m²`

Answer these questions:

1. What variance `σ²_uniform` does AMR assign to this measurement range?
2. In each P case, what does the covariance update rule `P⁺ = min(P⁻, σ²_uniform)` produce?
3. What does the clamp update do to the state (x̂⁺) in both cases?
4. Suppose `x̂⁻(x) = 2.450m` (state is OUTSIDE the range). What happens?

<details><summary>Solution</summary>

**1. Uniform variance:**

```
    σ²_uniform = (meas.range)² / 12
               = (0.060)² / 12
               = 0.0036 / 12
               = 0.0003 m²
    σ_uniform  = 0.01732m ≈ 1.73cm
```

**2. Covariance update:**

```
    Case A: P⁻ = 0.010 m²  (large — 10x bigger than σ²_uniform)
        P⁺ = min(0.010, 0.0003) = 0.0003 m²
        → Covariance tightened to the line-sensor range variance
        → After this update the filter is "as confident as the line-sensor allows"

    Case B: P⁻ = 0.0001 m²  (very tight — 3x smaller than σ²_uniform)
        P⁺ = min(0.0001, 0.0003) = 0.0001 m²
        → Covariance UNCHANGED — the state is already MORE precise than the measurement
        → The line-sensor cannot improve on what the filter already "knows"
```

**3. Clamp update (state estimate inside range in both cases):**

```
    x̂⁻ = 2.510m   (inside [2.470, 2.530])

    x̂⁺ = clamp(2.510, 2.470, 2.530) = 2.510m

    → No change to state position in either P case.
    → The state is already consistent with the measurement range; no correction needed.
```

**4. State outside range:**

```
    x̂⁻ = 2.450m   (below meas.min = 2.470m)

    x̂⁺ = clamp(2.450, 2.470, 2.530) = 2.470m

    → State SNAPPED to the nearest edge of the range.
    → The innovation here would be:
        ν = (meas centre) - x̂⁻ = 2.500 - 2.450 = 0.050m

    Check Mahalanobis gate first (using Case A, P = 0.010):
        d = 0.050 / sqrt(0.010 + 0.0003) = 0.050 / 0.1015 = 0.493
        → ACCEPTED ✓ (d < 2)

    After clamp: x̂⁺ = 2.470m (snapped from 2.450m to range boundary)
    Covariance: P⁺ = min(0.010, 0.0003) = 0.0003 m²
```

**Summary table for this measurement:**

| Scenario | x̂⁻ | P⁻ | x̂⁺ (after clamp) | P⁺ | Gate decision |
|---|---|---|---|---|---|
| State inside range, large P | 2.510m | 0.010 m² | 2.510m (unchanged) | 0.0003 m² | Accepted |
| State inside range, tight P | 2.510m | 0.0001 m² | 2.510m (unchanged) | 0.0001 m² | Accepted |
| State outside range, large P | 2.450m | 0.010 m² | 2.470m (snapped) | 0.0003 m² | Accepted (d=0.49) |

**Key takeaway:** A 6cm line-sensor range sets the covariance floor at `0.0003 m²` (σ = 1.73cm).
This is larger than the tight case `P = 0.0001 m²`. So in Case B, repeated valid line-sensor
updates cannot tighten P below 0.0003 m² — the floor is set by the physical measurement range.
However, **duplicate** line-sensor messages (with stale identical content) will keep executing the
clamp update and confirming `P⁺ = min(P, 0.0003)` — the covariance is correctly bounded but
the position is stale. A future fresh measurement then has a very tight P = 0.0003 m², and if
the genuine line position disagrees by more than `2 × sqrt(0.0003 + 0.005) ≈ 14.5cm`, it will
be rejected. This is the 14.5cm rejection threshold for a state that has been over-confirmed
by duplicates.
</details>

---

## Bonus — Connecting Back to SCL Bag Analysis

When you see `P_xx` trace in a bag that goes:

```
    0.010 → 0.0030 → 0.0010 → 0.0003 → 0.0003 → 0.0003 → [flat]
```

...and the flat region spans 3+ seconds, that is the duplicate flood signature. The covariance
has reached the `σ²_uniform` floor and is being confirmed there by stale readings. The robot's
position belief is locked at one value while the physical robot may have moved. The next
fresh line-sensor reading that reports a different position will be evaluated against
`d = ν / sqrt(0.0003 + 0.005) = ν / 0.0724`. Any drift above `2 × 0.0724 = 14.5cm`
causes silent rejection — and the robot continues drifting.

This is why line-sensor duplicate detection upstream of the estimator (sequence numbers, content
hashing, or timestamp monotonicity checks) would fix the silent-rejection failure class without
changing any estimator math.
</details>
