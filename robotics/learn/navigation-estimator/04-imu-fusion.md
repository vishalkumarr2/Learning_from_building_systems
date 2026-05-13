# 04 — IMU Fusion & Gyroscope Integration
### How OKS `imuCallback()` keeps theta honest

**Prerequisite:** `02-kalman-filter.md` — you need to understand Kalman gain and innovation before this makes sense. `03-sensorbar-correction.md` helps but is not required.
**Unlocks:** `05-slip-detection.md` — once you know how heading errors accumulate and how the Kalman filter corrects them, you can understand why slip events are so dangerous (they corrupt the odometry-based theta prediction that IMU then tries to correct).

---

## Why Should I Care? (OKS Project Context)

Heading error is the silent killer in AMR navigation. A robot with a 1° heading bias that drives 5 metres has drifted **87 mm sideways** before any correction fires. At 3°, it's 262 mm — enough to miss a bin slot entirely.

The the navigation estimator uses wheel odometry as its primary motion source (`predict()`), but wheel odometry alone can't maintain accurate heading over long distances — it accumulates angular error from unequal wheel slip, floor surface transitions, and asymmetric load. The IMU's gyroscope provides an independent angular velocity measurement that runs at ~100 Hz, far faster than any sensorbar update. `imuCallback()` fuses these two sources to keep θ (heading) well-estimated between sensorbar corrections.

Understanding this material lets you answer:

- **"Why is the robot's heading 8° off when the tile sequence shows only 2° of encoder drift?"** → The gyro bias estimate has gone stale, or the robot ran through vibration that corrupted the bias EMA. You can check by comparing `omega_z` readings against the robot's commanded angular velocity in the bag.
- **"Why did the estimator flag COLLISION when the robot just drove over a tile gap?"** → Because the floor discontinuity produced a gyro angular acceleration spike exceeding `collision_threshold`. You can calculate this from the geometry before even opening the bag.
- **"Why does OKS use full Kalman gain for theta but clamp for XY?"** → Because the IMU gives a point measurement in angular space (same state dimension), while sensorbar gives a range measurement that needs asymmetric handling. This note makes that distinction mathematically concrete.

---

# PART 1 — THE IMU SENSOR

## 1.1 What a MEMS Gyroscope Actually Measures

A gyroscope measures **angular velocity** — how fast the sensor is rotating, in radians per second. It does **not** directly measure angle; heading is derived by integrating angular velocity over time.

The OKS robot's IMU sits on the chassis with the **z-axis pointing upward**. This means the z-axis gyroscope (`omega_z`, ω_z) measures rotation in the horizontal plane — exactly the heading change we care about for 2-D navigation.

```
         IMU MOUNTING ON OKS AMR (top view)
         =====================================

                    FRONT  ↑ X
                           |
              +------------+------------+
              |            |            |
              |   +--------+--------+   |
              |   |    ω_z ↻       |   |   ← z-axis: pointing UP
         Y ←--+---+  [IMU BOX]     +---+      ω_z > 0 : CCW rotation
              |   |                |   |      ω_z < 0 : CW rotation
              |   +----------------+   |
              |                        |
              +------------------------+

         Raw IMU output:
           omega_x, omega_y  ← pitch/roll, NOT USED by the navigation estimator
           omega_z           ← yaw rate, fed into imuCallback()
           accel_x, accel_y  ← linear accel, NOT USED by the navigation estimator
           accel_z           ← gravity + vertical accel, NOT USED

         OKS only uses omega_z from the IMU.
         This is the angular velocity around the vertical axis.
```

> **Configuration note:** The **2D navigation estimator** is configured to use only ω_z.
> The ICM-42688-P outputs all 6 axes at full rate; the estimator simply ignores the others.
> If you see a future 3D estimator configuration, or need to investigate a gyro-related
> bug, the full IMU output is always available in the bag on `/robotXX/imu/data`.

**Key insight:** The gyroscope does not know where "north" is, what angle the robot is at, or even which direction it was pointing when powered on. It only knows *how fast the robot is currently rotating*. Heading is a computed quantity that accumulates from the first sample.

## 1.2 How MEMS Gyroscopes Work (Physical Intuition)

A MEMS (Micro-Electro-Mechanical Systems) gyroscope uses the **Coriolis effect**. A tiny vibrating proof mass is driven at a known frequency. When the chip rotates, the Coriolis force deflects the mass perpendicular to its drive direction. The deflection is proportional to angular velocity.

**What this means for errors:**
- Any vibration at the same frequency as the drive resonance looks like rotation → **vibration-induced noise**
- Temperature changes alter the resonant frequency of the proof mass → **temperature-dependent drift**
- Power supply noise modulates the drive amplitude → **electrical noise**
- Manufacturing imperfections create a persistent offset → **bias**

These error sources are why a gyroscope cannot simply be integrated indefinitely. They are the reason `imuCallback()` needs bias compensation and why the Kalman filter is essential.

## 1.3 Taxonomy of Gyro Errors

There are three fundamentally different error types, with different effects and different remedies:

```
         GYRO ERROR TAXONOMY
         ====================

  Error Type       |  Signal Character     |  Effect on θ       |  OKS remedy
  -----------------+-----------------------+--------------------+------------------
  White Noise      |  Zero-mean, random    |  Random walk        |  Kalman filter
  (ARW)            |  σ ∝ √(time)          |  √t growth          |  (averages out)
                   |                       |                     |
  Bias (constant)  |  Fixed DC offset      |  Linear drift       |  EMA bias
                   |  same sign always     |  θ_err = b × t     |  estimator
                   |                       |                     |
  Random Walk      |  Slowly drifting mean |  Quadratic drift    |  Sensorbar
  (in bias)        |  Correlated samples   |  accelerates over   |  correction
                   |                       |  many minutes       |  resets θ

  ARW = Angular Random Walk (units: rad/√s, or deg/√hr in spec sheets)
  Bias = "Zero-Rate Output" when static (units: rad/s or deg/hr)
```

**White noise** is the unavoidable background — random measurement uncertainty on every sample. It causes a **random walk** in the integrated angle: after N steps the expected angle error is σ_noise × √N × Δt. This cannot be compensated; the Kalman filter manages it by weighting the IMU measurement appropriately.

**Bias** is a systematic offset: the sensor reads `b` rad/s even when the robot is perfectly still. Over time `t`, this causes linear heading drift of `b × t`. This *can* be estimated and corrected — which is exactly what `imuCallback()` does via its EMA bias estimator.

**Random walk in the bias** is the slow, correlated drift of the bias value itself due to temperature and aging. This causes the bias estimate to become wrong over timescales of minutes to hours. It's the reason sensorbar corrections are ultimately necessary even with perfect bias compensation.

## 1.4 Worked Example: What Happens if You Ignore Bias

Suppose the gyro has a bias of `b = 0.01 rad/s` (a realistic value for a budget MEMS sensor).

```
        BIAS ACCUMULATION OVER TIME
        ============================

        True ω_z = 0 (robot stationary, or moving straight)
        Measured ω_z = 0.01 rad/s (pure bias)
        Integrated without correction: θ_err = b × t

         t = 1 s   → θ_err = 0.010 rad  =  0.57°    (negligible)
         t = 10 s  → θ_err = 0.100 rad  =  5.73°    (noticeable on long drives)
         t = 30 s  → θ_err = 0.300 rad  = 17.2°     (robot misses every target)
         t = 60 s  → θ_err = 0.600 rad  = 34.4°     (robot drives into walls)

        Position error for 3 m/s robot at t=10s:
          lateral_err = 3 × t × sin(θ_err) ≈ 3 × 10 × 0.100 = 3.0 m
          (completely off track after just 10 seconds!)
```

**This is why bias estimation is not optional.** Even a well-manufactured sensor with `b = 0.001 rad/s` accumulates 3.4° over 60 seconds. Without correction, any run longer than ~10 seconds will show systematic heading drift in the bags you analyse.

---

# PART 2 — INTEGRATION AND DRIFT

## 2.1 The Integration Formula

Heading angle is computed by summing angular velocity over time — the **discrete sum** approximation to a continuous integral:

```
                               N-1
  θ(t_N) = θ(t_0)  +   Σ   ω_z(t_i) × Δt_i
                              i=0

  where:
    θ(t_0)    = initial heading (from previous state)
    ω_z(t_i)  = measured angular velocity at step i [rad/s]
    Δt_i      = time between samples i and i+1 [s]
    N         = number of IMU samples

  At 100 Hz: Δt_i ≈ 0.010 s (10 ms)
```

In OKS `imuCallback()` this is one line:
```
theta += (omega_z - gyro_bias_estimate) * dt
```

Simple — but the devil is in what `gyro_bias_estimate` contains and how `dt` is computed.

## 2.2 Error Accumulation Mechanics

Every sample contributes an error term:
```
  Actual reading:  ω_z_measured = ω_z_true  +  bias  +  noise_i

  Integrated error after N steps:
    θ_err = Σ (bias × Δt)  +  Σ (noise_i × Δt)
          = bias × T          + Σ (noise_i × Δt)
          ↑                    ↑
    systematic drift      random walk
    (grows linearly       (grows as √T,
     with time T)          slower but unavoidable)
```

The systematic drift (bias term) dominates at short timescales. The random walk dominates at long timescales once bias is well-estimated. This is why the EMA bias estimator matters most during the first minute of operation.

## 2.3 Quantifying Bias-Driven Drift — Reference Table

```
        BIAS MAGNITUDE vs. HEADING ERROR OVER TIME
        =============================================

        Bias       | 10 s    | 30 s    | 60 s    | 120 s   | 5 min
        -----------+---------+---------+---------+---------+--------
        0.001 rad/s| 0.010 r | 0.030 r | 0.060 r | 0.120 r | 0.300 r
                   | 0.57°   | 1.72°   | 3.44°   | 6.88°   | 17.2°
        -----------+---------+---------+---------+---------+--------
        0.010 rad/s| 0.100 r | 0.300 r | 0.600 r | 1.200 r | 3.000 r
                   | 5.73°   | 17.2°   | 34.4°   | 68.8°   | 171.9°
        -----------+---------+---------+---------+---------+--------
        0.100 rad/s| 1.000 r | 3.000 r | 6.000 r | 12.00 r | 30.00 r
                   | 57.3°   | 171.9°  | 343.8°  | >>360°  | >>360°

        Lateral position error = drive_speed × time × sin(θ_err)
        At 0.5 m/s, b=0.01 r/s, t=30s:
          lateral = 0.5 × 30 × sin(0.3 rad) ≈ 4.4 m  ← robot crashes
```

**Key insight:** The middle row (`b = 0.010 rad/s`) represents a plausible worst-case for a cheap sensor without bias compensation. This is why every OKS bag with suspicious heading drift should have `omega_z` checked against `cmd_vel.angular.z` when the robot is stationary — any non-zero `omega_z` reading during zero-command periods is pure bias.

## 2.4 Visualizing Divergence

```
        TRUE HEADING vs. INTEGRATED HEADING (uncompensated bias)
        ===========================================================

  θ (rad)
   1.0 |                                      ╱╱╱ integrated
       |                                  ╱╱╱  (with bias)
   0.8 |                              ╱╱╱
       |                          ╱╱╱
   0.6 |                      ╱╱╱
       |                  ●●●●●●●●  ← actual heading
   0.4 |             ●●●●
       |         ●●●●
   0.2 |    ●●●●●
       |●●●●
   0.0 +----+----+----+----+----+----+----+----→ t (s)
       0    5   10   15   20   25   30   35   40

        Slope of integrated curve = bias (rad/s)
        Slope of true curve = actual angular velocity commanded

        After 40 s at 0.01 rad/s bias, integrated θ is 0.4 rad (23°) high.
        If the robot was actually driving straight (true ω = 0), every
        position update for 40 seconds has been computed with a wrong heading.
```

---

# PART 3 — BIAS ESTIMATION

## 3.1 What Bias Is and When It Can Be Estimated

Bias is the gyroscope's output when the robot is **truly stationary** — any non-zero reading at zero angular velocity is bias. The challenge: the robot is not always stationary. During motion, `omega_z` contains both true rotation and bias, and they cannot be separated from the IMU signal alone.

The OKS approach: **use a slow Exponential Moving Average (EMA) that updates continuously** but with a very small learning rate α. When the robot moves, the bias estimate barely changes (correct — we don't know the true bias from a motion reading). When the robot is stationary for many samples, the EMA gradually converges toward the true bias (correct — all `omega_z` readings are pure bias).

## 3.2 The EMA Update Rule

The bias estimator update in `imuCallback()` is:

```
  new_bias = (1 - α) × old_bias  +  α × omega_z_current

  Equivalently:
  new_bias = old_bias  +  α × (omega_z_current - old_bias)

  where:
    α  = learning rate, very small (e.g., 0.001 to 0.01)
    omega_z_current = raw gyro reading (includes bias + true rotation)
```

**This update is always applied**, not just when stationary. The reasoning: during motion, `omega_z_current - old_bias ≈ true_rotation ± noise`, which has zero mean over many samples if the robot rotates symmetrically. Over thousands of samples, the bias estimate drifts toward the true bias. The convergence is slow by design — α is chosen so the time constant τ = 1/α steps allows for robust averaging.

```
        EMA TIME CONSTANT
        ==================

          τ (in samples) = 1/α
          τ (in seconds)  = 1/(α × f_imu)

          α = 0.001, f_imu = 100 Hz:
            τ_samples = 1000 samples
            τ_seconds = 10 seconds

          This means the bias estimate has a "memory" of ~10 seconds.
          A sudden change in true bias (e.g., temperature step) takes
          ~3τ = 30 seconds to be 95% absorbed.

          During a 3 m/s emergency stop (maybe 0.5 s), the bias estimate
          barely changes — the angular deceleration contributes < α×0.5 = 0.0005
          to the estimate, negligible vs. the true bias of ~0.008 rad/s.
```

## 3.3 Worked Example: EMA Bias Convergence

Setup: α = 0.001, true bias = 0.008 rad/s, initial estimate = 0.005 rad/s.
Robot is stationary throughout, so `omega_z_measured = 0.008 rad/s` at each step.

```
  Update rule: bias_new = (1 - 0.001) × bias_old + 0.001 × 0.008

  Step 0: bias = 0.005000
  Step 1: bias = (0.999)(0.005000) + (0.001)(0.008)
               = 0.004995 + 0.000008
               = 0.005003     ← tiny step toward 0.008

  Step 2: bias = (0.999)(0.005003) + (0.001)(0.008)
               = 0.004998 + 0.000008
               = 0.005006

  Step 5: bias ≈ 0.005015
  Step 10: bias ≈ 0.005030
  Step 100: bias ≈ 0.005296
  Step 1000: bias ≈ 0.006918   ← ~1/3 of the way from 0.005 to 0.008
  Step 3000: bias ≈ 0.007716   ← 95% converged

  General formula after N stationary steps:
    bias(N) = bias_true - (bias_true - bias_0) × (1-α)^N
            = 0.008     - (0.008 - 0.005) × (0.999)^N
            = 0.008     - 0.003 × e^(-N/1000)   [approximately]
```

**Key insight:** With α = 0.001, the estimator needs ~3000 stationary samples (30 seconds at 100 Hz) to reach 95% of the true bias. During a typical warehouse run, the robot stops at pick/place locations frequently enough for this to converge. If a robot just arrived from cold storage and immediately starts a long run, the bias estimate will be wrong for the first minute of operation — this shows up in bags as systematic theta drift in the first few minutes.

## 3.4 Why α Must Be Very Small

If α is too large (e.g., 0.1), the bias estimate tracks `omega_z` closely. During fast rotation, `omega_z` is large — the bias estimate shoots up to that value and then the correction `omega_z - bias_estimate ≈ 0` — the robot appears to be stationary in the heading estimate. This destroys the whole point of IMU fusion.

If α is too small (e.g., 0.00001), the bias estimate barely moves even after hours of stationary time. The initial bias estimate at startup matters enormously, and the system cannot adapt to temperature-induced drift during a shift.

The choice of α is a tuning parameter that reflects the expected timescale of bias drift. For OKS AMRs operating at roughly constant temperature in a warehouse, α ≈ 0.001 is typical.

---

# PART 4 — KALMAN UPDATE FOR THETA

## 4.1 Why IMU Gets Full Kalman Gain (Not a Clamp)

In OKS, two different correction strategies are used depending on the measurement source:

```
        CORRECTION STRATEGY COMPARISON
        ================================

        Source          |  Measurement type      |  OKS correction method
        ----------------+------------------------+------------------------
        Sensorbar       |  Range in Y (lateral)  |  CLAMP (asymmetric)
        IMU gyro        |  Point in θ (heading)  |  Full Kalman gain

        WHY DIFFERENT?

        Sensorbar gives: "I can see the bar at distance d ± uncertainty"
          → This is a SIGNED range with physical limits; the bar is
            either detected or not. Innovation can be large and asymmetric.
          → Clamping prevents overcorrection when bar detection is noisy.

        IMU gives: θ_imu = Σ(ω_z - bias) × Δt
          → This is a POINT ESTIMATE in the same dimension as the state.
          → Both the prediction (encoder) and measurement (IMU) have
            Gaussian uncertainty. The Kalman formula is exact.
          → No reason to clamp; if both sensors agree, correction is tiny.
            If they disagree, one has failed — the collision detector handles that.
```

## 4.2 The State and Measurement

For the heading fusion step:
```
  State variable:   θ_enc  =  heading from encoder integration  (from predict())
  State variance:   P_θ    =  heading uncertainty from odometry model

  Measurement:      θ_imu  =  Σ (ω_z_i - bias_estimate) × Δt_i
                             (accumulated since last Kalman update)
  Measurement noise: R_θ   =  IMU angle noise variance
                             = σ_gyro² × T  (grows with integration time T)
```

## 4.3 The Kalman Update Equations

```
        THETA KALMAN UPDATE (imuCallback)
        ================================

        1. INNOVATION (how much IMU disagrees with odometry):
           ν = θ_imu - θ_enc

        2. INNOVATION COVARIANCE:
           S = P_θ + R_θ

        3. KALMAN GAIN:
           K = P_θ / S  =  P_θ / (P_θ + R_θ)

           Note: K ∈ [0, 1]
             K → 0 when P_θ << R_θ  (IMU noisy, trust odometry)
             K → 1 when P_θ >> R_θ  (IMU trusted, odometry uncertain)
             K = 0.5 when P_θ = R_θ (equal trust, split the difference)

        4. STATE UPDATE:
           θ_new = θ_enc + K × ν
                 = θ_enc + K × (θ_imu - θ_enc)

        5. COVARIANCE UPDATE:
           P_θ_new = (1 - K) × P_θ
```

## 4.4 Visualizing the Fusion

```
        HEADING ESTIMATION: GAUSSIAN PICTURE
        ======================================

  Probability
      |
      |      Encoder estimate           IMU estimate
      |      (wider, less certain)      (narrower after low noise run)
      |
      |        +-------+                   +--+
      |       /         \                 /    \
      |      /           \               /      \
      |     /             \             /        \
      |    /               \           /          \
      |   /    P_θ wide     \   ν     /  R_θ small \
      |  /                   \ ←→   /              \
      | /                     ●    ●                \
      +--+-------+-------+-------+-------+-------+--→ θ
         θ_enc-2σ   θ_enc   fused   θ_imu-σ  θ_imu

                         ↑
                     Fused result:
                     θ_new = θ_enc + K×ν
                     K is large because P_θ > R_θ
                     → fused result pulled toward θ_imu
```

When the two distributions are far apart (large ν), either:
1. The robot turned fast and there's legitimate disagreement (Kalman handles this — the posterior is between them, narrower than either source).
2. One sensor has failed — large ν with no commanded motion is suspicious, which is what the collision detector checks.

## 4.5 Worked Example: Single Kalman Update

```
  Given:
    θ_enc = 0.200 rad   (from odometry)
    P_θ   = 0.004 rad²  (odometry heading variance)
    θ_imu = 0.185 rad   (from IMU integration)
    R_θ   = 0.001 rad²  (IMU angular noise variance)

  Step 1 — Innovation:
    ν = 0.185 - 0.200 = -0.015 rad  (IMU says robot turned less than encoder)

  Step 2 — Innovation covariance:
    S = 0.004 + 0.001 = 0.005 rad²

  Step 3 — Kalman gain:
    K = 0.004 / 0.005 = 0.80

  Step 4 — State update:
    θ_new = 0.200 + 0.80 × (-0.015)
          = 0.200 - 0.012
          = 0.188 rad

  Step 5 — Covariance update:
    P_θ_new = (1 - 0.80) × 0.004 = 0.0008 rad²

  Interpretation:
    - K = 0.80 means IMU gets 80% weight (odometry was uncertain, IMU was precise)
    - Final heading 0.188 is 80% of the way from encoder to IMU estimate
    - Variance dropped from 0.004 to 0.0008 — fusion improved certainty 5×
    - The encoder had more uncertainty, possibly from a recent slip event
```

---

# PART 5 — COLLISION DETECTION

## 5.1 What the Collision Detector Checks

The collision detector in `imuCallback()` monitors **consecutive gyro reading differences**:

```
  angular_acceleration_estimate = (omega_z_current - omega_z_previous) / dt

  if |angular_acceleration_estimate| > collision_threshold:
      NavigatorState = COLLISION
```

Note this is computing an approximate **angular acceleration** (change in angular velocity per unit time), not the angular velocity itself. This is the right quantity: a collision typically causes a **sudden stop or angular jerk**, which shows up as a large Δω_z in a single timestep.

## 5.2 Physical Meaning

```
        NORMAL OPERATION vs. COLLISION EVENT
        ======================================

  Normal turn (smooth):
    ω_z profile:  0 → 0.1 → 0.3 → 0.5 → 0.5 → 0.3 → 0.1 → 0

    Δω_z per 10ms step ≈ 0.02 to 0.04 rad/s
    angular accel ≈ 2 to 4 rad/s²   ← well below 2.0 rad/s² threshold?
    (Depends on how fast the turn ramps up)

  Collision (robot hits wall at 0.5 m/s):
    ω_z profile:  0.0 → 0.0 → 0.0 → [IMPACT] → 2.5 → -0.3 → 0.0

    Δω_z in single step = 2.5 - 0.0 = 2.5 rad/s  in Δt = 0.01 s
    angular accel = 2.5 / 0.01 = 250 rad/s²  ← WAY above threshold

  Rough floor (tile gap at 0.3 m/s):
    ω_z profile:  0.0 → 0.0 → 0.4 → 0.0 → 0.0

    Δω_z = 0.4 rad/s in Δt = 0.01 s
    angular accel = 40 rad/s²  ← may exceed threshold!
    ↑ This is the false-positive source (covered in Part 6)
```

## 5.3 Consequence of COLLISION State

When COLLISION fires:
1. `NavigatorState` is set to `COLLISION` (a terminal state in the OKS FSM).
2. The robot stops accepting motion commands.
3. A manual restart (supervisor acknowledgement) is required before operation resumes.
4. The position estimate at the time of COLLISION is preserved but marked unreliable.

This is intentionally aggressive. A collision is a safety event — the robot may have moved objects or encountered a person. The OKS design treats any ambiguous sudden jerk as "assume the worst." The consequence for false positives (floor bumps, machinery vibration) is operational downtime, which is why floor quality matters for OKS deployments.

## 5.4 False Positive Sources and How to Identify Them

```
        FALSE POSITIVE COLLISION EVENTS
        =================================

  Source               | ω_z signature          | Distinguisher
  ---------------------+------------------------+---------------------------
  Floor tile gap       | Single spike, ω_z       | Correlated with robot
  (robot drives over)  | returns to near zero    | position on map;
                       | immediately             | repeatable at same location
                       |                         |
  External vibration   | Sustained oscillation,  | ω_z oscillates even when
  (forklift nearby)    | not single spike        | robot is stationary;
                       |                         | check cmd_vel = 0 during event
                       |                         |
  EMI / power spike    | Random spike, no        | No correlated acceleration;
                       | physical context        | check IMU power rail
                       |                         |
  Actual collision     | Large spike, possibly   | Robot was moving;
                       | followed by velocity    | cmd_vel → 0 afterwards;
                       | step-down               | robot position is wrong
```

**How to identify in bags:**
1. Open the incident bag and find the COLLISION timestamp.
2. Look back 50ms (5 IMU samples) at `omega_z`. Find the step where Δω_z exceeded threshold.
3. Check `cmd_vel.angular.z` at the same time. If it was zero and robot was accelerating linearly, this is likely a floor bump false positive.
4. Check the robot's tile position on the warehouse map. If this location has a reported tile edge, confirm false positive.
5. File as infrastructure issue, not estimator bug.

## 5.5 The Threshold Choice

The collision threshold is a design parameter trading off:
- **Lower threshold** → catches softer collisions, more false positives from floor bumps
- **Higher threshold** → misses some collisions, fewer false positives

For a robot driving at `v = 0.3 m/s` over a `2 cm` step bump:
```
  Approximate angular impulse from bump:
    The front wheel rises h = 0.02 m in time t_contact ≈ h/v = 0.02/0.3 ≈ 0.067 s
    The robot tips forward, then returns: angular velocity ≈ h/(L × t_contact)
      where L ≈ 0.3 m is wheelbase
    ω_z_peak ≈ 0.02 / (0.3 × 0.067) ≈ 0.99 rad/s

  Angular acceleration (in one IMU step of 10ms):
    Δω_z/Δt ≈ 0.99/0.01 = 99 rad/s²

  With collision_threshold = 2.0 rad/s² ... wait, the threshold is on
  Δω_z per sample (not divided by dt):
    Δω_z = 0.99 rad/s in one step → EXCEEDS 2.0 rad/s threshold if threshold
    is on raw Δω_z. Check the OKS code carefully — if threshold is 2.0 rad/s²:
      angular_accel = 0.99/0.01 = 99 rad/s² >> 2.0 rad/s²  → collision fires

  If threshold is on raw Δω_z (not divided by dt = 2.0 rad/s):
    Δω_z = 0.99 < 2.0 → collision does NOT fire

  → The threshold units determine sensitivity. Always check the code.
    See Section D of exercise 04-log-diagnosis.md for the worked calculation.
```

---

# PART 6 — OKS CODE CONNECTION

## 6.1 imuCallback() — Full Walkthrough

```cpp
// , imuCallback() — annotated with math from this note

void Estimator::imuCallback(const ImuMessage& msg) {
    double omega_z = msg.angular_velocity.z;     // [rad/s], raw gyro
    double dt = computeDt(msg.header.stamp);     // [s], ~0.010 s at 100 Hz

    // ── COLLISION DETECTION ────────────────────────────────────────────
    // Checks PART 5: Δω_z > threshold?
    double delta_omega = std::abs(omega_z - prev_omega_z_);
    if (delta_omega > collision_threshold_) {
        state_ = NavigatorState::COLLISION;      // terminal state → requires restart
        return;                                  // no further updates
    }
    prev_omega_z_ = omega_z;

    // ── BIAS ESTIMATION ───────────────────────────────────────────────
    // EMA update: PART 3, equation: bias = (1-α)×bias + α×ω_z
    gyro_bias_ += alpha_ * (omega_z - gyro_bias_);

    // ── THETA INTEGRATION ─────────────────────────────────────────────
    // PART 2: θ += (ω_z - bias) × dt
    double omega_corrected = omega_z - gyro_bias_;
    theta_imu_ += omega_corrected * dt;          // accumulates since last KF update

    // ── KALMAN UPDATE FOR THETA ───────────────────────────────────────
    // PART 4: standard KF equations
    double innovation = theta_imu_ - state_.theta;  // ν = θ_imu - θ_enc
    double S = cov_theta_ + R_theta_;               // S = P_θ + R_θ
    double K = cov_theta_ / S;                      // K = P_θ / S

    state_.theta += K * innovation;                 // θ_new = θ_enc + K×ν
    cov_theta_   *= (1.0 - K);                      // P_θ_new = (1-K)×P_θ

    theta_imu_ = state_.theta;                      // reset IMU accumulator
}
```

**Important:** `theta_imu_` is reset after each Kalman update. It doesn't accumulate indefinitely — it accumulates *between* Kalman updates and is then absorbed into the state. This prevents unbounded numerical drift in the accumulator.

## 6.2 Full Code-to-Math Mapping Table

| Concept | Math | OKS Code Location | Notes |
|---------|------|-------------------|-------|
| Angular velocity input | ω_z [rad/s] | `msg.angular_velocity.z` in `imuCallback()` | Only z-axis used |
| Sample interval | Δt [s] | `computeDt(msg.header.stamp)` | ~10 ms at 100 Hz |
| Bias EMA update | `b += α(ω_z − b)` | `gyro_bias_ += alpha_ * (omega_z - gyro_bias_)` | α very small, always applied |
| Corrected rate | ω_z − b | `omega_corrected = omega_z - gyro_bias_` | Fed into integration |
| Theta integration | `θ_imu += (ω_z−b)×Δt` | `theta_imu_ += omega_corrected * dt` | Accumulates between KF steps |
| Innovation | `ν = θ_imu − θ_enc` | `innovation = theta_imu_ - state_.theta` | Sign: positive if IMU > encoder |
| Kalman gain | `K = P_θ/(P_θ+R_θ)` | `K = cov_theta_ / S` | K ∈ [0,1] |
| Theta correction | `θ += K×ν` | `state_.theta += K * innovation` | Full gain, not clamped |
| Cov update | `P_θ *= (1−K)` | `cov_theta_ *= (1.0 - K)` | Fusion reduces uncertainty |
| Collision check | `|Δω_z| > thresh` | `delta_omega > collision_threshold_` | Before any update |
| COLLISION state | terminal FSM state | `state_ = NavigatorState::COLLISION` | Requires restart |
| Accumulator reset | after KF update | `theta_imu_ = state_.theta` | Prevents accumulator drift |

## 6.3 Why XY Uses Clamp but Theta Uses Kalman

This is a common question from engineers new to the OKS codebase.

```
        SENSORBAR (XY) vs. IMU (THETA): CORRECTION METHOD CHOICE
        ===========================================================

        Sensorbar correction:
          - Measurement: lateral position Y relative to bar feature
          - The bar is a PHYSICAL LINE; detection can be noisy or ambiguous
          - Large innovations can mean: bar was misidentified, robot is far
            off track, or the bar detection algorithm returned an outlier
          - A full Kalman update on a bad bar detection would jerk the
            robot estimate by meters
          - Solution: CLAMP the correction to max_bar_correction per step
          - Accept smaller improvements every step, reject huge jumps

        IMU correction:
          - Measurement: angle from angular velocity integration
          - Both encoder and IMU measure the SAME PHYSICAL QUANTITY (heading)
          - Their uncertainties (P_θ, R_θ) are well-characterised Gaussians
          - Large innovations (> ~0.1 rad) signal a sensor failure →
            handled by COLLISION detector BEFORE the Kalman update runs
          - If we reach the Kalman step, both sensors are healthy →
            full Kalman formula gives the optimal Bayes-fused estimate
          - Clamping would bias the estimate; full gain is correct here

        Summary: Clamp protects against outlier measurements.
                 Kalman is correct when measurements are well-modelled Gaussians.
                 The collision detector is the "outlier rejection" step for IMU.
```

## 6.4 Interaction with predict()

The IMU does **not** replace odometry as the primary motion source. The call sequence is:

```
        EVENT TIMELINE (one control loop iteration, ~10ms)
        ====================================================

        1. Odometry message arrives (encoder-based)
             └─ predict() called
                  - Compute delta_trans, delta_theta from encoder ticks
                  - Propagate (x, y, θ) using unicycle model    [PART 1, note 01]
                  - Inflate covariance: P += Q (process noise)

        2. IMU message arrives (~100 Hz, much faster than odom)
             └─ imuCallback() called
                  - Check collision (PART 5)
                  - Update bias estimate (PART 3)
                  - Integrate theta_imu (PART 2)
                  - Kalman update on theta (PART 4)

        3. Sensorbar message arrives (when robot is over bar)
             └─ onSensorbar() called
                  - Correct Y position (clamped)
                  - Reset lateral covariance

        KEY: delta_theta in predict() comes from ENCODERS, not IMU.
             IMU provides a CORRECTION on top of the encoder-predicted theta.
             IMU runs 10× faster than odom, tightening theta uncertainty
             between each odometry update.
```

---

## Summary — What to Remember

| Topic | Core Fact | OKS Implication |
|-------|-----------|-----------------|
| Gyro measures | Angular velocity ω_z [rad/s], **not** angle | Heading must be integrated; integration accumulates error |
| MEMS bias | A fixed DC offset in the gyro output | `b = 0.01 rad/s` → 34° drift in 60 s if uncorrected |
| EMA bias estimator | `b += α(ω_z − b)`, very small α | Converges in ~3000 steps (30 s); wrong at startup on cold robot |
| Integration formula | `θ += (ω_z − b) × Δt` each step | One line in `imuCallback()`; the bias subtraction is everything |
| Kalman gain for θ | `K = P_θ/(P_θ + R_θ)`, full gain | Odometry uncertain → K large → trust IMU more |
| IMU role | **Correction**, not primary source | Encoder predicts θ; IMU corrects between odometry updates |
| Collision detector | `|Δω_z| > threshold` | Fires on sudden angular jerk; floor bumps are false positive risk |
| COLLISION state | Terminal; requires manual restart | Aggressive by design: safety > uptime |
| Theta vs. XY correction | Theta: full KF; XY: clamp | IMU measurement is Gaussian; sensorbar outliers need rejection |
| Bias drift impact | Grows linearly with time | Always check `omega_z` during zero-cmd periods in incident bags |

---

## Common Bag Analysis Patterns (Quick Reference)

```
  SYMPTOM                    | LIKELY CAUSE              | WHERE TO LOOK IN BAG
  ---------------------------+---------------------------+----------------------
  Steady theta drift,        | Gyro bias not converged   | omega_z vs cmd_vel
  no collision               | or wrong initial estimate | when robot stationary
  ---------------------------+---------------------------+----------------------
  COLLISION at known floor   | Tile gap false positive   | omega_z spike, check
  location every run         | or bump                   | map tile coordinates
  ---------------------------+---------------------------+----------------------
  Large theta jump after     | Bias estimate corrupted   | alpha_ too large, or
  stop-start cycle           | by rotational manoeuvre   | bias drifted on restart
  ---------------------------+---------------------------+----------------------
  Heading OK during straight | Encoder-dominant; IMU     | cov_theta before/after
  run, drifts in turns       | gain K too low            | turns; check R_theta_
  ---------------------------+---------------------------+----------------------
  Random COLLISION events,   | EMI or vibration from     | omega_z oscillates
  no physical obstruction    | nearby machinery          | even when stationary
```
