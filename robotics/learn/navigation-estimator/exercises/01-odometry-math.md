# Exercises: Dead-Reckoning & Wheel Odometry

### Chapter 01: Dead-Reckoning and Wheel Odometry

**Self-assessment guide:** Write your answer on paper before expanding the details block. If you can answer 80% without peeking, you're ready for `02-kalman-filter.md`.

---

## Section A — Conceptual Questions

**A1.** A robot with perfect encoders drives in a perfect straight line for 10 metres on a concrete floor, then a shiny tile floor. Its position estimate ends up 8 cm off from where a tape measure says it should be. No line-sensor updates occurred. What is the most likely cause, and how would you distinguish it from a baseline calibration error?

<details><summary>Answer</summary>

The most likely cause is **wheel slip on the shiny tile floor**. When the floor surface changes from concrete (high friction) to tile (low friction), the wheels can slip during acceleration or deceleration — the encoder counts revolutions but the wheel is not making full ground contact. This is a **random error** that only appears on tile and doesn't accumulate on concrete.

To distinguish from baseline calibration error:
- **Baseline error** is *systematic*: every 10 m run drifts by the same amount in the same direction. The drift is proportional to how much the robot turned, not to floor surface. Run the robot in a straight line (no turns) on both surfaces — baseline error won't change because `Δθ = (s_R - s_L) / B` only matters when `s_R ≠ s_L`.
- **Wheel slip** is *stochastic*: the drift changes between runs and is correlated with the transition from one surface to another. The direction of drift is semi-random (whichever wheel slips first).

In a bag, slip shows up as a sudden `delta_trans` without corresponding `cmd_vel` — the robot moved (encoder counted) but the command did not request it. The the slip detector in `odometryCallback()` checks exactly this: did velocity drop faster than `max_velocity_decrease` allows?

</details>

**A2.** The unicycle model uses `θ + Δθ/2` (midpoint heading) for the position update, not `θ` (old heading) or `θ + Δθ` (new heading). Explain intuitively and mathematically why neither endpoint is correct, and why the midpoint is the best single-direction approximation for small arcs.

<details><summary>Answer</summary>

**Intuitive argument:** The robot starts pointing at `θ` and ends pointing at `θ + Δθ`. It turned continuously during the arc — it was not pointing at either endpoint for "most" of its travel. The midpoint `θ + Δθ/2` is the heading it was pointing at when it was halfway through the arc, which is the best single representative heading for the whole motion.

**Mathematical argument (Taylor expansion):**

The exact position update for a circular arc of radius R and angle Δθ is:
```
x' = x + R × sin(θ + Δθ) − R × sin(θ)
y' = y − R × cos(θ + Δθ) + R × cos(θ)
```

Using the sine/cosine subtraction identities and `s = R × Δθ`:
```
x' = x + s × sin(θ + Δθ) − sin(θ) / Δθ
```
The mean value theorem says `[sin(θ + Δθ) − sin(θ)] / Δθ = cos(θ + c)` for some `c ∈ (0, Δθ)`. The midpoint approximation uses `c = Δθ/2`, which gives a second-order accurate result. Using the endpoint (`c = 0` or `c = Δθ`) is only first-order accurate.

**Numerically:** For `Δθ = 0.1 rad` (about 6°), the midpoint error is O(Δθ³) ≈ 0.0001, while the endpoint error is O(Δθ²) ≈ 0.01 — ten times larger.

</details>

**A3.** AMR's `predict()` does NOT read `odom.twist.covariance` or `odom.pose.covariance` from the incoming ROS message. Your colleague says "this means AMR can't know how reliable the odometry is." Are they right? Explain what AMR uses instead and why the design choice makes sense.

<details><summary>Answer</summary>

Your colleague is partially right but misunderstands the design intent. AMR *can* assess odom reliability — it just does so differently.

**What AMR uses instead:**
1. **Fixed noise parameters** (`k_trans_noise`, `k_rot_noise`, etc.) tuned once during robot calibration. These encode the *expected* reliability of the warehouse robot's specific wheel/encoder hardware.
2. **Staleness detection** on the odom arrival timestamp. If odom hasn't arrived within `2× expected_period`, AMR knows something is wrong with the sensor, regardless of what the covariance field says.
3. **Slip detection** in `odometryCallback()` via velocity window comparison — a runtime check that detects actual slip events rather than relying on the publisher's covariance estimate.

**Why this design makes sense:**
- The base driver setting `odom.twist.covariance` correctly would require *it* to detect slip, which is harder at the driver level.
- Fixed parameters are simpler, more auditable, and less susceptible to bugs in the covariance publisher.
- The noise model parameters can be empirically calibrated from bags, then applied uniformly — no per-message noise injection.

**The trade-off:** AMR cannot adapt to surface-specific reliability (tile vs. concrete) without changing its parameters. Robots on the same floor type have well-matched noise models; robots crossing surface types may see systematic covariance under/over-estimation.

</details>

**A4.** A robot drives in a circle of radius 1 m (constant curvature) for a full 360°. At the end, it should be exactly back at (0, 0, 0°). Due to a 0.5% systematic wheel radius error on the right wheel only, it is not. Qualitatively describe the error: which direction is the position offset, and what is the heading error?

<details><summary>Answer</summary>

**Setup:** Right wheel radius is `r_R = r × 1.005`, left is `r_L = r`. The robot drives what it *thinks* is a circle of radius 1 m.

**Effect on s_R and s_L:** For every encoder tick, the right wheel travels 0.5% further than the robot believes. So `s_R` (as computed by the estimator) is 0.5% *underestimated* relative to true ground contact distance. This means:

- True `s_R > s_L` (right wheel covers more ground per tick than the estimator calculates)
- The estimator computes `Δθ = (s_R - s_L) / B` but uses the wrong `s_R`
- **Actual** `Δθ` is larger than estimated — the robot turns more per unit time than it thinks

**Over 360°:** The robot believes it has returned to heading 0°, but it has actually turned slightly more. The heading error is:
```
heading_error = 0.5% × (2π total turning) = 0.005 × 6.28 = 0.031 rad ≈ 1.8°
```

**Position offset:** Because the robot has turned slightly more (by ~1.8°), its final position is offset from (0,0) in the direction perpendicular to the final leg of the circle. The offset magnitude is approximately `circumference × 0.5% ≈ 6.28 × 0.005 ≈ 31 mm`. The offset direction depends on the last arc segment.

**Key insight:** Systematic wheel errors cause *heading drift* that then causes *position error*. They do not cause pure position error without heading error, because the arc geometry couples them.

</details>

**A5.** The AMR formula for lateral variance includes the term `min(cov_theta, 1e5) × delta_trans²`. After a slip detection event, AMR sets `cov_theta = INF`. Trace through what happens to `lateral_var` for the *next* prediction step, and explain why the `min(..., 1e5)` cap is necessary but also potentially dangerous.

<details><summary>Answer</summary>

**What happens:**

After slip, `cov_theta = INF`. At the next prediction step:
```
capped = min(INF, 1e5) = 1e5

lateral_var = k_trans_lat_noise × delta_trans
            + ...
            + 1e5 × delta_trans²
```

For a typical `delta_trans = 0.01 m`:
```
lateral contribution = 1e5 × (0.01)² = 1e5 × 0.0001 = 10 m²
```

This is an extremely large lateral variance — the robot knows its sideways position is effectively unknown (σ ≈ 3 m). **This is correct behaviour**: after a slip, the robot genuinely doesn't know where it is laterally.

**Why the cap is necessary:**

Without the cap, `INF × delta_trans²` = `NaN` in IEEE 754 arithmetic (0 × INF = NaN in some implementations, though INF × finite = INF in others). More critically, any subsequent Kalman update computation involving `lateral_var = INF` would produce `NaN` innovations and `NaN` gains. The `isFinite()` check in `executePoseGoal()` would then trigger `ESTIMATED_STATE_NOT_FINITE` and abort navigation.

**Why the cap is potentially dangerous:**

`1e5 m²` (σ ≈ 316 m) is finite but still astronomically large for a warehouse that is at most ~100 m wide. In practice this is fine — the line-sensor will soon correct the position. However, if line-sensor updates are *also* failing (e.g., after a slip onto an unmarked floor section), the robot will be in a state where it has huge but *finite* covariance indefinitely. The navigation stack will not trigger `ESTIMATED_STATE_NOT_FINITE`, so it will keep trying to navigate, but every goal will be executed with massive position uncertainty.

**When investigating bags:** If you see `cov_lateral` plateau near `1e5` for more than a few seconds without shrinking, the robot slipped AND line-sensor corrections are not working. Check for `is_reliable=False` on line-sensor messages, or that the robot is not over floor markings.

</details>

**A6.** A junior engineer proposes replacing the midpoint heading integration `(θ + Δθ/2)` with a Runge-Kutta 4th-order (RK4) integration for better accuracy. Is this a good idea for the robot? Consider the frequency of prediction steps, the magnitude of `Δθ` per step, and the cost of the improvement.

<details><summary>Answer</summary>

**Almost certainly not a good idea** for the robot's specific use case, for the following reasons:

**1. Magnitude of Δθ per step is small.** AMR receives odom at 50 Hz. At the robot's maximum angular velocity (typically ~1 rad/s), each step gives:
```
Δθ = 1 rad/s × 0.02 s = 0.02 rad per step
```
The midpoint rule error is O(Δθ³) = O(8×10⁻⁶) radians per step. Over 50 steps (1 second), accumulated midpoint error is ~0.0004 rad (0.02°). RK4 reduces this by another order of magnitude — from 0.02° to 0.002° per second. This is well below line-sensor measurement noise.

**2. Line-Sensor corrections dominate.** AMR corrects position every ~5–10 cm of travel. The dead-reckoning accumulates for at most 0.1 s before being corrected. The dead-reckoning accuracy between line-sensor updates is not the limiting factor.

**3. RK4 requires intermediate evaluations.** RK4 needs 4 function evaluations per step (k₁ through k₄). For odom at 50 Hz with a simple unicycle model, this quadruples computation with no measurable benefit.

**4. The real errors are not integration errors.** Wheel slip, baseline calibration, and floor height variation contribute far more than numerical integration error. Improving integration accuracy does not address these.

**Valid use case for higher-order integration:** A robot dead-reckoning for long stretches without corrections (e.g., GPS-denied navigation over tens of metres) would benefit from RK4. But AMR's line-sensor update frequency makes midpoint rule entirely adequate.

</details>

---

## Section B — Numerical Problems

**B1.** A robot starts at pose `(x=1.0 m, y=2.0 m, θ=30°)`. Its encoders read: left +450 ticks, right +460 ticks. Wheel radius `r = 0.060 m`, encoder resolution `N = 500 ticks/rev`, baseline `B = 0.35 m`.

Compute the final pose `(x', y', θ')` to 4 significant figures.

<details><summary>Solution</summary>

**Step 1 — Arc lengths:**
```
s_L = (450 / 500) × 2π × 0.060
    = 0.900 × 0.37699
    = 0.33929 m

s_R = (460 / 500) × 2π × 0.060
    = 0.920 × 0.37699
    = 0.34683 m
```

**Step 2 — Heading change:**
```
Δθ = (s_R - s_L) / B
   = (0.34683 - 0.33929) / 0.35
   = 0.00754 / 0.35
   = 0.02154 rad  ≈ 1.234°
```

**Step 3 — Forward distance:**
```
s = (s_R + s_L) / 2
  = (0.34683 + 0.33929) / 2
  = 0.34306 m
```

**Step 4 — Midpoint heading:**
```
θ = 30° = 0.52360 rad
θ_mid = 0.52360 + 0.02154/2
      = 0.52360 + 0.01077
      = 0.53437 rad  = 30.617°
```

**Step 5 — Position update:**
```
x' = 1.0 + 0.34306 × cos(0.53437)
   = 1.0 + 0.34306 × 0.85737
   = 1.0 + 0.29413
   = 1.2941 m

y' = 2.0 + 0.34306 × sin(0.53437)
   = 2.0 + 0.34306 × 0.51483
   = 2.0 + 0.17660
   = 2.1766 m

θ' = 0.52360 + 0.02154
   = 0.54514 rad  = 31.234°
```

**Final pose:** `(x' = 1.294 m, y' = 2.177 m, θ' = 31.23°)`

**Sanity check:** The robot moved approximately 0.343 m forward from (1.0, 2.0) while heading ~30°. That should give x ≈ 1.0 + 0.343×cos(30°) = 1.0 + 0.297 = 1.297. Close match (slight difference due to using midpoint heading at 30.6° vs. 30°). ✓

</details>

**B2.** A robot navigates a 50 m straight path in an warehouse. Its right wheel has a true radius of `r_R = 0.0503 m` but the software uses `r_R_nominal = 0.0500 m`. Left wheel is correct. Baseline B = 0.30 m. The robot drives at a constant velocity with equal encoder ticks on both wheels.

(a) Compute the heading drift rate in rad/m of path.
(b) How far along the path before lateral error exceeds 5 cm?
(c) What lateral error has accumulated at 50 m total travel?

<details><summary>Solution</summary>

**Setup:** Equal encoder ticks `n` on both wheels per step. True arc lengths:
```
s_L_true = (n/N) × 2π × 0.0500  (correct)
s_R_true = (n/N) × 2π × 0.0503  (1% larger than nominal)
```

But the software computes:
```
s_L_computed = s_L_true           (correct)
s_R_computed = (n/N) × 2π × 0.0500  (uses nominal, not true)
```

So the software *underestimates* s_R. With equal encoder ticks:
```
s_L_computed = s_R_computed  (they're identical in software)
```
This means `Δθ_computed = 0` — the software thinks the robot is going straight.

But truly:
```
s_R_true > s_L_true
Δθ_true = (s_R_true - s_L_true) / B per step
```

Per unit nominal distance `d` (i.e., per `s_nominal = d`):
```
s_R_true = d × (0.0503/0.0500) = d × 1.006
s_L_true = d

Δθ_true = (d × 1.006 - d × 1.000) / 0.30
         = d × 0.006 / 0.30
         = d × 0.02 rad/m
```

**(a) Heading drift rate = 0.02 rad/m**

(Note: The software sees 0 rad/m heading change because it uses the wrong radius. The robot actually turns 0.02 rad per meter of computed travel.)

**(b) Distance to 5 cm lateral error:**

Lateral error after distance `d`:
```
lateral_error = d × sin(Δθ_accumulated) ≈ d × (0.02 × d) = 0.02 × d²  (for small angles)
```

Set `0.02 × d² = 0.05`:
```
d² = 0.05 / 0.02 = 2.5
d = 1.58 m
```

**Lateral error exceeds 5 cm after only 1.58 m of travel.**

**(c) Lateral error at 50 m:**

```
Δθ_total = 0.02 × 50 = 1.0 rad  (≈ 57°)
lateral_error = 50 × sin(1.0) = 50 × 0.8415 = 42.1 m
```

This is physically impossible in a warehouse — the robot would have hit a wall long before 50 m. What actually happens: the robot spirals, turning ~57° over 50 m. The effective lateral displacement is less than 42 m because the robot's heading has changed direction, but the robot is clearly far off its intended straight path.

**Lesson:** A 0.6% wheel radius error (0.3 mm on a 50 mm wheel) causes navigation failure after ~1.5 m without line-sensor corrections. This illustrates why line-sensor corrections every 10–20 cm are essential.

</details>

**B3.** You are analysing a robot bag. The robot's `cov_theta` is 0.04 rad² at time T. Over the next 0.5 seconds, the robot drives straight (no turning) at 0.3 m/s. The AMR noise parameters are:
- `k_trans_noise = 0.002` (m²/m)
- `k_trans_lat_noise = 0.001` (m²/m)
- `k_rot_noise = 0.05` (rad²/rad)
- `k_time_pos_noise = 0.0001` (m²/s)
- `k_time_rot_noise = 0.0005` (rad²/s)

There are 25 prediction steps (50 Hz), `delta_t = 0.02 s` each. No line-sensor updates occur.

Compute the total `trans_var`, `theta_var`, and `lateral_var` accumulated over this 0.5 s window.

<details><summary>Solution</summary>

**Per-step values:**
```
delta_trans = 0.3 m/s × 0.02 s = 0.006 m per step
delta_theta = 0  (straight line)
delta_t     = 0.02 s
```

**Per-step variances:**
```
trans_var per step = 0.002 × 0.006 + 0 + 0.0001 × 0.02
                   = 0.000012 + 0 + 0.000002
                   = 0.000014 m²

theta_var per step = 0 + 0 + 0.0005 × 0.02
                   = 0.00001 rad²

lateral_var per step = 0.001 × 0.006 + 0 + min(cov_theta, 1e5) × 0.006²
```

For `lateral_var` we need the current `cov_theta`. It starts at 0.04 and grows each step:

After step k: `cov_theta(k) = 0.04 + k × 0.00001`

The `lateral_var` contribution from the cov_theta term for step k:
```
= min(0.04 + k × 0.00001, 1e5) × (0.006)²
≈ (0.04 + k × 0.00001) × 0.000036  (cov_theta stays near 0.04, far from 1e5 cap)
```

**Total trans_var over 25 steps:**
```
total_trans_var = 25 × 0.000014 = 0.000350 m²
σ_trans = √0.000350 = 0.0187 m  ≈ 1.87 cm
```

**Total theta_var over 25 steps:**
```
total_theta_var = 25 × 0.00001 = 0.000250 rad²
σ_theta = √0.000250 = 0.0158 rad  ≈ 0.91°
```

**Total lateral_var over 25 steps:**

Constant term: `25 × 0.001 × 0.006 = 0.000150 m²`

Cov_theta term (sum over steps k=0..24):
```
Σ (0.04 + k×0.00001) × 0.000036
= 0.000036 × Σ (0.04 + k×0.00001)
= 0.000036 × [25 × 0.04 + (0.00001) × (0+1+...+24)]
= 0.000036 × [1.0 + 0.00001 × 300]
= 0.000036 × [1.0 + 0.003]
= 0.000036 × 1.003
= 0.0000361 m²
```

**Total lateral_var = 0.000150 + 0.0000361 = 0.000186 m²**
σ_lateral = √0.000186 = 0.0136 m ≈ 1.36 cm

**Summary after 0.5 s, 0.15 m of travel:**
```
σ_trans   = 1.87 cm  (along-track uncertainty)
σ_theta   = 0.91°    (heading uncertainty)
σ_lateral = 1.36 cm  (cross-track uncertainty)
```

These are realistic values for an warehouse robot between line-sensor updates. The lateral uncertainty is almost as large as the along-track uncertainty, reflecting the `cov_theta × delta_trans²` coupling.

</details>

**B4.** A robot's uint16 encoder counter wraps from 65535 to 3 in a single odom tick. The encoder resolution is `N = 1024 ticks/rev` and wheel radius `r = 0.05 m`. The previous counter value was 65480 (stored as `uint16_t prev = 65480`).

(a) What is the correct `delta` if the cast is `(int16_t)(new_val - prev_val)`?
(b) What is the wrong `delta` if the cast is `(int32_t)(new_val - prev_val)` or `new_val - prev_val` in uint arithmetic?
(c) What `delta_trans` does each case produce?
(d) How large is the `trans_var` spike from the wrong calculation, using `k_trans_noise = 0.002`?

<details><summary>Solution</summary>

**Setup:**
```
prev_val  = 65480  (uint16_t)
new_val   = 3      (uint16_t, after wrap)
```

**(a) Correct int16 cast:**
```
diff = new_val - prev_val  (computed in uint16 arithmetic first)
     = 3 - 65480
     = 3 + 65536 - 65480  (uint16 wraps: 65536 ≡ 0, so 3 - 65480 wraps to 65536 + 3 - 65480)
     = 59  (as uint16)

(int16_t)(59) = 59  ticks  ← 59 ticks forward
```

Verification: From 65480 → 65535 is 55 ticks, then 65535→0 is 1 tick, then 0→3 is 3 ticks. Total = 55 + 1 + 3 = 59 ticks. ✓

**(b) Wrong int32 or uint arithmetic:**
```
(int32_t)(3 - 65480) = 3 - 65480 = -65477  (large negative in 32-bit!)
or as uint32:
(uint32_t)(3) - (uint32_t)(65480) = 4294967299 - 65480 = 4294901819  (enormous positive)
```

The int32 case: -65477 ticks. The uint case: ~4 billion ticks.

**(c) delta_trans:**

Correct (59 ticks):
```
delta_trans = (59 / 1024) × 2π × 0.05
            = 0.05762 × 0.31416
            = 0.01811 m  ≈ 18 mm  (normal step)
```

Wrong int32 (-65477 ticks):
```
delta_trans = (65477 / 1024) × 2π × 0.05
            = 63.94 × 0.31416
            = 20.09 m  ← 20 METRES in one tick!
```

(The sign just determines direction; magnitude is the same for the absolute value case.)

**(d) Variance spike from wrong delta_trans:**

Correct: `trans_var = 0.002 × 0.01811 = 3.62 × 10⁻⁵ m²` (normal)

Wrong: `trans_var = 0.002 × 20.09 = 0.04018 m²`

This is **1100×** the normal variance, producing σ_trans ≈ 0.20 m from a single bad tick. Depending on `k_trans_lat_noise` and current `cov_theta`, lateral variance would also spike.

More critically, a `delta_trans = 20 m` would pass through to the state update, teleporting the estimated position by ~20 metres. The next line-sensor update would either:
1. Reject it (Mahalanobis distance >> 2σ), leaving the state in the wrong place.
2. Or the position would be so far out of range that the robot stops navigating entirely.

**The fix** is always: `int16_t delta = static_cast<int16_t>(new_counter - prev_counter)` in the base driver, before passing to the estimator.

</details>

---

## Section C — Connect to robot

**C1.** You are debugging bag-XXXXX. The bag shows the robot's `cov_xx` and `cov_yy` growing steadily for 8 seconds without any shrink events (no line-sensor updates), then suddenly both spike to approximately `1e5 m²`, then remain there. No velocity commands were issued during this period. Describe:

(a) What sequence of events most likely produced this pattern?
(b) Which robot code path(s) were responsible for each phase?
(c) What would you check in the bag to confirm your hypothesis?
(d) Why did the covariance stop at `1e5` rather than continuing to grow or going to INF?

<details><summary>Answer</summary>

**(a) Most likely sequence of events:**

1. **Phase 1 (0–8 s, gradual growth):** Robot was moving (odom arriving normally). `predict()` ran every 20 ms, adding `k_trans_noise × delta_trans + ... + k_time_pos_noise × delta_t` to `cov_xx`/`cov_yy` on each step. No line-sensor corrections occurred (robot either passed no lines, or line-sensor was unreliable on that floor section).

2. **Phase 2 (sudden spike to 1e5):** A **slip event** was detected. `odometryCallback()` in `` computed that current velocity dropped faster than `max_velocity_decrease` allowed, given the recent velocity window in `_max_odom_vel_tracker`. Result: covariance was set to INF for X, Y, and Theta.

3. **Phase 3 (plateau at 1e5):** After slip, `cov_theta = INF`. On the next prediction step, `lateral_var` used `min(INF, 1e5) = 1e5`, capping lateral covariance. Translational variance also settled due to the staleness scaling with no new motion.

**(b) robot code paths:**

- Phase 1: ` predict()` lines ~396+, the linear noise model additions.
- Phase 2: ` odometryCallback()` lines ~285-310, slip detection block: `if (current_vel + max_velocity_decrease < max_vel) { set covariance INF }`.
- Phase 3: ` predict()` again, but now with `min(cov_theta, 1e5)` cap in `lateral_var` formula.

**(c) What to check in the bag:**

1. `/odom` topic: Look for `twist.linear.x` drop at the moment of the spike. If velocity was ~0.3 m/s and dropped to 0 in a single tick, that's the slip event.
2. `_max_odom_vel_tracker` window: Check if `cmd_vel` had a recent non-zero command before the velocity drop — slip detection only fires if there was a recent velocity command.
3. `/line-sensor` topic: Check `is_reliable` field on all messages in the 8-second window. If `is_reliable=True` but updates are not appearing in the estimator, there may be a different gating issue.
4. `estimated_state.status`: Should transition from `STATUS_OK` (0) to `STATUS_SLIP` (4) at the spike moment.

**(d) Why covariance stopped at 1e5 rather than growing further or going INF:**

After the slip event, if the robot came to rest (or odom became stale), `delta_trans = 0` and `delta_theta = 0`. The noise model adds:
```
trans_var  = 0 + 0 + k_time_pos_noise × delta_t  (only temporal drift)
theta_var  = 0 + 0 + k_time_rot_noise × delta_t
lateral_var = 0 + ... + min(1e5, 1e5) × 0² = 0  (delta_trans = 0!)
```

So after the robot stops, `lateral_var` stops growing (the cov_theta term is `× delta_trans²`, which is 0). The covariance plateau is the combination of: the INF-set-then-capped value from slip detection, plus a small amount of temporal drift that's too small to see on the scale of 1e5.

</details>

**C2.** The AMR noise parameter `k_trans_noise = 0.002` was tuned on a smooth concrete floor. The robot is now deployed on a new site with polished tile floors where wheel slip is 3× more common. You want to re-tune the parameters to reflect the new surface without access to the source code recompilation cycle — you'll update the YAML config file only.

(a) Which parameters should you increase, by roughly how much, and why?
(b) What is the risk of increasing them too much?
(c) There is one parameter you should NOT increase even though slip is higher. Which one, and why?
(d) A colleague suggests "just increase `k_rot_noise` to compensate for lateral drift from slip." Critique this suggestion using the lateral variance formula.

<details><summary>Answer</summary>

**(a) Parameters to increase and rationale:**

- **`k_trans_noise`**: This models slip per unit distance. If slip is 3× more common, increase by 3×: from 0.002 to ~0.006 m²/m. This correctly inflates along-track uncertainty on slippery floors.

- **`k_trans_lat_noise`**: Lateral slip (sideways skidding) is also more common on tile. Increase by similar factor: from whatever baseline to 3× baseline.

- **`k_rot_noise`**: Rotational slip (spinning in place due to high wheel friction differential) is more likely on tile. Increase by 2–3×.

The effect: The estimator becomes less confident between line-sensor updates, which is appropriate. The covariance will grow faster, so line-sensor measurements will have relatively more weight (smaller innovation variance relative to measurement noise).

**(b) Risk of increasing too much:**

If variance parameters are too high, the estimator becomes **over-conservative**. This causes:
1. **Line-Sensor updates accepted too readily**: The Mahalanobis gating rejects measurements if innovation > 2σ of prediction. If σ_prediction is inflated 10×, almost every measurement is within 2σ, including *bad* line-sensor readings (wrong line identification, ghost lines). The filter becomes susceptible to line-sensor false positives.
2. **Covariance blows up during long stretches without lines**: A 5 m stretch without line-sensor updates with 10× inflated noise parameters will result in covariance that is effectively infinite before the next correction.
3. **Navigation rejection**: If `cov_xx` or `cov_yy` exceed a navigation threshold before the next line-sensor update, the robot may refuse to navigate.

**(c) Parameter you should NOT increase:**

**`k_time_pos_noise`** (and `k_time_rot_noise`) — these model *temporal* drift (position uncertainty growing over time, even when stationary). Wheel slip on tile occurs during *motion*, not during rest. Inflating the time-based terms would cause covariance to grow even when the robot is parked, which is physically wrong and would cause the robot to delocalize itself while sitting still.

**(d) Critique of increasing `k_rot_noise` only:**

The lateral variance formula is:
```
lateral_var = k_trans_lat_noise × delta_trans
            + ...
            + min(cov_theta, 1e5) × delta_trans²
```

Increasing `k_rot_noise` makes `theta_var` grow faster per step, which increases `cov_theta` over time, which then feeds into the `min(cov_theta, 1e5) × delta_trans²` term. So yes, lateral variance *will* increase — but indirectly and with a delay (it takes multiple prediction steps for `cov_theta` to inflate enough to matter).

The problem with this approach:
1. **Wrong physical model**: Lateral slip on tile is not caused by heading uncertainty — it's a direct lateral motion that should be modelled in `k_trans_lat_noise`, not mediated through `cov_theta`.
2. **Coupled side effects**: Inflating `k_rot_noise` also inflates heading variance, which affects line-sensor theta update gating independently of lateral position. You may start rejecting valid heading corrections.
3. **Non-linear coupling**: The cov_theta → lateral effect is quadratic in `delta_trans`. This means the lateral variance only spikes at high speeds; at low speeds the effect is negligible. Direct slip is approximately linear in `delta_trans`.

**Correct approach**: Increase `k_trans_lat_noise` (direct lateral slip per metre) rather than hoping `cov_theta` inflation trickles down to lateral variance.

</details>
