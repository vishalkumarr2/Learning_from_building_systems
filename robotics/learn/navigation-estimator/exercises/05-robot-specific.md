# Exercises: Failure Mode Diagnosis

### Chapter 05: Reading Bags Like an Estimator Detective

**Self-assessment guide:** Work through each question before expanding the answer. Target: 80% correct without peeking → ready for real RCA work. For scenario questions, write down your three answers (status, root cause, next check) before opening.

---

## Section A — Conceptual Questions

**A1.** `ERROR_STATE_INVALID` appears in every AMR incident ticket involving slip. A new engineer says "so the bug is in 's isFinite() check — we should remove it so the robot can keep moving after a slip." Why is this wrong, and what would actually happen if isFinite() were removed?

<details><summary>Answer</summary>

**The isFinite() check is a safety gate, not a bug.** Removing it would be dangerous.

**What would happen without isFinite():**

When SLIPPED is detected, `cov_xx = cov_yy = cov_theta = INF`. The estimator still holds a position estimate `(x, y, theta)` but with infinite uncertainty.

`executePoseGoal()` then computes a velocity command using that state. The PID controller or path-following algorithm would try to compute: "how far am I from the goal?" using `(goal.x - state.x)` and `(goal.y - state.y)`. Those are still finite. So far so good.

But the Kalman gain computation uses `P` (the covariance matrix): `K = P × Hᵀ × (H × P × Hᵀ + R)⁻¹`. With `P = INF`, this produces `K = INF × (INF + R)⁻¹ = INF × (1/INF) = NaN` (not a number, because `INF/INF` is undefined). NaN propagates into the velocity command. The robot may issue a NaN velocity — the motor controller behaviour with a NaN command is undefined and platform-dependent. On some systems it would command full speed, on others zero, on others oscillate.

**The correct mental model:** After a slip, the robot genuinely does not know where it is. Allowing it to navigate with `cov=INF` is not "keep moving" — it is "move blindly with no position knowledge." The safe response is to stop and wait for re-localization via line-sensor.

**The fix is not to remove the check.** The fix is to address the root cause of the slip (floor, motor driver, etc.) and ensure there is a line-sensor line nearby for the robot to cross and recover.

</details>

---

**A2.** An incident bag shows `status=MISSED_LINE_UPDATES (8)` for 8 seconds, then `status=DELOCALIZED (1)` for 3 seconds, then the robot stops. During the entire 11 seconds, the line-sensor topic was publishing at 100 Hz with `is_reliable=True`. Explain the complete sequence of events in the estimator and why two different status codes appear.

<details><summary>Answer</summary>

**MISSED_LINE_UPDATES phase (0–8s):**

The line-sensor was physically detecting lines and publishing measurements, but the EKF was rejecting every measurement via the innovation gate (chi-squared / Mahalanobis distance check). This can happen because:
- The estimator's position estimate diverged slightly and each line-sensor measurement looked like an outlier
- A SPI conflict caused measurements to have corrupted position values, all failing the gate
- All measurements had the same timestamp (SPI burst issue), so only the first was processed

During this phase, the estimator kept running but received no useful line-sensor updates. The prediction step (`predict()`) continued adding motion noise each odom cycle, but no update step (`update()`) ran to reduce the covariance. So covariance grew monotonically.

**Why status=8 and not DELOCALIZED immediately:** The covariance had not yet crossed the DELOCALIZED threshold. Status=8 is reported as soon as line-sensor updates stop being accepted, which can happen with covariance still below the threshold.

**DELOCALIZED phase (8–11s):**

After 8 seconds without line-sensor corrections, the covariance finally crossed the `delocalized_threshold` (typically 0.03–0.1 depending on config). Status transitions to 1 (DELOCALIZED). The estimator continues running and reporting position but with lower confidence.

**The robot stops:**

Covariance has grown further, possibly to INF (if the DELOCALIZED phase is also accompanied by slip, or if this trajectory eventually leads to a executePoseGoal() call where isFinite() fails). Or the navigation planner refused to proceed with DELOCALIZED state.

**The critical takeaway:** Status=8 is a **warning** that DELOCALIZED is coming. If you see status=8 in a live system, investigate immediately — a stop is 8–30 seconds away unless the root cause is fixed.

</details>

---

**A3.** Describe the exact observable differences in a ROS bag between these two scenarios:

(a) Robot slips on a wet patch at T=30s, covariance goes to INF.  
(b) Robot dead-reckons for 20s without line-sensor updates, covariance grows to 0.04 (DELOCALIZED threshold).

Include: what `cov_xx` looks like over time, what `status` looks like, what the velocity signal looks like, and what happens when the robot next crosses a line-sensor line.

<details><summary>Answer</summary>

**(a) Slip scenario:**

| Signal | Before T=30s | At T=30s | After T=30s |
|--------|-------------|---------|-------------|
| `cov_xx` | 0.001–0.010 (normal) | **Steps to 1e5 or INF** in one sample | Stays at INF |
| `status` | 0 (OK) | **4 (SLIPPED)** | Stays at 4 |
| velocity | Normal (e.g. 0.4 m/s) | **Sudden drop to 0** within 50–100ms | 0 (robot stopped) |
| `cmd_vel` | Normally commanded | Goes to 0 when NAV_NOT_FINITE fires | 0 |

After line-sensor crossing: `cov_xx` **drops sharply** from INF back to the measurement quality (≈ line-sensor noise, e.g. 0.0001). Status → 0 (OK). This is an abrupt, large decrease.

**Bag signature:** `cov_xx` time series looks like a step function — flat, then vertical.

---

**(b) Delocalized scenario:**

| Signal | 0–20s | At threshold crossing | After |
|--------|-------|----------------------|-------|
| `cov_xx` | 0.001, slowly rising: 0.003, 0.007, 0.015, 0.024, **0.040** | 0.040 (threshold) | Continues rising |
| `status` | 0 (OK) during rise | **1 (DELOCALIZED)** | Stays at 1 |
| velocity | Normal, commanded motion | Normal (robot NOT stopped) | Robot continues moving |
| `cmd_vel` | Normal | Normal | Normal |

After line-sensor crossing: `cov_xx` **drops from ~0.04 to ~0.001** in one update. Status → 0 (OK). A moderate, expected decrease.

**Bag signature:** `cov_xx` time series looks like a ramp — gradual, consistent slope.

---

**Key distinguishing feature:**

The **slope and magnitude** of the covariance change. A step from 0.001 to 1e5 in one sample is SLIP. A ramp from 0.001 to 0.04 over 20 seconds is DELOCALIZED. You do not need to read the status code — the covariance shape tells you.

</details>

---

**A4.** COLLISION (status=5) is described as "terminal" while SLIPPED (status=4) is "not terminal." Both stop the robot. What is the architectural reason for this difference? (Hint: think about whether autonomous recovery is safe for each.)

<details><summary>Answer</summary>

**The architectural reason is safety under uncertainty:**

**SLIPPED:** The estimator knows *exactly* what went wrong — the velocity window check failed. The robot is stopped in a known position (the last valid estimate before the slip). The floor is the same floor it was on. After stopping, the robot's geometry has not changed, no obstacles were struck, and the mechanical state is the same. The only unknown is the exact position, which can be resolved by crossing a line-sensor line. **Autonomous recovery is safe** because the situation is well-defined.

**COLLISION:** The estimator knows something hit the robot (gyro spike), but it does NOT know:
- What hit it (shelf, person, forklift?)
- Whether the robot moved (was pushed off path?)
- Whether the robot or what it hit sustained damage
- Whether the collision obstacle is still in the way
- Whether the robot's mechanical state is intact (wheels aligned, sensors undamaged?)

Issuing an autonomous "recover by driving forward" command after a collision could:
- Drive into the obstacle again
- Move an injured person further
- Drive off an edge the collision pushed the robot near
- Send a mechanically damaged robot further into a facility

**A human must assess the situation** before the robot moves. The terminal state forces this verification. It is a deliberate safety design choice, not a limitation of the software.

**Analogy:** A car's airbag deployment (COLLISION) requires a mechanic to inspect and reset. Spinning wheels on ice (SLIP) is self-correcting once traction resumes. The severity and unknowns of the situation determine whether autonomous recovery is safe.

</details>

---

**A5.** Two engineers are debugging separate incidents. Engineer A says: "Status=8 means the SPI bus is broken." Engineer B says: "Status=8 means the line-sensor is physically off the lines." Who is right, and what is the precise condition that sets status=8?

<details><summary>Answer</summary>

**Neither is fully right.** Both are describing *causes* that can lead to status=8, but they have incorrectly equated the cause with the mechanism.

**The precise condition for status=8 (MISSED_LINE_UPDATES):**

Status is set to 8 when the estimator detects that a **statistically expected line-sensor update has not been incorporated** for longer than a threshold period. Specifically: the line-sensor topic is alive (messages arriving) but the EKF update step has not accepted any measurement.

This can happen for multiple reasons:
1. **SPI conflict (Engineer A's case):** SPI bus issues cause all measurements to have the same timestamp or corrupted values → all measurements rejected by innovation gate or deduplication logic
2. **Physical off-path (Engineer B's case):** But this would typically set status=3 (NO_LINE), not status=8. Status=8 specifically requires measurements to be *present but rejected*, not absent.
3. **Large estimator position error:** If the estimated position diverged badly, real line-sensor measurements look like outliers and are rejected by the Mahalanobis gate
4. **Dirty lines:** Measurements are so noisy they all fail the gate

**To distinguish the causes:**
- Check for SPI error messages in rosout → Engineer A's cause
- Check if line-sensor detections contain line position data or are empty → Engineer B's cause (empty = NO_LINE, not MISSED_UPDATES)
- Check if measurements have duplicate timestamps → SPI burst issue
- Check estimator position vs map → position divergence issue

The **status code is a mechanism** (updates not accepted). The **cause** requires additional investigation.

</details>

---

## Section B — Scenario Analysis

For each scenario below, you will see a synthetic ROS bag extract (text representation). Identify: **(1) which status code** this represents, **(2) the root cause**, **(3) what to check next**.

---

**B1. Log Extract: The Afternoon Stop**

```
[12:34:15.001] /estimated_state: status=0, x=45.23, y=12.44, cov_xx=0.0045
[12:34:15.051] /estimated_state: status=0, x=45.31, y=12.44, cov_xx=0.0046
[12:34:15.101] /odom: linear.x=0.401, angular.z=0.002
[12:34:15.101] /estimated_state: status=0, x=45.39, y=12.44, cov_xx=0.0047
[12:34:15.151] /odom: linear.x=0.392, angular.z=0.001
[12:34:15.201] /odom: linear.x=0.380, angular.z=0.001
[12:34:15.251] /odom: linear.x=0.028  <══ velocity drop
[12:34:15.261] /estimated_state: status=4, x=45.47, y=12.44, cov_xx=1e6
[12:34:15.263] /rosout [ERROR]: ERROR_STATE_INVALID
[12:34:15.263] /cmd_vel: linear.x=0.0  <══ stop command
[12:34:15.400] /line-sensor: rate=0Hz (no messages)
```

<details><summary>Answer</summary>

**(1) Status:** 4 — SLIPPED

**Evidence:**
- `cov_xx` jumps from 0.0047 to 1e6 in a single sample (not gradual)
- `status` transitions to 4 simultaneously with the covariance jump
- Velocity drops from 0.380 to 0.028 within one odom sample (~50ms) — too fast for normal deceleration (max_decel check fails)
- `ERROR_STATE_INVALID` follows immediately (isFinite(cov_xx=1e6) fails)

**(2) Root cause candidates:**
- **Most likely:** Floor surface change at this XY location (wet patch, tile type change, fresh wax)
- **Secondary:** Motor driver fault causing erroneous velocity reading (false slip detection)
- The line-sensor going silent *after* the stop is a consequence, not a cause

**(3) What to check next:**
- Check robot XY position at T=15.251 against the floor map — is there a known surface change here?
- Check maintenance logs — was floor cleaning scheduled for this area this afternoon?
- Check if other incidents have SLIPPED at the same XY coordinates (location-specific → floor issue)
- If no floor explanation: check motor driver health logs near this timestamp
- After fixing root cause: verify a line-sensor line is accessible for the robot to cross and recover

</details>

---

**B2. Log Extract: The Long Hallway**

```
[09:15:00.000] /estimated_state: status=0, cov_xx=0.0011, cov_yy=0.0012
[09:15:05.000] /estimated_state: status=0, cov_xx=0.0065, cov_yy=0.0063
[09:15:10.000] /estimated_state: status=0, cov_xx=0.0141, cov_yy=0.0138
[09:15:15.000] /estimated_state: status=0, cov_xx=0.0217, cov_yy=0.0213
[09:15:20.000] /estimated_state: status=0, cov_xx=0.0298, cov_yy=0.0291
[09:15:25.000] /estimated_state: status=1, cov_xx=0.0381, cov_yy=0.0373
                ↑ status just changed!
[09:15:25.000] /line-sensor: rate=98Hz, is_reliable=True, detections=[]
[09:15:25.000] /odom: rate=49Hz (normal)
[09:15:35.000] /estimated_state: status=1, cov_xx=0.0521, cov_yy=0.0508
[09:15:37.800] /line-sensor: is_reliable=True, detections=[line_at_offset=-0.012]
[09:15:37.810] /estimated_state: status=0, cov_xx=0.0014, cov_yy=0.0013
```

<details><summary>Answer</summary>

**(1) Status:** 1 — DELOCALIZED

**Evidence:**
- `cov_xx` rises smoothly and gradually over 25 seconds: 0.0011 → 0.0381
- The growth is linear/steady (approximately +0.0012/second), consistent with dead-reckoning noise accumulation
- Status changes to 1 when covariance crosses the DELOCALIZED threshold (~0.038 in this configuration)
- Robot does NOT stop — it continues moving (cmd_vel not zeroed)
- Covariance drops sharply back to 0.0014 when line-sensor crossing occurs at 09:15:37.8

**(2) Root cause:**
- **Primary:** Long path segment with no line-sensor line crossings — 25+ seconds of dead-reckoning
- The line-sensor was *alive and healthy* (`rate=98Hz, is_reliable=True`) but reporting empty detections (`detections=[]`) — this is NO_LINE during the segment, meaning the robot was on a section of floor with no lines, or the lines were not within the line-sensor detection range
- This is a **map or route design issue**: the route includes a floor segment without sufficient line density

**(3) What to check next:**
- Overlay the robot XY trajectory for T=09:15:00 to 09:15:37 onto the floor line map
- Identify the segment with no line crossings — is this a hallway, loading area, or blank floor section?
- Compare with the map: are lines missing from the map, or are they genuinely absent from the floor?
- Check if this route was recently changed and the new segment lacks the line density of the old route
- Fix: add line-sensor lines to the floor segment, or reroute through a section with better line coverage

</details>

---

**B3. Log Extract: The Invisible Shelf**

```
[16:22:04.001] /estimated_state: status=0, cov_xx=0.0031
[16:22:04.021] /imu/data: angular_velocity.z=0.012 rad/s  (normal)
[16:22:04.041] /imu/data: angular_velocity.z=0.008 rad/s  (normal)
[16:22:04.061] /imu/data: angular_velocity.z=2.847 rad/s  <══ SPIKE
[16:22:04.063] /estimated_state: status=5
[16:22:04.063] /cmd_vel: linear.x=0.0  (stop)
[16:22:04.081] /imu/data: angular_velocity.z=0.021 rad/s  (back to normal)
[16:22:04.100] /estimated_state: status=5
[16:22:30.000] /estimated_state: status=5  (still terminal, 26 seconds later)
[16:22:30.000] /rosout [INFO]: Waiting for operator reset
```

<details><summary>Answer</summary>

**(1) Status:** 5 — COLLISION (terminal)

**Evidence:**
- `angular_velocity.z` spikes from ~0.010 rad/s (normal rotation) to 2.847 rad/s — this is approximately 163 degrees/second angular velocity, consistent with a physical impact
- The spike is **impulsive** (one sample at 2.847, then immediately back to 0.021) — not sinusoidal/oscillatory
- `status` transitions to 5 simultaneously with the spike
- Robot stops immediately and **does not recover** — status=5 persists 26 seconds later
- Note: **`ERROR_STATE_INVALID` is NOT logged** — this uses the collision stop path, not the isFinite() path

**(2) Root cause:**
- **Most likely:** Physical collision with an obstacle — shelf leg, person, forklift, barrier
- The impulsive, single-sample nature of the omega_z spike is consistent with a rigid body impact
- A vibration false positive would show oscillatory, repeating pattern; EMI would often show unusual spike shapes or coincide with power events

**(3) What to check next:**
- Check camera footage or obstacle detection logs at T=16:22:04.061 — what was in the robot's path?
- Check the robot's XY position at the collision time against the obstacle map — was an obstacle added or moved without updating the map?
- Physically inspect the robot for damage: is the bumper intact? Any mechanical deformation?
- Check: was this a known blind spot in the robot's obstacle detection (e.g., low shelf legs below the scanner plane)?
- After investigation: requires **human verification of area and robot integrity** before any restart

</details>

---

**B4. Log Extract: The Healthy-Looking Failure**

```
[11:05:10.000] /estimated_state: status=0, cov_xx=0.0018
[11:05:10.000] /line-sensor: rate=102Hz, is_reliable=True, detections=[line_at=-0.001]
[11:05:15.000] /estimated_state: status=0, cov_xx=0.0072
[11:05:15.000] /line-sensor: rate=99Hz, is_reliable=True, detections=[line_at=-0.002]
[11:05:20.000] /estimated_state: status=8, cov_xx=0.0198
[11:05:20.000] /line-sensor: rate=101Hz, is_reliable=True, detections=[line_at=-0.004]
                ↑ status=8 but line-sensor looks healthy!
[11:05:25.000] /estimated_state: status=8, cov_xx=0.0312
[11:05:25.000] /line-sensor: rate=100Hz, is_reliable=True, detections=[line_at=-0.004]
                                                                      ↑ same value as 5s ago?
[11:05:30.000] /rosout [WARN]: SPI transaction timeout on line-sensor_controller
[11:05:30.000] /line-sensor: rate=97Hz, is_reliable=True, detections=[line_at=-0.004]
[11:05:35.000] /estimated_state: status=1, cov_xx=0.0401
```

<details><summary>Answer</summary>

**(1) Status:** 8 — MISSED_LINE_UPDATES (leading to DELOCALIZED)

**Evidence:**
- Line-Sensor topic appears healthy: rate=100Hz, `is_reliable=True`
- But `status` transitions to 8 at T=11:05:20 — the estimator has stopped accepting updates despite the topic being alive
- **Critical observation:** the `detections=[line_at=-0.004]` value is **identical** at T=11:05:20, T=11:05:25, and T=11:05:30 — over 10 seconds, the detected line offset never changed even slightly as the robot moved. This is the SPI duplicate timestamp signature: the SPI controller is sending the same measurement repeatedly with the same timestamp, and the estimator's duplicate detection discards all but the first.
- SPI warning at T=11:05:30 confirms: `SPI transaction timeout on line-sensor_controller`
- Covariance grows slowly (not a slip step-jump) → eventually crosses DELOCALIZED threshold at T=11:05:35

**(2) Root cause:**
- **SPI bus conflict or line-sensor controller fault** causing the line-sensor to repeat the same measurement with the same timestamp
- The estimator correctly identifies that `timestamp(t) == timestamp(t-1)` and discards the duplicate, preventing false updates from being incorporated. But this means NO updates are accepted, leading to status=8.
- This matches the documented dual-SPI conflict failure mode in this codebase (see repo memory: `dual-spi-conflict-reproduction.md`)

**(3) What to check next:**
- Check SPI error logs on the robot — is there a conflicting SPI device on the same bus as the line-sensor?
- Verify line-sensor measurement timestamps in the topic: `rostopic echo /line-sensor/detections | grep stamp` — are multiple consecutive messages reporting the same header.stamp?
- Check if the issue is intermittent or consistent (if SPI conflict, it may correlate with another SPI peripheral being active)
- Review `dual-spi-conflict-reproduction.md` in repo memory for the known reproduction steps
- Fix: resolve SPI bus conflict (device arbitration or physical bus separation); do not simply increase covariance thresholds

</details>

---

## Section C — "Connect 3" Challenge

**The scenario:**

You are given a bag from an incident at Site-A. The following time series was extracted:

```
    Time     cov_xx    status    line-sensor_rate    notes
    ──────   ──────    ──────    ──────────────    ─────────────────────────
    T=10.3s  0.0012    0 (OK)    100 Hz
    T=10.4s  0.0012    0 (OK)    0 Hz              ← LINE-SENSOR GAP (1 sample)
    T=10.5s  0.0013    0 (OK)    100 Hz            ← line-sensor resumes
    T=11.0s  0.0045    0 (OK)    100 Hz
    T=11.5s  0.0082    0 (OK)    100 Hz
    T=12.0s  0.0145    0 (OK)    100 Hz
    T=12.5s  0.0209    0 (OK)    100 Hz
    T=12.6s  0.0234    1 (DELOCALIZED) 100 Hz      ← threshold crossed
    T=12.8s  0.0004    0 (OK)    100 Hz            ← line-sensor crossing, cov reset
```

Note: line-sensor was not publishing for exactly 0.1 seconds (one sample) at T=10.4s.

**Question C1:** What happened at T=10.4s? Give a precise mechanistic explanation of why `cov_xx` is still 0.0012 at T=10.5s even though a line-sensor gap occurred.

<details><summary>Answer</summary>

**At T=10.4s:** The line-sensor topic published no message for one sample period (0.1s gap). This could be caused by:
- A brief SPI timeout that missed one transmission cycle
- A packet drop on the internal ROS network
- A single-cycle CPU scheduling delay causing the line-sensor publisher to miss its deadline

**Why cov_xx stays at 0.0012 at T=10.5s:**

The covariance contribution from a missing line-sensor update is **very small** compared to the contribution from motion. During 0.1 seconds without a line-sensor update, the prediction step ran (with odom still arriving at 50Hz) and added motion noise. But the motion noise per prediction step is proportional to `delta_trans × k_trans_noise`. For a robot moving at ~0.3 m/s:

```
    delta_trans per step (50Hz) ≈ 0.006 m
    k_trans_noise    ≈ 0.01 (typical)
    noise added per step        ≈ 0.006 × 0.01 = 0.00006

    Over 5 prediction steps in 0.1s: ≈ 0.0003
    Starting cov_xx = 0.0012
    Expected cov_xx at T=10.5 ≈ 0.0015
```

This is below the display precision in the data extract (rounds to 0.0012 or would show as 0.0013). The line-sensor also resumed at T=10.5s and provided a correction that pulled `cov_xx` back down, masking the 0.1s gap entirely. A single 0.1s line-sensor gap is **below the detection threshold** for this measurement — it is absorbed into normal covariance fluctuation.

**The key insight:** The covariance increase from a 0.1s line-sensor gap is negligible. The covariance increase visible from T=10.5s to T=12.6s (roughly +0.002 per 0.5s increment) represents **2.1 seconds of sustained dead-reckoning after T=10.5s** — meaning line-sensor updates resumed but the robot was NOT crossing any line-sensor lines during this period (just the topic publishing at 100Hz with no line crossings under the sensor, i.e., traversing blank floor).

</details>

---

**Question C2:** The line-sensor rate returned to 100Hz at T=10.5s. Yet status did NOT transition to `MISSED_LINE_UPDATES (8)`. Why not? What would need to be true for status=8 to appear here?

<details><summary>Answer</summary>

**Why status=8 did NOT appear:**

Status=8 (MISSED_LINE_UPDATES) requires that the estimator is **expecting** line-sensor updates and not receiving them. Specifically, it tracks whether any updates have been *accepted* by the EKF over a time window.

At T=10.5s, the line-sensor resumed publishing at 100Hz. However, the topic was publishing — it just had no **line detections** in the messages (the `detections=[]` list was empty). This is **NO_LINE behaviour, not MISSED_LINE_UPDATES**. The robot was traversing floor with no lines underneath the line-sensor, so the line-sensor correctly reported "I see nothing." An empty detection is not a "missed update" — it is an accurate report.

For status=8 to appear, the line-sensor would need to be **reporting line detections** (`detections=[line_at=xxx]`) but those detections would need to be **rejected by the EKF** (failing the innovation gate). In this scenario, the detections list appears to be empty during the T=10.5–12.6s window (inferred from the fact that covariance grows steadily without any corrections).

**Formally:** `MISSED_LINE_UPDATES = detections_available AND update_not_accepted`. In this scenario, `detections_available = False` (empty detections), so the condition for status=8 is not met.

**What would need to be true for status=8 here:**
- The floor between T=10.5s and T=12.6s would need to have line-sensor lines painted on it
- The line-sensor would need to be detecting those lines (non-empty detections)
- But the EKF would need to be rejecting those measurements (innovation too large, or duplicate timestamps from SPI issue)
- This combination → status=8

</details>

---

**Question C3:** If the line-sensor gap at T=10.4s had been **20 seconds** instead of 0.1 seconds (line-sensor offline from T=10.4s to T=30.4s), and the robot continued moving at the same speed, what would the failure sequence look like? Estimate:
- When would DELOCALIZED first appear?
- Would ERROR_STATE_INVALID fire? Under what condition?
- What is the earliest time at which the robot could recover?

<details><summary>Answer</summary>

**Estimating covariance growth during 20s line-sensor gap:**

From the observed data, covariance grew from 0.0013 to 0.0234 over approximately 2.1 seconds of dead-reckoning (T=10.5 to T=12.6s), giving a growth rate of approximately:

```
    Δcov_xx / Δt ≈ (0.0234 - 0.0013) / 2.1s ≈ 0.0105 per second
```

Starting from T=10.4s with `cov_xx = 0.0012`:

**When would DELOCALIZED first appear?**
```
    DELOCALIZED threshold ≈ 0.024 (from the data: status changes at cov_xx=0.0234)

    Time to threshold = (0.0234 - 0.0012) / 0.0105 ≈ 2.1 seconds

    DELOCALIZED would appear at approximately T=10.4 + 2.1 = T=12.5s
    (much earlier than in the original scenario because there are no line-sensor
    corrections to slow the growth at all during the 20s gap)
```

**Would ERROR_STATE_INVALID fire?**

This depends on whether covariance reaches INF or whether SLIPPED is triggered separately. If the robot only dead-reckons (no slip event), `cov_xx` grows continuously but through the Kalman noise model, not through the explicit `cov = INF` assignment. The Kalman filter produces **large but finite** covariance values during extended dead-reckoning.

ERROR_STATE_INVALID fires on **non-finite** (INF or NaN) values. Pure dead-reckoning covariance growth stays finite (just large). So NAV_NOT_FINITE would NOT fire from dead-reckoning alone, assuming no slip event.

**However:** If the robot then encounters a wet patch or motor issue during the 20s gap and the slip detector fires, `cov_xx` would be explicitly set to INF → NAV_NOT_FINITE would fire.

**Earliest possible recovery time:**

The robot can recover as soon as it crosses a line-sensor line AND the EKF accepts the measurement. With `cov_xx` now very large (but finite), the Kalman gain `K` would be very large, meaning the line-sensor measurement would completely dominate (almost ignore the prior estimate). The recovery would happen in **one sample** of the line-sensor.

```
    Earliest recovery = T=30.4s + (time to first line-sensor line crossing)
```

If a line-sensor line was at the same location as in the original scenario (T=12.8s corresponds to approximately 2.4s of travel distance at 0.3 m/s = ~0.72m past the gap start), the robot would cross that line at T=30.4s + 2.4s (if it keeps moving) = approximately T=32.8s.

**But:** With DELOCALIZED status starting at T≈12.5s, the navigation system may reduce speed or stop the robot before the line crossing. The recovery time depends on the planner's behaviour under DELOCALIZED status.

**Summary:**
```
    Original scenario (0.1s gap):  DELOCALIZED at T=12.6s, recovery at T=12.8s
    20s gap scenario:               DELOCALIZED at T≈12.5s, robot stopped or
                                    slowed, recovery only possible at T≥32.8s
                                    (18+ seconds of degraded or no operation)
```

This illustrates why even brief line-sensor gaps matter when they happen at the start of a long blank-floor segment — the covariance trajectory is the same, but the time to recovery increases dramatically.

</details>
