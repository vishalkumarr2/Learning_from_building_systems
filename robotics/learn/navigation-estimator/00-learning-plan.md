# Navigation & State Estimation — Learning Plan
### For: Software engineer investigating warehouse AMR navigation failures daily
### Goal: Read estimator covariance trajectories from first principles, diagnose EKF failures from logs/bags

---

## Why This Track Exists

You already know the the navigation estimator code deeply:
- `predict()` grows covariance from a fixed noise model (delta_trans, delta_theta, delta_t)
- `update()` shrinks covariance using line-sensor line measurements with Mahalanobis gating
- Slip detection sets covariance to INF via velocity window comparison
- `isFinite()` check in `executePoseGoal()` triggers `ESTIMATED_STATE_NOT_FINITE`

**What you're missing:** The mathematical theory that lets you look at a covariance trace in a bag
and immediately say "this blow-up is a slip, not a delocalization" or "this line-sensor update
was rejected because innovation exceeded 2-sigma." This track fills that gap.

---

## Dependency Graph

```
                              ┌────────────────────────┐
                              │ 05-failure-modes.md    │
                              │ (AMR diagnosis)        │
                              └────────────┬───────────┘
                                           │ requires all below
              ┌────────────────────────────┼────────────────────────────┐
              │                            │                            │
   ┌──────────▼──────────┐   ┌────────────▼────────────┐  ┌────────────▼──────────┐
   │ 03-measurement-     │   │ 04-imu-fusion.md        │  │ 02-kalman-filter.md   │
   │ models.md           │   │ (gyro, bias, line-sensor  │  │ (EKF predict/update)  │
   │ (line constraints,  │   │  theta update)          │  │                       │
   │  Mahalanobis)       │   └────────────┬────────────┘  └────────────┬──────────┘
   └──────────┬──────────┘                │                             │
              │                           └──────────┬──────────────────┘
              └──────────────────────────────────────┤
                                                     ▼
                                        ┌────────────────────────┐
                                        │ 01-dead-reckoning.md   │
                                        │ (odometry, unicycle,   │
                                        │  arc integration)      │
                                        └────────────────────────┘
```

**Reading order:** 01 → 02 → 03 → 04 → 05 (strictly sequential — each builds on the previous)

---

## Topic Breakdown

| # | File | Est. Time | What It Gives You |
|---|------|-----------|-------------------|
| 01 | `01-dead-reckoning.md` | 3 hrs | Derive `(x', y', θ')` from encoder counts; understand AMR `predict()` noise terms |
| 02 | `02-kalman-filter.md` | 4 hrs | KF → EKF equations; understand why covariance grows during motion and shrinks on measurement |
| 03 | `03-measurement-models.md` | 3 hrs | Line sensor as a constraint; Mahalanobis distance; when AMR rejects a line-sensor update |
| 04 | `04-imu-fusion.md` | 3 hrs | Gyro integration; bias estimation; how AMR theta update differs from XY update |
| 05 | `05-failure-modes.md` | 3 hrs | Recognise slip vs delocalize vs collision from covariance patterns in bags |

**Total: ~16 hours**

---

## Checkpoint Questions — "You're Done When You Can Answer..."

After each file, you must answer these without opening any notes:

### After `01-dead-reckoning.md`
- [ ] A robot moves 0.1m forward. Left wheel encoder reads 100 ticks, right reads 98. Wheel radius 0.05m, ticks/rev = 1000, base.3m. Compute Δx, Δy, Δθ.
- [ ] Why does arc integration accumulate heading error faster than position error?
- [ ] In `predict()`, which noise term (`k_trans_noise` vs `k_rot_pos_noise`) dominates during a tight turn?

### After `02-kalman-filter.md`
- [ ] Write the EKF prediction equations. What are `F`, `Q`, `P`?
- [ ] After 10 steps with no measurement, the covariance is large. One line-sensor update arrives. What happens to `P`?
- [ ] AMR `predict()` does NOT use the odom message's covariance fields. Why is this OK?

### After `03-measurement-models.md`
- [ ] Line-Sensor reads a line at 5cm left of expected. The covariance is large. Is the update accepted or rejected?
- [ ] What is Mahalanobis distance and why is it used instead of raw innovation?
- [ ] AMR uses `clamp(state, meas.min, meas.max)` instead of a standard Kalman gain. What does this mean geometrically?

### After `04-imu-fusion.md`
- [ ] What is gyro bias and why does it matter for heading estimation?
- [ ] AMR only uses IMU for theta, not XY. Why?
- [ ] If the IMU is missing, what happens to the robot theta update? (hint: check line-sensor code path)

### After `05-failure-modes.md`
- [ ] Given a bag where covariance(X,X) = INF appears at T+2s after a velocity spike, what is the most likely cause?
- [ ] The robot delocalized but the slip detection never fired. What else can cause `ERROR_STATE_INVALID`?
- [ ] How would you use `scripts/analysis/` to plot covariance over time and identify the divergence point?

---

## Weekly Schedule

### Week 1: Dead-Reckoning + Odometry Math (3 hrs)
**Read:** `01-dead-reckoning.md`
**Do:** `exercises/01-odometry-math.md`
**Connect:** Open `` `predict()` and label each noise term against the math you learned.

---

### Week 2: EKF Theory — Prediction Step (4 hrs)
**Read:** `02-kalman-filter.md` — KF derivation, then EKF linearisation
**Do:** `exercises/02-kalman-1d.md` — build a 1D KF by hand (position + velocity)
**Connect:** The AMR prediction step is a simplified EKF (no Jacobian for covariance — uses fixed
noise model instead). Understand why this is an approximation and when it breaks.

---

### Week 3: Measurement Update + Mahalanobis (3 hrs)
**Read:** `03-measurement-models.md`
**Do:** `exercises/03-ekf-unicycle.md` — implement EKF for a 2D unicycle in Python (30 lines)
**Connect:** Read `` `update()` . You can now read every line.

---

### Week 4: IMU Fusion + Theta (3 hrs)
**Read:** `04-imu-fusion.md`
**Do:** `exercises/04-log-diagnosis.md` — given a synthetic covariance trace, identify the failure
**Connect:** Look at `imuCallback()` callback. The gyro integration + bias correction will now make sense.

---

### Week 5: AMR Failure Modes + Bag Diagnosis (3 hrs)
**Read:** `05-failure-modes.md`
**Do:** `exercises/05-robot-specific.md` — use a real AMR bag (from `attachments/`) to find the
first moment covariance blows up, and name the cause.
**Goal:** At the end of this week, every `ERROR_STATE_INVALID` ticket you open should
have a 3-sentence hypothesis within 2 minutes of looking at the bag.

---

## AMR-Specific Reference

These code locations are your ground truth — the math in these notes should map 1:1:

| Concept | AMR Code Location |
|---------|------------------|
| Odometry noise model | `` `predict()` lines 396+ |
| Covariance set to INF (slip) | `` `odometryCallback()`  |
| Covariance set to INF (collision) | `` `imuCallback()` lines ~200–250 |
| Measurement update (line-sensor) | `` `update()`  |
| Mahalanobis gate (2-sigma) | `` `update()` — innovation check |
| isFinite check | `` `executePoseGoal()`  |
| Theta KF update | `` — Kalman gain for IMU theta |
| VelocityWindowTracker | `` |

---

## How This Connects to Other Tracks

```
electronics/05-spi-deep-dive.md     ─→  line-sensor data arrives over SPI (v3.1+)
zephyr/study-notes/02-sensors.md    ─→  how IMU data is read from ICM-42688
navigation-estimator/ (this track)  ─→  what the estimator does with that data
ros2-handson/ (next track)          ─→  how nav2 uses the estimated state for planning
```
