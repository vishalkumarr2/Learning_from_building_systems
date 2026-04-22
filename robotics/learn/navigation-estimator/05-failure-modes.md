# 05 — Failure Mode Diagnosis
### Reading bags like an estimator detective

**Prerequisite:** `01-dead-reckoning.md`, `02-kalman-filter.md`, `03-measurement-models.md`, `04-line-sensor-fusion.md` — you need to understand covariance, what makes it grow, and how line-sensor resets it before you can diagnose why it went wrong.
**Unlocks:** The end-to-end RCA workflow. Once you can diagnose any failure mode from a bag alone, you are ready to write post-mortems without guessing.

---

## Why Should I Care? (AMR Project Context)

Every day, robots at Site-A, Site-B, and other sites produce incident bags. They all share a common anatomy: the robot was fine, then something changed, then the estimator status transitioned to something non-OK, then the robot stopped. The `ERROR_STATE_INVALID` error message appears in the ticket and someone has to find out why.

Without this material, the investigation stalls at the symptom. You see `status=SLIPPED` and you write "the robot slipped" — but *why* did it slip? Wet floor? Motor driver fault? Bad odom message? Each of those has a completely different fix.

This note gives you the **causal chain vocabulary** to go from the error code all the way back to the physical event.

Specifically, after this note you can answer:
- **"a bag incident shows ERROR_STATE_INVALID at 03:29. What actually happened?"** → Look for the cov_xx jump 50-100 ms before the stop. If it's sudden → SLIP. If it's gradual → DELOCALIZED. The error message is the same either way.
- **"The robot stopped but there was no collision. Why is status=5?"** → EMI on IMU or power supply glitch causes a false gyro spike. Check the omega_z time series.
- **"Line-Sensor topic is healthy but status=MISSED_LINE_UPDATES. How is that possible?"** → Every measurement is failing the 2-sigma gate. Or every sample has the same timestamp. Topic alive ≠ updates accepted.
- **"Is status=DELOCALIZED dangerous?"** → No, the robot continues. Only COLLISION is truly terminal. Everything else either recovers or waits.

The the navigation estimator is deliberately transparent about its failures. The status codes are a gift. Most RCA failures are not "the code is wrong" — they are "the sensor gave a bad signal" or "the floor is different from what was assumed."

---

# PART 1 — THE DIAGNOSTIC MINDSET

## 1.1 Two Questions, Every Time

When you open an incident bag, you are trying to answer exactly two questions:

```
  QUESTION 1: What was the estimator state at the moment of failure?
              (Which status code? What were cov_xx, cov_yy, cov_theta?)

  QUESTION 2: What caused that state to become invalid?
              (Sensor fault? Environment? Software? Configuration?)
```

Every other thing you observe — velocity profiles, line-sensor rates, log messages — is evidence for answering Question 2. The status code answers Question 1 automatically.

> **Key insight:** The status code is an **effect**, not a cause. Writing "root cause: SLIPPED" in a post-mortem is like writing "root cause: the patient's blood pressure dropped." It describes the mechanism, not the trigger. The cause is the wet floor, or the motor driver fault, or the floor wax on Wednesday.

## 1.2 The Causal Chain

Every failure follows the same three-stage structure:

```
  PHYSICAL WORLD          ESTIMATOR INTERNALS        ROBOT BEHAVIOUR
  ══════════════          ═══════════════════        ════════════════

  Something               Estimator state            Behaviour
  physical happens   ──►  changes (status,      ──►  changes (stop,
                          covariance, etc.)          slow, reject cmd)

  EXAMPLES:
  ┌────────────────────┬────────────────────────┬──────────────────┐
  │ Wet floor patch    │ slip detected,         │ NAV_NOT_FINITE   │
  │                    │ cov → INF, status=4    │ stop             │
  ├────────────────────┼────────────────────────┼──────────────────┤
  │ Robot hits shelf   │ gyro spike,            │ COLLISION stop   │
  │ leg               │ status=5 (terminal)    │ requires restart │
  ├────────────────────┼────────────────────────┼──────────────────┤
  │ Long hallway,      │ cov slowly rises,      │ reduced          │
  │ no line crossings  │ threshold exceeded,    │ capability       │
  │                    │ status=1               │                  │
  ├────────────────────┼────────────────────────┼──────────────────┤
  │ SPI bus glitch     │ measurements rejected, │ status=8         │
  │ (SPI conflict)     │ no updates despite     │ cov growing      │
  │                    │ topic alive            │                  │
  └────────────────────┴────────────────────────┴──────────────────┘
```

You work **backwards**: from robot behaviour → estimator state → physical event.

## 1.3 The Diagnostic Template

Apply this for every failure bag. It takes about 90 seconds once you know what to look for.

```
  STEP 1: Find the earliest timestamp where status ≠ OK (0)
          ─────────────────────────────────────────────────
          rostopic echo /estimated_state | grep -A2 "status: [^0]"
          Or plot status field vs time in rqt_plot.

  STEP 2: Look 0.5 – 2 seconds BEFORE that timestamp
          ─────────────────────────────────────────────
          This is the "causal window". The trigger happened here.
          Most physical events precede the status change by 50–500ms.

  STEP 3: Check four channels in the causal window
          ─────────────────────────────────────────
          a) Velocity:     did cmd_vel or odom/twist/linear.x spike or drop?
          b) Covariance:   did cov_xx / cov_yy jump suddenly or rise slowly?
          c) Line-Sensor:    did the topic rate drop? did is_reliable flip?
          d) Odom rate:    are messages still arriving at expected frequency?

  STEP 4: Identify the triggering event
          ──────────────────────────────
          Sudden velocity drop   → SLIP candidate
          Gyro omega_z spike     → COLLISION candidate
          Covariance slow rise   → DELOCALIZED / dead-reckoning too long
          Line-Sensor rate drops   → NO_LINE or MISSED_LINE_UPDATES
          Odom messages stop     → MISSING_ODOM

  STEP 5: Find the ROOT CAUSE (sensor, environment, software)
          ──────────────────────────────────────────────────
          SLIP + wet floor patch → floor maintenance issue
          SLIP + no floor change → motor driver fault
          COLLISION + no obstacle → EMI or power supply glitch
          DELOCALIZED + map error → map doesn't show all lines
          MISSED_LINE_UPDATES + SPI error logs → SPI bus conflict
```

---

# PART 2 — SLIP (STATUS = 4)

## 2.1 What Slip Means Physically

A wheel slip event occurs when the driven wheel loses traction — the motor turns the wheel but the wheel is not gripping the floor. The encoder faithfully counts the wheel rotation, but the robot has not actually moved as far as the encoder claims.

```
    NORMAL TRACTION (good floor):
    ─────────────────────────────

    Motor input ──► Wheel rotation ──► Traction force ──► Forward motion
                         │                                      │
                   encoder counts N                    robot moves N×r×2π/ticks

    SLIP EVENT (wet/slippery floor):
    ─────────────────────────────────

    Motor input ──► Wheel rotation ──► [NO TRACTION]
                         │                    │
                   encoder still          wheel spins
                   counts N               but robot
                         │                stays still
                    velocity = N×r×2π/(ticks×dt)      (computed, wrong)
                    actual robot velocity ≈ 0          (real, never measured)
```

The estimator's only view of motion is through the encoder. It believes the robot moved. It updates the position estimate accordingly. But the robot didn't actually move. Position estimate diverges from reality.

## 2.2 The Trigger: Velocity Window Check

The the slip detector in `odometryCallback()` ( ) uses a **rolling velocity window**. Conceptually:

```
    VELOCITY WINDOW LOGIC
    ══════════════════════

    Maintain a circular buffer of recent velocity readings:
    [v(t-4), v(t-3), v(t-2), v(t-1), v(t)]

    At each new odom sample:
        v_current = odom.twist.linear.x
        v_window_avg = mean(circular_buffer)

        IF v_window_avg > min_speed_threshold:
            decel = (v_window_avg - v_current) / dt

            IF decel > max_decel_threshold:
                *** SLIP DETECTED ***
                cov_xx = cov_yy = cov_theta = INF
                status = SLIPPED (4)

    Intuition: a real robot cannot decelerate from 0.5 m/s to 0 m/s
    in one odom sample (dt ≈ 20ms). If velocity drops that fast,
    the encoder reading is wrong (slip) or the odom message is corrupted.
```

**The key check:** can the velocity have dropped this much in this time given the robot's physical deceleration limits? If not, something is wrong with the motion model.

## 2.3 Covariance Signature

When slip is detected, the estimator immediately sets covariance to effectively infinite for all three position components:

```
    cov_xx   → INF  (x position completely unknown)
    cov_yy   → INF  (y position completely unknown)
    cov_theta → INF  (heading completely unknown)
```

This is correct: after a slip event, the estimator has no idea where the robot is. The dead-reckoning from the slip point forward is entirely invalid. The only way to recover is to cross a line-sensor line and get an absolute measurement.

## 2.4 Bag Signature: What You See

```
    TIME (s)    VELOCITY (m/s)    cov_xx    status
    ────────    ──────────────    ──────    ──────
    10.00       0.45              0.0012    0 (OK)
    10.02       0.44              0.0013    0 (OK)
    10.04       0.43              0.0013    0 (OK)
    10.06       0.50  ← SPIKE     0.0014    0 (OK)   ← something happened
    10.08       0.02  ← DROP      1e5+      4         ← slip detected
    10.10       0.00              INF       4
    10.12       0.00              INF       4
          ...robot stopped via isFinite() check...
    10.14       ERROR_STATE_INVALID error logged

    WHAT TO LOOK FOR:
    • Velocity spike THEN sudden drop within 50-100ms
    • cov_xx jumps from small value (0.001-0.01) to 1e5 or INF in ONE sample
    • status changes to 4 simultaneously with covariance jump
    • Status=4 appears AFTER the velocity event, not before
```

## 2.5 Recovery Path

SLIPPED is **not terminal**. The robot recovers automatically when:

1. Robot crosses a line-sensor line with good measurement quality
2. The line measurement provides an absolute position fix
3. Covariance is reset to the post-measurement value
4. Status returns to OK

This is why post-slip behaviour often looks like: robot stops, waits for operator to move it to a line crossing, robot resumes. Or in autonomous recovery: planner sends robot toward a known line location.

## 2.6 Common Causes and False Positives

```
    ROOT CAUSE           INDICATOR IN BAG                FIX
    ─────────────────    ────────────────────────────    ─────────────────
    Wet floor patch      Floor material change in map,   Floor maintenance,
                         only happens at one location    anti-slip mats

    Floor wax (Wednesday cleaning is a recurring cause)

    Ramp transition      Always at ramp start/end        Reduce entry speed
                         coordinates in map

    Carpet edge          Location-specific, height       Add line at edge,
                         change visible in odom          height sensor check

    Motor driver fault   ← FALSE POSITIVE →              
                         Slip at random locations,       Replace motor driver
                         velocity spike without any
                         floor surface change

    Corrupt odom msg     ← FALSE POSITIVE →
                         Single huge velocity value      Check odom publisher,
                         in one sample, then normal      network health
```

> **Key insight:** True slip is **location-specific**. If SLIPPED events happen at the same XY coordinates every time, the floor is the cause. If they happen at random locations, suspect the motor driver or odom publisher.

---

# PART 3 — COLLISION (STATUS = 5)

## 3.1 What It Means: IMU Gyro Spike

The collision detector runs in `imuCallback()` in . It monitors the IMU's angular velocity (omega_z, the yaw rate). A physical collision causes a very specific signature:

```
    COLLISION ANGULAR VELOCITY PROFILE
    ════════════════════════════════════

    omega_z (rad/s)
         │
     3.0 │        ██
     2.5 │       ████
     2.0 │      ██████
     1.5 │     ████████
     1.0 │    ██████████
     0.5 │   ████████████
     0.0 ├───────────────────────────────────► time (ms)
    -0.5 │
         │      ^        ^
         │      |        |
         │   onset    duration
         │   (<5ms)  (50-100ms typical)

    Characteristics of a REAL collision:
    • Very fast onset: < 5ms rise time
    • Short duration: impact force, then dissipates
    • Large amplitude: typically > 2-3 rad/s for hard collision
    • Followed by zero (object stops robot, not oscillation)

    Characteristics of a FALSE POSITIVE (vibration/EMI):
    • Sinusoidal pattern: oscillates rather than impulsive
    • Repeatable frequency from nearby machinery
    • Same amplitude every time
```

## 3.2 Terminal State

COLLISION is the only **truly terminal** status. Unlike SLIPPED (which recovers on next line crossing) or DELOCALIZED (which recovers when covariance drops), COLLISION requires a human intervention or a supervised restart:

```
    STATUS RECOVERY MATRIX
    ══════════════════════

    Status       Terminal?   Recovery mechanism
    ────────     ─────────   ──────────────────
    1 DELOCAL.   No          Covariance decreases via line-sensor updates
    2 UNINIT.    No          Initial pose received from localization
    3 NO_LINE    No          Robot moves onto a line
    4 SLIPPED    No          Crosses line-sensor line with good measurement
    5 COLLISION  YES ★       Operator restart / supervised reset only
    6 INIT.ING   No          Initialization completes
    7 MISS.ODOM  No          Odom messages resume
    8 MISS.UPDS  No          Valid line-sensor measurements resume
```

The reason COLLISION is terminal: the estimator cannot know the robot's state after an impact. The robot may have been knocked off its path. The floor may have changed. An autonomous recovery attempt without human verification would be dangerous.

## 3.3 Bag Signature

```
    TIME (s)    omega_z (r/s)    status    cmd_vel
    ────────    ─────────────    ──────    ───────
    15.00       0.02             0 (OK)    0.3 m/s forward
    15.01       0.03             0 (OK)    0.3 m/s forward
    15.02       2.87 ← SPIKE     5         0.0  ← stop
    15.03       0.12             5         0.0
    15.04       0.01             5         0.0
          ...status stays at 5 forever...
          ...robot does NOT auto-recover...

    WHAT TO LOOK FOR:
    • omega_z spike > 2-3 rad/s, duration < 100ms
    • status = 5 appears simultaneously with or within 1 sample of spike
    • cmd_vel goes to zero and stays zero
    • No covariance spike (collision doesn't set cov to INF)
    • ERROR_STATE_INVALID does NOT fire (different stop path)
```

## 3.4 Collision vs ERROR_STATE_INVALID

This is a common confusion point:

```
    COLLISION STOP PATH:
    ─────────────────────

    imuCallback() detects gyro spike
           │
           ▼
    status = COLLISION (5)
           │
           ▼
    Collision stop command issued directly
    (separate code path, does not go through executePoseGoal())
           │
           ▼
    Robot stops

    NAV_NOT_FINITE STOP PATH:
    ──────────────────────────

    odometryCallback() detects slip
           │
           ▼
    cov → INF, status = SLIPPED (4)
           │
           ▼
    executePoseGoal() called for next motion command
           │
           ▼
    isFinite() check fails (cov_xx = INF)
           │
           ▼
    ERROR_STATE_INVALID logged
           │
           ▼
    Robot stops safely

    KEY DIFFERENCE:
    • COLLISION: stops via its own emergency path
    • SLIPPED: stops only when executePoseGoal() is called and checks state
    • You can tell them apart by which log message appears
```

## 3.5 False Positives

```
    CAUSE                   INDICATOR                       FIX
    ─────────────────────   ──────────────────────────      ───────────────
    Vibration from          omega_z oscillates at fixed     Increase collision
    nearby machinery        frequency, happens repeatedly   threshold, add
                            even with no actual impact      vibration isolation

    EMI on IMU              Random spikes, no correlation   Shield IMU cables,
                            with robot motion or impacts    check grounding

    Power supply glitch     omega_z spike coincides with    Check power rail,
                            voltage transient in log        bypass capacitors

    Hard floor bump          Does correlate with location   Map the bump,
    (door threshold, etc.)  — not truly a false positive,  slow robot at that
                            robot did experience jolt        location
```

---

# PART 4 — DELOCALIZED (STATUS = 1)

## 4.1 What It Means: Covariance Threshold Crossed

DELOCALIZED occurs when the estimator's position uncertainty grows large enough that it no longer trusts the estimate sufficiently for precise navigation:

```
    DELOCALIZATION GEOMETRY
    ════════════════════════

    Robot position uncertainty is an ellipse in 2D:

    Small covariance (OK):          Large covariance (DELOCALIZED):
    ─────────────────────           ──────────────────────────────

           ┌──┐                     ┌────────────────────┐
           │ ● │  ← robot            │                    │
           └──┘     95% confidence   │         ●          │  95% confidence
                    ellipse tiny     │                    │  ellipse huge
                                     └────────────────────┘

    The robot doesn't know if it's here... or here... or over there.
    Safe navigation requires knowing position to ±few centimetres.
    When uncertainty > threshold, status = 1.
```

## 4.2 Trigger: Slow Covariance Growth

Unlike SLIPPED (sudden), DELOCALIZED is **gradual**. Covariance grows continuously during dead-reckoning (between line-sensor line crossings) because:

```
    PREDICTION STEP (each odom sample):
    ─────────────────────────────────────

    cov(t+1) = F × cov(t) × Fᵀ + Q_motion

                                   ↑
                    This term adds noise every step.
                    It never subtracts.
                    Without line-sensor corrections, cov grows monotonically.

    TIME BETWEEN LINE CROSSINGS vs COVARIANCE:
    ═══════════════════════════════════════════

    cov_xx
      │
  0.05│                                         × ← DELOCALIZED threshold
      │                                    ×
  0.04│                               ×
      │                          ×
  0.03│                     ×
      │               ×
  0.02│         ×
      │    ×
  0.01│×
      ├─────────────────────────────────────────► time (s)
      0    2    4    6    8   10   12   14   16

    Each × = one odom sample adding motion noise.
    Line-Sensor crossing would RESET cov back to small value.
    Long gap between line crossings → monotonic growth → DELOCALIZED.
```

## 4.3 Not Terminal: Distinguishing from SLIPPED

```
    DELOCALIZED vs SLIPPED — KEY DIFFERENCES
    ══════════════════════════════════════════

    Feature          DELOCALIZED (1)          SLIPPED (4)
    ───────────      ────────────────────     ───────────────────
    Covariance       Slowly increases         Jumps to INF in ONE sample
    growth pattern   over seconds             (50ms transition)

    Duration before  Many seconds (5-30s      Microseconds: velocity
    status change    typical)                 event → immediate status change

    Recovery         Automatic: any line-sensor Same: needs line-sensor
                     crossing fixes it        crossing

    Operation        Robot continues, reduced Robot stops via isFinite()
    during state     capability               (cov=INF → not finite)

    Bag signature    cov_xx smoothly rising,  cov_xx flat then VERTICAL
                     threshold crossing       step up to 1e5+

    WARNING log      Yes, gradual warnings    Hard stop, error log

    cov_xx value     Moderate (0.01–0.1       Huge (1e5 or INF)
    at transition    range typically)
```

> **Key insight:** If you see `cov_xx = 1e5` and the robot stopped, it was SLIPPED. If you see `cov_xx = 0.05` and the robot is still moving but with warnings, it is DELOCALIZED. The scale is the giveaway.

## 4.4 Common Causes

```
    CAUSE                        HOW TO IDENTIFY
    ────────────────────         ─────────────────────────────────────
    Long straight path with      Plot robot XY trajectory vs line map.
    no line crossings            Gap between line crossings > ~10m?
                                 Did line get removed from map? Was it
                                 never in the map?

    Line-Sensor offline            is_reliable = False during the period.
    (temporary failure)          Check line-sensor topic rate.

    Lines not reflective         Line-Sensor detects nothing despite lines
    enough (dirty floor)         being present. is_reliable=True but
                                 no measurements arrive. → leads to
                                 MISSED_LINE_UPDATES first, then DELOCALIZED.

    Route change post-launch     New route bypasses area with dense lines.
                                 Was fine before, now DELOCALIZED on
                                 the new segment.
```

---

# PART 5 — NO_LINE (STATUS = 3) AND MISSED_LINE_UPDATES (STATUS = 8)

## 5.1 No_Line: Robot Cannot See Any Lines

NO_LINE means the line-sensor is functioning but reporting that no floor lines are visible. This typically means:

```
    NO_LINE CAUSES AND GEOMETRIES
    ══════════════════════════════

    CAUSE 1: Robot is physically off the line path
    ────────────────────────────────────────────────

    Map view:
                    ═══ LINE ═══════════════════ LINE ═══
                         ↑                          ↑
                     should be                should be
                     here                     here

                            ● ← robot is HERE (off path)

    The line-sensor correctly reports no lines — there are none under it.
    This is NOT a sensor failure. It is a navigation failure upstream
    (robot drove off path, map was wrong, path planning error).

    CAUSE 2: Blank floor section (loading dock, maintenance area)
    ──────────────────────────────────────────────────────────────
    Robot entered a floor area that genuinely has no lines.
    Line map doesn't cover this area. If the robot is supposed
    to be here, the map needs to be updated.

    CAUSE 3: Line-Sensor height wrong / floor gap
    ──────────────────────────────────────────────────────────────
    Floor dip or line-sensor mounting issue puts sensor too far
    from floor. Works normally except at this one location.
```

## 5.2 Missed_Line_Updates: The Subtle One

MISSED_LINE_UPDATES (status=8) is the most commonly misdiagnosed failure mode. The key feature that makes it hard:

```
    MISSED_LINE_UPDATES — WHAT MAKES IT HARD TO DIAGNOSE
    ══════════════════════════════════════════════════════

    Symptom surface:  ERROR_STATE_INVALID (same as SLIPPED)
                      Covariance growing (same as DELOCALIZED)

    What's different:
    ┌──────────────────────────────────────────────────────────────┐
    │  Topic: /line-sensor/detections                                │
    │  Rate: 100 Hz (looks completely healthy)                     │
    │  is_reliable: True                                           │
    │                                                              │
    │  But: estimator is REJECTING every measurement               │
    │       because they fail the innovation gate check            │
    │       OR because every sample has the same timestamp         │
    └──────────────────────────────────────────────────────────────┘

    This is the distinction:
    • NO_LINE:               topic publishes, no lines detected → no measurements
    • MISSED_LINE_UPDATES:   topic publishes, lines detected → measurements rejected
```

## 5.3 Why Measurements Get Rejected

The EKF measurement update includes an **innovation gate** (chi-squared test). A measurement is rejected if the innovation is too far from the expected value:

```
    INNOVATION GATE CHECK
    ══════════════════════

    For each line-sensor measurement z:
        innovation = z - H × x_predicted

        Mahalanobis distance:
        d² = innovationᵀ × S⁻¹ × innovation
        where S = H × P × Hᵀ + R

        IF d² > chi2_threshold (typically 5.99 for 95%, 2 DOF):
            REJECT measurement  ← counted as "missed update"

    When are many measurements rejected?
    ──────────────────────────────────────
    1. Estimator position estimate is very wrong:
       - After slip, position is off by large amount
       - Every real measurement looks like an outlier

    2. Line-Sensor reports wrong line positions:
       - SPI conflict producing corrupted measurements
       - Line coordinates systematically shifted

    3. All measurements have identical timestamp:
       - Only first sample in each burst counts
       - System counts subsequent ones as duplicates → rejected
       - This is the SPI conflict signature
```

## 5.4 Distinguishing NO_LINE from MISSED_LINE_UPDATES

```
    DIAGNOSTIC DECISION TREE
    ═════════════════════════

                Start: status=8 or cov growing despite topic alive?
                              │
                              ▼
              Is /line-sensor/detections publishing at normal rate?
              ┌───────────────────────────────────────────────────┐
              │ YES (rate OK)                  NO (rate dropped)   │
              └───────────────────────────────────────────────────┘
                     │                                   │
                     ▼                                   ▼
         Is is_reliable = True?                  Status = NO_LINE (3)
              │                                  Physical: off path or
              ▼                                  line-sensor mounting
          YES → MISSED_LINE_UPDATES (8)
                Check: SPI error logs?
                Check: measurement timestamps duplicated?
                Check: estimator position vs map (is it plausible?)

          NO  → Line-Sensor temporarily offline
                Check: SPI bus conflict, cable, power
                This leads to DELOCALIZED as cov grows
```

> **Key insight:** The topic being alive (rate=100Hz, is_reliable=True) is **necessary but not sufficient** for updates to be accepted. You must also verify that measurements are passing the innovation gate. This requires looking at the estimator's internal logging, not just the line-sensor topic.

---

# PART 6 — MISSING_ODOM (STATUS = 7) AND UNINITIALIZED (STATUS = 2)

## 6.1 Missing_Odom: Dead Motion Model

MISSING_ODOM occurs when the `/odom` topic stops delivering messages. The estimator uses odometry as the **motion model** — without it, the Kalman filter prediction step cannot run:

```
    EFFECT OF MISSING ODOM ON ESTIMATOR
    ═════════════════════════════════════

    Normal operation:
    ─────────────────
    odom@50Hz ──► predict() ──► covariance grows slowly
                                      position updated from wheel motion

    Missing odom:
    ─────────────
    No odom  ──► predict() skipped ──► position FROZEN
                                             covariance grows from
                                             time-based staleness term only
                                             (much slower than motion-based growth)

    After timeout (configurable, typically 0.5-1.0s):
    ──────────────────────────────────────────────────
    status = MISSING_ODOM (7)

    Unlike SLIP: covariance does not jump to INF.
    It grows from the staleness scaling, which is slower.
    Robot's last known position is assumed correct until
    covariance threshold is crossed.
```

## 6.2 Common Causes of Missing Odom

```
    CAUSE                           HOW TO IDENTIFY
    ─────────────────────────       ──────────────────────────────────
    Motor controller crash          /odom stops, motor logs show reset
    (firmware fault)                coincidence; other motor topics stop too

    Network partition               /odom stops on all topics simultaneously;
    (ROS node death)                multiple topics disappear together

    odom_publisher node crash       Only /odom stops; motor topics continue;
                                    ROS log shows node respawn

    CPU overload causing            /odom rate drops slowly, not sudden stop;
    scheduling delay                check CPU logs for saturation
```

## 6.3 Uninitialized: Pre-Startup State

UNINITIALIZED (status=2) means the estimator has never received an initial pose. It is a startup state, not a failure mode in normal operation:

```
    UNINITIALIZED OCCURRENCE PATTERN
    ══════════════════════════════════

    Normal startup:
    ───────────────
    Estimator starts → status=2 (UNINITIALIZED)
              │
              ▼
    Receives /initialpose or localization result
              │
              ▼
    status=6 (INITIALIZING) during first prediction steps
              │
              ▼
    status=0 (OK) — ready for operation

    Problematic occurrence:
    ────────────────────────
    If status=2 appears AFTER a period of normal operation,
    it means the estimator node RESTARTED (was killed and respawned)
    without receiving a new initial pose.

    Check: ROS node restart logs, watchdog events, system OOM events.
```

---

# PART 7 — ERROR_STATE_INVALID

## 7.1 Where It Comes From: Not the Estimator

This is the most important architectural fact in this entire document:

```
    ╔══════════════════════════════════════════════════════════╗
    ║  ERROR_STATE_INVALID                          ║
    ║  is logged by                          ║
    ║  NOT by                                     ║
    ║                                                          ║
    ║  The estimator's job is to maintain state.               ║
    ║  The estimator may SET cov to INF and keep running.      ║
    ║                                                          ║
    ║  's job is to issue velocity commands. ║
    ║  It checks state validity BEFORE issuing any command.    ║
    ║  If state is invalid, it refuses to issue the command.   ║
    ╚══════════════════════════════════════════════════════════╝
```

## 7.2 The isFinite() Check

In ``, the `executePoseGoal()` function (around ) contains:

```cpp
    // Pseudocode representation of the isFinite() check
    bool state_is_valid = isFinite(estimated_state.x)
                       && isFinite(estimated_state.y)
                       && isFinite(estimated_state.theta)
                       && isFinite(estimated_state.vx)
                       && isFinite(estimated_state.vy)
                       && isFinite(estimated_state.cov_xx)
                       && isFinite(estimated_state.cov_yy)
                       && isFinite(estimated_state.cov_theta);

    if (!state_is_valid) {
        log_error("ERROR_STATE_INVALID");
        // stop robot, do not proceed
        return;
    }
    // ... proceed with motion command
```

The check is exhaustive: **any single NaN or Inf** in any of those eight fields triggers the stop.

## 7.3 Which Fields Actually Go Infinite?

In practice, from the estimator code:

```
    FIELD           GOES INFINITE?    WHEN
    ─────────────   ──────────────    ────────────────────────────────
    x               No normally       Only if initial pose is Inf (bug)
    y               No normally       Only if initial pose is Inf (bug)
    theta           No normally       Only if initial pose is Inf (bug)
    vx              No normally       Only via numerical instability
    vy              No normally       Only via numerical instability
    cov_xx          YES ★             When SLIPPED → set to INF explicitly
    cov_yy          YES ★             When SLIPPED → set to INF explicitly
    cov_theta       YES ★             When SLIPPED → set to INF explicitly

    The most common cause of ERROR_STATE_INVALID
    is SLIPPED + robot attempting next motion command.
```

## 7.4 The Causal Chain for the Most Common Case

```
    MOST COMMON SEQUENCE LEADING TO ERROR_STATE_INVALID
    ════════════════════════════════════════════════════════════════

    T=0.00s   Robot driving normally, cov_xx=0.001
    T=0.06s   Robot hits wet floor patch
    T=0.08s   Velocity drops faster than max_decel
    T=0.08s   odometryCallback() slip check FAILS
    T=0.08s   cov_xx = cov_yy = cov_theta = INF  (estimator sets this)
    T=0.08s   status = SLIPPED (4)               (estimator sets this)
    T=0.08s   Estimator continues running         (it still works, just distrusts)
    T=0.10s   Planner sends next motion goal
    T=0.10s    calls executePoseGoal()
    T=0.10s   isFinite(cov_xx) returns FALSE      (INF fails isFinite)
    T=0.10s   ERROR_STATE_INVALID logged
    T=0.10s   Robot stops safely

    ROOT CAUSE: wet floor
    MECHANISM: slip → cov→INF → isFinite fails
    ERROR MESSAGE: ERROR_STATE_INVALID
    VISIBLE SYMPTOM: robot stopped
```

## 7.5 The Safe Design Intent

The isFinite() check is an intentional **safety gate**. The philosophy:

```
    THE SAFETY GATE PHILOSOPHY
    ═══════════════════════════

    WITHOUT the isFinite() check:
    ──────────────────────────────
    Robot state = (x=5.0, y=3.0, cov_xx=INF)
    Path planner: "command robot to move to (6.0, 3.0)"
    executePoseGoal() computes: delta = (1.0, 0.0)
    PID control: some NaN or Inf propagates into velocity command
    Robot: moves somewhere random at unknown speed
    Outcome: CRASH

    WITH the isFinite() check (actual design):
    ──────────────────────────────────────────
    Robot state = (x=5.0, y=3.0, cov_xx=INF)
    Path planner: "command robot to move to (6.0, 3.0)"
    executePoseGoal() checks isFinite(cov_xx) → FALSE
    Robot: stops immediately, logs error
    Operator: can safely intervene
    Outcome: SAFE STOP
```

> **Key insight:** `ERROR_STATE_INVALID` is not a bug. It is the correct safety response to a corrupted estimator state. **The bug is what caused the state to become corrupted.** Your RCA must go one level deeper.

---

# PART 8 — SYSTEMATIC BAG INVESTIGATION WORKFLOW

## 8.1 Setup: What to Load First

```
    BAG INVESTIGATION CHECKLIST
    ════════════════════════════

    □ 1. List all topics in the bag
         rosbag info <bagfile>
         Look for: /estimated_state, /odom, /line-sensor/detections,
                   /cmd_vel, /imu/data, /rosout

    □ 2. Get time bounds
         What is the bag start time? What is the incident time?
         What is the delta?

    □ 3. Check topic rates (are any missing or slow?)
         rostopic hz /line-sensor/detections  (expect ~100Hz)
         rostopic hz /odom                  (expect ~50Hz)
         rostopic hz /imu/data              (expect ~100Hz)

    □ 4. Find the first non-OK status
         rostopic echo /estimated_state | grep status
         Or use rqt_plot for visual timeline
```

## 8.2 The Investigation Decision Tree

```
    ┌─────────────────────────────────────────────────────┐
    │         OPEN BAG: find first status ≠ 0             │
    └─────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  status = 4 (SLIPPED)?        │──YES──► Section 8.3: Slip path
              └───────────────────────────────┘
                              │ NO
                              ▼
              ┌───────────────────────────────┐
              │  status = 5 (COLLISION)?      │──YES──► Section 8.4: Collision path
              └───────────────────────────────┘
                              │ NO
                              ▼
              ┌───────────────────────────────┐
              │  status = 1 (DELOCALIZED)?    │──YES──► Check: covariance growth
              └───────────────────────────────┘          rate slow? → DELOCALIZED
                              │ NO                        rate fast? → might be 8
                              ▼
              ┌───────────────────────────────┐
              │  status = 3 (NO_LINE)?        │──YES──► Check line-sensor rate
              └───────────────────────────────┘          rate=0? → off path
                              │ NO                        rate>0 with 0 detections
                              ▼                           → blank floor section
              ┌───────────────────────────────┐
              │  status = 8 (MISSED_UPDATES)? │──YES──► Check: rejection logs
              └───────────────────────────────┘          SPI error logs?
                              │ NO                        Duplicate timestamps?
                              ▼
              ┌───────────────────────────────┐
              │  status = 7 (MISSING_ODOM)?   │──YES──► Check odom topic
              └───────────────────────────────┘          Rate dropped? Node crash?
                              │ NO
                              ▼
              ┌───────────────────────────────┐
              │  status = 2 (UNINITIALIZED)?  │──YES──► Check: did node restart?
              └───────────────────────────────┘          Was initial pose sent?
```

## 8.3 Slip Investigation Steps

```
    STEP 1: Confirm slip signature
    ──────────────────────────────
    • Plot cov_xx vs time: look for step jump to 1e5+
    • Note the EXACT timestamp of the jump (call it T_slip)

    STEP 2: Go back to T_slip - 0.5s
    ─────────────────────────────────
    • Plot odom velocity vs time
    • Look for the velocity spike-then-drop pattern
    • Note: was the robot at a specific XY location? (check /amcl_pose or /estimated_state.x,y)

    STEP 3: Is it location-specific?
    ──────────────────────────────────
    • Compare XY at T_slip with floor map
    • Is there a known surface change here? (ramp, tile type, cleaning schedule)
    • Check other incident bags: same XY every time → floor issue

    STEP 4: Rule out false positive
    ────────────────────────────────
    • Check motor driver logs for errors near T_slip
    • Check if odom messages look corrupted (sudden large delta_x in one sample)
    • If velocity spike appears in ONE sample only and then normal → likely corrupt message
```

## 8.4 Collision Investigation Steps

```
    STEP 1: Confirm collision signature
    ────────────────────────────────────
    • Plot /imu/data angular_velocity.z vs time
    • Look for spike > 2 rad/s, duration < 100ms
    • Note timestamp T_collision

    STEP 2: Classify: real or false positive?
    ──────────────────────────────────────────
    • Is there any evidence of an obstacle in the path?
      (camera footage, operator report, obstacle detector topic)
    • Is the omega_z spike oscillatory (false positive) or impulsive (real)?
    • Does the spike coincide with a known floor feature (bump, threshold)?

    STEP 3: Check power and EMI
    ────────────────────────────
    • Was the spike random or location/time correlated?
    • Any power supply warnings in logs near T_collision?
    • Is there heavy machinery running nearby at that time?
```

---

# PART 9 — AMR CODE CONNECTION

## 9.1 Full Causal Chain Table

| Failure | Trigger Location | Code Mechanism | State Change | Robot Stop Path |
|---------|-----------------|----------------|--------------|-----------------|
| SLIPPED (4) | ``<br>`odometryCallback()`  | Rolling velocity window check; decel > max_decel | `cov_xx = cov_yy = cov_theta = INF`<br>`status = 4` | ``<br>`executePoseGoal()` L~160<br>`isFinite()` fails |
| COLLISION (5) | ``<br>`imuCallback()` | Gyro omega_z > threshold | `status = 5` (terminal) | Dedicated collision stop path (not isFinite) |
| DELOCALIZED (1) | ``<br>Post-prediction | `cov_xx` or `cov_yy` > delocalized_threshold | `status = 1` | Robot continues; reduced capability |
| NO_LINE (3) | ``<br>Line-Sensor callback | Line-Sensor reports 0 line detections | `status = 3` | No stop; continues dead-reckoning |
| MISSED_UPDATES (8) | ``<br>EKF update | All measurements rejected by innovation gate | `status = 8` | No stop; cov grows → may lead to DELOCALIZED |
| MISSING_ODOM (7) | ``<br>Odom watchdog | No odom message for > timeout | `status = 7` | No stop unless cov crosses threshold |
| NAV_NOT_FINITE | ``<br>`executePoseGoal()` L~160 | `isFinite()` check on any state field | — | **SAFE STOP** logged as error |

## 9.2 Status Code Quick Reference

```cpp
    // estimator.h (approximate)
    enum NavigatorStateStatus {
        OK                   = 0,   // All systems normal
        DELOCALIZED          = 1,   // Covariance exceeded threshold
        UNINITIALIZED        = 2,   // No initial pose received
        NO_LINE              = 3,   // Line-Sensor sees no lines
        SLIPPED              = 4,   // Velocity window check failed → cov→INF
        COLLISION            = 5,   // IMU gyro spike detected (TERMINAL)
        INITIALIZING         = 6,   // Startup state, receiving first pose
        MISSING_ODOM         = 7,   // Odom topic silent > timeout
        MISSED_LINE_UPDATES  = 8    // Line-Sensor alive but updates rejected
    };
```

## 9.3 The isFinite() Guard — Conceptual Implementation

```cpp
    //  executePoseGoal() — pseudocode
    void moveToPose(const Pose& target) {
        const NavigatorState& state = estimator_.getState();

        // SAFETY GATE: check ALL fields before issuing any velocity
        if (!std::isfinite(state.x)       ||
            !std::isfinite(state.y)       ||
            !std::isfinite(state.theta)   ||
            !std::isfinite(state.vx)      ||
            !std::isfinite(state.vy)      ||
            !std::isfinite(state.cov_xx)  ||
            !std::isfinite(state.cov_yy)  ||
            !std::isfinite(state.cov_theta)) {
            ROS_ERROR("ERROR_STATE_INVALID");
            stopRobot();        // safe stop
            return;             // do NOT issue velocity command
        }

        // Only reaches here if all fields are finite
        // ... compute and issue velocity command ...
    }
```

## 9.4 Covariance Propagation Through Slip

```
    MATHEMATICAL TRACE OF SLIP EVENT
    ══════════════════════════════════

    T=0: Normal state
    ─────────────────
    x = (x₀, y₀, θ₀)ᵀ
         ⎡ 0.001   0      0   ⎤
    P =  ⎢ 0       0.001  0   ⎥   (well-localized)
         ⎣ 0       0      0.0002⎦

    T=0.08s: Slip detected in odometryCallback()
    ─────────────────────────────────────
    x = (x₀, y₀, θ₀)ᵀ   ← position UNCHANGED (estimator doesn't know where robot went)
         ⎡ ∞   0   0 ⎤
    P =  ⎢ 0   ∞   0 ⎥   ← explicitly set to INF
         ⎣ 0   0   ∞ ⎦

    T=0.10s: executePoseGoal() called
    ────────────────────────────
    isFinite(P[0,0]) = isFinite(∞) = FALSE
    → ERROR_STATE_INVALID
    → robot stops

    T=later: Robot crosses a line-sensor line
    ────────────────────────────────────────
    Line-Sensor provides: z = measured_lateral_offset
    EKF update: P_new = (I - K×H) × P_old + measurement_noise_term
    With P_old = INF: K is computed differently (large gain)
    After update: P_new = R (measurement covariance only)
    → covariance reset to measurement quality
    → status → OK
```

---

## Summary — What to Remember

| Status | Name | Cov Change | Terminal? | Recovery | Root Cause Hunt |
|--------|------|-----------|-----------|----------|-----------------|
| 0 | OK | Normal | — | — | No failure |
| 1 | DELOCALIZED | Slow rise to threshold | No | Next line crossing | Long gap between lines; line-sensor offline |
| 2 | UNINITIALIZED | N/A | No | Receive initial pose | Node restart without re-init |
| 3 | NO_LINE | Slow rise | No | Robot onto a line | Off-path; blank floor; line-sensor height |
| 4 | SLIPPED | **Jumps to INF** | No | Next line crossing | Wet floor; motor driver; corrupt odom |
| 5 | COLLISION | Irrelevant | **YES** | Human restart | Real impact; EMI; power glitch; floor bump |
| 6 | INITIALIZING | N/A | No | Init completes | Normal startup |
| 7 | MISSING_ODOM | Slow rise (staleness) | No | Odom resumes | Motor ctrl crash; network; node death |
| 8 | MISSED_UPDATES | Slow rise | No | Valid measurements | SPI conflict; innovation gate rejections; duplicate timestamps |

**The diagnostic mantra:**

```
    ERROR_STATE_INVALID is never the root cause.
    SLIPPED is rarely the root cause.
    The root cause is physical: floor, sensor, power, network.
    Work backwards: error → mechanism → physics.
```

**The covariance scale key:**

```
    cov_xx value    Meaning
    ────────────    ────────────────────────────────────────
    < 0.001         Well-localized (recent line-sensor update)
    0.001 – 0.01    Normal dead-reckoning drift
    0.01 – 0.05     Elevated, approaching DELOCALIZED threshold
    > 0.05          DELOCALIZED (threshold dependent on config)
    1e4 – 1e6       SLIPPED (set explicitly by odometryCallback())
    INF             SLIPPED (set explicitly by odometryCallback())
```
