# Exercise04 — Nav2 Diagnostics: BT Traces and Parameter Tuning
### Companion exercises for `03-nav2-architecture.md`

**Estimated time:** 45 minutes  
**Prerequisite:** [03-nav2-architecture.md](../03-nav2-architecture.md)

**Self-assessment guide:** Section A and B are pure analysis — no code to run. Work through
each scenario before expanding the answer. If you can diagnose Section A in under
2 minutes per scenario, you can triage Nav2 failures in field logs at speed.

---

## Overview

Nav2 failures in the field rarely announce themselves clearly. Instead, you see a robot
standing still, spinning in place, or taking a 5-minute detour — and a wall of log output
that contains the answer if you know where to look. This exercise set practises reading
BehaviorTree execution traces and Nav2 YAML parameters to find the root cause without
needing to reproduce the issue.

---

## Section A — BT Trace Reading

Each scenario gives an abbreviated Nav2 log. Diagnose what went wrong and what to check.

---

### Scenario 1 — Robot Never Leaves Starting Position

```
[bt_navigator] BT tick
[bt_navigator]   ComputePathToPose: RUNNING
[bt_navigator]   ComputePathToPose: FAILURE
[bt_navigator]   ClearEntireCostmap: RUNNING
[bt_navigator]   ClearEntireCostmap: SUCCESS
[bt_navigator]   ComputePathToPose: RUNNING
[bt_navigator]   ComputePathToPose: FAILURE
[bt_navigator]   ClearEntireCostmap: RUNNING
[bt_navigator]   ClearEntireCostmap: SUCCESS
[bt_navigator]   ComputePathToPose: RUNNING
[bt_navigator]   ComputePathToPose: FAILURE
[bt_navigator] Goal aborted after 3 recovery attempts
[nav2_planner]  No path found between (1.2, 0.8) and (5.5, 3.1)
```

**Questions:**  
1. What is wrong?  
2. What three things would you check to diagnose this?  
3. `ClearEntireCostmap SUCCESS` appears — why does this not fix the issue?

<details><summary>Answer</summary>

**What is wrong:**
The global planner (`nav2_planner`) cannot find a valid path from the robot's position to
the goal. Clearing the costmap (removing transient obstacles) does not help — this means
either the goal is unreachable (in a wall, behind an impassable barrier), or the
`inflation_radius` is so large that every path through the environment is blocked.

**Three things to check:**

1. **Is the goal pose inside an obstacle?** Visualize the global costmap in RViz. If the
   goal marker is on a lethal (red) cell, the planner will always fail regardless of the
   path. This is the most common cause.

2. **Is `inflation_radius` too large for the environment?** If `inflation_radius` is
   wider than half the narrowest corridor, there is no passable path even in an open map.
   Check `global_costmap.inflation_layer.inflation_radius` against the corridor width.
   Rule: `inflation_radius` must be < (corridor_width / 2).

3. **Is the static map up to date?** An outdated map may show passable areas that are
   physically blocked (or vice versa). Check whether the `map_server` is publishing the
   correct map and that it matches the current environment layout.

**Why ClearEntireCostmap does not fix it:**
`ClearEntireCostmap` removes *transient* obstacles from the costmap — obstacles added by
the obstacle layer from sensor data. It does NOT remove:
- The static layer (loaded from the map file)
- The inflation layer around static obstacles

If the path is blocked by a permanent wall in the static map, or by inflation from a wall,
clearing transient obstacles does nothing. The planner will fail again immediately.

</details>

- [ ] Done

---

### Scenario 2 — Robot Oscillates Near Goal

```
[bt_navigator]   FollowPath: RUNNING
[bt_navigator]   FollowPath: RUNNING
[bt_navigator]   FollowPath: RUNNING
...  (50 more RUNNING ticks)
[bt_navigator]   FollowPath: FAILURE
[bt_navigator]   ComputePathToPose: RUNNING   (replanning to same goal)
[bt_navigator]   ComputePathToPose: SUCCESS
[bt_navigator]   FollowPath: RUNNING
...  (repeating cycle, robot oscillating ~0.5m from goal)
[nav2_controller] [WARN] Goal not reached after 30s timeout
```

**Questions:**  
1. Why is `FollowPath` reporting FAILURE near the goal?  
2. What parameter is most likely misconfigured?  
3. Why does `ComputePathToPose: SUCCESS` appear repeatedly? Is replanning the problem?

<details><summary>Answer</summary>

**Why FollowPath reports FAILURE near the goal:**
The controller is running `FollowPath` (tracking the path), but the `GoalReached` BT
condition never triggers because the robot never gets close enough to satisfy the goal
tolerance. The controller oscillates around the goal position — the DWB or RPP controller
generates velocity commands that overshoot, causing the robot to swing back and forth
around the tolerance boundary.

`FollowPath` eventually fails when the controller's own timeout (`goal_time_tolerance`,
or the controller's oscillation detector) triggers.

**Most likely misconfigured parameter:**
`xy_goal_tolerance` or `yaw_goal_tolerance` is too tight relative to the controller's
ability to stop precisely. For a typical differential drive robot:
- `xy_goal_tolerance: 0.05` m with a high-mass robot may be impossible to hit reliably.
- Try increasing to `0.25` m for initial testing, then tighten if needed.

Also check `goal_checker` type in the controller server config — some goal checkers apply
both XY and yaw tolerance simultaneously; if the robot arrives at the correct position
but with the wrong heading, it will oscillate trying to rotate in place.

**Why ComputePathToPose SUCCESS keeps appearing:**
Replanning is NOT the problem — it is the symptom. The BT's `GoalUpdated` or
`DistanceToGoal` condition triggers replanning because the controller failed. The planner
CAN find a path (short path from current position to the goal nearby), so it succeeds.
But the controller then fails again for the same reason. The loop continues indefinitely
until the overall timeout fires.

Replanning to the same nearby goal is correct behaviour; the root cause is in the
controller's goal tolerance settings.

</details>

- [ ] Done

---

### Scenario 3 — Robot Ignores an Obstacle and Runs Into It

```
[costmap_2d] [WARN] Laser scan dropped: stamp 2026-04-01T12:00:00.420Z is
             0.42s behind current time (threshold: 0.20s)
[costmap_2d] [WARN] Laser scan dropped: stamp 2026-04-01T12:00:00.423Z is
             0.42s behind current time (threshold: 0.20s)
...  (repeated at 10Hz — every scan is dropped)
[nav2_controller] FollowPath: RUNNING
[nav2_controller] FollowPath: RUNNING
[nav2_controller] FollowPath: FAILURE   (obstacle collision)
```

**Questions:**  
1. Why is the local costmap not showing the obstacle?  
2. What is the root cause — is this a sensor failure or a timing issue?  
3. What two things would you check or change to fix this?

<details><summary>Answer</summary>

**Why the local costmap does not show the obstacle:**
The `costmap_2d` obstacle layer receives laser scan messages but rejects every single one
with "stamp behind current time threshold". Because no scans are accepted, the obstacle
layer never marks any cells as occupied. The local costmap appears empty even though the
physical sensor is detecting the obstacle.

**Root cause — timing issue, not sensor failure:**
The laser scan's `header.stamp` is approximately 0.42 s behind the costmap's current time.
This means the scan was timestamped on the sensor's clock, and there is a ~420 ms lag
between the sensor clock and the ROS2 system clock.

This is typically caused by one of:
1. **Clock offset between a sensor driver running on a separate embedded system** (sensor
   publishes with its own MCU clock) and the main computer running ROS2.
2. **A bag replay** where the bag's timestamps are old but the system clock is real-time.
3. **High processing delay** — the scan is computed/republished by a pre-processing node
   that adds significant latency before costmap_2d sees it.

The sensor is working correctly (it produces valid data). The issue is time synchronization.

**Two fixes:**

1. **Increase the stamp tolerance (quick fix):**
   ```yaml
   local_costmap:
     obstacle_layer:
       observation_sources: scan
       scan:
         max_obstacle_height: 2.0
         expected_update_rate: 10.0
         observation_persistence: 0.0
         inf_is_valid: false
         # Increase tolerance from 0.2s to 0.5s:
         min_obstacle_height: 0.0
   ```
   In newer Nav2 versions, the parameter is `obstacle_max_range` + `observation_source`
   `raytrace_max_range`. The stamp tolerance is `obstacle_persistence_time`. Check your
   version's parameter names.
   
   The correct parameter is typically in `costmap_2d` plugin config as the tolerance field.

2. **Fix the clock synchronization (proper fix):**
   Use `chrony` or `PTP` to sync the sensor computer's clock to the navigation computer.
   If replaying a bag, use `--clock` flag: `ros2 bag play --clock <bag>` so nodes use
   bag time instead of wall clock time.

</details>

- [ ] Done

---

### Scenario 4 — Robot Takes a Very Long Detour

```
Environment: warehouse corridor, 4m wide, 30m long
inflation_radius: 3.5  (in global_costmap params)
[nav2_planner] Path found: length=67.2m
               (expected direct path: ~12m)
[nav2_planner] Path found: length=67.2m
```

The global costmap looks visually correct — no unexpected obstacles.

**Questions:**  
1. What is wrong with the `inflation_radius` value?  
2. Draw (in text) what the inflated costmap looks like in a 4m-wide corridor.  
3. What value should `inflation_radius` be set to for this environment?

<details><summary>Answer</summary>

**What is wrong:**
`inflation_radius = 3.5` m means every wall cell inflates its "danger zone" by 3.5 m in
all directions. In a 4 m wide corridor, both walls inflate toward the centre:
- Left wall inflates 3.5 m to the right.
- Right wall inflates 3.5 m to the left.

These inflation zones overlap in the middle of the corridor. The planner sees no
"free" (cost 0) cells anywhere in the corridor — the entire passable area has non-zero
cost (inflated lethal or near-lethal). The planner finds a path that goes *around* the
entire corridor through much wider areas, producing the 67 m detour.

**Text diagram of the inflated costmap:**

```
Wall (lethal) ████████████████████████████████████████████
Inflated      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
              ←────────── 3.5m ──────────→
Corridor      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  (only 4m total width)
              ←──────────── 4.0m ────────────→
Inflated      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
              ←────────── 3.5m ──────────→
Wall (lethal) ████████████████████████████████████████████

Overlap zone: 3.5 + 3.5 = 7m > 4m corridor width → FULLY INFLATED, no free cells
```

**Correct value:**
For a robot with footprint radius `r` in a corridor of width `w`, the inflation radius
should satisfy:
```
inflation_radius < (w / 2) - r
```

For a 4 m corridor and a typical robot with `r = 0.3 m`:
```
inflation_radius < (4.0 / 2) - 0.3 = 1.7 m
```

A safe value is `inflation_radius = 0.5` for conservative inflation, or up to `1.5` if you
want cost gradients that guide the robot away from walls without blocking the path entirely.

**Rule of thumb:** `inflation_radius` should be set to 1.5–2× the robot's collision radius
as a minimum to create a smooth cost gradient, but must never exceed `corridor_width / 2`.

</details>

- [ ] Done

---

## Section B — Parameter Tuning

Each YAML snippet below has one parameter problem. Identify it and give the corrected value.

---

**B1.** DWB controller that fails to turn in place (robot drives in large circles instead
of rotating toward the goal).

```yaml
controller_server:
  ros__parameters:
    controller_plugins: ["FollowPath"]
    FollowPath:
      plugin: "dwb_core::DWBLocalPlanner"
      min_vel_theta: 0.5      # minimum rotational velocity
      max_vel_theta: 1.5      # maximum rotational velocity
      min_speed_xy: 0.3       # minimum linear speed
      acc_lim_theta: 3.2
      decel_lim_theta: -3.2
```

<details><summary>Answer</summary>

**Problem:** `min_speed_xy: 0.3` prevents the controller from producing zero linear
velocity. In-place rotation requires `vx = 0, vy = 0, ω ≠ 0`. If `min_speed_xy` is
non-zero, DWB cannot generate a trajectory with zero linear speed — it must always move
forward or backward at least 0.3 m/s. The controller compensates by pairing rotation with
linear motion, producing the circular arc behaviour.

**Fix:**
```yaml
      min_speed_xy: 0.0    # allow pure rotation (vx=0, vy=0)
```

**Secondary check:** Also verify that `min_vel_x` is not set to a non-zero minimum.
On a differential drive robot, both `min_speed_xy` and `min_vel_x` must allow zero to
enable in-place turns.

</details>

- [ ] Done

---

**B2.** Global costmap that never shows sensor data (obstacle layer shows only static
map, never live sensor readings).

```yaml
global_costmap:
  global_costmap:
    ros__parameters:
      plugins: ["static_layer", "obstacle_layer", "inflation_layer"]
      obstacle_layer:
        plugin: "nav2_costmap_2d::ObstacleLayer"
        enabled: true
        observation_sources: scan
        scan:
          topic: /scan
          max_obstacle_height: 2.0
          clearing: true
          marking: true
          data_type: "LaserScan"
          raytrace_max_range: 10.0
          obstacle_max_range: 10.0
      update_frequency: 1.0
      publish_frequency: 1.0
      global_frame: map
      robot_base_frame: base_link
      static_map: true
```

<details><summary>Answer</summary>

**Problem:** `update_frequency: 1.0` — the global costmap only updates at 1 Hz. For a
robot moving at even 0.3 m/s, this means obstacles are added to the costmap only once per
second. Combined with `publish_frequency: 1.0`, the planner is working with stale data.

However, the deeper problem in most "never shows sensor data" cases is the **QoS mismatch
or TF issue**. In this config, check:

1. **`/scan` topic QoS:** The obstacle layer subscribes with `BEST_EFFORT`. If the laser
   driver publishes with `RELIABLE`, there is no mismatch (RELIABLE satisfies BEST_EFFORT
   requests). But if reversed, scans are silently dropped.

2. **Missing `rolling_window: true`:** For the *global* costmap, `static_map: true` and
   rolling window are mutually exclusive. If the obstacle layer is not marking on the
   static map, check whether the layer is within the static map bounds.

**For live sensor responsiveness, fix `update_frequency`:**
```yaml
      update_frequency: 10.0     # update 10× per second
      publish_frequency: 5.0     # publish 5× per second
```

**Note:** `update_frequency` for the global costmap is typically set lower (1–5 Hz) than
the local costmap (10 Hz) to reduce CPU load. But 1 Hz means a new obstacle is invisible
to the planner for up to 1 second after it appears.

</details>

- [ ] Done

---

**B3.** AMCL that loses localization when the robot is stationary (pose estimate slowly
drifts even while the robot is not moving).

```yaml
amcl:
  ros__parameters:
    min_particles: 500
    max_particles: 2000
    update_min_d: 0.0     # update on any movement, including zero
    update_min_a: 0.0     # update on any rotation, including zero
    resample_interval: 1
    laser_max_range: 10.0
    laser_min_range: 0.2
    set_initial_pose: true
    initial_pose:
      x: 0.0
      y: 0.0
      z: 0.0
      yaw: 0.0
```

<details><summary>Answer</summary>

**Problem:** `update_min_d: 0.0` and `update_min_a: 0.0` mean AMCL runs its particle filter
update on *every* incoming laser scan, even when the robot has not moved. This causes
two issues:

1. **Spurious particle diffusion:** Each MCL update cycle adds process noise to the
   particle cloud (to handle motion uncertainty). With zero thresholds, this noise is added
   continuously even at rest, gradually spreading the particle cloud and causing the pose
   estimate to drift.

2. **CPU waste:** Running the full particle filter update at laser scan rate (10 Hz) while
   stationary is expensive and unnecessary.

**Fix:** Set meaningful minimum movement thresholds. AMCL should only update when the robot
has moved enough for the laser data to provide genuinely new information.

```yaml
    update_min_d: 0.2    # update after moving at least 0.2m
    update_min_a: 0.1    # update after rotating at least ~6 degrees (radians)
```

These are the Nav2 default values. They ensure:
- AMCL only runs expensive updates when new information is available.
- Process noise is only injected when the robot is actually moving.
- The particle cloud stays tight when the robot is stationary.

</details>

- [ ] Done

---

## Section C — The "Robot Stuck" Diagnostic Checklist

**Task:** Without looking at any references, write out a step-by-step diagnostic checklist
for investigating a robot that has stopped navigating. Your checklist should have at least
6 steps and cover the four Nav2 layers (localization, costmap, planner, BT).

Write your checklist now, then expand the answer to compare.

<details><summary>Answer</summary>

### 6-Step Nav2 "Robot Stuck" Diagnostic Checklist

---

**Step 1 — Read the BT navigator status**
```bash
ros2 topic echo /bt_navigator/transition_event
ros2 action list
ros2 action info /navigate_to_pose
```
Is the `NavigateToPose` action still EXECUTING, ABORTED, CANCELED, or SUCCEEDED?
- `ABORTED` → the BT ran out of recovery attempts. Look for `[bt_navigator] Goal aborted`.
- `SUCCEEDED` → the robot thinks it arrived. Check goal tolerance settings.
- Action not listed → the BT navigator node crashed or was never activated.

---

**Step 2 — Check the last BT tick in the logs**
Look for the repeating pattern in the BT log output:
```
grep -E "FAILURE|RUNNING|SUCCESS" <log>
```
Identify which BT node is failing. Common patterns:
- `ComputePathToPose: FAILURE` → planner problem (Step 4)
- `FollowPath: FAILURE` repeating → controller problem (Step 5)
- `ComputePathToPose: RUNNING` never returns → planner timeout / stuck in computation

---

**Step 3 — Check localization**
```bash
ros2 topic echo /amcl_pose --once
ros2 run tf2_ros tf2_echo map base_link
```
Is the `map → base_link` transform being published and updating?
- Transform not available → AMCL crashed, or `map → odom` not being published.
- Transform frozen (same value for >5s) → AMCL stalled (check particle filter params).
- Transform updating but visually wrong in RViz → AMCL lost localization.

---

**Step 4 — Inspect the global costmap**
Open RViz, add the `/global_costmap/costmap` display.
- **Goal inside lethal cell (red)?** → goal is unreachable. Move the goal.
- **Entire corridor inflated?** → `inflation_radius` too large. Reduce it.
- **Costmap not updating?** → check `/scan` QoS, `update_frequency`, obstacle layer enabled.
- **Static map wrong?** → reload map server or fix the map file.

---

**Step 5 — Inspect the local costmap and controller**
```bash
ros2 topic echo /cmd_vel --once      # is the controller publishing velocity?
ros2 topic echo /local_costmap/costmap --once  # is the local costmap populated?
```
- `cmd_vel` zero → controller not running or goal already considered reached.
- `cmd_vel` non-zero → controller running, but something (e-stop, motor driver) not
  applying velocity.
- Local costmap empty despite obstacles nearby → sensor timestamp issue (check
  `[costmap_2d] scan dropped` warnings).
- Oscillation near goal → check `xy_goal_tolerance`, `min_speed_xy`.

---

**Step 6 — Check TF tree health**
```bash
ros2 run tf2_tools view_frames     # look for disconnected subtrees
ros2 run tf2_ros tf2_echo map base_link  # confirms full chain works
```
TF failures cause silent failures across all layers simultaneously. A single dead TF node
blocks the costmap, planner, and controller all at once. This is the first thing to verify
if multiple layers fail simultaneously.

Also check:
```bash
ros2 node list               # are all expected nodes present?
ros2 node info /nav2_controller  # are all expected topics subscribed?
```

---

**Summary table:**

| Symptom | Most likely cause | Step |
|---|---|---|
| Action not listed | BT navigator not active/crashed | 1 |
| `ComputePathToPose FAILURE` | Goal in wall, inflation too large | 2, 4 |
| `FollowPath FAILURE` near goal | Goal tolerance too tight | 2, 5 |
| Costmap empty | Sensor QoS mismatch, stamp threshold | 4, 5 |
| All layers failing simultaneously | TF tree broken | 6 |
| Robot spinning in place | Yaw goal tolerance, DWB min_speed_xy | 5 |

</details>

- [ ] Done
