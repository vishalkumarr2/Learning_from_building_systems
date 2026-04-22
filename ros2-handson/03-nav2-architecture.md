# 03 — Nav2 Architecture: BehaviorTree, Costmaps, Planners, Controllers
### Why the robot stops, why recovery behaviors trigger, why a path is never found
**Prerequisite:** 01-nodes-topics-actions.md (actions, lifecycle), 02-tf2-time-qos.md (TF2, QoS)
**Unlocks:** Interpreting Nav2 log output, diagnosing stuck navigation, tuning costmap inflation, choosing between planners and controllers, understanding BehaviorTree XML files

---

## Why Should I Care? (Context)

Nav2 is the standard ROS2 navigation stack for mobile robots. Nearly every AMR navigation failure traces back to one of four places:

1. **Localization** — the stack thinks the robot is somewhere it isn't
2. **Costmap** — there's an obstacle in the costmap that doesn't exist in reality (or vice versa)
3. **Planner** — no valid path exists given the current costmap state
4. **BehaviorTree** — the recovery loop ran out of retries and gave up

Without understanding how these four layers interact, navigation bugs look random. With this mental model, you can read a Nav2 log and immediately identify which layer failed and why.

---

# PART 1 — OVERVIEW: THE 4-LAYER STACK

---

## 1.1 The Layered Architecture

```
  ┌─────────────────────────────────────────────────────────┐
  │  Layer 4: BehaviorTree Executor (bt_navigator)          │
  │  Decides WHAT to do and in what sequence                │
  │  "Try planning → try following → try recovery → repeat" │
  └───────────────────────────┬─────────────────────────────┘
                              │ calls actions
  ┌───────────────────────────▼─────────────────────────────┐
  │  Layer 3: BT Action Plugins                             │
  │  ComputePathToPose → nav2_planner                       │
  │  FollowPath        → nav2_controller                    │
  │  ClearEntireCostmap → costmap_2d server                 │
  │  Spin / BackUp / Wait → recovery servers                │
  └───────────────────────────┬─────────────────────────────┘
                              │ reads/writes
  ┌───────────────────────────▼─────────────────────────────┐
  │  Layer 2: Costmap Servers                               │
  │  Global costmap: full-map obstacle grid for planning    │
  │  Local costmap: rolling window for controller           │
  └───────────────────────────┬─────────────────────────────┘
                              │ reads
  ┌───────────────────────────▼─────────────────────────────┐
  │  Layer 1: Localization                                  │
  │  AMCL: map + laser → /amcl_pose                         │
  │  robot_localization EKF: odom + IMU → filtered /odom    │
  └─────────────────────────────────────────────────────────┘
```

Each layer is independent — you can replace AMCL with a different localizer, or swap the planner, without touching the others. The BehaviorTree is the glue that orchestrates them as actions.

---

## 1.2 Nav2 Node Graph (Simplified)

```
                         ┌──────────────┐
  /map ──────────────────► map_server   │
                         └──────┬───────┘
                                │/map (TRANSIENT_LOCAL)
          ┌─────────────────────▼───────────────────────────┐
          │           costmap_2d_global                      │
          │  StaticLayer + ObstacleLayer + InflationLayer    │
          └──────────────────────┬──────────────────────────┘
                                 │/global_costmap/costmap
  /scan ──►┌───────────────────┐ │
           │ costmap_2d_local  │ │
           └────────────┬──────┘ │
                        │/local_costmap/costmap
                        │        │
                ┌───────▼───┐    │
  /odom ───────►│ nav2_     │    │
  /amcl_pose ──►│ controller│    │
                └───────┬───┘    │
                        │        │ ┌────────────┐
                        │        └─► nav2_planner│
                        │          └─────┬───────┘
                     /cmd_vel            │/plan
                        │          ┌─────▼───────┐
                        │          │ bt_navigator │◄── NavigateToPose action
                        └──────────┤             │
                                   └─────────────┘
```

---

# PART 2 — BEHAVIOR TREES

---

## 2.1 What a Behavior Tree IS

A BehaviorTree (BT) is a hierarchical state machine where each node returns one of three values:

```
SUCCESS   — I completed my task
FAILURE   — I could not complete my task
RUNNING   — I am still working (call me again next tick)
```

The tree is "ticked" at a configurable rate (default ~10Hz in Nav2). Each tick propagates from root to leaf, collecting return values.

**Node types:**
```
Sequence      — AND:  tick children left to right. Stop and return FAILURE
                      on first FAILURE child. Return SUCCESS only if ALL succeed.

Fallback      — OR:   tick children left to right. Stop and return SUCCESS
                      on first SUCCESS child. Return FAILURE only if ALL fail.

Decorator     — Modifier: wraps one child and changes its return value
                (Retry N times, Invert, ForceSuccess, RateController...)

Action        — Leaf: calls an actual ROS2 action server. Returns RUNNING
                      while the action is in progress.
```

---

## 2.2 The Default NavigateToPose BehaviorTree

```xml
<BehaviorTree>
  <RecoveryNode number_of_retries="6" name="NavigateRecovery">

    <PipelineSequence name="NavigateWithReplanning">
      <RateController hz="1.0">
        <ComputePathToPose goal="{goal}" path="{path}"/>
      </RateController>
      <FollowPath path="{path}" controller_id="FollowPath"/>
    </PipelineSequence>

    <ReactiveFallback name="RecoveryFallback">
      <GoalUpdated/>
      <SequenceWithMemory name="RecoveryActions">
        <ClearEntireCostmap name="ClearGlobalCostmap-Context"
                            service_name="global_costmap/clear_entirely_global_costmap"/>
        <Spin spin_dist="1.57"/>
        <Wait wait_duration="5.0"/>
        <BackUp backup_dist="0.30" backup_speed="0.05"/>
      </SequenceWithMemory>
    </ReactiveFallback>

  </RecoveryNode>
</BehaviorTree>
```

**Walk-through — what happens when `ComputePathToPose` returns FAILURE:**

```
Tick 1:
  RecoveryNode ticks PipelineSequence
  → RateController ticks ComputePathToPose → FAILURE (no path found)
  → PipelineSequence returns FAILURE
  RecoveryNode tries fallback: ReactiveFallback
  → GoalUpdated? No → FAILURE
  → SequenceWithMemory:
      ClearEntireCostmap → SUCCESS (costmap cleared)
      Spin → RUNNING (still spinning)

Tick 2-N: (Spin still running)
  Spin → RUNNING

Tick N+1: (Spin complete)
  Wait 5s → RUNNING

...eventually:
  BackUp → SUCCESS
  SequenceWithMemory → SUCCESS
  ReactiveFallback → SUCCESS
  RecoveryNode: recovery done, retry PipelineSequence

If this fails 6 times total: RecoveryNode returns FAILURE → navigation aborted.
```

---

## 2.3 PipelineSequence vs Sequence

This is one of the most misunderstood Nav2 internals:

```
Sequence:           Ticks children from the beginning each tick.
                    If child 1 is RUNNING, re-ticks child 1, skips child 2.

PipelineSequence:   Re-ticks ALL previously-RUNNING children each tick.
                    If child 1 returns SUCCESS, also ticks child 2 on same tick.
                    Effect: planner and controller run concurrently.
```

```
PipelineSequence during navigation:
  Tick 1:  RateController(ComputePathToPose) → SUCCESS   ← path computed
           FollowPath → RUNNING                          ← start following
  Tick 2:  RateController: 1Hz, skip this tick
           FollowPath → RUNNING                          ← still following
  Tick 3:  RateController: 1Hz, skip
           FollowPath → RUNNING
  ...
  Tick 10: RateController: 1Hz, replan now
           ComputePathToPose → SUCCESS                   ← updated path
           FollowPath → RUNNING (receives updated path)
```

**Key insight:** The `RateController hz="1.0"` causes the path to be replanned every second, not every tick. This prevents the planner from consuming CPU on every 10Hz tick.

---

## 2.4 The "Robot Stuck in Recovery Loop" Pattern

```
Robot behavior:                        BT state:
  spinning...                            Spin → RUNNING
  waiting...                             Wait → RUNNING
  backing up...                          BackUp → RUNNING
  trying to plan...                      ComputePathToPose → FAILURE
  spinning again...                      (retry 2 of 6)
  ...
  "Goal aborted"                         RecoveryNode: retries exhausted
```

**Root cause checklist:**
1. Is there actually a valid path? Check if the goal pose is inside an obstacle in the global costmap.
2. Is the global costmap showing a phantom obstacle? Old sensor data, mis-configured inflation radius.
3. Is the robot's localization correct? If AMCL has diverged, the costmap obstacle locations don't match reality.
4. Is `inflation_radius` too large for the corridor width? Robot inflation footprint blocks its own path.

---

# PART 3 — COSTMAPS

---

## 3.1 Costmap Values

```
Cost value  Meaning
──────────────────────────────────────────────────────────
  0          FREE — robot can be here
  1–252      INFLATED — near an obstacle; higher = closer
  253        INSCRIBED — robot center would touch an obstacle
  254        LETHAL — obstacle cell; robot cannot be here
  255        UNKNOWN — not yet observed
```

The planner's job: find a path through cells with cost < 253. The controller's job: follow that path while the local costmap stays clear.

---

## 3.2 Costmap Layers

```
Final costmap (master)
    │
    ├── StaticLayer     ← reads from /map (the pre-built occupancy grid)
    │                     FREE cells in map = 0, OCCUPIED = 254
    │
    ├── ObstacleLayer   ← reads from /scan, /depth/points
    │                     marks new obstacles seen by sensors
    │                     can clear cells when sensor sees free space
    │
    └── InflationLayer  ← expands all obstacles outward by inflation_radius
                          creates cost gradient (253 at robot_radius, 1 at inflation_radius)
```

```
Before InflationLayer:           After InflationLayer:
                                  inflation_radius = 0.5m
  ░░░░░░░░░░░░░░░░                ░░░░░░░░░░░░░░░░
  ░░░░░░░░░░░░░░░░                ░░1 1 1 1 1░░░░░
  ░░░░░████░░░░░░░                ░░1 ╔═════╗1░░░░
  ░░░░░████░░░░░░░   ────────►    ░░1 ║     ║1░░░░
  ░░░░░████░░░░░░░                ░░1 ║wall ║1░░░░
  ░░░░░░░░░░░░░░░░                ░░1 ╚═════╝1░░░░
  ░░░░░░░░░░░░░░░░                ░░1 1 1 1 1░░░░░
                                  ░░░░░░░░░░░░░░░░

  ░ = FREE (0)                    Numbers = cost gradient (1–252)
  █ = LETHAL (254)                ╔╗╚╝ = INSCRIBED/LETHAL border
```

---

## 3.3 Global vs Local Costmap

| Property | Global Costmap | Local Costmap |
|---|---|---|
| Size | Full map (e.g., 50m × 30m) | Rolling window around robot (e.g., 4m × 4m) |
| Update rate | Slow (~1Hz) | Fast (~5–10Hz, matches sensor rate) |
| Layers | Static + Obstacle + Inflation | Obstacle + Inflation only (no static) |
| Purpose | Path planning | Reactive collision avoidance |
| Frame | `map` frame | Robot-centered (`odom` or `base_link`) |

**Key insight:** The local costmap does NOT use the static map layer. It only sees what the sensors currently see. This is why the robot can drive through a previously-mapped obstacle if the sensor isn't observing it right now.

---

## 3.4 inflation_radius Tuning

```
Too small (e.g., exactly robot_radius):
  Robot path hugs walls → robot scrapes obstacles
  Paths through narrow corridors may work but margins are zero

Too large (e.g., 1.5× corridor width):
  Inflation from both walls overlaps → entire corridor is high-cost
  Planner may declare NO PATH even though the robot could fit
  Robot refuses to enter narrow sections

Correct:
  inflation_radius = robot_radius + safety_margin
  safety_margin = 0.1–0.3m depending on robot speed and controller accuracy
```

**Diagnosis:** If the planner reports "no path found" but the corridor is clearly passable, reduce `inflation_radius` or check that the `robot_radius` parameter matches the actual robot footprint.

---

# PART 4 — PLANNERS AND CONTROLLERS

---

## 4.1 The Planner: Global Path Planning

The planner receives a goal pose and the global costmap, and returns a path (a list of poses from current position to goal).

**NavFn Planner** (Dijkstra / A*):
```
Strengths:  Reliable, well-tested, handles most warehouse maps
Weaknesses: Doesn't account for robot kinematics (can produce paths a
            differential-drive robot can't physically follow)
Parameters: use_astar: true/false, allow_unknown: true/false
```

**SmacPlanner (SE2 or Hybrid-A*)**:
```
Strengths:  Kinematically-feasible paths (respects min turn radius)
            Better for non-holonomic robots in tight spaces
Weaknesses: Slower than NavFn, more parameters to tune
Parameters: minimum_turning_radius, motion_model_for_search: DUBIN/REEDS_SHEPP
```

**When path planning fails:**
```bash
# Check planner logs:
ros2 topic echo /plan       # should publish a path
ros2 topic echo /planning_server/goal_status

# Common log messages:
# [nav2_planner] Failed to create a plan from X to Y
#   → goal inside obstacle, or costmap not initialized
# [nav2_planner] Timeout exceeded
#   → map too large, NavFn too slow, reduce resolution or switch to SmacPlanner
```

---

## 4.2 The Controller: Local Path Following

The controller runs at ~20Hz, receives the planned path + local costmap, and outputs `/cmd_vel`.

### DWB (Dynamic Window Approach)

```
Algorithm:
  1. Sample N velocity commands (vx, vθ) within the dynamic window
     (window = what's kinematically reachable in the next timestep)
  2. Project each sampled velocity forward for T seconds → trajectory
  3. Score each trajectory with multiple critics
  4. Send the velocity with the best total score

Trajectory Critics:
  PathAlign    — stay close to the planned path heading
  PathDist     — stay close to the planned path laterally
  GoalAlign    — point toward goal when close
  ObstacleDist — penalize trajectories near obstacles

Key parameters:
  max_vel_x: 0.5         # max forward speed (m/s)
  min_vel_x: -0.1        # allows slight backward motion
  max_vel_theta: 1.0     # max angular velocity (rad/s)
  vx_samples: 20         # number of vx samples → affects CPU
  vtheta_samples: 20     # number of vθ samples
```

**DWB failure modes:**
- Robot oscillates: `PathAlign` weight too high, fighting `PathDist`. Balance the weights.
- Robot stops near obstacles: `ObstacleDist` too aggressive. Reduce `obstacle_scale`.
- Robot can't rotate in place: `min_speed_xy` > 0. Set to 0.0 for diff-drive.

### RPP (Regulated Pure Pursuit)

```
Algorithm:
  1. Find the "lookahead point" on the path at lookahead_dist ahead
  2. Compute the curvature (1/radius) to reach that point
  3. Scale linear speed by curvature (slow down on curves)
     and by proximity to obstacles (slow down near obstacles)
  4. Output (vx, vθ)

Key parameters:
  desired_linear_vel: 0.5    # target forward speed (m/s)
  lookahead_dist: 0.6        # how far ahead to look on the path (m)
  max_angular_accel: 3.2     # rad/s²
  use_cost_regulated_linear_velocity_scaling: true  # slow near obstacles
  regulated_linear_scaling_min_speed: 0.25          # never go below this
```

**RPP failure modes:**
- Robot overshoots corners: increase `lookahead_dist` or decrease speed.
- Robot cuts corners into obstacles: decrease `lookahead_dist`.
- Robot stops short of goal: `goal_dist_tol` too large. Check `xy_goal_tolerance`.

---

## 4.3 DWB vs RPP: Decision Guide

```
Use RPP when:                           Use DWB when:
  Wide, clear corridors                   Cluttered environments
  Industrial/warehouse predictable paths  Dynamic obstacles
  Simpler parameter tuning needed         Need fine-grained trajectory scoring
  Differential drive, stable odometry     Need custom trajectory critics
  Speed regulation near obstacles         Multi-objective optimization needed
```

**Starting point for a warehouse AMR:** RPP with `desired_linear_vel: 0.5`, `lookahead_dist: 0.6`. Tune from there.

---

## 4.4 Controller Failure Modes

```bash
# Controller stopped publishing /cmd_vel:
ros2 topic hz /cmd_vel         # should be ~20Hz when navigating

# Common log messages:
# [nav2_controller] Could not find any valid trajectories
#   → DWB: all sampled trajectories are in collision
#   → Fix: check local costmap for phantom obstacles
# [nav2_controller] Goal not reached within tolerance
#   → Robot is circling near goal but not converging
#   → Fix: increase xy_goal_tolerance or check localization accuracy
# [nav2_controller] Exceeded time tolerance
#   → Robot took too long to reach goal (e.g., blocked by real obstacle)
```

---

# PART 5 — ROBOT LOCALIZATION (EKF NODE)

---

## 5.1 What robot_localization Does

The `robot_localization` package provides an Extended Kalman Filter (EKF) that fuses multiple odometry sources into a single, smooth `/odom` output.

```
Sensor inputs:
  /wheel_odom  (nav_msgs/Odometry)   ← from motor controller
  /imu/data    (sensor_msgs/Imu)     ← from IMU

                    ┌──────────────┐
  /wheel_odom ─────►│              │
                    │  EKF node    ├──► /odometry/filtered  (smooth /odom)
  /imu/data ───────►│              │      └──► TF: odom → base_link
                    └──────────────┘
```

**Why bother with EKF if you already have wheel odometry?**
- Wheel odometry alone: noisy, susceptible to wheel slip
- IMU alone: drifts (gyro bias accumulates)
- EKF fusion: IMU provides fast, high-frequency correction; wheels provide long-term position

---

## 5.2 Key Parameters

```yaml
ekf_node:
  ros__parameters:
    frequency: 50.0                  # EKF update rate (Hz)
    two_d_mode: true                 # robot moves in 2D plane only

    # Which states each sensor provides (x, y, z, roll, pitch, yaw, vx, vy, vz, vroll, vpitch, vyaw, ax, ay, az)
    odom0: /wheel_odom
    odom0_config: [true,  true,  false,   # x, y, z
                   false, false, true,    # roll, pitch, yaw
                   true,  false, false,   # vx, vy, vz
                   false, false, true,    # vroll, vpitch, vyaw
                   false, false, false]   # ax, ay, az

    imu0: /imu/data
    imu0_config: [false, false, false,
                  true,  true,  true,    # fuse orientation from IMU
                  false, false, false,
                  true,  true,  true,    # fuse angular velocity from IMU
                  true,  false, false]   # fuse linear accel X only

    process_noise_covariance: [...]      # how much to trust the process model
    initial_estimate_covariance: [...]   # initial uncertainty
```

---

## 5.3 AMCL: Map-Based Localization

AMCL (Adaptive Monte Carlo Localization) uses a particle filter to localize the robot against a known map.

```
Input:   /scan  (laser measurement)
Input:   /map   (pre-built occupancy grid from SLAM)
Input:   /initialpose (initial guess, or use /tf odom→base)
Output:  /amcl_pose  (geometry_msgs/PoseWithCovarianceStamped)
Output:  TF: map → odom  (the localization correction)
```

```
Particle filter intuition:
  100 particles spread around initial pose
    │
    ▼ Robot moves (odom update)
  All particles moved according to motion model
    │
    ▼ Robot observes scan
  Particles consistent with scan get HIGH weight
  Particles inconsistent with scan get LOW weight
    │
    ▼ Resample: high-weight particles survive, low-weight die
  Particles converge on true pose
```

**AMCL failure modes:**

```bash
# Localization diverged (particles spread all over map):
# - Robot was lifted and moved (external intervention)
# - Wheel slip caused large odometry error → particles fell off the true pose
# Fix: publish a new /initialpose from rviz2 (2D Pose Estimate button)

# AMCL not converging:
# - Global localization needed: ros2 service call /reinitialize_global_localization
# - scan → map mismatch: check if map matches current environment

# Common log message:
# [amcl] Particle filter variance too high
#   → Localization is uncertain; Nav2 may still navigate but with degraded accuracy
```

---

## 5.4 Localization Failure → Replanning Loop

```
Failure chain:
  robot_localization EKF publishes stale odom
       │
       ▼
  TF odom→base_link not updating at full rate
       │
       ▼
  Nav2 controller believes robot hasn't moved
       │
       ▼
  Controller keeps commanding velocity to "catch up"
       │
       ▼
  Robot moves faster than intended → overshoots goal
  OR: replanning loop (controller sees progress=zero → BT triggers recovery)

Diagnosis:
  ros2 topic hz /odometry/filtered    # should be ~50Hz
  ros2 topic hz /tf                   # should reflect odom publisher rate

  Low rate or gaps → EKF is stalling → check IMU and wheel odom topics
```

---

## Summary — What to Remember

| Nav2 Component | Purpose | Key Failure Mode |
|---|---|---|
| BehaviorTree executor (`bt_navigator`) | Orchestrates planning, following, recovery in a tree structure | Recovery loop exhausts retries → goal aborted |
| `PipelineSequence` | Runs planner + controller concurrently; replanner fires at `RateController` hz | Confused with `Sequence`; results in either over-replanning or never replanning |
| `RecoveryNode` | Retries the navigation sequence N times after recovery actions | Wrong `number_of_retries`; robot gives up too early or loops forever |
| Global costmap | Full-map obstacle grid for path planning | Stale obstacle from old sensor data → path planning fails; fix with `ClearEntireCostmap` |
| Local costmap | Rolling window for reactive control | Phantom obstacle → controller stops → BT triggers recovery |
| `inflation_radius` | Expands obstacles to keep robot body clear | Too large → robot can't enter corridors; too small → robot scrapes walls |
| StaticLayer | Loads pre-built map into costmap | Not loaded (TRANSIENT_LOCAL QoS mismatch) → costmap shows only sensor obstacles |
| ObstacleLayer | Adds real-time sensor obstacles to costmap | Not updating (scan QoS mismatch or sensor driver down) → robot drives into real obstacles |
| NavFn planner | Dijkstra/A* global path | Slow on large maps; ignores robot kinematics |
| SmacPlanner | Hybrid-A* kinematically feasible path | Slower; needs `minimum_turning_radius` tuned |
| DWB controller | Samples trajectories, scores with critics | CPU heavy; oscillates if `PathAlign` weight unbalanced |
| RPP controller | Regulated pure pursuit | Overshoots corners if `lookahead_dist` too large |
| `robot_localization` EKF | Fuses wheel odom + IMU → smooth `/odom` | Stale odom if IMU stops; replanning loop if `/odom` gaps |
| AMCL | Particle filter localization against known map | Diverges after wheel slip or robot being moved; fix with `/initialpose` |
