# 08 — Trajectory Tracking and Path Following
### From Nav2's cmd_vel to smooth wheel motion — the Jetson layer

**Prerequisite:** `07-cascaded-loops.md` (cascaded PID on MCU)
**Unlocks:** `09-two-layer-architecture.md` (MCU↔Jetson coordination)

---

## Why Should I Care? (Practical Context)

The MCU cascade (Lessons 05-07) controls individual motors. But the warehouse robot has TWO driven wheels (differential drive). The **Jetson** runs ROS2 Nav2, which plans paths and outputs `cmd_vel` — a linear+angular velocity command. Something must translate `cmd_vel` into **left and right wheel speed commands** and ensure the robot follows the planned path accurately.

**This is where 80% of "the robot drifts" bugs live:**
- Pure pursuit lookahead too short → oscillation around the path
- DWB max_vel_theta too high → aggressive rotation → line-sensor triggers
- No velocity profiling → jerky starts and stops → load shifts

---

# PART 1 — DIFFERENTIAL DRIVE KINEMATICS

## 1.1 Forward Kinematics

A differential drive robot with wheel radius $r$ and wheelbase (distance between wheels) $L$:

```
          ↑ x (forward)
          │
     ┌────┼────┐
     │    │    │
  ωL │    ●    │ ωR      ● = center of rotation
     │    │    │
     └────┼────┘
          │
     ←─ L ─→
```

Given left wheel speed $\omega_L$ and right wheel speed $\omega_R$:

$$v = \frac{r(\omega_R + \omega_L)}{2} \quad \text{(linear velocity)}$$

$$\omega = \frac{r(\omega_R - \omega_L)}{L} \quad \text{(angular velocity)}$$

## 1.2 Inverse Kinematics

Given desired $v$ and $\omega$ (from Nav2 `cmd_vel`):

$$\omega_L = \frac{v - \omega L/2}{r}$$

$$\omega_R = \frac{v + \omega L/2}{r}$$

**AMR parameters:** $r = 0.065$ m, $L = 0.43$ m

```python
# Inverse kinematics: cmd_vel → wheel speeds
def cmd_vel_to_wheels(v, omega, wheel_radius=0.065, wheelbase=0.43):
    omega_left = (v - omega * wheelbase / 2.0) / wheel_radius
    omega_right = (v + omega * wheelbase / 2.0) / wheel_radius
    return omega_left, omega_right

# Example: v = 0.5 m/s, turning at 0.2 rad/s
wL, wR = cmd_vel_to_wheels(0.5, 0.2)
# wL = (0.5 - 0.2*0.215) / 0.065 = 7.03 rad/s
# wR = (0.5 + 0.2*0.215) / 0.065 = 8.35 rad/s
```

## 1.3 Odometry (Dead Reckoning)

Integrating wheel speeds gives position estimate:

$$\Delta x = v \cdot \cos(\theta) \cdot \Delta t$$
$$\Delta y = v \cdot \sin(\theta) \cdot \Delta t$$
$$\Delta \theta = \omega \cdot \Delta t$$

**AMR odometry runs at 50 Hz on the Jetson**, fusing encoder feedback with IMU data in the navigation estimator.

**Why odometry drifts:** Wheel slip, wheel radius errors, wheelbase calibration errors. After 100 m of driving, pure odometry can be off by 1-3 m. That's why AMR uses LiDAR AMCL or fusion estimators.

---

# PART 2 — PATH FOLLOWING ALGORITHMS

## 2.1 Pure Pursuit

**Idea:** Pick a "lookahead point" on the path at distance $L_d$ ahead of the robot, and steer toward it.

```
Path:  ─────────●─────────────
                 │ L_d (lookahead distance)
                 │
                ╔╗  ← Robot
                ╚╝
```

The required curvature to reach the lookahead point:

$$\kappa = \frac{2 \sin(\alpha)}{L_d}$$

where $\alpha$ is the angle between the robot's heading and the lookahead point.

Angular velocity command:

$$\omega = v \cdot \kappa = \frac{2v \sin(\alpha)}{L_d}$$

```python
import numpy as np

def pure_pursuit(robot_x, robot_y, robot_theta, path, lookahead_dist, velocity):
    """
    Compute cmd_vel using pure pursuit.
    Returns: (v, omega)
    """
    # Find lookahead point on path
    lookahead_point = find_lookahead_point(robot_x, robot_y, path, lookahead_dist)
    if lookahead_point is None:
        return 0.0, 0.0  # No valid point → stop
    
    # Transform lookahead point to robot frame
    dx = lookahead_point[0] - robot_x
    dy = lookahead_point[1] - robot_y
    
    # Angle to lookahead point relative to robot heading
    alpha = np.arctan2(dy, dx) - robot_theta
    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))  # Normalize to [-π, π]
    
    # Curvature
    curvature = 2.0 * np.sin(alpha) / lookahead_dist
    
    # Cmd_vel
    omega = velocity * curvature
    return velocity, omega
```

**Tuning $L_d$ (lookahead distance):**

| $L_d$ | Effect | AMR usage |
|-------|--------|-----------|
| Small (0.2 m) | Tracks path tightly, oscillates at high speed | Docking, precision approach |
| Medium (0.5 m) | Good balance | Normal corridor driving |
| Large (1.0 m) | Smooth but cuts corners | High-speed transit |

**AMR uses regulated pure pursuit** (Nav2's RPP plugin), which adaptively adjusts $L_d$ based on curvature and speed.

## 2.2 Stanley Controller

**Idea:** Control both the **heading error** and the **cross-track error** (distance from path).

$$\delta = \theta_e + \arctan\left(\frac{k \cdot e_{cross}}{v}\right)$$

where:
- $\theta_e$ = heading error (angle between robot heading and path tangent)
- $e_{cross}$ = cross-track error (perpendicular distance to path)
- $k$ = gain parameter
- $v$ = forward velocity

**Advantage:** Converges to the path exponentially. **Disadvantage:** Assumes bicycle model (front-wheel steering), needs adaptation for differential drive.

## 2.3 DWB (Dynamic Window Based) — Nav2's Default

The DWB planner samples many (v, ω) pairs in the robot's velocity space, simulates each trajectory forward, and picks the one that:
1. Avoids obstacles
2. Stays close to the global path
3. Makes progress toward the goal
4. Is kinematically feasible (respects max acceleration)

```
Velocity space:
  ω ↑
    │  ╳ ╳ ╳ ╳         ╳ = infeasible (obstacle collision)
    │  ╳ ○ ○ ╳         ○ = feasible
    │  ╳ ○ ★ ╳         ★ = best scored trajectory
    │  ╳ ○ ○ ╳
    └──────────→ v
    
  ★ is selected → becomes cmd_vel
```

**Key DWB parameters (AMR tuning):**

| Parameter | Value | Effect |
|-----------|-------|--------|
| `max_vel_x` | 0.5 m/s | Max forward speed |
| `max_vel_theta` | 0.5 rad/s | Max rotation speed |
| `max_accel_x` | 0.3 m/s² | Acceleration limit → smooth starts |
| `max_accel_theta` | 0.5 rad/s² | Angular accel → prevents jerky turns |
| `sim_time` | 2.0 s | How far ahead to simulate |
| `vx_samples` | 10 | Velocity space resolution |
| `vtheta_samples` | 20 | Angular vel resolution |

**Common AMR DWB bugs:**
- `max_vel_theta` too high → robot rotates aggressively → triggers line-sensor safety → e-stop
- `sim_time` too short → robot doesn't see obstacles ahead → last-second swerve
- `max_accel_x` too high → wheels slip on smooth floor → odometry error → position drift

---

# PART 3 — VELOCITY PROFILING

## 3.1 The Problem

If the Nav2 planner outputs a raw path (sequence of waypoints), and the robot tries to follow it at constant speed, what happens at corners?

**Without profiling:** Robot hits corner at full speed → needs impossible angular acceleration → either misses the turn or jerks violently.

**With profiling:** Plan a velocity profile that decelerates before corners, stops at the goal, and respects acceleration limits.

## 3.2 Trapezoidal Velocity Profile

The simplest: accelerate, cruise, decelerate.

```
Speed
  ▲
  │     ┌───────────────┐
  │    /│               │\
  │   / │               │ \
  │  /  │               │  \
  │ /   │               │   \
  └─────┴───────────────┴─────→ Time
    acc    cruise         dec
```

$$d_{accel} = \frac{v_{max}^2}{2a_{max}}$$

If the total distance is too short for full speed, use a triangular profile (accelerate then immediately decelerate).

```python
def trapezoidal_profile(distance, v_max, a_max):
    """Generate trapezoidal velocity profile."""
    # Distance needed to accelerate to v_max and decelerate
    d_accel = v_max**2 / (2 * a_max)
    
    if 2 * d_accel >= distance:
        # Triangular profile: can't reach v_max
        v_peak = (a_max * distance) ** 0.5
        t_accel = v_peak / a_max
        return {
            'type': 'triangular',
            'v_peak': v_peak,
            't_accel': t_accel,
            't_total': 2 * t_accel
        }
    else:
        # Trapezoidal: full cruise segment
        d_cruise = distance - 2 * d_accel
        t_accel = v_max / a_max
        t_cruise = d_cruise / v_max
        return {
            'type': 'trapezoidal',
            'v_max': v_max,
            't_accel': t_accel,
            't_cruise': t_cruise,
            't_total': 2 * t_accel + t_cruise
        }
```

## 3.3 S-Curve Profile (Smoother)

Trapezoidal profiles have discontinuous acceleration → jerky motion. S-curve profiles limit jerk (rate of change of acceleration):

```
Acceleration
  ▲
  │  ╱─────╲
  │ ╱       ╲
  ├─╱─────────╲─→ Time
  │             ╲       ╱
  │              ╲─────╱
  ▼
```

**AMR uses S-curve for docking** (final 0.5 m approach) because precision matters more than speed. For corridor driving, trapezoidal is sufficient.

---

# PART 4 — FROM CMD_VEL TO WHEEL COMMANDS

## 4.1 The Complete Pipeline on Jetson

```
Nav2 Planner → Global Path
       │
       ▼
Nav2 Controller (RPP/DWB) → cmd_vel (v, ω) at 20 Hz
       │
       ▼
Velocity Smoother → smoothed cmd_vel (acceleration-limited)
       │
       ▼
Inverse Kinematics → (ω_left, ω_right)
       │
       ▼
SPI → MCU → cascaded PID → motor PWM
```

## 4.2 Velocity Smoother

Nav2's velocity smoother enforces acceleration and deceleration limits on `cmd_vel`:

```python
class VelocitySmoother:
    def __init__(self, max_accel, max_decel, dt):
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.dt = dt
        self.current_v = 0.0
        self.current_omega = 0.0
    
    def smooth(self, target_v, target_omega):
        # Linear velocity
        dv = target_v - self.current_v
        max_dv = self.max_accel * self.dt if dv > 0 else self.max_decel * self.dt
        dv = max(min(dv, max_dv), -max_dv)
        self.current_v += dv
        
        # Angular velocity (same logic)
        dw = target_omega - self.current_omega
        max_dw = self.max_accel * self.dt  # Angular accel limit
        dw = max(min(dw, max_dw), -max_dw)
        self.current_omega += dw
        
        return self.current_v, self.current_omega
```

## 4.3 Handling the Zero-Velocity Dead Zone

At very low speeds, the motor has static friction (stiction). The PID commands a tiny duty, but the motor doesn't move. The position error grows, integral winds up, and eventually the motor lurches forward.

**Solutions used in our robot:**
1. **Minimum speed threshold:** If $|v_{cmd}| < v_{min}$, clamp to 0 (accept position holding, not creeping)
2. **Feedforward dither:** Add a small oscillating signal to overcome stiction
3. **Friction compensation:** Measure the static friction torque and add it as feedforward

---

# PART 5 — COORDINATE FRAMES AND TRANSFORMS

## 5.1 Frames in our robot

```
map → odom → base_link → wheel_left, wheel_right
 │      │        │
 │      │        └── Robot center (between wheels, on ground)
 │      └── Odometry frame (continuous, drift-allowed)
 └── Map frame (corrected by AMCL/localization)
```

**All cmd_vel is in `base_link` frame:** $v$ is forward, $\omega$ is counter-clockwise viewed from above.

## 5.2 Transform Timing Matters

The path following controller needs to know "where is the robot now?" to compute errors. This requires a TF lookup from `map` → `base_link`. If this transform is **stale** (the TF hasn't been updated recently), the controller uses an outdated position and makes wrong corrections.

**AMR TF staleness budget:**
- Acceptable: < 50 ms
- Warning: 50–100 ms → reduced speed mode
- Critical: > 100 ms → stop and wait

---

## Checkpoint Questions

1. Derive the inverse kinematics for a differential drive: given (v, ω), find (ω_left, ω_right).
2. In pure pursuit, what happens to tracking if $L_d$ is set too small? Too large?
3. Why does DWB sample many (v, ω) pairs instead of just computing the optimal one analytically?
4. A trapezoidal profile has a total distance of 0.5 m, $v_{max} = 0.5$ m/s, $a_{max} = 0.3$ m/s². Is this trapezoidal or triangular?
5. The velocity smoother limits acceleration to 0.3 m/s². Nav2 commands an instant jump from 0 to 0.5 m/s. How long does the smoother take to reach 0.5 m/s?
6. Why does AMR use pure pursuit for docking instead of DWB?
7. The robot drifts left even though cmd_vel is straight (v=0.5, ω=0). List three possible causes.

---

## Key Takeaways

- **Differential drive inverse kinematics** converts (v, ω) → (ω_left, ω_right)
- **Pure pursuit** is simple and effective; lookahead distance $L_d$ is the key tuning knob
- **DWB** is more sophisticated (obstacle avoidance, multi-objective scoring) but harder to tune
- **Velocity profiling** prevents impossible acceleration demands at corners and stops
- **The pipeline:** Nav2 path → controller → velocity smoother → inverse kinematics → SPI → MCU PID
- **Stale transforms** cause path-tracking errors — timing matters as much as algorithm choice
