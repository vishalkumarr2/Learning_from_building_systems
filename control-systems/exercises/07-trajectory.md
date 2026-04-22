# Exercise 7 — Trajectory Tracking

**Covers:** Lesson 08
**Difficulty:** Hard

---

## Problem 1: Differential Drive Kinematics

An OKS robot has wheel separation $L = 0.4$ m and wheel radius $r = 0.075$ m.

a) The commanded linear velocity is $v = 0.5$ m/s and angular velocity is $\omega = 0.3$ rad/s. Calculate the left and right wheel speeds ($\omega_L$, $\omega_R$) in rad/s.

b) If the left wheel spins at 8 rad/s and the right at 12 rad/s, what are $v$ and $\omega$?

c) What is the maximum linear velocity if each wheel is limited to 15 rad/s? What is the maximum angular velocity while driving straight?

---

## Problem 2: Pure Pursuit Implementation

Implement pure pursuit in Python:

```python
import numpy as np

def pure_pursuit(robot_x, robot_y, robot_theta, 
                 path, lookahead_distance):
    """
    Args:
        robot_x, robot_y: current position (m)
        robot_theta: heading (rad)
        path: Nx2 array of (x, y) waypoints
        lookahead_distance: L_d (m)
    
    Returns:
        linear_vel, angular_vel
    """
    # Your implementation here
    pass
```

Test with a circular path of radius 2.0 m. The robot starts at (2, 0) facing north. Use $L_d = 0.5$ m, $v = 0.3$ m/s.

a) Simulate for 20 seconds. Does the robot follow the circle?

b) Try $L_d = 0.1$ m and $L_d = 2.0$ m. How does lookahead affect tracking?

c) At what speed does the robot start cutting corners on the circle?

---

## Problem 3: Velocity Profiling

A path consists of three segments:
1. Straight line: 3.0 m
2. 90° turn, radius 0.5 m
3. Straight line: 2.0 m

Robot limits: $v_{max} = 0.8$ m/s, $a_{max} = 0.5$ m/s², $\omega_{max} = 1.0$ rad/s.

a) What is the maximum speed through the turn? (Hint: $v = \omega \times r$)

b) Design a velocity profile: compute the speed at every 0.1 m along the path.

c) How far before the turn must the robot start decelerating? Use $v^2 = v_0^2 - 2a \cdot d$.

d) Sketch the velocity profile as a function of distance along path.

---

## Problem 4: Coordinate Frames

The robot is at position $(3.0, 2.0)$ in the map frame, heading $\theta = \pi/4$ (45°).

A waypoint is at $(5.0, 4.0)$ in the map frame.

a) Transform the waypoint to the robot's body frame.

b) What is the cross-track error (lateral distance to the waypoint)?

c) What heading error does the robot have if it should be pointing at the waypoint?

---

## Problem 5: DWB vs Pure Pursuit

Compare DWB (Dynamic Window B) and Pure Pursuit for these scenarios. For each, state which you'd prefer and why:

a) Long straight corridor, 0.8 m/s

b) Tight 180° turn, radius 0.3 m

c) Dynamic obstacle appears 2 m ahead

d) Uneven floor causing wheel slip

e) Very precise docking at a charging station

---

## Solutions

<details>
<summary>Click to reveal solutions</summary>

**Problem 1a:**

$v_R = \frac{v + \omega L/2}{r} = \frac{0.5 + 0.3 \times 0.2}{0.075} = \frac{0.56}{0.075} = 7.47$ rad/s

$v_L = \frac{v - \omega L/2}{r} = \frac{0.5 - 0.06}{0.075} = \frac{0.44}{0.075} = 5.87$ rad/s

**Problem 1b:**

$v = \frac{r(\omega_L + \omega_R)}{2} = \frac{0.075 \times 20}{2} = 0.75$ m/s

$\omega = \frac{r(\omega_R - \omega_L)}{L} = \frac{0.075 \times 4}{0.4} = 0.75$ rad/s

**Problem 3:**

a) $v_{turn} = \omega_{max} \times r = 1.0 \times 0.5 = 0.5$ m/s

c) Deceleration distance: $d = \frac{v_{max}^2 - v_{turn}^2}{2 a_{max}} = \frac{0.64 - 0.25}{1.0} = 0.39$ m.

Start decelerating 0.39 m before the turn begins.

**Problem 4a:**

Translation: $\Delta x = 5-3 = 2$, $\Delta y = 4-2 = 2$

Rotation by $-\theta = -\pi/4$:

$x_{body} = 2\cos(-\pi/4) + 2\sin(-\pi/4) = 1.414 - 1.414 = 0$

$y_{body} = -2\sin(-\pi/4) + 2\cos(-\pi/4) = 1.414 + 1.414 = 2.828$

The waypoint is directly ahead, 2.83 m away.

b) Cross-track error = $|x_{body}| = 0$ m (waypoint is on the centerline).

c) Heading error = $\arctan(x_{body}/y_{body}) = \arctan(0) = 0$ rad. Perfect alignment.

</details>
