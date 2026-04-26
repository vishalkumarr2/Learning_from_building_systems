# 05 — Holonomic vs Non-Holonomic vs Underactuated

## Why some robots can move directly sideways and others must maneuver

This page exists because many planning and control interviews quietly test one deeper idea:

> Is the robot free to move in every local direction, or is its motion constrained by its current heading and actuation?

That distinction explains why some robots can follow a path like a cursor on a screen, while others must swing, reverse, or re-approach to reach the same pose.

---

## 1. Mathematical definition

Let the robot configuration be:

$$
q \in \mathbb{R}^n
$$

### Holonomic

A robot is **holonomic** if its constraints can be expressed as configuration-only constraints, so the allowed motion directions match the independent coordinates of its configuration space.

In practical robotics language, that means:

> The robot can move instantaneously in every independent local direction that its configuration space allows.

For a planar omnidirectional robot, you can model:

$$
q = (x, y, \theta)
$$

with directly commanded body motion:

$$
\dot{x} = u_x, \quad \dot{y} = u_y, \quad \dot{\theta} = u_\theta
$$

Here the robot can translate in $x$, translate in $y$, and rotate independently.

### Non-holonomic

A robot is **non-holonomic** if it has velocity constraints that cannot be integrated into pure position constraints.

These are often written as:

$$
A(q)\dot{q} = 0
$$

If those constraints cannot be converted into a configuration-only condition like:

$$
h(q) = 0
$$

then they are non-holonomic.

For a unicycle or differential-drive robot:

$$
q = (x, y, \theta)
$$

$$
\dot{x} = v\cos\theta
$$

$$
\dot{y} = v\sin\theta
$$

$$
\dot{\theta} = \omega
$$

This implies the sideways-velocity constraint:

$$
-\sin\theta\,\dot{x} + \cos\theta\,\dot{y} = 0
$$

That equation means the robot cannot move sideways instantaneously.

### Underactuated

A robot is **underactuated** when it has fewer independent control inputs than the full number of configuration or state variables that matter for motion.

That is a different question from holonomy.

The common interview-safe summary is:

- **Holonomic / non-holonomic** asks: what motion directions are locally feasible?
- **Underactuated** asks: how many independent controls do I have relative to the motion I want to control?

A quadrotor is the classic example: it is usually discussed as **underactuated**, not simply as a holonomic mobile robot.

---

## 2. ELI5 intuition

### Holonomic robot

Think of a robot that moves like a computer game character on ice-free tiles:

- forward
- backward
- left
- right
- rotate

If it needs to shift a little to the left, it just does it.

### Non-holonomic robot

Now think of a real car.

It can:

- go forward
- go backward
- turn

But it cannot do this:

> "Move 20 cm directly sideways into the next parking spot right now."

It has to maneuver.

### One-line kid version

- **Holonomic**: "I can go directly where I want."
- **Non-holonomic**: "I may get there, but I have to steer my way there."

---

## 3. Real-world use cases

### Holonomic robots

- **Mecanum AMRs** in factories where sideways docking is useful.
- **Omni-wheel service robots** in labs, hospitals, or demos.
- **Cartesian gantries** where motion is independently commanded along axes.

### Non-holonomic robots

- **Differential-drive warehouse AMRs** because they are simple and robust.
- **Forklifts** because approach angle and reversing matter.
- **Autonomous cars** because steering limits define feasible motion.
- **Tuggers and AGVs with trailers** because turning constraints dominate maneuvering.

### Underactuated systems

- **Quadrotors** because thrust and attitude are controlled indirectly rather than commanding arbitrary translational motion directly.
- **Fixed-wing aircraft** because they cannot strafe like a holonomic robot.
- **Acrobots / pendulum systems** because not every joint is directly actuated.

---

## 4. Why this matters in planning

If the robot is approximately holonomic in the plane, a planner can often think more simply:

- free space on the map
- path length
- obstacle avoidance

If the robot is non-holonomic, the planner must also care about:

- heading
- curvature
- turning radius
- reverse policy

That is why a 2D costmap path is not always a drivable path.

This is exactly the intuition behind:

- **NavFn** searching mostly in `(x, y)`
- **Smac / Hybrid-A\*** searching in `(x, y, \theta)`

**Related reading:** [04-navfn-vs-smac-search-spaces.md](04-navfn-vs-smac-search-spaces.md)

---

## 5. Comparison table

| Type | Core idea | Mathematical clue | Sideways motion right now? | Typical example | Planning implication |
| --- | --- | --- | --- | --- | --- |
| Holonomic | Can move directly in each independent local direction | Direct local velocity in all relevant DOFs | Usually yes, if that DOF exists | Omni-wheel base | Geometry-first planning often works |
| Non-holonomic | Has non-integrable velocity constraints | `A(q)\dot{q} = 0` | No | Differential drive, car, forklift | Heading and curvature matter |
| Underactuated | Has fewer independent controls than motion variables of interest | Fewer independent inputs than desired motion authority | Not the defining test | Quadrotor, fixed-wing aircraft | Dynamics and control coupling matter |

---

## 6. Tricky interview questions

### 1. Is every wheeled robot non-holonomic?

No. Differential-drive and car-like robots are non-holonomic, but mecanum and omni-wheel robots are often modeled as holonomic in the plane.

### 2. If a non-holonomic robot can still reach the goal, why call it constrained?

Because the constraint is about **instantaneous feasible velocity directions**, not necessarily eventual reachability.

### 3. Why does heading matter so much for non-holonomic robots?

Because heading determines which local velocities are physically feasible.

### 4. Is underactuated the same as non-holonomic?

No. Underactuation is about missing independent control inputs. Non-holonomy is about motion constraints that forbid arbitrary instantaneous directions.

### 5. Is a quadrotor holonomic?

In full dynamics, the better answer is that it is **underactuated**. In simplified planning layers, people may sometimes approximate its position control as if it were free in 3D.

### 6. Give the cleanest one-sentence distinction

A holonomic robot can move instantaneously in every allowed local direction; a non-holonomic robot must maneuver because some velocity directions are forbidden.

---

## 7. Practical takeaway

When someone says a path is valid, ask:

1. Valid in the map?
2. Valid for the robot's heading and turning radius?
3. Valid under the platform's actuation limits?

That is the bridge from geometry to real robot motion.

**Companion exercise:** [Exercise08 — Holonomic, Non-Holonomic, and Underactuated Interview Traps](exercises/08-holonomic-non-holonomic-underactuated-interview-traps.md)