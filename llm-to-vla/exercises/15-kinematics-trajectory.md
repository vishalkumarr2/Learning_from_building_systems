# Exercise 15 — Forward/Inverse Kinematics & Trajectory Planning
> Phase VI–VII · Days 93-95 · ~8 hours · **Robotics-specific**

[← Exercise 14: Data Collection](14-data-collection-teleoperation.md)

---

## Objectives

By completing this exercise you will:
- Implement forward kinematics using DH parameters from scratch
- Solve inverse kinematics with Jacobian pseudoinverse and damped least-squares
- Understand joint-space vs task-space trajectory representations
- Generate smooth trajectories with velocity and acceleration limits
- Implement cubic and quintic polynomial interpolation
- Connect kinematics to VLA action spaces (why action space choice matters)

## Prerequisites
- Exercise 12 (MuJoCo manipulation — Jacobian-based control)
- Linear algebra fundamentals (matrix exponentials, SVD)
- Study Note 13 (Imitation Learning — action space discussion)

## Setup

```bash
pip install numpy matplotlib scipy mujoco
```

> **Note**: This exercise is pure Python + NumPy for understanding.
> In production, use libraries like `pinocchio`, `ikfast`, or MoveIt 2.

---

## Part 1: Forward Kinematics from DH Parameters (~2 hours)

### 1.1 — Denavit-Hartenberg Convention

Every serial robot can be described by 4 parameters per joint:

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| Link length | $a_i$ | Distance along $x_i$ from $z_{i-1}$ to $z_i$ |
| Link twist | $\alpha_i$ | Angle about $x_i$ from $z_{i-1}$ to $z_i$ |
| Link offset | $d_i$ | Distance along $z_{i-1}$ from $x_{i-1}$ to $x_i$ |
| Joint angle | $\theta_i$ | Angle about $z_{i-1}$ from $x_{i-1}$ to $x_i$ |

The transformation from frame $i-1$ to frame $i$:

$$T_i = \begin{bmatrix} \cos\theta_i & -\sin\theta_i\cos\alpha_i & \sin\theta_i\sin\alpha_i & a_i\cos\theta_i \\ \sin\theta_i & \cos\theta_i\cos\alpha_i & -\cos\theta_i\sin\alpha_i & a_i\sin\theta_i \\ 0 & \sin\alpha_i & \cos\alpha_i & d_i \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

### 1.2 — Implement the DH Transform

```python
"""
Forward Kinematics using Denavit-Hartenberg parameters.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class DHParams:
    """DH parameters for a single joint."""
    a: float       # link length
    alpha: float   # link twist (radians)
    d: float       # link offset
    theta: float   # joint angle (radians) — variable for revolute joints


def dh_transform(params: DHParams) -> np.ndarray:
    """
    Compute the 4x4 homogeneous transformation matrix from DH parameters.

    Args:
        params: DH parameters for one joint

    Returns:
        4x4 transformation matrix T_i
    """
    ct = np.cos(params.theta)
    st = np.sin(params.theta)
    ca = np.cos(params.alpha)
    sa = np.sin(params.alpha)
    a = params.a
    d = params.d

    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,   sa,       ca,      d     ],
        [0,   0,        0,       1     ],
    ])


class SerialRobot:
    """
    Serial robot arm defined by DH parameters.
    Supports forward kinematics, Jacobian computation, and IK.
    """

    def __init__(self, dh_table: list[dict], joint_limits: list[tuple] = None):
        """
        Args:
            dh_table: List of {'a', 'alpha', 'd', 'theta_offset'} per joint.
                      For revolute joints, theta = q + theta_offset.
            joint_limits: List of (q_min, q_max) per joint.
        """
        self.n_joints = len(dh_table)
        self.dh_table = dh_table
        self.joint_limits = joint_limits or [(-np.pi, np.pi)] * self.n_joints

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """
        Compute end-effector pose given joint angles.

        Args:
            q: Joint angles [n_joints]

        Returns:
            4x4 homogeneous transformation matrix (base to EE)
        """
        T = np.eye(4)

        for i in range(self.n_joints):
            params = DHParams(
                a=self.dh_table[i]['a'],
                alpha=self.dh_table[i]['alpha'],
                d=self.dh_table[i]['d'],
                theta=q[i] + self.dh_table[i].get('theta_offset', 0.0),
            )
            T = T @ dh_transform(params)

        return T

    def get_position(self, q: np.ndarray) -> np.ndarray:
        """Extract end-effector position [x, y, z] from FK."""
        T = self.forward_kinematics(q)
        return T[:3, 3]

    def get_orientation(self, q: np.ndarray) -> np.ndarray:
        """Extract end-effector rotation matrix [3x3] from FK."""
        T = self.forward_kinematics(q)
        return T[:3, :3]

    def get_all_frames(self, q: np.ndarray) -> list[np.ndarray]:
        """Get transformation matrices for all intermediate frames (for visualization)."""
        frames = [np.eye(4)]
        T = np.eye(4)

        for i in range(self.n_joints):
            params = DHParams(
                a=self.dh_table[i]['a'],
                alpha=self.dh_table[i]['alpha'],
                d=self.dh_table[i]['d'],
                theta=q[i] + self.dh_table[i].get('theta_offset', 0.0),
            )
            T = T @ dh_transform(params)
            frames.append(T.copy())

        return frames


# ---------- Define a 3-DOF Planar Robot ----------
def create_3dof_planar(l1: float = 0.3, l2: float = 0.25, l3: float = 0.15):
    """Create a 3-DOF planar robot (all revolute, Z-axis rotation)."""
    dh_table = [
        {'a': l1, 'alpha': 0.0, 'd': 0.0, 'theta_offset': 0.0},
        {'a': l2, 'alpha': 0.0, 'd': 0.0, 'theta_offset': 0.0},
        {'a': l3, 'alpha': 0.0, 'd': 0.0, 'theta_offset': 0.0},
    ]
    limits = [(-np.pi, np.pi)] * 3
    return SerialRobot(dh_table, limits)


# ---------- Define a 6-DOF Robot (simplified Puma-like) ----------
def create_6dof_robot():
    """Create a 6-DOF robot with Puma-560-like DH parameters."""
    dh_table = [
        {'a': 0.0,    'alpha': -np.pi/2, 'd': 0.15, 'theta_offset': 0.0},
        {'a': 0.30,   'alpha': 0.0,      'd': 0.0,  'theta_offset': 0.0},
        {'a': 0.02,   'alpha': -np.pi/2, 'd': 0.0,  'theta_offset': 0.0},
        {'a': 0.0,    'alpha':  np.pi/2, 'd': 0.30, 'theta_offset': 0.0},
        {'a': 0.0,    'alpha': -np.pi/2, 'd': 0.0,  'theta_offset': 0.0},
        {'a': 0.0,    'alpha': 0.0,      'd': 0.06, 'theta_offset': 0.0},
    ]
    limits = [
        (-2.79, 2.79), (-1.92, 1.92), (-2.36, 2.36),
        (-5.03, 5.03), (-2.09, 2.09), (-6.28, 6.28),
    ]
    return SerialRobot(dh_table, limits)
```

### 1.3 — Visualize the Robot

```python
"""Visualization of robot arm configurations."""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_robot_2d(robot: SerialRobot, q: np.ndarray, ax=None, color='steelblue'):
    """Plot a planar robot in 2D."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    frames = robot.get_all_frames(q)
    positions = np.array([f[:3, 3] for f in frames])

    # Draw links
    ax.plot(positions[:, 0], positions[:, 1], 'o-',
            color=color, linewidth=3, markersize=8)

    # Draw joints
    for i, pos in enumerate(positions):
        ax.plot(pos[0], pos[1], 'ko', markersize=10)

    # Draw end-effector
    ax.plot(positions[-1, 0], positions[-1, 1], 'r*', markersize=15)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')

    return ax


def plot_workspace(robot: SerialRobot, n_samples: int = 5000):
    """Monte Carlo workspace visualization."""
    positions = []
    for _ in range(n_samples):
        q = np.array([
            np.random.uniform(lo, hi) for lo, hi in robot.joint_limits
        ])
        pos = robot.get_position(q)
        positions.append(pos)

    positions = np.array(positions)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(positions[:, 0], positions[:, 1], s=1, alpha=0.3, c='steelblue')
    ax.set_aspect('equal')
    ax.set_title(f'Reachable Workspace ({robot.n_joints}-DOF)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

### 1.4 — Exercise: FK Verification

```python
# TODO:
# 1. Create the 3-DOF planar robot
# 2. Verify FK for these configurations:
#    a) q = [0, 0, 0] → EE should be at (l1+l2+l3, 0, 0)
#    b) q = [π/2, 0, 0] → EE should be at (0, l1+l2+l3, 0)
#    c) q = [0, π/2, 0] → compute expected position manually
# 3. Plot the robot in 5 different configurations
# 4. Plot the workspace — what shape is it?
# 5. Create the 6-DOF robot and plot its 3D workspace

# YOUR CODE HERE
```

---

## Part 2: Inverse Kinematics (~2.5 hours)

### 2.1 — Analytical IK for Planar Robots

For a 2-DOF planar arm, IK has a closed-form solution:

```python
def analytical_ik_2dof(target_x: float, target_y: float,
                        l1: float, l2: float) -> list[tuple]:
    """
    Analytical inverse kinematics for 2-DOF planar arm.

    Returns:
        List of (q1, q2) solutions (0, 1, or 2 solutions).
    """
    d_sq = target_x**2 + target_y**2
    d = np.sqrt(d_sq)

    # Reachability check
    if d > l1 + l2 or d < abs(l1 - l2):
        return []  # unreachable

    # Cosine law for q2
    cos_q2 = (d_sq - l1**2 - l2**2) / (2 * l1 * l2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)

    solutions = []
    for sign in [1, -1]:  # elbow up / elbow down
        q2 = sign * np.arccos(cos_q2)
        # q1 from geometry
        k1 = l1 + l2 * np.cos(q2)
        k2 = l2 * np.sin(q2)
        q1 = np.arctan2(target_y, target_x) - np.arctan2(k2, k1)
        solutions.append((q1, q2))

    return solutions
```

### 2.2 — Numerical IK: Jacobian Methods

For general robots, we solve IK iteratively using the Jacobian:

```python
"""
Numerical Inverse Kinematics solvers.
"""
import numpy as np


def compute_jacobian(robot: SerialRobot, q: np.ndarray, delta: float = 1e-6) -> np.ndarray:
    """
    Compute the geometric Jacobian numerically (finite differences).

    Args:
        robot: SerialRobot instance
        q: Current joint configuration [n_joints]
        delta: Finite difference step size

    Returns:
        Jacobian matrix [6 x n_joints] (position + orientation rows)
    """
    n = robot.n_joints
    J = np.zeros((6, n))

    T_current = robot.forward_kinematics(q)
    pos_current = T_current[:3, 3]

    for i in range(n):
        q_perturbed = q.copy()
        q_perturbed[i] += delta

        T_perturbed = robot.forward_kinematics(q_perturbed)
        pos_perturbed = T_perturbed[:3, 3]

        # Position Jacobian (finite difference)
        J[:3, i] = (pos_perturbed - pos_current) / delta

        # Orientation Jacobian (approximate from rotation difference)
        R_diff = T_perturbed[:3, :3] @ T_current[:3, :3].T
        # Extract axis-angle from rotation difference
        J[3:, i] = np.array([
            R_diff[2, 1] - R_diff[1, 2],
            R_diff[0, 2] - R_diff[2, 0],
            R_diff[1, 0] - R_diff[0, 1],
        ]) / (2 * delta)

    return J


def ik_jacobian_pseudoinverse(
    robot: SerialRobot,
    target_pos: np.ndarray,
    q_init: np.ndarray = None,
    max_iter: int = 200,
    tol: float = 1e-4,
    step_size: float = 0.5,
) -> tuple[np.ndarray, bool, int]:
    """
    Solve IK using Jacobian pseudoinverse (Moore-Penrose).

    J† = J^T (J J^T)^{-1}
    Δq = α * J† * e

    Args:
        robot: SerialRobot instance
        target_pos: Desired [x, y, z] position
        q_init: Initial joint guess (random if None)
        max_iter: Maximum iterations
        tol: Position error tolerance
        step_size: Learning rate α

    Returns:
        (q_solution, converged, iterations)
    """
    if q_init is None:
        q_init = np.zeros(robot.n_joints)

    q = q_init.copy()

    for iteration in range(max_iter):
        current_pos = robot.get_position(q)
        error = target_pos - current_pos
        error_norm = np.linalg.norm(error)

        if error_norm < tol:
            return q, True, iteration

        # Compute position Jacobian (top 3 rows only)
        J = compute_jacobian(robot, q)[:3, :]

        # Pseudoinverse: J† = J^T (J J^T)^{-1}
        JJT = J @ J.T
        try:
            J_pinv = J.T @ np.linalg.inv(JJT)
        except np.linalg.LinAlgError:
            # Singular — fall back to damped
            J_pinv = J.T @ np.linalg.inv(JJT + 0.01 * np.eye(3))

        # Update
        dq = step_size * J_pinv @ error
        q = q + dq

        # Clip to joint limits
        for i in range(robot.n_joints):
            q[i] = np.clip(q[i], *robot.joint_limits[i])

    return q, False, max_iter


def ik_damped_least_squares(
    robot: SerialRobot,
    target_pos: np.ndarray,
    q_init: np.ndarray = None,
    max_iter: int = 200,
    tol: float = 1e-4,
    damping: float = 0.05,
    step_size: float = 1.0,
) -> tuple[np.ndarray, bool, int]:
    """
    Solve IK using Damped Least Squares (Levenberg-Marquardt).

    Δq = J^T (J J^T + λ²I)^{-1} * e

    More robust near singularities than pseudoinverse.

    Args:
        damping: λ — regularization parameter

    Returns:
        (q_solution, converged, iterations)
    """
    if q_init is None:
        q_init = np.zeros(robot.n_joints)

    q = q_init.copy()

    for iteration in range(max_iter):
        current_pos = robot.get_position(q)
        error = target_pos - current_pos
        error_norm = np.linalg.norm(error)

        if error_norm < tol:
            return q, True, iteration

        J = compute_jacobian(robot, q)[:3, :]

        # Damped least squares
        JJT = J @ J.T
        damped = JJT + damping**2 * np.eye(3)
        dq = step_size * J.T @ np.linalg.solve(damped, error)

        q = q + dq

        # Clip to joint limits
        for i in range(robot.n_joints):
            q[i] = np.clip(q[i], *robot.joint_limits[i])

    return q, False, max_iter


def ik_with_nullspace(
    robot: SerialRobot,
    target_pos: np.ndarray,
    q_init: np.ndarray = None,
    q_preferred: np.ndarray = None,
    nullspace_gain: float = 0.1,
    max_iter: int = 200,
    tol: float = 1e-4,
    damping: float = 0.05,
) -> tuple[np.ndarray, bool, int]:
    """
    IK with nullspace optimization — reach target while keeping joints
    close to a preferred configuration.

    Δq = J† * e + (I - J† J) * (q_preferred - q) * gain

    This is key for redundant robots (>6 DOF) where multiple IK solutions exist.
    """
    if q_init is None:
        q_init = np.zeros(robot.n_joints)
    if q_preferred is None:
        q_preferred = np.zeros(robot.n_joints)

    q = q_init.copy()
    n = robot.n_joints

    for iteration in range(max_iter):
        current_pos = robot.get_position(q)
        error = target_pos - current_pos

        if np.linalg.norm(error) < tol:
            return q, True, iteration

        J = compute_jacobian(robot, q)[:3, :]

        # Damped pseudoinverse
        JJT = J @ J.T + damping**2 * np.eye(3)
        J_pinv = J.T @ np.linalg.inv(JJT)

        # Primary task: reach target
        dq_primary = J_pinv @ error

        # Nullspace projection: (I - J† J)
        nullspace_proj = np.eye(n) - J_pinv @ J

        # Secondary task: move toward preferred config
        dq_secondary = nullspace_proj @ (nullspace_gain * (q_preferred - q))

        q = q + dq_primary + dq_secondary

        for i in range(n):
            q[i] = np.clip(q[i], *robot.joint_limits[i])

    return q, False, max_iter
```

### 2.3 — Exercise: Compare IK Methods

```python
# TODO:
# 1. Create the 3-DOF planar robot
# 2. Define 10 target positions within the workspace
# 3. Solve IK with:
#    a) Analytical (2-DOF version, ignore 3rd joint)
#    b) Jacobian pseudoinverse
#    c) Damped least squares (try λ = 0.01, 0.05, 0.1)
# 4. Compare:
#    - Convergence speed (iterations)
#    - Final position error
#    - Behavior near workspace boundary (singularities)
# 5. Plot: convergence curves for each method
# 6. For the 6-DOF robot: solve IK for 5 random targets
#    - Use nullspace to prefer q_preferred = [0,0,0,0,0,0] (upright)
#    - Visualize the solution vs. solution without nullspace

# YOUR CODE HERE
```

---

## Part 3: Trajectory Generation (~2 hours)

### 3.1 — Joint-Space vs Task-Space Trajectories

| Aspect | Joint-Space | Task-Space |
|--------|-------------|------------|
| Planning in | q₁, q₂, ..., qₙ | x, y, z, R |
| Interpolation | Polynomial in q | Polynomial in Cartesian |
| IK needed? | No (already in joint) | Yes, at every timestep |
| Obstacle avoidance | Hard | Natural |
| Straight-line EE path? | Not guaranteed | Yes |
| Singularity issues | None during execution | Can hit singularities |

### 3.2 — Polynomial Trajectory Interpolation

```python
"""
Trajectory generation: joint-space polynomial interpolation.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class TrajectoryPoint:
    """Waypoint with optional velocity/acceleration constraints."""
    position: np.ndarray
    velocity: np.ndarray = None
    acceleration: np.ndarray = None
    time: float = 0.0


class CubicTrajectory:
    """
    Cubic polynomial trajectory between two points.

    q(t) = a₀ + a₁t + a₂t² + a₃t³

    Boundary conditions: q(0), q(T), q̇(0), q̇(T)
    """

    def __init__(self, q_start: np.ndarray, q_end: np.ndarray,
                 v_start: np.ndarray = None, v_end: np.ndarray = None,
                 duration: float = 1.0):
        self.n_joints = len(q_start)
        self.duration = duration

        if v_start is None:
            v_start = np.zeros(self.n_joints)
        if v_end is None:
            v_end = np.zeros(self.n_joints)

        # Solve for coefficients per joint
        # q(0) = a0 = q_start
        # q̇(0) = a1 = v_start
        # q(T) = a0 + a1*T + a2*T² + a3*T³ = q_end
        # q̇(T) = a1 + 2*a2*T + 3*a3*T² = v_end
        T = duration
        self.coeffs = np.zeros((self.n_joints, 4))  # [a0, a1, a2, a3]

        for j in range(self.n_joints):
            a0 = q_start[j]
            a1 = v_start[j]
            a2 = (3 * (q_end[j] - q_start[j]) / T**2
                   - 2 * v_start[j] / T - v_end[j] / T)
            a3 = (-2 * (q_end[j] - q_start[j]) / T**3
                   + (v_end[j] + v_start[j]) / T**2)
            self.coeffs[j] = [a0, a1, a2, a3]

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate trajectory at time t.

        Returns:
            (position, velocity, acceleration) at time t
        """
        t = np.clip(t, 0, self.duration)
        pos = np.zeros(self.n_joints)
        vel = np.zeros(self.n_joints)
        acc = np.zeros(self.n_joints)

        for j in range(self.n_joints):
            a0, a1, a2, a3 = self.coeffs[j]
            pos[j] = a0 + a1*t + a2*t**2 + a3*t**3
            vel[j] = a1 + 2*a2*t + 3*a3*t**2
            acc[j] = 2*a2 + 6*a3*t

        return pos, vel, acc

    def sample(self, dt: float = 0.01) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample the full trajectory at fixed timestep."""
        times = np.arange(0, self.duration + dt, dt)
        positions = []
        velocities = []
        accelerations = []

        for t in times:
            p, v, a = self.evaluate(t)
            positions.append(p)
            velocities.append(v)
            accelerations.append(a)

        return np.array(positions), np.array(velocities), np.array(accelerations)


class QuinticTrajectory:
    """
    Quintic (5th order) polynomial trajectory.

    q(t) = a₀ + a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵

    Boundary conditions: q, q̇, q̈ at start and end.
    Guarantees continuous acceleration (smoother than cubic).
    """

    def __init__(self, q_start: np.ndarray, q_end: np.ndarray,
                 v_start: np.ndarray = None, v_end: np.ndarray = None,
                 a_start: np.ndarray = None, a_end: np.ndarray = None,
                 duration: float = 1.0):
        self.n_joints = len(q_start)
        self.duration = duration

        if v_start is None:
            v_start = np.zeros(self.n_joints)
        if v_end is None:
            v_end = np.zeros(self.n_joints)
        if a_start is None:
            a_start = np.zeros(self.n_joints)
        if a_end is None:
            a_end = np.zeros(self.n_joints)

        T = duration
        self.coeffs = np.zeros((self.n_joints, 6))

        for j in range(self.n_joints):
            a0 = q_start[j]
            a1 = v_start[j]
            a2 = a_start[j] / 2.0

            # Solve 3x3 system for a3, a4, a5
            M = np.array([
                [T**3,     T**4,      T**5],
                [3*T**2,   4*T**3,    5*T**4],
                [6*T,      12*T**2,   20*T**3],
            ])
            rhs = np.array([
                q_end[j] - a0 - a1*T - a2*T**2,
                v_end[j] - a1 - 2*a2*T,
                a_end[j] - 2*a2,
            ])
            a3, a4, a5 = np.linalg.solve(M, rhs)
            self.coeffs[j] = [a0, a1, a2, a3, a4, a5]

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate position, velocity, acceleration at time t."""
        t = np.clip(t, 0, self.duration)
        pos = np.zeros(self.n_joints)
        vel = np.zeros(self.n_joints)
        acc = np.zeros(self.n_joints)

        for j in range(self.n_joints):
            a = self.coeffs[j]
            pos[j] = a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3 + a[4]*t**4 + a[5]*t**5
            vel[j] = a[1] + 2*a[2]*t + 3*a[3]*t**2 + 4*a[4]*t**3 + 5*a[5]*t**4
            acc[j] = 2*a[2] + 6*a[3]*t + 12*a[4]*t**2 + 20*a[5]*t**3

        return pos, vel, acc

    def sample(self, dt: float = 0.01) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample the full trajectory."""
        times = np.arange(0, self.duration + dt, dt)
        positions = np.array([self.evaluate(t)[0] for t in times])
        velocities = np.array([self.evaluate(t)[1] for t in times])
        accelerations = np.array([self.evaluate(t)[2] for t in times])
        return positions, velocities, accelerations
```

### 3.3 — Multi-Waypoint Trajectories

```python
class MultiWaypointTrajectory:
    """
    Chain multiple cubic segments through a sequence of waypoints.
    Ensures velocity continuity at waypoint boundaries.
    """

    def __init__(self, waypoints: list[np.ndarray], durations: list[float] = None):
        """
        Args:
            waypoints: List of joint configurations [n_waypoints x n_joints]
            durations: Time for each segment (auto-computed if None)
        """
        self.waypoints = [np.asarray(w) for w in waypoints]
        self.n_segments = len(waypoints) - 1
        self.n_joints = len(waypoints[0])

        if durations is None:
            # Auto duration: proportional to angular distance
            durations = []
            for i in range(self.n_segments):
                dist = np.linalg.norm(self.waypoints[i+1] - self.waypoints[i])
                durations.append(max(dist / 1.0, 0.5))  # min 0.5s per segment

        self.durations = durations
        self.total_duration = sum(durations)

        # Compute intermediate velocities using parabolic blending
        velocities = self._compute_velocities()

        # Build cubic segments
        self.segments = []
        for i in range(self.n_segments):
            seg = CubicTrajectory(
                q_start=self.waypoints[i],
                q_end=self.waypoints[i+1],
                v_start=velocities[i],
                v_end=velocities[i+1],
                duration=durations[i],
            )
            self.segments.append(seg)

    def _compute_velocities(self) -> list[np.ndarray]:
        """Compute velocities at each waypoint for smooth transitions."""
        n = len(self.waypoints)
        velocities = [np.zeros(self.n_joints)]  # start at rest

        for i in range(1, n - 1):
            # Average velocity from adjacent segments
            v_before = (self.waypoints[i] - self.waypoints[i-1]) / self.durations[i-1]
            v_after = (self.waypoints[i+1] - self.waypoints[i]) / self.durations[i]

            # Use average if same direction, zero if direction changes
            v = np.zeros(self.n_joints)
            for j in range(self.n_joints):
                if v_before[j] * v_after[j] > 0:  # same direction
                    v[j] = 0.5 * (v_before[j] + v_after[j])
                # else: v[j] = 0 (direction change → zero velocity)
            velocities.append(v)

        velocities.append(np.zeros(self.n_joints))  # end at rest
        return velocities

    def evaluate(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate trajectory at global time t."""
        t = np.clip(t, 0, self.total_duration)

        # Find which segment
        elapsed = 0.0
        for i, seg in enumerate(self.segments):
            if t <= elapsed + seg.duration:
                local_t = t - elapsed
                return seg.evaluate(local_t)
            elapsed += seg.duration

        # End of trajectory
        return self.segments[-1].evaluate(self.segments[-1].duration)

    def sample(self, dt: float = 0.01) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample full multi-segment trajectory."""
        times = np.arange(0, self.total_duration + dt, dt)
        results = [self.evaluate(t) for t in times]
        positions = np.array([r[0] for r in results])
        velocities = np.array([r[1] for r in results])
        accelerations = np.array([r[2] for r in results])
        return positions, velocities, accelerations
```

### 3.4 — Task-Space Trajectory (Cartesian)

```python
class CartesianTrajectory:
    """
    Task-space straight-line trajectory with IK at each step.
    Useful when you need the end-effector to follow a specific path.
    """

    def __init__(self, robot: SerialRobot,
                 start_pos: np.ndarray, end_pos: np.ndarray,
                 duration: float = 2.0, dt: float = 0.01,
                 q_init: np.ndarray = None):
        self.robot = robot
        self.dt = dt
        self.duration = duration
        self.n_steps = int(duration / dt)

        # Generate Cartesian path (linear interpolation)
        alphas = np.linspace(0, 1, self.n_steps)
        self.cartesian_path = np.array([
            (1 - a) * start_pos + a * end_pos for a in alphas
        ])

        # Solve IK at each step
        self.joint_path = np.zeros((self.n_steps, robot.n_joints))
        q = q_init if q_init is not None else np.zeros(robot.n_joints)

        for i, target in enumerate(self.cartesian_path):
            q_sol, converged, _ = ik_damped_least_squares(
                robot, target, q_init=q, max_iter=50, damping=0.05
            )
            if converged:
                q = q_sol
            self.joint_path[i] = q

    def get_joint_trajectory(self) -> np.ndarray:
        """Return the joint-space trajectory [n_steps x n_joints]."""
        return self.joint_path

    def get_cartesian_trajectory(self) -> np.ndarray:
        """Return the planned Cartesian path [n_steps x 3]."""
        return self.cartesian_path

    def verify_tracking(self) -> np.ndarray:
        """Compute actual EE positions and tracking error."""
        actual_positions = np.array([
            self.robot.get_position(q) for q in self.joint_path
        ])
        errors = np.linalg.norm(actual_positions - self.cartesian_path, axis=1)
        return errors
```

### 3.5 — Exercise: Build and Compare Trajectories

```python
# TODO:
# 1. Create a 3-DOF planar robot
# 2. Define 4 waypoints (joint configurations)
# 3. Generate trajectories using:
#    a) Cubic polynomial (point-to-point)
#    b) Quintic polynomial (same endpoints)
#    c) Multi-waypoint cubic (through all 4 points)
# 4. Plot for each:
#    - Joint positions over time
#    - Joint velocities over time
#    - Joint accelerations over time
# 5. Compare cubic vs quintic: which has continuous acceleration?
# 6. Generate a Cartesian straight-line trajectory:
#    - Start: (0.3, 0.2, 0)
#    - End: (-0.2, 0.3, 0)
#    - Plot the EE path — is it straight?
#    - Plot the tracking error
# 7. BONUS: What happens near a singularity (arm fully extended)?

# YOUR CODE HERE
```

---

## Part 4: Velocity & Acceleration Limits (~1 hour)

### 4.1 — Time-Optimal Scaling

Real robots have joint velocity and acceleration limits. We need to slow down
trajectories that exceed them:

```python
def time_scale_trajectory(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    dt: float,
    vel_limits: np.ndarray,
    acc_limits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Scale trajectory duration to respect velocity and acceleration limits.

    Args:
        positions: [T x n_joints]
        velocities: [T x n_joints]
        accelerations: [T x n_joints]
        dt: Original timestep
        vel_limits: Max absolute velocity per joint [n_joints]
        acc_limits: Max absolute acceleration per joint [n_joints]

    Returns:
        (scaled_pos, scaled_vel, scaled_acc, new_dt)
    """
    # Find the worst violation ratio
    vel_ratios = np.abs(velocities) / vel_limits[None, :]
    acc_ratios = np.abs(accelerations) / acc_limits[None, :]

    max_vel_ratio = np.max(vel_ratios) if vel_ratios.size > 0 else 0
    max_acc_ratio = np.max(acc_ratios) if acc_ratios.size > 0 else 0

    # Time must be scaled by max(vel_ratio, sqrt(acc_ratio))
    # Because: v scales as 1/s, a scales as 1/s²
    scale_factor = max(1.0, max_vel_ratio, np.sqrt(max_acc_ratio))

    new_dt = dt * scale_factor

    # Rescale velocities and accelerations
    scaled_vel = velocities / scale_factor
    scaled_acc = accelerations / scale_factor**2

    print(f"Time scaling: {scale_factor:.2f}x (dt: {dt:.4f} → {new_dt:.4f})")
    if scale_factor > 1:
        print(f"  Velocity violation: {max_vel_ratio:.2f}x limit")
        print(f"  Acceleration violation: {max_acc_ratio:.2f}x limit")

    return positions, scaled_vel, scaled_acc, new_dt


def trapezoidal_velocity_profile(
    q_start: np.ndarray,
    q_end: np.ndarray,
    v_max: np.ndarray,
    a_max: np.ndarray,
    dt: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a trapezoidal velocity profile (bang-coast-bang).
    Fastest profile respecting velocity and acceleration limits.

    Each joint is planned independently, then synchronized to the slowest.
    """
    n_joints = len(q_start)
    delta_q = q_end - q_start
    directions = np.sign(delta_q)

    # Compute time for each joint
    joint_times = np.zeros(n_joints)
    for j in range(n_joints):
        dq = abs(delta_q[j])
        if dq < 1e-10:
            joint_times[j] = 0.0
            continue

        # Time to reach v_max
        t_accel = v_max[j] / a_max[j]
        # Distance during accel + decel
        d_accel = a_max[j] * t_accel**2  # = v_max² / a_max

        if d_accel >= dq:
            # Triangle profile (never reaches v_max)
            t_total = 2 * np.sqrt(dq / a_max[j])
        else:
            # Trapezoidal: accel + coast + decel
            d_coast = dq - d_accel
            t_coast = d_coast / v_max[j]
            t_total = 2 * t_accel + t_coast

        joint_times[j] = t_total

    # Synchronize: all joints finish at the same time
    T = max(joint_times)
    n_steps = int(T / dt) + 1
    times = np.linspace(0, T, n_steps)

    positions = np.zeros((n_steps, n_joints))
    velocities = np.zeros((n_steps, n_joints))
    accelerations = np.zeros((n_steps, n_joints))

    for j in range(n_joints):
        dq = abs(delta_q[j])
        if dq < 1e-10:
            positions[:, j] = q_start[j]
            continue

        # Scale v_max and a_max for this joint to match synchronized time T
        # Recompute profile to finish in exactly T
        t_accel_j = T / 2 - np.sqrt(T**2 / 4 - dq / a_max[j]) if dq < a_max[j] * (T/2)**2 else T / 2
        t_accel_j = min(t_accel_j, T / 2)
        v_peak = dq / (T - t_accel_j) if T > t_accel_j else 0

        for i, t in enumerate(times):
            if t < t_accel_j:
                # Accelerating
                accelerations[i, j] = directions[j] * v_peak / t_accel_j
                velocities[i, j] = directions[j] * v_peak * t / t_accel_j
                positions[i, j] = q_start[j] + directions[j] * 0.5 * v_peak * t**2 / t_accel_j
            elif t < T - t_accel_j:
                # Coasting
                accelerations[i, j] = 0.0
                velocities[i, j] = directions[j] * v_peak
                d_accel_phase = 0.5 * v_peak * t_accel_j
                positions[i, j] = q_start[j] + directions[j] * (d_accel_phase + v_peak * (t - t_accel_j))
            else:
                # Decelerating
                t_remaining = T - t
                accelerations[i, j] = -directions[j] * v_peak / t_accel_j
                velocities[i, j] = directions[j] * v_peak * t_remaining / t_accel_j
                positions[i, j] = q_end[j] - directions[j] * 0.5 * v_peak * t_remaining**2 / t_accel_j

    return positions, velocities, accelerations
```

### 4.2 — Exercise: Trajectory with Limits

```python
# TODO:
# 1. Define velocity limits: [2.0, 2.0, 3.0] rad/s
#    Acceleration limits: [5.0, 5.0, 8.0] rad/s²
# 2. Generate a cubic trajectory from q=[0,0,0] to q=[π/2, -π/3, π/4]
#    with duration=0.5s
# 3. Check: does it violate limits? Apply time_scale_trajectory
# 4. Generate the same motion with trapezoidal_velocity_profile
# 5. Plot both side by side:
#    - Position, velocity, acceleration per joint
# 6. Which is faster? Which is smoother?
# 7. Connection to Project 08: why does the action interpolator need
#    acceleration limits? What happens if a VLA outputs a joint command
#    that violates limits?

# YOUR CODE HERE
```

---

## Part 5: Connecting Kinematics to VLA Action Spaces (~0.5 hours)

### 5.1 — Why Action Space Choice Matters

VLA policies output actions. The choice of action space fundamentally determines
what the policy must learn:

| Action Space | Representation | VLA Difficulty | Precision |
|--------------|----------------|----------------|-----------|
| Joint position | q (absolute) | Must learn full IK implicitly | High |
| Joint velocity | dq/dt | Must learn velocity control | Medium |
| EE position (absolute) | [x, y, z, quat] | Must learn FK implicitly | High |
| EE delta (relative) | [Δx, Δy, Δz] | Easiest for policy | Medium |
| EE delta + gripper | [Δx, Δy, Δz, grip] | Most common for VLAs | Medium |

### 5.2 — The VLA Pipeline Connection

```python
"""
Demonstrate how kinematics connects VLA outputs to robot commands.
This is what happens inside a VLA deployment pipeline (see Project 08).
"""


def vla_to_robot_command(
    vla_action: np.ndarray,     # VLA output: [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]
    current_q: np.ndarray,       # Current joint configuration
    robot: SerialRobot,
    control_mode: str = "resolved_rate",
    dt: float = 0.1,
) -> np.ndarray:
    """
    Convert VLA action to joint-space command.

    This is the core of what the Action Interpolator in Project 08 does.
    """
    if control_mode == "resolved_rate":
        # Convert EE velocity to joint velocity via Jacobian
        J = compute_jacobian(robot, current_q)[:3, :]
        ee_velocity = vla_action[:3] / dt

        # Damped least squares for robustness
        JJT = J @ J.T + 0.01 * np.eye(3)
        joint_velocity = J.T @ np.linalg.solve(JJT, ee_velocity)

        # Integrate to get new joint position
        q_target = current_q + joint_velocity * dt

    elif control_mode == "ik_target":
        # Compute target EE position, solve IK
        current_pos = robot.get_position(current_q)
        target_pos = current_pos + vla_action[:3]

        q_target, converged, _ = ik_damped_least_squares(
            robot, target_pos, q_init=current_q
        )
        if not converged:
            q_target = current_q  # stay put if IK fails

    else:
        raise ValueError(f"Unknown control mode: {control_mode}")

    # Clip to joint limits
    for i in range(robot.n_joints):
        q_target[i] = np.clip(q_target[i], *robot.joint_limits[i])

    return q_target


# Exercise reflection:
# When a VLA outputs Δx = 0.05m but the robot is near a singularity,
# the required joint velocities can be enormous. This is why:
# 1. Safety gates (Project 08) clip velocities
# 2. Action interpolators smooth abrupt commands
# 3. Understanding kinematics prevents dangerous deployments
```

---

## Summary & Key Takeaways

| Concept | Why It Matters for VLAs |
|---------|------------------------|
| Forward kinematics (DH) | Predict where the robot will be given joint angles |
| Inverse kinematics | Convert VLA task-space actions to joint commands |
| Jacobian | Real-time velocity mapping, singularity detection |
| Nullspace optimization | Redundancy resolution — same EE pose, different configs |
| Polynomial interpolation | Smooth action execution between VLA inference steps |
| Velocity/acceleration limits | Safety-critical: prevent hardware damage |
| Trapezoidal profiles | Time-optimal motion within limits |
| Action space choice | Determines what the VLA must learn vs. what the controller handles |

### Connection to Other Exercises

- **Exercise 12**: Used Jacobian control for MuJoCo manipulation
- **Exercise 14**: Data collection requires understanding reachable workspace
- **Project 08**: Action interpolator implements trajectory generation between VLA commands
- **Study Note 16**: Deployment chapter covers real-time control integration

### Further Reading
- Siciliano et al., *Robotics: Modelling, Planning and Control* (2009) — Ch. 2-4
- Lynch & Park, *Modern Robotics* (2017) — Ch. 4-6, free textbook
- Craig, *Introduction to Robotics* (2005) — DH convention standard reference
- [MoveIt 2 Concepts](https://moveit.ros.org/documentation/concepts/) — IK solvers in practice
- [Pinocchio library](https://github.com/stack-of-tasks/pinocchio) — fast C++ kinematics/dynamics

---

## Checklist

- [ ] Implemented DH transform and forward kinematics
- [ ] Verified FK for known configurations
- [ ] Visualized robot workspace (Monte Carlo)
- [ ] Solved IK with analytical method (2-DOF)
- [ ] Solved IK with Jacobian pseudoinverse
- [ ] Solved IK with damped least-squares
- [ ] Understood nullspace optimization for redundant robots
- [ ] Generated cubic and quintic polynomial trajectories
- [ ] Built multi-waypoint trajectory with velocity continuity
- [ ] Generated Cartesian straight-line trajectories with IK
- [ ] Applied velocity/acceleration limits
- [ ] Generated trapezoidal velocity profiles
- [ ] Connected kinematics to VLA action space choices
- [ ] Understand why action interpolation is critical for safe deployment
