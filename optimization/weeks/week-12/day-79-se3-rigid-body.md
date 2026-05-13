# Day 79: SE(3) — 3D Rigid Body Transforms

> **Phase VI — Lie Groups & Manifold Optimization | Week 12 | 2.5 hours**
> *The 6-DOF transform group that chains every frame in a robot's kinematic tree.*

## Navigation
- **Previous:** [Day 78: SE(2) — 2D Rigid Motions](day-78-se2-rigid-motions.md)
- **Next:** [Day 80: Exponential and Logarithmic Maps](day-80-exponential-map.md)
- **Week:** [Week 12 Overview](../README.md)
- **Chapter:** [Chapter 09: Lie Groups Reference](../../09-lie-groups.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
ROS `tf2` maintains a tree of SE(3) transforms: `world → odom → base_link → laser_link → camera_optical`. Every sensor reading requires chaining SE(3) elements. When the navigation estimator publishes `odom → base_link`, it is publishing an SE(3) element. Incorrect composition order is a frequent source of transform bugs in multi-sensor fusion.

---

## Theory (45 min)

### 79.1 The 4×4 Homogeneous Matrix

An SE(3) element combines rotation $R \in SO(3)$ and translation $\mathbf{t} \in \mathbb{R}^3$:

$$T = \begin{pmatrix} R & \mathbf{t} \\ \mathbf{0}^T & 1 \end{pmatrix} \in \mathbb{R}^{4 \times 4}$$

Transforming a point: $\mathbf{p}' = R\mathbf{p} + \mathbf{t}$, or in homogeneous coordinates:

$$\begin{pmatrix} \mathbf{p}' \\ 1 \end{pmatrix} = T \begin{pmatrix} \mathbf{p} \\ 1 \end{pmatrix}$$

### 79.2 Twist Coordinates and Screw Motions

The Lie algebra $\mathfrak{se}(3)$ element (a **twist**) is parameterized by $\xi = (\mathbf{v}, \boldsymbol{\omega})^T \in \mathbb{R}^6$:

$$\hat{\xi} = \begin{pmatrix} [\boldsymbol{\omega}]_\times & \mathbf{v} \\ \mathbf{0}^T & 0 \end{pmatrix} \in \mathbb{R}^{4 \times 4}$$

where $[\boldsymbol{\omega}]_\times$ is the $3\times3$ skew-symmetric matrix. The 6 parameters: $\mathbf{v}$ is linear velocity, $\boldsymbol{\omega}$ is angular velocity.

### 79.3 Chasles' Theorem

**Every rigid body displacement can be realized as a rotation about an axis combined with a translation along that axis** — a **screw motion**. The axis is called the **screw axis**, and the ratio $d/\theta$ (translation per radian) is the **pitch**.

- Pure rotation: pitch = 0
- Pure translation: pitch = ∞ ($\theta = 0$)
- General: finite pitch

### 79.4 Group Operations

$$\text{Composition:} \quad T_1 T_2 = \begin{pmatrix} R_1 R_2 & R_1 \mathbf{t}_2 + \mathbf{t}_1 \\ \mathbf{0}^T & 1 \end{pmatrix}$$

$$\text{Inverse:} \quad T^{-1} = \begin{pmatrix} R^T & -R^T \mathbf{t} \\ \mathbf{0}^T & 1 \end{pmatrix}$$

**Convention matters:** $T_A^B$ transforms points from frame $B$ to frame $A$. Chaining: $T_A^C = T_A^B \cdot T_B^C$.

---

## Implementation (60 min)

```python
import numpy as np

def hat3(omega):
    """Skew-symmetric matrix from 3-vector."""
    return np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])

def se3_from_Rt(R, t):
    """Build 4x4 SE(3) matrix from R (3x3) and t (3,)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def se3_compose(T1, T2):
    """Compose two SE(3) elements."""
    return T1 @ T2

def se3_inverse(T):
    """Compute SE(3) inverse."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def se3_transform_point(T, p):
    """Transform a 3D point by SE(3)."""
    return T[:3, :3] @ p + T[:3, 3]

def rodrigues(omega):
    """SO(3) exponential map via Rodrigues' formula."""
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3)
    K = hat3(omega / theta)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

# --- Demo: transform chain ---
# world -> odom (robot moved 2m forward)
T_w_o = se3_from_Rt(np.eye(3), np.array([2.0, 0.0, 0.0]))

# odom -> base_link (robot rotated 30° about z)
angle = np.pi / 6
R_o_b = rodrigues(np.array([0, 0, angle]))
T_o_b = se3_from_Rt(R_o_b, np.array([0.5, 0.1, 0.0]))

# base_link -> laser (fixed offset)
T_b_l = se3_from_Rt(np.eye(3), np.array([0.3, 0.0, 0.15]))

# Chain: world -> laser
T_w_l = se3_compose(T_w_o, se3_compose(T_o_b, T_b_l))
print("World -> Laser:\n", np.round(T_w_l, 4))

# Point in laser frame -> world frame
p_laser = np.array([1.0, 0.0, 0.0])
p_world = se3_transform_point(T_w_l, p_laser)
print("Laser point in world:", np.round(p_world, 4))

# Relative transform: given world poses of two robots
T_w_A = se3_from_Rt(rodrigues([0, 0, 0.5]), [3, 1, 0])
T_w_B = se3_from_Rt(rodrigues([0, 0, -0.2]), [5, 2, 0])
T_A_B = se3_compose(se3_inverse(T_w_A), T_w_B)
print("Relative A->B:\n", np.round(T_A_B, 4))
```

---

## Practice Problems (45 min)

**Problem 1:** Given $T_w^o$ (world→odom) and $T_o^b$ (odom→base), compute $T_w^b$ and $T_b^w$.

<details><summary>Answer</summary>

$T_w^b = T_w^o \cdot T_o^b$ (chain left-to-right). $T_b^w = (T_w^b)^{-1} = \begin{pmatrix} R^T & -R^T\mathbf{t} \\ 0 & 1 \end{pmatrix}$ where $R, \mathbf{t}$ are extracted from $T_w^b$.
</details>

**Problem 2:** A sensor at $T_b^s$ (base→sensor) observes a landmark at position $\mathbf{p}^s$ in sensor frame. Express the landmark in world frame given $T_w^b$.

<details><summary>Answer</summary>

$\mathbf{p}^w = T_w^b \cdot T_b^s \cdot \begin{pmatrix}\mathbf{p}^s \\ 1\end{pmatrix}$. Chain the transforms: $T_w^s = T_w^b \cdot T_b^s$, then $\mathbf{p}^w = R_{ws}\mathbf{p}^s + \mathbf{t}_{ws}$.
</details>

**Problem 3:** Verify the group axioms (closure, associativity, identity, inverse) for SE(3) using three random SE(3) elements.

<details><summary>Answer</summary>

Numerically: generate $T_1, T_2, T_3$ with random rotations/translations. Check: (1) $T_1 T_2$ has $\det(R_{12}) = 1$; (2) $(T_1 T_2) T_3 \approx T_1 (T_2 T_3)$; (3) $T_1 I = I T_1 = T_1$; (4) $T_1 T_1^{-1} \approx I$.
</details>

---

## Expert Challenges

**Challenge 1:** Prove that SE(3) has dimension 6: 3 for rotation + 3 for translation, and that the constraint $R^T R = I, \det R = 1$ removes exactly $16 - 6 = 10$ degrees of freedom from the 4×4 matrix.

<details><summary>Answer</summary>

A 4×4 matrix has 16 entries. Constraints: $R^T R = I$ gives 6 independent equations (the upper triangle of the symmetric $3\times3$ identity), $\det R = 1$ is dependent given $R^T R = I$. Last row fixed $(0,0,0,1)$ gives 4 constraints. Total: $16 - 6 - 4 = 6$ DOF.
</details>

**Challenge 2:** Implement a function that extracts roll, pitch, yaw from an SE(3) matrix using the ZYX convention, handling gimbal lock.

<details><summary>Answer</summary>

```python
def se3_to_rpy(T):
    R = T[:3, :3]
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        roll = np.arctan2(R[2,1], R[2,2])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = np.arctan2(R[1,0], R[0,0])
    else:  # gimbal lock
        roll = np.arctan2(-R[1,2], R[1,1])
        pitch = np.arctan2(-R[2,0], sy)
        yaw = 0.0
    return roll, pitch, yaw
```
</details>

**Challenge 3:** Show that the set of pure translations $\{(I, \mathbf{t}) : \mathbf{t} \in \mathbb{R}^3\}$ forms a **normal subgroup** of SE(3).

<details><summary>Answer</summary>

Let $N = \{(I, \mathbf{t})\}$ and $g = (R, \mathbf{s}) \in SE(3)$. Then $g (I, \mathbf{t}) g^{-1} = (R, \mathbf{s})(I, \mathbf{t})(R^T, -R^T\mathbf{s}) = (I, R\mathbf{t})$. Since $R\mathbf{t} \in \mathbb{R}^3$, we have $g N g^{-1} \subseteq N$. ✓
</details>

---

## Connections
- **Back:** [Day 78: SE(2)](day-78-se2-rigid-motions.md) — SE(3) is the 3D generalization
- **Back:** [Day 73: SO(3)](../week-11/day-73-so3-rodrigues.md) — rotation subgroup of SE(3)
- **Forward:** [Day 80: Exp/Log Maps](day-80-exponential-map.md) — mapping between $\mathfrak{se}(3)$ and SE(3)
- **OKS:** `tf2` transform tree is a directed graph of SE(3) elements

## Self-Check
- [ ] I can construct an SE(3) matrix from rotation + translation and decompose it back
- [ ] I understand the transform chain convention $T_A^C = T_A^B \cdot T_B^C$
- [ ] I can compute relative transforms between two poses
- [ ] I know what a screw motion is and can state Chasles' theorem
