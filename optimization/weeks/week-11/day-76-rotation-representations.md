# Day 76: Rotation Representations Compared

> **Phase VI — Lie Groups & Manifold Optimization | Week 11 | 2.5 hours**
> *Every representation has a fatal flaw. The art is knowing which flaw you can tolerate.*

## Navigation
- **Previous:** [Day 75: Quaternion Interpolation and Averaging](day-75-quaternion-operations.md)
- **Next:** [Day 77: Week 11 Review](day-77-week11-review.md)
- **Week:** [Week 11 Overview](../README.md)
- **Chapter:** [Chapter 09: Lie Groups Reference](../../09-lie-groups.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
The OKS stack uses multiple representations simultaneously: quaternions in `tf2` messages, Euler angles for human-readable logging, rotation matrices inside the estimator, and axis-angle for incremental IMU updates. Conversion bugs between representations are a recurring source of navigation failures. Understanding the tradeoffs prevents choosing the wrong tool for the job.

---

## Theory (45 min)

### 76.1 Four Representations

| Representation | Parameters | Constraints | DOF |
|---------------|-----------|-------------|-----|
| Rotation matrix $R$ | 9 | $R^TR=I$, $\det=1$ | 3 |
| Euler angles $(\phi,\theta,\psi)$ | 3 | None | 3 |
| Axis-angle $\omega = \theta\hat{n}$ | 3 | $\|\hat{n}\|=1$, $\theta\in[0,\pi]$ | 3 |
| Quaternion $q = (w,x,y,z)$ | 4 | $\|q\|=1$ | 3 |

### 76.2 Euler Angles and Gimbal Lock

Euler angles decompose rotation into three sequential rotations about fixed or body axes. The most common convention (ZYX / yaw-pitch-roll):

$$R = R_z(\psi)\,R_y(\theta)\,R_x(\phi)$$

**Gimbal lock** occurs when $\theta = \pm 90°$: the first and third axes align, losing one degree of freedom. At $\theta = \pi/2$:

$$R = \begin{pmatrix} 0 & -\sin(\psi-\phi) & \cos(\psi-\phi) \\ 0 & \cos(\psi-\phi) & \sin(\psi-\phi) \\ -1 & 0 & 0 \end{pmatrix}$$

Only $\psi - \phi$ is observable — individual yaw and roll are indistinguishable.

### 76.3 Axis-Angle Singularities

Axis-angle has a singularity at $\theta = 0$ (axis undefined) and a discontinuity at $\theta = \pi$ (axis direction ambiguous). The exponential map handles these via Taylor expansion and eigenvector extraction, respectively.

### 76.4 Comprehensive Comparison

| Criterion | Euler | Axis-Angle | Quaternion | Matrix |
|-----------|-------|-----------|-----------|--------|
| **Memory** | 3 floats | 3 floats | 4 floats | 9 floats |
| **Singularities** | Gimbal lock | θ=0, θ=π | None | None |
| **Composition** | 3 trig evals | exp/log | 16 multiplies | 27 multiplies |
| **Interpolation** | Non-trivial | Via exp/log | SLERP | Via log/exp |
| **Human-readable** | Yes | Somewhat | No | No |
| **Normalization** | N/A | Normalize axis | $\|q\|=1$ | Gram-Schmidt |
| **Double cover** | No | No | Yes ($q=-q$) | No |

### 76.5 When to Use What

- **Storage/communication:** Quaternion (4 floats, no singularity, easy normalization)
- **Human display:** Euler angles (interpretable yaw/pitch/roll)
- **Computation:** Rotation matrix (direct vector transformation)
- **Incremental updates:** Axis-angle (natural for angular velocity integration)

---

## Implementation (60 min)

```python
import numpy as np
from code.lie_groups.lie_groups import (
    so3_exp, so3_log, hat,
    angle_axis_to_quaternion, quaternion_to_rotation
)

def euler_to_rotation(phi, theta, psi):
    """ZYX Euler angles (roll, pitch, yaw) → rotation matrix."""
    Rx = np.array([[1, 0, 0], [0, np.cos(phi), -np.sin(phi)],
                   [0, np.sin(phi), np.cos(phi)]])
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                   [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def rotation_to_euler(R):
    """Rotation matrix → ZYX Euler angles (roll, pitch, yaw).
    WARNING: singular at pitch = ±π/2."""
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    if not singular:
        phi = np.arctan2(R[2,1], R[2,2])
        theta = np.arctan2(-R[2,0], sy)
        psi = np.arctan2(R[1,0], R[0,0])
    else:
        phi = np.arctan2(-R[1,2], R[1,1])
        theta = np.arctan2(-R[2,0], sy)
        psi = 0.0
    return phi, theta, psi

def axis_angle_to_quaternion_manual(omega):
    """Axis-angle → quaternion (manual implementation)."""
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.array([1.0, 0, 0, 0])
    axis = omega / theta
    return np.array([np.cos(theta/2), *(np.sin(theta/2) * axis)])

# --- Gimbal lock demonstration ---
print("=== Gimbal Lock Demonstration ===")
for pitch_deg in [0, 45, 89, 89.99, 90]:
    pitch = np.radians(pitch_deg)
    R = euler_to_rotation(np.radians(30), pitch, np.radians(60))
    phi_r, theta_r, psi_r = rotation_to_euler(R)
    print(f"  pitch={pitch_deg:6.2f}° → recovered: roll={np.degrees(phi_r):7.2f}°,"
          f" pitch={np.degrees(theta_r):7.2f}°, yaw={np.degrees(psi_r):7.2f}°")

# --- Conversion roundtrips ---
print(f"\n=== Conversion Roundtrips ===")
omega_test = np.array([0.3, -0.7, 1.2])
R_from_aa = so3_exp(omega_test)
q_from_aa = angle_axis_to_quaternion(omega_test)
R_from_q = quaternion_to_rotation(q_from_aa)
euler = rotation_to_euler(R_from_aa)
R_from_euler = euler_to_rotation(*euler)

print(f"axis-angle → matrix → euler → matrix error: "
      f"{np.max(np.abs(R_from_aa - R_from_euler)):.2e}")
print(f"axis-angle → quat → matrix error: "
      f"{np.max(np.abs(R_from_aa - R_from_q)):.2e}")

# --- Composition speed comparison ---
import time

N = 100_000
omega_a = np.array([0.1, 0.2, -0.15])
omega_b = np.array([-0.3, 0.1, 0.25])
R_a, R_b = so3_exp(omega_a), so3_exp(omega_b)
q_a = angle_axis_to_quaternion(omega_a)
q_b = angle_axis_to_quaternion(omega_b)

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

t0 = time.perf_counter()
for _ in range(N):
    _ = R_a @ R_b
t_mat = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(N):
    _ = quat_mul(q_a, q_b)
t_quat = time.perf_counter() - t0

print(f"\n=== Composition Benchmark ({N} iterations) ===")
print(f"  Matrix multiply: {t_mat*1000:.1f} ms")
print(f"  Quaternion multiply: {t_quat*1000:.1f} ms")
print(f"  Ratio (mat/quat): {t_mat/t_quat:.2f}x")
```

---

## Practice Problems (45 min)

**Problem 1:** A robot has Euler angles (roll=0°, pitch=89°, yaw=45°). Apply a small additional pitch of 2°. What happens to the recovered Euler angles?

<details><summary>Answer</summary>

At pitch=91° (past gimbal lock at 90°), the Euler angle extraction becomes numerically unstable. Roll and yaw will show large jumps even though the actual rotation changed by only 2°. The quaternion representation handles this smoothly — no jumps.
</details>

**Problem 2:** You need to store 1 million orientations. Compare memory usage for each representation.

<details><summary>Answer</summary>

- Euler: $3 \times 4 = 12$ bytes each → 12 MB
- Axis-angle: $3 \times 4 = 12$ bytes → 12 MB
- Quaternion: $4 \times 4 = 16$ bytes → 16 MB
- Matrix: $9 \times 4 = 36$ bytes → 36 MB

Euler/axis-angle use least memory but have singularities. Quaternions cost 33% more but are singularity-free. Matrices cost 3× more.
</details>

**Problem 3:** Convert the quaternion $q = (0.5, 0.5, 0.5, 0.5)$ to Euler angles (ZYX) and to axis-angle.

<details><summary>Answer</summary>

$\|q\|=1$ ✓. Rotation angle: $\theta = 2\arccos(0.5) = 120°$. Axis: $(0.5, 0.5, 0.5)/\sin(60°) = (1,1,1)/\sqrt{3}$.

Axis-angle: $\omega = (120° / \sqrt{3})(1,1,1) \approx (1.209, 1.209, 1.209)$ rad.

Euler: compute $R$ first, then extract — roll ≈ 90°, pitch ≈ 0°, yaw ≈ 90° (convention-dependent).
</details>

---

## Expert Challenges

**Challenge 1:** Prove that no 3-parameter representation of SO(3) can be globally singularity-free (topological obstruction).

<details><summary>Answer</summary>

SO(3) is not homeomorphic to any subset of $\mathbb{R}^3$ because SO(3) $\cong \mathbb{RP}^3$ is compact and has fundamental group $\mathbb{Z}/2$, while any open subset of $\mathbb{R}^3$ is simply connected. By the hairy ball theorem generalized: you cannot parameterize a compact non-contractible manifold by a single chart without singularities. This is why 4-parameter quaternions (with 1 constraint) or 9-parameter matrices (with 6 constraints) are needed for global representations.
</details>

**Challenge 2:** Implement a function that detects when a rotation is near gimbal lock and switches from Euler to quaternion representation automatically.

<details><summary>Answer</summary>

```python
def safe_orientation(R, gimbal_threshold=np.radians(85)):
    _, theta, _ = rotation_to_euler(R)
    if abs(theta) > gimbal_threshold:
        q = angle_axis_to_quaternion(so3_log(R))
        return {"repr": "quaternion", "value": q, "warning": "near gimbal lock"}
    phi, theta, psi = rotation_to_euler(R)
    return {"repr": "euler_deg", "value": np.degrees([phi, theta, psi])}
```
</details>

**Challenge 3:** The condition number of the Euler → matrix Jacobian blows up at gimbal lock. Compute this Jacobian and show where it becomes singular.

<details><summary>Answer</summary>

The Jacobian $J = \partial R / \partial(\phi,\theta,\psi)$ has 9 rows (matrix entries) and 3 columns. At $\theta = \pm\pi/2$, $\partial R/\partial\phi$ and $\partial R/\partial\psi$ become linearly dependent (they produce the same infinitesimal rotation), so $\text{rank}(J) = 2$ instead of 3. The condition number $\to \infty$.
</details>

---

## Connections
- **Prerequisites:** [Day 72](day-72-so2-rotations.md) SO(2), [Day 73](day-73-so3-rodrigues.md) Rodrigues, [Day 74](day-74-quaternions.md) quaternions, [Day 75](day-75-quaternion-operations.md) SLERP
- **Forward:** [Day 77](day-77-week11-review.md) — unified review and full conversion pipeline
- **OKS:** Internal navigation uses quaternions; sensorbar heading uses SO(2); user-facing logs show Euler; IMU increments use axis-angle

## Self-Check
- [ ] I can explain gimbal lock geometrically and identify it numerically
- [ ] I can convert between all four representations correctly
- [ ] I know which representation to choose for each use case (storage, display, computation, update)
- [ ] I understand the topological reason why 3-parameter representations must have singularities
