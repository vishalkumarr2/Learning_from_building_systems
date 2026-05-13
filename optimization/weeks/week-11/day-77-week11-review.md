# Day 77: Week 11 Review and Exercise Session

> **Phase VI — Lie Groups & Manifold Optimization | Week 11 | 2.5 hours**
> *Seven days, one unified rotation toolkit. Prove you own it.*

## Navigation
- **Previous:** [Day 76: Rotation Representations Compared](day-76-rotation-representations.md)
- **Next:** [Day 78: SE(2) — Planar Rigid Motions (Week 12)](../week-12/day-78-se2-rigid-motions.md)
- **Week:** [Week 11 Overview](../README.md)
- **Chapter:** [Chapter 09: Lie Groups Reference](../../09-lie-groups.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
This review consolidates everything needed to reason about robot orientation: from abstract group axioms to the practical choice of quaternions in `tf2`, from Rodrigues' formula in the IMU driver to SLERP in the trajectory planner. The full pipeline — read IMU quaternion → convert → compose → display Euler — runs every control cycle on every OKS AMR.

---

## Theory (45 min)

### Week 11 Concept Map

```
Group Axioms (Day 71)
    ├── Matrix Groups: GL(n) ⊃ O(n) ⊃ SO(n)
    │
    ├── SO(2) (Day 72)
    │   ├── 2×2 rotation matrix
    │   ├── exp/log maps (angle ↔ matrix)
    │   └── Abelian — rotations commute
    │
    └── SO(3) (Days 73–76)
        ├── Rodrigues' Formula (Day 73)
        │   ├── axis-angle → matrix via exp map
        │   └── hat/vee operators
        ├── Quaternions (Day 74)
        │   ├── unit quaternion ↔ rotation
        │   └── double cover (q = -q)
        ├── SLERP & Averaging (Day 75)
        │   ├── geodesic interpolation
        │   └── Fréchet mean on S³
        └── Representations Compared (Day 76)
            ├── Euler angles (gimbal lock)
            ├── axis-angle (singularity at θ=0,π)
            ├── quaternion (double cover)
            └── matrix (9 params, 6 constraints)
```

### Key Formulas Summary

| Formula | Expression |
|---------|-----------|
| SO(2) exp | $R(\theta) = \begin{pmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{pmatrix}$ |
| SO(2) log | $\theta = \text{atan2}(R_{10}, R_{00})$ |
| Hat operator | $[\omega]_\times = \begin{pmatrix}0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0\end{pmatrix}$ |
| Rodrigues | $R = I + \sin\theta\,[\hat{n}]_\times + (1-\cos\theta)\,[\hat{n}]_\times^2$ |
| SO(3) log | $\theta = \arccos\frac{\text{tr}(R)-1}{2}$, $\omega = \frac{\theta}{2\sin\theta}\text{vee}(R-R^T)$ |
| Quat → angle | $\theta = 2\arccos\|w\|$ |
| SLERP | $q(t) = \frac{\sin((1-t)\Omega)}{\sin\Omega}q_0 + \frac{\sin(t\Omega)}{\sin\Omega}q_1$ |

---

## Implementation (60 min)

### Full Rotation Converter Tool

```python
import numpy as np
from code.lie_groups.lie_groups import (
    so3_exp, so3_log, hat, vee,
    angle_axis_to_quaternion, quaternion_to_rotation, quaternion_slerp
)

class RotationConverter:
    """Unified rotation representation converter."""

    @staticmethod
    def euler_to_matrix(roll, pitch, yaw):
        """ZYX Euler (radians) → 3×3 rotation matrix."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr]
        ])

    @staticmethod
    def matrix_to_euler(R):
        """3×3 rotation matrix → ZYX Euler angles (roll, pitch, yaw)."""
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        if sy > 1e-6:
            roll = np.arctan2(R[2,1], R[2,2])
            pitch = np.arctan2(-R[2,0], sy)
            yaw = np.arctan2(R[1,0], R[0,0])
        else:
            roll = np.arctan2(-R[1,2], R[1,1])
            pitch = np.arctan2(-R[2,0], sy)
            yaw = 0.0
        return roll, pitch, yaw

    @staticmethod
    def axis_angle_to_matrix(omega):
        return so3_exp(np.asarray(omega, dtype=float))

    @staticmethod
    def matrix_to_axis_angle(R):
        return so3_log(R)

    @staticmethod
    def quaternion_to_matrix(q):
        return quaternion_to_rotation(np.asarray(q, dtype=float))

    @staticmethod
    def matrix_to_quaternion(R):
        tr = np.trace(R)
        if tr > 0:
            s = 2.0 * np.sqrt(tr + 1.0)
            return np.array([0.25*s, (R[2,1]-R[1,2])/s,
                             (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s])
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = 2.0 * np.sqrt(1+R[0,0]-R[1,1]-R[2,2])
            return np.array([(R[2,1]-R[1,2])/s, 0.25*s,
                             (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
        elif R[1,1] > R[2,2]:
            s = 2.0 * np.sqrt(1+R[1,1]-R[0,0]-R[2,2])
            return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s,
                             0.25*s, (R[1,2]+R[2,1])/s])
        else:
            s = 2.0 * np.sqrt(1+R[2,2]-R[0,0]-R[1,1])
            return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s,
                             (R[1,2]+R[2,1])/s, 0.25*s])

    def convert(self, value, from_repr, to_repr):
        """Convert between any two representations.
        Representations: 'euler', 'axis_angle', 'quaternion', 'matrix'
        """
        # First convert to matrix as canonical form
        if from_repr == 'euler':
            R = self.euler_to_matrix(*value)
        elif from_repr == 'axis_angle':
            R = self.axis_angle_to_matrix(value)
        elif from_repr == 'quaternion':
            R = self.quaternion_to_matrix(value)
        elif from_repr == 'matrix':
            R = np.array(value)
        else:
            raise ValueError(f"Unknown repr: {from_repr}")
        # Then convert from matrix to target
        if to_repr == 'matrix':
            return R
        elif to_repr == 'euler':
            return self.matrix_to_euler(R)
        elif to_repr == 'axis_angle':
            return self.matrix_to_axis_angle(R)
        elif to_repr == 'quaternion':
            return self.matrix_to_quaternion(R)
        else:
            raise ValueError(f"Unknown repr: {to_repr}")

# --- Demo: full pipeline ---
rc = RotationConverter()

print("=== Full Conversion Pipeline ===")
# Start with IMU quaternion (ROS convention: x,y,z,w → our convention: w,x,y,z)
imu_quat_ros = [0.0, 0.0, 0.383, 0.924]  # ~45° around z
imu_quat = [imu_quat_ros[3], *imu_quat_ros[:3]]  # convert to w,x,y,z

R = rc.convert(imu_quat, 'quaternion', 'matrix')
euler = rc.convert(imu_quat, 'quaternion', 'euler')
aa = rc.convert(imu_quat, 'quaternion', 'axis_angle')

print(f"IMU quaternion (w,x,y,z): [{imu_quat[0]:.3f}, {imu_quat[1]:.3f},"
      f" {imu_quat[2]:.3f}, {imu_quat[3]:.3f}]")
print(f"Rotation matrix:\n{np.round(R, 4)}")
print(f"Euler (deg): roll={np.degrees(euler[0]):.1f}, pitch={np.degrees(euler[1]):.1f},"
      f" yaw={np.degrees(euler[2]):.1f}")
print(f"Axis-angle: {np.round(aa, 4)} (angle={np.degrees(np.linalg.norm(aa)):.1f}°)")

# --- Compose: apply an additional rotation ---
delta_omega = np.array([0.0, 0.0, np.radians(15)])  # 15° yaw increment
R_delta = rc.axis_angle_to_matrix(delta_omega)
R_composed = R @ R_delta
euler_new = rc.matrix_to_euler(R_composed)
print(f"\nAfter +15° yaw: roll={np.degrees(euler_new[0]):.1f}°,"
      f" pitch={np.degrees(euler_new[1]):.1f}°, yaw={np.degrees(euler_new[2]):.1f}°")

# --- Roundtrip error check ---
print(f"\n=== Roundtrip Error Check ===")
reprs = ['euler', 'axis_angle', 'quaternion']
for src in reprs:
    if src == 'euler':
        val = (0.3, -0.2, 1.1)
    elif src == 'axis_angle':
        val = np.array([0.3, -0.2, 1.1])
    else:
        val = angle_axis_to_quaternion([0.3, -0.2, 1.1])
    for dst in reprs:
        if dst == src:
            continue
        intermediate = rc.convert(val, src, dst)
        back = rc.convert(intermediate, dst, src)
        if src == 'euler':
            err = max(abs(a-b) for a, b in zip(val, back))
        else:
            err = np.max(np.abs(np.array(val) - np.array(back)))
        print(f"  {src:12s} → {dst:12s} → {src:12s}: max error = {err:.2e}")
```

---

## Practice Problems (45 min)

### Exercise Set 08 — Section A: Rapid-Fire

**Q1:** What is $\det(R)$ for any $R \in \text{SO}(3)$?
<details><summary>Answer</summary>+1</details>

**Q2:** How many DOF does a 3D rotation have?
<details><summary>Answer</summary>3</details>

**Q3:** What is the identity quaternion?
<details><summary>Answer</summary>$(1, 0, 0, 0)$</details>

**Q4:** If $q$ represents a rotation, what does $-q$ represent?
<details><summary>Answer</summary>The same rotation (double cover).</details>

**Q5:** At what Euler pitch angle does gimbal lock occur?
<details><summary>Answer</summary>$\pm 90°$ ($\pm\pi/2$)</details>

**Q6:** What is $\text{tr}(R)$ for a rotation by angle $\theta$?
<details><summary>Answer</summary>$1 + 2\cos\theta$</details>

### Exercise Set 08 — Section B: Worked Problems

**B1:** An IMU outputs $q = (0.707, 0, 0, 0.707)$. What is this rotation in Euler angles?

<details><summary>Answer</summary>

$w = 0.707 = \cos(\theta/2)$, so $\theta = 90°$. Axis from vector part: $(0,0,1)$ (z-axis). This is a pure 90° yaw. Euler: roll=0°, pitch=0°, yaw=90°.
</details>

**B2:** Compose two rotations: 45° around x followed by 90° around z. Express the result as axis-angle.

<details><summary>Answer</summary>

$R = R_z(90°) R_x(45°)$. Compute numerically: $\omega = \text{so3\_log}(R) \approx (0.459, -0.459, 1.209)$ rad, angle $\approx 104.5°$ around axis $\approx (0.34, -0.34, 0.88)$.
</details>

**B3:** SLERP between $q_0 = (1,0,0,0)$ and $q_1 = (0,1,0,0)$ at $t=0.5$. What rotation does this represent?

<details><summary>Answer</summary>

$q_1$ represents 180° around x. The midpoint is 90° around x: $q(0.5) = (\cos45°, \sin45°, 0, 0) = (0.707, 0.707, 0, 0)$.
</details>

---

## Expert Challenges

**Challenge 1:** Prove that SO(3) is a 3-dimensional smooth manifold.

<details><summary>Answer</summary>

SO(3) is defined by $f(R) = R^TR - I = 0$ and $\det(R) = 1$. The constraint $R^TR = I$ gives 6 independent equations (the upper triangle of the symmetric matrix $R^TR - I$). Since $R$ has 9 entries and 6 constraints, $\dim(\text{SO}(3)) = 9 - 6 = 3$. The Jacobian of the constraint map has full rank 6 everywhere on SO(3), so by the implicit function theorem, SO(3) is a smooth 3-dimensional submanifold of $\mathbb{R}^{3\times 3}$.
</details>

**Challenge 2:** Derive the Haar (uniform) measure on SO(3) in axis-angle coordinates: $d\mu = \frac{2(1-\cos\theta)}{\pi\theta^2}\,d\theta\,d\hat{n}$.

<details><summary>Answer</summary>

The volume element comes from the Jacobian of the exponential map. In axis-angle coordinates $(\theta, \hat{n})$, the metric tensor determinant gives $\sqrt{g} = \frac{2(1-\cos\theta)}{\theta^2}$. Integrating: $\text{Vol}(\text{SO}(3)) = \int_0^\pi \frac{2(1-\cos\theta)}{\theta^2} \cdot 4\pi\theta^2\,d\theta = 8\pi^2$. Normalizing gives the stated density.
</details>

**Challenge 3:** Implement uniform random sampling on SO(3) using the quaternion method (4 Gaussian random variables, normalize).

<details><summary>Answer</summary>

```python
def random_rotation():
    """Uniformly random rotation via quaternion."""
    q = np.random.randn(4)
    q = q / np.linalg.norm(q)
    return quaternion_to_rotation(q)

# Verify: average trace should be 1 + 2*E[cos θ] = 1 (for uniform)
traces = [np.trace(random_rotation()) for _ in range(10000)]
print(f"Mean trace: {np.mean(traces):.3f} (expected: ~1.0 for uniform)")
```
This works because the standard 4D Gaussian, when normalized, is uniform on $S^3$.
</details>

---

## Connections
- **All Week 11 Days:** [71](day-71-group-axioms.md) → [72](day-72-so2-rotations.md) → [73](day-73-so3-rodrigues.md) → [74](day-74-quaternions.md) → [75](day-75-quaternion-operations.md) → [76](day-76-rotation-representations.md) → **77**
- **Forward:** Week 12 extends to SE(2) and SE(3) — rigid-body motions (rotation + translation)
- **OKS pipeline:** IMU publishes quaternion → estimator composes via matrix multiply → planner interpolates via SLERP → logger displays Euler angles

## Self-Check
- [ ] I can convert between all four rotation representations without reference
- [ ] I can identify gimbal lock and explain the topological reason it is unavoidable with 3 parameters
- [ ] I can implement SLERP and explain its constant-velocity property
- [ ] I understand the full OKS orientation pipeline: IMU → quaternion → matrix → compose → Euler display
- [ ] I can state the Rodrigues formula and derive it from the matrix exponential
