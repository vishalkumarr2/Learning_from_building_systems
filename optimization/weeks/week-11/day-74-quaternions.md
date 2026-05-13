# Day 74: Quaternions for Rotation

> **Phase VI — Lie Groups & Manifold Optimization | Week 11 | 2.5 hours**
> *Four numbers, zero gimbal lock, double the coverage.*

## Navigation
- **Previous:** [Day 73: SO(3) — Rodrigues' Formula](day-73-so3-rodrigues.md)
- **Next:** [Day 75: Quaternion Interpolation and Averaging](day-75-quaternion-operations.md)
- **Week:** [Week 11 Overview](../README.md)
- **Chapter:** [Chapter 09: Lie Groups Reference](../../09-lie-groups.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
ROS `tf2` represents all orientations as quaternions `[x, y, z, w]`. The OKS IMU publishes orientation in quaternion form. Every `geometry_msgs/Quaternion` in the navigation stack is a unit quaternion. Understanding the double-cover property explains why `q` and `-q` represent the same rotation — a frequent source of sign-flip bugs in sensor fusion.

---

## Theory (45 min)

### 74.1 Hamilton's Quaternions

A quaternion is $q = w + xi + yj + zk$ where $i, j, k$ satisfy:

$$i^2 = j^2 = k^2 = ijk = -1$$

From which: $ij = k$, $jk = i$, $ki = j$, and $ji = -k$, $kj = -i$, $ik = -j$.

We write $q = (w, \mathbf{v})$ with scalar part $w$ and vector part $\mathbf{v} = (x, y, z)$.

**Quaternion multiplication:**

$$(w_1, \mathbf{v}_1)(w_2, \mathbf{v}_2) = (w_1 w_2 - \mathbf{v}_1 \cdot \mathbf{v}_2,\; w_1\mathbf{v}_2 + w_2\mathbf{v}_1 + \mathbf{v}_1 \times \mathbf{v}_2)$$

Note: multiplication is **non-commutative** (due to the cross product term).

### 74.2 Unit Quaternions and Rotations

A **unit quaternion** has $\|q\| = \sqrt{w^2 + x^2 + y^2 + z^2} = 1$.

For axis $\hat{n}$ and angle $\theta$:

$$q = \left(\cos\frac{\theta}{2},\; \sin\frac{\theta}{2}\,\hat{n}\right)$$

The rotation of vector $\mathbf{p}$ is: $\mathbf{p}' = q \mathbf{p} q^*$ (sandwich product), where $q^* = (w, -\mathbf{v})$ is the conjugate.

### 74.3 Double Cover

Both $q$ and $-q$ represent the **same rotation**:

$$q \mathbf{p} q^* = (-q) \mathbf{p} (-q)^*$$

This means unit quaternions form a **double cover** of SO(3): the map $\phi: S^3 \to \text{SO}(3)$ is 2-to-1. Topologically, SO(3) $\cong$ $S^3 / \{q \sim -q\} \cong \mathbb{RP}^3$.

### 74.4 Quaternion to/from Rotation Matrix

**Quaternion → matrix:**

$$R = \begin{pmatrix} 1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy) \\ 2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx) \\ 2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2) \end{pmatrix}$$

**Matrix → quaternion** (Shepperd's method): compute from the trace and diagonal entries, choosing the numerically stable branch.

### 74.5 Advantages over Euler Angles

| Property | Euler Angles | Quaternions |
|----------|-------------|-------------|
| Parameters | 3 | 4 (with 1 constraint) |
| Singularities | Gimbal lock at $\pm 90°$ pitch | None |
| Composition | Trig functions | Simple multiply |
| Interpolation | Non-trivial | SLERP (geodesic) |
| Normalization | N/A | Normalize to unit length |

---

## Implementation (60 min)

```python
import numpy as np
from code.lie_groups.lie_groups import (
    angle_axis_to_quaternion, quaternion_to_rotation,
    so3_exp, so3_log
)

def quaternion_multiply(q1, q2):
    """Multiply two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_conjugate(q):
    """Conjugate: (w, -v)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_rotate_vector(q, v):
    """Rotate vector v by unit quaternion q: q v q*."""
    p = np.array([0.0, v[0], v[1], v[2]])
    result = quaternion_multiply(quaternion_multiply(q, p), quaternion_conjugate(q))
    return result[1:]

def rotation_matrix_to_quaternion(R):
    """Shepperd's method: SO(3) → unit quaternion [w,x,y,z]."""
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)

# --- Basic quaternion operations ---
omega_z = np.array([0, 0, np.pi / 2])  # 90° around z
q = angle_axis_to_quaternion(omega_z)
print("=== Axis-angle → Quaternion ===")
print(f"ω = [0, 0, π/2] → q = [{q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}]")
print(f"Expected: [cos(π/4), 0, 0, sin(π/4)] = [{np.cos(np.pi/4):.4f}, 0, 0, {np.sin(np.pi/4):.4f}]")

# --- Quaternion rotation of a vector ---
v = np.array([1.0, 0.0, 0.0])
v_rot = quaternion_rotate_vector(q, v)
print(f"\n=== Rotating [1,0,0] by 90° around z ===")
print(f"Result: {np.round(v_rot, 4)} (expected: [0, 1, 0])")

# --- Double cover demonstration ---
q_neg = -q
R_pos = quaternion_to_rotation(q)
R_neg = quaternion_to_rotation(q_neg)
print(f"\n=== Double Cover ===")
print(f"q and -q give same R? {np.allclose(R_pos, R_neg)}")

# --- Composition via quaternion multiply ---
q1 = angle_axis_to_quaternion([np.pi/4, 0, 0])  # 45° around x
q2 = angle_axis_to_quaternion([0, np.pi/4, 0])  # 45° around y
q12 = quaternion_multiply(q1, q2)
R12 = quaternion_to_rotation(q12)
R_direct = so3_exp([np.pi/4, 0, 0]) @ so3_exp([0, np.pi/4, 0])
print(f"\n=== Composition ===")
print(f"Quaternion multiply matches matrix multiply? {np.allclose(R12, R_direct)}")

# --- Roundtrip: R → q → R ---
R_test = so3_exp([0.3, -0.7, 1.2])
q_test = rotation_matrix_to_quaternion(R_test)
R_back = quaternion_to_rotation(q_test)
print(f"\n=== Roundtrip R → q → R ===")
print(f"Max error: {np.max(np.abs(R_test - R_back)):.2e}")
```

---

## Practice Problems (45 min)

**Problem 1:** Convert a $180°$ rotation around the axis $(1, 0, 0)$ to a quaternion. What is $q$? What is $-q$? Verify both give the same rotation matrix.

<details><summary>Answer</summary>

$\theta = \pi$, $\hat{n} = (1,0,0)$. $q = (\cos(\pi/2), \sin(\pi/2)(1,0,0)) = (0, 1, 0, 0)$. $-q = (0, -1, 0, 0)$.

Both yield $R = \text{diag}(1, -1, -1)$. ✓
</details>

**Problem 2:** Compute the quaternion product of $q_1 = (\frac{\sqrt2}{2}, \frac{\sqrt2}{2}, 0, 0)$ and $q_2 = (\frac{\sqrt2}{2}, 0, \frac{\sqrt2}{2}, 0)$ by hand.

<details><summary>Answer</summary>

Using the formula: $w = \frac{1}{2} - 0 - 0 - 0 = \frac{1}{2}$, $x = 0 + \frac{1}{2} + 0 - 0 = \frac{1}{2}$, $y = 0 - 0 + \frac{1}{2} + 0 = \frac{1}{2}$, $z = 0 + 0 - 0 + \frac{1}{2} = \frac{1}{2}$.

$q_1 q_2 = (\frac{1}{2}, \frac{1}{2}, \frac{1}{2}, \frac{1}{2})$. This is a $120°$ rotation around axis $(1,1,1)/\sqrt{3}$.
</details>

**Problem 3:** Why does ROS use the convention `[x, y, z, w]` while most math textbooks use `[w, x, y, z]`? Does it affect any computation?

<details><summary>Answer</summary>

It is purely a storage convention. ROS uses `[x,y,z,w]` for consistency with `geometry_msgs` (vector part first). All quaternion math is identical — you just need to keep track of which index holds $w$. Bugs arise when mixing conventions without conversion.
</details>

---

## Expert Challenges

**Challenge 1:** Prove that the set of unit quaternions $S^3$ with quaternion multiplication forms a group.

<details><summary>Answer</summary>

Closure: $\|q_1 q_2\| = \|q_1\|\|q_2\| = 1$ (quaternion norm is multiplicative). Associativity: inherited from quaternion algebra. Identity: $(1, 0, 0, 0)$. Inverse: $q^{-1} = q^*/\|q\|^2 = q^*$ for unit quaternions. ✓
</details>

**Challenge 2:** Derive the quaternion-to-rotation-matrix formula from the sandwich product $q \mathbf{p} q^*$.

<details><summary>Answer</summary>

Expand $q p q^*$ with $p = (0, \mathbf{v})$ and collect terms by $v_1, v_2, v_3$. The coefficient of each $v_i$ in the output gives column $i$ of $R$. After algebraic expansion: $R_{11} = 1 - 2(y^2+z^2)$, $R_{12} = 2(xy - wz)$, etc.
</details>

**Challenge 3:** The quaternion group and SO(3) have different topologies. Explain why a continuous path from $q$ to $-q$ on $S^3$ corresponds to a loop in SO(3) (the "plate trick" or "belt trick").

<details><summary>Answer</summary>

A path from $q$ to $-q$ on $S^3$ maps to a closed loop in SO(3) (since $q$ and $-q$ map to the same rotation). This loop is **not contractible** — SO(3) has fundamental group $\pi_1(\text{SO}(3)) = \mathbb{Z}/2$. However, going from $q$ to $-q$ and back (a $4\pi$ rotation) *is* contractible. This is the mathematical basis of the Dirac belt trick.
</details>

---

## Connections
- **Prerequisites:** [Day 73](day-73-so3-rodrigues.md) — axis-angle representation and Rodrigues' formula
- **Forward:** [Day 75](day-75-quaternion-operations.md) — SLERP and averaging; [Day 76](day-76-rotation-representations.md) — comparison of all representations
- **OKS:** Every `tf2::Quaternion` in the OKS stack is a unit quaternion; IMU drivers publish orientations as quaternions

## Self-Check
- [ ] I can convert between axis-angle and quaternion representations in both directions
- [ ] I understand the double-cover property and why `q` and `-q` encode the same rotation
- [ ] I can multiply quaternions and know why the result represents composition
- [ ] I can implement Shepperd's method for numerically stable matrix-to-quaternion conversion
