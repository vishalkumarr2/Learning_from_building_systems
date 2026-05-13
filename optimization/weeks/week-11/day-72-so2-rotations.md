# Day 72: SO(2) — 2D Rotations

> **Phase VI — Lie Groups & Manifold Optimization | Week 11 | 2.5 hours**
> *The simplest Lie group is hiding in every AMR heading estimate.*

## Navigation
- **Previous:** [Day 71: Group Axioms and Matrix Groups](day-71-group-axioms.md)
- **Next:** [Day 73: SO(3) — Rodrigues' Formula](day-73-so3-rodrigues.md)
- **Week:** [Week 11 Overview](../README.md)
- **Chapter:** [Chapter 09: Lie Groups Reference](../../09-lie-groups.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Every OKS AMR operates on a 2D plane. The robot heading $\theta$ is an element of SO(2). The navigation estimator fuses wheel odometry and sensorbar data to estimate heading — both produce rotation increments that compose via SO(2) multiplication. Getting angle wrapping wrong is a classic source of estimator bugs in the OKS stack.

---

## Theory (45 min)

### 72.1 Rotation Matrices in 2D

A rotation by angle $\theta$ in the plane is represented by:

$$R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

**Properties:**
- $R(\theta)^T R(\theta) = I$ (orthogonal)
- $\det R(\theta) = \cos^2\theta + \sin^2\theta = 1$
- $R(\theta)$ maps the unit circle to itself

### 72.2 Group Structure of SO(2)

SO(2) is a **commutative** (abelian) group:

$$R(\alpha) R(\beta) = R(\alpha + \beta) = R(\beta) R(\alpha)$$

This is the *only* SO(n) that is abelian — 3D rotations do not commute.

| Property | Expression |
|----------|-----------|
| Composition | $R(\alpha) R(\beta) = R(\alpha + \beta)$ |
| Identity | $R(0) = I$ |
| Inverse | $R(\theta)^{-1} = R(-\theta) = R(\theta)^T$ |
| Periodicity | $R(\theta + 2\pi) = R(\theta)$ |

### 72.3 Exponential and Logarithmic Maps

The **Lie algebra** $\mathfrak{so}(2)$ consists of $2 \times 2$ skew-symmetric matrices:

$$\mathfrak{so}(2) = \left\{ \begin{pmatrix} 0 & -\theta \\ \theta & 0 \end{pmatrix} : \theta \in \mathbb{R} \right\}$$

The **exponential map** connects algebra to group:

$$\exp\left(\begin{pmatrix} 0 & -\theta \\ \theta & 0 \end{pmatrix}\right) = R(\theta)$$

The **logarithmic map** is the inverse: $\log(R) \to \theta = \text{atan2}(R_{10}, R_{00})$.

### 72.4 Angle Wrapping and the Topology of SO(2)

SO(2) is topologically a **circle** $S^1$. The angle parameterization $\theta \in (-\pi, \pi]$ has a discontinuity at $\pm\pi$. Computing angular differences requires wrapping:

$$\Delta\theta = \text{atan2}(\sin(\theta_2 - \theta_1), \cos(\theta_2 - \theta_1))$$

This is exactly what `so2_log(R_1^T R_2)` computes — no manual wrapping needed.

---

## Implementation (60 min)

```python
import numpy as np
from code.lie_groups.lie_groups import so2_exp, so2_log, so2_compose

# --- SO(2) basics ---
theta1 = np.pi / 3   # 60 degrees
theta2 = np.pi / 4   # 45 degrees

R1 = so2_exp(theta1)
R2 = so2_exp(theta2)
R12 = so2_compose(R1, R2)

print("=== SO(2) Composition ===")
print(f"R(60°) @ R(45°) = R({np.degrees(so2_log(R12)):.1f}°)")
print(f"Expected: R(105°)")

# --- Verify group properties ---
I = np.eye(2)
print(f"\nR @ R^T = I? {np.allclose(R1 @ R1.T, I)}")
print(f"det(R) = {np.linalg.det(R1):.6f}")

# --- Angle wrapping via log map ---
def angle_diff_naive(a, b):
    """Wrong way: simple subtraction."""
    return b - a

def angle_diff_so2(a, b):
    """Right way: via SO(2) log map."""
    return so2_log(so2_exp(a).T @ so2_exp(b))

a = 3.0   # near +π
b = -3.0  # near -π (actually close to a on the circle)
print(f"\n=== Angle Wrapping ===")
print(f"Naive diff: {angle_diff_naive(a, b):.4f} rad ({np.degrees(angle_diff_naive(a, b)):.1f}°)")
print(f"SO(2) diff: {angle_diff_so2(a, b):.4f} rad ({np.degrees(angle_diff_so2(a, b)):.1f}°)")

# --- Interpolation on SO(2) ---
def so2_interpolate(R1, R2, t):
    """Geodesic interpolation: R(t) = R1 * exp(t * log(R1^T R2))."""
    delta = so2_log(R1.T @ R2)
    return so2_compose(R1, so2_exp(t * delta))

R_start = so2_exp(0.0)
R_end = so2_exp(np.pi * 0.75)
print(f"\n=== SO(2) Interpolation ===")
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    R_t = so2_interpolate(R_start, R_end, t)
    print(f"  t={t:.2f}: θ = {np.degrees(so2_log(R_t)):7.2f}°")

# --- Rotating a vector ---
v = np.array([1.0, 0.0])
R90 = so2_exp(np.pi / 2)
print(f"\n=== Rotating [1,0] by 90° ===")
print(f"Result: {R90 @ v}  (expected [0, 1])")
```

---

## Practice Problems (45 min)

**Problem 1:** Compute $R(30°) \cdot R(60°)$ by hand using the angle-addition formula for cosine and sine. Verify it equals $R(90°)$.

<details><summary>Answer</summary>

$R(30°)R(60°)$ entry $(0,0)$: $\cos30\cos60 - \sin30\sin60 = \frac{\sqrt3}{2}\cdot\frac{1}{2} - \frac{1}{2}\cdot\frac{\sqrt3}{2} = 0 = \cos90°$. ✓

Similarly for all entries. The composition is $R(90°)$ as predicted by $R(\alpha)R(\beta) = R(\alpha+\beta)$.
</details>

**Problem 2:** An AMR heading reads $\theta_1 = 170°$ at time $t_1$ and $\theta_2 = -170°$ at time $t_2$. What is the actual angular change? Use both the naive and SO(2) methods.

<details><summary>Answer</summary>

Naive: $-170 - 170 = -340°$ (wrong — implies nearly a full turn).
Convert to radians: $\theta_1 = 170 \cdot \pi/180$, $\theta_2 = -170 \cdot \pi/180$.
SO(2): $\Delta\theta = \text{atan2}(\sin(-340°), \cos(-340°)) = \text{atan2}(\sin(20°), \cos(20°)) = 20°$.
The robot turned just 20° clockwise.
</details>

**Problem 3:** Show that the average of two angles cannot be computed as $(\theta_1 + \theta_2)/2$ in general. Give a counter-example and the correct formula.

<details><summary>Answer</summary>

Counter-example: $\theta_1 = 179°$, $\theta_2 = -179°$. Arithmetic mean = $0°$, but the true angular midpoint is $180°$ (or $-180°$).

Correct: compute via $\text{atan2}\!\left(\frac{\sin\theta_1 + \sin\theta_2}{2}, \frac{\cos\theta_1 + \cos\theta_2}{2}\right)$, or use SO(2) interpolation at $t=0.5$.
</details>

---

## Expert Challenges

**Challenge 1:** Prove that SO(2) is isomorphic to the unit circle $S^1 = \{z \in \mathbb{C} : |z| = 1\}$ under complex multiplication.

<details><summary>Answer</summary>

Define $\phi: S^1 \to \text{SO}(2)$ by $\phi(e^{i\theta}) = R(\theta)$. Then $\phi(e^{i\alpha} \cdot e^{i\beta}) = \phi(e^{i(\alpha+\beta)}) = R(\alpha+\beta) = R(\alpha)R(\beta) = \phi(e^{i\alpha})\phi(e^{i\beta})$. The map is bijective and preserves the group operation, so it is an isomorphism. ✓
</details>

**Challenge 2:** Derive the exponential map $\exp: \mathfrak{so}(2) \to \text{SO}(2)$ using the matrix power series $e^A = \sum_{k=0}^{\infty} A^k / k!$.

<details><summary>Answer</summary>

Let $\Omega = \begin{pmatrix} 0 & -\theta \\ \theta & 0 \end{pmatrix}$. Then $\Omega^2 = -\theta^2 I$, $\Omega^3 = -\theta^2 \Omega$, etc.

$$e^\Omega = I\sum_{k\,\text{even}} \frac{(-1)^{k/2} \theta^k}{k!} + \frac{\Omega}{\theta}\sum_{k\,\text{odd}} \frac{(-1)^{(k-1)/2} \theta^k}{k!} = I\cos\theta + \frac{\Omega}{\theta}\sin\theta = R(\theta)$$
</details>

**Challenge 3:** The **winding number** of a path $\theta(t)$ for $t \in [0, T]$ is $n = \frac{1}{2\pi}\int_0^T \dot{\theta}(t)\,dt$. If a robot drives in a figure-eight, what is the winding number? Why does this matter for SLAM?

<details><summary>Answer</summary>

For a figure-eight, $n = 0$ — the two loops cancel. This matters for SLAM because accumulated heading drift is proportional to total rotation; a figure-eight path accumulates less drift than a single loop of the same total path length.
</details>

---

## Connections
- **Prerequisites:** [Day 71](day-71-group-axioms.md) — group axioms and matrix groups
- **Forward:** [Day 73](day-73-so3-rodrigues.md) extends to 3D rotations, [Day 76](day-76-rotation-representations.md) compares all representations
- **OKS:** SO(2) directly represents robot heading; every `cmd_vel` angular velocity integrates on SO(2)

## Self-Check
- [ ] I can write the 2D rotation matrix from memory and verify its properties
- [ ] I understand why angle wrapping bugs occur and how SO(2) log avoids them
- [ ] I can interpolate between two rotations on SO(2) using the exponential map
- [ ] I can explain why SO(2) is abelian but SO(3) is not
