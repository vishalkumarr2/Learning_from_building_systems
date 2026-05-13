# Day 73: SO(3) — Rodrigues' Formula

> **Phase VI — Lie Groups & Manifold Optimization | Week 11 | 2.5 hours**
> *Three dimensions, three degrees of freedom, one elegant formula.*

## Navigation
- **Previous:** [Day 72: SO(2) — 2D Rotations](day-72-so2-rotations.md)
- **Next:** [Day 74: Quaternions for Rotation](day-74-quaternions.md)
- **Week:** [Week 11 Overview](../README.md)
- **Chapter:** [Chapter 09: Lie Groups Reference](../../09-lie-groups.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
The OKS AMR body frame has a full 3D orientation relative to the world frame. The IMU measures angular velocity as a 3-vector $\omega \in \mathbb{R}^3$. Converting this to a rotation matrix requires the exponential map on SO(3) — which is precisely the Rodrigues formula. The navigation estimator uses this every cycle.

---

## Theory (45 min)

### 73.1 The Rotation Group SO(3)

$$\text{SO}(3) = \{R \in \mathbb{R}^{3\times 3} : R^T R = I,\; \det(R) = +1\}$$

**Key facts:**
- SO(3) has **3 degrees of freedom** (9 entries, 6 constraints from $R^TR = I$)
- SO(3) is **non-abelian**: $R_x R_z \neq R_z R_x$ in general
- SO(3) is **compact** and **connected**

### 73.2 Axis-Angle Representation

Every rotation in 3D can be described by:
- An **axis** $\hat{n} \in \mathbb{R}^3$ with $\|\hat{n}\| = 1$
- An **angle** $\theta \in [0, \pi]$

Combined into a single vector: $\omega = \theta \hat{n} \in \mathbb{R}^3$.

The axis is the eigenvector of $R$ corresponding to eigenvalue 1 (the fixed axis of rotation).

### 73.3 The Hat Operator and so(3)

The **hat operator** maps $\omega \in \mathbb{R}^3$ to a skew-symmetric matrix:

$$[\omega]_\times = \hat{\omega} = \begin{pmatrix} 0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{pmatrix}$$

Key property: $[\omega]_\times v = \omega \times v$ (cross product as matrix multiplication).

The Lie algebra $\mathfrak{so}(3)$ is the set of all $3 \times 3$ skew-symmetric matrices.

### 73.4 Rodrigues' Rotation Formula

The **exponential map** from $\mathfrak{so}(3)$ to SO(3):

$$R = \exp([\omega]_\times) = I + \frac{\sin\theta}{\theta}[\omega]_\times + \frac{1 - \cos\theta}{\theta^2}[\omega]_\times^2$$

where $\theta = \|\omega\|$ and $[\omega]_\times = \theta [\hat{n}]_\times$.

Equivalently, with unit axis $\hat{n}$ and angle $\theta$:

$$R = I + \sin\theta\,[\hat{n}]_\times + (1 - \cos\theta)\,[\hat{n}]_\times^2$$

**Small-angle approximation** ($\theta \to 0$): $R \approx I + [\omega]_\times$.

### 73.5 Logarithmic Map

Given $R \in \text{SO}(3)$, recover $\omega$:

$$\theta = \arccos\!\left(\frac{\text{tr}(R) - 1}{2}\right), \qquad [\omega]_\times = \frac{\theta}{2\sin\theta}(R - R^T)$$

Special cases: $\theta \approx 0$ (identity) and $\theta \approx \pi$ (half-turn) require separate treatment.

---

## Implementation (60 min)

```python
import numpy as np
from code.lie_groups.lie_groups import hat, vee, so3_exp, so3_log

# --- Rodrigues' formula demonstration ---
# Rotate 90° around the z-axis
omega_z = np.array([0, 0, np.pi / 2])
R_z = so3_exp(omega_z)
print("=== 90° rotation around z-axis ===")
print(f"R =\n{np.round(R_z, 4)}")
print(f"Expected: [[0,-1,0],[1,0,0],[0,0,1]]")

# Rotate 180° around the x-axis
omega_x = np.array([np.pi, 0, 0])
R_x = so3_exp(omega_x)
print(f"\n=== 180° rotation around x-axis ===")
print(f"R =\n{np.round(R_x, 4)}")

# --- Verify SO(3) properties ---
print(f"\n=== SO(3) Properties ===")
print(f"R^T R = I? {np.allclose(R_z.T @ R_z, np.eye(3))}")
print(f"det(R)   = {np.linalg.det(R_z):.6f}")

# --- Log map: recover axis-angle ---
omega_recovered = so3_log(R_z)
print(f"\n=== Log map recovery ===")
print(f"Original ω:  {omega_z}")
print(f"Recovered ω: {np.round(omega_recovered, 6)}")

# --- Composition (non-commutativity) ---
omega1 = np.array([np.pi/4, 0, 0])  # 45° around x
omega2 = np.array([0, np.pi/4, 0])  # 45° around y
R1 = so3_exp(omega1)
R2 = so3_exp(omega2)
print(f"\n=== Non-commutativity of SO(3) ===")
print(f"R1 @ R2 == R2 @ R1? {np.allclose(R1 @ R2, R2 @ R1)}")
diff = so3_log((R1 @ R2).T @ (R2 @ R1))
print(f"Rotation between R1R2 and R2R1: {np.degrees(np.linalg.norm(diff)):.2f}°")

# --- Small-angle Taylor expansion accuracy ---
print(f"\n=== Small-angle accuracy ===")
for eps in [1e-1, 1e-3, 1e-6, 1e-10]:
    omega_small = np.array([eps, 0, 0])
    R_exact = so3_exp(omega_small)
    R_approx = np.eye(3) + hat(omega_small)  # first-order
    error = np.linalg.norm(R_exact - R_approx)
    print(f"  |ω|={eps:.0e}: error = {error:.2e}")

# --- Hat and vee are inverses ---
omega_test = np.array([1.0, -2.0, 3.0])
print(f"\n=== Hat/Vee roundtrip ===")
print(f"vee(hat(ω)) == ω? {np.allclose(vee(hat(omega_test)), omega_test)}")
```

---

## Practice Problems (45 min)

**Problem 1:** Compute the rotation matrix for a $90°$ rotation around the axis $\hat{n} = (0, 1, 0)$ using Rodrigues' formula by hand.

<details><summary>Answer</summary>

$\theta = \pi/2$, $[\hat{n}]_\times = \begin{pmatrix} 0&0&1\\0&0&0\\-1&0&0 \end{pmatrix}$.

$[\hat{n}]_\times^2 = \begin{pmatrix} -1&0&0\\0&0&0\\0&0&-1 \end{pmatrix}$.

$R = I + \sin(\pi/2)[\hat{n}]_\times + (1-\cos(\pi/2))[\hat{n}]_\times^2 = \begin{pmatrix} 0&0&1\\0&1&0\\-1&0&0 \end{pmatrix}$.
</details>

**Problem 2:** Given $R_1$ (45° around x) and $R_2$ (45° around y), compute $R_1 R_2$ and find its axis-angle representation.

<details><summary>Answer</summary>

$R_1 R_2 = \text{so3\_exp}([π/4,0,0]) \cdot \text{so3\_exp}([0,π/4,0])$. Using the code: $\omega = \text{so3\_log}(R_1 R_2)$, yielding angle $\approx 55.7°$ around axis $\approx (0.63, 0.63, 0.45)$ (not aligned with any principal axis).
</details>

**Problem 3:** Verify that $\text{tr}(R) = 1 + 2\cos\theta$ for any rotation $R$ with angle $\theta$.

<details><summary>Answer</summary>

The eigenvalues of $R$ are $1, e^{i\theta}, e^{-i\theta}$. The trace is the sum of eigenvalues: $1 + e^{i\theta} + e^{-i\theta} = 1 + 2\cos\theta$. ✓
</details>

---

## Expert Challenges

**Challenge 1:** Derive Rodrigues' formula from the matrix exponential power series $e^{[\omega]_\times} = \sum_{k=0}^\infty \frac{[\omega]_\times^k}{k!}$ using the identity $[\hat{n}]_\times^3 = -[\hat{n}]_\times$.

<details><summary>Answer</summary>

For unit $\hat{n}$: $K = [\hat{n}]_\times$ satisfies $K^3 = -K$, so $K^4 = -K^2$, $K^5 = K$, etc.

$$e^{\theta K} = I + \theta K + \frac{\theta^2}{2}K^2 + \frac{\theta^3}{6}K^3 + \cdots = I + \left(\theta - \frac{\theta^3}{6}+\cdots\right)K + \left(\frac{\theta^2}{2}-\frac{\theta^4}{24}+\cdots\right)K^2$$

$$= I + \sin\theta\,K + (1-\cos\theta)\,K^2$$
</details>

**Challenge 2:** Show that the exponential map $\exp: \mathfrak{so}(3) \to \text{SO}(3)$ is surjective but not injective.

<details><summary>Answer</summary>

**Surjective:** Every $R \in \text{SO}(3)$ has an axis-angle representation, so $R = \exp([\theta\hat{n}]_\times)$.

**Not injective:** $\exp([\theta\hat{n}]_\times) = \exp([(\theta + 2\pi)\hat{n}]_\times)$ for any integer multiple of $2\pi$. Also, $\exp([\pi\hat{n}]_\times) = \exp([-\pi\hat{n}]_\times)$.
</details>

**Challenge 3:** Implement a function that computes the geodesic distance between two SO(3) elements: $d(R_1, R_2) = \|\log(R_1^T R_2)\|$.

<details><summary>Answer</summary>

```python
def so3_distance(R1, R2):
    omega = so3_log(R1.T @ R2)
    return np.linalg.norm(omega)

# Test: distance from identity to 90° rotation = π/2
d = so3_distance(np.eye(3), so3_exp([0, 0, np.pi/2]))
print(f"Distance: {d:.4f}, expected: {np.pi/2:.4f}")
```
</details>

---

## Connections
- **Prerequisites:** [Day 71](day-71-group-axioms.md) — SO(3) is a matrix group; [Day 72](day-72-so2-rotations.md) — SO(2) is the 2D case
- **Forward:** [Day 74](day-74-quaternions.md) gives an alternative parameterization; [Day 76](day-76-rotation-representations.md) compares all representations
- **OKS:** Rodrigues is the core formula for converting IMU angular velocity to rotation matrix updates

## Self-Check
- [ ] I can write Rodrigues' formula and apply it to a given axis-angle
- [ ] I understand why SO(3) is non-abelian and can demonstrate with an example
- [ ] I can recover the axis-angle from a rotation matrix using the log map
- [ ] I know when the small-angle approximation is valid and its error bound
