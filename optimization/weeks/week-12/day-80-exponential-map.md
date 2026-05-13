# Day 80: Exponential and Logarithmic Maps

> **Phase VI — Lie Groups & Manifold Optimization | Week 12 | 2.5 hours**
> *The exp map is the bridge between the flat world of optimization and the curved world of rigid transforms.*

## Navigation
- **Previous:** [Day 79: SE(3) — 3D Rigid Body Transforms](day-79-se3-rigid-body.md)
- **Next:** [Day 81: Lie Algebra Deep Dive](day-81-lie-algebra.md)
- **Week:** [Week 12 Overview](../README.md)
- **Chapter:** [Chapter 09: Lie Groups Reference](../../09-lie-groups.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Pose-graph optimization in SLAM requires perturbing SE(3) poses. You cannot add a small vector to a 4×4 matrix and get a valid SE(3) element. Instead, you perturb in the Lie algebra via $T' = T \cdot \exp(\delta\xi)$. This is the fundamental operation in every manifold optimizer the OKS stack uses.

---

## Theory (45 min)

### 80.1 The Exponential Map: $\mathfrak{g} \to G$

For matrix Lie groups, the exponential map is the matrix exponential:

$$\exp(\hat{\xi}) = \sum_{k=0}^{\infty} \frac{\hat{\xi}^k}{k!} = I + \hat{\xi} + \frac{\hat{\xi}^2}{2!} + \cdots$$

This maps an element of the Lie algebra (tangent space at identity) to a group element.

### 80.2 Closed-Form for SO(3): Rodrigues' Formula

For $\boldsymbol{\omega} \in \mathbb{R}^3$ with $\theta = \|\boldsymbol{\omega}\|$ and $\mathbf{a} = \boldsymbol{\omega}/\theta$:

$$\exp([\boldsymbol{\omega}]_\times) = I + \sin\theta \, [\mathbf{a}]_\times + (1 - \cos\theta) \, [\mathbf{a}]_\times^2$$

The logarithmic map inverts this. Given $R \in SO(3)$:

$$\theta = \arccos\left(\frac{\text{tr}(R) - 1}{2}\right), \quad [\boldsymbol{\omega}]_\times = \frac{\theta}{2\sin\theta}(R - R^T)$$

### 80.3 Closed-Form for SE(3): The $V$ Matrix

For twist $\xi = (\mathbf{v}, \boldsymbol{\omega})^T$ with $\theta = \|\boldsymbol{\omega}\|$:

$$\exp(\hat{\xi}) = \begin{pmatrix} \exp([\boldsymbol{\omega}]_\times) & V\mathbf{v} \\ \mathbf{0}^T & 1 \end{pmatrix}$$

where the **$V$ matrix** is:

$$V = I + \frac{1 - \cos\theta}{\theta^2} [\boldsymbol{\omega}]_\times + \frac{\theta - \sin\theta}{\theta^3} [\boldsymbol{\omega}]_\times^2$$

For the log map, invert: $\mathbf{v} = V^{-1}\mathbf{t}$ where:

$$V^{-1} = I - \frac{1}{2}[\boldsymbol{\omega}]_\times + \left(\frac{1}{\theta^2} - \frac{1 + \cos\theta}{2\theta\sin\theta}\right)[\boldsymbol{\omega}]_\times^2$$

### 80.4 Small-Angle Approximations

When $\theta \to 0$, the Taylor expansions give:

| Map | Exact | Small-angle |
|-----|-------|-------------|
| $\exp([\omega]_\times)$ | Rodrigues | $I + [\omega]_\times$ |
| $V$ | Full formula | $I + \frac{1}{2}[\omega]_\times$ |
| $V^{-1}$ | Full formula | $I - \frac{1}{2}[\omega]_\times$ |

These are critical for numerical stability near $\theta = 0$.

---

## Implementation (60 min)

```python
import numpy as np

def hat3(omega):
    """R^3 -> so(3): skew-symmetric matrix."""
    return np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])

def vee3(Omega):
    """so(3) -> R^3: extract vector from skew-symmetric."""
    return np.array([Omega[2,1], Omega[0,2], Omega[1,0]])

def so3_exp(omega):
    """Exponential map so(3) -> SO(3) via Rodrigues."""
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3) + hat3(omega)
    K = hat3(omega / theta)
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

def so3_log(R):
    """Logarithmic map SO(3) -> so(3)."""
    cos_angle = np.clip((np.trace(R) - 1) / 2, -1, 1)
    theta = np.arccos(cos_angle)
    if theta < 1e-10:
        return vee3(R - R.T) / 2
    return theta / (2 * np.sin(theta)) * vee3(R - R.T)

def V_matrix(omega):
    """Compute V matrix for SE(3) exp map."""
    theta = np.linalg.norm(omega)
    K = hat3(omega)
    if theta < 1e-10:
        return np.eye(3) + 0.5 * K
    K_unit = K / theta
    return (np.eye(3)
            + ((1 - np.cos(theta)) / theta**2) * K
            + ((theta - np.sin(theta)) / theta**3) * K @ K)

def V_inv_matrix(omega):
    """Compute V^{-1} for SE(3) log map."""
    theta = np.linalg.norm(omega)
    K = hat3(omega)
    if theta < 1e-10:
        return np.eye(3) - 0.5 * K
    half_theta = theta / 2
    return (np.eye(3)
            - 0.5 * K
            + (1/theta**2 - (1 + np.cos(theta))/(2*theta*np.sin(theta))) * K @ K)

def se3_exp(xi):
    """Exponential map: R^6 -> SE(3). xi = (v, omega)."""
    v, omega = xi[:3], xi[3:]
    R = so3_exp(omega)
    V = V_matrix(omega)
    t = V @ v
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def se3_log(T):
    """Logarithmic map: SE(3) -> R^6. Returns (v, omega)."""
    R = T[:3, :3]
    t = T[:3, 3]
    omega = so3_log(R)
    Vi = V_inv_matrix(omega)
    v = Vi @ t
    return np.concatenate([v, omega])

# --- Verification ---
xi = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
T = se3_exp(xi)
xi_back = se3_log(T)

print("Original twist:", np.round(xi, 6))
print("Round-trip:    ", np.round(xi_back, 6))
print("Max error:     ", np.max(np.abs(xi - xi_back)))

# Small-angle test
xi_small = np.array([0.01, -0.005, 0.008, 1e-8, 2e-8, 3e-8])
T_small = se3_exp(xi_small)
xi_small_back = se3_log(T_small)
print("\nSmall angle error:", np.max(np.abs(xi_small - xi_small_back)))
```

---

## Practice Problems (45 min)

**Problem 1:** Compute $\exp([\boldsymbol{\omega}]_\times)$ by hand for $\boldsymbol{\omega} = (0, 0, \pi/2)$ (90° about z-axis).

<details><summary>Answer</summary>

$\theta = \pi/2$, $\mathbf{a} = (0,0,1)$. $K = \begin{pmatrix}0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0\end{pmatrix}$. $\exp = I + \sin(\pi/2)K + (1-\cos(\pi/2))K^2 = I + K + K^2 = \begin{pmatrix}0&-1&0\\1&0&0\\0&0&1\end{pmatrix}$.
</details>

**Problem 2:** For $\boldsymbol{\omega} = (0, 0, 0)$ (pure translation), show that $V = I$ and the exp map gives $T = \begin{pmatrix}I & \mathbf{v} \\ 0 & 1\end{pmatrix}$.

<details><summary>Answer</summary>

When $\theta = 0$: $\exp([0]_\times) = I$, $V = I + 0 = I$, so $\mathbf{t} = V\mathbf{v} = \mathbf{v}$. The result is a pure translation matrix.
</details>

**Problem 3:** Verify numerically that $\exp(\hat{\xi}_1)\exp(\hat{\xi}_2) \neq \exp(\hat{\xi}_1 + \hat{\xi}_2)$ for non-commuting twists.

<details><summary>Answer</summary>

```python
xi1 = np.array([1, 0, 0, 0, 0, 0.5])
xi2 = np.array([0, 1, 0, 0.5, 0, 0])
T_product = se3_exp(xi1) @ se3_exp(xi2)
T_sum = se3_exp(xi1 + xi2)
print("Differ:", np.max(np.abs(T_product - T_sum)))  # > 0
```

They differ because the BCH formula has higher-order correction terms involving $[\xi_1, \xi_2]$.
</details>

---

## Expert Challenges

**Challenge 1:** Derive the $V$ matrix by explicitly summing the Taylor series $\sum_{k=0}^{\infty} \frac{[\omega]_\times^k}{(k+1)!}$.

<details><summary>Answer</summary>

Using $[\omega]_\times^2 = \mathbf{a}\mathbf{a}^T - I$ (for unit $\mathbf{a}$), the even and odd powers collapse into cosine and sine series: $V = I + \frac{1-\cos\theta}{\theta^2}[\omega]_\times + \frac{\theta-\sin\theta}{\theta^3}[\omega]_\times^2$.
</details>

**Challenge 2:** Explain the "wrapping" issue: for what values of $\theta$ does $\log(\exp(\omega))$ fail to return $\omega$?

<details><summary>Answer</summary>

$\log$ returns $\theta \in [0, \pi]$. If $\|\omega\| > \pi$, the exp map wraps around (since $\exp(\omega) = \exp(\omega - 2\pi\hat{a})$ for axis $\hat{a}$). The log map always returns the shortest-path rotation. At $\theta = \pi$, the log has a branch cut and axis direction is ambiguous.
</details>

**Challenge 3:** Implement a numerically stable version of $V^{-1}$ that uses Taylor expansion for $\theta < 10^{-4}$ and the closed form otherwise.

<details><summary>Answer</summary>

```python
def V_inv_stable(omega):
    theta = np.linalg.norm(omega)
    K = hat3(omega)
    if theta < 1e-4:
        # Taylor: V^-1 ≈ I - 1/2 K + 1/12 K^2
        return np.eye(3) - 0.5 * K + (1/12) * K @ K
    half = theta / 2
    return (np.eye(3) - 0.5 * K
            + (1/theta**2 - (1+np.cos(theta))/(2*theta*np.sin(theta))) * K @ K)
```
</details>

---

## Connections
- **Back:** [Day 73: Rodrigues](../week-11/day-73-so3-rodrigues.md) — SO(3) exp map is a special case
- **Back:** [Day 78: SE(2)](day-78-se2-rigid-motions.md) — SE(2) exp map uses a 2D $V$ matrix
- **Forward:** [Day 81: Lie Algebra](day-81-lie-algebra.md) — deeper structure of the tangent space
- **Forward:** [Day 83: Jacobians](day-83-jacobians.md) — $V$ matrix reappears as the left Jacobian
- **OKS:** incremental pose update $T_{k+1} = T_k \cdot \exp(\delta\xi)$ is the core of manifold optimization

## Self-Check
- [ ] I can state the closed-form exp map for both SO(3) and SE(3)
- [ ] I understand the role of the $V$ matrix and can compute it
- [ ] I can perform exp → log round-trips and know when they fail (wrapping)
- [ ] I can handle the small-angle ($\theta \to 0$) case numerically
