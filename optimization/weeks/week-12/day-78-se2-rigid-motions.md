# Day 78: SE(2) — 2D Rigid Motions

> **Phase VI — Lie Groups & Manifold Optimization | Week 12 | 2.5 hours**
> *Every AMR pose is a point in SE(2) — learn the group that describes planar motion.*

## Navigation
- **Previous:** [Day 77: Week 11 Review](../week-11/day-77-week11-review.md)
- **Next:** [Day 79: SE(3) — 3D Rigid Body Transforms](day-79-se3-rigid-body.md)
- **Week:** [Week 12 Overview](../README.md)
- **Chapter:** [Chapter 09: Lie Groups Reference](../../09-lie-groups.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
An OKS AMR moving on a warehouse floor has pose $(x, y, \theta)$. This triple is **not** a vector — you cannot simply add two poses. The correct composition rule comes from SE(2), the group of 2D rigid motions. The navigation estimator internally composes incremental SE(2) transforms from wheel odometry and sensorbar data every control cycle.

---

## Theory (45 min)

### 78.1 Homogeneous Coordinates

A 2D rigid motion combines rotation $R \in SO(2)$ and translation $\mathbf{t} \in \mathbb{R}^2$. In homogeneous coordinates:

$$T = \begin{pmatrix} R & \mathbf{t} \\ 0 & 1 \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta & t_x \\ \sin\theta & \cos\theta & t_y \\ 0 & 0 & 1 \end{pmatrix} \in \mathbb{R}^{3 \times 3}$$

A point $\mathbf{p} = (p_x, p_y)$ transforms as:

$$\begin{pmatrix} p_x' \\ p_y' \\ 1 \end{pmatrix} = T \begin{pmatrix} p_x \\ p_y \\ 1 \end{pmatrix}$$

### 78.2 SE(2) as a Semidirect Product

SE(2) is the **special Euclidean group** in 2D:

$$\text{SE}(2) = \text{SO}(2) \ltimes \mathbb{R}^2$$

The semidirect product means rotation and translation interact non-trivially. For elements $(R_1, \mathbf{t}_1)$ and $(R_2, \mathbf{t}_2)$:

$$\text{Composition:} \quad (R_1, \mathbf{t}_1) \circ (R_2, \mathbf{t}_2) = (R_1 R_2, \; R_1 \mathbf{t}_2 + \mathbf{t}_1)$$

Note: the rotation $R_1$ **acts on** $\mathbf{t}_2$ before addition. This is why pose composition is not simple vector addition.

### 78.3 Group Properties

| Property | Matrix Form | Component Form |
|----------|-------------|----------------|
| **Identity** | $I_3$ | $(I_2, \mathbf{0})$ |
| **Inverse** | $T^{-1}$ | $(R^T, -R^T \mathbf{t})$ |
| **Composition** | $T_1 T_2$ | $(R_1 R_2, R_1 \mathbf{t}_2 + \mathbf{t}_1)$ |

The inverse is **not** simply negating parameters:

$$T^{-1} = \begin{pmatrix} R^T & -R^T \mathbf{t} \\ 0 & 1 \end{pmatrix}$$

### 78.4 The Lie Algebra $\mathfrak{se}(2)$

The Lie algebra consists of $3 \times 3$ matrices of the form:

$$\hat{\xi} = \begin{pmatrix} 0 & -\theta & v_x \\ \theta & 0 & v_y \\ 0 & 0 & 0 \end{pmatrix} \in \mathfrak{se}(2)$$

where $\xi = (v_x, v_y, \theta)^T \in \mathbb{R}^3$ is the **twist vector**. The hat operator $\hat{\cdot}$ maps $\mathbb{R}^3 \to \mathfrak{se}(2)$ and vee $(\cdot)^\vee$ is its inverse.

---

## Implementation (60 min)

```python
import numpy as np

def se2_from_xyt(x, y, theta):
    """Build SE(2) matrix from (x, y, theta)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, x],
        [s,  c, y],
        [0,  0, 1]
    ])

def se2_compose(T1, T2):
    """Compose two SE(2) elements."""
    return T1 @ T2

def se2_inverse(T):
    """Compute SE(2) inverse: (R^T, -R^T t)."""
    R = T[:2, :2]
    t = T[:2, 2]
    Rt = R.T
    return np.block([
        [Rt, (-Rt @ t).reshape(2, 1)],
        [np.zeros((1, 2)), np.ones((1, 1))]
    ])

def se2_exp(xi):
    """Exponential map: se(2) -> SE(2). xi = (vx, vy, theta)."""
    vx, vy, theta = xi
    if abs(theta) < 1e-10:
        return se2_from_xyt(vx, vy, theta)
    # V matrix for SE(2)
    s, c = np.sin(theta), np.cos(theta)
    V = (1.0 / theta) * np.array([
        [s,       -(1 - c)],
        [(1 - c),  s      ]
    ])
    t = V @ np.array([vx, vy])
    return se2_from_xyt(t[0], t[1], theta)

def se2_log(T):
    """Logarithmic map: SE(2) -> se(2). Returns xi = (vx, vy, theta)."""
    theta = np.arctan2(T[1, 0], T[0, 0])
    t = T[:2, 2]
    if abs(theta) < 1e-10:
        return np.array([t[0], t[1], theta])
    s, c = np.sin(theta), np.cos(theta)
    V = (1.0 / theta) * np.array([
        [s,       -(1 - c)],
        [(1 - c),  s      ]
    ])
    v = np.linalg.solve(V, t)
    return np.array([v[0], v[1], theta])

# --- Demo ---
T1 = se2_from_xyt(1.0, 0.0, np.pi / 4)
T2 = se2_from_xyt(0.0, 2.0, np.pi / 6)

T12 = se2_compose(T1, T2)
T1_inv = se2_inverse(T1)
identity_check = se2_compose(T1, T1_inv)
print("T1 @ T2:\n", T12)
print("T1 @ T1_inv (should be I):\n", np.round(identity_check, 10))

# Round-trip exp/log
xi = np.array([0.5, -0.3, 0.8])
T = se2_exp(xi)
xi_back = se2_log(T)
print("Original xi:", xi)
print("Round-trip xi:", np.round(xi_back, 10))
```

---

## Practice Problems (45 min)

**Problem 1:** A robot at pose $(1, 2, \pi/4)$ moves forward 1 m in its body frame. Compute the new world pose using SE(2) composition.

<details><summary>Answer</summary>

Body-frame motion: $T_\text{inc} = \text{se2}(1, 0, 0)$ (1 m along local x-axis). World pose: $T_\text{new} = T_\text{old} \cdot T_\text{inc}$. Result: $(1 + \cos(\pi/4), 2 + \sin(\pi/4), \pi/4) \approx (1.707, 2.707, 0.785)$.
</details>

**Problem 2:** Prove that $T^{-1} T = I$ using the component form $(R^T, -R^T \mathbf{t})$.

<details><summary>Answer</summary>

$(R^T, -R^T\mathbf{t}) \circ (R, \mathbf{t}) = (R^T R, R^T \mathbf{t} + (-R^T \mathbf{t})) = (I, \mathbf{0})$. ✓
</details>

**Problem 3:** Compute $\text{se2\_log}(\text{se2\_exp}(\xi))$ for $\xi = (1, -1, \pi)$. Is the round-trip exact?

<details><summary>Answer</summary>

Yes, for $|\theta| < \pi$ the round-trip is exact up to floating-point precision. At $\theta = \pi$, the $V$ matrix is still well-defined so the round-trip works, but $\theta = 2\pi$ wraps to $0$.
</details>

---

## Expert Challenges

**Challenge 1:** Show that SE(2) is **not** abelian by constructing two elements $T_1, T_2$ where $T_1 T_2 \neq T_2 T_1$.

<details><summary>Answer</summary>

Let $T_1 = \text{se2}(1, 0, \pi/2)$, $T_2 = \text{se2}(0, 1, 0)$. Then $T_1 T_2$ translates $(0,1)$ *after* a 90° rotation, giving final position $(0, 1)$. But $T_2 T_1$ translates first, then rotates, giving $(0, 2)$. The positions differ.
</details>

**Challenge 2:** Derive the $V$ matrix in the SE(2) exponential map from the Taylor series $\exp(\hat{\xi}) = \sum_{k=0}^\infty \frac{\hat{\xi}^k}{k!}$.

<details><summary>Answer</summary>

Writing $\hat{\xi} = \begin{pmatrix} \omega^\wedge & \mathbf{v} \\ 0 & 0 \end{pmatrix}$, the series separates into rotation (Rodrigues) and translation blocks. The translation block sums to $V\mathbf{v}$ where $V = \frac{1}{\theta}\begin{pmatrix} \sin\theta & -(1-\cos\theta) \\ (1-\cos\theta) & \sin\theta \end{pmatrix}$.
</details>

**Challenge 3:** Implement an SE(2) interpolation function that computes $T(s) = T_1 \cdot \exp(s \cdot \log(T_1^{-1} T_2))$ for $s \in [0, 1]$.

<details><summary>Answer</summary>

```python
def se2_interp(T1, T2, s):
    T_rel = se2_compose(se2_inverse(T1), T2)
    xi = se2_log(T_rel)
    return se2_compose(T1, se2_exp(s * xi))
```

This traces the geodesic from $T_1$ to $T_2$ on SE(2).
</details>

---

## Connections
- **Back:** [Day 72: SO(2)](../week-11/day-72-so2-rotations.md) — SE(2) extends SO(2) with translation
- **Back:** [Day 71: Group Axioms](../week-11/day-71-group-axioms.md) — SE(2) satisfies all group axioms
- **Forward:** [Day 79: SE(3)](day-79-se3-rigid-body.md) — the 3D generalization
- **OKS:** Planar robot pose is SE(2); the nav estimator composes incremental SE(2) transforms

## Self-Check
- [ ] I can build an SE(2) matrix from $(x, y, \theta)$ and extract parameters back
- [ ] I understand why pose composition is matrix multiplication, not vector addition
- [ ] I can compute the SE(2) inverse and verify $T T^{-1} = I$
- [ ] I can apply exp/log maps and explain the role of the $V$ matrix
