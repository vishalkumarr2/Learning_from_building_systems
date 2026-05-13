# Day 85: Manifold Gauss-Newton

> **Phase VI — Lie Groups & Manifold Optimization | Week 13 | 2.5 hours**
> *Standard Gauss-Newton breaks on manifolds — learn the retraction trick that makes it work.*

## Navigation
- **Previous:** [Day 84: Week 12 Review](../week-12/day-84-week12-review.md)
- **Next:** [Day 86: Pose-Graph SLAM on SE(2)](day-86-pose-graph-se2.md)
- **Week:** [Week 13 Overview](../README.md)
- **Chapter:** [Chapter 09: Lie Groups Reference](../../09-lie-groups.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
The OKS navigation estimator refines robot poses on SE(2)/SE(3) — a manifold, not a vector space. Naive $R + \Delta R$ leaves SO(3), producing invalid rotations. Manifold Gauss-Newton uses Exp/Log maps to keep every iterate on the manifold, which is exactly how pose refinement works in the robot's localization stack.

---

## Theory (45 min)

### 85.1 Why Standard Gauss-Newton Fails on Manifolds

In Euclidean space, Gauss-Newton updates via $x_{k+1} = x_k + \delta$. On SO(3), this breaks:

$$R + \Delta R \notin \text{SO}(3)$$

Adding a matrix to a rotation matrix destroys orthogonality. The fundamental issue: manifolds are not vector spaces, so additive updates are meaningless.

**Key insight**: instead of updating in the ambient space, we linearize in the *tangent space* and map back via the exponential map.

### 85.2 Retraction: The ⊕ Operator

Define the **retraction** (or box-plus) operator:

$$x \oplus \delta = x \cdot \text{Exp}(\delta)$$

where $\delta \in \mathbb{R}^n$ is a tangent-space perturbation and $\text{Exp}: \mathbb{R}^n \to \mathcal{M}$ is the exponential map. For SO(3), $\delta \in \mathbb{R}^3$ and $\text{Exp}$ is Rodrigues' formula.

The inverse operation (box-minus):

$$x_1 \ominus x_2 = \text{Log}(x_2^{-1} \cdot x_1)$$

### 85.3 Tangent-Space Linearization

Given a cost $f(x) = \frac{1}{2}\|r(x)\|^2$ where $x \in \mathcal{M}$, we perturb around the current estimate $x_k$:

$$r(x_k \oplus \delta) \approx r(x_k) + J_k \delta$$

where $J_k = \frac{\partial r(x_k \oplus \delta)}{\partial \delta}\big|_{\delta=0}$ is the Jacobian with respect to the tangent-space perturbation.

The normal equations become:

$$J_k^T J_k \, \delta^* = -J_k^T r(x_k)$$

### 85.4 Manifold Gauss-Newton Update Rule

The complete update:

$$\delta^* = -(J_k^T J_k)^{-1} J_k^T r_k, \qquad x_{k+1} = x_k \cdot \text{Exp}(\delta^*)$$

Equivalently with the pseudoinverse: $\delta^* = -J_k^\dagger r_k$.

**Convergence**: under standard regularity conditions, manifold GN inherits quadratic convergence near the optimum, just like its Euclidean counterpart.

---

## Implementation (60 min)

```python
"""Manifold Gauss-Newton on SE(2)."""
import numpy as np

# --- SE(2) helpers ---
def se2_exp(xi):
    """Exp map: R^3 -> SE(2). xi = [dx, dy, dtheta]."""
    dx, dy, dth = xi
    c, s = np.cos(dth), np.sin(dth)
    if abs(dth) < 1e-10:
        return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
    V = np.array([[s / dth, -(1 - c) / dth],
                  [(1 - c) / dth, s / dth]])
    t = V @ np.array([dx, dy])
    return np.array([[c, -s, t[0]], [s, c, t[1]], [0, 0, 1]])

def se2_log(T):
    """Log map: SE(2) -> R^3."""
    th = np.arctan2(T[1, 0], T[0, 0])
    c, s = np.cos(th), np.sin(th)
    if abs(th) < 1e-10:
        return np.array([T[0, 2], T[1, 2], th])
    V_inv = np.array([[s / th, (1 - c) / th],
                      [-(1 - c) / th, s / th]]) / (s**2 / th + (1 - c)**2 / th) * th
    # Simpler: invert V directly
    half_th = th / 2
    V_inv = (half_th / np.tan(half_th)) * np.eye(2) + np.array([[0, half_th], [-half_th, 0]])
    rho = V_inv @ T[:2, 2]
    return np.array([rho[0], rho[1], th])

def se2_inv(T):
    R = T[:2, :2]
    t = T[:2, 2]
    T_inv = np.eye(3)
    T_inv[:2, :2] = R.T
    T_inv[:2, 2] = -R.T @ t
    return T_inv

# --- Manifold Gauss-Newton for circle-of-poses fitting ---
def make_circle_poses(n, radius, noise_std=0.05):
    """Generate n poses arranged on a circle with noise."""
    poses = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        x = radius * np.cos(angle) + np.random.randn() * noise_std
        y = radius * np.sin(angle) + np.random.randn() * noise_std
        th = angle + np.pi / 2 + np.random.randn() * noise_std
        c, s = np.cos(th), np.sin(th)
        T = np.array([[c, -s, x], [s, c, y], [0, 0, 1]])
        poses.append(T)
    return poses

def residual_and_jacobian(T_est, T_meas):
    """Residual: Log(T_meas^{-1} T_est), Jacobian w.r.t. tangent perturbation."""
    err_T = se2_inv(T_meas) @ T_est
    r = se2_log(err_T)
    # Jacobian of Log(T_meas^{-1} (T_est Exp(delta))) at delta=0
    # Approximate as identity (right Jacobian ≈ I for small residuals)
    J = np.eye(3)
    return r, J

def manifold_gauss_newton(poses_init, poses_meas, max_iter=20, tol=1e-8):
    """Refine poses_init to match poses_meas using manifold GN."""
    poses = [T.copy() for T in poses_init]
    n = len(poses)
    costs = []
    for it in range(max_iter):
        total_r, total_J = [], []
        for i in range(n):
            r_i, J_i = residual_and_jacobian(poses[i], poses_meas[i])
            row = np.zeros((3, 3 * n))
            row[:, 3*i:3*i+3] = J_i
            total_r.append(r_i)
            total_J.append(row)
        r = np.concatenate(total_r)
        J = np.vstack(total_J)
        cost = 0.5 * r @ r
        costs.append(cost)
        H = J.T @ J + 1e-6 * np.eye(3 * n)
        delta = np.linalg.solve(H, -J.T @ r)
        if np.linalg.norm(delta) < tol:
            break
        for i in range(n):
            poses[i] = poses[i] @ se2_exp(delta[3*i:3*i+3])
    return poses, costs

# --- Demo ---
np.random.seed(42)
n_poses, radius = 8, 2.0
true_poses = make_circle_poses(n_poses, radius, noise_std=0.0)
meas_poses = make_circle_poses(n_poses, radius, noise_std=0.1)
init_poses = make_circle_poses(n_poses, radius, noise_std=0.3)

refined, costs = manifold_gauss_newton(init_poses, meas_poses)
print(f"Iterations: {len(costs)}, Final cost: {costs[-1]:.6f}")
print(f"Cost reduction: {costs[0]:.4f} -> {costs[-1]:.6f}")
```

---

## Practice Problems (45 min)

### Problem 1: Derive the Update Rule
Starting from $r(x_k \oplus \delta) \approx r_k + J_k \delta$, derive the normal equations and show that the update $x_{k+1} = x_k \cdot \text{Exp}(\delta^*)$ stays on the manifold.

<details><summary>Answer</summary>

Minimizing $\|r_k + J_k \delta\|^2$ gives $\frac{\partial}{\partial \delta}(r_k + J_k\delta)^T(r_k + J_k\delta) = 0$, yielding $J_k^T J_k \delta^* = -J_k^T r_k$. Since $\text{Exp}: \mathbb{R}^3 \to \text{SE}(2)$ maps to the group and group multiplication is closed, $x_k \cdot \text{Exp}(\delta^*) \in \text{SE}(2)$.
</details>

### Problem 2: SO(3) Rotation Averaging
Given $N$ noisy rotations $\{R_i\}$, find the Fréchet mean $R^* = \arg\min_R \sum_i \|\text{Log}(R^{-1} R_i)\|^2$ using manifold GN.

<details><summary>Answer</summary>

```python
def rotation_average(rotations, max_iter=50):
    R = rotations[0].copy()
    for _ in range(max_iter):
        deltas = [so3_log(R.T @ Ri) for Ri in rotations]
        mean_delta = np.mean(deltas, axis=0)
        if np.linalg.norm(mean_delta) < 1e-10:
            break
        R = R @ so3_exp(mean_delta)
    return R
```
The Jacobian of each $\text{Log}(R^{-1}R_i)$ w.r.t. tangent perturbation is approximately $-I_3$, so the GN step simplifies to averaging the log-map residuals.
</details>

### Problem 3: Convergence Analysis
Run manifold GN on the circle-of-poses problem with increasing noise levels (σ = 0.01, 0.1, 0.5, 1.0). Plot iterations-to-convergence vs noise. At what noise level does convergence fail?

<details><summary>Answer</summary>

Convergence typically degrades beyond σ ≈ 0.5–1.0 rad because the linearization in the tangent space becomes inaccurate for large perturbations. The basin of convergence shrinks as noise grows, and the algorithm may converge to a local minimum or diverge when initial error exceeds ~π/2 in rotation.
</details>

---

## Expert Challenges

### Challenge 1: Left vs Right Perturbation
Implement manifold GN with *left* perturbation ($x_{k+1} = \text{Exp}(\delta) \cdot x_k$) and compare convergence with the right perturbation version. When does the choice matter?

<details><summary>Answer</summary>

Left perturbation: $r(x) = \text{Log}(T_{\text{meas}}^{-1} \cdot \text{Exp}(\delta) \cdot T_k)$. The Jacobian changes — it involves the right Jacobian $J_r$ instead of $J_l$. For small residuals both converge identically; the difference appears when composing transforms in chains where the perturbation frame matters.
</details>

### Challenge 2: Damped Manifold GN (Levenberg-Marquardt)
Add a damping parameter $\lambda$ to get $(J^TJ + \lambda I)\delta = -J^Tr$. Implement adaptive $\lambda$ (increase on cost increase, decrease on decrease).

<details><summary>Answer</summary>

Start $\lambda = 0.01$. If cost decreases, accept step and $\lambda \leftarrow \lambda / 10$. If cost increases, reject step and $\lambda \leftarrow \lambda \times 10$. This interpolates between GN ($\lambda \to 0$) and gradient descent ($\lambda \to \infty$), ensuring convergence from poor initializations.
</details>

### Challenge 3: Numerical Jacobian Verification
Compute the Jacobian $\frac{\partial r(x \oplus \delta)}{\partial \delta}$ numerically using central differences on the manifold and compare with the analytical Jacobian.

<details><summary>Answer</summary>

```python
def numerical_jacobian(x, r_func, eps=1e-7):
    n = 3  # tangent space dim
    J = np.zeros((3, n))
    for i in range(n):
        ei = np.zeros(n); ei[i] = eps
        r_plus = r_func(x @ se2_exp(ei))
        r_minus = r_func(x @ se2_exp(-ei))
        J[:, i] = (r_plus - r_minus) / (2 * eps)
    return J
```
This perturbs along each tangent-space basis direction using the Exp map, ensuring we stay on the manifold.
</details>

---

## Connections
- **Prerequisites**: Day 80 (Exp/Log maps), Day 83 (Jacobians on Lie groups)
- **Forward**: Day 86 builds on this to solve full pose-graph SLAM on SE(2)
- **OKS**: every pose refinement step in the navigation estimator is a manifold GN iteration

## Self-Check
- [ ] I can explain why $R + \Delta R \notin \text{SO}(3)$ and why retraction is needed
- [ ] I can write the manifold GN update rule from memory
- [ ] I can implement manifold GN on SE(2) and verify convergence
- [ ] I understand the role of the tangent-space Jacobian vs the ambient-space Jacobian
