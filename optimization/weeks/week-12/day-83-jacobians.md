# Day 83: Left and Right Jacobians

> **Phase VI — Lie Groups & Manifold Optimization | Week 12 | 2.5 hours**
> *The Jacobians of the exponential map are the missing link between Lie algebra perturbations and manifold optimization.*

## Navigation
- **Previous:** [Day 82: Adjoint Representation and BCH](day-82-adjoint-bch.md)
- **Next:** [Day 84: Week 12 Review + Exercises](day-84-week12-review.md)
- **Week:** [Week 12 Overview](../README.md)
- **Chapter:** [Chapter 09: Lie Groups Reference](../../09-lie-groups.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Gauss-Newton optimization on SE(3) manifolds requires linearizing $\exp(\boldsymbol{\phi} + \delta\boldsymbol{\phi})$ around the current estimate. The left/right Jacobians provide this linearization. Without them, SLAM and pose-graph optimization cannot compute the correct gradient update. Every manifold optimizer in the OKS stack (from pose-graph SLAM to hand-eye calibration) relies on these Jacobians.

---

## Theory (45 min)

### 83.1 The Linearization Problem

We need to relate a perturbation $\delta\boldsymbol{\phi}$ in the Lie algebra to a perturbation on the group:

$$\exp(\boldsymbol{\phi} + \delta\boldsymbol{\phi}) \approx \exp(\boldsymbol{\phi}) \cdot \exp(J_l^{-1}(\boldsymbol{\phi}) \, \delta\boldsymbol{\phi})$$

$$\exp(\boldsymbol{\phi} + \delta\boldsymbol{\phi}) \approx \exp(J_r^{-1}(\boldsymbol{\phi}) \, \delta\boldsymbol{\phi}) \cdot \exp(\boldsymbol{\phi})$$

$J_l$ is the **left Jacobian** and $J_r$ is the **right Jacobian**.

### 83.2 Left Jacobian of SO(3)

For $\boldsymbol{\phi} \in \mathbb{R}^3$ with $\theta = \|\boldsymbol{\phi}\|$ and $\mathbf{a} = \boldsymbol{\phi}/\theta$:

$$J_l(\boldsymbol{\phi}) = \frac{\sin\theta}{\theta} I + \left(1 - \frac{\sin\theta}{\theta}\right)\mathbf{a}\mathbf{a}^T + \frac{1 - \cos\theta}{\theta} [\mathbf{a}]_\times$$

**Series form:** $J_l(\boldsymbol{\phi}) = \sum_{k=0}^{\infty} \frac{1}{(k+1)!} [\boldsymbol{\phi}]_\times^k = I + \frac{1}{2!}[\boldsymbol{\phi}]_\times + \frac{1}{3!}[\boldsymbol{\phi}]_\times^2 + \cdots$

### 83.3 Right Jacobian

The right Jacobian relates to the left by a sign flip:

$$J_r(\boldsymbol{\phi}) = J_l(-\boldsymbol{\phi})$$

Or equivalently: $J_r(\boldsymbol{\phi}) = R(\boldsymbol{\phi})^T J_l(\boldsymbol{\phi})$ where $R = \exp([\boldsymbol{\phi}]_\times)$.

**Closed form for SO(3):**

$$J_r(\boldsymbol{\phi}) = \frac{\sin\theta}{\theta} I + \left(1 - \frac{\sin\theta}{\theta}\right)\mathbf{a}\mathbf{a}^T - \frac{1 - \cos\theta}{\theta} [\mathbf{a}]_\times$$

### 83.4 Inverse Jacobians

The inverse $J_l^{-1}$ has closed form:

$$J_l^{-1}(\boldsymbol{\phi}) = \frac{\theta/2}{\tan(\theta/2)} I + \left(1 - \frac{\theta/2}{\tan(\theta/2)}\right)\mathbf{a}\mathbf{a}^T - \frac{\theta}{2}[\mathbf{a}]_\times$$

And $J_r^{-1}(\boldsymbol{\phi}) = J_l^{-1}(-\boldsymbol{\phi})$.

### 83.5 Connection to Optimization

In pose-graph SLAM, the residual between poses $T_i$ and $T_j$ is:

$$\mathbf{e}_{ij} = \log(T_{ij}^{-1} T_i^{-1} T_j)$$

The Jacobian of this residual with respect to a left perturbation $T_i \leftarrow T_i \exp(\delta\xi_i)$ involves $J_r^{-1}$:

$$\frac{\partial \mathbf{e}_{ij}}{\partial \delta\xi_i} = -J_r^{-1}(\mathbf{e}_{ij})$$

---

## Implementation (60 min)

```python
import numpy as np

def hat3(w):
    return np.array([[0,-w[2],w[1]], [w[2],0,-w[0]], [-w[1],w[0],0]])

def so3_exp(w):
    theta = np.linalg.norm(w)
    if theta < 1e-10:
        return np.eye(3) + hat3(w)
    K = hat3(w/theta)
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*K@K

def so3_left_jacobian(phi):
    """Left Jacobian J_l(phi) for SO(3)."""
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        return np.eye(3) + 0.5 * hat3(phi)
    a = phi / theta
    K = hat3(a)
    return ((np.sin(theta)/theta) * np.eye(3)
            + (1 - np.sin(theta)/theta) * np.outer(a, a)
            + ((1 - np.cos(theta))/theta) * K)

def so3_right_jacobian(phi):
    """Right Jacobian J_r(phi) = J_l(-phi)."""
    return so3_left_jacobian(-phi)

def so3_left_jacobian_inv(phi):
    """Inverse of left Jacobian J_l^{-1}(phi)."""
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        return np.eye(3) - 0.5 * hat3(phi)
    a = phi / theta
    half_theta = theta / 2
    cot_half = half_theta / np.tan(half_theta)
    K = hat3(a)
    return (cot_half * np.eye(3)
            + (1 - cot_half) * np.outer(a, a)
            - half_theta * K)

def so3_right_jacobian_inv(phi):
    """Inverse of right Jacobian."""
    return so3_left_jacobian_inv(-phi)

# --- Numerical verification against finite differences ---
def numerical_left_jacobian(phi, eps=1e-7):
    """Compute J_l numerically: d/d(delta) exp(phi + delta)|_{delta=0}."""
    J = np.zeros((3, 3))
    R0 = so3_exp(phi)
    for i in range(3):
        delta = np.zeros(3)
        delta[i] = eps
        R_plus = so3_exp(phi + delta)
        R_minus = so3_exp(phi - delta)
        # Left perturbation: exp(phi+d) = exp(phi) exp(J_l^{-1} d)
        # => J_l maps delta to the tangent of exp(phi+d) exp(-phi)
        dR = R_plus @ R0.T
        # Extract rotation vector (log map)
        cos_a = np.clip((np.trace(dR) - 1) / 2, -1, 1)
        angle = np.arccos(cos_a)
        if angle < 1e-10:
            w = np.array([dR[2,1]-dR[1,2], dR[0,2]-dR[2,0], dR[1,0]-dR[0,1]]) / 2
        else:
            w = angle / (2*np.sin(angle)) * np.array([
                dR[2,1]-dR[1,2], dR[0,2]-dR[2,0], dR[1,0]-dR[0,1]])
        J[:, i] = w / eps
    return J

# Test
phi = np.array([0.5, -0.3, 0.8])
Jl_analytic = so3_left_jacobian(phi)
Jl_numeric = numerical_left_jacobian(phi)

print("Left Jacobian (analytic):\n", np.round(Jl_analytic, 6))
print("Left Jacobian (numeric):\n", np.round(Jl_numeric, 6))
print("Max error:", np.max(np.abs(Jl_analytic - Jl_numeric)))

# Verify J_l * J_l^{-1} = I
Jl_inv = so3_left_jacobian_inv(phi)
print("\nJ_l @ J_l_inv (should be I):\n", np.round(Jl_analytic @ Jl_inv, 6))

# Verify J_r = R^T J_l
Jr = so3_right_jacobian(phi)
R = so3_exp(phi)
print("\nJ_r == R^T J_l:", np.allclose(Jr, R.T @ Jl_analytic))

# Verify relationship: J_r(phi) = J_l(-phi)
print("J_r(phi) == J_l(-phi):", np.allclose(Jr, so3_left_jacobian(-phi)))
```

---

## Practice Problems (45 min)

**Problem 1:** Compute $J_l(\boldsymbol{\phi})$ for $\boldsymbol{\phi} = (0, 0, \pi/2)$. Verify that $J_l J_l^{-1} = I$.

<details><summary>Answer</summary>

$\theta = \pi/2$, $\mathbf{a} = (0,0,1)$. $\sin(\pi/2)/(\pi/2) = 2/\pi$, $(1-\cos(\pi/2))/(\pi/2) = 2/\pi$. So $J_l = \frac{2}{\pi}I + (1-\frac{2}{\pi})\mathbf{e}_3\mathbf{e}_3^T + \frac{2}{\pi}E_3$. Numerically verify $J_l J_l^{-1} = I$.
</details>

**Problem 2:** Show that $J_l(\mathbf{0}) = I$ (the Jacobian at zero rotation is the identity).

<details><summary>Answer</summary>

As $\theta \to 0$: $\sin\theta/\theta \to 1$, $(1-\cos\theta)/\theta \to 0$. So $J_l \to I + 0 + 0 = I$. Alternatively, from the series: $J_l = I + \frac{1}{2!}\cdot 0 + \cdots = I$.
</details>

**Problem 3:** For the SLAM residual $\mathbf{e} = \log(T_m^{-1} T_i^{-1} T_j)$, compute the Jacobian with respect to a right perturbation of $T_j$: $T_j \leftarrow T_j \exp(\delta\xi_j)$.

<details><summary>Answer</summary>

$\mathbf{e}(\delta\xi_j) = \log(T_m^{-1} T_i^{-1} T_j \exp(\delta\xi_j))$. Using the BCH/Jacobian identity: $\frac{\partial \mathbf{e}}{\partial \delta\xi_j}\bigg|_{\delta=0} = J_r^{-1}(\mathbf{e})$.
</details>

---

## Expert Challenges

**Challenge 1:** Derive the series expansion $J_l = \sum_{k=0}^{\infty} \frac{[\phi]_\times^k}{(k+1)!}$ from the definition $J_l = \frac{\partial}{\partial \delta}\exp(\phi+\delta)\exp(-\phi)\big|_{\delta=0}$.

<details><summary>Answer</summary>

$\exp(\phi+\delta) = \sum_n \frac{(\hat\phi + \hat\delta)^n}{n!}$. Differentiating w.r.t. $\delta$ at $\delta=0$ and right-multiplying by $\exp(-\phi)$, collect terms: the $k$-th power of $\hat\phi$ appears with coefficient $\frac{1}{(k+1)!}$, giving $J_l\delta = \sum_{k=0}^\infty \frac{\hat\phi^k}{(k+1)!}\hat\delta$, hence $J_l = \sum_{k=0}^\infty \frac{[\phi]_\times^k}{(k+1)!}$.
</details>

**Challenge 2:** Prove that $\det(J_l(\phi)) > 0$ for all $\theta < 2\pi$, i.e., the Jacobian is always invertible within the principal domain.

<details><summary>Answer</summary>

$J_l$ has eigenvalues $\frac{\sin\theta}{\theta}$ (with multiplicity 2 in the plane perpendicular to $\mathbf{a}$) and $1$ (along $\mathbf{a}$). Since $\frac{\sin\theta}{\theta} > 0$ for $\theta \in (0, \pi)$ and $= \frac{\sin\theta}{\theta}$ passes through zero only at $\theta = k\pi$, $\det(J_l) = \left(\frac{\sin\theta}{\theta}\right)^2 > 0$ for $\theta \in (0, \pi)$. At $\theta = \pi$, the $\mathbf{a}\mathbf{a}^T$ term compensates.
</details>

**Challenge 3:** Extend the SO(3) left Jacobian to SE(3). The 6×6 left Jacobian has the block structure:

$$\mathcal{J}_l = \begin{pmatrix} J_l & Q \\ \mathbf{0} & J_l \end{pmatrix}$$

where $Q$ involves the $V$ matrix derivatives. Implement this and verify numerically.

<details><summary>Answer</summary>

```python
def se3_left_jacobian(xi):
    """6x6 left Jacobian for SE(3) (approximate)."""
    v, w = xi[:3], xi[3:]
    Jl = so3_left_jacobian(w)
    # Q matrix (simplified first-order approximation)
    theta = np.linalg.norm(w)
    if theta < 1e-10:
        Q = 0.5 * hat3(v)
    else:
        K = hat3(w); V_hat = hat3(v)
        Q = (0.5 * V_hat
             + (theta - np.sin(theta))/theta**3 * (K@V_hat + V_hat@K + K@V_hat@K)
             + ((1-theta**2/2-np.cos(theta))/theta**4)
               * (K@K@V_hat + V_hat@K@K - 3*K@V_hat@K))
    J6 = np.zeros((6,6))
    J6[:3,:3] = Jl
    J6[:3,3:] = Q
    J6[3:,3:] = Jl
    return J6
```
</details>

---

## Connections
- **Back:** [Day 80: Exp/Log](day-80-exponential-map.md) — Jacobians are derivatives of the exp map
- **Back:** [Day 82: Adjoint](day-82-adjoint-bch.md) — $J_l^{-1}$ appears in the BCH formula
- **Forward:** [Day 84: Review](day-84-week12-review.md) — Jacobians complete the SE(3) toolkit
- **OKS:** SLAM pose-graph optimization uses $J_r^{-1}$ to compute the Gauss-Newton update step

## Self-Check
- [ ] I can state the closed-form left and right Jacobians for SO(3)
- [ ] I understand the relation $J_r(\phi) = J_l(-\phi)$
- [ ] I can verify Jacobians against finite differences
- [ ] I know why Jacobians are needed for optimization on manifolds (SLAM)
