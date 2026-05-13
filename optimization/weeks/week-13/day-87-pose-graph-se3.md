# Day 87: Pose-Graph SLAM on SE(3)

> **Phase VI — Lie Groups & Manifold Optimization | Week 13 | 2.5 hours**
> *Extend pose-graph optimization to 3D with robust kernels for outlier resilience.*

## Navigation
- **Previous:** [Day 86: Pose-Graph SLAM on SE(2)](day-86-pose-graph-se2.md)
- **Next:** [Day 88: Ceres & GTSAM for Manifold Optimization](day-88-ceres-gtsam-manifold.md)
- **Week:** [Week 13 Overview](../README.md)
- **Chapter:** [Chapter 09: Lie Groups Reference](../../09-lie-groups.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
While OKS AMRs primarily operate in 2D, 3D pose-graph SLAM becomes critical for multi-floor warehouses with ramps/elevators, or when fusing lidar point clouds that capture 3D structure. Robust kernels protect against spurious loop closures — a real problem when perceptual aliasing causes false place matches in repetitive warehouse aisles.

---

## Theory (45 min)

### 87.1 SE(3) Edge Error

For poses $T_i, T_j \in \text{SE}(3)$ and measurement $Z_{ij} \in \text{SE}(3)$:

$$e_{ij} = \text{Log}\!\left(Z_{ij}^{-1} \cdot T_i^{-1} \cdot T_j\right) \in \mathbb{R}^6$$

The 6-vector $e_{ij}$ contains 3 rotation and 3 translation components. The cost for a single edge:

$$c_{ij} = e_{ij}^T \, \Omega_{ij} \, e_{ij}$$

where $\Omega_{ij} \in \mathbb{R}^{6 \times 6}$ is the information matrix encoding measurement confidence for all 6 DOF.

### 87.2 Information Matrices in 6D

A typical structure for $\Omega_{ij}$:

$$\Omega_{ij} = \begin{pmatrix} \Omega_R & 0 \\ 0 & \Omega_t \end{pmatrix}, \quad \Omega_R = \sigma_R^{-2} I_3, \quad \Omega_t = \sigma_t^{-2} I_3$$

Rotation and translation uncertainties are often decoupled. Lidar-based edges have high translational confidence ($\sigma_t \sim 0.01$m) but lower rotational confidence. IMU-based edges have the opposite pattern.

### 87.3 Robust Kernels

Outlier edges (false loop closures) cause catastrophic failure with squared costs. Robust kernels replace $c_{ij}$ with $\rho(c_{ij})$ where $\rho$ grows sub-quadratically:

| Kernel | $\rho(s)$ | Behavior |
|--------|-----------|----------|
| **Huber** | $s$ if $s \le k^2$; $2k\sqrt{s} - k^2$ otherwise | Linear beyond $k$ |
| **Cauchy** | $k^2 \log(1 + s/k^2)$ | Logarithmic growth |
| **Tukey** | $k^2(1-(1-s/k^2)^3)/6$ if $s \le k^2$; $k^2/6$ | Hard rejection beyond $k$ |

The **weighted residual** becomes $w_{ij} \cdot e_{ij}$ where $w_{ij} = \sqrt{\rho'(c_{ij})}$.

### 87.4 Manifold Levenberg-Marquardt

Combine robust kernels with damping for reliable convergence:

$$(H + \lambda \text{diag}(H)) \, \delta = -b$$

where $H$ and $b$ incorporate the robust weights. Adaptive $\lambda$: decrease on successful steps, increase on rejected steps. The gain ratio $\varrho = (\text{actual decrease}) / (\text{predicted decrease})$ guides the adaptation.

---

## Implementation (60 min)

```python
"""SE(3) Pose-Graph SLAM with Robust Kernels."""
import numpy as np

# --- SO(3) / SE(3) helpers ---
def so3_exp(phi):
    angle = np.linalg.norm(phi)
    if angle < 1e-10:
        return np.eye(3) + skew(phi)
    axis = phi / angle
    K = skew(axis)
    return np.eye(3) + np.sin(angle)*K + (1 - np.cos(angle))*(K @ K)

def so3_log(R):
    cos_a = np.clip((np.trace(R) - 1) / 2, -1, 1)
    angle = np.arccos(cos_a)
    if angle < 1e-10:
        return np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / 2
    return angle / (2 * np.sin(angle)) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def se3_exp(xi):
    rho, phi = xi[:3], xi[3:]
    R = so3_exp(phi)
    angle = np.linalg.norm(phi)
    if angle < 1e-10:
        V = np.eye(3)
    else:
        K = skew(phi / angle)
        V = np.eye(3) + (1-np.cos(angle))/angle**2 * K + (angle-np.sin(angle))/angle**3 * (K@K)
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = V @ rho
    return T

def se3_log(T):
    R, t = T[:3,:3], T[:3,3]
    phi = so3_log(R)
    angle = np.linalg.norm(phi)
    if angle < 1e-10:
        V_inv = np.eye(3)
    else:
        K = skew(phi / angle)
        half = angle / 2
        V_inv = np.eye(3) - 0.5*skew(phi) + (1 - half/np.tan(half))/angle**2 * (skew(phi) @ skew(phi))
    rho = V_inv @ t
    return np.concatenate([rho, phi])

def se3_inv(T):
    Ti = np.eye(4); Ti[:3,:3] = T[:3,:3].T; Ti[:3,3] = -T[:3,:3].T @ T[:3,3]
    return Ti

# --- Robust kernels ---
def huber_weight(sq_err, k=1.0):
    if sq_err <= k**2:
        return 1.0
    return k / np.sqrt(sq_err)

def cauchy_weight(sq_err, k=1.0):
    return 1.0 / (1.0 + sq_err / k**2)

# --- Pose-graph optimizer ---
def se3_pose_graph(poses_init, edges, z_list, omega_list,
                   kernel='none', k_param=1.0, max_iter=30, tol=1e-8):
    n = len(poses_init)
    poses = [T.copy() for T in poses_init]
    dim = 6
    costs = []

    for it in range(max_iter):
        N = dim * n
        H = np.zeros((N, N))
        b = np.zeros(N)
        cost = 0.0

        for (i, j), z_ij, omega in zip(edges, z_list, omega_list):
            e = se3_log(se3_inv(z_ij) @ se3_inv(poses[i]) @ poses[j])
            sq = e @ omega @ e
            cost += sq

            # Robust weight
            if kernel == 'huber':
                w = huber_weight(sq, k_param)
            elif kernel == 'cauchy':
                w = cauchy_weight(sq, k_param)
            else:
                w = 1.0

            A, B = -np.eye(dim), np.eye(dim)
            w_omega = w * omega
            ii, jj = dim*i, dim*j
            H[ii:ii+dim, ii:ii+dim] += A.T @ w_omega @ A
            H[ii:ii+dim, jj:jj+dim] += A.T @ w_omega @ B
            H[jj:jj+dim, ii:ii+dim] += B.T @ w_omega @ A
            H[jj:jj+dim, jj:jj+dim] += B.T @ w_omega @ B
            b[ii:ii+dim] += A.T @ w_omega @ e
            b[jj:jj+dim] += B.T @ w_omega @ e

        costs.append(cost)
        # Anchor pose 0
        H[:dim, :] = 0; H[:, :dim] = 0
        H[:dim, :dim] = np.eye(dim) * 1e8
        b[:dim] = 0

        delta = np.linalg.solve(H + 1e-4*np.eye(N), -b)
        if np.linalg.norm(delta) < tol:
            break
        for k in range(n):
            poses[k] = poses[k] @ se3_exp(delta[dim*k:dim*k+dim])

    return poses, costs

# --- Demo: 6-pose loop with one outlier ---
np.random.seed(42)
n = 6
true_poses = [np.eye(4)]
for i in range(n - 1):
    dT = se3_exp(np.array([0.5, 0, 0, 0, 0, np.pi/3]))
    true_poses.append(true_poses[-1] @ dT)

edges, z_list, omega_list = [], [], []
omega = np.diag([100]*3 + [400]*3)
for i in range(n - 1):
    z = se3_inv(true_poses[i]) @ true_poses[i+1]
    noise = np.random.randn(6) * 0.02
    z_noisy = z @ se3_exp(noise)
    edges.append((i, i+1)); z_list.append(z_noisy); omega_list.append(omega)

# Loop closure (clean)
z_lc = se3_inv(true_poses[-1]) @ true_poses[0]
edges.append((n-1, 0)); z_list.append(z_lc @ se3_exp(np.random.randn(6)*0.02)); omega_list.append(omega)

# Outlier edge (corrupted)
z_outlier = se3_exp(np.random.randn(6) * 2.0)
edges.append((1, 4)); z_list.append(z_outlier); omega_list.append(omega)

init_poses = [np.eye(4)]
for i in range(n - 1):
    init_poses.append(init_poses[-1] @ z_list[i])

# Compare no kernel vs Huber
_, costs_none = se3_pose_graph(init_poses, edges, z_list, omega_list, kernel='none')
_, costs_huber = se3_pose_graph(init_poses, edges, z_list, omega_list, kernel='huber', k_param=1.0)

print(f"No kernel  — final cost: {costs_none[-1]:.4f}")
print(f"Huber k=1  — final cost: {costs_huber[-1]:.4f}")
```

---

## Practice Problems (45 min)

### Problem 1: SE(3) Error Derivation
Derive the error $e_{ij} = \text{Log}(Z_{ij}^{-1} T_i^{-1} T_j)$ and show that $e_{ij} = 0$ iff $T_j = T_i \cdot Z_{ij}$.

<details><summary>Answer</summary>

If $T_j = T_i Z_{ij}$, then $T_i^{-1} T_j = Z_{ij}$, so $Z_{ij}^{-1} T_i^{-1} T_j = I_4$, and $\text{Log}(I) = 0_6$. Conversely, $e_{ij} = 0 \Rightarrow Z_{ij}^{-1} T_i^{-1} T_j = I \Rightarrow T_j = T_i Z_{ij}$.
</details>

### Problem 2: Robust Kernel Comparison
Add a Cauchy kernel to the optimizer. On the outlier dataset, compare Huber (k=1.0), Cauchy (k=0.5), and no kernel. Which best rejects the outlier?

<details><summary>Answer</summary>

Cauchy with small $k$ most aggressively down-weights outliers. Ranking: Cauchy < Huber < none for final cost on the inlier edges. However, Cauchy can be too aggressive — if $k$ is too small, it also down-weights legitimate measurements with moderate noise.
</details>

### Problem 3: 6×6 Information Design
Design an information matrix for a lidar-based loop closure with σ_t = 0.02m and σ_R = 0.5°. What does the condition number of $\Omega$ tell you?

<details><summary>Answer</summary>

$\Omega = \text{diag}(1/0.02^2, 1/0.02^2, 1/0.02^2, 1/(0.5\pi/180)^2, 1/(0.5\pi/180)^2, 1/(0.5\pi/180)^2) = \text{diag}(2500, 2500, 2500, 13132, 13132, 13132)$. Condition number ≈ 5.3, well-conditioned. High condition numbers cause GN to under-update poorly constrained DOFs.
</details>

---

## Expert Challenges

### Challenge 1: Switchable Constraints
Implement switchable constraints (Sünderhauf & Protzel, 2012): augment each loop closure with a switch variable $s_{ij} \in [0,1]$ that scales the information matrix. The optimizer learns to set $s \to 0$ for outliers.

<details><summary>Answer</summary>

Add variable $s_{ij}$ per loop closure. Modified cost: $s_{ij}^2 \, e_{ij}^T \Omega e_{ij} + \Omega_s (s_{ij} - 1)^2$. The prior $\Omega_s$ penalizes deviation from $s=1$. During optimization, outlier edges drive $s_{ij} \to 0$, effectively removing them. Joint optimization over poses and switches.
</details>

### Challenge 2: SE(3) Manifold LM
Implement full Levenberg-Marquardt with adaptive $\lambda$ using gain ratio $\varrho$. Test convergence from poor initialization.

<details><summary>Answer</summary>

Gain ratio: $\varrho = (F(x_k) - F(x_k \oplus \delta)) / (\delta^T(\lambda\delta - b))$. If $\varrho > 0.75$: $\lambda \leftarrow \lambda/3$. If $\varrho < 0.25$: $\lambda \leftarrow 2\lambda$. Accept step only if $\varrho > 0$. This provides robust convergence even when initialization is far from optimal.
</details>

### Challenge 3: Covariance Propagation
After optimizing, compute the marginal covariance of the relative transform $T_i^{-1} T_j$ for any pair $(i,j)$.

<details><summary>Answer</summary>

Marginal covariance of relative transform: $\Sigma_{ij}^{\text{rel}} = F_{ij} \Sigma F_{ij}^T$ where $F_{ij} = [-\text{Ad}(T_j^{-1}T_i) \;\; I_6]$ selects and transforms the relevant blocks from the joint covariance $\Sigma = H^{-1}$. In practice, extract blocks $\Sigma_{ii}, \Sigma_{ij}, \Sigma_{jj}$ without inverting the full Hessian using sparse methods.
</details>

---

## Connections
- **Prerequisites**: Day 86 (SE(2) pose graph), Day 79 (SE(3) basics), Day 80 (Exp/Log)
- **Forward**: Day 88 connects to production solvers (Ceres, GTSAM) that implement all of this
- **OKS**: 3D SLAM backend for lidar-based warehouse mapping, robust to perceptual aliasing

## Self-Check
- [ ] I can write the SE(3) edge error from memory
- [ ] I can explain why robust kernels are needed and compare Huber vs Cauchy
- [ ] I can implement SE(3) pose-graph optimization with robust weighting
- [ ] I understand the 6×6 information matrix structure and its physical meaning
