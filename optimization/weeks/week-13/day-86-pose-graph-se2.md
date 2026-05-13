# Day 86: Pose-Graph SLAM on SE(2)

> **Phase VI — Lie Groups & Manifold Optimization | Week 13 | 2.5 hours**
> *Turn odometry and loop closures into a graph and optimize all poses simultaneously.*

## Navigation
- **Previous:** [Day 85: Manifold Gauss-Newton](day-85-manifold-gauss-newton.md)
- **Next:** [Day 87: Pose-Graph SLAM on SE(3)](day-87-pose-graph-se3.md)
- **Week:** [Week 13 Overview](../README.md)
- **Chapter:** [Chapter 09: Lie Groups Reference](../../09-lie-groups.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
OKS AMRs build a map of the warehouse floor by accumulating odometry and detecting loop closures (revisiting known locations). Pose-graph optimization on SE(2) corrects the accumulated drift, producing a globally consistent map. This is the foundation of 2D SLAM used in the OKS mapping pipeline.

---

## Theory (45 min)

### 86.1 Graph-Based SLAM Formulation

A **pose graph** consists of:
- **Nodes** $x_i \in \text{SE}(2)$: robot poses at discrete times
- **Edges** $(i, j)$ with measurement $z_{ij} \in \text{SE}(2)$: relative transform between poses $i$ and $j$

Edges come from two sources:
- **Odometry edges**: sequential measurements $z_{i,i+1}$ from wheel encoders
- **Loop-closure edges**: non-sequential measurements $z_{i,j}$ from place recognition

### 86.2 Error Function on SE(2)

The error for edge $(i, j)$ measures how well poses $x_i, x_j$ agree with measurement $z_{ij}$:

$$e_{ij} = \text{Log}\!\left(z_{ij}^{-1} \cdot x_i^{-1} \cdot x_j\right) \in \mathbb{R}^3$$

This is zero when $x_i^{-1} x_j = z_{ij}$ exactly. The full cost function:

$$F(\mathbf{x}) = \sum_{(i,j)} e_{ij}^T \, \Omega_{ij} \, e_{ij}$$

where $\Omega_{ij} \in \mathbb{R}^{3\times 3}$ is the **information matrix** (inverse covariance) of the measurement.

### 86.3 Information Matrices and Sparsity

Each edge contributes a $3 \times 3$ block to the Hessian at positions $(i,i)$, $(i,j)$, $(j,i)$, $(j,j)$. The full Hessian $H \in \mathbb{R}^{3n \times 3n}$ is **sparse**: only connected nodes have non-zero blocks. For a chain of $n$ poses with a few loop closures, $H$ is nearly tridiagonal.

### 86.4 Anchoring the First Node

The pose graph has a gauge freedom — translating/rotating all poses simultaneously does not change edge errors. We fix this by **anchoring** $x_0$: either hold it constant (remove its variables) or add a strong prior $e_0 = \text{Log}(x_0^{-1} \cdot x_{\text{prior}})$ with large $\Omega_0$.

### 86.5 Jacobians of the Edge Error

Perturbing $x_i \to x_i \cdot \text{Exp}(\delta_i)$ and $x_j \to x_j \cdot \text{Exp}(\delta_j)$:

$$\frac{\partial e_{ij}}{\partial \delta_i} \approx -J_r^{-1}(e_{ij}) \cdot \text{Ad}(x_j^{-1} x_i), \qquad \frac{\partial e_{ij}}{\partial \delta_j} \approx J_r^{-1}(e_{ij})$$

For small errors, $J_r^{-1} \approx I$, giving the approximate Jacobians $A_{ij} \approx -\text{Ad}(\cdot)$ and $B_{ij} \approx I$.

---

## Implementation (60 min)

```python
"""Pose-Graph SLAM on SE(2) — 10-node loop."""
import numpy as np

def se2_exp(xi):
    dx, dy, dth = xi
    c, s = np.cos(dth), np.sin(dth)
    if abs(dth) < 1e-10:
        return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
    V = np.array([[s/dth, -(1-c)/dth], [(1-c)/dth, s/dth]])
    t = V @ np.array([dx, dy])
    return np.array([[c, -s, t[0]], [s, c, t[1]], [0, 0, 1]])

def se2_log(T):
    th = np.arctan2(T[1, 0], T[0, 0])
    if abs(th) < 1e-10:
        return np.array([T[0, 2], T[1, 2], th])
    half = th / 2
    V_inv = np.eye(2) * (half / np.tan(half))
    V_inv += np.array([[0, half], [-half, 0]])
    rho = V_inv @ T[:2, 2]
    return np.array([rho[0], rho[1], th])

def se2_inv(T):
    R = T[:2, :2]; t = T[:2, 2]
    Ti = np.eye(3); Ti[:2, :2] = R.T; Ti[:2, 2] = -R.T @ t
    return Ti

# --- Build ground-truth poses in a loop ---
def build_loop_poses(n, step=1.0):
    """n poses forming a closed polygon."""
    poses = [np.eye(3)]
    dth = 2 * np.pi / n
    for i in range(n - 1):
        dT = se2_exp(np.array([step, 0, dth]))
        poses.append(poses[-1] @ dT)
    return poses

def add_noise_se2(T, sigma_t=0.05, sigma_r=0.02):
    noise = np.array([np.random.randn()*sigma_t, np.random.randn()*sigma_t,
                      np.random.randn()*sigma_r])
    return T @ se2_exp(noise)

# --- Pose graph optimization ---
def pose_graph_optimize(poses_init, edges, measurements, omegas,
                        anchor=0, max_iter=30, tol=1e-8):
    n = len(poses_init)
    poses = [T.copy() for T in poses_init]
    dim = 3
    costs = []

    for it in range(max_iter):
        total_dim = dim * n
        H = np.zeros((total_dim, total_dim))
        b = np.zeros(total_dim)
        cost = 0.0

        for (i, j), z_ij, omega in zip(edges, measurements, omegas):
            e_ij = se2_log(se2_inv(z_ij) @ se2_inv(poses[i]) @ poses[j])
            cost += e_ij @ omega @ e_ij

            # Approximate Jacobians (small-error regime)
            A_ij = -np.eye(dim)  # d e / d delta_i
            B_ij = np.eye(dim)   # d e / d delta_j

            # Accumulate into H and b
            ii, jj = dim*i, dim*j
            H[ii:ii+dim, ii:ii+dim] += A_ij.T @ omega @ A_ij
            H[ii:ii+dim, jj:jj+dim] += A_ij.T @ omega @ B_ij
            H[jj:jj+dim, ii:ii+dim] += B_ij.T @ omega @ A_ij
            H[jj:jj+dim, jj:jj+dim] += B_ij.T @ omega @ B_ij
            b[ii:ii+dim] += A_ij.T @ omega @ e_ij
            b[jj:jj+dim] += B_ij.T @ omega @ e_ij

        costs.append(cost)

        # Anchor: fix first pose by zeroing its rows/cols
        ai = dim * anchor
        H[ai:ai+dim, :] = 0; H[:, ai:ai+dim] = 0
        H[ai:ai+dim, ai:ai+dim] = np.eye(dim) * 1e6
        b[ai:ai+dim] = 0

        delta = np.linalg.solve(H, -b)
        if np.linalg.norm(delta) < tol:
            break
        for k in range(n):
            poses[k] = poses[k] @ se2_exp(delta[dim*k:dim*k+dim])

    return poses, costs

# --- Build problem ---
np.random.seed(42)
n = 10
true_poses = build_loop_poses(n, step=1.0)

# Odometry edges (sequential)
edges, measurements, omegas = [], [], []
odom_omega = np.diag([100, 100, 400])
for i in range(n - 1):
    z = se2_inv(true_poses[i]) @ true_poses[i+1]
    z_noisy = add_noise_se2(z, 0.05, 0.02)
    edges.append((i, i+1)); measurements.append(z_noisy); omegas.append(odom_omega)

# Loop closure: last -> first
z_lc = se2_inv(true_poses[-1]) @ true_poses[0]
z_lc_noisy = add_noise_se2(z_lc, 0.05, 0.02)
edges.append((n-1, 0)); measurements.append(z_lc_noisy); omegas.append(odom_omega)

# Noisy initial poses (odometry only, no loop closure)
init_poses = [np.eye(3)]
for i in range(n - 1):
    init_poses.append(init_poses[-1] @ measurements[i])

optimized, costs = pose_graph_optimize(init_poses, edges, measurements, omegas)

print(f"Iterations: {len(costs)}")
print(f"Cost: {costs[0]:.4f} -> {costs[-1]:.6f}")
for i in range(n):
    err = np.linalg.norm(se2_log(se2_inv(true_poses[i]) @ optimized[i]))
    print(f"  Pose {i}: error = {err:.4f}")
```

---

## Practice Problems (45 min)

### Problem 1: Build a Pose Graph
Given odometry $\Delta T_{01}, \Delta T_{12}, \Delta T_{23}$ and a loop closure $z_{30}$, write out the full cost function with edge errors and information matrices.

<details><summary>Answer</summary>

$F = \sum_{k=0}^{2} e_{k,k+1}^T \Omega_{\text{odom}} e_{k,k+1} + e_{30}^T \Omega_{\text{lc}} e_{30}$ where $e_{ij} = \text{Log}(z_{ij}^{-1} x_i^{-1} x_j)$. The loop closure typically has lower information (higher uncertainty) than odometry: $\Omega_{\text{lc}} \preceq \Omega_{\text{odom}}$.
</details>

### Problem 2: Analyze Sparsity
For a 100-pose chain with 3 loop closures, how many non-zero $3\times3$ blocks does the Hessian $H$ have? What fraction of $H$ is non-zero?

<details><summary>Answer</summary>

Each odometry edge contributes 4 blocks (ii, ij, ji, jj). 99 odometry edges → 396 blocks. 3 loop closures → 12 blocks. Total: 408 blocks out of $100^2 = 10000$ possible, so ~4.1% non-zero. A sparse solver exploits this structure for $O(n)$ cost on chain-like graphs.
</details>

### Problem 3: Effect of Loop Closures
Run the pose graph with and without the loop closure edge. Compare final pose errors. How much does a single loop closure reduce drift?

<details><summary>Answer</summary>

Without loop closure, accumulated drift grows linearly — the last pose has the largest error (~0.5m for 10 poses at σ=0.05m). With a single loop closure connecting last to first, drift is distributed across all poses, reducing maximum error by roughly $\sqrt{n}$ factor.
</details>

---

## Expert Challenges

### Challenge 1: Sparse Solver
Replace `np.linalg.solve` with a sparse Cholesky solver using `scipy.sparse`. Benchmark on a 500-pose graph.

<details><summary>Answer</summary>

Use `scipy.sparse.csc_matrix` for $H$ and `scipy.sparse.linalg.spsolve` for the linear system. For a 500-pose chain, sparse solve is ~50× faster than dense. Ordering (AMD/RCM) further reduces fill-in.
</details>

### Challenge 2: Covariance Recovery
After optimization, the inverse of the Hessian $H^{-1}$ gives the marginal covariances. Extract the $3\times3$ block for each pose and visualize uncertainty ellipses.

<details><summary>Answer</summary>

$\Sigma = H^{-1}$. The $(i,i)$ diagonal block $\Sigma_{ii}$ gives the marginal covariance of pose $i$. Poses near the anchor have small uncertainty; poses far from anchor (or between loop closures) have larger uncertainty. Visualize as 2σ ellipses from the $(x,y)$ subblock of $\Sigma_{ii}$.
</details>

### Challenge 3: Incremental SLAM
Instead of batch optimization, implement incremental updates: add one pose + edges at a time and re-optimize. Compare convergence with batch.

<details><summary>Answer</summary>

Incremental: after adding pose $x_k$ with edge $(k-1, k)$, initialize $x_k = x_{k-1} \cdot z_{k-1,k}$ and run 1–3 GN iterations on the full graph. This amortizes computation and converges faster than re-optimizing from scratch each time, since previous poses are already near-optimal.
</details>

---

## Connections
- **Prerequisites**: Day 85 (Manifold GN), Day 78 (SE(2) basics)
- **Forward**: Day 87 extends this to SE(3) with robust kernels
- **OKS**: 2D pose-graph optimization is the backend of warehouse floor mapping

## Self-Check
- [ ] I can formulate the pose-graph SLAM cost function on SE(2)
- [ ] I can explain anchoring and why it is necessary
- [ ] I can implement and run a 10-node pose graph optimizer
- [ ] I understand the sparsity structure of the Hessian
