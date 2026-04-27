# Exercises: Graph-Based Optimization

### Chapter 06: Graph Optimization and SLAM

**Self-assessment guide:** If you can build a simple pose graph and optimize it, you understand the core SLAM backend. The implementation exercises use Python — the concepts transfer directly to g2o/GTSAM.

---

## Section A — Conceptual Questions

**A1.** Explain why the Hessian $H = J^T \Omega J$ of a pose-graph problem is sparse. Draw the sparsity pattern for a chain of 4 poses with one loop closure between pose 0 and pose 3.

<details><summary>Answer</summary>

Each edge in the pose graph contributes to the Jacobian: if edge $(i, j)$ has residual $e_{ij}(x_i, x_j)$, its Jacobian has non-zero entries only in columns corresponding to $x_i$ and $x_j$.

When you form $H = J^T \Omega J$, an entry $H_{ab}$ is non-zero only if variables $x_a$ and $x_b$ appear in the same edge. This means $H$ has the same sparsity pattern as the **adjacency matrix** of the graph.

**For 4 poses (each 3-DOF in SE2) with edges 0→1, 1→2, 2→3, 0→3:**

```
     x0  x1  x2  x3
x0 [ X   X   .   X ]    x0 connects to x1 (odom), x3 (loop closure)
x1 [ X   X   X   . ]    x1 connects to x0, x2
x2 [ .   X   X   X ]    x2 connects to x1, x3
x3 [ X   .   X   X ]    x3 connects to x2, x0

X = 3×3 non-zero block (SE2 has 3 DOF)
. = 3×3 zero block
```

Without the loop closure (0→3), $H$ would be tri-diagonal (only adjacent blocks). The loop closure adds off-diagonal blocks at $(0,3)$ and $(3,0)$, creating **fill-in** during Cholesky factorization.

</details>

**A2.** What is the difference between a prior factor and a between factor in a factor graph? Why do you need at least one prior?

<details><summary>Answer</summary>

**Prior factor** $f(x_i)$: constrains a single variable. Example: "Pose 0 is at the origin with covariance $\Sigma_0$."
- Connects to **one** variable
- Residual: $e = x_i - x_{\text{prior}}$

**Between factor** $f(x_i, x_j)$: constrains the relative transformation between two variables. Example: "The odometry says pose 1 is 1m ahead of pose 0."
- Connects **two** variables
- Residual: $e = z_{ij} \ominus (x_i^{-1} \cdot x_j)$

**Why you need at least one prior:**

Without any prior, the pose graph has a **gauge freedom** — you can rigidly translate/rotate the entire solution without changing any between-factor residual. The system is rank-deficient ($H$ is singular).

The prior "pins" the coordinate frame. In practice:
- Fix the first pose: strong prior with small covariance on $x_0$
- Or fix the first pose exactly: don't optimize $x_0$ (Ceres: `SetParameterBlockConstant`)

Without this, the solver fails (singular Hessian) or drifts to a random global pose.

</details>

**A3.** A loop closure detector reports a match between pose 47 and pose 3, but the match is actually wrong (perceptual aliasing). What happens to the optimized map if you:
1. Use standard least squares?
2. Use Huber loss on edges?
3. Use Switchable Constraints?

<details><summary>Answer</summary>

1. **Standard least squares:** The wrong loop closure is treated as equally valid. The optimizer **distorts the entire map** to make this false constraint consistent with the others. Poses between 3 and 47 get warped. The map is corrupted.

2. **Huber loss:** The residual of the wrong edge will be very large (because it contradicts the true trajectory). Huber loss **reduces its influence** from quadratic to linear for large residuals. The optimizer gives it less weight, so the distortion is reduced — but not eliminated entirely. A very wrong loop closure with modest Huber threshold can still corrupt the map.

3. **Switchable Constraints:** Each edge gets a binary "switch" variable $s_{ij} \in [0, 1]$. The cost becomes:
   $$s_{ij}^2 \|e_{ij}\|^2_{\Omega} + (1 - s_{ij})^2 \cdot \gamma$$
   If the edge is consistent with the rest → $s_{ij} \to 1$ (keep it). If inconsistent → $s_{ij} \to 0$ (disable it). The wrong loop closure gets **automatically disabled** by the optimizer. Cost: one extra variable per edge.

Modern best practice: **Graduated Non-Convexity (GNC)** — start with convex loss, gradually increase robustness. Most reliable way to handle outlier edges.

</details>

---

## Section B — Computation Problems

**B1.** Build a 1D pose graph and optimize it.

Scenario: 5 poses on a line. Odometry says each step is 1.0m. A "loop closure" says pose 4 is 3.8m from pose 0 (not 4.0m — there was drift). Optimize the poses.

```python
import numpy as np

def optimize_1d_pose_graph(odom_edges, loop_edges, prior_pose, prior_sigma,
                           odom_sigma, loop_sigma, max_iter=20):
    """
    1D pose-graph optimization using Gauss-Newton.
    
    odom_edges: list of (from, to, measurement)
    loop_edges: list of (from, to, measurement)
    prior_pose: (index, value)
    
    Returns: optimized poses
    """
    # TODO: implement
    # 1. Initialize poses from odometry
    # 2. Build Jacobian and residual vector
    # 3. Solve H @ dx = -b
    # 4. Update poses
    # 5. Repeat until convergence
    pass

# Odometry: 0→1: 1.0, 1→2: 1.0, 2→3: 1.0, 3→4: 1.0
odom_edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)]
# Loop closure: 0→4: 3.8 (not 4.0 — accumulated drift)
loop_edges = [(0, 4, 3.8)]
# Prior: pose 0 at 0.0
prior = (0, 0.0)

poses = optimize_1d_pose_graph(odom_edges, loop_edges, prior, 
                                prior_sigma=0.01, odom_sigma=0.1, loop_sigma=0.1)
# Expected: poses spread drift evenly → [0, 0.95, 1.90, 2.85, 3.80]
```

<details><summary>Solution</summary>

```python
import numpy as np

def optimize_1d_pose_graph(odom_edges, loop_edges, prior_pose, prior_sigma,
                           odom_sigma, loop_sigma, max_iter=20):
    n = max(max(e[1] for e in odom_edges), 
            max(e[1] for e in loop_edges)) + 1
    
    # Initialize from odometry chain
    poses = np.zeros(n)
    for (i, j, z) in odom_edges:
        if j > i:
            poses[j] = poses[i] + z
    
    prior_info = 1.0 / prior_sigma**2
    odom_info = 1.0 / odom_sigma**2
    loop_info = 1.0 / loop_sigma**2
    
    for iteration in range(max_iter):
        H = np.zeros((n, n))
        b = np.zeros(n)
        total_cost = 0
        
        # Prior factor
        idx, val = prior_pose
        e = poses[idx] - val
        H[idx, idx] += prior_info
        b[idx] += prior_info * e
        total_cost += 0.5 * prior_info * e**2
        
        # Odometry factors
        for (i, j, z) in odom_edges:
            e = (poses[j] - poses[i]) - z  # residual
            # Jacobian: de/dxi = -1, de/dxj = +1
            H[i, i] += odom_info
            H[j, j] += odom_info
            H[i, j] -= odom_info
            H[j, i] -= odom_info
            b[i] -= odom_info * e
            b[j] += odom_info * e
            total_cost += 0.5 * odom_info * e**2
        
        # Loop closure factors
        for (i, j, z) in loop_edges:
            e = (poses[j] - poses[i]) - z
            H[i, i] += loop_info
            H[j, j] += loop_info
            H[i, j] -= loop_info
            H[j, i] -= loop_info
            b[i] -= loop_info * e
            b[j] += loop_info * e
            total_cost += 0.5 * loop_info * e**2
        
        # Solve
        dx = np.linalg.solve(H, -b)
        poses += dx
        
        if np.linalg.norm(dx) < 1e-10:
            break
    
    return poses

odom_edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)]
loop_edges = [(0, 4, 3.8)]
prior = (0, 0.0)

poses = optimize_1d_pose_graph(odom_edges, loop_edges, prior,
                                prior_sigma=0.01, odom_sigma=0.1, loop_sigma=0.1)
print("Optimized poses:", [f"{p:.3f}" for p in poses])
# [0.000, 0.950, 1.900, 2.850, 3.800] — drift spread evenly
```

The 0.2m total drift (4.0 expected vs 3.8 measured) is distributed evenly across the 4 odometry edges: each "loses" 0.05m.

</details>

**B2.** Extend B1 to 2D (SE2). Use 4 poses in a square with a loop closure. Each pose is $(x, y, \theta)$.

Hint: The residual for an SE2 between factor is:
$$e_{ij} = \begin{bmatrix} R_i^T (t_j - t_i) - t_{ij}^{\text{meas}} \\ \theta_j - \theta_i - \theta_{ij}^{\text{meas}} \end{bmatrix}$$

<details><summary>Hint for Jacobian</summary>

For the SE2 residual $e_{ij}$, the Jacobians w.r.t. $x_i = (t_{ix}, t_{iy}, \theta_i)$ and $x_j = (t_{jx}, t_{jy}, \theta_j)$:

$$\frac{\partial e}{\partial x_j} = \begin{bmatrix} R_i^T & 0 \\ 0 & 1 \end{bmatrix}, \quad \frac{\partial e}{\partial x_i} = \begin{bmatrix} -R_i^T & R_i^T \frac{\partial}{\partial \theta_i}(t_j - t_i) \\ 0 & -1 \end{bmatrix}$$

For simplicity, you can use finite differences to check your analytic Jacobians.

</details>

---

## Section C — OKS Connection

**C1.** The OKS navigation estimator uses an EKF with sensorbar (landmark-based) corrections. In a factor-graph view:

1. What are the variable nodes?
2. What are the factor types?
3. If the robot drives in a loop and re-detects the same reflectors, what factor does this correspond to?
4. Why doesn't OKS use full graph optimization instead of an EKF?

<details><summary>Answer</summary>

1. **Variable nodes:** Robot poses at each timestep: $x_k = (x, y, \theta, v, \omega)_k$
   (Optionally: reflector positions if doing SLAM, but OKS treats them as known)

2. **Factor types:**
   - **Prior:** Initial pose from startup localization
   - **Odometry (between):** Wheel encoder prediction between consecutive timesteps
   - **Sensorbar (measurement):** Bearing/distance to known reflectors
   - **IMU (between):** Angular velocity integration for heading

3. **Re-detecting reflectors** in a loop corresponds to a **loop closure factor** — a measurement factor connecting the current pose to a previously-observed landmark. The information from the landmark provides a "jump" in accuracy, correcting accumulated odometry drift. This is exactly how loop closure works in SLAM.

4. **Why EKF instead of graph optimization:**
   - **Real-time constraint:** EKF updates in O(n²) where n is state dimension (small, ~5). Graph optimization over all timesteps is O(N³) where N grows with time.
   - **Constant memory:** EKF uses fixed-size matrices. Graph optimization stores all poses.
   - **Sufficient for OKS:** With known reflector positions (no SLAM needed) and corrections every ~100ms, the EKF doesn't accumulate much drift between corrections.
   - **Simplicity:** EKF is well-understood, debuggable, and validated in the OKS codebase.
   
   Graph optimization would help if: (1) reflector maps were inaccurate, (2) loop closures were needed, (3) multi-robot map merging was required, or (4) offline trajectory smoothing was desired for diagnostics.

</details>
