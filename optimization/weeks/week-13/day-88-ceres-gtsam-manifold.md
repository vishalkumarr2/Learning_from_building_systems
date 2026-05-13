# Day 88: Ceres & GTSAM for Manifold Optimization

> **Phase VI — Lie Groups & Manifold Optimization | Week 13 | 2.5 hours**
> *Bridge from theory to production — see how real solvers handle manifold constraints.*

## Navigation
- **Previous:** [Day 87: Pose-Graph SLAM on SE(3)](day-87-pose-graph-se3.md)
- **Next:** [Day 89: IMU Preintegration](day-89-imu-preintegration.md)
- **Week:** [Week 13 Overview](../README.md)
- **Chapter:** [Chapter 09: Lie Groups Reference](../../09-lie-groups.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
The OKS software stack uses GTSAM in its ROS-based navigation pipeline for factor-graph inference, and Ceres for sensor calibration. Understanding how these solvers handle manifold constraints — local parameterizations in Ceres, on-manifold factors in GTSAM — is essential for debugging and extending the robot's estimation pipeline.

---

## Theory (45 min)

### 88.1 Ceres Solver: Local Parameterization

Ceres Solver models optimization on manifolds via **local parameterizations** (renamed **manifolds** in Ceres 2.2+). The idea:

- The state lives in an **ambient space** $\mathbb{R}^n$ (e.g., quaternion $\in \mathbb{R}^4$)
- A local parameterization maps a **tangent-space perturbation** $\delta \in \mathbb{R}^k$ to a new point
- `Plus(x, delta) -> x_new` implements $x \oplus \delta$
- `ComputeJacobian(x) -> J` gives $\frac{\partial (x \oplus \delta)}{\partial \delta}\big|_{\delta=0}$

For SO(3) via quaternion:

$$\text{Plus}(q, \delta) = q \otimes \text{Exp}_q(\delta), \quad \delta \in \mathbb{R}^3, \; q \in \mathbb{R}^4$$

This lets Ceres optimize 4-parameter quaternions with only 3 degrees of freedom, avoiding the singularity of minimal representations.

### 88.2 GTSAM: On-Manifold Factors

GTSAM takes a different approach — manifold structure is built into the variable types:

- `Rot3` represents SO(3) directly
- `Pose3` represents SE(3) directly
- `between(x_i, x_j, z_ij, noise_model)` creates a factor with error: $\text{Log}(z_{ij}^{-1} \cdot x_i^{-1} \cdot x_j)$

The factor graph is optimized using Levenberg-Marquardt or Gauss-Newton, with all retraction/lifting handled internally. GTSAM also provides **iSAM2** for incremental optimization using Bayes tree factorization.

### 88.3 Automatic Differentiation on Manifolds

Both solvers support automatic differentiation, but with manifold awareness:

**Ceres**: auto-diff operates in the ambient space; the local parameterization projects the gradient to the tangent space. The chain rule is:

$$\frac{\partial r}{\partial \delta} = \frac{\partial r}{\partial x} \cdot \frac{\partial (x \oplus \delta)}{\partial \delta}$$

**GTSAM**: uses expression-based auto-diff where the chain rule is applied through a tree of expressions, each aware of its manifold structure. Jacobians are computed in the tangent space directly.

### 88.4 Factor Graph Representation

A factor graph $G = (\mathcal{F}, \mathcal{X}, \mathcal{E})$ consists of:
- **Variable nodes** $\mathcal{X} = \{x_1, \ldots, x_n\}$ — states to estimate
- **Factor nodes** $\mathcal{F} = \{f_1, \ldots, f_m\}$ — soft constraints from measurements
- **Edges** $\mathcal{E}$ connecting factors to the variables they depend on

The joint probability is:

$$p(\mathcal{X}) \propto \prod_{k=1}^{m} f_k(\mathcal{X}_k) = \prod_{k=1}^{m} \exp\!\left(-\tfrac{1}{2}\|h_k(\mathcal{X}_k) - z_k\|_{\Sigma_k}^2\right)$$

MAP estimation = minimizing the sum of squared Mahalanobis residuals.

### 88.5 Comparison of Approaches

| Aspect | Ceres | GTSAM |
|--------|-------|-------|
| Manifold handling | Local parameterization / Manifold API | Native manifold types |
| Auto-diff | Jet-based, ambient space | Expression-based, tangent space |
| Incremental | Not built-in | iSAM2 (Bayes tree) |
| Sparsity | Automatic sparse Schur complement | Elimination via variable ordering |
| Language | C++ (Python bindings) | C++ (Python bindings) |
| Typical use | Calibration, BA | SLAM, state estimation |

---

## Implementation (60 min)

```python
"""Python equivalents of Ceres/GTSAM patterns for SE(3) optimization."""
import numpy as np

# Reuse SE(3) helpers from Day 87
def skew(v):
    return np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])

def so3_exp(phi):
    a = np.linalg.norm(phi)
    if a < 1e-10: return np.eye(3) + skew(phi)
    K = skew(phi/a)
    return np.eye(3) + np.sin(a)*K + (1-np.cos(a))*(K@K)

def so3_log(R):
    ca = np.clip((np.trace(R)-1)/2, -1, 1)
    a = np.arccos(ca)
    if a < 1e-10: return np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])/2
    return a/(2*np.sin(a))*np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])

def se3_exp(xi):
    rho, phi = xi[:3], xi[3:]
    R = so3_exp(phi); a = np.linalg.norm(phi)
    if a < 1e-10: V = np.eye(3)
    else:
        K = skew(phi/a)
        V = np.eye(3)+(1-np.cos(a))/a**2*K+(a-np.sin(a))/a**3*(K@K)
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = V @ rho
    return T

def se3_log(T):
    R, t = T[:3,:3], T[:3,3]; phi = so3_log(R); a = np.linalg.norm(phi)
    if a < 1e-10: return np.concatenate([t, phi])
    K = skew(phi)
    V_inv = np.eye(3) - 0.5*K + (1-a/(2*np.tan(a/2)))/a**2*(K@K)
    return np.concatenate([V_inv @ t, phi])

def se3_inv(T):
    Ti = np.eye(4); Ti[:3,:3] = T[:3,:3].T; Ti[:3,3] = -T[:3,:3].T @ T[:3,3]
    return Ti

# --- Ceres-style local parameterization ---
class SE3Manifold:
    """Mimics Ceres Manifold for SE(3)."""
    ambient_size = 12   # flattened [R|t]
    tangent_size = 6

    @staticmethod
    def plus(x_flat, delta):
        T = np.eye(4); T[:3,:3] = x_flat[:9].reshape(3,3); T[:3,3] = x_flat[9:]
        T_new = T @ se3_exp(delta)
        return np.concatenate([T_new[:3,:3].ravel(), T_new[:3,3]])

    @staticmethod
    def minus(y_flat, x_flat):
        Tx = np.eye(4); Tx[:3,:3] = x_flat[:9].reshape(3,3); Tx[:3,3] = x_flat[9:]
        Ty = np.eye(4); Ty[:3,:3] = y_flat[:9].reshape(3,3); Ty[:3,3] = y_flat[9:]
        return se3_log(se3_inv(Tx) @ Ty)

# --- GTSAM-style factor graph ---
class BetweenFactor:
    """Mimics gtsam.BetweenFactorPose3."""
    def __init__(self, i, j, z_ij, sigma):
        self.i, self.j = i, j
        self.z_ij = z_ij
        self.omega = np.diag(1.0 / np.array(sigma)**2)

    def error(self, poses):
        e = se3_log(se3_inv(self.z_ij) @ se3_inv(poses[self.i]) @ poses[self.j])
        return e

    def weighted_error(self, poses):
        e = self.error(poses)
        return e @ self.omega @ e

class FactorGraph:
    """Minimal GTSAM-style factor graph."""
    def __init__(self):
        self.factors = []
        self.initial = {}

    def add(self, factor):
        self.factors.append(factor)

    def insert(self, key, value):
        self.initial[key] = value.copy()

    def optimize(self, max_iter=30, tol=1e-8):
        n = len(self.initial)
        poses = [self.initial[k] for k in sorted(self.initial)]
        dim = 6
        for it in range(max_iter):
            N = dim * n
            H = np.zeros((N, N)); b = np.zeros(N); cost = 0
            for f in self.factors:
                e = f.error(poses)
                cost += e @ f.omega @ e
                A, B = -np.eye(dim), np.eye(dim)
                ii, jj = dim*f.i, dim*f.j
                H[ii:ii+dim, ii:ii+dim] += A.T@f.omega@A
                H[ii:ii+dim, jj:jj+dim] += A.T@f.omega@B
                H[jj:jj+dim, ii:ii+dim] += B.T@f.omega@A
                H[jj:jj+dim, jj:jj+dim] += B.T@f.omega@B
                b[ii:ii+dim] += A.T@f.omega@e
                b[jj:jj+dim] += B.T@f.omega@e
            # Anchor node 0
            H[:dim,:]=0; H[:,:dim]=0; H[:dim,:dim]=np.eye(dim)*1e8; b[:dim]=0
            delta = np.linalg.solve(H+1e-4*np.eye(N), -b)
            if np.linalg.norm(delta) < tol: break
            for k in range(n):
                poses[k] = poses[k] @ se3_exp(delta[dim*k:dim*k+dim])
        return poses, cost

# --- Demo ---
np.random.seed(42)
graph = FactorGraph()
sigma = [0.01]*3 + [0.005]*3  # translation then rotation

true_poses = [np.eye(4)]
for i in range(4):
    dT = se3_exp(np.array([1,0,0, 0,0,0.3]))
    true_poses.append(true_poses[-1] @ dT)

for i in range(4):
    z = se3_inv(true_poses[i]) @ true_poses[i+1]
    z_noisy = z @ se3_exp(np.random.randn(6)*0.02)
    graph.add(BetweenFactor(i, i+1, z_noisy, sigma))
    graph.insert(i, true_poses[i] @ se3_exp(np.random.randn(6)*0.05))
graph.insert(4, true_poses[4] @ se3_exp(np.random.randn(6)*0.05))

result, final_cost = graph.optimize()
print(f"Final cost: {final_cost:.6f}")
for i, T in enumerate(result):
    err = np.linalg.norm(se3_log(se3_inv(true_poses[i]) @ T))
    print(f"  Pose {i}: error = {err:.5f}")
```

---

## Practice Problems (45 min)

### Problem 1: Ceres vs GTSAM API Translation
Given a Ceres cost function for a between-factor, translate it to the equivalent GTSAM factor. What are the key API differences?

<details><summary>Answer</summary>

Ceres: define `CostFunction::Evaluate(parameters, residuals, jacobians)` with `LocalParameterization` for SE(3). GTSAM: define `NoiseModelFactor::evaluateError(values, jacobians)` with `Pose3` variables. Key difference: Ceres works in ambient space + projects; GTSAM works in tangent space directly. GTSAM factors return tangent-space errors natively.
</details>

### Problem 2: Custom Prior Factor
Design a prior factor that constrains a single pose $x_i$ to a known value $\bar{x}$ with covariance $\Sigma$. Write the error and Jacobian.

<details><summary>Answer</summary>

Error: $e = \text{Log}(\bar{x}^{-1} \cdot x_i) \in \mathbb{R}^6$. Jacobian w.r.t. perturbation $\delta_i$: $J \approx I_6$ (for small errors). Weighted cost: $e^T \Sigma^{-1} e$. This is equivalent to `PriorFactorPose3` in GTSAM or a unary cost function in Ceres with appropriate local parameterization.
</details>

### Problem 3: Solver Output Analysis
After running the optimizer, the solver reports: 10 iterations, final cost 0.0023, gradient norm 1e-9. What does each number tell you about the solution quality?

<details><summary>Answer</summary>

- **10 iterations**: reasonable convergence speed, not hitting max iterations
- **Final cost 0.0023**: sum of squared Mahalanobis errors; low means measurements are consistent with the solution. Compare with $\chi^2$ expectation: for $m$ edges with 6-DOF errors, expected cost ≈ $6m$ under Gaussian noise
- **Gradient norm 1e-9**: well below any practical threshold; the optimizer found a stationary point (local minimum or saddle, but GN on convex-like problems gives global minimum)
</details>

---

## Expert Challenges

### Challenge 1: iSAM2 Concept
Explain the Bayes tree data structure used by iSAM2. How does it enable incremental updates when new factors are added?

<details><summary>Answer</summary>

A Bayes tree represents the Cholesky factorization of the information matrix as a directed tree of cliques. When a new factor is added, only the cliques along the path from the affected variables to the root need to be re-eliminated. This gives $O(\log n)$ amortized update cost for sparse graphs, vs $O(n)$ for batch re-optimization.
</details>

### Challenge 2: Schur Complement for Landmarks
In bundle adjustment with poses and landmarks, use the Schur complement to eliminate landmarks and solve a reduced system in poses only.

<details><summary>Answer</summary>

Partition the Hessian: $\begin{pmatrix} H_{pp} & H_{pl} \\ H_{lp} & H_{ll} \end{pmatrix}\begin{pmatrix}\delta_p \\ \delta_l\end{pmatrix} = \begin{pmatrix} b_p \\ b_l \end{pmatrix}$. Since $H_{ll}$ is block-diagonal (landmarks don't share factors), its inverse is cheap. The Schur complement system: $(H_{pp} - H_{pl}H_{ll}^{-1}H_{lp})\delta_p = b_p - H_{pl}H_{ll}^{-1}b_l$. Solve for $\delta_p$ (small system), then back-substitute for $\delta_l$.
</details>

### Challenge 3: Mixed Manifold Variables
Design a factor graph with both SE(3) poses and $\mathbb{R}^3$ landmarks. How do the Jacobians differ for the two variable types?

<details><summary>Answer</summary>

For SE(3) poses: Jacobians are computed w.r.t. 6D tangent-space perturbations using Exp map. For $\mathbb{R}^3$ landmarks: standard Euclidean Jacobians (no retraction needed, $\oplus$ is just vector addition). The Hessian has 6×6 blocks for poses and 3×3 blocks for landmarks. Mixed-manifold optimization is straightforward — each variable type uses its own Plus/Minus operators independently.
</details>

---

## Connections
- **Prerequisites**: Days 85–87 (manifold GN, pose graphs)
- **Forward**: Day 89 (IMU preintegration) uses factor graphs with time-varying states
- **OKS**: GTSAM powers the navigation stack; Ceres handles calibration routines

## Self-Check
- [ ] I can explain Ceres local parameterization and GTSAM on-manifold factors
- [ ] I can implement a minimal factor graph optimizer matching GTSAM's API
- [ ] I understand how auto-diff works on manifolds in both frameworks
- [ ] I can design custom factors (prior, between, landmark) for a factor graph
