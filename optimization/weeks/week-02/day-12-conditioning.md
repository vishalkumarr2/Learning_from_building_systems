# Day 12: Function Landscapes & Conditioning

> **Phase I — Mathematical Foundations | Week 2 | 2.5 hours**
> *A well-conditioned problem is half-solved; an ill-conditioned one fights you at every step.*

## Navigation
- **Previous:** [Day 11: Automatic Differentiation](day-11-automatic-differentiation.md)
- **Next:** [Day 13: Iterative Methods (CG)](day-13-iterative-methods.md)
- **Week:** [Week 2 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
When OKS SLAM's Hessian has eigenvalues spanning $10^6$, gradient descent crawls along narrow valleys while overshooting across ridges. The condition number $\kappa(H)$ quantifies this. Preconditioning and variable scaling transform the problem into a rounder landscape where solvers converge fast. Ceres's `JACOBI` scaling and `CLUSTER_JACOBI` preconditioner exist for exactly this reason.

---

## Theory (45 min)

### 12.1 Condition Number

For a symmetric positive definite matrix $H$:

$$\kappa(H) = \frac{\lambda_{\max}}{\lambda_{\min}}$$

**Meaning:** A perturbation of $\delta$ in the input (RHS) causes up to $\kappa \cdot \delta$ perturbation in the solution.

| $\kappa(H)$ | Interpretation |
|-------------|----------------|
| $\approx 1$ | Perfectly conditioned (circular contours) |
| $10^3$ | Moderately ill-conditioned |
| $10^6$ | Ill-conditioned (common in SLAM) |
| $> 10^{15}$ | Effectively singular ($\approx \frac{1}{\epsilon_{\text{mach}}}$) |

### 12.2 Geometric View: Contour Plots

For $f(x) = \frac{1}{2} x^T H x$:
- $\kappa = 1$: circular contours → all directions have equal curvature → gradient points to minimum
- $\kappa \gg 1$: highly elliptical contours → gradient zig-zags → slow convergence

**Gradient descent iterations ∝ $\kappa$** for quadratic problems.

### 12.3 Sources of Ill-Conditioning in Robotics

1. **Scale mismatch:** Translation parameters (meters) vs rotation parameters (radians) in the same system
2. **Gauge freedom:** Absolute position is unobservable → Hessian is rank-deficient
3. **Poorly excited directions:** Some parameters barely affect the cost
4. **Distant landmarks:** Range observations have $\sim 1/r^2$ information

### 12.4 Preconditioning

Transform $Hx = g$ into $M^{-1}Hx = M^{-1}g$ where $M \approx H$.

$$\kappa(M^{-1}H) \ll \kappa(H)$$

**Common preconditioners:**
| Preconditioner | $M$ | Cost | Effectiveness |
|---------------|-----|------|---------------|
| Jacobi | $\text{diag}(H)$ | $O(n)$ | Moderate |
| Block Jacobi | Block-diagonal of $H$ | $O(n \cdot b^2)$ | Good for block-structured problems |
| Incomplete Cholesky | Sparse approximate $LL^T$ | $O(n \cdot \text{nnz})$ | Very good |
| Exact | $H$ itself | $O(n^3)$ | Perfect (but defeats the purpose) |

### 12.5 Variable Scaling

Before optimizing, scale variables so all have similar magnitude:

$$\tilde{x}_i = x_i / s_i$$

where $s_i$ is a characteristic scale. This is equivalent to preconditioning with $M = \text{diag}(s_1^2, \ldots, s_n^2)$.

**Ceres:** Uses `ParameterBlockLocalParameterization` and `ScaledLoss` to handle scaling. The `JACOBI` scaling option divides each column of the Jacobian by its norm.

### 12.6 When Conditioning Goes Wrong

An ill-conditioned Hessian means:
- Direct solvers lose digits of precision: $\log_{10}(\text{digits lost}) \approx \log_{10}(\kappa)$
- Iterative solvers converge slowly: CG needs $O(\sqrt{\kappa})$ iterations
- Gradient descent needs $O(\kappa)$ iterations
- The solver may report convergence but the solution is inaccurate

---

## Implementation (60 min)

### Exercise 1: Condition Number Visualization
```python
# See: code/week02/conditioning.py
```
Create quadratic functions with $\kappa = 1, 10, 100, 1000$. Plot contours and gradient descent paths.

### Exercise 2: Digits Lost
Solve $Hx = g$ for matrices with various $\kappa$. Measure actual digits of precision in the solution.

### Exercise 3: Preconditioning
Implement Jacobi preconditioning. Show it reduces the effective condition number and speeds up CG convergence.

### Exercise 4: Variable Scaling
Take a SLAM-like problem with mixed translation/rotation parameters. Show that scaling improves convergence.

---

## Practice (45 min)

1. A $2 \times 2$ matrix has eigenvalues 1 and 1000. Draw its contour plot and the gradient descent path from $(1, 1)$.
2. You solve $Ax = b$ and get 8 correct digits. The true condition number is $10^8$. How many digits could be correct?
3. Why does adding a regularizer $\lambda I$ to $H$ improve conditioning? What does it cost?
4. In SLAM, translation parameters are in meters and rotation in radians. If a robot moves 100m with 0.01 rad rotations, what's the rough condition number due to scale mismatch?
5. Explain why the Gauss-Newton Hessian $J^T J$ can be ill-conditioned even if $J$ is well-conditioned.
6. CG converges in $O(\sqrt{\kappa})$ iterations. If $\kappa = 10^6$, how many iterations? After Jacobi preconditioning reduces it to $10^3$?
7. Why is incomplete Cholesky a good preconditioner for sparse systems?
8. What's the condition number of a diagonal matrix? Of the identity matrix?

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Automatic scaling for SLAM</summary>

**Problem:** Design an automatic variable-scaling scheme for a pose-graph SLAM problem. Given the Jacobian structure (some columns correspond to x/y translation, some to θ rotation), propose and implement a scaling that brings all columns to similar norm.

**Solution:** Column scaling: $\tilde{J}_{\cdot,j} = J_{\cdot,j} / \|J_{\cdot,j}\|$. This is what Ceres `JACOBI` scaling does. For SLAM: translation columns have norm $\propto 1/\sigma_t$ (measurement noise), rotation columns have norm $\propto r/\sigma_\theta$ where $r$ is the range. If $\sigma_t = 0.01$m and $\sigma_\theta = 0.001$rad at $r = 10$m, the norms differ by $\sim 100\times$. After column scaling, $\kappa(J^TJ)$ drops dramatically. Caveat: the scaled solution $\tilde{x}$ must be unscaled: $x_j = \tilde{x}_j / s_j$.
</details>

<details>
<summary>🎯 Challenge 2: Condition number of the normal equations</summary>

**Problem:** Show that $\kappa(J^T J) = \kappa(J)^2$. Why does this mean you should avoid forming the normal equations explicitly when the Jacobian is moderately ill-conditioned?

**Solution:** $J^T J$ has singular values $\sigma_i^2$ where $\sigma_i$ are the singular values of $J$. So $\kappa(J^TJ) = \sigma_{\max}^2/\sigma_{\min}^2 = \kappa(J)^2$. If $\kappa(J) = 10^4$ (fine), then $\kappa(J^TJ) = 10^8$ (losing 8 digits). Alternatives: (1) QR factorization of $J$ directly ($O(mn^2)$ but numerically stable), (2) use the augmented system $\begin{pmatrix}I & J \\ J^T & 0\end{pmatrix}\begin{pmatrix}r\\x\end{pmatrix}=\begin{pmatrix}b\\0\end{pmatrix}$ which has $\kappa = \kappa(J)$. In practice, for SLAM's sparse $J^TJ$ the sparsity advantage of normal equations usually outweighs the conditioning cost, especially with LM regularization $J^TJ + \lambda I$.
</details>

<details>
<summary>🎯 Challenge 3: Preconditioner design for BA</summary>

**Problem:** In bundle adjustment, the Schur complement $S$ is a dense camera-only matrix. Design a good preconditioner for $S$. Hint: consider the block structure where each camera contributes a $6 \times 6$ diagonal block.

**Solution:** **Block Jacobi preconditioner:** $M = \text{blockdiag}(S_{11}, S_{22}, \ldots, S_{kk})$ where $S_{ii}$ is the $6 \times 6$ block corresponding to camera $i$. This is cheap to compute ($k$ inversions of $6 \times 6$ matrices) and captures the dominant diagonal structure. Ceres calls this `CLUSTER_JACOBI`. For even better preconditioning, cluster nearby cameras and use the block for each cluster (Ceres `CLUSTER_TRIDIAGONAL`). The ideal preconditioner would be $S$ itself, but that defeats the purpose of using CG on $S$.
</details>

---

## Self-Assessment Checklist

- [ ] I can compute and interpret the condition number of a matrix
- [ ] I understand the geometric effect of ill-conditioning on optimization
- [ ] I can implement Jacobi preconditioning and measure its effect
- [ ] I know the sources of ill-conditioning in robotics problems

---

## Key Takeaways

1. **Condition number** $\kappa = \lambda_{\max}/\lambda_{\min}$ determines how hard a linear system (and the associated optimization) is to solve.
2. **Ill-conditioning** makes gradient-based methods slow and solutions inaccurate.
3. **Preconditioning** transforms the problem to reduce $\kappa$ — Jacobi (diagonal) is the simplest and often sufficient.
4. **Variable scaling** is the easiest fix — ensure all parameters have similar magnitude before optimizing.
