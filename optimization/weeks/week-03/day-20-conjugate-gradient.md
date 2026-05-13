# Day 20: Conjugate Gradient for Optimization

> **Phase II — Unconstrained Optimization | Week 3 | 2.5 hours**
> *O(n) per step, n-step convergence for quadratics — the middle ground.*

## Navigation
- **Previous:** [Day 19: Quasi-Newton BFGS](day-19-bfgs.md)
- **Next:** [Day 21: Week 3 Review & Optimizer Shootout](day-21-week-review.md)
- **Week:** [Week 3 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Ceres uses Conjugate Gradient internally via `ITERATIVE_SCHUR` and `CGNR` linear solvers. When solving $J^TJ\delta = -J^Tr$ at each Gauss-Newton step, CG avoids forming $J^TJ$ explicitly — just matrix-vector products. For sparse SLAM with 100K parameters, this is essential. CG for optimization (nonlinear CG) is also the core of Newton-CG.

---

## Theory (45 min)

### 20.1 Linear CG (Review from Day 13)

For $Ax = b$ with $A \succ 0$: generate conjugate directions $d_0, d_1, \ldots$ satisfying $d_i^T A d_j = 0$.

**Key property:** Linear CG converges in exactly $n$ steps for $n \times n$ systems.

### 20.2 Nonlinear CG

Extend CG to general $\min f(x)$:

$$x_{k+1} = x_k + \alpha_k d_k$$

$$d_{k+1} = -\nabla f_{k+1} + \beta_k d_k$$

The magic is in choosing $\beta_k$:

**Fletcher-Reeves:**
$$\beta_k^{FR} = \frac{\|\nabla f_{k+1}\|^2}{\|\nabla f_k\|^2}$$

**Polak-Ribière:**
$$\beta_k^{PR} = \frac{\nabla f_{k+1}^T (\nabla f_{k+1} - \nabla f_k)}{\|\nabla f_k\|^2}$$

**Polak-Ribière+ (recommended):**
$$\beta_k^{PR+} = \max(0, \beta_k^{PR})$$

This automatic restart (setting $\beta = 0$ when PR is negative) is crucial for robustness.

### 20.3 Why CG Works

For a quadratic $f = \frac{1}{2}x^TAx - b^Tx$:
- CG directions are A-conjugate: $d_i^T A d_j = 0$
- At step $k$, $x_k$ minimizes $f$ over $x_0 + \text{span}\{d_0, \ldots, d_{k-1}\}$
- **Converges in at most $n$ steps** (exact arithmetic)
- **Much faster** if eigenvalues are clustered: $O(\sqrt{\kappa})$ steps

For nonlinear functions: locally, $f \approx$ quadratic near the minimum, so CG inherits these properties.

### 20.4 Restart Strategies

Nonlinear CG can lose conjugacy. Remedies:
1. **Periodic restart:** Every $n$ steps, reset $d = -g$
2. **PR+ automatic restart:** $\beta^{PR+} = \max(0, \beta^{PR})$ — restarts when directions become poor
3. **Powell restart:** Restart when $|g_{k+1}^T g_k| > 0.2 \|g_{k+1}\|^2$ — detects loss of orthogonality

### 20.5 Convergence Rate Comparison

| Method | Per-step cost | Storage | Convergence |
|--------|--------------|---------|-------------|
| GD     | $O(n)$       | $O(n)$ | Linear, $O(\kappa)$ steps |
| CG     | $O(n)$       | $O(n)$ | $n$-step for quadratic, $O(\sqrt{\kappa})$ |
| L-BFGS | $O(mn)$      | $O(mn)$ | Superlinear |
| Newton | $O(n^3)$     | $O(n^2)$ | Quadratic |

CG has the same $O(n)$ cost as GD but much better convergence on well-conditioned problems.

---

## Implementation (60 min)

### Exercise 1: Nonlinear CG (Fletcher-Reeves and Polak-Ribière+)
```python
# See: code/week03/conjugate_gradient.py
```
Implement both CG variants with Wolfe line search.

### Exercise 2: Quadratic Convergence Verification
Test CG on a quadratic $f = \frac{1}{2}x^TAx - b^Tx$ with $n = 50$. Verify it converges in exactly $n$ steps (or fewer with clustered eigenvalues).

### Exercise 3: L-BFGS vs CG
Compare on large-scale Rosenbrock ($n = 100$). Plot iterations vs $\|g\|$.

### Exercise 4: Restart Strategy Comparison
Compare FR, PR, PR+, and periodic restart on a nonlinear function. Show PR+ is most robust.

---

## Practice (45 min)

1. For $f(x) = \frac{1}{2}x^TAx$, show CG directions satisfy $d_i^TAd_j = 0$.
2. Why does FR sometimes cycle (never converge)? Give intuition.
3. Why is PR+ better than pure PR? What does the $\max(0, \cdot)$ do geometrically?
4. If you have a preconditioner $M \approx H$, how does preconditioned CG change the $\beta$ formula?
5. Compare CG with 3 different $\alpha$ choices: exact, Armijo, Wolfe. Which gives the best conjugacy?
6. For a diagonal system with eigenvalues $\{1, 1, 1, \ldots, 1, 100\}$: how many CG steps?
7. CG needs $O(n)$ per step — same as GD. So why is CG better? (Hint: directions, not step size.)
8. When does CG beat L-BFGS? When does L-BFGS beat CG?

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Preconditioned nonlinear CG</summary>

**Problem:** For $f(x)$ with condition number $\kappa$, CG needs $O(\sqrt{\kappa})$ steps. Use a preconditioner $M \approx H^{-1}$: replace $g$ with $Mg$ in the CG update. Implement using diagonal Hessian approximation $M = \text{diag}(H)^{-1}$ and compare iteration count.

**Solution:** Preconditioned CG: $d_{k+1} = -Mg_{k+1} + \beta_k d_k$ with $\beta_k^{PR} = g_{k+1}^T(Mg_{k+1}-Mg_k) / (g_k^T Mg_k)$. The diagonal preconditioner captures scale differences. For Rosenbrock with scaling $[1, 100]$, this reduces iterations by ~3-5×. For SLAM, the block-diagonal of $J^TJ$ is a natural preconditioner (each pose block is easy to invert).
</details>

<details>
<summary>🎯 Challenge 2: Truncated Newton (Newton-CG)</summary>

**Problem:** Use CG to *inexactly* solve $H_k \delta = -g_k$ (the Newton system). Stop CG early when: (a) $\|r_j\| \leq \eta_k\|g_k\|$ (forcing sequence), (b) negative curvature detected ($d^THd < 0$), or (c) boundary of trust region reached. Implement and compare with full Newton.

**Solution:** The forcing sequence $\eta_k = \min(0.5, \sqrt{\|g_k\|})$ gives superlinear convergence (Eisenstat-Walker choice). Negative curvature direction is valuable: step to trust region boundary along it. This combines Newton's local rate with CG's $O(n)$ per-step cost. Used in large-scale optimization libraries (L-BFGS-B, IPOPT inner solver).
</details>

<details>
<summary>🎯 Challenge 3: CG on the normal equations</summary>

**Problem:** For least squares $\min \|Ax - b\|^2$, apply CG to the normal equations $A^TAx = A^Tb$ without forming $A^TA$. Implement CGNR (CG on Normal equations, Residual form) and compare with direct SVD solve. This is exactly what Ceres does with `CGNR` solver type.

**Solution:** CGNR: the matrix-vector product $A^TA v$ is computed as two operations: $w = Av$ then $A^Tw$. Never form $A^TA$. For sparse $A$ (Jacobian in SLAM), this preserves sparsity. Cost: 2 sparse mat-vecs per CG step. With preconditioning (Jacobi or block-diagonal), CGNR converges in $O(\sqrt{\kappa(A^TA)})$ steps. Ceres config: `linear_solver_type = CGNR; preconditioner_type = JACOBI`.
</details>

---

## Self-Assessment Checklist

- [ ] I can implement nonlinear CG with Fletcher-Reeves and Polak-Ribière+ updates
- [ ] I understand n-step convergence on quadratics and $O(\sqrt{\kappa})$ bound
- [ ] I know why PR+ is preferred over FR (automatic restart, no cycling)
- [ ] I can explain when CG is preferred over L-BFGS and vice versa

---

## Key Takeaways

1. **Nonlinear CG** extends linear CG to general optimization — $O(n)$ per step, $O(n)$ storage.
2. **PR+** ($\beta = \max(0, \beta^{PR})$) is the recommended variant — automatic restart, no cycling.
3. **n-step convergence** for quadratics; $O(\sqrt{\kappa})$ for well-conditioned problems.
4. **CG vs L-BFGS:** CG is simpler and cheaper per step; L-BFGS usually converges in fewer iterations but needs $O(mn)$ memory.
