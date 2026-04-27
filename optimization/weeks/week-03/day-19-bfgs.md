# Day 19: Quasi-Newton Methods — BFGS and L-BFGS

> **Phase II — Unconstrained Optimization | Week 3 | 2.5 hours**
> *Superlinear convergence without computing the Hessian.*

## Navigation
- **Previous:** [Day 18: Trust Region Methods](day-18-trust-region.md)
- **Next:** [Day 20: Conjugate Gradient for Optimization](day-20-conjugate-gradient.md)
- **Week:** [Week 3 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Ceres' `LINE_SEARCH` solver uses L-BFGS by default (`line_search_direction_type = LBFGS`). For problems where the Jacobian sparsity makes trust region expensive but gradients are cheap (e.g., trajectory smoothing), L-BFGS is the go-to. The key insight: you approximate the Hessian from gradient differences alone — $O(n)$ memory, no Hessian required.

---

## Theory (45 min)

### 19.1 The Secant Condition

After step $k$, we have:
- $s_k = x_{k+1} - x_k$ (step)
- $y_k = \nabla f_{k+1} - \nabla f_k$ (gradient change)

We want $B_{k+1}$ to satisfy the secant equation:

$$B_{k+1} s_k = y_k$$

This says: the new Hessian approximation should be consistent with the most recent curvature information.

### 19.2 The BFGS Update

The rank-2 BFGS update:

$$B_{k+1} = B_k - \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k} + \frac{y_k y_k^T}{y_k^T s_k}$$

Or equivalently, the *inverse* Hessian update (more useful — avoids solving a linear system):

$$H_{k+1} = \left(I - \frac{s_k y_k^T}{y_k^T s_k}\right) H_k \left(I - \frac{y_k s_k^T}{y_k^T s_k}\right) + \frac{s_k s_k^T}{y_k^T s_k}$$

**Properties:**
- If $H_k \succ 0$ and $y_k^T s_k > 0$ (guaranteed by Wolfe conditions), then $H_{k+1} \succ 0$
- Convergence rate: **superlinear** (faster than linear, slower than quadratic)
- Cost: $O(n^2)$ per iteration (matrix-vector products)
- Storage: $O(n^2)$ (full dense matrix)

### 19.3 L-BFGS: Limited-Memory BFGS

Store only the last $m$ pairs $(s_k, y_k)$ instead of the full $n \times n$ matrix.

**Two-loop recursion** (Nocedal):
```
q = ∇f_k
for i = k-1, k-2, ..., k-m:
    αᵢ = ρᵢ sᵢᵀ q
    q = q - αᵢ yᵢ
r = H⁰_k q                    # H⁰ = γI, γ = (sₖ₋₁ᵀyₖ₋₁)/(yₖ₋₁ᵀyₖ₋₁)
for i = k-m, k-m+1, ..., k-1:
    β = ρᵢ yᵢᵀ r
    r = r + sᵢ (αᵢ - β)
return r = H_k ∇f_k            # This is the search direction
```

**Properties:**
- Storage: $O(mn)$ — typically $m = 3\text{-}20$
- Cost per step: $O(mn)$ — just $m$ dot products
- Works for $n = 10^6$ easily
- **Default in Ceres** `line_search_direction_type = LBFGS` with $m = 10$

### 19.4 Convergence Rates

| Method | Convergence | Per-step cost | Storage |
|--------|-------------|---------------|---------|
| GD     | Linear      | $O(n)$        | $O(n)$  |
| Newton | Quadratic   | $O(n^3)$      | $O(n^2)$ |
| BFGS   | Superlinear | $O(n^2)$      | $O(n^2)$ |
| L-BFGS | Superlinear | $O(mn)$       | $O(mn)$ |

### 19.5 Practical Considerations

- **Wolfe conditions required:** Without $c_2$ (curvature), $y^Ts$ may be negative → PD lost
- **Initial $H_0$:** Usually $H_0 = \gamma I$ where $\gamma = s^Ty / (y^Ty)$ (scaled identity)
- **Restart:** If direction becomes poor (e.g., $g^Td > 0$), restart with $H_0 = I$
- **Powell damping:** Modify $y_k$ to ensure $y_k^Ts_k > \varepsilon s_k^TB_ks_k$ — used in Ceres

---

## Implementation (60 min)

### Exercise 1: Full BFGS with Wolfe Line Search
```python
# See: code/week03/bfgs.py
```
Implement BFGS with the inverse Hessian update and strong Wolfe line search.

### Exercise 2: L-BFGS Two-Loop Recursion
Implement L-BFGS with $m = 10$. Compare memory: BFGS $O(n^2)$ vs L-BFGS $O(mn)$.

### Exercise 3: Comparison Shootout
Test GD, Newton, BFGS, L-BFGS on Rosenbrock — iterations, function evals, time, final error.

### Exercise 4: High-Dimensional Test
Test L-BFGS on $n = 1000$ extended Rosenbrock. BFGS should run out of memory (or be slow); L-BFGS should work.

---

## Practice (45 min)

1. **Derive** the BFGS update from the optimization problem: $\min_{B} \|B - B_k\|_W$ subject to $Bs = y$ and $B = B^T$. (The norm $\|\cdot\|_W$ is weighted Frobenius.)
2. Why do Wolfe conditions guarantee $y^Ts > 0$? (Hint: curvature condition $\nabla f_{k+1}^Td_k \geq c_2 \nabla f_k^T d_k$.)
3. What happens to BFGS if you skip the curvature condition (Armijo only)?
4. How does L-BFGS approximate the full BFGS matrix? What is lost?
5. For $m=1$ in L-BFGS, what method do you recover? (Hint: a specific secant method.)
6. Explain Powell damping: when $y^Ts$ is too small, modify $y$ to $\theta y + (1-\theta)Bs$.
7. Why is the initial scaling $\gamma = s^Ty/(y^Ty)$ important? What if you use $\gamma = 1$ always?
8. Compare memory for $n = 100{,}000$: BFGS needs $n^2 = 10^{10}$ doubles = 80 GB vs L-BFGS ($m=10$): $2mn = 2M$ doubles = 16 MB.

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: SR1 update — when BFGS isn't enough</summary>

**Problem:** The symmetric rank-1 (SR1) update: $B_{k+1} = B_k + \frac{(y-Bs)(y-Bs)^T}{(y-Bs)^Ts}$ can approximate indefinite Hessians (BFGS cannot). Implement SR1 with trust region (SR1 + TR is a powerful combination). Show it converges on a function where the Hessian is indefinite near the solution.

**Solution:** SR1 can model negative curvature — useful for finding saddle points or for problems with indefinite Hessians. Skip the update when $|(y-Bs)^Ts| < \varepsilon\|s\|\|y-Bs\|$ (denominator too small). SR1+TR is used in Ceres' `SUBSPACE_DOGLEG` variant and in advanced SLAM solvers.
</details>

<details>
<summary>🎯 Challenge 2: BFGS on Riemannian manifolds</summary>

**Problem:** For optimization on $SO(3)$ or $SE(3)$, the standard BFGS update doesn't work because $s_k = x_{k+1} - x_k$ is meaningless on a manifold. Implement Riemannian BFGS using logarithmic map for $s_k$ and parallel transport for $y_k$.

**Solution:** $s_k = \text{Log}_{x_k}(x_{k+1})$ (tangent vector), $y_k = \mathcal{T}_{x_{k+1} \to x_k}(\text{grad} f_{k+1}) - \text{grad} f_k$ where $\mathcal{T}$ is parallel transport. The BFGS update stays the same but operates in the tangent space. This is critical for Lie group SLAM (Day 41). Libraries: `pymanopt`, `geoopt` (PyTorch).
</details>

<details>
<summary>🎯 Challenge 3: Compact representation and preconditioning</summary>

**Problem:** The L-BFGS matrix can be written in compact form: $H_k = \delta_k I + V_k M_k V_k^T$ where $V_k$ and $M_k$ are $n \times 2m$ and $2m \times 2m$ matrices. Use this to apply L-BFGS as a preconditioner for CG in a Newton-CG method. Implement and compare with unpreconditioned Newton-CG.

**Solution:** The compact form enables $H^{-1}v$ products in $O(mn)$ without the two-loop recursion — useful as a preconditioner. For Newton-CG: each CG step needs $Hv$ (Hessian-vector product via AD) and we precondition with L-BFGS. This combines the best of both: Newton's quadratic convergence with L-BFGS's cheap preconditioning. Convergence: CG needs ~5-10 steps instead of ~100.
</details>

---

## Self-Assessment Checklist

- [ ] I can implement BFGS with the inverse Hessian update
- [ ] I can implement L-BFGS two-loop recursion from scratch
- [ ] I understand why Wolfe conditions are needed for BFGS (PD maintenance)
- [ ] I know when to choose L-BFGS over BFGS over Newton

---

## Key Takeaways

1. **BFGS** approximates the Hessian from gradient differences — no second derivatives needed.
2. **L-BFGS** uses only the last $m$ pairs — $O(mn)$ memory, works at scale.
3. **Superlinear convergence** — faster than GD, cheaper than Newton. The best trade-off.
4. **Wolfe conditions** are essential — they guarantee $y^Ts > 0$ which keeps $H \succ 0$.
