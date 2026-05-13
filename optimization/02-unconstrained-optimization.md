# 02 — Unconstrained Optimization

> **Goal:** Understand the core algorithms that find minima of smooth functions without constraints — the building blocks for everything in chapters 03–07.

---

## 1. Problem Statement

$$\min_{x \in \mathbb{R}^n} f(x)$$

No constraints on $x$. The function $f$ is assumed smooth (at least twice differentiable).

**Strategy:** Generate a sequence $x_0, x_1, x_2, \ldots$ that converges to a local minimizer $x^*$.

Every method follows this template:
```
x_{k+1} = x_k + α_k · d_k
```
where $d_k$ is the **search direction** and $\alpha_k$ is the **step size** (learning rate).

---

## 2. Gradient Descent

### The Algorithm

The simplest choice: move in the steepest descent direction.

$$x_{k+1} = x_k - \alpha_k \nabla f(x_k)$$

```python
def gradient_descent(f, grad_f, x0, alpha=0.01, tol=1e-6, max_iter=10000):
    x = x0.copy()
    for k in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        x = x - alpha * g
    return x, k
```

### Step Size Selection

| Strategy | Formula | Pros | Cons |
|----------|---------|------|------|
| Fixed | $\alpha_k = \alpha$ (constant) | Simple | Must tune; too large → diverge, too small → crawl |
| Exact line search | $\alpha_k = \arg\min_\alpha f(x_k - \alpha g_k)$ | Optimal per step | Expensive (1D optimization each step) |
| Backtracking (Armijo) | Start large, shrink until sufficient decrease | Practical, cheap | Needs two parameters ($c$, $\rho$) |
| Barzilai-Borwein | $\alpha_k = \frac{s_{k-1}^T y_{k-1}}{y_{k-1}^T y_{k-1}}$ | Fast, adaptive | Non-monotone (can increase $f$) |

### Backtracking Line Search (Armijo Condition)

Accept step $\alpha_k$ if:

$$f(x_k - \alpha_k g_k) \leq f(x_k) - c \cdot \alpha_k \|g_k\|^2$$

where $c \in (0, 1)$ (typically $c = 10^{-4}$). Start with $\alpha = 1$ and shrink by factor $\rho \in (0,1)$ until satisfied.

```python
def backtracking_line_search(f, x, g, alpha=1.0, c=1e-4, rho=0.5):
    fx = f(x)
    while f(x - alpha * g) > fx - c * alpha * np.dot(g, g):
        alpha *= rho
    return alpha
```

### Convergence Rate

For a strongly convex function with condition number $\kappa = \lambda_{\max}(H) / \lambda_{\min}(H)$:

$$f(x_{k+1}) - f(x^*) \leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^2 (f(x_k) - f(x^*))$$

- **Linear convergence** — each step reduces error by a constant factor
- **Bad conditioning ($\kappa \gg 1$) → slow convergence** — the zigzag pattern along narrow valleys
- For $\kappa = 100$: factor $\approx 0.96$ per step → need ~250 steps for 4 digits of accuracy

**OKS relevance:** This is why pure gradient descent is never used in the navigation estimator. $\kappa$ for the sensorbar problem can be $>10^4$.

---

## 3. Newton's Method

### The Algorithm

Use the second-order Taylor model to compute the step:

$$\Delta x = -[\nabla^2 f(x_k)]^{-1} \nabla f(x_k) = -H_k^{-1} g_k$$

```python
def newtons_method(f, grad_f, hess_f, x0, tol=1e-8, max_iter=100):
    x = x0.copy()
    for k in range(max_iter):
        g = grad_f(x)
        H = hess_f(x)
        if np.linalg.norm(g) < tol:
            break
        dx = np.linalg.solve(H, -g)  # Never invert H explicitly!
        x = x + dx
    return x, k
```

### Why Newton is Fast

| Property | Gradient Descent | Newton's Method |
|----------|-----------------|-----------------|
| Step direction | $-g$ (steepest descent) | $-H^{-1}g$ (accounts for curvature) |
| Convergence | Linear: $O(\kappa \log(1/\epsilon))$ | Quadratic: $O(\log\log(1/\epsilon))$ |
| Per-step cost | $O(n)$ | $O(n^3)$ (Hessian solve) |
| Condition-dependent? | Yes (severely) | No (self-correcting) |

**Quadratic convergence** means the number of correct digits **doubles** each step. Once you're close, convergence is explosive:
- Step $k$: error = $10^{-2}$
- Step $k+1$: error = $10^{-4}$
- Step $k+2$: error = $10^{-8}$

### When Newton Fails

1. **Far from minimum:** The quadratic model is poor → Newton step may go uphill
2. **Hessian not PD:** At a saddle point, $H$ has negative eigenvalues → step goes wrong way
3. **Hessian computation:** Need exact second derivatives — expensive or unavailable

**Solutions:** Line search along Newton direction, or trust region (see §5).

---

## 4. Quasi-Newton Methods (BFGS, L-BFGS)

### Idea

Approximate the Hessian $H_k$ (or its inverse $B_k = H_k^{-1}$) using only gradient information. No second derivatives needed.

### BFGS Update

Given step $s_k = x_{k+1} - x_k$ and gradient change $y_k = g_{k+1} - g_k$:

$$B_{k+1} = \left(I - \frac{s_k y_k^T}{y_k^T s_k}\right) B_k \left(I - \frac{y_k s_k^T}{y_k^T s_k}\right) + \frac{s_k s_k^T}{y_k^T s_k}$$

- Maintains positive definiteness automatically
- Converges **superlinearly** (faster than linear, slower than quadratic)
- Per-step cost: $O(n^2)$ — better than $O(n^3)$ for Newton

### L-BFGS (Limited-Memory BFGS)

For large $n$: don't store the full $n \times n$ matrix $B_k$. Instead, store the last $m$ pairs $(s_k, y_k)$ (typically $m = 5$–$20$) and compute $B_k g_k$ on the fly.

- Per-step cost: $O(mn)$ — linear in $n$!
- The default optimizer in many ML frameworks and scipy's `minimize(method='L-BFGS-B')`

**OKS relevance:** When Ceres uses `DENSE_QR` or `DENSE_NORMAL_CHOLESKY` for small problems, it's effectively doing Newton. For large problems, it uses iterative solvers that share ideas with L-BFGS.

---

## 5. Trust Region Methods

### Idea

Instead of choosing a direction then a step size (line search), choose both simultaneously by solving:

$$\min_{\Delta x} \quad m_k(\Delta x) = f_k + g_k^T \Delta x + \frac{1}{2} \Delta x^T B_k \Delta x$$
$$\text{subject to} \quad \|\Delta x\| \leq \Delta_k$$

where $\Delta_k$ is the **trust region radius**.

### Algorithm

```
1. Solve the trust-region subproblem for Δx
2. Compute actual vs predicted reduction:
   ρ_k = (f(x_k) - f(x_k + Δx)) / (m_k(0) - m_k(Δx))
3. If ρ_k > threshold: accept step, possibly expand trust region
   If ρ_k ≈ 0 or < 0: reject step, shrink trust region
4. Update x_k+1 and Δ_{k+1}
```

### Trust Region vs Line Search

| Aspect | Line Search | Trust Region |
|--------|------------|--------------|
| Step direction | Fixed (Newton, GD) | Adapted to region |
| Step length | Varies along direction | Bounded by radius |
| Far from minimum | Can take bad steps | Naturally conservative |
| Hessian not PD | Needs modification | Handled by constraint |

### Levenberg-Marquardt as Trust Region

The LM algorithm (Chapter 03) can be interpreted as a trust-region method where:

$$\Delta x = -(J^T J + \lambda I)^{-1} J^T r$$

- $\lambda$ large → small trust region → gradient-descent-like (safe but slow)
- $\lambda$ small → large trust region → Gauss-Newton-like (fast but risky)

This is exactly what Ceres does. The `TrustRegionStrategy` in Ceres toggles between `LEVENBERG_MARQUARDT` and `DOGLEG`.

---

## 6. Conjugate Gradient Method

### Idea

Improve upon gradient descent by choosing directions that are **conjugate** with respect to $A$:

$$d_i^T A d_j = 0 \quad \text{for } i \neq j$$

For a quadratic $f(x) = \frac{1}{2}x^TAx - b^Tx$, CG solves $Ax = b$ in exactly $n$ steps (in exact arithmetic).

### Algorithm (for $Ax = b$)

```python
def conjugate_gradient(A, b, x0, tol=1e-8, max_iter=None):
    x = x0.copy()
    r = b - A @ x           # residual
    d = r.copy()             # search direction
    for k in range(max_iter or len(b)):
        Ad = A @ d
        alpha = r.dot(r) / d.dot(Ad)
        x = x + alpha * d
        r_new = r - alpha * Ad
        if np.linalg.norm(r_new) < tol:
            break
        beta = r_new.dot(r_new) / r.dot(r)
        d = r_new + beta * d
        r = r_new
    return x
```

### Why CG Matters for Large-Scale Problems

- Never forms or stores $A$ explicitly — only needs matrix-vector products $Av$
- Converges in $\leq n$ steps for $n \times n$ problem
- For clustered eigenvalues: converges much faster
- Used as the **inner solver** in truncated Newton methods and Ceres' `ITERATIVE_SCHUR`

**OKS relevance:** Ceres uses CG-based solvers (`CGNR`, `ITERATIVE_SCHUR`) for large SLAM-like problems where the Hessian is too large to factorize directly.

---

## 7. Comparison Summary

| Method | Direction | Step | Convergence | Per-step cost | When to use |
|--------|-----------|------|-------------|---------------|-------------|
| Gradient descent | $-g$ | Line search | Linear | $O(n)$ | ML training, huge $n$ |
| Newton | $-H^{-1}g$ | Line search / TR | Quadratic | $O(n^3)$ | Small $n$, fast convergence needed |
| BFGS | $-B_k g$ | Line search | Superlinear | $O(n^2)$ | Medium $n$, no Hessian available |
| L-BFGS | $-B_k g$ (implicit) | Line search | Superlinear | $O(mn)$ | Large $n$, limited memory |
| Trust region (Newton) | TR subproblem | Radius | Quadratic | $O(n^3)$ | Robust, general purpose |
| Conjugate gradient | Conjugate dirs | Exact (quadratic) | $\leq n$ steps | $O(n)$ per step | Huge sparse systems |
| LM (next chapter) | $(J^TJ+\lambda I)^{-1}J^Tr$ | Trust region | Superlinear | $O(mn^2)$ | **Least-squares (Ceres!)** |

---

## 8. Practical Decision Tree

```
Is f(x) a sum of squares: Σ r_i(x)²?
├── YES → Use Gauss-Newton / LM (Chapter 03)
│         (exploits structure, much faster)
│
└── NO → Is n large (> 10,000)?
    ├── YES → L-BFGS or CG
    │         (can't store Hessian)
    │
    └── NO → Do you have the Hessian?
        ├── YES → Newton + trust region
        │         (quadratic convergence)
        │
        └── NO → BFGS
                  (builds Hessian approximation from gradients)
```

---

## Summary: What to Carry Forward

| Concept | Where it's used next |
|---------|---------------------|
| Line search (Armijo) | 03 (step acceptance in LM) |
| Newton step $-H^{-1}g$ | 03 (Gauss-Newton as approximate Newton) |
| Trust region | 03 (LM is a trust-region method), 06 (pose-graph) |
| Condition number → convergence | 07 (numerical methods) |
| CG solver | 06 (large pose-graph), 07 (sparse systems) |
| Quadratic model | 03 (the key approximation in Gauss-Newton) |
