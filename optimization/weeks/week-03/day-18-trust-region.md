# Day 18: Trust Region Methods

> **Phase II — Unconstrained Optimization | Week 3 | 2.5 hours**
> *Restrict the step to where your model is trustworthy.*

## Navigation
- **Previous:** [Day 17: Newton's Method](day-17-newtons-method.md)
- **Next:** [Day 19: Quasi-Newton BFGS](day-19-bfgs.md)
- **Week:** [Week 3 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Ceres' default strategy is `LEVENBERG_MARQUARDT` — which is a trust region method applied to least squares. When you see `trust_region_strategy_type` in a Ceres solver config, that's this. Trust regions handle indefinite Hessians, saddle points, and far-from-solution starts — all common in SLAM. The `DOGLEG` option in Ceres is exactly what you'll implement today.

---

## Theory (45 min)

### 18.1 The Trust Region Idea

Instead of choosing a *direction* then a *step size* (line search), trust region methods:
1. Build a quadratic model $m_k(\delta)$
2. Restrict $\|\delta\| \leq \Delta_k$ (trust region radius)
3. Solve the constrained subproblem
4. Update $\Delta_k$ based on model quality

### 18.2 The Subproblem

$$\min_\delta \; m_k(\delta) = f_k + g_k^T \delta + \frac{1}{2}\delta^T H_k \delta \quad \text{s.t.} \quad \|\delta\| \leq \Delta_k$$

### 18.3 Cauchy Point

The simplest solution: steepest descent restricted to the trust region.

$$\delta_C = -\tau \frac{\Delta}{\|g\|} g$$

where $\tau = \min\left(1, \frac{\|g\|^3}{\Delta \, g^T H g}\right)$ if $g^THg > 0$, else $\tau = 1$.

**Guarantee:** Using the Cauchy point gives global convergence (but slow — linear rate like GD).

### 18.4 Dogleg Method

Interpolate between the Cauchy point and the full Newton step:

1. **Newton step** $\delta_N = -H^{-1}g$ (if $H \succ 0$)
2. **Cauchy step** $\delta_C$ (steepest descent clipped to $\Delta$)
3. If $\|\delta_N\| \leq \Delta$: take Newton step
4. If $\|\delta_C\| \geq \Delta$: take scaled gradient step on boundary
5. Otherwise: interpolate along the dogleg path (Cauchy → Newton) to hit the boundary

### 18.5 Radius Update

Compute the actual-to-predicted reduction ratio:

$$\rho_k = \frac{f(x_k) - f(x_k + \delta_k)}{m_k(0) - m_k(\delta_k)}$$

| $\rho_k$ | Action |
|----------|--------|
| $< 0.25$ | Shrink: $\Delta_{k+1} = 0.25 \Delta_k$ |
| $\in [0.25, 0.75)$ | Keep: $\Delta_{k+1} = \Delta_k$ |
| $\geq 0.75$ and $\|\delta\| = \Delta$ | Expand: $\Delta_{k+1} = 2\Delta_k$ |

If $\rho_k < 0$ (model was *worse*): reject step, shrink radius, try again.

### 18.6 Trust Region vs Line Search

| Aspect | Line Search | Trust Region |
|--------|-------------|--------------|
| Strategy | Pick direction, then step size | Pick step within ball |
| Handles indefinite $H$? | Needs modification | Naturally |
| Robustness | Can fail to find step | Always produces a step |
| Default in Ceres | `LINE_SEARCH` | `TRUST_REGION` (default) |

---

## Implementation (60 min)

### Exercise 1: Cauchy Point Trust Region
```python
# See: code/week03/trust_region.py
```
Implement the simplest trust region using only the Cauchy point.

### Exercise 2: Dogleg Trust Region
Implement the full dogleg method with Newton/Cauchy interpolation.

### Exercise 3: Comparison
Compare trust region vs line search Newton on Rosenbrock from various starting points.

### Exercise 4: Saddle Point
Create a function with a saddle point (e.g., $f = x^2 - y^2$). Show trust region navigates away while pure Newton may converge to the saddle.

### Exercise 5: Visualization
On a contour plot: show the trust region circle, Cauchy point, Newton point, dogleg path, and how the radius adapts over iterations.

---

## Practice (45 min)

1. **Trust region vs line search:** When to prefer which? Create a comparison table.
2. **Connection to Levenberg-Marquardt:** Show that $(H + \lambda I)\delta = -g$ with $\lambda$ chosen to satisfy $\|\delta\| = \Delta$ is equivalent to the trust region subproblem.
3. What happens if the model is perfect ($\rho = 1$ always)? The trust region expands to infinity → becomes Newton.
4. What's the minimum number of function evaluations per trust region iteration?
5. For a diagonal Hessian, derive the trust region solution analytically.
6. Why does the Cauchy point guarantee global convergence? (Sufficient decrease ∝ $\|g\|^2$.)
7. Derive the dogleg path parametrically: $p(\tau) = (1-\tau)\delta_C + \tau\delta_N$ for $\tau \in [0,1]$... but actually it's piecewise! Explain.
8. In Ceres, `DOGLEG` vs `LEVENBERG_MARQUARDT`: when to switch? (Hint: DOGLEG needs $H \succ 0$.)

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Steihaug-Toint CG for the trust region subproblem</summary>

**Problem:** For large-scale problems where forming $H$ is prohibitive, solve the trust region subproblem approximately using CG, stopping when: (a) CG converges, (b) the iterate leaves the trust region, or (c) negative curvature is detected. Implement this "truncated CG" approach.

**Solution:** Run CG on $H\delta = -g$. At each CG step $k$: (1) check $\|\delta_k\| \leq \Delta$ — if boundary hit, interpolate to boundary; (2) check $d_k^T H d_k > 0$ — if negative curvature, step to boundary along $d_k$. This gives a point at least as good as Cauchy (so global convergence holds) but approaches Newton quality when CG succeeds. Cost: $O(n)$ per CG step × few CG steps. Ceres uses this for `ITERATIVE_SCHUR`.
</details>

<details>
<summary>🎯 Challenge 2: Exact trust region solution via secular equation</summary>

**Problem:** The exact solution of the trust region subproblem is $(H + \lambda^*I)\delta = -g$ with $\lambda^* \geq 0$ satisfying complementarity: $\lambda^*(\|\delta\| - \Delta) = 0$. Find $\lambda^*$ by solving the scalar secular equation $\|\delta(\lambda)\| = \Delta$. Implement using Newton's method on the secular equation.

**Solution:** Define $\phi(\lambda) = \frac{1}{\|\delta(\lambda)\|} - \frac{1}{\Delta}$ where $\delta(\lambda) = -(H+\lambda I)^{-1}g$. Use eigendecomposition $H = Q\Lambda Q^T$: then $\delta(\lambda) = -\sum_i \frac{q_i^T g}{\lambda_i + \lambda} q_i$ and $\|\delta\|^2 = \sum_i \frac{(q_i^Tg)^2}{(\lambda_i+\lambda)^2}$. Newton on $\phi(\lambda) = 0$ converges quickly. Handle the hard case: $\lambda^* = -\lambda_{\min}$ and $q_{\min}^T g = 0$.
</details>

<details>
<summary>🎯 Challenge 3: Two-dimensional subspace trust region</summary>

**Problem:** Instead of the full trust region subproblem, restrict the search to the 2D subspace spanned by $g$ and $H^{-1}g$. This is cheaper than the full solve and captures both steepest descent and Newton directions. Implement and compare quality vs dogleg.

**Solution:** Let $S = [g, H^{-1}g]$ (orthogonalized). Project: $\hat{H} = S^THS$ (2×2), $\hat{g} = S^Tg$ (2×1). Solve the 2D trust region subproblem $(|\hat{\delta}\| \leq \Delta)$ exactly (secular equation in 2D is cheap). Map back: $\delta = S\hat{\delta}$. This captures more curvature than dogleg (which only interpolates linearly) at cost $O(n)$ for forming $S^THS$. This is what Ceres uses in the `SUBSPACE_DOGLEG` variant.
</details>

---

## Self-Assessment Checklist

- [ ] I can implement trust region with Cauchy point and dogleg methods
- [ ] I understand the ratio $\rho$ and how it drives radius adaptation
- [ ] I can explain why trust regions handle indefinite Hessians naturally
- [ ] I know the connection between trust region and Levenberg-Marquardt

---

## Key Takeaways

1. **Trust region** = constrained step within $\|\delta\| \leq \Delta$ — safer than line search.
2. **Cauchy point** = GD in a ball — simple, globally convergent, slow.
3. **Dogleg** = interpolate Cauchy↔Newton — fast near solution, robust far away.
4. **Radius update** via $\rho = \text{actual} / \text{predicted}$ — expand when model is good, shrink when bad.
