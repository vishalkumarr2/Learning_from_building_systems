# Day 6: Optimality Conditions

> **Phase I — Mathematical Foundations | Week 1 | 2.5 hours**
> *Necessary conditions tell you "not here." Sufficient conditions tell you "definitely here."*

## Navigation
- **Previous:** [Day 5: Convexity](day-05-convexity.md)
- **Next:** [Day 7: Week 1 Review](day-07-week-review.md)
- **Week:** [Week 1 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
When Ceres or g2o reports "convergence," it means the optimality conditions are satisfied below a threshold. Understanding these conditions helps you diagnose: did the solver find a true minimum, or did it stop at a saddle point?

---

## Theory (45 min)

### 6.1 First-Order Necessary Condition (FONC)

If $x^*$ is a local minimizer of $f$ and $f$ is differentiable:

$$\boxed{\nabla f(x^*) = 0}$$

Any minimizer must be a **stationary point** (zero gradient). But not every stationary point is a minimizer — it could be a maximum or saddle.

### 6.2 Second-Order Necessary Condition (SONC)

If $x^*$ is a local minimizer:

$$\nabla f(x^*) = 0 \quad \text{AND} \quad H(x^*) \succeq 0 \quad \text{(PSD)}$$

The Hessian can't have negative eigenvalues at a minimum (you'd be able to decrease $f$ along the corresponding eigenvector).

### 6.3 Second-Order Sufficient Condition (SOSC)

If $\nabla f(x^*) = 0$ **and** $H(x^*) \succ 0$ (PD), then $x^*$ is a **strict local minimum**.

| Condition | $\nabla f = 0$ | Hessian | Conclusion |
|-----------|:-:|:-:|:-:|
| FONC (necessary) | ✓ | — | Must hold at any minimum |
| SONC (necessary) | ✓ | PSD | Must hold at any minimum |
| SOSC (sufficient) | ✓ | PD | **Guarantees** local minimum |

### 6.4 The Gap: Degenerate Cases

When $\nabla f = 0$ and $H$ is PSD (but not PD):
- $f(x) = x^4$ at $x = 0$: minimum despite $f''(0) = 0$
- $f(x) = x^3$ at $x = 0$: inflection point, NOT a minimum

Higher-order analysis is needed for degenerate cases.

### 6.5 Saddle Points

$x^*$ is a **saddle point** if $\nabla f(x^*) = 0$ and $H(x^*)$ is **indefinite** (has both positive and negative eigenvalues).

In high dimensions, saddle points vastly outnumber local minima. If $H$ has $n$ eigenvalues, there are $2^n$ sign combinations — most are saddle-type.

### 6.6 For Convex Functions

For convex $f$:
- FONC ($\nabla f = 0$) is both necessary AND sufficient for global optimality
- No need to check the Hessian
- This is the power of convexity

---

## Implementation (60 min)

### Exercise 1: Finding and Classifying Stationary Points
```python
# See: code/week01/optimality_conditions.py
```
Find all stationary points of a 2D function, classify each.

### Exercise 2: Saddle Point Detection
Identify and visualize saddle points — show descent directions (negative curvature eigenvectors).

### Exercise 3: Optimality Diagnostics
Given a "solution" from an optimizer, verify the optimality conditions and report:
- Gradient norm (should be ~0)
- Hessian eigenvalues (should be ≥0)
- Condition number (large = poorly conditioned)

### Exercise 4: Escape from Saddle Points
Demonstrate that adding noise to gradient descent helps escape saddle points.

---

## Practice (45 min)

For each function, find all stationary points and classify them:

1. $f(x) = x^3 - 3x$ → two stationary points, classify each
2. $f(x,y) = x^2 + y^2 - 2x - 4y + 5$ → complete the square
3. $f(x,y) = x^3 + y^3 - 3xy$ → three stationary points
4. $f(x,y) = \sin(x)\sin(y)$ on $[0, 2\pi]^2$ → multiple, many saddles
5. $f(x,y) = x^2 y^2$ → degenerate Hessian at origin
6. $f(x) = x^4 - 2x^2 + 1$ → minima and maximum
7. $f(x,y,z) = x^2 + y^2 - z^2$ → saddle at origin
8. $f(x) = \|Ax - b\|^2$ where $A$ is rank-deficient → infinitely many minimizers

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Convergence criteria in Ceres</summary>

**Problem:** Ceres Solver reports convergence based on three thresholds: (1) gradient tolerance, (2) relative function change, (3) relative parameter change. Map each to the optimality conditions. Which is the most theoretically justified?

**Solution:** (1) Gradient tolerance checks FONC: $\|\nabla f\| < \epsilon_g$. This is the most theoretically justified — directly checks stationarity. (2) Function change $|f_{k} - f_{k-1}|/|f_k| < \epsilon_f$ checks if progress has stalled, but could trigger at a saddle. (3) Parameter change $\|\Delta x\| < \epsilon_x$ also checks stalling. In practice, (1) is primary but all three are needed — (2) and (3) catch slow convergence that (1) might miss due to numerical noise.
</details>

<details>
<summary>🎯 Challenge 2: Saddle points in high dimensions</summary>

**Problem:** For a random quadratic $f(x) = \frac{1}{2}x^T A x$ where $A$ has entries drawn from $\mathcal{N}(0,1)$, what fraction of eigenvalues are positive? What does this imply about random non-convex problems?

**Solution:** For a GOE (Gaussian Orthogonal Ensemble) random symmetric matrix, eigenvalues follow the Wigner semicircle law centered at 0. About half are positive, half negative. So a random quadratic in $\mathbb{R}^n$ is a saddle point with probability approaching 1 as $n \to \infty$ — there are $2^n$ possible sign patterns but only 1 gives PD (minimum). In neural networks ($n \sim 10^6$), local minima are exponentially rare compared to saddle points. This is why SGD with momentum works — it naturally escapes saddle points.
</details>

<details>
<summary>🎯 Challenge 3: When SOSC fails but it's still a minimum</summary>

**Problem:** Show that $f(x,y) = x^4 + y^4$ has a minimum at the origin, but SOSC fails. What additional analysis proves it?

**Solution:** $\nabla f(0,0) = 0$ ✓. $H(0,0) = \text{diag}(0, 0)$ — PSD but not PD, so SOSC doesn't apply. However, for all $(x,y) \neq (0,0)$: $f(x,y) = x^4 + y^4 > 0 = f(0,0)$. Direct verification shows it's a global minimum. The issue is that SOSC is sufficient but not necessary — the function is "flatter than quadratic" at the origin. Fourth-order analysis: the 4th-order terms $x^4 + y^4$ are positive definite, confirming the minimum.
</details>

---

## Self-Assessment Checklist

- [ ] I can state FONC, SONC, and SOSC precisely
- [ ] I can find all stationary points and classify them (minimum, maximum, saddle)
- [ ] I understand the gap between necessary and sufficient conditions
- [ ] I can interpret solver convergence diagnostics in terms of optimality conditions

---

## Key Takeaways

1. **FONC** ($\nabla f = 0$): necessary at any minimum, but saddle points also satisfy it.
2. **SOSC** ($\nabla f = 0$, $H \succ 0$): guarantees a local minimum. The gold standard.
3. **For convex functions**, FONC alone is sufficient — the simplicity that makes optimization tractable.
4. **In practice**, check gradient norm + Hessian eigenvalues to verify your solution.
