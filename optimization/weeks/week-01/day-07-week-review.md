# Day 7: Week 1 Review and Comprehensive Exercises

> **Phase I — Mathematical Foundations | Week 1 | 2.5 hours**
> *If you can do these 10 problems, you own the math. If you can't, re-read the day that's weak.*

## Navigation
- **Previous:** [Day 6: Optimality Conditions](day-06-optimality-conditions.md)
- **Next:** [Day 8: Matrix Decompositions (Week 2)](../week-02/day-08-matrix-decompositions.md)
- **Week:** [Week 1 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
This review consolidates the exact mathematical toolkit used in every OKS optimization pipeline: the Jacobian linearizations in SLAM, the Hessian approximations in Gauss-Newton, the convexity analysis that tells you when initialization matters, and the optimality checks that tell you if the solver succeeded.

---

## Week 1 Summary

| Day | Topic | Key Concept |
|-----|-------|-------------|
| 1 | Vectors & Matrices | Four subspaces, projection, rank |
| 2 | Eigenvalues & SVD | PD classification, condition number |
| 3 | Gradients & Jacobians | Chain rule, numerical verification |
| 4 | Hessians & Taylor | Quadratic model, Newton step |
| 5 | Convexity | Local = global, strong convexity |
| 6 | Optimality Conditions | FONC/SONC/SOSC, saddle points |

---

## Comprehensive Exercise Set (60 min)

Complete all 10 problems. The companion code provides scaffolding and verification.

```python
# See: code/week01/exercise_set_1.py
```

### Problem 1: SVD and Rank
Given $A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 1 & 1 & 1 \end{pmatrix}$:
(a) Compute the rank.
(b) Find the null space.
(c) Find the best rank-1 approximation via SVD.

### Problem 2: Positive Definiteness
Prove that $A^T A$ is always PSD. When is it PD?

### Problem 3: Gradient of Least Squares
Derive $\nabla_x \|Ax - b\|^2 = 2A^T(Ax - b)$ from scratch (no lookup).

### Problem 4: Hessian Classification Pipeline
For $f(x,y) = x^3 - 3xy^2$:
(a) Find all stationary points.
(b) Compute the Hessian at each.
(c) Classify each (min/max/saddle).

### Problem 5: Taylor Approximation
Compute the second-order Taylor expansion of $f(x,y) = e^{x+y}$ around $(0,0)$. Compare to the actual function at $(0.1, 0.1)$ and $(1, 1)$.

### Problem 6: Convexity Proof
Show that $f(x) = \max(x_1, x_2, \ldots, x_n)$ is convex using the definition.

### Problem 7: Newton Step
For $f(x,y) = x^2 + 10y^2$:
(a) Compute the Newton step from $(5, 5)$.
(b) How many Newton steps are needed? (It's a quadratic — think about it.)
(c) Compare to gradient descent with $\alpha = 1/(2L)$.

### Problem 8: Condition Number Impact
Generate random matrices $A \in \mathbb{R}^{100 \times 100}$ with condition numbers $\kappa = 10, 100, 1000, 10000$. Solve $Ax = b$ with small perturbation $\delta b$. Plot $\|\delta x\|/\|x\|$ vs $\kappa$.

### Problem 9: Saddle vs Minimum Detection
Write a function that takes any twice-differentiable $f: \mathbb{R}^n \to \mathbb{R}$ and a point $x$, and classifies it as: (a) not stationary, (b) strict local min, (c) strict local max, (d) saddle, (e) degenerate (inconclusive).

### Problem 10: End-to-End Optimization
Minimize the Rosenbrock function using only the tools from this week:
(a) Start from $(-1.2, 1.0)$.
(b) Apply Newton's method (Hessian + gradient).
(c) Track convergence: $\|\nabla f\|_k$ vs iteration.
(d) Verify SOSC at the solution.

---

## Self-Assessment Checklist

- [ ] I can compute SVD, rank, null space, and best low-rank approximation
- [ ] I can classify stationary points using gradient + Hessian (FONC/SOSC)
- [ ] I understand convexity and why local = global for convex functions
- [ ] I can derive and verify gradients/Jacobians/Hessians analytically and numerically
- [ ] I can perform a Newton step on a quadratic and explain why it converges in 1 step
- [ ] I can explain how condition number affects solver accuracy

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Degenerate Hessian analysis</summary>

**Problem:** For $f(x,y) = (x^2 - y)^2$, find all stationary points, compute the Hessian at each, and classify. What makes this problem tricky for standard SOSC?

**Solution:** $\nabla f = (4x(x^2-y), -2(x^2-y))$. Setting to zero: $x^2 = y$, giving the curve $(t, t^2)$ for all $t$. The Hessian along this curve is $H = \begin{pmatrix} 4(3t^2-t^2) & -4t \\ -4t & 2 \end{pmatrix} = \begin{pmatrix} 8t^2 & -4t \\ -4t & 2 \end{pmatrix}$. At $t=0$: $H = \begin{pmatrix} 0 & 0 \\ 0 & 2 \end{pmatrix}$, which is PSD but singular — SOSC fails. The entire curve is a valley of global minima ($f=0$), but SOSC can only confirm isolated strict minima.
</details>

<details>
<summary>🎯 Challenge 2: SVD-based pseudoinverse solver</summary>

**Problem:** Implement a least-squares solver using SVD that handles rank-deficient systems gracefully. For $Ax = b$ where $A$ may be rank-deficient, return the minimum-norm solution. Test on a system where direct `np.linalg.solve` fails but your SVD solver succeeds.

**Solution:** Compute $A = U\Sigma V^T$. Set $\Sigma^+ = \text{diag}(1/\sigma_i)$ for $\sigma_i > \epsilon$, zero otherwise. Then $x = V\Sigma^+ U^T b$. This gives the minimum-norm least-squares solution. Test: $A = \begin{pmatrix} 1 & 1 \\ 2 & 2 \end{pmatrix}$, $b = (3, 6)^T$ — rank 1, infinite solutions, SVD returns the one with smallest $\|x\|$.
</details>

<details>
<summary>🎯 Challenge 3: End-to-end optimization pipeline</summary>

**Problem:** For the Rosenbrock function $f(x,y) = 100(y-x^2)^2 + (1-x)^2$ starting from $(-1.2, 1.0)$: (a) compute the gradient and Hessian symbolically, (b) perform 5 Newton steps manually tracking $\|x_k - x^*\|$ and $\|\nabla f_k\|$, (c) verify SOSC at the solution, (d) estimate the convergence rate from your iteration data.

**Solution:** At $(1,1)$: $H = \begin{pmatrix} 802 & -400 \\ -400 & 200 \end{pmatrix}$, eigenvalues $\approx 1001.6, 0.4$ — PD, so SOSC holds. Newton converges quadratically: once in the basin, the error squares each step. From $(-1.2, 1.0)$, convergence is not immediate because the initial point is far from the basin of quadratic convergence. Track $\|x_{k+1}-x^*\|/\|x_k-x^*\|^2$ to verify the quadratic rate near convergence.
</details>

---

## Key Takeaways

1. **Linear algebra is the language** — every optimization concept (gradient, Hessian, convergence) is expressed in terms of vectors, matrices, and their properties.
2. **Classification pipeline** works: find stationary points (FONC: $\nabla f = 0$), then check Hessian (SOSC: $\nabla^2 f \succ 0$).
3. **Convexity is the dividing line** — convex problems have unique global minima; non-convex problems require careful initialization.
4. **Condition number predicts difficulty** — high $\kappa$ means slow convergence and numerical sensitivity.

---

## Key Connections Forward

These concepts directly feed into:

- **Week 2:** Sparse matrices (efficiency of Jacobians), automatic differentiation (computing gradients), conjugate gradient (solving $Hx = g$)
- **Week 3-4:** Gradient descent, Newton/quasi-Newton methods, trust regions
- **Week 5-6:** Gauss-Newton and LM (Hessian approximation), Ceres Solver
- **Week 7-8:** Constrained optimization (Lagrange multipliers, KKT conditions)
- **Week 9-10:** Factor graphs (sparse Jacobians), SLAM (all of the above)
