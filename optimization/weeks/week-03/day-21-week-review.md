# Day 21: Week 3 Review & Optimizer Shootout

> **Phase II — Unconstrained Optimization | Week 3 | 2.5 hours**
> *Benchmark everything you've built. Know when to use each method.*

## Navigation
- **Previous:** [Day 20: Conjugate Gradient](day-20-conjugate-gradient.md)
- **Next:** [Day 22: Convergence Theory](../week-04/day-22-convergence-theory.md)
- **Week:** [Week 3 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Every Ceres `Solver::Options` choice maps to this week's content: `minimizer_type` (LINE_SEARCH vs TRUST_REGION), `line_search_direction_type` (STEEPEST_DESCENT, LBFGS, BFGS), `trust_region_strategy_type` (LEVENBERG_MARQUARDT, DOGLEG). Understanding which method wins on which problem is the difference between a 10-second solve and a failed solve.

---

## Theory Review (45 min)

### Week 3 Method Comparison Table

| Method | Convergence | Per-step | Storage | Hessian? | Best for |
|--------|-------------|----------|---------|----------|----------|
| GD | Linear: $O(\kappa)$ | $O(n)$ | $O(n)$ | No | Simple problems, huge $n$ |
| GD + Wolfe | Linear | $O(n)$ | $O(n)$ | No | When gradients cheap |
| Newton | Quadratic | $O(n^3)$ | $O(n^2)$ | Yes | Small dense problems |
| Newton + LS | Quadratic (local) | $O(n^3)$ | $O(n^2)$ | Yes | Medium problems |
| Trust Region (Cauchy) | Linear | $O(n)$ | $O(n)$ | Yes | Robust, handles indefinite $H$ |
| Trust Region (Dogleg) | Superlinear | $O(n^3)$ | $O(n^2)$ | Yes | Ceres default |
| BFGS | Superlinear | $O(n^2)$ | $O(n^2)$ | No | Medium $n$, no Hessian |
| L-BFGS | Superlinear | $O(mn)$ | $O(mn)$ | No | Large $n$, Ceres LINE_SEARCH |
| CG (PR+) | $n$-step (quad) | $O(n)$ | $O(n)$ | No | Very large $n$, sparse |

### Decision Tree

```
Is n small (< 1000)?
├── Yes: Can you compute H?
│   ├── Yes: Trust Region Dogleg or Newton + LS
│   └── No:  BFGS
└── No:
    ├── Is H sparse? → Sparse Newton / Trust Region (Ceres default)
    ├── Can afford O(mn)? → L-BFGS
    └── Minimal memory? → CG (PR+)
```

### Key Formulas Recap

| Method | Update Rule |
|--------|-------------|
| GD | $x_{k+1} = x_k - \alpha_k \nabla f_k$ |
| Newton | $x_{k+1} = x_k - H_k^{-1}\nabla f_k$ |
| Trust Region | $\min_\delta m(\delta)$ s.t. $\|\delta\| \leq \Delta$, update $\Delta$ via $\rho$ |
| BFGS | $H_{k+1} = (I-\rho sy^T)H_k(I-\rho ys^T) + \rho ss^T$ |
| L-BFGS | Two-loop recursion with $m$ stored $(s,y)$ pairs |
| CG | $d_{k+1} = -g_{k+1} + \beta_k d_k$, $\beta^{PR+} = \max(0, g_{k+1}^T(g_{k+1}-g_k)/\|g_k\|^2)$ |

---

## Implementation (60 min)

### Optimizer Shootout
```python
# See: code/week03/optimizer_shootout.py
```

Benchmark all 6 methods on 5 test functions across 5 metrics:

**Test Functions:**
1. Rosenbrock (2D) — narrow curved valley
2. Styblinski-Tang (5D) — many local minima
3. Beale (2D) — flat regions
4. Extended Rosenbrock (20D) — high dimensional
5. Quadratic (50D, $\kappa = 1000$) — controlled conditioning

**Metrics:**
1. Iterations to convergence
2. Function evaluations
3. Wall-clock time
4. Final error $\|x - x^*\|$
5. Final gradient norm $\|\nabla f\|$

---

## Practice (45 min)

### Comprehensive Review Problems

1. **Match the method:** For each scenario, choose the best optimizer:
   - 3D problem with cheap Hessian → ?
   - 100,000-variable SLAM → ?
   - Noisy gradient, no Hessian → ?
   - Dense 500-variable problem → ?

2. **Failure modes:** For each method, describe its worst case:
   - GD on ill-conditioned quadratic → ?
   - Newton at a saddle point → ?
   - BFGS without Wolfe conditions → ?
   - CG (FR) on non-quadratic → ?

3. **Rate ordering:** Sort by convergence speed near the solution: GD, BFGS, Newton, CG.

4. **Memory ordering:** Sort by memory usage for $n = 10,000$: Newton, L-BFGS ($m=10$), CG, GD, BFGS.

5. **Ceres mapping:** Match each Ceres setting to this week's topic:
   - `STEEPEST_DESCENT` → ?
   - `LBFGS` → ?
   - `LEVENBERG_MARQUARDT` → ?
   - `DOGLEG` → ?

6. **Phase II gate:** Solve by hand: $\min f(x,y) = (x-1)^2 + 10(y-x^2)^2$ — apply one step of GD, Newton, and BFGS from $(0,0)$.

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Unified optimizer interface</summary>

**Problem:** Design a Python optimizer class hierarchy that supports all 6 methods with a common interface: `minimize(f, grad, x0, options)`. Each method should track iterations, function evals, gradient evals, and time. Compare your design with `scipy.optimize.minimize`.

**Solution:** Abstract base class `Optimizer` with `step()` and `converged()` methods. Factory pattern: `create_optimizer("bfgs", options)`. This is exactly what you'll build in the Day 28 capstone (`miniopt` library). The key abstraction: all methods compute a search direction $d_k$ and a step length $\alpha_k$ — the difference is how they compute $d_k$.
</details>

<details>
<summary>🎯 Challenge 2: Adaptive method switching</summary>

**Problem:** Build a meta-optimizer that starts with L-BFGS for global progress, then switches to Newton when $\|g\| < \varepsilon$ (close to solution). Implement and show it combines L-BFGS's robustness with Newton's quadratic finish.

**Solution:** Monitor $\|g\|$: when below threshold, start computing Hessian and switch to Newton. The transition should be seamless (same $x$, fresh $H$). Handle the case where Hessian computation fails (fall back to L-BFGS). This is the idea behind "two-phase" solvers. Ceres does something similar: it uses trust region (global) with careful damping that approaches Newton (local).
</details>

<details>
<summary>🎯 Challenge 3: Benchmarking framework with statistical rigor</summary>

**Problem:** Run each optimizer on each function 20 times with random starting points. Report mean ± std for all metrics. Generate performance profiles (Dolan-Moré): for each method, plot the fraction of problems solved within factor $\tau$ of the best method. This is the standard way to compare optimizers in the literature.

**Solution:** Performance profiles: for solver $s$ on problem $p$, compute ratio $r_{p,s} = t_{p,s} / \min_s t_{p,s}$. Plot $\rho_s(\tau) = \text{fraction of problems where } r_{p,s} \leq \tau$. The solver with highest $\rho_s(1)$ is the most efficient; the one reaching $\rho_s = 1$ first is the most robust. See Dolan & Moré (2002). This is the gold standard for optimizer comparison papers.
</details>

---

## Self-Assessment Checklist

- [ ] I can implement all 6 optimization methods from scratch
- [ ] I know the convergence rate, cost, and storage of each method
- [ ] I can choose the right method for a given problem size and structure
- [ ] I can map each method to its Ceres solver configuration

---

## Key Takeaways

1. **No single method wins everywhere** — choose based on $n$, Hessian availability, and structure.
2. **Trust region** (Ceres default) is the most robust for nonlinear least squares.
3. **L-BFGS** is the workhorse for large unconstrained problems without Hessian.
4. **Newton** gives quadratic convergence but needs $O(n^3)$ — worth it only for small/sparse problems.
