# Day 28: Week 4 Review & Capstone — Build `miniopt`

> **Phase II — Unconstrained Optimization | Week 4 | 2.5 hours**
> *Consolidate everything into a working optimization library with a test suite.*

## Navigation
- **Previous:** [Day 27: scipy.optimize Deep Dive](day-27-scipy-optimize.md)
- **Next:** [Day 29: Linear Least Squares](../week-05/day-29-linear-least-squares.md)
- **Week:** [Week 4 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
This capstone cements your ability to build, test, and select optimization algorithms — a skill you'll use every time you prototype a new calibration, estimation, or control algorithm for the OKS stack. The `miniopt` library mirrors how real solver libraries are structured (Ceres, OSQP, scipy) with a unified interface and method selection.

---

## Week 4 Review (60 min)

### Key Concepts

| Day | Topic | Core Idea |
|-----|-------|-----------|
| 22 | Convergence Theory | Rates: linear ($r < 1$), superlinear, quadratic. Lipschitz conditions. |
| 23 | Momentum Methods | Heavy ball, Nesterov acceleration → $O(1/k^2)$ for convex |
| 24 | Stochastic Methods | SGD, mini-batch, Adam — variance reduction, adaptive learning rates |
| 25 | Regularization | L2 shrinks, L1 sparsifies, proximal operators, ISTA/FISTA |
| 26 | Global Optimization | Random restart, SA, basin-hopping — when local isn't enough |
| 27 | scipy.optimize | L-BFGS-B default, supply gradients, check `.success` |

### Method Selection Guide

```
Problem → Method
│
├─ Smooth, small n (<100)
│  ├─ Gradients available → BFGS
│  └─ No gradients → Nelder-Mead
│
├─ Smooth, large n (>100)
│  ├─ Gradients available → L-BFGS-B
│  └─ Stochastic/online → Adam
│
├─ Noisy objective → SGD with momentum
│
├─ Sparse solution desired → ISTA/FISTA (L1)
│
├─ Many local minima → Random restart or basin-hopping
│
└─ Least-squares structure → Gauss-Newton (Week 5!)
```

### Phase II Checklist

- [ ] Implemented gradient descent, Newton, BFGS, L-BFGS from scratch
- [ ] Understand line search (Wolfe conditions) and trust regions
- [ ] Can analyze convergence rate (linear/superlinear/quadratic)
- [ ] Know momentum (Nesterov) and stochastic methods (Adam)
- [ ] Understand regularization (L1/L2) and proximal operators
- [ ] Know when to use global optimization vs local
- [ ] Can choose the right scipy method for any given problem
- [ ] Built `miniopt` with unified interface and test suite

---

## Capstone: Build `miniopt` (90 min)

### Requirements

Build a Python library `miniopt` with:

1. **Unified interface:** `miniopt.minimize(f, x0, method='bfgs', **kwargs)`
2. **Methods:** At least 6:
   - `'gd'` — Gradient descent with backtracking
   - `'momentum'` — Heavy ball method
   - `'nesterov'` — Nesterov accelerated gradient
   - `'bfgs'` — BFGS quasi-Newton
   - `'lbfgs'` — L-BFGS (memory-limited)
   - `'adam'` — Adam optimizer
3. **Result dataclass:** `OptimizeResult` with fields: `x`, `fun`, `nit`, `nfev`, `ngrad`, `success`, `message`, `history`
4. **Standard test functions:** Rosenbrock, Beale, Styblinski-Tang, Booth, sphere
5. **pytest test suite:** At least 15 tests covering correctness, convergence, edge cases

### Implementation

```python
# See: code/week04/miniopt.py
# See: code/week04/test_miniopt.py
```

### Test Expectations

```
pytest test_miniopt.py -v
```

Tests should cover:
- Each method converges on a quadratic
- BFGS converges on Rosenbrock
- L-BFGS produces similar results to BFGS
- Adam handles noisy objectives
- OptimizeResult has correct fields
- Method names are validated
- Edge cases: zero-dimensional, already at minimum

---

## Practice (Remaining time)

1. Draw the inheritance diagram for `miniopt`: what's shared across methods? What differs?
2. For each method in `miniopt`, state its convergence rate class.
3. Which `miniopt` methods would you use on the OKS wheel calibration problem? Why?
4. If you were adding one more method to `miniopt`, what would it be? (Hint: Week 5)
5. Compare your `miniopt.minimize(method='bfgs')` with `scipy.optimize.minimize(method='BFGS')` on Rosenbrock 5D.

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Method auto-selection</summary>

**Problem:** Add a `method='auto'` option that selects the best method based on problem characteristics:
- If `jac` is provided and `n < 100` → BFGS
- If `jac` is provided and `n >= 100` → L-BFGS
- If no `jac` and `n < 20` → Nelder-Mead (numerical gradient)
- If `stochastic=True` → Adam

Test that `'auto'` selects correctly for various problem sizes.

**Solution:** Inspect function signature and keyword arguments to determine properties. Write tests that verify: (a) auto picks BFGS for 10D with gradients, (b) auto picks L-BFGS for 200D, (c) auto picks Adam for stochastic.
</details>

<details>
<summary>🎯 Challenge 2: Convergence visualization</summary>

**Problem:** Add a `miniopt.plot_convergence(results)` function that takes a list of `OptimizeResult` objects (from different methods on the same problem) and plots:
1. $f(x_k)$ vs iteration (log scale)
2. $\|x_k - x^*\|$ vs iteration (log scale)
3. $\|\nabla f(x_k)\|$ vs iteration (log scale)
4. Wall time vs accuracy achieved

**Solution:** Each method's `history` list stores `(f, x, grad_norm)` tuples. Matplotlib subplots with shared x-axis. Color-code by method. This visualization makes it immediately clear which method converges fastest and to what accuracy.
</details>

<details>
<summary>🎯 Challenge 3: Benchmark against scipy</summary>

**Problem:** Run `miniopt` and `scipy.optimize.minimize` on the same 5 test functions in dimensions $n = 2, 5, 10, 50$. Compare: iterations to convergence ($10^{-6}$ tolerance), total function evaluations, wall time, and final accuracy. Identify where your implementation beats scipy and where it doesn't.

**Solution:** scipy will generally win on wall time (C implementation) but iteration counts should be similar for BFGS. Your Adam may beat scipy's quasi-Newton on noisy problems. The comparison teaches that algorithmic choice matters more than implementation speed for most robotics problems.
</details>

---

## Self-Assessment Checklist

- [ ] `miniopt.minimize()` works with all 6 methods
- [ ] All pytest tests pass
- [ ] I can articulate when to use each method
- [ ] I'm ready for Phase III (nonlinear least squares)

---

## Key Takeaways

1. **Unified interfaces** make algorithm comparison easy and method swapping painless.
2. **Testing** optimization code requires known-solution test functions, not just "it runs."
3. **Phase II complete** — you now have a working optimizer library built from scratch.
4. **Next:** Phase III applies these methods to the specific structure of least-squares problems.
