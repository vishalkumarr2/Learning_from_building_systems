# Day 27: scipy.optimize Deep Dive

> **Phase II — Unconstrained Optimization | Week 4 | 2.5 hours**
> *Mastering the Swiss Army knife of scientific optimization in Python.*

## Navigation
- **Previous:** [Day 26: Global Optimization](day-26-global-optimization.md)
- **Next:** [Day 28: Week 4 Review & Capstone](day-28-week-review.md)
- **Week:** [Week 4 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
`scipy.optimize` is the default Python solver for quick prototyping and offline analysis at Rapyuta. When analyzing OKS logs, you'll minimize residuals with `least_squares`, calibrate sensor models with `minimize`, and fit distributions with `curve_fit`. For production, Ceres/OSQP are preferred — but scipy is where you prototype, validate, and debug. Knowing every method and when to use each saves hours of trial and error.

---

## Theory (45 min)

### 27.1 The `minimize` Interface

```python
scipy.optimize.minimize(fun, x0, method=..., jac=..., hess=...,
                        bounds=..., constraints=..., callback=..., options=...)
```

Returns `OptimizeResult` with `.x`, `.fun`, `.success`, `.nfev`, `.nit`, `.jac`, `.hess_inv`.

### 27.2 Method Taxonomy

| Method | Order | Gradient? | Hessian? | Bounds? | Best for |
|--------|-------|-----------|----------|---------|----------|
| `Nelder-Mead` | 0 | No | No | No | Small $n$, noisy, no derivatives |
| `Powell` | 0 | No | No | No | Moderate $n$, no derivatives |
| `CG` | 1 | Yes | No | No | Large $n$, smooth |
| `BFGS` | quasi-2 | Yes | No | No | General purpose, $n < 1000$ |
| `L-BFGS-B` | quasi-2 | Yes | No | **Yes** | Large $n$ with bounds |
| `TNC` | quasi-2 | Yes | No | **Yes** | Large $n$ with bounds |
| `Newton-CG` | 2 | Yes | Yes | No | Accurate Hessian available |
| `trust-constr` | 2 | Yes | Optional | **Yes** | Constraints + bounds |
| `trust-ncg` | 2 | Yes | Yes | No | Sparse Hessian, positive definite |
| `trust-krylov` | 2 | Yes | Yes | No | Very large $n$ |
| `dogleg` | 2 | Yes | Yes | No | Small $n$, positive definite Hessian |

### 27.3 Choosing the Right Method

**Decision tree:**
1. Can you compute gradients? → No: `Nelder-Mead` (small $n$) or `Powell`
2. Can you compute Hessians? → No: `BFGS` or `L-BFGS-B`
3. Need bounds? → `L-BFGS-B` or `TNC`
4. Need constraints? → `trust-constr` or `SLSQP`
5. Very large $n$ ($>10^4$)? → `L-BFGS-B` (memory-limited)
6. Least-squares structure? → Use `least_squares` instead

### 27.4 Supplying Derivatives

Three options:
1. **Analytic Jacobian:** `jac=my_gradient_function` — fastest, most accurate
2. **Finite differences:** `jac='2-point'` or `'3-point'` — automatic, $O(n)$ extra evaluations
3. **Automatic differentiation:** Use JAX's `jax.grad` then pass to scipy

**Hessian options:** `hess=callable`, `hessp=callable` (Hessian-vector product), or BFGS approximation.

### 27.5 Callbacks and Monitoring

```python
def callback(xk):
    """Called at each iteration with current x."""
    print(f"  x = {xk}, f(x) = {fun(xk)}")
```

Useful for: logging convergence, implementing early stopping, saving intermediate results.

### 27.6 Least Squares Preview

`scipy.optimize.least_squares(fun, x0, method='trf', jac=..., bounds=...)`

Solves: $\min \sum r_i(x)^2$ — exploits the least-squares structure (Gauss-Newton internally).

Methods:
- `'trf'` (Trust Region Reflective) — handles bounds, default
- `'dogbox'` — handles rectangular bounds efficiently
- `'lm'` — Levenberg-Marquardt, no bounds but fastest

**Week 5 goes deep on this.** Today: just know when to use `least_squares` vs `minimize`.

### 27.7 Common Pitfalls

1. **Not supplying gradients** when available — 10× slower
2. **Using `Nelder-Mead` for $n > 20$** — scales terribly
3. **Ignoring `success` flag** — always check `result.success`
4. **Not scaling variables** — different magnitudes → poor conditioning
5. **Forgetting bounds** — using `L-BFGS-B` without bounds when needed

---

## Implementation (60 min)

### Exercise 1: Every Method Comparison
```python
# See: code/week04/scipy_optimize_deep_dive.py
```
Run all applicable scipy methods on the Rosenbrock function. Compare iterations, function evaluations, and time.

### Exercise 2: Callback Logging
Implement a callback that logs $f(x_k)$, $\|x_k - x^*\|$, and $\|\nabla f(x_k)\|$ at each iteration. Plot convergence curves for BFGS vs CG vs Newton-CG.

### Exercise 3: Bounds and Scaling
Solve a problem where variables have different scales ($x_1 \in [0, 1]$, $x_2 \in [0, 10000]$). Show that without scaling, BFGS converges slowly. With proper scaling, convergence is fast.

### Exercise 4: Analytic Derivatives Speedup
Compare `jac=None` (finite differences), `jac='3-point'`, and `jac=analytic` for Rosenbrock. Measure total function evaluations and wall time.

### Exercise 5: Custom Optimizer Wrapping
Wrap your Week 3 BFGS implementation in a scipy-compatible interface. Compare with scipy's built-in BFGS on the same problem.

### Exercise 6: Wheel Radius Calibration
Practical: estimate wheel radius from encoder ticks and measured distances. Use `least_squares` for best results. Compare with `minimize` on the same problem.

---

## Practice (45 min)

1. What is the difference between `BFGS` and `L-BFGS-B`? When would you choose each?
2. For a 50-dimensional problem with analytic gradients and no bounds, which method would you try first? Second?
3. `Nelder-Mead` is derivative-free. Why is it still useful despite being slow?
4. What does `OptimizeResult.nfev` tell you that `.nit` doesn't?
5. If `result.success == False`, what should you do? (List 3 concrete steps.)
6. When should you use `least_squares` instead of `minimize`? What's the advantage?
7. How does `trust-constr` handle both equality and inequality constraints?
8. What is the memory advantage of L-BFGS-B? How does the `m` parameter (history size) affect it?
9. For the OKS sensor calibration problem with 6 parameters and 100 measurements, which scipy method would you use?
10. Why does scipy's `minimize` default to BFGS when gradients are provided?

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Benchmarking suite</summary>

**Problem:** Build a benchmarking function that tests all applicable scipy methods on a given problem and returns a comparison table: method name, final $f$, iterations, function evaluations, gradient evaluations, time, and success status. Run it on: (a) Rosenbrock 10D, (b) Styblinski-Tang 10D, (c) a quadratic with condition number $10^4$.

**Solution:** The suite should auto-detect which methods are applicable (e.g., skip methods requiring Hessian if not provided). Key insight: BFGS wins on smooth, moderate-dimensional problems. L-BFGS-B wins for $n > 100$. Nelder-Mead wins on noisy/non-smooth problems. For ill-conditioned quadratics, Newton-CG with exact Hessian beats everything.
</details>

<details>
<summary>🎯 Challenge 2: JAX + scipy hybrid</summary>

**Problem:** Use JAX's automatic differentiation with scipy's minimize. Write a function that: (a) defines the objective in JAX, (b) uses `jax.grad` for the Jacobian and `jax.hessian` for the Hessian, (c) passes these to `scipy.optimize.minimize(method='Newton-CG')`. Show the speedup over finite-difference Jacobians on Rosenbrock 50D.

**Solution:** `jax.grad(f)` returns a function. Wrap it to convert jax arrays to numpy for scipy compatibility. For Rosenbrock 50D: analytic gradient via JAX → 50× fewer function evaluations than '2-point' finite differences. With Hessian: Newton-CG converges in ~30 iterations vs BFGS's ~200.
</details>

<details>
<summary>🎯 Challenge 3: Constrained optimization preview</summary>

**Problem:** Use `trust-constr` to solve a constrained calibration problem: minimize calibration error subject to physical constraints (wheel radius between 5 and 15 cm, wheelbase between 30 and 60 cm). Compare with unconstrained BFGS followed by clipping, and show that `trust-constr` gives better results because it respects constraints during optimization, not just at the end.

**Solution:** `trust-constr` with `LinearConstraint` or `NonlinearConstraint` objects. The key difference: constrained optimization follows the constraint boundary when the unconstrained optimum is infeasible, finding the best feasible point. Clipping after BFGS gives a suboptimal feasible point because it doesn't explore along the constraint surface.
</details>

---

## Self-Assessment Checklist

- [ ] I know which scipy method to use for a given problem type
- [ ] I can supply analytic derivatives and measure the speedup
- [ ] I can use callbacks to monitor and debug optimization runs
- [ ] I understand when `least_squares` is better than `minimize`

---

## Key Takeaways

1. **`L-BFGS-B`** is the default workhorse — bounded, memory-efficient, gradient-based.
2. **Always supply gradients** when available — finite differences waste evaluations.
3. **Always check `result.success`** — silent failures are the most dangerous.
4. **`least_squares`** exploits problem structure — use it when you have residuals.
