# 03 — Least Squares and the Ceres Solver

> **Goal:** Master the most important optimization structure in robotics — least squares — from normal equations through Gauss-Newton and Levenberg-Marquardt to the Ceres Solver API.

> **This is the highest-value chapter for OKS work.** Every sensorbar fit, every calibration, every SLAM backend runs a nonlinear least-squares solver.

---

## 1. Linear Least Squares

### Problem

Given $A \in \mathbb{R}^{m \times n}$ ($m > n$, overdetermined) and $b \in \mathbb{R}^m$:

$$\min_x \|Ax - b\|_2^2 = \min_x \sum_{i=1}^m (a_i^T x - b_i)^2$$

There are more equations ($m$) than unknowns ($n$), so no exact solution exists. We find the $x$ that minimizes the sum of squared residuals.

### Normal Equations

Setting the gradient to zero:

$$\nabla_x \|Ax - b\|^2 = 2A^T(Ax - b) = 0$$

$$\boxed{A^T A x^* = A^T b}$$

This is the **normal equation**. The solution: $x^* = (A^TA)^{-1}A^Tb$.

### Geometric Interpretation

$Ax^*$ is the **orthogonal projection** of $b$ onto the column space of $A$. The residual $r = b - Ax^*$ is orthogonal to every column of $A$: $A^T r = 0$.

### Solving Methods

| Method | Cost | Numerical stability | When to use |
|--------|------|---------------------|-------------|
| Normal equations ($A^TAx = A^Tb$) | $O(mn^2 + n^3)$ | Poor for ill-conditioned $A$ | Quick-and-dirty, small problems |
| QR factorization ($A = QR$) | $O(mn^2)$ | Good | Default choice |
| SVD ($A = U\Sigma V^T$) | $O(mn^2)$ | Best | Rank-deficient or near-singular |

```python
import numpy as np

# All three methods — same result, different stability
# Method 1: Normal equations
x_ne = np.linalg.solve(A.T @ A, A.T @ b)

# Method 2: QR (what np.linalg.lstsq uses internally)
x_qr, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)

# Method 3: SVD
U, s, Vt = np.linalg.svd(A, full_matrices=False)
x_svd = Vt.T @ np.diag(1/s) @ U.T @ b
```

---

## 2. Weighted Least Squares

### Problem

Different measurements have different reliability. Weight them:

$$\min_x \sum_{i=1}^m w_i (a_i^T x - b_i)^2 = \min_x \|W^{1/2}(Ax - b)\|^2$$

where $W = \text{diag}(w_1, \ldots, w_m)$.

Normal equations become: $A^T W A x = A^T W b$.

**OKS relevance:** The EKF measurement update is exactly a weighted least-squares problem. The weights are the inverse measurement covariance: $W = R^{-1}$. Sensorbar measurements with high uncertainty get low weight.

---

## 3. Nonlinear Least Squares (NLLS)

### Problem

When the model is nonlinear in the parameters:

$$\min_x \sum_{i=1}^m r_i(x)^2 = \min_x \|r(x)\|^2$$

where $r : \mathbb{R}^n \to \mathbb{R}^m$ is the vector of **residuals**.

**Example:** Fitting a sensorbar measurement model $f(\theta, d; p)$ where the residuals are:

$$r_i(p) = \text{measured}_i - f(\theta_i, d_i; p)$$

Cannot solve with normal equations because $r(x)$ is nonlinear. Need iterative methods.

---

## 4. Gauss-Newton Method

### Key Insight

Linearize each residual around the current estimate $x_k$:

$$r(x_k + \Delta x) \approx r(x_k) + J_k \Delta x$$

where $J_k = \frac{\partial r}{\partial x}\bigg|_{x_k}$ is the Jacobian of residuals.

Substitute into the cost:

$$\|r(x_k) + J_k \Delta x\|^2$$

This is now a **linear** least-squares problem in $\Delta x$! The normal equations give:

$$\boxed{J_k^T J_k \Delta x = -J_k^T r_k}$$

### Algorithm

```python
def gauss_newton(residual_fn, jacobian_fn, x0, tol=1e-8, max_iter=50):
    x = x0.copy()
    for k in range(max_iter):
        r = residual_fn(x)
        J = jacobian_fn(x)
        # Solve normal equations: (J^T J) Δx = -J^T r
        dx = np.linalg.solve(J.T @ J, -J.T @ r)
        x = x + dx
        if np.linalg.norm(dx) < tol:
            break
    return x, k
```

### Connection to Newton's Method

For $f(x) = \frac{1}{2}\|r(x)\|^2$:

- Gradient: $\nabla f = J^T r$
- Hessian: $\nabla^2 f = J^T J + \sum_i r_i \nabla^2 r_i$

Gauss-Newton **drops the second term** $\sum_i r_i \nabla^2 r_i$, approximating:

$$H \approx J^T J$$

This is valid when:
- Residuals $r_i$ are small at the solution (good fit)
- Or the problem is "nearly linear" (second derivatives of $r_i$ are small)

**When it fails:** Large residuals → $J^TJ$ is a poor Hessian approximation → oscillation or divergence.

---

## 5. Levenberg-Marquardt (LM)

### The Fix

Add a damping term to the Gauss-Newton normal equations:

$$\boxed{(J_k^T J_k + \lambda_k I) \Delta x = -J_k^T r_k}$$

### What $\lambda$ Does

| $\lambda$ | Step behaviour | Equivalent to |
|-----------|----------------|---------------|
| $\lambda \to 0$ | Gauss-Newton step | Fast but risky |
| $\lambda \to \infty$ | $\Delta x \approx -\frac{1}{\lambda} J^T r$ = gradient descent | Safe but slow |
| Moderate $\lambda$ | Blend | Best of both worlds |

### Adaptive $\lambda$ Strategy

```
1. Compute trial step Δx from (J^T J + λI) Δx = -J^T r
2. Compute gain ratio: ρ = (‖r(x)‖² - ‖r(x + Δx)‖²) / predicted_reduction
3. If ρ > threshold (e.g., 0.25):
   - Accept step: x ← x + Δx
   - Decrease λ (more Gauss-Newton-like)
4. If ρ < threshold:
   - Reject step: keep x
   - Increase λ (more gradient-descent-like)
```

### Trust Region Interpretation

LM is equivalent to solving:

$$\min_{\Delta x} \|J_k \Delta x + r_k\|^2 \quad \text{s.t.} \quad \|\Delta x\| \leq \Delta_k$$

The Lagrange multiplier for the constraint is $\lambda_k$. Large $\lambda$ = small trust region.

---

## 6. Robust Estimation: Loss Functions

### Problem with $\ell_2$

Squared error $r_i^2$ gives outliers **quadratic** influence. A single bad measurement can ruin the fit.

### Robust Loss Functions

Replace $\frac{1}{2}r^2$ with $\rho(r)$:

| Loss | $\rho(r)$ | Influence $\rho'(r)$ | Behaviour |
|------|-----------|----------------------|-----------|
| Squared ($\ell_2$) | $\frac{1}{2}r^2$ | $r$ | Unbounded — outliers dominate |
| Huber | $\begin{cases} \frac{1}{2}r^2 & |r| \leq \delta \\ \delta|r| - \frac{1}{2}\delta^2 & |r| > \delta \end{cases}$ | Capped at $\pm\delta$ | Linear growth for outliers |
| Cauchy | $\frac{c^2}{2}\log(1 + (r/c)^2)$ | $\frac{r}{1 + (r/c)^2}$ | Logarithmic growth → strong outlier rejection |
| Tukey | $\begin{cases} \frac{c^2}{6}[1-(1-(r/c)^2)^3] & |r| \leq c \\ \frac{c^2}{6} & |r| > c \end{cases}$ | Zero beyond $c$ | Completely ignores outliers |

### In Ceres

```cpp
// Standard squared error
problem.AddResidualBlock(cost_fn, nullptr, params);

// Cauchy loss with c = 40.0 (used in OKS sensorbar)
problem.AddResidualBlock(cost_fn, new ceres::CauchyLoss(40.0), params);

// Huber loss with δ = 1.0
problem.AddResidualBlock(cost_fn, new ceres::HuberLoss(1.0), params);
```

**OKS sensorbar uses `CauchyLoss(c=40.0)`:** This means measurements with residual $|r| \gg 40$ have their influence heavily down-weighted. This is critical because sensorbar readings can have large outliers from reflective floor markings or partial occlusions.

### Iteratively Reweighted Least Squares (IRLS)

Robust estimation can be converted to weighted least squares by iterating:

1. Solve WLS with current weights
2. Update weights: $w_i = \rho'(r_i) / r_i$
3. Repeat until convergence

Ceres handles this internally through its `LossFunction` mechanism.

---

## 7. The Ceres Solver — Architecture and API

### Core Concepts

```
Problem
├── ParameterBlock(s)    ← the unknowns: x, y, θ, ...
├── ResidualBlock(s)     ← one per measurement/constraint
│   ├── CostFunction     ← computes r(x) and optionally J
│   └── LossFunction     ← ρ(r) — null means squared error
└── Solver::Options      ← algorithm choice, convergence criteria
```

### CostFunction

Computes residuals and Jacobians:

```cpp
struct SensorbarCostFunctor {
    SensorbarCostFunctor(double measured_y, double line_x)
        : measured_y_(measured_y), line_x_(line_x) {}

    template <typename T>
    bool operator()(const T* const params, T* residual) const {
        // params[0] = theta, params[1] = offset
        T predicted_y = params[1] + T(line_x_) * ceres::tan(params[0]);
        residual[0] = T(measured_y_) - predicted_y;
        return true;
    }

    double measured_y_;
    double line_x_;
};

// Auto-differentiation — Ceres computes Jacobians for you
ceres::CostFunction* cost =
    new ceres::AutoDiffCostFunction<SensorbarCostFunctor, 1, 2>(
        new SensorbarCostFunctor(measurement, x_position));
```

### Solver Options

```cpp
ceres::Solver::Options options;
options.linear_solver_type = ceres::DENSE_QR;           // Small problems
// options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY; // Large sparse
// options.linear_solver_type = ceres::ITERATIVE_SCHUR;  // SLAM problems

options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;  // Default
// options.trust_region_strategy_type = ceres::DOGLEG;

options.max_num_iterations = 50;
options.function_tolerance = 1e-6;
options.gradient_tolerance = 1e-10;
options.parameter_tolerance = 1e-8;
```

### Solver Output

```cpp
ceres::Solver::Summary summary;
ceres::Solve(options, &problem, &summary);

// Key fields for debugging:
summary.initial_cost        // Cost before optimization
summary.final_cost          // Cost after optimization
summary.iterations          // How many LM steps
summary.termination_type    // CONVERGENCE, NO_CONVERGENCE, FAILURE
summary.message             // Human-readable status
```

### Linear Solver Selection Guide

| Solver | Problem structure | Typical use |
|--------|-------------------|-------------|
| `DENSE_QR` | Small ($n < 100$), dense | Sensorbar fit, calibration |
| `DENSE_NORMAL_CHOLESKY` | Small, normal equations OK | Fast for small dense NLLS |
| `SPARSE_NORMAL_CHOLESKY` | Large, sparse Jacobian | Bundle adjustment |
| `CGNR` | Large, sparse, iterative | When factorization is too expensive |
| `ITERATIVE_SCHUR` | BA / SLAM structure | Exploits graph sparsity pattern |

---

## 8. Automatic Differentiation

### The Problem

Computing Jacobians by hand is:
- Error-prone (especially for complex residuals)
- Tedious to maintain as the model changes

### Three Approaches

| Approach | How | Accuracy | Speed | Ceres support |
|----------|-----|----------|-------|---------------|
| Analytic | Derive $\frac{\partial r}{\partial x}$ by hand | Exact | Fastest | `SizedCostFunction` |
| Numeric | Finite differences: $\frac{f(x+h)-f(x)}{h}$ | ~$O(h)$ | Slow ($2n$ evaluations) | `NumericDiffCostFunction` |
| Automatic | Dual numbers / operator overloading | Exact (to machine precision) | ~2–5× cost function | `AutoDiffCostFunction` ✅ |

**Always prefer `AutoDiffCostFunction`** — exact Jacobians with minimal effort. Use `NumericDiffCostFunction` only for black-box functions you can't template.

### How Auto-Diff Works (Dual Numbers)

A dual number: $a + b\epsilon$ where $\epsilon^2 = 0$.

$$f(a + \epsilon) = f(a) + f'(a)\epsilon$$

Evaluate $f$ once with dual numbers → get both $f(a)$ and $f'(a)$ for free. Ceres extends this to arbitrary dimensions via `Jet<T, N>`.

---

## 9. Debugging Ceres: Common OKS Failure Modes

| Symptom | Likely cause | What to check |
|---------|--------------|---------------|
| `NO_CONVERGENCE` after max iterations | Poor initial guess or ill-conditioned | Increase `max_num_iterations`, check initial parameter values |
| `NUMERICAL_FAILURE` | Jacobian has NaN/Inf | Check for division by zero in cost function, `tan()` at ±π/2 |
| `final_cost ≈ initial_cost` | Solver didn't move | Gradient near zero at start (local min) or `parameter_tolerance` too tight |
| Cost decreasing but slowly | High condition number | Try different `linear_solver_type`, add regularization |
| Oscillating cost | Trust region too large/small | Check `min_lm_diagonal`, `max_lm_diagonal` |
| Good fit but wrong parameters | Multiple local minima | Non-convex problem — try different initial guesses |

---

## Summary: What to Carry Forward

| Concept | Where it's used next |
|---------|---------------------|
| Normal equations $A^TAx = A^Tb$ | 07 (conditioning), 06 (pose-graph normal equations) |
| Gauss-Newton as approximate Newton | 06 (graph optimization uses GN on the pose-graph cost) |
| LM damping ($\lambda$) | 06 (robust pose-graph optimization) |
| Robust loss functions | 06 (outlier loop closures in SLAM) |
| Ceres API | 06 (Ceres as a graph optimizer), ongoing RCA work |
| Auto-differentiation | Any new cost function you write |
