# 07 — Numerical Methods and Practical Concerns

> **Goal:** Understand the numerical pitfalls that make textbook algorithms fail in practice — conditioning, convergence, sparsity, and the tricks that real solvers use.

---

## 1. Floating-Point Arithmetic

### The Basics

IEEE 754 double precision: 64 bits → ~15–16 significant decimal digits.

$$\text{Machine epsilon:} \quad \epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$$

Every floating-point operation introduces relative error $\leq \epsilon_{\text{mach}}$.

### Catastrophic Cancellation

Subtracting nearly equal numbers amplifies relative error:

```python
# BAD: catastrophic cancellation
a = 1.0000000000001
b = 1.0000000000000
diff = a - b  # Only ~1 digit of accuracy left!

# BETTER: reformulate to avoid subtraction
# Use log1p(x) instead of log(1+x) for small x
# Use expm1(x) instead of exp(x)-1 for small x
```

### Implications for Optimization

- Finite-difference Jacobians: $\frac{f(x+h) - f(x)}{h}$ — cancellation in numerator
- Convergence tests: `|x_new - x_old| < tol` fails near machine precision
- Residual computation: summing many small terms → use compensated summation

---

## 2. Condition Number

### Definition

For a matrix $A$:

$$\kappa(A) = \|A\| \cdot \|A^{-1}\| = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}$$

### What It Means

$\kappa(A)$ measures how much the solution $x$ can change when $b$ is perturbed in $Ax = b$:

$$\frac{\|\Delta x\|}{\|x\|} \leq \kappa(A) \frac{\|\Delta b\|}{\|b\|}$$

| $\kappa(A)$ | Interpretation | Digits lost |
|-------------|----------------|-------------|
| $10^0$ = 1 | Perfect conditioning | 0 |
| $10^3$ | Mild | ~3 |
| $10^6$ | Moderate | ~6 |
| $10^{10}$ | Severe | ~10 (only 5–6 digits reliable) |
| $10^{16}$ | Singular to working precision | All |

### For Optimization

The Hessian condition number determines convergence:
- Gradient descent: $O(\kappa)$ iterations
- Conjugate gradient: $O(\sqrt{\kappa})$ iterations
- Newton: independent of $\kappa$ (but solving $H\Delta x = -g$ is sensitive to $\kappa(H)$)

### Checking in Practice

```python
import numpy as np

A = np.array([[1, 1], [1, 1.0001]])  # Nearly singular
print(f"Condition number: {np.linalg.cond(A):.0f}")  # ~20,000

# For the normal equations:
J = compute_jacobian(x)
H = J.T @ J
print(f"Hessian condition: {np.linalg.cond(H):.0f}")
# Note: cond(J^T J) = cond(J)^2  ← squaring makes things worse!
```

**OKS relevance:** The normal equations $J^TJ \Delta x = -J^Tr$ square the condition number. This is why Ceres prefers QR factorization (`DENSE_QR`) over normal equations for ill-conditioned problems.

---

## 3. Solving Linear Systems

### The Core Operation

Every iteration of Newton/GN/LM requires solving:

$$H \Delta x = -g$$

How you solve this determines practical performance.

### Direct Methods

| Method | Cost | Requirements | Stability |
|--------|------|-------------|-----------|
| LU decomposition | $O(n^3)$ | General $H$ | Good with pivoting |
| Cholesky $H = LL^T$ | $O(n^3/3)$ | $H \succ 0$ (positive definite) | Excellent |
| QR of $J$ | $O(mn^2)$ | Overdetermined LS | Better than normal eqs |
| Sparse Cholesky | $O(\text{fill-in})$ | Sparse $H \succ 0$ | Excellent |

### When to Use What

```
Is H sparse?
├── YES → How sparse?
│   ├── Very sparse (SLAM, large-scale) → Sparse Cholesky (CHOLMOD)
│   │   or Iterative (CG/LSQR)
│   │
│   └── Moderately sparse → Sparse direct (SuiteSparse)
│
└── NO (dense) → Is H positive definite?
    ├── YES → Cholesky (fastest)
    └── NO → LU with pivoting, or add damping (LM)
```

### Iterative Methods

For very large systems where even sparse Cholesky is too expensive:

| Method | Applies to | Per-iteration | Convergence |
|--------|-----------|---------------|-------------|
| Conjugate Gradient | $H \succ 0$, symmetric | $O(\text{nnz}(H))$ | $O(\sqrt{\kappa})$ |
| LSQR | Rectangular $J$ | $O(\text{nnz}(J))$ | $O(\sqrt{\kappa})$ |
| GMRES | General non-symmetric | $O(\text{nnz}(H) \cdot k)$ | Depends |

**Preconditioning** dramatically speeds up iterative solvers by reducing the effective $\kappa$.

---

## 4. Sparsity and Fill-In

### The Sparsity Pattern

In SLAM/graph optimization, $H$ has a specific sparsity pattern determined by the graph structure.

```
Original H:     After Cholesky L:
[X . . X]       [X . . .]
[. X X .]       [. X X .]
[. X X X]  →    [. X X X]
[X . X X]       [X F X X]  ← F = "fill-in" (new non-zero)
```

**Fill-in** = new non-zero entries created during factorization. More fill-in → more computation.

### Variable Ordering

The **ordering of variables** dramatically affects fill-in.

| Ordering | Fill-in | Method |
|----------|---------|--------|
| Natural (as given) | Can be terrible | Default |
| AMD (Approximate Min Degree) | Good general-purpose | `SuiteSparse`, Ceres default |
| METIS (nested dissection) | Best for very large | Graph partitioning |
| COLAMD (column AMD) | Good for non-square | Used with QR |

```cpp
// In Ceres:
options.linear_solver_ordering_type = ceres::AMD;   // default
options.linear_solver_ordering_type = ceres::NESDIS; // nested dissection
```

### The Schur Complement Trick (Bundle Adjustment)

In BA/SLAM, variables split into:
- **Poses** (few connections each) — "camera" variables
- **Landmarks** (connected to many poses) — "point" variables

The landmark block of $H$ is block-diagonal → can be eliminated cheaply:

$$H_{pp} - H_{pl} H_{ll}^{-1} H_{lp}) \Delta x_p = -g_p + H_{pl} H_{ll}^{-1} g_l$$

This **Schur complement** system is much smaller and denser.

**Ceres exploits this** with `ITERATIVE_SCHUR` and `SCHUR_JACOBI` preconditioners.

---

## 5. Convergence Analysis

### Convergence Rates

| Rate | Definition | Example |
|------|-----------|---------|
| Linear | $\|x_{k+1} - x^*\| \leq c \|x_k - x^*\|$, $c < 1$ | Gradient descent |
| Superlinear | $\|x_{k+1} - x^*\| / \|x_k - x^*\| \to 0$ | BFGS, LM |
| Quadratic | $\|x_{k+1} - x^*\| \leq c \|x_k - x^*\|^2$ | Newton |

### Convergence Diagnostics

What to monitor during optimization:

```python
for k in range(max_iter):
    # Compute step
    ...
    
    # Track these quantities:
    cost_k = 0.5 * np.sum(r**2)
    grad_norm = np.linalg.norm(J.T @ r)
    step_norm = np.linalg.norm(dx)
    
    # Convergence criteria (Ceres uses all three):
    # 1. Function tolerance: relative change in cost
    if abs(cost_k - cost_prev) / cost_prev < func_tol:
        break  # Cost barely changing
    
    # 2. Gradient tolerance: near-zero gradient
    if grad_norm < grad_tol:
        break  # At a stationary point
    
    # 3. Parameter tolerance: step is tiny
    if step_norm < param_tol * np.linalg.norm(x):
        break  # Parameters barely moving
```

### Common Convergence Failures

| Symptom | Cause | Fix |
|---------|-------|-----|
| Cost oscillates | Step too large / trust region issue | Reduce initial trust region, use LM |
| Cost decreases but never converges | Very flat minimum | Relax `function_tolerance` |
| Cost stalls early | Stuck at saddle / local min | Better initialization, multiple starts |
| Very slow decrease | Ill-conditioning | Precondition, rescale variables |
| NaN/Inf in cost or gradient | Numerical issue in residual | Check residual function for edge cases |

---

## 6. Scaling and Preconditioning

### Variable Scaling

If variables have very different magnitudes ($x_1 \approx 10^6$, $x_2 \approx 10^{-3}$), the Hessian is ill-conditioned.

**Fix:** Scale variables so they're all $O(1)$:

$$\tilde{x} = Dx \quad \text{where } D = \text{diag}(\text{typical values})$$

```python
# Before optimization, scale:
x_scale = np.array([1e6, 1e-3, 1.0])  # typical magnitudes
x_scaled = x0 / x_scale

# Optimize in scaled space
result = optimize(f_scaled, x_scaled)

# Unscale
x_final = result * x_scale
```

### Jacobian Scaling in Ceres

```cpp
// Ceres can auto-scale the Jacobian columns:
options.jacobi_scaling = true;  // Normalizes each column of J
```

This is equivalent to a diagonal preconditioner on the normal equations.

### Preconditioning for CG

For iterative solving of $H\Delta x = -g$:

Solve $M^{-1}H \Delta x = -M^{-1}g$ where $M \approx H$ but cheap to invert.

| Preconditioner | Cost | Quality |
|---------------|------|---------|
| Jacobi (diagonal of $H$) | $O(n)$ | Weak but cheap |
| Block Jacobi | $O(nb^3)$ ($b$ = block size) | Medium |
| Incomplete Cholesky | $O(\text{nnz})$ | Good |
| Schur-Jacobi (for BA) | Problem-dependent | Excellent for SLAM |

---

## 7. Automatic Differentiation: Implementation Details

### Forward Mode (What Ceres Uses)

A **dual number**: $a + b\epsilon$ where $\epsilon^2 = 0$.

For $n$ parameters, Ceres uses $\text{Jet}<T, N>$:

```cpp
// Ceres Jet: value + N partial derivatives
template <typename T, int N>
struct Jet {
    T a;           // function value
    T v[N];        // partial derivatives ∂f/∂x_i
};

// Arithmetic automatically propagates derivatives:
// (a + v·ε) * (b + w·ε) = ab + (aw + bv)·ε
```

**Cost:** Each operation costs ~$(N+1)$ floating-point ops instead of 1.

For a cost function with $n$ parameters and $m$ residuals:
- Forward AD: $O(mn)$ — evaluate $m$ residuals, each tracking $n$ derivatives
- This is the same complexity as analytic Jacobians

### Reverse Mode (What PyTorch/JAX Use)

For $f : \mathbb{R}^n \to \mathbb{R}$ (scalar output):
- Forward mode: $n$ passes to get full gradient
- **Reverse mode: 1 pass** (backpropagation!)

For $f : \mathbb{R}^n \to \mathbb{R}^m$:
- Forward is better when $n < m$ (few params, many residuals)
- Reverse is better when $n > m$ (many params, few outputs)

**Ceres uses forward mode** because NLLS has $m \gg n$ (many residuals, few parameters per block).

---

## 8. Numerical Differentiation (When You Must)

When the residual function is a black box (Fortran code, hardware-in-the-loop):

### Central Differences

$$\frac{\partial f}{\partial x_i} \approx \frac{f(x + h e_i) - f(x - h e_i)}{2h}$$

- Accuracy: $O(h^2)$ — much better than forward differences $O(h)$
- Cost: $2n$ evaluations for the full gradient
- Optimal $h \approx \epsilon_{\text{mach}}^{1/3} \approx 6 \times 10^{-6}$ for central

### In Ceres

```cpp
// When you can't template the cost function:
ceres::CostFunction* cost =
    new ceres::NumericDiffCostFunction<MyCostFunctor,
        ceres::CENTRAL,  // FORWARD, CENTRAL, or RIDDERS
        1,               // num residuals
        3>(              // num params
            new MyCostFunctor());
```

**Ridders' method** is the most accurate (extrapolates multiple step sizes) but 4× more expensive than central.

---

## 9. Common Numerical Pitfalls in OKS

| Pitfall | Where | Symptom | Fix |
|---------|-------|---------|-----|
| $\tan(\theta)$ at $\pm\pi/2$ | Sensorbar model | NaN in residual | Use $\sin/\cos$ parameterization |
| Division by small distance | Distance-based costs | Huge gradients | Add epsilon: $d + \epsilon$ |
| Angle wrapping | Pose error computation | 2π jumps in residual | Use `atan2`, normalize to $[-\pi, \pi]$ |
| Quaternion normalization | 3D rotation | Drift off unit sphere | Use manifold, or renormalize each step |
| Covariance matrix inversion | Information matrix from EKF | Singular $P$ → crash | Use pseudo-inverse, or regularize $P + \epsilon I$ |
| Log of negative number | In robust loss computation | NaN propagation | Guard with `max(val, epsilon)` |

---

## 10. Cheat Sheet: Solver Selection

```
PROBLEM TYPE?
│
├── Nonlinear Least Squares (sum of squares)
│   ├── n < 100, dense → Ceres DENSE_QR
│   ├── n < 1000, dense → Ceres DENSE_NORMAL_CHOLESKY
│   ├── Large, sparse (not SLAM) → Ceres SPARSE_NORMAL_CHOLESKY
│   └── SLAM / BA structure → Ceres ITERATIVE_SCHUR or g2o/GTSAM
│
├── Convex (LP, QP, SOCP, SDP)
│   ├── LP → HiGHS, Gurobi
│   ├── QP → OSQP (MPC), Gurobi
│   ├── SOCP → ECOS
│   └── SDP → SCS, MOSEK
│
├── General NLP (nonlinear + constraints)
│   ├── Medium → SQP (SNOPT, fmincon)
│   └── Large → Interior-point (IPOPT + CasADi)
│
└── Real-time / embedded
    ├── QP for MPC → qpOASES, OSQP, FORCES
    └── Simple NLLS → hand-coded LM
```

---

## Summary: Key Takeaways

1. **Condition number is king** — check $\kappa(J^TJ)$ before blaming the algorithm
2. **Never form normal equations** for ill-conditioned problems — use QR or Ceres' built-in solvers
3. **Sparsity matters** — exploit it or pay $O(n^3)$ instead of $O(n)$
4. **Scale your variables** — all should be $O(1)$
5. **Auto-diff > numeric diff** — exact, faster, less error-prone
6. **Convergence diagnostics** — monitor cost, gradient norm, and step size
7. **Guard against numerical edge cases** — atan2, epsilon denominators, manifold constraints
