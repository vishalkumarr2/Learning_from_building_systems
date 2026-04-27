# 01 — Mathematical Foundations for Optimization

> **Goal:** Build the calculus and linear algebra vocabulary needed to read any optimizer's cost function, gradient, and convergence proof.

---

## 1. Vectors, Norms, and Inner Products

### Vector Norms

A norm $\| \cdot \|$ measures the "size" of a vector.

| Norm | Formula | Geometric meaning |
|------|---------|-------------------|
| $\ell_1$ (Manhattan) | $\|x\|_1 = \sum_i \|x_i\|$ | Diamond in 2D |
| $\ell_2$ (Euclidean) | $\|x\|_2 = \sqrt{\sum_i x_i^2}$ | Circle in 2D |
| $\ell_\infty$ (Chebyshev) | $\|x\|_\infty = \max_i \|x_i\|$ | Square in 2D |

**Why it matters:** The choice of norm changes what "closest" means. In robust estimation (Ceres), switching from $\ell_2$ to $\ell_1$-like norms (Huber/Cauchy loss) reduces sensitivity to outliers.

### Inner Product and Angle

$$\langle x, y \rangle = x^T y = \|x\| \|y\| \cos\theta$$

Two vectors are **orthogonal** when $x^T y = 0$. Orthogonality shows up everywhere:
- Gradient is orthogonal to level sets
- Residual is orthogonal to column space in least-squares
- Conjugate directions in CG solver

---

## 2. Gradient — The Direction of Steepest Ascent

For a scalar function $f : \mathbb{R}^n \to \mathbb{R}$, the **gradient** is:

$$\nabla f(x) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

**Key properties:**
1. $\nabla f(x)$ points in the direction of steepest **ascent** of $f$
2. $-\nabla f(x)$ points in the direction of steepest **descent** — this is why gradient descent works
3. $\nabla f(x) = 0$ at any local minimum, maximum, or saddle point (stationary point)

### Example: Quadratic Cost

$$f(x) = \frac{1}{2} x^T A x - b^T x$$

$$\nabla f(x) = Ax - b$$

Setting $\nabla f = 0$ gives $Ax = b$ — this is the normal equation for least-squares!

### Jacobian (Vector-Valued Functions)

For $f : \mathbb{R}^n \to \mathbb{R}^m$, the **Jacobian** $J \in \mathbb{R}^{m \times n}$:

$$J_{ij} = \frac{\partial f_i}{\partial x_j}$$

In Ceres: `CostFunction::Evaluate()` returns both residuals $f(x)$ and Jacobians $J$.

---

## 3. Hessian — Curvature Information

The **Hessian** is the matrix of second derivatives:

$$H_{ij} = \nabla^2 f(x) = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

**Key properties:**
- $H$ is symmetric (assuming $f$ is twice continuously differentiable)
- Eigenvalues of $H$ describe curvature in each direction
- $H \succ 0$ (positive definite) ⟹ the point is a strict local minimum

### Curvature Interpretation

| Eigenvalues of $H$ | Shape at the point | Optimization implication |
|---------------------|---------------------|--------------------------|
| All positive | Bowl (minimum) | Converged — stop here |
| All negative | Cap (maximum) | Ascent point — run away |
| Mixed signs | Saddle point | Not a minimum, keep going |
| Some near zero | Flat valley | Slow convergence, ill-conditioned |

**OKS relevance:** When the sensorbar has too few measurements, the Hessian near the solution has near-zero eigenvalues → the solver oscillates or converges slowly.

---

## 4. Taylor Expansion — The Optimizer's Workhorse

Every optimization method is built on Taylor approximation:

### First-order (linear model):
$$f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x$$

This is what **gradient descent** uses.

### Second-order (quadratic model):
$$f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x + \frac{1}{2} \Delta x^T H \Delta x$$

This is what **Newton's method** uses. Setting the gradient of this quadratic to zero:

$$H \Delta x = -\nabla f(x)$$

This is the **Newton step**: $\Delta x = -H^{-1} \nabla f(x)$

---

## 5. Convexity — The Property That Makes Optimization Easy

### Definition

A function $f$ is **convex** if for all $x, y$ and $\lambda \in [0,1]$:

$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda) f(y)$$

Geometrically: the line segment between any two points on the graph lies **above** the function.

### Equivalent Conditions

| Condition | Requires |
|-----------|----------|
| $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda) f(y)$ | $f$ only |
| $f(y) \geq f(x) + \nabla f(x)^T (y-x)$ | First derivative |
| $\nabla^2 f(x) \succeq 0$ (PSD Hessian) | Second derivative |

### Why Convexity Matters

For convex functions:
- **Every local minimum is a global minimum** — no getting stuck in bad local optima
- Gradient descent converges to the optimum (given appropriate step size)
- Strong duality holds → can solve dual problem instead
- Efficient algorithms exist with polynomial-time guarantees

### Common Convex Functions

| Function | Domain | Convex? | OKS use |
|----------|--------|---------|---------|
| $\|x\|_2^2$ | $\mathbb{R}^n$ | ✅ Strongly convex | Least-squares cost |
| $\|x\|_1$ | $\mathbb{R}^n$ | ✅ Convex (not strongly) | L1 regularization |
| $\max(x_1, \ldots, x_n)$ | $\mathbb{R}^n$ | ✅ Convex | Minimax problems |
| $x \log x$ | $\mathbb{R}_{++}$ | ✅ Convex | Entropy, KL divergence |
| $e^x$ | $\mathbb{R}$ | ✅ Convex | — |
| $\sin(x)$ | $\mathbb{R}$ | ❌ Non-convex | — |

### Composition Rules (Preserving Convexity)

- Sum of convex functions → convex
- $\max$ of convex functions → convex
- Affine transformation: $f(Ax + b)$ is convex if $f$ is convex
- Non-negative weighted sum: $\sum \alpha_i f_i(x)$, $\alpha_i \geq 0$ → convex

**OKS relevance:** The sum of squared residuals $\sum_i r_i(x)^2$ is convex in $x$ when each $r_i$ is affine. When $r_i$ is nonlinear (sensorbar model), the problem is **non-convex** — which is why Ceres needs good initial guesses and LM damping.

---

## 6. Optimality Conditions

### Unconstrained

| Condition | Statement | What it tells you |
|-----------|-----------|-------------------|
| First-order necessary | $\nabla f(x^*) = 0$ | Stationary point (could be min, max, or saddle) |
| Second-order necessary | $\nabla^2 f(x^*) \succeq 0$ | Not a maximum (could still be saddle) |
| Second-order sufficient | $\nabla f(x^*) = 0$ and $\nabla^2 f(x^*) \succ 0$ | Strict local minimum ✅ |

### Constrained (Preview — detailed in 04)

For $\min f(x)$ subject to $g_i(x) \leq 0$ and $h_j(x) = 0$:

**KKT conditions** (first-order necessary, given constraint qualification):
1. **Stationarity:** $\nabla f(x^*) + \sum_i \lambda_i \nabla g_i(x^*) + \sum_j \nu_j \nabla h_j(x^*) = 0$
2. **Primal feasibility:** $g_i(x^*) \leq 0$, $h_j(x^*) = 0$
3. **Dual feasibility:** $\lambda_i \geq 0$
4. **Complementary slackness:** $\lambda_i g_i(x^*) = 0$

---

## 7. Positive Definite Matrices — The Optimizer's Best Friend

A symmetric matrix $A$ is **positive definite** ($A \succ 0$) if:

$$x^T A x > 0 \quad \forall x \neq 0$$

Equivalent conditions:
- All eigenvalues are positive
- All leading minors are positive (Sylvester's criterion)
- $A = L L^T$ for some lower-triangular $L$ with positive diagonal (Cholesky decomposition)

**Why this matters for optimization:**
- The Hessian $H \succ 0$ guarantees a local minimum
- Newton step $\Delta x = -H^{-1} g$ is a descent direction when $H \succ 0$
- In Gauss-Newton, $J^T J$ is always PSD (positive semi-definite) — adding $\lambda I$ in LM makes it PD
- Cholesky factorization is the fastest way to solve $H \Delta x = -g$ when $H \succ 0$

---

## 8. Matrix Calculus Cheat Sheet

These identities appear constantly in optimization derivations:

| Expression | Result |
|------------|--------|
| $\nabla_x (a^T x)$ | $a$ |
| $\nabla_x (x^T A x)$ | $(A + A^T) x$ (or $2Ax$ if $A$ symmetric) |
| $\nabla_x \|Ax - b\|^2$ | $2A^T(Ax - b)$ |
| $\nabla_x \text{tr}(AXB)$ | $A^T B^T$ |
| $\frac{\partial}{\partial X} \log \det(X)$ | $X^{-T}$ |

The third one — $\nabla_x \|Ax - b\|^2 = 2A^T(Ax - b)$ — is the gradient of the least-squares cost. Setting it to zero gives the **normal equations**: $A^T A x = A^T b$.

---

## Summary: What to Carry Forward

| Concept | Where it's used next |
|---------|---------------------|
| Gradient = steepest ascent | 02-unconstrained (gradient descent) |
| Hessian = curvature | 02-unconstrained (Newton's method) |
| Taylor quadratic model | 02-unconstrained (trust region), 03-least-squares (Gauss-Newton) |
| Convexity → global minimum | 05-convex-optimization (all of it) |
| PD matrices → Cholesky | 03-least-squares (solving normal equations), 07-numerical |
| Normal equations | 03-least-squares (the starting point) |
| KKT conditions | 04-constrained (the entire chapter) |
