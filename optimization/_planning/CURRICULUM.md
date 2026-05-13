# Optimization Mastery: Zero to Expert in 16 Weeks

### 2.5 hours/day × 7 days/week × 16 weeks = ~280 hours
### From "what is a gradient?" to "I can debug a production SLAM backend and design multi-robot coordination"

---

## Design Philosophy

1. **Every concept earns its place through code** — no topic without an implementation exercise
2. **Python first, C++ when it matters** — prototype in NumPy/SciPy, productionize in Ceres/g2o/OSQP
3. **Robotics grounding throughout** — every week ties back to real robot problems (calibration, navigation, SLAM, MPC)
4. **Spaced repetition** — hard topics revisited across multiple weeks with increasing depth
5. **Daily structure**: ~45 min theory → ~60 min implementation → ~45 min exercises/review
6. **Phase gates** — self-assessment quiz before each new phase to verify prerequisites
7. **Manifold awareness** — Lie groups taught explicitly before SLAM and calibration need them

---

## Expert Review Applied (10-Reviewer Panel)

This curriculum was reviewed by a panel of 10 specialists. Key improvements incorporated:

| Priority | Change | Location |
|----------|--------|----------|
| **CRITICAL** | Lie groups & manifold optimization — dedicated day | Day 41 (was Ceres Advanced) |
| **HIGH** | Fisher Information, covariance estimation, observability | Day 35 (expanded) |
| **HIGH** | Schur complement, variable ordering, marginalization | Day 9 (expanded) |
| **MEDIUM** | MPC recursive feasibility & stability guarantees | Day 53 (expanded) |
| **MEDIUM** | Temporal calibration | Day 64 (expanded) |
| **MEDIUM** | Phase-gate self-assessments | Start of each Phase |
| **MEDIUM** | Proximal operators (ISTA/FISTA) | Day 25 (expanded) |
| **LOW** | SOCP/SDP exercises | Day 50 (added) |
| **LOW** | Unit testing numerical code (pytest) | Day 28 (added) |
| **LOW** | SGD/Adam marked as optional enrichment | Day 24 (reframed) |
| **LOW** | Distributed optimization awareness | Day 68 (added) |

---

## Curriculum Overview

| Phase | Weeks | Topic | Hours | Outcome |
|-------|-------|-------|-------|---------|
| **I** | 1–2 | Mathematical Foundations | 35 | Fluent in gradients, Hessians, convexity, linear algebra for optimization |
| **II** | 3–4 | Unconstrained Optimization | 35 | Can implement GD, Newton, BFGS, trust region from scratch |
| **III** | 5–6 | Nonlinear Least Squares + Ceres | 35 | Can formulate and solve NLS problems in Python and Ceres C++ |
| **IV** | 7–8 | Constrained + Convex Optimization | 35 | Can formulate QP/LP, understand KKT, use CVXPY and OSQP |
| **V** | 9–10 | Robotics Applications (SLAM, MPC, Calibration) | 35 | Can build pose-graph SLAM, MPC controller, calibration pipeline |
| **VI** | 11–13 | Lie Groups & Manifold Optimization | 52.5 | SO(3)/SE(3) exp/log, manifold GN, pose-graph SLAM, IMU preintegration, mini VIO |
| **VII** | 14–16 | Game Theory for Optimization & Robotics | 52.5 | Nash equilibria, zero-sum LP, cooperative games, mechanism design, no-regret learning, congestion games |

---

## Tools & Setup (Do Before Day 1)

### Python Stack
```bash
python3 -m venv optim-env
source optim-env/bin/activate
pip install numpy scipy matplotlib sympy autograd jax jaxlib
pip install cvxpy osqp quadprog
pip install jupyter notebook
```

### C++ Stack (Weeks 5+)
```bash
# Ceres Solver
sudo apt install libceres-dev

# g2o (for SLAM)
sudo apt install libg2o-dev

# Eigen (already a Ceres dep)
sudo apt install libeigen3-dev

# Build tools
sudo apt install cmake build-essential
```

### Textbooks (Free Online)
| Book | Use For | Link |
|------|---------|------|
| Boyd & Vandenberghe — *Convex Optimization* | Weeks 7-8, reference throughout | [web.stanford.edu/~boyd/cvxbook](https://web.stanford.edu/~boyd/cvxbook/) |
| Nocedal & Wright — *Numerical Optimization* | Weeks 3-6, algorithm internals | Springer (library access) |
| Grisetti et al. — *A Tutorial on Graph-Based SLAM* | Week 9 | IEEE paper (free preprint) |
| Ceres Solver Tutorial | Week 5-6 | [ceres-solver.org/tutorial.html](http://ceres-solver.org/tutorial.html) |

---

# PHASE I: MATHEMATICAL FOUNDATIONS (Weeks 1–2)

> **Goal:** Fluent in the math language that every optimization paper and solver uses.
> If you skip this, everything after Week 3 will feel like magic.

> **🏎️ FAST-TRACK OPTION:** If you have a strong linear algebra and calculus background,
> take this 10-question prerequisites quiz. Score ≥ 8/10 → skip directly to Day 7 (Week 1 Review).
>
> 1. Compute eigenvalues of $\begin{bmatrix} 3 & 1 \\ 1 & 3 \end{bmatrix}$ and interpret geometrically
> 2. When is a symmetric matrix positive definite? Give 3 equivalent conditions.
> 3. Compute $\nabla f$ and $\nabla^2 f$ for $f(x,y) = x^2 e^y + \sin(xy)$
> 4. State the chain rule for $f(g(x))$ where $f: \mathbb{R}^m \to \mathbb{R}$, $g: \mathbb{R}^n \to \mathbb{R}^m$
> 5. What does the SVD $A = U \Sigma V^T$ tell you about the rank of $A$?
> 6. What is the condition number of a matrix and why does it matter for solving $Ax = b$?
> 7. Compute the Taylor expansion of $e^x$ around $x=0$ to 3rd order
> 8. What is the difference between a local minimum and a global minimum? Give a condition for each.
> 9. What is a convex function? Give 2 equivalent definitions.
> 10. For a quadratic $f(x) = \frac{1}{2}x^T A x - b^T x$, what is the minimizer?

---

## Week 1 — Linear Algebra + Calculus for Optimization

### Day 1: Vectors, Matrices, Transformations (2.5 hrs)
**Theory (45 min):**
- Vector spaces, basis, span, linear independence
- Matrix as linear transformation
- Column space, null space, rank
- Orthogonality and projection

**Implementation (60 min):**
```python
# Exercise: Implement from scratch (no numpy.linalg)
# 1. Gram-Schmidt orthogonalization
# 2. QR decomposition via Gram-Schmidt
# 3. Projection of vector onto subspace
```

**Practice (45 min):**
- 10 hand-computation problems: rank, null space, column space
- Verify each answer with `numpy.linalg`

**Reading:** Gilbert Strang — *Introduction to Linear Algebra* Ch 1-3 (skim if familiar)

---

### Day 2: Eigenvalues, Positive Definiteness, SVD (2.5 hrs)
**Theory (45 min):**
- Eigenvalue decomposition: $A = Q \Lambda Q^T$ for symmetric matrices
- Positive definite / semidefinite / indefinite classification
- SVD: $A = U \Sigma V^T$ — geometric interpretation
- Condition number: $\kappa(A) = \sigma_{\max} / \sigma_{\min}$

**Implementation (60 min):**
```python
# Exercise:
# 1. Power iteration for dominant eigenvalue
# 2. Check positive definiteness via eigenvalues AND Cholesky attempt
# 3. Compute SVD, reconstruct matrix, verify low-rank approximation
# 4. Compute condition number, show how it predicts numerical error
```

**Practice (45 min):**
- Classify 10 matrices as PD/PSD/indefinite
- For each: verify via eigenvalues, determinant test, and Cholesky

**Key insight:** The Hessian of a function at a minimum is positive definite. This is *the* bridge from linear algebra to optimization.

---

### Day 3: Multivariable Calculus — Gradients and Jacobians (2.5 hrs)
**Theory (45 min):**
- Partial derivatives, gradient vector $\nabla f$
- Jacobian matrix of vector-valued functions
- Chain rule for composed functions: $\frac{\partial}{\partial x} f(g(x)) = J_f \cdot J_g$
- Directional derivative: $D_v f = \nabla f \cdot v$

**Implementation (60 min):**
```python
# Exercise:
# 1. Compute gradient of f(x,y) = x²y + sin(xy) by hand, verify with sympy
# 2. Compute Jacobian of f(x,y,z) = [x²+y, yz, xz²] by hand, verify with sympy
# 3. Implement numerical gradient via finite differences:
#    grad_f ≈ [f(x+h*e_i) - f(x-h*e_i)] / 2h  for each dimension
# 4. Compare numerical vs analytic gradient — measure error vs step size h
```

**Practice (45 min):**
- 8 gradient computations by hand (polynomial, trig, exponential, composed)
- Gradient of Rosenbrock function: $f(x,y) = (1-x)^2 + 100(y-x^2)^2$

---

### Day 4: Hessians and Taylor Expansion (2.5 hrs)
**Theory (45 min):**
- Hessian matrix: $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$
- Second-order Taylor expansion: $f(x+\delta) \approx f(x) + \nabla f^T \delta + \frac{1}{2} \delta^T H \delta$
- Quadratic model interpretation
- Connection: minimizing quadratic model → Newton step: $\delta^* = -H^{-1} \nabla f$

**Implementation (60 min):**
```python
# Exercise:
# 1. Compute Hessian of Rosenbrock function by hand
# 2. Implement numerical Hessian via finite differences of gradients
# 3. Verify: at the minimum (1,1), Hessian should be PD
# 4. Plot: actual function vs quadratic Taylor approximation at several points
# 5. Show: better approximation near expansion point, worse far away
```

**Practice (45 min):**
- 6 Hessian computations by hand
- For each: classify as PD/PSD/indefinite → is the point a min/max/saddle?

---

### Day 5: Convexity — The Central Concept (2.5 hrs)
**Theory (45 min):**
- Convex set: $\lambda x + (1-\lambda)y \in S$ for $\lambda \in [0,1]$
- Convex function: $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda) f(y)$
- First-order condition: $f(y) \geq f(x) + \nabla f(x)^T (y-x)$
- Second-order condition: $H \succeq 0$ everywhere
- **Key theorem:** For convex functions, every local minimum is a global minimum

**Implementation (60 min):**
```python
# Exercise:
# 1. Classify functions as convex/concave/neither:
#    x², log(x), x³, ||x||, max(x₁,x₂), x₁x₂
# 2. For each: verify numerically via second-order condition (eigenvalues of Hessian)
# 3. Visualize: convex function with tangent planes (first-order condition)
# 4. Show: GD on convex function always converges to global min
#          GD on non-convex function gets stuck at local min
```

**Practice (45 min):**
- 12 convexity classification problems
- Operations that preserve convexity: sum, max, composition with affine

---

### Day 6: Optimality Conditions (2.5 hrs)
**Theory (45 min):**
- Necessary condition (first-order): $\nabla f(x^*) = 0$
- Sufficient condition (second-order): $\nabla f(x^*) = 0$ AND $H(x^*) \succ 0$
- Saddle points: $\nabla f = 0$ but $H$ indefinite
- Why convexity simplifies everything: first-order sufficient

**Implementation (60 min):**
```python
# Exercise:
# 1. Find all critical points of f(x,y) = x⁴ + y⁴ - 2x² - 2y²
#    Classify each as min/max/saddle using Hessian eigenvalues
# 2. Implement critical point finder:
#    - Start from random points, run Newton's method on ∇f = 0
#    - Classify each found point
#    - Visualize on contour plot
# 3. Monkey saddle: f(x,y) = x³ - 3xy² — show Hessian test is inconclusive at origin
```

**Practice (45 min):**
- 8 problems: find and classify critical points

---

### Day 7: Week 1 Review + Comprehensive Exercise Set (2.5 hrs)
**Review (45 min):**
- Flashcard drill: definitions, theorems, key formulas
- Fill-in-the-gaps quiz covering all 6 days

**Comprehensive Exercise (105 min):**
```python
# EXERCISE SET 1: Foundations Comprehensive (10 problems)
#
# 1. Given A (4×3), find rank, null space, column space
# 2. Prove: if A is PSD, then x^T A x ≥ 0 for all x
# 3. Compute gradient and Hessian of f(x) = x^T A x + b^T x + c
# 4. Show: Hessian of least-squares ||Ax-b||² is 2A^TA (always PSD)
# 5. Taylor-expand f(x,y) = e^(x+2y) around (0,0) to second order
# 6. Classify convexity: f(x) = log(1 + e^x) — "softplus"
# 7. Find all critical points of Himmelblau's function
# 8. Condition number: solve Ax=b with well-conditioned and ill-conditioned A
#    Show how relative error amplifies with κ(A)
# 9. Implement: gradient checker that compares analytic vs numerical gradient
#    and reports max relative error
# 10. Rosenbrock function: plot contours, gradient field, and Hessian eigenvectors
```

---

## Week 2 — Advanced Foundations + Intro to Iterative Methods

### Day 8: Matrix Decompositions for Optimization (2.5 hrs)
**Theory (45 min):**
- Cholesky: $A = LL^T$ for PD matrices — fastest way to solve $Ax = b$
- LDL^T: for PSD matrices
- QR decomposition for least squares
- When to use what: Cholesky (PD systems), QR (rectangular), SVD (rank-deficient)

**Implementation (60 min):**
```python
# Exercise:
# 1. Implement Cholesky decomposition (without numpy.linalg.cholesky)
# 2. Solve Ax=b using your Cholesky: forward sub on Ly=b, back sub on L^T x=y
# 3. Benchmark: Cholesky solve vs numpy.linalg.solve vs numpy.linalg.inv @ b
# 4. Show: Cholesky fails on non-PD matrix (as expected)
```

**Practice (45 min):**
- Time complexity analysis: which decomposition for which problem shape?

---

### Day 9: Sparse Matrices & Schur Complement (2.5 hrs)
**Theory (45 min):**
- Sparse vs dense: memory and computation
- Storage formats: COO, CSR, CSC
- Sparse Cholesky: fill-in, ordering strategies (AMD, METIS)
- Why SLAM matrices are sparse (only neighboring poses interact)
- **Schur complement elimination**: partition variables → eliminate subset → dense reduced system
  - In BA: eliminate landmarks → camera-only system (why Ceres has `SPARSE_SCHUR`)
  - In SLAM: marginalize old poses → keep recent window
- **Variable ordering**: bad ordering → fill-in explosion; AMD/METIS find good orderings
- **Marginalization** (preview): remove variables while preserving their information
  - Sparsity-preserving vs dense marginalization (the "fill-in" problem)

**Implementation (60 min):**
```python
# Exercise:
# 1. Create a tridiagonal system (like 1D pose graph) — 1000×1000
# 2. Solve dense vs scipy.sparse — compare time and memory
# 3. Visualize sparsity pattern with plt.spy()
# 4. Show: banded structure → O(n) solve vs O(n³) dense
```

**Practice (45 min):**
- Create sparsity patterns for: tridiagonal, block-diagonal, pose-graph, banded

---

### Day 10: Numerical Differentiation Deep Dive (2.5 hrs)
**Theory (45 min):**
- Forward difference: $\frac{f(x+h) - f(x)}{h}$ — $O(h)$ error
- Central difference: $\frac{f(x+h) - f(x-h)}{2h}$ — $O(h^2)$ error
- Complex-step derivative: $\frac{\text{Im}(f(x+ih))}{h}$ — no cancellation error!
- Truncation vs rounding error tradeoff — optimal $h \approx \epsilon^{1/2}$ (forward), $\epsilon^{1/3}$ (central)

**Implementation (60 min):**
```python
# Exercise:
# 1. Plot error vs h for forward, central, and complex-step derivatives of sin(x)
#    Show the V-shaped error curve (truncation vs rounding)
# 2. Find optimal h for each method experimentally
# 3. Implement complex-step Jacobian for vector functions
# 4. Compare all three methods on a function with very small gradient
#    (where rounding error dominates)
```

**Practice (45 min):**
- When would you use numerical derivatives vs analytic? vs automatic differentiation?
- Real-world scenario: checking a hand-derived Jacobian

---

### Day 11: Automatic Differentiation (2.5 hrs)
**Theory (45 min):**
- Forward mode AD: dual numbers $a + b\epsilon$, $\epsilon^2 = 0$
- Reverse mode AD: computation graph, backpropagation
- Forward vs reverse: when each is efficient (Jacobian shape)
- AD in optimization: Ceres AutoDiff, JAX, PyTorch autograd

**Implementation (60 min):**
```python
# Exercise:
# 1. Implement dual number class with +, *, sin, cos, exp
# 2. Use it to compute gradient of f(x,y) = sin(x*y) + x²
# 3. Compare: your dual numbers vs JAX grad vs numerical finite diff
# 4. Time comparison on a function with 100 parameters:
#    numerical (O(100) evaluations) vs AD (O(1) pass)
```
```python
import jax
import jax.numpy as jnp

# JAX automatic differentiation:
def f(x):
    return jnp.sin(x[0] * x[1]) + x[0]**2

grad_f = jax.grad(f)
hess_f = jax.hessian(f)
```

**Practice (45 min):**
- Derive forward-mode AD for a 3-node computation graph by hand
- Map: which optimization tool uses which AD type?

---

### Day 12: Function Landscapes and Conditioning (2.5 hrs)
**Theory (45 min):**
- Condition number of optimization: $\kappa(H)$ = ratio of eigenvalues
- Ill-conditioned = elongated contours = slow convergence
- Preconditioning: transform variables to improve conditioning
- Scaling: why normalizing variables matters in practice

**Implementation (60 min):**
```python
# Exercise:
# 1. Create two quadratics: well-conditioned (κ=2) and ill-conditioned (κ=1000)
# 2. Run gradient descent on both — count iterations to convergence
# 3. Visualize: overlay GD trajectory on contour plots
# 4. Apply diagonal preconditioning — show dramatic speedup on ill-conditioned
# 5. Real example: least-squares where columns of A have very different scales
#    Show: without scaling, solver is 100× slower
```

**Practice (45 min):**
- Given a Hessian, predict GD convergence rate
- Design a preconditioner for a given problem

---

### Day 13: Iterative Methods for Linear Systems (2.5 hrs)
**Theory (45 min):**
- Why iterative? Dense direct solvers are $O(n^3)$ — too slow for large systems
- Conjugate Gradient (CG) method for PD systems
- Convergence: CG converges in at most $n$ steps, much faster if $\kappa$ is small
- Preconditioned CG: $M^{-1}Ax = M^{-1}b$

**Implementation (60 min):**
```python
# Exercise:
# 1. Implement CG from scratch (the classic algorithm, ~15 lines)
# 2. Solve a 1000×1000 PD system: CG vs dense Cholesky — compare time
# 3. Plot: residual vs iteration for well-conditioned vs ill-conditioned
# 4. Implement Jacobi preconditioner, show CG convergence improvement
# 5. Real scenario: sparse 10000×10000 system from pose-graph structure
#    Dense solve: impossible. CG: seconds.
```

**Practice (45 min):**
- When to use CG vs Cholesky vs LU?
- Connection: CG is actually minimizing a quadratic — it IS optimization!

---

### Day 14: Week 2 Review + Comprehensive Exercise Set (2.5 hrs)
**Review (45 min):**
- Concept map: connect all Week 1-2 topics
- Identify: which foundation supports which optimization method

**Comprehensive Exercise (105 min):**
```python
# EXERCISE SET 2: Advanced Foundations (8 problems)
#
# 1. Implement a gradient checker:
#    Input: f(x), analytic_grad(x), x₀
#    Output: max relative error between analytic and numerical gradient
#    Use central differences with optimal h
#
# 2. Sparse tridiagonal system (n=5000):
#    Solve with: (a) dense LU, (b) sparse Cholesky, (c) CG
#    Report time and memory for each
#
# 3. Implement forward-mode AD for f: R^n → R using dual numbers
#    Test on: f(x) = ||Ax - b||² where A is 50×20
#
# 4. Condition number study:
#    Generate random PD matrices with κ = 1, 10, 100, 1000, 10000
#    For each: solve Ax=b, perturb b by 1e-8, measure solution change
#    Plot: relative error amplification vs κ
#
# 5. Implement preconditioned CG with incomplete Cholesky preconditioner
#
# 6. Rosenbrock (extended to n=50 dimensions):
#    Compute condition number of Hessian at (1,...,1)
#    Explain why GD is slow on this function
#
# 7. Complex-step derivative for a function involving matrix operations:
#    f(θ) = ||R(θ)p - q||² where R is a 2D rotation matrix
#
# 8. Build a numerical differentiation test suite that you'll reuse
#    throughout the course to verify every analytic derivative you write
```

---

# PHASE II: UNCONSTRAINED OPTIMIZATION (Weeks 3–4)

> **Goal:** Implement every major unconstrained optimizer from scratch, understand convergence guarantees, and know when to use which method.

---

## Week 3 — First-Order and Second-Order Methods

> **🛣️ PHASE II GATE — Self-Assessment (15 min before Day 15)**
> Answer these 10 questions. If you score < 7/10, review the relevant Phase I day.
> 1. Write the gradient of $f(x,y) = x^2 y + \sin(xy)$
> 2. For $f(x) = x^T A x$, what is the Hessian? When is $f$ convex?
> 3. What does $\kappa(A) = 10^6$ mean practically for solving $Ax = b$?
> 4. Name 3 differences between forward-mode and reverse-mode AD
> 5. Given a 1000×1000 sparse matrix with 5000 nonzeros, which format is best for matrix-vector multiply?
> 6. What is the normal equation for least squares? When does it fail?
> 7. Compute $\nabla f$ for $f = \|Ax - b\|^2$ in matrix notation
> 8. What is the Schur complement of a 2×2 block matrix?
> 9. How does `jax.grad` handle a function $f: \mathbb{R}^n \to \mathbb{R}$?
> 10. What is the condition for a critical point to be a local minimum?

### Day 15: Gradient Descent — The Workhorse (2.5 hrs)
**Theory (45 min):**
- Algorithm: $x_{k+1} = x_k - \alpha_k \nabla f(x_k)$
- Fixed step size: convergence condition $\alpha < 2/L$ where $L$ = Lipschitz constant of $\nabla f$
- Convergence rate: $O(1/k)$ for convex, $O(\rho^k)$ for strongly convex ($\rho = \frac{\kappa-1}{\kappa+1}$)
- Why GD is slow: zig-zagging on ill-conditioned problems

**Implementation (60 min):**
```python
# Exercise:
# 1. Implement GD with fixed step size
# 2. Implement GD with backtracking line search (Armijo condition)
# 3. Test on: (a) quadratic f(x) = x^T A x, (b) Rosenbrock, (c) log-sum-exp
# 4. Plot convergence: f(x_k) - f* vs iteration for each
# 5. Vary condition number: show GD iterations grow linearly with κ
```

**Practice (45 min):**
- Derive convergence rate for quadratic case by hand
- Predict: how many GD iterations for κ=100 to reach ε=1e-6?

---

### Day 16: Line Search Methods (2.5 hrs)
**Theory (45 min):**
- Exact line search: $\alpha^* = \arg\min_\alpha f(x - \alpha \nabla f)$
- Armijo (sufficient decrease) condition: $f(x + \alpha d) \leq f(x) + c_1 \alpha \nabla f^T d$
- Wolfe conditions: Armijo + curvature condition
- Backtracking: start large, shrink until Armijo holds
- Strong Wolfe conditions: guarantees for quasi-Newton methods

**Implementation (60 min):**
```python
# Exercise:
# 1. Implement backtracking line search
# 2. Implement Wolfe line search (zoom algorithm from Nocedal & Wright)
# 3. Compare: fixed step vs Armijo vs Wolfe on Rosenbrock
# 4. Pathological case: show where Armijo alone is insufficient
# 5. Visualize: 1D slice along search direction with Armijo/Wolfe zones marked
```

**Practice (45 min):**
- Prove: Armijo + backtracking always terminates (given descent direction)
- Parameter sensitivity: effect of c₁ (Armijo) and c₂ (Wolfe)

---

### Day 17: Newton's Method (2.5 hrs)
**Theory (45 min):**
- Newton step: $x_{k+1} = x_k - H^{-1} \nabla f$
- Derivation: minimize second-order Taylor model
- Quadratic convergence near solution: $\|x_{k+1} - x^*\| \leq C \|x_k - x^*\|^2$
- Problem: Hessian must be PD (otherwise step may ascend!)
- Problem: Hessian is $O(n^2)$ to store, $O(n^3)$ to invert

**Implementation (60 min):**
```python
# Exercise:
# 1. Implement Newton's method with analytic Hessian
# 2. Test on Rosenbrock: count iterations (should be ~10-20 vs GD's ~10000)
# 3. Show: quadratic convergence by plotting log(error) vs iteration
# 4. Failure case: start far from minimum where Hessian is indefinite
#    Show: pure Newton diverges, but damped Newton (with line search) works
# 5. Implement modified Newton: replace H with H + μI when H not PD
```

**Practice (45 min):**
- Derive: Newton's method on f(x) = ax² + bx + c converges in 1 step
- When does Newton fail? List 3 scenarios

---

### Day 18: Trust Region Methods (2.5 hrs)
**Theory (45 min):**
- Trust region idea: restrict step to $\|\delta\| \leq \Delta$ (radius)
- Subproblem: $\min_\delta m(\delta) = f + \nabla f^T \delta + \frac{1}{2}\delta^T H \delta$ s.t. $\|\delta\| \leq \Delta$
- Cauchy point: steepest descent restricted to trust region
- Dogleg method: interpolate between Cauchy and Newton steps
- Radius update: expand if good agreement, shrink if bad
- Ratio: $\rho = \frac{f(x_k) - f(x_k + \delta)}{m(0) - m(\delta)}$

**Implementation (60 min):**
```python
# Exercise:
# 1. Implement trust region with Cauchy point (simplest version)
# 2. Implement dogleg trust region method
# 3. Compare: trust region vs line search Newton on Rosenbrock
# 4. Pathological case: near saddle point — show trust region handles it better
# 5. Visualize: trust region circle on contour plot, show how radius adapts
```

**Practice (45 min):**
- Trust region vs line search: when to prefer which? (table of tradeoffs)
- Connection to Levenberg-Marquardt (preview for Week 5)

---

### Day 19: Quasi-Newton Methods — BFGS (2.5 hrs)
**Theory (45 min):**
- Problem: computing Hessian is expensive
- Idea: approximate Hessian from gradient differences
- Secant condition: $B_{k+1} s_k = y_k$ where $s_k = x_{k+1} - x_k$, $y_k = \nabla f_{k+1} - \nabla f_k$
- BFGS update formula (rank-2 update to $B$)
- Superlinear convergence: faster than GD, close to Newton, no Hessian needed
- L-BFGS: store only last $m$ vectors instead of full matrix

**Implementation (60 min):**
```python
# Exercise:
# 1. Implement BFGS with Wolfe line search
# 2. Implement L-BFGS (m=10) using two-loop recursion
# 3. Compare on Rosenbrock: GD vs Newton vs BFGS vs L-BFGS
#    Plot: function value vs (a) iterations, (b) gradient evaluations, (c) wall time
# 4. High-dimensional test (n=1000): show L-BFGS scales, Newton doesn't
# 5. Show: BFGS with Armijo (not Wolfe) can fail — demonstrate
```

**Practice (45 min):**
- Memory comparison: Newton O(n²), BFGS O(n²), L-BFGS O(mn)
- Derive: BFGS update maintains symmetry and positive definiteness

---

### Day 20: Conjugate Gradient for Optimization (2.5 hrs)
**Theory (45 min):**
- Nonlinear CG: Fletcher-Reeves, Polak-Ribière
- Relation to linear CG (from Day 13)
- CG as "momentum" for gradient descent
- Advantages: O(n) per step, no matrix storage
- Restart strategies

**Implementation (60 min):**
```python
# Exercise:
# 1. Implement nonlinear CG (Polak-Ribière variant)
# 2. Compare: GD vs CG vs L-BFGS on extended Rosenbrock (n=100)
# 3. Show: CG converges in n steps for n-dimensional quadratic
# 4. Implement automatic restart (when conjugacy is lost)
# 5. Real benchmark: compare all methods on a 500-dimensional problem
```

**Practice (45 min):**
- Method selection flowchart: given problem size and gradient cost, which method?

---

### Day 21: Week 3 Review + Optimizer Shootout (2.5 hrs)
**Review (45 min):**
- Fill in comparison table:

| Method | Per-iter cost | Memory | Convergence | Needs Hessian? |
|--------|--------------|--------|-------------|----------------|
| GD | O(n) | O(n) | Linear | No |
| Newton | O(n³) | O(n²) | Quadratic | Yes |
| BFGS | O(n²) | O(n²) | Superlinear | No |
| L-BFGS | O(mn) | O(mn) | Superlinear | No |
| CG | O(n) | O(n) | Variable | No |
| Trust Region | O(n³) | O(n²) | Quadratic | Yes |

**Comprehensive Exercise (105 min):**
```python
# OPTIMIZER SHOOTOUT
# Implement a benchmarking framework:
#
# Functions: Rosenbrock, Styblinski-Tang, Rastrigin, Ackley, Beale
# Methods: GD, Newton, BFGS, L-BFGS, CG, Trust-Region-Newton
# Metrics: iterations, gradient evals, function evals, wall time, final error
#
# Generate a report table and identify:
# 1. Which method wins on convex problems?
# 2. Which method wins on ill-conditioned problems?
# 3. Which method wins on high-dimensional problems (n=1000)?
# 4. Which method is most robust to bad starting points?
```

---

## Week 4 — Advanced Unconstrained + Convergence Theory

### Day 22: Convergence Theory Deep Dive (2.5 hrs)
**Theory (60 min):**
- Formal convergence definitions: linear, superlinear, quadratic
- $\|x_{k+1} - x^*\| \leq C \|x_k - x^*\|^p$
- Convergence rate of GD: depends on $\kappa$
- Convergence rate of Newton: quadratic (local)
- Global convergence: guarantees that iterates approach *a* critical point (not necessarily the minimum)
- Zoutendijk's theorem: sufficient decrease + GD-related → $\sum \|\nabla f_k\|^2 < \infty$

**Implementation (60 min):**
```python
# Exercise:
# 1. Empirically measure convergence rate:
#    Plot log(||x_k - x*||) vs k for each method
#    Fit the convergence order p
# 2. Show: Newton has p=2 (quadratic), BFGS has p≈1.6 (superlinear)
# 3. Verify Zoutendijk: plot cumulative sum of ||∇f||² for GD with Armijo
```

**Practice (30 min):**
- Prove: GD with exact line search on quadratic converges in n steps for n eigenvalues

---

### Day 23: Momentum Methods and Acceleration (2.5 hrs)
**Theory (45 min):**
- Heavy ball method: $x_{k+1} = x_k - \alpha \nabla f_k + \beta (x_k - x_{k-1})$
- Nesterov acceleration: $O(1/k^2)$ for convex (vs GD's $O(1/k)$)
- Optimal first-order methods: matching lower bounds
- Connection to control theory: momentum as PD controller

**Implementation (60 min):**
```python
# Exercise:
# 1. Implement heavy ball method
# 2. Implement Nesterov accelerated gradient
# 3. Compare: GD vs heavy ball vs Nesterov on ill-conditioned quadratic
# 4. Show: oscillation problem with heavy ball (too much momentum)
# 5. Nesterov on Rosenbrock: compare convergence curves
```

**Practice (45 min):**
- Derive optimal momentum parameter for quadratic case
- Robotics connection: estimation smoothing as optimization with momentum

---

### Day 24: Stochastic and Mini-Batch Methods (2.5 hrs) ⚡ ENRICHMENT
> **Note:** SGD/Adam are primarily ML tools. Robotics optimization rarely uses stochastic methods.
> Treat this as enrichment — skim if time-constrained, deep-dive if working on learning-based robotics.

**Theory (45 min):**
- SGD: replace ∇f with noisy estimate from subset of data
- Convergence: $O(1/\sqrt{k})$ — slower but each step is cheap
- Mini-batch: variance reduction
- Adam, AdaGrad, RMSProp (brief survey — ML focus, but robotics uses them for learning-based methods)
- When robotics does use stochastic methods: learned cost functions, neural network training for perception

**Implementation (60 min):**
```python
# Exercise:
# 1. Implement SGD for sum-of-functions: f(x) = Σ fᵢ(x)
# 2. Implement mini-batch GD with batch size parameter
# 3. Compare: GD vs SGD vs mini-batch on linear regression (n=10000 samples)
# 4. Learning rate schedules: constant, 1/k, cosine annealing
# 5. Implement Adam optimizer from scratch
```

**Practice (45 min):**
- When is SGD better than full-batch GD? (compute budget argument)
- Robotics application: online parameter estimation

---

### Day 25: Multi-Objective and Regularization (2.5 hrs)
**Theory (45 min):**
- Regularization as multi-objective: $\min f(x) + \lambda \|x\|^2$ (Tikhonov/Ridge)
- L1 regularization: sparsity, LASSO
- Pareto optimality (brief)
- Regularization path: how solution changes with λ

**Implementation (60 min):**
```python
# Exercise:
# 1. Ridge regression: solve closed-form and iteratively, compare
# 2. L1 proximal gradient (ISTA): implement for LASSO
# 3. Regularization path: plot solution components vs λ
# 4. Cross-validation to select λ
# 5. Robotics example: regularized sensor calibration (prevent overfitting to noise)
```

**Practice (45 min):**
- Connection: Tikhonov = Bayesian prior (Gaussian), L1 = Laplace prior
- When to regularize in robotics: calibration, map fitting, trajectory smoothing
- **Proximal operators** (foundation for modern solvers):
  - Proximal operator: $\text{prox}_{\lambda f}(v) = \arg\min_x [f(x) + \frac{1}{2\lambda}\|x - v\|^2]$
  - ISTA (Iterative Shrinkage-Thresholding): proximal gradient for L1
  - FISTA: accelerated proximal gradient (Nesterov acceleration)
  - Why this matters: OSQP and SCS use ADMM (splitting + proximal ops) internally

---

### Day 26: Global Optimization Methods (2.5 hrs)
**Theory (45 min):**
- Problem: local methods find local optima, not global
- Random restart: simple but effective
- Simulated annealing: probabilistic acceptance of worse solutions
- Genetic/evolutionary algorithms (brief)
- Basin-hopping
- When global optimization is needed vs when good initialization suffices

**Implementation (60 min):**
```python
# Exercise:
# 1. Implement random restart + L-BFGS
# 2. Implement simulated annealing
# 3. Test on Rastrigin (many local minima): local method vs global
# 4. scipy.optimize: compare minimize() vs differential_evolution() vs basinhopping()
# 5. Robotics scenario: finding initial alignment for scan matching
```

**Practice (45 min):**
- In robotics, is global optimization common? (Usually no — good initialization from sensors)
- When does it appear? (Loop closure detection, place recognition)

---

### Day 27: scipy.optimize Deep Dive (2.5 hrs)
**Theory (30 min):**
- `scipy.optimize.minimize`: interface, method selection
- Methods: Nelder-Mead, Powell, CG, BFGS, L-BFGS-B, TNC, Newton-CG, trust-constr
- `scipy.optimize.least_squares`: for NLS problems (preview)

**Implementation (90 min):**
```python
# Exercise:
# 1. Solve same problem with every scipy method — compare results and timing
# 2. Callback function: log iteration history, plot convergence
# 3. Bounds: L-BFGS-B for box-constrained problems
# 4. Jacobian/Hessian supply: show speedup of providing analytic derivatives
# 5. Wrap your custom optimizer in scipy.optimize.minimize interface
# 6. Robotics problem: calibrate wheel radius using scipy.optimize.minimize
#    Given: measured distances vs wheel encoder counts
#    Solve: min Σ(d_measured - π * r * ticks)²
```

**Practice (30 min):**
- Decision tree: which scipy method for which problem class?

---

### Day 28: Week 4 Review + Capstone Exercise (2.5 hrs)
**Comprehensive Exercise (150 min):**
```python
# PHASE II CAPSTONE: Build Your Own Optimization Library
#
# Create a Python package `miniopt` with:
#
# 1. Unified interface: miniopt.minimize(f, x0, method='bfgs', **kwargs)
# 2. Methods: gd, newton, bfgs, lbfgs, cg, trust_region
# 3. Line search: backtracking (Armijo), Wolfe
# 4. Options: max_iter, tol, callback, verbose
# 5. Return: OptimizeResult(x, fun, grad, nit, nfev, success)
#
# Test suite: all methods pass on 5 standard test functions
# 6. **pytest tests**: write proper unit tests (test_miniopt.py)
#    - Test convergence on Rosenbrock, quadratic, Styblinski-Tang
#    - Test that OptimizeResult fields are populated correctly
#    - Test that invalid inputs raise clear errors
#    - Tolerance-based assertions: assert abs(result.fun - known_min) < 1e-6
#    - Run: pytest test_miniopt.py -v
#
# This library will be your reference implementation for the rest of the course.
```

---

# PHASE III: NONLINEAR LEAST SQUARES + CERES (Weeks 5–6)

> **Goal:** Master the optimization formulation most used in robotics.
> Sensor calibration, SLAM, bundle adjustment — all are nonlinear least squares.

---

## Week 5 — Least Squares Theory + Python Implementation

> **🛣️ PHASE III GATE — Self-Assessment (15 min before Day 29)**
> 1. Implement gradient descent with backtracking line search (pseudocode)
> 2. What is the convergence rate of Newton's method? When does it fail?
> 3. Why does BFGS approximate the Hessian instead of computing it?
> 4. What is the Armijo condition? Write it mathematically.
> 5. Trust region vs line search: give one advantage of each
> 6. What does L-BFGS store instead of the full Hessian?
> 7. Name the 3 components of a conjugate gradient step
> 8. When is Nesterov acceleration better than heavy ball?
> 9. What is regularization and why does it help ill-conditioned problems?
> 10. Your `miniopt` library: which method would you recommend for a 10000-variable problem?

### Day 29: Linear Least Squares (2.5 hrs)
**Theory (45 min):**
- Problem: $\min_x \|Ax - b\|^2$
- Normal equations: $A^T A x = A^T b$
- Geometric interpretation: projection onto column space
- Solution via QR, SVD, Cholesky
- Weighted least squares: $\min_x (Ax-b)^T W (Ax-b)$
- Rank-deficient case: minimum-norm solution via SVD

**Implementation (60 min):**
```python
# Exercise:
# 1. Solve linear LS four ways: normal equations, QR, SVD, numpy.lstsq
# 2. Weighted LS: fit a line to noisy data with known per-point uncertainties
# 3. Rank-deficient: A is 100×50 with rank 30 — compare solutions
# 4. Show: normal equations are numerically unstable when κ(A) is large
#    QR and SVD are stable
# 5. Robotics: linear odometry calibration — estimate wheel radii from straight-line data
```

---

### Day 30: Nonlinear Least Squares Formulation (2.5 hrs)
**Theory (45 min):**
- Problem: $\min_x \sum_i \|r_i(x)\|^2$ where $r_i$ are nonlinear residuals
- Jacobian: $J_{ij} = \frac{\partial r_i}{\partial x_j}$
- Gradient: $\nabla f = J^T r$
- Approximate Hessian: $H \approx J^T J$ (Gauss-Newton approximation)
- Why this works: $J^T J$ ignores second-order residual terms (small near solution)
- When it fails: large residuals → need full Hessian

**Implementation (60 min):**
```python
# Exercise:
# 1. Formulate circle fitting as NLS:
#    Given points (xᵢ, yᵢ), find center (cx, cy) and radius r
#    Residuals: rᵢ = sqrt((xᵢ-cx)² + (yᵢ-cy)²) - r
# 2. Compute Jacobian analytically
# 3. Verify with numerical Jacobian
# 4. Visualize: residuals at initial guess vs at solution
```

**Practice (45 min):**
- List 5 robotics problems that are NLS: calibration, scan matching, ICP, BA, pose estimation
- For each: what are the residuals? what are the parameters?

---

### Day 31: Gauss-Newton Method (2.5 hrs)
**Theory (45 min):**
- Gauss-Newton step: $\delta = -(J^T J)^{-1} J^T r$
- Equivalently: solve $J^T J \delta = -J^T r$ (normal equations of linearized problem)
- Convergence: superlinear when residuals are small at solution
- Failure: diverges when residuals are large or $J^T J$ is singular

**Implementation (60 min):**
```python
# Exercise:
# 1. Implement Gauss-Newton from scratch
# 2. Test on circle fitting — should converge in 5-10 iterations
# 3. Test on exponential curve fitting: y = a * exp(b * x) + c
# 4. Failure case: fit a model that's wrong (large residuals at solution)
#    Show: GN oscillates or diverges
# 5. Regularized GN: $(J^T J + \mu I) \delta = -J^T r$ — show it stabilizes
```

**Practice (45 min):**
- Connection to Newton: GN is Newton with approximate Hessian
- When is the approximation good? (small residuals, overdetermined system)

---

### Day 32: Levenberg-Marquardt Algorithm (2.5 hrs)
**Theory (45 min):**
- LM = Gauss-Newton + trust region: $(J^T J + \lambda I) \delta = -J^T r$
- When $\lambda \to 0$: Gauss-Newton step (aggressive)
- When $\lambda \to \infty$: gradient descent step (conservative)
- Damping strategy: adjust $\lambda$ based on actual vs predicted reduction
- Connection to trust region: $\lambda$ is the Lagrange multiplier for $\|\delta\| \leq \Delta$
- **This is what Ceres uses by default**

**Implementation (60 min):**
```python
# Exercise:
# 1. Implement LM from scratch with adaptive damping
# 2. Compare: GN vs LM on the failure case from Day 31 — LM should work
# 3. Plot: damping parameter λ vs iteration (should decrease near solution)
# 4. Implement gain ratio: ρ = (f(xk) - f(xk+δ)) / (predicted decrease)
# 5. Sensitivity: what happens with bad initial λ?
```

**Practice (45 min):**
- Why is LM the default in most robotics software?
- Derive: at the solution, LM and GN produce the same step

---

### Day 33: Robust Estimation (2.5 hrs)
**Theory (45 min):**
- Problem: outliers dominate $\sum r_i^2$ (quadratic penalty)
- Loss functions: $\rho(r)$ instead of $r^2$
  - Huber: quadratic for small $r$, linear for large $r$
  - Cauchy: $\log(1 + r^2/c^2)$
  - Tukey bisquare: completely rejects outliers beyond threshold
- Iteratively Reweighted Least Squares (IRLS): solve weighted LS repeatedly
- M-estimators: general framework

**Implementation (60 min):**
```python
# Exercise:
# 1. Implement Huber, Cauchy, Tukey loss functions and their weights
# 2. Circle fitting with outliers:
#    Standard LS: pulled toward outliers
#    Huber LS: partially resists
#    Cauchy LS: strongly resists
# 3. Implement IRLS loop
# 4. Line fitting with 30% outliers: RANSAC vs Huber LS comparison
# 5. Robotics: ICP scan matching with outlier rejection via Huber loss
```

**Practice (45 min):**
- How to choose the loss function threshold?
- Connection: RANSAC, M-estimators, and their tradeoffs

---

### Day 34: scipy.optimize.least_squares (2.5 hrs)
**Theory (30 min):**
- API: `least_squares(fun, x0, jac, bounds, loss, method)`
- Methods: 'trf' (trust region reflective), 'dogbox', 'lm' (Levenberg-Marquardt)
- Bounds: box constraints on parameters
- Sparse Jacobians: `jac_sparsity` for large problems

**Implementation (90 min):**
```python
# Exercise:
# 1. Solve circle fitting with scipy.optimize.least_squares
# 2. Compare methods: trf vs lm — when does each win?
# 3. Bounds: constrain radius to be positive, center to be in a region
# 4. Sparse Jacobian: 2D grid deformation with 10000 points
# 5. Custom loss: provide Huber loss function
# 6. Full robotics example: camera calibration
#    - Generate synthetic checkerboard images
#    - Residuals: reprojection error
#    - Parameters: focal length, principal point, distortion
#    - Solve with least_squares, compare with/without distortion model
```

**Practice (30 min):**
- Jacobian sparsity: draw the sparsity pattern for BA, pose graph, odometry calibration

---

### Day 35: Week 5 Review + Covariance & Fisher Information + NLS Capstone (2.5 hrs)

**Theory — Uncertainty in NLS (45 min):**
- **Fisher Information Matrix**: $\mathcal{I} = J^T \Omega J$ — how much information the data carries about parameters
- **Cramér-Rao Lower Bound**: best achievable covariance $\Sigma \succeq \mathcal{I}^{-1}$
- **Covariance from NLS**: at solution, $\Sigma \approx (J^T \Omega J)^{-1}$
- **Observability analysis**: rank of $J^T J$ — full rank = all parameters observable
  - Degenerate configurations: pure-translation motion can’t calibrate rotation
  - Rank-deficient $J^T J$ = some parameters are unobservable
- **Marginalization vs conditioning**: uncertainty on a subset of parameters
- **Confidence ellipses**: visualize 2D parameter uncertainty as ellipses from covariance

```python
# UNCERTAINTY EXERCISES (before capstone):
# 1. After solving circle fitting NLS, compute covariance from (J^T J)^{-1}
#    Plot 95% confidence ellipse for (cx, cy)
# 2. Observability test: fit a line y = mx + b from 2 points vs 100 points
#    Compare Fisher information — more points = tighter bounds
# 3. Degenerate case: all points on one side of circle → show rank deficiency
#    Visualize: huge uncertainty ellipse in the poorly-observed direction
# 4. Data collection strategy: which additional measurement reduces uncertainty most?
#    Hint: maximize det(J^T J) — D-optimal design
```

```python
# CAPSTONE: Robot Wheel Calibration Pipeline
#
# Scenario: Differential drive robot, both wheels slightly different radius.
# Data: commanded velocities + measured pose changes (with noise + outliers)
#
# 1. Formulate as NLS: residuals = measured_pose - predicted_pose(r_left, r_right, baseline)
# 2. Implement Jacobian analytically
# 3. Solve with your LM implementation
# 4. Solve with scipy.optimize.least_squares
# 5. Add outlier rejection with Huber loss
# 6. Uncertainty estimation: covariance from J^T J at solution
# 7. Plot: parameter estimates converge to true values
# 8. Monte Carlo: run 100 trials with different noise → histogram of estimates
```

---

## Week 6 — Ceres Solver Mastery (C++)

### Day 36: Ceres Architecture and First Problem (2.5 hrs)
**Theory (45 min):**
- Ceres Solver architecture: Problem → CostFunction → LossFunction → Solver
- AutoDiff vs AnalyticDiff vs NumericDiff
- Jet types for automatic differentiation
- Problem construction: AddResidualBlock, AddParameterBlock

**Implementation (60 min):**
```cpp
// Exercise 1: Hello Ceres — minimize (10 - x)²
#include <ceres/ceres.h>

struct CostFunctor {
  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    residual[0] = 10.0 - x[0];
    return true;
  }
};

// Exercise 2: Circle fitting in Ceres (translate Python version to C++)
// Exercise 3: Curve fitting: y = exp(mx + c) with AutoDiff
```

**Practice (45 min):**
- Build and run, read solver summary output, interpret convergence info
- Experiment with solver options: max_iterations, function_tolerance, gradient_tolerance

---

### Day 37: Ceres AutoDiff Deep Dive (2.5 hrs)
**Theory (45 min):**
- Jet<T, N>: dual number with N partial derivatives
- How Ceres evaluates derivatives: template magic
- When AutoDiff fails: non-differentiable operations (if/else on parameters)
- NumericDiff as fallback: forward, central, Ridders

**Implementation (60 min):**
```cpp
// Exercise:
// 1. Implement cost function with AutoDiff for 3D point triangulation
//    Given: 2 camera poses + 2 pixel observations
//    Parameter: 3D point [X, Y, Z]
//    Residual: reprojection error in each camera
//
// 2. Same problem with AnalyticDiff — derive and implement Jacobian
//    Compare: runtime and accuracy vs AutoDiff
//
// 3. NumericDiff wrapper for a black-box function (e.g., calling external library)
```

**Practice (45 min):**
- When to use which: AutoDiff (default) > Analytic (performance-critical) > Numeric (last resort)
- Read Ceres docs: local parameterizations for rotations (Quaternion, AngleAxis)

---

### Day 38: Loss Functions and Robust Estimation in Ceres (2.5 hrs)
**Theory (30 min):**
- Built-in losses: HuberLoss, CauchyLoss, ArctanLoss, TukeyLoss
- Scaling parameter interpretation
- Custom loss functions

**Implementation (90 min):**
```cpp
// Exercise:
// 1. Circle fitting with outliers — compare NULL loss vs HuberLoss vs CauchyLoss
// 2. Experiment with loss function scale parameter
// 3. Implement custom loss function (Geman-McClure)
// 4. 2D scan matching with Ceres:
//    - Source points + target points with correspondence
//    - Parameter: 2D transform (tx, ty, θ)
//    - Residual: transformed source - target for each pair
//    - Add HuberLoss for outlier robustness
// 5. Compare: with/without loss on data with 20% outlier correspondences
```

---

### Day 39: Ceres for Real Robotics Problems (2.5 hrs)
**Implementation (150 min):**
```cpp
// Exercise 1: IMU-Odometry Calibration
// Given: wheel odometry poses + IMU measurements
// Estimate: wheel radii, baseline, IMU-to-base transform
// Cost: pose difference between wheel prediction and IMU prediction
//
// Exercise 2: 2D Lidar Extrinsic Calibration
// Given: robot poses from odometry + lidar scans
// Estimate: lidar-to-base transform (x, y, θ)
// Cost: scan matching residual at each pose
//
// Exercise 3: Pose Graph Optimization (preview of Week 9)
// Given: odometry edges + loop closure edges
// Estimate: all robot poses
// Cost: relative pose error for each edge
```

---

### Day 40: Solver Configuration, Advanced Features & Debugging (2.5 hrs)
**Theory (45 min):**
- Solver types: DENSE_QR, DENSE_NORMAL_CHOLESKY, SPARSE_NORMAL_CHOLESKY, CGNR, DENSE_SCHUR, SPARSE_SCHUR, ITERATIVE_SCHUR
- When to use which: problem structure matters!
- Trust region: LEVENBERG_MARQUARDT vs DOGLEG
- Convergence criteria: function_tolerance, gradient_tolerance, parameter_tolerance
- Debugging: solver summary, iteration callbacks, ordering
- Constant parameters: SetParameterBlockConstant (gauge freedom)
- Bounds: SetParameterLowerBound/Upper
- Covariance estimation: Ceres::Covariance class

**Implementation (60 min):**
```cpp
// Exercise:
// 1. Same problem, all solver types — benchmark and compare
// 2. Large problem (5000 residuals): show SPARSE_NORMAL_CHOLESKY >> DENSE
// 3. Bundle adjustment structure: show SPARSE_SCHUR wins
// 4. Iteration callback: log and plot convergence in real time
// 5. Debugging: intentionally create a bad problem (rank-deficient Jacobian)
//    Read the warning messages, understand what they mean
// 6. Covariance: after solving circle fit, compute Ceres::Covariance
//    Visualize as confidence ellipses (connects to Day 35 Fisher Information)
// 7. Fix parameters: hold one camera fixed in BA (gauge freedom)
```

**Practice (45 min):**
- Ceres solver selection flowchart
- Common pitfalls: wrong initial values, unscaled parameters, singular Jacobian

---

### Day 41: Lie Groups & Manifold Optimization for Robotics (2.5 hrs)
> **⚠️ CRITICAL TOPIC** — Without this, SLAM (Week 9) and calibration (Week 10) will feel like magic.
> This is the mathematical foundation for optimizing over rotations and poses.

**Theory (60 min):**
- The problem: rotations live on a manifold, not in Euclidean space
  - You can’t just “add a delta” to a rotation matrix — it leaves SO(3)
  - Need: perturbation in tangent space, then map back to manifold
- **SO(2)**: rotation group in 2D, angle wrapping, exp/log maps
- **SO(3)**: 3D rotations — Lie group structure
  - Representations: rotation matrix, quaternion, angle-axis, Euler angles
  - Exponential map: $\text{so}(3) \to SO(3)$ — Rodrigues’ formula
  - Logarithmic map: $SO(3) \to \text{so}(3)$ — extract angle-axis
  - Tangent space $\text{so}(3)$: skew-symmetric matrices / 3-vectors
- **SE(2)**, **SE(3)**: rigid body transformations (rotation + translation)
  - Composition: $T_{AC} = T_{AB} \cdot T_{BC}$
  - Inverse: $T^{-1}$
  - Exp/Log maps for SE(3)
- **Optimization on manifolds**:
  - Perturbation model: $x \oplus \delta = \text{Exp}(\delta) \cdot x$ (left perturbation)
  - Jacobians on manifolds: $\frac{\partial r}{\partial \delta}$ where $\delta \in \mathbb{R}^{\text{dim}}$
  - Retraction: how Ceres, g2o, GTSAM handle this internally
  - Over-parameterization: quaternions have 4 params for 3 DoF
- Connection to Ceres:
  - `EigenQuaternionManifold`, `QuaternionManifold`, `SphereManifold`
  - Why `SetManifold` is needed: tells Ceres the tangent space dimension

**Implementation (60 min):**
```python
# Exercise (Python first, then connect to Ceres):
#
# 1. Implement SO(2): exp(theta) -> R, log(R) -> theta
#    Verify: exp(log(R)) = R, log(exp(theta)) = theta
#
# 2. Implement SO(3): Rodrigues formula
#    exp(omega) -> R (3x3), log(R) -> omega (3-vec)
#    Test: random rotations, verify exp(log(R)) = R
#
# 3. Implement SE(2) class with compose, inverse, exp, log
#    Compute the Jacobian of composition: d(T1 * T2) / d(delta_T1)
#
# 4. Optimization on SO(3):
#    Given: noisy rotation measurements R1...Rn
#    Find: R_mean that minimizes sum ||Log(Ri^{-1} * R)||^2
#    Solve with GN on manifold: perturbation in tangent space
#
# 5. Connect to Ceres (C++):
#    Redo exercise 4 in Ceres with QuaternionManifold
#    Compare: your manifold GN vs Ceres
```
```cpp
// Exercise 6 (C++): Bundle adjustment mini-problem with manifolds
// - 5 cameras, 50 3D points, ~200 observations
// - Camera: angle-axis rotation (3) + translation (3) + focal length (1)
// - Point: 3 DoF
// - Use EigenQuaternionManifold or AngleAxisManifold
// - Use Schur complement solver (cameras are few, points are many)
// - Fix camera 0 (SetParameterBlockConstant) for gauge freedom
```

**Practice (30 min):**
- Lie group cheat sheet: fill in exp/log/compose/inverse for SO(2), SO(3), SE(2), SE(3)
- Why quaternions need local parameterization: 4 params, 3 DoF, unit constraint
- Preview: how GTSAM uses manifold traits for automatic Jacobian computation

---

### Day 42: Week 6 Review + Ceres Capstone (2.5 hrs)
```cpp
// CAPSTONE: Multi-Sensor Calibration Pipeline
//
// Scenario: Robot with wheel encoders + IMU + 2D lidar
// Data: robot drove a path, collected all sensor data
//
// Estimate simultaneously:
// - Left/right wheel radii, baseline
// - IMU-to-base transform (3 DoF: x, y, θ)
// - Lidar-to-base transform (3 DoF: x, y, θ)
// - IMU biases (gyro bias, accel bias)
//
// Residual blocks:
// - Odometry residuals (wheel model vs measured pose change)
// - IMU preintegration residuals
// - Lidar scan matching residuals
//
// Requirements:
// - AutoDiff for all cost functions
// - Huber loss on lidar residuals (outlier scans)
// - SPARSE_NORMAL_CHOLESKY solver
// - Covariance estimation on final parameters
// - Convergence plot
```

---

# PHASE IV: CONSTRAINED + CONVEX OPTIMIZATION (Weeks 7–8)

> **Goal:** Handle the constraints that real robotics problems always have:
> velocity limits, obstacle avoidance, joint limits, force bounds.

---

## Week 7 — Constrained Optimization Theory

> **🛣️ PHASE IV GATE — Self-Assessment (15 min before Day 43)**
> 1. Write the Gauss-Newton update step. How does it differ from Newton?
> 2. When does LM behave like gradient descent? When like Gauss-Newton?
> 3. What is a loss function in robust estimation? Name two.
> 4. Explain IRLS in one sentence.
> 5. In `scipy.optimize.least_squares`, what does `jac_sparsity` do?
> 6. Write a Ceres `CostFunctor` for $r(x) = 10 - x$ (template syntax)
> 7. What is the tangent space of SO(3)? How many dimensions?
> 8. Why does bundle adjustment use `SPARSE_SCHUR`?
> 9. What is covariance estimation from NLS? Write the formula.
> 10. When should you use `SetParameterBlockConstant` in Ceres?

### Day 43: Equality Constraints — Lagrange Multipliers (2.5 hrs)
**Theory (45 min):**
- Problem: $\min f(x)$ s.t. $h(x) = 0$
- Lagrangian: $L(x, \lambda) = f(x) + \lambda^T h(x)$
- Necessary conditions: $\nabla_x L = 0$, $h(x) = 0$
- Geometric interpretation: gradients of $f$ and $h$ are parallel at optimum
- Constraint qualification: when do Lagrange conditions hold?

**Implementation (60 min):**
```python
# Exercise:
# 1. Minimize x² + y² subject to x + y = 1
#    Solve: by hand, then verify with scipy.optimize.minimize(constraints=...)
# 2. Minimize distance to origin subject to being on an ellipse
# 3. Implement: find Lagrange multiplier system, solve with Newton
# 4. Robotics: minimum energy trajectory subject to reaching goal position
# 5. Visualize: contours of f, constraint curve, optimal point, gradient alignment
```

---

### Day 44: Inequality Constraints — KKT Conditions (2.5 hrs)
**Theory (60 min):**
- Problem: $\min f(x)$ s.t. $g(x) \leq 0$, $h(x) = 0$
- KKT conditions (necessary for local optimum):
  1. Stationarity: $\nabla f + \mu^T \nabla g + \lambda^T \nabla h = 0$
  2. Primal feasibility: $g(x) \leq 0$, $h(x) = 0$
  3. Dual feasibility: $\mu \geq 0$
  4. Complementary slackness: $\mu_i g_i(x) = 0$
- Active set: constraints where $g_i(x) = 0$
- When KKT are sufficient: convex problems

**Implementation (60 min):**
```python
# Exercise:
# 1. Min x² + y² s.t. x + y ≥ 1 — solve KKT by hand, verify with solver
# 2. Min f(x,y) with 3 inequality constraints — identify active set
# 3. Implement: KKT checker (given a candidate solution, verify all conditions)
# 4. Robotics: minimize actuator effort s.t. velocity limits and position bounds
```

**Practice (30 min):**
- Complementary slackness intuition: at optimum, either constraint is active OR multiplier is zero
- Why this matters: active constraints determine the solution structure

---

### Day 45: Penalty and Barrier Methods (2.5 hrs)
**Theory (45 min):**
- Quadratic penalty: $\min f(x) + \frac{\mu}{2} \|h(x)\|^2 + \frac{\mu}{2} \|\max(0, g(x))\|^2$
- Problem: need $\mu \to \infty$, but ill-conditioning
- Augmented Lagrangian: adds explicit multiplier estimates, better conditioning
- Log barrier: $\min f(x) - \frac{1}{t} \sum \log(-g_i(x))$ for inequality constraints
- Interior point = barrier method with increasing $t$ (central path)

**Implementation (60 min):**
```python
# Exercise:
# 1. Implement quadratic penalty method
# 2. Show: solution approaches feasible as μ increases, but conditioning worsens
# 3. Implement augmented Lagrangian (ADMM-style)
# 4. Implement log barrier method
# 5. Compare all three on a simple constrained QP
# 6. Plot: penalty function landscape for different μ values
```

**Practice (45 min):**
- Interior point methods: path through strictly feasible interior → solution
- Connection to SVM: hinge loss as penalty for constraint violation

---

### Day 46: Linear Programming (2.5 hrs)
**Theory (45 min):**
- Standard form: $\min c^T x$ s.t. $Ax \leq b$, $x \geq 0$
- Geometry: feasible region is polytope, optimum at vertex
- Simplex method: walk along vertices
- Duality: primal-dual relationship, strong duality
- Interior point for LP: modern alternative to simplex

**Implementation (60 min):**
```python
# Exercise:
# 1. LP by hand: 2 variables, plot feasible region, find optimum graphically
# 2. Implement simplex method (basic version) from scratch
# 3. Solve with scipy.optimize.linprog — compare simplex vs interior point
# 4. Robotics: task assignment as LP — assign n robots to n tasks, minimize total cost
# 5. Transportation problem: move goods from warehouses to stores, minimize shipping
```

**Practice (45 min):**
- LP duality: interpret dual variables as prices/shadow prices
- Can LP model robot path planning? (not really — but LP relaxation is useful)

---

### Day 47: Quadratic Programming (2.5 hrs)
**Theory (45 min):**
- QP: $\min \frac{1}{2} x^T P x + q^T x$ s.t. $Gx \leq h$, $Ax = b$
- Convex QP: $P \succeq 0$ → globally optimal solution
- KKT system for QP: linear system! (unlike general NLP)
- Active set methods: identify binding constraints
- Interior point methods for QP

**Implementation (60 min):**
```python
# Exercise:
# 1. Implement QP via KKT system (for equality-constrained case)
# 2. Solve with OSQP, qpOASES, scipy
# 3. Benchmark: problem size vs solve time for each solver
# 4. Portfolio optimization: min variance s.t. expected return ≥ target
# 5. Robotics: minimum effort joint torque s.t. task-space velocity constraint
#    τ = J^T f where J is Jacobian — QP in f
```

**Practice (45 min):**
- Why QP is the sweet spot: general enough for many constraints, solvable in milliseconds
- MPC preview: MPC is just a sequence of QPs

---

### Day 48: Nonlinear Programming (2.5 hrs)
**Theory (45 min):**
- NLP: $\min f(x)$ s.t. $g(x) \leq 0$, $h(x) = 0$ (all possibly nonlinear)
- Sequential Quadratic Programming (SQP): solve a QP at each iteration
- SQP = Newton's method applied to KKT conditions
- Interior point methods for NLP (IPOPT)
- Local vs global solutions: NLP is NP-hard in general

**Implementation (60 min):**
```python
# Exercise:
# 1. scipy.optimize.minimize with constraints:
#    method='SLSQP', method='trust-constr'
# 2. Compare: SQP vs interior point on nonlinear constrained problem
# 3. Robotics: trajectory optimization with obstacle avoidance
#    min Σ||u_k||² s.t. ||p_k - obstacle|| ≥ d for all k
# 4. Multiple local optima: show two different initializations → different solutions
```

**Practice (45 min):**
- NLP solver landscape: IPOPT, SNOPT, Knitro, WORHP
- When to formulate as NLP vs break into simpler subproblems

---

### Day 49: Week 7 Review + Constrained Optimization Capstone (2.5 hrs)
```python
# CAPSTONE: Constrained Robot Arm Motion
#
# 2-link planar robot arm:
# - Joint angles: θ₁, θ₂
# - End effector: (x, y) via forward kinematics
# - Actuator torque limits: |τᵢ| ≤ τ_max
# - Joint limits: θ_min ≤ θᵢ ≤ θ_max
# - Workspace obstacle: ||end_effector - obstacle|| ≥ d
#
# Tasks:
# 1. Formulate inverse kinematics as constrained NLP
# 2. Formulate minimum-torque trajectory (N waypoints) as QP
# 3. Add obstacle avoidance → becomes NLP
# 4. Solve with scipy and compare SLSQP vs trust-constr
# 5. Visualize: arm motion, joint trajectories, torque profiles, constraint satisfaction
```

---

## Week 8 — Convex Optimization + MPC

### Day 50: Convex Optimization Theory (2.5 hrs)
**Theory (60 min):**
- Convex optimization = convex objective + convex constraints
- Hierarchy: LP ⊂ QP ⊂ QCQP ⊂ SOCP ⊂ SDP ⊂ General Convex
- Each level: more expressive but harder to solve
- Duality: strong duality for convex problems
- SOCP (Second-Order Cone Programming): extends QP with norm constraints
  - Example: robust least squares $\min \|Ax - b\|_2 + \lambda \|x\|_2$ s.t. $\|Cx - d\| \leq 1$
- SDP (Semidefinite Programming): variable is a positive semidefinite matrix
  - Example: rotation averaging $\min \sum \|R_i - R_j R_{ij}\|_F^2$ s.t. $R \in SO(3)$ (SDP relaxation)
- Slater's condition: when strong duality holds

**Implementation (60 min):**
```python
# Exercise:
# 1. Classify problems: which level of convex hierarchy?
# 2. For each: formulate in standard form, solve, verify
# 3. Show: a problem that looks non-convex but can be reformulated as convex
#    Example: minimize ||Ax - b||₁ → LP reformulation
```

**Practice (30 min):**
- Decision tree: is my problem convex? How to check?

---

### Day 51: CVXPY — Convex Modeling Language (2.5 hrs)
**Theory (30 min):**
- CVXPY: express optimization in math-like syntax
- Disciplined Convex Programming (DCP): rules for valid convex programs
- Solvers: SCS, OSQP, ECOS, MOSEK

**Implementation (120 min):**
```python
import cvxpy as cp

# Exercise 1: Least-squares with CVXPY
x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A @ x - b))
prob = cp.Problem(objective)
prob.solve()

# Exercise 2: LASSO (L1 regularized least-squares)
objective = cp.Minimize(cp.sum_squares(A @ x - b) + lam * cp.norm1(x))

# Exercise 3: Robust regression (Huber loss)
objective = cp.Minimize(cp.sum(cp.huber(A @ x - b, M=1.0)))

# Exercise 4: Portfolio optimization
# min risk s.t. return ≥ target, sum(weights) = 1, weights ≥ 0

# Exercise 5: Minimum-time trajectory
# min T s.t. dynamics, velocity limits, start/end conditions

# Exercise 6: Robotics: minimum-snap trajectory for drone
# Polynomial trajectory through waypoints, minimize snap (4th derivative)
# Smooth motion = small snap

# Exercise 7 (SOCP): Robust least-squares
# min ||Ax - b||₂ s.t. ||x||₂ ≤ budget  (second-order cone constraint)
# Compare: unconstrained solution vs cone-constrained

# Exercise 8 (SDP): Rotation averaging relaxation
# Given noisy relative rotations R_ij, find absolute rotations R_i
# SDP relaxation: relax R ∈ SO(3) to R^T R ⪰ I
# Solve with CVXPY (SDP), project back to SO(3)
```

---

### Day 52: OSQP and Real-Time QP Solving (2.5 hrs)
**Theory (45 min):**
- OSQP: Operator Splitting QP solver
- ADMM algorithm: split problem into easy subproblems
- Warm starting: use previous solution as initial guess (critical for MPC!)
- Parameters: rho, sigma, max_iter, polish
- Code generation: compile QP to C code for embedded systems

**Implementation (60 min):**
```python
import osqp
import scipy.sparse as sp

# Exercise:
# 1. Formulate QP in OSQP format: min 0.5 x^T P x + q^T x s.t. l ≤ Ax ≤ u
# 2. Solve, warm start, resolve with slightly changed parameters — compare time
# 3. MPC-style: solve 100 QPs in sequence, each warm-started from previous
#    Show: warm starting is 5-10× faster
# 4. Large QP (n=5000): benchmark OSQP vs CVXPY vs scipy
# 5. Code generation: osqp.codegen() → C code for embedded
```

**Practice (45 min):**
- OSQP vs qpOASES vs ECOS: when to use which?
- Real-time requirements: MPC needs QP solved in <1ms for 100Hz control

---

### Day 53: Model Predictive Control — Theory (2.5 hrs)
**Theory (60 min):**
- MPC idea: optimize trajectory over horizon, apply first control, repeat
- Formulation for linear system $x_{k+1} = Ax_k + Bu_k$:
  $\min \sum_{k=0}^{N-1} [x_k^T Q x_k + u_k^T R u_k] + x_N^T Q_f x_N$
  s.t. $x_{k+1} = Ax_k + Bu_k$, $u_{\min} \leq u_k \leq u_{\max}$, $x_{\min} \leq x_k \leq x_{\max}$
- This is a QP! Stack all variables: $z = [x_0, u_0, x_1, u_1, ...]$
- Horizon length tradeoff: longer = better planning, slower solve
- Stability: terminal cost $Q_f$ and terminal constraint
- **Recursive feasibility**: if feasible at step $k$, must be feasible at step $k+1$
  - Terminal constraint set $\mathcal{X}_f$: control-invariant set
  - Terminal cost $Q_f$: solve DARE (discrete algebraic Riccati equation)
  - $Q_f$ as Lyapunov function: guarantees closed-loop stability
- **Infeasibility handling**: what to do when QP solver says "infeasible"
  - Soft constraints: slack variables with large penalty
  - Priority: hard constraints (dynamics) > safety > performance
  - Backup controller: switch to safe fallback if MPC fails
- **Robust MPC** (awareness): tube MPC, scenario MPC — handles model uncertainty

**Implementation (60 min):**
```python
# Exercise:
# 1. Formulate MPC for 1D double integrator (position + velocity)
# 2. Stack into QP: identify P, q, A, l, u matrices
# 3. Solve with OSQP
# 4. Simulate: MPC loop — solve QP, apply first control, propagate, repeat
# 5. Compare: no constraints vs with velocity limit vs with acceleration limit
# 6. Plot: position, velocity, control input over time
```

**Practice (30 min):**
- Why receding horizon? Why not solve once for entire trajectory?
- Computational budget: how does QP size scale with horizon and state dimension?

---

### Day 54: MPC Implementation — 2D Mobile Robot (2.5 hrs)
**Implementation (150 min):**
```python
# Full MPC controller for differential-drive robot:
#
# State: [x, y, θ, v, ω]
# Control: [a, α] (linear/angular acceleration)
# Dynamics: nonlinear (bicycle model) → linearize around current state
#
# Constraints:
# - |v| ≤ v_max, |ω| ≤ ω_max
# - |a| ≤ a_max, |α| ≤ α_max
# - Obstacle avoidance: ||[x,y] - obs|| ≥ d (nonlinear → linearize)
#
# Implementation:
# 1. Linearize dynamics around current trajectory
# 2. Formulate as QP
# 3. Solve with OSQP (warm-started from previous solution)
# 4. Apply first control
# 5. Re-linearize and repeat
#
# Visualization:
# - Animated robot following reference trajectory
# - Obstacle avoidance in real-time
# - Control inputs vs limits
# - Solve time per iteration
```

---

### Day 55: Nonlinear MPC and CasADi (2.5 hrs)
**Theory (45 min):**
- Linear MPC: fast (QP) but needs linearization
- Nonlinear MPC: solve NLP directly — more accurate but slower
- CasADi: symbolic framework for NLP with automatic differentiation
- Multiple shooting vs single shooting vs collocation

**Implementation (60 min):**
```python
# Exercise:
# 1. pip install casadi
# 2. Formulate the same mobile robot MPC as NLP in CasADi
# 3. Solve with IPOPT (interior point NLP solver)
# 4. Compare: linearized QP-MPC vs nonlinear CasADi-MPC
#    - Trajectory accuracy
#    - Solve time
# 5. When does the difference matter? (tight turns, near constraints)
```

**Practice (45 min):**
- MPC in OKS robots: linear MPC for velocity tracking, nonlinear for tight maneuvers
- Solver selection: OSQP for linear MPC, IPOPT/ACADOS for nonlinear MPC

---

### Day 56: Week 8 Review + MPC Capstone (2.5 hrs)
```python
# CAPSTONE: Warehouse Robot MPC Controller
#
# Scenario: OKS-like AMR navigating a warehouse grid
#
# 1. Define robot model (differential drive, velocity/acceleration limits)
# 2. Define environment: grid with static obstacles, narrow corridors
# 3. Given a global path (waypoints), implement MPC to track it
# 4. Constraints:
#    - Velocity and acceleration limits
#    - Obstacle clearance (linearized distance constraints)
#    - Corridor width (stay in lane)
# 5. Implementation:
#    - QP-based MPC with OSQP
#    - Warm starting between timesteps
#    - Horizon of N=20 steps at 10Hz control rate
# 6. Test scenarios:
#    - Straight corridor with speed
#    - Sharp 90° turn (velocity must slow)
#    - Narrow passage (tight constraints)
#    - Moving obstacle (re-plan)
# 7. Metrics: tracking error, constraint violations, solve time, smoothness
```

---

# PHASE V: ROBOTICS APPLICATIONS (Weeks 9–10)

> **Goal:** Apply everything to the three big robotics optimization problems:
> SLAM (graph optimization), calibration (NLS), and motion planning (MPC/NLP).

---

## Week 9 — Graph-Based SLAM

> **🛣️ PHASE V GATE — Self-Assessment (15 min before Day 57)**
> 1. Write the KKT conditions for $\min f(x)$ s.t. $g(x) \leq 0$
> 2. What is complementary slackness?
> 3. Penalty method: why can't $\mu$ be infinite from the start?
> 4. Formulate MPC for a double integrator as a QP.
> 5. What does OSQP warm-starting do? Why is it critical for MPC?
> 6. What is DCP (Disciplined Convex Programming) in CVXPY?
> 7. LP ⊂ QP ⊂ SOCP ⊂ SDP — give one example of each.
> 8. What is recursive feasibility in MPC?
> 9. CasADi: how does it differ from Ceres for trajectory optimization?
> 10. In NMPC, why is "direct collocation" preferred over "single shooting" for long horizons?

### Day 57: Factor Graphs — Theory (2.5 hrs)
**Theory (60 min):**
- Factor graph: bipartite graph with variable nodes and factor nodes
- Joint probability: $p(x) \propto \prod_i f_i(x_{S_i})$
- MAP estimation: $\max_x \prod f_i = \min_x \sum -\log f_i$
- Gaussian factors → NLS problem!
- Information matrix: $\Lambda = J^T J$ — sparse due to graph structure
- Bayes tree: efficient incremental factorization

**Implementation (60 min):**
```python
# Exercise:
# 1. Draw factor graphs for:
#    (a) Odometry chain: x0 —odom— x1 —odom— x2 —odom— x3
#    (b) GPS at every 5th pose: add unary factors
#    (c) Loop closure: add factor between x0 and x3
# 2. Write out the NLS problem for each graph
# 3. Compute and visualize the information matrix sparsity pattern
# 4. Show: loop closure creates fill-in in the information matrix
```

**Practice (30 min):**
- Why factor graphs? They make the structure of optimization problems explicit
- Connection: factor graph = Bayesian network = NLS structure

---

### Day 58: 1D Pose Graph SLAM from Scratch (2.5 hrs)
**Implementation (150 min):**
```python
# Build 1D SLAM optimizer from absolute zero:
#
# Setup:
# - Robot drives along a line: x0 → x1 → x2 → ... → x9
# - Odometry: noisy relative position measurements between consecutive poses
# - Loop closure: noisy measurement between x0 and x9 (robot returned to start)
# - Prior: x0 = 0 (anchor)
#
# Step 1: Define variables (10 scalar poses)
# Step 2: Define residuals for each factor:
#   - Prior: r_prior = x0 - 0
#   - Odometry: r_odom(i) = (x_{i+1} - x_i) - z_odom(i)
#   - Loop closure: r_loop = (x_9 - x_0) - z_loop
# Step 3: Build Jacobian (sparse! tridiagonal + one off-diagonal entry)
# Step 4: Solve with Gauss-Newton: (J^T Ω J) Δx = -J^T Ω r
# Step 5: Iterate until convergence
#
# Visualize:
# - Before optimization: poses from odometry only (drift!)
# - After optimization: loop closure pulls poses into consistency
# - Information matrix sparsity pattern
```

---

### Day 59: 2D Pose Graph SLAM (2.5 hrs)
**Implementation (150 min):**
```python
# Extend to 2D: poses are (x, y, θ)
#
# Key difference: rotation makes everything nonlinear!
# Relative pose: T_ij = T_i^{-1} * T_j (SE(2) composition)
# Residual: r_ij = Log(Z_ij^{-1} * T_i^{-1} * T_j) where Log maps SE(2) → R³
#
# Implementation:
# 1. SE(2) class: compose, inverse, log, exp
# 2. Residual function for relative pose factor
# 3. Jacobian computation (analytical — derive on paper first!)
# 4. Build sparse Jacobian and information matrix
# 5. Gauss-Newton solver with sparse Cholesky (scipy.sparse)
# 6. Test: 50 poses in a loop, with 2-3 loop closures
#
# Visualize:
# - Trajectory before/after optimization
# - Covariance ellipses at each pose
# - Convergence: total error vs iteration
```

---

### Day 60: g2o Framework (2.5 hrs)
**Theory (30 min):**
- g2o: General Graph Optimization — C++ framework
- Vertex types: SE2, SE3
- Edge types: relative pose constraints
- Solver backends: CSparse, Cholmod, PCG
- Python bindings available

**Implementation (120 min):**
```python
# Exercise (using g2o Python bindings or C++):
#
# 1. Install and set up g2o
# 2. Recreate the 2D SLAM problem from Day 59 using g2o
# 3. Compare: your implementation vs g2o (results should match!)
# 4. Large-scale: 1000 poses, 50 loop closures
#    Your code: slow. g2o: fast (optimized sparse solver).
# 5. 3D pose graph: SE3 vertices (bonus if using C++)
#
# Or with GTSAM:
# pip install gtsam
import gtsam
# Use gtsam.NonlinearFactorGraph, gtsam.Values, gtsam.LevenbergMarquardtOptimizer
```

---

### Day 61: GTSAM and Incremental SLAM (2.5 hrs)
**Theory (45 min):**
- GTSAM: factor graph library with focus on SLAM and SFM
- iSAM2: incremental smoothing and mapping
  - Don't re-solve everything when a new factor arrives
  - Bayes tree: only update affected variables
  - O(affected variables) instead of O(all variables)
- Why incremental matters: online SLAM must be fast

**Implementation (60 min):**
```python
import gtsam

# Exercise:
# 1. Build 2D SLAM with GTSAM's factor graph
# 2. Add noise models (diagonal, full covariance)
# 3. Batch solve with LevenbergMarquardtOptimizer
# 4. Incremental solve with ISAM2:
#    - Add factors one at a time (as if receiving data online)
#    - After each addition, get updated estimate
# 5. Compare: batch vs incremental — results should be similar, incremental faster
# 6. Landmark SLAM: add point landmarks observed from multiple poses
```

**Practice (45 min):**
- g2o vs GTSAM: when to use which? (g2o: simpler, GTSAM: more features + incremental)
- Factor graph for OKS: what factors would a real warehouse SLAM system use?

---

### Day 62: Robust SLAM and Outlier Rejection (2.5 hrs)
**Theory (45 min):**
- Problem: wrong loop closures (perceptual aliasing) → catastrophic failure
- Switchable constraints: add binary switch variable to each loop closure
- Max-mixture models: mixture of inlier and outlier distributions
- DCS (Dynamic Covariance Scaling)
- Graduated Non-Convexity (GNC): start convex, gradually tighten

**Implementation (60 min):**
```python
# Exercise:
# 1. Create SLAM scenario with 3 correct and 2 wrong loop closures
# 2. Standard solver: completely wrong map
# 3. With Huber loss on loop closures: partially helps
# 4. Switchable constraints: implement in your custom solver
# 5. GTSAM robust factors: use gtsam.noiseModel.mEstimator
# 6. Compare all approaches: which correctly rejects wrong closures?
```

**Practice (45 min):**
- Why is robust SLAM hard? (wrong closures look just like correct ones locally)
- State of the art: Kimera, Hydra — what robustness strategies do they use?

---

### Day 63: Week 9 Review + SLAM Capstone (2.5 hrs)
```python
# CAPSTONE: Full 2D SLAM Pipeline
#
# 1. Generate synthetic warehouse environment:
#    - Grid of corridors (like OKS warehouse)
#    - Robot does 3 loops around the warehouse
#    - Odometry: differential drive model with noise
#    - Lidar: simulated 2D scans (point clouds)
#
# 2. Frontend (given — focus on backend):
#    - ICP scan matching between consecutive poses → odometry factors
#    - Place recognition → loop closure candidates (some wrong!)
#
# 3. Your backend:
#    - Factor graph with odometry + loop closure factors
#    - Robust estimation (Huber or switchable constraints)
#    - Solve with Gauss-Newton (your code) and verify with GTSAM
#
# 4. Evaluation:
#    - Absolute Trajectory Error (ATE) vs ground truth
#    - With/without loop closures
#    - With/without robust estimation
#    - Runtime comparison: your solver vs GTSAM
#
# 5. Visualization:
#    - Trajectory: odometry-only vs optimized vs ground truth
#    - Map: accumulated scans before and after optimization
#    - Rejected loop closures highlighted
```

---

## Week 10 — Calibration, Trajectory Optimization, and Capstone

### Day 64: Multi-Sensor Calibration as Optimization (2.5 hrs)
**Theory (45 min):**
- Extrinsic calibration: find transform between sensors
- Intrinsic calibration: find sensor parameters (focal length, wheel radius, ...)
- Hand-eye calibration: $AX = XB$ problem
- Observability: do we have enough data to calibrate? (Fisher information)
- **Temporal calibration**: estimate time offset $\Delta t$ between sensors
  - Why: cameras, lidars, IMUs have different timestamps — even 5ms offset corrupts calibration
  - Method: add $\Delta t$ as optimization variable, interpolate measurements
  - Joint spatio-temporal calibration: solve for extrinsic + time offset together

**Implementation (60 min):**
```python
# Exercise:
# 1. Camera intrinsic calibration:
#    Generate synthetic checkerboard observations from multiple viewpoints
#    Estimate: fx, fy, cx, cy, k1, k2 (distortion)
#    Solve with scipy.optimize.least_squares
#
# 2. Lidar-to-camera extrinsic:
#    Given: 3D lidar points + 2D image points of known targets
#    Estimate: rotation + translation (6 DoF)
#    Solve with Ceres (C++) or scipy (Python)
#
# 3. IMU bias calibration:
#    Given: static IMU data (robot not moving)
#    Estimate: gyro bias, accel bias, gravity direction
```

**Practice (45 min):**
- What makes calibration hard? (initialization, local minima, degenerate configurations)
- Data collection strategy: how to move to maximize Fisher information?

---

### Day 65: Trajectory Optimization (2.5 hrs)
**Theory (45 min):**
- Direct collocation: discretize trajectory, optimize all points simultaneously
- Direct shooting: integrate dynamics forward, optimize controls
- Polynomial trajectories: minimum-jerk, minimum-snap
- Bezier curves and B-splines as trajectory representations

**Implementation (60 min):**
```python
# Exercise:
# 1. Minimum-jerk trajectory between two points:
#    5th order polynomial, minimize ∫ jerk² dt
#    Closed-form solution — derive and implement
#
# 2. Minimum-snap trajectory through waypoints (drone planning):
#    Piecewise polynomial, continuity constraints
#    Solve as QP with CVXPY
#
# 3. Direct collocation for double integrator with obstacles:
#    NLP with CasADi, solved by IPOPT
#
# 4. Compare: polynomial vs collocation for a zigzag path
```

**Practice (45 min):**
- Trajectory optimization in OKS: smoothing the global path for the local planner
- Connection to MPC: trajectory optimization = open-loop, MPC = closed-loop

---

### Day 66: Numerical Debugging and Production Issues (2.5 hrs)
**Theory (45 min):**
- Common failures: divergence, slow convergence, infeasible, unbounded
- Diagnosis checklist:
  1. Is the problem well-posed? (check Jacobian rank)
  2. Is the initial guess reasonable? (residual at x0)
  3. Is the Jacobian correct? (gradient checker)
  4. Is the problem scaled? (column norms of Jacobian)
  5. Is there numerical cancellation? (condition number)
- Solver output interpretation: Ceres summary, OSQP info, IPOPT logs

**Implementation (60 min):**
```python
# Exercise: Debug These Broken Optimizations
#
# 1. Problem: solver diverges → diagnosis: wrong Jacobian sign
# 2. Problem: solver converges to wrong answer → diagnosis: local minimum, need better init
# 3. Problem: solver is very slow → diagnosis: ill-conditioned, need scaling
# 4. Problem: solver says "infeasible" → diagnosis: conflicting constraints
# 5. Problem: solver oscillates → diagnosis: discontinuous cost function
#
# For each: identify the issue, fix it, verify solution
```

**Practice (45 min):**
- Build your debugging checklist (you'll use this in production)
- Real OKS scenario: estimator calibration diverged — walk through diagnosis

---

### Day 67: Real-Time Optimization (2.5 hrs)
**Theory (45 min):**
- Real-time requirements: solve within control loop period (1-10ms)
- Warm starting: previous solution as initial guess
- Early termination: good-enough solution in limited iterations
- Code generation: compile problem to fast C code
- Explicit MPC: precompute solution as piecewise affine function of state

**Implementation (60 min):**
```python
# Exercise:
# 1. MPC loop at 100Hz: measure solve time per iteration
#    Must be <10ms consistently!
# 2. Warm starting benchmark: cold start vs warm start solve time
# 3. Iteration-limited solve: 5, 10, 20 max iterations — quality vs time tradeoff
# 4. OSQP code generation: generate C solver, call from Python via ctypes
# 5. Compile-time: benchmark generated C solver vs Python OSQP
```

**Practice (45 min):**
- Real-time constraints in OKS: MPC runs at 10-50Hz, must be deterministic
- WCET (Worst Case Execution Time): why average isn't enough

---

### Day 68: Optimization Software Landscape (2.5 hrs)
**Theory (60 min):**
Survey of production optimization tools:

| Tool | Type | Language | Strength |
|------|------|----------|----------|
| Ceres | NLS | C++ | Robotics calibration, SLAM |
| g2o | Graph NLS | C++ | SLAM backends |
| GTSAM | Factor graph | C++ | Incremental SLAM, SFM |
| OSQP | QP | C/Python | Real-time MPC |
| CVXPY | Convex | Python | Prototyping, modeling |
| CasADi | NLP/AD | Python/C++ | Nonlinear MPC, optimal control |
| IPOPT | NLP | C++ | General large-scale NLP |
| ACADOS | NMPC | C/Python | Embedded nonlinear MPC |
| Eigen | Dense LA | C++ | Foundation for all above |
| SuiteSparse | Sparse LA | C | Backend for Ceres, g2o |
| OR-Tools | LP/MIP | C++/Python | Task assignment, scheduling |

- **Distributed & multi-agent optimization** (growth direction):
  - Distributed SLAM: each robot optimizes locally, share constraints at boundaries
  - Consensus optimization (ADMM-based): robots agree on shared variables
  - Multi-agent task allocation: combinatorial optimization (MILP / auction)
  - Reference: Choudhary et al., "Distributed Mapping with Privacy and Communication Constraints"

**Implementation (90 min):**
```python
# Exercise: Solve the SAME problem in multiple tools
#
# Problem: 2D robot trajectory optimization with obstacles
# - 50 timesteps, differential drive
# - 3 circular obstacles
# - Start and goal positions
#
# Solve with:
# 1. scipy.optimize.minimize (SLSQP)
# 2. CVXPY (convex relaxation)
# 3. CasADi + IPOPT
# 4. OSQP (linearized version)
#
# Compare: solution quality, solve time, ease of formulation
```

---

### Day 69–70: Final Capstone Project (5 hrs total)
```
# FINAL CAPSTONE: Integrated Robotics Optimization Pipeline
#
# Build a complete optimization stack for a simulated warehouse robot:
#
# === Module 1: Calibration (Ceres C++ or scipy Python) ===
# - Input: raw sensor data (wheel encoders, IMU, lidar)
# - Output: calibrated parameters (wheel radii, baseline, sensor transforms)
# - Method: NLS with Huber loss, Levenberg-Marquardt
#
# === Module 2: SLAM Backend (custom or GTSAM) ===
# - Input: odometry factors + loop closure factors
# - Output: optimized pose graph (trajectory + map)
# - Method: Gauss-Newton on factor graph, robust to wrong loop closures
#
# === Module 3: MPC Controller (OSQP or CasADi) ===
# - Input: reference path from global planner, current state
# - Output: velocity commands
# - Method: QP-MPC with velocity/acceleration limits and obstacle avoidance
# - Runs at 10Hz with warm starting
#
# === Integration ===
# 1. Calibrate sensors → use calibrated parameters in SLAM and MPC
# 2. Run SLAM → get corrected map and trajectory
# 3. Plan path on SLAM map → reference for MPC
# 4. MPC tracks path while avoiding obstacles
#
# === Deliverables ===
# - Working code for all three modules
# - Convergence plots for each optimization
# - Runtime benchmarks
# - Written report: which optimization method for which module, and why
```

---

# Appendix A: Daily Checklist Template

```
## Day [N]: [Topic]
Date: ____

### Theory (45 min)
- [ ] Read assigned material
- [ ] Took notes on key concepts
- [ ] Can explain main idea in one sentence

### Implementation (60 min)
- [ ] Completed all exercises
- [ ] Code runs and produces correct output
- [ ] Compared with reference solution / library

### Practice (45 min)
- [ ] Completed problem set
- [ ] Reviewed mistakes
- [ ] Updated flashcards

### Reflection
- Hardest concept today:
- Connection to previous material:
- Still unclear:
```

---

# Appendix B: Prerequisite Check

Before starting, you should be comfortable with:

- [ ] Python (NumPy, Matplotlib, basic classes)
- [ ] Linear algebra basics (matrix multiply, determinant, inverse)
- [ ] Calculus (derivatives, partial derivatives, chain rule)
- [ ] C++ basics (for Weeks 5-6 Ceres exercises) — structs, templates, CMake

If not, spend 3-5 days on a crash course before Day 1.

---

# Appendix C: Recommended Reading Order

| Priority | Resource | When |
|----------|----------|------|
| **Essential** | Boyd & Vandenberghe (free PDF) | Weeks 7-8, reference throughout |
| **Essential** | Ceres Solver tutorial | Week 6 |
| **Essential** | Grisetti SLAM tutorial paper | Week 9 |
| **High** | Nocedal & Wright Ch 1-7 | Weeks 3-5, algorithm detail |
| **Medium** | GTSAM tutorial | Week 9 |
| **Medium** | Rawlings MPC textbook | Week 8 |
| **Reference** | OSQP paper | Week 8 |
| **Reference** | g2o paper | Week 9 |

---

# Appendix D: Progress Tracker

| Week | Phase | Status | Hours | Notes |
|------|-------|--------|-------|-------|
| 1 | Math Foundations I | ☐ | 17.5 | |
| 2 | Math Foundations II | ☐ | 17.5 | |
| 3 | Unconstrained I | ☐ | 17.5 | |
| 4 | Unconstrained II | ☐ | 17.5 | |
| 5 | Least Squares (Python) | ☐ | 17.5 | |
| 6 | Ceres Solver (C++) | ☐ | 17.5 | |
| 7 | Constrained Optimization | ☐ | 17.5 | |
| 8 | Convex + MPC | ☐ | 17.5 | |
| 9 | SLAM | ☐ | 17.5 | |
| 10 | Applications + Capstone | ☐ | 17.5 | |
| 11 | Lie Groups: Group Theory & Rotations | ☐ | 17.5 | Phase VI |
| 12 | Lie Groups: Rigid Motions & Lie Algebra | ☐ | 17.5 | |
| 13 | Lie Groups: Manifold Optimization + Capstone | ☐ | 17.5 | |
| 14 | Game Theory: Normal Form & Zero-Sum | ☐ | 17.5 | Phase VII |
| 15 | Game Theory: Cooperative & Dynamic | ☐ | 17.5 | |
| 16 | Game Theory: Algorithmic GT + Capstone | ☐ | 17.5 | |
| **Total** | | | **280** | |
