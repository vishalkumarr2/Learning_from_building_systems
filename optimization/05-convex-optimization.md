# 05 — Convex Optimization

> **Goal:** Understand the class of problems with guaranteed global optima — linear programs, quadratic programs, conic programs — and the interior-point methods that solve them in polynomial time.

---

## 1. What Makes Convex Special?

### Definition

A problem is **convex** if:
- The objective $f(x)$ is a convex function
- The inequality constraints $g_i(x) \leq 0$ are convex functions
- The equality constraints $h_j(x) = 0$ are **affine** ($h_j(x) = a_j^T x - b_j$)

### The Fundamental Theorem

> **Any local minimum of a convex problem is the global minimum.**

This means:
- No getting stuck in local minima
- KKT conditions are necessary **and sufficient**
- Strong duality holds (under mild conditions)
- Polynomial-time algorithms exist

### The Hierarchy

```
General nonlinear optimization (NP-hard in general)
  └── Convex optimization (polynomial time)
        ├── Linear Programming (LP)
        ├── Quadratic Programming (QP)
        ├── Second-Order Cone Programming (SOCP)
        ├── Semidefinite Programming (SDP)
        └── Geometric Programming (GP)
```

Each level includes the ones above it: LP ⊂ QP ⊂ SOCP ⊂ SDP ⊂ Convex.

---

## 2. Linear Programming (LP)

### Standard Form

$$\min_x c^T x \quad \text{s.t.} \quad Ax \leq b, \quad x \geq 0$$

Everything is linear — objective and constraints.

### Where LP Appears

| Application | Variables | Constraints |
|-------------|-----------|-------------|
| Task allocation | Assign robots to tasks | Capacity, precedence |
| Network flow | Flow on edges | Conservation, capacity |
| Diet problem | Quantities of foods | Nutritional requirements |
| Warehouse layout | Item placement | Space constraints |

### Solving LPs

| Method | Complexity | Practical speed |
|--------|-----------|-----------------|
| Simplex | Exponential worst case | Very fast in practice |
| Interior-point | $O(n^{3.5} \log(1/\epsilon))$ | Fast for large LPs |

### LP Duality

Primal: $\min c^Tx$ s.t. $Ax \leq b$, $x \geq 0$

Dual: $\max b^Ty$ s.t. $A^Ty \leq c$, $y \geq 0$

Strong duality always holds for LP (if feasible).

---

## 3. Quadratic Programming (QP)

### Standard Form

$$\min_x \frac{1}{2} x^T Q x + c^T x \quad \text{s.t.} \quad Ax \leq b, \quad Cx = d$$

where $Q \succeq 0$ (positive semidefinite) for convexity.

### Where QP Appears in OKS

| Application | Q matrix | Constraints |
|-------------|----------|-------------|
| **MPC** | State/input cost matrices | Velocity/acceleration limits, input bounds |
| Least-squares with bounds | $J^TJ$ | Parameter bounds |
| Portfolio optimization | Covariance matrix | Budget, position limits |

### MPC as a QP

At each control step, MPC solves:

$$\min_{u_0, \ldots, u_{N-1}} \sum_{k=0}^{N-1} \left[ x_k^T Q x_k + u_k^T R u_k \right] + x_N^T Q_f x_N$$
$$\text{s.t.} \quad x_{k+1} = Ax_k + Bu_k \quad \text{(dynamics)}$$
$$\phantom{\text{s.t.}} \quad u_{\min} \leq u_k \leq u_{\max} \quad \text{(actuator limits)}$$
$$\phantom{\text{s.t.}} \quad x_{\min} \leq x_k \leq x_{\max} \quad \text{(state limits)}$$

Stack all decision variables into one vector → standard QP form.

**OKS relevance:** The robot's MPC runs a QP at each control cycle to compute wheel velocities that track the desired path while respecting max velocity/acceleration.

### QP Solvers

| Solver | Method | Speed | Best for |
|--------|--------|-------|----------|
| OSQP | ADMM | Very fast | Medium QPs, MPC |
| qpOASES | Active-set | Fast warm-start | Real-time MPC |
| Gurobi | Barrier + simplex | Very fast | Large QPs (commercial) |
| ECOS | Interior-point | Good | Embedded |

```python
import osqp
import numpy as np
from scipy import sparse

# Min 0.5 x^T P x + q^T x  s.t.  l <= Ax <= u
P = sparse.csc_matrix([[4, 1], [1, 2]])
q = np.array([1, 1])
A = sparse.csc_matrix([[1, 1], [1, 0], [0, 1]])
l = np.array([1, 0, 0])
u = np.array([1, 0.7, 0.7])

solver = osqp.OSQP()
solver.setup(P, q, A, l, u, verbose=False)
result = solver.solve()
print(result.x)  # Optimal x
```

---

## 4. Second-Order Cone Programming (SOCP)

### Standard Form

$$\min_x c^Tx \quad \text{s.t.} \quad \|A_ix + b_i\|_2 \leq c_i^Tx + d_i, \quad i = 1, \ldots, m$$

The constraint says: the vector $A_ix + b_i$ must lie inside a (rotated) ice-cream cone.

### Why SOCP Matters

Many practical constraints are SOC representable:

| Constraint | SOCP form |
|------------|-----------|
| Norm bound: $\|x\| \leq t$ | Direct SOC constraint |
| Robust LP: uncertain $a_i$ | Ellipsoidal uncertainty → SOC |
| Quadratic-over-linear: $\frac{\|x\|^2}{t} \leq s$ | Rotated SOC |
| Euclidean distance | $\|p_1 - p_2\| \leq d$ |

**OKS relevance:** Path planning with distance constraints (stay $\geq d$ from obstacles) can be formulated as SOCP.

---

## 5. Semidefinite Programming (SDP)

### Standard Form

$$\min_X \langle C, X \rangle \quad \text{s.t.} \quad \langle A_i, X \rangle = b_i, \quad X \succeq 0$$

where $X$ is a matrix variable constrained to be positive semidefinite.

### Where SDP Appears

| Application | Why |
|-------------|-----|
| Relaxation of combinatorial problems | Graph partition, max-cut |
| Sensor network localization | Distance constraints → SDP relaxation |
| Control (LMIs) | Stability analysis via Lyapunov |
| Sum-of-squares polynomials | Proving nonnegativity |

### Relationship to Other Problems

$$\text{LP} \subset \text{QP} \subset \text{SOCP} \subset \text{SDP}$$

Any LP can be written as a QP (with $Q = 0$), any QP as an SOCP, any SOCP as an SDP.

---

## 6. Interior-Point Methods

### The Main Algorithm for Convex Optimization

**Idea:** Replace inequality constraints with a log-barrier, then solve a sequence of smooth unconstrained problems.

### Algorithm Outline

For $\min f(x)$ s.t. $g_i(x) \leq 0$:

$$\min_x t \cdot f(x) + \phi(x) \quad \text{where} \quad \phi(x) = -\sum_{i=1}^m \log(-g_i(x))$$

```
1. Initialize: x strictly feasible, t = t_0
2. Centering step: minimize t·f(x) + φ(x) using Newton's method
   → get x*(t), the "central path" point
3. Increase t: t ← μ · t  (μ > 1, typically μ = 10-20)
4. If m/t < ε: STOP (duality gap < ε)
5. Go to 2
```

### Why It Works

- The barrier keeps $x$ strictly inside the feasible region
- As $t \to \infty$, the barrier term vanishes → $x^*(t)$ converges to the true optimum
- The duality gap at each step is exactly $m/t$
- Total Newton steps: $O(\sqrt{m} \log(m/\epsilon))$ — polynomial!

### Central Path

The set $\{x^*(t) \mid t > 0\}$ is a smooth curve through the interior of the feasible region, converging to the optimal point.

---

## 7. CVXPy — Disciplined Convex Programming in Python

### Philosophy

Don't formulate the standard form manually. Describe the problem in a natural way, and CVXPy:
1. Verifies convexity (rejects non-convex problems)
2. Transforms to standard form
3. Calls the appropriate solver

### Basic Usage

```python
import cvxpy as cp
import numpy as np

# Variables
x = cp.Variable(2)

# Objective
objective = cp.Minimize(cp.sum_squares(x - np.array([1, 2])))

# Constraints
constraints = [
    x >= 0,
    x[0] + x[1] <= 3,
    cp.norm(x) <= 2.5
]

# Solve
problem = cp.Problem(objective, constraints)
problem.solve()

print(f"Status: {problem.status}")
print(f"Optimal x: {x.value}")
print(f"Optimal cost: {problem.value}")
```

### DCP Rules (Disciplined Convex Programming)

CVXPy checks that your problem follows composition rules:

| Expression | Convexity | Examples |
|------------|-----------|---------|
| Affine | Both convex and concave | $a^Tx + b$ |
| $\|x\|_2$ | Convex | Norm constraints |
| `cp.sum_squares(x)` | Convex | Least-squares objective |
| `cp.log(x)` | Concave | Log constraints |
| `cp.maximum(f, g)` | Convex if $f, g$ convex | Max of convex functions |
| `cp.quad_form(x, P)` | Convex if $P \succeq 0$ | QP objective |

**Key rule:** Convex ∘ affine = convex. So `cp.norm(A @ x - b)` is valid.

### Example: Weighted Least Squares with Constraints

```python
# Fit y = Ax with non-negative parameters and bounded norm
A = np.random.randn(100, 10)
y = np.random.randn(100)
W = np.diag(np.random.rand(100))  # measurement weights

x = cp.Variable(10)
objective = cp.Minimize(cp.quad_form(A @ x - y, W))
constraints = [x >= 0, cp.norm(x) <= 5]
cp.Problem(objective, constraints).solve()
```

---

## 8. Convexity Detection and Reformulation

### When a Problem Looks Nonconvex but Isn't

| Original form | Convex reformulation |
|---------------|---------------------|
| $\min \|Ax - b\|_2$ | Already convex (norm of affine) |
| $\min \frac{\|x\|^2}{t}$, $t > 0$ | Rotated SOC: $\|x\|^2 \leq ts$ |
| $\min \max_i (a_i^Tx + b_i)$ | Epigraph: $\min t$ s.t. $a_i^Tx + b_i \leq t$ |
| $\min \log\det(X^{-1})$ s.t. $X \succ 0$ | SDP with $\log\det$ is concave → maximize |
| $\min \prod x_i^{-1}$ | Geometric program → convex via log transform |

### Convex Relaxation

For non-convex problems, solve a convex relaxation to get:
- A **lower bound** on the optimal value
- An approximate solution (sometimes tight, sometimes loose)

Common relaxations:
- Boolean $x \in \{0, 1\}^n$ → LP relaxation: $0 \leq x \leq 1$
- Rank constraint $\text{rank}(X) = 1$ → SDP relaxation: $X \succeq 0$

---

## 9. Practical Guide: Which Solver When?

| Problem type | Preferred solver | Python interface |
|-------------|-----------------|------------------|
| LP | HiGHS, Gurobi, GLPK | `scipy.optimize.linprog`, CVXPy |
| QP | OSQP, Gurobi, qpOASES | OSQP Python, CVXPy |
| SOCP | ECOS, Gurobi, MOSEK | CVXPy |
| SDP | SCS, MOSEK | CVXPy |
| General convex | SCS (free), MOSEK (commercial) | CVXPy |
| MPC (real-time) | OSQP, FORCES | OSQP, CasADi |
| Non-convex NLP | IPOPT, SNOPT | CasADi, pyomo |

**CVXPy auto-selects** a solver based on problem type. Override with:

```python
problem.solve(solver=cp.OSQP)  # or cp.ECOS, cp.SCS, cp.MOSEK
```

---

## Summary: What to Carry Forward

| Concept | Where it's used next |
|---------|---------------------|
| QP formulation | MPC in control (Track 4), exercises |
| Interior-point methods | Understanding Ceres' solver internals |
| CVXPy | Exercises 03 (constrained fitting), 04 (MPC) |
| Convexity checking | Knowing when your problem has a guaranteed global solution |
| Solver selection | 06 (choosing graph optimization backend) |
| Duality | Understanding why certain relaxations work in SLAM |
