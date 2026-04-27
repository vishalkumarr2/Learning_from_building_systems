# 04 — Constrained Optimization

> **Goal:** Add constraints to the picture — equality and inequality constraints, Lagrange multipliers, KKT conditions, duality, and practical algorithms.

---

## 1. Problem Formulation

### Standard Form

$$\min_{x \in \mathbb{R}^n} f(x)$$
$$\text{subject to} \quad g_i(x) \leq 0, \quad i = 1, \ldots, m$$
$$\phantom{\text{subject to}} \quad h_j(x) = 0, \quad j = 1, \ldots, p$$

- $f$ — objective function
- $g_i$ — inequality constraints (≤ 0 by convention)
- $h_j$ — equality constraints

**Feasible set:** $\mathcal{F} = \{x \mid g_i(x) \leq 0, \; h_j(x) = 0 \; \forall i, j\}$

### Examples in Robotics

| Problem | Objective | Constraints |
|---------|-----------|-------------|
| Path planning | Minimize path length | Obstacle avoidance $g_i(x) \leq 0$, kinematics $h_j(x) = 0$ |
| MPC | Minimize tracking error | Velocity/acceleration limits, actuator bounds |
| Calibration | Minimize residual | Known geometric constraints (e.g., unit quaternion) |
| Robot arm IK | Minimize joint motion | Joint angle limits, collision avoidance |

---

## 2. Lagrange Multipliers (Equality Constraints Only)

### Setup

$$\min_x f(x) \quad \text{s.t.} \quad h(x) = 0$$

### Geometric Insight

At the constrained optimum $x^*$:
- $\nabla f(x^*)$ must be parallel to $\nabla h(x^*)$
- Otherwise, you could move along the constraint surface and decrease $f$

So there exists a scalar $\lambda^*$ such that:

$$\nabla f(x^*) + \lambda^* \nabla h(x^*) = 0$$

### The Lagrangian

$$\mathcal{L}(x, \lambda) = f(x) + \lambda^T h(x)$$

The **necessary conditions** for optimality:

$$\nabla_x \mathcal{L} = \nabla f + \sum_j \lambda_j \nabla h_j = 0$$
$$\nabla_\lambda \mathcal{L} = h(x) = 0$$

These are $n + p$ equations in $n + p$ unknowns ($x$ and $\lambda$).

### Example: Minimize on a Sphere

$$\min_{x} x_1 + x_2 + x_3 \quad \text{s.t.} \quad x_1^2 + x_2^2 + x_3^2 = 1$$

Lagrangian: $\mathcal{L} = x_1 + x_2 + x_3 + \lambda(x_1^2 + x_2^2 + x_3^2 - 1)$

$$\frac{\partial \mathcal{L}}{\partial x_i} = 1 + 2\lambda x_i = 0 \implies x_i = -\frac{1}{2\lambda}$$

From constraint: $3 \cdot \frac{1}{4\lambda^2} = 1 \implies \lambda = \pm\frac{\sqrt{3}}{2}$

Minimum at $x^* = (-\frac{1}{\sqrt{3}}, -\frac{1}{\sqrt{3}}, -\frac{1}{\sqrt{3}})$.

---

## 3. KKT Conditions (Inequality + Equality Constraints)

The Karush-Kuhn-Tucker conditions are the **necessary conditions** for constrained optimality (and sufficient when the problem is convex).

### The Conditions

Given the Lagrangian:

$$\mathcal{L}(x, \lambda, \mu) = f(x) + \sum_j \lambda_j h_j(x) + \sum_i \mu_i g_i(x)$$

At the optimum $(x^*, \lambda^*, \mu^*)$:

| Condition | Formula | Meaning |
|-----------|---------|---------|
| **Stationarity** | $\nabla_x \mathcal{L} = 0$ | Gradient balance |
| **Primal feasibility** | $g_i(x^*) \leq 0$, $h_j(x^*) = 0$ | Solution is feasible |
| **Dual feasibility** | $\mu_i^* \geq 0$ | Multipliers for inequalities are non-negative |
| **Complementary slackness** | $\mu_i^* g_i(x^*) = 0 \; \forall i$ | Either constraint inactive or multiplier zero |

### Complementary Slackness — The Key Insight

For each inequality constraint:
- **Active** ($g_i(x^*) = 0$): constraint is "tight", $\mu_i^* \geq 0$ (constraint is pushing)
- **Inactive** ($g_i(x^*) < 0$): constraint doesn't matter, so $\mu_i^* = 0$

**You never need to check inactive constraints** — their multipliers are simply zero.

### Interpretation of Multipliers

$\mu_i^*$ tells you: "How much would the objective improve if constraint $i$ were relaxed by $\epsilon$?"

$$\frac{\partial f^*}{\partial b_i} = -\mu_i^*$$

where $g_i(x) \leq b_i$ and $b_i = 0$ at the original problem.

**OKS relevance:** In MPC, the multiplier on the velocity constraint tells you how much the tracking error would decrease if the robot could go faster. Active constraints at their limits → the robot is saturating.

---

## 4. Duality

### The Lagrangian Dual Function

$$q(\lambda, \mu) = \min_x \mathcal{L}(x, \lambda, \mu)$$

For any $\lambda$ and $\mu \geq 0$, $q(\lambda, \mu) \leq f^*$ (weak duality).

### The Dual Problem

$$\max_{\lambda, \mu} q(\lambda, \mu) \quad \text{s.t.} \quad \mu \geq 0$$

| Property | Primal | Dual |
|----------|--------|------|
| Direction | Minimize | Maximize |
| Variables | $x$ (original) | $\lambda, \mu$ (multipliers) |
| Constraints | $g_i \leq 0, h_j = 0$ | $\mu \geq 0$ |
| Bound | Upper bound on $f^*$ | Lower bound on $f^*$ |

### Strong Duality

For **convex** problems satisfying a constraint qualification (e.g., Slater's condition: there exists a strictly feasible point):

$$d^* = p^* \quad \text{(zero duality gap)}$$

This means:
1. Solving the dual gives the exact primal optimum
2. KKT conditions are both necessary and sufficient
3. We can solve whichever (primal or dual) is easier

---

## 5. Penalty and Barrier Methods

### Penalty Method (Equality Constraints)

Replace $\min f(x)$ s.t. $h(x) = 0$ with:

$$\min_x f(x) + \frac{\rho}{2} \|h(x)\|^2$$

As $\rho \to \infty$, the solution approaches the constrained optimum.

**Problem:** Large $\rho$ → ill-conditioned Hessian → numerical issues.

### Log-Barrier Method (Inequality Constraints)

Replace $\min f(x)$ s.t. $g_i(x) \leq 0$ with:

$$\min_x f(x) - \frac{1}{t} \sum_i \log(-g_i(x))$$

The barrier $-\log(-g_i)$ approaches $+\infty$ as $g_i \to 0^-$, keeping $x$ strictly inside the feasible region.

As $t \to \infty$, the barrier vanishes and the solution approaches the constrained optimum.

**This is the basis for interior-point methods** (Chapter 05).

```python
def barrier_method(f, g_list, x0, t=1.0, mu=10.0, tol=1e-6):
    """
    f: objective, g_list: list of g_i(x) <= 0 constraints
    """
    x = x0.copy()
    m = len(g_list)
    while m / t > tol:
        # Minimize: t * f(x) + barrier
        def phi(x):
            barrier = -sum(np.log(-g(x)) for g in g_list)
            return t * f(x) + barrier
        x = minimize_unconstrained(phi, x)  # Use Newton/BFGS
        t *= mu
    return x
```

---

## 6. Sequential Quadratic Programming (SQP)

### Idea

Generalize Newton's method to constrained problems by solving a QP subproblem at each step:

$$\min_{\Delta x} \nabla f_k^T \Delta x + \frac{1}{2} \Delta x^T H_k \Delta x$$
$$\text{s.t.} \quad g_i(x_k) + \nabla g_i(x_k)^T \Delta x \leq 0$$
$$\phantom{\text{s.t.}} \quad h_j(x_k) + \nabla h_j(x_k)^T \Delta x = 0$$

This linearizes the constraints and uses a quadratic model of the objective.

### Algorithm

```
1. At x_k, form QP subproblem (linearize constraints, quadratic objective)
2. Solve QP → get step Δx and multiplier estimates λ, μ
3. Line search or trust region on a merit function
4. Update x_{k+1} = x_k + α Δx
5. Update Hessian estimate (BFGS on Lagrangian Hessian)
```

**SQP is the workhorse for medium-sized NLPs.** It's what SNOPT, IPOPT (in SQP mode), and many MPC solvers use.

---

## 7. Augmented Lagrangian Method (ALM)

### Combine Penalty + Lagrange Multipliers

$$\mathcal{L}_A(x, \lambda, \rho) = f(x) + \lambda^T h(x) + \frac{\rho}{2}\|h(x)\|^2$$

### Algorithm

```
1. Minimize L_A(x, λ_k, ρ_k) over x (unconstrained subproblem)
2. Update multipliers: λ_{k+1} = λ_k + ρ_k h(x_{k+1})
3. Optionally increase ρ_k
4. Repeat until convergence
```

**Advantage over pure penalty:** The multiplier estimate $\lambda_k$ means you don't need $\rho \to \infty$ — finite $\rho$ suffices, avoiding ill-conditioning.

---

## 8. Solver Landscape

| Solver | Method | Best for | Language |
|--------|--------|----------|----------|
| IPOPT | Interior-point | Large NLP | C++ / Python (via pyomo, casadi) |
| SNOPT | SQP | Medium NLP, sparse | Fortran / MATLAB |
| OSQP | ADMM | Convex QP | C / Python |
| ECOS | Interior-point | Convex SOCP | C / Python |
| qpOASES | Active-set QP | Real-time MPC | C++ |
| FORCES | Code-gen QP/NLP | Embedded MPC | C |
| `scipy.optimize.minimize` | SQP/trust-constr | General Python | Python |

**OKS relevance:**
- MPC uses QP solvers (OSQP, qpOASES) at the inner loop
- Trajectory optimization uses NLP solvers (IPOPT, SNOPT)
- Ceres doesn't handle general constraints (it's NLLS only), but you can add bounds via `SetParameterLowerBound`/`SetParameterUpperBound`

---

## 9. Constrained Optimization in Ceres

Ceres is primarily for unconstrained/bound-constrained least squares, but you can handle constraints:

### Parameter Bounds

```cpp
problem.SetParameterLowerBound(params, 0, -M_PI);   // θ ≥ -π
problem.SetParameterUpperBound(params, 0,  M_PI);   // θ ≤ π
```

### Local Parameterizations (Manifold Constraints)

For unit quaternions ($\|q\| = 1$), don't add a constraint — use a manifold:

```cpp
// Ceres 2.x
problem.SetManifold(quaternion, new ceres::EigenQuaternionManifold);
```

This tells Ceres to update the quaternion on the 3-sphere rather than in $\mathbb{R}^4$.

### Soft Constraints (as Cost Terms)

Turn a constraint into a penalty:

```cpp
// Soft constraint: x[0] ≈ target_value
struct SoftConstraint {
    template <typename T>
    bool operator()(const T* x, T* residual) const {
        residual[0] = T(weight) * (x[0] - T(target));
        return true;
    }
    double weight, target;
};
```

---

## Summary: What to Carry Forward

| Concept | Where it's used next |
|---------|---------------------|
| KKT conditions | 05 (optimality in convex programs, interior-point methods) |
| Duality | 05 (LP/QP duals, SDP) |
| Log-barrier | 05 (interior-point algorithm) |
| SQP | 05 (QP subproblems), MPC formulation in control |
| Penalty methods | 06 (graph optimization with soft constraints) |
| Ceres bounds + manifolds | 06 (rotation manifolds in SLAM) |
