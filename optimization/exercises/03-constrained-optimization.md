# Exercises: Constrained Optimization

### Chapter 04: Constrained Optimization

**Self-assessment guide:** Focus on understanding KKT conditions and formulating real robotics problems as constrained optimization. The math is less important than the modelling skill.

---

## Section A — Conceptual Questions

**A1.** State the KKT conditions for the problem:

$$\min_x f(x) \quad \text{s.t.} \quad g_i(x) \leq 0, \quad h_j(x) = 0$$

Then apply them to: $\min_{x,y} x^2 + y^2$ subject to $x + y = 1$.

<details><summary>Answer</summary>

**KKT conditions** (necessary for optimality):

1. **Stationarity:** $\nabla f(x^*) + \sum_i \mu_i \nabla g_i(x^*) + \sum_j \lambda_j \nabla h_j(x^*) = 0$
2. **Primal feasibility:** $g_i(x^*) \leq 0$, $h_j(x^*) = 0$
3. **Dual feasibility:** $\mu_i \geq 0$ (for inequality constraints)
4. **Complementary slackness:** $\mu_i g_i(x^*) = 0$ (either $\mu_i = 0$ or $g_i = 0$)

**Applied to** $\min x^2 + y^2$ s.t. $x + y = 1$:

Only an equality constraint, so no $\mu_i$. KKT:
- $\nabla f + \lambda \nabla h = 0$: $(2x, 2y) + \lambda(1, 1) = 0$ → $x = y = -\lambda/2$
- $h = 0$: $x + y = 1$ → $-\lambda = 1$ → $\lambda = -1$
- Solution: $x^* = y^* = 1/2$, $\lambda^* = -1$

The minimum of $x^2 + y^2$ on the line $x + y = 1$ is the point closest to the origin: $(0.5, 0.5)$.

$\lambda = -1$ means: relaxing the constraint by $\epsilon$ (to $x + y = 1 + \epsilon$) would decrease the optimal cost by $|\lambda| \cdot \epsilon = \epsilon$.

</details>

**A2.** Explain the difference between a penalty method and a barrier method. When would you choose each?

<details><summary>Answer</summary>

**Penalty method:** Adds a term to the objective for constraint violations:
$$\min_x f(x) + \frac{\mu}{2}\sum_i [\max(0, g_i(x))]^2$$

- Approaches the constrained solution **from outside** (violates constraints, then reduces violation)
- Works for equality and inequality constraints
- Problem: as $\mu \to \infty$, the problem becomes ill-conditioned

**Barrier method:** Adds a term that goes to infinity at the constraint boundary:
$$\min_x f(x) - \frac{1}{t}\sum_i \log(-g_i(x))$$

- Stays **strictly inside** the feasible region at all times
- Only works for inequality constraints (need a feasible starting point)
- Foundation of interior-point methods

**When to choose:**
- **Penalty:** Don't have a feasible initial point; have equality constraints; simple implementation
- **Barrier/Interior-point:** Have a feasible start; need high-accuracy solutions; industrial solvers (IPOPT, MOSEK) use this internally

</details>

**A3.** A robot arm has joint limits: $-\pi \leq q_i \leq \pi$ for each joint $q_i$. You want to minimize the distance to a target. Formulate this as a constrained optimization problem and list which solver type (LP, QP, NLP) applies.

<details><summary>Answer</summary>

**Variables:** Joint angles $q = (q_1, ..., q_n)$

**Objective:** $\min_q \|FK(q) - p_{\text{target}}\|^2$ where $FK(q)$ is the forward kinematics (nonlinear function of $q$).

**Constraints:** $-\pi \leq q_i \leq \pi$ (box constraints) → equivalently: $q_i - \pi \leq 0$ and $-q_i - \pi \leq 0$.

**Solver type:** **NLP** — the objective is nonlinear (FK involves sin/cos of joint angles), and the constraints are linear (box constraints).

Suitable solvers: IPOPT, SNOPT, or even Ceres (with parameter bounds via `SetParameterLowerBound`/`SetParameterUpperBound`).

Actually, Ceres doesn't support general constraints, but box constraints on parameters are supported directly — and box constraints are all we need here.

</details>

---

## Section B — Computation Problems

**B1.** Implement the penalty method for:

$$\min_{x,y} (x - 3)^2 + (y - 2)^2 \quad \text{s.t.} \quad x + y \leq 3, \quad x \geq 0, \quad y \geq 0$$

```python
import numpy as np
from scipy.optimize import minimize

def objective(xy):
    x, y = xy
    return (x - 3)**2 + (y - 2)**2

def penalty_objective(xy, mu):
    """
    Objective + penalty for constraint violations.
    Constraints: x+y <= 3, x >= 0, y >= 0
    """
    x, y = xy
    cost = objective(xy)
    # TODO: add penalty terms for each constraint
    # Penalty for g(x) <= 0: (mu/2) * max(0, g(x))^2
    pass

# Solve with increasing penalty parameter
# TODO: loop mu from 1 to 1e6, solve each, report convergence
```

<details><summary>Solution</summary>

```python
import numpy as np
from scipy.optimize import minimize

def objective(xy):
    x, y = xy
    return (x - 3)**2 + (y - 2)**2

def penalty_objective(xy, mu):
    x, y = xy
    cost = objective(xy)
    # g1: x + y <= 3  →  x + y - 3 <= 0
    cost += (mu / 2) * max(0, x + y - 3)**2
    # g2: x >= 0  →  -x <= 0
    cost += (mu / 2) * max(0, -x)**2
    # g3: y >= 0  →  -y <= 0
    cost += (mu / 2) * max(0, -y)**2
    return cost

x0 = np.array([0.0, 0.0])
for mu in [1, 10, 100, 1000, 10000]:
    result = minimize(lambda xy: penalty_objective(xy, mu), x0, method='Nelder-Mead')
    x, y = result.x
    print(f"mu={mu:6d}: x={x:.4f}, y={y:.4f}, "
          f"cost={objective(result.x):.4f}, x+y={x+y:.4f}")
    x0 = result.x  # warm start

# Expected: converges to (2, 1) or close — the unconstrained minimum (3,2)
# violates x+y <= 3, so the solution is on the constraint boundary.
```

The unconstrained minimum is $(3, 2)$ but $3 + 2 = 5 > 3$, so the constraint $x + y \leq 3$ is active. The constrained solution is at approximately $(2, 1)$.

</details>

**B2.** Use `scipy.optimize.linprog` to solve a simple resource allocation:

A warehouse robot can carry boxes of type A (weight 2kg, value 5) or type B (weight 3kg, value 7). Capacity: 12kg. Max 3 of type A, max 4 of type B. Maximize total value.

<details><summary>Answer</summary>

```python
from scipy.optimize import linprog

# linprog minimizes c^T x, so negate for maximization
c = [-5, -7]  # negate values

# Inequality constraints: A_ub @ x <= b_ub
A_ub = [
    [2, 3],     # weight constraint: 2a + 3b <= 12
]
b_ub = [12]

# Bounds for each variable
x_bounds = [(0, 3), (0, 4)]  # 0 <= a <= 3, 0 <= b <= 4

result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=x_bounds, method='highs')
print(f"Type A: {result.x[0]:.1f}, Type B: {result.x[1]:.1f}")
print(f"Total value: {-result.fun:.1f}")

# Note: LP gives fractional solutions. For integer solutions, 
# need integer programming (scipy doesn't support this directly).
```

LP solution: $a = 1.5, b = 3.0$, value $= 28.5$. Since we can't carry half a box, the integer solution requires enumeration or an IP solver.

</details>

---

## Section C — OKS Connection

**C1.** The OKS MPC controller needs to compute velocity commands $(v, \omega)$ that:
- Track a reference path (minimize tracking error)
- Respect velocity limits: $|v| \leq v_{\max}$, $|\omega| \leq \omega_{\max}$
- Respect acceleration limits: $|\dot{v}| \leq a_{\max}$, $|\dot{\omega}| \leq \alpha_{\max}$
- Avoid obstacles (distance $\geq d_{\min}$)

1. Classify each constraint as linear or nonlinear
2. What optimization problem type is this? (LP/QP/SOCP/NLP)
3. Why is the obstacle constraint the hardest?

<details><summary>Answer</summary>

1. **Classification:**
   - Velocity limits: **linear** (box constraints on variables)
   - Acceleration limits: **linear** (box constraints on $v_{k+1} - v_k$)
   - Tracking error (objective): **nonlinear** if using the full kinematic model ($x_{k+1} = x_k + v_k \cos\theta_k \cdot dt$)
   - Obstacle avoidance: **nonlinear** ($\sqrt{(x - x_{\text{obs}})^2 + (y - y_{\text{obs}})^2} \geq d_{\min}$)

2. **NLP** — the kinematic model constraints are nonlinear (cos/sin of heading), and obstacle constraints are nonlinear distance functions. If you linearize the kinematics around the current trajectory, it becomes a QP (this is what many practical MPC implementations do).

3. **Obstacle constraint difficulty:**
   - It's non-convex (the feasible region has "holes")
   - In a cluttered environment, you need constraints for many obstacles
   - The constraint can appear/disappear as the robot moves (active set changes)
   - For moving obstacles, the constraint depends on time
   
   This is why many robots use a simpler approach: plan a path that avoids obstacles (using A*/RRT), then track that path with MPC that only handles kinematic constraints. OKS does this — path planning handles obstacles, the controller handles dynamics.

</details>
