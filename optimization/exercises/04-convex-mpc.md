# Exercises: Convex Optimization and MPC

### Chapter 05: Convex Optimization

**Self-assessment guide:** These exercises connect convex optimization directly to robot control. If you can formulate the MPC QP and interpret the dual variables, you have the key skills.

---

## Section A — Conceptual Questions

**A1.** Prove that a QP with a positive semidefinite Hessian is convex. What does this guarantee about its solutions?

$$\min_x \frac{1}{2}x^T Q x + c^T x \quad \text{s.t.} \quad Ax \leq b$$

<details><summary>Answer</summary>

The objective $f(x) = \frac{1}{2}x^T Q x + c^T x$ has Hessian $\nabla^2 f = Q$.

$Q \succeq 0$ (PSD) means $v^T Q v \geq 0$ for all $v$, which is exactly the condition for $f$ to be convex.

The constraint set $\{x : Ax \leq b\}$ is a polyhedron, which is always convex (intersection of half-spaces).

A convex function minimized over a convex set is a convex optimization problem. This guarantees:
1. **Every local minimum is a global minimum**
2. **The solution set is convex** (if multiple solutions exist, any convex combination is also optimal)
3. **KKT conditions are sufficient** (not just necessary)
4. **Efficient solvers exist** with polynomial-time complexity guarantees

If $Q \succ 0$ (strictly PD), the solution is **unique**.

</details>

**A2.** What is strong duality, and why does it matter practically?

<details><summary>Answer</summary>

For a convex optimization problem, **strong duality** means the optimal value of the dual problem equals the optimal value of the primal problem:

$$f^* = \min_x f(x) \text{ s.t. constraints} = \max_{\lambda, \mu} g(\lambda, \mu) \text{ (dual)}$$

**Why it matters:**
1. **Dual variables have physical meaning**: $\lambda_i^* = \frac{\partial f^*}{\partial b_i}$ — how much the optimal cost changes if you relax constraint $i$ by a small amount
2. **Certificates of optimality**: if you find primal $x$ and dual $(\lambda, \mu)$ with the same objective, both are optimal
3. **Decomposition**: some large problems are easier to solve in the dual
4. **Sensitivity analysis**: dual variables tell you which constraints are "expensive"

**For MPC:** The dual variable of a velocity constraint tells you how much better the tracking would be if you could go faster. This is useful for higher-level planning — "this segment needs more speed budget."

</details>

**A3.** Why is OSQP the solver of choice for MPC on embedded systems? What are its limitations?

<details><summary>Answer</summary>

**Advantages:**
- Solves QPs, which is what linear MPC produces
- **First-order method** (ADMM) — no matrix factorizations per iteration, just matrix-vector products
- Fixed memory footprint — no dynamic allocation (critical for embedded)
- Warm-starting — previous solution is close to the next (exploits temporal coherence)
- Handles infeasible problems gracefully (returns certificate)
- Open source, C code, no dependencies

**Limitations:**
- Only handles QPs (not nonlinear — need CasADi/IPOPT for NMPC)
- Lower accuracy than interior-point methods (typically $10^{-3}$ to $10^{-6}$)
- Convergence speed degrades with ill-conditioning
- Cannot handle second-order cone or semidefinite constraints
- For very high-accuracy solutions, MOSEK or Gurobi are faster

**For OKS:** OSQP or similar QP solvers are appropriate because the control problem is linearized at each timestep, producing a QP. The ~ms solve time fits within the control loop.

</details>

---

## Section B — Computation Problems

**B1.** Formulate and solve a simple MPC problem using CVXPY:

A 1D robot with state $(p, v)$ (position, velocity) and input $u$ (force/acceleration):

$$p_{k+1} = p_k + v_k \cdot dt, \quad v_{k+1} = v_k + u_k \cdot dt$$

Objective: drive to position $p = 10$ from $p_0 = 0, v_0 = 0$ in $N = 20$ steps with $dt = 0.1$.

Constraints: $|v| \leq 3$, $|u| \leq 2$.

```python
import cvxpy as cp
import numpy as np

N = 20      # horizon
dt = 0.1
p_target = 10.0

# Decision variables
p = cp.Variable(N + 1)  # position
v = cp.Variable(N + 1)  # velocity
u = cp.Variable(N)      # control input

# TODO: Define objective, constraints, solve
# Objective: minimize sum of tracking errors + small control effort
# cost = sum((p - p_target)^2) + 0.01 * sum(u^2)

# TODO: Plot the optimal trajectory (p, v, u vs time)
```

<details><summary>Solution</summary>

```python
import cvxpy as cp
import numpy as np

N = 20
dt = 0.1
p_target = 10.0

p = cp.Variable(N + 1)
v = cp.Variable(N + 1)
u = cp.Variable(N)

cost = 0
constraints = []

# Initial conditions
constraints += [p[0] == 0, v[0] == 0]

for k in range(N):
    # Dynamics
    constraints += [p[k+1] == p[k] + v[k] * dt]
    constraints += [v[k+1] == v[k] + u[k] * dt]
    # Velocity limits
    constraints += [v[k] >= -3, v[k] <= 3]
    # Control limits
    constraints += [u[k] >= -2, u[k] <= 2]

constraints += [v[N] >= -3, v[N] <= 3]

# Objective: track target + small control effort + terminal velocity penalty
cost = cp.sum_squares(p - p_target) + 0.01 * cp.sum_squares(u) + 10 * cp.square(v[N])

prob = cp.Problem(cp.Minimize(cost), constraints)
prob.solve(solver=cp.OSQP)

print(f"Status: {prob.status}")
print(f"Final position: {p.value[-1]:.2f}")
print(f"Final velocity: {v.value[-1]:.2f}")
print(f"Optimal cost: {prob.value:.2f}")

# Print trajectory
for k in range(0, N+1, 5):
    t = k * dt
    print(f"t={t:.1f}: p={p.value[k]:.2f}, v={v.value[k]:.2f}", end="")
    if k < N:
        print(f", u={u.value[k]:.2f}")
    else:
        print()
```

The robot accelerates at max ($u = 2$), coasts at max velocity ($v = 3$), then decelerates to stop near $p = 10$.

</details>

**B2.** Extend B1 to 2D with a unicycle model (linearized). Add an obstacle at $(5, 2)$ with radius $1$.

Hint: For a linearized unicycle at each step, you can approximate the nonlinear dynamics. Or use the double integrator model in 2D (simpler, captures the key MPC structure).

<details><summary>Hint</summary>

For the 2D extension with obstacle avoidance, the obstacle constraint $\|(p_x, p_y) - (5, 2)\| \geq 1$ is non-convex. Two approaches:

1. **Linearize the constraint** at the current position:
   $$\hat{d}^T (p_k - p_{\text{obs}}) \geq 1$$ 
   where $\hat{d}$ is the unit vector from obstacle to current position.

2. **Use SOC constraint** (exact but second-order cone):
   ```python
   constraints += [cp.norm(cp.vstack([px[k] - 5, py[k] - 2])) >= 1]
   ```
   This makes it an SOCP, still convex!

Note: CVXPY can handle SOC constraints directly.

</details>

---

## Section C — OKS Connection

**C1.** The OKS controller receives a reference path as a sequence of waypoints $(x_i, y_i, \theta_i)$. It must produce $(v, \omega)$ commands at 10 Hz.

1. How would you formulate this as an MPC problem?
2. What is the prediction horizon in time if you use $N = 10$ steps?
3. Why does the controller need to re-solve the MPC at every timestep (receding horizon)?
4. What happens if the QP solver doesn't converge in time?

<details><summary>Answer</summary>

1. **MPC Formulation:**
   - State: $(x, y, \theta, v, \omega)$ — position, heading, velocities
   - Input: $(a, \alpha)$ — linear and angular acceleration commands
   - Dynamics: linearized unicycle (or discrete kinematic model)
   - Objective: $\sum_{k=0}^{N-1} \|s_k - s_{\text{ref},k}\|^2_Q + \|u_k\|^2_R + \|s_N - s_{\text{ref},N}\|^2_{Q_f}$
   - Constraints: velocity bounds, acceleration bounds, (optionally) obstacle margins

2. $N = 10$ at 10 Hz = **1 second** prediction horizon.

3. **Receding horizon** is necessary because:
   - The model is imperfect — actual trajectory diverges from predicted
   - Disturbances (wheel slip, delays) are not predicted
   - New information (updated path, new obstacles) arrives
   - Only the first control action is applied; the rest are replanned
   - This provides implicit feedback — errors are corrected at the next solve

4. **If the solver doesn't converge:**
   - Option 1: Use the last successful solution (stale control — dangerous if sustained)
   - Option 2: Apply a safe default (e.g., decelerate, stop)
   - Option 3: Use a warm-started partial solution (OSQP can return after fixed iterations)
   - OKS likely has a watchdog: if no valid command in $X$ ms → emergency stop
   - This is why solve time must be **deterministic** and **bounded** — real-time constraint

</details>

**C2.** The dual variable of the velocity constraint in the MPC tells you the "shadow price" of the speed limit. In what real OKS scenario would this information be useful?

<details><summary>Answer</summary>

**Scenario: Dynamic speed zones in the warehouse.**

Different areas have different speed limits (near humans: 0.3 m/s, open aisles: 1.0 m/s). If the dual variable of the speed constraint is large in a particular zone, it means:
- The robot is being significantly slowed down by that constraint
- Relaxing the speed limit there would substantially improve throughput

**Practical use:**
- Fleet optimization can identify bottleneck zones where speed limits are too conservative
- Safety analysis: large dual variables mean the robot is "pushing against" the limit — if the limit were suddenly removed (software bug), the robot would immediately accelerate
- Warehouse layout optimization: if a speed zone has consistently high dual variables, consider redesigning the zone (wider aisle, better visibility) to allow higher speeds safely

This connects optimization theory to real operational decisions.

</details>
