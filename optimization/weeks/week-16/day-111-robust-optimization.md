# Day 111: Robust Optimization as Games Against Nature

> **Phase VII — Game Theory for Optimization & Robotics | Week 16 | 2.5 hours**
> *When uncertainty is the adversary, minimax gives you the strategy that survives the worst case.*

## Navigation
- Previous: [Day 110: Multi-Robot Games + Adversarial ML](day-110-multi-robot-adversarial.md)
- Next: [Day 112: Capstone — Multi-Robot Delivery Game](day-112-week16-capstone.md)
- [Week 16 Overview](../../../CURRICULUM.md)
- [Chapter 10: Game Theory Reference](../../../10-game-theory.md)
- [Full Curriculum](../../../00-learning-plan.md)

## OKS Relevance
OKS AMRs face persistent uncertainty: obstacle positions shift, sensor readings drift, motor responses vary with battery level, and network latency fluctuates. Robust optimization frames the controller as a player against "nature" who picks the worst-case disturbance. This yields path plans that work even under worst-case obstacle motion and motor controllers that maintain stability despite parameter uncertainty — exactly the guarantees needed for 24/7 warehouse autonomy.

---

## Theory (45 min)

### 111.1 Minimax Formulation

The **robust optimization** problem:

$$\min_{x \in \mathcal{X}} \max_{\delta \in \mathcal{U}} f(x, \delta)$$

where $x$ is the decision, $\delta$ is the uncertainty, and $\mathcal{U}$ is the **uncertainty set**. This is a two-player zero-sum game: the decision-maker minimizes, nature maximizes.

Common uncertainty sets:
- **Box**: $\mathcal{U} = \{\delta : \|\delta\|_\infty \leq \rho\}$
- **Ellipsoidal**: $\mathcal{U} = \{\delta : \delta^\top \Sigma^{-1} \delta \leq \rho^2\}$
- **Polyhedral**: $\mathcal{U} = \{\delta : A\delta \leq b\}$

### 111.2 Sion's Minimax Theorem

**Theorem (Sion 1958)**: If $\mathcal{X}$ is compact convex, $\mathcal{U}$ is compact convex, $f$ is convex in $x$ and concave in $\delta$ (or more generally, quasi-convex-concave), then:

$$\min_{x \in \mathcal{X}} \max_{\delta \in \mathcal{U}} f(x, \delta) = \max_{\delta \in \mathcal{U}} \min_{x \in \mathcal{X}} f(x, \delta)$$

This guarantees the saddle point exists and the minimax equals the maximin — the game has a well-defined value.

### 111.3 Robust Linear Programming

A nominal LP: $\min c^\top x$ s.t. $Ax \leq b$. With uncertain $A$:

$$\min_{x} c^\top x \quad \text{s.t. } (A + \Delta)x \leq b \;\; \forall \Delta \in \mathcal{U}_A$$

For **ellipsoidal uncertainty** $\Delta_i \in \{\delta : \|\delta\| \leq \rho_i\}$ on each row:

$$a_i^\top x + \rho_i \|x\| \leq b_i$$

This is a **second-order cone program** (SOCP) — efficiently solvable.

### 111.4 H∞ Control and Robust MPC

**H∞ control** minimizes the worst-case gain from disturbance $w$ to output $z$:

$$\min_K \left\| T_{zw}(K) \right\|_\infty = \min_K \sup_\omega \bar{\sigma}(T_{zw}(e^{j\omega}))$$

For a system $\dot{x} = Ax + Bu + Ew$, $z = Cx + Du$, finding the H∞-optimal controller is equivalent to solving a minimax game:

$$\min_u \max_w \int_0^\infty (z^\top z - \gamma^2 w^\top w) \, dt$$

The controller achieves $\|T_{zw}\|_\infty < \gamma$ iff the associated Riccati equation has a stabilizing solution.

**Robust MPC**: At each step, solve:

$$\min_{\mathbf{u}} \max_{\mathbf{w} \in \mathcal{W}^N} \sum_{k=0}^{N-1} \ell(x_k, u_k) + V_f(x_N)$$

subject to $x_{k+1} = Ax_k + Bu_k + w_k$, constraints $\forall w \in \mathcal{W}$.

---

## Implementation (60 min)

```python
"""Robust Optimization: minimax formulations and robust control."""
import numpy as np
from scipy.optimize import minimize, linprog

def robust_linear_problem(c: np.ndarray, A: np.ndarray, b: np.ndarray,
                          rho: float) -> dict:
    """Solve robust LP with ellipsoidal uncertainty on constraint matrix.
    
    min c^T x  s.t. a_i^T x + rho * ||x|| <= b_i for all i
    
    Approximated via penalty-based approach.
    """
    m, n = A.shape
    
    def objective(x):
        return c @ x
    
    def constraint_i(x, i):
        return b[i] - A[i] @ x - rho * np.linalg.norm(x)
    
    constraints = [{'type': 'ineq', 'fun': constraint_i, 'args': (i,)}
                   for i in range(m)]
    
    x0 = np.zeros(n)
    result = minimize(objective, x0, method='SLSQP', constraints=constraints)
    
    return {
        'x': result.x,
        'obj': result.fun,
        'success': result.success,
    }

def minimax_game(f, grad_x_f, grad_delta_f, x0, delta0,
                 x_bounds, delta_bounds, lr=0.01, T=1000):
    """Gradient descent-ascent for minimax: min_x max_delta f(x, delta).
    
    Args:
        f: objective function
        grad_x_f: gradient w.r.t. x
        grad_delta_f: gradient w.r.t. delta
        x_bounds, delta_bounds: box constraints
    """
    x = np.array(x0, dtype=float)
    delta = np.array(delta0, dtype=float)
    
    trajectory = [(x.copy(), delta.copy(), f(x, delta))]
    
    for t in range(T):
        # Gradient descent on x
        gx = grad_x_f(x, delta)
        x -= lr * gx
        x = np.clip(x, *x_bounds)
        
        # Gradient ascent on delta
        gd = grad_delta_f(x, delta)
        delta += lr * gd
        delta = np.clip(delta, *delta_bounds)
        
        trajectory.append((x.copy(), delta.copy(), f(x, delta)))
    
    return {
        'x': x, 'delta': delta,
        'value': f(x, delta),
        'trajectory': trajectory,
    }

def robust_mpc_step(A: np.ndarray, B: np.ndarray, Q: np.ndarray,
                    R: np.ndarray, x0: np.ndarray, w_bound: float,
                    N: int = 5) -> np.ndarray:
    """One step of robust MPC via minimax over disturbance.
    
    Simplified: optimize u[0] against worst-case constant disturbance.
    """
    nx, nu = B.shape
    
    def cost(u_flat):
        """Worst-case cost for given control sequence."""
        u_seq = u_flat.reshape(N, nu)
        worst_cost = -np.inf
        
        # Sample worst-case disturbances (approximate)
        for _ in range(50):
            w_seq = w_bound * np.random.choice([-1, 1], (N, nx))
            x = x0.copy()
            total = 0.0
            for k in range(N):
                total += x @ Q @ x + u_seq[k] @ R @ u_seq[k]
                x = A @ x + B @ u_seq[k] + w_seq[k]
            total += x @ Q @ x  # terminal
            worst_cost = max(worst_cost, total)
        
        return worst_cost
    
    u0 = np.zeros(N * nu)
    result = minimize(cost, u0, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-6})
    return result.x[:nu]  # return first control action

# Demo 1: Robust LP
print("=== Robust Linear Problem ===")
c = np.array([1.0, 2.0])
A = np.array([[1.0, 1.0], [2.0, 0.5]])
b = np.array([4.0, 6.0])

for rho in [0.0, 0.1, 0.5, 1.0]:
    result = robust_linear_problem(c, A, b, rho)
    print(f"  rho={rho:.1f}: x={np.round(result['x'], 3)}, "
          f"obj={result['obj']:.3f}")

# Demo 2: Minimax on quadratic
print("\n=== Minimax Game ===")
# f(x, delta) = (x + delta)^2 + x^2 - delta^2
# Player min over x, nature max over delta
def f(x, d): return (x[0] + d[0])**2 + x[0]**2 - d[0]**2
def gx(x, d): return np.array([2*(x[0] + d[0]) + 2*x[0]])
def gd(x, d): return np.array([2*(x[0] + d[0]) - 2*d[0]])

result_mm = minimax_game(f, gx, gd, [2.0], [1.0],
                         (-5, 5), (-2, 2), lr=0.05, T=500)
print(f"  Saddle point: x={result_mm['x'][0]:.4f}, "
      f"delta={result_mm['delta'][0]:.4f}, value={result_mm['value']:.4f}")

# Demo 3: Robust MPC
print("\n=== Robust MPC ===")
A_sys = np.array([[1.0, 0.1], [0.0, 1.0]])
B_sys = np.array([[0.0], [0.1]])
Q = np.eye(2)
R = np.array([[0.1]])
x0 = np.array([2.0, 0.5])

for w_bound in [0.0, 0.1, 0.5]:
    u_opt = robust_mpc_step(A_sys, B_sys, Q, R, x0, w_bound)
    print(f"  w_bound={w_bound:.1f}: u_opt={u_opt[0]:.4f}")

# Compare nominal vs robust trajectories
print("\n  Trajectory comparison (w_bound=0.3):")
np.random.seed(0)
x_nom, x_rob = x0.copy(), x0.copy()
for k in range(10):
    w = 0.3 * np.random.choice([-1, 1], 2)
    u_nom = robust_mpc_step(A_sys, B_sys, Q, R, x_nom, 0.0)
    u_rob = robust_mpc_step(A_sys, B_sys, Q, R, x_rob, 0.3)
    x_nom = A_sys @ x_nom + B_sys.flatten() * u_nom[0] + w
    x_rob = A_sys @ x_rob + B_sys.flatten() * u_rob[0] + w
print(f"    Nominal final: {np.round(x_nom, 3)}")
print(f"    Robust final:  {np.round(x_rob, 3)}")
```

---

## Practice Problems (45 min)

**P1**: For the robust LP with $c = (1, 1)$, $A = I_2$, $b = (3, 3)$, and ellipsoidal uncertainty $\rho = 0.5$, find the robust optimal solution analytically.

<details><summary>Answer</summary>

Robust constraint: $x_i + 0.5\|x\| \leq 3$. By symmetry, $x_1 = x_2 = x^*$. Then $x^* + 0.5\sqrt{2}x^* \leq 3$, so $x^*(1 + \sqrt{2}/2) \leq 3$, giving $x^* = 3/(1 + \sqrt{2}/2) \approx 1.76$. Nominal solution is $x^* = 3$. Robustness costs: objective increases from 6 to $\approx 3.51$.
</details>

**P2**: Prove that for a linear objective $f(x, \delta) = c^\top x + \delta^\top x$ with $\|\delta\| \leq \rho$, the robust problem reduces to $\min_x c^\top x + \rho\|x\|$.

<details><summary>Answer</summary>

$\max_{\|\delta\| \leq \rho} \delta^\top x = \rho \|x\|$ by Cauchy-Schwarz, achieved at $\delta^* = \rho x / \|x\|$. Substituting: $\min_x \max_\delta (c^\top x + \delta^\top x) = \min_x (c^\top x + \rho\|x\|)$. This is a regularized optimization problem — robustness acts as a norm penalty, connecting to regularization in machine learning.
</details>

**P3**: Explain the relationship between $H_\infty$ control and the minimax game. What does $\gamma$ represent?

<details><summary>Answer</summary>

$\gamma$ is the **worst-case amplification gain**: the maximum ratio $\|z\|/\|w\|$ over all disturbances. The controller plays $u$ to minimize $\gamma$; nature plays $w$ to maximize $\|z\|$. Achieving $\|T_{zw}\|_\infty < \gamma$ means the controller guarantees the disturbance never gets amplified by more than $\gamma$. Smaller $\gamma$ = more robust but more conservative; at $\gamma = \gamma^*_{\min}$, the game reaches its value.
</details>

---

## Expert Challenges

**E1**: Implement **distributionally robust optimization** (DRO): $\min_x \max_{P \in \mathcal{P}} \mathbb{E}_P[f(x, \xi)]$ where $\mathcal{P}$ is a Wasserstein ball around the empirical distribution.

<details><summary>Hint</summary>

For linear $f$ and Wasserstein-1 ball of radius $\rho$: DRO reduces to $\min_x \{c^\top x + \rho \|\nabla_\xi f(x, \cdot)\|_*\}$ where $\|\cdot\|_*$ is the dual norm. Implement by sampling $\xi$, computing the worst-case expectation via LP, and optimizing $x$ in the outer loop.
</details>

**E2**: Design a robust path planner for an AMR. The obstacle positions are uncertain within ellipsoidal sets. Formulate as minimax and solve.

<details><summary>Answer</summary>

Decision: waypoints $x_1, \ldots, x_K$. Uncertainty: obstacle center $o_j \in \{o_j^0 + \delta : \|\delta\| \leq r_j\}$. Constraint: $\|x_k - o_j\| \geq d_{\text{safe}} + r_j$ for all $k, j$ (worst-case obstacle position). This is a robust constraint: $\min_{k} \|x_k - o_j^0\| - r_j \geq d_{\text{safe}}$. Solve via sequential convex programming with robust constraints.
</details>

**E3**: Compare robust optimization vs. stochastic optimization for a robot's charging schedule. When does robustness help vs. hurt?

<details><summary>Answer</summary>

Robust: $\min_u \max_{d \in \mathcal{U}} \text{cost}(u, d)$ — prepares for worst-case demand. Stochastic: $\min_u \mathbb{E}_d[\text{cost}(u, d)]$ — optimizes average case. Robustness helps when: (1) failure cost is very high (safety-critical), (2) distribution is poorly known, (3) worst-case is not too far from average. It hurts when: (1) worst case is extremely unlikely, (2) being overly conservative wastes resources, (3) the uncertainty set is too large. For charging: robust for minimum-charge constraints (safety), stochastic for scheduling optimization (efficiency).
</details>

---

## Connections
- **Back**: [Day 110: Adversarial ML](day-110-multi-robot-adversarial.md) — adversarial training is minimax
- **Back**: [Day 96: Zero-Sum & Minimax](../week-14/day-96-zero-sum-minimax.md) — zero-sum foundations
- **Forward**: [Day 112: Capstone](day-112-week16-capstone.md) — robust rerouting under edge failures
- **OKS**: Robust path planning, H∞ motor control, worst-case sensor uncertainty

## Self-Check
- [ ] I can formulate a robust optimization problem as a minimax game
- [ ] I understand Sion's minimax theorem and when saddle points exist
- [ ] I know how ellipsoidal uncertainty transforms LP into SOCP
- [ ] I can connect H∞ control to game theory and explain the role of $\gamma$
