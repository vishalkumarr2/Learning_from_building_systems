# Day 101: Bargaining Theory

> **Phase VII — Game Theory for Optimization & Robotics | Week 15 | 2.5 hours**
> *Two players, one surplus — axioms alone determine the split.*

## Navigation
- **Previous:** [Day 100: Shapley Value and Fair Division](day-100-shapley-value.md)
- **Next:** [Day 102: Dynamic Games and Backward Induction](day-102-dynamic-games.md)
- [Week 15 Overview](../../../CURRICULUM.md)
- [Chapter 10: Game Theory Reference](../../../10-game-theory.md)
- [Full Curriculum](../../../00-learning-plan.md)

## OKS Relevance
A robot and the BEC scheduler negotiate over charging time: the robot wants a full charge (more uptime), the BEC wants to minimize station occupancy (serve more robots). Bargaining theory provides principled solutions that balance both objectives — Nash for product-fairness, Kalai-Smorodinsky for proportional fairness.

---

## Theory (45 min)

### 101.1 The Bargaining Problem

A **bargaining problem** is a pair $(F, d)$ where:
- $F \subset \mathbb{R}^2$ is the **feasible set** — compact, convex set of achievable utility pairs.
- $d = (d_1, d_2) \in F$ is the **disagreement point** — the outcome if no agreement is reached.
- There exists $u \in F$ with $u > d$ (componentwise) — cooperation is beneficial.

A **bargaining solution** $f: \mathcal{B} \to \mathbb{R}^2$ maps each problem to an outcome in $F$.

### 101.2 Nash Bargaining Solution (NBS)

Nash (1950) proposed four axioms: **Pareto efficiency**, **symmetry**, **independence of irrelevant alternatives (IIA)**, and **invariance to affine transformations**.

The unique solution satisfying all four:

$$f^N(F, d) = \arg\max_{u \in F,\, u \geq d} (u_1 - d_1)(u_2 - d_2)$$

This is the **Nash product** — maximizing the product of surpluses over the disagreement point.

For a linear feasible set $u_1 + u_2 \leq c$ with $d = (0, 0)$: $f^N = (c/2, c/2)$.

**Asymmetric Nash bargaining** with bargaining powers $\alpha, \beta > 0$, $\alpha + \beta = 1$:

$$f^{AN} = \arg\max_{u \in F,\, u \geq d} (u_1 - d_1)^\alpha (u_2 - d_2)^\beta$$

### 101.3 Kalai-Smorodinsky Solution (KS)

Replace IIA with **monotonicity**: if $F \subseteq G$ and the maximum achievable utility for each player is the same, then the solution in $G$ should be at least as good for both.

The **utopia point** $\bar{u}$: $\bar{u}_i = \max\{u_i : u \in F, u \geq d\}$.

The KS solution is the maximal point in $F$ on the line from $d$ to $\bar{u}$:

$$f^{KS}(F, d) = \max\{t \geq 0 : d + t(\bar{u} - d) \in F\}$$

This ensures proportionality: gains are split in proportion to ideal gains.

### 101.4 Rubinstein Alternating Offers

A dynamic model: players alternate proposals over discrete rounds with discount factors $\delta_1, \delta_2 \in (0,1)$.

**Subgame perfect equilibrium** (Rubinstein 1982): Player 1 offers $x_1^*$ in round 1, immediately accepted:

$$x_1^* = \frac{1 - \delta_2}{1 - \delta_1 \delta_2}, \quad x_2^* = \frac{\delta_2(1 - \delta_1)}{1 - \delta_1 \delta_2}$$

As $\delta_1, \delta_2 \to 1$, the split converges to the Nash bargaining solution.

---

## Implementation (60 min)

```python
"""Bargaining theory — Day 101"""
import numpy as np
from scipy.optimize import minimize_scalar, minimize
import matplotlib.pyplot as plt


def nash_bargaining(F_boundary, d, n_points=1000):
    """Compute Nash bargaining solution on a parametric boundary.
    
    Args:
        F_boundary: function t -> (u1, u2) parametrizing Pareto frontier, t in [0,1]
        d: disagreement point (d1, d2)
    """
    best_t, best_product = 0, -np.inf
    for t in np.linspace(0, 1, n_points):
        u1, u2 = F_boundary(t)
        if u1 >= d[0] and u2 >= d[1]:
            product = (u1 - d[0]) * (u2 - d[1])
            if product > best_product:
                best_product = product
                best_t = t
    return F_boundary(best_t)


def kalai_smorodinsky(F_boundary, d, n_points=1000):
    """Compute Kalai-Smorodinsky solution."""
    # Find utopia point
    u1_max = max(F_boundary(t)[0] for t in np.linspace(0, 1, n_points)
                 if F_boundary(t)[1] >= d[1])
    u2_max = max(F_boundary(t)[1] for t in np.linspace(0, 1, n_points)
                 if F_boundary(t)[0] >= d[0])
    utopia = (u1_max, u2_max)
    
    # Find intersection of line d->utopia with Pareto frontier
    direction = (utopia[0] - d[0], utopia[1] - d[1])
    best_t_line, best_scale = 0, 0
    for t in np.linspace(0, 1, n_points):
        u1, u2 = F_boundary(t)
        if abs(direction[0]) > 1e-10:
            s = (u1 - d[0]) / direction[0]
        else:
            s = (u2 - d[1]) / direction[1]
        predicted_u2 = d[1] + s * direction[1]
        if abs(u2 - predicted_u2) < 0.05 * max(abs(u2_max), 1) and s > best_scale:
            best_scale = s
            best_t_line = t
    return F_boundary(best_t_line), utopia


def rubinstein_equilibrium(delta1, delta2, pie=1.0):
    """Rubinstein alternating offers equilibrium."""
    x1 = pie * (1 - delta2) / (1 - delta1 * delta2)
    x2 = pie - x1
    return x1, x2


def simulate_rubinstein(delta1, delta2, rounds=20, pie=1.0):
    """Simulate alternating offers with impatience."""
    x1_eq, x2_eq = rubinstein_equilibrium(delta1, delta2, pie)
    offers, values = [], []
    for r in range(rounds):
        if r % 2 == 0:  # Player 1 proposes
            offer = (x1_eq, pie - x1_eq)
        else:  # Player 2 proposes
            offer = (pie - x2_eq, x2_eq)
        discount = delta1**r if r % 2 == 0 else delta2**r
        offers.append(offer)
        values.append((offer[0] * delta1**r, offer[1] * delta2**r))
    return offers, values, (x1_eq, x2_eq)


def visualize_bargaining(F_boundary, d, n_points=500):
    """Visualize all bargaining solutions."""
    ts = np.linspace(0, 1, n_points)
    frontier = np.array([F_boundary(t) for t in ts])
    
    nbs = nash_bargaining(F_boundary, d)
    ks, utopia = kalai_smorodinsky(F_boundary, d)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill(frontier[:, 0], frontier[:, 1], alpha=0.15, color='blue')
    ax.plot(frontier[:, 0], frontier[:, 1], 'b-', linewidth=2, label='Pareto frontier')
    ax.plot(*d, 'kx', markersize=12, markeredgewidth=3, label=f'Disagreement {d}')
    ax.plot(*nbs, 'ro', markersize=10, label=f'Nash ({nbs[0]:.2f}, {nbs[1]:.2f})')
    ax.plot(*ks, 'gs', markersize=10, label=f'KS ({ks[0]:.2f}, {ks[1]:.2f})')
    ax.plot(*utopia, 'm^', markersize=10, label=f'Utopia {utopia[0]:.2f}, {utopia[1]:.2f}')
    ax.plot([d[0], utopia[0]], [d[1], utopia[1]], 'g--', alpha=0.5)
    ax.set_xlabel('Player 1 utility'); ax.set_ylabel('Player 2 utility')
    ax.set_title('Bargaining Solutions'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('bargaining.png', dpi=100); plt.show()


# --- Demo ---
if __name__ == "__main__":
    # Pareto frontier: quarter-circle
    F = lambda t: (4 * np.cos(t * np.pi / 2), 4 * np.sin(t * np.pi / 2))
    d = (1.0, 1.0)
    
    nbs = nash_bargaining(F, d)
    ks, utopia = kalai_smorodinsky(F, d)
    print(f"Nash bargaining:    ({nbs[0]:.3f}, {nbs[1]:.3f})")
    print(f"Kalai-Smorodinsky:  ({ks[0]:.3f}, {ks[1]:.3f})")
    print(f"Utopia point:       ({utopia[0]:.3f}, {utopia[1]:.3f})")
    
    # Rubinstein
    for d1, d2 in [(0.9, 0.9), (0.9, 0.5), (0.5, 0.9)]:
        x1, x2 = rubinstein_equilibrium(d1, d2)
        print(f"Rubinstein δ=({d1},{d2}): P1={x1:.3f}, P2={x2:.3f}")
    
    visualize_bargaining(F, d)
```

---

## Practice Problems (45 min)

**P1.** For $F = \{(u_1, u_2) : u_1 + u_2 \leq 10, u_1, u_2 \geq 0\}$ and $d = (2, 3)$, find the Nash bargaining solution.

<details><summary>Answer</summary>

Maximize $(u_1 - 2)(u_2 - 3)$ subject to $u_1 + u_2 = 10$. Substituting $u_2 = 10 - u_1$: maximize $(u_1 - 2)(7 - u_1)$. FOC: $7 - 2u_1 + 2 = 0 \Rightarrow u_1 = 4.5, u_2 = 5.5$. NBS = $(4.5, 5.5)$.
</details>

**P2.** With $F = \{(u_1, u_2) : u_1^2 + u_2^2 \leq 16\}$ and $d = (0,0)$, compare NBS and KS.

<details><summary>Answer</summary>

Symmetric problem. Utopia = $(4, 4)$ (infeasible but defines direction). KS line is $u_1 = u_2$, intersects circle at $(2\sqrt{2}, 2\sqrt{2})$. NBS: maximize $u_1 u_2$ on circle → same point by symmetry. **Both coincide at** $(2\sqrt{2}, 2\sqrt{2})$.
</details>

**P3.** In Rubinstein's model, if $\delta_1 = 0.95$ and $\delta_2 = 0.80$, what is the equilibrium split? Who benefits from patience?

<details><summary>Answer</summary>

$x_1 = (1 - 0.80)/(1 - 0.76) = 0.20/0.24 = 0.833$, $x_2 = 0.167$. Player 1 (more patient) gets more. **Patience is power.**
</details>

---

## Expert Challenges

**E1.** Prove that the Nash bargaining solution is the only solution satisfying all four Nash axioms.

<details><summary>Hint</summary>

Use affine invariance to normalize to a symmetric problem, apply symmetry to get equal split, then use IIA to extend to general problems.
</details>

**E2.** Implement a multi-round Rubinstein simulation with stochastic breakdown: at each round, negotiations fail with probability $p$. Show convergence to NBS as $p \to 0$.

<details><summary>Hint</summary>

Breakdown probability acts like discounting: $\delta_i' = \delta_i(1-p)$. As $p \to 0$, equilibrium converges to Nash product maximizer.
</details>

**E3.** For an asymmetric NBS with powers $\alpha = 0.7, \beta = 0.3$ on the linear feasible set $u_1 + u_2 \leq 1, d = (0,0)$, find the solution and interpret.

<details><summary>Hint</summary>

Maximize $u_1^{0.7} u_2^{0.3}$ subject to $u_1 + u_2 = 1$. Lagrange: $0.7/u_1 = 0.3/u_2$ → $u_1 = 0.7, u_2 = 0.3$. Power = share.
</details>

---

## Connections
- **Prerequisites:** Cooperative games (Day 99) — bargaining is the 2-player cooperative case; Nash equilibrium (Day 93) — Nash used equilibrium ideas
- **Forward:** Dynamic games (Day 102) — Rubinstein embeds bargaining in extensive form; Mechanism design (Day 104) — mechanisms must satisfy participation constraints
- **OKS:** Robot-BEC charging negotiation; bandwidth allocation between competing robot tasks

## Self-Check
- [ ] I can compute the Nash bargaining solution by maximizing the Nash product
- [ ] I understand the difference between NBS and Kalai-Smorodinsky
- [ ] I can derive Rubinstein's equilibrium for given discount factors
- [ ] I understand how patience affects bargaining power
