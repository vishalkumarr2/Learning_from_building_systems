# Day 99: Cooperative Games and the Core

> **Phase VII — Game Theory for Optimization & Robotics | Week 15 | 2.5 hours**
> *When players form coalitions, the question shifts from "what's my best move?" to "how do we split the gains?"*

## Navigation
- **Previous:** [Day 98: Week 14 Review](../week-14/day-98-week14-review.md)
- **Next:** [Day 100: Shapley Value and Fair Division](day-100-shapley-value.md)
- **Week:** [Week 15 Overview](../README.md)
- **Chapter:** [Chapter 10: Game Theory Reference](../../10-game-theory.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
A fleet of OKS robots is a natural cooperative game. A coalition of 3 robots can handle a multi-pick order faster than any single robot. The core tells us which reward allocations are stable — no subset of robots has an incentive to break away and form its own team. This is the foundation for stable task-sharing protocols in warehouse robotics.

---

## Theory (45 min)

### 99.1 Transferable Utility (TU) Games

A **cooperative game with transferable utility** is a pair $(N, v)$ where:
- $N = \{1, 2, \ldots, n\}$ is the set of players.
- $v : 2^N \to \mathbb{R}$ is the **characteristic function** with $v(\emptyset) = 0$.

For any coalition $S \subseteq N$, $v(S)$ is the total value the coalition can achieve by cooperating, regardless of what outsiders do.

A game is **superadditive** if for all disjoint $S, T \subseteq N$:

$$v(S \cup T) \geq v(S) + v(T)$$

Superadditivity means cooperation never hurts — the grand coalition $N$ achieves the highest total value.

### 99.2 Imputations and the Core

An **allocation** (payoff vector) is $x = (x_1, \ldots, x_n) \in \mathbb{R}^n$.

An **imputation** satisfies:
- **Efficiency:** $\sum_{i \in N} x_i = v(N)$
- **Individual rationality:** $x_i \geq v(\{i\})$ for all $i$

The **core** is the set of imputations that no coalition can improve upon:

$$\text{Core}(v) = \left\{ x \in \mathbb{R}^n : \sum_{i \in N} x_i = v(N), \; \sum_{i \in S} x_i \geq v(S) \;\; \forall S \subseteq N \right\}$$

### 99.3 Bondareva-Shapley Theorem

A game has a **nonempty core** if and only if it is **balanced**.

A collection of coalitions $\mathcal{B}$ with weights $\delta_S \geq 0$ is a **balanced collection** if $\sum_{S \in \mathcal{B}: i \in S} \delta_S = 1$ for all $i \in N$.

**Theorem (Bondareva-Shapley):** The core of $(N, v)$ is nonempty iff for every balanced collection $(\mathcal{B}, \delta)$:

$$\sum_{S \in \mathcal{B}} \delta_S \, v(S) \leq v(N)$$

For **convex games** ($v(S \cup T) + v(S \cap T) \geq v(S) + v(T)$ for all $S, T$), the core is always nonempty and equals the convex hull of marginal vectors.

---

## Implementation (60 min)

```python
"""Cooperative games and the core — Day 99"""
import numpy as np
from itertools import combinations
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def characteristic_function(coalitions_values: dict) -> callable:
    """Create a characteristic function from a dictionary."""
    def v(S):
        return coalitions_values.get(frozenset(S), 0)
    return v


def all_coalitions(N):
    """Generate all non-empty subsets of N."""
    players = list(N)
    for r in range(1, len(players) + 1):
        for combo in combinations(players, r):
            yield frozenset(combo)


def is_in_core(x, N, v):
    """Check if allocation x is in the core."""
    n = len(N)
    players = sorted(N)
    # Efficiency check
    if abs(sum(x[i] for i in range(n)) - v(frozenset(players))) > 1e-9:
        return False, "Fails efficiency"
    # Coalition rationality
    for S in all_coalitions(N):
        indices = [players.index(p) for p in S]
        if sum(x[i] for i in indices) < v(S) - 1e-9:
            return False, f"Blocked by coalition {set(S)}"
    return True, "In core"


def find_core_allocation(N, v):
    """Find a core allocation via LP (minimize x_1 subject to core constraints)."""
    players = sorted(N)
    n = len(players)
    coalitions = list(all_coalitions(N))
    # min c^T x  s.t.  A_ub x <= b_ub,  A_eq x = b_eq
    c = np.zeros(n); c[0] = 1  # minimize x_1 (arbitrary)
    A_ub, b_ub = [], []
    for S in coalitions:
        if S == frozenset(players):
            continue
        row = np.zeros(n)
        for p in S:
            row[players.index(p)] = -1  # -sum >= -v(S)
        A_ub.append(row)
        b_ub.append(-v(S))
    A_eq = np.ones((1, n))
    b_eq = np.array([v(frozenset(players))])
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=[(None, None)] * n, method='highs')
    if result.success:
        return result.x
    return None


def visualize_core_3player(v_dict):
    """Visualize core for a 3-player game on the simplex."""
    v = characteristic_function(v_dict)
    V_N = v(frozenset([1, 2, 3]))
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    # Simplex corners: player gets all surplus
    corners = np.array([[V_N, 0, 0], [0, V_N, 0], [0, 0, V_N]])
    # Project to 2D using barycentric coords
    proj = lambda x: np.array([x[1] + x[2]/2, x[2]*np.sqrt(3)/2])
    tri = np.array([proj(c) for c in corners])
    triangle = Polygon(tri, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(triangle)
    # Sample feasible points
    pts = []
    for i in np.linspace(0, V_N, 80):
        for j in np.linspace(0, V_N - i, 80):
            k = V_N - i - j
            x = [i, j, k]
            ok = True
            for S in all_coalitions({1, 2, 3}):
                idx = [p - 1 for p in S]
                if sum(x[p] for p in idx) < v(S) - 1e-9:
                    ok = False; break
            if ok and all(xi >= -1e-9 for xi in x):
                pts.append(proj(x))
    if pts:
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], s=1, c='blue', alpha=0.4, label='Core')
    for i, label in enumerate(['P1', 'P2', 'P3']):
        ax.annotate(label, tri[i], fontsize=12, fontweight='bold')
    ax.set_title(f'Core of 3-Player Game (v(N)={V_N})')
    ax.legend(); ax.set_aspect('equal'); ax.axis('off')
    plt.tight_layout(); plt.savefig('core_3player.png', dpi=100); plt.show()


# --- Demo ---
if __name__ == "__main__":
    v_dict = {
        frozenset(): 0,
        frozenset([1]): 0, frozenset([2]): 0, frozenset([3]): 0,
        frozenset([1, 2]): 5, frozenset([1, 3]): 6, frozenset([2, 3]): 7,
        frozenset([1, 2, 3]): 10
    }
    v = characteristic_function(v_dict)
    N = {1, 2, 3}
    alloc = find_core_allocation(N, v)
    print(f"Core allocation: {alloc}")
    print(f"In core? {is_in_core(alloc, N, v)}")
    print(f"Equal split in core? {is_in_core([10/3]*3, N, v)}")
    visualize_core_3player(v_dict)
```

---

## Practice Problems (45 min)

**P1.** Consider a 3-player game with $v(\{1\})=v(\{2\})=v(\{3\})=0$, $v(\{1,2\})=4$, $v(\{1,3\})=5$, $v(\{2,3\})=3$, $v(N)=8$. Is the allocation $(3, 2, 3)$ in the core?

<details><summary>Answer</summary>

Check: $3+2+3=8=v(N)$ ✓. $3+2=5 \geq 4$ ✓. $3+3=6 \geq 5$ ✓. $2+3=5 \geq 3$ ✓. All individual $\geq 0$ ✓. **Yes, it is in the core.**
</details>

**P2.** For the glove game — 2 left-glove owners and 1 right-glove owner, $v(S) = \min(L(S), R(S))$ — show the core is $\{(0, 0, 1)\}$.

<details><summary>Answer</summary>

$N=\{L_1,L_2,R\}$. $v(\{L_1,R\})=v(\{L_2,R\})=1$, $v(\{L_1,L_2\})=0$, $v(N)=1$. Core: $x_1+x_2+x_3=1$; $x_1+x_3\geq1$, $x_2+x_3\geq1$, $x_1+x_2\geq0$. From the first two: $x_3\geq1-x_1$ and $x_3\geq1-x_2$. Since $x_1+x_2+x_3=1$, we get $x_1=x_2=0$, $x_3=1$.
</details>

**P3.** Prove that every convex game is superadditive.

<details><summary>Answer</summary>

For disjoint $S, T$: $v(S \cup T) + v(S \cap T) \geq v(S) + v(T)$. Since $S \cap T = \emptyset$, $v(\emptyset)=0$, so $v(S \cup T) \geq v(S) + v(T)$.
</details>

---

## Expert Challenges

**E1.** Construct a 3-player superadditive game with an empty core. Verify emptiness using the Bondareva-Shapley theorem.

<details><summary>Hint</summary>

Try $v(\{i\})=0$, $v(\{i,j\})=\alpha$ for all pairs, $v(N)=\beta$ with $3\alpha/2 > \beta$. The balanced collection with each pair weighted $1/2$ violates the condition.
</details>

**E2.** Implement a function to check whether a game is convex and prove that convex games always have a nonempty core.

<details><summary>Hint</summary>

Check $v(S \cup \{i\}) - v(S) \leq v(T \cup \{i\}) - v(T)$ for all $S \subseteq T \subseteq N \setminus \{i\}$. Convexity implies balancedness.
</details>

**E3.** For the airport cost-sharing game (runway costs $c_1 < c_2 < c_3$ for planes of increasing size), show the core is $\{x : x_i \leq c_i - c_{i-1}\}$ and find its center.

<details><summary>Hint</summary>

$v(S) = \max_{i \in S} c_i$. The core constraints reduce to incremental cost bounds.
</details>

---

## Connections
- **Prerequisites:** Normal form games (Day 92), LP formulation (Day 97) — core is an LP feasibility problem
- **Forward:** Shapley value (Day 100) always exists even when core is empty; Bargaining (Day 101) for 2-player cooperative solutions
- **OKS:** Multi-robot task coalitions use core stability to ensure no subset of robots prefers to defect from the fleet schedule

## Self-Check
- [ ] I can define a TU game and compute its characteristic function
- [ ] I can check whether an allocation is in the core
- [ ] I understand the Bondareva-Shapley theorem and can verify balancedness
- [ ] I can formulate core membership as an LP
