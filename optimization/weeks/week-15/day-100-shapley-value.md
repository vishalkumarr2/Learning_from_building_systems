# Day 100: Shapley Value and Fair Division

> **Phase VII — Game Theory for Optimization & Robotics | Week 15 | 2.5 hours**
> *The unique fair way to split the pie — backed by four irrefutable axioms.*

## Navigation
- **Previous:** [Day 99: Cooperative Games and the Core](day-99-cooperative-core.md)
- **Next:** [Day 101: Bargaining Theory](day-101-bargaining.md)
- [Week 15 Overview](../../../CURRICULUM.md)
- [Chapter 10: Game Theory Reference](../../../10-game-theory.md)
- [Full Curriculum](../../../00-learning-plan.md)

## OKS Relevance
When three robots complete a multi-pick order, how much credit does each deserve? The Shapley value answers this: evaluate each robot's marginal contribution across every possible joining order. This same principle powers SHAP values in ML — explaining which sensor or subsystem contributed most to a prediction or fault diagnosis.

---

## Theory (45 min)

### 100.1 The Shapley Axioms

For a TU game $(N, v)$, a **value** is a function $\phi: \mathcal{G}^N \to \mathbb{R}^n$ assigning a payoff to each player. Shapley (1953) showed exactly one value satisfies:

1. **Efficiency:** $\sum_{i \in N} \phi_i(v) = v(N)$
2. **Symmetry:** If $v(S \cup \{i\}) = v(S \cup \{j\})$ for all $S$, then $\phi_i = \phi_j$
3. **Null player:** If $v(S \cup \{i\}) = v(S)$ for all $S$, then $\phi_i = 0$
4. **Additivity:** $\phi_i(v + w) = \phi_i(v) + \phi_i(w)$

### 100.2 The Shapley Formula

$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\;(n - |S| - 1)!}{n!} \left[ v(S \cup \{i\}) - v(S) \right]$$

**Interpretation:** Average the marginal contribution $v(S \cup \{i\}) - v(S)$ over all possible orderings in which player $i$ joins.

Equivalently, using permutations $\pi$ of $N$:

$$\phi_i(v) = \frac{1}{n!} \sum_{\pi \in \Pi(N)} \left[ v(P_i^\pi \cup \{i\}) - v(P_i^\pi) \right]$$

where $P_i^\pi$ is the set of players preceding $i$ in permutation $\pi$.

### 100.3 Properties and Computational Complexity

- Computing the exact Shapley value requires evaluating $v$ for $2^n$ coalitions — **exponential**.
- For **weighted voting games**, polynomial-time algorithms exist via generating functions.
- The Shapley value always lies in the core for **convex games**.
- **Nucleolus** is an alternative solution that minimizes the worst dissatisfaction; always in the core (when nonempty).
- **SHAP values** in ML are Shapley values where "players" are features and $v(S)$ is the prediction using only features in $S$.

---

## Implementation (60 min)

```python
"""Shapley value computation — Day 100"""
import numpy as np
from itertools import permutations, combinations
from math import factorial


def shapley_value(N, v):
    """Compute exact Shapley value for a TU game.
    
    Args:
        N: set of players
        v: characteristic function v(S) -> float, S is frozenset
    Returns:
        dict mapping player -> Shapley value
    """
    players = sorted(N)
    n = len(players)
    phi = {i: 0.0 for i in players}
    
    for i in players:
        others = [p for p in players if p != i]
        for size in range(0, n):
            for S_tuple in combinations(others, size):
                S = frozenset(S_tuple)
                s = len(S)
                weight = factorial(s) * factorial(n - s - 1) / factorial(n)
                marginal = v(S | {i}) - v(S)
                phi[i] += weight * marginal
    return phi


def shapley_via_permutations(N, v):
    """Compute Shapley value by averaging over all permutations (brute force)."""
    players = sorted(N)
    n = len(players)
    phi = {i: 0.0 for i in players}
    count = 0
    for perm in permutations(players):
        count += 1
        predecessors = set()
        for p in perm:
            marginal = v(frozenset(predecessors | {p})) - v(frozenset(predecessors))
            phi[p] += marginal
            predecessors.add(p)
    return {i: phi[i] / count for i in players}


def verify_axioms(N, v, phi):
    """Verify Shapley axioms hold for computed values."""
    players = sorted(N)
    # Efficiency
    total = sum(phi[i] for i in players)
    eff = abs(total - v(frozenset(players))) < 1e-9
    print(f"Efficiency: sum={total:.4f}, v(N)={v(frozenset(players)):.4f} -> {'PASS' if eff else 'FAIL'}")
    
    # Null player check
    for i in players:
        is_null = all(
            abs(v(frozenset(S) | {i}) - v(frozenset(S))) < 1e-9
            for r in range(len(players))
            for S in combinations([p for p in players if p != i], r)
        )
        if is_null:
            print(f"Null player {i}: phi={phi[i]:.4f} -> {'PASS' if abs(phi[i]) < 1e-9 else 'FAIL'}")


def voting_game_shapley(weights, quota):
    """Shapley-Shubik power index for a weighted voting game."""
    n = len(weights)
    N = set(range(n))
    def v(S):
        return 1.0 if sum(weights[i] for i in S) >= quota else 0.0
    return shapley_value(N, v)


# --- Demo ---
if __name__ == "__main__":
    # Example: 3-player game
    v_dict = {
        frozenset(): 0,
        frozenset([1]): 0, frozenset([2]): 0, frozenset([3]): 0,
        frozenset([1, 2]): 5, frozenset([1, 3]): 6, frozenset([2, 3]): 7,
        frozenset([1, 2, 3]): 10
    }
    v = lambda S: v_dict.get(frozenset(S), 0)
    N = {1, 2, 3}
    
    phi_exact = shapley_value(N, v)
    phi_perm = shapley_via_permutations(N, v)
    print(f"Shapley (formula):      {phi_exact}")
    print(f"Shapley (permutations): {phi_perm}")
    verify_axioms(N, v, phi_exact)
    
    # UN Security Council-style voting
    # 5 permanent (veto) + 10 non-permanent, quota = 39 (with vetoes)
    weights = [7]*5 + [1]*10  # permanent members have high weight
    quota = 39
    power = voting_game_shapley(weights, quota)
    print(f"\nVoting game power index:")
    for i, p in power.items():
        label = "Permanent" if i < 5 else "Non-perm"
        print(f"  Player {i} ({label}): {p:.4f}")
```

---

## Practice Problems (45 min)

**P1.** Compute the Shapley value for the glove game: $N=\{L_1, L_2, R\}$, $v(S) = \min(\text{left gloves in } S, \text{right gloves in } S)$.

<details><summary>Answer</summary>

Orderings and marginal contributions for $R$: $R$ is pivotal in 4 of 6 orderings. $\phi_R = 4/6 = 2/3$. By symmetry $\phi_{L_1} = \phi_{L_2} = 1/6$. The Shapley value is $(1/6, 1/6, 2/3)$.
</details>

**P2.** Three towns share an airport runway. Costs: Town A needs 100m ($1M), Town B needs 200m ($3M), Town C needs 300m ($6M). Use Shapley to allocate costs.

<details><summary>Answer</summary>

$v(\{A\})=1$, $v(\{B\})=3$, $v(\{C\})=6$, $v(\{A,B\})=3$, $v(\{A,C\})=6$, $v(\{B,C\})=6$, $v(N)=6$. Shapley gives each player their average marginal cost: $\phi_A = 1/3$, $\phi_B = 1$, $\phi_C = 14/3 \approx 4.67$. Total = 6. ✓
</details>

**P3.** Show that the Shapley value of a convex game lies in the core.

<details><summary>Answer</summary>

For convex games, marginal contributions are increasing: $v(S \cup \{i\}) - v(S) \leq v(T \cup \{i\}) - v(T)$ for $S \subseteq T$. The Shapley value, as a weighted average of marginal vectors, lies in the convex hull of marginal vectors, which equals the core for convex games.
</details>

---

## Expert Challenges

**E1.** Implement a Monte Carlo approximation of the Shapley value using random permutation sampling. Compare accuracy vs. exact for $n=10$ players with $K=1000$ samples.

<details><summary>Hint</summary>

Sample random permutations, compute marginal contribution for each player in each permutation, average. Error decreases as $O(1/\sqrt{K})$.
</details>

**E2.** Compute the nucleolus for the 3-player game above and compare with the Shapley value. When do they coincide?

<details><summary>Hint</summary>

The nucleolus minimizes $\max_S (v(S) - \sum_{i \in S} x_i)$ subject to efficiency. Solve via a sequence of LPs. They coincide for symmetric games.
</details>

**E3.** Connect Shapley values to SHAP: for a linear model $f(x) = \beta_0 + \sum \beta_i x_i$, show $\text{SHAP}_i = \beta_i (x_i - \mathbb{E}[x_i])$.

<details><summary>Hint</summary>

For linear models, $v(S) = \beta_0 + \sum_{i \in S} \beta_i x_i + \sum_{j \notin S} \beta_j \mathbb{E}[x_j]$. Marginal contribution is always $\beta_i(x_i - \mathbb{E}[x_i])$.
</details>

---

## Connections
- **Prerequisites:** Core and cooperative games (Day 99) — Shapley value always exists; LP formulations (Day 97)
- **Forward:** Bargaining (Day 101) extends to bilateral settings; Mechanism design (Day 104) uses Shapley for cost sharing
- **OKS:** Fair credit assignment for multi-robot task completion; SHAP-based fault attribution in diagnostics

## Self-Check
- [ ] I can state the four Shapley axioms and explain their intuition
- [ ] I can compute the Shapley value by hand for small games
- [ ] I understand the connection between Shapley values and SHAP
- [ ] I can implement both exact and permutation-based Shapley computation
