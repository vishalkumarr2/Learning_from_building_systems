# Day 95: Equilibrium Selection

> **Phase VII — Game Theory for Optimization & Robotics | Week 14 | 2.5 hours**
> *When a game has multiple equilibria, the real challenge is choosing which one to play.*

## Navigation
- **Previous:** [Day 94: Mixed-Strategy Nash Equilibrium](day-94-mixed-nash.md)
- **Next:** [Day 96: Zero-Sum Games and Minimax](day-96-zero-sum-minimax.md)
- [Week 14 Overview](../../../CURRICULUM.md)
- [Chapter 10: Game Theory Reference](../../../10-game-theory.md)
- [Full Curriculum](../../../00-learning-plan.md)

## OKS Relevance
A warehouse fleet may have multiple stable task allocations (multiple NE). The fleet controller must pick one — the "equilibrium selection" problem. Risk-dominant solutions tolerate communication failures; Pareto-dominant ones maximize throughput. Correlated equilibrium uses a central dispatcher signal to coordinate robots beyond what decentralized NE can achieve.

---

## Theory (45 min)

### 95.1 The Coordination Problem

When a game has multiple NE, players face a **coordination problem**: which equilibrium will they play? Without communication, miscoordination is likely. Different refinement criteria resolve the ambiguity.

### 95.2 Pareto Dominance

NE $s^*$ **Pareto dominates** NE $s'$ if $u_i(s^*) \geq u_i(s')$ for all $i$ with strict inequality for some $i$. The **Pareto-dominant NE** is preferred by all players.

In the Stag Hunt: (Stag, Stag) yields (4,4) and Pareto dominates (Hare, Hare) at (2,2).

### 95.3 Risk Dominance (Harsanyi & Selten, 1988)

For a 2×2 game with two pure NE $(s_1, s_1)$ and $(s_2, s_2)$, the **risk-dominant** equilibrium has the larger **Nash product** of deviation losses:

$$\text{NE } (s_1, s_1) \text{ risk-dominates if } (u_1(s_1,s_1) - u_1(s_2,s_1))(u_2(s_1,s_1) - u_2(s_1,s_2)) \geq (u_1(s_2,s_2) - u_1(s_1,s_2))(u_2(s_2,s_2) - u_2(s_2,s_1))$$

Intuitively: risk dominance picks the equilibrium that is "safer" against uncertainty about the other player's choice.

In the Stag Hunt: (Hare, Hare) is risk-dominant despite being Pareto-inferior — it's the safe choice.

### 95.4 Focal Points (Schelling, 1960)

A **focal point** is an equilibrium that stands out due to cultural, contextual, or labeling cues — even though the payoff structure doesn't distinguish it. Example: "Meet somewhere in New York City" → Grand Central Station.

Focal points are powerful in human play but hard to formalize for robots without explicit coordination protocols.

### 95.5 Correlated Equilibrium (Aumann, 1974)

A **correlated equilibrium** is a probability distribution $\pi$ over strategy profiles such that when a mediator privately recommends $s_i$ to player $i$, no player wants to deviate:

$$\sum_{s_{-i}} \pi(s_i, s_{-i}) \cdot u_i(s_i, s_{-i}) \geq \sum_{s_{-i}} \pi(s_i, s_{-i}) \cdot u_i(s_i', s_{-i}) \quad \forall i, s_i, s_i'$$

Every NE is a correlated equilibrium (product distributions). But correlated equilibria can achieve **higher total welfare** by correlating players' actions.

The set of correlated equilibria is a **convex polytope** defined by linear constraints — computable via LP.

---

## Implementation (60 min)

```python
import numpy as np
from scipy.optimize import linprog

def classify_equilibria(A, B, equilibria):
    """Classify NE by Pareto dominance and risk dominance."""
    payoffs = [(A[eq], B[eq]) for eq in equilibria]
    n = len(equilibria)

    # Pareto dominance
    pareto_dominant = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if payoffs[j][0] >= payoffs[i][0] and payoffs[j][1] >= payoffs[i][1]:
                if payoffs[j][0] > payoffs[i][0] or payoffs[j][1] > payoffs[i][1]:
                    dominated = True; break
        if not dominated:
            pareto_dominant.append(equilibria[i])
    return pareto_dominant

def risk_dominant_2x2(A, B):
    """Find the risk-dominant NE for a 2×2 symmetric coordination game.

    Assumes pure NE at (0,0) and (1,1). Returns 0 or 1.
    """
    # Nash product for (0,0): deviation loss for each player
    loss1_from_00 = A[0,0] - A[1,0]  # P1's loss from deviating at (0,0)
    loss2_from_00 = B[0,0] - B[0,1]  # P2's loss from deviating at (0,0)
    np_00 = loss1_from_00 * loss2_from_00

    loss1_from_11 = A[1,1] - A[0,1]
    loss2_from_11 = B[1,1] - B[1,0]
    np_11 = loss1_from_11 * loss2_from_11

    print(f"  Nash product (0,0): {np_00:.2f}")
    print(f"  Nash product (1,1): {np_11:.2f}")
    return 0 if np_00 >= np_11 else 1

def correlated_equilibrium_lp(A, B, objective="welfare"):
    """Find a correlated equilibrium maximizing total welfare via LP.

    Returns the probability distribution over strategy profiles.
    """
    m, n = A.shape
    num_vars = m * n  # π(i,j) for each (i,j)

    # Objective: maximize total welfare = Σ π(i,j) * (A[i,j] + B[i,j])
    c = -(A + B).flatten()  # negative for minimization

    # Incentive constraints
    A_ub = []
    b_ub = []

    # Player 1 constraints: for each i, i'
    for i in range(m):
        for i_prime in range(m):
            if i == i_prime:
                continue
            row = np.zeros(num_vars)
            for j in range(n):
                idx = i * n + j
                row[idx] = A[i_prime, j] - A[i, j]  # gain from deviating i→i'
            A_ub.append(row)
            b_ub.append(0.0)

    # Player 2 constraints: for each j, j'
    for j in range(n):
        for j_prime in range(n):
            if j == j_prime:
                continue
            row = np.zeros(num_vars)
            for i in range(m):
                idx = i * n + j
                row[idx] = B[i, j_prime] - B[i, j]
            A_ub.append(row)
            b_ub.append(0.0)

    # Probability constraints: sum = 1, all >= 0
    A_eq = np.ones((1, num_vars))
    b_eq = np.array([1.0])
    bounds = [(0, None)] * num_vars

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method='highs')

    if result.success:
        pi = result.x.reshape(m, n)
        welfare = -result.fun
        return pi, welfare
    return None, None

# --- Demo: Stag Hunt ---
A_sh = np.array([[4, 0], [3, 2]])
B_sh = np.array([[4, 3], [0, 2]])

print("=== Stag Hunt: Equilibrium Selection ===")
equilibria = [(0,0), (1,1)]
pareto = classify_equilibria(A_sh, B_sh, equilibria)
print(f"Pareto-dominant NE: {pareto}")
print("Risk dominance:")
rd = risk_dominant_2x2(A_sh, B_sh)
print(f"  Risk-dominant NE: ({rd},{rd})")

# Correlated equilibrium
print("\n=== Stag Hunt: Correlated Equilibrium ===")
pi, welfare = correlated_equilibrium_lp(A_sh, B_sh)
print(f"Optimal CE distribution:\n{np.round(pi, 4)}")
print(f"Total welfare: {welfare:.4f}")

# --- Demo: Battle of the Sexes ---
A_bos = np.array([[3, 0], [0, 2]])
B_bos = np.array([[2, 0], [0, 3]])
print("\n=== Battle of Sexes: Correlated Equilibrium ===")
pi_bos, w_bos = correlated_equilibrium_lp(A_bos, B_bos)
print(f"Optimal CE distribution:\n{np.round(pi_bos, 4)}")
print(f"Total welfare: {w_bos:.4f}")
print(f"(Compare: pure NE give welfare 5 each, mixed NE gives 2.4)")
```

**Expected output:**
```
=== Stag Hunt: Equilibrium Selection ===
Pareto-dominant NE: [(0, 0)]
Risk dominance:
  Nash product (0,0): 1.00
  Nash product (1,1): 4.00
  Risk-dominant NE: (1,1)

=== Stag Hunt: Correlated Equilibrium ===
Optimal CE distribution:
[[1. 0.]
 [0. 0.]]
Total welfare: 8.0000

=== Battle of Sexes: Correlated Equilibrium ===
Optimal CE distribution:
[[0.5 0. ]
 [0.  0.5]]
Total welfare: 5.0000
(Compare: pure NE give welfare 5 each, mixed NE gives 2.4)
```

---

## Practice Problems (45 min)

**P1.** In the Stag Hunt, the Pareto-dominant NE is (Stag, Stag) but the risk-dominant NE is (Hare, Hare). Explain the tension: when should a robot fleet choose the risky-but-better option vs. the safe option?

<details><summary>Answer</summary>

(Stag, Stag) requires mutual trust — if one robot deviates, the cooperator gets 0. (Hare, Hare) guarantees at least 2 regardless. With reliable communication, Pareto dominance wins — use (Stag, Stag). With unreliable communication or unknown partners, risk dominance is safer — use (Hare, Hare). In OKS: if the fleet controller can broadcast coordination signals reliably, go for the higher-throughput assignment. If communication is lossy, fall back to individually-safe allocations.
</details>

**P2.** Verify that the Battle of Sexes CE with $\pi(0,0) = \pi(1,1) = 1/2$ satisfies all incentive constraints.

<details><summary>Answer</summary>

Player 1, recommended Row 0 (happens with prob 1/2): opponents play Col 0. Payoff for obeying: 3. Payoff for deviating to Row 1: 0. 3 ≥ 0. ✓
Player 1, recommended Row 1: opponents play Col 1. Payoff: 2 vs. 0. ✓
Player 2, recommended Col 0: opponent plays Row 0. Payoff: 2 vs. 0. ✓
Player 2, recommended Col 1: opponent plays Row 1. Payoff: 3 vs. 0. ✓
All constraints satisfied.
</details>

**P3.** For a 3-player 2-strategy coordination game where all players choosing the same strategy yields payoff 3 each, and any mismatch yields 0, how many pure NE exist?

<details><summary>Answer</summary>

Two pure NE: all play strategy 0 (payoff 3 each) and all play strategy 1 (payoff 3 each). Any unilateral deviation breaks coordination → payoff 0. Both are NE. Both are Pareto efficient with equal payoffs, so neither Pareto-dominates the other. A correlated equilibrium could randomize 50/50 between them.
</details>

---

## Expert Challenges

**E1.** Prove that the set of correlated equilibria forms a convex polytope. What does this imply about computing optimal CE?

<details><summary>Answer</summary>

The incentive constraints are linear in $\pi(s)$, the normalization $\sum_s \pi(s) = 1$ is linear, and non-negativity $\pi(s) \geq 0$ is linear. A set defined by finitely many linear inequalities and equalities is a convex polytope. Implication: optimizing any linear objective (total welfare, fairness, etc.) over correlated equilibria is an LP — solvable in polynomial time. This is much easier than finding Nash equilibria (which is PPAD-complete).
</details>

**E2.** In a game with $k$ pure NE, what is the maximum number of CE? Characterize the relationship.

<details><summary>Answer</summary>

Every NE is a CE (put all probability on that NE). Every convex combination of NE is also a CE. But the CE polytope is generally larger — it includes non-product distributions. The CE set is a superset of the convex hull of NE. The maximum number of extreme CE is bounded by the number of vertices of the polytope, which depends on the game structure.
</details>

**E3.** Design a correlated equilibrium for an intersection game where two robots must choose Yield or Go. Going simultaneously causes a crash (payoff -10), both yielding wastes time (-1), and one going while the other yields is optimal for the goer (2, 0).

<details><summary>Answer</summary>

$A = \begin{pmatrix} -1 & 0 \\ 2 & -10 \end{pmatrix}$, $B = \begin{pmatrix} -1 & 2 \\ 0 & -10 \end{pmatrix}$. The traffic-light CE: $\pi(\text{Go,Yield}) = \pi(\text{Yield,Go}) = 1/2$. Each robot, when told "Go," knows the other yields (payoff 2 vs. -1 for deviating). When told "Yield," knows the other goes (payoff 0 vs. -10 for deviating). Total welfare = $(1/2)(0+2)+(1/2)(2+0) = 2$, better than mixed NE.
</details>

---

## Connections
- **Back:** Day 93 (multiple pure NE), Day 94 (mixed NE as alternative selection), Day 29–31 (LP for CE)
- **Forward:** Day 96 (zero-sum — unique NE, no selection needed), Week 15 (mechanism design as equilibrium engineering)
- **OKS:** Fleet coordination protocol = correlated equilibrium mediator; dispatcher signals resolve multi-robot deadlocks

## Self-Check
- [ ] I can distinguish Pareto dominance from risk dominance and explain when each matters
- [ ] I understand correlated equilibrium as a generalization of NE with a mediator
- [ ] I can formulate the CE problem as an LP and solve it with scipy
- [ ] I can explain focal points and their limitation for autonomous systems
