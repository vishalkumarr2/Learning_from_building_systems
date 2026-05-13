# Day 93: Pure-Strategy Nash Equilibrium

> **Phase VII — Game Theory for Optimization & Robotics | Week 14 | 2.5 hours**
> *A Nash equilibrium is a profile where no player can unilaterally improve — the fixed point of rational play.*

## Navigation
- **Previous:** [Day 92: Normal Form Games and Dominance](day-92-normal-form-games.md)
- **Next:** [Day 94: Mixed-Strategy Nash Equilibrium](day-94-mixed-nash.md)
- [Week 14 Overview](../../../CURRICULUM.md)
- [Chapter 10: Game Theory Reference](../../../10-game-theory.md)
- [Full Curriculum](../../../00-learning-plan.md)

## OKS Relevance
When multiple robots choose charging stations, a stable assignment is one where no single robot benefits by switching. This is a pure Nash equilibrium of the station-selection game. Finding it avoids oscillation in real-time scheduling — robots converge to a consistent plan without central coordination.

---

## Theory (45 min)

### 93.1 Best Response

Player $i$'s **best response** to $s_{-i}$ is any strategy maximizing their payoff:

$$BR_i(s_{-i}) = \arg\max_{s_i \in S_i} u_i(s_i, s_{-i})$$

Best response is generally a **set** (multiple strategies may tie). A strategy profile $(s_1^*, \ldots, s_n^*)$ where each player best-responds to the others is special — that's an equilibrium.

### 93.2 Nash Equilibrium Definition

A strategy profile $s^* = (s_1^*, \ldots, s_n^*)$ is a **Nash Equilibrium (NE)** if for every player $i$:

$$u_i(s_i^*, s_{-i}^*) \geq u_i(s_i, s_{-i}^*) \quad \forall s_i \in S_i$$

Equivalently: $s_i^* \in BR_i(s_{-i}^*)$ for all $i$. No player has a profitable **unilateral** deviation.

### 93.3 Finding Pure NE: The Underline Method

For a bimatrix $(A, B)$:
1. For each column $j$, underline the maximum entry in $A_{:,j}$ (Player 1's best response to $j$).
2. For each row $i$, underline the maximum entry in $B_{i,:}$ (Player 2's best response to $i$).
3. Any cell $(i,j)$ with **both** entries underlined is a pure NE.

### 93.4 Multiple and Zero Equilibria

- **Stag Hunt** has two pure NE: (Stag, Stag) and (Hare, Hare).
- **Matching Pennies** has zero pure NE — best responses never coincide.
- **Battle of the Sexes** has two pure NE on the diagonal.

Games can have 0, 1, or many pure NE. The number and structure vary.

### 93.5 Pareto Efficiency vs Nash Equilibrium

A profile $s$ is **Pareto efficient** if no other profile makes every player at least as well off and at least one strictly better. NE need not be Pareto efficient — the Prisoner's Dilemma's unique NE (Defect, Defect) is Pareto dominated by (Cooperate, Cooperate).

$$\text{NE} \not\subseteq \text{Pareto efficient} \quad \text{and} \quad \text{Pareto efficient} \not\subseteq \text{NE}$$

---

## Implementation (60 min)

```python
import numpy as np

def best_response(payoff, player, opponent_action):
    """Return set of best-response actions for player given opponent's choice.

    player: 1 (row) or 2 (column).
    """
    if player == 1:
        col = payoff[:, opponent_action]
        max_val = np.max(col)
        return set(np.where(col == max_val)[0])
    else:
        row = payoff[opponent_action, :]
        max_val = np.max(row)
        return set(np.where(row == max_val)[0])

def find_pure_nash(A, B):
    """Find all pure-strategy Nash equilibria using the underline method."""
    m, n = A.shape
    equilibria = []
    for i in range(m):
        for j in range(n):
            # i must be a best response to j for player 1
            if A[i, j] == np.max(A[:, j]):
                # j must be a best response to i for player 2
                if B[i, j] == np.max(B[i, :]):
                    equilibria.append((i, j))
    return equilibria

def is_pareto_efficient(A, B, profile):
    """Check if a profile is Pareto efficient."""
    i, j = profile
    u1, u2 = A[i, j], B[i, j]
    m, n = A.shape
    for r in range(m):
        for c in range(n):
            if A[r, c] >= u1 and B[r, c] >= u2:
                if A[r, c] > u1 or B[r, c] > u2:
                    return False
    return True

# --- Demo: Classic Games ---
games = {
    "Prisoner's Dilemma": (np.array([[-1,-3],[0,-2]]), np.array([[-1,0],[-3,-2]])),
    "Stag Hunt": (np.array([[4,0],[3,2]]), np.array([[4,3],[0,2]])),
    "Battle of Sexes": (np.array([[3,0],[0,2]]), np.array([[2,0],[0,3]])),
    "Matching Pennies": (np.array([[1,-1],[-1,1]]), np.array([[-1,1],[1,-1]])),
}

for name, (A, B) in games.items():
    ne = find_pure_nash(A, B)
    print(f"\n=== {name} ===")
    print(f"  Pure NE: {ne}")
    for eq in ne:
        pareto = is_pareto_efficient(A, B, eq)
        print(f"  {eq}: payoffs=({A[eq]:.0f},{B[eq]:.0f}), Pareto efficient={pareto}")

# Best response mapping for Stag Hunt
print("\n=== Stag Hunt: Best Response Map ===")
A_sh, B_sh = games["Stag Hunt"]
for j in range(2):
    br1 = best_response(A_sh, 1, j)
    print(f"  BR1(P2={j}) = {br1}")
for i in range(2):
    br2 = best_response(B_sh, 2, i)
    print(f"  BR2(P1={i}) = {br2}")
```

**Expected output:**
```
=== Prisoner's Dilemma ===
  Pure NE: [(1, 1)]
  (1, 1): payoffs=(-2,-2), Pareto efficient=False

=== Stag Hunt ===
  Pure NE: [(0, 0), (1, 1)]
  (0, 0): payoffs=(4,4), Pareto efficient=True
  (1, 1): payoffs=(2,2), Pareto efficient=False

=== Battle of Sexes ===
  Pure NE: [(0, 0), (1, 1)]
  (0, 0): payoffs=(3,2), Pareto efficient=True
  (1, 1): payoffs=(2,3), Pareto efficient=True

=== Matching Pennies ===
  Pure NE: []

=== Stag Hunt: Best Response Map ===
  BR1(P2=0) = {0}
  BR1(P2=1) = {1}
  BR2(P1=0) = {0}
  BR2(P1=1) = {1}
```

---

## Practice Problems (45 min)

**P1.** Find all pure NE in this 3×3 game using the underline method:

$$A = \begin{pmatrix} 2 & 0 & 1 \\ 3 & 1 & 0 \\ 1 & 2 & 3 \end{pmatrix}, \quad B = \begin{pmatrix} 1 & 3 & 0 \\ 0 & 2 & 1 \\ 2 & 0 & 3 \end{pmatrix}$$

<details><summary>Answer</summary>

Player 1 best responses (max in each column): Col 0→Row 1 (3), Col 1→Row 2 (2), Col 2→Row 2 (3).
Player 2 best responses (max in each row): Row 0→Col 1 (3), Row 1→Col 2 (1), Row 2→Col 2 (3).
Underlined cells: (1,0), (2,1), (2,2). Check both underlined: (2,2) has A=3 (max in col 2) and B=3 (max in row 2). **NE = {(2, 2)}** with payoffs (3, 3).
</details>

**P2.** Construct a 2×2 symmetric game with exactly two pure NE that are both Pareto efficient.

<details><summary>Answer</summary>

Battle of the Sexes variant: $A = \begin{pmatrix} 3 & 0 \\ 0 & 3 \end{pmatrix}$, $B = A^T = \begin{pmatrix} 3 & 0 \\ 0 & 3 \end{pmatrix}$. NE: (0,0) and (1,1), both yield payoff (3,3). Both are Pareto efficient since no other profile gives both players ≥ 3.
</details>

**P3.** In the Prisoner's Dilemma, the NE is not Pareto efficient. Explain intuitively why rational individual play leads to a collectively suboptimal outcome.

<details><summary>Answer</summary>

Each player reasons: "Regardless of what the other does, Defect gives me a higher payoff." Both defect, reaching (-2, -2). But if both cooperated, they'd get (-1, -1). The tension: individual rationality (no incentive to deviate) conflicts with collective welfare. Without binding agreements, the dominant strategy leads to an inferior joint outcome.
</details>

---

## Expert Challenges

**E1.** Prove: every game in which IESDS yields a unique profile has that profile as the unique NE.

<details><summary>Answer</summary>

If IESDS reduces to $(s_1^*, s_2^*)$, then for player 1, all strategies except $s_1^*$ were eliminated because they were dominated at some stage. In the original game, $s_1^*$ must be a best response to $s_2^*$ (the only surviving opponent strategy). Symmetrically for player 2. So $(s_1^*, s_2^*)$ is a NE. Uniqueness: any NE survives IESDS (a NE strategy is never dominated), so any NE must be among the survivors. Since only one profile survives, it is the unique NE.
</details>

**E2.** Construct a 3×3 game with exactly three pure Nash equilibria. Verify computationally.

<details><summary>Answer</summary>

```python
A = np.array([[3, 0, 0],
              [0, 3, 0],
              [0, 0, 3]])
B = A.copy()  # symmetric coordination game
ne = find_pure_nash(A, B)
print(f"NE: {ne}")  # [(0,0), (1,1), (2,2)]
```

Three NE on the diagonal: (0,0), (1,1), (2,2) — each giving payoff (3,3). Off-diagonal gives 0 to both, so no deviation is profitable.
</details>

**E3.** A congestion game has 3 robots and 2 stations; cost at station $k$ is $c_k \cdot n_k$ where $n_k$ is the number of robots at station $k$ and $c_1=1, c_2=2$. Enumerate all profiles and find all pure NE.

<details><summary>Answer</summary>

Profiles (station for each robot): (1,1,1), (1,1,2), (1,2,1), (2,1,1), (1,2,2), (2,1,2), (2,2,1), (2,2,2). Cost for robot at station $k$: $c_k \cdot n_k$. NE: (1,1,2) and permutations — 2 at station 1 (cost 2 each) and 1 at station 2 (cost 2). No robot can reduce cost by switching: moving from station 1→2 gives cost $2·2=4$; moving from station 2→1 gives cost $1·3=3 > 2$. So configurations with split (2,1) are NE.
</details>

---

## Connections
- **Back:** Day 92 (normal form, dominance) — NE refines beyond dominance
- **Forward:** Day 94 (mixed NE — what happens when no pure NE exists), Day 95 (selecting among multiple NE)
- **OKS:** Station assignment as congestion game — pure NE gives stable fleet allocations

## Self-Check
- [ ] I can apply the underline method to find all pure NE in a bimatrix
- [ ] I can compute best-response sets for each player and verify a NE
- [ ] I understand why NE need not be Pareto efficient (Prisoner's Dilemma example)
- [ ] My code correctly finds 0, 1, or multiple pure NE in test games
