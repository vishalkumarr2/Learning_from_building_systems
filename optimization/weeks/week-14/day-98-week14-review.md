# Day 98: Week 14 Review — Game Theory Foundations

> **Phase VII — Game Theory for Optimization & Robotics | Week 14 | 2.5 hours**
> *From payoff matrices to LP duality — a unified solver for any two-player game.*

## Navigation
- **Previous:** [Day 97: Zero-Sum Games via LP and Duality](day-97-zero-sum-lp.md)
- **Next:** [Day 99: Cooperative Games and the Core](../week-15/day-99-cooperative-core.md)
- [Week 14 Overview](../../../CURRICULUM.md)
- [Chapter 10: Game Theory Reference](../../../10-game-theory.md)
- [Full Curriculum](../../../00-learning-plan.md)

## OKS Relevance
This week's tools model every multi-robot interaction as a game: intersection coordination (Day 92), charging station selection (Day 93), stochastic patrol (Day 94), fleet protocols (Day 95), adversarial planning (Day 96), and robust control via LP (Day 97). The complete solver below handles arbitrary $m \times n$ games — the computational backbone for game-theoretic fleet management.

---

## Theory (45 min)

### 98.1 Concept Map

```
Normal Form Game (Day 92)
  ├── Dominance & IESDS ──→ simplify game
  ├── Pure Nash Equilibrium (Day 93)
  │     ├── Best response / underline method
  │     └── Pareto efficiency check
  ├── Mixed Nash Equilibrium (Day 94)
  │     ├── Indifference principle (2×2)
  │     └── Support enumeration (general)
  ├── Equilibrium Selection (Day 95)
  │     ├── Pareto dominance
  │     ├── Risk dominance
  │     └── Correlated equilibrium (LP)
  └── Zero-Sum Games (Days 96-97)
        ├── Saddle points
        ├── Minimax theorem
        └── LP formulation ←→ Duality
```

### 98.2 Key Formulas Summary

| Concept | Formula |
|---------|---------|
| Nash equilibrium | $s_i^* \in BR_i(s_{-i}^*)$ for all $i$ |
| Mixed NE (2×2) | $p^* = \frac{h-g}{(e-g)-(f-h)}$ from $B$; $q^* = \frac{d-b}{(a-b)-(c-d)}$ from $A$ |
| Maximin | $\underline{v} = \max_i \min_j A_{ij}$ |
| Minimax | $\overline{v} = \min_j \max_i A_{ij}$ |
| Minimax theorem | $\max_p \min_q p^T A q = \min_q \max_p p^T A q = v^*$ |
| Row player LP | $\max v$ s.t. $A^T p \geq v \cdot \mathbf{1}$, $\sum p = 1$, $p \geq 0$ |
| Risk dominance | Compare Nash products of deviation losses |
| Correlated equilibrium | LP: max welfare s.t. incentive constraints |

### 98.3 Decision Tree: Solving a Two-Player Game

1. **Write the bimatrix** $(A, B)$.
2. **Apply IESDS** — remove dominated strategies.
3. **Check zero-sum:** is $B = -A$?
   - **Yes →** find saddle points or solve via LP (Day 97).
   - **No →** continue.
4. **Find pure NE** via underline method.
5. **Find mixed NE** via indifference (2×2) or support enumeration (general).
6. **Multiple NE?** Apply selection criteria (Pareto, risk dominance, CE).

---

## Implementation (60 min)

### Complete Game Solver

```python
import numpy as np
from scipy.optimize import linprog
from itertools import combinations

class GameSolver:
    """Complete solver for two-player normal-form games."""

    def __init__(self, A, B):
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float)
        self.m, self.n = self.A.shape

    def is_zero_sum(self, tol=1e-10):
        return np.allclose(self.A + self.B, 0, atol=tol)

    def iesds(self):
        """Iterated elimination of strictly dominated strategies."""
        rows, cols = list(range(self.m)), list(range(self.n))
        changed = True
        while changed:
            changed = False
            # Row player
            remove = []
            for i in rows:
                for i2 in rows:
                    if i == i2: continue
                    if np.all(self.A[i2, cols] > self.A[i, cols]):
                        remove.append(i); break
            for r in set(remove): rows.remove(r); changed = True
            # Column player
            remove = []
            for j in cols:
                for j2 in cols:
                    if j == j2: continue
                    r_idx = np.array(rows)
                    if np.all(self.B[r_idx, j2] > self.B[r_idx, j]):
                        remove.append(j); break
            for c in set(remove): cols.remove(c); changed = True
        return rows, cols

    def find_pure_nash(self):
        """Find all pure-strategy NE."""
        equilibria = []
        for i in range(self.m):
            for j in range(self.n):
                if (self.A[i, j] == np.max(self.A[:, j]) and
                    self.B[i, j] == np.max(self.B[i, :])):
                    equilibria.append((i, j))
        return equilibria

    def find_saddle_points(self):
        """For zero-sum games: find saddle points."""
        row_mins = np.min(self.A, axis=1)
        col_maxs = np.max(self.A, axis=0)
        return [(i, j) for i in range(self.m) for j in range(self.n)
                if self.A[i,j] == row_mins[i] and self.A[i,j] == col_maxs[j]]

    def solve_zero_sum_lp(self):
        """Solve zero-sum game via LP. Returns (value, p*, q*)."""
        m, n = self.m, self.n
        c = np.zeros(m + 1); c[-1] = -1
        A_ub = np.zeros((n, m + 1))
        A_ub[:, :m] = -self.A.T; A_ub[:, -1] = 1.0
        A_eq = np.zeros((1, m + 1)); A_eq[0, :m] = 1.0
        bounds = [(0, None)] * m + [(None, None)]
        res = linprog(c, A_ub=A_ub, b_ub=np.zeros(n),
                      A_eq=A_eq, b_eq=[1.0], bounds=bounds, method='highs')
        p = res.x[:m]; v = res.x[-1]

        c2 = np.zeros(n + 1); c2[-1] = 1
        A_ub2 = np.zeros((m, n + 1))
        A_ub2[:, :n] = self.A; A_ub2[:, -1] = -1.0
        A_eq2 = np.zeros((1, n + 1)); A_eq2[0, :n] = 1.0
        bounds2 = [(0, None)] * n + [(None, None)]
        res2 = linprog(c2, A_ub=A_ub2, b_ub=np.zeros(m),
                       A_eq=A_eq2, b_eq=[1.0], bounds=bounds2, method='highs')
        q = res2.x[:n]
        return v, p, q

    def solve(self):
        """Full solve: returns a summary dict."""
        result = {"zero_sum": self.is_zero_sum()}
        rows, cols = self.iesds()
        result["iesds_rows"] = rows
        result["iesds_cols"] = cols
        result["pure_ne"] = self.find_pure_nash()

        if self.is_zero_sum():
            sps = self.find_saddle_points()
            result["saddle_points"] = sps
            v, p, q = self.solve_zero_sum_lp()
            result["game_value"] = v
            result["p_star"] = p
            result["q_star"] = q
        return result

# --- Demo: Solve multiple games ---
games = {
    "Prisoner's Dilemma": (
        [[-1,-3],[0,-2]], [[-1,0],[-3,-2]]
    ),
    "Matching Pennies": (
        [[1,-1],[-1,1]], [[-1,1],[1,-1]]
    ),
    "3×3 Zero-Sum": (
        [[3,1,4],[2,5,0],[4,2,3]], [[-3,-1,-4],[-2,-5,0],[-4,-2,-3]]
    ),
    "Stag Hunt": (
        [[4,0],[3,2]], [[4,3],[0,2]]
    ),
}

for name, (A, B) in games.items():
    solver = GameSolver(A, B)
    res = solver.solve()
    print(f"\n=== {name} ===")
    print(f"  Zero-sum: {res['zero_sum']}")
    print(f"  IESDS survivors: rows={res['iesds_rows']}, cols={res['iesds_cols']}")
    print(f"  Pure NE: {res['pure_ne']}")
    if res['zero_sum']:
        print(f"  Game value: {res['game_value']:.4f}")
        print(f"  P1 strategy: {np.round(res['p_star'], 3)}")
        print(f"  P2 strategy: {np.round(res['q_star'], 3)}")
```

**Expected output:**
```
=== Prisoner's Dilemma ===
  Zero-sum: False
  IESDS survivors: rows=[1], cols=[1]
  Pure NE: [(1, 1)]

=== Matching Pennies ===
  Zero-sum: True
  IESDS survivors: rows=[0, 1], cols=[0, 1]
  Pure NE: []
  Game value: 0.0000
  P1 strategy: [0.5 0.5]
  P2 strategy: [0.5 0.5]

=== 3×3 Zero-Sum ===
  Zero-sum: True
  IESDS survivors: rows=[0, 1, 2], cols=[0, 1, 2]
  Pure NE: []
  Game value: 2.7143
  P1 strategy: [0.429 0.143 0.429]
  P2 strategy: [0.286 0.429 0.286]

=== Stag Hunt ===
  Zero-sum: False
  IESDS survivors: rows=[0, 1], cols=[0, 1]
  Pure NE: [(0, 0), (1, 1)]
```

---

## Practice Problems (45 min)

### Exercise Set 09 — Section A: Rapid-Fire

**A1.** In a zero-sum game, if Player 1's maximin value is 3, what can you say about Player 2's minimax value?

<details><summary>Answer</summary>
In mixed strategies, minimax = maximin = 3 (minimax theorem). In pure strategies, minimax ≥ 3.
</details>

**A2.** True or False: every 2×2 game has at least one pure Nash equilibrium.

<details><summary>Answer</summary>
False. Matching Pennies has no pure NE. But by Nash's theorem, it has a mixed NE.
</details>

**A3.** If IESDS eliminates all but one strategy for each player, is that profile always a NE?

<details><summary>Answer</summary>
Yes. The unique IESDS survivor is always a NE (any NE survives IESDS, and there's only one survivor).
</details>

### Exercise Set 09 — Section B: Computation

**B1.** Solve: $A = \begin{pmatrix} 2 & -1 \\ -1 & 3 \end{pmatrix}$ (zero-sum). Find $v^*$, $p^*$, $q^*$.

<details><summary>Answer</summary>

2×2 zero-sum: $p^* = \frac{3-(-1)}{(2-(-1))-((-1)-3)} = 4/7$. $q^* = \frac{3-(-1)}{(2-(-1))-((-1)-3)} = 4/7$. $v^* = 2(4/7) + (-1)(3/7) = 5/7 \approx 0.714$.
</details>

**B2.** Find all NE (pure and mixed) for Battle of the Sexes.

<details><summary>Answer</summary>

Pure NE: (0,0) payoff (3,2) and (1,1) payoff (2,3). Mixed NE: $p^*=3/5$, $q^*=2/5$, payoffs (6/5, 6/5). Total: 3 NE.
</details>

### Exercise Set 09 — Section C: Applied

**C1.** Two warehouse zones compete for a shared robot. Model as a game: each zone requests High or Low priority. If both High, the robot splits time (payoff 3 each). If one High and one Low, the High zone gets full service (5) and Low gets backup (1). Both Low yields medium service (4 each). Find the NE and recommend a fleet policy.

<details><summary>Answer</summary>

$A = B^T = \begin{pmatrix} 3 & 5 \\ 1 & 4 \end{pmatrix}$. IESDS: High dominates Low for each zone (3>1, 5>4). Unique NE: (High, High) with payoff (3,3). But (Low, Low) gives (4,4) — Pareto superior. Policy recommendation: implement a correlated equilibrium (dispatcher alternates priority) to achieve (4,4) on average.
</details>

---

## Expert Challenges

**E1.** Prove the minimax theorem from LP strong duality (complete proof, not sketch).

<details><summary>Answer</summary>

**Primal (P1):** $\max v$ s.t. $A^T p \geq v \mathbf{1}$, $\mathbf{1}^T p = 1$, $p \geq 0$.
**Dual (P2):** $\min w$ s.t. $Aq \leq w \mathbf{1}$, $\mathbf{1}^T q = 1$, $q \geq 0$.
Both feasible: $p = \mathbf{1}/m$, $v = \min_j (A^T \mathbf{1}/m)_j$ works for primal; similarly for dual. By LP strong duality (finite optima exist when both feasible): $v^* = w^*$.
Now: $v^* = \max_p \min_j \sum_i A_{ij} p_i = \max_p \min_{q \in \Delta_n} p^T A q$ (min over simplex attained at vertex). Similarly $w^* = \min_q \max_p p^T A q$. Therefore $\max_p \min_q p^T A q = \min_q \max_p p^T A q$. ∎
</details>

**E2.** Design a game with exactly 3 Nash equilibria (two pure, one mixed). Prove this is the complete set.

<details><summary>Answer</summary>

Battle of the Sexes: $A = \begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix}$, $B = \begin{pmatrix} 2 & 0 \\ 0 & 3 \end{pmatrix}$. Pure NE: (0,0) and (1,1) by the underline method. Mixed NE: $p^*=3/5$, $q^*=2/5$. Completeness: support enumeration over all $2^2 \times 2^2 = 16$ support pairs yields only these three. For any 2×2 game with two diagonal pure NE, the indifference equations yield exactly one interior mixed NE.
</details>

**E3.** Model the following warehouse scenario: Robot A and Robot B must each choose an aisle (1, 2, or 3) for picking. If they choose the same aisle, congestion halves their throughput. Payoff = base throughput of chosen aisle minus congestion penalty. Base throughputs: aisle 1=10, aisle 2=8, aisle 3=6. Congestion penalty if same aisle: -5 each. Find all pure NE.

<details><summary>Answer</summary>

```python
base = [10, 8, 6]
A = np.zeros((3, 3)); B = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        A[i,j] = base[i] - (5 if i == j else 0)
        B[i,j] = base[j] - (5 if i == j else 0)

solver = GameSolver(A, B)
print(f"Pure NE: {solver.find_pure_nash()}")
# NE: (0,1), (1,0) — robots split between aisles 1 and 2
```

Payoffs at (0,1): A gets 10, B gets 8. At (1,0): A gets 8, B gets 10. Neither wants to switch — moving to the other's aisle causes congestion. (0,2) is not NE since B prefers aisle 1 (payoff 10 > 6).
</details>

---

## Connections
- **Week 14 recap:** Normal form → dominance (92) → pure NE (93) → mixed NE (94) → selection (95) → zero-sum (96) → LP (97)
- **Forward:** Week 15 extends to dynamic games (extensive form, backward induction) and mechanism design
- **OKS integration:** Every multi-robot decision maps to a game; the GameSolver class provides the computational tool

## Self-Check
- [ ] I can classify any 2-player game as zero-sum or general and choose the right solution method
- [ ] I can trace the full pipeline: bimatrix → IESDS → pure NE → mixed NE → LP (if zero-sum)
- [ ] I understand LP duality = minimax theorem and can explain both directions
- [ ] My GameSolver correctly handles all test cases from the week
- [ ] I can model a real warehouse scenario as a game and interpret the equilibrium
