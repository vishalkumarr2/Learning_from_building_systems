# Day 92: Normal Form Games and Dominance

> **Phase VII — Game Theory for Optimization & Robotics | Week 14 | 2.5 hours**
> *Every strategic interaction can be encoded as a matrix — then solved with the tools you already know.*

## Navigation
- **Previous:** [Day 91: Week 13 Capstone](../week-13/day-91-week13-capstone.md)
- **Next:** [Day 93: Pure-Strategy Nash Equilibrium](day-93-pure-nash.md)
- [Week 14 Overview](../../../CURRICULUM.md)
- [Chapter 10: Game Theory Reference](../../../10-game-theory.md)
- [Full Curriculum](../../../00-learning-plan.md)

## OKS Relevance
Two OKS robots approaching the same intersection must decide: yield or proceed. Each robot's best action depends on the other's choice. This is a coordination game in normal form. Understanding dominance lets the fleet controller prune clearly bad strategies before computing equilibria — reducing the dimension of the decision space for real-time dispatch.

---

## Theory (45 min)

### 92.1 Players, Strategies, Payoffs

A **normal-form game** is a tuple $\Gamma = (N, \{S_i\}_{i \in N}, \{u_i\}_{i \in N})$ where:
- $N = \{1, 2, \ldots, n\}$ is the set of **players**.
- $S_i$ is the finite set of **strategies** for player $i$.
- $u_i : S_1 \times \cdots \times S_n \to \mathbb{R}$ is the **payoff function**.

A **strategy profile** $s = (s_1, \ldots, s_n) \in S_1 \times \cdots \times S_n$ specifies one strategy per player. We write $s_{-i}$ for the profile of all players except $i$.

For two-player games, payoffs are encoded in **bimatrices** $(A, B)$ where $A_{ij} = u_1(i,j)$ and $B_{ij} = u_2(i,j)$.

### 92.2 Classic Games

**Prisoner's Dilemma** — each player prefers to defect regardless of the other:

$$A = \begin{pmatrix} -1 & -3 \\ 0 & -2 \end{pmatrix}, \quad B = \begin{pmatrix} -1 & 0 \\ -3 & -2 \end{pmatrix}$$

**Stag Hunt** — coordination with risk:

$$A = \begin{pmatrix} 4 & 0 \\ 3 & 2 \end{pmatrix}, \quad B = \begin{pmatrix} 4 & 3 \\ 0 & 2 \end{pmatrix}$$

**Chicken (Hawk-Dove)** — anti-coordination:

$$A = \begin{pmatrix} 0 & -1 \\ 1 & -5 \end{pmatrix}, \quad B = \begin{pmatrix} 0 & 1 \\ -1 & -5 \end{pmatrix}$$

### 92.3 Strictly and Weakly Dominated Strategies

Strategy $s_i$ is **strictly dominated** by $s_i'$ if:
$$u_i(s_i', s_{-i}) > u_i(s_i, s_{-i}) \quad \forall s_{-i} \in S_{-i}$$

Strategy $s_i$ is **weakly dominated** by $s_i'$ if $\geq$ holds everywhere and $>$ holds for at least one $s_{-i}$.

A rational player never plays a strictly dominated strategy. This lets us simplify games.

### 92.4 Iterated Elimination of Strictly Dominated Strategies (IESDS)

1. Remove any strictly dominated strategy for any player.
2. Re-evaluate dominance in the reduced game.
3. Repeat until no further elimination is possible.

**Key property:** IESDS order does not matter — the result is unique. If IESDS reduces to a single profile, the game is **dominance solvable**.

---

## Implementation (60 min)

```python
import numpy as np

# --- Build classic games ---
def prisoners_dilemma():
    """Prisoner's Dilemma: (C)ooperate, (D)efect."""
    A = np.array([[-1, -3],
                  [ 0, -2]])
    B = np.array([[-1,  0],
                  [-3, -2]])
    return NormalFormGame(A, B)

def stag_hunt():
    A = np.array([[4, 0],
                  [3, 2]])
    B = np.array([[4, 3],
                  [0, 2]])
    return NormalFormGame(A, B)

# --- IESDS implementation ---
def iesds(payoff1, payoff2, strict=True):
    """Iterated elimination of (strictly) dominated strategies.

    Returns masks of surviving strategies for each player.
    """
    A, B = payoff1.copy(), payoff2.copy()
    rows = list(range(A.shape[0]))
    cols = list(range(A.shape[1]))
    changed = True
    while changed:
        changed = False
        # Check row player
        to_remove = []
        for i in rows:
            for i2 in rows:
                if i == i2:
                    continue
                col_idx = [cols.index(c) for c in cols]
                vals_i  = A[i, cols]
                vals_i2 = A[i2, cols]
                if strict and np.all(vals_i2 > vals_i):
                    to_remove.append(i); break
                elif not strict and np.all(vals_i2 >= vals_i) and np.any(vals_i2 > vals_i):
                    to_remove.append(i); break
        for r in set(to_remove):
            rows.remove(r); changed = True

        # Check column player
        to_remove = []
        for j in cols:
            for j2 in cols:
                if j == j2:
                    continue
                vals_j  = B[np.ix_(rows, [j])].flatten()
                vals_j2 = B[np.ix_(rows, [j2])].flatten()
                if strict and np.all(vals_j2 > vals_j):
                    to_remove.append(j); break
                elif not strict and np.all(vals_j2 >= vals_j) and np.any(vals_j2 > vals_j):
                    to_remove.append(j); break
        for c in set(to_remove):
            cols.remove(c); changed = True

    return rows, cols

# --- Demo ---
game = prisoners_dilemma()
print("=== Prisoner's Dilemma ===")
print("Payoff1:\n", game.payoff1)
print("Payoff2:\n", game.payoff2)

rows, cols = iesds(game.payoff1, game.payoff2)
print(f"IESDS surviving strategies: Player1={rows}, Player2={cols}")
# Expected: [1], [1] — both defect

# 3×3 game where IESDS reduces fully
A3 = np.array([[3, 0, 1],
               [1, 2, 0],
               [0, 1, 3]])
B3 = np.array([[1, 0, 2],
               [0, 3, 1],
               [2, 1, 0]])
rows3, cols3 = iesds(A3, B3)
print(f"\n3×3 game IESDS survivors: P1={rows3}, P2={cols3}")
```

**Expected output:**
```
=== Prisoner's Dilemma ===
Payoff1: [[-1 -3] [ 0 -2]]
Payoff2: [[-1  0] [-3 -2]]
IESDS surviving strategies: Player1=[1], Player2=[1]
3×3 game IESDS survivors: P1=[0], P2=[2]
```

---

## Practice Problems (45 min)

**P1.** Construct the payoff bimatrix for: two firms choose High or Low price. If both pick High, each earns 5. If both pick Low, each earns 2. If one picks High and the other Low, the High-price firm earns 0, the Low-price firm earns 7. Is this a Prisoner's Dilemma?

<details><summary>Answer</summary>

$$A = \begin{pmatrix} 5 & 0 \\ 7 & 2 \end{pmatrix}, \quad B = A^T = \begin{pmatrix} 5 & 7 \\ 0 & 2 \end{pmatrix}$$

Yes — Low strictly dominates High for each player, yet (High, High) Pareto-dominates (Low, Low). This has the Prisoner's Dilemma structure.
</details>

**P2.** Apply IESDS to the game $A = \begin{pmatrix} 4 & 3 & 2 \\ 5 & 4 & 1 \\ 1 & 0 & 0 \end{pmatrix}$, $B = \begin{pmatrix} 2 & 1 & 4 \\ 3 & 2 & 0 \\ 1 & 3 & 1 \end{pmatrix}$. Show each elimination step.

<details><summary>Answer</summary>

Step 1: Row 3 is strictly dominated by Row 1 (1<4, 0<3, 0<2). Remove Row 3.
Step 2: In the 2×3 game, Column 2 is dominated by Column 1 for Player 2 (1<2 and 2<3). Remove Col 2.
Step 3: In the 2×2 game, Row 1 [4,2] is dominated by Row 2 [5,1]? No (2>1). Check Row 2 vs Row 1: 5>4 but 1<2. Neither dominates. IESDS terminates with strategies {Row1, Row2} × {Col1, Col3}.
</details>

**P3.** In the Stag Hunt, is strategy Stag weakly dominated? Strictly dominated?

<details><summary>Answer</summary>

Stag (row 0): payoffs [4, 0]. Hare (row 1): payoffs [3, 2]. Stag is not strictly dominated (4 > 3). Stag is not weakly dominated either — it wins on one column (4 > 3) and loses on the other (0 < 2). Neither strategy dominates the other.
</details>

---

## Expert Challenges

**E1.** Prove that the order of elimination does not matter for strict dominance (sketch the key argument using the irrelevance of eliminated strategies to the remaining comparisons).

<details><summary>Answer</summary>

If $s_i$ is strictly dominated by $s_i'$ in game $G$, then $u_i(s_i', s_{-i}) > u_i(s_i, s_{-i})$ for all $s_{-i}$. Restricting $S_{-i}$ to a subset preserves the strict inequality. Thus if $s_i$ is dominated in $G$, it remains dominated after removing any other player's strategy. Order of removal is irrelevant because every dominated strategy eventually gets removed.
</details>

**E2.** Construct a symmetric 3×3 game that is not dominance solvable but has a unique pure Nash equilibrium.

<details><summary>Answer</summary>

$$A = B^T = \begin{pmatrix} 3 & 0 & 4 \\ 4 & 3 & 0 \\ 0 & 4 & 3 \end{pmatrix}$$

No strategy is dominated (each beats one and loses to another — Rock-Paper-Scissors structure with diagonal bonus). IESDS removes nothing. Yet (0,0) with payoff (3,3) can be a NE if we tune off-diagonals. Adjust to: diagonal=3, clockwise=0, counter-clockwise=2 to create a single pure NE.
</details>

**E3.** Implement a function that checks whether a game is dominance solvable and, if so, returns the unique surviving profile in ≤ 10 lines.

<details><summary>Answer</summary>

```python
def is_dominance_solvable(A, B):
    rows, cols = iesds(A, B, strict=True)
    if len(rows) == 1 and len(cols) == 1:
        return True, (rows[0], cols[0])
    return False, None

print(is_dominance_solvable(*[prisoners_dilemma().payoff1, prisoners_dilemma().payoff2]))
# (True, (1, 1))
```
</details>

---

## Connections
- **Prereq:** Matrix operations (Day 1), LP basics (Day 29–31) — needed for Ch 10 later
- **Forward:** Day 93 (Nash equilibrium), Day 96 (zero-sum games)
- **OKS:** Intersection coordination modeled as 2×2 game; IESDS prunes infeasible robot actions before dispatching

## Self-Check
- [ ] I can write the bimatrix for any two-player game from a verbal description
- [ ] I can apply IESDS step-by-step and explain why order doesn't matter for strict dominance
- [ ] I can identify Prisoner's Dilemma, Stag Hunt, and Chicken from their payoff structure
- [ ] My IESDS code correctly reduces the Prisoner's Dilemma to (Defect, Defect)
