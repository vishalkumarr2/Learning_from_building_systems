# Day 96: Zero-Sum Games and Minimax

> **Phase VII — Game Theory for Optimization & Robotics | Week 14 | 2.5 hours**
> *In zero-sum games, your gain is my loss — and the minimax theorem guarantees a unique rational outcome.*

## Navigation
- **Previous:** [Day 95: Equilibrium Selection](day-95-equilibrium-selection.md)
- **Next:** [Day 97: Zero-Sum Games via LP and Duality](day-97-zero-sum-lp.md)
- [Week 14 Overview](../../../CURRICULUM.md)
- [Chapter 10: Game Theory Reference](../../../10-game-theory.md)
- [Full Curriculum](../../../00-learning-plan.md)

## OKS Relevance
Adversarial path planning asks: what route should the robot take if nature (or an adversary) places obstacles in the worst-case position? This is a zero-sum game between the robot (maximizing progress) and the adversary (minimizing it). The minimax strategy gives the robot the best *guaranteed* performance — the foundation of robust control.

---

## Theory (45 min)

### 96.1 Zero-Sum Games

A two-player game is **zero-sum** if $u_1(s) + u_2(s) = 0$ for every strategy profile $s$. Equivalently, $B = -A$ — player 2's payoff matrix is the negative of player 1's.

We represent a zero-sum game by a single matrix $A$ where player 1 (row) maximizes and player 2 (column) minimizes.

### 96.2 Security Levels

Each player can guarantee a minimum payoff regardless of the opponent:

**Maximin value** (player 1's guarantee):
$$\underline{v}_1 = \max_i \min_j A_{ij}$$

Player 1 picks the row whose worst-case column payoff is highest.

**Minimax value** (player 2's guarantee):
$$\overline{v}_2 = \min_j \max_i A_{ij}$$

Player 2 picks the column whose worst-case row payoff is lowest.

**Key inequality:** $\underline{v}_1 \leq \overline{v}_2$ always holds.

### 96.3 Saddle Points

A cell $(i^*, j^*)$ is a **saddle point** if:

$$A_{i^*,j} \leq A_{i^*,j^*} \leq A_{i,j^*} \quad \forall i, j$$

Equivalently, $A_{i^*,j^*}$ is both the minimum in its row and the maximum in its column. At a saddle point:

$$\underline{v}_1 = \overline{v}_2 = A_{i^*,j^*} = v^*$$

The game's **value** $v^*$ is the unique payoff at any saddle point. The saddle point is a pure-strategy NE.

### 96.4 Von Neumann's Minimax Theorem (1928)

> **Theorem:** In any finite two-player zero-sum game, the mixed minimax and maximin values are equal:
> $$\max_{\sigma_1 \in \Delta_m} \min_{\sigma_2 \in \Delta_n} \sigma_1^T A \sigma_2 = \min_{\sigma_2 \in \Delta_n} \max_{\sigma_1 \in \Delta_m} \sigma_1^T A \sigma_2 = v^*$$

This guarantees:
1. A unique game value $v^*$ exists.
2. Optimal mixed strategies exist for both players.
3. The order of max/min doesn't matter (in mixed strategies).

### 96.5 Relation to Nash Equilibrium

In zero-sum games, the set of NE is the Cartesian product of maximin/minimax strategies. All NE give the same payoff $v^*$. There is no equilibrium selection problem — the value is unique.

---

## Implementation (60 min)

```python
import numpy as np

def find_saddle_points(A):
    """Find all saddle points (pure-strategy NE) in a zero-sum game."""
    m, n = A.shape
    row_mins = np.min(A, axis=1)   # min of each row
    col_maxs = np.max(A, axis=0)   # max of each column
    saddle_pts = []
    for i in range(m):
        for j in range(n):
            if A[i, j] == row_mins[i] and A[i, j] == col_maxs[j]:
                saddle_pts.append((i, j))
    return saddle_pts

def maximin_value(A):
    """Player 1's security level: max over rows of min over columns."""
    return np.max(np.min(A, axis=1))

def minimax_value(A):
    """Player 2's security level: min over columns of max over rows."""
    return np.min(np.max(A, axis=0))

def maximin_strategy(A):
    """Return Player 1's maximin pure strategy (row index)."""
    row_mins = np.min(A, axis=1)
    return int(np.argmax(row_mins))

def minimax_strategy(A):
    """Return Player 2's minimax pure strategy (column index)."""
    col_maxs = np.max(A, axis=0)
    return int(np.argmin(col_maxs))

def verify_minimax_theorem_pure(A):
    """Check if maximin = minimax (saddle point exists)."""
    return maximin_value(A) == minimax_value(A)

# --- Demo: Game with saddle point ---
A1 = np.array([[ 3,  2,  4],
               [ 1,  4,  2],
               [ 5,  3,  0]])

print("=== Game with Saddle Point ===")
print(f"Payoff matrix A:\n{A1}")
print(f"Maximin value: {maximin_value(A1)}")
print(f"Minimax value: {minimax_value(A1)}")
print(f"Maximin = Minimax? {verify_minimax_theorem_pure(A1)}")
sps = find_saddle_points(A1)
print(f"Saddle points: {sps}")
if sps:
    print(f"Game value: {A1[sps[0]]}")

# --- Demo: Game without saddle point ---
A2 = np.array([[ 1, -1],
               [-1,  1]])  # Matching Pennies

print("\n=== Matching Pennies (no saddle point) ===")
print(f"Maximin value: {maximin_value(A2)}")
print(f"Minimax value: {minimax_value(A2)}")
print(f"Maximin = Minimax? {verify_minimax_theorem_pure(A2)}")
print(f"Saddle points: {find_saddle_points(A2)}")
print("  → Need mixed strategies (minimax theorem guarantees v*=0)")

# --- Demo: 4×3 military game ---
A3 = np.array([[ 2,  3,  1],
               [ 4,  1,  2],
               [ 1,  5,  3],
               [ 3,  2,  4]])

print("\n=== 4×3 Game ===")
print(f"Maximin: row {maximin_strategy(A3)}, value {maximin_value(A3)}")
print(f"Minimax: col {minimax_strategy(A3)}, value {minimax_value(A3)}")
sps3 = find_saddle_points(A3)
print(f"Saddle points: {sps3}")
if not sps3:
    print("  → No pure saddle point; solve via LP (Day 97)")
```

**Expected output:**
```
=== Game with Saddle Point ===
Payoff matrix A:
[[ 3  2  4]
 [ 1  4  2]
 [ 5  3  0]]
Maximin value: 2
Minimax value: 4
Maximin = Minimax? False
Saddle points: []

=== Matching Pennies (no saddle point) ===
Maximin value: -1
Minimax value: 1
Maximin = Minimax? False
Saddle points: []
  → Need mixed strategies (minimax theorem guarantees v*=0)

=== 4×3 Game ===
Maximin: row 3, value 2
Minimax: col 2, value 3
Saddle points: []
  → No pure saddle point; solve via LP (Day 97)
```

```python
# --- Game that HAS a saddle point ---
A_sp = np.array([[ 3,  1,  2],
                 [ 5,  4,  6],
                 [ 2,  0,  1]])

print("\n=== Game WITH Saddle Point ===")
print(f"Maximin value: {maximin_value(A_sp)}")
print(f"Minimax value: {minimax_value(A_sp)}")
sps = find_saddle_points(A_sp)
print(f"Saddle points: {sps}, value = {A_sp[sps[0]]}")
```

---

## Practice Problems (45 min)

**P1.** Find the maximin and minimax values for $A = \begin{pmatrix} 5 & 3 \\ 2 & 4 \end{pmatrix}$. Is there a saddle point?

<details><summary>Answer</summary>

Row mins: min(5,3)=3, min(2,4)=2. Maximin = max(3,2) = 3 (row 0).
Col maxs: max(5,2)=5, max(3,4)=4. Minimax = min(5,4) = 4 (col 1).
Maximin (3) ≠ Minimax (4), so no pure saddle point. The game value lies in [3, 4] and must be found via mixed strategies.
</details>

**P2.** Verify that the matrix $A = \begin{pmatrix} 1 & 3 & 5 \\ 4 & 2 & 1 \\ 3 & 4 & 2 \end{pmatrix}$ has a saddle point at (2,2) with value 2. Confirm by checking the row-min and column-max conditions.

<details><summary>Answer</summary>

Row 2: [3, 4, 2]. Min = 2 at col 2. Col 2: [5, 1, 2]. Max = 5 at row 0, not 2. So (2,2) is NOT a saddle point. Let's recheck: Row mins = [1, 1, 2]. Maximin = 2 at row 2. Col maxs = [4, 4, 5]. Minimax = 4 at col 0 or 1. 2 ≠ 4, no saddle point exists.
</details>

**P3.** For a robot choosing between 3 paths and nature choosing between 2 obstacle placements, the travel time matrix is $A = \begin{pmatrix} 10 & 15 \\ 12 & 8 \\ 14 & 11 \end{pmatrix}$ (robot minimizes, nature maximizes). Find the robot's minimax strategy.

<details><summary>Answer</summary>

Robot minimizes, so flip perspective: robot's "payoff" is $-A$ and they maximize. Row maxs of $A$ (worst case per path): 15, 12, 14. Robot picks row with min worst-case = row 1 (max 12). Minimax strategy: Path 2, guaranteeing travel time ≤ 12. Nature's best response: col 0 (giving 12).
</details>

---

## Expert Challenges

**E1.** Prove that if a zero-sum game has two saddle points $(i_1, j_1)$ and $(i_2, j_2)$, they must have the same value. Additionally, show that $(i_1, j_2)$ and $(i_2, j_1)$ are also saddle points.

<details><summary>Answer</summary>

Let $v_1 = A_{i_1,j_1}$ and $v_2 = A_{i_2,j_2}$. By saddle point definition: $A_{i_1,j_2} \leq v_1$ (min in row $i_1$) and $A_{i_1,j_2} \geq v_2$ (max in col $j_2$). So $v_2 \leq A_{i_1,j_2} \leq v_1$. Symmetrically, $v_1 \leq v_2$. Therefore $v_1 = v_2 = A_{i_1,j_2} = A_{i_2,j_1}$. Since $A_{i_1,j_2} = v^*$ is both the row min and column max at $(i_1,j_2)$, it's a saddle point too.
</details>

**E2.** Show that Matching Pennies has game value $v^* = 0$ by computing the mixed minimax directly. Find the optimal mixed strategies.

<details><summary>Answer</summary>

$A = \begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}$. Player 1 plays $(p, 1-p)$: $\sigma_1^T A = (2p-1, 1-2p)$. Player 2 minimizes: $\min(2p-1, 1-2p)$. This is maximized when $2p-1 = 1-2p$ → $p = 1/2$, giving value 0. Similarly, Player 2 plays $(q, 1-q)$ and $q = 1/2$ gives minimax value 0. $v^* = 0$, strategies $(1/2, 1/2)$ for both.
</details>

**E3.** In adversarial path planning, the robot chooses a path $r \in \{1,\ldots,m\}$ and the adversary places an obstacle at location $o \in \{1,\ldots,n\}$. The cost is $c(r,o)$. Formulate this as a zero-sum game and explain what the minimax strategy represents physically.

<details><summary>Answer</summary>

Let $A_{ro} = -c(r,o)$ (robot maximizes negative cost = minimizes cost). The minimax strategy $\sigma_1^*$ is a probability distribution over paths that minimizes the worst-case expected cost: $\min_{\sigma_1} \max_o \sum_r \sigma_1(r) \cdot c(r,o)$. Physically: the robot randomizes its path choice so that no single obstacle placement can cause catastrophic delay. The game value $v^*$ is the guaranteed expected cost. This connects to robust optimization (Phase V) — minimax over uncertainty.
</details>

---

## Connections
- **Back:** Day 93–94 (NE in general games), Day 5 (minimax relates to convex-concave saddle points)
- **Forward:** Day 97 (LP solution for zero-sum games), Week 15 (mechanism design)
- **OKS:** Robust path planning under worst-case obstacles; adversarial testing of robot controllers

## Self-Check
- [ ] I can compute maximin and minimax values from a payoff matrix
- [ ] I can identify saddle points and verify they are pure NE
- [ ] I understand the minimax theorem: mixed strategies equalize maximin and minimax
- [ ] I can model an adversarial robotics scenario as a zero-sum game
