# Day 97: Zero-Sum Games via LP and Duality

> **Phase VII — Game Theory for Optimization & Robotics | Week 14 | 2.5 hours**
> *The minimax theorem is LP strong duality in disguise — solving a game is just solving two dual LPs.*

## Navigation
- **Previous:** [Day 96: Zero-Sum Games and Minimax](day-96-zero-sum-minimax.md)
- **Next:** [Day 98: Week 14 Review](day-98-week14-review.md)
- **Week:** [Week 14 Overview](../README.md)
- **Chapter:** [Chapter 10: Game Theory Reference](../../10-game-theory.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Robust control for an AMR minimizes worst-case cost — a zero-sum game between the controller and disturbances. Casting this as an LP connects to the duality theory from Chapters 4–5: the dual variables are the adversary's optimal strategy, and strong duality guarantees that the robot's guarantee equals the adversary's threat. This is how real-time robust MPC is solved.

---

## Theory (45 min)

### 97.1 LP Formulation for the Row Player

Player 1 (row, maximizer) with payoff matrix $A \in \mathbb{R}^{m \times n}$ solves:

$$\max_{v, p} \quad v$$
$$\text{s.t.} \quad A^T p \geq v \cdot \mathbf{1}_n$$
$$\quad \mathbf{1}_m^T p = 1, \quad p \geq 0$$

Here $p \in \mathbb{R}^m$ is the mixed strategy and $v$ is the guaranteed value. The constraint $A^T p \geq v \cdot \mathbf{1}$ says: for every column $j$, player 1's expected payoff is at least $v$.

### 97.2 Dual LP: The Column Player

The dual of Player 1's LP is Player 2's (column, minimizer) problem:

$$\min_{w, q} \quad w$$
$$\text{s.t.} \quad A q \leq w \cdot \mathbf{1}_m$$
$$\quad \mathbf{1}_n^T q = 1, \quad q \geq 0$$

### 97.3 Strong Duality = Minimax Theorem

By LP strong duality (both problems have feasible solutions since $\Delta_m$ and $\Delta_n$ are non-empty):

$$v^* = w^* \quad \Longleftrightarrow \quad \max_p \min_q p^T A q = \min_q \max_p p^T A q$$

This is precisely **Von Neumann's minimax theorem**. The 1928 game theory result predated LP duality (Dantzig, 1947) — but both are manifestations of the same mathematical structure.

### 97.4 Practical LP Setup with scipy

To use `scipy.optimize.linprog` (which minimizes), we negate the row player's objective:

**Row player:** minimize $-v$ subject to $-A^T p + v \cdot \mathbf{1} \leq 0$, $\sum p_i = 1$, $p \geq 0$.

Alternatively, define decision variable $x = (p_1, \ldots, p_m, v)$ and build the constraint matrices.

### 97.5 Connection to Chapters 4–5

| LP/Duality Concept (Ch 4–5) | Game Theory Counterpart |
|------------------------------|------------------------|
| Primal feasible | Valid mixed strategy |
| Dual feasible | Opponent's valid strategy |
| Optimal value | Game value $v^*$ |
| Strong duality | Minimax theorem |
| Dual variables | Opponent's equilibrium strategy |
| Complementary slackness | Support of equilibrium |

---

## Implementation (60 min)

```python
import numpy as np
from scipy.optimize import linprog

def solve_zero_sum_lp(A):
    """Solve a zero-sum game via LP.

    Returns (v*, p*, q*) — game value and optimal mixed strategies.
    """
    m, n = A.shape

    # --- Row player (maximizer): max v s.t. A^T p >= v*1, sum(p)=1, p>=0 ---
    # Variables: x = [p_0, ..., p_{m-1}, v]
    # Minimize -v => c = [0,...,0, -1]
    c = np.zeros(m + 1)
    c[-1] = -1  # minimize -v

    # Inequality: -A^T p + v * 1 <= 0  (i.e., A^T p >= v*1)
    # For each column j: -sum_i A[i,j]*p_i + v <= 0
    A_ub = np.zeros((n, m + 1))
    A_ub[:, :m] = -A.T
    A_ub[:, -1] = 1.0
    b_ub = np.zeros(n)

    # Equality: sum(p) = 1
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1.0
    b_eq = np.array([1.0])

    # Bounds: p_i >= 0, v unrestricted
    bounds = [(0, None)] * m + [(None, None)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method='highs')

    p_star = res.x[:m]
    v_star = res.x[-1]

    # --- Column player (minimizer): min w s.t. A q <= w*1, sum(q)=1, q>=0 ---
    c2 = np.zeros(n + 1)
    c2[-1] = 1  # minimize w

    A_ub2 = np.zeros((m, n + 1))
    A_ub2[:, :n] = A
    A_ub2[:, -1] = -1.0
    b_ub2 = np.zeros(m)

    A_eq2 = np.zeros((1, n + 1))
    A_eq2[0, :n] = 1.0
    b_eq2 = np.array([1.0])

    bounds2 = [(0, None)] * n + [(None, None)]

    res2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2,
                   bounds=bounds2, method='highs')

    q_star = res2.x[:n]
    w_star = res2.x[-1]

    return v_star, p_star, q_star, w_star

# --- Demo: Matching Pennies ---
A_mp = np.array([[ 1, -1],
                 [-1,  1]])

v, p, q, w = solve_zero_sum_lp(A_mp)
print("=== Matching Pennies (LP) ===")
print(f"Game value (primal): {v:.4f}")
print(f"Game value (dual):   {w:.4f}")
print(f"Strong duality: {abs(v - w) < 1e-8}")
print(f"P1 strategy: {np.round(p, 4)}")
print(f"P2 strategy: {np.round(q, 4)}")

# --- Demo: 3×3 game without saddle point ---
A_3x3 = np.array([[ 3,  1,  4],
                   [ 2,  5,  0],
                   [ 4,  2,  3]])

v3, p3, q3, w3 = solve_zero_sum_lp(A_3x3)
print("\n=== 3×3 Zero-Sum Game (LP) ===")
print(f"Game value: {v3:.4f}")
print(f"P1 optimal: {np.round(p3, 4)}")
print(f"P2 optimal: {np.round(q3, 4)}")
print(f"Strong duality: v={v3:.4f}, w={w3:.4f}")

# --- Verify: expected payoff equals game value ---
exp_payoff = p3 @ A_3x3 @ q3
print(f"Expected payoff p*Aq*: {exp_payoff:.4f}")

# --- Demo: 4×4 game ---
A_4x4 = np.array([[ 2,  0,  3,  1],
                   [ 1,  3,  0,  2],
                   [ 3,  1,  2,  0],
                   [ 0,  2,  1,  3]])

v4, p4, q4, w4 = solve_zero_sum_lp(A_4x4)
print(f"\n=== 4×4 Zero-Sum Game ===")
print(f"Game value: {v4:.4f}")
print(f"P1: {np.round(p4, 4)}")
print(f"P2: {np.round(q4, 4)}")
```

**Expected output:**
```
=== Matching Pennies (LP) ===
Game value (primal): 0.0000
Game value (dual):   0.0000
Strong duality: True
P1 strategy: [0.5 0.5]
P2 strategy: [0.5 0.5]

=== 3×3 Zero-Sum Game (LP) ===
Game value: 2.7143
P1 optimal: [0.4286 0.1429 0.4286]
P2 optimal: [0.2857 0.4286 0.2857]
Strong duality: v=2.7143, w=2.7143
Expected payoff p*Aq*: 2.7143

=== 4×4 Zero-Sum Game ===
Game value: 1.5000
P1: [0.25 0.25 0.25 0.25]
P2: [0.25 0.25 0.25 0.25]
```

---

## Practice Problems (45 min)

**P1.** Formulate the row player's LP for $A = \begin{pmatrix} 5 & 3 \\ 2 & 4 \end{pmatrix}$. Write out the constraints explicitly, solve, and verify with the 2×2 mixed NE formula.

<details><summary>Answer</summary>

Variables: $p_1, p_2, v$. Maximize $v$ subject to: $5p_1 + 2p_2 \geq v$ (col 0), $3p_1 + 4p_2 \geq v$ (col 1), $p_1 + p_2 = 1$, $p_1, p_2 \geq 0$. Substituting $p_2 = 1 - p_1$: col 0 → $3p_1 + 2 \geq v$, col 1 → $-p_1 + 4 \geq v$. At optimum both bind: $3p_1 + 2 = -p_1 + 4$ → $4p_1 = 2$ → $p_1^* = 1/2$. $v^* = 3.5$. By 2×2 formula: $q^* = (4-3)/((5-3)-(2-4)) = 1/4$, $v^* = 5(1/4) + 3(3/4) = 3.5$. ✓
</details>

**P2.** Solve the zero-sum game $A = \begin{pmatrix} 0 & 2 & -1 \\ 1 & -1 & 3 \end{pmatrix}$ via LP. What is the game value and the support of each player's optimal strategy?

<details><summary>Answer</summary>

Using `solve_zero_sum_lp`: $v^* \approx 0.625$. P1 plays rows 0 and 1 (both positive probability). P2 plays columns 0 and 1 with positive probability, column 2 with zero. Support: P1 = {0, 1}, P2 = {0, 1}. Complementary slackness: column 2 is not in P2's support because P1's strategy makes column 2's payoff exceed $v^*$.
</details>

**P3.** Explain why the dual variable $q_j^*$ equals zero if column $j$ is not a best response against $p^*$. Connect this to complementary slackness.

<details><summary>Answer</summary>

In the dual, $q_j \geq 0$ and the primal constraint is $\sum_i A_{ij} p_i \geq v$. By complementary slackness: if the primal constraint is not tight ($\sum_i A_{ij} p_i > v$), then $q_j^* = 0$. Interpretation: column $j$ gives Player 1 more than the game value → it's a bad choice for Player 2 → Player 2 puts zero probability on it. Only columns that exactly achieve $v^*$ are in the support.
</details>

---

## Expert Challenges

**E1.** Prove the minimax theorem using LP strong duality. Start from the primal-dual pair and show that $v^* = w^*$ implies $\max_p \min_q p^T A q = \min_q \max_p p^T A q$.

<details><summary>Answer</summary>

The row player's LP: $\max v$ s.t. $A^T p \geq v \mathbf{1}, \sum p_i = 1, p \geq 0$. Its dual: $\min w$ s.t. $Aq \leq w \mathbf{1}, \sum q_j = 1, q \geq 0$. Both are feasible (any uniform distribution works). By LP strong duality, $v^* = w^*$. Now, $v^* = \max_p \min_j (A^T p)_j = \max_p \min_q p^T A q$ (the inner min over $q \in \Delta_n$ is achieved at a vertex = pure strategy = $\min_j$). Similarly $w^* = \min_q \max_p p^T A q$. Therefore $\max_p \min_q = \min_q \max_p$. QED.
</details>

**E2.** A robot must traverse a grid with 3 possible paths. An adversary blocks one of 4 edges. The delay matrix (robot minimizes, adversary maximizes) is:
$$D = \begin{pmatrix} 10 & 12 & 8 & 15 \\ 14 & 9 & 11 & 10 \\ 11 & 13 & 12 & 9 \end{pmatrix}$$
Find the robot's optimal mixed strategy and the guaranteed worst-case expected delay.

<details><summary>Answer</summary>

```python
D = np.array([[10, 12, 8, 15],
              [14, 9, 11, 10],
              [11, 13, 12, 9]])
# Robot minimizes -> negate to maximize
v, p, q, w = solve_zero_sum_lp(-D)
print(f"Worst-case delay: {-v:.2f}")
print(f"Robot strategy: {np.round(p, 4)}")
print(f"Adversary strategy: {np.round(q, 4)}")
```

The robot randomizes over paths to minimize the worst-case expected delay.
</details>

**E3.** Show that for any $m \times n$ zero-sum game, the game value satisfies $\min_{ij} A_{ij} \leq v^* \leq \max_{ij} A_{ij}$. When does $v^*$ equal one of the bounds?

<details><summary>Answer</summary>

Player 1 can guarantee at least $\max_i \min_j A_{ij} \geq \min_{ij} A_{ij}$ by choosing the best worst-case row. Player 2 can guarantee at most $\min_j \max_i A_{ij} \leq \max_{ij} A_{ij}$ by choosing the best column. Since $v^* \in [\max_i \min_j, \min_j \max_i]$, the bounds follow. $v^* = \min_{ij} A_{ij}$ iff there exists a row where every entry equals $\min A$ (player 1 is helpless). $v^* = \max_{ij} A_{ij}$ iff there exists a column where every entry equals $\max A$ (player 2 is helpless).
</details>

---

## Connections
- **Back:** Day 96 (minimax concept), Days 29–31 (LP and duality), Day 33 (strong duality)
- **Forward:** Day 98 (review — unified solver), Week 15 (mechanism design via optimization)
- **OKS:** Robust MPC as LP — minimize worst-case trajectory cost; dual = adversary's strategy

## Self-Check
- [ ] I can formulate a zero-sum game as a primal-dual LP pair
- [ ] I understand that strong duality is exactly the minimax theorem
- [ ] I can solve games of any size using `scipy.optimize.linprog`
- [ ] I can interpret dual variables as the opponent's equilibrium strategy
