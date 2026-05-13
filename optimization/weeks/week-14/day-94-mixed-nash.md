# Day 94: Mixed-Strategy Nash Equilibrium

> **Phase VII — Game Theory for Optimization & Robotics | Week 14 | 2.5 hours**
> *When pure strategies fail, randomization makes every player indifferent — and that's the equilibrium.*

## Navigation
- **Previous:** [Day 93: Pure-Strategy Nash Equilibrium](day-93-pure-nash.md)
- **Next:** [Day 95: Equilibrium Selection](day-95-equilibrium-selection.md)
- [Week 14 Overview](../../../CURRICULUM.md)
- [Chapter 10: Game Theory Reference](../../../10-game-theory.md)
- [Full Curriculum](../../../00-learning-plan.md)

## OKS Relevance
Deterministic robot patrol routes are predictable and exploitable (by adversaries or by congestion). Mixed strategies assign randomized probabilities over routes, ensuring unpredictability. In warehouse pick optimization, stochastic task selection prevents bottlenecks at popular aisles — each robot randomizes according to a mixed NE.

---

## Theory (45 min)

### 94.1 Mixed Strategies

A **mixed strategy** for player $i$ is a probability distribution $\sigma_i \in \Delta(S_i)$ over pure strategies:

$$\sigma_i = (p_1, p_2, \ldots, p_{|S_i|}), \quad p_k \geq 0, \quad \sum_k p_k = 1$$

The **support** of $\sigma_i$ is $\text{supp}(\sigma_i) = \{s_k : p_k > 0\}$.

**Expected payoff** under mixed profile $(\sigma_1, \sigma_2)$:

$$U_1(\sigma_1, \sigma_2) = \sigma_1^T A \sigma_2 = \sum_{i,j} p_i q_j A_{ij}$$

### 94.2 The Indifference Principle

At a mixed NE, each player must be **indifferent** among all strategies in their support. If player 1 mixes over strategies $\{i_1, i_2\}$, then:

$$\sum_j q_j A_{i_1,j} = \sum_j q_j A_{i_2,j}$$

Player 2's mixing probabilities $q$ make player 1 indifferent; player 1's mixing probabilities $p$ make player 2 indifferent. Each player's randomization is pinned by the **other** player's payoffs.

### 94.3 Nash's Existence Theorem

> **Theorem (Nash, 1950):** Every finite game has at least one Nash equilibrium in mixed strategies.

This is a consequence of Brouwer's fixed-point theorem applied to the best-response correspondence. The NE may be pure (a degenerate mixed strategy).

### 94.4 Computing Mixed NE in 2×2 Games

For a 2×2 game with $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$, $B = \begin{pmatrix} e & f \\ g & h \end{pmatrix}$, player 1 mixes $(p, 1-p)$ and player 2 mixes $(q, 1-q)$.

**Player 2's indifference** (from Player 1's mix): $p \cdot e + (1-p) \cdot g = p \cdot f + (1-p) \cdot h$

$$p^* = \frac{h - g}{(e - g) - (f - h)}$$

**Player 1's indifference** (from Player 2's mix): $q \cdot a + (1-q) \cdot b = q \cdot c + (1-q) \cdot d$

$$q^* = \frac{d - b}{(a - b) - (c - d)}$$

Valid mixed NE requires $p^*, q^* \in (0, 1)$.

---

## Implementation (60 min)

```python
import numpy as np

def find_mixed_nash_2x2(A, B):
    """Find the mixed-strategy NE for a 2×2 game.

    Returns (p*, q*) where p* is P1's probability on row 0,
    q* is P2's probability on column 0. Returns None if no interior mixed NE.
    """
    # Player 1's mixing prob from Player 2's indifference
    e, f, g, h = B[0,0], B[0,1], B[1,0], B[1,1]
    denom_p = (e - g) - (f - h)
    if abs(denom_p) < 1e-12:
        return None  # degenerate
    p_star = (h - g) / denom_p

    # Player 2's mixing prob from Player 1's indifference
    a, b, c, d = A[0,0], A[0,1], A[1,0], A[1,1]
    denom_q = (a - b) - (c - d)
    if abs(denom_q) < 1e-12:
        return None
    q_star = (d - b) / denom_q

    if 0 < p_star < 1 and 0 < q_star < 1:
        return p_star, q_star
    return None

def expected_payoffs(A, B, p, q):
    """Compute expected payoffs given mixing probabilities."""
    sigma1 = np.array([p, 1-p])
    sigma2 = np.array([q, 1-q])
    u1 = sigma1 @ A @ sigma2
    u2 = sigma1 @ B @ sigma2
    return u1, u2

def verify_mixed_ne(A, B, p, q, tol=1e-9):
    """Verify that (p, q) is a mixed NE by checking indifference."""
    sigma2 = np.array([q, 1-q])
    payoffs_p1 = A @ sigma2  # payoff for each of P1's pure strategies
    # P1 should be indifferent: payoffs_p1[0] ≈ payoffs_p1[1]
    p1_indiff = abs(payoffs_p1[0] - payoffs_p1[1]) < tol

    sigma1 = np.array([p, 1-p])
    payoffs_p2 = sigma1 @ B  # payoff for each of P2's pure strategies
    p2_indiff = abs(payoffs_p2[0] - payoffs_p2[1]) < tol

    return p1_indiff and p2_indiff

# --- Demo: Matching Pennies ---
A_mp = np.array([[ 1, -1],
                 [-1,  1]])
B_mp = np.array([[-1,  1],
                 [ 1, -1]])

result = find_mixed_nash_2x2(A_mp, B_mp)
print("=== Matching Pennies ===")
print(f"Mixed NE: p*={result[0]:.4f}, q*={result[1]:.4f}")
u1, u2 = expected_payoffs(A_mp, B_mp, *result)
print(f"Expected payoffs: ({u1:.4f}, {u2:.4f})")
print(f"Verified: {verify_mixed_ne(A_mp, B_mp, *result)}")

# --- Demo: Battle of the Sexes ---
A_bos = np.array([[3, 0],
                  [0, 2]])
B_bos = np.array([[2, 0],
                  [0, 3]])

result_bos = find_mixed_nash_2x2(A_bos, B_bos)
print("\n=== Battle of the Sexes ===")
print(f"Mixed NE: p*={result_bos[0]:.4f}, q*={result_bos[1]:.4f}")
u1, u2 = expected_payoffs(A_bos, B_bos, *result_bos)
print(f"Expected payoffs: ({u1:.4f}, {u2:.4f})")
print(f"Verified: {verify_mixed_ne(A_bos, B_bos, *result_bos)}")

# --- Demo: Chicken ---
A_ch = np.array([[ 0, -1],
                 [ 1, -5]])
B_ch = np.array([[ 0,  1],
                 [-1, -5]])

result_ch = find_mixed_nash_2x2(A_ch, B_ch)
print("\n=== Chicken ===")
print(f"Mixed NE: p*={result_ch[0]:.4f}, q*={result_ch[1]:.4f}")
print(f"Verified: {verify_mixed_ne(A_ch, B_ch, *result_ch)}")
```

**Expected output:**
```
=== Matching Pennies ===
Mixed NE: p*=0.5000, q*=0.5000
Expected payoffs: (0.0000, 0.0000)
Verified: True

=== Battle of the Sexes ===
Mixed NE: p*=0.6000, q*=0.4000
Expected payoffs: (1.2000, 1.2000)
Verified: True

=== Chicken ===
Mixed NE: p*=0.8000, q*=0.8000
Verified: True
```

---

## Practice Problems (45 min)

**P1.** Compute the mixed NE of the game $A = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix}$, $B = \begin{pmatrix} 1 & 0 \\ 0 & 2 \end{pmatrix}$ by hand. Verify that both players are indifferent.

<details><summary>Answer</summary>

Player 2 indifference: $p \cdot 1 + (1-p) \cdot 0 = p \cdot 0 + (1-p) \cdot 2$ → $p = 2 - 2p$ → $p^* = 2/3$.
Player 1 indifference: $q \cdot 2 + (1-q) \cdot 0 = q \cdot 0 + (1-q) \cdot 1$ → $2q = 1 - q$ → $q^* = 1/3$.
Verify P1: $U_1(\text{Row 0}) = (1/3)(2) + (2/3)(0) = 2/3$. $U_1(\text{Row 1}) = (1/3)(0) + (2/3)(1) = 2/3$. ✓ Indifferent.
</details>

**P2.** In Matching Pennies, explain why the mixed NE must be $(1/2, 1/2)$ by symmetry. What is each player's expected value of the game?

<details><summary>Answer</summary>

Matching Pennies is symmetric under swapping H/T for the column player. The unique NE must respect this symmetry, so $q^* = 1/2$. Similarly for the row player: $p^* = 1/2$. Expected payoff: $U_1 = (1/2)(1/2)(1) + (1/2)(1/2)(-1) + (1/2)(1/2)(-1) + (1/2)(1/2)(1) = 0$. Both players get 0 in expectation.
</details>

**P3.** A 2×2 game has $A = \begin{pmatrix} 5 & 1 \\ 3 & 4 \end{pmatrix}$, $B = \begin{pmatrix} 2 & 3 \\ 4 & 1 \end{pmatrix}$. Does a mixed NE exist? If so, find it. What are the expected payoffs?

<details><summary>Answer</summary>

$q^* = (4-1)/((5-1)-(3-4)) = 3/5$. $p^* = (1-4)/((2-4)-(3-1)) = -3/(-4) = 3/4$. Both in $(0,1)$ so mixed NE exists: $p^*=3/4$, $q^*=3/5$. Expected payoff for P1: $(3/5)(5)+(2/5)(1) = 17/5 = 3.4$ (from row 0, using indifference any row gives same).
</details>

---

## Expert Challenges

**E1.** Prove: in a 2×2 game with two pure NE on the diagonal, there always exists a third mixed NE.

<details><summary>Answer</summary>

If $(0,0)$ and $(1,1)$ are pure NE, then $A_{00} \geq A_{10}$, $A_{11} \geq A_{01}$, $B_{00} \geq B_{01}$, $B_{11} \geq B_{10}$. The denominators in the mixed NE formulas are: $(e-g)-(f-h)$ and $(a-b)-(c-d)$. From the NE conditions, $(a-c) \geq 0$ and $(d-b) \geq 0$ with the off-diagonal terms being smaller. This ensures $p^*, q^* \in (0,1)$, giving a valid interior mixed NE. (Formal proof uses the intermediate value theorem on the best-response correspondence.)
</details>

**E2.** Implement support enumeration for finding all NE (pure and mixed) in an arbitrary $m \times n$ game. Test on a 3×3 game.

<details><summary>Answer</summary>

```python
from itertools import combinations

def support_enumeration(A, B):
    """Find all NE by enumerating support pairs."""
    m, n = A.shape
    equilibria = []
    for k1 in range(1, m+1):
        for k2 in range(1, n+1):
            for supp1 in combinations(range(m), k1):
                for supp2 in combinations(range(n), k2):
                    ne = _solve_support(A, B, supp1, supp2)
                    if ne is not None:
                        equilibria.append(ne)
    return equilibria

def _solve_support(A, B, supp1, supp2):
    """Solve for NE with given supports using indifference + normalization."""
    k1, k2 = len(supp1), len(supp2)
    # P2's strategy q makes P1 indifferent over supp1
    # Build system: A[supp1, :][:, supp2] @ q = v * ones, sum(q) = 1
    As = A[np.ix_(list(supp1), list(supp2))]
    M = np.vstack([As[1:] - As[0:1], np.ones((1, k2))])
    rhs = np.append(np.zeros(k1-1), 1.0)
    try:
        q_supp = np.linalg.solve(M[:k2, :], rhs[:k2]) if M.shape[0] >= k2 else None
        if q_supp is None or np.any(q_supp < -1e-10):
            return None
    except np.linalg.LinAlgError:
        return None
    q = np.zeros(A.shape[1]); q[list(supp2)] = q_supp
    if np.any(q < -1e-10):
        return None
    # Similarly for p
    Bs = B[np.ix_(list(supp1), list(supp2))]
    Mt = np.vstack([Bs.T[1:] - Bs.T[0:1], np.ones((1, k1))])
    rhst = np.append(np.zeros(k2-1), 1.0)
    try:
        p_supp = np.linalg.solve(Mt[:k1, :], rhst[:k1]) if Mt.shape[0] >= k1 else None
        if p_supp is None or np.any(p_supp < -1e-10):
            return None
    except np.linalg.LinAlgError:
        return None
    p = np.zeros(A.shape[0]); p[list(supp1)] = p_supp
    if np.any(p < -1e-10):
        return None
    return (p, q)

# Test on 3×3 coordination game
A3 = np.array([[3,0,0],[0,3,0],[0,0,3]])
B3 = A3.copy()
eqs = support_enumeration(A3, B3)
print(f"Found {len(eqs)} equilibria")
```
</details>

**E3.** Show that in the Battle of the Sexes, the mixed NE gives both players a *lower* expected payoff than either pure NE. Interpret this economically.

<details><summary>Answer</summary>

Pure NE payoffs: (3,2) and (2,3). Mixed NE: $p^*=3/5$, $q^*=2/5$, expected payoffs $(6/5, 6/5) = (1.2, 1.2)$. Both get 1.2 < 2, which is worse than either pure NE. Interpretation: randomization creates miscoordination — sometimes players end up at different locations, getting 0. The mixed NE is a "fair" but inefficient outcome reflecting the coordination failure.
</details>

---

## Connections
- **Back:** Day 93 (pure NE, best response), Day 5 (convexity — NE existence via fixed-point)
- **Forward:** Day 95 (selecting among equilibria), Day 96 (zero-sum mixed strategies via minimax)
- **OKS:** Stochastic patrol scheduling; mixed NE in robot task selection prevents herding

## Self-Check
- [ ] I can solve for the mixed NE of any 2×2 game using the indifference principle
- [ ] I understand why Player 1's mixing probability depends on Player 2's payoffs (not their own)
- [ ] I can verify a mixed NE by checking both players' indifference conditions
- [ ] I know Nash's existence theorem and its implication: every finite game has at least one NE
