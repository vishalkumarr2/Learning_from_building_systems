# Day 106: Computing Nash Equilibria

> **Phase VII — Game Theory for Optimization & Robotics | Week 16 | 2.5 hours**
> *Finding equilibria is fundamentally hard — understanding why shapes how we design real-time robot decision systems.*

## Navigation
- **Previous:** [Day 105: Week 15 Review](../week-15/day-105-week15-review.md)
- **Next:** [Day 107: Learning in Games — Fictitious Play](day-107-fictitious-play.md)
- **Week:** [Week 16 Overview](../README.md)
- **Chapter:** [Chapter 10: Game Theory Reference](../../10-game-theory.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
When multiple AMRs must compute equilibrium strategies in real-time — for corridor negotiation, charging station allocation, or task bidding — the computational complexity of Nash equilibrium directly limits what's feasible onboard. Understanding PPAD-completeness explains why we rely on approximate or structured equilibria (potential games, zero-sum) rather than general NE computation.

---

## Theory (45 min)

### 106.1 The Lemke-Howson Algorithm

For a 2-player bimatrix game $(A, B)$ with $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{m \times n}$, define the **best response polytopes**:

$$P = \{x \in \mathbb{R}^m_+ : B^\top x \leq \mathbf{1}\}, \quad Q = \{y \in \mathbb{R}^n_+ : A y \leq \mathbf{1}\}$$

Each vertex of $P \times Q$ carries **labels** from $\{1, \ldots, m+n\}$:
- Label $i \in \{1,\ldots,m\}$: strategy $i$ of player 1 has zero probability ($x_i = 0$)
- Label $m+j \in \{m+1,\ldots,m+n\}$: strategy $j$ of player 2 is a best response ($(B^\top x)_j = 1$)

A pair $(x, y)$ is a **Nash equilibrium** iff it is **completely labeled** — every label $1, \ldots, m+n$ appears at least once.

**Pivot procedure**: Start from the artificial equilibrium $(0, 0)$. Drop one label $k$, then pivot along edges of the polytope, alternating between $P$ and $Q$, until label $k$ reappears. The endpoint is a NE.

### 106.2 PPAD Complexity

The class **PPAD** (Polynomial Parity Arguments on Directed graphs) captures problems where:
- A directed graph on exponentially many nodes is implicitly defined
- Each node has in-degree and out-degree ≤ 1
- Given a source, find another endpoint

**Theorem (Daskalakis, Goldberg, Papadimitriou 2009)**: Computing a Nash equilibrium is PPAD-complete, even for 2-player games.

This means: no polynomial-time algorithm is known (or likely exists) for finding exact NE in general bimatrix games. The Lemke-Howson algorithm runs in worst-case exponential time.

### 106.3 Approximate Nash Equilibria

An **$\varepsilon$-Nash equilibrium** is a strategy profile where no player can improve by more than $\varepsilon$:

$$u_i(s_i^*, s_{-i}^*) \geq u_i(s_i, s_{-i}^*) - \varepsilon \quad \forall s_i, \forall i$$

**Support enumeration** checks all possible support pairs $(S_1, S_2)$ — there are $O(2^{m+n})$ pairs but works well for small games. For each support pair, solving for NE reduces to a linear system.

The **Tsaknakis-Spirakis** algorithm finds a $0.3393$-NE in polynomial time. Finding $\varepsilon$-NE for $\varepsilon < $ some constant remains PPAD-hard.

---

## Implementation (60 min)

```python
"""Computing Nash Equilibria: Lemke-Howson and Support Enumeration."""
import numpy as np
from itertools import combinations
from typing import Optional

def support_enumeration(A: np.ndarray, B: np.ndarray) -> list:
    """Find all Nash equilibria via support enumeration (exact, exponential)."""
    m, n = A.shape
    equilibria = []
    
    for k1 in range(1, m + 1):
        for k2 in range(1, n + 1):
            for S1 in combinations(range(m), k1):
                for S2 in combinations(range(n), k2):
                    result = _solve_support(A, B, list(S1), list(S2))
                    if result is not None:
                        equilibria.append(result)
    return equilibria

def _solve_support(A, B, S1, S2) -> Optional[tuple]:
    """Solve for NE given supports S1, S2."""
    m, n = A.shape
    k1, k2 = len(S1), len(S2)
    
    # Player 2's mix must make player 1 indifferent on S1
    # A[S1, :][:, S2] @ y[S2] = v1 * ones, sum(y[S2]) = 1
    A_sub = A[np.ix_(S1, S2)]
    M1 = np.column_stack([A_sub, -np.ones(k1)])
    b1 = np.zeros(k1)
    M1_full = np.vstack([M1, np.append(np.ones(k2), 0).reshape(1, -1)])
    b1_full = np.append(b1, 1.0)
    
    try:
        sol1 = np.linalg.lstsq(M1_full, b1_full, rcond=None)[0]
        y_S2 = sol1[:k2]
        if np.any(y_S2 < -1e-10):
            return None
    except np.linalg.LinAlgError:
        return None
    
    # Player 1's mix must make player 2 indifferent on S2
    B_sub = B[np.ix_(S1, S2)]
    M2 = np.column_stack([B_sub.T, -np.ones(k2)])
    b2 = np.zeros(k2)
    M2_full = np.vstack([M2, np.append(np.ones(k1), 0).reshape(1, -1)])
    b2_full = np.append(b2, 1.0)
    
    try:
        sol2 = np.linalg.lstsq(M2_full, b2_full, rcond=None)[0]
        x_S1 = sol2[:k1]
        if np.any(x_S1 < -1e-10):
            return None
    except np.linalg.LinAlgError:
        return None
    
    # Verify best response conditions
    x = np.zeros(m); x[S1] = np.maximum(x_S1, 0)
    y = np.zeros(n); y[S2] = np.maximum(y_S2, 0)
    x /= x.sum(); y /= y.sum()
    
    payoff1 = A @ y
    if np.max(payoff1) - np.min(payoff1[list(S1)]) > 1e-8:
        return None
    
    return (x, y)

# Demo: Matching Pennies
A = np.array([[1, -1], [-1, 1]])
B = -A
eqs = support_enumeration(A, B)
print(f"Matching Pennies: found {len(eqs)} NE")
for x, y in eqs:
    print(f"  x = {x}, y = {y}")
    print(f"  Payoffs: ({x @ A @ y:.4f}, {x @ B @ y:.4f})")

# Battle of the Sexes
A_bos = np.array([[3, 0], [0, 2]])
B_bos = np.array([[2, 0], [0, 3]])
eqs_bos = support_enumeration(A_bos, B_bos)
print(f"\nBattle of Sexes: found {len(eqs_bos)} NE")
for x, y in eqs_bos:
    print(f"  x = {np.round(x, 4)}, y = {np.round(y, 4)}")

# Timing comparison
import time
sizes = [3, 4, 5, 6]
for sz in sizes:
    A_r = np.random.randn(sz, sz)
    B_r = np.random.randn(sz, sz)
    t0 = time.time()
    eqs_r = support_enumeration(A_r, B_r)
    dt = time.time() - t0
    print(f"  {sz}x{sz}: {len(eqs_r)} NE, {dt:.4f}s")
```

---

## Practice Problems (45 min)

**P1**: For the game $A = \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix}$, $B = \begin{pmatrix} 1 & 0 \\ 0 & 2 \end{pmatrix}$, run support enumeration by hand. List all NE.

<details><summary>Answer</summary>

Pure NE: $(1, 1)$ with payoff $(2, 1)$ and $(2, 2)$ with payoff $(1, 2)$. Mixed NE: $x = (2/3, 1/3)$, $y = (2/3, 1/3)$ with expected payoffs $(2/3, 2/3)$. Found by solving indifference: $2y_1 = y_2 = 1 - y_1 \Rightarrow y_1 = 2/3$.
</details>

**P2**: Explain why Lemke-Howson always terminates. What graph structure guarantees this?

<details><summary>Answer</summary>

The Lemke-Howson path traces edges of a graph where each node has degree ≤ 2 (each vertex of the product polytope has exactly one "missing" label or is completely labeled). This forms disjoint paths and cycles. Starting from the artificial equilibrium (degree 1 node), we must reach another degree 1 node — a genuine NE. Finiteness follows because the polytope has finitely many vertices.
</details>

**P3**: Why can't we use binary search to find NE? How does this relate to PPAD?

<details><summary>Answer</summary>

NE is not an optimization problem — there's no single objective to minimize. PPAD problems have guaranteed existence (by Brouwer/Nash) but no efficiently verifiable direction of improvement. Binary search requires a monotone structure (better/worse), which NE lacks. The PPAD class captures this: solutions exist by parity arguments on directed graphs, but finding them requires following potentially exponential paths.
</details>

---

## Expert Challenges

**E1**: Implement Lemke-Howson pivoting for a $3 \times 3$ game. Track the pivot path and verify it terminates at a NE.

<details><summary>Hint</summary>

Set up the labeled polytopes explicitly. Maintain a basis for each polytope. When dropping label $k$, pivot in the polytope that "owns" that label, then pivot in the other polytope on the newly duplicate label. Continue until $k$ reappears.
</details>

**E2**: Prove that finding an $\varepsilon$-NE with $\varepsilon = 0$ is PPAD-complete but finding one with $\varepsilon = 1/\text{poly}(n)$ is also hard. What's the best known polynomial-time approximation?

<details><summary>Answer</summary>

Rubinstein (2016) showed that for any constant $\varepsilon < 1/3$, finding $\varepsilon$-NE is PPAD-hard under ETH. The Tsaknakis-Spirakis algorithm achieves $0.3393$-NE in polynomial time. The gap between $0.3393$ and the hardness threshold remains open.
</details>

**E3**: For zero-sum games, NE can be found in polynomial time via LP. Write down the LP and explain why PPAD-hardness doesn't apply.

<details><summary>Answer</summary>

For zero-sum $(A, -A)$: $\max_{x} \min_{y} x^\top A y = \max_{x,v} \{v : A^\top x \geq v \cdot \mathbf{1}, x \geq 0, \sum x_i = 1\}$. This is an LP, solvable in polynomial time. PPAD-hardness applies to general-sum games; zero-sum games have the minimax structure that reduces to LP, bypassing the PPAD barrier entirely.
</details>

---

## Connections
- **Back**: Builds on [Day 97: Zero-Sum LP](../week-14/day-97-zero-sum-lp.md) — minimax and LP for zero-sum
- **Forward**: [Day 107: Fictitious Play](day-107-fictitious-play.md) uses iterative methods that bypass NE computation
- **OKS**: Real-time robot negotiation must use structured games (zero-sum, potential) where polynomial algorithms exist

## Self-Check
- [ ] I can explain why NE computation is PPAD-complete and what that implies
- [ ] I can run support enumeration on a small bimatrix game
- [ ] I understand the Lemke-Howson pivot procedure and its labeled polytope formulation
- [ ] I know when polynomial-time NE algorithms exist (zero-sum, potential games)
