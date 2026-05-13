# Game Theory for Optimization

> **A comprehensive reference — strategic interaction, equilibria, mechanism design,
> and their connections to optimization, robotics, and machine learning.**
>
> Game theory studies decision-making when multiple agents interact, each with their own
> objectives. This guide covers both non-cooperative and cooperative theory, algorithmic
> aspects, and applications in multi-robot systems and adversarial ML. Use it alongside
> the daily lessons or as a standalone study guide.

## Prerequisites
- [Day 4: Linear Programming](weeks/week-01/day-04-linear-programming.md)
- [Day 5: LP Duality](weeks/week-01/day-05-lp-duality.md)
- [Chapter 05: Convex Optimization](05-convex-optimization.md)
- [Chapter 08: Matrix Essentials](08-matrix-essentials.md)
- [Chapter 09: Lie Groups](09-lie-groups.md) (for robotics examples)

---

## 1. What Is a Game? (Optimization with Multiple Agents)

Classical optimization asks: given an objective $f(x)$, find $x^*$ that minimizes $f$.
Game theory asks: given $n$ agents, each choosing a strategy to optimize *their own*
objective, what happens when all agents act simultaneously?

### 1.1 Players, Strategies, Payoffs

A **normal-form game** $G = (N, \{S_i\}, \{u_i\})$ consists of:

| Component | Symbol | Definition |
|-----------|--------|------------|
| **Players** | $N = \{1, 2, \ldots, n\}$ | The set of decision-making agents |
| **Strategy set** | $S_i$ | Available actions for player $i$ |
| **Strategy profile** | $s = (s_1, s_2, \ldots, s_n) \in S$ | One strategy chosen per player |
| **Payoff function** | $u_i : S \to \mathbb{R}$ | Utility player $i$ receives given profile $s$ |

We write $s_{-i} = (s_1, \ldots, s_{i-1}, s_{i+1}, \ldots, s_n)$ for the strategies of all
players *except* player $i$, so $s = (s_i, s_{-i})$.

### 1.2 Normal-Form Representation

For two-player finite games, we display payoffs as a **bimatrix**. Row player is Player 1,
column player is Player 2. Each cell contains $(u_1, u_2)$.

**Example — Prisoner's Dilemma:**

|  | Cooperate | Defect |
|--|-----------|--------|
| **Cooperate** | $(-1, -1)$ | $(-3, 0)$ |
| **Defect** | $(0, -3)$ | $(-2, -2)$ |

Here $S_1 = S_2 = \{\text{Cooperate}, \text{Defect}\}$ and the unique Nash equilibrium is
(Defect, Defect) with payoffs $(-2, -2)$ — even though (Cooperate, Cooperate) at $(-1, -1)$
is Pareto-superior.

### 1.3 Connection to Multi-Objective Optimization

In standard optimization, we have one objective. In a game, each player has a separate
objective. The key differences:

| Property | Single-Agent Optimization | Game Theory |
|----------|--------------------------|-------------|
| Objective | Minimize $f(x)$ | Each $i$ minimizes $f_i(x_i; x_{-i})$ |
| Solution concept | Global/local minimum | Equilibrium (Nash, correlated, etc.) |
| Efficiency | Optimal by definition | May be suboptimal (prisoner's dilemma) |
| Constraints | Feasibility | Incentive compatibility |

### 1.4 Why Nash Equilibrium ≠ Pareto Optimum

A strategy profile $s^*$ is a **Nash equilibrium (NE)** if no player can improve by
unilateral deviation:

$$u_i(s_i^*, s_{-i}^*) \geq u_i(s_i, s_{-i}^*) \quad \forall s_i \in S_i, \; \forall i \in N$$

A profile $s$ is **Pareto optimal** if there is no $s'$ such that $u_i(s') \geq u_i(s)$
for all $i$ with strict inequality for at least one.

**Key insight:** NE is a *stability* concept (no one wants to deviate); Pareto optimality is
an *efficiency* concept (no waste). The prisoner's dilemma shows they can diverge — the NE
wastes utility that cooperation would capture.

> **OKS Relevance:** When multiple AMRs compete for shared resources (corridors, charging
> stations, elevators), each robot locally optimizes its own path. The resulting "traffic
> equilibrium" may be inefficient — two robots may deadlock in a corridor when a slight
> detour by one would benefit both. This is precisely the Nash-vs-Pareto gap.

---

## 2. Dominance and Solution Concepts

### 2.1 Strictly and Weakly Dominated Strategies

Strategy $s_i$ **strictly dominates** $s_i'$ if:

$$u_i(s_i, s_{-i}) > u_i(s_i', s_{-i}) \quad \forall s_{-i} \in S_{-i}$$

Strategy $s_i$ **weakly dominates** $s_i'$ if the inequality holds with $\geq$ everywhere
and $>$ for at least one $s_{-i}$.

**A rational player never plays a strictly dominated strategy.**

### 2.2 Iterated Elimination of Strictly Dominated Strategies (IESDS)

Algorithm:
1. Find and remove any strictly dominated strategy for any player
2. In the reduced game, repeat step 1
3. Continue until no more eliminations are possible

IESDS is **order-independent** for strict dominance: the surviving set is the same regardless
of which dominated strategy you eliminate first.

**Example:**

|  | L | C | R |
|--|---|---|---|
| **T** | $(3,1)$ | $(0,1)$ | $(0,0)$ |
| **M** | $(1,1)$ | $(1,1)$ | $(5,0)$ |
| **B** | $(0,1)$ | $(4,1)$ | $(0,0)$ |

- For Player 2: R is dominated by L (payoffs: 0,0,0 vs 1,1,1). Eliminate R.
- Reduced game: T dominates M (payoffs: (3,0) vs (1,1)). Eliminate M.
- Remaining: T dominates B (3 > 0 and 0 < 4 — wait, T does not dominate B in column C).
- We stop. IESDS alone does not always solve the game.

### 2.3 Best Response Functions

Player $i$'s **best response** to $s_{-i}$ is:

$$BR_i(s_{-i}) = \arg\max_{s_i \in S_i} u_i(s_i, s_{-i})$$

This is a set-valued function (there may be ties). A Nash equilibrium is a **mutual best
response**: $s_i^* \in BR_i(s_{-i}^*)$ for all $i$.

### 2.4 Pure-Strategy Nash Equilibrium

A **pure-strategy NE** is a strategy profile where each player plays a single deterministic
action and no one wants to deviate.

**Finding pure NE in bimatrix games:**
1. For each column, mark the row(s) with the highest payoff for Player 1 (best response)
2. For each row, mark the column(s) with the highest payoff for Player 2
3. Any cell marked by both players is a pure NE

**Example — Battle of the Sexes:**

|  | Opera | Football |
|--|-------|----------|
| **Opera** | $(3, 2)$ | $(0, 0)$ |
| **Football** | $(0, 0)$ | $(2, 3)$ |

- Player 1 best responses: Opera→Opera (3>0), Football→Football (2>0)
- Player 2 best responses: Opera→Opera (2>0), Football→Football (3>0)
- Pure NE: (Opera, Opera) and (Football, Football)

### 2.5 Mixed-Strategy Nash Equilibrium

A **mixed strategy** for player $i$ is a probability distribution over pure strategies:

$$\sigma_i \in \Delta(S_i) = \left\{ p \in \mathbb{R}^{|S_i|}_{\geq 0} : \sum_{s_i \in S_i} p(s_i) = 1 \right\}$$

The **expected payoff** under mixed profile $\sigma = (\sigma_1, \ldots, \sigma_n)$ is:

$$u_i(\sigma) = \sum_{s \in S} \left(\prod_{j \in N} \sigma_j(s_j)\right) u_i(s)$$

**Nash's Existence Theorem (1950):** Every finite game (finite players, finite strategies)
has at least one mixed-strategy Nash equilibrium.

**Key property:** At a mixed NE, each player must be indifferent among all strategies played
with positive probability. If strategy $s_i$ is in the support of $\sigma_i^*$, then:

$$u_i(s_i, \sigma_{-i}^*) = \max_{s_i' \in S_i} u_i(s_i', \sigma_{-i}^*)$$

**Computing mixed NE (2×2 games):**

For a $2 \times 2$ game with payoff matrices $A$ (Player 1) and $B$ (Player 2):

|  | L | R |
|--|---|---|
| **T** | $(a_{11}, b_{11})$ | $(a_{12}, b_{12})$ |
| **B** | $(a_{21}, b_{21})$ | $(a_{22}, b_{22})$ |

Let Player 1 play T with probability $p$ and Player 2 play L with probability $q$.

Player 2 is indifferent when:
$$p \cdot b_{11} + (1-p) \cdot b_{21} = p \cdot b_{12} + (1-p) \cdot b_{22}$$
$$p = \frac{b_{22} - b_{21}}{(b_{11} - b_{12}) - (b_{21} - b_{22})}$$

Player 1 is indifferent when:
$$q \cdot a_{11} + (1-q) \cdot a_{12} = q \cdot a_{21} + (1-q) \cdot a_{22}$$
$$q = \frac{a_{22} - a_{12}}{(a_{11} - a_{21}) - (a_{12} - a_{22})}$$

```python
import numpy as np

def mixed_ne_2x2(A, B):
    """Compute mixed NE for a 2x2 bimatrix game.
    A: 2x2 payoff matrix for Player 1 (row player)
    B: 2x2 payoff matrix for Player 2 (column player)
    Returns: (p, q) — probability of T for P1, probability of L for P2
    """
    # Player 1's mixing probability (makes Player 2 indifferent)
    denom_p = (B[0,0] - B[0,1]) - (B[1,0] - B[1,1])
    if abs(denom_p) < 1e-12:
        return None  # No fully mixed NE
    p = (B[1,1] - B[1,0]) / denom_p

    # Player 2's mixing probability (makes Player 1 indifferent)
    denom_q = (A[0,0] - A[1,0]) - (A[0,1] - A[1,1])
    if abs(denom_q) < 1e-12:
        return None
    q = (A[1,1] - A[0,1]) / denom_q

    if 0 <= p <= 1 and 0 <= q <= 1:
        return (p, q)
    return None

# Battle of the Sexes
A = np.array([[3, 0], [0, 2]])
B = np.array([[2, 0], [0, 3]])
result = mixed_ne_2x2(A, B)
# result = (0.6, 0.4) — P1 plays Opera 60%, P2 plays Opera 40%
```

---

## 3. Zero-Sum Games and the Minimax Theorem

### 3.1 Two-Player Zero-Sum Games

A game is **zero-sum** if $u_1(s) + u_2(s) = 0$ for all $s$. We write the payoff matrix $A$
for Player 1; Player 2's payoff is $-A$.

Zero-sum games model pure competition: one player's gain is exactly the other's loss.

### 3.2 Minimax Strategies and Saddle Points

Player 1 (row) wants to maximize; Player 2 (column) wants to minimize.

- Player 1's **maximin**: $\max_i \min_j a_{ij}$ — "guarantee the best worst case"
- Player 2's **minimax**: $\min_j \max_i a_{ij}$ — "limit the opponent's best outcome"

**Saddle point:** A cell $(i^*, j^*)$ where:
$$a_{i^*j^*} = \max_i \min_j a_{ij} = \min_j \max_i a_{ij}$$

At a saddle point, neither player can improve by unilateral deviation — it is a pure NE.

**Example — Matching Pennies** (no saddle point):

$$A = \begin{pmatrix} 1 & -1 \\ -1 & 1 \end{pmatrix}$$

- Maximin: $\max(\min(1,-1), \min(-1,1)) = \max(-1,-1) = -1$
- Minimax: $\min(\max(1,-1), \max(-1,1)) = \min(1,1) = 1$
- Maximin $\neq$ minimax → no pure saddle point, game requires mixed strategies.

### 3.3 Von Neumann's Minimax Theorem (1928)

**Theorem:** For every two-player zero-sum game with finite strategies:

$$\max_{\sigma_1 \in \Delta(S_1)} \min_{\sigma_2 \in \Delta(S_2)} \sigma_1^T A \sigma_2
= \min_{\sigma_2 \in \Delta(S_2)} \max_{\sigma_1 \in \Delta(S_1)} \sigma_1^T A \sigma_2
= v$$

The common value $v$ is the **value of the game**. This is the foundational result that
makes zero-sum games "solvable" — the maximin and minimax strategies coincide in mixed
strategies.

For Matching Pennies: $v = 0$, and the equilibrium is each player choosing Head/Tail with
probability $\frac{1}{2}$.

### 3.4 LP Formulation of Zero-Sum Games

**Player 1's problem** (maximize the guarantee):

$$\max_{v, \sigma_1} \; v$$
$$\text{s.t. } \sum_{i} \sigma_1(i) \cdot a_{ij} \geq v \quad \forall j$$
$$\sum_i \sigma_1(i) = 1, \quad \sigma_1 \geq 0$$

**Player 2's problem** (minimize the opponent's guarantee):

$$\min_{w, \sigma_2} \; w$$
$$\text{s.t. } \sum_{j} a_{ij} \cdot \sigma_2(j) \leq w \quad \forall i$$
$$\sum_j \sigma_2(j) = 1, \quad \sigma_2 \geq 0$$

```python
from scipy.optimize import linprog

def solve_zero_sum_lp(A):
    """Solve a zero-sum game via LP (Player 1's perspective).
    A: m x n payoff matrix for row player.
    Returns: (value, mixed_strategy_p1)
    """
    m, n = A.shape
    # Variables: [sigma_1(0), ..., sigma_1(m-1), v]
    # Maximize v  =>  minimize -v
    c = np.zeros(m + 1)
    c[-1] = -1  # minimize -v

    # Constraints: sum_i sigma_1(i) * a_{ij} >= v  for each j
    # Rewrite: -sum_i sigma_1(i) * a_{ij} + v <= 0
    A_ub = np.zeros((n, m + 1))
    for j in range(n):
        A_ub[j, :m] = -A[:, j]
        A_ub[j, -1] = 1
    b_ub = np.zeros(n)

    # Equality: sum sigma_1 = 1
    A_eq = np.zeros((1, m + 1))
    A_eq[0, :m] = 1
    b_eq = np.array([1.0])

    bounds = [(0, None)] * m + [(None, None)]
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    value = -result.fun
    strategy = result.x[:m]
    return value, strategy

# Rock-Paper-Scissors
A = np.array([[0, -1, 1],
              [1,  0, -1],
              [-1, 1,  0]])
value, sigma = solve_zero_sum_lp(A)
# value ≈ 0, sigma ≈ [1/3, 1/3, 1/3]
```

### 3.5 Connection to LP Duality

Player 1's LP and Player 2's LP are **dual** to each other. By strong LP duality:

- Primal optimal = Dual optimal → the minimax theorem follows
- Complementary slackness → if $\sigma_1^*(i) > 0$ then constraint $j$ is tight for Player 2

This bridges directly to [Chapter 04–05](04-constrained-optimization.md): LP duality is the
engine behind the minimax theorem. Von Neumann proved minimax in 1928; Dantzig invented
the simplex method in 1947 partly inspired by this connection.

> **OKS Relevance:** The minimax framework applies directly to worst-case path planning.
> When an AMR must traverse an area with uncertain obstacles (an adversarial environment),
> finding the minimax path means finding the route that minimizes the maximum possible cost
> — a robust planning strategy against worst-case scenarios.

---

## 4. Cooperative Games

In cooperative games, players can form **coalitions** and make binding agreements.
The question shifts from "what will each player do?" to "how should the coalition
divide the payoff?"

### 4.1 Coalitional Games and the Characteristic Function

A **coalitional game** $(N, v)$ consists of:
- Player set $N = \{1, \ldots, n\}$
- **Characteristic function** $v : 2^N \to \mathbb{R}$ with $v(\emptyset) = 0$

$v(S)$ is the total payoff coalition $S \subseteq N$ can guarantee by cooperating, regardless
of what $N \setminus S$ does.

**Superadditivity:** $v(S \cup T) \geq v(S) + v(T)$ for disjoint $S, T$ — "cooperating
never hurts." Most interesting games are superadditive.

### 4.2 The Core — Stability Concept

An **imputation** is a payoff vector $x = (x_1, \ldots, x_n)$ with:
- **Efficiency:** $\sum_{i \in N} x_i = v(N)$
- **Individual rationality:** $x_i \geq v(\{i\})$ for all $i$

The **core** is the set of imputations where no coalition can do better by breaking away:

$$\text{Core}(v) = \left\{ x : \sum_{i \in N} x_i = v(N), \; \sum_{i \in S} x_i \geq v(S) \; \forall S \subseteq N \right\}$$

The core may be empty (some games have no stable allocation). Checking non-emptiness
is a linear feasibility problem with $2^n$ constraints (though it can often be reduced).

### 4.3 Shapley Value — Axioms and Formula

The **Shapley value** $\phi_i(v)$ is the unique payoff allocation satisfying:

| Axiom | Statement |
|-------|-----------|
| **Efficiency** | $\sum_i \phi_i(v) = v(N)$ |
| **Symmetry** | If $i$ and $j$ contribute equally to every coalition, $\phi_i = \phi_j$ |
| **Null player** | If $i$ adds nothing to any coalition, $\phi_i = 0$ |
| **Additivity** | $\phi_i(v + w) = \phi_i(v) + \phi_i(w)$ |

**Formula:**

$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n - |S| - 1)!}{n!} \left[ v(S \cup \{i\}) - v(S) \right]$$

Intuition: average the marginal contribution of player $i$ over all possible orderings in
which players join the grand coalition.

```python
from itertools import combinations
import math

def shapley_value(n, v):
    """Compute Shapley values for an n-player coalitional game.
    n: number of players (0-indexed)
    v: function mapping frozenset -> float (characteristic function)
    Returns: list of Shapley values [phi_0, ..., phi_{n-1}]
    """
    players = list(range(n))
    phi = [0.0] * n
    for i in players:
        others = [j for j in players if j != i]
        for size in range(len(others) + 1):
            for S in combinations(others, size):
                S_set = frozenset(S)
                weight = math.factorial(len(S)) * math.factorial(n - len(S) - 1) / math.factorial(n)
                marginal = v(S_set | {i}) - v(S_set)
                phi[i] += weight * marginal
    return phi

# Example: 3-player majority game — coalition wins if |S| >= 2
def majority_game(S):
    return 1.0 if len(S) >= 2 else 0.0

phi = shapley_value(3, majority_game)
# phi = [1/3, 1/3, 1/3] — symmetric game, equal Shapley values
```

### 4.4 Bargaining: Nash, Kalai-Smorodinsky

**Nash Bargaining Solution (NBS):** Given feasible set $F \subset \mathbb{R}^2$ and
disagreement point $d$, the NBS maximizes:

$$\max_{x \in F} (x_1 - d_1)(x_2 - d_2) \quad \text{s.t. } x \geq d$$

This is the product of surplus gains — it satisfies axioms of symmetry, Pareto optimality,
invariance to affine transformations, and independence of irrelevant alternatives.

**Kalai-Smorodinsky Solution:** Replaces IIA with monotonicity. It picks the point on the
Pareto frontier where each player gets the same fraction of their maximum achievable payoff.

### 4.5 Connection to SHAP in Machine Learning

SHAP (SHapley Additive exPlanations) uses the Shapley value to explain ML predictions:
- Players = features
- $v(S)$ = model prediction using only features in $S$
- $\phi_i$ = contribution of feature $i$ to the prediction

This is a direct application of cooperative game theory to ML interpretability, connecting
Shapley's 1953 result to modern explainable AI.

> **OKS Relevance:** Cooperative game theory applies to multi-robot coalition formation.
> When a fleet of AMRs must decide which robots serve which zones, the Shapley value
> gives a fair way to allocate "credit" for completed tasks — useful for load balancing
> and performance analysis.

---

## 5. Dynamic Games

### 5.1 Extensive Form and Game Trees

Dynamic games unfold over time. The **extensive form** represents a game as a tree:
- **Nodes:** decision points for players
- **Edges:** actions available at each node
- **Terminal nodes:** outcomes with payoff vectors
- **Root:** the starting point

```
            Player 1
           /        \
         L            R
        /              \
    Player 2         Player 2
    /    \            /    \
   l      r         l      r
 (2,1)  (0,0)    (0,2)  (1,1)
```

### 5.2 Information Sets

An **information set** groups nodes where a player cannot distinguish between them (same
available actions, no knowledge of which node they are at).

- **Perfect information:** every information set is a singleton (chess, go)
- **Imperfect information:** some information sets contain multiple nodes (poker)

### 5.3 Backward Induction

For games of **perfect information** with finite horizon, solve from the terminal nodes
backward:

1. At each terminal node, payoffs are known
2. At each penultimate decision node, the player picks the action maximizing their payoff
3. Continue backward to the root

**Example (from the tree above):**
- Player 2 at left node: picks $l$ (payoff 1 > 0)
- Player 2 at right node: picks $l$ (payoff 2 > 1)
- Player 1 at root: L gives payoff 2 (from P2 picking $l$), R gives payoff 0 (from P2
  picking $l$). Player 1 picks L.
- Outcome: (L, l, l) with payoffs $(2, 1)$.

### 5.4 Subgame Perfect Equilibrium (SPE)

A **subgame** is a portion of the game tree that forms a complete game by itself (starts at
a singleton information set and includes all successors).

**Definition:** A strategy profile is a **subgame perfect equilibrium** if it constitutes a
Nash equilibrium in every subgame.

SPE eliminates **non-credible threats** — strategies that are NE of the full game but
involve irrational play in some subgame. Backward induction yields an SPE in finite perfect-
information games.

### 5.5 Repeated Games and the Folk Theorem

When a **stage game** is played repeatedly:
- **Finitely repeated:** backward induction often forces defection in every round
  (e.g., finitely repeated prisoner's dilemma has unique SPE: always defect)
- **Infinitely repeated** (or unknown horizon with discount $\delta \in (0,1)$):
  cooperation can be sustained

**Folk Theorem (informal):** In an infinitely repeated game with sufficiently patient
players ($\delta$ close to 1), *any* feasible and individually rational payoff vector can be
sustained as a Nash equilibrium of the repeated game.

**Trigger strategies** (e.g., Grim Trigger: cooperate until the opponent defects, then
defect forever) enforce cooperation through the threat of permanent punishment.

The discount factor threshold for cooperation in the prisoner's dilemma:
$$\delta \geq \frac{u(\text{temptation}) - u(\text{cooperate})}{u(\text{temptation}) - u(\text{punishment})}$$

### 5.6 Stackelberg Games (Leader-Follower)

In a **Stackelberg game**, one player (the leader) moves first, and the other (the follower)
observes and best-responds.

**Structure:**
1. Leader chooses $x \in X$
2. Follower observes $x$ and chooses $y^*(x) = \arg\max_{y \in Y} u_F(x, y)$
3. Leader anticipates this: $\max_{x \in X} u_L(x, y^*(x))$

This is a **bilevel optimization** problem:

$$\max_x \; u_L(x, y^*(x)) \quad \text{s.t. } y^*(x) \in \arg\max_y u_F(x, y)$$

The leader's payoff in a Stackelberg equilibrium is at least as good as in the corresponding
Nash equilibrium (first-mover advantage).

> **OKS Relevance:** Multi-robot coordination often has implicit leader-follower dynamics.
> The fleet management system (RCS) acts as a "leader" assigning tasks; individual robots
> are "followers" that optimize their local execution. Stackelberg models also apply to
> elevator allocation: the BEC controller (leader) schedules battery swaps, and robots
> (followers) plan around the schedule.

---

## 6. Mechanism Design

Mechanism design is **"inverse game theory"** — instead of analyzing a given game, we
*design* the rules of the game to achieve a desired outcome.

### 6.1 Revelation Principle

**Setting:** Players have private types $\theta_i$ (e.g., valuations). A **mechanism**
$(M, g)$ specifies message spaces $M_i$ and an outcome function $g : M \to O$.

**Revelation Principle:** For any mechanism that implements a social choice function in
equilibrium, there exists a **direct revelation mechanism** where:
1. Each player reports their type $\hat{\theta}_i$
2. Truth-telling ($\hat{\theta}_i = \theta_i$) is an equilibrium

This simplifies mechanism design: we only need to consider truthful mechanisms.

### 6.2 Incentive Compatibility

A mechanism is **incentive compatible (IC)** if reporting truthfully is optimal:

$$u_i(\theta_i, g(\theta_i, \theta_{-i})) \geq u_i(\theta_i, g(\hat{\theta}_i, \theta_{-i})) \quad \forall \hat{\theta}_i, \forall \theta_{-i}$$

- **Dominant-strategy IC (DSIC):** truth-telling is optimal regardless of others' reports
- **Bayesian IC (BIC):** truth-telling is optimal in expectation over others' types

### 6.3 VCG Mechanism (Vickrey-Clarke-Groves)

The VCG mechanism achieves efficient allocation with DSIC:

1. **Allocation rule:** Choose outcome $o^*$ that maximizes total reported value:
   $o^* = \arg\max_o \sum_i v_i(o, \hat{\theta}_i)$

2. **Payment rule:** Player $i$ pays:
   $$p_i = \max_o \sum_{j \neq i} v_j(o, \hat{\theta}_j) - \sum_{j \neq i} v_j(o^*, \hat{\theta}_j)$$

   Intuition: $i$ pays the "externality" they impose — how much worse off the others are
   due to $i$'s presence.

**Properties:** Allocatively efficient, DSIC, individually rational (under Clarke pivot rule).

### 6.4 Auction Types

| Auction | Rules | Equilibrium Strategy (private values) | Revenue |
|---------|-------|--------------------------------------|---------|
| **First-price sealed-bid** | Highest bid wins, pays own bid | Shade bid: $b_i = \frac{n-1}{n}\theta_i$ | $\frac{n-1}{n+1}\bar{\theta}$ |
| **Second-price (Vickrey)** | Highest bid wins, pays second-highest | Bid true value: $b_i = \theta_i$ | $\frac{n-1}{n+1}\bar{\theta}$ |
| **English (ascending)** | Open ascending, last bidder wins | Stay until price = value | $\frac{n-1}{n+1}\bar{\theta}$ |
| **Dutch (descending)** | Open descending, first bidder wins | Same as first-price sealed | $\frac{n-1}{n+1}\bar{\theta}$ |

The Vickrey auction is a special case of VCG for single-item allocation.

### 6.5 Revenue Equivalence Theorem

**Theorem (Myerson, 1981):** Under the independent private values model with risk-neutral
bidders, symmetric equilibria, and the same allocation rule, all four standard auction
formats yield the same expected revenue.

Conditions for revenue equivalence:
1. Values are independently drawn from the same distribution
2. Bidders are risk-neutral
3. Bidder with the lowest type gets zero surplus
4. The item goes to the bidder with the highest value

> **OKS Relevance:** Multi-robot task allocation can be modeled as an auction. Each robot
> "bids" its estimated cost to complete a task, and the auctioneer (RCS) assigns tasks to
> minimize total cost. Second-price auctions ensure robots truthfully report their costs.
> This is used in market-based multi-robot coordination systems.

---

## 7. Algorithmic Game Theory

### 7.1 Computing Nash Equilibria — Lemke-Howson

For two-player bimatrix games $(A, B)$ with $A \in \mathbb{R}^{m \times n}$, finding a Nash
equilibrium is equivalent to a **linear complementarity problem (LCP)**.

The **Lemke-Howson algorithm** (1964):
1. Set up the complementarity conditions from the best-response polytopes
2. Start at a vertex of the polytope (a pure strategy pair)
3. Pivot along edges following complementary pairs
4. Terminate at a vertex satisfying all conditions — this is a NE

**Complexity:** Lemke-Howson runs in exponential time in the worst case, but works well
in practice for small games.

### 7.2 PPAD Complexity Class

**PPAD** (Polynomial Parity Arguments on Directed graphs) is a complexity class containing
problems whose solutions are guaranteed to exist by a parity argument.

**Daskalakis, Goldberg, Papadimitriou (2009):** Computing a Nash equilibrium of a
two-player game is **PPAD-complete**.

Implications:
- Unless PPAD = P (considered unlikely), there is no polynomial-time algorithm
- Nash equilibrium exists (guaranteed by Nash's theorem) but may be hard to *find*
- This is unlike LP (polynomial) or NP-complete problems (existence not guaranteed)

### 7.3 Potential Games

A game is an **exact potential game** if there exists $\Phi : S \to \mathbb{R}$ such that:

$$u_i(s_i, s_{-i}) - u_i(s_i', s_{-i}) = \Phi(s_i, s_{-i}) - \Phi(s_i', s_{-i}) \quad \forall i, s_i, s_i', s_{-i}$$

**Properties:**
- Every local maximum of $\Phi$ is a pure NE
- Best-response dynamics converge to a NE in finite steps
- Examples: congestion games, coordination games

Potential games turn multi-agent equilibrium problems into single-agent optimization of $\Phi$.

### 7.4 Congestion Games and Wardrop Equilibrium

A **congestion game** has:
- Shared resources $E$
- Each player chooses a subset of resources
- Cost of resource $e$ depends on the number of users: $c_e(x_e)$

**Rosenthal's theorem:** Every congestion game is a potential game with:
$$\Phi(s) = \sum_{e \in E} \sum_{k=1}^{x_e(s)} c_e(k)$$

**Wardrop equilibrium** (continuous limit): In a traffic network, flows distribute so that
all used paths between any origin-destination pair have equal cost, and no unused path has
lower cost. This is the solution to a convex optimization problem:

$$\min_f \sum_e \int_0^{f_e} c_e(t) \, dt \quad \text{s.t. flow conservation constraints}$$

### 7.5 Price of Anarchy and Price of Stability

For a game with social welfare $W(s) = \sum_i u_i(s)$:

$$\text{Price of Anarchy (PoA)} = \frac{W(s^{\text{OPT}})}{\min_{s \in NE} W(s)}$$

$$\text{Price of Stability (PoS)} = \frac{W(s^{\text{OPT}})}{\max_{s \in NE} W(s)}$$

| Metric | Measures | Value Range | Interpretation |
|--------|----------|-------------|----------------|
| PoA | Worst-case NE efficiency | $\geq 1$ | Ratio of optimal to worst equilibrium |
| PoS | Best-case NE efficiency | $\geq 1$ | Ratio of optimal to best equilibrium |

**Classic result (Roughgarden & Tardos, 2002):** For selfish routing with linear latency
functions $c_e(x) = a_e x + b_e$, the Price of Anarchy is at most $\frac{4}{3}$.

```python
def price_of_anarchy(payoff_matrix_A, payoff_matrix_B, ne_profiles, all_profiles):
    """Compute Price of Anarchy for a bimatrix game.
    ne_profiles: list of NE strategy profiles (row, col) indices
    all_profiles: all possible (row, col) pairs
    """
    def social_welfare(i, j):
        return payoff_matrix_A[i, j] + payoff_matrix_B[i, j]

    opt_welfare = max(social_welfare(i, j) for i, j in all_profiles)
    worst_ne_welfare = min(social_welfare(i, j) for i, j in ne_profiles)

    if worst_ne_welfare <= 0:
        return float('inf')
    return opt_welfare / worst_ne_welfare
```

> **OKS Relevance:** The Price of Anarchy quantifies the cost of decentralized AMR
> decision-making. If each robot greedily picks the shortest path, the fleet-level
> performance may degrade — the PoA tells us how much. A PoA close to 1 means selfish
> routing is nearly optimal; high PoA signals the need for centralized coordination (RCS).

---

## 8. Learning in Games

### 8.1 Fictitious Play and Convergence

**Fictitious play (Brown, 1951):** Each player maintains an empirical frequency of the
opponent's past actions and best-responds to it.

At round $t$:
1. Player $i$ observes the empirical frequency of $j$'s actions: $\hat{\sigma}_j^t = \frac{1}{t}\sum_{\tau=1}^t \mathbf{1}[s_j^\tau = \cdot]$
2. Player $i$ plays $s_i^{t+1} \in BR_i(\hat{\sigma}_{-i}^t)$

**Convergence results:**
- Converges in 2-player zero-sum games (Robinson, 1951)
- Converges in potential games
- Does **not** converge in all games (Shapley, 1964 gave a $3 \times 3$ counterexample)

### 8.2 Regret Matching

**Regret** for not having played action $a$ instead of the chosen action at time $t$:

$$R_i^T(a) = \sum_{t=1}^T \left[ u_i(a, s_{-i}^t) - u_i(s_i^t, s_{-i}^t) \right]$$

**Regret matching (Hart & Mas-Colell, 2000):** Play action $a$ with probability
proportional to its positive regret:

$$\sigma_i^{t+1}(a) = \frac{\max(0, R_i^t(a))}{\sum_{a'} \max(0, R_i^t(a'))}$$

**Theorem:** If all players use regret matching, the time-averaged strategy profile converges
to the set of **correlated equilibria**.

### 8.3 Multiplicative Weights / No-Regret Learning

The **Multiplicative Weights Update (MWU)** algorithm:

$$w_i^{t+1}(a) = w_i^t(a) \cdot (1 - \eta \cdot \ell_i^t(a))$$
$$\sigma_i^{t+1}(a) = \frac{w_i^{t+1}(a)}{\sum_{a'} w_i^{t+1}(a')}$$

where $\ell_i^t(a)$ is the loss of action $a$ at time $t$ and $\eta > 0$ is the learning rate.

**No-regret guarantee:** After $T$ rounds, the average regret satisfies:

$$\frac{1}{T} R_i^T \leq O\left(\sqrt{\frac{\ln |S_i|}{T}}\right)$$

**Key result:** If all players use no-regret algorithms, the time-averaged play converges to
a **coarse correlated equilibrium** (a relaxation of Nash).

### 8.4 Connection to Online Convex Optimization

Game learning is a special case of **online convex optimization (OCO)**:

| OCO Component | Game Theory Interpretation |
|---------------|---------------------------|
| Decision set $K$ | Simplex $\Delta(S_i)$ |
| Loss function $\ell_t$ | Payoff from opponents' play |
| Regret bound | Quality of learning algorithm |
| Follow-the-regularized-leader | Smooth best response |

The OCO framework unifies fictitious play, MWU, and gradient-based methods under a single
lens. Online mirror descent with negative entropy gives exactly the MWU update.

### 8.5 Bandit Algorithms as Game Players

When players observe only their *own* payoff (not the full payoff matrix), we enter the
**multi-armed bandit** setting:

- **Explore-then-commit:** Play each action $K$ times, then commit to the best
- **UCB1 (Upper Confidence Bound):** Play $a^t = \arg\max_a \left[\hat{\mu}_a + \sqrt{\frac{2 \ln t}{n_a}}\right]$
- **EXP3 (Exponential weights for Exploration and Exploitation):** Adversarial bandit
  algorithm with $O(\sqrt{T |S_i| \ln |S_i|})$ regret

In games, EXP3 provides no-regret guarantees even when the "environment" (other players)
is adversarial — connecting bandit theory to game-theoretic learning.

```python
import numpy as np

def exp3(payoff_fn, n_actions, T, gamma=0.1):
    """EXP3 algorithm for adversarial bandits.
    payoff_fn: callable(action, t) -> reward in [0,1]
    n_actions: number of available actions
    T: number of rounds
    gamma: exploration parameter
    """
    weights = np.ones(n_actions)
    cumulative_rewards = np.zeros(n_actions)

    for t in range(T):
        # Compute mixed strategy
        total = weights.sum()
        probs = (1 - gamma) * weights / total + gamma / n_actions

        # Sample action
        action = np.random.choice(n_actions, p=probs)

        # Observe reward
        reward = payoff_fn(action, t)

        # Update weights
        estimated_reward = reward / probs[action]
        weights[action] *= np.exp(gamma * estimated_reward / n_actions)
        cumulative_rewards[action] += reward

    return cumulative_rewards
```

> **OKS Relevance:** Learning in games applies to adaptive fleet management. As warehouse
> conditions change (order patterns, congestion), robots can use no-regret learning to
> adaptively select routes and strategies. The guardian system's anomaly detection can be
> viewed as a game against adversarial sensor faults — the system "learns" to distinguish
> genuine anomalies from noise.

---

## 9. Robotics and ML Applications

### 9.1 Multi-Robot Task Allocation via Auctions

**Market-based approaches** to task allocation:

1. **Single-item auctions:** Tasks are announced one at a time; robots bid their cost
2. **Combinatorial auctions:** Robots bid on bundles of tasks (capturing synergies)
3. **Sequential single-item:** Approximate combinatorial allocation with lower complexity

**Algorithm (Consensus-Based Bundle Algorithm — CBBA):**
1. Each robot builds a bundle of desired tasks (greedy inclusion)
2. Robots exchange bundles and resolve conflicts via consensus
3. Converges to a conflict-free assignment in $O(n_t)$ iterations ($n_t$ = number of tasks)

| Method | Optimality | Communication | Complexity |
|--------|-----------|---------------|------------|
| Centralized (Hungarian) | Optimal | $O(n^2)$ to center | $O(n^3)$ |
| Sequential auction | Suboptimal | $O(n \cdot n_t)$ | $O(n \cdot n_t^2)$ |
| CBBA | Within 50% of optimal | $O(n^2 \cdot n_t)$ | $O(n \cdot n_t^2)$ |

### 9.2 Pursuit-Evasion as Minimax

The pursuit-evasion problem: pursuers try to catch an evader in an environment.

**Formulation:** Two-player zero-sum game where:
- Pursuer minimizes capture time
- Evader maximizes escape time / capture time
- Minimax strategy gives the optimal pursuit policy under worst-case evasion

**Hamilton-Jacobi-Isaacs (HJI) equation** for continuous pursuit-evasion:

$$\min_u \max_d \left[ \nabla V \cdot f(x, u, d) \right] = 0$$

where $V(x)$ is the value function, $u$ is the pursuer control, $d$ is the evader control,
and $f$ is the dynamics.

### 9.3 GANs as Zero-Sum Games

**Generative Adversarial Networks (Goodfellow et al., 2014):**

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

| Game Theory Concept | GAN Counterpart |
|---------------------|-----------------|
| Player 1 (row) | Discriminator $D$ |
| Player 2 (column) | Generator $G$ |
| Payoff matrix | $V(D, G)$ (continuous) |
| Nash equilibrium | $D^*(x) = \frac{1}{2}$ everywhere, $G$ generates $p_{\text{data}}$ |
| Minimax theorem | Theoretical convergence guarantee |

**Training challenges mirror game-theory difficulties:**
- Mode collapse ↔ cycling in non-convergent games
- Training instability ↔ absence of pure NE
- Spectral normalization ↔ bounding the strategy space

### 9.4 Adversarial Robustness

**Adversarial examples** are a game between a classifier and an attacker:

- Classifier chooses model parameters $\theta$
- Attacker chooses perturbation $\delta$ with $\|\delta\|_p \leq \epsilon$

**Robust optimization formulation:**

$$\min_\theta \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \max_{\|\delta\|_p \leq \epsilon} \mathcal{L}(f_\theta(x + \delta), y) \right]$$

This is a minimax problem. PGD adversarial training (Madry et al., 2018) alternates between:
1. **Inner maximization:** find worst-case perturbation (attacker's move)
2. **Outer minimization:** update model to be robust (defender's move)

### 9.5 Robust Optimization as Games Against Nature

When uncertainty is non-stochastic (set-based), robust optimization models it as a game
between the decision-maker and "nature" (worst-case adversary):

$$\min_x \max_{\xi \in \mathcal{U}} f(x, \xi)$$

| Uncertainty Set $\mathcal{U}$ | Tractability | Resulting Problem |
|-------------------------------|-------------|-------------------|
| Box: $\|\xi\|_\infty \leq \rho$ | LP remains LP | Worst case at vertex |
| Ellipsoidal: $\xi^T \Sigma^{-1} \xi \leq 1$ | LP becomes SOCP | Second-order cone |
| Polyhedral: $C\xi \leq d$ | LP remains LP | Dual formulation |

This connects directly to [Chapter 05: Convex Optimization](05-convex-optimization.md) —
robust linear programs are solvable via conic duality.

### 9.6 Multi-Agent Reinforcement Learning (MARL)

MARL extends single-agent RL to multiple interacting agents:

| Setting | Game Type | Algorithm Examples |
|---------|-----------|-------------------|
| **Cooperative** | Common reward | QMIX, MAPPO, CTDE |
| **Competitive** | Zero-sum | Self-play, PSRO, NeuRD |
| **Mixed** | General-sum | Independent learners, mean-field |

**Key challenges:**
- **Non-stationarity:** each agent's policy changes, making the environment non-stationary
  for every other agent
- **Credit assignment:** in cooperative settings, attributing team reward to individual actions
- **Scalability:** state-action space grows exponentially with the number of agents
- **Equilibrium selection:** multiple equilibria, which one will learning find?

**Centralized Training, Decentralized Execution (CTDE):**
- During training: access global state and all agents' actions
- During execution: each agent uses only local observations
- Bridges cooperative game theory (centralized) with practical deployment (decentralized)

> **OKS Relevance:** MARL directly applies to fleet-level AMR coordination. Robots must
> cooperatively navigate a warehouse, avoid collisions, and maximize throughput. The CTDE
> paradigm mirrors the OKS architecture: RCS has a global view during planning (centralized),
> but each robot executes locally based on its sensors and local state (decentralized).
> Guardian's multi-robot deadlock detection is essentially identifying non-convergent game
> dynamics in real time.

---

## 10. Notation Reference Table

| Symbol | Meaning |
|--------|---------|
| $N = \{1, \ldots, n\}$ | Set of players |
| $S_i$ | Strategy set of player $i$ |
| $S = S_1 \times \cdots \times S_n$ | Strategy space (all profiles) |
| $s = (s_1, \ldots, s_n)$ | Strategy profile |
| $s_{-i}$ | Strategies of all players except $i$ |
| $u_i(s)$ | Payoff function of player $i$ |
| $\Delta(S_i)$ | Simplex over $S_i$ (mixed strategies) |
| $\sigma_i \in \Delta(S_i)$ | Mixed strategy of player $i$ |
| $\sigma = (\sigma_1, \ldots, \sigma_n)$ | Mixed strategy profile |
| $BR_i(s_{-i})$ | Best response correspondence of player $i$ |
| $v$ | Value of a zero-sum game |
| $A$ | Payoff matrix (row player in zero-sum games) |
| $(A, B)$ | Bimatrix game payoff matrices |
| $v(S)$ | Characteristic function (coalitional games) |
| $\phi_i(v)$ | Shapley value of player $i$ |
| $\Phi(s)$ | Potential function |
| $R_i^T(a)$ | Cumulative regret of action $a$ after $T$ rounds |
| $\delta$ | Discount factor (repeated games) |
| $\theta_i$ | Private type of player $i$ (mechanism design) |
| $\text{PoA}$ | Price of Anarchy |
| $\text{PoS}$ | Price of Stability |

**Convention summary:**
- Payoff matrices: row player is Player 1, column player is Player 2
- In a bimatrix $(A, B)$: $A_{ij}$ is Player 1's payoff, $B_{ij}$ is Player 2's payoff
- Zero-sum: Player 2's payoff is $-A$
- Mixed strategy: probability vector $\sigma_i \in \Delta(S_i)$
- Expected payoff: $u_i(\sigma) = \sum_{s \in S} \left(\prod_{j \in N} \sigma_j(s_j)\right) u_i(s)$
- Nash equilibrium: $\sigma^*$ such that $u_i(\sigma_i^*, \sigma_{-i}^*) \geq u_i(\sigma_i, \sigma_{-i}^*)$ for all $\sigma_i \in \Delta(S_i)$ and all $i \in N$

---

## 11. Recommended Resources

### Textbooks

| Book | Focus | Level |
|------|-------|-------|
| **Osborne & Rubinstein**, *A Course in Game Theory* (1994) | Classic non-cooperative theory | Graduate |
| **Osborne**, *An Introduction to Game Theory* (2003) | Accessible introduction | Undergraduate |
| **Shoham & Leyton-Brown**, *Multiagent Systems* (2009) | Algorithmic and computational | Graduate |
| **Nisan, Roughgarden, Tardos, Vazirani**, *Algorithmic Game Theory* (2007) | CS perspective, PoA, mechanisms | Graduate |
| **Fudenberg & Tirole**, *Game Theory* (1991) | Comprehensive, economics-oriented | Graduate |
| **Myerson**, *Game Theory: Analysis of Conflict* (1991) | Mechanism design, Bayesian games | Graduate |
| **Maschler, Solan, Zamir**, *Game Theory* (2013) | Modern comprehensive treatment | Graduate |

### Key Papers

| Paper | Contribution |
|-------|-------------|
| Nash (1950), "Equilibrium points in n-person games" | Existence of mixed NE |
| Von Neumann (1928), "Zur Theorie der Gesellschaftsspiele" | Minimax theorem |
| Shapley (1953), "A value for n-person games" | Shapley value axiomatization |
| Daskalakis, Goldberg, Papadimitriou (2009) | PPAD-completeness of Nash |
| Roughgarden & Tardos (2002) | Price of Anarchy for selfish routing |
| Vickrey (1961), "Counterspeculation, auctions, and sealed tenders" | Second-price auction (VCG precursor) |
| Hart & Mas-Colell (2000) | Regret matching → correlated equilibria |
| Goodfellow et al. (2014) | GANs as minimax games |
| Madry et al. (2018) | PGD adversarial training as robust optimization |

### Software Libraries

| Library | Language | Purpose |
|---------|----------|---------|
| **Nashpy** | Python | Two-player game solving (support enumeration, vertex enumeration, Lemke-Howson) |
| **Gambit** | C++/Python | Extensive and normal-form game computation |
| **OpenSpiel** | C++/Python | RL and game theory research (DeepMind) |
| **PettingZoo** | Python | Multi-agent RL environments |
| **RLlib** | Python | Scalable multi-agent RL (Ray) |
| **CVXPY** | Python | LP/SOCP formulations of zero-sum and robust games |

### Quick Start with Nashpy

```python
import nashpy as nash
import numpy as np

# Define a 2-player game (Battle of the Sexes)
A = np.array([[3, 0], [0, 2]])  # Player 1 payoffs
B = np.array([[2, 0], [0, 3]])  # Player 2 payoffs

game = nash.Game(A, B)

# Find all Nash equilibria via support enumeration
equilibria = list(game.support_enumeration())
for eq in equilibria:
    sigma1, sigma2 = eq
    print(f"P1: {sigma1}, P2: {sigma2}")
    print(f"Expected payoffs: {game[sigma1, sigma2]}")

# Output:
# P1: [1. 0.], P2: [1. 0.]       — (Opera, Opera)
# P1: [0. 1.], P2: [0. 1.]       — (Football, Football)
# P1: [0.6 0.4], P2: [0.4 0.6]   — Mixed NE
```

---

## Summary of Key Connections

```
Game Theory in the Optimization Landscape
==========================================

                    LP Duality (Ch 04-05)
                         |
                    Minimax Theorem (§3)
                    /           \
           Zero-Sum Games    Robust Optimization (§9.5)
                |                    |
         LP Formulation      Convex Optimization (Ch 05)
                |
        Algorithmic GT (§7)
           /        \
    Potential Games   PoA/PoS
         |                \
    Congestion Games    Network Design
         |
    Convex Program (Wardrop)

    Mechanism Design (§6)        Learning in Games (§8)
         |                            |
    Auction Theory              Online Convex Opt
         |                            |
    Task Allocation (§9.1)      No-Regret Algorithms
                                      |
                                MARL (§9.6)
```

**Core takeaway:** Game theory extends optimization from a single decision-maker to multiple
interacting agents. The minimax theorem connects to LP duality, potential games reduce
multi-agent problems to single-agent optimization, and learning in games connects to online
convex optimization. In robotics, these ideas underpin task allocation, adversarial planning,
and multi-agent coordination.
