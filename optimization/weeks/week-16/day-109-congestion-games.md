# Day 109: Potential Games and Congestion Games

> **Phase VII — Game Theory for Optimization & Robotics | Week 16 | 2.5 hours**
> *When selfish agents optimize a shared potential function, decentralized behavior can be surprisingly efficient — or catastrophically wasteful.*

## Navigation
- Previous: [Day 108: No-Regret Learning](day-108-no-regret.md)
- Next: [Day 110: Multi-Robot Games + Adversarial ML](day-110-multi-robot-adversarial.md)
- [Week 16 Overview](../../../CURRICULUM.md)
- [Chapter 10: Game Theory Reference](../../../10-game-theory.md)
- [Full Curriculum](../../../00-learning-plan.md)

## OKS Relevance
Warehouse aisle routing is a textbook congestion game: each AMR chooses a path, and delay increases with the number of robots on each segment. The Price of Anarchy determines how much worse decentralized routing is compared to a centralized dispatcher. Understanding this gap drives OKS fleet architecture decisions — when to let robots self-route vs. when central coordination is worth the overhead.

---

## Theory (45 min)

### 109.1 Exact Potential Games

A game is an **exact potential game** if there exists a function $\Phi: \mathcal{S} \to \mathbb{R}$ such that for every player $i$, every strategy pair $(s_i, s_i')$, and every opponent profile $s_{-i}$:

$$u_i(s_i, s_{-i}) - u_i(s_i', s_{-i}) = \Phi(s_i, s_{-i}) - \Phi(s_i', s_{-i})$$

**Key property**: Any sequence of unilateral improvements (a player switching to increase their payoff) also increases $\Phi$. Since $\Phi$ is bounded, this process terminates at a **pure Nash equilibrium**. Every potential game has at least one pure NE.

### 109.2 Congestion Games (Rosenthal 1973)

A **congestion game** has:
- Players $N = \{1, \ldots, n\}$
- Resources $E = \{e_1, \ldots, e_m\}$
- Strategy sets $\mathcal{S}_i \subseteq 2^E$ (subsets of resources)
- Delay functions $d_e: \mathbb{N} \to \mathbb{R}$ for each resource $e$

Player $i$'s cost: $c_i(s) = \sum_{e \in s_i} d_e(n_e(s))$ where $n_e(s) = |\{j : e \in s_j\}|$.

**Rosenthal's Theorem**: Every congestion game is a potential game with:

$$\Phi(s) = \sum_{e \in E} \sum_{k=1}^{n_e(s)} d_e(k)$$

### 109.3 Price of Anarchy and Braess's Paradox

The **social cost** is $C(s) = \sum_i c_i(s)$. The **Price of Anarchy** (PoA) measures the worst-case ratio:

$$\text{PoA} = \frac{\max_{s \in \text{NE}} C(s)}{\min_{s} C(s)} = \frac{\text{worst NE cost}}{\text{optimal cost}}$$

The **Price of Stability** (PoS) uses the best NE instead.

**Braess's Paradox**: Adding a road to a network can increase equilibrium travel time for all users. Consider a network with $n$ users, origin $O$, destination $D$:
- Upper path: $O \to A \to D$ with costs $d_{OA}(x) = x/n$, $d_{AD}(x) = 1$
- Lower path: $O \to B \to D$ with costs $d_{OB}(x) = 1$, $d_{BD}(x) = x/n$

Without bridge $A \to B$: equilibrium splits traffic equally, cost = $3/2$ per user.
Add bridge $A \to B$ with $d_{AB}(x) = 0$: everyone uses $O \to A \to B \to D$, cost = $2$ per user.

**Theorem (Roughgarden & Tardos 2002)**: For linear delay functions $d_e(x) = a_e x + b_e$, the PoA $\leq 4/3$.

---

## Implementation (60 min)

```python
"""Congestion Games: equilibrium computation and Price of Anarchy."""
import numpy as np
from itertools import product
from typing import Callable

class CongestionGame:
    """A congestion game with n players and m resources."""
    
    def __init__(self, n_players: int, strategies: list,
                 delay_fns: dict):
        """
        Args:
            n_players: number of players
            strategies: list of lists, strategies[i] = list of resource subsets
            delay_fns: dict mapping resource -> delay function (int -> float)
        """
        self.n = n_players
        self.strategies = strategies
        self.delay_fns = delay_fns
        self.resources = list(delay_fns.keys())
    
    def congestion(self, profile: tuple) -> dict:
        """Compute congestion on each resource."""
        counts = {e: 0 for e in self.resources}
        for i, si in enumerate(profile):
            for e in self.strategies[i][si]:
                counts[e] += 1
        return counts
    
    def cost(self, profile: tuple, player: int) -> float:
        """Cost for player given strategy profile."""
        counts = self.congestion(profile)
        return sum(self.delay_fns[e](counts[e])
                   for e in self.strategies[player][profile[player]])
    
    def social_cost(self, profile: tuple) -> float:
        """Total cost across all players."""
        return sum(self.cost(profile, i) for i in range(self.n))
    
    def potential(self, profile: tuple) -> float:
        """Rosenthal potential function."""
        counts = self.congestion(profile)
        return sum(sum(self.delay_fns[e](k) for k in range(1, counts[e] + 1))
                   for e in self.resources)
    
    def find_all_ne(self) -> list:
        """Find all pure Nash equilibria by exhaustive search."""
        equilibria = []
        n_strats = [len(s) for s in self.strategies]
        
        for profile in product(*[range(k) for k in n_strats]):
            is_ne = True
            for i in range(self.n):
                current_cost = self.cost(profile, i)
                for alt in range(n_strats[i]):
                    if alt != profile[i]:
                        alt_profile = list(profile)
                        alt_profile[i] = alt
                        if self.cost(tuple(alt_profile), i) < current_cost - 1e-10:
                            is_ne = False
                            break
                if not is_ne:
                    break
            if is_ne:
                equilibria.append(profile)
        return equilibria
    
    def price_of_anarchy(self) -> float:
        """Compute Price of Anarchy."""
        nes = self.find_all_ne()
        n_strats = [len(s) for s in self.strategies]
        
        worst_ne = max(self.social_cost(ne) for ne in nes)
        opt = min(self.social_cost(p)
                  for p in product(*[range(k) for k in n_strats]))
        return worst_ne / opt if opt > 0 else float('inf')


def braess_paradox_demo():
    """Demonstrate Braess's paradox with 2 players."""
    # Without bridge: 2 paths (upper, lower)
    # Upper: O->A (x/2), A->D (1)
    # Lower: O->B (1), B->D (x/2)
    delay_no_bridge = {
        'OA': lambda x: x / 2, 'AD': lambda x: 1.0,
        'OB': lambda x: 1.0, 'BD': lambda x: x / 2,
    }
    # Player strategies: 0 = upper (OA, AD), 1 = lower (OB, BD)
    strats_no = [
        [['OA', 'AD'], ['OB', 'BD']],  # Player 0
        [['OA', 'AD'], ['OB', 'BD']],  # Player 1
    ]
    game_no = CongestionGame(2, strats_no, delay_no_bridge)
    
    # With bridge: add path O->A->B->D
    delay_bridge = {
        'OA': lambda x: x / 2, 'AD': lambda x: 1.0,
        'OB': lambda x: 1.0, 'BD': lambda x: x / 2,
        'AB': lambda x: 0.0,  # free bridge
    }
    strats_br = [
        [['OA', 'AD'], ['OB', 'BD'], ['OA', 'AB', 'BD']],
        [['OA', 'AD'], ['OB', 'BD'], ['OA', 'AB', 'BD']],
    ]
    game_br = CongestionGame(2, strats_br, delay_bridge)
    
    print("=== Braess's Paradox ===")
    nes_no = game_no.find_all_ne()
    for ne in nes_no:
        print(f"  No bridge NE: {ne}, social cost = {game_no.social_cost(ne):.2f}")
    
    nes_br = game_br.find_all_ne()
    for ne in nes_br:
        print(f"  With bridge NE: {ne}, social cost = {game_br.social_cost(ne):.2f}")
    
    print(f"  PoA without bridge: {game_no.price_of_anarchy():.3f}")
    print(f"  PoA with bridge: {game_br.price_of_anarchy():.3f}")

braess_paradox_demo()

# Warehouse aisle congestion game
print("\n=== Warehouse Aisle Congestion ===")
delay_aisle = {
    'aisle_A': lambda x: 1 + 2 * x,
    'aisle_B': lambda x: 3 + x,
    'aisle_C': lambda x: 2 + 1.5 * x,
}
# 3 robots, each chooses one aisle
strats_wh = [[['aisle_A'], ['aisle_B'], ['aisle_C']]] * 3
game_wh = CongestionGame(3, strats_wh, delay_aisle)

nes_wh = game_wh.find_all_ne()
print(f"Found {len(nes_wh)} NE")
for ne in nes_wh:
    print(f"  NE: {ne} (aisles: {[['A','B','C'][s] for s in ne]}), "
          f"social cost = {game_wh.social_cost(ne):.2f}")
print(f"  PoA: {game_wh.price_of_anarchy():.3f}")
```

---

## Practice Problems (45 min)

**P1**: Verify that the warehouse aisle game above is a potential game by computing $\Phi$ for two strategy profiles that differ by one player's deviation.

<details><summary>Answer</summary>

Take profiles $(0, 0, 1)$ and $(1, 0, 1)$ (player 0 switches A→B). $\Delta c_0 = d_B(1) - d_A(2) = 4 - 5 = -1$. $\Delta\Phi = [d_B(1)] - [d_A(2)] = 4 - 5 = -1$. These match, confirming the potential property. The potential is $\Phi(s) = \sum_e \sum_{k=1}^{n_e} d_e(k)$.
</details>

**P2**: Construct a 4-player, 2-resource congestion game with linear delays. Compute all NE and the PoA.

<details><summary>Answer</summary>

Resources $\{A, B\}$ with $d_A(x) = x$, $d_B(x) = 2$. Each player picks A or B. Equilibrium: 2 on A (cost 2 each), 2 on B (cost 2 each) — total 8. Optimum: same split, cost 8. If 3 on A: cost 3+3+3+2=11, not NE (player on A deviates to B for 2<3). PoA = 8/8 = 1. With $d_B(x) = 3$: equilibrium has 3 on A, 1 on B; PoA = (3·3+3)/(2·2+2·3) = 12/10 = 1.2.
</details>

**P3**: Prove that every finite potential game has at least one pure NE.

<details><summary>Answer</summary>

Consider the strategy profile $s^*$ that maximizes $\Phi$. For any player $i$ and deviation $s_i' \neq s_i^*$: $\Phi(s_i^*, s_{-i}^*) \geq \Phi(s_i', s_{-i}^*)$. By the potential property: $u_i(s_i^*, s_{-i}^*) - u_i(s_i', s_{-i}^*) = \Phi(s_i^*, s_{-i}^*) - \Phi(s_i', s_{-i}^*) \geq 0$. So no player benefits from deviating — $s^*$ is a pure NE.
</details>

---

## Expert Challenges

**E1**: Prove the Roughgarden-Tardos bound: PoA $\leq 4/3$ for linear congestion games.

<details><summary>Hint</summary>

Use the variational inequality characterization of Wardrop equilibrium. For linear $d_e(x) = a_e x + b_e$, show that at NE flow $f^*$: $\sum_e d_e(f_e^*) f_e^* \leq \sum_e d_e(f_e^*) f_e^{\text{opt}}$. Then use the inequality $x(ax+b) \leq 4/3 \cdot [ax^2/2 + bx - a(x-y)^2/2]$ for the social cost bound.
</details>

**E2**: Implement best-response dynamics for a 10-player congestion game and verify convergence to a NE.

<details><summary>Answer</summary>

Starting from any profile, repeatedly let an unsatisfied player switch to a better strategy. Since each switch increases $\Phi$, the process terminates. With 10 players and 5 resources with random linear delays, convergence typically occurs within 20-50 rounds. The final profile is verifiably a NE.
</details>

**E3**: Model OKS warehouse routing as a congestion game with 6 robots, 4 corridors, and quadratic delay. Compute the PoA and suggest when centralized routing is justified.

<details><summary>Answer</summary>

With quadratic delays $d_e(x) = a_e x^2$, PoA can exceed $4/3$. For specific parameters: PoA ≈ 1.5 means decentralized routing wastes 50% more time than optimal. Centralized routing is justified when: (1) PoA > 1.3, (2) congestion is frequent (>30% of time), (3) communication latency < replanning latency. For OKS, congestion is intermittent; a hybrid approach (decentralized + congestion alerts) balances overhead vs. efficiency.
</details>

---

## Connections
- **Back**: [Day 108: No-Regret Learning](day-108-no-regret.md) — no-regret dynamics converge in potential games
- **Forward**: [Day 110: Multi-Robot Games](day-110-multi-robot-adversarial.md) — task allocation extends congestion model
- **OKS**: Aisle congestion modeling, fleet routing architecture decisions, centralized vs. decentralized tradeoffs

## Self-Check
- [ ] I can define a potential game and construct the Rosenthal potential for congestion games
- [ ] I understand Braess's paradox and can construct examples
- [ ] I can compute Price of Anarchy for small games
- [ ] I know the PoA bound for linear congestion games and its implications for fleet routing
