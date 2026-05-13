# Day 102: Dynamic Games and Backward Induction

> **Phase VII — Game Theory for Optimization & Robotics | Week 15 | 2.5 hours**
> *When moves happen in sequence, the future determines the present — solve from the end.*

## Navigation
- **Previous:** [Day 101: Bargaining Theory](day-101-bargaining.md)
- **Next:** [Day 103: Repeated Games and Folk Theorem](day-103-repeated-games.md)
- [Week 15 Overview](../../../CURRICULUM.md)
- [Chapter 10: Game Theory Reference](../../../10-game-theory.md)
- [Full Curriculum](../../../00-learning-plan.md)

## OKS Relevance
Two robots approach an intersection. Robot A arrives first and decides: proceed or yield. Robot B observes A's choice and reacts. This sequential structure is an extensive-form game. Backward induction finds the subgame-perfect strategy — eliminating non-credible threats like "I'll crash into you if you don't yield," which no rational robot would execute.

---

## Theory (45 min)

### 102.1 Extensive Form Games

An **extensive-form game** consists of:
- A **game tree** $T$ with nodes (decision points) and edges (actions).
- **Player assignment** at each decision node.
- **Information sets** — groups of nodes where a player cannot distinguish between them.
- **Payoff vectors** at terminal nodes.

A game has **perfect information** if every information set is a singleton (each player knows exactly where they are in the tree).

### 102.2 Strategies in Extensive Form

A **strategy** for player $i$ specifies an action at every information set of player $i$ — even those unreached. This is why extensive-form games can have more Nash equilibria than seem reasonable.

A **subgame** is a subtree starting at a node that:
1. Is a singleton information set
2. Contains all successors of that node
3. Does not break any information set

### 102.3 Backward Induction and SPE

**Backward induction** in perfect-information games:
1. Start at terminal nodes.
2. At each decision node, choose the action maximizing that player's payoff.
3. Replace the subtree with the resulting payoff vector.
4. Repeat until the root.

**Kuhn's Theorem:** Every finite game of perfect information has a pure-strategy Nash equilibrium findable by backward induction.

A **subgame perfect equilibrium (SPE)** is a strategy profile that induces a Nash equilibrium in every subgame. Backward induction finds all SPE in perfect-information games.

### 102.4 Non-Credible Threats

A Nash equilibrium may rely on **non-credible threats** — actions a player would never actually take if that part of the game were reached. SPE eliminates these by requiring rationality at every decision point, not just on the equilibrium path.

**Example:** Entry deterrence. Incumbent threatens to fight if entrant enters, but fighting is costly. The threat is non-credible; SPE predicts entry + accommodate.

---

## Implementation (60 min)

```python
"""Dynamic games and backward induction — Day 102"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GameNode:
    """Node in an extensive-form game tree."""
    player: Optional[int] = None  # None for terminal
    actions: dict = field(default_factory=dict)  # action_name -> child GameNode
    payoffs: Optional[tuple] = None  # terminal payoffs (p1, p2, ...)
    name: str = ""


def backward_induction(node: GameNode, n_players: int = 2) -> tuple:
    """Solve a perfect-information game by backward induction.
    
    Returns:
        (payoff_vector, action_dict) — equilibrium payoffs and strategy profile
    """
    if node.payoffs is not None:
        return node.payoffs, {}
    
    best_payoff = None
    best_action = None
    all_strategies = {}
    
    for action_name, child in node.actions.items():
        child_payoff, child_strategies = backward_induction(child, n_players)
        if best_payoff is None or child_payoff[node.player] > best_payoff[node.player]:
            best_payoff = child_payoff
            best_action = action_name
            all_strategies = dict(child_strategies)
    
    all_strategies[node.name] = best_action
    return best_payoff, all_strategies


def build_entry_game():
    """Entry deterrence game.
    
    Entrant: Enter or Stay Out
    If Enter -> Incumbent: Fight or Accommodate
    """
    fight = GameNode(payoffs=(-1, -1), name="fight_outcome")
    accommodate = GameNode(payoffs=(2, 1), name="accommodate_outcome")
    incumbent = GameNode(player=1, actions={"Fight": fight, "Accommodate": accommodate},
                         name="Incumbent")
    stay_out = GameNode(payoffs=(0, 3), name="stay_out_outcome")
    entrant = GameNode(player=0, actions={"Enter": incumbent, "Stay Out": stay_out},
                       name="Entrant")
    return entrant


def build_centipede(rounds=4, gain=1, loss=2):
    """Build a centipede game with alternating players."""
    pot = [1, 1]
    # Build from the end
    terminal = GameNode(payoffs=tuple(pot), name=f"end")
    current = terminal
    for r in range(rounds - 1, -1, -1):
        player = r % 2
        take_payoffs = list(pot)
        take_payoffs[player] += gain
        take_payoffs[1 - player] -= loss // 2  # simplified
        take = GameNode(payoffs=tuple(take_payoffs), name=f"take_{r}")
        node = GameNode(player=player,
                        actions={"Take": take, "Pass": current},
                        name=f"P{player+1}_round{r}")
        pot[0] += 1; pot[1] += 1
        current = node
    return current


def build_stackelberg():
    """Stackelberg duopoly (discrete quantities).
    
    Leader chooses q1 in {1,2,3,4}, Follower best-responds q2.
    Payoffs: pi_i = (10 - q1 - q2) * qi - 2*qi
    """
    root_actions = {}
    for q1 in range(1, 5):
        follower_actions = {}
        for q2 in range(1, 5):
            p = 10 - q1 - q2
            pi1 = max(0, p * q1 - 2 * q1)
            pi2 = max(0, p * q2 - 2 * q2)
            follower_actions[f"q2={q2}"] = GameNode(
                payoffs=(pi1, pi2), name=f"outcome_q1={q1}_q2={q2}")
        root_actions[f"q1={q1}"] = GameNode(
            player=1, actions=follower_actions, name=f"Follower_q1={q1}")
    return GameNode(player=0, actions=root_actions, name="Leader")


def print_game_tree(node, indent=0):
    """Pretty-print a game tree."""
    prefix = "  " * indent
    if node.payoffs is not None:
        print(f"{prefix}→ Payoffs: {node.payoffs}")
        return
    print(f"{prefix}[{node.name}] Player {node.player + 1} chooses:")
    for action, child in node.actions.items():
        print(f"{prefix}  {action}:")
        print_game_tree(child, indent + 2)


# --- Demo ---
if __name__ == "__main__":
    # Entry deterrence
    print("=== Entry Deterrence ===")
    game = build_entry_game()
    print_game_tree(game)
    payoff, strategy = backward_induction(game)
    print(f"SPE payoffs: {payoff}")
    print(f"SPE strategy: {strategy}")
    print("(Entrant enters, Incumbent accommodates — 'fight' is non-credible)\n")
    
    # Stackelberg duopoly
    print("=== Stackelberg Duopoly ===")
    game = build_stackelberg()
    payoff, strategy = backward_induction(game)
    print(f"SPE payoffs: {payoff}")
    print(f"SPE strategy: {strategy}")
    
    # Compare with simultaneous Cournot
    # Cournot NE: q1 = q2 = (a-c)/(3) = (10-2)/3 ≈ 2.67
    print(f"Cournot NE would give q1=q2≈2.67, leader advantage in Stackelberg\n")
    
    # Centipede
    print("=== Centipede Game (4 rounds) ===")
    game = build_centipede(4)
    payoff, strategy = backward_induction(game)
    print(f"SPE payoffs: {payoff}")
    print(f"SPE: immediate take (paradox of backward induction)")
```

---

## Practice Problems (45 min)

**P1.** Solve by backward induction: Player 1 chooses L or R. If L: Player 2 chooses A (payoffs 3,1) or B (payoffs 0,0). If R: Player 2 chooses C (payoffs 1,2) or D (payoffs 2,3).

<details><summary>Answer</summary>

P2 at L-node: A gives 1 > B gives 0, so A. P2 at R-node: D gives 3 > C gives 2, so D. P1 chooses: L→(3,1) vs R→(2,3). P1 picks L. **SPE: (L; A, D) with payoffs (3,1).**
</details>

**P2.** Find all Nash equilibria of the entry game. Which are subgame perfect?

<details><summary>Answer</summary>

NE: {(Enter, Accommodate), (Stay Out, Fight)}. Only **(Enter, Accommodate) is SPE** — "Fight" is non-credible because if entry happens, Incumbent prefers Accommodate (1 > -1).
</details>

**P3.** In a 3-stage sequential game, Player 1 moves first, then Player 2, then Player 1 again. How many strategies does each player have if each has 2 actions at each node?

<details><summary>Answer</summary>

P1 has 2 nodes (root + 2 possible stage-3 nodes, but only reaches based on P2's choice). P1 strategies: $2 \times 2^2 = 8$ (action at root × action at each of 2 possible third-stage nodes). P2 has $2^2 = 4$ strategies (action at each of 2 possible nodes).
</details>

---

## Expert Challenges

**E1.** Implement a Stackelberg game with continuous quantities. Solve for the leader's optimal quantity using backward induction analytically and verify numerically.

<details><summary>Hint</summary>

Follower's BR: $q_2^*(q_1) = (a - c - q_1)/2$. Leader maximizes $(a - q_1 - q_2^*(q_1) - c) q_1$. Compare with Cournot.
</details>

**E2.** The centipede game predicts immediate defection, but experiments show cooperation. Implement a bounded-rationality model where players look ahead only $k$ steps and show how cooperation emerges.

<details><summary>Hint</summary>

With $k$-step lookahead, players in early rounds don't see the terminal defection incentive. Cooperation survives for $\text{rounds} - k$ steps.
</details>

**E3.** Model robot intersection priority as a sequential game with imperfect information (robot B doesn't observe A's speed, only direction). Compare SPE with perfect vs imperfect info.

<details><summary>Hint</summary>

With imperfect info, B's information set groups nodes where A chose different speeds. B must use the same action at all nodes in an information set.
</details>

---

## Connections
- **Prerequisites:** Normal form games (Day 92) — every extensive game has a normal form; Nash equilibrium (Day 93) — SPE refines NE
- **Forward:** Repeated games (Day 103) — repeat a stage game over time; Mechanism design (Day 104) — sequential revelation
- **OKS:** Intersection priority protocols, sequential task assignment, leader-follower fleet coordination

## Self-Check
- [ ] I can convert a game tree to extensive form and identify subgames
- [ ] I can apply backward induction to find subgame perfect equilibria
- [ ] I understand why SPE eliminates non-credible threats
- [ ] I can compare Stackelberg (sequential) vs Cournot (simultaneous) outcomes
