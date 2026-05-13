# Day 103: Repeated Games and Folk Theorem

> **Phase VII — Game Theory for Optimization & Robotics | Week 15 | 2.5 hours**
> *Play once, defect. Play forever, cooperate — the shadow of the future changes everything.*

## Navigation
- **Previous:** [Day 102: Dynamic Games and Backward Induction](day-102-dynamic-games.md)
- **Next:** [Day 104: Mechanism Design and Auctions](day-104-mechanism-design.md)
- **Week:** [Week 15 Overview](../README.md)
- **Chapter:** [Chapter 10: Game Theory Reference](../../10-game-theory.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Robots in a warehouse interact repeatedly at shared corridors, intersections, and charging stations. A robot that aggressively cuts off others gains short-term speed but faces retaliation tomorrow. The folk theorem guarantees that cooperative norms (yield at intersections, share corridors) can be sustained as equilibria when robots value future interactions — exactly the regime of a permanent fleet.

---

## Theory (45 min)

### 103.1 Repeated Game Framework

Given a **stage game** $G$, the **$T$-fold repeated game** $G(T)$ plays $G$ for $T$ rounds. Players observe all previous actions (perfect monitoring).

In the **infinitely repeated game** $G(\infty, \delta)$, player $i$'s payoff is:

$$U_i = (1 - \delta) \sum_{t=0}^{\infty} \delta^t u_i(a^t)$$

where $\delta \in (0, 1)$ is the **discount factor** and the $(1-\delta)$ normalization makes payoffs comparable to stage-game payoffs.

### 103.2 Trigger Strategies

**Grim trigger:** Cooperate until any player defects, then defect forever.

For the Prisoner's Dilemma with payoffs $R$ (reward), $T$ (temptation), $P$ (punishment), $S$ (sucker), where $T > R > P > S$:

Cooperation under grim trigger is an SPE iff:

$$\delta \geq \frac{T - R}{T - P}$$

**Tit-for-Tat (TFT):** Start cooperating. Then copy opponent's last action.
- Simple, retaliatory, forgiving, clear.
- Won Axelrod's tournament (1980).

**Win-Stay-Lose-Shift (Pavlov):** Repeat last action if payoff was $R$ or $T$; switch if $P$ or $S$.

### 103.3 The Folk Theorem

The **feasible payoff set** $F$ is the convex hull of all pure-strategy payoff vectors.

The **minmax value** for player $i$: $\underline{v}_i = \min_{a_{-i}} \max_{a_i} u_i(a_i, a_{-i})$.

A payoff vector $v$ is **individually rational** if $v_i \geq \underline{v}_i$ for all $i$.

**Folk Theorem (Friedman 1971):** Any feasible, individually rational payoff vector $v$ with $v_i > \underline{v}_i$ can be sustained as a Nash equilibrium of $G(\infty, \delta)$ for $\delta$ sufficiently close to 1.

**Stronger version (Fudenberg-Maskin 1986):** Under full dimension condition, any feasible strictly individually rational payoff is achievable in **subgame perfect equilibrium** for large $\delta$.

### 103.4 Finite Repetition

For finitely repeated games with a unique NE in the stage game: backward induction gives the stage NE in every round. **No cooperation emerges.**

Exception: if the stage game has multiple NE, cooperation can be sustained in early rounds by threatening to switch to a worse NE in the final rounds.

---

## Implementation (60 min)

```python
"""Repeated games and folk theorem — Day 103"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import product


# --- Strategy definitions ---
def always_cooperate(history, player):
    return 'C'

def always_defect(history, player):
    return 'D'

def tit_for_tat(history, player):
    if not history:
        return 'C'
    opponent = 1 - player
    return history[-1][opponent]

def grim_trigger(history, player):
    opponent = 1 - player
    if any(h[opponent] == 'D' for h in history):
        return 'D'
    return 'C'

def pavlov(history, player):
    if not history:
        return 'C'
    my_last = history[-1][player]
    opp_last = history[-1][1 - player]
    if (my_last == 'C' and opp_last == 'C') or (my_last == 'D' and opp_last == 'D'):
        return my_last  # win-stay
    return 'D' if my_last == 'C' else 'C'  # lose-shift

def random_strategy(history, player):
    return np.random.choice(['C', 'D'])


# --- Game engine ---
PD_PAYOFFS = {
    ('C', 'C'): (3, 3), ('C', 'D'): (0, 5),
    ('D', 'C'): (5, 0), ('D', 'D'): (1, 1)
}

def play_repeated_game(strat1, strat2, rounds=200, payoffs=None):
    """Simulate a repeated game."""
    if payoffs is None:
        payoffs = PD_PAYOFFS
    history = []
    scores = [0, 0]
    score_history = [[], []]
    for _ in range(rounds):
        a1 = strat1(history, 0)
        a2 = strat2(history, 1)
        p1, p2 = payoffs[(a1, a2)]
        scores[0] += p1; scores[1] += p2
        history.append((a1, a2))
        score_history[0].append(scores[0])
        score_history[1].append(scores[1])
    return scores, history, score_history


def run_tournament(strategies, rounds=200, payoffs=None):
    """Round-robin tournament."""
    names = list(strategies.keys())
    n = len(names)
    total_scores = {name: 0 for name in names}
    
    for i in range(n):
        for j in range(n):
            scores, _, _ = play_repeated_game(
                strategies[names[i]], strategies[names[j]], rounds, payoffs)
            total_scores[names[i]] += scores[0]
            total_scores[names[j]] += scores[1]
    
    ranking = sorted(total_scores.items(), key=lambda x: -x[1])
    return ranking


def threshold_delta(T, R, P):
    """Minimum discount factor for grim trigger cooperation in PD."""
    return (T - R) / (T - P)


def plot_payoff_evolution(strat1, strat2, name1, name2, rounds=100):
    """Plot cumulative and per-round average payoffs."""
    scores, history, score_hist = play_repeated_game(strat1, strat2, rounds)
    avg1 = [score_hist[0][t] / (t + 1) for t in range(rounds)]
    avg2 = [score_hist[1][t] / (t + 1) for t in range(rounds)]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(score_hist[0], label=name1); ax1.plot(score_hist[1], label=name2)
    ax1.set_title('Cumulative Payoffs'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(avg1, label=name1); ax2.plot(avg2, label=name2)
    ax2.axhline(y=3, color='g', linestyle='--', alpha=0.5, label='Mutual C')
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='Mutual D')
    ax2.set_title('Average Payoff per Round'); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('repeated_game.png', dpi=100); plt.show()


def visualize_folk_theorem():
    """Visualize feasible and individually rational regions for PD."""
    fig, ax = plt.subplots(figsize=(7, 6))
    # Feasible set: convex hull of payoff vectors
    payoff_vecs = [(3,3), (0,5), (5,0), (1,1)]
    from matplotlib.patches import Polygon
    hull = Polygon([(0,5),(3,3),(5,0),(1,1)], alpha=0.2, color='blue', label='Feasible')
    ax.add_patch(hull)
    # Minmax = 1 for both (can guarantee 1 by playing D)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    ax.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='Minmax = 1')
    ax.fill_between([1, 5], 1, 5, alpha=0.1, color='green', label='IR region')
    for v, label in zip(payoff_vecs, ['(C,C)', '(C,D)', '(D,C)', '(D,D)']):
        ax.plot(*v, 'ko', markersize=8)
        ax.annotate(label, v, textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel('Player 1 payoff'); ax.set_ylabel('Player 2 payoff')
    ax.set_title('Folk Theorem: Achievable Payoffs (PD)')
    ax.legend(); ax.set_xlim(-0.5, 6); ax.set_ylim(-0.5, 6); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig('folk_theorem.png', dpi=100); plt.show()


# --- Demo ---
if __name__ == "__main__":
    delta_min = threshold_delta(T=5, R=3, P=1)
    print(f"Minimum δ for grim trigger cooperation: {delta_min:.3f}")
    
    strategies = {
        'TFT': tit_for_tat, 'Grim': grim_trigger, 'AllC': always_cooperate,
        'AllD': always_defect, 'Pavlov': pavlov, 'Random': random_strategy
    }
    ranking = run_tournament(strategies, rounds=200)
    print("\nTournament Results:")
    for name, score in ranking:
        print(f"  {name:8s}: {score}")
    
    plot_payoff_evolution(tit_for_tat, grim_trigger, 'TFT', 'Grim')
    visualize_folk_theorem()
```

---

## Practice Problems (45 min)

**P1.** For PD with $T=5, R=3, S=0, P=1$, find the minimum $\delta$ for: (a) grim trigger, (b) TFT.

<details><summary>Answer</summary>

(a) Grim: $\delta \geq (5-3)/(5-1) = 1/2$. (b) TFT: defect gives $5 + \delta \cdot 0 + \delta^2 \cdot 5 + \ldots$ vs cooperate $3/(1-\delta)$. Deviation to D then back to C (opponent retaliates one round): $\delta \geq (5-3)/(5-0) = 2/5$. TFT requires **lower** $\delta$.
</details>

**P2.** In a finitely repeated PD (100 rounds), what does backward induction predict? Why is this empirically wrong?

<details><summary>Answer</summary>

Backward induction: defect in round 100 (last round, no future). Then round 99 has no incentive. Unravels to **defect every round**. Empirically wrong because humans use bounded rationality, reputation, and uncertainty about opponent's type.
</details>

**P3.** Design a strategy that beats TFT against AllD but loses to TFT against itself.

<details><summary>Answer</summary>

**Suspicious TFT (STFT):** Start with D, then copy opponent. Against AllD: STFT gets (1,1,1,...) same as TFT-vs-AllD since both defect. Against TFT: STFT starts D → TFT retaliates → alternating (D,C,D,C...) scoring lower than mutual cooperation. So STFT doesn't clearly beat TFT, illustrating TFT's robustness.
</details>

---

## Expert Challenges

**E1.** Prove the folk theorem for the PD: show that for any payoff vector $(v_1, v_2)$ with $v_i > 1$ in the feasible set, there exists a strategy profile and $\bar{\delta}$ such that for all $\delta > \bar{\delta}$, it is an SPE.

<details><summary>Hint</summary>

Construct a strategy that plays the target action profile on-path and reverts to mutual minimax for $L$ rounds upon deviation. Choose $L(\delta)$ to make deviation unprofitable.
</details>

**E2.** Implement Axelrod's ecological tournament: after each generation, strategies replicate proportional to their score. Plot population dynamics over 100 generations.

<details><summary>Hint</summary>

After round-robin, strategy $i$'s share = score$_i$ / total. Next generation's population uses these shares. Run tournament again with weighted encounters.
</details>

**E3.** Extend to imperfect public monitoring: with probability $\epsilon$, the observed action is flipped. How does this affect the minimum $\delta$ for cooperation?

<details><summary>Hint</summary>

With noise $\epsilon$, expected punishment is triggered even without defection. Grim trigger becomes too harsh. Use forgiving strategies and solve for modified $\delta$ threshold: $\delta \geq (T-R)/((T-P)(1-2\epsilon))$ approximately.
</details>

---

## Connections
- **Prerequisites:** Nash equilibrium (Day 93) — stage game NE; Backward induction (Day 102) — finite case unravels
- **Forward:** Mechanism design (Day 104) — repeated interaction enables enforcement; Week 15 review (Day 105)
- **OKS:** Corridor-sharing norms among robots, reputation-based priority at intersections, cooperative fleet behavior

## Self-Check
- [ ] I can compute the threshold $\delta$ for cooperation under grim trigger
- [ ] I understand the folk theorem and can describe the set of achievable payoffs
- [ ] I can implement and compare repeated-game strategies
- [ ] I know why finite repetition unravels cooperation (and when it doesn't)
