# Day 107: Learning in Games — Fictitious Play

> **Phase VII — Game Theory for Optimization & Robotics | Week 16 | 2.5 hours**
> *When computing equilibria is intractable, let agents learn their way there through repeated interaction.*

## Navigation
- **Previous:** [Day 106: Computing Nash Equilibria](day-106-computing-nash.md)
- **Next:** [Day 108: No-Regret Learning](day-108-no-regret.md)
- **Week:** [Week 16 Overview](../README.md)
- **Chapter:** [Chapter 10: Game Theory Reference](../../10-game-theory.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
OKS AMRs operate in a decentralized fleet where each robot makes local decisions about routing, task acceptance, and charging. Fictitious play models exactly this: each robot best-responds to the historical distribution of other robots' actions, without centralized coordination. Understanding when this converges (and when it cycles) determines whether decentralized fleet policies stabilize.

---

## Theory (45 min)

### 107.1 Fictitious Play Definition

In a repeated game, player $i$ at time $t$ maintains the **empirical frequency** of opponent $j$'s past actions:

$$\hat{\sigma}_j^t(a_j) = \frac{1}{t} \sum_{\tau=1}^{t} \mathbf{1}[a_j^\tau = a_j]$$

Player $i$ then plays a **best response** to $\hat{\sigma}_{-i}^t$:

$$a_i^{t+1} \in \arg\max_{a_i} \sum_{a_{-i}} u_i(a_i, a_{-i}) \hat{\sigma}_{-i}^t(a_{-i})$$

This is **fictitious play** (Brown, 1951): each player acts as if opponents play a fixed mixed strategy equal to their historical frequency.

### 107.2 Convergence Results

**Convergence** means the empirical frequencies $\hat{\sigma}^t$ converge to a Nash equilibrium.

| Game class | Convergence? |
|---|---|
| Zero-sum games | ✓ (Robinson 1951) |
| $2 \times 2$ games | ✓ |
| Potential games | ✓ (Monderer & Shapley 1996) |
| Supermodular games | ✓ |
| General $3 \times 3$ | ✗ — Shapley (1964) counter-example cycles |
| Generic $n \times n$ | Unknown in general |

**Shapley's counter-example**: The game with payoff matrices
$$A = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0 \end{pmatrix}, \quad B = \begin{pmatrix} 0 & 0 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \end{pmatrix}$$
produces cycling behavior — the empirical frequencies trace a limit cycle and never converge.

### 107.3 Rate of Convergence

When fictitious play converges, the rate is typically $O(1/t)$ for the empirical frequencies approaching NE. In zero-sum games, the **payoff convergence** satisfies:

$$\left| \max_x \min_y x^\top A y - \hat{x}^{t\top} A \hat{y}^t \right| = O\left(\frac{1}{\sqrt{t}}\right)$$

**Connection to online learning**: Fictitious play is a special case of "follow the leader" — play the best response to all past data. This connects to the regret minimization framework (Day 108).

---

## Implementation (60 min)

```python
"""Fictitious Play: simulation and convergence analysis."""
import numpy as np
import matplotlib.pyplot as plt

def fictitious_play(A: np.ndarray, B: np.ndarray, T: int = 1000,
                    seed: int = 42) -> dict:
    """Simulate fictitious play for a 2-player game.
    
    Args:
        A, B: payoff matrices (m x n)
        T: number of rounds
        seed: random seed for tie-breaking
    
    Returns:
        dict with action histories and empirical frequencies
    """
    rng = np.random.default_rng(seed)
    m, n = A.shape
    
    # Counts of each action
    count1 = np.zeros(m)
    count2 = np.zeros(n)
    
    history1, history2 = [], []
    freq1_history, freq2_history = [], []
    
    # Initial actions (random)
    a1, a2 = rng.integers(m), rng.integers(n)
    count1[a1] += 1
    count2[a2] += 1
    history1.append(a1)
    history2.append(a2)
    
    for t in range(1, T):
        # Empirical frequencies
        freq2 = count2 / count2.sum()
        freq1 = count1 / count1.sum()
        
        # Best response to empirical frequency
        payoffs1 = A @ freq2
        a1 = rng.choice(np.where(np.isclose(payoffs1, payoffs1.max()))[0])
        
        payoffs2 = B.T @ freq1
        a2 = rng.choice(np.where(np.isclose(payoffs2, payoffs2.max()))[0])
        
        count1[a1] += 1
        count2[a2] += 1
        history1.append(a1)
        history2.append(a2)
        
        freq1_history.append(count1 / count1.sum())
        freq2_history.append(count2 / count2.sum())
    
    return {
        'history1': history1, 'history2': history2,
        'freq1': np.array(freq1_history), 'freq2': np.array(freq2_history),
        'final_freq1': count1 / count1.sum(),
        'final_freq2': count2 / count2.sum(),
    }

def plot_convergence(result: dict, title: str = "Fictitious Play"):
    """Plot empirical frequency evolution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for i in range(result['freq1'].shape[1]):
        axes[0].plot(result['freq1'][:, i], label=f'Action {i}')
    axes[0].set_title(f'{title} — Player 1 frequencies')
    axes[0].set_xlabel('Round'); axes[0].legend()
    
    for j in range(result['freq2'].shape[1]):
        axes[1].plot(result['freq2'][:, j], label=f'Action {j}')
    axes[1].set_title(f'{title} — Player 2 frequencies')
    axes[1].set_xlabel('Round'); axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('fictitious_play_convergence.png', dpi=100)
    plt.show()

# Example 1: Matching Pennies (zero-sum) — should converge to (0.5, 0.5)
A_mp = np.array([[1, -1], [-1, 1]])
B_mp = -A_mp
result_mp = fictitious_play(A_mp, B_mp, T=5000)
print("Matching Pennies:")
print(f"  Final freq1: {result_mp['final_freq1']}")
print(f"  Final freq2: {result_mp['final_freq2']}")
print(f"  NE is (0.5, 0.5) for both — error: "
      f"{np.linalg.norm(result_mp['final_freq1'] - 0.5):.4f}")

# Example 2: Shapley's cycling game — should NOT converge
A_sh = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
B_sh = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
result_sh = fictitious_play(A_sh, B_sh, T=5000)
print("\nShapley's game:")
print(f"  Final freq1: {np.round(result_sh['final_freq1'], 4)}")
print(f"  Final freq2: {np.round(result_sh['final_freq2'], 4)}")
print(f"  NE is (1/3, 1/3, 1/3) — error: "
      f"{np.linalg.norm(result_sh['final_freq1'] - 1/3):.4f}")

# Example 3: Coordination game (potential game) — should converge
A_coord = np.array([[2, 0], [0, 1]])
B_coord = np.array([[2, 0], [0, 1]])
result_coord = fictitious_play(A_coord, B_coord, T=2000)
print("\nCoordination game:")
print(f"  Final freq1: {result_coord['final_freq1']}")
print(f"  Final freq2: {result_coord['final_freq2']}")

plot_convergence(result_mp, "Matching Pennies")
```

---

## Practice Problems (45 min)

**P1**: Run fictitious play on the Prisoner's Dilemma. Does it converge? To which NE? Explain why.

<details><summary>Answer</summary>

PD has a dominant strategy (Defect, Defect). After the first round, both players' best responses are always Defect regardless of empirical frequency. So $\hat{\sigma}^t \to (D, D)$ immediately. FP converges trivially because the dominant strategy NE is reached in one step.
</details>

**P2**: For Shapley's cycling game, plot the trajectory of $(\hat{\sigma}_1^t, \hat{\sigma}_2^t)$ in the simplex. Describe the cycling pattern.

<details><summary>Answer</summary>

The trajectory spirals outward from the center $(1/3, 1/3, 1/3)$ in a roughly hexagonal path. Player 1 cycles Rock→Paper→Scissors and player 2 cycles with a phase offset. The frequencies oscillate with slowly decaying amplitude, but the instantaneous actions keep cycling. In the limit, the time-average approaches $(1/3, 1/3, 1/3)$ but the path never settles.
</details>

**P3**: Prove that in a $2 \times 2$ zero-sum game, the fictitious play value $v_t = \hat{x}^{t\top} A \hat{y}^t$ satisfies $|v_t - v^*| = O(1/\sqrt{t})$.

<details><summary>Answer</summary>

At each step, player $i$'s best response guarantees payoff $\geq v^*$ against $\hat{\sigma}_{-i}^t$. The empirical payoff accumulates: $\sum_\tau u_i(a_i^\tau, \hat{\sigma}_{-i}^\tau) \geq t \cdot v^*$. The deviation from exact best response at each step contributes $O(1)$, giving total regret $O(\sqrt{t})$ by standard arguments. Dividing by $t$ yields $O(1/\sqrt{t})$ convergence rate for the average payoff.
</details>

---

## Expert Challenges

**E1**: Implement **smooth fictitious play** where players best-respond with a logit (softmax) perturbation: $\sigma_i^{t+1}(a_i) \propto \exp(\beta \cdot u_i(a_i, \hat{\sigma}_{-i}^t))$. Show this converges in Shapley's game for large enough $\beta$.

<details><summary>Hint</summary>

Smooth FP converges in all games for finite $\beta$ (Fudenberg & Levine 1998). The smoothing breaks the cycling by adding entropy. Implement with temperature parameter $1/\beta$ and show the trajectory spirals inward.
</details>

**E2**: Prove that fictitious play converges in all potential games. What property of the potential function drives convergence?

<details><summary>Answer</summary>

In a potential game with potential $\Phi$, each best-response step increases $\Phi$ (or keeps it equal). Since $\Phi$ is bounded and the strategy space is finite, the sequence of action profiles must eventually cycle through a set of profiles. The empirical frequencies then converge because the cycling set forms a "sink" in the best-response dynamics, which corresponds to NE in potential games by the finite improvement property.
</details>

**E3**: Design a 2-robot corridor negotiation scenario. Model it as a repeated game and simulate FP. Does the fleet stabilize?

<details><summary>Answer</summary>

Model: 2 robots, 2 corridors. If both choose the same corridor, they incur delay cost 3. Different corridors give cost 1 each. This is an anti-coordination game: $A = B = \begin{pmatrix} -3 & -1 \\ -1 & -3 \end{pmatrix}$. FP converges to the mixed NE $(0.5, 0.5)$ — each robot randomizes equally. In practice this means persistent oscillation in corridor choice, motivating explicit coordination protocols.
</details>

---

## Connections
- **Back**: [Day 106: Computing NE](day-106-computing-nash.md) — FP avoids explicit NE computation
- **Forward**: [Day 108: No-Regret Learning](day-108-no-regret.md) — FP is "follow the leader"; no-regret improves the guarantee
- **OKS**: Decentralized fleet learning — robots adjusting routing based on historical fleet behavior

## Self-Check
- [ ] I can simulate fictitious play and interpret the empirical frequency trajectory
- [ ] I know which game classes guarantee FP convergence and which don't
- [ ] I understand the connection between FP and best-response dynamics
- [ ] I can explain Shapley's counter-example and why cycling occurs
