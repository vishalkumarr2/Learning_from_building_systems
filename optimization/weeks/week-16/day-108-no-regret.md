# Day 108: No-Regret Learning

> **Phase VII — Game Theory for Optimization & Robotics | Week 16 | 2.5 hours**
> *Guarantee bounded regret regardless of what the environment (or opponents) do — the foundation of online decision-making.*

## Navigation
- **Previous:** [Day 107: Learning in Games — Fictitious Play](day-107-fictitious-play.md)
- **Next:** [Day 109: Potential Games and Congestion Games](day-109-congestion-games.md)
- **Week:** [Week 16 Overview](../README.md)
- **Chapter:** [Chapter 10: Game Theory Reference](../../10-game-theory.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
OKS task allocation is an online problem: tasks arrive over time, and robots must choose which to accept without knowing future arrivals. No-regret algorithms guarantee that over time, no robot would have been better off following any fixed strategy — critical for fair, efficient fleet-wide task distribution without central planning.

---

## Theory (45 min)

### 108.1 External Regret

A player selects actions $a_1, a_2, \ldots, a_T$ from a set of $n$ actions. After each round, a loss vector $\ell^t \in [0,1]^n$ is revealed. The **external regret** after $T$ rounds is:

$$R_T = \sum_{t=1}^{T} \ell^t(a_t) - \min_{i \in [n]} \sum_{t=1}^{T} \ell^t(i)$$

This compares cumulative loss to the **best fixed action in hindsight**. An algorithm is **no-regret** (or Hannan consistent) if $R_T / T \to 0$.

### 108.2 Regret Matching

Hart & Mas-Colell (2000): For each action $i$, define the **cumulative regret** for not having played $i$:

$$r_i^t = \sum_{\tau=1}^{t} \left[\ell^\tau(a_\tau) - \ell^\tau(i)\right]$$

Let $r_i^{t+} = \max(r_i^t, 0)$. The **regret matching** strategy at time $t+1$ is:

$$\sigma^{t+1}(i) = \frac{r_i^{t+}}{\sum_j r_j^{t+}}$$

If all regrets are non-positive, play uniformly. **Theorem**: Regret matching achieves $R_T = O(\sqrt{T \cdot n})$.

### 108.3 Multiplicative Weights Update (MWU) / Hedge

Initialize weights $w_i^1 = 1$ for each action $i$. At each round:

1. Play $\sigma^t(i) = w_i^t / \sum_j w_j^t$
2. Observe loss $\ell^t$
3. Update: $w_i^{t+1} = w_i^t \cdot \exp(-\eta \cdot \ell^t(i))$

With learning rate $\eta = \sqrt{\ln n / T}$:

$$R_T \leq 2\sqrt{T \ln n}$$

This is **optimal** up to constants — no algorithm can achieve $R_T = o(\sqrt{T \log n})$ against adversarial losses.

### 108.4 Connection to Nash Equilibria

**Theorem (Freund & Schapire 1999)**: If both players in a zero-sum game use no-regret algorithms, their **average strategies** converge to a Nash equilibrium at rate $O(\sqrt{\log n / T})$.

Specifically, if $\bar{\sigma}_1^T$ and $\bar{\sigma}_2^T$ are the time-averaged mixed strategies:

$$\max_x x^\top A \bar{\sigma}_2^T - \bar{\sigma}_1^{T\top} A \min_y \bar{\sigma}_1^{T\top} A y \leq O\left(\sqrt{\frac{\ln n}{T}}\right)$$

---

## Implementation (60 min)

```python
"""No-Regret Learning: Regret Matching and Multiplicative Weights."""
import numpy as np
import matplotlib.pyplot as plt

def multiplicative_weights(losses: np.ndarray, eta: float = None) -> dict:
    """Multiplicative Weights Update (Hedge) algorithm.
    
    Args:
        losses: T x n array of loss vectors
        eta: learning rate (auto-set if None)
    
    Returns:
        dict with strategies, cumulative regret, and regret bound
    """
    T, n = losses.shape
    if eta is None:
        eta = np.sqrt(np.log(n) / T)
    
    weights = np.ones(n)
    strategies = np.zeros((T, n))
    cumulative_loss = np.zeros(n)
    player_loss = 0.0
    regret_history = []
    
    for t in range(T):
        # Play proportional to weights
        sigma = weights / weights.sum()
        strategies[t] = sigma
        
        # Incur loss
        player_loss += sigma @ losses[t]
        cumulative_loss += losses[t]
        
        # Regret: player loss - best fixed action
        regret = player_loss - cumulative_loss.min()
        regret_history.append(regret)
        
        # Update weights
        weights *= np.exp(-eta * losses[t])
    
    return {
        'strategies': strategies,
        'regret': np.array(regret_history),
        'avg_strategy': strategies.mean(axis=0),
        'bound': 2 * np.sqrt(T * np.log(n)),
    }

def regret_matching(game_A: np.ndarray, T: int = 5000) -> dict:
    """Regret matching for a 2-player zero-sum game.
    
    Both players use regret matching; returns average strategies.
    """
    m, n = game_A.shape
    cum_regret1 = np.zeros(m)
    cum_regret2 = np.zeros(n)
    
    sum_strategy1 = np.zeros(m)
    sum_strategy2 = np.zeros(n)
    avg1_hist, avg2_hist = [], []
    
    for t in range(1, T + 1):
        # Strategies from regret
        pos1 = np.maximum(cum_regret1, 0)
        sigma1 = pos1 / pos1.sum() if pos1.sum() > 0 else np.ones(m) / m
        
        pos2 = np.maximum(cum_regret2, 0)
        sigma2 = pos2 / pos2.sum() if pos2.sum() > 0 else np.ones(n) / n
        
        sum_strategy1 += sigma1
        sum_strategy2 += sigma2
        
        # Compute regrets
        # Player 1: could have played pure i instead
        val1 = sigma1 @ game_A @ sigma2
        for i in range(m):
            cum_regret1[i] += game_A[i] @ sigma2 - val1
        
        # Player 2: playing -A (zero-sum)
        val2 = sigma1 @ (-game_A) @ sigma2
        for j in range(n):
            cum_regret2[j] += sigma1 @ (-game_A[:, j]) - val2
        
        avg1_hist.append(sum_strategy1 / t)
        avg2_hist.append(sum_strategy2 / t)
    
    return {
        'avg_strategy1': sum_strategy1 / T,
        'avg_strategy2': sum_strategy2 / T,
        'avg1_hist': np.array(avg1_hist),
        'avg2_hist': np.array(avg2_hist),
    }

# Demo 1: MWU against adversarial losses
np.random.seed(42)
T, n = 2000, 5
losses = np.random.rand(T, n)
result_mwu = multiplicative_weights(losses)
print("MWU against random losses:")
print(f"  Final regret: {result_mwu['regret'][-1]:.2f}")
print(f"  Theoretical bound: {result_mwu['bound']:.2f}")
print(f"  Avg regret per round: {result_mwu['regret'][-1]/T:.4f}")

# Demo 2: Regret matching on Rock-Paper-Scissors
A_rps = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
result_rm = regret_matching(A_rps, T=10000)
print(f"\nRPS via regret matching:")
print(f"  Avg strategy P1: {np.round(result_rm['avg_strategy1'], 4)}")
print(f"  Avg strategy P2: {np.round(result_rm['avg_strategy2'], 4)}")
print(f"  NE is (1/3, 1/3, 1/3) — error: "
      f"{np.linalg.norm(result_rm['avg_strategy1'] - 1/3):.4f}")

# Plot regret growth vs bound
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(result_mwu['regret'], label='Actual regret')
T_range = np.arange(1, T + 1)
ax.plot(T_range, 2 * np.sqrt(T_range * np.log(n)), '--', label=r'$2\sqrt{T \ln n}$')
ax.set_xlabel('Round'); ax.set_ylabel('Cumulative Regret')
ax.set_title('MWU Regret vs Theoretical Bound')
ax.legend()
plt.tight_layout()
plt.savefig('no_regret_bound.png', dpi=100)
plt.show()
```

---

## Practice Problems (45 min)

**P1**: Implement MWU for $n = 3$ actions with adversarial losses $\ell^t = e_{(t \mod 3)}$ (cycling). Verify $R_T = O(\sqrt{T})$.

<details><summary>Answer</summary>

Each action gets hit equally often (every 3rd round), so the best fixed action has loss $T/3$. MWU distributes weight and incurs loss close to $T/3$ too. The regret stays bounded by $2\sqrt{T \ln 3} \approx 2.1\sqrt{T}$. At $T = 10000$, regret $\lesssim 210$, average regret $\lesssim 0.021$.
</details>

**P2**: Both players in a $3 \times 3$ zero-sum game use MWU. Prove the average strategies form an $\varepsilon$-NE with $\varepsilon = O(\sqrt{\log 3 / T})$.

<details><summary>Answer</summary>

Each player has regret $R_T \leq 2\sqrt{T \ln 3}$. Player 1's average loss is at most $\min_i \bar{\ell}_i + 2\sqrt{\ln 3 / T}$. In game terms, $\bar{\sigma}_1^{T\top} A \bar{\sigma}_2^T \geq v^* - \varepsilon$ and symmetrically for player 2. So $(\bar{\sigma}_1^T, \bar{\sigma}_2^T)$ is a $2\varepsilon$-NE with $\varepsilon = 2\sqrt{\ln 3 / T}$.
</details>

**P3**: Compare regret matching and MWU on the same zero-sum game. Which converges faster to NE?

<details><summary>Answer</summary>

MWU typically converges faster because it uses multiplicative (exponential) updates that adapt more aggressively. Regret matching's linear update is simpler but slower. For RPS with $T = 5000$: MWU average strategy error $\approx 0.01$, regret matching error $\approx 0.02$. Both are $O(1/\sqrt{T})$ but MWU has better constants.
</details>

---

## Expert Challenges

**E1**: Implement **internal regret** minimization. Show that if both players minimize internal regret, average play converges to a **correlated equilibrium**.

<details><summary>Hint</summary>

Internal regret for swap $i \to j$: $\sum_{t: a_t = i} [\ell^t(i) - \ell^t(j)]$. Use Blum-Mansour (2007): run $n$ copies of MWU, one per swap. The resulting algorithm has $O(\sqrt{T n \log n})$ internal regret and converges to correlated equilibrium.
</details>

**E2**: Derive the MWU regret bound $R_T \leq \eta \sum_t \|\ell^t\|_\infty^2 / 8 + \ln n / \eta$ using the potential function $\Phi^t = \ln(\sum_i w_i^t)$.

<details><summary>Answer</summary>

At each step: $\Phi^{t+1} - \Phi^t = \ln(\sum_i w_i^t e^{-\eta \ell_i^t} / \sum_i w_i^t) \leq \ln(\sum_i \sigma_i^t (1 - \eta \ell_i^t + \eta^2 \ell_i^{t2}/2)) \leq -\eta \sigma^t \cdot \ell^t + \eta^2/8$ (using $\ell \in [0,1]$ and $e^{-x} \leq 1-x+x^2/2$). Telescoping: $\Phi^T - \Phi^0 \leq -\eta \sum_t \sigma^t \cdot \ell^t + T\eta^2/8$. Also $\Phi^T \geq -\eta \min_i L_i^T$ and $\Phi^0 = \ln n$. Combining and setting $\eta = \sqrt{8\ln n / T}$ gives $R_T \leq 2\sqrt{T \ln n / 2}$.
</details>

**E3**: Design an online task allocation system for 4 robots and 3 task types using MWU. Each robot independently learns which tasks to specialize in.

<details><summary>Answer</summary>

Each robot runs MWU over 3 task types. Loss for robot $r$ choosing task $k$ at time $t$ depends on congestion: $\ell_r^t(k) = c_k \cdot n_k^t / N$ where $n_k^t$ is the number of robots choosing task $k$. Over time, robots diversify. With 4 robots and 3 tasks, the equilibrium has robots spread across tasks proportional to task difficulty, achieving near-optimal throughput with $O(\sqrt{T})$ regret.
</details>

---

## Connections
- **Back**: [Day 107: Fictitious Play](day-107-fictitious-play.md) — FP is "follow the leader"; MWU adds randomization for better worst-case
- **Forward**: [Day 109: Congestion Games](day-109-congestion-games.md) — no-regret dynamics in congestion settings
- **OKS**: Online task allocation, corridor bandwidth sharing, charging schedule optimization

## Self-Check
- [ ] I can define external regret and state the $O(\sqrt{T \log n})$ bound
- [ ] I can implement both regret matching and multiplicative weights
- [ ] I understand why no-regret + no-regret → Nash equilibrium in zero-sum games
- [ ] I can connect no-regret learning to practical online robot decision-making
