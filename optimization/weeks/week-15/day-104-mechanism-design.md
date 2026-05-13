# Day 104: Mechanism Design and Auctions

> **Phase VII — Game Theory for Optimization & Robotics | Week 15 | 2.5 hours**
> *Don't just play the game — design the rules so that self-interested agents produce the outcome you want.*

## Navigation
- **Previous:** [Day 103: Repeated Games and Folk Theorem](day-103-repeated-games.md)
- **Next:** [Day 105: Week 15 Review](day-105-week15-review.md)
- **Week:** [Week 15 Overview](../README.md)
- **Chapter:** [Chapter 10: Game Theory Reference](../../10-game-theory.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
The fleet controller must allocate tasks to robots — each robot privately knows its battery level, distance to task, and current load. Mechanism design provides auction-based allocation where truthful reporting is the dominant strategy. The VCG mechanism ensures efficient task assignment even when robots have private information.

---

## Theory (45 min)

### 104.1 The Mechanism Design Problem

**Setting:** A principal designs a game (mechanism) for $n$ agents with private types $\theta_i \in \Theta_i$.

A **mechanism** $(M, g)$ consists of:
- Message spaces $M_i$ for each agent.
- An outcome function $g: M_1 \times \cdots \times M_n \to X$ mapping messages to outcomes.

The **revelation principle** says: for any mechanism and equilibrium, there exists a **direct** mechanism where agents report types and truth-telling is an equilibrium, achieving the same outcome.

### 104.2 Incentive Compatibility

A direct mechanism is:
- **Dominant Strategy Incentive Compatible (DSIC):** Truth-telling is a dominant strategy:
$$u_i(\theta_i, g(\theta_i, \theta_{-i})) \geq u_i(\theta_i, g(\theta_i', \theta_{-i})) \quad \forall \theta_i, \theta_i', \theta_{-i}$$

- **Bayesian Incentive Compatible (BIC):** Truth-telling maximizes expected utility given beliefs.

**Individual Rationality (IR):** Each agent gets non-negative utility from participation.

### 104.3 The VCG Mechanism

The **Vickrey-Clarke-Groves** mechanism achieves efficient outcomes with truthful reporting.

- **Social choice:** $x^*(\theta) = \arg\max_{x \in X} \sum_i v_i(x, \theta_i)$
- **Payments:** $p_i(\theta) = -\sum_{j \neq i} v_j(x^*(\theta), \theta_j) + h_i(\theta_{-i})$

The standard choice $h_i(\theta_{-i}) = \max_{x} \sum_{j \neq i} v_j(x, \theta_j)$ gives the **Clarke pivot rule:**

$$p_i = \underbrace{\max_x \sum_{j \neq i} v_j(x, \theta_j)}_{\text{welfare without } i} - \underbrace{\sum_{j \neq i} v_j(x^*(\theta), \theta_j)}_{\text{others' welfare with } i}$$

Agent $i$ pays the **externality** they impose on others.

### 104.4 Auction Theory

| Auction | Rule | Equilibrium (IPV) |
|---------|------|-------------------|
| **First-price sealed** | Pay your bid | Shade bid: $b_i = \frac{n-1}{n}\theta_i$ |
| **Second-price (Vickrey)** | Pay 2nd highest | Bid truthfully: $b_i = \theta_i$ |
| **English (ascending)** | Open ascending | Stay until $\theta_i$ |
| **Dutch (descending)** | First to stop wins | Equivalent to first-price |

**Revenue Equivalence Theorem (Myerson 1981):** Under independent private values, symmetric equilibrium, and risk-neutral bidders, all standard auctions yield the same expected revenue:

$$E[\text{Revenue}] = E[\theta^{(2:n)}] = \frac{n-1}{n+1} \cdot \bar{\theta}$$

for uniform $\theta_i \sim U[0, \bar{\theta}]$, where $\theta^{(2:n)}$ is the second-highest valuation.

---

## Implementation (60 min)

```python
"""Mechanism design and auctions — Day 104"""
import numpy as np
from itertools import combinations


def vickrey_auction(bids):
    """Second-price sealed-bid auction.
    
    Returns: (winner, payment)
    """
    sorted_idx = np.argsort(bids)[::-1]
    winner = sorted_idx[0]
    payment = bids[sorted_idx[1]]  # second-highest bid
    return winner, payment


def first_price_auction(valuations, n_simulations=10000):
    """Simulate first-price auction with equilibrium bidding.
    
    In symmetric IPV with uniform [0,1], equilibrium bid = (n-1)/n * valuation.
    """
    n = len(valuations)
    # Equilibrium bids
    bids = [(n - 1) / n * v for v in valuations]
    winner = np.argmax(bids)
    payment = bids[winner]
    return winner, payment, bids


def vcg_mechanism(valuations, items):
    """VCG mechanism for multi-item allocation.
    
    Args:
        valuations: dict of agent -> {item: value}
        items: list of items to allocate
    Returns:
        allocation, payments
    """
    agents = list(valuations.keys())
    
    # Find efficient allocation (maximize total welfare)
    best_alloc = None
    best_welfare = -np.inf
    
    # Simple: each item to one agent (no bundles)
    from itertools import product as iproduct
    for assignment in iproduct(agents, repeat=len(items)):
        welfare = sum(valuations[assignment[j]].get(items[j], 0) for j in range(len(items)))
        if welfare > best_welfare:
            best_welfare = welfare
            best_alloc = dict(zip(items, assignment))
    
    # Compute VCG payments (Clarke pivot)
    payments = {}
    for i in agents:
        # Welfare of others with i
        others_welfare_with = sum(
            valuations[best_alloc[item]].get(item, 0)
            for item in items if best_alloc[item] != i
        )
        # Welfare of others without i (re-optimize excluding i)
        other_agents = [a for a in agents if a != i]
        best_without = -np.inf
        for assignment in iproduct(other_agents, repeat=len(items)):
            w = sum(valuations[assignment[j]].get(items[j], 0) for j in range(len(items)))
            if w > best_without:
                best_without = w
        payments[i] = best_without - others_welfare_with
    
    return best_alloc, payments


def simulate_revenue_equivalence(n_bidders=5, n_trials=50000):
    """Empirically verify revenue equivalence theorem."""
    rev_first, rev_second = [], []
    
    for _ in range(n_trials):
        vals = np.random.uniform(0, 1, n_bidders)
        
        # Vickrey (second-price): bid truthfully
        _, pay_v = vickrey_auction(vals)
        rev_second.append(pay_v)
        
        # First-price: equilibrium shading
        bids_fp = (n_bidders - 1) / n_bidders * vals
        winner_fp = np.argmax(bids_fp)
        rev_first.append(bids_fp[winner_fp])
    
    theoretical = (n_bidders - 1) / (n_bidders + 1)
    return {
        'first_price': np.mean(rev_first),
        'second_price': np.mean(rev_second),
        'theoretical': theoretical
    }


def prove_vickrey_ic():
    """Demonstrate incentive compatibility of Vickrey auction."""
    true_value = 8.0
    second_highest = 6.0  # fixed for demo
    
    print("=== Vickrey IC Demonstration ===")
    print(f"True value: {true_value}, Second-highest bid: {second_highest}\n")
    
    for bid in [4, 6, 7, 8, 9, 10, 12]:
        if bid > second_highest:
            won = True
            utility = true_value - second_highest  # pay second-highest
        else:
            won = False
            utility = 0
        print(f"  Bid {bid:2d}: {'WIN' if won else 'LOSE'}, utility = {utility:.1f}")
    
    print(f"\nOptimal bid = {true_value} (truthful): no benefit from over/under-bidding")


# --- Demo ---
if __name__ == "__main__":
    # Vickrey auction
    bids = [10, 7, 8, 3, 9]
    winner, payment = vickrey_auction(bids)
    print(f"Vickrey: winner=Agent {winner} (bid={bids[winner]}), pays={payment}")
    
    # VCG for task allocation
    valuations = {
        'R1': {'task_A': 10, 'task_B': 5},
        'R2': {'task_A': 8, 'task_B': 7},
        'R3': {'task_A': 3, 'task_B': 9},
    }
    alloc, payments = vcg_mechanism(valuations, ['task_A', 'task_B'])
    print(f"\nVCG allocation: {alloc}")
    print(f"VCG payments: {payments}")
    
    # Revenue equivalence
    rev = simulate_revenue_equivalence(n_bidders=5)
    print(f"\nRevenue equivalence (5 bidders, U[0,1]):")
    print(f"  First-price:  {rev['first_price']:.4f}")
    print(f"  Second-price: {rev['second_price']:.4f}")
    print(f"  Theoretical:  {rev['theoretical']:.4f}")
    
    prove_vickrey_ic()
```

---

## Practice Problems (45 min)

**P1.** In a Vickrey auction with bids [12, 9, 15, 7], who wins and what do they pay? Why is bidding 15 optimal for the winner if their true value is 15?

<details><summary>Answer</summary>

Winner: bidder 3 (bid=15), pays 12 (second-highest). Utility = 15-12 = 3. If they bid less (say 11), they lose to bidder 1 → utility 0. If they bid more (say 20), still pay 12 → same utility. **Truthful bidding is weakly dominant.**
</details>

**P2.** Compute VCG payments: 2 agents, 1 item. Agent A values it at 10, Agent B at 7.

<details><summary>Answer</summary>

Efficient allocation: give to A. Without A, B gets the item (welfare = 7). With A present, B gets 0. Payment for A = 7 - 0 = 7. Without B, A gets item (welfare = 10). With B, A still gets it (welfare for B = 0). Payment for B = 0. **A pays 7** (= B's value, same as Vickrey).
</details>

**P3.** In a first-price auction with 3 bidders and values drawn from $U[0, 100]$, what is the equilibrium bid for a bidder with value 60?

<details><summary>Answer</summary>

Equilibrium bid = $\frac{n-1}{n} \cdot v = \frac{2}{3} \cdot 60 = 40$. Shade by $1/n$ of your value.
</details>

---

## Expert Challenges

**E1.** Implement Myerson's optimal auction for 2 bidders with $\theta_i \sim U[0,1]$. Show it sets a reserve price of $r = 1/2$ and compute expected revenue.

<details><summary>Hint</summary>

Virtual valuation: $\psi(\theta) = \theta - (1-F(\theta))/f(\theta) = 2\theta - 1$. Allocate when $\psi > 0$, i.e., $\theta > 1/2$. Expected revenue = $5/12 > 1/3$ (Vickrey without reserve).
</details>

**E2.** Design a mechanism for allocating 3 tasks to 3 robots where each robot can handle at most 1 task. Prove your mechanism is DSIC and compute payments.

<details><summary>Hint</summary>

This is a combinatorial VCG with assignment constraints. Use the Hungarian algorithm for efficient allocation, then compute Clarke pivot payments.
</details>

**E3.** Show that no mechanism can simultaneously satisfy DSIC, efficiency, individual rationality, and budget balance (Green-Laffont impossibility).

<details><summary>Hint</summary>

VCG achieves the first three but the principal may run a deficit ($\sum p_i \leq 0$ is possible). Budget balance requires $\sum p_i = 0$. Construct a 2-agent example where these conflict.
</details>

---

## Connections
- **Prerequisites:** Cooperative games (Day 99) — VCG uses Shapley-like marginal contributions; Dynamic games (Day 102) — sequential mechanisms
- **Forward:** Week 15 review (Day 105) — combine all tools; application to fleet scheduling
- **OKS:** Auction-based task allocation, truthful battery-level reporting, VCG for multi-robot warehouse optimization

## Self-Check
- [ ] I can state the revelation principle and explain its importance
- [ ] I understand why the Vickrey auction is incentive-compatible
- [ ] I can compute VCG payments using the Clarke pivot rule
- [ ] I know the revenue equivalence theorem and its conditions
