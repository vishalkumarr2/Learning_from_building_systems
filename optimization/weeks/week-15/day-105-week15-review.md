# Day 105: Week 15 Review — Cooperative, Dynamic & Mechanism Design

> **Phase VII — Game Theory for Optimization & Robotics | Week 15 | 2.5 hours**
> *From coalition stability to auction design — a complete toolkit for strategic multi-agent systems.*

## Navigation
- **Previous:** [Day 104: Mechanism Design and Auctions](day-104-mechanism-design.md)
- **Next:** [Day 106: Computing Nash Equilibria](../week-16/day-106-computing-nash.md)
- **Week:** [Week 15 Overview](../README.md)
- **Chapter:** [Chapter 10: Game Theory Reference](../../10-game-theory.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
This week's tools complete the game-theoretic toolkit for fleet robotics: cooperative games model task coalitions (Day 99), Shapley ensures fair credit (Day 100), bargaining resolves bilateral disputes (Day 101), dynamic games handle sequential decisions (Day 102), repeated games enable cooperative norms (Day 103), and mechanism design creates truthful allocation protocols (Day 104). The review integrates these into a unified BEC scheduling model.

---

## Theory (45 min)

### 105.1 Concept Map

```
Cooperative Games (Day 99)
  ├── Characteristic function v(S)
  ├── Core (stable allocations) ←→ LP feasibility
  └── Bondareva-Shapley (nonempty core ⟺ balanced)
         │
Shapley Value (Day 100)
  ├── 4 axioms → unique fair allocation
  ├── φᵢ = avg marginal contribution
  └── Always in core for convex games
         │
Bargaining (Day 101)
  ├── Nash: max product of surpluses
  ├── Kalai-Smorodinsky: proportional to utopia
  └── Rubinstein: patience = power
         │
Dynamic Games (Day 102)
  ├── Extensive form & game trees
  ├── Backward induction → SPE
  └── Eliminates non-credible threats
         │
Repeated Games (Day 103)
  ├── Grim trigger, TFT, Pavlov
  ├── Folk theorem: any IR payoff for δ→1
  └── Finite: unraveling (backward induction)
         │
Mechanism Design (Day 104)
  ├── Revelation principle
  ├── VCG: efficient + truthful
  ├── Vickrey auction: DSIC
  └── Revenue equivalence theorem
```

### 105.2 Key Formulas Summary

| Concept | Formula |
|---------|---------|
| Core | $\{x : \sum x_i = v(N), \sum_{i \in S} x_i \geq v(S) \; \forall S\}$ |
| Shapley value | $\phi_i = \sum_{S} \frac{\|S\|!(n-\|S\|-1)!}{n!} [v(S \cup \{i\}) - v(S)]$ |
| Nash bargaining | $\arg\max_{u \geq d} (u_1 - d_1)(u_2 - d_2)$ |
| Rubinstein split | $x_1^* = \frac{1-\delta_2}{1-\delta_1\delta_2}$ |
| Grim trigger $\delta$ | $\delta \geq \frac{T-R}{T-P}$ |
| VCG payment | $p_i = \max_x \sum_{j \neq i} v_j(x) - \sum_{j \neq i} v_j(x^*)$ |
| First-price bid | $b_i = \frac{n-1}{n} \theta_i$ (uniform IPV) |
| Revenue equivalence | $E[\text{Rev}] = E[\theta^{(2:n)}]$ |

### 105.3 When to Use What

| Situation | Tool | Key insight |
|-----------|------|-------------|
| Stable team formation | Core (Day 99) | No subset wants to deviate |
| Fair reward splitting | Shapley (Day 100) | Average marginal contribution |
| Two-party negotiation | NBS/KS (Day 101) | Axiom-driven fair split |
| Sequential decisions | SPE (Day 102) | Only credible threats matter |
| Long-term interaction | Folk theorem (Day 103) | Future rewards enable cooperation |
| Private info allocation | VCG (Day 104) | Truthful + efficient |

---

## Implementation (60 min)

```python
"""Week 15 unified game theory toolkit — Day 105"""
import numpy as np
from itertools import combinations, permutations, product as iproduct
from math import factorial
from scipy.optimize import linprog


class CooperativeGame:
    """Complete cooperative game analysis toolkit."""
    
    def __init__(self, N, v_dict):
        self.N = sorted(N)
        self.n = len(self.N)
        self.v_dict = v_dict
    
    def v(self, S):
        return self.v_dict.get(frozenset(S), 0)
    
    def shapley_value(self):
        phi = {}
        for i in self.N:
            phi[i] = 0.0
            others = [p for p in self.N if p != i]
            for size in range(self.n):
                for S in combinations(others, size):
                    S = frozenset(S)
                    weight = factorial(len(S)) * factorial(self.n - len(S) - 1) / factorial(self.n)
                    phi[i] += weight * (self.v(S | {i}) - self.v(S))
        return phi
    
    def is_in_core(self, x):
        if abs(sum(x.values()) - self.v(frozenset(self.N))) > 1e-9:
            return False
        for r in range(1, self.n):
            for S in combinations(self.N, r):
                if sum(x[i] for i in S) < self.v(frozenset(S)) - 1e-9:
                    return False
        return True
    
    def core_allocation_lp(self):
        n = self.n
        c = np.zeros(n); c[0] = 1
        A_ub, b_ub = [], []
        for r in range(1, n):
            for S in combinations(self.N, r):
                row = np.zeros(n)
                for p in S:
                    row[self.N.index(p)] = -1
                A_ub.append(row)
                b_ub.append(-self.v(frozenset(S)))
        A_eq = np.ones((1, n))
        b_eq = [self.v(frozenset(self.N))]
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=[(None, None)] * n, method='highs')
        if res.success:
            return {self.N[i]: res.x[i] for i in range(n)}
        return None


def backward_induction(node):
    """Solve game tree by backward induction."""
    if node.get('payoffs') is not None:
        return node['payoffs'], {}
    best_pay, best_act, strats = None, None, {}
    player = node['player']
    for action, child in node['actions'].items():
        pay, child_strats = backward_induction(child)
        if best_pay is None or pay[player] > best_pay[player]:
            best_pay, best_act = pay, action
            strats = dict(child_strats)
    strats[node.get('name', '')] = best_act
    return best_pay, strats


def vickrey_auction(bids):
    idx = np.argsort(bids)[::-1]
    return idx[0], bids[idx[1]]


def vcg_payments(valuations, allocation):
    """Compute VCG payments for a given allocation."""
    agents = list(valuations.keys())
    items = list(allocation.keys())
    payments = {}
    for i in agents:
        others = [a for a in agents if a != i]
        welfare_others_with = sum(
            valuations[allocation[item]].get(item, 0)
            for item in items if allocation[item] != i)
        best_without = -np.inf
        for assign in iproduct(others, repeat=len(items)):
            w = sum(valuations[assign[j]].get(items[j], 0) for j in range(len(items)))
            best_without = max(best_without, w)
        payments[i] = best_without - welfare_others_with
    return payments


def bec_scheduling_demo():
    """Model BEC battery-swap scheduling as a mechanism design problem."""
    print("=== BEC Scheduling as Mechanism Design ===\n")
    
    # Robots report urgency (private: true battery remaining)
    robots = {'R1': {'urgency': 9, 'true_battery': 5},
              'R2': {'urgency': 7, 'true_battery': 15},
              'R3': {'urgency': 8, 'true_battery': 10}}
    
    # VCG: allocate slot to highest-urgency robot
    bids = {r: robots[r]['urgency'] for r in robots}
    sorted_r = sorted(bids, key=bids.get, reverse=True)
    winner = sorted_r[0]
    payment = bids[sorted_r[1]]
    
    print(f"Bids (urgency): {bids}")
    print(f"Winner: {winner} (pays {payment} priority units)")
    print(f"Truthful because: Vickrey mechanism — overbidding risks overpaying,")
    print(f"  underbidding risks losing the slot when you need it most.\n")
    
    # Cooperative game: coalition of robots sharing BEC slots
    v_dict = {
        frozenset(): 0,
        frozenset(['R1']): 3, frozenset(['R2']): 2, frozenset(['R3']): 2,
        frozenset(['R1', 'R2']): 7, frozenset(['R1', 'R3']): 6,
        frozenset(['R2', 'R3']): 5, frozenset(['R1', 'R2', 'R3']): 10
    }
    game = CooperativeGame({'R1', 'R2', 'R3'}, v_dict)
    shapley = game.shapley_value()
    core_alloc = game.core_allocation_lp()
    
    print(f"Shapley value:  {shapley}")
    print(f"Core allocation: {core_alloc}")
    print(f"Shapley in core: {game.is_in_core(shapley)}")


# --- Demo ---
if __name__ == "__main__":
    bec_scheduling_demo()
    
    # Quick Stackelberg via backward induction
    print("\n=== Stackelberg Duopoly ===")
    game_tree = {
        'name': 'Leader', 'player': 0,
        'actions': {
            f'q1={q1}': {
                'name': f'Follower_q1={q1}', 'player': 1,
                'actions': {
                    f'q2={q2}': {'payoffs': (max(0,(10-q1-q2)*q1 - 2*q1),
                                              max(0,(10-q1-q2)*q2 - 2*q2))}
                    for q2 in range(1, 6)
                }
            } for q1 in range(1, 6)
        }
    }
    payoff, strategy = backward_induction(game_tree)
    print(f"SPE: {strategy}, payoffs: {payoff}")
```

---

## Practice Problems (45 min)

**P1.** (Exercise Set 09D) For the 3-robot BEC game: $v(\{R_i\})=0$, $v(\{R_1,R_2\})=6$, $v(\{R_1,R_3\})=8$, $v(\{R_2,R_3\})=5$, $v(N)=12$. Compute: (a) Shapley value, (b) verify it's in the core, (c) find a core allocation that differs from Shapley.

<details><summary>Answer</summary>

(a) $\phi_{R_1} = (0+6+8+6+8+12)/6$ marginal averaging → $\phi = (4.5, 2.5, 5.0)$. (b) Check all coalition constraints. (c) $(5, 2, 5)$ is also in core: $5+2=7≥6$, $5+5=10≥8$, $2+5=7≥5$. ✓
</details>

**P2.** (Exercise Set 09E) Two robots negotiate corridor bandwidth. Feasible: $b_1 + b_2 \leq 10$ Mbps. Disagreement: each gets 2 Mbps. Robot 1's utopia is 10, Robot 2's is 8 (due to hardware limits). Find NBS and KS.

<details><summary>Answer</summary>

NBS: maximize $(b_1-2)(b_2-2)$ on $b_1+b_2=10$. FOC: $b_1=b_2=5$. NBS = (5,5). KS: utopia = (10,8), direction (8,6) from d=(2,2). Line: $b_2-2 = (6/8)(b_1-2)$. On $b_1+b_2=10$: $b_1 \approx 5.71, b_2 \approx 4.29$. KS favors robot 1 (higher potential).
</details>

**P3.** Design a Vickrey auction for 4 charging slots among 6 robots. Each robot bids on their preferred slot. Find winners and payments.

<details><summary>Answer</summary>

Run 4 independent Vickrey auctions (one per slot). Each slot goes to highest bidder at second-highest price. If robots want multiple slots, use VCG over the combinatorial allocation.
</details>

---

## Expert Challenges

**E1.** Prove the folk theorem for the IPD with grim trigger: for any target payoff $(v_1, v_2)$ with $v_i > 1$ (minmax), construct the supporting strategy and compute the minimum $\delta$.

<details><summary>Hint</summary>

Mix C and D to achieve target $v_i$. On deviation, switch to permanent mutual D. Compare one-shot gain from deviation vs discounted loss. Solve for $\delta$.
</details>

**E2.** Design a revenue-maximizing auction for selling 1 BEC charging slot to 3 robots with valuations $\theta_i \sim U[0, 10]$. Set the optimal reserve price via Myerson's theory.

<details><summary>Hint</summary>

Virtual valuation $\psi(\theta) = 2\theta - 10$. Positive when $\theta > 5$. Optimal reserve = 5. Expected revenue with reserve exceeds standard Vickrey.
</details>

**E3.** Model a Stackelberg game where the fleet controller (leader) sets corridor speed limits, and robots (followers) choose routes. Solve for the SPE and compare with the simultaneous-move Nash.

<details><summary>Hint</summary>

Leader anticipates follower BR. Substitute BR into leader's objective. Leader commits to speed limit; followers route to minimize travel time. SPE gives leader a strategic advantage (Stackelberg leadership benefit).
</details>

---

## Connections
- **Week 14:** Normal form (92), Nash (93-94), equilibrium selection (95), zero-sum (96-97) — foundations
- **Week 15:** Cooperative (99), Shapley (100), bargaining (101), dynamic (102), repeated (103), mechanism design (104) — advanced tools
- **Forward:** Multi-agent optimization, distributed algorithms, fleet-level decision-making
- **OKS:** Complete toolkit: coalition formation → fair division → negotiation → sequential priority → norm enforcement → truthful allocation

## Self-Check
- [ ] I can draw the concept map connecting all Week 15 topics
- [ ] I can select the right tool for a given multi-agent problem
- [ ] I can compute core allocations, Shapley values, and VCG payments
- [ ] I understand the folk theorem's implications for repeated fleet interactions
- [ ] I can model a BEC scheduling problem using mechanism design
