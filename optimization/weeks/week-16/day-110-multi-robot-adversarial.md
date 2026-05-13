# Day 110: Multi-Robot Games + Adversarial ML

> **Phase VII — Game Theory for Optimization & Robotics | Week 16 | 2.5 hours**
> *Multi-robot coordination is a game; adversarial robustness is a game — game theory unifies both.*

## Navigation
- Previous: [Day 109: Potential Games and Congestion Games](day-109-congestion-games.md)
- Next: [Day 111: Robust Optimization as Games Against Nature](day-111-robust-optimization.md)
- [Week 16 Overview](../../../CURRICULUM.md)
- [Chapter 10: Game Theory Reference](../../../10-game-theory.md)
- [Full Curriculum](../../../00-learning-plan.md)

## OKS Relevance
OKS fleet coordination combines two game-theoretic challenges: (1) allocating tasks across robots competing for limited resources (auction games), and (2) ensuring perception and navigation are robust to worst-case sensor failures or adversarial conditions (adversarial games). This day bridges multi-agent coordination with adversarial robustness — both critical for warehouse-scale fleet operation.

---

## Theory (45 min)

### 110.1 Task Allocation as Combinatorial Auctions

With $n$ robots and $m$ tasks, each robot $i$ has a **valuation** $v_i(S)$ for completing task bundle $S \subseteq M$. The **combinatorial auction** finds an allocation $(S_1, \ldots, S_n)$ maximizing social welfare:

$$\max_{S_1, \ldots, S_n} \sum_{i=1}^n v_i(S_i) \quad \text{s.t. } S_i \cap S_j = \emptyset \; \forall i \neq j$$

This is NP-hard in general, but with **submodular valuations** ($v_i(S \cup \{j\}) - v_i(S)$ is decreasing in $|S|$), greedy allocation achieves a $(1 - 1/e)$-approximation.

**VCG mechanism**: The Vickrey-Clarke-Groves mechanism incentivizes truthful bidding. Robot $i$ pays:

$$p_i = \max_{S_{-i}} \sum_{j \neq i} v_j(S_j) - \sum_{j \neq i} v_j(S_j^*)$$

i.e., the externality $i$ imposes on others. Under VCG, truthful reporting is a dominant strategy.

### 110.2 Pursuit-Evasion Games

A **pursuit-evasion game** involves $p$ pursuers trying to capture $e$ evaders in a graph or continuous space. The game is zero-sum: pursuers maximize capture probability, evaders minimize it.

For a discrete graph $G = (V, E)$, the **lion and man** problem: at each step, pursuer moves along an edge, evader moves along an edge. The **search number** $s(G)$ is the minimum number of pursuers needed to guarantee capture.

**Continuous pursuit-evasion**: Pursuer dynamics $\dot{x}_p = u_p$, evader $\dot{x}_e = u_e$ with $\|u_p\| \leq v_p$, $\|u_e\| \leq v_e$. If $v_p > v_e$, the pursuer can always capture. The optimal pursuit strategy involves **Apollonius circles**.

### 110.3 Adversarial ML as Zero-Sum Games

A **GAN** (Generative Adversarial Network) is a zero-sum game between generator $G$ and discriminator $D$:

$$\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

At Nash equilibrium: $G$ generates the true data distribution, $D$ outputs $1/2$ everywhere.

**Adversarial robustness**: For a classifier $f_\theta$, the adversary solves:

$$\max_{\|\delta\| \leq \varepsilon} \ell(f_\theta(x + \delta), y)$$

The defender trains with **adversarial training** (Madry et al.):

$$\min_\theta \; \mathbb{E}_{(x,y)} \left[ \max_{\|\delta\| \leq \varepsilon} \ell(f_\theta(x + \delta), y) \right]$$

This is a minimax problem — directly a two-player zero-sum game between the model and the adversary.

---

## Implementation (60 min)

```python
"""Multi-Robot Auctions and Adversarial Games."""
import numpy as np
from itertools import combinations

def greedy_auction(valuations: np.ndarray) -> tuple:
    """Greedy task allocation for additive valuations.
    
    Args:
        valuations: n_robots x m_tasks matrix, v[i][j] = value of task j to robot i
    
    Returns:
        allocation (list of lists), total welfare
    """
    n, m = valuations.shape
    allocated = [[] for _ in range(n)]
    assigned = set()
    
    # Greedy: repeatedly assign highest-value (robot, task) pair
    pairs = []
    for i in range(n):
        for j in range(m):
            pairs.append((valuations[i, j], i, j))
    pairs.sort(reverse=True)
    
    for val, robot, task in pairs:
        if task not in assigned:
            allocated[robot].append(task)
            assigned.add(task)
    
    welfare = sum(sum(valuations[i, j] for j in allocated[i]) for i in range(n))
    return allocated, welfare

def vcg_payments(valuations: np.ndarray, allocation: list) -> np.ndarray:
    """Compute VCG payments for each robot."""
    n, m = valuations.shape
    payments = np.zeros(n)
    
    # Welfare without each robot
    for i in range(n):
        # Remove robot i, re-allocate
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        v_others = valuations[mask]
        _, welfare_without_i = greedy_auction(v_others)
        
        # Others' welfare in original allocation
        welfare_others_with_i = sum(
            sum(valuations[j, t] for t in allocation[j])
            for j in range(n) if j != i
        )
        
        payments[i] = welfare_without_i - welfare_others_with_i
    
    return payments

def pursuit_evasion_grid(grid_size: int = 5, n_steps: int = 50,
                         seed: int = 42) -> dict:
    """Simple pursuit-evasion on a grid."""
    rng = np.random.default_rng(seed)
    
    # Pursuer starts at (0, 0), evader at (grid_size-1, grid_size-1)
    pursuer = np.array([0, 0], dtype=float)
    evader = np.array([grid_size - 1, grid_size - 1], dtype=float)
    
    v_p, v_e = 1.2, 1.0  # pursuer slightly faster
    trajectories_p, trajectories_e = [pursuer.copy()], [evader.copy()]
    
    for step in range(n_steps):
        # Pursuer: move toward evader
        direction_p = evader - pursuer
        dist = np.linalg.norm(direction_p)
        if dist < 0.5:
            break  # captured
        pursuer += v_p * direction_p / dist
        
        # Evader: move away from pursuer + random
        direction_e = evader - pursuer
        noise = rng.normal(0, 0.3, 2)
        move_e = direction_e / np.linalg.norm(direction_e) + noise
        evader += v_e * move_e / np.linalg.norm(move_e)
        evader = np.clip(evader, 0, grid_size)
        
        trajectories_p.append(pursuer.copy())
        trajectories_e.append(evader.copy())
    
    return {
        'pursuer': np.array(trajectories_p),
        'evader': np.array(trajectories_e),
        'captured': step < n_steps - 1,
        'steps': len(trajectories_p),
    }

def adversarial_minimax_demo():
    """GAN-style minimax on a simple function."""
    # min_x max_y  x*y - x^2/2 - y^2/2
    # NE at x=0, y=0
    lr = 0.05
    x, y = 2.0, -1.0
    trajectory = [(x, y)]
    
    for t in range(200):
        # Gradient ascent for y (maximizer)
        grad_y = x - y
        y += lr * grad_y
        
        # Gradient descent for x (minimizer)
        grad_x = y - x
        x -= lr * grad_x
        
        trajectory.append((x, y))
    
    trajectory = np.array(trajectory)
    return trajectory

# Demo 1: Task auction
print("=== Task Allocation Auction ===")
np.random.seed(42)
n_robots, m_tasks = 4, 6
valuations = np.random.randint(1, 20, (n_robots, m_tasks))
print(f"Valuations:\n{valuations}")

allocation, welfare = greedy_auction(valuations)
for i, tasks in enumerate(allocation):
    task_vals = [valuations[i, t] for t in tasks]
    print(f"  Robot {i}: tasks {tasks}, values {task_vals}")
print(f"  Total welfare: {welfare}")

payments = vcg_payments(valuations, allocation)
print(f"  VCG payments: {payments}")
print(f"  Utilities: {[sum(valuations[i,t] for t in allocation[i]) - payments[i] for i in range(n_robots)]}")

# Demo 2: Pursuit-evasion
print("\n=== Pursuit-Evasion ===")
pe_result = pursuit_evasion_grid()
print(f"  Captured: {pe_result['captured']} in {pe_result['steps']} steps")

# Demo 3: Minimax
print("\n=== Minimax GAN-style ===")
traj = adversarial_minimax_demo()
print(f"  Start: ({traj[0,0]:.2f}, {traj[0,1]:.2f})")
print(f"  End:   ({traj[-1,0]:.4f}, {traj[-1,1]:.4f})")
print(f"  NE at (0, 0) — error: {np.linalg.norm(traj[-1]):.4f}")
```

---

## Practice Problems (45 min)

**P1**: For 3 robots and 4 tasks with valuations $V = \begin{pmatrix} 5 & 3 & 8 & 2 \\ 4 & 7 & 1 & 6 \\ 6 & 2 & 4 & 5 \end{pmatrix}$, find the optimal allocation and VCG payments.

<details><summary>Answer</summary>

Optimal: Robot 0 gets task 2 (val 8), Robot 1 gets task 1 (val 7) and task 3 (val 6), Robot 2 gets task 0 (val 6). Total welfare = 27. VCG payments: Robot 0 pays $v_{\text{others without 0}} - v_{\text{others with 0}}$. Without Robot 0, tasks {0,1,2,3} assigned to robots {1,2}: best is 7+6+6+4=23 minus others' welfare with 0: 7+6+6=19. Payment₀ = 23-19 = 4. Similarly for others.
</details>

**P2**: In a pursuit-evasion game on a cycle graph with 6 nodes, what is the minimum number of pursuers to guarantee capture?

<details><summary>Answer</summary>

For a cycle $C_n$, the search number is 2. One pursuer cannot guarantee capture (evader can always move away). Two pursuers can trap the evader: one chases clockwise, the other counterclockwise, reducing the evader's escape space to zero. For $C_6$: two pursuers starting at opposite nodes converge in at most 3 steps.
</details>

**P3**: Show that the GAN minimax objective has a saddle point when $G$ generates the true distribution.

<details><summary>Answer</summary>

If $p_G = p_{\text{data}}$, the objective becomes $V(D) = \mathbb{E}_x[\log D(x) + \log(1-D(x))]$. Maximizing over $D$: $D^*(x) = 1/2$ everywhere, giving $V = -2\log 2$. This is a global saddle: $G$ cannot improve (already matching data), $D$ cannot improve (already optimal). The minimax theorem applies because the objective is convex in $D$'s parameters and concave in $G$'s (in function space).
</details>

---

## Expert Challenges

**E1**: Implement a **combinatorial auction** with complementary valuations $v_i(S)$ that is not additive. Compare greedy, VCG, and an LP relaxation.

<details><summary>Hint</summary>

Use $v_i(\{a,b\}) > v_i(\{a\}) + v_i(\{b\})$ for some pairs (super-additive). The LP relaxation of the winner determination problem gives a fractional solution; round it and compare social welfare.
</details>

**E2**: Implement FGSM (Fast Gradient Sign Method) adversarial attack and adversarial training for a simple 2-layer neural network on a toy 2D classification problem.

<details><summary>Answer</summary>

FGSM: $\delta = \varepsilon \cdot \text{sign}(\nabla_x \ell(f_\theta(x), y))$. Adversarial training: alternate between finding worst-case $\delta$ and updating $\theta$. On a 2D spiral dataset, standard accuracy drops from 95% to 30% under attack; adversarial training recovers to 75% robust accuracy at $\varepsilon = 0.1$.
</details>

**E3**: Model a warehouse scenario where one robot is "adversarial" (e.g., a malfunctioning robot that moves erratically). Design a pursuit-evasion strategy for a helper robot to intercept it.

<details><summary>Answer</summary>

Model as zero-sum: helper minimizes time-to-intercept, adversarial robot maximizes it. With grid graph and speed ratio $v_h/v_a > 1$, the lion-and-man strategy guarantees capture. For speed ratio 1.2 on a 10×10 grid, expected capture time ≈ 15 steps. The helper uses the Apollonius circle approach: move to cut off the evader's escape directions rather than pursuing directly.
</details>

---

## Connections
- **Back**: [Day 109: Congestion Games](day-109-congestion-games.md) — auctions extend congestion with strategic bidding
- **Forward**: [Day 111: Robust Optimization](day-111-robust-optimization.md) — adversarial robustness formalized as games against nature
- **OKS**: Fleet task allocation, pursuit of malfunctioning robots, robust perception in adversarial environments

## Self-Check
- [ ] I can formulate task allocation as a combinatorial auction and compute VCG payments
- [ ] I understand pursuit-evasion games and the role of speed ratios
- [ ] I can explain the GAN minimax objective and its Nash equilibrium
- [ ] I see the connection between adversarial ML and zero-sum game theory
