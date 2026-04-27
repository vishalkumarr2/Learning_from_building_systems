# Day 26: Global Optimization Methods

> **Phase II — Unconstrained Optimization | Week 4 | 2.5 hours**
> *When local methods aren't enough — escaping local minima systematically.*

## Navigation
- **Previous:** [Day 25: Regularization](day-25-regularization.md)
- **Next:** [Day 27: scipy.optimize Deep Dive](day-27-scipy-optimize.md)
- **Week:** [Week 4 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Most OKS optimization is local (sensor calibration, pose estimation — all start near the solution). But global optimization appears in two critical scenarios: (1) **loop closure detection** in SLAM where the initial alignment between scans may be far from correct, and (2) **scan matching initialization** where ICP needs a reasonable starting point. Understanding when you need global search vs when good initialization suffices is a key engineering judgment.

---

## Theory (45 min)

### 26.1 The Problem: Local Minima

Local optimization finds $x^*$ with $\nabla f(x^*) = 0$ and $\nabla^2 f(x^*) \succeq 0$. But there may be many such points, and only one (or few) are the **global** minimum.

For convex problems: local = global (no issue). For nonconvex: GD, Newton, BFGS all find local minima.

### 26.2 Random Restart

Simplest approach: run a local optimizer from $K$ random starting points.

$$x^*_{\text{best}} = \arg\min_{k=1}^K f(x^*_k)$$

**Properties:**
- Probability of finding global minimum increases with $K$
- Each restart is independent — embarrassingly parallel
- Works well when basins of attraction are large

**Cost:** $K \times$ (cost of one local solve).

### 26.3 Simulated Annealing

Inspired by metallurgy — allow "uphill" moves with decreasing probability:

1. Propose: $x_{\text{new}} = x + \varepsilon$ (random perturbation)
2. Accept if $f(x_{\text{new}}) < f(x)$ (downhill)
3. Accept with probability $\exp(-\Delta f / T)$ if uphill
4. Cool: $T \leftarrow \gamma T$ where $\gamma \in (0.9, 0.999)$

**Temperature schedule** controls exploration vs exploitation:
- High $T$: almost random walk (exploration)
- Low $T$: almost greedy descent (exploitation)

**Guarantee:** Converges to global minimum in probability as $T \to 0$ slowly enough. In practice: convergence is slow.

### 26.4 Basin-Hopping

Combination of random perturbation + local optimization:

1. Start at $x_0$, run local optimizer → $x^*_0$
2. Perturb: $x_1 = x^*_0 + \varepsilon$
3. Run local optimizer → $x^*_1$
4. Accept/reject via Metropolis criterion
5. Repeat

**Advantage:** Each evaluation is at a local minimum → cleaner landscape. `scipy.optimize.basinhopping` implements this.

### 26.5 Evolutionary/Genetic Algorithms (Brief)

- Maintain a **population** of candidate solutions
- **Selection:** keep the best
- **Crossover:** combine two solutions
- **Mutation:** randomly perturb
- `scipy.optimize.differential_evolution` — a robust implementation

### 26.6 When Do You Need Global Optimization?

| Scenario | Local suffices? | Why |
|----------|----------------|-----|
| Sensor calibration | Yes | Prior from design specs is close |
| Pose estimation | Yes | IMU prediction is good initial guess |
| ICP scan matching | Usually | Previous frame is good seed |
| Loop closure | **No** | Initial alignment may be far off |
| Robot placement | **No** | Combinatorial landscape |
| Neural network training | **No** | But SGD works via implicit exploration |

**Rule of thumb:** If your initialization is within the basin of attraction of the global minimum, use local methods. If not, you need global search.

---

## Implementation (60 min)

### Exercise 1: Random Restart + L-BFGS
```python
# See: code/week04/global_optimization.py
```
Implement random restart with `scipy.optimize.minimize(method='L-BFGS-B')`.

### Exercise 2: Simulated Annealing
Implement simulated annealing from scratch with exponential cooling.

### Exercise 3: Test on Rastrigin Function
Rastrigin: $f(x) = 10n + \sum_i [x_i^2 - 10\cos(2\pi x_i)]$ — many local minima. Compare local method (finds nearest local min) vs global methods.

### Exercise 4: scipy Global Optimizers
Compare `minimize()`, `differential_evolution()`, and `basinhopping()` on the same problem.

### Exercise 5: Scan Matching Initialization
Robotics scenario: find initial rotation/translation for ICP. Random restart over rotation angles, L-BFGS for refinement.

---

## Practice (45 min)

1. How many random restarts are needed to find a global minimum with 95% probability if $p = 0.05$ per trial?
2. Why does simulated annealing's cooling schedule matter? What happens if you cool too fast? Too slow?
3. Compare the computational cost: 100 random restarts of L-BFGS vs 10000 iterations of simulated annealing.
4. In robotics, is global optimization common? Why or why not?
5. Basin-hopping vs differential evolution: which is better for continuous problems? For mixed-integer?
6. Why does the Rastrigin function have $10^n$ local minima in $n$ dimensions?
7. How does scan matching avoid global optimization in practice? (Hint: motion model prediction.)
8. Design a simple test: given a function, how do you determine whether you need global optimization?

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Multi-start with clustering</summary>

**Problem:** Implement multi-start optimization that clusters converged solutions to identify distinct local minima. On Rastrigin in 2D: (a) find all local minima in $[-5, 5]^2$, (b) identify the basin of attraction for each, (c) estimate what fraction of starting points converge to the global minimum.

**Solution:** Run 1000 random starts, collect all converged points. Cluster with `scipy.cluster.hierarchy` or DBSCAN (tolerance 0.1). Each cluster = one local minimum. The basin fraction = cluster_size/1000. For 2D Rastrigin: 25 local minima in $[-5,5]^2$, global minimum's basin is ~4% of the area, so ~4% of random starts find it.
</details>

<details>
<summary>🎯 Challenge 2: Adaptive simulated annealing</summary>

**Problem:** Implement adaptive SA where the perturbation size adapts based on acceptance rate. Target 30-50% acceptance: if acceptance is too high, increase perturbation; if too low, decrease. Show this converges faster than fixed-perturbation SA on Rastrigin.

**Solution:** Track acceptance rate over a window of 100 steps. If rate > 0.5: multiply perturbation by 1.1. If rate < 0.3: multiply by 0.9. This automatically handles the transition from exploration (large perturbation) to exploitation (small perturbation). Converges 2-5× faster than fixed SA on Rastrigin.
</details>

<details>
<summary>🎯 Challenge 3: Global optimization for ICP initialization</summary>

**Problem:** Generate two 2D point clouds (source and target) with known transformation $(R, t)$. Add noise and outliers. Use random restart + ICP to recover the transformation. Measure: (a) how often does single-start ICP fail, (b) how many restarts are needed for 99% success, (c) does basin-hopping improve over random restart?

**Solution:** With rotation up to 180°: single-start ICP fails ~40% of the time (converges to wrong rotation by 180°). With 10 random restarts over $[0, 2\pi)$ rotation: ~98% success. Basin-hopping converges faster because it reuses successful basins. This demonstrates why real SLAM systems use feature-based initial alignment before ICP refinement.
</details>

---

## Self-Assessment Checklist

- [ ] I can implement random restart and simulated annealing
- [ ] I understand when global optimization is needed vs when good initialization suffices
- [ ] I can use scipy's `basinhopping` and `differential_evolution`
- [ ] I can design a multi-start strategy for a robotics initialization problem

---

## Key Takeaways

1. **Local methods** find local minima; **global methods** search more broadly.
2. **Random restart** is simple, parallel, and often sufficient.
3. **In robotics**, good initialization (from sensors/prediction) usually avoids global search.
4. **When needed:** loop closure, place recognition, initial alignment — use `basinhopping` or multi-start.
