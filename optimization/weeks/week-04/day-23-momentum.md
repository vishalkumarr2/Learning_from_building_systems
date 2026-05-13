# Day 23: Momentum Methods and Acceleration

> **Phase II — Unconstrained Optimization | Week 4 | 2.5 hours**
> *From GD's zigzag to Nesterov's optimal rate — the physics of convergence acceleration.*

## Navigation
- **Previous:** [Day 22: Convergence Theory](day-22-convergence-theory.md)
- **Next:** [Day 24: Stochastic Methods](day-24-stochastic-methods.md)
- **Week:** [Week 4 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
The OKS estimator's Kalman filter is itself a form of momentum-based update — the state prediction carries forward information from previous steps. Understanding momentum methods explains why the estimator can track fast-moving robots without overshooting, and why tuning the process noise covariance $Q$ (which controls how much the estimator "trusts" its momentum) is critical for navigation stability.

---

## Theory (45 min)

### 23.1 The Zigzag Problem

Gradient descent on ill-conditioned problems zigzags:

$$x_{k+1} = x_k - \alpha \nabla f(x_k)$$

With $\kappa = 100$: the gradient points "mostly sideways" relative to the optimal direction. GD takes ~230 steps for $10^{-2}$ reduction.

### 23.2 Heavy Ball Method (Polyak, 1964)

Add momentum from previous step:

$$x_{k+1} = x_k - \alpha \nabla f(x_k) + \beta(x_k - x_{k-1})$$

**Optimal parameters for quadratic** $f(x) = \frac{1}{2}x^TAx$:

$$\alpha = \frac{4}{(\sqrt{\lambda_{\max}} + \sqrt{\lambda_{\min}})^2}, \quad \beta = \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^2$$

**Rate:** $O\left(\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k\right)$ vs GD's $O\left(\left(\frac{\kappa-1}{\kappa+1}\right)^k\right)$.

For $\kappa = 100$: heavy ball factor = $0.818$ vs GD's factor = $0.98$. Dramatically faster!

### 23.3 Nesterov Accelerated Gradient (1983)

"Look ahead" before computing the gradient:

$$y_k = x_k + \frac{k-1}{k+2}(x_k - x_{k-1})$$
$$x_{k+1} = y_k - \alpha \nabla f(y_k)$$

**Rate for convex:** $O(1/k^2)$ vs GD's $O(1/k)$ — optimal for first-order methods.

**Rate for strongly convex:** $O\left(\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k\right)$ — matches lower bound.

### 23.4 Connection to Control Theory

The heavy ball method is a **PD controller** for optimization:

$$x_{k+1} = x_k - \alpha \underbrace{\nabla f(x_k)}_{\text{proportional}} + \beta \underbrace{(x_k - x_{k-1})}_{\text{derivative}}$$

- **P term** (gradient): pushes toward minimum
- **D term** (momentum): dampens oscillations

This is the same trade-off as PID tuning in robot control:
- Too much P (too large $\alpha$): oscillations
- Too much D (too large $\beta$): overshooting, instability
- Optimal balance: critical damping

### 23.5 Practical Considerations

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| Heavy ball | Simple, fast on quadratics | Requires eigenvalue bounds, can oscillate |
| Nesterov | Optimal rate, adaptive | Slightly more complex, can overshoot on nonconvex |
| BFGS | Adapts to curvature | $O(n^2)$ memory |

**When to use momentum:** Problems with high $\kappa$ where second-order methods are too expensive ($n > 10^4$).

---

## Implementation (60 min)

### Exercise 1: Heavy Ball Method
```python
# See: code/week04/momentum_methods.py
```
Implement heavy ball with optimal parameters for quadratic.

### Exercise 2: Nesterov Accelerated Gradient
Implement Nesterov with adaptive momentum parameter $\frac{k-1}{k+2}$.

### Exercise 3: GD vs Heavy Ball vs Nesterov
Compare all three on an ill-conditioned quadratic ($\kappa = 1000$). Plot convergence.

### Exercise 4: Oscillation Problem
Show heavy ball with $\beta$ too large causes oscillation/divergence.

### Exercise 5: Nesterov on Rosenbrock
Apply Nesterov to the nonconvex Rosenbrock function. Compare convergence curves with GD and heavy ball.

---

## Practice (45 min)

1. Derive the optimal $\alpha$ and $\beta$ for heavy ball on $f(x) = \frac{1}{2}x^TAx$ by analyzing the eigenvalues of the iteration matrix.
2. Compute: for $\kappa = 10000$, how many iterations does GD need vs Nesterov for $10^{-6}$ error?
3. Why does Nesterov evaluate the gradient at $y_k$ (the "look-ahead" point) instead of $x_k$?
4. How does momentum relate to the exponential moving average of gradients?
5. For robot trajectory optimization: if the cost function has $\kappa \approx 10^4$, which method would you use and why?
6. Explain why heavy ball can diverge on nonconvex functions even if the step size is small.
7. What happens to Nesterov's momentum parameter $\frac{k-1}{k+2}$ as $k \to \infty$?
8. Connection: how is the Kalman filter gain $K_k$ analogous to an adaptive momentum parameter?

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Optimal first-order method</summary>

**Problem:** Implement the "optimal" method of Nesterov for strongly convex functions with known $\mu$ and $L$:
$$x_{k+1} = y_k - \frac{1}{L}\nabla f(y_k), \quad y_{k+1} = x_{k+1} + \frac{\sqrt{L} - \sqrt{\mu}}{\sqrt{L} + \sqrt{\mu}}(x_{k+1} - x_k)$$
Compare its convergence to standard Nesterov on quadratics with $\kappa \in \{100, 1000, 10000\}$.

**Solution:** The strongly convex version uses a fixed momentum parameter $\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}$ instead of the adaptive $\frac{k-1}{k+2}$. This gives exponential convergence matching the lower bound. On $\kappa=10000$: ~200 iterations instead of GD's ~10000.
</details>

<details>
<summary>🎯 Challenge 2: Momentum and oscillation analysis</summary>

**Problem:** For the 2D quadratic $f(x,y) = 50x^2 + y^2$ ($\kappa=100$), plot the full trajectory of GD, heavy ball, and Nesterov starting from $(1, 1)$. Annotate the zigzag pattern of GD and the spiral/damped pattern of momentum methods.

**Solution:** GD zigzags along the steep $x$-direction. Heavy ball with optimal $\beta$ spirals inward smoothly. With $\beta$ too large, heavy ball overshoots and spirals outward before recovering. Nesterov shows the characteristic "overshoot then correct" pattern — it briefly goes past the minimum then pulls back, which is actually optimal.
</details>

<details>
<summary>🎯 Challenge 3: Adaptive restart for Nesterov</summary>

**Problem:** Nesterov can overshoot on nonconvex functions. Implement the "function restart" heuristic: if $f(x_{k+1}) > f(x_k)$, reset momentum to zero. Show this prevents oscillation on Rosenbrock while preserving acceleration on well-conditioned regions.

**Solution:** The restart heuristic detects when momentum causes the function value to increase. By resetting $x_{k-1} = x_k$ (zero momentum), the method falls back to GD for one step. On Rosenbrock's narrow valley, this prevents the momentum from shooting across the valley. The restarted method typically converges in 2-3× fewer iterations than vanilla Nesterov on Rosenbrock.
</details>

---

## Self-Assessment Checklist

- [ ] I can implement heavy ball and Nesterov from scratch
- [ ] I understand why momentum improves GD's convergence from $O(\kappa)$ to $O(\sqrt{\kappa})$
- [ ] I can compute optimal momentum parameters for quadratics
- [ ] I recognize the PD-controller analogy and its connection to robot estimation

---

## Key Takeaways

1. **Momentum** turns GD's $O(\kappa)$ into $O(\sqrt{\kappa})$ — the square root is huge for ill-conditioned problems.
2. **Heavy ball** is simple but needs eigenvalue bounds; **Nesterov** is adaptive and optimal.
3. **Control theory connection:** momentum = derivative term in PD control.
4. **Practical:** Use momentum for large-scale problems where $\kappa$ is high and Hessians are unavailable.
