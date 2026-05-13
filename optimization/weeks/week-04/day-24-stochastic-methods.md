# Day 24: Stochastic and Mini-Batch Methods

> **Phase II — Unconstrained Optimization | Week 4 | 2.5 hours** ⚡ ENRICHMENT
> *SGD, Adam, and friends — ML's workhorses, briefly visiting robotics.*

## Navigation
- **Previous:** [Day 23: Momentum Methods](day-23-momentum.md)
- **Next:** [Day 25: Regularization](day-25-regularization.md)
- **Week:** [Week 4 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
While OKS robots primarily use deterministic optimization (Ceres, OSQP), stochastic methods appear in two key areas: (1) **online parameter estimation** where sensor data arrives as a stream and parameters must be updated incrementally, and (2) **learning-based components** like terrain classifiers or neural SLAM modules. Understanding SGD/Adam also builds intuition for adaptive step size selection.

> **Note:** SGD/Adam are primarily ML tools. Robotics optimization rarely uses stochastic methods. Treat this as enrichment — skim if time-constrained, deep-dive if working on learning-based robotics.

---

## Theory (45 min)

### 24.1 Stochastic Gradient Descent (SGD)

Given $f(x) = \frac{1}{N}\sum_{i=1}^N f_i(x)$, replace the full gradient with a stochastic estimate:

$$x_{k+1} = x_k - \alpha_k \nabla f_{i_k}(x)$$

where $i_k$ is randomly sampled.

**Key property:** $\mathbb{E}[\nabla f_{i_k}(x)] = \nabla f(x)$ — unbiased estimator.

**Convergence rate:** $O(1/\sqrt{k})$ for convex, $O(1/k)$ for strongly convex with decaying $\alpha_k = O(1/k)$.

### 24.2 Mini-Batch Gradient

Use a batch $B_k$ of size $b$:

$$x_{k+1} = x_k - \alpha_k \cdot \frac{1}{b}\sum_{i \in B_k} \nabla f_i(x_k)$$

Variance reduction: $\text{Var}[\text{estimate}] = \frac{\sigma^2}{b}$.

Trade-off: larger $b$ → lower variance but more compute per step.

### 24.3 Adaptive Methods

**AdaGrad** — adapt per-parameter:
$$x_{k+1,j} = x_{k,j} - \frac{\alpha}{\sqrt{G_{k,j} + \varepsilon}} g_{k,j}$$
where $G_{k,j} = \sum_{t=1}^k g_{t,j}^2$.

**RMSProp** — exponential moving average instead of full history:
$$v_{k,j} = \beta v_{k-1,j} + (1-\beta) g_{k,j}^2$$
$$x_{k+1,j} = x_{k,j} - \frac{\alpha}{\sqrt{v_{k,j} + \varepsilon}} g_{k,j}$$

**Adam** — RMSProp + momentum with bias correction:
$$m_k = \beta_1 m_{k-1} + (1-\beta_1) g_k \quad \text{(first moment)}$$
$$v_k = \beta_2 v_{k-1} + (1-\beta_2) g_k^2 \quad \text{(second moment)}$$
$$\hat{m}_k = \frac{m_k}{1 - \beta_1^k}, \quad \hat{v}_k = \frac{v_k}{1 - \beta_2^k}$$
$$x_{k+1} = x_k - \frac{\alpha}{\sqrt{\hat{v}_k} + \varepsilon} \hat{m}_k$$

Default: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$, $\alpha = 10^{-3}$.

### 24.4 Learning Rate Schedules

| Schedule | Formula | When |
|----------|---------|------|
| Constant | $\alpha_k = \alpha_0$ | Simple, but won't converge to exact minimum |
| Step decay | $\alpha_k = \alpha_0 \cdot \gamma^{\lfloor k/s \rfloor}$ | Every $s$ steps, decay by $\gamma$ |
| Inverse | $\alpha_k = \alpha_0 / (1 + dk)$ | Theoretically optimal for convex |
| Cosine annealing | $\alpha_k = \frac{\alpha_0}{2}(1 + \cos(\pi k / T))$ | Popular in deep learning |
| Warm-up + decay | Linearly increase, then decay | Transformer training |

### 24.5 When Robotics Uses Stochastic Methods

- **Online calibration:** streaming sensor data → stochastic parameter update
- **Neural network training:** perception, learned cost functions
- **Reinforcement learning:** policy gradient methods are inherently stochastic
- **Real-time estimation:** particle filters use stochastic sampling

---

## Implementation (60 min)

### Exercise 1: SGD for Sum-of-Functions
```python
# See: code/week04/stochastic_methods.py
```
Implement SGD for $f(x) = \frac{1}{N}\sum_i f_i(x)$ where each $f_i$ is a single data point's contribution.

### Exercise 2: Mini-Batch GD
Implement mini-batch GD with configurable batch size. Show variance reduction as batch size increases.

### Exercise 3: GD vs SGD vs Mini-Batch on Linear Regression
Generate $N = 10000$ samples, compare full GD, SGD, and mini-batch ($b = 32, 128, 512$) on linear regression.

### Exercise 4: Learning Rate Schedules
Implement constant, $1/k$, and cosine annealing schedules. Show their effect on convergence.

### Exercise 5: Adam from Scratch
Implement Adam optimizer with bias correction. Compare to SGD on a nonconvex toy problem.

---

## Practice (45 min)

1. When is SGD better than full-batch GD? (Compute budget: $N = 10^6$ and you can only do 100 passes.)
2. Derive: why does mini-batch variance scale as $\sigma^2/b$?
3. What is the optimal batch size for a given compute budget?
4. Why does Adam often converge faster than SGD in early iterations but may not find as good a minimum?
5. In robotics online estimation: if 100 sensor readings arrive per second, is it better to batch them or process one at a time?
6. Why does AdaGrad's learning rate go to zero? How does RMSProp fix this?
7. Compare Adam to L-BFGS: when would you prefer each?
8. What is the "noise ball" — why does SGD with constant step size orbit the minimum?

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Variance reduction with SVRG</summary>

**Problem:** Implement Stochastic Variance-Reduced Gradient (SVRG): periodically compute the full gradient, then use it to reduce variance of stochastic updates. Show that SVRG converges linearly (like GD) while using stochastic updates (like SGD).

**Solution:** SVRG outer loop: compute $\tilde{\mu} = \nabla f(\tilde{x})$ at a reference point. Inner loop: $g_k = \nabla f_{i_k}(x_k) - \nabla f_{i_k}(\tilde{x}) + \tilde{\mu}$. The correction term $-\nabla f_{i_k}(\tilde{x}) + \tilde{\mu}$ ensures $\mathbb{E}[g_k] = \nabla f(x_k)$ with variance → 0 as $x_k \to \tilde{x}$. Converges linearly for strongly convex $f$.
</details>

<details>
<summary>🎯 Challenge 2: Adam convergence failure</summary>

**Problem:** Construct a simple 1D problem where Adam with default parameters converges to the *wrong* point (non-convergence of Adam, Reddi et al., 2018). Then implement AMSGrad (the fix) and show it converges correctly.

**Solution:** The classic counter-example alternates gradients: $g_k = C$ for $k \mod 3 \neq 1$ and $g_k = -\sqrt{C}$ otherwise, with $C > 1$. The running average in Adam can make the effective update direction wrong. AMSGrad fixes this by taking $\hat{v}_k = \max(\hat{v}_{k-1}, v_k)$ — it never decreases the denominator, ensuring convergence.
</details>

<details>
<summary>🎯 Challenge 3: Stochastic optimization for robot calibration</summary>

**Problem:** Simulate an online wheel calibration scenario: encoder ticks and true distances arrive in a stream. Implement an SGD-based online estimator for wheel radius. Compare to batch least squares (computing periodically with all data). Show that SGD with averaging converges to the same answer.

**Solution:** Model: $d_i = \pi r \cdot \text{ticks}_i + \varepsilon_i$. Loss: $\frac{1}{2}(d_i - \pi r \cdot \text{ticks}_i)^2$. SGD update: $r \leftarrow r - \alpha_k \cdot \text{ticks}_i \cdot \pi \cdot (d_i - \pi r \cdot \text{ticks}_i)$. With Polyak averaging ($\bar{r}_k = \frac{1}{k}\sum r_j$), this converges to the batch LS solution. Practical benefit: handles non-stationary wheel wear by weighting recent data more.
</details>

---

## Self-Assessment Checklist

- [ ] I can implement SGD, mini-batch GD, and Adam from scratch
- [ ] I understand the variance-compute tradeoff in batch size selection
- [ ] I know when stochastic methods are appropriate in robotics (and when they're not)
- [ ] I can implement and compare different learning rate schedules

---

## Key Takeaways

1. **SGD** trades accuracy per step for speed per step — useful when $N$ is huge.
2. **Adam** adapts per-parameter learning rates — the default choice for neural networks.
3. **Robotics rarely needs SGD** — problems are usually small enough for batch methods.
4. **Online estimation** is the main robotics use case for stochastic optimization ideas.
