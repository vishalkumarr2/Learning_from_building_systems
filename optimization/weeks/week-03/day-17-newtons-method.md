# Day 17: Newton's Method

> **Phase II — Unconstrained Optimization | Week 3 | 2.5 hours**
> *Quadratic convergence: the gold standard when you can afford the Hessian.*

## Navigation
- **Previous:** [Day 16: Line Search Methods](day-16-line-search.md)
- **Next:** [Day 18: Trust Region Methods](day-18-trust-region.md)
- **Week:** [Week 3 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Ceres uses Gauss-Newton (an approximation of Newton) as its default inner loop. Newton's method on the full problem $\min f(x)$ is the theoretical ideal — quadratic convergence means ~5 iterations vs ~10,000 for GD. The catch: you need the Hessian ($O(n^2)$ storage, $O(n^3)$ solve). For SLAM with 100,000 variables, that's impossible — which is why we'll learn sparse and approximate alternatives.

---

## Theory (45 min)

### 17.1 The Newton Step

Minimize the second-order Taylor expansion at $x_k$:

$$m_k(\delta) = f_k + \nabla f_k^T \delta + \frac{1}{2} \delta^T H_k \delta$$

Setting $\nabla_\delta m_k = 0$:

$$H_k \delta = -\nabla f_k \quad \Rightarrow \quad x_{k+1} = x_k - H_k^{-1} \nabla f_k$$

### 17.2 Quadratic Convergence

Near a minimum $x^*$ where $H(x^*) \succ 0$:

$$\|x_{k+1} - x^*\| \leq C \|x_k - x^*\|^2$$

**Example:** If $\|x_0 - x^*\| = 0.1$, then errors go: $0.1, 0.01, 0.0001, 10^{-8}, 10^{-16}$ — just 5 iterations to machine precision.

### 17.3 Derivation for Quadratics

For $f(x) = \frac{1}{2}x^TAx - b^Tx$: $\nabla f = Ax - b$, $H = A$.

Newton: $x_1 = x_0 - A^{-1}(Ax_0 - b) = A^{-1}b = x^*$.

**Newton converges in exactly 1 step on any quadratic** — because the Taylor model is exact.

### 17.4 When Newton Fails

1. **Hessian not PD:** Step may ascend (maximize instead of minimize)
2. **Far from solution:** Quadratic model is poor, step overshoots
3. **Cost:** $O(n^2)$ to store $H$, $O(n^3)$ to solve $H\delta = -g$ — prohibitive for large $n$
4. **Saddle points:** Newton converges to saddle points as readily as to minima

### 17.5 Modified Newton

When $H$ is not PD, add regularization:

$$(\bar{H} + \mu I)\delta = -\nabla f$$

Choose $\mu$ to make $\bar{H} + \mu I \succ 0$. This is a damped Newton step — blends between Newton (small $\mu$) and gradient descent (large $\mu$).

### 17.6 Newton with Line Search

To handle poor initial points:

$$x_{k+1} = x_k + \alpha_k \delta_k$$

where $\delta_k$ is the Newton direction and $\alpha_k$ satisfies Wolfe conditions. Ensures global convergence while preserving quadratic convergence near the solution.

---

## Implementation (60 min)

### Exercise 1: Pure Newton
```python
# See: code/week03/newtons_method.py
```
Implement Newton's method with analytic Hessian.

### Exercise 2: Rosenbrock Test
Test on Rosenbrock: count iterations (~10-20 vs GD's ~10,000). Show quadratic convergence by plotting $\log\|x_k - x^*\|$ vs iteration.

### Exercise 3: Quadratic Convergence Verification
Plot $\log(\text{error}_k)$ vs $k$ — the slope should double each step (characteristic of order-2 convergence).

### Exercise 4: Failure Case
Start far from minimum where Hessian is indefinite. Show pure Newton diverges, but damped Newton (with line search) works.

### Exercise 5: Modified Newton
Implement $H + \mu I$ regularization. When $H$ has negative eigenvalues, shift them to be positive. Show it stabilizes the divergent case.

---

## Practice (45 min)

1. **Derive:** Newton's method on $f(x) = ax^2 + bx + c$ converges in 1 step. Show it.
2. **When does Newton fail?** List 3 scenarios with concrete examples.
3. For $f(x,y) = x^4 + y^4$, compute the Hessian at $(0,0)$. Why does Newton struggle there?
4. Compare per-iteration cost: GD is $O(n)$, Newton is $O(n^3)$. At what $n$ does Newton become impractical?
5. For Rosenbrock at $(-1, 1)$: compute $H$, check if PD, compute Newton step.
6. What is the relationship between Newton's method and the inverse function theorem?
7. If the Hessian is diagonal, what is the Newton step? (Element-wise $g_i / H_{ii}$.)
8. Why does damped Newton ($\alpha < 1$) help? Sketch the 1D case.

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Newton's method on a neural network loss</summary>

**Problem:** Consider a 2-layer network with 10 parameters. Compute the full Hessian and run Newton. Compare with GD + Adam. Newton should converge in very few steps — but what's the catch for networks with 10M parameters?

**Solution:** For 10 parameters, Newton converges in ~5-10 iterations while Adam takes ~100s. But for $n = 10^7$: $H$ is $10^7 \times 10^7 = 10^{14}$ entries = 800 TB. Solving $H\delta = -g$ is $O(n^3) = O(10^{21})$ FLOPs. This is why deep learning uses first-order methods (SGD, Adam) — Newton is information-optimal but computationally infeasible at scale.
</details>

<details>
<summary>🎯 Challenge 2: Cubic regularization of Newton</summary>

**Problem:** Instead of the quadratic model, use $m(\delta) = f + g^T\delta + \frac{1}{2}\delta^T H\delta + \frac{\sigma}{3}\|\delta\|^3$. Show that this method has $O(1/k^2)$ global convergence rate (better than GD's $O(1/k)$) and naturally handles indefinite Hessians.

**Solution:** The cubic term penalizes large steps, so even if $H$ is indefinite, the model has a unique minimum. The subproblem reduces to $(H + \lambda I)\delta = -g$ where $\lambda = \sigma\|\delta\|$ — this is the Levenberg-Marquardt form! The $O(1/k^2)$ rate was proven by Nesterov & Polyak (2006). The method is "ARC" (Adaptive Regularization with Cubics). Key insight: the cubic model is a better local approximation than quadratic when far from the solution.
</details>

<details>
<summary>🎯 Challenge 3: Hessian-free Newton (Newton-CG)</summary>

**Problem:** Instead of forming $H$ explicitly, use CG to solve $H\delta = -g$ using only Hessian-vector products $Hv$. Implement Newton-CG where $Hv$ is computed via AD. Compare memory: full Newton $O(n^2)$ vs Newton-CG $O(n)$.

**Solution:** $Hv$ can be computed as $\nabla((\nabla f)^T v)$ — a gradient of a scalar, which AD handles in $O(n)$ time. Run CG for $k \ll n$ steps (truncated Newton). Early termination when negative curvature detected: $d^T H d < 0$ → use current CG direction as is. This scales Newton to $n = 10^6+$ variables. Libraries: `jax.jvp` for forward-mode Hv, `scipy.sparse.linalg.LinearOperator` + `cg` for the solve.
</details>

---

## Self-Assessment Checklist

- [ ] I can implement Newton's method and verify quadratic convergence
- [ ] I know when Newton fails (indefinite $H$, far from solution, large $n$)
- [ ] I can implement modified Newton with $H + \mu I$ regularization
- [ ] I understand Newton with line search for global convergence

---

## Key Takeaways

1. **Newton's step** is $\delta = -H^{-1}g$ — solves the quadratic model exactly.
2. **Quadratic convergence** $\|e_{k+1}\| \leq C\|e_k\|^2$ — doubles correct digits each step.
3. **Fails** when $H \not\succ 0$ or far from solution — fix with regularization or line search.
4. **Cost** $O(n^3)$ per step — prohibitive for large problems, motivating quasi-Newton (Day 19).
