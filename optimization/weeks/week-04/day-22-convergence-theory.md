# Day 22: Convergence Theory Deep Dive

> **Phase II — Unconstrained Optimization | Week 4 | 2.5 hours**
> *Formal definitions of convergence speed — the language for comparing methods.*

## Navigation
- **Previous:** [Day 21: Week 3 Review](../week-03/day-21-week-review.md)
- **Next:** [Day 23: Momentum Methods](day-23-momentum.md)
- **Week:** [Week 4 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
When the Ceres solver log shows "cost: 1.4e+03 → 2.1e-01 in 12 iterations," understanding convergence rates tells you whether those 12 iterations are good (near quadratic) or a sign of trouble (linear with $\kappa = 10^6$). Convergence theory also explains *why* trust region methods converge globally while pure Newton can diverge — crucial for solver configuration.

---

## Theory (60 min)

### 22.1 Convergence Order Definitions

Given a sequence $\{x_k\}$ converging to $x^*$:

**Linear convergence (Q-linear):**
$$\|x_{k+1} - x^*\| \leq C \|x_k - x^*\|, \quad C \in (0, 1)$$

**Superlinear convergence (Q-superlinear):**
$$\lim_{k\to\infty} \frac{\|x_{k+1} - x^*\|}{\|x_k - x^*\|} = 0$$

**Quadratic convergence (Q-quadratic):**
$$\|x_{k+1} - x^*\| \leq C \|x_k - x^*\|^2$$

**R-rate (root-convergence):** The "average" rate. $\{x_k\}$ converges R-linearly at rate $\sigma$ if $\|x_k - x^*\| \leq M \sigma^k$ for some $M > 0$.

### 22.2 Method-Specific Rates

| Method | Rate | Constant | Condition |
|--------|------|----------|-----------|
| GD (quadratic) | Linear | $C = \frac{\kappa - 1}{\kappa + 1}$ | $\kappa = \lambda_{\max}/\lambda_{\min}$ |
| GD (strongly convex) | Linear | $C = 1 - \frac{2\mu L}{(\mu + L)^2}$ | $\mu$-strong, $L$-smooth |
| Newton (local) | Quadratic | $C = \frac{L_H}{2\mu}$ | $H$ Lipschitz, $\mu$-strong |
| BFGS (local) | Superlinear | $\to 0$ | Dennis-Moré condition |
| CG ($n$-dim quad) | $n$-step | — | Exact for quadratics |

### 22.3 GD Convergence on Quadratics

For $f(x) = \frac{1}{2}x^TAx - b^Tx$ with eigenvalues $\lambda_1 \leq \cdots \leq \lambda_n$:

$$f(x_k) - f(x^*) \leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^{2k} (f(x_0) - f(x^*))$$

where $\kappa = \lambda_n/\lambda_1$.

**Example:** $\kappa = 100 \Rightarrow$ convergence factor = $0.98^{2k}$. Need $k \approx 230$ to reduce error by $10^{-2}$.

### 22.4 Zoutendijk's Theorem

If $f$ is bounded below, $\nabla f$ is Lipschitz continuous, and we use sufficient decrease + angle condition:

$$\sum_{k=0}^{\infty} \|\nabla f_k\|^2 \cos^2 \theta_k < \infty$$

where $\theta_k$ is the angle between $d_k$ and $-\nabla f_k$.

**Consequence:** For any method with $\cos\theta_k > c > 0$ (bounded away from $90°$), $\nabla f_k \to 0$. This is *global convergence* — the method converges to *a* critical point.

### 22.5 Local vs Global Convergence

- **Global convergence:** Any starting point → converges to *a* critical point (may be saddle)
- **Local convergence:** If you start close enough to $x^*$ → converges to $x^*$ with known rate
- Newton: quadratic local rate but no global guarantee without modifications
- Trust region: global convergence + fast local rate = best of both

---

## Implementation (60 min)

### Exercise 1: Empirical Convergence Rate Measurement
```python
# See: code/week04/convergence_theory.py
```
Run GD, Newton, BFGS on a quadratic. Plot $\log\|x_k - x^*\|$ vs $k$. Fit the convergence order $p$.

### Exercise 2: Condition Number Effect
Vary $\kappa \in \{10, 100, 1000, 10000\}$ and measure GD iterations. Verify the linear rate depends on $\kappa$.

### Exercise 3: Zoutendijk Verification
Run GD with Armijo line search. Plot the cumulative sum $\sum_{k=0}^K \|\nabla f_k\|^2$ — it should converge (finite sum).

### Exercise 4: Superlinear Verification for BFGS
Run BFGS. Compute ratios $\|x_{k+1}-x^*\| / \|x_k-x^*\|$ and show they $\to 0$ (superlinear).

---

## Practice (45 min)

1. If $\|x_{k+1} - x^*\| = 0.5\|x_k - x^*\|$ and $\|x_0 - x^*\| = 1$, how many iterations for $\|x_k - x^*\| < 10^{-8}$?
2. If Newton converges quadratically with $C = 1$ and $\|x_0 - x^*\| = 0.1$, how many iterations for $10^{-16}$?
3. Why can't GD achieve superlinear convergence? (Hint: fixed direction approximation.)
4. Prove: quadratic convergence implies $\|x_k - x^*\| \leq C^{2^k - 1} \|x_0 - x^*\|^{2^k}$.
5. What is the convergence rate of Newton on $f(x) = x^4$ near $x^* = 0$? (Careful — $\nabla^2 f(0) = 0$!)
6. How does preconditioning affect GD convergence? (Hint: effective $\kappa$.)
7. Explain the Dennis-Moré condition for BFGS superlinear convergence.
8. When Ceres reports "TR: cost 1.4e3 → 2.1e-1 in 12 iterations," estimate the convergence rate.

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Convergence order estimator</summary>

**Problem:** Build a tool that automatically estimates convergence order from a sequence $\{e_k\}$. Use the ratio $e_{k+1}/e_k^p$ and find $p$ via least squares on $\log e_{k+1} \approx p \log e_k + \log C$. Apply to GD, BFGS, Newton and verify theoretical rates.

**Solution:** Plot $\log e_{k+1}$ vs $\log e_k$ — the slope is the convergence order. For GD: slope ≈ 1.0 (linear). Newton: slope ≈ 2.0 (quadratic). BFGS: slope starts ~1.0 then increases toward 2.0 (superlinear — the slope isn't constant). Discard early iterates (global vs local phase).
</details>

<details>
<summary>🎯 Challenge 2: Convergence on degenerate problems</summary>

**Problem:** Study convergence on $f(x, y) = x^2 + y^4$ near $(0, 0)$. Newton should be slower than expected because $\nabla^2 f(0,0)$ is singular. How does each method behave? What is the actual convergence rate of Newton here?

**Solution:** Newton on $f = y^4$ gives $y_{k+1} = y_k - y_k^4/(4y_k^3) = y_k - y_k/4 = 3y_k/4$ — only linear, not quadratic! The Hessian is singular at the solution. Regularized Newton (LM-style) helps: $(H + \mu I)\delta = -g$ with small $\mu$ restores superlinear convergence. This is exactly why Ceres uses LM damping.
</details>

<details>
<summary>🎯 Challenge 3: Complexity lower bounds</summary>

**Problem:** For the class of $L$-smooth, $\mu$-strongly convex functions, the worst-case bound for any first-order method is $O(\sqrt{\kappa}\log(1/\varepsilon))$ (Nesterov's lower bound). Verify numerically: construct the worst-case function (a rotated quadratic with eigenvalues $\mu$ and $L$) and show GD needs $O(\kappa)$ while optimal methods need $O(\sqrt{\kappa})$.

**Solution:** The worst case for GD is a 2D quadratic with eigenvalues $\mu$ and $L$: GD zigzags, taking $O(\kappa)$ steps. Nesterov achieves $O(\sqrt{\kappa})$. For $\kappa = 10000$: GD needs ~10000 steps, Nesterov needs ~100. This is the fundamental justification for momentum methods (Day 23).
</details>

---

## Self-Assessment Checklist

- [ ] I can define linear, superlinear, and quadratic convergence precisely
- [ ] I can compute the convergence rate constant for GD on a quadratic
- [ ] I understand Zoutendijk's theorem and what "global convergence" means
- [ ] I can empirically measure convergence order from iteration data

---

## Key Takeaways

1. **Linear** ($C < 1$), **superlinear** ($C_k \to 0$), **quadratic** ($p = 2$) — these classify all methods.
2. **GD's rate** depends on $\kappa$ — bad for ill-conditioned problems.
3. **Zoutendijk guarantees** any reasonable method converges to a critical point.
4. **Local vs global:** Trust region gives both; Newton gives only local.
