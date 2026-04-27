# Day 15: Gradient Descent — The Workhorse

> **Phase II — Unconstrained Optimization | Week 3 | 2.5 hours**
> *Simple, universal, and the foundation for everything that follows.*

## Navigation
- **Previous:** [Day 14: Week 2 Review](../week-02/day-14-week-review.md)
- **Next:** [Day 16: Line Search Methods](day-16-line-search.md)
- **Week:** [Week 3 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Gradient descent underlies every optimizer you'll use. Ceres starts each LM iteration with a gradient computation $g = J^Tr$. Understanding why plain GD is slow on SLAM problems (ill-conditioned $J^TJ$, $\kappa \sim 10^6$) explains why LM/trust-region methods exist. The convergence rate $\rho = (\kappa - 1)/(\kappa + 1)$ predicts exactly how many iterations you'd waste.

---

## Phase II Gate — Self-Assessment (15 min)

Before starting Phase II, verify Phase I mastery. Score yourself:

1. Write the gradient of $f(x,y) = x^2 y + \sin(xy)$
2. For $f(x) = x^T A x$, what is the Hessian? When is $f$ convex?
3. What does $\kappa(A) = 10^6$ mean practically for solving $Ax = b$?
4. Name 3 differences between forward-mode and reverse-mode AD
5. Given a 1000×1000 sparse matrix with 5000 nonzeros, which format is best for matvec?
6. What is the normal equation for least squares? When does it fail?
7. Compute $\nabla f$ for $f = \|Ax - b\|^2$ in matrix notation
8. What is the Schur complement of a 2×2 block matrix?
9. How does `jax.grad` handle $f: \mathbb{R}^n \to \mathbb{R}$?
10. What is the condition for a critical point to be a local minimum?

**Score < 7/10?** Review the relevant Phase I day before continuing.

---

## Theory (45 min)

### 15.1 The Algorithm

$$x_{k+1} = x_k - \alpha_k \nabla f(x_k)$$

That's it. Move in the direction of steepest descent, with step size $\alpha_k$.

### 15.2 Fixed Step Size

For $L$-smooth functions ($\|\nabla f(x) - \nabla f(y)\| \leq L\|x-y\|$):

$$\alpha < \frac{2}{L}$$

guarantees convergence. For quadratic $f(x) = \frac{1}{2}x^TAx$, $L = \lambda_{\max}(A)$.

**Optimal fixed step:** $\alpha^* = \frac{2}{\lambda_{\max} + \lambda_{\min}}$

### 15.3 Convergence Rates

| Assumption | Rate | Iterations for $\epsilon$ accuracy |
|-----------|------|-------------------------------------|
| Convex | $O(1/k)$ | $O(1/\epsilon)$ |
| Strongly convex | $O(\rho^k)$, $\rho = \frac{\kappa-1}{\kappa+1}$ | $O(\kappa \log(1/\epsilon))$ |
| Quadratic | Exact: $O(\rho^k)$ | Proportional to $\kappa$ |

### 15.4 Why GD is Slow

For quadratic $f(x) = \frac{1}{2}x^TAx - b^Tx$:

- Gradient $\nabla f = Ax - b$
- Each GD step overshoots across the narrow direction and undershoots along the wide direction
- The trajectory **zig-zags** along the valley floor
- Number of iterations ∝ $\kappa(A)$

**Example:** $\kappa = 1000$ → need ~7000 iterations for $10^{-6}$ accuracy. Newton needs ~5.

### 15.5 Backtracking Line Search (Preview)

Instead of fixed $\alpha$, choose $\alpha_k$ to satisfy the **Armijo condition**:

$$f(x_k - \alpha \nabla f_k) \leq f(x_k) - c_1 \alpha \|\nabla f_k\|^2$$

Start with $\alpha = 1$ (or some large value) and halve until the condition holds. Typical $c_1 = 10^{-4}$.

---

## Implementation (60 min)

### Exercise 1: Fixed Step Size GD
```python
# See: code/week03/gradient_descent.py
```
Implement GD with fixed step size. Test on $f(x) = \frac{1}{2}x^TAx$ for various $\kappa$.

### Exercise 2: Backtracking Line Search
Implement GD with Armijo backtracking. Compare convergence with fixed step on Rosenbrock.

### Exercise 3: Standard Test Functions
Test on: (a) quadratic, (b) Rosenbrock $f = (1-x)^2 + 100(y-x^2)^2$, (c) log-sum-exp.

### Exercise 4: Convergence Plots
Plot $f(x_k) - f^*$ vs iteration for each test function and each step-size strategy.

### Exercise 5: Condition Number Scaling
Vary $\kappa \in \{1, 10, 100, 1000\}$: show GD iterations grow linearly with $\kappa$.

---

## Practice (45 min)

1. Derive the convergence rate for GD on $f(x) = \frac{1}{2}x^TAx$ (diagonalize and analyze each eigendirection).
2. $\kappa = 100$: predict iterations for $\epsilon = 10^{-6}$. Then run it and compare.
3. Why can't you just use a very large step size? What happens?
4. For Rosenbrock, what is the Lipschitz constant $L$ at $(0,0)$? At $(1,1)$?
5. If computing $\nabla f$ costs 10ms and you need $10^4$ iterations, what's the total wall time?
6. GD with momentum (heavy ball) will reduce the $\kappa$ dependence. Can you guess how?
7. Why does GD zig-zag? Draw the contour plot and gradient vectors.
8. Fixed step vs backtracking: which is cheaper per iteration? Which needs fewer iterations?

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Tight convergence analysis</summary>

**Problem:** For $f(x) = \frac{1}{2}x^T\text{diag}(\lambda_1, \lambda_2)x$ with $\lambda_1 = 1, \lambda_2 = \kappa$, show that GD with optimal step size $\alpha = 2/(\kappa + 1)$ satisfies $f(x_k) \leq \left(\frac{\kappa-1}{\kappa+1}\right)^{2k} f(x_0)$.

**Solution:** In each eigendirection: $x_k^{(i)} = (1 - \alpha\lambda_i)^k x_0^{(i)}$. With $\alpha = 2/(\kappa+1)$: for $\lambda_1 = 1$, ratio $= 1 - 2/(\kappa+1) = (\kappa-1)/(\kappa+1) \equiv \rho$. For $\lambda_2 = \kappa$, ratio $= 1 - 2\kappa/(\kappa+1) = -(\kappa-1)/(\kappa+1) = -\rho$. So $|\text{ratio}| = \rho$ in both directions, and $f(x_k)/f(x_0) \leq \rho^{2k}$.
</details>

<details>
<summary>🎯 Challenge 2: GD on non-smooth functions</summary>

**Problem:** Run GD on $f(x) = |x_1| + |x_2|$ (L1 norm, non-smooth at origin). What happens? Why? What's the fix?

**Solution:** The gradient $\nabla f = [\text{sign}(x_1), \text{sign}(x_2)]$ is discontinuous at the origin and magnitude-1 everywhere else. GD with fixed step oscillates across the origin without converging. With diminishing step size ($\alpha_k = 1/k$), it converges but slowly. The correct approach is the **proximal gradient method**: $x_{k+1} = \text{prox}_{\alpha\|\cdot\|_1}(x_k - \alpha \nabla g(x_k))$ where $g$ is the smooth part. The prox of L1 is soft-thresholding: $\text{prox}_{\alpha\|\cdot\|_1}(v)_i = \text{sign}(v_i)\max(|v_i| - \alpha, 0)$.
</details>

<details>
<summary>🎯 Challenge 3: Steepest descent in non-Euclidean norms</summary>

**Problem:** Standard GD uses the Euclidean norm for "steepest descent." Show that using the norm $\|v\|_P = \sqrt{v^TPv}$ where $P \succ 0$ gives the update $x_{k+1} = x_k - \alpha P^{-1}\nabla f(x_k)$. Choosing $P = H(x_k)$ recovers Newton's method.

**Solution:** Steepest descent in $\|\cdot\|_P$: $\Delta x = \arg\min_{\|v\|_P \leq 1} \nabla f^T v$. By Cauchy-Schwarz in $P$-norm: $\nabla f^T v \geq -\|\nabla f\|_{P^{-1}} \|v\|_P$. Equality at $v = -P^{-1}\nabla f / \|P^{-1}\nabla f\|_P$. So steepest descent direction is $-P^{-1}\nabla f$. With $P = H$: direction is $-H^{-1}\nabla f$ = Newton step. This is the **preconditioned gradient** — connecting Day 12's preconditioning to optimization.
</details>

---

## Self-Assessment Checklist

- [ ] I can implement GD with fixed step and backtracking line search
- [ ] I understand the convergence rate and its dependence on $\kappa$
- [ ] I can explain why GD zig-zags on ill-conditioned problems
- [ ] I know the Armijo condition and can implement backtracking

---

## Key Takeaways

1. **GD** is $x_{k+1} = x_k - \alpha \nabla f$ — simple but convergence depends on $\kappa$.
2. **Fixed step size** requires knowing $L$ (Lipschitz constant); **backtracking** adapts automatically.
3. **Iterations ∝ $\kappa$** for strongly convex functions — this motivates Newton and quasi-Newton methods.
4. **Backtracking (Armijo)** is the minimum viable line search — always use it over fixed step.
