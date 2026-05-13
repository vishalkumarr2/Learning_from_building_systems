# Day 16: Line Search Methods

> **Phase II — Unconstrained Optimization | Week 3 | 2.5 hours**
> *Choosing the right step size is half the battle.*

## Navigation
- **Previous:** [Day 15: Gradient Descent](day-15-gradient-descent.md)
- **Next:** [Day 17: Newton's Method](day-17-newtons-method.md)
- **Week:** [Week 3 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Every Ceres solver iteration uses a line search or trust region to pick the step size. The Wolfe conditions guarantee convergence for BFGS — which is why Ceres requires them internally when using the LINE_SEARCH minimizer. Understanding Armijo vs Wolfe helps you debug cases where the solver stalls: often a line-search failure is the root cause of "solver did not converge."

---

## Theory (45 min)

### 16.1 Exact Line Search

Given a search direction $d_k$, the ideal step:

$$\alpha^* = \arg\min_{\alpha > 0} f(x_k + \alpha d_k)$$

This is a 1D optimization — solvable for quadratics, but impractical in general.

### 16.2 Armijo Condition (Sufficient Decrease)

Accept $\alpha$ if:

$$f(x_k + \alpha d_k) \leq f(x_k) + c_1 \alpha \nabla f_k^T d_k$$

where $c_1 \in (0, 1)$, typically $c_1 = 10^{-4}$.

**Interpretation:** The function must decrease by at least a fraction of the predicted decrease from the linear model.

### 16.3 Curvature Condition

$$\nabla f(x_k + \alpha d_k)^T d_k \geq c_2 \nabla f_k^T d_k$$

where $c_2 \in (c_1, 1)$, typically $c_2 = 0.9$.

**Interpretation:** The gradient along $d_k$ must have flattened enough — prevents stopping too early.

### 16.4 Wolfe Conditions

**Wolfe** = Armijo + curvature:

$$f(x_k + \alpha d_k) \leq f(x_k) + c_1 \alpha \nabla f_k^T d_k$$
$$\nabla f(x_k + \alpha d_k)^T d_k \geq c_2 \nabla f_k^T d_k$$

**Strong Wolfe** = Armijo + strong curvature (absolute value):

$$|\nabla f(x_k + \alpha d_k)^T d_k| \leq c_2 |\nabla f_k^T d_k|$$

### 16.5 Backtracking Algorithm

```
α ← α₀  (e.g., 1.0)
while f(x + αd) > f(x) + c₁ α ∇f^T d:
    α ← ρ α     (ρ ∈ (0,1), e.g., 0.5)
return α
```

Simple, cheap, guaranteed to terminate for any descent direction.

### 16.6 Zoom Algorithm (Nocedal & Wright)

For Wolfe conditions — a more sophisticated two-phase algorithm:
1. **Bracket:** Find interval $[\alpha_{\text{lo}}, \alpha_{\text{hi}}]$ containing an acceptable step
2. **Zoom:** Bisect or interpolate within the bracket until Wolfe conditions hold

### 16.7 Why Wolfe Matters

- **Armijo alone** is sufficient for GD convergence
- **Wolfe conditions** are needed for BFGS convergence (secant condition requires curvature)
- **Strong Wolfe** prevents pathological cases in conjugate gradient

---

## Implementation (60 min)

### Exercise 1: Backtracking Line Search
```python
# See: code/week03/line_search.py
```
Implement backtracking with Armijo condition. Test parameter sensitivity ($c_1$, $\rho$, $\alpha_0$).

### Exercise 2: Wolfe Line Search
Implement the zoom algorithm from Nocedal & Wright, Algorithm 3.5/3.6.

### Exercise 3: Comparison
Compare fixed step vs Armijo vs Wolfe on Rosenbrock.

### Exercise 4: Pathological Case
Show where Armijo alone gives bad step sizes (too short), but Wolfe corrects it.

### Exercise 5: Visualization
For a 1D slice along the search direction: plot $\phi(\alpha) = f(x + \alpha d)$ with Armijo/Wolfe acceptance regions marked.

---

## Practice (45 min)

1. **Prove:** Armijo + backtracking always terminates given a descent direction and $f$ bounded below.
2. **Parameter sensitivity:** What happens with $c_1 = 0.5$ (too strict) vs $c_1 = 10^{-8}$ (too loose)?
3. Show that $c_1 < c_2$ is required for the Wolfe conditions to be satisfiable.
4. For quadratic $f(x) = \frac{1}{2}x^TAx$, derive the exact line search step: $\alpha^* = \frac{g^Tg}{g^TAg}$.
5. Why does backtracking start with $\alpha = 1$ for Newton's method? (Because Newton's step is unit-length optimal.)
6. How many function evaluations does backtracking need per iteration? (1 + number of shrinks.)
7. Wolfe line search needs gradient evaluations — is it worth the cost?
8. Draw the 1D plot showing acceptable Armijo region, Wolfe region, and strong Wolfe region.

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Quadratic interpolation line search</summary>

**Problem:** Instead of halving $\alpha$, use the current function values to fit a quadratic and find its minimum. Implement this and compare convergence speed vs simple backtracking.

**Solution:** Given $\phi(0) = f(x_k)$, $\phi'(0) = \nabla f_k^T d_k$, and $\phi(\alpha_0) = f(x_k + \alpha_0 d_k)$, fit quadratic $\phi(\alpha) \approx a\alpha^2 + b\alpha + c$. Then $\alpha_{\text{new}} = -b/(2a) = -\phi'(0) \alpha_0^2 / (2(\phi(\alpha_0) - \phi(0) - \phi'(0)\alpha_0))$. Safeguard: ensure $\alpha_{\text{new}} \in [0.1\alpha_0, 0.5\alpha_0]$. This typically needs fewer iterations than simple halving because it uses curvature information.
</details>

<details>
<summary>🎯 Challenge 2: Goldstein conditions</summary>

**Problem:** An alternative to Wolfe: the **Goldstein conditions** require $f(x + \alpha d)$ to lie between two linear bounds. State them and implement. When are they preferred over Wolfe?

**Solution:** Goldstein conditions: $(1 - c) \alpha \nabla f^T d \leq f(x + \alpha d) - f(x) \leq c \alpha \nabla f^T d$ with $0 < c < 1/2$. The lower bound prevents $\alpha$ from being too small, the upper bound (Armijo) prevents too large. Preferred for methods that don't need curvature info (GD, trust region). Not suitable for BFGS/CG because they don't guarantee the curvature condition needed for the secant update.
</details>

<details>
<summary>🎯 Challenge 3: Line search failure detection</summary>

**Problem:** In production code (Ceres, scipy), what happens when line search fails? Implement a line search that gracefully handles: (a) $f$ returning NaN/Inf, (b) gradient returning NaN, (c) step size underflow ($\alpha < 10^{-16}$). Return appropriate diagnostics.

**Solution:** Add guards: if $f(x + \alpha d)$ is NaN/Inf → shrink $\alpha$ aggressively ($\rho = 0.1$ instead of 0.5). If $\alpha < \alpha_{\min}$ → declare line search failure, return the current $x$ with a flag. If gradient is NaN → fall back to function-value-only line search (no curvature condition). Ceres does this internally — check `LineSearchMinimizer::DoLineSearch()` source. Scipy raises `LineSearchWarning`.
</details>

---

## Self-Assessment Checklist

- [ ] I can implement backtracking (Armijo) and Wolfe line searches
- [ ] I understand why Wolfe conditions are needed for BFGS convergence
- [ ] I know the zoom algorithm structure for strong Wolfe
- [ ] I can visualize and interpret line search acceptance regions

---

## Key Takeaways

1. **Backtracking (Armijo)** is the simplest line search — guaranteed to terminate, sufficient for GD.
2. **Wolfe conditions** add a curvature requirement — needed for BFGS and CG convergence guarantees.
3. **Strong Wolfe** = Armijo + $|g^Td| \leq c_2|g_0^Td|$ — most restrictive, best guarantees.
4. **Zoom algorithm** is the standard implementation for Wolfe — see Nocedal & Wright §3.5.
