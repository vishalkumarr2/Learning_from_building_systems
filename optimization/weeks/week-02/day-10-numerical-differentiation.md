# Day 10: Numerical Differentiation Deep Dive

> **Phase I — Mathematical Foundations | Week 2 | 2.5 hours**
> *Trust but verify: every hand-derived Jacobian should be checked numerically.*

## Navigation
- **Previous:** [Day 9: Sparse Matrices & Schur Complement](day-09-sparse-schur.md)
- **Next:** [Day 11: Automatic Differentiation](day-11-automatic-differentiation.md)
- **Week:** [Week 2 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Ceres provides `NumericDiffCostFunction` as a fallback when analytic or auto-diff Jacobians aren't available. Understanding the accuracy limits of numerical differentiation tells you why Ceres auto-diff is almost always preferred, and why the numerical Jacobian checker is indispensable for debugging.

---

## Theory (45 min)

### 10.1 Forward Difference

$$f'(x) \approx \frac{f(x+h) - f(x)}{h} + O(h)$$

First-order accurate. Error has two components:
- **Truncation error:** $O(h)$ — decreases as $h \to 0$
- **Rounding error:** $O(\epsilon_{\text{mach}} / h)$ — increases as $h \to 0$

### 10.2 Central Difference

$$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h} + O(h^2)$$

Second-order accurate. Two function evaluations per component.

### 10.3 Complex-Step Derivative

$$f'(x) = \frac{\text{Im}(f(x + ih))}{h} + O(h^2)$$

**No cancellation error!** Works because $f(x+ih) = f(x) + ihf'(x) - \frac{h^2}{2}f''(x) + \ldots$

The imaginary part gives $hf'(x) + O(h^3)$. Since there's no subtraction of nearly-equal numbers, $h$ can be tiny ($10^{-30}$) without rounding problems.

**Limitation:** Requires $f$ to be implementable with complex arithmetic (no `abs`, `max`, boolean branches on values).

### 10.4 Optimal Step Size

| Method | Optimal $h$ | Achievable error |
|--------|------------|-----------------|
| Forward | $\sqrt{\epsilon_{\text{mach}}} \approx 10^{-8}$ | $\sqrt{\epsilon} \approx 10^{-8}$ |
| Central | $\epsilon^{1/3} \approx 10^{-5.3}$ | $\epsilon^{2/3} \approx 10^{-10.7}$ |
| Complex-step | As small as you want ($10^{-30}$) | Machine precision $\approx 10^{-16}$ |

### 10.5 Jacobian Computation

For $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian $J \in \mathbb{R}^{m \times n}$ requires:
- **Forward/central:** $n$ (or $2n$) function evaluations (perturb each input dimension)
- **Cost:** $O(mn)$ per function evaluation → total $O(mn^2)$

This is why AD (Day 11) is preferred for large $n$.

---

## Implementation (60 min)

### Exercise 1: Error vs Step Size
```python
# See: code/week02/numerical_differentiation.py
```
Plot error vs $h$ for forward, central, and complex-step on $f(x) = \sin(x)$. Show the V-shaped error curve.

### Exercise 2: Find Optimal $h$ Experimentally
For each method, sweep $h$ from $10^{-1}$ to $10^{-16}$ and find the $h$ that minimizes error.

### Exercise 3: Complex-Step Jacobian
Implement a complex-step Jacobian for vector functions $f: \mathbb{R}^n \to \mathbb{R}^m$.

### Exercise 4: When Rounding Dominates
Test all three methods on $f(x) = x + 10^{-12}$ near $x=1$. The gradient is 1, but numerical methods struggle when $f(x)$ values are very close.

---

## Practice (45 min)

1. Derive the truncation error for the central difference formula using Taylor expansion.
2. Why can't you use $h = 10^{-100}$ with forward differences but you can with complex-step?
3. What is the cost of computing a numerical Jacobian for $f: \mathbb{R}^{1000} \to \mathbb{R}^{50}$ using central differences?
4. Ceres `NumericDiffCostFunction` uses central differences by default. What step size does it use?
5. You derived a Jacobian by hand and want to verify it. Write the pseudocode for a gradient checker.
6. Why does the complex-step method fail on $f(x) = |x|$?
7. For a function that takes 1 second to evaluate with 500 parameters, how long would a numerical Jacobian take?
8. Higher-order finite differences: derive the 4th-order central difference formula. What's its optimal $h$?

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Automatic step size selection</summary>

**Problem:** Implement a gradient checker that automatically selects the optimal step size $h$ for each component by monitoring the error vs $h$ curve. Start with $h = 10^{-1}$ and halve until the error starts increasing (rounding regime).

**Solution:** For each component $i$: compute forward differences with $h, h/2, h/4, \ldots$. The sequence of derivative estimates converges, then diverges as rounding takes over. The optimal $h$ is where the Richardson extrapolation error estimate $|D(h) - D(h/2)|$ is minimized. In practice, starting with $h = |x_i| \cdot 10^{-4}$ (or $10^{-4}$ if $x_i = 0$) and using central differences works well. Ceres uses `epsilon = sqrt(machine_epsilon) * max(abs(x_i), 1)`.
</details>

<details>
<summary>🎯 Challenge 2: Second derivatives via finite differences</summary>

**Problem:** Compute the Hessian numerically. (a) Use the formula $H_{ij} \approx \frac{f(x+h e_i+h e_j)-f(x+h e_i)-f(x+h e_j)+f(x)}{h^2}$. (b) What is its error order? (c) How many function evaluations for the full Hessian? (d) Can you use the complex-step trick?

**Solution:** (a) This is the second-order forward difference for mixed partials. (b) Error is $O(h)$ — first-order only. For $O(h^2)$, use the central formula: $\frac{f(x+h e_i+h e_j)-f(x+h e_i-h e_j)-f(x-h e_i+h e_j)+f(x-h e_i-h e_j)}{4h^2}$. (c) Forward: $n(n+1)/2 + 1$ evaluations. Central: $2n(n+1)$ evaluations. (d) Complex-step can compute the Hessian using a "hyper-dual" number approach or by applying complex-step to the gradient (which itself can be computed analytically).
</details>

<details>
<summary>🎯 Challenge 3: Verifying a Ceres cost function Jacobian</summary>

**Problem:** You've written a Ceres cost function with an analytic Jacobian. Before deploying, you want to verify it. Design a test that: (1) evaluates the analytic Jacobian, (2) computes a numerical Jacobian, (3) reports the maximum relative error, (4) flags components where the error exceeds a threshold.

**Solution:** In Ceres, use `ceres::GradientChecker` which does exactly this. In Python, the pattern is: for each parameter block, perturb each element by $h = \sqrt{\epsilon} \cdot \max(|x_i|, 1)$, compute central differences, compare to the analytic Jacobian. Report per-element relative errors $|J_a - J_n| / \max(|J_a|, |J_n|, 1)$. Flag if any exceed $10^{-4}$ (for central differences). This is the single most important debugging tool for optimization — a wrong Jacobian is the #1 cause of solver failure.
</details>

---

## Self-Assessment Checklist

- [ ] I can implement forward, central, and complex-step derivatives
- [ ] I understand the truncation vs rounding tradeoff and optimal $h$
- [ ] I can implement a gradient checker for vector functions
- [ ] I know when numerical differentiation fails and why AD is preferred

---

## Key Takeaways

1. **Central difference** ($O(h^2)$) is the workhorse — used by Ceres for gradient checking.
2. **Complex-step** eliminates cancellation error — use it when the function supports complex inputs.
3. **Optimal $h$** exists because of the truncation-rounding tradeoff — $10^{-8}$ for forward, $10^{-5}$ for central.
4. **Always verify analytic Jacobians numerically** — this catches more bugs than any other practice.
