# Day 4: Hessians and Taylor Expansion

> **Phase I — Mathematical Foundations | Week 1 | 2.5 hours**
> *The Hessian is the curvature map. Newton's method uses it to jump straight to the bottom of the bowl.*

## Navigation
- **Previous:** [Day 3: Gradients and Jacobians](day-03-gradients-jacobians.md)
- **Next:** [Day 5: Convexity](day-05-convexity.md)
- **Week:** [Week 1 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Every Gauss-Newton and Levenberg-Marquardt step in SLAM approximates the Hessian via $J^T J$. The quality of this approximation determines convergence speed. Taylor expansion is how we linearize nonlinear sensor models.

---

## Theory (45 min)

### 4.1 The Hessian Matrix

For $f: \mathbb{R}^n \to \mathbb{R}$:
$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

$$H(x) = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\
\vdots & & \ddots
\end{pmatrix}$$

**Symmetry:** If second derivatives are continuous (Schwarz's theorem), $H = H^T$.

### 4.2 Second-Order Taylor Expansion

$$f(x + \delta) \approx f(x) + \nabla f(x)^T \delta + \frac{1}{2} \delta^T H(x) \delta$$

This is a **quadratic model** of $f$ around $x$. It's exact for quadratic functions, and a good approximation nearby for smooth functions.

### 4.3 The Newton Step

Minimizing the quadratic model with respect to $\delta$:
$$\nabla_\delta \left[ f(x) + \nabla f^T \delta + \frac{1}{2} \delta^T H \delta \right] = \nabla f + H\delta = 0$$
$$\boxed{\delta^* = -H^{-1} \nabla f}$$

This is **Newton's step** — the theoretical foundation of most second-order optimization methods.

### 4.4 Hessians of Common Forms

| Function | Hessian |
|----------|---------|
| $f(x) = a^T x + c$ | $H = 0$ |
| $f(x) = x^T A x$ | $H = A + A^T$ (= $2A$ if symmetric) |
| $f(x) = \|Ax - b\|^2$ | $H = 2A^T A$ (always PSD!) |

**Key insight for NLS:** The Hessian of a least-squares cost $\sum r_i(x)^2$ has the form $H \approx 2J^T J$ (Gauss-Newton approximation), which ignores second-order residual terms. This works well when residuals are small.

---

## Implementation (60 min)

### Exercise 1: Hessian of Rosenbrock
Compute analytically, then verify numerically.

### Exercise 2: Numerical Hessian
Implement via finite differences of the gradient.

### Exercise 3: Classify Critical Points
Evaluate Hessian at critical points — PD means minimum, indefinite means saddle.

### Exercise 4: Taylor Approximation Quality
Plot actual function vs quadratic approximation at several points.

### Exercise 5: Newton Step Visualization
Show the Newton step jumping toward the minimum on a contour plot.

```python
# See: code/week01/hessians_taylor.py
```

---

## Practice (45 min)

Compute Hessians by hand and classify the critical point:

1. $f(x,y) = x^2 + y^2$ at $(0,0)$ → PD (minimum)
2. $f(x,y) = x^2 - y^2$ at $(0,0)$ → indefinite (saddle)
3. $f(x,y) = (x+y)^2$ at $(0,0)$ → PSD (degenerate)
4. $f(x,y) = x^4 + y^4 - 2x^2 - 2y^2$ — find ALL critical points, classify each
5. **Rosenbrock** $f(x,y) = (1-x)^2 + 100(y-x^2)^2$ at $(1,1)$ — verify minimum
6. $f(x,y,z) = x^2 + 2y^2 + 3z^2 + xy$ — classify origin

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Gauss-Newton Hessian approximation</summary>

**Problem:** For $f(x) = \sum_i r_i(x)^2$, show that $H = 2(J^T J + \sum_i r_i \nabla^2 r_i)$ where $J$ is the Jacobian of residuals. Why is $H \approx 2J^T J$ a good approximation?

**Solution:** Differentiating: $\nabla f = 2J^T r$, $H = 2(J^T J + \sum_i r_i H_i)$. The Gauss-Newton approximation drops $\sum_i r_i H_i$ because: (1) at the solution, residuals $r_i$ are small (good fit), so these terms are small; (2) $J^T J$ is always PSD, so the approximate Hessian is always PSD, ensuring descent. This is why GN is so effective for well-fitting models.
</details>

<details>
<summary>🎯 Challenge 2: Why Newton fails on non-convex problems</summary>

**Problem:** Apply Newton's method to $f(x) = x^4$ starting at $x_0 = 1$. What happens?

**Solution:** $f'(x) = 4x^3$, $f''(x) = 12x^2$. Newton step: $\delta = -f'/f'' = -4x^3/(12x^2) = -x/3$. So $x_{k+1} = x_k - x_k/3 = 2x_k/3$. The method converges, but only linearly (rate 2/3), not quadratically! This is because the Hessian vanishes at the minimum ($f''(0) = 0$), so the quadratic model becomes degenerate. Newton converges quadratically only when the Hessian at the minimum is nonsingular.
</details>

<details>
<summary>🎯 Challenge 3: Numerical Hessian accuracy</summary>

**Problem:** Derive the optimal step size $h$ for computing the Hessian via finite differences of the gradient, considering both truncation and rounding error.

**Solution:** The Hessian approximation $H_{ij} \approx [\nabla_j f(x + h e_i) - \nabla_j f(x - h e_i)] / 2h$ has truncation error $O(h^2)$ and rounding error $O(\epsilon/h^2)$ (where $\epsilon \sim 10^{-16}$). Balancing: $h^2 \sim \epsilon/h^2$ gives $h \sim \epsilon^{1/4} \approx 10^{-4}$ for float64.
</details>

---

## Self-Assessment Checklist

- [ ] I can compute the Hessian by hand for polynomial and trigonometric functions
- [ ] I understand the Taylor quadratic model and where it breaks down
- [ ] I can derive the Newton step from the quadratic model
- [ ] I see why Gauss-Newton drops the second-order residual terms

---

## Key Takeaways

1. **Hessian = curvature.** PD Hessian = upward-curving bowl = minimum.
2. **Taylor expansion is the basis of all iterative optimization** — approximate locally, step, repeat.
3. **Newton's step** $\delta = -H^{-1}\nabla f$ uses curvature to predict the minimum location.
4. **GN approximation** $H \approx 2J^T J$ avoids computing second derivatives and guarantees PSD.
