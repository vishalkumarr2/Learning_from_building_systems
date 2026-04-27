# Day 3: Multivariable Calculus — Gradients and Jacobians

> **Phase I — Mathematical Foundations | Week 1 | 2.5 hours**
> *The gradient points uphill. Optimization is the art of going downhill efficiently.*

## Navigation
- **Previous:** [Day 2: Eigenvalues, PD, SVD](day-02-eigenvalues-svd.md)
- **Next:** [Day 4: Hessians and Taylor Expansion](day-04-hessians-taylor.md)
- **Week:** [Week 1 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Every optimization step in SLAM, calibration, and MPC requires computing gradients and Jacobians. The Jacobian of the sensor model maps state perturbations to measurement changes — it's the linearization that makes nonlinear problems tractable.

---

## Theory (45 min)

### 3.1 Gradient Vector

For $f: \mathbb{R}^n \to \mathbb{R}$:
$$\nabla f(x) = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}$$

**Key property:** $\nabla f(x)$ points in the direction of steepest ascent. The magnitude $\|\nabla f\|$ is the steepness.

**Directional derivative:** $D_v f(x) = \nabla f(x) \cdot v$ — rate of change along direction $v$.

### 3.2 Jacobian Matrix

For $f: \mathbb{R}^n \to \mathbb{R}^m$:
$$J_f(x) = \begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix} \in \mathbb{R}^{m \times n}$$

Row $i$ = gradient of $f_i$. Column $j$ = how all outputs change when $x_j$ changes.

### 3.3 Chain Rule for Composed Functions

For $h(x) = f(g(x))$:
$$J_h(x) = J_f(g(x)) \cdot J_g(x)$$

This is fundamental to: backpropagation, AD, and computing gradients of complex residuals.

### 3.4 Gradient of Common Forms

| Function | Gradient |
|----------|----------|
| $f(x) = a^T x$ | $\nabla f = a$ |
| $f(x) = x^T A x$ | $\nabla f = (A + A^T)x$ |
| $f(x) = \|Ax - b\|^2$ | $\nabla f = 2A^T(Ax - b)$ |
| $f(x) = \log \det(X)$ | $\nabla f = X^{-T}$ |

---

## Implementation (60 min)

### Exercise 1: Symbolic Gradient Verification
```python
# See: code/week01/gradients_jacobians.py
```
Compute gradients by hand, verify with SymPy.

### Exercise 2: Jacobian of a Vector Function
Compute the Jacobian of $f(x,y,z) = [x^2+y, \; yz, \; xz^2]$ by hand, verify with SymPy.

### Exercise 3: Numerical Gradient via Finite Differences
Implement the central difference gradient:
$$\frac{\partial f}{\partial x_i} \approx \frac{f(x + h e_i) - f(x - h e_i)}{2h}$$

### Exercise 4: Numerical vs Analytic Gradient — Error Analysis
Compare numerical and analytic gradients. Plot error vs step size $h$ to see the truncation/rounding tradeoff.

---

## Practice (45 min)

Compute gradients by hand (verify with code):

1. $f(x,y) = x^2 y + \sin(xy)$
2. $f(x,y) = e^{x+2y}$
3. $f(x,y,z) = x^2 + 2y^2 + 3z^2 + 4xy$
4. $f(x) = \log(1 + e^x)$ (softplus — appears in logistic regression)
5. $f(x,y) = \sqrt{x^2 + y^2}$ (distance to origin — gradient magnitude?)
6. **Rosenbrock:** $f(x,y) = (1-x)^2 + 100(y-x^2)^2$ — the classic test function
7. $f(x) = x^T A x + b^T x + c$ — generic quadratic
8. Chain rule: gradient of $g(h(x))$ where $h: \mathbb{R}^3 \to \mathbb{R}^2$ and $g: \mathbb{R}^2 \to \mathbb{R}$

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Gradient of a norm</summary>

**Problem:** Compute $\nabla_x \|Ax - b\|_2$ (not the squared norm!). What happens at $Ax = b$?

**Solution:** Using chain rule: $\nabla_x \|Ax - b\|_2 = \frac{A^T(Ax - b)}{\|Ax - b\|_2}$. At $Ax = b$, the gradient is undefined (0/0) — the norm is not differentiable at zero. This is why we usually minimize $\|Ax - b\|^2$ instead (differentiable everywhere). Using the non-squared norm leads to $\ell_1$-like behavior and requires subgradient methods.
</details>

<details>
<summary>🎯 Challenge 2: Jacobian sparsity in SLAM</summary>

**Problem:** In a pose-graph SLAM with $n$ poses and $m$ relative-pose measurements, what is the sparsity pattern of the measurement Jacobian? Why does this matter for efficiency?

**Solution:** Each measurement only involves 2 poses, so each row of the Jacobian has at most $2 \times \text{pose\_dim}$ nonzero entries out of $n \times \text{pose\_dim}$ columns. The Jacobian is extremely sparse. The normal equations $J^T J$ inherit this sparsity (block structure). Exploiting this sparsity is why Ceres, g2o, and GTSAM can solve thousand-pose problems in milliseconds.
</details>

<details>
<summary>🎯 Challenge 3: Gradient checker design</summary>

**Problem:** Design a robust gradient checker that works for functions with very small gradients (where relative error is meaningless). What should the comparison metric be?

**Solution:** Use the relative error formula: $\frac{|g_\text{analytic} - g_\text{numerical}|}{|g_\text{analytic}| + |g_\text{numerical}| + \epsilon}$ where $\epsilon \sim 10^{-7}$. The denominator prevents division by near-zero. Threshold: error < $10^{-5}$ for float64 with central differences. Use $h \approx \epsilon^{1/3} \approx 10^{-5}$ for central differences. Run at multiple random points — a single point might be at a saddle where gradients are zero anyway.
</details>

---

## Self-Assessment Checklist

- [ ] I can compute gradients and Jacobians by hand for polynomial, trig, and exponential functions
- [ ] I can implement a numerical gradient checker and choose the right step size $h$
- [ ] I understand the chain rule for composed functions (Jacobian multiplication)
- [ ] I can write down the gradient of $\|Ax - b\|^2$ from memory

---

## Key Takeaways

1. **Gradient = direction of steepest ascent.** Negate it → steepest descent.
2. **Jacobian linearizes** a nonlinear function — the entire optimization toolkit depends on this.
3. **Chain rule as matrix multiplication** — this is what AD automates.
4. **Always check your analytic gradient** with finite differences before trusting it.
