# Day 5: Convexity — The Central Concept

> **Phase I — Mathematical Foundations | Week 1 | 2.5 hours**
> *If your problem is convex, every local minimum is THE global minimum. No other property matters more.*

## Navigation
- **Previous:** [Day 4: Hessians and Taylor Expansion](day-04-hessians-taylor.md)
- **Next:** [Day 6: Optimality Conditions](day-06-optimality-conditions.md)
- **Week:** [Week 1 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Least-squares problems (SLAM, calibration) are convex when the model is linear. When nonlinear, they're locally convex near the solution — which is why good initialization matters. MPC with linear dynamics + quadratic cost → convex QP (solvable in real-time).

---

## Theory (45 min)

### 5.1 Convex Sets

A set $S$ is **convex** if for all $x, y \in S$ and $\lambda \in [0, 1]$:
$$\lambda x + (1 - \lambda) y \in S$$

*Geometrically:* the line segment between any two points in $S$ lies entirely in $S$.

**Examples:** Hyperplanes, halfspaces, balls, ellipsoids, polyhedra, cones.
**Non-examples:** Star shapes, non-connected sets, donut shapes.

### 5.2 Convex Functions

$f: \mathbb{R}^n \to \mathbb{R}$ is **convex** if for all $x, y$ and $\lambda \in [0,1]$:
$$f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

*Geometrically:* the function lies below the chord between any two points.

### 5.3 Conditions for Convexity

| Condition | Statement |
|-----------|-----------|
| **Zeroth-order** | $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$ |
| **First-order** | $f(y) \geq f(x) + \nabla f(x)^T(y-x)$ (tangent lies below) |
| **Second-order** | $H(x) \succeq 0$ for all $x$ (Hessian is PSD everywhere) |

### 5.4 The Fundamental Theorem

> **For convex functions, every local minimum is a global minimum.**

This is why convexity is so powerful: you don't need to worry about getting trapped in local minima. Any descent method will find THE answer.

### 5.5 Operations Preserving Convexity

- **Non-negative sum:** $\alpha f + \beta g$ (for $\alpha, \beta \geq 0$)
- **Pointwise maximum:** $\max(f_1, \ldots, f_k)$
- **Composition with affine:** $f(Ax + b)$ if $f$ is convex
- **Perspective:** $tf(x/t)$ for $t > 0$

### 5.6 Strong Convexity

$f$ is **$m$-strongly convex** if $H(x) \succeq mI$ for all $x$ (all eigenvalues $\geq m > 0$).

Strongly convex functions have a *unique* minimum and GD converges at rate $O\left(\left(\frac{L-m}{L+m}\right)^{2k}\right)$ where $L$ is the Lipschitz constant of $\nabla f$.

---

## Implementation (60 min)

### Exercise 1: Convexity Classification
```python
# See: code/week01/convexity.py
```
Classify functions as convex/concave/neither using the second-order condition.

### Exercise 2: Tangent Plane Visualization (First-Order Condition)
For a convex function, show that the tangent plane always lies below the function.

### Exercise 3: GD on Convex vs Non-Convex
Run gradient descent on both types. Show: convex always finds global min; non-convex gets stuck.

### Exercise 4: Convexity-Preserving Operations
Demonstrate that sum, max, and affine composition of convex functions remain convex.

---

## Practice (45 min)

Classify each function as convex, concave, or neither. Verify with the Hessian:

1. $f(x) = x^2$ ✓ convex
2. $f(x) = \log(x)$ for $x > 0$
3. $f(x) = x^3$
4. $f(x) = \|x\|_2$ (Euclidean norm)
5. $f(x_1, x_2) = \max(x_1, x_2)$
6. $f(x_1, x_2) = x_1 x_2$
7. $f(x) = e^x$
8. $f(x) = -\log(x)$
9. $f(x) = \|Ax - b\|^2$ — always convex! (the basis of least squares)
10. $f(x) = \log(1 + e^x)$ (softplus)
11. $f(x,y) = (1-x)^2 + 100(y-x^2)^2$ (Rosenbrock — convex?)
12. $f(x) = \lambda_{\max}(X)$ (maximum eigenvalue — convex!)

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Proving least-squares is convex</summary>

**Problem:** Show that $f(x) = \|Ax - b\|^2$ is always convex, for any $A$ and $b$.

**Solution:** $f(x) = x^T A^T A x - 2b^T A x + b^T b$. The Hessian is $H = 2A^T A$. For any vector $v$: $v^T A^T A v = \|Av\|^2 \geq 0$. So $H$ is PSD everywhere → $f$ is convex. It's *strictly* convex iff $A$ has full column rank (so $A^TA$ is PD). If $A$ is rank-deficient, there are infinitely many minimizers.
</details>

<details>
<summary>🎯 Challenge 2: Non-convex SLAM and initialization</summary>

**Problem:** The SLAM cost function $\sum \|z_{ij} - h(x_i, x_j)\|^2$ is not globally convex because $h$ involves rotation. Why does SLAM still work, and what goes wrong with bad initialization?

**Solution:** While the global problem is non-convex (rotation is periodic), it is locally convex near the true solution — the Hessian at the optimum is PD if the measurements are informative. With good initialization (e.g., from odometry), iterative methods converge to the right basin. With bad initialization, the solver can converge to a local minimum where the map is "flipped" or "folded." This is why robust initialization (e.g., chordal relaxation, rotation averaging) is critical.
</details>

<details>
<summary>🎯 Challenge 3: Strong convexity and convergence rate</summary>

**Problem:** For $f(x) = \frac{1}{2}x^T A x$ with $A$ having eigenvalues $[1, 100]$, compute the GD convergence rate with optimal step size. How many iterations to reduce error by $10^{-6}$?

**Solution:** $m = 1$, $L = 100$, optimal step size $\alpha = 2/(L + m) = 2/101$. Rate: $\rho = (L-m)/(L+m) = 99/101 \approx 0.98$. Iterations: $0.98^{2k} = 10^{-6}$ → $k \approx 342$. Compare with Newton: 5-10 iterations. This is why condition number matters so much for first-order methods.
</details>

---

## Self-Assessment Checklist

- [ ] I can classify functions as convex/concave/neither using the Hessian
- [ ] I understand the fundamental theorem: local min = global min for convex functions
- [ ] I can identify which optimization problems are convex (quadratic, least-squares, LP, QP)
- [ ] I see why non-convex problems (SLAM with rotations) need good initialization

---

## Key Takeaways

1. **Convexity is the dividing line** between easy and hard optimization.
2. **Least squares is always convex** — this is why it's the workhorse of robotics.
3. **Non-convex problems** require good initialization or convex relaxation.
4. **Strong convexity** gives convergence rate guarantees.
