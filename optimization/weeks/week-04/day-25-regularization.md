# Day 25: Multi-Objective and Regularization

> **Phase II — Unconstrained Optimization | Week 4 | 2.5 hours**
> *Taming ill-posed problems with regularization — the bridge from optimization to Bayesian estimation.*

## Navigation
- **Previous:** [Day 24: Stochastic Methods](day-24-stochastic-methods.md)
- **Next:** [Day 26: Global Optimization](day-26-global-optimization.md)
- **Week:** [Week 4 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Regularization is *everywhere* in the OKS stack. The Ceres solver's `LossFunction` applies robust kernels (a form of regularization). OSQP solves regularized QPs internally. The navigation estimator uses process noise $Q$ as a Tikhonov-like regularizer on state transitions. Map building uses L2 regularization to prevent overfitting to noisy sensor data. Understanding regularization deeply means understanding *why* solver parameters are tuned the way they are.

---

## Theory (45 min)

### 25.1 Regularization as Multi-Objective

Original problem may be ill-posed or underdetermined. Add a penalty:

$$\min_x \underbrace{f(x)}_{\text{data fit}} + \lambda \underbrace{R(x)}_{\text{regularizer}}$$

### 25.2 Tikhonov / Ridge Regularization (L2)

$$\min_x \|Ax - b\|^2 + \lambda \|x\|^2$$

**Closed-form:** $x^* = (A^TA + \lambda I)^{-1} A^Tb$

**Effect:** Shrinks solution toward zero. Makes $A^TA + \lambda I$ invertible even if $A$ is rank-deficient.

**Bayesian interpretation:** Gaussian prior $x \sim \mathcal{N}(0, \sigma^2_p I)$ gives $\lambda = \sigma^2_n / \sigma^2_p$.

### 25.3 L1 / LASSO Regularization

$$\min_x \|Ax - b\|^2 + \lambda \|x\|_1$$

**Effect:** Promotes sparsity — many components of $x^*$ are exactly zero.

**No closed-form** — requires iterative methods (proximal gradient).

**Bayesian interpretation:** Laplace prior $p(x_j) \propto \exp(-|x_j|/b)$.

### 25.4 The Regularization Path

As $\lambda$ varies from $\infty$ to $0$:
- $\lambda \to \infty$: $x^* \to 0$ (pure prior)
- $\lambda \to 0$: $x^*$ = unconstrained solution (pure data fit)
- The path $x^*(\lambda)$ traces a smooth curve for L2, piecewise-linear for L1

### 25.5 Proximal Operators

The **proximal operator** of $f$:
$$\text{prox}_{\lambda f}(v) = \arg\min_x \left[f(x) + \frac{1}{2\lambda}\|x - v\|^2\right]$$

**Key proximal operators:**
| $f(x)$ | $\text{prox}_{\lambda f}(v)$ |
|---------|-------------------------------|
| $\|x\|_1$ | Soft-thresholding: $\text{sign}(v_j)\max(|v_j| - \lambda, 0)$ |
| $\|x\|^2$ | Scaling: $v / (1 + 2\lambda)$ |
| $\delta_C(x)$ (indicator) | Projection onto $C$ |

### 25.6 ISTA and FISTA

**ISTA** (Iterative Shrinkage-Thresholding):
$$x_{k+1} = \text{prox}_{\alpha\lambda\|\cdot\|_1}\left(x_k - \alpha \nabla g(x_k)\right)$$

where $g(x) = \|Ax - b\|^2$ (smooth part) and $\lambda\|x\|_1$ (nonsmooth part).

**FISTA** (Fast ISTA): add Nesterov acceleration to ISTA → $O(1/k^2)$ convergence.

### 25.7 Connection to Modern Solvers

Why this matters for the rest of the curriculum:
- **OSQP** and **SCS** use ADMM internally, which is based on proximal operators
- **ADMM** splits problems into subproblems solved via proximal operators
- Regularization is how Ceres handles underdetermined problems
- Understanding proximal operators unlocks Phase IV (constrained optimization)

---

## Implementation (60 min)

### Exercise 1: Ridge Regression
```python
# See: code/week04/regularization.py
```
Solve ridge regression both closed-form and iteratively (GD on regularized objective). Compare solutions.

### Exercise 2: L1 Proximal Gradient (ISTA)
Implement ISTA for LASSO. Show that it produces sparse solutions.

### Exercise 3: Regularization Path
Plot solution components $x_j$ vs $\lambda$ for both L2 and L1. Show L1's kink points where components enter/leave.

### Exercise 4: Cross-Validation for λ
Implement k-fold cross-validation to select optimal $\lambda$. Plot validation error vs $\lambda$.

### Exercise 5: Regularized Sensor Calibration
Robot calibration problem: estimate sensor parameters from noisy data. Without regularization: overfits to noise. With L2: stable solution.

---

## Practice (45 min)

1. Derive the closed-form solution for ridge regression from first principles (set gradient to zero).
2. Show that the ridge solution $x^*(\lambda)$ has smaller norm than the unconstrained solution.
3. Why does L1 produce sparse solutions but L2 doesn't? (Hint: geometry of the constraint set.)
4. Compute $\text{prox}_{\lambda\|\cdot\|_1}(v)$ for $v = [3, -0.5, 1.5]$ with $\lambda = 1$.
5. If $A$ has condition number $\kappa = 10^6$, what $\lambda$ makes $(A^TA + \lambda I)$ well-conditioned?
6. In Ceres, `HuberLoss` is used on residuals. How is this related to regularization?
7. Why is the regularization path piecewise-linear for L1 but smooth for L2?
8. Explain: "regularization = Bayesian prior." What does $\lambda$ correspond to in the Bayesian framework?
9. If you add L2 regularization to the OKS odometry calibration, what does it mean physically?
10. What is the proximal operator for the indicator function of a box constraint $x \in [l, u]$?

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: FISTA implementation</summary>

**Problem:** Implement FISTA (accelerated proximal gradient) for LASSO. Compare convergence of ISTA ($O(1/k)$) vs FISTA ($O(1/k^2)$) on a problem with $A \in \mathbb{R}^{100 \times 500}$ (more variables than equations — L1 selects a sparse subset).

**Solution:** FISTA adds Nesterov momentum to ISTA:
```
y_k = x_k + ((t_{k-1} - 1)/t_k) * (x_k - x_{k-1})
x_{k+1} = prox(y_k - α∇g(y_k))
t_{k+1} = (1 + sqrt(1 + 4*t_k²)) / 2
```
On the 100×500 problem, FISTA reaches $10^{-6}$ accuracy in ~200 iterations vs ISTA's ~2000. The acceleration is most visible in the final convergence phase.
</details>

<details>
<summary>🎯 Challenge 2: Elastic net</summary>

**Problem:** Implement elastic net: $\min \|Ax-b\|^2 + \lambda_1\|x\|_1 + \lambda_2\|x\|^2$. This combines L1 sparsity with L2 stability. Show that it selects groups of correlated features (unlike pure L1 which picks one arbitrarily).

**Solution:** The proximal gradient step becomes: compute gradient of smooth part ($\|Ax-b\|^2 + \lambda_2\|x\|^2$), then apply soft-thresholding (L1 prox). The L2 term makes the problem strongly convex, ensuring a unique solution even with correlated features. Create data with groups of correlated features and show elastic net selects the whole group while LASSO picks one per group.
</details>

<details>
<summary>🎯 Challenge 3: Regularization in robot SLAM</summary>

**Problem:** In graph SLAM, the optimization $\min \sum \|z_{ij} - h(x_i, x_j)\|^2$ can be underdetermined (gauge freedom). Show this by constructing a small SLAM problem where the Hessian $J^TJ$ is singular. Then show that adding a single anchor constraint (L2 regularization on one pose) makes it well-conditioned.

**Solution:** 3-pose SLAM with 2 relative measurements: $z_{01}$ and $z_{12}$. The system has rank deficiency of 3 (can translate and rotate all poses). $J^TJ$ has 3 zero eigenvalues. Adding $(x_0 - x_{\text{anchor}})^2$ with small weight fixes the gauge. This is exactly what Ceres/g2o do with "anchor nodes."
</details>

---

## Self-Assessment Checklist

- [ ] I can implement both L2 and L1 regularization and explain their effects
- [ ] I understand proximal operators and can implement ISTA for LASSO
- [ ] I can select $\lambda$ using cross-validation
- [ ] I see the connection: regularization = Bayesian prior = solver stabilization

---

## Key Takeaways

1. **L2** (Tikhonov) shrinks, **L1** (LASSO) sparsifies — choose based on the problem structure.
2. **Proximal operators** are the building blocks of modern convex solvers (OSQP, SCS, ADMM).
3. **Cross-validation** selects $\lambda$; the Bayesian view gives interpretability.
4. **In robotics:** regularization prevents overfitting to noise and stabilizes underdetermined problems.
