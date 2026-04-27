# Day 2: Eigenvalues, Positive Definiteness, SVD

> **Phase I — Mathematical Foundations | Week 1 | 2.5 hours**
> *PD matrices are the foundation of optimization: every well-behaved minimum has a PD Hessian.*

## Navigation
- **Previous:** [Day 1: Vectors, Matrices, Transformations](day-01-vectors-matrices.md)
- **Next:** [Day 3: Gradients and Jacobians](day-03-gradients-jacobians.md)
- **Week:** [Week 1 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
The information matrix in SLAM (inverse covariance) must be PD. SVD reveals which directions of a state have strong measurements vs weak. Condition number predicts whether your optimizer will converge reliably.

---

## Theory (45 min)

### 2.1 Eigenvalue Decomposition

For symmetric $A \in \mathbb{R}^{n \times n}$:
$$A = Q \Lambda Q^T$$

where $Q$ is orthogonal ($Q^T Q = I$) and $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$.

**Geometric interpretation:** Eigenvectors define the principal axes. Eigenvalues define the stretching along each axis.

### 2.2 Positive Definiteness Classification

| Classification | Eigenvalues | Meaning |
|----------------|------------|---------|
| Positive definite (PD) | All $\lambda_i > 0$ | Bowl shape — unique minimum |
| Positive semidefinite (PSD) | All $\lambda_i \geq 0$ | Flat valleys — degenerate minimum |
| Indefinite | Mixed signs | Saddle point |
| Negative definite | All $\lambda_i < 0$ | Hill — maximum |

**Tests for PD:**
1. All eigenvalues positive
2. All leading principal minors positive (Sylvester's criterion)
3. Cholesky decomposition succeeds
4. $x^T A x > 0$ for all $x \neq 0$

### 2.3 Singular Value Decomposition

For any $A \in \mathbb{R}^{m \times n}$:
$$A = U \Sigma V^T$$

- $U \in \mathbb{R}^{m \times m}$: left singular vectors (column space basis)
- $\Sigma \in \mathbb{R}^{m \times n}$: singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
- $V \in \mathbb{R}^{n \times n}$: right singular vectors (row space basis)

**SVD reveals everything:** rank = # nonzero $\sigma_i$, null space = columns of $V$ for zero $\sigma_i$.

### 2.4 Condition Number

$$\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}} = \frac{\lambda_{\max}}{\lambda_{\min}} \quad (\text{for PD } A)$$

**Meaning:** A small perturbation $\delta b$ to $Ax = b$ causes solution change bounded by $\kappa(A) \cdot \|\delta b\| / \|b\|$.

- $\kappa \approx 1$: well-conditioned → stable
- $\kappa \gg 1$: ill-conditioned → small input changes cause large output changes
- $\kappa = \infty$: singular

**Optimization connection:** Condition number of the Hessian determines gradient descent convergence rate: $O\left(\left(\frac{\kappa-1}{\kappa+1}\right)^{2k}\right)$ steps.

---

## Implementation (60 min)

### Exercise 1: Power Iteration
```python
# See: code/week01/eigenvalues_svd.py
```
Implement power iteration to find the dominant eigenvalue/eigenvector.

### Exercise 2: PD Classification
Check positive definiteness via: (a) eigenvalues, (b) Cholesky attempt, (c) quadratic form test.

### Exercise 3: SVD and Low-Rank Approximation
Compute SVD, reconstruct with $k < r$ singular values, measure approximation error.

### Exercise 4: Condition Number → Numerical Error
Solve $Ax = b$, perturb $b$, show solution error amplifies by $\kappa(A)$.

---

## Practice (45 min)

Classify the following matrices as PD/PSD/indefinite. For each, verify via eigenvalues, determinant test, and Cholesky:

1. $A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$
2. $A = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}$
3. $A = \begin{pmatrix} 4 & 2 \\ 2 & 1 \end{pmatrix}$
4. $A = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix}$
5. $I_{3 \times 3}$

6–10. Additional 5 random symmetric matrices (generate and classify).

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Why is the Hessian at a minimum PD?</summary>

**Problem:** Prove that if $x^*$ is a strict local minimum of $f$, then $H(x^*) \succeq 0$ (PSD). Under what additional condition is it PD?

**Solution:** By Taylor expansion: $f(x^* + \delta) \approx f(x^*) + \nabla f(x^*)^T \delta + \frac{1}{2}\delta^T H \delta$. Since $\nabla f(x^*) = 0$, we get $f(x^* + \delta) - f(x^*) \approx \frac{1}{2}\delta^T H \delta$. For this to be $\geq 0$ for all $\delta$, we need $H \succeq 0$. If $f$ is *strictly* convex near $x^*$, then $H \succ 0$.
</details>

<details>
<summary>🎯 Challenge 2: Condition number and GD convergence</summary>

**Problem:** For $f(x) = \frac{1}{2}x^T A x - b^T x$ with PD $A$ having condition number $\kappa$, how many gradient descent iterations are needed to reduce the error by a factor of $10^{-6}$?

**Solution:** GD converges as $\|x_k - x^*\| \leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^k \|x_0 - x^*\|$. For $\kappa = 100$: convergence rate $\approx 0.98$, so $0.98^k = 10^{-6}$ gives $k \approx 683$ iterations. For $\kappa = 2$: rate $\approx 0.33$, so $k \approx 12$. This motivates preconditioning and second-order methods.
</details>

<details>
<summary>🎯 Challenge 3: SVD of a measurement Jacobian</summary>

**Problem:** A robot has a range sensor giving measurements $z_i = \|p_{\text{robot}} - p_{\text{landmark}_i}\| + \epsilon_i$. The Jacobian $J$ has singular values $[5.2, 3.1, 0.01]$. What does this tell you about observability?

**Solution:** The third direction has singular value near zero — it's nearly unobservable. This means you can't reliably estimate the state in that direction. The condition number $\kappa = 5.2/0.01 = 520$ is high, meaning the estimate will be highly sensitive to noise in that direction. Geometrically, the landmarks may be nearly collinear, giving poor depth resolution in one direction.
</details>

---

## Self-Assessment Checklist

- [ ] I can classify any symmetric matrix as PD/PSD/indefinite using 3 different tests
- [ ] I understand SVD geometrically and can compute low-rank approximations
- [ ] I can predict numerical trouble from the condition number
- [ ] I see the bridge: PD Hessian ↔ local minimum ↔ optimization

---

## Key Takeaways

1. **PD = bowl = minimum.** The entire field of optimization revolves around this.
2. **SVD is the swiss army knife** — rank, null space, condition number, low-rank approximation.
3. **Condition number predicts convergence** — if $\kappa(H)$ is bad, simple methods will struggle.
4. **Every SLAM information matrix must be PD** — otherwise the MAP estimate is ill-defined.
