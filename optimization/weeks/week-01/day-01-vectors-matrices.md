# Day 1: Vectors, Matrices, Transformations

> **Phase I — Mathematical Foundations | Week 1 | 2.5 hours**
> *Every optimization method manipulates vectors and matrices. Today we build the language.*

## Navigation
- **Previous:** [Curriculum Home](../../CURRICULUM.md)
- **Next:** [Day 2: Eigenvalues, PD, SVD](day-02-eigenvalues-svd.md)
- **Week:** [Week 1 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Robot pose = vector in $\mathbb{R}^3$ (x, y, θ). Transforming between sensor frames = matrix multiplication. Null space of a measurement Jacobian tells you which directions are unobservable — critical for SLAM and calibration.

---

## Theory (45 min)

### 1.1 Vector Spaces, Basis, Span

A **vector space** $V$ over $\mathbb{R}$ is a set with addition and scalar multiplication satisfying the usual axioms. The key concept:

**Linear independence:** Vectors $\{v_1, \ldots, v_k\}$ are linearly independent if:
$$c_1 v_1 + c_2 v_2 + \cdots + c_k v_k = 0 \implies c_1 = c_2 = \cdots = c_k = 0$$

**Basis:** A linearly independent set that spans $V$. Every vector in $V$ has a *unique* representation in terms of the basis.

**Dimension:** Number of vectors in any basis of $V$.

### 1.2 Matrices as Linear Transformations

A matrix $A \in \mathbb{R}^{m \times n}$ defines a linear map $T: \mathbb{R}^n \to \mathbb{R}^m$ where $T(x) = Ax$.

Geometric interpretation: $A$ maps the standard basis vectors $e_1, \ldots, e_n$ to the *columns* of $A$.

### 1.3 Four Fundamental Subspaces

For $A \in \mathbb{R}^{m \times n}$:

| Subspace | Definition | Dimension |
|----------|-----------|-----------|
| Column space $C(A)$ | $\{Ax : x \in \mathbb{R}^n\}$ | $r$ (rank) |
| Null space $N(A)$ | $\{x : Ax = 0\}$ | $n - r$ |
| Row space $C(A^T)$ | $\{A^T y : y \in \mathbb{R}^m\}$ | $r$ |
| Left null space $N(A^T)$ | $\{y : A^T y = 0\}$ | $m - r$ |

**Fundamental theorem of linear algebra:** $C(A) \perp N(A^T)$ and $C(A^T) \perp N(A)$.

### 1.4 Orthogonality and Projection

**Orthogonal projection** of $b$ onto column space of $A$:
$$\hat{b} = A(A^T A)^{-1} A^T b$$

This is the closest point to $b$ in $C(A)$ — the essence of least squares!

**Projection matrix:** $P = A(A^T A)^{-1} A^T$. Key properties: $P^2 = P$, $P^T = P$.

---

## Implementation (60 min)

### Exercise 1: Gram-Schmidt Orthogonalization
Implement from scratch — no `numpy.linalg`:

```python
# See: code/week01/gram_schmidt.py
```

### Exercise 2: QR Decomposition via Gram-Schmidt
Use your Gram-Schmidt to decompose $A = QR$.

### Exercise 3: Projection onto Subspace
Given a subspace defined by column vectors, project an arbitrary vector onto it.

### Exercise 4: Verify with NumPy
For each exercise above, compare your output with `numpy.linalg.qr`, `numpy.linalg.lstsq`.

---

## Practice (45 min)

### Hand Computation Problems

1. Find the rank, null space, and column space of:
$$A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 1 & 3 & 4 \end{pmatrix}$$

2. Are the vectors $[1, 2, 3]$, $[4, 5, 6]$, $[7, 8, 9]$ linearly independent?

3. Project $b = [1, 2, 3]$ onto the column space of $A = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{pmatrix}$.

4. Compute $A^T A$ and $A A^T$ for a $3 \times 2$ matrix. Note the dimensions.

5. Given a rotation matrix $R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$, verify $R^T R = I$.

6–10. Additional rank, null-space, and projection problems (verify all with NumPy).

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Prove that projection is idempotent</summary>

**Problem:** Show that applying the projection matrix twice gives the same result: $P^2 = P$.

**Solution:**
$$P^2 = A(A^TA)^{-1}A^T \cdot A(A^TA)^{-1}A^T = A(A^TA)^{-1}(A^TA)(A^TA)^{-1}A^T = A(A^TA)^{-1}A^T = P$$

The middle $(A^TA)^{-1}(A^TA)$ cancels to $I$.
</details>

<details>
<summary>🎯 Challenge 2: What happens when A is rank-deficient?</summary>

**Problem:** If $A$ has rank $r < n$, then $A^TA$ is singular. How do you compute the projection?

**Solution:** Use the pseudoinverse: $P = A A^+ = A(A^TA)^+ A^T$, or equivalently use the thin SVD: $A = U_r \Sigma_r V_r^T$, then $P = U_r U_r^T$. This projects onto the column space regardless of rank.

**Robotics connection:** Rank-deficient Jacobians appear when a robot is at a kinematic singularity.
</details>

<details>
<summary>🎯 Challenge 3: Null space and unobservability</summary>

**Problem:** In a SLAM system, the measurement Jacobian $H$ has a null space corresponding to gauge freedom (absolute position/orientation). What dimension is this null space for 2D SLAM?

**Solution:** For 2D SLAM with poses $(x, y, \theta)$: the null space has dimension 3 — corresponding to translation (2 DOF) and rotation (1 DOF) of the entire map. This is why we need to "anchor" at least one pose.
</details>

---

## Self-Assessment Checklist

- [ ] I can compute rank, null space, and column space by hand
- [ ] I understand why projection gives the "closest point" in a subspace
- [ ] I can implement Gram-Schmidt without looking at references
- [ ] I see the connection: projection → least squares → optimization

---

## Key Takeaways

1. **Rank** tells you how many independent constraints you have — fundamental for optimization
2. **Null space** tells you what you *cannot* determine — the unobservable directions
3. **Projection** is the geometric heart of least squares
4. **Four fundamental subspaces** partition $\mathbb{R}^n$ and $\mathbb{R}^m$ into orthogonal complements

## Reading
- Gilbert Strang — *Introduction to Linear Algebra* Ch 1-3 (skim if familiar)
- Boyd & Vandenberghe §A.1-A.3 (Appendix: Linear Algebra review)
