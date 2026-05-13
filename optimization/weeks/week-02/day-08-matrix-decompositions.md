# Day 8: Matrix Decompositions for Optimization

> **Phase I — Mathematical Foundations | Week 2 | 2.5 hours**
> *Cholesky is to PD systems what butter is to toast — the natural, fast choice.*

## Navigation
- **Previous:** [Day 7: Week 1 Review](../week-01/day-07-week-review.md)
- **Next:** [Day 9: Sparse Matrices & Schur Complement](day-09-sparse-schur.md)
- **Week:** [Week 2 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
Every Ceres Solver call internally decomposes the Hessian approximation $J^T J$. Cholesky is the default for dense problems; sparse Cholesky (with ordering) for SLAM. Knowing which decomposition to use — and why — lets you configure Ceres and g2o for 10× speedups.

---

## Theory (45 min)

### 8.1 Cholesky Decomposition: $A = LL^T$

For a **positive definite** matrix $A$:

$$A = LL^T$$

where $L$ is lower triangular with positive diagonal entries.

**Algorithm** (column-by-column):

$$l_{jj} = \sqrt{a_{jj} - \sum_{k=1}^{j-1} l_{jk}^2}, \qquad l_{ij} = \frac{1}{l_{jj}}\left(a_{ij} - \sum_{k=1}^{j-1} l_{ik} l_{jk}\right)$$

**Cost:** $\frac{1}{3}n^3$ flops — exactly half the cost of LU.

**Solving $Ax = b$:**
1. Forward substitution: $Ly = b$ → $O(n^2)$
2. Back substitution: $L^T x = y$ → $O(n^2)$

### 8.2 LDL^T Decomposition

For **symmetric** matrices (including PSD):

$$A = LDL^T$$

where $L$ is unit lower triangular, $D$ is diagonal. Avoids square roots. The sign pattern of $D$ reveals the inertia (number of +, −, 0 eigenvalues).

### 8.3 QR Decomposition for Least Squares

For $A \in \mathbb{R}^{m \times n}$ with $m \geq n$:

$$A = QR = Q \begin{pmatrix} R_1 \\ 0 \end{pmatrix}$$

The least-squares solution $\min_x \|Ax - b\|^2$:

$$x = R_1^{-1} Q_1^T b$$

More numerically stable than the normal equations $A^T A x = A^T b$ (avoids squaring the condition number).

### 8.4 Decision Table

| Problem | Decomposition | Cost | When |
|---------|-------------|------|------|
| $Ax=b$, $A$ PD | Cholesky | $\frac{n^3}{3}$ | **Default for optimization** |
| $Ax=b$, $A$ symmetric | LDL^T | $\frac{n^3}{3}$ | Indefinite or PSD systems |
| $\min\|Ax-b\|^2$, $A$ full rank | QR | $\frac{2mn^2}{3}$ | Rectangular, well-conditioned |
| $\min\|Ax-b\|^2$, $A$ rank-deficient | SVD | $2mn^2 + 11n^3$ | Rank-deficient, minimum-norm |
| General $Ax=b$ | LU | $\frac{2n^3}{3}$ | Non-symmetric |

---

## Implementation (60 min)

### Exercise 1: Implement Cholesky from Scratch
```python
# See: code/week02/matrix_decompositions.py
```
Write the column-by-column Cholesky algorithm. Verify against `np.linalg.cholesky`.

### Exercise 2: Solve via Cholesky
Solve $Ax = b$ by forward + back substitution using your Cholesky factor. Compare accuracy to `np.linalg.solve`.

### Exercise 3: Benchmark Decompositions
For $n = 100, 500, 1000$: time Cholesky solve vs `numpy.linalg.solve` vs `numpy.linalg.inv(A) @ b`. Show that Cholesky is fastest and `inv` is slowest (and least accurate).

### Exercise 4: Cholesky Fails on Non-PD
Show that Cholesky correctly detects a non-PD matrix (it encounters a negative value under the square root).

### Exercise 5: QR vs Normal Equations
Compare QR vs normal equations on an ill-conditioned rectangular system. Show QR gives better accuracy.

---

## Practice (45 min)

1. What is the cost of solving $Ax = b$ if we already have $L$ from Cholesky?
2. Why does Cholesky cost half of LU? (Hint: symmetry eliminates half the work.)
3. If $A = LL^T$ and we want $A^{-1}$, how do we compute it from $L$?
4. In Ceres, why is `DENSE_QR` recommended for small problems and `DENSE_SCHUR` for large?
5. How does QR avoid squaring the condition number that normal equations suffer from?
6. What happens if you try Cholesky on $A^T A$ when $A$ is rank-deficient?
7. Derive: det($A$) from Cholesky factor $L$. (Hint: $\det(A) = \det(L)^2$.)
8. Why is `L @ L.T @ x = b` slower than `cho_solve` in SciPy?

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Modified Cholesky for indefinite matrices</summary>

**Problem:** Newton's method sometimes produces an indefinite Hessian. Implement a modified Cholesky that adds a minimal diagonal perturbation $E$ so that $A + E = LDL^T$ with $D > 0$. Use the Gill-Murray-Wright strategy.

**Solution:** During the LDL^T factorization, if a pivot $d_{jj}$ would be too small or negative, replace it with $\max(|d_{jj}|, \beta^2 \cdot \max_j |a_{jj}|, \delta)$ where $\delta$ is a small positive constant. This ensures the modified Hessian is PD while keeping the perturbation small. In Ceres, this is handled automatically by trust region methods (which add $\lambda I$ to the Hessian).
</details>

<details>
<summary>🎯 Challenge 2: Rank-revealing decompositions</summary>

**Problem:** You have a 1000×100 Jacobian $J$ where the last 5 columns are nearly linearly dependent (degenerate parameters). Compare: (a) normal equations $J^T J x = J^T b$, (b) QR with column pivoting, (c) SVD truncated at rank 95. Which gives the most useful solution?

**Solution:** (a) Normal equations amplify the ill-conditioning: $\kappa(J^T J) = \kappa(J)^2$, likely causing garbage for the 5 degenerate parameters. (b) QR with column pivoting (`np.linalg.qr` with pivoting or `scipy.linalg.qr(J, pivoting=True)`) reveals which columns are nearly dependent and gives the upper-triangular $R$ with decreasing diagonal — you can truncate. (c) SVD gives the minimum-norm solution and cleanly separates the 95 "real" parameters from the 5 degenerate ones. In robotics calibration, SVD is the gold standard for identifying unobservable parameters.
</details>

<details>
<summary>🎯 Challenge 3: Block Cholesky for structured problems</summary>

**Problem:** In bundle adjustment, $H$ has block structure: $H = \begin{pmatrix} H_{cc} & H_{cl} \\ H_{lc} & H_{ll} \end{pmatrix}$ where $H_{ll}$ is block-diagonal (landmarks are independent). Show that the Schur complement $S = H_{cc} - H_{cl} H_{ll}^{-1} H_{lc}$ can be formed in $O(m \cdot k^3)$ time where $m$ is the number of landmarks and $k$ is the landmark dimension.

**Solution:** Since $H_{ll} = \text{blkdiag}(H_1, \ldots, H_m)$, we have $H_{ll}^{-1} = \text{blkdiag}(H_1^{-1}, \ldots, H_m^{-1})$. Each $H_i^{-1}$ costs $O(k^3)$ (for 3D points, $k=3$, so it's $27$ flops). The Schur complement accumulates: $S = H_{cc} - \sum_{i=1}^{m} H_{cl,i} H_i^{-1} H_{lc,i}$. Each term is a rank-$k$ update to $S$, costing $O(c^2 k)$ where $c$ is the total camera parameter count. This is the core of Ceres's `SPARSE_SCHUR` solver.
</details>

---

## Self-Assessment Checklist

- [ ] I can implement Cholesky decomposition from scratch
- [ ] I know when to use Cholesky vs QR vs SVD vs LU
- [ ] I understand why solving via decomposition is better than computing $A^{-1}$
- [ ] I can explain the block Cholesky structure in bundle adjustment

---

## Key Takeaways

1. **Cholesky ($LL^T$)** is the fastest solver for PD systems — the default in optimization.
2. **QR** avoids condition number squaring — use for rectangular least squares.
3. **Never compute $A^{-1}$ explicitly** — use decomposition + substitution.
4. **The decomposition choice directly maps to Ceres solver options** — understanding this is not academic, it's practical.
