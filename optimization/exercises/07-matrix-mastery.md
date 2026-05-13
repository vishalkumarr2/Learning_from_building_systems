# Exercises: Matrix Mastery

### Chapter 08: Matrix Essentials for Optimization

**Self-assessment guide:** These exercises cover the full landscape of matrix theory needed for optimization and robotics — from hand computation to Python verification to real-world scenarios. Complete them in order; later exercises build on earlier ones.

**Difficulty key:** ⭐ Foundational | ⭐⭐ Intermediate | ⭐⭐⭐ Advanced | ⭐⭐⭐⭐ Challenge

---

## Section A — Matrix Classification and Structure

**A1.** ⭐ Classify each matrix as symmetric / skew-symmetric / orthogonal / none.
State which spectral classes (PD, PSD, indefinite, etc.) are even *possible* for that structural type.

$$A = \begin{pmatrix} 3 & 1 \\ 1 & 4 \end{pmatrix}, \quad
B = \begin{pmatrix} 0 & -2 \\ 2 & 0 \end{pmatrix}, \quad
C = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}$$

<details><summary>Answer</summary>

**$A$:** Symmetric ($A = A^T$). Can be PD, PSD, indefinite, ND, NSD.

**$B$:** Skew-symmetric ($B = -B^T$). Eigenvalues are purely imaginary ($\pm 2i$), so **definiteness does not apply** (definiteness is defined for symmetric matrices only). However, $B$ is real and has no real eigenvalues.

**$C$:** Orthogonal ($C^T C = I$, verify: each column has norm 1, dot product = 0). Eigenvalues lie on the unit circle. Not symmetric, so definiteness classification doesn't directly apply. $\kappa(C) = 1$.

**Key insight:** Only symmetric (or Hermitian) matrices have a meaningful PD/PSD/indefinite classification. Orthogonal matrices are always perfectly conditioned.

</details>

**A2.** ⭐ Given the block matrix:

$$M = \begin{pmatrix} 2I_3 & J \\ J^T & 5I_2 \end{pmatrix}$$

where $J \in \mathbb{R}^{3 \times 2}$. Is $M$ symmetric? What conditions on $J$ make $M$ positive definite?

<details><summary>Answer</summary>

**Symmetric?** Yes. $(2I_3)^T = 2I_3$, $(5I_2)^T = 5I_2$, and the off-diagonal blocks are $J$ and $J^T$, so $M = M^T$.

**PD condition via Schur complement:**
$M \succ 0 \iff 2I_3 \succ 0$ (always true) **and** $5I_2 - J^T (2I_3)^{-1} J \succ 0$

$$5I_2 - \frac{1}{2} J^T J \succ 0 \iff J^T J \prec 10 I_2 \iff \sigma_{\max}(J)^2 < 10 \iff \sigma_{\max}(J) < \sqrt{10}$$

So $M$ is PD whenever all singular values of $J$ are less than $\sqrt{10} \approx 3.16$.

</details>

**A3.** ⭐⭐ A matrix $A \in \mathbb{R}^{5 \times 5}$ has eigenvalues $\{8, 4, 2, 0.5, 0.001\}$.

1. What is $\kappa_2(A)$?
2. How many digits of accuracy can you expect when solving $Ax = b$ in double precision?
3. If you form the normal equations $A^T A x = A^T b$, what becomes the new $\kappa$?
4. Would you recommend solving via Cholesky on $A^T A$ or via QR on $A$?

<details><summary>Answer</summary>

1. $\kappa_2(A) = \sigma_{\max}/\sigma_{\min} = 8 / 0.001 = 8000$. Since $A$ is symmetric, eigenvalues = singular values.

2. Digits lost $\approx \log_{10}(8000) \approx 3.9$. With 16 digits of double precision, expect about **12 reliable digits**. Quite acceptable.

3. $\kappa(A^T A) = \kappa(A)^2 = 8000^2 = 64{,}000{,}000$. Digits lost $\approx 7.8$. Only about **8 reliable digits**.

4. **QR factorization** of $A$ is recommended. It works with $\kappa(A)$ directly (not squared), giving 12-digit accuracy. The normal equations lose about 4 extra digits unnecessarily.

**Rule:** If $\kappa(A) > 10^4$, avoid the normal equations. Use QR or SVD.

</details>

---

## Section B — Positive Definiteness

**B1.** ⭐ Apply Sylvester's criterion to determine if each matrix is positive definite:

$$P = \begin{pmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2 \end{pmatrix}, \qquad
Q = \begin{pmatrix} 1 & 2 \\ 2 & 3 \end{pmatrix}$$

<details><summary>Answer</summary>

**Matrix $P$:**
- $\Delta_1 = 2 > 0$ ✓
- $\Delta_2 = 2 \cdot 2 - (-1)(-1) = 4 - 1 = 3 > 0$ ✓
- $\Delta_3 = \det(P) = 2(4-1) - (-1)(-2-0) + 0 = 6 - 2 + 0 = 4 > 0$ ✓

All leading principal minors positive → **$P$ is PD**.

(This is actually the tridiagonal matrix from 1D Laplacian discretization — always PD!)

**Matrix $Q$:**
- $\Delta_1 = 1 > 0$ ✓
- $\Delta_2 = \det(Q) = 1 \cdot 3 - 2 \cdot 2 = 3 - 4 = -1 < 0$ ✗

**$Q$ is NOT positive definite.** ($Q$ is indefinite; eigenvalues $\approx 4.24$ and $-0.24$.)

</details>

**B2.** ⭐⭐ Prove: if $A \succ 0$ and $B \succ 0$, then $A + B \succ 0$.

<details><summary>Answer</summary>

For any $x \neq 0$:
$$x^T(A + B)x = x^TAx + x^TBx$$

Since $A \succ 0$: $x^TAx > 0$.
Since $B \succ 0$: $x^TBx > 0$.

Therefore $x^T(A+B)x > 0$ for all $x \neq 0$, hence $A+B \succ 0$. ∎

**Corollary:** The set of PD matrices is a **convex cone** — any positive linear combination of PD matrices is PD. This is why adding a regularization term $\lambda I$ to a PSD Hessian makes it PD: $H + \lambda I \succ 0$ for any $\lambda > 0$ when $H \succeq 0$.

</details>

**B3.** ⭐⭐ A noisy sensor gives you this estimated covariance matrix:

$$\hat{\Sigma} = \begin{pmatrix} 4.0 & 1.0 & 3.1 \\ 1.0 & 2.0 & -0.5 \\ 3.1 & -0.5 & 1.0 \end{pmatrix}$$

1. Is $\hat{\Sigma}$ PD? (Try Cholesky.)
2. If not, use Higham's algorithm (eigenvalue clipping) to find the nearest PD matrix.
3. Why does it matter that covariance matrices be PD?

<details><summary>Answer</summary>

**1. Cholesky test:** Attempting $\hat{\Sigma} = LL^T$:
- $L_{11} = \sqrt{4} = 2$
- $L_{21} = 1/2 = 0.5$, $L_{31} = 3.1/2 = 1.55$
- $L_{22} = \sqrt{2 - 0.25} = \sqrt{1.75} = 1.323$
- $L_{32} = (-0.5 - 0.5 \cdot 1.55)/1.323 = -1.275/1.323 = -0.964$
- $L_{33} = \sqrt{1.0 - 1.55^2 - (-0.964)^2} = \sqrt{1.0 - 2.4025 - 0.929} = \sqrt{-2.332}$

**Negative under square root → $\hat{\Sigma}$ is NOT PD.** (The $(3,3)$ entry is too small relative to the off-diagonal correlations.)

**2. Higham's algorithm:**
- Eigendecompose: $\hat{\Sigma} = Q\Lambda Q^T$ with eigenvalues $\approx \{5.93, 1.53, -0.46\}$
- Clip: $\tilde{\Lambda} = \text{diag}(5.93, 1.53, 0.01)$ (replace $-0.46$ with small $\epsilon = 0.01$)
- Reconstruct: $\tilde{\Sigma} = Q\tilde{\Lambda}Q^T$

```python
import numpy as np
Sigma_hat = np.array([[4.0, 1.0, 3.1], [1.0, 2.0, -0.5], [3.1, -0.5, 1.0]])
eigvals, Q = np.linalg.eigh(Sigma_hat)
eigvals_clipped = np.maximum(eigvals, 0.01)
Sigma_pd = Q @ np.diag(eigvals_clipped) @ Q.T
print(np.linalg.cholesky(Sigma_pd))  # Now succeeds
```

**3. Why PD matters:** A covariance matrix that isn't PD implies some direction has zero or negative variance, which is physically meaningless. The Mahalanobis distance $d^2 = (x-\mu)^T \Sigma^{-1}(x-\mu)$ requires $\Sigma \succ 0$. Non-PD covariance will cause Kalman filter updates, optimization weights, and uncertainty ellipsoid calculations to break.

</details>

**B4.** ⭐⭐⭐ The Schur complement test says $M = \begin{pmatrix} A & B \\ B^T & C \end{pmatrix} \succ 0$ iff $A \succ 0$ and $C - B^T A^{-1} B \succ 0$.

In bundle adjustment, the Hessian is:
$$H = \begin{pmatrix} H_{cc} & H_{cl} \\ H_{lc} & H_{ll} \end{pmatrix}$$

where $H_{cc}$ is the camera-camera block ($6c \times 6c$) and $H_{ll}$ is the landmark-landmark block ($3l \times 3l$), with $l \gg c$.

1. What is the Schur complement $S$ that eliminates landmarks?
2. What are the dimensions of $S$?
3. Why is this drastically more efficient than solving the full system?

<details><summary>Answer</summary>

**1.** $S = H_{cc} - H_{cl} H_{ll}^{-1} H_{lc}$

**2.** $S$ is $6c \times 6c$ (camera-only system).

**3.** Efficiency comparison:
- **Full system:** Solve $(6c + 3l) \times (6c + 3l)$ system. If $l = 10{,}000$ landmarks and $c = 50$ cameras, that's $30{,}300 \times 30{,}300$.
- **Schur complement:** Solve $300 \times 300$ system for cameras, then back-substitute for landmarks.
- **Key:** $H_{ll}$ is **block-diagonal** (each landmark only connects to itself), so $H_{ll}^{-1}$ is trivially computed as independent $3 \times 3$ inversions. The expensive part is forming $S$, but it's $O(c^2 l)$ rather than $O((c+l)^3)$.

This is the "Schur trick" that makes BA tractable for thousands of landmarks. Ceres Solver's `SPARSE_SCHUR` and `ITERATIVE_SCHUR` implement this.

**Physical interpretation:** Eliminating landmarks via the Schur complement has a clear
geometric meaning — it marginalizes out all landmark positions, leaving a reduced system
that operates purely on camera poses. The Schur complement $S$ is the **marginal
information matrix** for cameras. PD of $S$ guarantees the camera-only system is
well-posed (no unobservable directions). $H_{ll}$ being block-diagonal reflects the
crucial structural fact that each landmark connects only to the cameras that observe it,
never directly to other landmarks.

</details>

**B5.** ⭐ Check diagonal dominance for each matrix. Is it strictly diagonally dominant?
Is it PD? Comment on the relationship.

$$D_1 = \begin{pmatrix} 5 & 1 & 1 \\ 1 & 4 & 1 \\ 1 & 1 & 6 \end{pmatrix}, \qquad
D_2 = \begin{pmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2 \end{pmatrix}, \qquad
D_3 = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}$$

<details><summary>Answer</summary>

**Diagonal dominance check** — is $|a_{ii}| > \sum_{j \neq i} |a_{ij}|$ for each row?

**$D_1$:** Row 1: $5 > 1+1=2$ ✓ | Row 2: $4 > 1+1=2$ ✓ | Row 3: $6 > 1+1=2$ ✓
→ **Strictly diagonally dominant.** By the Gershgorin + symmetry argument → PD. ✓

**$D_2$:** Row 1: $2 > 1+0=1$ ✓ | Row 2: $2 > 1+1=2$? NO (equality, not strict) | Row 3: $2 > 0+1=1$ ✓
→ **NOT strictly diagonally dominant.** But eigenvalues are $2-\sqrt{2}, 2, 2+\sqrt{2}$ — all positive → **still PD**. ✓

**$D_3$:** Row 1: $1 > 2$? NO | Row 2: $1 > 2$? NO
→ **NOT diagonally dominant.** Eigenvalues: $3, -1$ → **NOT PD (indefinite).**

**Key takeaway:** Strict diagonal dominance is **sufficient** for PD (for symmetric matrices
with positive diagonal), but **not necessary**. $D_2$ shows a matrix can fail the diagonal
dominance test yet still be PD — other tests (eigenvalues, Cholesky, Sylvester) are needed
to confirm.

</details>

---

## Section C — SVD

**C1.** ⭐ Compute the full SVD of $A = \begin{pmatrix} 3 & 0 \\ 0 & -2 \end{pmatrix}$ by hand.

<details><summary>Answer</summary>

**Step 1:** $A^T A = \begin{pmatrix} 9 & 0 \\ 0 & 4 \end{pmatrix}$. Eigenvalues: $\sigma_1^2 = 9$, $\sigma_2^2 = 4$.

**Step 2:** Singular values: $\sigma_1 = 3$, $\sigma_2 = 2$.

**Step 3:** Right singular vectors (eigenvectors of $A^T A$):
$v_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$, $v_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$. So $V = I$.

**Step 4:** Left singular vectors: $u_i = Av_i / \sigma_i$.
$u_1 = \frac{1}{3}\begin{pmatrix} 3 \\ 0 \end{pmatrix} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$,
$u_2 = \frac{1}{2}\begin{pmatrix} 0 \\ -2 \end{pmatrix} = \begin{pmatrix} 0 \\ -1 \end{pmatrix}$.

$$A = \underbrace{\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}}_{U} \underbrace{\begin{pmatrix} 3 & 0 \\ 0 & 2 \end{pmatrix}}_{\Sigma} \underbrace{\begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}}_{V^T}$$

**Note:** $U$ is orthogonal (not a rotation — $\det(U) = -1$). Singular values are always non-negative, so the sign of $-2$ is absorbed into $U$.

</details>

**C2.** ⭐⭐ Compute the SVD of $A = \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}$ by hand. What is the rank? Identify the four fundamental subspaces (column space, null space, row space, left null space) from the SVD.

<details><summary>Answer</summary>

**$A^TA$:**
$A^TA = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$

Eigenvalues: $\lambda_1 = 2, \lambda_2 = 0$. So $\sigma_1 = \sqrt{2}, \sigma_2 = 0$.

**Rank = 1** (one nonzero singular value).

Eigenvectors of $A^TA$:
$v_1 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ 1 \end{pmatrix}$ (for $\lambda = 2$),
$v_2 = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 \\ -1 \end{pmatrix}$ (for $\lambda = 0$).

$u_1 = Av_1 / \sigma_1 = \frac{1}{\sqrt{2}} \begin{pmatrix} 2/\sqrt{2} \\ 0 \end{pmatrix} / \sqrt{2} = \begin{pmatrix} 1 \\ 0 \end{pmatrix}$

Complete $U$ with $u_2 = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$.

$$A = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} \begin{pmatrix} \sqrt{2} & 0 \\ 0 & 0 \end{pmatrix} \begin{pmatrix} 1/\sqrt{2} & 1/\sqrt{2} \\ 1/\sqrt{2} & -1/\sqrt{2} \end{pmatrix}$$

**Four subspaces:**
- **Column space** $C(A) = \text{span}(u_1) = \text{span}\{(1, 0)^T\}$ — the first row direction
- **Row space** $C(A^T) = \text{span}(v_1) = \text{span}\{(1, 1)^T / \sqrt{2}\}$
- **Null space** $N(A) = \text{span}(v_2) = \text{span}\{(1, -1)^T / \sqrt{2}\}$
- **Left null space** $N(A^T) = \text{span}(u_2) = \text{span}\{(0, 1)^T\}$

**Python verification:**
```python
import numpy as np
A = np.array([[1, 1], [0, 0]])
U, s, Vt = np.linalg.svd(A)
print(f"σ = {s}")         # [1.414..., 0.]
print(f"U =\n{U}")        # [[1, 0], [0, 1]] (or sign-flipped)
print(f"V^T =\n{Vt}")     # [[0.707, 0.707], [0.707, -0.707]]
print(f"Rank = {np.sum(s > 1e-10)}")  # 1
# Verify: null space is V[:, 1:]
print(f"Null space basis: {Vt[1, :]}")  # [0.707, -0.707] → span{(1,-1)}
```

</details>

**C3.** ⭐⭐ A $100 \times 50$ matrix $A$ has singular values that decay as $\sigma_i = 10 / i^2$.

1. What is $\sigma_1$? $\sigma_{50}$? $\kappa(A)$?
2. What rank-$k$ approximation captures 99% of $\|A\|_F^2$? (Energy = $\sum \sigma_i^2$.)
3. What is the Frobenius-norm error of keeping only the top 5 singular values?

<details><summary>Answer</summary>

**1.**
$\sigma_1 = 10$, $\sigma_{50} = 10/2500 = 0.004$, $\kappa = 10/0.004 = 2500$.

**2.** Total energy: $\|A\|_F^2 = \sum_{i=1}^{50} (10/i^2)^2 = 100 \sum_{i=1}^{50} 1/i^4$

$\sum_{i=1}^{50} 1/i^4 \approx \pi^4/90 \approx 1.0823$ (converges quickly). So total $\approx 108.23$.

Energy from top $k$: $E_k = 100 \sum_{i=1}^{k} 1/i^4$.

$E_1 = 100$, $E_2 = 100 + 6.25 = 106.25$, $E_3 = 106.25 + 1.23 = 107.48$

$E_3/108.23 \approx 99.3\%$. So **rank 3** captures 99% of the energy.

**3.** Error: $\|A - A_5\|_F = \sqrt{\sum_{i=6}^{50} \sigma_i^2} = 10\sqrt{\sum_{i=6}^{50} 1/i^4}$

$\sum_{i=6}^{50} 1/i^4 \approx 1.0823 - \sum_{i=1}^{5} 1/i^4 = 1.0823 - (1 + 0.0625 + 0.0123 + 0.0039 + 0.0016) = 1.0823 - 1.0803 = 0.0020$

Error $\approx 10\sqrt{0.002} \approx 0.45$.

**Lesson:** Rapidly decaying singular values mean low-rank approximation works extremely well. This is common in robotics data (e.g., point clouds, correlation matrices).

</details>

**C4.** ⭐⭐⭐ Using the pseudoinverse, find the minimum-norm least-squares solution to:

$$\begin{pmatrix} 1 & 2 \\ 1 & 2 \\ 0 & 0 \end{pmatrix} x = \begin{pmatrix} 3 \\ 3 \\ 1 \end{pmatrix}$$

Explain why there's no exact solution and why the pseudoinverse gives the "best" approximate answer.

<details><summary>Answer</summary>

**Step 1: SVD of $A$.**

$A^TA = \begin{pmatrix} 2 & 4 \\ 4 & 8 \end{pmatrix}$. Eigenvalues: $\lambda_1 = 10, \lambda_2 = 0$. So $\sigma_1 = \sqrt{10}$, rank = 1.

$v_1 = \frac{1}{\sqrt{5}}(1, 2)^T$, $v_2 = \frac{1}{\sqrt{5}}(2, -1)^T$.

$u_1 = Av_1/\sigma_1 = \frac{1}{\sqrt{10}}\begin{pmatrix}5\\5\\0\end{pmatrix}/\sqrt{10} = \frac{1}{\sqrt{2}}\begin{pmatrix}1\\1\\0\end{pmatrix}$

Complete: $u_2 = \frac{1}{\sqrt{2}}(1,-1,0)^T$, $u_3 = (0,0,1)^T$.

**Step 2: Pseudoinverse.**
$A^+ = V\Sigma^+U^T = \frac{1}{\sqrt{5}}\begin{pmatrix}1\\2\end{pmatrix} \cdot \frac{1}{\sqrt{10}} \cdot \frac{1}{\sqrt{2}}(1, 1, 0)$

$= \frac{1}{10}\begin{pmatrix}1 & 1 & 0 \\ 2 & 2 & 0\end{pmatrix}$

**Step 3: Solution.**
$x^+ = A^+b = \frac{1}{10}\begin{pmatrix}6\\12\end{pmatrix} = \begin{pmatrix}0.6\\1.2\end{pmatrix}$

**Why no exact solution:** $b = (3,3,1)^T$ but $C(A) = \text{span}\{(1,1,0)^T\}$. The component $(0,0,1)^T$ is in $C(A)^\perp$ — no $x$ can produce it. The pseudoinverse projects $b$ onto $C(A)$ and solves for the minimum-norm $x$.

**Residual:** $Ax^+ = (3, 3, 0)^T$, residual $= (0, 0, 1)^T$, $\|r\| = 1$. This is the smallest possible residual.

</details>

---

## Section D — Matrix Norms

**D1.** ⭐ Compute $\|A\|_1$, $\|A\|_\infty$, $\|A\|_F$, and $\|A\|_2$ for:

$$A = \begin{pmatrix} 3 & -1 \\ 2 & 4 \end{pmatrix}$$

<details><summary>Answer</summary>

**$\|A\|_1$ (max column sum):**
Column 1: $|3| + |2| = 5$. Column 2: $|-1| + |4| = 5$. $\|A\|_1 = 5$.

**$\|A\|_\infty$ (max row sum):**
Row 1: $|3| + |-1| = 4$. Row 2: $|2| + |4| = 6$. $\|A\|_\infty = 6$.

**$\|A\|_F$ (Frobenius):**
$\sqrt{9 + 1 + 4 + 16} = \sqrt{30} \approx 5.48$.

**$\|A\|_2$ (spectral / largest singular value):**
$A^TA = \begin{pmatrix} 13 & 5 \\ 5 & 17 \end{pmatrix}$. Eigenvalues: $\lambda = 15 \pm \sqrt{4+25} = 15 \pm \sqrt{29}$.
$\lambda_{\max} = 15 + \sqrt{29} \approx 20.39$. $\sigma_{\max} = \sqrt{20.39} \approx 4.52$.

So $\|A\|_2 \approx 4.52$.

**Check relationship:** $\|A\|_2 \leq \|A\|_F$ → $4.52 \leq 5.48$ ✓

</details>

**D2.** ⭐⭐ Prove that $\|A\|_2 \leq \|A\|_F \leq \sqrt{r} \|A\|_2$ where $r = \text{rank}(A)$.

<details><summary>Answer</summary>

Let the singular values be $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$.

$\|A\|_2 = \sigma_1$ and $\|A\|_F = \sqrt{\sum_{i=1}^r \sigma_i^2}$.

**Left inequality:** $\sigma_1 \leq \sqrt{\sigma_1^2 + \sigma_2^2 + \cdots + \sigma_r^2}$ since adding non-negative terms under the square root increases it. ✓

**Right inequality:** $\sigma_i \leq \sigma_1$ for all $i$, so:
$$\sum_{i=1}^r \sigma_i^2 \leq \sum_{i=1}^r \sigma_1^2 = r \sigma_1^2$$

Therefore $\|A\|_F = \sqrt{\sum \sigma_i^2} \leq \sqrt{r} \sigma_1 = \sqrt{r} \|A\|_2$. ✓

**When are the bounds tight?**
- Left bound: tight when $r = 1$ (rank-1 matrix).
- Right bound: tight when all nonzero singular values are equal ($\sigma_1 = \sigma_2 = \cdots = \sigma_r$), e.g., $\alpha I$.

</details>

---

## Section E — Conditioning and Regularization

**E1.** ⭐⭐ You're solving a SLAM problem and the Jacobian $J$ has singular values:

$$\sigma = [100, 50, 20, 5, 0.01, 0.0001]$$

1. What is $\kappa(J)$? $\kappa(J^TJ)$?
2. Using truncated SVD with threshold $\tau = 0.1$, what is the numerical rank?
3. What does the null space ($v_5, v_6$) tell you about the SLAM problem?
4. If you apply Tikhonov regularization with $\lambda = 1$, what is the new effective condition number?

<details><summary>Answer</summary>

**1.**
$\kappa(J) = 100 / 0.0001 = 10^6$ (ill-conditioned but manageable with QR).
$\kappa(J^TJ) = (10^6)^2 = 10^{12}$ (catastrophic with normal equations!).

**2.** Numerical rank = number of $\sigma_i > 0.1$: that's $\{100, 50, 20, 5\}$ → **rank 4**.

**3.** The two directions $v_5, v_6$ (corresponding to $\sigma_5 = 0.01$ and $\sigma_6 = 0.0001$) represent **nearly unobservable state dimensions**. In SLAM, these are typically:
- Absolute position (gauge freedom — can shift entire map)
- Absolute orientation (can rotate entire map)

These modes can't be determined from relative measurements alone.

**4.** With Tikhonov: $\kappa_{\text{eff}} = \frac{\sigma_1^2 + \lambda}{\sigma_6^2 + \lambda} = \frac{10000 + 1}{10^{-8} + 1} \approx \frac{10001}{1} \approx 10001$.

Dramatically improved from $10^{12}$! But the solution is biased — the unobservable modes are pulled toward zero.

</details>

**E2.** ⭐⭐⭐ *Catastrophic cancellation in action.* Consider:

$$A = \begin{pmatrix} 1 & 1 \\ 1 & 1 + \epsilon \end{pmatrix}, \quad b = \begin{pmatrix} 2 \\ 2 + \epsilon \end{pmatrix}$$

1. What is the true solution $x$ for any $\epsilon > 0$?
2. Compute $\kappa(A)$ as a function of $\epsilon$.
3. For $\epsilon = 10^{-12}$, solve in Python and compare to the true solution.

<details><summary>Answer</summary>

**1.** By inspection or elimination: $x_1 + x_2 = 2$ and $x_1 + (1+\epsilon)x_2 = 2 + \epsilon$.
Subtracting: $\epsilon x_2 = \epsilon$, so $x_2 = 1$, $x_1 = 1$.

True solution: $x = (1, 1)^T$ for all $\epsilon > 0$.

**2.** Eigenvalues of $A$: $\lambda = 1 + \frac{\epsilon}{2} \pm \sqrt{1 + \frac{\epsilon^2}{4}}$.
For small $\epsilon$: $\lambda_{\max} \approx 2 + \epsilon/2$, $\lambda_{\min} \approx \epsilon/2$.
$\kappa(A) \approx (2 + \epsilon/2)/(\epsilon/2) \approx 4/\epsilon$ for small $\epsilon$.

For $\epsilon = 10^{-12}$: $\kappa \approx 4 \times 10^{12}$. Almost all digits are lost!

**3.** Python verification:

```python
import numpy as np

eps = 1e-12
A = np.array([[1.0, 1.0], [1.0, 1.0 + eps]])
b = np.array([2.0, 2.0 + eps])

# Direct solve
x = np.linalg.solve(A, b)
print(f"Solution: {x}")
print(f"True solution: [1.0, 1.0]")
print(f"Error: {np.linalg.norm(x - np.array([1.0, 1.0]))}")
print(f"Condition number: {np.linalg.cond(A):.2e}")
```

You'll see errors on the order of $10^{-4}$ to $10^{-2}$ — the solution is wildly inaccurate despite the true answer being the simple vector $(1, 1)^T$.

</details>

**E3.** ⭐⭐⭐ *The L-curve in practice.* For the system $Ax = b$ where:

$$A = \begin{pmatrix} 1 & 0 \\ 0 & 10^{-8} \end{pmatrix}, \quad b = \begin{pmatrix} 1 + \delta_1 \\ 10^{-8} + \delta_2 \end{pmatrix}$$

with noise $\delta_1, \delta_2 \sim \mathcal{N}(0, 10^{-4})$. True solution: $x = (1, 1)^T$.

Write Python code to:
1. Solve with Tikhonov regularization for $\lambda \in [10^{-16}, 10^{0}]$
2. Plot the L-curve ($\log\|x_\lambda\|$ vs $\log\|Ax_\lambda - b\|$)
3. Identify the optimal $\lambda$ at the L-curve corner

<details><summary>Answer</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

A = np.diag([1.0, 1e-8])
x_true = np.array([1.0, 1.0])
np.random.seed(42)
noise = np.random.normal(0, 1e-4, 2)
b = A @ x_true + noise

lambdas = np.logspace(-16, 0, 200)
residuals = []
sol_norms = []

for lam in lambdas:
    x_lam = np.linalg.solve(A.T @ A + lam * np.eye(2), A.T @ b)
    residuals.append(np.linalg.norm(A @ x_lam - b))
    sol_norms.append(np.linalg.norm(x_lam))

plt.figure(figsize=(8, 6))
plt.loglog(residuals, sol_norms, 'b-', linewidth=1.5)
plt.xlabel(r'$\|Ax_\lambda - b\|_2$')
plt.ylabel(r'$\|x_\lambda\|_2$')
plt.title('L-Curve for Tikhonov Regularization')

# Mark the corner (maximum curvature)
log_r = np.log(residuals)
log_n = np.log(sol_norms)
# Approximate curvature from discrete differences
dr = np.gradient(log_r)
dn = np.gradient(log_n)
ddr = np.gradient(dr)
ddn = np.gradient(dn)
curvature = np.abs(dr * ddn - dn * ddr) / (dr**2 + dn**2)**1.5
corner_idx = np.argmax(curvature[10:-10]) + 10  # avoid edges
plt.plot(residuals[corner_idx], sol_norms[corner_idx], 'ro', markersize=10)
plt.annotate(f'λ = {lambdas[corner_idx]:.2e}', 
             xy=(residuals[corner_idx], sol_norms[corner_idx]),
             xytext=(10, 10), textcoords='offset points')

plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('l_curve.png', dpi=150)
plt.show()

# Compare solutions
for lam in [0, 1e-12, lambdas[corner_idx], 1e-2]:
    if lam == 0:
        x = np.linalg.solve(A.T @ A, A.T @ b)
    else:
        x = np.linalg.solve(A.T @ A + lam * np.eye(2), A.T @ b)
    print(f"λ={lam:.1e}: x={x}, error={np.linalg.norm(x - x_true):.4e}")
```

The L-curve should show:
- **Top-left:** Large $\lambda$ → small solution norm but large residual (over-regularized)
- **Bottom-right:** Small $\lambda$ → small residual but large solution norm (noise amplification)
- **Corner:** The balance point — optimal regularization

</details>

---

## Section F — Special Matrices in Robotics

**F1.** ⭐ Verify that $R(\pi/4) = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}$ satisfies all rotation matrix properties: $R^TR = I$, $\det(R) = +1$, $\kappa(R) = 1$.

<details><summary>Answer</summary>

**$R^TR = I$:**
$R^TR = \frac{1}{2}\begin{pmatrix}1 & 1 \\ -1 & 1\end{pmatrix}\begin{pmatrix}1 & -1 \\ 1 & 1\end{pmatrix} = \frac{1}{2}\begin{pmatrix}2 & 0 \\ 0 & 2\end{pmatrix} = I$ ✓

**$\det(R)$:**
$\det(R) = \frac{1}{2}(1 \cdot 1 - (-1) \cdot 1) = \frac{1}{2}(1 + 1) = 1$ ✓

**$\kappa(R)$:**
Since $R^TR = I$, both singular values equal 1. $\kappa = 1/1 = 1$. ✓

**Eigenvalues:** $\lambda = e^{\pm i\pi/4} = \frac{1}{\sqrt{2}} \pm \frac{i}{\sqrt{2}}$. Both have $|\lambda| = 1$.

</details>

**F2.** ⭐⭐ Using Rodrigues' rotation formula, compute the 3D rotation matrix for axis
$\hat{\omega} = (0, 0, 1)^T$ (z-axis) and angle $\theta = \pi/3$ (60°). Then verify:
1. $R^TR = I_3$
2. $\det(R) = +1$
3. The rotation acts correctly on the vector $p = (1, 0, 0)^T$

<details><summary>Answer</summary>

**Rodrigues' formula:** $R = I + (\sin\theta)[\hat{\omega}]_\times + (1-\cos\theta)[\hat{\omega}]_\times^2$

**Skew-symmetric matrix:** $[\hat{\omega}]_\times = \begin{pmatrix} 0 & -1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix}$

$[\hat{\omega}]_\times^2 = \begin{pmatrix} -1 & 0 & 0 \\ 0 & -1 & 0 \\ 0 & 0 & 0 \end{pmatrix}$

With $\theta = \pi/3$: $\sin(\pi/3) = \sqrt{3}/2$, $\cos(\pi/3) = 1/2$:

$$R = \begin{pmatrix}1&0&0\\0&1&0\\0&0&1\end{pmatrix} + \frac{\sqrt{3}}{2}\begin{pmatrix}0&-1&0\\1&0&0\\0&0&0\end{pmatrix} + \frac{1}{2}\begin{pmatrix}-1&0&0\\0&-1&0\\0&0&0\end{pmatrix}$$

$$R = \begin{pmatrix} 1/2 & -\sqrt{3}/2 & 0 \\ \sqrt{3}/2 & 1/2 & 0 \\ 0 & 0 & 1 \end{pmatrix}$$

**1. Orthogonality:**
$R^TR$: Row/column dot products give $I_3$ since each row has unit norm
($1/4 + 3/4 = 1$) and rows are orthogonal ($1 \cdot \sqrt{3}/4 - \sqrt{3} \cdot 1/4 = 0$). ✓

**2. Determinant:**
$\det(R) = 1 \cdot (1/4 + 3/4) = 1$ (expanding along the third row). ✓

**3. Action on $p = (1,0,0)^T$:**
$Rp = (1/2, \sqrt{3}/2, 0)^T$ — a 60° rotation in the $xy$-plane as expected. ✓

**Python verification:**
```python
import numpy as np
theta = np.pi / 3
omega_hat = np.array([0, 0, 1.0])
K = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0.0]])  # skew-symmetric
R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
print(f"R^T R = I? {np.allclose(R.T @ R, np.eye(3))}")  # True
print(f"det(R) = {np.linalg.det(R):.6f}")                # 1.000000
print(f"R @ [1,0,0] = {R @ [1,0,0]}")                    # [0.5, 0.866, 0]
```

</details>

**F3.** ⭐⭐ A 2D pose-graph SLAM problem has 4 poses in a loop. The information matrix (graph Laplacian) is:

$$\Lambda = \begin{pmatrix} 2 & -1 & 0 & -1 \\ -1 & 2 & -1 & 0 \\ 0 & -1 & 2 & -1 \\ -1 & 0 & -1 & 2 \end{pmatrix}$$

1. Verify this is a valid graph Laplacian (each row sums to zero).
2. What are the eigenvalues? How many zero eigenvalues and what do they represent?
3. Why can't you directly solve $\Lambda x = b$?
4. How is this issue resolved in practice?

<details><summary>Answer</summary>

**1.** Row sums: $2-1+0-1=0$, $-1+2-1+0=0$, $0-1+2-1=0$, $-1+0-1+2=0$ ✓.

Each row sums to zero → valid Laplacian.

**2.** For a cycle graph $C_4$, eigenvalues are:
$\lambda_k = 2 - 2\cos(2\pi k/4)$ for $k = 0, 1, 2, 3$.

$\lambda_0 = 0$, $\lambda_1 = 2$, $\lambda_2 = 4$, $\lambda_3 = 2$.

So eigenvalues: $\{0, 2, 2, 4\}$. **One zero eigenvalue**, corresponding to eigenvector $\mathbf{1} = (1,1,1,1)^T$.

**Physical meaning:** Adding a constant to all poses doesn't change any relative measurement. This is the **gauge freedom** — the absolute position is unobservable from relative measurements alone.

**3.** $\Lambda$ is singular (has a zero eigenvalue), so $\Lambda x = b$ has no unique solution (or no solution, depending on $b$).

**4.** In practice, you either:
- **Pin one node** (anchor): set $x_1 = 0$ and solve the reduced $(n-1) \times (n-1)$ system
- **Add a prior**: replace $\Lambda$ with $\Lambda + \epsilon I$ (Tikhonov regularization)
- **Use the pseudoinverse**: $x = \Lambda^+ b$ (minimum-norm solution, centered at zero mean)

</details>

**F4.** ⭐⭐⭐ The Jacobian of a 2D robot observing a landmark at relative position $(r, \theta)$ is:

$$J = \begin{pmatrix} \frac{\partial r}{\partial x} & \frac{\partial r}{\partial y} & \frac{\partial r}{\partial \phi} \\ \frac{\partial \theta}{\partial x} & \frac{\partial \theta}{\partial y} & \frac{\partial \theta}{\partial \phi} \end{pmatrix} = \begin{pmatrix} \cos\alpha & \sin\alpha & 0 \\ -\sin\alpha / d & \cos\alpha / d & -1 \end{pmatrix}$$

where $d$ is the distance and $\alpha$ is the bearing.

1. What are the singular values of $J$ as a function of $d$?
2. How does $\kappa(J)$ change as the robot moves closer or further from the landmark?
3. At what distance does the problem become ill-conditioned?

<details><summary>Answer</summary>

**1.** Compute $J^TJ$:

Let $c = \cos\alpha$, $s = \sin\alpha$.

$J^TJ = \begin{pmatrix} c^2 + s^2/d^2 & cs - cs/d^2 & s/d \\ cs - cs/d^2 & s^2 + c^2/d^2 & -c/d \\ s/d & -c/d & 1 \end{pmatrix}$

$= \begin{pmatrix} 1 + (s/d)^2 - s^2 & cs(1 - 1/d^2) & s/d \\ cs(1-1/d^2) & 1 + (c/d)^2 - c^2 & -c/d \\ s/d & -c/d & 1 \end{pmatrix}$

This simplifies when we note that $J$ has only 2 rows and 3 columns, so rank $\leq 2$. The smallest singular value of $J$ is always 0 (the orientation $\phi$ and position are coupled, but a single landmark with range+bearing provides only 2 constraints for 3 unknowns).

For the two nonzero singular values: $\sigma_1 \approx \sqrt{1 + 1/d^2}$ and $\sigma_2 \approx 1$ (for the range measurement).

**2.** As $d \to 0$: $\sigma_1 \to \infty$ (bearing measurement becomes very sensitive), $\sigma_2 \approx 1$. The problem is well-conditioned for the observable directions.

As $d \to \infty$: $\sigma_1 \to 1$, $\sigma_2 \to 1$. Both singular values approach 1 — well-conditioned but the information content is low (measurements barely constrain the state).

**3.** Since rank($J$) = 2 for a single landmark, you always need multiple landmarks to fully constrain the robot pose. The conditioning depends on the **geometry** of landmarks, not just distance. Collinear landmarks (all at similar $\alpha$) cause ill-conditioning regardless of distance.

</details>

---

## Section G — Python Verification Problems

**G1.** ⭐⭐ Write a function `classify_matrix(A, tol=1e-10)` that returns a dictionary with keys: `"symmetric"`, `"pd"`, `"psd"`, `"indefinite"`, `"orthogonal"`, `"singular"`, `"condition_number"`, `"rank"`.

<details><summary>Answer</summary>

```python
import numpy as np

def classify_matrix(A, tol=1e-10):
    """Classify a matrix's properties."""
    n = A.shape[0]
    result = {}
    
    # Symmetry
    result["symmetric"] = np.allclose(A, A.T, atol=tol)
    
    # Orthogonality
    if A.shape[0] == A.shape[1]:
        result["orthogonal"] = np.allclose(A.T @ A, np.eye(n), atol=tol)
    else:
        result["orthogonal"] = False
    
    # SVD for rank and condition number
    U, s, Vt = np.linalg.svd(A)
    result["rank"] = int(np.sum(s > tol * s[0] * max(A.shape)))
    result["singular"] = (result["rank"] < min(A.shape))
    result["condition_number"] = s[0] / s[-1] if s[-1] > tol else np.inf
    
    # Definiteness (only for symmetric)
    if result["symmetric"]:
        eigvals = np.linalg.eigvalsh(A)
        if np.all(eigvals > tol):
            result["pd"] = True
            result["psd"] = True
            result["indefinite"] = False
        elif np.all(eigvals > -tol):
            result["pd"] = False
            result["psd"] = True
            result["indefinite"] = False
        else:
            result["pd"] = False
            result["psd"] = False
            result["indefinite"] = True
    else:
        result["pd"] = None
        result["psd"] = None
        result["indefinite"] = None
    
    return result

# Test
A = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]])
print(classify_matrix(A))
# {'symmetric': True, 'orthogonal': False, 'rank': 3, 'singular': False,
#  'condition_number': ~8.8, 'pd': True, 'psd': True, 'indefinite': False}
```

</details>

**G2.** ⭐⭐ Implement `nearest_pd(A, epsilon=1e-6)` that takes any symmetric matrix and returns the nearest PD matrix using Higham's algorithm. Verify with Cholesky.

<details><summary>Answer</summary>

```python
import numpy as np

def nearest_pd(A, epsilon=1e-6):
    """Find nearest positive definite matrix to A (Higham 1988)."""
    # Ensure symmetry
    A_sym = (A + A.T) / 2
    
    # Eigendecomposition
    eigvals, Q = np.linalg.eigh(A_sym)
    
    # Clip eigenvalues
    eigvals_clipped = np.maximum(eigvals, epsilon)
    
    # Reconstruct
    A_pd = Q @ np.diag(eigvals_clipped) @ Q.T
    
    # Ensure perfect symmetry (numerical safety)
    A_pd = (A_pd + A_pd.T) / 2
    
    return A_pd

# Test with the noisy covariance from B3
Sigma_hat = np.array([[4.0, 1.0, 3.1],
                       [1.0, 2.0, -0.5],
                       [3.1, -0.5, 1.0]])

# Original: Cholesky fails
try:
    np.linalg.cholesky(Sigma_hat)
    print("Original is PD")
except np.linalg.LinAlgError:
    print("Original is NOT PD")

# Fixed: Cholesky succeeds
Sigma_fixed = nearest_pd(Sigma_hat)
L = np.linalg.cholesky(Sigma_fixed)
print(f"Fixed matrix:\n{Sigma_fixed}")
print(f"Eigenvalues: {np.linalg.eigvalsh(Sigma_fixed)}")
print(f"Cholesky succeeded, L diag: {np.diag(L)}")
```

</details>

**G3.** ⭐⭐⭐ Implement `tikhonov_solve(A, b, lam)` and `truncated_svd_solve(A, b, k)`. Compare their behavior on an ill-conditioned system.

<details><summary>Answer</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

def tikhonov_solve(A, b, lam):
    """Tikhonov-regularized least-squares solution."""
    n = A.shape[1]
    return np.linalg.solve(A.T @ A + lam * np.eye(n), A.T @ b)

def truncated_svd_solve(A, b, k):
    """Truncated SVD solution keeping top k singular values."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    # Keep only top k
    x = np.zeros(A.shape[1])
    for i in range(k):
        x += (U[:, i] @ b / s[i]) * Vt[i, :]
    return x

# Create ill-conditioned problem
n = 20
np.random.seed(42)
# Singular values with sharp decay
s_true = np.logspace(0, -10, n)
U, _ = np.linalg.qr(np.random.randn(n, n))
V, _ = np.linalg.qr(np.random.randn(n, n))
A = U @ np.diag(s_true) @ V.T

x_true = np.ones(n)
b_clean = A @ x_true
noise = 1e-6 * np.random.randn(n)
b = b_clean + noise

print(f"Condition number: {np.linalg.cond(A):.2e}")

# Compare methods
x_direct = np.linalg.solve(A, b)
x_tikh = tikhonov_solve(A, b, lam=1e-8)
x_tsvd = truncated_svd_solve(A, b, k=10)

print(f"Direct solve error:        {np.linalg.norm(x_direct - x_true):.4e}")
print(f"Tikhonov (λ=1e-8) error:   {np.linalg.norm(x_tikh - x_true):.4e}")
print(f"Truncated SVD (k=10) error: {np.linalg.norm(x_tsvd - x_true):.4e}")

# Sweep lambda for Tikhonov
lambdas = np.logspace(-14, 2, 100)
errors = [np.linalg.norm(tikhonov_solve(A, b, lam) - x_true) for lam in lambdas]

plt.figure(figsize=(8, 5))
plt.semilogx(lambdas, errors, 'b-')
plt.xlabel('λ (regularization)')
plt.ylabel('‖x_λ - x_true‖')
plt.title('Tikhonov Regularization: Error vs λ')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tikhonov_error.png', dpi=150)
plt.show()
```

**Expected output:** Direct solve has large error due to noise amplification in small singular value directions. Both Tikhonov and truncated SVD dramatically reduce error by damping or removing those directions.

</details>

**G4.** ⭐⭐⭐ Write a function `svd_analysis(A)` that prints a comprehensive report: rank, condition number, energy distribution, 4 fundamental subspaces, and a singular value bar chart.

<details><summary>Answer</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

def svd_analysis(A, tol=None):
    """Comprehensive SVD analysis of a matrix."""
    m, n = A.shape
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    
    if tol is None:
        tol = max(m, n) * s[0] * np.finfo(float).eps
    
    r = np.sum(s > tol)
    energy = np.cumsum(s**2) / np.sum(s**2)
    
    print("=" * 60)
    print(f"SVD Analysis: {m}×{n} matrix")
    print("=" * 60)
    print(f"\nRank: {r} (out of min(m,n) = {min(m,n)})")
    print(f"Condition number: {s[0]/s[min(m,n)-1]:.4e}" if s[min(m,n)-1] > 0 
          else f"Condition number: ∞ (singular)")
    
    print(f"\nSingular values:")
    for i, si in enumerate(s[:min(10, len(s))]):
        bar = "█" * max(1, int(40 * si / s[0]))
        print(f"  σ_{i+1:2d} = {si:12.6e}  {bar}  ({energy[i]*100:.1f}% energy)")
    if len(s) > 10:
        print(f"  ... ({len(s) - 10} more)")
    
    print(f"\nFour Fundamental Subspaces:")
    print(f"  Column space C(A):  dim = {r}, basis = first {r} columns of U")
    print(f"  Row space C(Aᵀ):   dim = {r}, basis = first {r} rows of Vᵀ")
    print(f"  Null space N(A):    dim = {n - r}", end="")
    if n - r > 0:
        print(f", basis = last {n-r} rows of Vᵀ")
    else:
        print(" (trivial)")
    print(f"  Left null N(Aᵀ):   dim = {m - r}", end="")
    if m - r > 0:
        print(f", basis = last {m-r} columns of U")
    else:
        print(" (trivial)")
    
    # Plot singular values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.bar(range(1, len(s)+1), s, color='steelblue')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Singular Value')
    ax1.set_title('Singular Value Spectrum')
    ax1.set_yscale('log')
    
    ax2.plot(range(1, len(s)+1), energy * 100, 'ro-')
    ax2.axhline(y=99, color='gray', linestyle='--', alpha=0.5, label='99% threshold')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Energy (%)')
    ax2.set_title('Energy Distribution')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('svd_analysis.png', dpi=150)
    plt.show()
    
    return {"U": U, "s": s, "Vt": Vt, "rank": r, "condition": s[0]/max(s[-1], 1e-300)}

# Example: a SLAM-like Jacobian
np.random.seed(42)
# 10 measurements, 6 state variables, rank 5 (one unobservable direction)
A = np.random.randn(10, 5) @ np.random.randn(5, 6)  # rank ≤ 5
result = svd_analysis(A)
```

</details>

---

## Section G2 — Matrix Calculus

**G5.** ⭐⭐ Derive the gradient of the least-squares cost $f(x) = \|Ax - b\|^2$ from first principles.

*Hint:* Expand the squared norm as $(Ax-b)^T(Ax-b)$, distribute, then differentiate each term with respect to $x$.

<details><summary>Answer</summary>

**Expand:**
$$f(x) = (Ax - b)^T(Ax - b) = x^T A^T A x - 2b^T A x + b^T b$$

**Differentiate term by term** using matrix calculus identities:
- $\nabla_x (x^T A^T A x) = 2 A^T A x$ (since $A^T A$ is symmetric)
- $\nabla_x (-2 b^T A x) = -2 A^T b$
- $\nabla_x (b^T b) = 0$

$$\boxed{\nabla_x f = 2 A^T A x - 2 A^T b = 2 A^T(Ax - b)}$$

Setting $\nabla_x f = 0$ gives the normal equations: $A^T A x = A^T b$.

**Key identities used:**
- $\nabla_x (x^T M x) = (M + M^T)x = 2Mx$ when $M$ is symmetric
- $\nabla_x (c^T x) = c$

**Python verification:**
```python
import numpy as np
A = np.random.randn(5, 3); b = np.random.randn(5)
x = np.random.randn(3)
# Analytical gradient
grad_analytical = 2 * A.T @ (A @ x - b)
# Numerical gradient (finite differences)
eps = 1e-7
grad_numerical = np.zeros(3)
for i in range(3):
    x_plus = x.copy(); x_plus[i] += eps
    x_minus = x.copy(); x_minus[i] -= eps
    grad_numerical[i] = (np.linalg.norm(A @ x_plus - b)**2 
                        - np.linalg.norm(A @ x_minus - b)**2) / (2*eps)
print(f"Match: {np.allclose(grad_analytical, grad_numerical)}")  # True
```

</details>

**G6.** ⭐⭐ In Gauss-Newton optimization, the cost is $f(x) = \frac{1}{2}\|r(x)\|^2$ where
$r(x)$ is a vector of residuals. Using the chain rule:

1. Show that $\nabla f = J^T r$ where $J = \partial r / \partial x$.
2. Show that the Gauss-Newton approximation to the Hessian is $H_{GN} = J^T J$.
3. Why is $J^T J$ always PSD? Under what condition is it PD?

<details><summary>Answer</summary>

**1. Gradient via chain rule:**
$$f(x) = \frac{1}{2} r(x)^T r(x)$$
$$\nabla_x f = \frac{\partial r}{\partial x}^T r(x) = J^T r$$

This follows from the chain rule: $\frac{\partial f}{\partial x_i} = \sum_k r_k \frac{\partial r_k}{\partial x_i}$, which in matrix form is $J^T r$.

**2. Gauss-Newton Hessian:**
The exact Hessian is: $H = J^T J + \sum_k r_k \nabla^2 r_k$

The Gauss-Newton approximation **drops the second-order term** $\sum_k r_k \nabla^2 r_k$,
giving $H_{GN} = J^T J$. This is valid when:
- Residuals $r_k$ are small (near the solution), or
- Residuals are nearly linear (so $\nabla^2 r_k \approx 0$)

**3. PSD/PD of $J^TJ$:**
For any $v \neq 0$: $v^T J^T J v = \|Jv\|^2 \geq 0$. So $J^T J \succeq 0$ (PSD). ✓

$J^T J \succ 0$ (PD) iff $\|Jv\|^2 > 0$ for all $v \neq 0$, i.e., $N(J) = \{0\}$,
i.e., $J$ has **full column rank**. If $J$ is rank-deficient, $J^T J$ is singular and the
Gauss-Newton step $\Delta x = -(J^T J)^{-1} J^T r$ doesn't exist — this is exactly when
Levenberg-Marquardt adds damping: $(J^T J + \lambda I)\Delta x = -J^T r$.

</details>

---

## Section H — Synthesis and Robotics Scenarios

**H1.** ⭐⭐⭐ *Full pipeline exercise.* A SLAM system estimates 3 poses ($x, y, \theta$ each) from 4 relative measurements. The Jacobian is:

$$J = \begin{pmatrix}
1 & 0 & 0 & -1 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & -1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & -1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & -1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & -1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & -1 \\
-1 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & -1 & 0 & 0 & 0 & 0 & 0 & 1 & 0
\end{pmatrix}$$

(4 measurements: pose1→pose2 (3D), pose2→pose3 (3D), and a partial loop closure pose3→pose1 (only x, y).)

1. Compute $J^TJ$ and verify it is the graph Laplacian.
2. Find the null space of $J$. What is its physical interpretation?
3. Pin pose 1 at the origin and solve the reduced system for a given $b$.

<details><summary>Answer</summary>

```python
import numpy as np

# Construct J
J = np.zeros((8, 9))
# Measurement 1→2 (rows 0-2)
J[0, 0], J[0, 3] = 1, -1
J[1, 1], J[1, 4] = 1, -1
J[2, 2], J[2, 5] = 1, -1
# Measurement 2→3 (rows 3-5)
J[3, 3], J[3, 6] = 1, -1
J[4, 4], J[4, 7] = 1, -1
J[5, 5], J[5, 8] = 1, -1
# Loop closure 3→1 (rows 6-7, only x,y)
J[6, 6], J[6, 0] = 1, -1
J[7, 7], J[7, 1] = 1, -1

# 1. J^T J
H = J.T @ J
print("J^T J (information matrix):")
print(H.astype(int))

# Verify: each row sums to zero? Not exactly — it's a weighted Laplacian
# because pose 3's theta has no loop constraint
print(f"Row sums: {H.sum(axis=1)}")  # theta_3 row sum != 0

# 2. Null space
U, s, Vt = np.linalg.svd(J)
print(f"\nSingular values: {np.round(s, 4)}")
null_dim = np.sum(s < 1e-10)
print(f"Null space dimension: {null_dim}")
if null_dim > 0:
    null_vectors = Vt[-null_dim:]
    print(f"Null space basis:\n{np.round(null_vectors, 3)}")

# Physical interpretation: the null space vectors correspond to
# global transformations that don't change any measurement.
# For full 3-constraint connections: translate + rotate the whole graph.
# With partial loop closure: only translation in x,y is unobservable.

# 3. Pin pose 1 at origin
# Remove first 3 columns (x1, y1, theta1 = 0)
J_reduced = J[:, 3:]  # 8x6
b = np.array([0.5, 0.3, 0.1, -0.2, 0.4, 0.05, 0.3, 0.1])

x_reduced = np.linalg.lstsq(J_reduced, b, rcond=None)[0]
x_full = np.zeros(9)
x_full[3:] = x_reduced
print(f"\nSolution with pose 1 pinned:")
print(f"Pose 1: x={x_full[0]:.3f}, y={x_full[1]:.3f}, θ={x_full[2]:.3f}")
print(f"Pose 2: x={x_full[3]:.3f}, y={x_full[4]:.3f}, θ={x_full[5]:.3f}")
print(f"Pose 3: x={x_full[6]:.3f}, y={x_full[7]:.3f}, θ={x_full[8]:.3f}")
```

**Physical interpretation of null space:** Since the loop closure only constrains $x$ and $y$ (not $\theta$), the null space has dimension ≥ 1 containing a mode where all $\theta$ values shift by the same amount. The global translation is also unobservable (3 DOF). With the partial loop closure, some of this gauge freedom is broken.

</details>

**H2.** ⭐⭐⭐⭐ *Matrix conditioning in Levenberg-Marquardt.* Implement a simplified LM solver that:

1. Computes the Gauss-Newton Hessian $H = J^T J$
2. Checks $\kappa(H)$ and logs it
3. Adds $\lambda I$ (LM damping / Tikhonov)
4. Monitors how $\lambda$ affects the condition number
5. Solves and updates

Test on a simple 2D curve fitting: $y = a \cdot e^{-bx} + c$ with noisy data.

<details><summary>Answer</summary>

```python
import numpy as np
import matplotlib.pyplot as plt

def levenberg_marquardt(residual_fn, jacobian_fn, x0, max_iter=50, 
                        lam0=1e-3, tol=1e-8):
    """Simplified LM solver with conditioning diagnostics."""
    x = x0.copy()
    lam = lam0
    n = len(x)
    
    history = {"cost": [], "lambda": [], "kappa_H": [], "kappa_damped": [], "x": []}
    
    for iteration in range(max_iter):
        r = residual_fn(x)
        J = jacobian_fn(x)
        cost = 0.5 * r @ r
        
        # Gauss-Newton Hessian
        H = J.T @ J
        g = J.T @ r
        
        # Condition numbers
        kappa_H = np.linalg.cond(H)
        kappa_damped = np.linalg.cond(H + lam * np.eye(n))
        
        history["cost"].append(cost)
        history["lambda"].append(lam)
        history["kappa_H"].append(kappa_H)
        history["kappa_damped"].append(kappa_damped)
        history["x"].append(x.copy())
        
        # Solve damped system
        dx = np.linalg.solve(H + lam * np.eye(n), -g)
        
        # Evaluate new cost
        x_new = x + dx
        cost_new = 0.5 * residual_fn(x_new) @ residual_fn(x_new)
        
        # Accept/reject and update lambda
        rho = (cost - cost_new) / (0.5 * dx @ (lam * dx - g) + 1e-30)
        
        if rho > 0:
            x = x_new
            lam = max(lam / 3, 1e-15)
            if np.linalg.norm(dx) < tol:
                print(f"Converged at iteration {iteration}")
                break
        else:
            lam = min(lam * 2, 1e10)
        
        if iteration % 5 == 0:
            print(f"Iter {iteration:3d}: cost={cost:.6e}, λ={lam:.2e}, "
                  f"κ(H)={kappa_H:.2e}, κ(H+λI)={kappa_damped:.2e}")
    
    return x, history

# Test: fit y = a * exp(-b*x) + c
np.random.seed(42)
x_data = np.linspace(0, 5, 50)
a_true, b_true, c_true = 3.0, 1.5, 0.5
y_data = a_true * np.exp(-b_true * x_data) + c_true + 0.1 * np.random.randn(50)

def residual(params):
    a, b, c = params
    return a * np.exp(-b * x_data) + c - y_data

def jacobian(params):
    a, b, c = params
    J = np.zeros((len(x_data), 3))
    J[:, 0] = np.exp(-b * x_data)           # dr/da
    J[:, 1] = -a * x_data * np.exp(-b * x_data)  # dr/db
    J[:, 2] = np.ones(len(x_data))           # dr/dc
    return J

# Start from poor initial guess
x0 = np.array([1.0, 0.5, 0.0])
x_opt, hist = levenberg_marquardt(residual, jacobian, x0)
print(f"\nOptimized: a={x_opt[0]:.4f}, b={x_opt[1]:.4f}, c={x_opt[2]:.4f}")
print(f"True:      a={a_true}, b={b_true}, c={c_true}")

# Plot diagnostics
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].semilogy(hist["cost"])
axes[0, 0].set_title("Cost")
axes[0, 0].set_xlabel("Iteration")

axes[0, 1].semilogy(hist["lambda"])
axes[0, 1].set_title("λ (damping)")
axes[0, 1].set_xlabel("Iteration")

axes[1, 0].semilogy(hist["kappa_H"], label="κ(H)")
axes[1, 0].semilogy(hist["kappa_damped"], label="κ(H+λI)")
axes[1, 0].set_title("Condition Number")
axes[1, 0].set_xlabel("Iteration")
axes[1, 0].legend()

axes[1, 1].scatter(x_data, y_data, s=10, alpha=0.5, label="Data")
axes[1, 1].plot(x_data, x_opt[0]*np.exp(-x_opt[1]*x_data) + x_opt[2], 
                'r-', linewidth=2, label="Fit")
axes[1, 1].set_title("Curve Fit Result")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('lm_diagnostics.png', dpi=150)
plt.show()
```

**Key observations:**
- Early iterations: large $\lambda$ → well-conditioned (gradient descent-like)
- Late iterations: small $\lambda$ → near-Newton, higher $\kappa$ but faster convergence
- $\kappa(H + \lambda I)$ is always much smaller than $\kappa(H)$

</details>

**H3.** ⭐⭐⭐⭐ *Comprehensive matrix diagnosis.* Write a "matrix doctor" function that takes any matrix $A$ and outputs:

1. Classification (symmetric, orthogonal, triangular, sparse, etc.)
2. Definiteness (if symmetric)
3. SVD summary (rank, condition, energy profile)
4. Specific warnings (ill-conditioned, near-singular, etc.)
5. Recommended solver based on all the above

<details><summary>Answer</summary>

```python
import numpy as np
from scipy import sparse

def matrix_doctor(A, verbose=True):
    """Comprehensive matrix diagnosis with solver recommendation."""
    m, n = A.shape
    is_square = (m == n)
    report = {}
    
    # --- Structure ---
    report["shape"] = (m, n)
    report["square"] = is_square
    report["symmetric"] = is_square and np.allclose(A, A.T, atol=1e-12)
    report["skew_symmetric"] = is_square and np.allclose(A, -A.T, atol=1e-12)
    report["diagonal"] = is_square and np.allclose(A, np.diag(np.diag(A)), atol=1e-12)
    report["upper_triangular"] = np.allclose(A, np.triu(A), atol=1e-12)
    report["lower_triangular"] = np.allclose(A, np.tril(A), atol=1e-12)
    
    nnz_total = m * n
    nnz_actual = np.count_nonzero(np.abs(A) > 1e-15)
    report["sparsity"] = 1 - nnz_actual / nnz_total
    report["sparse"] = report["sparsity"] > 0.5
    
    if is_square:
        report["orthogonal"] = np.allclose(A.T @ A, np.eye(n), atol=1e-10)
    
    # --- SVD ---
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    tol = max(m, n) * s[0] * np.finfo(float).eps
    rank = int(np.sum(s > tol))
    kappa = s[0] / s[rank-1] if rank > 0 else np.inf
    
    report["rank"] = rank
    report["full_rank"] = (rank == min(m, n))
    report["condition_number"] = kappa
    report["singular_values"] = s
    
    energy = np.cumsum(s**2) / np.sum(s**2) if np.sum(s**2) > 0 else np.zeros_like(s)
    k_99 = int(np.searchsorted(energy, 0.99) + 1)
    report["rank_for_99pct_energy"] = min(k_99, len(s))
    
    # --- Definiteness ---
    if report["symmetric"]:
        eigvals = np.linalg.eigvalsh(A)
        report["eigenvalues"] = eigvals
        if np.all(eigvals > tol):
            report["definiteness"] = "positive definite"
        elif np.all(eigvals > -tol):
            report["definiteness"] = "positive semidefinite"
        elif np.all(eigvals < -tol):
            report["definiteness"] = "negative definite"
        elif np.all(eigvals < tol):
            report["definiteness"] = "negative semidefinite"
        else:
            report["definiteness"] = "indefinite"
    
    # --- Warnings ---
    warnings = []
    if kappa > 1e12:
        warnings.append(f"SEVERELY ILL-CONDITIONED (κ={kappa:.1e})")
    elif kappa > 1e6:
        warnings.append(f"ILL-CONDITIONED (κ={kappa:.1e})")
    elif kappa > 1e3:
        warnings.append(f"Moderately conditioned (κ={kappa:.1e})")
    
    if not report["full_rank"]:
        warnings.append(f"RANK DEFICIENT: rank={rank}, expected={min(m,n)}")
    
    if report.get("definiteness") == "indefinite":
        warnings.append("INDEFINITE: not a valid Hessian at a minimum")
    
    report["warnings"] = warnings
    
    # --- Solver Recommendation ---
    if not is_square or m != n:
        if rank < min(m, n):
            solver = "SVD / pseudoinverse (rank-deficient)"
        elif m > n:
            solver = "QR factorization (overdetermined least-squares)"
        else:
            solver = "SVD (underdetermined, minimum-norm)"
    elif report.get("definiteness") == "positive definite":
        if report["sparse"] and n > 100:
            solver = "Sparse Cholesky (CHOLMOD) with AMD ordering"
        else:
            solver = "Dense Cholesky (fastest for dense PD)"
    elif report.get("definiteness") == "positive semidefinite":
        solver = "LDLᵀ or regularized Cholesky (H + εI)"
    elif report.get("definiteness") == "indefinite":
        solver = "Modified Cholesky or LDLᵀ with pivoting"
    elif report["symmetric"]:
        solver = "LDLᵀ with pivoting"
    else:
        solver = "LU with partial pivoting"
    
    report["recommended_solver"] = solver
    
    # --- Print ---
    if verbose:
        print("=" * 60)
        print("MATRIX DOCTOR REPORT")
        print("=" * 60)
        
        props = []
        if report["symmetric"]: props.append("Symmetric")
        if report["skew_symmetric"]: props.append("Skew-symmetric")
        if report.get("orthogonal"): props.append("Orthogonal")
        if report["diagonal"]: props.append("Diagonal")
        if report["upper_triangular"]: props.append("Upper triangular")
        if report["lower_triangular"]: props.append("Lower triangular")
        if report["sparse"]: props.append(f"Sparse ({report['sparsity']*100:.0f}%)")
        
        print(f"Shape: {m}×{n}")
        print(f"Properties: {', '.join(props) if props else 'None special'}")
        if "definiteness" in report:
            print(f"Definiteness: {report['definiteness']}")
        print(f"Rank: {rank}/{min(m,n)} | κ = {kappa:.2e}")
        print(f"99% energy at rank {report['rank_for_99pct_energy']}")
        
        if warnings:
            print(f"\n⚠ WARNINGS:")
            for w in warnings:
                print(f"  • {w}")
        else:
            print("\n✓ No issues detected")
        
        print(f"\n→ Recommended solver: {solver}")
        print("=" * 60)
    
    return report

# Demo
A1 = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]])
matrix_doctor(A1)

print()

# Ill-conditioned Jacobian
J = np.random.randn(10, 5) * np.array([100, 10, 1, 0.01, 0.0001])
matrix_doctor(J)
```

</details>

---

## Reference Answers — Quick Check Table

| Exercise | Key Result |
|----------|-----------|
| A1 | $A$: symmetric, $B$: skew-symmetric, $C$: orthogonal |
| A2 | PD when $\sigma_{\max}(J) < \sqrt{10}$ |
| A3 | $\kappa = 8000$, QR preferred over normal equations |
| B1 | $P$ is PD, $Q$ is indefinite |
| B3 | $\hat{\Sigma}$ not PD, fix via eigenvalue clipping |
| B4 | Schur complement is $6c \times 6c$, vastly smaller than full system |
| C1 | $\sigma_1 = 3$, $\sigma_2 = 2$, sign absorbed into $U$ |
| C2 | Rank 1, null space = $\text{span}\{(1,-1)^T\}$ |
| C3 | $\kappa = 2500$, rank 3 for 99% energy |
| D1 | $\|A\|_1 = 5$, $\|A\|_\infty = 6$, $\|A\|_F \approx 5.48$, $\|A\|_2 \approx 4.52$ |
| E1 | $\kappa(J) = 10^6$, numerical rank = 4, null space = gauge freedom |
| E2 | True $x = (1,1)$, $\kappa \approx 4/\epsilon$, fails for $\epsilon = 10^{-12}$ |
| F1 | All rotation properties verified |
| F2 | Eigenvalues $\{0, 2, 2, 4\}$, zero = gauge freedom |

---

*Next: Implement these exercises in Python → [code/matrix_mastery/](../code/matrix_mastery/)*
