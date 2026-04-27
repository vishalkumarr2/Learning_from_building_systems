# Matrix Essentials for Optimization

> **A comprehensive reference — matrix types, criteria, decompositions, and their role in optimization.**
>
> This guide consolidates and extends the matrix material from Weeks 1–2 into a single,
> deep reference. Use it as a companion to the daily lessons or as a standalone study guide.

## Prerequisites
- [Day 1: Vectors, Matrices, Transformations](weeks/week-01/day-01-vectors-matrices.md)
- [Day 2: Eigenvalues, PD, SVD](weeks/week-01/day-02-eigenvalues-svd.md)
- [Day 8: Matrix Decompositions](weeks/week-02/day-08-matrix-decompositions.md)
- [Day 12: Conditioning](weeks/week-02/day-12-conditioning.md)

---

## 1. Matrix Taxonomy

Not all matrices are created equal. Recognizing the *type* of matrix you're working with
immediately tells you which algorithms, decompositions, and guarantees apply.

### 1.1 Structural Classification

| Type | Definition | Key Property | Example in Robotics |
|------|-----------|--------------|---------------------|
| **Symmetric** | $A = A^T$ | Real eigenvalues, orthogonal eigenvectors | Hessian $\nabla^2 f$, covariance $\Sigma$ |
| **Skew-symmetric** | $A = -A^T$ | Eigenvalues are purely imaginary | Cross-product matrix $[\omega]_\times$ |
| **Orthogonal** | $A^T A = I$ | Preserves norms and angles, $\det = \pm 1$ | Rotation matrices $R \in SO(3)$ |
| **Diagonal** | $a_{ij} = 0$ for $i \neq j$ | Eigenvalues = diagonal entries | Scaling, singular value matrix $\Sigma$ |
| **Triangular** | $a_{ij} = 0$ above (or below) diagonal | $O(n^2)$ solve via substitution | Cholesky factor $L$ |
| **Band** | $a_{ij} = 0$ for $|i-j| > k$ | $O(nk^2)$ factorization | Tridiagonal systems in 1D optimization |
| **Sparse** | Most entries are zero | Specialized storage and algorithms | SLAM Jacobians, graph Laplacians |
| **Block** | Partitioned into sub-matrices | Exploit structure for efficiency | Bundle adjustment Hessian |

### 1.2 Spectral Classification

| Type | Eigenvalue Condition | Geometric Shape | Optimization Meaning |
|------|---------------------|-----------------|---------------------|
| **Positive definite (PD)** | All $\lambda_i > 0$ | Ellipsoidal bowl | Unique minimum exists |
| **Positive semidefinite (PSD)** | All $\lambda_i \geq 0$ | Bowl with flat valley | Minimum exists but may not be unique |
| **Indefinite** | Mixed $\lambda_i$ signs | Saddle surface | Saddle point — not a minimum |
| **Negative definite (ND)** | All $\lambda_i < 0$ | Inverted bowl | Maximum (minimize $-f$) |
| **Negative semidefinite (NSD)** | All $\lambda_i \leq 0$ | Inverted bowl + flat | Maximum with degeneracy |

### 1.3 The Definiteness Decision Tree

```
Is A symmetric?
├── No → Not applicable (definiteness is defined for symmetric matrices)
└── Yes → Compute eigenvalues
    ├── All λ > 0 → Positive Definite
    ├── All λ ≥ 0, some = 0 → Positive Semidefinite
    ├── All λ < 0 → Negative Definite
    ├── All λ ≤ 0, some = 0 → Negative Semidefinite
    └── Mixed signs → Indefinite
```

---

## 2. Positive Definiteness — The Central Criterion

In optimization, PD matrices appear everywhere: the Hessian at a minimum is PD,
information matrices in SLAM are PD, and the convergence rate of most solvers depends
on the eigenvalue spectrum of a PD matrix.

### 2.1 Six Tests for Positive Definiteness

Given a symmetric matrix $A \in \mathbb{R}^{n \times n}$:

| # | Test | Computational Cost | Notes |
|---|------|-------------------|-------|
| 1 | **Eigenvalue test:** All $\lambda_i > 0$ | $O(n^3)$ | Definitive but expensive |
| 2 | **Cholesky test:** $A = LL^T$ succeeds | $O(n^3/3)$ | Fastest in practice, also gives solver |
| 3 | **Sylvester's criterion:** All leading principal minors $> 0$ | $O(n^4)$ naive | Good for small matrices and proofs |
| 4 | **Quadratic form:** $x^TAx > 0$ for all $x \neq 0$ | Not computational | The *definition*; used in proofs |
| 5 | **Pivot test:** All pivots in Gaussian elimination are positive | $O(n^3/3)$ | Equivalent to Cholesky |
| 6 | **Diagonal dominance** (sufficient, not necessary): $a_{ii} > \sum_{j \neq i} |a_{ij}|$ | $O(n^2)$ | Quick sufficient check |

**Practical recommendation:** Try Cholesky (#2). If it succeeds, the matrix is PD. If it fails
(negative value under square root), the matrix is not PD. This is also the fastest method.

### 2.2 Sylvester's Criterion — Worked Example

A matrix is PD if and only if all **leading principal minors** are positive.

For $A = \begin{pmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{pmatrix}$:

- $\Delta_1 = a_{11} > 0$
- $\Delta_2 = a_{11}a_{22} - a_{12}a_{21} > 0$
- $\Delta_3 = \det(A) > 0$

**Example:** $A = \begin{pmatrix} 4 & 2 & 1 \\ 2 & 5 & 3 \\ 1 & 3 & 6 \end{pmatrix}$

- $\Delta_1 = 4 > 0$ ✓
- $\Delta_2 = 4 \cdot 5 - 2 \cdot 2 = 16 > 0$ ✓
- $\Delta_3 = 4(30 - 9) - 2(12 - 3) + 1(6 - 5) = 84 - 18 + 1 = 67 > 0$ ✓

Therefore $A$ is positive definite.

### 2.3 The PSD Cone and Matrix Inequalities

The set of all $n \times n$ PSD matrices forms a **convex cone** $\mathcal{S}_+^n$:

- If $A, B \succeq 0$ then $\alpha A + \beta B \succeq 0$ for $\alpha, \beta \geq 0$ (closure under conic combinations)
- $0 \in \mathcal{S}_+^n$ (contains the origin)

**Löwner (matrix) ordering:** $A \succeq B$ means $A - B$ is PSD. This defines a partial order on symmetric matrices.

Key inequalities:
- $A \succeq 0 \iff x^TAx \geq 0$ for all $x$
- $A \succeq B$ and $B \succeq C \implies A \succeq C$ (transitivity)
- $A \succeq 0 \implies \text{tr}(A) \geq 0$ and $\det(A) \geq 0$
- If $A \succ 0$ then $A^{-1} \succ 0$

**Optimization relevance:** Semidefinite programming (SDP) optimizes over the PSD cone.
The constraint "$H \succeq 0$" is convex, enabling efficient algorithms.

### 2.4 Schur Complement Criterion

For a block matrix $M = \begin{pmatrix} A & B \\ B^T & C \end{pmatrix}$:

$$M \succ 0 \iff A \succ 0 \text{ and } C - B^T A^{-1} B \succ 0$$

The matrix $S = C - B^T A^{-1} B$ is the **Schur complement** of $A$ in $M$.

**Why this matters:** In bundle adjustment, the Hessian has block structure
$\begin{pmatrix} H_{cc} & H_{cl} \\ H_{lc} & H_{ll} \end{pmatrix}$.
The Schur complement eliminates landmarks, leaving a smaller camera-only system.
Positive definiteness of $S$ guarantees the reduced system is well-posed.

### 2.5 Nearest Positive Definite Matrix

Given a symmetric matrix $A$ that is *not* PD (e.g., a noisy Hessian estimate),
find the closest PD matrix.

**Algorithm (Higham, 1988):**

1. Compute eigendecomposition: $A = Q \Lambda Q^T$
2. Replace negative eigenvalues: $\tilde{\Lambda} = \text{diag}(\max(\lambda_i, \epsilon))$
3. Reconstruct: $\tilde{A} = Q \tilde{\Lambda} Q^T$

This gives the nearest PSD matrix in the Frobenius norm. Adding a small $\epsilon > 0$
to the zero eigenvalues makes it PD.

**Optimization use:** When Newton's method produces an indefinite Hessian, this (or
adding $\lambda I$, or modified Cholesky) makes the search direction a descent direction.

---

## 3. SVD Masterclass

The Singular Value Decomposition is the most important matrix factorization.
It applies to *any* matrix (no restrictions on shape or symmetry) and reveals
rank, range, null space, condition number, and best low-rank approximation.

### 3.1 Full Derivation

For any $A \in \mathbb{R}^{m \times n}$ with rank $r$:

$$A = U \Sigma V^T$$

**Construction:**
1. Form $A^TA$ (symmetric PSD, $n \times n$). Its eigenvalues are $\sigma_1^2 \geq \cdots \geq \sigma_r^2 > 0 = \cdots = 0$.
2. Eigenvectors of $A^TA$ give $V = [v_1 | \cdots | v_n]$ (right singular vectors).
3. $\sigma_i = \sqrt{\lambda_i(A^TA)}$ are the **singular values**.
4. $u_i = Av_i / \sigma_i$ for $i = 1, \ldots, r$ give the left singular vectors.
5. Complete $U$ with an orthonormal basis for $N(A^T)$.

### 3.2 Geometric Interpretation

SVD decomposes any linear transformation into three steps:

$$A = \underbrace{U}_{\text{rotate output}} \cdot \underbrace{\Sigma}_{\text{stretch}} \cdot \underbrace{V^T}_{\text{rotate input}}$$

- $V^T$ rotates the input to align with the principal axes
- $\Sigma$ stretches along each axis by $\sigma_i$
- $U$ rotates the result into the output space

**The unit sphere** $\|x\| = 1$ maps to an **ellipsoid** with semi-axes $\sigma_1, \ldots, \sigma_r$ oriented along $u_1, \ldots, u_r$.

### 3.3 Four Forms of SVD

| Form | $U$ | $\Sigma$ | $V$ | Use |
|------|-----|---------|-----|-----|
| **Full** | $m \times m$ | $m \times n$ | $n \times n$ | Theoretical completeness |
| **Thin/Economy** | $m \times n$ | $n \times n$ | $n \times n$ | Most computations |
| **Compact** | $m \times r$ | $r \times r$ | $n \times r$ | Rank-$r$ representation |
| **Truncated** | $m \times k$ | $k \times k$ | $n \times k$ | Best rank-$k$ approximation |

**NumPy:**
- `np.linalg.svd(A)` → full SVD
- `np.linalg.svd(A, full_matrices=False)` → thin SVD

### 3.4 The Pseudoinverse (Moore-Penrose)

For $A = U\Sigma V^T$, the pseudoinverse is:

$$A^+ = V \Sigma^+ U^T$$

where $\Sigma^+$ has entries $1/\sigma_i$ for $\sigma_i > 0$ and $0$ otherwise.

**Properties:**
- $AA^+A = A$ and $A^+AA^+ = A^+$ (generalized inverse conditions)
- $x^+ = A^+ b$ is the **minimum-norm least-squares** solution to $Ax = b$
- If $A$ is invertible: $A^+ = A^{-1}$
- If $A$ has full column rank: $A^+ = (A^TA)^{-1}A^T$
- If $A$ has full row rank: $A^+ = A^T(AA^T)^{-1}$

**NumPy:** `np.linalg.pinv(A)` computes $A^+$ via SVD.

### 3.5 Eckart-Young-Mirsky Theorem

The **best rank-$k$ approximation** to $A$ (in both Frobenius and spectral norms) is:

$$A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^T = U_k \Sigma_k V_k^T$$

**Error bounds:**
- Spectral norm: $\|A - A_k\|_2 = \sigma_{k+1}$
- Frobenius norm: $\|A - A_k\|_F = \sqrt{\sigma_{k+1}^2 + \cdots + \sigma_r^2}$

**Applications:**
- **Data compression:** Keep top-$k$ singular values, discard noise in the small ones
- **Noise filtering:** If true signal has rank $k$, truncated SVD removes noise
- **PCA:** The top-$k$ left singular vectors of centered data $X$ are the principal components
- **Latent Semantic Analysis:** Low-rank approximation of term-document matrices

### 3.6 Numerical Rank

In theory, rank = number of nonzero singular values. In practice, due to floating-point
noise, we define:

$$\text{numerical rank} = \#\{i : \sigma_i > \tau\}$$

where $\tau$ is a threshold. Common choices:
- $\tau = \epsilon_{\text{mach}} \cdot \sigma_1 \cdot \max(m, n)$ (NumPy default for `np.linalg.matrix_rank`)
- $\tau = \epsilon_{\text{mach}} \cdot \sigma_1$ (more aggressive)

**Robotics connection:** In SLAM, the numerical rank of the Jacobian reveals how many
state dimensions are actually observable given the current measurements.

---

## 4. Matrix Norms

Norms measure the "size" of a matrix. Different norms answer different questions.

### 4.1 Vector Norms (Review)

| Norm | Definition | Geometric Meaning |
|------|-----------|-------------------|
| $\|x\|_1 = \sum_i |x_i|$ | Manhattan / taxicab distance | Diamond-shaped unit ball |
| $\|x\|_2 = \sqrt{\sum_i x_i^2}$ | Euclidean distance | Circular unit ball |
| $\|x\|_\infty = \max_i |x_i|$ | Chebyshev distance | Square unit ball |
| $\|x\|_p = \left(\sum_i |x_i|^p\right)^{1/p}$ | General $p$-norm | Interpolates between above |

### 4.2 Matrix Norms

| Norm | Definition | Computation |
|------|-----------|------------|
| **Frobenius** | $\|A\|_F = \sqrt{\sum_{ij} a_{ij}^2} = \sqrt{\text{tr}(A^TA)}$ | $= \sqrt{\sum_i \sigma_i^2}$ |
| **Spectral (2-norm)** | $\|A\|_2 = \max_{\|x\|=1} \|Ax\|_2$ | $= \sigma_{\max}(A)$ |
| **Nuclear (trace)** | $\|A\|_* = \sum_i \sigma_i$ | Sum of singular values |
| **Max norm** | $\|A\|_{\max} = \max_{ij} |a_{ij}|$ | Largest absolute entry |

**Relationships:** $\|A\|_2 \leq \|A\|_F \leq \sqrt{r} \cdot \|A\|_2$ where $r = \text{rank}(A)$.

### 4.3 Induced (Operator) Norms

The matrix $p$-norm is the maximum stretching factor:

$$\|A\|_p = \max_{x \neq 0} \frac{\|Ax\|_p}{\|x\|_p}$$

| $p$ | Formula |
|-----|---------|
| $p = 1$ | $\|A\|_1 = \max_j \sum_i |a_{ij}|$ (max column sum) |
| $p = 2$ | $\|A\|_2 = \sigma_{\max}(A)$ (largest singular value) |
| $p = \infty$ | $\|A\|_\infty = \max_i \sum_j |a_{ij}|$ (max row sum) |

### 4.4 Submultiplicativity

For any consistent matrix norm: $\|AB\| \leq \|A\| \cdot \|B\|$.

**Consequence:** $\|A^k\| \leq \|A\|^k$. This bounds the error growth in iterative methods.

### 4.5 Norms in Optimization

- **Frobenius norm** → measures overall "size" of a matrix; used in regularization ($\min \|X\|_F$)
- **Spectral norm** → measures worst-case amplification; determines condition number
- **Nuclear norm** → convex relaxation of rank; used in matrix completion / low-rank recovery
- **$\ell_1$ on singular values** = nuclear norm → promotes low rank (like $\ell_1$ promotes sparsity)

---

## 5. Ill-Conditioning — Diagnosis and Remedies

### 5.1 Condition Number Revisited

$$\kappa(A) = \|A\| \cdot \|A^{-1}\| = \frac{\sigma_{\max}}{\sigma_{\min}}$$

**Rules of thumb:**

| $\kappa(A)$ | Digits Lost | Status | Action |
|-------------|-------------|--------|--------|
| $1$ | $0$ | Perfect | None needed |
| $10^3$ | $\sim 3$ | Moderate | Monitor |
| $10^6$ | $\sim 6$ | Ill-conditioned | Precondition or regularize |
| $10^{12}$ | $\sim 12$ | Severely ill-conditioned | Regularize; direct solve unreliable |
| $> 10^{16}$ | All | Effectively singular | Rank-deficient; use pseudoinverse |

**Forward error bound:** $\frac{\|\delta x\|}{\|x\|} \leq \kappa(A) \cdot \frac{\|\delta b\|}{\|b\|}$

If you store numbers with $d$ digits of precision, and $\kappa(A) \approx 10^k$, then
the solution has at most $d - k$ reliable digits.

### 5.2 Sources of Ill-Conditioning in Robotics

| Source | Why It Happens | Typical $\kappa$ |
|--------|---------------|-----------------|
| **Scale mismatch** | Translation (meters) vs rotation (radians) in same state | $10^2 - 10^4$ |
| **Near-singularity** | Near-collinear landmarks, degenerate configurations | $10^6 - 10^{12}$ |
| **Gauge freedom** | Absolute position/orientation unobservable in SLAM | $\infty$ (rank-deficient) |
| **Poor excitation** | Parameters that barely affect measurements | $10^4 - 10^8$ |
| **Large dynamic range** | Nearby + distant features in same BA problem | $10^3 - 10^6$ |
| **Normal equations** | Forming $J^TJ$ squares the condition number | $\kappa(J)^2$ |

### 5.3 Catastrophic Cancellation — A Worked Example

Consider solving:

$$\begin{pmatrix} 1 & 1 \\ 1 & 1 + 10^{-15} \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \begin{pmatrix} 2 \\ 2 + 10^{-15} \end{pmatrix}$$

True solution: $x = [1, 1]^T$. Condition number $\kappa \approx 4 \times 10^{15}$.

In double precision ($\epsilon_{\text{mach}} \approx 10^{-16}$): the solver produces wildly
wrong results because we lose *all* significant digits. Tiny changes to the RHS produce
completely different solutions.

### 5.4 Regularization Techniques

When the problem is ill-conditioned or rank-deficient, **regularization** adds
prior information to stabilize the solution.

#### Tikhonov Regularization (Ridge Regression)

$$\min_x \|Ax - b\|^2 + \lambda \|x\|^2$$

Solution: $x_\lambda = (A^TA + \lambda I)^{-1} A^T b$

**Effect:** Condition number becomes $\kappa(A^TA + \lambda I) = \frac{\sigma_1^2 + \lambda}{\sigma_n^2 + \lambda}$

For large $\lambda$: $\kappa \to 1$ (well-conditioned but biased).
For small $\lambda$: $\kappa \to \kappa(A^TA)$ (unbiased but ill-conditioned).

**Robotics analogy:** Levenberg-Marquardt adds $\lambda I$ to $J^TJ$ — this IS Tikhonov regularization of the Gauss-Newton step.

#### Truncated SVD

$$x_k = \sum_{i=1}^{k} \frac{u_i^T b}{\sigma_i} v_i$$

Keep only the $k$ largest singular values/vectors. Small $\sigma_i$ amplify noise,
so discarding them improves stability at the cost of a biased solution.

**When to use which:**
- **Tikhonov** when all parameters are meaningful but noisy → smooth damping
- **Truncated SVD** when some parameters are genuinely unobservable → hard cutoff
- **Both combined** (damped truncated SVD): discard the worst, damp the rest

#### L-Curve Method for Choosing $\lambda$

Plot $\|\text{residual}\|$ vs $\|\text{solution}\|$ as $\lambda$ varies. The optimal $\lambda$
is at the "corner" of the L-shaped curve — balancing data fit against solution regularity.

### 5.5 Preconditioning Strategies

Transform $Hx = g$ into $\tilde{H}\tilde{x} = \tilde{g}$ with $\kappa(\tilde{H}) \ll \kappa(H)$.

| Preconditioner | Construction | Cost | Best For |
|---------------|-------------|------|----------|
| **Jacobi** | $M = \text{diag}(H)$ | $O(n)$ | Scale-mismatched problems |
| **Block Jacobi** | $M = \text{blockdiag}(H)$ | $O(n \cdot b^3)$ | Block-structured Hessians |
| **SSOR** | Symmetric SOR based on $H = D + L + L^T$ | $O(\text{nnz})$ | General sparse |
| **Incomplete Cholesky** | Sparse $\tilde{L}\tilde{L}^T \approx H$ | $O(\text{nnz})$ | Sparse PD systems |
| **AMG** | Algebraic multigrid hierarchy | $O(n)$ apply | Very large sparse systems |

---

## 6. Special Matrices in Robotics & Optimization

### 6.1 Rotation Matrices — $SO(2)$ and $SO(3)$

**$SO(2)$:** 2D rotation by angle $\theta$:
$$R(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

**Properties:** $R^TR = I$, $\det(R) = +1$, eigenvalues $= e^{\pm i\theta}$, $\kappa(R) = 1$.

**$SO(3)$:** 3D rotation parameterized by axis-angle $\omega = \theta \hat{n}$:
$$R = I + \sin\theta [\hat{n}]_\times + (1 - \cos\theta)[\hat{n}]_\times^2$$
(Rodrigues' formula)

where $[\hat{n}]_\times$ is the $3 \times 3$ skew-symmetric cross-product matrix.

**Key fact:** Rotation matrices are **always perfectly conditioned** ($\kappa = 1$) and orthogonal.
They never cause numerical trouble by themselves — trouble arises when rotation parameters
are mixed with translation parameters of very different scale.

### 6.2 Covariance Matrices

$$\Sigma = \mathbb{E}[(x - \mu)(x - \mu)^T]$$

**Properties:** Always symmetric PSD. Diagonal entries = variances. Off-diagonal = covariances.

**Eigenvalue interpretation:**
- Eigenvectors = principal directions of uncertainty
- Eigenvalues = variance along each principal direction
- $\sqrt{\lambda_i}$ = standard deviation along $i$-th principal direction
- The uncertainty ellipsoid is $\{x : (x-\mu)^T \Sigma^{-1} (x-\mu) \leq c^2\}$

### 6.3 Information Matrices (Precision Matrices)

$$\Lambda = \Sigma^{-1}$$

**Properties:** PD (if $\Sigma$ is PD). In SLAM, $\Lambda$ is **sparse** because only
spatially adjacent variables share measurements, while $\Sigma$ is typically dense.

**Why SLAM uses information form:**
- Sparse $\Lambda$ → efficient storage and computation
- Adding a new measurement = **adding** to $\Lambda$ (additive update)
- Marginalizing a variable = Schur complement (fill-in, but manageable)

### 6.4 Jacobian Matrices

$$J_{ij} = \frac{\partial f_i}{\partial x_j}$$

**Properties:** Generally rectangular ($m \times n$, often $m > n$). Not necessarily symmetric.

**What the SVD of $J$ tells you:**
- **Rank** = number of independently constrained state dimensions
- **Null space** ($V$ columns for $\sigma_i = 0$) = unobservable directions
- **Condition number** $\kappa(J)$ = how sensitive the solution is to measurement noise
- **Left null space** ($U$ columns for $\sigma_i = 0$) = redundant measurements

### 6.5 Hessian Matrices

$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

**Properties:** Always symmetric (by Schwarz's theorem if $f$ is $C^2$). At a local minimum:
$H \succeq 0$ (PSD). At a strict local minimum of a strongly convex function: $H \succ 0$ (PD).

**Gauss-Newton approximation:** For least-squares $f(x) = \frac{1}{2}\|r(x)\|^2$:
$$H \approx J^T J$$
(drops second-order residual terms). This approximation is PSD by construction, even if the true Hessian is indefinite.

### 6.6 Graph Laplacian

For a graph with $n$ nodes, weighted adjacency $W$, and degree matrix $D = \text{diag}(\sum_j w_{ij})$:

$$L = D - W$$

**Properties:**
- Symmetric, PSD
- Smallest eigenvalue is always $0$ (eigenvector = $\mathbf{1}$, the constant vector)
- Number of zero eigenvalues = number of connected components
- Second-smallest eigenvalue (Fiedler value) = algebraic connectivity

**Optimization use:** Pose-graph SLAM can be viewed as minimizing a quadratic form on a graph Laplacian. Spectral methods use the Laplacian eigenvectors for initialization.

---

## 7. Matrix Calculus Quick Reference

### 7.1 Common Derivatives

| Expression | Derivative w.r.t. $x$ |
|-----------|----------------------|
| $a^Tx$ | $a$ |
| $x^TAx$ | $(A + A^T)x$ (or $2Ax$ if $A$ symmetric) |
| $\|Ax - b\|^2$ | $2A^T(Ax - b)$ |
| $\text{tr}(AX)$ | $A^T$ (w.r.t. $X$) |
| $\log\det(X)$ | $X^{-T}$ (w.r.t. $X$, for PD $X$) |
| $x^TA^{-1}x$ | $-A^{-T}xx^TA^{-T}$ (w.r.t. $A$) |

### 7.2 The Chain Rule for Matrix-Valued Functions

If $f : \mathbb{R}^n \to \mathbb{R}$ is composed as $f = g \circ h$ where $h : \mathbb{R}^n \to \mathbb{R}^m$:

$$\nabla_x f = J_h^T \nabla_y g$$

This is the foundation of backpropagation and the adjoint method in automatic differentiation.

---

## 8. Summary — The Matrix Criteria Cheat Sheet

### When You Encounter a Matrix, Ask:

1. **Is it symmetric?** → Eigendecomposition exists; definiteness is meaningful
2. **Is it PD?** → Try Cholesky. If yes: unique minimum, use Cholesky solver
3. **What's the condition number?** → Predicts convergence speed and accuracy
4. **What's the rank?** → SVD reveals observable vs unobservable directions
5. **Is it sparse?** → Use sparse decompositions, iterative methods
6. **Does it have block structure?** → Exploit with Schur complement

### Algorithm Selection Flowchart

```
Problem: Solve Hx = g (where H is your Hessian)

Is H symmetric?
├── No → Use LU decomposition (O(2n³/3))
└── Yes
    ├── Is H positive definite?
    │   ├── Yes → Use Cholesky (O(n³/3)) ← FASTEST
    │   └── No / Unknown
    │       ├── Is H PSD? → Use LDLᵀ (handles zeros)
    │       └── Is H indefinite? → Modified Cholesky or add λI
    └── Is H sparse?
        ├── Yes
        │   ├── PD? → Sparse Cholesky with ordering (CHOLMOD)
        │   └── General? → Sparse LU (UMFPACK) or CG with preconditioner
        └── No → Dense decomposition as above
```

---

## Further Reading

- **Gilbert Strang** — *Linear Algebra and Its Applications*, Ch 6 (Positive Definite Matrices)
- **Golub & Van Loan** — *Matrix Computations*, Ch 2 (Norms), Ch 5 (SVD), Ch 4 (Cholesky)
- **Trefethen & Bau** — *Numerical Linear Algebra*, Ch 3–5 (Norms, SVD, Conditioning)
- **Boyd & Vandenberghe** — *Convex Optimization*, §A.5 (Matrix inequalities), Ch 4.6 (SDP)
- **Nocedal & Wright** — *Numerical Optimization*, Ch 3.4 (Modified Newton methods)

---

## Exercises

See [Exercise Set 7: Matrix Mastery](exercises/07-matrix-mastery.md) for 26 graded problems
covering all topics in this guide — from hand computation through Python implementation
to real-world robotics scenarios.
