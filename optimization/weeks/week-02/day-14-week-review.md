# Day 14: Week 2 Review — Computational Foundations

> **Phase I — Mathematical Foundations | Week 2 | 2.5 hours**
> *Integrate everything: decompositions, sparsity, differentiation, conditioning, and iterative solvers.*

## Navigation
- **Previous:** [Day 13: Iterative Methods (CG)](day-13-iterative-methods.md)
- **Next:** [Day 15: Gradient Descent](../week-03/day-15-gradient-descent.md)
- **Week:** [Week 2 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
This review pulls together the computational pipeline that Ceres and GTSAM use internally. When you call `ceres::Solve()`, this is what happens: the Jacobian $J$ is assembled (sparse), the normal equations $J^TJ \Delta x = -J^Tg$ are formed (exploiting sparsity via Schur complement), the system is solved (sparse Cholesky or preconditioned CG), and variable scaling ensures good conditioning. Every topic from this week is part of that pipeline.

---

## Week 2 Concept Map

```
Matrix Decompositions (Day 8)
   ├── LU, Cholesky, QR
   ├── Pivoting for stability
   └── Cholesky ← used by Ceres for normal equations
           │
Sparse Matrices & Schur (Day 9)
   ├── CSR/CSC storage
   ├── Fill-reducing ordering (AMD, METIS)
   └── Schur complement for BA
           │
Numerical Differentiation (Day 10)
   ├── Forward, central, complex-step
   ├── Truncation vs round-off trade-off
   └── Gradient/Jacobian/Hessian checking
           │
Automatic Differentiation (Day 11)
   ├── Forward mode (dual numbers)
   ├── Reverse mode (backpropagation)
   └── Ceres Jet type, JAX
           │
Conditioning (Day 12)
   ├── κ(A) = λ_max/λ_min
   ├── Sources: scale mismatch, gauge freedom
   └── Preconditioning, variable scaling
           │
Iterative Methods — CG (Day 13)
   ├── Krylov subspace, A-conjugacy
   ├── O(√κ) convergence
   └── Preconditioned CG → Ceres ITERATIVE_SCHUR
```

---

## Comprehensive Problems (120 min)

### Problem 1: Gradient Checker (20 min)
Build a function `check_gradient(f, grad_f, x)` that:
1. Computes the analytic gradient `grad_f(x)`
2. Computes the numerical gradient using central differences
3. Reports the relative error for each component
4. Returns PASS if all relative errors < $10^{-5}$

Test it on $f(x) = \|Ax - b\|^2$ and intentionally introduce a sign error to verify it catches bugs.

### Problem 2: Sparse Tridiagonal System (15 min)
Create a tridiagonal system $Ax = b$ with $n = 50000$:
$$A_{ij} = \begin{cases} 4 & i = j \\ -1 & |i-j| = 1 \\ 0 & \text{otherwise} \end{cases}$$

Solve using: (a) `scipy.sparse.linalg.spsolve`, (b) your CG implementation, (c) PCG with Jacobi. Report times and verify solutions match.

### Problem 3: Forward-Mode AD for Jacobians (15 min)
Using your dual number implementation from Day 11, compute the full $m \times n$ Jacobian of:
$$f(x_1, x_2, x_3) = \begin{pmatrix} x_1^2 + \sin(x_2 x_3) \\ e^{x_1} \cos(x_2) \\ x_3 / (1 + x_1^2) \end{pmatrix}$$

Verify against both symbolic and numerical Jacobians.

### Problem 4: Condition Number Study (20 min)
For $n = 100$, create SPD matrices with $\kappa \in \{10, 100, 10^3, 10^4, 10^5, 10^6\}$:
1. Solve $Ax = b$ and measure digits of precision
2. Run gradient descent and count iterations to $10^{-6}$ relative error
3. Run CG and count iterations
4. Plot all three metrics vs $\log_{10}(\kappa)$

Verify: GD iterations ∝ $\kappa$, CG iterations ∝ $\sqrt{\kappa}$.

### Problem 5: Preconditioned CG Deep Dive (20 min)
For the matrix from Problem 4 with $\kappa = 10^4$:
1. CG without preconditioning: record iteration count
2. Jacobi preconditioning: record iteration count
3. "Perfect" preconditioning ($M = A$): verify 1 iteration
4. Compute $\kappa(M^{-1}A)$ for each case

### Problem 6: Schur Complement Solver (15 min)
Create a block system:
$$\begin{pmatrix} B & E \\ E^T & C \end{pmatrix} \begin{pmatrix} x_c \\ x_l \end{pmatrix} = \begin{pmatrix} g_c \\ g_l \end{pmatrix}$$

Where $B$ is $10 \times 10$ (cameras), $C$ is $90 \times 90$ block-diagonal (landmarks), $E$ is $10 \times 90$.

Solve using: (a) full dense solve, (b) Schur complement eliminate landmarks first, then back-substitute. Verify answers match.

### Problem 7: AD vs Numerical Derivatives (15 min)
For $f(x) = \text{softmax}(x)^T \text{softmax}(x)$ at $x = [1, 2, 3, 4, 5]$:
1. Forward-mode AD gradient (JAX or your dual numbers)
2. Central difference gradient with $h = 10^{-7}$
3. Complex-step gradient with $h = 10^{-30}$
4. Compare accuracy and time

### Problem 8: Full Pipeline Integration (20 min)
Simulate a mini-SLAM problem:
- 5 robot poses (each $[x, y, \theta]$), 20 landmarks (each $[l_x, l_y]$)
- Random odometry and observation measurements with noise
- Assemble the sparse information matrix $H = J^T \Sigma^{-1} J$
- Compute $\kappa(H)$ (should be large due to gauge freedom)
- Add a prior on pose 0 to fix the gauge: $H \leftarrow H + \text{diag}(10, 10, 10, 0, \ldots, 0)$
- Solve with CG and report the condition number improvement

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Sparse Cholesky with fill-reducing ordering</summary>

**Problem:** Create a 2D grid Laplacian ($n \times n$ grid, 5-point stencil) for $n=20$. Compute Cholesky with natural ordering and with AMD (approximate minimum degree) ordering. Compare: (a) number of nonzeros in $L$, (b) factorization time, (c) why the difference grows with $n$.

**Solution:** Natural ordering of a 2D grid produces bandwidth $n$, so $L$ has $O(n^3)$ nonzeros. AMD reorders to reduce fill: for a 2D grid it achieves $O(n^2 \log n)$ fill. Use `scipy.sparse.linalg.splu` with `permc_spec='MMD_AT_PLUS_A'` vs `'NATURAL'`. For $n=20$: natural fill $\approx 7600$ nonzeros, AMD fill $\approx 3200$. The ratio grows as $O(n/\log n)$.
</details>

<details>
<summary>🎯 Challenge 2: Forward vs reverse AD efficiency crossover</summary>

**Problem:** Implement both forward-mode (dual numbers) and reverse-mode AD for $f: \mathbb{R}^n \to \mathbb{R}^m$. Empirically find the crossover point where reverse mode becomes faster than forward mode as a function of $n$ and $m$. Test on $f(x) = Ax$ (linear) and $f(x) = \|Ax - b\|^2$ (scalar-valued).

**Solution:** Forward mode computes one directional derivative per pass (cost $O(m \cdot \text{eval})$ per input), total for full Jacobian: $O(n \cdot m \cdot \text{eval})$. Reverse mode computes one row of the Jacobian per pass, total: $O(m \cdot n \cdot \text{eval})$ but with $O(n \cdot m)$ memory. For scalar-valued functions ($m=1$): reverse mode always wins — one backward pass gives the entire gradient. For $m = n$: roughly equal. For $m \gg n$: forward mode wins. Crossover at $m \approx n$.
</details>

<details>
<summary>🎯 Challenge 3: PCG for a SLAM information matrix</summary>

**Problem:** Build a synthetic 1D SLAM information matrix (50 poses, 100 loop closures). Implement PCG with three preconditioners: (a) Jacobi (diagonal), (b) block-diagonal (3×3 blocks per pose), (c) incomplete Cholesky. Compare iteration counts and explain why block-diagonal dramatically outperforms Jacobi for SLAM.

**Solution:** SLAM information matrices have strong block structure — each pose contributes a 3×3 block on the diagonal. Jacobi ignores intra-pose correlations, so it reduces $\kappa$ modestly (perhaps $10\times$). Block-diagonal captures the local structure, reducing effective $\kappa$ by $100\times$ or more. Incomplete Cholesky is best but most expensive per iteration. Iteration counts: Jacobi ~500, block-diagonal ~50, IC ~15 for a typical 50-pose problem.
</details>

---

## Self-Assessment Checklist

- [ ] I can choose the right decomposition (LU vs Cholesky vs QR) for a given problem
- [ ] I understand sparse matrix storage and can use Schur complement for BA
- [ ] I can compute derivatives numerically, verify them, and use AD for exact derivatives
- [ ] I know what condition number means and how preconditioning helps
- [ ] I can implement CG and preconditioned CG for large sparse systems
- [ ] I'm ready for Phase II: unconstrained optimization methods

---

## Key Takeaways from Week 2

1. **Decompositions** (Cholesky, QR) are the backbone of direct solvers — know when each applies.
2. **Sparsity** makes large problems tractable — CSC storage + fill-reducing ordering + Schur complement.
3. **AD** gives exact derivatives at machine precision — forward mode via dual numbers, reverse mode for gradients of scalar functions with many inputs.
4. **Conditioning** determines how hard a problem is — $\kappa$ controls both accuracy and convergence speed.
5. **CG** is the iterative solver of choice for SPD systems — $O(\sqrt{\kappa})$ convergence, improved with preconditioning.

---

## Phase I Completion

With Week 2 complete, you have the mathematical toolkit for optimization:
- **Linear algebra:** vectors, matrices, eigenvalues, SVD, decompositions
- **Calculus:** gradients, Jacobians, Hessians, Taylor expansions
- **Differentiation:** numerical, automatic (forward + reverse)
- **Computation:** sparse solvers, conditioning, CG/PCG

**Phase II** (Weeks 3-4) builds on all of this: gradient descent uses gradients and step-size selection; Newton's method uses Hessians and Cholesky; trust region methods use CG (Steihaug); BFGS approximates the Hessian without forming it.
