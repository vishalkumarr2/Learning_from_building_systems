# Day 13: Iterative Methods — Conjugate Gradient

> **Phase I — Mathematical Foundations | Week 2 | 2.5 hours**
> *CG solves PD systems in n steps — but with preconditioning, it often needs far fewer.*

## Navigation
- **Previous:** [Day 12: Conditioning & Function Landscapes](day-12-conditioning.md)
- **Next:** [Day 14: Week 2 Review](day-14-week-review.md)
- **Week:** [Week 2 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
When solving the Schur complement system $Sx_c = g_c$ in BA, $S$ is dense and moderately sized (hundreds to thousands). Ceres uses CG with block-Jacobi preconditioning on $S$ (the `ITERATIVE_SCHUR` strategy). For large-scale SLAM, preconditioned CG on the full sparse system is sometimes preferred over direct factorization. CG's convergence rate depends directly on $\kappa(S)$ — everything from Day 12 feeds into this.

---

## Theory (45 min)

### 13.1 Why Iterative?

| Method | Cost per step | Total cost ($n \times n$ system) | Memory |
|--------|-------------|-------|--------|
| Dense Cholesky | — | $O(n^3)$ | $O(n^2)$ |
| Sparse Cholesky | — | $O(n \cdot \text{nnz}^2)$ | Fill-in dependent |
| CG | $O(\text{nnz})$ | $O(k \cdot \text{nnz})$, $k \ll n$ | $O(n)$ |

CG wins when: (1) $n$ is huge, (2) the matrix is sparse, (3) you only need an approximate solution, or (4) you can't afford the memory for factorization.

### 13.2 CG for Positive Definite Systems

**Goal:** Solve $Ax = b$ where $A$ is SPD ($A = A^T$, $x^TAx > 0$ for $x \neq 0$).

**Key insight:** Solving $Ax = b$ is equivalent to minimizing:

$$f(x) = \frac{1}{2}x^TAx - b^Tx$$

CG minimizes $f$ over successive **Krylov subspaces** $\mathcal{K}_k(A, r_0) = \text{span}\{r_0, Ar_0, A^2r_0, \ldots, A^{k-1}r_0\}$.

### 13.3 The CG Algorithm

```
r₀ = b - Ax₀
p₀ = r₀

for k = 0, 1, 2, ...
    αₖ = (rₖᵀrₖ) / (pₖᵀApₖ)         # step size
    xₖ₊₁ = xₖ + αₖpₖ                  # update solution
    rₖ₊₁ = rₖ - αₖApₖ                 # update residual
    βₖ = (rₖ₊₁ᵀrₖ₊₁) / (rₖᵀrₖ)       # CG coefficient
    pₖ₊₁ = rₖ₊₁ + βₖpₖ                # update direction
```

**Properties:**
- Exact solution in at most $n$ steps (in exact arithmetic)
- Search directions $p_k$ are **A-conjugate**: $p_i^T A p_j = 0$ for $i \neq j$
- Residuals are orthogonal: $r_i^T r_j = 0$ for $i \neq j$
- Each step only needs one matrix-vector product $Ap_k$

### 13.4 Convergence Rate

After $k$ iterations:

$$\frac{\|x_k - x^*\|_A}{\|x_0 - x^*\|_A} \leq 2 \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^k$$

where $\kappa = \kappa(A) = \lambda_{\max}/\lambda_{\min}$.

**Rule of thumb:** To reduce error by factor $10^{-p}$, need $k \approx \frac{p}{2}\sqrt{\kappa}$ iterations.

| $\kappa$ | Iterations for $10^{-6}$ accuracy |
|----------|----------------------------------|
| $10$ | ~10 |
| $10^2$ | ~30 |
| $10^4$ | ~300 |
| $10^6$ | ~3000 |

### 13.5 Preconditioned CG (PCG)

Replace $Ax = b$ with $M^{-1}Ax = M^{-1}b$ where $M \approx A$:

```
r₀ = b - Ax₀
z₀ = M⁻¹r₀
p₀ = z₀

for k = 0, 1, 2, ...
    αₖ = (rₖᵀzₖ) / (pₖᵀApₖ)
    xₖ₊₁ = xₖ + αₖpₖ
    rₖ₊₁ = rₖ - αₖApₖ
    zₖ₊₁ = M⁻¹rₖ₊₁                    # preconditioner solve
    βₖ = (rₖ₊₁ᵀzₖ₊₁) / (rₖᵀzₖ)
    pₖ₊₁ = zₖ₊₁ + βₖpₖ
```

Now convergence depends on $\kappa(M^{-1}A)$ instead of $\kappa(A)$.

### 13.6 CG as an Optimizer

CG is also used for unconstrained optimization (nonlinear CG):
- **Fletcher-Reeves:** $\beta_k = \frac{\|g_{k+1}\|^2}{\|g_k\|^2}$
- **Polak-Ribière:** $\beta_k = \frac{g_{k+1}^T(g_{k+1} - g_k)}{\|g_k\|^2}$

These will appear in Week 3 (Day 20). For now, focus on CG for **linear systems**.

---

## Implementation (60 min)

### Exercise 1: CG from Scratch
```python
# See: code/week02/iterative_methods.py
```
Implement the CG algorithm. Verify it converges in exactly $n$ steps for an $n \times n$ system.

### Exercise 2: Convergence Monitoring
Track $\|r_k\|$ vs iteration. Show the dependence on $\kappa$.

### Exercise 3: Preconditioned CG
Implement PCG with Jacobi preconditioning. Compare iteration counts.

### Exercise 4: Sparse PD Benchmark
Create a large sparse PD system (1D Laplacian, $n = 10000$). Compare: (a) dense solve, (b) sparse direct solve, (c) CG, (d) PCG.

---

## Practice (45 min)

1. Prove that the CG search directions are A-conjugate (i.e., $p_i^T A p_j = 0$).
2. Why does CG require $A$ to be SPD? What goes wrong if $A$ has a negative eigenvalue?
3. CG needs one matrix-vector product per iteration. For a sparse matrix with nnz nonzeros, what's the cost per iteration?
4. You're solving a $10^6 \times 10^6$ sparse system with $\kappa = 10^4$. Estimate the CG iteration count.
5. The Jacobi preconditioner is $M = \text{diag}(A)$. When is it effective? When does it fail?
6. Why is CG memory-efficient compared to GMRES for symmetric systems?
7. In Ceres, `ITERATIVE_SCHUR` uses CG on the Schur complement. Why not use CG on the full system?
8. If CG converges in 50 iterations for a $10000 \times 10000$ system, what does this tell you about the eigenvalue distribution?

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: CG as polynomial optimization</summary>

**Problem:** Show that CG finds the best polynomial approximation $p_k(\lambda)$ of degree $k$ such that $\|x_k - x^*\|_A = \min_{p_k: p_k(0)=1} \max_{\lambda \in [\lambda_{\min}, \lambda_{\max}]} |p_k(\lambda)| \cdot \|x_0 - x^*\|_A$. Why does this explain superlinear convergence when eigenvalues cluster?

**Solution:** CG's error at step $k$ satisfies $\|e_k\|_A \leq \min_{p_k(0)=1} \max_i |p_k(\lambda_i)| \cdot \|e_0\|_A$ where the min is over degree-$k$ polynomials with $p(0) = 1$. If eigenvalues cluster into $m$ groups, a degree-$m$ polynomial can nearly vanish at all eigenvalues, giving fast convergence in $\sim m \ll n$ steps. This is why preconditioning that clusters eigenvalues (even without reducing $\kappa$ much) can dramatically accelerate CG.
</details>

<details>
<summary>🎯 Challenge 2: Truncated CG for trust regions</summary>

**Problem:** In trust-region methods (Week 3), we need to solve $H \Delta x = -g$ subject to $\|\Delta x\| \leq \Delta$. Implement Steihaug-Toint truncated CG: run CG but stop early if (a) the iterate reaches the trust region boundary, or (b) negative curvature is detected ($p_k^T H p_k \leq 0$).

**Solution:** At each CG step, check: (1) if $p_k^T H p_k \leq 0$, go to the trust region boundary along $p_k$ (negative curvature = not PD in this direction). (2) if $\|x_{k+1}\| > \Delta$, find $\tau$ such that $\|x_k + \tau p_k\| = \Delta$ (quadratic in $\tau$, take positive root). The truncated CG-Steihaug method provides an approximate solution to the trust-region subproblem in at most $n$ CG steps, naturally handling indefinite Hessians.
</details>

<details>
<summary>🎯 Challenge 3: Block CG for multiple right-hand sides</summary>

**Problem:** In SLAM, you sometimes need to solve $AX = B$ where $B \in \mathbb{R}^{n \times s}$ ($s$ RHS vectors). Standard CG solves each column independently. Design Block CG that shares Krylov subspace information across columns.

**Solution:** Block CG maintains $s$ search directions simultaneously and orthogonalizes across all of them. The key modification: $P_k \in \mathbb{R}^{n \times s}$, $R_k \in \mathbb{R}^{n \times s}$, and the step size becomes a matrix $\alpha_k = (P_k^TAP_k)^{-1}P_k^TR_k$. When the RHS vectors share structure (common in SLAM covariance recovery), Block CG converges in fewer matrix-vector products total than $s$ independent CG runs. Deflation of converged columns is important for efficiency.
</details>

---

## Self-Assessment Checklist

- [ ] I can implement CG from scratch and verify $n$-step convergence
- [ ] I understand the convergence bound and its dependence on $\kappa$
- [ ] I can implement and apply Jacobi preconditioning
- [ ] I know when to choose CG over direct solvers

---

## Key Takeaways

1. **CG** solves SPD systems with $O(\text{nnz})$ per iteration — unbeatable for large sparse systems.
2. **Convergence** depends on $\sqrt{\kappa}$ — preconditioning is essential for ill-conditioned problems.
3. **Preconditioned CG** replaces $\kappa(A)$ with $\kappa(M^{-1}A)$ — Jacobi is the simplest useful preconditioner.
4. **Ceres uses PCG** for the Schur complement in BA (`ITERATIVE_SCHUR` + `CLUSTER_JACOBI`).
