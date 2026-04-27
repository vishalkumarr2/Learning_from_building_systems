# Day 9: Sparse Matrices & Schur Complement

> **Phase I — Mathematical Foundations | Week 2 | 2.5 hours**
> *In SLAM, 99.9% of the Hessian is zero. Exploit it or drown.*

## Navigation
- **Previous:** [Day 8: Matrix Decompositions](day-08-matrix-decompositions.md)
- **Next:** [Day 10: Numerical Differentiation](day-10-numerical-differentiation.md)
- **Week:** [Week 2 Overview](../README.md)
- **Curriculum:** [CURRICULUM.md](../../CURRICULUM.md)

## OKS Relevance
OKS SLAM generates enormous but sparse systems: each pose only connects to its neighbors and observed landmarks. The Schur complement trick eliminates landmarks cheaply, leaving a camera-only system. This is exactly what Ceres `SPARSE_SCHUR` does. Understanding sparsity patterns lets you choose the right solver and variable ordering.

---

## Theory (45 min)

### 9.1 Sparse vs Dense

A $10000 \times 10000$ matrix:
- **Dense:** 800 MB, solving costs ~$10^{12}$ flops
- **Sparse** (0.1% nonzero): 0.8 MB, solving costs ~$10^6$ flops (banded)

### 9.2 Storage Formats

| Format | Best for | Memory |
|--------|----------|--------|
| COO (Coordinate) | Construction, conversion | 3 × nnz |
| CSR (Compressed Sparse Row) | Row slicing, matrix-vector products | 2 × nnz + n |
| CSC (Compressed Sparse Column) | Column slicing, direct solvers | 2 × nnz + n |

### 9.3 Sparse Cholesky and Fill-in

When factoring $A = LL^T$, entries that are zero in $A$ may become nonzero in $L$ — **fill-in**. Bad variable ordering → dense $L$. Good ordering → minimal fill-in.

**Ordering strategies:**
- **AMD** (Approximate Minimum Degree): fast, good general-purpose
- **METIS** (graph partitioning): better for large graphs, higher setup cost
- **COLAMD**: for rectangular matrices (QR)

### 9.4 Why SLAM is Sparse

In a pose graph, pose $i$ only constrains pose $j$ if there's a direct measurement between them. The information matrix (Hessian) has nonzeros only for connected pose pairs → **banded/sparse structure**.

### 9.5 Schur Complement Elimination

Partition variables into two groups:

$$\begin{pmatrix} H_{aa} & H_{ab} \\ H_{ba} & H_{bb} \end{pmatrix} \begin{pmatrix} x_a \\ x_b \end{pmatrix} = \begin{pmatrix} g_a \\ g_b \end{pmatrix}$$

Eliminate $x_b$ to get the **Schur complement**:

$$\boxed{S = H_{aa} - H_{ab} H_{bb}^{-1} H_{ba}}$$

$$S \, x_a = g_a - H_{ab} H_{bb}^{-1} g_b$$

Then back-substitute: $x_b = H_{bb}^{-1}(g_b - H_{ba} x_a)$.

**In bundle adjustment:**
- $x_a$ = camera poses (few, coupled)
- $x_b$ = landmarks (many, independent ⇒ $H_{bb}$ is block-diagonal ⇒ cheap to invert)
- $S$ is dense but small (camera-only)

### 9.6 Marginalization Preview

Schur complement = **marginalization** in probabilistic terms. Eliminating $x_b$ from the joint distribution $p(x_a, x_b)$ yields the marginal $p(x_a)$ whose information matrix is exactly $S$.

Key issue: marginalization **fills in** the information matrix — connected variables that were conditionally independent given $x_b$ become directly coupled.

---

## Implementation (60 min)

### Exercise 1: Tridiagonal System
```python
# See: code/week02/sparse_schur.py
```
Create a 1000×1000 tridiagonal PD system. Compare dense vs sparse solve.

### Exercise 2: Sparsity Visualization
Use `plt.spy()` to visualize sparsity patterns for: tridiagonal, block-diagonal, pose-graph, and arrow-shaped matrices.

### Exercise 3: Schur Complement
Implement Schur complement elimination for a block-structured system. Verify it gives the same answer as solving the full system.

### Exercise 4: Fill-in with Ordering
Create a sparse PD matrix. Factor with natural ordering vs AMD ordering. Compare fill-in (count nonzeros in $L$).

---

## Practice (45 min)

1. For a chain of $n$ poses (1D pose graph), what is the sparsity pattern of the information matrix?
2. How many nonzeros in a tridiagonal $n \times n$ matrix? What fraction of entries is nonzero?
3. If $H_{bb}$ is $3000 \times 3000$ block-diagonal with $3 \times 3$ blocks, what is the cost of computing $H_{bb}^{-1}$?
4. Why does Ceres use `SPARSE_SCHUR` for BA but `SPARSE_NORMAL_CHOLESKY` for pose graphs?
5. What happens to the Schur complement's sparsity when you marginalize landmarks in BA?
6. In a 100-pose SLAM graph where each pose sees 3 landmarks, draw the sparsity pattern.
7. Explain: why is AMD ordering $O(n \cdot \text{nnz})$ but saves $O(n^2)$ in the factorization?
8. What is the "fill-reducing ordering" doing intuitively?

---

## Expert Challenges

<details>
<summary>🎯 Challenge 1: Sliding window marginalization</summary>

**Problem:** You're running online SLAM with a sliding window of 10 poses. Each step: (1) add new pose, (2) marginalize oldest pose. After 100 steps, what happened to the sparsity of the marginal information matrix? Why does this cause problems?

**Solution:** Each marginalization fills in connections between all variables that were connected to the marginalized pose. After many marginalizations, the information matrix of the window becomes increasingly dense — the "fill-in accumulation" problem. After 100 steps, the 10-pose window's information matrix may be nearly fully dense, destroying the sparsity that makes SLAM efficient. Solutions: (1) only marginalize when necessary, (2) sparsification after marginalization (drop small entries, accept approximation), (3) use iSAM2 which avoids explicit marginalization.
</details>

<details>
<summary>🎯 Challenge 2: Variable ordering for a simple SLAM graph</summary>

**Problem:** Consider 5 poses (P1-P5) in a chain, with P5 connected back to P1 (loop closure). The landmarks (L1, L2) are each seen by two poses. Find the optimal variable elimination ordering to minimize fill-in.

**Solution:** Eliminate landmarks first (they have low connectivity): L1, L2. This fills in connections between the poses that see each landmark. Then eliminate poses in order P1-P5. The loop closure between P1 and P5 means eliminating P1 creates fill-in to P5. The optimal ordering is approximately: L1, L2, then P2, P3, P4, then P1 or P5 (the loop nodes last, since they have highest degree after landmark elimination). METIS or AMD would approximate this.
</details>

<details>
<summary>🎯 Challenge 3: Implement a minimal sparse Cholesky</summary>

**Problem:** Implement a sparse Cholesky factorization for banded matrices (bandwidth $w$). Show it's $O(n w^2)$ vs $O(n^3)$ for dense. Test on the 1D pose graph system.

**Solution:** For a banded matrix with bandwidth $w$, the Cholesky factor $L$ is also banded with the same bandwidth. The column-$j$ computation only touches rows $j$ to $\min(j+w, n)$, giving $O(w^2)$ per column and $O(nw^2)$ total. For a tridiagonal system ($w=1$), this is $O(n)$ — linear time!
</details>

---

## Self-Assessment Checklist

- [ ] I can create and manipulate sparse matrices in SciPy
- [ ] I can derive and apply the Schur complement formula
- [ ] I understand fill-in and the role of variable ordering
- [ ] I can explain why BA uses Schur complement to eliminate landmarks

---

## Key Takeaways

1. **SLAM matrices are sparse** — exploiting sparsity is not optional, it's the difference between $O(n)$ and $O(n^3)$.
2. **Schur complement** eliminates a variable block cheaply when it has special structure (block-diagonal).
3. **Variable ordering** controls fill-in — bad ordering makes sparse problems behave like dense ones.
4. **Marginalization = Schur complement** — removing variables fills in the remaining system.
