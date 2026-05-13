# Day 18: The Polyhedral Model

> **Phase II · Week 3 · Day 18 of 70 · 2.5 hours**

*"The polyhedral model turns 'can I legally reorder these loops?' from a gut feeling into a theorem you can prove with linear algebra."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 17: Graph-Level Optimizations](day-17-graph-level-optimizations.md) | [Day 19: Loop Tiling & Scheduling](day-19-loop-tiling-scheduling.md) | [Week 3: IRs & Passes](README.md) | [Phase II: Compiler Fundamentals](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Graph-level optimizations decide **which** operators to fuse. But once you've decided to fuse a `matmul + bias + relu` into one kernel, you still need to decide **how to execute the loops** inside that kernel — which loop goes outermost, how to tile for cache, whether to parallelize across GPU threads. The **polyhedral model** is the mathematical framework that tells you which loop transformations are **legal** (preserve correctness) and which are **profitable** (improve performance). TVM, Halide, MLIR's Affine dialect, and Pluto all build on this foundation.

---

## 1. The Problem: Loop Nest Optimization

Consider matrix multiply — three nested loops with a specific data access pattern:

```c
// Naive matmul: C[i][j] += A[i][k] * B[k][j]
for (int i = 0; i < M; i++)
  for (int j = 0; j < N; j++)
    for (int k = 0; k < K; k++)
      C[i][j] += A[i][k] * B[k][j];
```

Questions a compiler must answer:
1. Can we **reorder** loops (e.g., `i,k,j` instead of `i,j,k`)?
2. Can we **tile** loops (e.g., process 32×32 blocks)?
3. Can we **parallelize** the outer loop?
4. Can we **vectorize** the inner loop?

The polyhedral model answers all of these by modeling the computation as a **geometric object** in an integer lattice.

---

## 2. Three Components of the Polyhedral Model

### 2a. Iteration Domain

The set of all loop iterations, represented as an **integer polyhedron** — a set of integer points bounded by affine inequalities.

For the matmul above:

$$\mathcal{D}_S = \{(i, j, k) \in \mathbb{Z}^3 \mid 0 \leq i < M, \; 0 \leq j < N, \; 0 \leq k < K\}$$

This is a 3D rectangular box in iteration space:

```
        k ▲
          │    ┌────────────┐
          │   /            /│
     K ───┤  /            / │
          │ /            /  │
          │┌────────────┐   │
          ││            │   │
          ││  Iteration │  /
          ││   Domain   │ / N
          ││            │/
          │└────────────┘
          └──────────────────▶ j
         /
        / i
       ▼
       M
```

### 2b. Access Relations (Array Subscripts)

Each memory access is an **affine function** of the loop iterators:

| Statement: `C[i][j] += A[i][k] * B[k][j]` | Access | Affine Map |
|:--------------------------------------------|:-------|:-----------|
| Read A | $A[i][k]$ | $(i, j, k) \mapsto (i, k)$ |
| Read B | $B[k][j]$ | $(i, j, k) \mapsto (k, j)$ |
| Read/Write C | $C[i][j]$ | $(i, j, k) \mapsto (i, j)$ |

In matrix form, the access to $A[i][k]$:

$$\begin{pmatrix} \text{row} \\ \text{col} \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} i \\ j \\ k \end{pmatrix}$$

### 2c. Schedule (Execution Order)

A **schedule** maps each iteration point to a **timestamp** — when it should execute. The original schedule for `i,j,k` order:

$$\theta(i, j, k) = (i, j, k)$$

Meaning: iteration $(i, j, k)$ executes at time $(i, j, k)$ in lexicographic order. A different schedule, say loop order `i,k,j`:

$$\theta'(i, j, k) = (i, k, j)$$

A schedule is **legal** if it preserves all data dependencies.

---

## 3. Dependence Analysis

Two iterations **depend** on each other if they access the same memory location and at least one is a write. For matmul:

```c
C[i][j] += A[i][k] * B[k][j];
//  ^^ write                       ^^ same statement, different k
```

Iteration $(i, j, k_1)$ and $(i, j, k_2)$ both write to $C[i][j]$, creating a **dependence**:

$$\delta = (i, j, k_2) - (i, j, k_1) = (0, 0, k_2 - k_1)$$

The dependence vector is $(0, 0, +)$ — positive in the $k$ direction. This means:
- **Loops $i$ and $j$ are independent** — can be freely reordered, parallelized, or tiled
- **Loop $k$ carries a dependence** — iterations must execute in $k$-order (or use atomic reduction)

```
  Dependence Analysis for Matmul
  ───────────────────────────────
  Statement S: C[i][j] += A[i][k] * B[k][j]

  Dependencies (S → S):
    Flow (RAW):  (i,j,k) → (i,j,k+1)  on C[i][j]
    direction:   (0, 0, +)

  Conclusion:
    i-loop: parallelizable  ✓
    j-loop: parallelizable  ✓
    k-loop: reduction (must be sequential or atomic)

  Legal loop orders: any permutation that keeps k innermost
    ✓  i, j, k          ✓  j, i, k
    ✗  k, i, j  (would violate k-dependence if k is now parallel)
```

### The Legality Test

A schedule $\theta'$ is legal if and only if, for every dependence $(s_1, s_2)$ with dependence vector $\delta$:

$$\theta'(s_2) - \theta'(s_1) \succ \mathbf{0} \quad \text{(lexicographically positive)}$$

This ensures that the dependent iteration $s_2$ always executes after $s_1$.

---

## 4. Affine Transformations

Any loop transformation expressible as an affine function of the original iterators is a **polyhedral transformation**:

### Loop Interchange

Swap loop $i$ and $j$:

$$T_{\text{interchange}} = \begin{pmatrix} 0 & 1 & 0 \\ 1 & 0 & 0 \\ 0 & 0 & 1 \end{pmatrix}, \quad \begin{pmatrix} i' \\ j' \\ k' \end{pmatrix} = T \begin{pmatrix} i \\ j \\ k \end{pmatrix} = \begin{pmatrix} j \\ i \\ k \end{pmatrix}$$

### Loop Tiling (Strip-Mining + Interchange)

Split loop $i$ into blocks of size $B$:

$$i = B \cdot i_o + i_i, \quad 0 \leq i_i < B$$

The tiled loop nest `io, jo, ii, ji, k`:

```c
for (int io = 0; io < M/B; io++)
  for (int jo = 0; jo < N/B; jo++)
    for (int ii = 0; ii < B; ii++)
      for (int ji = 0; ji < B; ji++)
        for (int k = 0; k < K; k++)
          C[io*B+ii][jo*B+ji] += A[io*B+ii][k] * B[k][jo*B+ji];
```

This is an affine transformation from $(i, j, k) \to (i_o, j_o, i_i, j_i, k)$:

$$i_o = \lfloor i / B \rfloor, \quad i_i = i \mod B, \quad j_o = \lfloor j / B \rfloor, \quad j_i = j \mod B$$

### Loop Skewing

Used to expose parallelism in wavefront patterns:

$$\begin{pmatrix} i' \\ j' \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} i \\ j \end{pmatrix}$$

This transforms a diagonal dependence pattern into one where outer iterations are independent.

---

## 5. ISL: The Integer Set Library

**ISL** is the standard library for polyhedral operations. TVM, MLIR, Pluto, and PPCG all use it.

```python
# Python bindings (islpy)
import islpy as isl

# Define an iteration domain
domain = isl.Set("{ S[i,j,k] : 0 <= i < 128 and 0 <= j < 128 and 0 <= k < 64 }")
print(f"Domain: {domain}")
print(f"Dimension: {domain.dim(isl.dim_type.set)}")

# Define an access relation
read_A = isl.Map("{ S[i,j,k] -> A[i,k] }")
print(f"Read A: {read_A}")

# Apply domain to get accessed elements
accessed = read_A.intersect_domain(domain)
accessed_set = accessed.range()
print(f"A elements accessed: {accessed_set}")

# Define a schedule (original order i,j,k)
schedule = isl.Map("{ S[i,j,k] -> [i,j,k] }")

# Define a tiled schedule
tiled = isl.Map("{ S[i,j,k] -> [floor(i/32), floor(j/32), i - 32*floor(i/32), j - 32*floor(j/32), k] }")
print(f"Tiled schedule: {tiled}")
```

### Dependence Analysis with ISL

```python
# Dependence: S(i,j,k) writes C[i,j], S(i,j,k') reads C[i,j] where k' > k
write_C = isl.Map("{ S[i,j,k] -> C[i,j] }")
read_C = isl.Map("{ S[i,j,k] -> C[i,j] }")

# RAW dependence: same C element, source before sink
# In ISL notation:
dep = isl.Map("{ S[i,j,k] -> S[i,j,k'] : k' > k and 0 <= k < 64 and 0 <= k' < 64 }")
print(f"Dependence: {dep}")

# Check if a transformation is legal:
# New schedule must map source before sink for all dependencies
```

---

## 6. Connection to TVM Schedule Primitives

TVM's schedule primitives are **polyhedral transformations** with user-friendly names:

| TVM Primitive | Polyhedral Transformation | Effect |
|:-------------|:-------------------------|:-------|
| `split(i, 32)` | Strip-mine axis $i$ by factor 32 | Creates `i_outer, i_inner` |
| `reorder(j, i)` | Loop interchange | Swaps execution order |
| `tile(i, j, 32, 32)` | 2D strip-mine + interchange | Creates `io, jo, ii, ji` |
| `fuse(i, j)` | Collapse dimensions | Linearize two loops |
| `parallel(i)` | Mark as independent | Execute on threads |
| `vectorize(i)` | Mark innermost as SIMD | Use vector instructions |
| `compute_at(B, i)` | Change producer schedule | Fuse producer into consumer loop |

```python
# TVM example (conceptual)
import tvm
from tvm import te

M, N, K = 1024, 1024, 1024
A = te.placeholder((M, K), name='A')
B = te.placeholder((K, N), name='B')
k = te.reduce_axis((0, K), name='k')
C = te.compute((M, N), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')

s = te.create_schedule(C.op)

# Polyhedral transformations via TVM API
i, j = s[C].op.axis
k, = s[C].op.reduce_axis

# Tile: strip-mine i and j
io, ii = s[C].split(i, factor=32)
jo, ji = s[C].split(j, factor=32)

# Reorder: interchange for locality
s[C].reorder(io, jo, ii, ji, k)

# Parallelize outer loops
s[C].parallel(io)

# Vectorize innermost
s[C].vectorize(ji)

print(tvm.lower(s, [A, B, C], simple_mode=True))
```

---

## Hands-On Exercises

### Exercise 1: Dependence Vectors by Hand (20 min)

For each loop nest, determine the dependence vectors and which loops can be parallelized:

```c
// (a) Element-wise: Y[i] = X[i] * 2
for (int i = 0; i < N; i++)
    Y[i] = X[i] * 2;

// (b) Stencil: Y[i] = X[i-1] + X[i] + X[i+1]
for (int i = 1; i < N-1; i++)
    Y[i] = X[i-1] + X[i] + X[i+1];

// (c) Scan (prefix sum): Y[i] = Y[i-1] + X[i]
for (int i = 1; i < N; i++)
    Y[i] = Y[i-1] + X[i];

// (d) 2D convolution
for (int i = 0; i < H; i++)
  for (int j = 0; j < W; j++)
    for (int ki = 0; ki < 3; ki++)
      for (int kj = 0; kj < 3; kj++)
        O[i][j] += I[i+ki][j+kj] * K[ki][kj];
```

Expected answers:
| Case | Deps | Parallelizable? |
|:-----|:-----|:---------------|
| (a) | None (X≠Y) | Fully parallel |
| (b) | ? | ? |
| (c) | ? | ? |
| (d) | ? | ? |

### Exercise 2: Tiling a Matrix Multiply (25 min)

```python
import numpy as np
import time

def matmul_naive(A, B):
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i][j] += A[i][k] * B[k][j]
    return C

def matmul_tiled(A, B, block=32):
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N))
    for io in range(0, M, block):
        for jo in range(0, N, block):
            for ko in range(0, K, block):
                # Tile bounds
                ie = min(io + block, M)
                je = min(jo + block, N)
                ke = min(ko + block, K)
                for ii in range(io, ie):
                    for ji in range(jo, je):
                        for ki in range(ko, ke):
                            C[ii][ji] += A[ii][ki] * B[ki][ji]
    return C

# Test correctness and measure performance
N = 256  # Keep small for pure Python
A = np.random.randn(N, N)
B = np.random.randn(N, N)

t0 = time.perf_counter()
C1 = matmul_naive(A, B)
t_naive = time.perf_counter() - t0

t0 = time.perf_counter()
C2 = matmul_tiled(A, B, block=32)
t_tiled = time.perf_counter() - t0

print(f"Naive:  {t_naive:.3f}s")
print(f"Tiled:  {t_tiled:.3f}s")
print(f"Speedup: {t_naive/t_tiled:.2f}x")
print(f"Correct: {np.allclose(C1, C2)}")
```

### Exercise 3: ISL Exploration (Optional, 20 min)

```python
# pip install islpy
import islpy as isl

# 1. Create the iteration domain for 2D convolution
#    i: 0..63, j: 0..63, ki: 0..2, kj: 0..2
domain = isl.Set("{ S[i,j,ki,kj] : 0<=i<64 and 0<=j<64 and 0<=ki<3 and 0<=kj<3 }")

# 2. Write the access relation for I[i+ki][j+kj]
access_I = isl.Map("{ S[i,j,ki,kj] -> I[i+ki, j+kj] }")

# 3. What is the footprint? (set of I elements accessed)
footprint = access_I.intersect_domain(domain).range()
print(f"I footprint: {footprint}")
# Should be { I[r,c] : 0 <= r <= 66 and 0 <= c <= 66 }

# 4. Create a tiled schedule
tiled = isl.Map("{ S[i,j,ki,kj] -> [floor(i/8), floor(j/8), "
                "i - 8*floor(i/8), j - 8*floor(j/8), ki, kj] }")

# 5. What is the data reuse within one tile?
# Compute the I footprint for a single tile (io=0, jo=0)
tile_domain = isl.Set("{ S[i,j,ki,kj] : 0<=i<8 and 0<=j<8 and 0<=ki<3 and 0<=kj<3 }")
tile_footprint = access_I.intersect_domain(tile_domain).range()
print(f"Tile footprint: {tile_footprint}")
# Should be { I[r,c] : 0 <= r <= 10 and 0 <= c <= 10 } = 11x11 = 121 elements
# But total accesses = 8*8*3*3 = 576 → reuse ratio = 576/121 ≈ 4.8x
```

---

## Key Takeaways

1. **Iteration domains** are integer polyhedra — sets of loop iterations bounded by affine constraints
2. **Access relations** map iterations to array elements via affine functions — these determine data reuse and locality
3. **Schedules** assign timestamps to iterations — different schedules = different loop orders, all expressible as affine transforms
4. **Dependence analysis** determines which schedules are legal — a transformation is safe iff it preserves the lexicographic ordering of dependent iterations
5. **TVM's schedule primitives** (`split`, `reorder`, `tile`, `fuse`) are exactly polyhedral transformations with user-friendly APIs
6. **ISL** is the standard computational geometry library underpinning TVM, MLIR, Pluto, and other polyhedral tools

---

## Further Reading

- Bastoul, C. — *Code Generation in the Polyhedral Model Is Easier Than You Think* (PACT 2004)
- Verdoolaege, S. — *isl: An Integer Set Library for the Polyhedral Model* (ICMS 2010)
- Bondhugula, U. et al. — *A Practical Automatic Polyhedral Parallelizer and Locality Optimizer* (PLDI 2008) — the Pluto algorithm
- Grosser, T. et al. — *Polly — Performing Polyhedral Optimizations on a Low-Level Intermediate Representation* (2012)
- Ragan-Kelley, J. et al. — *Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation* (PLDI 2013)

---

## Tomorrow's Preview

**Day 19** puts the polyhedral model to work with **loop tiling and scheduling** — we'll tile matrix multiply for L1/L2 cache, implement multi-level tiling strategies, explore TVM's `AutoScheduler` (Ansor), and see how autotuning searches the space of legal schedules to find hardware-optimal configurations automatically.
