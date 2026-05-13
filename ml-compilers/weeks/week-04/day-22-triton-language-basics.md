# Day 22: Triton Language Basics

> **Phase II · Week 4 · Day 22 of 70 · 2.5 hours**

*"Triton's insight is radical: let the programmer think in blocks, not threads — and let the compiler handle the scheduling, memory coalescing, and shared memory that make CUDA so painful."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 21: Mini-Project — FX Pass](../week-03/day-21-mini-project-fx-pass.md) | [Day 23: Triton Matrix Multiplication](day-23-triton-matmul.md) | [Week 4: Triton & Kernel Engineering](README.md) | [Phase II: Compiler Fundamentals](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

CUDA gives you maximum control — and maximum ways to shoot yourself in the foot. You manage thread blocks, shared memory, bank conflicts, warp divergence, and memory coalescing *by hand*. Triton offers a different deal: write kernels in Python at the **block level**, and let the Triton compiler handle the low-level scheduling. The result is 80–95% of hand-tuned CUDA performance with 3–5× less code. This is why `torch.compile` (Inductor) generates Triton kernels as its primary GPU backend.

---

## 1. The Gap Triton Fills

### The GPU Programming Spectrum

```
Abstraction                     Performance    Effort
──────────────────────────────────────────────────────
PyTorch Eager                   ████░░░░░░     ★☆☆☆☆
  torch.compile (Inductor)      ███████░░░     ★★☆☆☆
  Triton kernels                ████████░░     ★★★☆☆  ◄── sweet spot
  CUDA C++ (hand-tuned)         ██████████     ★★★★★
  PTX / SASS assembly           ██████████     ★★★★★
──────────────────────────────────────────────────────
```

### Why Not Just Use CUDA?

| Pain Point | CUDA | Triton |
|:--|:--|:--|
| Thread indexing | Manual `threadIdx.x + blockIdx.x * blockDim.x` | Automatic via `tl.program_id` + block size |
| Shared memory | Explicit `__shared__`, `__syncthreads()` | Compiler manages automatically |
| Memory coalescing | Manual layout planning | Block loads are coalesced by construction |
| Warp divergence | Manual branch analysis | Masking model eliminates divergence |
| Autotuning | External scripts (e.g., cuBLAS heuristics) | Built-in `@triton.autotune` |
| Language | C++, separate compilation | Python, JIT compiled |

---

## 2. The Block-Based Programming Model

Triton's core abstraction: each **program instance** operates on a **block** of data, not a single element.

```
CUDA model (thread-centric):
┌─────────────────────────────────────────────┐
│ Grid of Thread Blocks                       │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐          │
│  │ T0  │ │ T1  │ │ T2  │ │ T3  │ ...       │
│  │ one │ │ one │ │ one │ │ one │           │
│  │ elem│ │ elem│ │ elem│ │ elem│           │
│  └─────┘ └─────┘ └─────┘ └─────┘          │
│  Each thread processes 1 element            │
└─────────────────────────────────────────────┘

Triton model (block-centric):
┌─────────────────────────────────────────────┐
│ Grid of Programs                            │
│  ┌───────────┐ ┌───────────┐               │
│  │ Program 0 │ │ Program 1 │ ...            │
│  │ [x0..x127]│ │[x128..255]│               │
│  │ BLOCK_SIZE│ │ BLOCK_SIZE│               │
│  │ elements  │ │ elements  │               │
│  └───────────┘ └───────────┘               │
│  Each program processes BLOCK_SIZE elements │
└─────────────────────────────────────────────┘
```

**Key insight:** Triton programs are parameterized by `BLOCK_SIZE` (a compile-time constant). The compiler decides how to map blocks to warps, threads, and registers internally.

---

## 3. Core Primitives: `tl.load`, `tl.store`, and Masks

### The Building Blocks

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,        # Pointer to input tensor x
    y_ptr,        # Pointer to input tensor y
    out_ptr,      # Pointer to output tensor
    n_elements,   # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Compile-time constant
):
    # Step 1: Which block am I?
    pid = tl.program_id(axis=0)
    
    # Step 2: Compute offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Step 3: Create mask for out-of-bounds elements
    mask = offsets < n_elements
    
    # Step 4: Load data (masked — OOB loads return 0)
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Step 5: Compute
    output = x + y
    
    # Step 6: Store results (masked — OOB stores are no-ops)
    tl.store(out_ptr + offsets, output, mask=mask)
```

### How Masking Works

```
Tensor: [a, b, c, d, e, f, g]   n_elements = 7
BLOCK_SIZE = 4

Program 0 (pid=0):
  offsets = [0, 1, 2, 3]
  mask    = [T, T, T, T]    (all < 7)
  loads   = [a, b, c, d]    ← all valid

Program 1 (pid=1):
  offsets = [4, 5, 6, 7]
  mask    = [T, T, T, F]    (7 is NOT < 7)
  loads   = [e, f, g, 0]    ← last element masked
  stores  = [e+.., f+.., g+.., skip]
```

This eliminates the need for separate boundary-check logic — the mask pattern handles it uniformly.

---

## 4. Writing the Host-Side Launcher

The kernel alone isn't enough — you need Python code to set up the grid and call it:

```python
import torch

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Element-wise addition using Triton kernel."""
    assert x.is_cuda and y.is_cuda
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # Grid: how many program instances to launch
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # Launch kernel
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output

# Usage
x = torch.rand(1_000_000, device='cuda')
y = torch.rand(1_000_000, device='cuda')
z = add(x, y)

# Verify correctness
assert torch.allclose(z, x + y)
```

### Grid Calculation

$$\text{num\_programs} = \left\lceil \frac{N}{\text{BLOCK\_SIZE}} \right\rceil = \texttt{triton.cdiv}(N, \text{BLOCK\_SIZE})$$

For $N = 1{,}000{,}000$ and $\text{BLOCK\_SIZE} = 1024$: $\lceil 1{,}000{,}000 / 1024 \rceil = 977$ programs.

---

## 5. Reduction Kernel: Summing a Vector

Reductions are more interesting — they require cooperation within a block:

```python
@triton.jit
def sum_kernel(
    x_ptr,
    out_ptr,          # Pointer to scalar output
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load block of data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Block-level reduction — Triton handles the tree reduce
    block_sum = tl.sum(x, axis=0)
    
    # Atomic add to global accumulator
    tl.atomic_add(out_ptr, block_sum)


def triton_sum(x: torch.Tensor) -> torch.Tensor:
    output = torch.zeros(1, device=x.device, dtype=x.dtype)
    n = x.numel()
    grid = (triton.cdiv(n, 1024),)
    sum_kernel[grid](x, output, n, BLOCK_SIZE=1024)
    return output
```

### Reduction Pattern

```
Block data: [x0, x1, x2, x3, x4, x5, x6, x7]

tl.sum() compiles to a warp-level tree reduction:
  Step 1: [x0+x1, x2+x3, x4+x5, x6+x7]
  Step 2: [x0+x1+x2+x3, x4+x5+x6+x7]
  Step 3: [x0+x1+x2+x3+x4+x5+x6+x7]

Then atomic_add to global accumulator
(tl.sum uses shuffle instructions — no shared memory needed)
```

---

## 6. How Triton Compiles to PTX

Triton doesn't interpret Python — it compiles through a multi-stage pipeline:

```
Python with @triton.jit
        │
        ▼
  ┌──────────────┐
  │ Triton AST   │  Parse decorated function
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Triton IR    │  SSA form, block-level ops
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  MLIR        │  Triton dialect → GPU dialect
  │  (LLVM-based)│  Tiling, memory planning
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ LLVM IR      │  Standard LLVM passes
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ PTX Assembly │  NVIDIA GPU assembly
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ cubin        │  Binary loaded by CUDA driver
  └──────────────┘
```

### Inspecting the Generated Code

```python
# Compile and inspect PTX
compiled = add_kernel.warmup(
    torch.float32, torch.float32, torch.float32,
    1024, BLOCK_SIZE=1024, grid=(1,)
)

# View generated LLVM IR
print(compiled.asm['llir'])

# View generated PTX
print(compiled.asm['ptx'])

# View generated SASS (native GPU assembly)
# Requires ptxas/cuobjdump
```

### What the Compiler Does For You

The Triton compiler automatically handles:

| Optimization | What it does |
|:--|:--|
| **Shared memory allocation** | Promotes reused loads to shared memory |
| **Memory coalescing** | Reorders accesses for 128-byte transactions |
| **Vectorization** | Merges scalar loads into `LDG.128` |
| **Software pipelining** | Overlaps loads with compute via `cp.async` |
| **Register allocation** | Maps block elements to registers |
| **Warp scheduling** | Distributes block across warps |

---

## 7. CUDA vs Triton: Side-by-Side Comparison

### Vector Addition

**CUDA C++ (25 lines + build system):**

```cuda
__global__ void add_kernel(float* x, float* y, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] + y[idx];
    }
}

void add(float* x, float* y, float* out, int n) {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    add_kernel<<<grid_size, block_size>>>(x, y, out, n);
}
```

**Triton (15 lines, pure Python):**

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)
```

For simple kernels, CUDA and Triton generate near-identical PTX. The advantage of Triton grows with kernel complexity — matmuls, attention, and fused operations where shared memory management dominates the CUDA code.

---

## Hands-On Exercises

### Exercise 1: Fused Multiply-Add Kernel (20 min)

Write a Triton kernel that computes $\text{out} = a \cdot x + y$ (FMA) for vectors. Verify it matches `torch.addcmul`.

```python
@triton.jit
def fma_kernel(a_ptr, x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n
    # TODO: load a, x, y — compute a*x + y — store
```

### Exercise 2: Softmax Kernel (40 min)

Implement a row-wise softmax in Triton. Each program handles one row:

$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

```python
@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_cols,
    input_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # Load row
    row = tl.load(input_ptr + row_start + col_offsets, mask=mask, other=-float('inf'))
    
    # Numerically stable softmax
    row_max = tl.max(row, axis=0)
    row_exp = tl.exp(row - row_max)
    row_sum = tl.sum(row_exp, axis=0)
    softmax_out = row_exp / row_sum
    
    tl.store(output_ptr + row_idx * output_row_stride + col_offsets, softmax_out, mask=mask)
```

### Exercise 3: Benchmark (20 min)

Compare your vector add kernel against `torch.add` for sizes $[2^{10}, 2^{14}, 2^{18}, 2^{22}]$:

```python
import triton.testing

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],
        x_vals=[2**i for i in range(10, 23)],
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='vector-add-performance',
        args={},
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: x + y)
    else:
        ms = triton.testing.do_bench(lambda: add(x, y))
    
    gbps = 3 * x.numel() * x.element_size() / ms * 1e-6  # 2 reads + 1 write
    return gbps

benchmark.run(print_data=True, show_plots=True)
```

---

## Key Takeaways

1. **Block-level abstraction** — Triton programs operate on blocks of data, not individual elements; the compiler maps blocks to warps and threads.
2. **Masking replaces bounds checks** — `tl.load(..., mask=...)` and `tl.store(..., mask=...)` handle boundary conditions uniformly.
3. **Compiler does the hard work** — Shared memory, coalescing, vectorization, and pipelining are handled automatically by the MLIR-based compilation pipeline.
4. **Python ergonomics, GPU performance** — Write kernels in Python, get 80–95% of hand-tuned CUDA speed with a fraction of the code.
5. **Triton → MLIR → LLVM → PTX** — The compilation chain is fully inspectable at every stage.

---

## Further Reading

- **Triton paper:** *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations* (Tillet et al., 2019)
- **Official tutorials:** [triton-lang.org/main/getting-started/tutorials](https://triton-lang.org/main/getting-started/tutorials)
- **Triton internals talk:** Philippe Tillet's talks at GTC and LLVM Dev meetings
- **Source code:** [github.com/triton-lang/triton](https://github.com/triton-lang/triton) — read `python/triton/language/core.py` for primitive definitions

---

## Tomorrow

**Day 23** tackles the kernel that matters most: **matrix multiplication**. We'll implement a tiled GEMM in Triton with block pointers, accumulator patterns, and `@triton.autotune` — and benchmark it against cuBLAS. The matmul kernel is where Triton's block model truly shines, replacing hundreds of lines of CUDA shared-memory management with clean, declarative tiling.
