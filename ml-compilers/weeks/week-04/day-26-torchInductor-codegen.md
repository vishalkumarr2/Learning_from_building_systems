# Day 26: TorchInductor Code Generation

> **Phase II · Week 4 · Day 26 of 70 · 2.5 hours**

*"Inductor is where abstract math meets real hardware. Its job is deceptively simple: take a graph of tensor operations and produce the fastest possible Triton kernel. The magic is in the decisions it makes along the way — what to fuse, how to schedule, and what shape the generated code takes."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 25: torch.compile Internals](day-25-torch-compile-internals.md) | [Day 27: Custom Triton Backend](day-27-custom-triton-backend.md) | [Week 4: Triton & Kernel Engineering](README.md) | [Phase II: Compiler Fundamentals](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Yesterday we traced the full `torch.compile` pipeline end-to-end. Today we zoom into **Stage 3 — Inductor** and study how it actually turns an FX graph into Triton source code. This is where every fusion decision, memory layout choice, and scheduling trade-off lives. Understanding Inductor's codegen lets you: (1) read the generated Triton to diagnose performance issues, (2) write custom lowerings when Inductor's defaults are suboptimal, and (3) understand *why* certain patterns run fast and others don't.

---

## 1. Inductor's Internal Pipeline

The Inductor backend receives an FX graph of ATen primitives from AOTAutograd. It transforms that graph through four stages before emitting code:

```
FX Graph (ATen ops)
     │
     ▼
┌──────────────────────────────────────────────────────┐
│  1. LOWERING                                          │
│  ─────────────────                                    │
│  ATen ops → Inductor IR nodes                         │
│  aten.add → pointwise(lambda x, y: x + y)            │
│  aten.sum → reduction(lambda x, y: x + y, init=0)    │
│  aten.mm  → extern_kernel(triton_matmul)              │
│                                                       │
│  Registry: torch._inductor.lowering                   │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  2. FUSION                                            │
│  ──────────────                                       │
│  Group IR nodes into SchedulerNode clusters            │
│  Rules:                                               │
│    • pointwise + pointwise → fuse (always)            │
│    • pointwise + reduction → fuse (same shape)        │
│    • reduction + pointwise → fuse (broadcast ok)      │
│    • extern kernels → never fuse                      │
│                                                       │
│  Goal: minimize kernel launches & memory traffic      │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  3. SCHEDULING                                        │
│  ─────────────────                                    │
│  For each fused group:                                │
│    • Choose loop ordering (row-major vs column-major) │
│    • Determine block sizes and grid dimensions        │
│    • Handle reductions (single-block vs multi-block)  │
│    • Insert synchronization points                    │
│                                                       │
│  Key file: torch._inductor.scheduler                  │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  4. CODE GENERATION                                   │
│  ──────────────────────                               │
│  Emit Triton Python source (GPU) or C++ (CPU)         │
│  Compile via Triton JIT or gcc/clang                  │
│  Cache in __pycache__/torch_inductor/                 │
│                                                       │
│  Key file: torch._inductor.codegen.triton             │
└──────────────────────────────────────────────────────┘
```

---

## 2. Lowering: ATen → Inductor IR

Lowering translates each ATen operation into an Inductor IR node. The registry maps operation names to lowering functions:

```python
# Simplified view of how lowerings are registered
# (from torch/_inductor/lowering.py)

@register_lowering(aten.add)
def lower_add(a, b):
    return pointwise(
        args=[a, b],
        fn=lambda x, y: ops.add(x, y),
    )

@register_lowering(aten.sum, type_promotion_kind=None)
def lower_sum(x, dims, keepdim=False):
    return reduction(
        x,
        reduction_type="sum",
        dims=dims,
        keepdim=keepdim,
    )
```

### Inductor IR Node Types

| IR Node | Description | Example Ops |
|:--|:--|:--|
| `Pointwise` | Element-wise, same output shape | add, mul, relu, sin, cast |
| `Reduction` | Collapses dimension(s) | sum, mean, amax, argmax |
| `TensorBox` | Wraps a storage + layout | inputs, constants |
| `ExternKernel` | Calls pre-compiled kernel | mm, conv2d, SDPA |
| `TemplateKernel` | Specialized Triton template | fused GEMM + epilogue |
| `FallbackKernel` | Runs via ATen (no codegen) | unsupported ops |

### The ops Namespace

Inside a lowering function, `ops.*` provides a hardware-neutral set of operations:

```python
# ops.* is Inductor's "virtual instruction set"
ops.add(x, y)           # addition
ops.mul(x, y)           # multiplication
ops.maximum(x, y)       # element-wise max
ops.where(cond, a, b)   # conditional select
ops.exp(x)              # exponential
ops.log(x)              # natural log
ops.to_dtype(x, dtype)  # type cast
ops.load(name, index)   # memory load
ops.store(name, index, val)  # memory store
```

These get translated to Triton intrinsics during code generation:
- `ops.add(x, y)` → `x + y`
- `ops.maximum(x, y)` → `tl.maximum(x, y)`
- `ops.exp(x)` → `tl.exp(x)`

---

## 3. Fusion: The Scheduler's Key Decision

Fusion is Inductor's most impactful optimization. Each fusion eliminates a kernel launch (~5 µs) and an entire round-trip to global memory.

### Fusion Cost Model

```
Memory traffic WITHOUT fusion:         WITH fusion:
─────────────────────────               ────────────────
Kernel 1: y = x * 2                     Kernel 1 (fused):
  Load x:    N × 4 bytes                  Load x:    N × 4 bytes
  Store y:   N × 4 bytes                  Store z:   N × 2 bytes
Kernel 2: z = relu(y).half()             ────────────────
  Load y:    N × 4 bytes                 Total: N × 6 bytes
  Store z:   N × 2 bytes
─────────────────────                   
Total: N × 14 bytes                     Savings: 57% less memory traffic
                                        + 1 fewer kernel launch
```

### Fusion Rules

```python
# Inductor's fusion decision tree (simplified):

def can_fuse(node_a, node_b):
    # Rule 1: both must be schedulable (not extern)
    if is_extern(node_a) or is_extern(node_b):
        return False
    
    # Rule 2: no circular dependencies
    if creates_cycle(node_a, node_b):
        return False
    
    # Rule 3: compatible iteration ranges
    if not compatible_ranges(node_a, node_b):
        return False
    
    # Rule 4: pointwise+pointwise always fuses
    if is_pointwise(node_a) and is_pointwise(node_b):
        return True
    
    # Rule 5: reduction fusion requires matching dims
    if is_reduction(node_a) or is_reduction(node_b):
        return check_reduction_compatibility(node_a, node_b)
    
    return False
```

### What Blocks Fusion

| Blocker | Why | Example |
|:--|:--|:--|
| Extern kernel | Pre-compiled, can't inline | `torch.mm`, `conv2d` |
| Shape mismatch | Different iteration domains | `(B,M,K)` op + `(B,N)` op |
| Circular dependency | A depends on B and B on A | Rare, usually a graph issue |
| Reduction boundary | Reduction output feeds non-matching op | `sum(x, dim=1)` → `(B,)` + `(B,M)` op |

---

## 4. Kernel Categories in Generated Code

Inductor generates three kinds of Triton kernels:

### 4.1 Pointwise Kernels

The simplest and most common. One program instance per block of elements:

```python
# Generated by Inductor for: y = relu(x * 2 + 1).half()
@triton.jit
def triton_poi_fused_mul_add_relu_to_half_0(
    in_ptr0,      # input tensor
    out_ptr0,     # output tensor
    xnumel,       # total number of elements
    XBLOCK: tl.constexpr,
):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    
    x0 = tl.load(in_ptr0 + xindex, xmask)
    x1 = x0 * 2.0                           # mul
    x2 = x1 + 1.0                           # add
    x3 = tl.maximum(x2, 0.0)                # relu
    x4 = x3.to(tl.float16)                  # cast
    tl.store(out_ptr0 + xindex, x4, xmask)
```

**Naming convention:** `triton_poi_fused_{op_names}_{kernel_id}`

### 4.2 Reduction Kernels

Handle operations that collapse dimensions. Two-level: in-block reduce, then cross-block:

```python
# Generated for: y = x.sum(dim=1)  where x is (M, N)
@triton.jit
def triton_red_fused_sum_0(
    in_ptr0,
    out_ptr0,
    xnumel,       # outer dimension (M)
    rnumel,       # reduction dimension (N)
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    
    accumulator = tl.zeros([XBLOCK, RBLOCK], dtype=tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + tl.arange(0, RBLOCK)
        rmask = rindex < rnumel
        
        tmp = tl.load(
            in_ptr0 + (xindex[:, None] * rnumel + rindex[None, :]),
            xmask[:, None] & rmask[None, :],
        )
        accumulator += tmp
    
    result = tl.sum(accumulator, axis=1)
    tl.store(out_ptr0 + xindex, result, xmask)
```

**Naming convention:** `triton_red_fused_{op_names}_{kernel_id}`

### 4.3 Template Kernels

For complex patterns (GEMM + epilogue), Inductor uses hand-written Triton templates:

```python
# Template: GEMM + bias + ReLU
# (used when Inductor detects: linear(x) followed by fused pointwise)
#
# These live in torch/_inductor/codegen/triton_templates/
# and are specialized at codegen time with concrete shapes
```

**Naming convention:** `triton_tem_{template_name}_{kernel_id}`

---

## 5. Reading Generated Code

The most practical Inductor skill: reading what it actually produces.

```python
import torch

# Tell Inductor to write generated code to disk
torch._inductor.config.debug = True

@torch.compile
def layer_norm_gelu(x, weight, bias):
    x = torch.layer_norm(x, [x.shape[-1]], weight, bias)
    return torch.nn.functional.gelu(x)

x = torch.randn(32, 128, device="cuda")
w = torch.randn(128, device="cuda")
b = torch.randn(128, device="cuda")
out = layer_norm_gelu(x, w, b)
```

```bash
# Find the generated code
TORCH_LOGS="output_code" python script.py 2>&1 | head -100

# Or look in the cache directory:
ls /tmp/torchinductor_$USER/
# ├── triton/
# │   ├── 0/
# │   │   ├── triton_poi_fused_...py     ← generated Triton kernels
# │   │   └── triton_red_fused_...py
# │   └── ...
# └── wrapper_code/
#     └── ...py                           ← wrapper that calls kernels
```

### The Wrapper Code

Inductor generates two things: Triton kernels *and* a Python wrapper that orchestrates them:

```python
# Simplified wrapper (auto-generated by Inductor)
def call(args):
    arg0, arg1, arg2 = args   # x, weight, bias
    
    # Allocate output buffer
    buf0 = empty_strided((32, 128), (128, 1), device='cuda', dtype=torch.float32)
    
    # Launch fused layer_norm + gelu kernel
    triton_red_fused_native_layer_norm_gelu_0.run(
        arg0, arg1, arg2, buf0,
        32,          # xnumel (batch)
        128,         # rnumel (hidden dim)
        grid=grid(32),
        XBLOCK=1,
        RBLOCK=128,
    )
    
    return (buf0,)
```

---

## 6. Writing Custom Lowerings

When Inductor's default lowering for an op is suboptimal, you can register your own:

```python
import torch
from torch._inductor.lowering import register_lowering, pointwise
from torch._inductor import ir

# Custom lowering for a hypothetical fused op
@register_lowering(torch.ops.mylib.fused_silu)
def lower_fused_silu(x):
    """SiLU(x) = x * sigmoid(x) as a single pointwise kernel."""
    
    def fn(x):
        # ops.* expressions get compiled to Triton intrinsics
        sigmoid_x = ops.sigmoid(x)
        return ops.mul(x, sigmoid_x)
    
    return pointwise(
        args=[x],
        fn=fn,
    )
```

### When to Write a Custom Lowering

| Scenario | Default Inductor Behavior | Custom Lowering Benefit |
|:--|:--|:--|
| Fused activation | Separate mul + sigmoid | Single pass, 2× less memory |
| Custom dtype logic | Falls back to ATen | Stays in Triton |
| Approximate math | Uses `tl.exp` (exact) | Use fast-math approximation |
| Domain-specific op | FallbackKernel (no fusion) | Enables fusion with neighbors |

### Testing Your Lowering

```python
import torch

# Register the custom op first (via torch.library)
@torch.library.custom_op("mylib::fused_silu", mutates_args=())
def fused_silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)  # eager fallback

@fused_silu.register_fake
def fused_silu_fake(x):
    return torch.empty_like(x)

# Now test with torch.compile
@torch.compile
def model(x):
    return torch.ops.mylib.fused_silu(x)

x = torch.randn(1024, device="cuda")
result = model(x)
expected = x * torch.sigmoid(x)
assert torch.allclose(result, expected, atol=1e-5)
```

---

## Hands-On Exercises

### Exercise 1: Read Generated Code (20 min)

```python
import torch

torch._inductor.config.debug = True

@torch.compile
def f(x):
    y = x.cos()
    y = y + 1.0
    y = y * x.sin()
    return y.sum()

x = torch.randn(1024, device="cuda")
result = f(x)

# TODO:
# 1. Set TORCH_LOGS="output_code" and run
# 2. Find the generated Triton kernel(s)
# 3. How many kernels were generated? Why?
# 4. Identify which ops were fused together
# 5. What is the reduction strategy for sum()?
```

### Exercise 2: Fusion Experiment (25 min)

```python
import torch
import torch._inductor.config as inductor_config

# Compare fusion vs no-fusion
inductor_config.debug = True

@torch.compile
def fused_chain(x):
    y = x * 2
    y = y + 1
    y = torch.relu(y)
    y = y.half()
    return y

@torch.compile
def broken_chain(x):
    y = x * 2
    y = y + 1
    z = torch.mm(y, y.T)   # extern kernel breaks the chain
    z = torch.relu(z)
    z = z.half()
    return z

x = torch.randn(256, 256, device="cuda")

# TODO:
# 1. Run both functions, inspect generated code
# 2. Count kernels in each case
# 3. Why does mm break the fusion chain?
# 4. Measure wallclock time difference
```

### Exercise 3: Custom Lowering (30 min)

```python
import torch
from torch._inductor.lowering import register_lowering, pointwise

# Implement a custom lowering for "squared ReLU":
#   squared_relu(x) = relu(x)^2
#
# Default Inductor: relu and square are separate pointwise nodes
# (they'll fuse anyway — but this exercise teaches the mechanics)

# Step 1: Define the custom op
@torch.library.custom_op("mylib::squared_relu", mutates_args=())
def squared_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x) ** 2

@squared_relu.register_fake
def squared_relu_fake(x):
    return torch.empty_like(x)

# Step 2: TODO — write the custom lowering
# @register_lowering(torch.ops.mylib.squared_relu)
# def lower_squared_relu(x):
#     ...

# Step 3: Verify
@torch.compile
def model(x):
    return torch.ops.mylib.squared_relu(x)

x = torch.randn(1024, device="cuda")
result = model(x)
expected = torch.relu(x) ** 2
print(f"Correct: {torch.allclose(result, expected)}")
```

---

## Key Takeaways

1. **Four-stage pipeline** — Inductor transforms ATen ops through lowering → fusion → scheduling → code generation, each stage narrowing the abstraction.
2. **Lowerings map ATen → IR** — Every supported ATen op has a lowering function that expresses it in terms of `ops.*` primitives. Custom lowerings let you override defaults.
3. **Fusion is the biggest win** — Merging pointwise ops into a single kernel reduces kernel launches and memory traffic by $n\times$ for $n$ fused ops.
4. **Three kernel types** — Pointwise (`poi`), reduction (`red`), and template (`tem`) kernels handle different iteration patterns.
5. **Read the generated code** — `TORCH_LOGS="output_code"` and `torch._inductor.config.debug = True` are your best debugging tools.
6. **Extern kernels break fusion** — `mm`, `conv2d`, and other pre-compiled kernels act as fusion barriers, splitting the graph into separate kernel groups.

---

## Further Reading

- **Inductor source code:** `torch/_inductor/` — start with `compile_fx.py`, `lowering.py`, `scheduler.py`
- **Inductor design doc:** [dev-discuss.pytorch.org — TorchInductor overview](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-target-backends/747)
- **Custom ops + torch.compile:** [pytorch.org/tutorials/advanced/custom_ops_landing_page.html](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
- **Triton code generation internals:** `torch/_inductor/codegen/triton.py`
- **Mark Saroufim's Inductor walkthrough:** YouTube — "PyTorch Compiler Stack" (2024)

---

## Tomorrow

**Day 27** takes us from reading Inductor's generated code to **writing our own torch.compile backend**. We'll use the backend API to intercept the FX graph and generate custom Triton kernels, learning when to customize vs. let Inductor do its thing.
