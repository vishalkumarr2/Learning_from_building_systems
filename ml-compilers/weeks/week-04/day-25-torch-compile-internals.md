# Day 25: torch.compile Internals

> **Phase II · Week 4 · Day 25 of 70 · 2.5 hours**

*"torch.compile is three compilers in a trench coat: Dynamo captures the graph, AOTAutograd differentiates it, and Inductor generates the code. Understanding each piece tells you where to look when things break."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 24: Triton Flash Attention](day-24-triton-flash-attention.md) | [Day 26: TorchInductor Codegen](day-26-torchInductor-codegen.md) | [Week 4: Triton & Kernel Engineering](README.md) | [Phase II: Compiler Fundamentals](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

`torch.compile` is how PyTorch delivers compiler-level optimizations without forcing users to rewrite their code. A single decorator turns eager-mode Python into optimized Triton kernels — often 2–3× faster with zero code changes. But when it doesn't work (graph breaks, dynamic shapes, mysterious slowdowns), you need to understand the internals to diagnose the problem. Today we trace a model through the full pipeline: **Dynamo** → **AOTAutograd** → **Inductor** → **Triton/C++**, and learn the debugging tools that make the opaque system transparent.

---

## 1. The Three-Stage Pipeline

```
User writes:
  @torch.compile
  def f(x):
      return x.sin() + x.cos()

What actually happens:

┌─────────────────────────────────────────────────────────────┐
│  Stage 1: TorchDynamo                                       │
│  ─────────────────                                          │
│  • Intercepts Python bytecode execution                     │
│  • Traces operations into an FX Graph                       │
│  • Inserts "guards" for assumptions (dtypes, shapes, etc.)  │
│  • Handles graph breaks when tracing fails                  │
│  Output: FX Graph (ATen-level IR)                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: AOTAutograd                                       │
│  ────────────────────                                       │
│  • Splits graph into forward + backward                     │
│  • Runs autograd ahead-of-time (not at runtime)             │
│  • Decomposes high-level ops into primitives                │
│  Output: Joint forward/backward FX Graphs                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 3: TorchInductor (Backend Compiler)                  │
│  ─────────────────────────────────────                      │
│  • Fuses operations (pointwise, reductions)                 │
│  • Schedules memory access patterns                         │
│  • Generates Triton kernels (GPU) or C++/OpenMP (CPU)       │
│  • Applies post-scheduling optimizations                    │
│  Output: Compiled executable code                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. TorchDynamo: Graph Capture via Bytecode Analysis

### How Dynamo Works

Unlike `torch.jit.trace` (which only captures tensor operations) or `torch.jit.script` (which requires a Python subset), Dynamo works at the **Python bytecode level**:

```python
import torch
import dis

def f(x, flag=True):
    y = x.sin()
    if flag:
        y = y + x.cos()
    return y

# View the bytecode Dynamo intercepts:
dis.dis(f)
```

```
  2           0 LOAD_FAST                0 (x)
              2 LOAD_METHOD              0 (sin)
              4 CALL_METHOD              0
              6 STORE_FAST               2 (y)

  3           8 LOAD_FAST                1 (flag)
             10 POP_JUMP_IF_FALSE       24

  4          12 LOAD_FAST                2 (y)
             14 LOAD_FAST                0 (x)
             16 LOAD_METHOD              1 (cos)
             18 CALL_METHOD              0
             20 BINARY_ADD
             22 STORE_FAST               2 (y)

  5     >>   24 LOAD_FAST                2 (y)
             26 RETURN_VALUE
```

Dynamo replaces Python's frame evaluation function (`_PyEval_EvalFrameDefault`) with its own evaluator that:

1. **Traces** tensor operations into an FX graph
2. **Records guards** for non-tensor values (e.g., `flag is True`)
3. **Compiles** the graph when the frame returns
4. **Caches** the compiled code, keyed by guards

```
Frame evaluation flow:

  Python calls f(x, flag=True)
       │
       ▼
  Dynamo intercepts frame
       │
       ├─ Cache hit? (guards match) → Run cached compiled code
       │
       └─ Cache miss → Trace new graph
            │
            ├─ Tensor ops → Add to FX graph
            │   LOAD_FAST x → track as graph input
            │   CALL_METHOD sin → add sin node
            │   BINARY_ADD → add add node
            │
            ├─ Non-tensor values → Record as guards
            │   flag == True → guard: "flag is True"
            │
            ├─ Unsupported ops → Graph break!
            │   (split into subgraphs)
            │
            └─ Frame returns → Compile graph → Cache it
```

---

## 3. Guards: When Dynamo Recompiles

Guards are conditions that must hold for the cached compiled code to be valid:

```python
@torch.compile
def f(x, multiplier):
    return x * multiplier

# First call: traces graph, records guards
f(torch.randn(4), 2.0)
# Guard: multiplier == 2.0, x.dtype == float32, x.shape == (4,)

# Second call: guards pass → use cached code
f(torch.randn(4), 2.0)  # Cache HIT

# Third call: multiplier changed → guard fails → retrace!
f(torch.randn(4), 3.0)  # Cache MISS → recompile
```

### Common Guard Types

| Guard Type | Example | Triggers Recompile When |
|:--|:--|:--|
| **Value** | `n == 10` | Python scalar changes |
| **Type** | `type(x) is Tensor` | Input type changes |
| **Shape** | `x.shape[0] == 32` | Tensor dimensions change |
| **Dtype** | `x.dtype == float32` | Tensor dtype changes |
| **Device** | `x.device == cuda:0` | Tensor device changes |
| **Attribute** | `x.requires_grad == True` | Grad tracking changes |

### Excessive Recompilation

```python
# BAD: recompiles every call because `n` is a guard
@torch.compile
def f(x, n):
    return x[:n]  # n is a Python int → guarded

for n in range(100):
    f(torch.randn(100), n)  # 100 recompilations!

# BETTER: use dynamic shapes
@torch.compile(dynamic=True)
def f(x, n):
    return x[:n]  # n is symbolically traced
```

---

## 4. Graph Breaks: When Dynamo Can't Trace

A **graph break** occurs when Dynamo encounters an operation it can't trace. The function is split into subgraphs with Python code between them:

```python
@torch.compile
def f(x):
    y = x.sin()            # ← Subgraph 1: sin
    print(f"Shape: {y.shape}")  # ← Graph break! (print is side effect)
    z = y.cos()            # ← Subgraph 2: cos
    return z
```

```
Without graph break:              With graph break:
  ┌──────────────┐                 ┌──────────────┐
  │ sin → cos    │                 │ sin          │  Subgraph 1 (compiled)
  │ (one kernel) │                 └──────┬───────┘
  └──────────────┘                        │
                                   Python: print(...)  (interpreted)
                                          │
                                   ┌──────┴───────┐
                                   │ cos          │  Subgraph 2 (compiled)
                                   └──────────────┘
```

### Common Graph Break Causes

| Cause | Example | Fix |
|:--|:--|:--|
| `print()` / logging | `print(x.shape)` | Remove or use `torch._dynamo.config.suppress_errors` |
| Data-dependent control flow | `if x.sum() > 0:` | Use `torch.where` instead |
| Unsupported Python builtins | `sorted(tensor_list)` | Rewrite with torch ops |
| Custom autograd functions | `MyFunction.apply(x)` | Register with `torch.library` |
| Non-standard module | Third-party ops | Wrap with `torch._dynamo.allow_in_graph` |

### Detecting Graph Breaks

```python
# Method 1: fullgraph=True (error on any break)
@torch.compile(fullgraph=True)
def f(x):
    y = x.sin()
    print(y.shape)  # Raises error instead of silently breaking
    return y.cos()

# Method 2: TORCH_LOGS (see what Dynamo produces)
# TORCH_LOGS="graph_breaks" python my_script.py
```

---

## 5. AOTAutograd: Forward/Backward Splitting

After Dynamo captures the forward graph, AOTAutograd creates the backward graph **ahead of time**:

```python
# User's function
def f(x):
    return x.sin().exp()

# After Dynamo → FX Graph:
# graph():
#   %x = placeholder
#   %sin = call_function[torch.sin](%x)
#   %exp = call_function[torch.exp](%sin)
#   return %exp

# After AOTAutograd → Two graphs:

# Forward graph (primals → output + saved tensors):
# graph():
#   %x = placeholder
#   %sin = call_function[aten.sin](%x)
#   %exp = call_function[aten.exp](%sin)
#   return (%exp, %sin, %x)  ← saves sin, x for backward
#                    ↑ these are "saved for backward"

# Backward graph (grad_output + saved → grad_input):
# graph():
#   %grad_out = placeholder          # dL/d(exp)
#   %sin_saved = placeholder          # saved from forward
#   %x_saved = placeholder            # saved from forward
#   %exp_grad = mul(%grad_out, exp(%sin_saved))   # d(exp)/d(sin) * dL/d(exp)
#   %sin_grad = mul(%exp_grad, cos(%x_saved))     # d(sin)/d(x) * chain
#   return %sin_grad
```

### Decomposition to Primitives

AOTAutograd also **decomposes** high-level ops into primitives that Inductor can fuse:

```
High-level:                    Decomposed:
  torch.nn.functional.gelu     aten.mul(x, 0.5 * (1 + aten.erf(x / sqrt(2))))
  torch.layer_norm              aten.mean → aten.var → aten.sub → aten.div
  torch.softmax                aten.exp(x - aten.amax(x)) / aten.sum(...)
```

This is crucial: Inductor doesn't need to know about `gelu` or `layer_norm` — it just sees primitive ops it can fuse.

---

## 6. TorchInductor: Code Generation

Inductor takes the decomposed graphs and generates executable code:

```
Inductor pipeline:

  FX Graph (ATen primitives)
       │
       ▼
  ┌─────────────────┐
  │ Lowering         │  Convert ATen ops to Inductor IR
  │                  │  (PointwiseOp, ReductionOp, etc.)
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Fusion           │  Group ops into fused kernels
  │                  │  (pointwise + pointwise → 1 kernel)
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Scheduling       │  Decide memory layout, loop ordering
  │                  │  Handle reductions, broadcasts
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ Code Generation  │  Emit Triton (GPU) or C++ (CPU)
  │                  │  Compile and cache
  └─────────────────┘
```

### What Inductor Fuses

```python
# This entire sequence becomes ONE Triton kernel:
def f(x):
    y = x * 2          # pointwise
    y = y + 1           # pointwise
    y = torch.relu(y)   # pointwise
    y = y.to(torch.float16)  # pointwise
    return y

# After torch.compile, Inductor generates:
# @triton.jit
# def fused_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
#     xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
#     x0 = tl.load(in_ptr0 + xindex)       # one load
#     x1 = x0 * 2.0                         # fused mul
#     x2 = x1 + 1.0                         # fused add
#     x3 = tl.maximum(x2, 0.0)              # fused relu
#     tl.store(out_ptr0 + xindex, x3.to(tl.float16))  # one store
```

4 separate kernel launches → 1 fused kernel. Memory traffic cut by 4×.

---

## 7. Debugging with TORCH_LOGS

The `TORCH_LOGS` environment variable is your window into the pipeline:

```bash
# See everything (very verbose)
TORCH_LOGS="all" python script.py

# See specific stages:
TORCH_LOGS="dynamo" python script.py         # Graph capture details
TORCH_LOGS="aot" python script.py            # AOTAutograd graphs
TORCH_LOGS="inductor" python script.py       # Inductor decisions
TORCH_LOGS="output_code" python script.py    # Generated Triton/C++ code
TORCH_LOGS="graph_breaks" python script.py   # Where graph breaks occur
TORCH_LOGS="guards" python script.py         # Guard conditions
TORCH_LOGS="recompiles" python script.py     # Recompilation events

# Combine multiple:
TORCH_LOGS="graph_breaks,output_code" python script.py
```

### Practical Debugging Workflow

```python
import torch
import torch._dynamo as dynamo

# 1. Check for graph breaks
@torch.compile(fullgraph=True)  # Will error on breaks
def model(x):
    ...

# 2. Inspect the captured graph
def my_backend(gm: torch.fx.GraphModule, example_inputs):
    print("Captured graph:")
    gm.print_readable()
    return gm  # Return unmodified for inspection

@torch.compile(backend=my_backend)
def f(x):
    return x.sin() + x.cos()

f(torch.randn(4, device='cuda'))

# 3. View generated code
torch._inductor.config.debug = True
# Code will be written to /tmp/torchinductor_<user>/

# 4. Count graph breaks
dynamo.reset()
explanation = dynamo.explain(f)(torch.randn(4))
print(explanation)
```

### Reading `explain()` Output

```python
explanation = torch._dynamo.explain(model)(x)
print(explanation)

# Output:
# Graph count: 3              ← 3 subgraphs (2 graph breaks)
# Graph break reasons:
#   1. print() at line 15     ← first break
#   2. unsupported: sorted()  ← second break
# Guard count: 12
# Shape guard count: 4
# Op count: 47
```

---

## 8. Dynamic Shapes

By default, `torch.compile` creates guards on exact shapes. This causes recompilation when shapes change:

```python
# Static shapes (default): recompiles for each new shape
@torch.compile
def f(x):
    return x.sum()

f(torch.randn(32, 64))   # Compile for (32, 64)
f(torch.randn(64, 128))  # Recompile for (64, 128)!

# Dynamic shapes: uses symbolic integers
@torch.compile(dynamic=True)
def f(x):
    return x.sum()

f(torch.randn(32, 64))   # Compile with symbolic s0, s1
f(torch.randn(64, 128))  # Cache HIT — same symbolic graph
```

### How Symbolic Shapes Work

```
Static mode:
  Guard: x.shape[0] == 32 AND x.shape[1] == 64

Dynamic mode:
  Guard: x.shape[0] >= 2  (lower bound only)
  Graph uses: s0 = x.size(0), s1 = x.size(1)
  All operations use symbolic ints: grid = cdiv(s0 * s1, BLOCK)
```

---

## Hands-On Exercises

### Exercise 1: Pipeline Walkthrough (30 min)

Trace a simple model through all three stages:

```python
import torch

class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 32)
    
    def forward(self, x):
        x = self.linear(x)
        x = torch.relu(x)
        x = x * 0.5
        return x

model = TinyModel().cuda()
x = torch.randn(8, 64, device='cuda')

# TODO:
# 1. Use dynamo.explain() to inspect graph capture
# 2. Use a custom backend to print the FX graph
# 3. Set TORCH_LOGS="output_code" and inspect the generated Triton
# 4. Find the generated code in /tmp/torchinductor_*/
```

### Exercise 2: Fix Graph Breaks (20 min)

This function has three graph breaks. Find and fix them:

```python
@torch.compile(fullgraph=True)
def broken_model(x, training=True):
    y = x.sin()
    print(f"Input norm: {x.norm()}")         # Break 1
    if x.sum() > 0:                          # Break 2
        y = y * 2
    results = sorted([y, y.cos()], key=lambda t: t.sum())  # Break 3
    return results[0]
```

### Exercise 3: Dynamic Shapes Experiment (20 min)

```python
import time

@torch.compile
def static_fn(x):
    return x.layer_norm([x.shape[-1]])

@torch.compile(dynamic=True)
def dynamic_fn(x):
    return x.layer_norm([x.shape[-1]])

# Measure compilation time for 10 different shapes
shapes = [(2**i, 256) for i in range(4, 14)]
# TODO: time each call, count recompilations
# Which mode is faster overall?
```

---

## Key Takeaways

1. **Three-stage pipeline** — Dynamo captures the graph (bytecode analysis), AOTAutograd splits forward/backward, Inductor generates Triton/C++ code.
2. **Guards control caching** — compiled code is cached with guard conditions; when guards fail, recompilation occurs. This is the #1 cause of unexpected slowness.
3. **Graph breaks split subgraphs** — `print()`, data-dependent control flow, and unsupported ops cause graph breaks that prevent fusion opportunities. Use `fullgraph=True` to catch them.
4. **Decomposition enables fusion** — AOTAutograd decomposes `gelu`, `layer_norm`, etc. into primitives that Inductor can fuse into single kernels.
5. **`TORCH_LOGS` is essential** — `output_code`, `graph_breaks`, `guards`, and `recompiles` are the four most useful log channels for debugging.
6. **Dynamic shapes reduce recompilation** — `torch.compile(dynamic=True)` uses symbolic integers to generate shape-generic code at the cost of slightly more complex guards.

---

## Further Reading

- **TorchDynamo deep dive:** Jason Ansel's talk at PyTorch Conference 2022
- **AOTAutograd design doc:** [pytorch.org/docs/main/torch.compiler_aot_autograd.html](https://pytorch.org/docs/main/torch.compiler_aot_autograd.html)
- **Inductor architecture:** [dev-discuss.pytorch.org — TorchInductor overview](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-target-backends/747)
- **TORCH_LOGS reference:** [pytorch.org/docs/main/logging.html](https://pytorch.org/docs/main/logging.html)
- **torch.compile troubleshooting:** [pytorch.org/docs/main/torch.compiler_troubleshooting.html](https://pytorch.org/docs/main/torch.compiler_troubleshooting.html)

---

## Tomorrow

**Day 26** goes one level deeper into **TorchInductor's code generation**. We'll study how Inductor's scheduler decides which ops to fuse, how it generates Triton kernel source code, and how the wrapper code orchestrates kernel launches. We'll read real Inductor-generated code and learn to modify the codegen to add custom optimizations.
