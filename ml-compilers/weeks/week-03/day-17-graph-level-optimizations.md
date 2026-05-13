# Day 17: Graph-Level Optimizations

> **Phase II · Week 3 · Day 17 of 70 · 2.5 hours**

*"The fastest operation is the one you never execute — and the second fastest is the one you fuse into another."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 16: Computation Graphs as IR](day-16-computation-graphs-ir.md) | [Day 18: The Polyhedral Model](day-18-polyhedral-model.md) | [Week 3: IRs & Passes](README.md) | [Phase II: Compiler Fundamentals](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

A raw computation graph from `torch.trace` or `jit.script` is **naively correct but naively slow**. Every node becomes a separate kernel launch on GPU (~5–10µs overhead each), every intermediate tensor gets allocated and freed, and redundant computations waste both time and memory. Graph-level optimization passes transform this graph into an equivalent but dramatically faster one — often reducing kernel count by 3–10× and memory by 2–5×. These passes are the "middle-end" of the ML compiler, and they operate purely on the dataflow structure without knowing anything about hardware.

---

## 1. Constant Folding

**Idea**: if all inputs to a node are known at compile time, evaluate the node during compilation and replace it with the result.

```
  Before:                          After:
  ┌───┐                           ┌───┐
  │ 2 │──┐                        │ 6 │──▶ [mul] ──▶ output
  ├───┤   ▼                       └───┘      ▲
  │ 3 │──▶ [mul] ──▶ [add] ──▶ output        │
  └───┘              ▲                        x
                     │
                     x

  2 * 3 = 6 (computed at compile time)
  Graph shrinks: 2 nodes → 1 node
```

In FX:

```python
import torch
from torch.fx import symbolic_trace, Node

class WithConstants(torch.nn.Module):
    def forward(self, x):
        scale = torch.tensor(2.0) * torch.tensor(3.0)  # constant!
        return x * scale

traced = symbolic_trace(WithConstants())
print("Before:", len([n for n in traced.graph.nodes if n.op == 'call_function']))

# Constant folding pass
from torch.fx.passes.tools_common import legalize_graph
from torch.fx.experimental.const_fold import split_const_subgraphs

const_split = split_const_subgraphs(traced)
print("After split — constant subgraphs extracted for precomputation")
```

### When It Applies in ML

- **Fixed input shapes** baked into shape computation nodes
- **Weight preprocessing**: `weight.T` computed once at compile time
- **Scale factors**: `1.0 / sqrt(d_model)` in attention

---

## 2. Dead Code Elimination (DCE)

**Idea**: if a node's output is never used by any other node or the graph output, remove it.

```
  Before:                          After:
  x ──▶ [relu] ──▶ [log] ──▶ out  x ──▶ [relu] ──▶ out
           │
           └──▶ [exp] (dead!)      [exp] removed — no users
```

In FX:

```python
import torch
from torch.fx import symbolic_trace

class HasDeadCode(torch.nn.Module):
    def forward(self, x):
        a = torch.relu(x)
        b = torch.exp(x)      # computed but never used!
        c = torch.log(a)
        return c

traced = symbolic_trace(HasDeadCode())
print("Before DCE:")
traced.graph.print_tabular()

# FX built-in DCE
traced.graph.eliminate_dead_code()
traced.recompile()

print("\nAfter DCE:")
traced.graph.print_tabular()
# `exp` node is gone!
```

DCE is cheap (single pass, $O(N)$) and always safe. Run it after every other optimization pass to clean up orphaned nodes.

---

## 3. Common Subexpression Elimination (CSE)

**Idea**: if two nodes compute the same operation on the same inputs, keep one and redirect all users to it.

```
  Before:                              After:
  x ──▶ [relu] ──▶ [mul] ──▶ a        x ──▶ [relu] ──┬──▶ [mul] ──▶ a
  x ──▶ [relu] ──▶ [add] ──▶ b                        └──▶ [add] ──▶ b
  (relu computed twice!)               (relu computed once, shared)
```

```python
import torch
from torch.fx import symbolic_trace

class HasCSE(torch.nn.Module):
    def forward(self, x):
        a = torch.relu(x)
        b = torch.relu(x)  # redundant!
        return a * 2 + b * 3

traced = symbolic_trace(HasCSE())

# Manual CSE pass
def cse_pass(graph):
    seen = {}  # (op, target, args) → node
    replacements = 0
    for node in graph.nodes:
        if node.op == 'call_function':
            key = (node.target, tuple(id(a) for a in node.args))
            if key in seen:
                node.replace_all_uses_with(seen[key])
                replacements += 1
            else:
                seen[key] = node
    graph.eliminate_dead_code()
    return replacements

n = cse_pass(traced.graph)
traced.recompile()
print(f"Eliminated {n} redundant nodes")
```

### CSE in Practice

CSE catches patterns like:
- Repeated `x.shape[0]` lookups
- Identical normalization denominators: $\frac{1}{\sqrt{d}}$ computed multiple times in multi-head attention
- Gradient checkpointing re-computations that overlap with forward

---

## 4. Operator Fusion

**The single most important optimization in ML compilers.**

Without fusion, a chain of elementwise ops generates one kernel launch per op, with intermediate tensors written to and read from global memory:

```
  Unfused (3 kernel launches, 2 intermediate tensors):
  ┌──────────┐    ┌──────────┐    ┌──────────┐
  │ mul      │──▶ │ add      │──▶ │ relu     │
  │ DRAM→reg │    │ DRAM→reg │    │ DRAM→reg │
  │ reg→DRAM │    │ reg→DRAM │    │ reg→DRAM │
  └──────────┘    └──────────┘    └──────────┘
     ~10µs           ~10µs           ~10µs      = 30µs + memory

  Fused (1 kernel launch, 0 intermediate tensors):
  ┌──────────────────────────────────────┐
  │ mul → add → relu (all in registers) │
  │ DRAM→reg               reg→DRAM     │
  └──────────────────────────────────────┘
     ~12µs                                = 12µs, 2.5x faster
```

### Fusion Categories

| Pattern | Example | Savings |
|:--------|:--------|:--------|
| **Elementwise chain** | `relu(x * 2 + 1)` | Eliminate intermediates |
| **Reduction + elementwise** | `softmax = exp(x) / sum(exp(x))` | One pass over data |
| **MatMul + bias + activation** | `relu(Wx + b)` | Vendor-fused GEMM (cuBLAS) |
| **Attention pattern** | Q @ K.T / √d → softmax → @ V | FlashAttention kernel |

### Implementing a Fusion Pass in FX

```python
import torch
from torch.fx import symbolic_trace, Node
import operator

class FusableModel(torch.nn.Module):
    def forward(self, x):
        a = x * 2.0
        b = a + 1.0
        c = torch.relu(b)
        return c

traced = symbolic_trace(FusableModel())

def find_fusable_chains(graph):
    """Find chains of elementwise ops that can be fused."""
    ELEMENTWISE = {torch.relu, torch.sigmoid, torch.tanh,
                   operator.mul, operator.add, operator.sub}
    chains = []
    visited = set()
    
    for node in graph.nodes:
        if node in visited or node.op != 'call_function':
            continue
        if node.target not in ELEMENTWISE:
            continue
        
        # Walk forward collecting the chain
        chain = [node]
        visited.add(node)
        current = node
        
        while True:
            users = list(current.users)
            if len(users) != 1:
                break
            nxt = users[0]
            if nxt.op != 'call_function' or nxt.target not in ELEMENTWISE:
                break
            chain.append(nxt)
            visited.add(nxt)
            current = nxt
        
        if len(chain) > 1:
            chains.append(chain)
    
    return chains

chains = find_fusable_chains(traced.graph)
for i, chain in enumerate(chains):
    ops = [n.target.__name__ if hasattr(n.target, '__name__') else str(n.target)
           for n in chain]
    print(f"Fusable chain {i}: {' → '.join(ops)}")
```

---

## 5. Layout Transformation (NCHW → NHWC)

Memory layout determines cache efficiency. For convolutions:

```
  NCHW (PyTorch default):        NHWC (cuDNN preferred):
  Memory order:                   Memory order:
  [batch][channel][height][width] [batch][height][width][channel]

  For 3×3 conv accessing a        For 3×3 conv accessing a
  spatial window:                  spatial window:
  ┌─C0─┐ ┌─C1─┐ ┌─C2─┐          ┌──H0,W0──┐
  │ ... │ │ ... │ │ ... │          │ C0 C1 C2│ ← contiguous!
  └─────┘ └─────┘ └─────┘          ├──H0,W1──┤
  Channels scattered in memory     │ C0 C1 C2│
  → poor spatial locality           └─────────┘
                                   Channels are contiguous
                                   → vector-friendly
```

The layout transform pass inserts transpose operations at graph boundaries and propagates the preferred layout through the graph:

```python
def layout_transform_pass(graph, preferred='channels_last'):
    """Insert layout conversions for conv2d operations."""
    for node in graph.nodes:
        if node.target == torch.nn.functional.conv2d:
            # Insert: input = input.to(memory_format=channels_last)
            # Insert: weight = weight.to(memory_format=channels_last)
            # After conv: output = output.to(memory_format=contiguous)
            pass  # Actual implementation modifies graph nodes
```

Performance impact: NHWC can be **1.3–2× faster** for convolutions on NVIDIA Tensor Cores.

---

## 6. Algebraic Simplification

Replace expensive operations with mathematically equivalent cheaper ones:

| Before | After | Rule |
|:-------|:------|:-----|
| `x * 1.0` | `x` | Multiplicative identity |
| `x + 0.0` | `x` | Additive identity |
| `x * 0.0` | `zeros_like(x)` | Zero product |
| `x ** 2` | `x * x` | Strength reduction |
| `exp(log(x))` | `x` | Inverse functions |
| `x / x` | `ones_like(x)` | Self-division (if `x ≠ 0`) |
| `relu(relu(x))` | `relu(x)` | Idempotent function |
| `x.T.T` | `x` | Double transpose |

```python
def algebraic_simplify(graph):
    """Apply algebraic simplification rules."""
    simplified = 0
    for node in list(graph.nodes):
        if node.op != 'call_function':
            continue
        
        # Rule: relu(relu(x)) → relu(x)
        if node.target == torch.relu:
            arg = node.args[0]
            if isinstance(arg, Node) and arg.target == torch.relu:
                node.replace_all_uses_with(arg)
                simplified += 1
        
        # Rule: x * 1.0 → x
        if node.target == operator.mul:
            for i, arg in enumerate(node.args):
                if isinstance(arg, (int, float)) and arg == 1.0:
                    other = node.args[1 - i]
                    node.replace_all_uses_with(other)
                    simplified += 1
                    break
    
    graph.eliminate_dead_code()
    return simplified
```

---

## Hands-On Exercises

### Exercise 1: Build a Complete Optimization Pipeline (30 min)

```python
import torch
from torch.fx import symbolic_trace
import operator

class UnoptimizedModel(torch.nn.Module):
    def forward(self, x):
        # Constant folding target
        scale = 2.0 * 3.0
        
        # CSE target
        a = torch.relu(x)
        b = torch.relu(x)  # redundant
        
        # Algebraic simplification target
        c = a * 1.0         # identity
        
        # Dead code target
        d = torch.exp(x)    # unused
        
        # Fusion target
        e = b * scale
        f = e + 1.0
        g = torch.relu(f)
        
        return c + g

traced = symbolic_trace(UnoptimizedModel())
print(f"Before: {len(list(traced.graph.nodes))} nodes")

# TODO: Apply in order:
# 1. Constant folding
# 2. CSE
# 3. Algebraic simplification
# 4. DCE (cleanup)
# Count nodes after each pass
```

### Exercise 2: Measure Fusion Impact (25 min)

```python
import torch
import time

def unfused(x):
    x = x * 2.0
    x = x + 1.0
    x = torch.relu(x)
    x = x * 0.5
    x = torch.sigmoid(x)
    return x

compiled = torch.compile(unfused, mode="reduce-overhead")

x = torch.randn(4096, 4096, device='cuda')

# Warm up
for _ in range(20):
    unfused(x)
    compiled(x)
torch.cuda.synchronize()

# Benchmark
def bench(fn, x, n=200):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000

print(f"Eager:    {bench(unfused, x):.3f} ms  (5 kernel launches)")
print(f"Compiled: {bench(compiled, x):.3f} ms  (1 fused kernel)")
```

### Exercise 3: Write a Pattern-Matching Pass (20 min)

```python
import torch
from torch.fx import symbolic_trace

class GeluApprox(torch.nn.Module):
    """GELU written as individual ops — ripe for pattern matching."""
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            0.7978845608 * (x + 0.044715 * x * x * x)))

traced = symbolic_trace(GeluApprox())

# Task: Write an FX pass that detects this pattern
# and replaces it with a single torch.nn.functional.gelu call.
#
# Hints:
# 1. Look for the tanh node
# 2. Check if its input matches 0.7978... * (x + 0.044... * x^3)
# 3. If so, replace the entire subgraph with F.gelu(x)
```

---

## Key Takeaways

1. **Constant folding** eliminates compile-time-known computations — shapes, scales, dtype conversions
2. **DCE** is cheap ($O(N)$) and should run after every pass — it cleans up nodes orphaned by other transforms
3. **CSE** finds and deduplicates identical computations — especially valuable in attention and normalization patterns
4. **Operator fusion** is the single most impactful optimization — it eliminates kernel launch overhead and intermediate memory traffic
5. **Layout transforms** (NCHW→NHWC) can unlock 1.3–2× speedups on Tensor Cores with zero algorithm changes
6. **Algebraic simplification** catches identity ops, inverse functions, and strength-reducible patterns

---

## Further Reading

- Jia, Z. et al. — *TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph Substitutions* (SOSP 2019)
- Niu, W. et al. — *DNNFusion: Accelerating Deep Neural Networks Execution with Advanced Operator Fusion* (PLDI 2021)
- PyTorch docs — [FX Graph Transformations](https://pytorch.org/docs/stable/fx.html)
- Zheng, L. et al. — *Ansor: Generating High-Performance Tensor Programs for Deep Learning* (OSDI 2020)

---

## Tomorrow's Preview

**Day 18** goes deeper — literally into the **loop nests** inside each operator. The **polyhedral model** gives us a mathematical framework for reasoning about iteration spaces, data dependencies, and legal loop transformations. We'll see how TVM's schedule primitives (`split`, `reorder`, `tile`) map to affine transformations in a polyhedral framework.
