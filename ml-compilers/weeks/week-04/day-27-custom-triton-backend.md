# Day 27: Writing a Custom Triton Backend

> **Phase II · Week 4 · Day 27 of 70 · 2.5 hours**

*"The backend API is torch.compile's escape hatch. When Inductor's fusion heuristics aren't right for your workload, you don't fight the compiler — you replace it."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 26: TorchInductor Codegen](day-26-torchInductor-codegen.md) | [Day 28: Stop & Reflect #2](day-28-stop-and-reflect-2.md) | [Week 4: Triton & Kernel Engineering](README.md) | [Phase II: Compiler Fundamentals](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

`torch.compile` defaults to Inductor, but the backend is pluggable. The backend API lets you intercept the FX graph *after* Dynamo + AOTAutograd and generate your own code — Triton, CUDA, C++, or even target a different accelerator entirely. This is how companies like OpenAI (Triton), NVIDIA (TensorRT), Intel (IPEX), and Hugging Face (Optimum) integrate their specialized compilers into the PyTorch ecosystem. Today you'll build a custom backend from scratch and understand when it's worth the effort.

---

## 1. The Backend API Contract

A `torch.compile` backend is a callable that takes an FX graph and example inputs, and returns a compiled function:

```
torch.compile(model, backend=my_backend)

Contract:
                                ┌─────────────────────────┐
  FX Graph (GraphModule)  ────▶ │   Your Backend          │
  Example Inputs (list)   ────▶ │   ──────────────        │ ────▶  Callable
                                │   Analyze graph          │
                                │   Generate code          │
                                │   Compile & return       │
                                └─────────────────────────┘

Signature:
  def my_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor])
      -> Callable
```

### What You Receive

The `gm` (GraphModule) contains:
- **`gm.graph`** — the FX graph with ATen-level ops (after decomposition)
- **`gm.code`** — Python source that executes the graph
- **`gm.forward()`** — callable that runs the graph eagerly

The `example_inputs` are concrete tensors with the shapes/dtypes that triggered compilation.

```python
def my_backend(gm, example_inputs):
    # gm.graph.print_tabular() shows:
    #   opcode    name    target                  args
    #   ─────────────────────────────────────────────────
    #   placeholder  arg0   arg0                   ()
    #   call_function  sin   aten.sin.default       (arg0,)
    #   call_function  cos   aten.cos.default       (arg0,)
    #   call_function  add   aten.add.Tensor        (sin, cos)
    #   output   output  output                    ((add,),)
    
    print(gm.graph)
    return gm.forward   # fallback: just run eagerly
```

---

## 2. Minimal Backend: Graph Printer

The simplest useful backend — prints every graph that Dynamo captures:

```python
import torch
from torch._dynamo import register_backend

@register_backend
def debug_backend(gm, example_inputs):
    """Print the graph and fall back to eager execution."""
    print("=" * 60)
    print("CAPTURED GRAPH:")
    print("=" * 60)
    gm.graph.print_tabular()
    print(f"\nInput shapes: {[x.shape for x in example_inputs]}")
    print(f"Input dtypes: {[x.dtype for x in example_inputs]}")
    print("=" * 60)
    
    # Return the GraphModule's forward — runs the ATen ops eagerly
    return gm.forward

# Use it
@torch.compile(backend="debug_backend")
def f(x):
    return x.sin() + x.cos()

x = torch.randn(8, 8, device="cuda")
result = f(x)
```

This is invaluable for understanding *what* Dynamo captures. Every graph break produces a separate call to your backend.

---

## 3. Pattern-Matching Backend

A more practical backend that detects specific patterns and replaces them with optimized kernels:

```python
import torch
import triton
import triton.language as tl
from torch._dynamo import register_backend

# Step 1: Write an optimized Triton kernel for SiLU
@triton.jit
def silu_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    # SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    result = x * tl.sigmoid(x)
    tl.store(out_ptr + offsets, result, mask=mask)


def run_silu_kernel(x):
    """Wrapper that launches the Triton kernel."""
    output = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    silu_kernel[grid](x, output, n, BLOCK_SIZE=1024)
    return output


# Step 2: Pattern detection
def find_silu_pattern(graph):
    """Find sigmoid(x) * x patterns in the FX graph."""
    matches = []
    for node in graph.nodes:
        if node.op != "call_function":
            continue
        # Pattern: mul(sigmoid(x), x) or mul(x, sigmoid(x))
        if node.target == torch.ops.aten.mul.Tensor:
            a, b = node.args
            if (hasattr(a, 'target') and 
                a.target == torch.ops.aten.sigmoid.default and
                a.args[0] is b):
                matches.append((node, b))  # (mul_node, input_x)
            elif (hasattr(b, 'target') and 
                  b.target == torch.ops.aten.sigmoid.default and
                  b.args[0] is a):
                matches.append((node, a))
    return matches


# Step 3: The backend
@register_backend
def silu_opt_backend(gm, example_inputs):
    """Replace SiLU patterns with a custom Triton kernel."""
    matches = find_silu_pattern(gm.graph)
    
    if not matches:
        # No pattern found — fall back to Inductor
        from torch._inductor.compile_fx import compile_fx
        return compile_fx(gm, example_inputs)
    
    print(f"[silu_opt] Found {len(matches)} SiLU pattern(s), replacing...")
    
    # For simplicity, handle single-pattern case
    # Production code would rewrite the graph properly
    def optimized_forward(*args):
        # Run the Triton kernel directly
        return run_silu_kernel(args[0])
    
    return optimized_forward


# Step 4: Test it
@torch.compile(backend="silu_opt_backend")
def model(x):
    return torch.sigmoid(x) * x  # this IS SiLU

x = torch.randn(4096, device="cuda")
result = model(x)
expected = torch.nn.functional.silu(x)
print(f"Correct: {torch.allclose(result, expected, atol=1e-5)}")
```

---

## 4. Proper Graph Rewriting

Real backends rewrite the FX graph instead of replacing `forward()`:

```python
import torch
from torch.fx import GraphModule, Graph, Node

def rewrite_silu(gm: GraphModule) -> GraphModule:
    """Rewrite SiLU pattern in-place in the FX graph."""
    graph = gm.graph
    
    for node in list(graph.nodes):  # copy list since we modify
        if node.op != "call_function":
            continue
        
        if node.target != torch.ops.aten.mul.Tensor:
            continue
        
        a, b = node.args
        
        # Check: mul(sigmoid(x), x)
        sigmoid_node, input_node = None, None
        if (isinstance(a, Node) and a.target == torch.ops.aten.sigmoid.default
            and a.args[0] is b):
            sigmoid_node, input_node = a, b
        elif (isinstance(b, Node) and b.target == torch.ops.aten.sigmoid.default
              and b.args[0] is a):
            sigmoid_node, input_node = b, a
        
        if sigmoid_node is None:
            continue
        
        # Replace the pattern with silu
        with graph.inserting_before(node):
            new_node = graph.call_function(
                torch.ops.aten.silu.default,
                args=(input_node,),
            )
        
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        
        # Remove dead sigmoid node if unused
        if len(sigmoid_node.users) == 0:
            graph.erase_node(sigmoid_node)
    
    graph.lint()  # validate graph integrity
    gm.recompile()
    return gm
```

### Graph Rewriting Safety Rules

```
DO:                                      DON'T:
─────────────────────────                ──────────────────────────
✓ Copy node list before iteration        ✗ Modify graph while iterating
✓ Call graph.lint() after changes         ✗ Leave dangling nodes
✓ Use replace_all_uses_with()             ✗ Manually reconnect edges
✓ Check node.users before erasing         ✗ Erase nodes with live users
✓ Call gm.recompile() when done           ✗ Forget to recompile
```

---

## 5. When to Customize vs. Use Inductor

```
                    Use Inductor                Write Custom Backend
                    ───────────                 ────────────────────
Pointwise fusion    ██████████  excellent        unnecessary
Reductions          ████████░░  good             rarely needed
GEMM epilogues      ██████░░░░  template-based   custom template wins
Custom hardware     ░░░░░░░░░░  GPU/CPU only     FPGA, TPU, custom ASIC
Custom dtypes       ██░░░░░░░░  limited          full control
Exotic patterns     ████░░░░░░  may not fuse     pattern-specific kernel
Cross-layer fusion  ██░░░░░░░░  limited          can fuse across layers
Quantization        ████░░░░░░  basic support    specialized backends win
```

### Decision Flowchart

```
Is Inductor producing optimal code for your pattern?
 │
 ├── YES → Use Inductor. Don't over-engineer.
 │
 └── NO → Can you fix it with a custom lowering? (Day 26)
      │
      ├── YES → Register lowering. Stays within Inductor.
      │
      └── NO → Does the issue require different scheduling/fusion?
           │
           ├── YES → Write a custom backend.
           │
           └── NO → File a PyTorch issue. Inductor improves fast.
```

---

## 6. Backend Composition: Chaining with Inductor

You don't have to replace Inductor entirely. A common pattern is pre-processing the graph and then handing off to Inductor:

```python
from torch._inductor.compile_fx import compile_fx

@register_backend
def preprocess_then_inductor(gm, example_inputs):
    """Apply custom graph transforms, then use Inductor for codegen."""
    
    # Step 1: Custom transforms
    gm = rewrite_silu(gm)           # our pattern rewrite
    gm = fold_constants(gm)          # constant folding
    gm = remove_dead_code(gm)        # DCE
    
    # Step 2: Hand off to Inductor for the heavy lifting
    return compile_fx(gm, example_inputs)
```

This is the recommended approach for most use cases:
- You handle **domain-specific patterns** that Inductor misses
- Inductor handles **fusion, scheduling, and code generation**

---

## 7. Testing & Debugging Custom Backends

### Correctness Testing

```python
import torch
from torch.testing import assert_close

def test_backend_correctness():
    """Golden-model testing: compare custom backend vs eager."""
    
    def model(x):
        return torch.sigmoid(x) * x + x.cos()
    
    x = torch.randn(256, 256, device="cuda", requires_grad=True)
    x_copy = x.clone().detach().requires_grad_(True)
    
    # Eager reference
    eager_out = model(x)
    eager_out.sum().backward()
    
    # Compiled with custom backend
    compiled = torch.compile(model, backend="silu_opt_backend")
    compiled_out = compiled(x_copy)
    compiled_out.sum().backward()
    
    # Compare forward
    assert_close(compiled_out, eager_out, atol=1e-5, rtol=1e-5)
    
    # Compare gradients
    assert_close(x_copy.grad, x.grad, atol=1e-5, rtol=1e-5)
    
    print("✓ Forward and backward match!")

test_backend_correctness()
```

### Debugging Tips

```python
# 1. Use TORCH_LOGS to see what Dynamo captures
# TORCH_LOGS="dynamo,aot" python script.py

# 2. Print the graph in your backend
def debug_backend(gm, example_inputs):
    print("Graph nodes:")
    for node in gm.graph.nodes:
        print(f"  {node.op:15s} {node.name:20s} target={node.target}")
    return gm.forward

# 3. Verify graph validity
gm.graph.lint()  # raises if graph is broken

# 4. Test with fullgraph to ensure no graph breaks
@torch.compile(backend="my_backend", fullgraph=True)
def f(x): ...

# 5. Check for dynamic shapes
print(f"Shapes: {[x.shape for x in example_inputs]}")
# If shapes are symbolic (torch.SymInt), handle gracefully
```

### Common Backend Bugs

| Bug | Symptom | Fix |
|:--|:--|:--|
| Missing output | `RuntimeError: output mismatch` | Ensure return matches graph output |
| Wrong dtype | Silent numerical errors | Check `example_inputs` dtypes |
| Dangling nodes | `graph.lint()` fails | Use `replace_all_uses_with` + erase |
| Gradient mismatch | Wrong backward values | Test with `requires_grad=True` |
| Shape assumptions | Fails on recompilation | Use symbolic shapes or `dynamic=True` |

---

## Hands-On Exercises

### Exercise 1: Build a Logging Backend (20 min)

```python
import torch

# TODO: Create a backend that:
# 1. Logs every graph it receives (ops, shapes, dtypes)
# 2. Counts total ops and fusion opportunities
# 3. Falls back to Inductor for actual execution
# 4. Prints a summary: "Graph: N ops, M fusable groups"

# Test with:
@torch.compile(backend="your_backend")
def model(x, y):
    a = x + y
    b = a * 2
    c = torch.relu(b)
    d = c.sum()
    return d

x = torch.randn(128, 128, device="cuda")
y = torch.randn(128, 128, device="cuda")
result = model(x, y)
```

### Exercise 2: Constant Folding Pass (25 min)

```python
import torch

# TODO: Write a backend that constant-folds scalar operations
# before handing to Inductor.
#
# Example: if the graph contains `mul(x, 2.0)` followed by 
# `mul(result, 3.0)`, rewrite to `mul(x, 6.0)`.
#
# Hints:
# - Walk the graph looking for chains of scalar multiplies
# - Create new constant nodes with graph.call_function()
# - Use replace_all_uses_with() to rewire

@torch.compile(backend="const_fold_backend")
def f(x):
    y = x * 2.0
    y = y * 3.0    # should fold to x * 6.0
    y = y + 0.0    # should eliminate (identity)
    return y
```

### Exercise 3: Pattern-Matching GELU Backend (30 min)

```python
# TODO: Build a backend that detects the GELU approximation pattern:
#   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
# and replaces it with torch.ops.aten.gelu.default
#
# Steps:
# 1. Write find_gelu_pattern() to detect the subgraph
# 2. Rewrite the graph to use a single GELU node
# 3. Hand off to Inductor
# 4. Verify numerical correctness

# This is hard! Start by printing the decomposed graph of:
@torch.compile(backend="debug_backend")
def approx_gelu(x):
    return 0.5 * x * (1 + torch.tanh(
        (2.0 / 3.14159265) ** 0.5 * (x + 0.044715 * x ** 3)
    ))
```

---

## Key Takeaways

1. **Backends are callables** — `(GraphModule, List[Tensor]) → Callable`. That's the entire API contract.
2. **Start with Inductor fallback** — Use `compile_fx(gm, example_inputs)` as your baseline and only override specific patterns.
3. **Graph rewriting is the core skill** — `replace_all_uses_with()`, `graph.erase_node()`, `graph.lint()`, and `gm.recompile()` are your primary tools.
4. **Pattern detection + replacement** — Most custom backends follow the same structure: find a pattern in the FX graph, replace it with an optimized implementation.
5. **Test forward AND backward** — Always verify gradients match, not just forward outputs.
6. **Composition over replacement** — Chain your transforms with Inductor rather than reimplementing all of code generation.

---

## Further Reading

- **Backend API docs:** [pytorch.org/docs/main/torch.compiler_custom_backends.html](https://pytorch.org/docs/main/torch.compiler_custom_backends.html)
- **Custom backends tutorial:** [pytorch.org/tutorials/intermediate/custom_backend.html](https://pytorch.org/tutorials/intermediate/custom_backend.html)
- **TensorRT backend example:** `torch_tensorrt` — real-world production backend
- **FX Graph manipulation:** [pytorch.org/docs/main/fx.html](https://pytorch.org/docs/main/fx.html)
- **Edward Yang's blog:** "Backends for torch.compile" (2023)

---

## Tomorrow

**Day 28** is our second **Stop & Reflect** session. We'll build a concept map connecting everything from Phase II: FX graphs → IR → compiler passes → scheduling → Triton → torch.compile → custom backends. A 10-question self-check quiz and "Ready for TVM" checklist will ensure you've internalized the foundations before we move to Phase III.
