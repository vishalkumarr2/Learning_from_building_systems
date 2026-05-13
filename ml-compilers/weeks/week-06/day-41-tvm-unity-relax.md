# Day 41: TVM Unity & Relax

> **Phase III · Week 6 · Day 41 of 70 · 2.5 hours**

*"The best IRs aren't designed to be permanent — they're designed to be composable, so the next one can build on what came before."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 40: TVM for Edge Devices](day-40-tvm-for-edge-devices.md) | [Day 42: Stop & Reflect #3](day-42-stop-and-reflect-3.md) | [Week 6: TVM Tuning & Backends](README.md) | [Phase III: Apache TVM Deep Dive](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

TVM's original design split the compiler into distinct layers — Relay for graph-level IR, TE for compute declarations, TIR for low-level loops — each with its own abstraction and tooling. This served well, but created friction: Relay can't express dynamic shapes naturally, TE schedules can't compose with TIR transforms, and adding a new abstraction layer requires threading it through every pass. **TVM Unity** is the initiative to unify these layers into a single, composable framework. **Relax** is the new graph-level IR born from this effort, designed from scratch with first-class dynamic shapes, dataflow blocks, and direct interoperability with TIR. Understanding Unity and Relax shows you where TVM (and the ML compiler field) is headed.

---

## 1. The Problem with the Current Stack

### Relay's Pain Points

Relay was a breakthrough when introduced (2018), but the ML landscape has evolved:

```
Current TVM Stack — Layered but Rigid
═══════════════════════════════════════

   ┌──────────────────────────────────┐
   │         Relay (Graph IR)          │  ← FP types, static shapes, 
   │  • Functional, pure              │     limited dynamism
   │  • Type-checked, shape-inferred  │
   └──────────┬───────────────────────┘
              │ Lower (one-way door)
   ┌──────────▼───────────────────────┐
   │     TE (Tensor Expression)        │  ← Separate language, 
   │  • Compute + Schedule             │     can't inspect Relay context
   │  • Generates TIR                  │
   └──────────┬───────────────────────┘
              │ Compile
   ┌──────────▼───────────────────────┐
   │        TIR (Low-Level IR)         │  ← No way to "look up" at Relay
   │  • Loops, buffers, intrinsics     │     graph structure
   │  • Target-specific transforms     │
   └──────────────────────────────────┘
   
   Problems:
   ✗ Dynamic shapes require workarounds (Any, shape_func)
   ✗ TE schedules are isolated from graph context
   ✗ Cross-layer optimization is hard (e.g., layout + tiling)
   ✗ Adding a new abstraction = rewriting passes
```

### Concrete Limitations

| Issue | Relay Behavior | Real-World Need |
|:--|:--|:--|
| Dynamic batch | Requires `Any` + `shape_func` hacks | LLMs: batch size varies per request |
| Dynamic sequence length | Relay can't reason about it natively | Transformers: seq_len is fundamental |
| Cross-layer optimization | TE can't see Relay graph | Layout + tiling should co-optimize |
| Custom operator fusion | Must write Relay fusion patterns | Want TIR-level custom fusion rules |
| Debugging | Relay and TIR have separate debugging | Need unified tracing across layers |

---

## 2. TVM Unity Vision

### The Core Insight

Instead of *separate* IRs that communicate through lowering, Unity proposes a **single IR ecosystem** where different abstraction levels coexist and compose:

```
TVM Unity — Composable Abstractions
════════════════════════════════════

   ┌──────────────────────────────────────────────┐
   │            Relax (Graph + Dataflow)            │
   │                                                │
   │  R.call_tir(                                   │
   │    tir_matmul,     ← Direct TIR reference      │
   │    [A, B], out     ← Explicit memory            │
   │  )                                              │
   │                                                │
   │  ┌──────────────────────────────────────┐      │
   │  │         TIR (Low-Level)               │      │
   │  │  @T.prim_func                         │      │
   │  │  def tir_matmul(A, B, C):             │      │
   │  │    for i, j, k in T.grid(M, N, K):    │      │
   │  │      C[i, j] += A[i, k] * B[k, j]    │      │
   │  └──────────────────────────────────────┘      │
   │                                                │
   │  Key: Relax and TIR live in the SAME module    │
   │  and can be transformed TOGETHER               │
   └──────────────────────────────────────────────┘
```

### Unity Principles

1. **Co-existence** — Relax (high-level) and TIR (low-level) functions live in the same `IRModule`
2. **Interoperability** — Relax can call TIR functions directly via `call_tir`
3. **Incremental lowering** — Lower one function at a time, not all-or-nothing
4. **First-class dynamic shapes** — Shape expressions are part of the type system
5. **TVMScript everywhere** — A single Python-based syntax for Relax, TIR, and transforms

---

## 3. Relax: The Next-Generation IR

### Key Design Differences from Relay

| Feature | Relay | Relax |
|:--|:--|:--|
| Shape representation | Inferred statically, `Any` for dynamic | First-class symbolic shapes (`m`, `n`) |
| Execution model | Pure functional (copy semantics) | Dataflow blocks (explicit mutation regions) |
| Memory management | Implicit (compiler decides) | Explicit allocation via `R.builtin.alloc_tensor` |
| TIR integration | Lowering pass (one-way) | `call_tir` (direct reference, composable) |
| Syntax | Relay text format | TVMScript (Python-native) |
| Custom ops | Register externally | Inline TIR or `call_packed` |

### Relax Program Structure

```python
from tvm.script import relax as R, tir as T
import tvm

@tvm.script.ir_module
class MyModule:
    # TIR function: low-level matmul kernel
    @T.prim_func
    def matmul(
        A: T.Buffer((T.int64(128), T.int64(256)), "float32"),
        B: T.Buffer((T.int64(256), T.int64(512)), "float32"),
        C: T.Buffer((T.int64(128), T.int64(512)), "float32"),
    ):
        for i, j, k in T.grid(128, 256, 512):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] += A[vi, vk] * B[vk, vj]
    
    # Relax function: high-level graph
    @R.function
    def main(
        x: R.Tensor((128, 256), dtype="float32"),
        w: R.Tensor((256, 512), dtype="float32"),
    ) -> R.Tensor((128, 512), dtype="float32"):
        with R.dataflow():
            # Allocate output buffer explicitly
            out = R.call_tir(
                MyModule.matmul,  # Direct TIR reference
                [x, w],
                out_sinfo=R.Tensor((128, 512), dtype="float32"),
            )
            y = R.nn.relu(out)
            R.output(y)
        return y
```

### Dataflow Blocks

Relax introduces **dataflow blocks** to delineate where the compiler can freely reorder and optimize:

```python
@R.function
def transformer_layer(x: R.Tensor, w_q: R.Tensor, w_k: R.Tensor,
                      w_v: R.Tensor, w_o: R.Tensor):
    with R.dataflow():
        # Inside a dataflow block:
        # ✓ Compiler can reorder operations
        # ✓ Compiler can fuse operations
        # ✓ No side effects allowed
        q = R.matmul(x, w_q)
        k = R.matmul(x, w_k)
        v = R.matmul(x, w_v)
        attn = R.nn.attention(q, k, v)
        out = R.matmul(attn, w_o)
        R.output(out)
    
    # Outside dataflow block:
    # ✓ Side effects allowed (print, assert, control flow)
    # ✗ Compiler won't reorder across block boundaries
    R.print(out.shape)
    return out
```

---

## 4. First-Class Dynamic Shapes

### The Breakthrough

Relay's `Any` was a placeholder — the compiler couldn't reason about it. Relax makes shape variables **symbolic**:

```python
@R.function
def dynamic_matmul(
    x: R.Tensor(("m", "k"), dtype="float32"),  # m, k are symbolic
    w: R.Tensor(("k", "n"), dtype="float32"),   # k must match
) -> R.Tensor(("m", "n"), dtype="float32"):     # output shape is m×n
    with R.dataflow():
        out = R.matmul(x, w)
        R.output(out)
    return out
```

The shape algebra is part of the type system:

$$\text{matmul}: \mathbb{R}^{m \times k} \times \mathbb{R}^{k \times n} \rightarrow \mathbb{R}^{m \times n}$$

```
Shape inference in Relax:

  x: Tensor[m, k]   w: Tensor[k, n]
           \             /
            \           /
          matmul(x, w)
               │
               ▼
       result: Tensor[m, n]     ← symbolic, not "Any"!
               │
         reshape(result, (m*n,))
               │
               ▼
       flat: Tensor[m*n]        ← symbolic arithmetic works
```

### Why This Matters for LLMs

Large language models have dynamic shapes everywhere:

```python
@R.function
def llm_forward(
    tokens: R.Tensor(("batch", "seq_len"), dtype="int32"),
    kv_cache: R.Tensor(("batch", "num_heads", "past_len", "head_dim"), 
                        dtype="float32"),
):
    # Relax tracks: new_len = past_len + seq_len
    # This enables the compiler to:
    # 1. Pre-allocate KV cache correctly
    # 2. Generate specialized kernels for different seq_len
    # 3. Apply optimizations that depend on shape relationships
    ...
```

---

## 5. TVMScript: A Unified Syntax

TVMScript provides a single Python-like syntax for both Relax and TIR:

```python
from tvm.script import ir as I, relax as R, tir as T

@I.ir_module
class ConvReLUModule:
    """A module with both TIR kernels and Relax graph."""
    
    @T.prim_func(private=True)
    def conv2d_nchw(
        data: T.Buffer((1, 3, 224, 224), "float32"),
        weight: T.Buffer((16, 3, 3, 3), "float32"),
        output: T.Buffer((1, 16, 222, 222), "float32"),
    ):
        for n, c, h, w, rc, rh, rw in T.grid(1, 16, 222, 222, 3, 3, 3):
            with T.block("conv"):
                vn, vc, vh, vw = T.axis.remap("SSSS", [n, c, h, w])
                vrc, vrh, vrw = T.axis.remap("RRR", [rc, rh, rw])
                with T.init():
                    output[vn, vc, vh, vw] = T.float32(0)
                output[vn, vc, vh, vw] += (
                    data[vn, vrc, vh + vrh, vw + vrw] *
                    weight[vc, vrc, vrh, vrw]
                )
    
    @R.function
    def main(
        x: R.Tensor((1, 3, 224, 224), dtype="float32"),
        w: R.Tensor((16, 3, 3, 3), dtype="float32"),
    ) -> R.Tensor((1, 16, 222, 222), dtype="float32"):
        with R.dataflow():
            conv_out = R.call_tir(
                ConvReLUModule.conv2d_nchw,
                [x, w],
                out_sinfo=R.Tensor((1, 16, 222, 222), dtype="float32"),
            )
            relu_out = R.nn.relu(conv_out)
            R.output(relu_out)
        return relu_out
```

### Comparison: Relay vs Relax Syntax

```python
# ─── Relay (old) ─────────────────────────────────
relay_mod = tvm.IRModule()
x = relay.var("x", shape=(1, 784), dtype="float32")
w = relay.var("w", shape=(784, 10), dtype="float32")
y = relay.nn.dense(x, w)
y = relay.nn.softmax(y)
relay_mod["main"] = relay.Function([x, w], y)

# ─── Relax (new) ─────────────────────────────────
@I.ir_module
class RelaxMod:
    @R.function
    def main(
        x: R.Tensor((1, 784), "float32"),
        w: R.Tensor((784, 10), "float32"),
    ):
        with R.dataflow():
            y = R.matmul(x, w)
            z = R.nn.softmax(y)
            R.output(z)
        return z
```

---

## 6. Migration Path & Current Status

### Relay → Relax Migration

```
Migration Strategy
══════════════════

  Phase 1: Coexistence (current)
  ┌──────────────────────────────────┐
  │  Relay frontend importers still  │
  │  work. Convert Relay → Relax     │
  │  using relay_translator.         │
  └──────────────────────────────────┘
            │
  Phase 2: Relax-first (in progress)
  ┌──────────────────────────────────┐
  │  New features built on Relax.    │
  │  Relay maintained but frozen.    │
  │  New model importers target      │
  │  Relax directly.                 │
  └──────────────────────────────────┘
            │
  Phase 3: Relay deprecated (future)
  ┌──────────────────────────────────┐
  │  Relay kept for backward compat. │
  │  All active development on Relax │
  └──────────────────────────────────┘
```

### Converting Relay to Relax

```python
from tvm.relax.testing import relay_translator

# Existing Relay module
relay_mod, params = relay.frontend.from_pytorch(scripted_model, input_shapes)

# Convert to Relax
relax_mod = relay_translator.from_relay(
    relay_mod["main"],
    target=tvm.target.Target("llvm"),
    relay_params=params,
)

# Now you can use Relax passes and features
# including dynamic shapes, dataflow blocks, etc.
```

### What Works Today (as of 2026)

| Feature | Status | Notes |
|:--|:--|:--|
| TVMScript for Relax | ✅ Stable | Full syntax support |
| `call_tir` integration | ✅ Stable | Relax ↔ TIR interop |
| Dynamic shapes | ✅ Stable | Symbolic shape inference |
| Relay → Relax translator | ✅ Stable | Most models convert cleanly |
| MetaSchedule + Relax | ✅ Stable | Auto-tuning for Relax modules |
| PyTorch → Relax (direct) | ✅ Stable | Via `relax.frontend.from_pytorch` |
| ONNX → Relax (direct) | ✅ Stable | Via `relax.frontend.from_onnx` |
| LLM support (PagedKVCache) | ✅ Stable | Dynamic seq + paged attention |
| µTVM + Relax | 🔄 In progress | AOT for microcontrollers |

---

## Hands-On Exercises

### Exercise 1: Write a Relax Module (20 min)

```python
from tvm.script import ir as I, relax as R, tir as T
import tvm

# TODO: Create a Relax module with:
# 1. A TIR prim_func for element-wise add
# 2. A Relax function that calls it via call_tir
# 3. Build and run it

@I.ir_module
class AddModule:
    @T.prim_func(private=True)
    def elemwise_add(
        A: T.Buffer((128,), "float32"),
        B: T.Buffer((128,), "float32"),
        C: T.Buffer((128,), "float32"),
    ):
        for i in range(128):
            with T.block("add"):
                vi = T.axis.spatial(128, i)
                C[vi] = A[vi] + B[vi]
    
    @R.function
    def main(
        a: R.Tensor((128,), "float32"),
        b: R.Tensor((128,), "float32"),
    ) -> R.Tensor((128,), "float32"):
        with R.dataflow():
            c = R.call_tir(
                AddModule.elemwise_add, [a, b],
                out_sinfo=R.Tensor((128,), "float32"),
            )
            R.output(c)
        return c

# Build and test
target = tvm.target.Target("llvm")
mod = tvm.relax.transform.LegalizeOps()(AddModule)
ex = tvm.relax.build(AddModule, target)
vm = tvm.relax.VirtualMachine(ex, tvm.cpu())

import numpy as np
a = tvm.nd.array(np.ones(128, dtype="float32"))
b = tvm.nd.array(np.ones(128, dtype="float32") * 2)
result = vm["main"](a, b)
print(result.numpy())  # Should be all 3.0
```

### Exercise 2: Dynamic Shapes in Relax (20 min)

```python
# Create a module that handles dynamic batch size
@I.ir_module
class DynamicBatchModule:
    @R.function
    def main(
        x: R.Tensor(("batch", 784), "float32"),
        w: R.Tensor((784, 10), "float32"),
    ) -> R.Tensor(("batch", 10), "float32"):
        with R.dataflow():
            y = R.matmul(x, w)
            z = R.nn.softmax(y, axis=-1)
            R.output(z)
        return z

# TODO: Build this module, then test with batch=1, batch=4, batch=16
# Verify that the same compiled module handles all batch sizes
```

### Exercise 3: Compare Relay and Relax (20 min)

```python
# Take the MobileNetV2 model from Day 35
# 1. Import via relay.frontend.from_pytorch
# 2. Convert to Relax via relay_translator
# 3. Print both IRs — observe the structural differences
# 4. Compile both and verify identical outputs

# Bonus: Try adding dynamic batch to the Relax version
# (which would be painful or impossible in Relay)
```

---

## Key Takeaways

1. **TVM Unity** unifies Relay, TE, and TIR into a single composable framework — different abstraction levels coexist in one `IRModule`
2. **Relax** replaces Relay with first-class symbolic shapes, dataflow blocks, and explicit `call_tir` integration
3. **Dynamic shapes** are no longer a workaround — shape variables like `("batch", "seq_len")` are part of the type system and support symbolic arithmetic
4. **Dataflow blocks** mark regions where the compiler can freely optimize, while code outside blocks can have side effects
5. **TVMScript** provides a single Python-native syntax for both Relax and TIR, making the whole stack readable and writable as Python
6. **Migration** from Relay is incremental — existing Relay models can be translated to Relax, and both coexist during the transition

---

## Further Reading

- [TVM Unity: The Next Step in ML Compilation](https://tvm.apache.org/2021/12/15/tvm-unity) — vision blog post
- [Relax: TVM's Next-Gen Graph-Level IR (RFC)](https://github.com/apache/tvm-rfcs/blob/main/rfcs/0089-relax.md) — design rationale
- [MLC-LLM: LLM Compilation with Relax](https://mlc.ai/mlc-llm/) — real-world Relax usage for LLMs
- [TVMScript Documentation](https://tvm.apache.org/docs/reference/api/python/script.html) — syntax reference
- [Machine Learning Compilation (MLC) Course](https://mlc.ai/) — comprehensive course using Relax
- [Relax API Reference](https://tvm.apache.org/docs/reference/api/python/relax/index.html) — detailed API docs

---

## Tomorrow: Stop & Reflect #3

Day 42 is your third reflection checkpoint. You'll build a concept map linking everything from Weeks 5–6 (Relay → TE → TIR → tuning → BYOC → quantization → edge → Relax), take a 10-question self-check quiz, and develop a decision framework for choosing between TVM, `torch.compile`, Triton, and ONNX Runtime.
