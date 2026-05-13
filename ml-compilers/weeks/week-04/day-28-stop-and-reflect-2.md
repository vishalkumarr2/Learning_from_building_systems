# Day 28: Stop & Reflect #2

> **Phase II · Week 4 · Day 28 of 70 · 2.5 hours**

*"A compiler is a pipeline of lowering steps, each trading generality for specificity. If you can trace an operation from Python source to GPU instructions, you understand the whole stack."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 27: Custom Triton Backend](day-27-custom-triton-backend.md) | [Day 29: TVM Architecture Overview](../week-05/day-29-tvm-architecture-overview.md) | [Week 4: Triton & Kernel Engineering](README.md) | [Phase II: Compiler Fundamentals](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

This is the end of Phase II. Over the past two weeks, you've gone from Python-level graph capture to GPU-level Triton kernels. Before moving to TVM and the broader compiler ecosystem, we consolidate everything into a single mental model. The goal: you should be able to trace any PyTorch operation from `model(x)` through every compiler stage down to the Triton kernel that runs on hardware.

---

## 1. The Complete Map: Python → GPU

```
Layer 0: USER CODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  @torch.compile
  def model(x):
      return F.gelu(x @ W + b)

        │  torch.compile decorator
        ▼

Layer 1: TORCHDYNAMO (Graph Capture)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Intercepts Python bytecode via PEP 523
  • Traces into FX Graph (symbolic execution)
  • Inserts guards (dtype, shape, device, value)
  • Handles graph breaks (unsupported ops → subgraphs)
  
  Output: FX Graph with high-level ATen ops
          call_function  aten.mm.default     (x, W)
          call_function  aten.add.Tensor     (mm, b)
          call_function  aten.gelu.default   (add)

        │  Sent to AOTAutograd
        ▼

Layer 2: AOTAUTOGRAD (Differentiation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Splits graph into forward + backward
  • Decomposes high-level ops into primitives
    gelu → mul, erf, add, mul (5 primitive ops)
  • Produces joint forward/backward graph
  
  Output: Decomposed FX Graph (ATen primitives only)
          aten.mm, aten.add, aten.mul, aten.erf, ...

        │  Passed to backend (default: Inductor)
        ▼

Layer 3: TORCHINDUCTOR (Compilation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  3a. Lowering:   ATen ops → Inductor IR nodes
                  (Pointwise, Reduction, ExternKernel)
  
  3b. Fusion:     Group fusable nodes
                  pointwise + pointwise → 1 kernel
                  mm stays as ExternKernel (unfused)
  
  3c. Scheduling: Choose loop ordering, block sizes
                  Handle reduction strategies
  
  3d. Codegen:    Emit Triton source code (GPU)
                  or C++/OpenMP code (CPU)

  Output: triton_poi_fused_add_mul_erf_0.py
          + wrapper code orchestrating kernel launches

        │  Triton JIT compiler
        ▼

Layer 4: TRITON COMPILER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • Triton IR → Triton GPU IR → LLVM IR → PTX
  • Block-level ops → thread-level scheduling
  • Auto-manages shared memory, coalescing
  • Applies autotuning for block sizes

  Output: PTX assembly (→ cubin via ptxas)

        │  CUDA driver
        ▼

Layer 5: GPU HARDWARE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • SMs execute warps of 32 threads
  • L1/L2 cache hierarchy
  • HBM bandwidth: 1-3 TB/s
  • Compute: hundreds of TFLOPS
```

---

## 2. Concept Map: How Everything Connects

```
                    ┌───────────────────────┐
                    │   Python Source Code   │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │    FX Graph (ATen)     │◄── Day 15-17: FX framework
                    │  nodes, edges, ops    │    Day 18-19: passes & transforms
                    └───────────┬───────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                   │
    ┌─────────▼──────┐  ┌──────▼───────┐  ┌───────▼───────┐
    │   Dynamo       │  │  AOTAutograd  │  │ Graph Breaks  │
    │   (capture)    │  │  (grad split) │  │ (subgraphs)   │
    │   Day 25       │  │  Day 25       │  │ Day 25        │
    └─────────┬──────┘  └──────┬───────┘  └───────────────┘
              │                │
              └────────┬───────┘
                       │
             ┌─────────▼─────────┐
             │  Decomposed Graph  │◄── primitives only
             │  (ATen primitives) │    no gelu, layer_norm
             └─────────┬─────────┘
                       │
         ┌─────────────┼──────────────┐
         │             │              │
  ┌──────▼──────┐ ┌───▼────┐  ┌──────▼──────┐
  │  Inductor   │ │ Custom │  │  Other      │
  │  (default)  │ │Backend │  │  Backends   │
  │  Day 26     │ │Day 27  │  │  (TRT,etc.) │
  └──────┬──────┘ └───┬────┘  └─────────────┘
         │            │
         └─────┬──────┘
               │
    ┌──────────▼──────────┐
    │   Inductor IR       │◄── Pointwise, Reduction,
    │   (loop-level)      │    ExternKernel, Template
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │   Fusion Groups     │◄── minimize kernels & memory
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │   Triton Source      │◄── Day 22-24: Triton language
    │   (Python)           │    blocks, masks, tiling
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │   Triton Compiler    │◄── IR → LLVM → PTX
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │   GPU Execution      │◄── warps, SMs, HBM
    └─────────────────────┘
```

---

## 3. Key Mental Models

### Mental Model 1: The Lowering Ladder

Every compiler is a series of lowering steps. Each step trades generality for specificity:

```
Generality          ←──────────────────────→          Specificity
                                                        
  Python            FX Graph         Inductor IR       Triton        PTX
  ──────            ────────         ──────────        ──────        ───
  • Any code        • Tensor ops     • Loop nests      • Blocks      • Instructions
  • Dynamic         • Static shapes  • Memory layout   • Masks       • Registers
  • Objects         • No objects     • Fusion groups   • Tiling      • Warps
  • Control flow    • DAG only       • Scheduled       • JIT'd       • Native
```

**Key insight:** information is *lost* at each step (you can't reconstruct arbitrary Python from PTX), but the remaining information is enough for the next stage to make good decisions.

### Mental Model 2: The Fusion Funnel

```
     N separate operations
     ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐ ┌──┐
     │op│ │op│ │op│ │op│ │op│ │op│    6 ops, 6 kernel launches
     └──┘ └──┘ └──┘ └──┘ └──┘ └──┘    12 memory round-trips
                    │
                fusion
                    │
              ┌─────┴─────┐
              │ fused_kern │ │extern│    2 kernel launches
              │ (4 ops)    │ │(mm)  │    4 memory round-trips
              └────────────┘ └──────┘
```

Fusion is the single most impactful optimization. It reduces:
- Kernel launch overhead: $O(n) \to O(1)$
- Memory traffic: from $2n \times \text{tensor\_size}$ to $2 \times \text{tensor\_size}$
- The roofline shifts from memory-bound to compute-bound

### Mental Model 3: The Guard-Compile-Cache Loop

```
  First call:                    Second call (same shape):
  ┌─────────┐                    ┌─────────┐
  │ Dynamo  │──▶ trace ──▶ compile ──▶ cache     │ Dynamo  │──▶ check guards
  └─────────┘         (slow, ~1s)      │   hit   └─────────┘        │
                                       ▼                            ▼
                                   ┌───────┐                   ┌───────┐
                                   │cached │                   │cached │
                                   │kernel │                   │kernel │ ← fast!
                                   └───────┘                   └───────┘

  Different shape:
  ┌─────────┐
  │ Dynamo  │──▶ check guards ──▶ FAIL ──▶ recompile ──▶ new cache entry
  └─────────┘
```

---

## 4. Self-Check Quiz (10 Questions)

Test yourself before checking the answers. Write your answers first, then compare.

### Questions

**Q1:** What are the three stages of `torch.compile`'s pipeline, and what does each produce?

**Q2:** What is a "graph break" and why does it reduce optimization opportunities?

**Q3:** In this code, how many Triton kernels will Inductor generate?
```python
@torch.compile
def f(x, W):
    y = x @ W           # matmul
    y = y + 1            # add
    y = torch.relu(y)    # relu
    y = y * 0.5          # mul
    return y.sum()       # sum
```

**Q4:** Why does AOTAutograd decompose `torch.gelu()` into primitive ops?

**Q5:** What is the difference between a `Pointwise` kernel and a `Reduction` kernel in Inductor's codegen?

**Q6:** In Triton, what does `tl.program_id(0)` return and how does it relate to the CUDA concept of `blockIdx.x`?

**Q7:** You write a custom `torch.compile` backend. What is the function signature it must implement?

**Q8:** The fused kernel for `y = relu(x * 2 + 1)` loads from memory once and stores once. Without fusion, how many loads and stores would there be?

**Q9:** What does `torch._inductor.config.debug = True` do, and where does the output go?

**Q10:** When should you write a custom `torch.compile` backend instead of a custom lowering?

---

### Answers

<details>
<summary><strong>Click to reveal answers</strong></summary>

**A1:** (1) **TorchDynamo** — captures Python into an FX Graph of ATen ops. (2) **AOTAutograd** — splits into forward/backward graphs and decomposes high-level ops into primitives. (3) **TorchInductor** — lowers to Inductor IR, fuses, schedules, and generates Triton/C++ code.

**A2:** A graph break occurs when Dynamo encounters an operation it can't trace symbolically (e.g., `print()`, data-dependent `if`, unsupported Python). It splits the graph into subgraphs, each compiled separately. This prevents fusion across the break boundary and adds Python overhead between subgraphs.

**A3:** **Three kernels:** (1) ExternKernel for `mm` (calls cuBLAS), (2) Fused pointwise kernel for `add + relu + mul`, (3) Reduction kernel for `sum`. The `mm` acts as a fusion barrier — the pointwise ops after it fuse together, and `sum` is a different kernel type (reduction).

**A4:** So that Inductor can fuse the primitive ops with neighboring operations. Inductor doesn't need special-case code for `gelu` — it just sees `mul`, `erf`, `add`, `mul` and fuses them like any other pointwise chain. This composability is the key insight.

**A5:** A **Pointwise** kernel has a 1:1 mapping between input and output elements — each program instance processes a block of elements independently. A **Reduction** kernel collapses one or more dimensions, requiring an accumulation loop within each program instance (and potentially cross-block reduction for large dimensions).

**A6:** `tl.program_id(0)` returns the index of the current *program instance* in the grid's first dimension. It's semantically equivalent to CUDA's `blockIdx.x`. Each program instance processes one *block* of data (unlike CUDA where each thread processes one element).

**A7:** `def my_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable` — it receives the FX graph and concrete example inputs, and returns a callable that takes the same inputs and produces the outputs.

**A8:** Without fusion, three separate kernels:
- Kernel 1 (`x * 2`): 1 load + 1 store
- Kernel 2 (`+ 1`): 1 load + 1 store  
- Kernel 3 (`relu`): 1 load + 1 store
Total: **3 loads, 3 stores** (6 memory ops vs. 2 with fusion — a 3× reduction).

**A9:** It tells Inductor to write the generated Triton/C++ source code to disk (in `/tmp/torchinductor_$USER/`) and print additional debug information. You can inspect the actual generated kernels, wrapper code, and fusion decisions.

**A10:** Use a **custom lowering** when you want to change how a *single op* is translated to Inductor IR (it stays within Inductor's fusion and scheduling). Use a **custom backend** when you need to change *fusion decisions, scheduling, or code generation strategy* — things that require replacing Inductor's pipeline, not just one op within it.

</details>

---

## 5. Phase II Concept Connections

Map each concept to where it lives in the pipeline:

| Concept | Pipeline Stage | Key Insight |
|:--|:--|:--|
| FX Graph | Dynamo output / IR | DAG of tensor ops, no Python control flow |
| Graph Breaks | Dynamo | Split graph → fewer fusion opportunities |
| Guards | Dynamo | Control caching and recompilation |
| Decomposition | AOTAutograd | High-level → primitives enables fusion |
| Forward/Backward split | AOTAutograd | Autograd runs ahead-of-time, not at runtime |
| Lowering | Inductor Stage 1 | ATen ops → Inductor IR nodes |
| Fusion | Inductor Stage 2 | Multiple ops → single kernel |
| Scheduling | Inductor Stage 3 | Loop ordering, block sizes, reductions |
| Codegen | Inductor Stage 4 | IR → Triton source code |
| Block programming | Triton | Programs process blocks, not elements |
| Autotuning | Triton | Search over BLOCK_SIZE, num_warps, etc. |
| Tiling | Triton (matmul) | Fit sub-blocks into SRAM for data reuse |
| Flash Attention | Triton pattern | Online softmax + tiling = $O(N)$ memory |
| Custom backend | torch.compile API | Replace or augment Inductor |

---

## 6. "Ready for TVM" Checklist

Before moving to Phase III (TVM, MLIR, and the broader compiler ecosystem), verify you can confidently answer these:

```
□ I can explain the full torch.compile pipeline (3 stages + Triton)
□ I can read an FX graph and identify ops, shapes, and data flow
□ I can write an FX graph transformation pass
□ I can write a Triton kernel for a pointwise operation
□ I can write a Triton kernel for matrix multiplication (tiled)
□ I understand why tiling improves arithmetic intensity
□ I can explain how Flash Attention achieves O(N) memory
□ I know what Inductor's fusion decisions look like and why
□ I can read Inductor-generated Triton code
□ I can register a custom torch.compile backend
□ I understand guards, graph breaks, and recompilation
□ I know when to use a custom lowering vs. a custom backend
```

**If you checked 10+:** You're ready for TVM. Phase III will feel like a natural extension.

**If you checked 7–9:** Review the gaps. Re-read the specific day's material and redo one exercise.

**If you checked <7:** Spend today's remaining time on the weakest areas. Consider re-doing Day 22 (Triton basics) or Day 25 (torch.compile) exercises before moving on.

---

## 7. Reflection Prompts

Spend 10 minutes journaling on these. Writing forces clarity.

1. **What was the single most surprising thing you learned in Phase II?** Was it how much work `torch.compile` does behind a single decorator? How Triton manages to be competitive with CUDA? Something else?

2. **Which concept felt hardest to internalize?** Graph breaks and guards? Triton's block model? Inductor fusion rules? Identifying the hard parts tells you where to invest more time.

3. **Can you explain the full pipeline to a colleague?** Try it out loud: "When you call `torch.compile`, first Dynamo..." If you get stuck, that's the gap to fill.

4. **What would you build with these tools?** A custom kernel for your workload? A profiling backend? A specialized fusion pass? Having a project in mind makes Phase III more concrete.

5. **How does this connect to your broader goals?** Whether you're building ML infrastructure, optimizing models for production, or working on compiler research — where does this knowledge plug in?

---

## Phase II Summary: What You Learned

```
Week 3 (Days 15-21): Compiler Infrastructure
──────────────────────────────────────────────
  • FX Graph representation and tracing
  • Writing compiler passes (constant folding, DCE, pattern matching)
  • Operator fusion theory and practice
  • Shape propagation and type inference
  • Mini-project: building an FX optimization pass

Week 4 (Days 22-28): Triton & Kernel Engineering
──────────────────────────────────────────────────
  • Triton's block-based programming model
  • Matrix multiplication with tiling and SRAM reuse
  • Flash Attention algorithm and implementation
  • torch.compile internals (Dynamo → AOTAutograd → Inductor)
  • Inductor code generation and fusion
  • Writing custom torch.compile backends
  • This reflection (connecting it all together)
```

---

## Key Takeaways

1. **Compilers are lowering pipelines** — each stage translates a higher-level representation into a lower one, making decisions that shape performance.
2. **Fusion is king** — the single most impactful optimization in ML compilers. Everything else (tiling, scheduling, autotuning) is secondary.
3. **The FX graph is the lingua franca** — Dynamo, AOTAutograd, Inductor, and custom backends all communicate through FX graphs. Mastering graph manipulation is the highest-leverage skill.
4. **Triton is the GPU sweet spot** — 80-95% of CUDA performance, 3-5× less code, Python-native. This is why Inductor targets it.
5. **torch.compile is composable** — you don't have to replace the whole stack. Custom lowerings, graph transforms, and backend composition let you intervene at the right level.

---

## Further Reading

- **Phase II blog post:** Jason Ansel — "The PyTorch Compiler" (PyTorch Conference 2023)
- **Triton paper:** Tillet et al., "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations" (2019)
- **Flash Attention paper:** Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022)
- **Inductor design:** [dev-discuss.pytorch.org — TorchInductor overview](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-target-backends/747)
- **Compiler textbooks:** Cooper & Torczon, "Engineering a Compiler" (for classical foundations)

---

## Next: Phase III

**Day 29** begins Phase III with **TVM Architecture Overview**. TVM takes the compiler ideas we've learned — IR, passes, scheduling, code generation — and applies them to a broader set of hardware targets: CPUs, GPUs, FPGAs, custom accelerators. Where Inductor is PyTorch-specific, TVM is framework-agnostic and hardware-agnostic. The patterns you've learned transfer directly.
