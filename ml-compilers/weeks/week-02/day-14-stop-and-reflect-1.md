# Day 14: Stop & Reflect #1

> **Phase I · Week 2 · Day 14 of 70 · 2.5 hours**

*"You don't really understand something until you can explain it without opening your notes."*

---

| ← Previous | Next → | 📅 Week | �phase Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 13: Operator Fusion](day-13-operator-fusion-fundamentals.md) | [Day 15: Compiler 101 for ML](../week-03/day-15-compiler-101-for-ml.md) | [Week 2: PyTorch Internals](README.md) | Phase I: Foundations | [Curriculum Home](../../STUDY-PLAN.md) |

---

## Why This Matters

Two weeks and 14 days of dense material. Before we move into Phase II (Compiler Fundamentals), we need to **consolidate**. Research on learning shows that active recall, concept mapping, and identifying misconceptions produces 2–3× better retention than re-reading. This session is not optional downtime — it's where the material clicks into a coherent mental model.

---

## 1. Phase I Concept Map

Every concept from Days 1–13 connects. Trace the arrows to see how knowledge builds:

```
                        ┌─────────────────────────────┐
                        │    ML Systems & Compilers    │
                        │      WHY do we need them?    │
                        └──────────────┬──────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                      │
             ┌──────▼──────┐                        ┌──────▼──────┐
             │  Hardware   │                        │  Software   │
             │ Constraints │                        │  Execution  │
             └──────┬──────┘                        └──────┬──────┘
                    │                                      │
        ┌───────────┼───────────┐               ┌──────────┼──────────┐
        │           │           │               │          │          │
   ┌────▼───┐ ┌────▼────┐ ┌────▼────┐    ┌─────▼────┐ ┌──▼───┐ ┌───▼───┐
   │GPU Arch│ │ Memory  │ │Compute  │    │  Eager   │ │Graph │ │Fusion │
   │(Day1-3)│ │Hierarchy│ │ vs BW   │    │  Mode    │ │ Mode │ │(Day13)│
   └────┬───┘ │(Day4-5) │ │(Day6-7) │    │ (Day12) │ │(Day12│ └───┬───┘
        │     └────┬────┘ └────┬────┘    └─────┬────┘ └──┬───┘     │
        │          │           │               │         │          │
        │     ┌────▼────┐     │          ┌─────▼────┐    │          │
        │     │ HBM     │     │          │Dispatcher│    │          │
        │     │ L2/SRAM │     │          │ per-op   │    │          │
        │     │Registers│     │          │ overhead │    │          │
        │     └────┬────┘     │          └──────────┘    │          │
        │          │           │                          │          │
        │          └─────┬─────┘                          │          │
        │                │                                │          │
   ┌────▼────────────────▼──────┐              ┌──────────▼──────────▼──┐
   │     Roofline Model         │              │   Graph Capture         │
   │  AI = FLOPs / Bytes        │              │ trace/script/fx/dynamo │
   │  compute vs memory bound   │              │ (Day 12)               │
   │  (Day 6-7)                 │              └───────────┬────────────┘
   └────────────┬───────────────┘                          │
                │                                          │
                └──────────────────┬───────────────────────┘
                                   │
                          ┌────────▼────────┐
                          │   FUSION        │
                          │ Eliminate HBM   │
                          │ round-trips by  │
                          │ keeping interme-│
                          │ diates in regs  │
                          │ (Day 13)        │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │   Profiling     │
                          │ (Days 8-11)     │
                          │ Measure before  │
                          │ & after to      │
                          │ validate gains  │
                          └─────────────────┘
```

### Key Insight Chains

1. **Hardware → Software path:** GPU has limited memory bandwidth → element-wise ops are memory-bound → fusion reduces memory traffic → graph mode enables the compiler to find fusion opportunities

2. **Measurement path:** Profiling (nsight, torch.profiler) → identify bottleneck (compute vs memory) → Roofline model quantifies gap → fusion/optimization targets the gap → re-profile to validate

3. **Abstraction path:** Eager mode (Python-friendly) → graph capture (multiple methods) → IR representation → compiler optimizations → generated kernel code

---

## 2. The Five Mental Models

These are the core frameworks you should now carry in your head:

### Model 1: The Memory Hierarchy Ladder

```
Speed:   Fastest ◄──────────────────────────────► Slowest
Size:    Smallest ◄─────────────────────────────► Largest

┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│Registers │  │  SRAM    │  │   L2     │  │   HBM    │
│ ~20 TB/s │  │ ~19 TB/s │  │ ~6 TB/s  │  │ ~2 TB/s  │
│  256 KB  │  │  20 MB   │  │  40 MB   │  │  80 GB   │
│(per SM)  │  │(per SM:  │  │ (shared) │  │ (global) │
│          │  │ 192 KB)  │  │          │  │          │
└──────────┘  └──────────┘  └──────────┘  └──────────┘

The goal of every optimization: keep data as far LEFT as possible.
Fusion keeps intermediates in registers instead of writing to HBM.
```

### Model 2: Arithmetic Intensity and the Roofline

$$\text{Attainable FLOPS} = \min\left(\text{Peak FLOPS},\; \text{BW} \times \text{AI}\right)$$

- Below the ridge point → **memory-bound** → optimize for fewer bytes
- Above the ridge point → **compute-bound** → optimize for fewer FLOPs
- **Fusion increases AI** by reducing bytes while keeping FLOPs constant

### Model 3: The Eager-Graph Spectrum

```
Full Python ◄──────────────────────────────► Full Optimization

  Eager       jit.trace   jit.script   FX      torch.compile
  │              │            │         │            │
  │  No graph    │  Frozen    │ Parsed  │ Symbolic   │ Dynamic
  │  capture     │  snapshot  │ subset  │ proxy      │ bytecode
  │              │            │         │            │ interception
  │              │            │         │            │
  Flexibility ★★★★★          ★★★       ★★           ★★★★
  Optimization   ☆            ★★        ★★           ★★★★★
```

### Model 4: The Fusion Decision Tree

```
Can these ops be fused?
         │
    Are they element-wise    YES ──► FUSE (pointwise fusion)
    with same shapes?
         │ NO
         │
    Is one a producer and    YES ──► Is the consumer element-wise?
    the other a consumer?                │
         │ NO                       YES ──► FUSE (vertical fusion)
         │                          NO  ──► Is it matmul epilogue?
    Do they share                            │
    the same input?                     YES ──► EPILOGUE FUSE
         │                              NO  ──► CANNOT FUSE
    YES ──► HORIZONTAL FUSE
    NO  ──► CANNOT FUSE

Special barrier: REDUCTIONS must complete before downstream fusion.
```

### Model 5: The Profiling Loop

```
    ┌─► Profile (measure) ─► Identify bottleneck ─► Hypothesize fix ─┐
    │                                                                  │
    └──────────── Validate improvement ◄── Apply optimization ◄───────┘
    
    NEVER optimize without measuring first.
    NEVER claim improvement without measuring after.
```

---

## 3. Self-Assessment Quiz (10 Questions)

Answer each question before checking. Rate your confidence (1–5).

### Questions

**Q1.** An A100 has 2 TB/s memory bandwidth and 312 TFLOPS FP16 compute. What is the arithmetic intensity ridge point?

**Q2.** A `relu()` on an FP32 tensor performs 1 FLOP per element and accesses 8 bytes per element (4 read + 4 write). Is this operation compute-bound or memory-bound on an A100?

**Q3.** You have a chain: `y = sigmoid(relu(x * w + b))`. How many HBM read/write operations occur unfused (assuming each intermediate is written and read back)? How many with full fusion?

**Q4.** What is the key difference between `torch.jit.trace` and `torch.jit.script`?

**Q5.** Why does `torch.jit.trace` give wrong results for models with data-dependent `if` statements?

**Q6.** What is a "graph break" in `torch.compile`, and why is it bad?

**Q7.** Can `x / x.sum()` be fused into a single kernel? Why or why not?

**Q8.** What does the Roofline model tell you that a simple latency measurement does not?

**Q9.** Name three things a compiler can do with a computation graph that it cannot do in eager mode.

**Q10.** A colleague claims their custom CUDA kernel is "optimal" because it uses 100% of compute FLOPS. But the operation is element-wise ReLU. What's wrong with this claim?

---

### Answers

<details>
<summary>Click to reveal answers</summary>

**A1.** Ridge point = $\frac{312 \times 10^{12}}{2 \times 10^{12}} = 156$ FLOPs/byte. Operations below 156 FLOPs/byte are memory-bound.

**A2.** Memory-bound. AI = 1 FLOP / 8 bytes = 0.125 FLOPs/byte, which is far below the 156 FLOPs/byte ridge point. The GPU's compute units are idle >99% of the time waiting for data.

**A3.** **Unfused:** Let each tensor be $N$ elements of 4 bytes.
- `x * w`: read $x$, read $w$, write $t_1$ → 12N bytes
- `+ b`: read $t_1$, read $b$, write $t_2$ → 12N bytes
- `relu()`: read $t_2$, write $t_3$ → 8N bytes
- `sigmoid()`: read $t_3$, write $y$ → 8N bytes
- Total: **40N bytes**, 4 kernel launches

**Fused:** read $x$, $w$, $b$, write $y$ → **16N bytes**, 1 kernel launch. Savings: 60%.

**A4.** `trace` executes the model with real inputs and records operations — control flow is baked in as whatever path was taken during tracing. `script` parses the Python AST and preserves control flow as `if/else` in the IR, but only supports a subset of Python.

**A5.** Because tracing runs the model once with example inputs, it follows only one branch of the `if`. The resulting graph always takes that branch, regardless of future inputs. The conditional is "baked in" at trace time.

**A6.** A graph break is where TorchDynamo cannot trace through a Python operation (e.g., `print()`, unsupported library call) and splits the graph. Each break creates a boundary across which the compiler cannot optimize — no fusion, no memory planning, extra Python overhead between graph segments.

**A7.** No, it cannot be fused into a **single** kernel in general. `x.sum()` is a reduction that must read all elements of $x$ before producing a scalar. Only after the reduction completes can the division proceed. The reduction creates a fusion barrier. (However, the element-wise part `x / scalar` and the reduction `x.sum()` can each be internally optimized.)

**A8.** The Roofline model tells you **whether you're memory-bound or compute-bound** and how close you are to the hardware's theoretical peak. A latency measurement tells you how long something took but not *why* it was slow or how much headroom remains. The Roofline identifies whether to optimize for fewer bytes (memory-bound) or fewer FLOPs (compute-bound).

**A9.** Any three of: (1) Operator fusion, (2) Memory/buffer planning and reuse, (3) Shape specialization / kernel selection, (4) Dead code elimination, (5) Constant folding, (6) Layout optimization (e.g., channels-last), (7) Operation reordering for data locality.

**A10.** ReLU is memory-bound (AI = 0.125 FLOPs/byte). Achieving 100% compute utilization is meaningless because compute is not the bottleneck — memory bandwidth is. The correct metric is % of peak memory bandwidth utilized. A truly optimal ReLU kernel should be hitting close to 2 TB/s bandwidth, not 312 TFLOPS compute.

</details>

---

## 4. Common Misconceptions

### Misconception 1: "More FLOPs = Slower"

**Reality:** An operation with 10× the FLOPs can be *faster* than one with fewer FLOPs if the high-FLOP operation is compute-bound and efficiently utilizing the GPU, while the low-FLOP operation is memory-bound and bottlenecked by bandwidth. Matrix multiplication (many FLOPs, high AI) can be faster than a chain of element-wise ops (few FLOPs, low AI) on the same data.

### Misconception 2: "Graph Mode Replaces Eager Mode"

**Reality:** `torch.compile` uses eager mode as a **fallback** for anything it can't trace. Graph breaks insert eager regions between compiled graphs. The two modes coexist — graph mode is an optimization *layer*, not a replacement.

### Misconception 3: "Fusion Always Helps"

**Reality:** Fusion helps for **memory-bound** operations. For compute-bound operations (large matmuls), fusion of the matmul itself is not the optimization — the matmul kernel already has high arithmetic intensity. Epilogue fusion (fusing a ReLU after matmul) helps, but fusing two large matmuls together rarely does.

### Misconception 4: "torch.compile is Free Performance"

**Reality:** `torch.compile` has:
- **Compilation overhead:** First call triggers compilation (seconds to minutes)
- **Graph breaks:** Non-trivial models may have many breaks, limiting optimization scope
- **Guard overhead:** Dynamic shapes require recompilation when shapes change
- **Correctness risks:** Subtle differences in floating-point behavior under fusion

### Misconception 5: "Registers Are Just Fast Memory"

**Reality:** Registers are *not addressable memory* — they're operand storage for ALU instructions. You can't take a pointer to a register. The compiler assigns variables to registers; you control this indirectly through code structure. In GPU programming (Triton/CUDA), register pressure affects occupancy, which affects latency hiding.

---

## 5. "Ready for Phase II" Checklist

Check each box honestly. If you can't check ≥8/10, revisit the indicated days.

- [ ] **I can draw the GPU memory hierarchy** from registers to HBM with approximate bandwidths (Days 1–3)
- [ ] **I can calculate arithmetic intensity** for any element-wise operation and determine if it's compute or memory bound (Days 6–7)
- [ ] **I can use the Roofline model** to explain why fusion helps element-wise ops but not large matmuls (Days 6–7, 13)
- [ ] **I can explain the difference** between eager and graph execution with concrete examples (Day 12)
- [ ] **I can list 3+ graph capture methods** in PyTorch and their tradeoffs (Day 12)
- [ ] **I can identify fusion opportunities** in a chain of operations and know when fusion is blocked (Day 13)
- [ ] **I can calculate memory traffic savings** from fusing a chain of N element-wise ops (Day 13)
- [ ] **I can use `torch.profiler`** to capture and interpret a Chrome trace (Days 8–11)
- [ ] **I can run `torch.compile`** and inspect generated Triton code with `TORCH_LOGS` (Days 12–13)
- [ ] **I understand why reductions are fusion barriers** and can explain with an example (Day 13)

### Scoring

| Score | Assessment | Action |
|:--|:--|:--|
| 10/10 | Ready for Phase II | Proceed to Day 15 |
| 8–9/10 | Minor gaps | Review indicated days, then proceed |
| 5–7/10 | Significant gaps | Re-do the exercises from weak areas |
| <5/10 | Foundation incomplete | Revisit Week 1 before continuing |

---

## 6. Reflection Prompts

Spend 10 minutes writing brief answers to these. Writing forces clarity.

1. **What was the single most surprising thing you learned in Phase I?** Why did it surprise you — what was your prior mental model?

2. **Draw the data flow for a single `y = relu(x + bias)` call** from Python through the dispatcher to GPU execution. Where does time go in eager mode? Where does fusion save time?

3. **If you were designing a new ML framework from scratch**, would you start with eager or graph mode? What would you sacrifice?

4. **Name one topic where you feel shaky.** Write one specific question you still have. (This becomes your personal focus for Phase II review.)

5. **Explain to an imaginary colleague** (who knows Python but not ML systems) why `torch.compile` can make their training loop 2× faster without changing any model code.

---

## Phase I → Phase II Bridge

What changes in Phase II:

```
Phase I (Days 1-14):                    Phase II (Days 15-28):
─────────────────                       ──────────────────────
"What happens on the GPU"              "How compilers generate code"
                                       
GPU architecture                  ──►  Compiler IR design
Memory hierarchy                  ──►  Optimization passes
Profiling & measurement           ──►  Lowering & code generation
Eager vs graph mode               ──►  Compiler frontend/backend
Operator fusion (intuition)       ──►  Formal fusion algorithms

You now know WHY things are slow.       You'll learn HOW compilers fix it.
```

---

## Key Takeaways

1. **Phase I gave you the hardware mental model** — memory hierarchy, bandwidth limits, arithmetic intensity, and the Roofline framework
2. **Profiling is the foundation** — never optimize without measuring first, never claim improvement without measuring after
3. **Graph capture enables optimization** — the compiler needs to see multiple ops to optimize across them
4. **Fusion is the #1 optimization** for memory-bound workloads — it eliminates intermediate HBM traffic
5. **The ecosystem is converging** on `torch.compile` as the primary optimization path, but understanding *why* it works requires the hardware foundations from Phase I

---

## Further Reading

- Hennessy & Patterson, "Computer Architecture: A Quantitative Approach" — Chapter 4 (Memory Hierarchy)
- Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model" (2009)
- [PyTorch Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — useful context for Phase II

---

## Tomorrow's Teaser

**Day 15: Compiler 101 for ML** opens Phase II. We leave the GPU and zoom out to the classic compiler pipeline — lexing, parsing, IR, optimization passes, code generation — and see how ML compilers (XLA, TVM, TorchInductor) map onto this structure. The concepts you've built about *what to optimize* meet the machinery of *how to optimize it*.
