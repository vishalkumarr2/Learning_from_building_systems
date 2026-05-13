# Day 49: Stop & Reflect #4 — Phase III Complete

> **Phase III · Week 7 · Day 49 of 70 · 2.5 hours**

*"You don't truly understand a system until you can draw its map from memory and explain where every edge leads."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 48: Compiler Testing & Verification](day-48-compiler-testing-verification.md) | [Day 50: Model Formats & ONNX](../week-08/day-50-model-formats-onnx.md) | [Week 7: TVM Advanced & MLC](README.md) | [Phase III: Apache TVM Deep Dive](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

You've completed **Phase III** — three weeks (Days 29–48) spanning the entire Apache TVM stack, the broader compiler ecosystem (MLIR, XLA, ONNX Runtime), MLC-LLM for LLM deployment, the contribution workflow, and compiler testing. That's **20 dense lessons** of interconnected material. Before moving to Phase IV (inference optimization and deployment), you must consolidate: verify recall, identify gaps, and build the mental frameworks that let you make sound tool-selection decisions in practice. This session tests you, connects the dots, and certifies readiness.

---

## 1. Full Concept Map: The TVM & Ecosystem Universe

Build this map from memory first, then verify against the reference:

```
Complete TVM Ecosystem — Phase III Concept Map
═══════════════════════════════════════════════════════════════════════════════

  Input Models
  ├── PyTorch (torch.export / fx)
  ├── ONNX (onnx.load)
  ├── TFLite (flatbuffer)
  └── Hugging Face (for MLC-LLM)
        │
        ▼  Frontend Import (Day 30)
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                        Relay IR  (Days 30-31)                          │
  │  • Typed functional graph: let-bound, SSA                              │
  │  • Ops: nn.conv2d, nn.dense, add, reshape, ...                         │
  │  • Passes: FuseOps, FoldConstant, SimplifyInference, ToMixedPrecision  │
  │  • Quantization: QNN dialect (Day 39)                                  │
  │  • Pattern matching: DFPatternCallback                                 │
  └────┬──────────────┬──────────────────────────┬─────────────────────────┘
       │              │                          │
       ▼              ▼                          ▼
  ┌──────────┐  ┌──────────────┐         ┌───────────────────────────┐
  │TE (Day32)│  │ BYOC (Day 38)│         │   Relax IR (Day 41)      │
  │ Compute  │  │ Partition →  │         │ • Symbolic shapes (n, m)  │
  │ + Schedule│  │ cuDNN/TRT/  │         │ • Mutable state (KV cache)│
  │           │  │ DNNL/ACL    │         │ • Dataflow blocks         │
  └─────┬────┘  └──────────────┘         │ • call_tir bridge to TIR  │
        │                                └───────────┬───────────────┘
        ▼                                            │
  ┌──────────────────────────────────────────────────┘
  │            TIR — Tensor IR  (Day 33)
  │  • Low-level: loops, buffers, index arithmetic
  │  • Schedule primitives: split, reorder, bind, vectorize
  │  • Buffer scopes: global, shared, local
  └────┬───────────────────────────────────────────────────┐
       │                                                    │
       ▼  Tuning (Days 36-37)                              ▼  Code Generation
  ┌────────────────────────────────┐              ┌────────────────────────┐
  │  AutoTVM     → template-based │              │  LLVM → x86/ARM/RISC-V│
  │  AutoScheduler → Ansor (auto) │              │  CUDA → NVIDIA GPU     │
  │  MetaSchedule → unified, TIR  │              │  Metal → Apple GPU     │
  │                                │              │  Vulkan → cross-GPU    │
  │  Cost model: XGBoost/NN       │              │  OpenCL → mobile GPU   │
  │  Search: evolutionary/SA      │              │  C codegen → µTVM      │
  └────────────────────────────────┘              │  WebGPU → browser      │
                                                   └──────────┬─────────────┘
                                                              │
                                          ┌───────────────────▼──────────────┐
                                          │        Runtime  (Day 34, 40)     │
                                          │  Graph Executor │ AOT Executor   │
                                          │  VM Executor    │ µTVM (bare metal)
                                          │  RPC (remote)   │ WASM (browser) │
                                          └──────────────────────────────────┘

  ═══════ Broader Ecosystem (Week 7) ═══════

  MLIR (Day 43)              XLA / StableHLO (Day 44)
  ├── Dialect system          ├── HLO IR (functional)
  ├── Linalg, Tensor,         ├── XLA compiler (TPU/GPU/CPU)
  │   Affine, SCF, Arith      ├── StableHLO = portable HLO
  ├── Torch-MLIR              └── torch_xla for PyTorch
  │   (PyTorch → MLIR)
  └── IREE (MLIR → deploy)

  ONNX Runtime (Day 45)     MLC-LLM (Day 46)
  ├── Execution Providers     ├── HF model → Relax IR
  │   (CPU, CUDA, TRT, etc.)  ├── Quantization (q4f16_1)
  ├── Graph optimizations     ├── Fused dequant-matmul
  └── Broad model support     ├── Auto-tuning per target
                               └── Universal deploy (phone→browser)
```

---

## 2. Comparison Matrix: ML Compiler Ecosystem

Rate each system across the dimensions that matter for deployment decisions:

```
ML Compiler Comparison Matrix
═════════════════════════════════════════════════════════════════════════════

  Dimension          │ TVM       │ XLA       │ Triton    │ ORT       │ MLIR
  ═══════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪══════════
  Primary use case   │ Multi-    │ TPU +     │ NVIDIA GPU│ Production│ Compiler
                     │ target    │ Google    │ kernels   │ inference │ infra-
                     │ deploy    │ ecosystem │           │           │ structure
  ───────────────────┼───────────┼───────────┼───────────┼───────────┼──────────
  Input format       │ Relay/    │ StableHLO │ Python DSL│ ONNX      │ Dialects
                     │ Relax/ONNX│ (from PT) │ (triton)  │           │ (varied)
  ───────────────────┼───────────┼───────────┼───────────┼───────────┼──────────
  GPU performance    │ ★★★★☆    │ ★★★★☆    │ ★★★★★    │ ★★★☆☆    │ N/A
  (NVIDIA)           │ (tuned)   │ (fused)   │ (hand opt)│ (EP dep.) │ (infra)
  ───────────────────┼───────────┼───────────┼───────────┼───────────┼──────────
  CPU performance    │ ★★★★☆    │ ★★★☆☆    │ ✗         │ ★★★★☆    │ N/A
                     │ (LLVM)    │ (XLA CPU) │ (GPU only)│ (MLAS)    │
  ───────────────────┼───────────┼───────────┼───────────┼───────────┼──────────
  Edge / mobile      │ ★★★★★    │ ★★☆☆☆    │ ✗         │ ★★★☆☆    │ ★★★★☆
                     │ (µTVM)    │ (TF Lite) │           │ (NNAPI)   │ (IREE)
  ───────────────────┼───────────┼───────────┼───────────┼───────────┼──────────
  Browser / WASM     │ ★★★★★    │ ✗         │ ✗         │ ★★★☆☆    │ ★★☆☆☆
                     │ (WebGPU)  │           │           │ (WASM EP) │
  ───────────────────┼───────────┼───────────┼───────────┼───────────┼──────────
  Auto-tuning        │ ★★★★★    │ ★★★☆☆    │ ★★★☆☆    │ ✗         │ ✗
                     │ MetaSched │ (internal)│ (Autotune)│           │
  ───────────────────┼───────────┼───────────┼───────────┼───────────┼──────────
  Ease of use        │ ★★☆☆☆    │ ★★★★☆    │ ★★★★☆    │ ★★★★★    │ ★☆☆☆☆
                     │ (steep)   │ (torch_xla│ (Python)  │ (1-liner) │ (C++ API)
  ───────────────────┼───────────┼───────────┼───────────┼───────────┼──────────
  LLM support        │ ★★★★☆    │ ★★★★☆    │ ★★★★★    │ ★★★☆☆    │ ★★☆☆☆
                     │ (MLC-LLM) │ (Pax/Jax) │ (vLLM)   │ (GenAI)   │
  ───────────────────┼───────────┼───────────┼───────────┼───────────┼──────────
  Community size     │ ★★★☆☆    │ ★★★★☆    │ ★★★★☆    │ ★★★★★    │ ★★★★★
                     │ (~900)    │ (Google)  │ (OpenAI)  │ (MSFT)    │ (LLVM)
  ───────────────────┼───────────┼───────────┼───────────┼───────────┼──────────
  Open governance    │ ★★★★★    │ ★★☆☆☆    │ ★★★☆☆    │ ★★★☆☆    │ ★★★★★
                     │ (Apache)  │ (Google)  │ (OpenAI)  │ (MSFT)    │ (LLVM)
  ═══════════════════╪═══════════╪═══════════╪═══════════╪═══════════╪══════════

  Decision Flowchart:
  ┌─────────────────────────────────────────────────────────────────┐
  │ What's your target?                                             │
  │                                                                 │
  │ NVIDIA GPU only ──→ Triton (best perf) or TensorRT (production)│
  │ Google TPU      ──→ XLA + JAX (only option)                     │
  │ Apple Silicon   ──→ TVM/MLC-LLM (Metal) or CoreML              │
  │ Web browser     ──→ TVM/MLC-LLM (WebGPU) or ORT (WASM)        │
  │ Multi-target    ──→ TVM (best coverage) or ORT (easier)        │
  │ Edge / MCU      ──→ TVM µTVM or IREE (MLIR)                    │
  │ "Just works"    ──→ ONNX Runtime (broadest model support)      │
  │ LLM serving     ──→ vLLM/Triton (NVIDIA) or MLC-LLM (anywhere)│
  └─────────────────────────────────────────────────────────────────┘
```

---

## 3. Self-Assessment Quiz

Answer from memory. Score yourself honestly — each question is worth 1 point.

### Questions

**Q1.** What are the three major IRs in TVM's compilation stack, and what level of abstraction does each operate at?

**Q2.** Explain why `(a + b) + c ≠ a + (b + c)` in floating-point arithmetic. Give a concrete example with numbers.

**Q3.** What is the purpose of `FuseOps` in Relay? Name two types of operators it fuses together.

**Q4.** A TIR `PrimFunc` uses `tir.Split(i, factor=32)`. What does this produce, and why is the factor choice important?

**Q5.** What is BYOC? Name two external libraries TVM can offload to via BYOC.

**Q6.** In MLC-LLM, what does "q4f16_1" mean? Break down each part.

**Q7.** Why does Relax exist when Relay already works? Name two capabilities Relax has that Relay lacks.

**Q8.** What tolerance would you use for `np.allclose` when comparing a matmul with $K=4096$ in float32? Justify with the formula.

**Q9.** Explain the difference between AutoTVM, AutoScheduler (Ansor), and MetaSchedule. When would you choose each?

**Q10.** You need to deploy a model on both NVIDIA GPUs and Apple M-series chips. Which ML compiler ecosystem would you choose and why?

### Answers

<details>
<summary>Click to reveal answers</summary>

**A1.** (1) **Relay** — graph-level functional IR for operator fusion, constant folding, type inference. (2) **TIR** — low-level loop IR for memory layout, tiling, vectorization. (3) **Relax** — next-gen graph IR with symbolic shapes, mutable state, call_tir bridge to TIR.

**A2.** IEEE 754 rounds after each operation. Example: `(1e8 + 1.0) + (-1e8)` → `1e8 + (-1e8) = 0.0` (the 1.0 was absorbed). But `1e8 + (1.0 + (-1e8))` → `1e8 + (-99999999.0) = 1.0`.

**A3.** FuseOps combines adjacent operators into a single kernel to eliminate intermediate memory reads/writes. It fuses: (1) element-wise ops together (relu + add), (2) injective ops into their producer (conv2d + relu), (3) reduction ops with their element-wise consumers.

**A4.** `Split(i, factor=32)` produces two loop variables: `i_outer` (range: `ceil(N/32)`) and `i_inner` (range: `32`). Factor 32 is chosen to match hardware (warp size, vector width, cache line size).

**A5.** BYOC = Bring Your Own Codegen. It partitions subgraphs and delegates them to external libraries: cuDNN, TensorRT, DNNL (oneDNN), ACL (ARM Compute Library).

**A6.** q4f16_1: **q4** = 4-bit quantized weights, **f16** = float16 computation/activations, **1** = group size configuration (group quantization with per-group scale factors).

**A7.** Relax supports: (1) **symbolic shapes** — dimensions can be variables like `(batch, seq_len, hidden)` instead of fixed integers; (2) **mutable state** — KV caches for LLMs can be updated in-place without functional purity hacks.

**A8.** $\text{atol} \approx \sqrt{K} \cdot \epsilon_{\text{machine}} = \sqrt{4096} \times 5.96 \times 10^{-8} \approx 64 \times 6 \times 10^{-8} \approx 3.8 \times 10^{-6}$. In practice, use `atol=1e-4` with `rtol=1e-5` to account for different summation orders across backends.

**A9.** **AutoTVM**: uses hand-written schedule templates, good when you know the search space. **AutoScheduler (Ansor)**: generates schedules automatically without templates, better for new ops. **MetaSchedule**: unified framework working at TIR level, supports both template-based and auto-generated schedules, integrates with Relax. Choose MetaSchedule for new projects; AutoScheduler if MetaSchedule doesn't support your target; AutoTVM only for legacy codebases.

**A10.** TVM (via MLC-LLM pipeline). Reasoning: TVM is the only framework that can compile to both CUDA (NVIDIA) and Metal (Apple) from a single IR. Alternatives: ONNX Runtime requires different execution providers per target and doesn't optimize as aggressively; Triton is NVIDIA-only; XLA is primarily TPU/CUDA.

</details>

### Scoring

```
Score Interpretation
════════════════════

  10/10  ★★★★★  Phase III mastery — proceed to Phase IV immediately
   8-9   ★★★★☆  Strong understanding — review weak areas, then proceed
   6-7   ★★★☆☆  Adequate — revisit 2-3 weakest days before moving on
   4-5   ★★☆☆☆  Gaps remain — re-read key lessons (30, 33, 36, 41, 46)
   0-3   ★☆☆☆☆  Insufficient — redo Phase III before advancing
```

---

## 4. Phase III Concept Connections

### Cross-Cutting Themes

```
Recurring Patterns Across Phase III
════════════════════════════════════

  Theme 1: PROGRESSIVE LOWERING
  ─────────────────────────────
  Relay/Relax (graph) → TIR (loops) → LLVM IR → machine code
  MLIR:  Torch → Linalg → Affine → SCF → LLVM
  XLA:   StableHLO → HLO → target code
  Pattern: always start high, lower through well-defined stages

  Theme 2: SEPARATION OF CONCERNS
  ────────────────────────────────
  TVM:   "what to compute" (TE) vs "how to compute" (Schedule)
  MLIR:  "semantics" (Linalg) vs "mapping" (Affine/SCF)
  Triton: "algorithm" (Python) vs "tiling" (compiler)
  Pattern: separate algorithm specification from optimization

  Theme 3: HARDWARE ABSTRACTION
  ─────────────────────────────
  TVM targets:     "llvm", "cuda", "metal", "vulkan"
  MLIR backends:   LLVM, SPIRV, GPU dialects
  ORT providers:   CPUExecutionProvider, CUDAExecutionProvider
  Pattern: single IR → multiple backends through target descriptors

  Theme 4: SEARCH OVER PROGRAMS
  ─────────────────────────────
  AutoTVM:      search schedule templates
  MetaSchedule: search TIR transformations
  Halide:       search schedule space
  Pattern: correct programs form a space; search finds the fast one

  Theme 5: FUSION IS KING
  ───────────────────────
  Relay FuseOps:  fuse element-wise chains
  XLA:            whole-graph fusion
  Triton:         programmer-defined fusion
  MLC-LLM:       dequantize-matmul fusion
  Pattern: eliminate intermediate memory traffic
```

---

## 5. "Ready for Phase IV" Checklist

Phase IV covers inference optimization and deployment (model formats, quantization-at-scale, serving, edge deployment). Verify readiness:

### Core Knowledge (must be confident)

- [ ] Can draw the TVM compilation pipeline from import to target code generation
- [ ] Can write a TE compute declaration and apply split/reorder/bind schedules
- [ ] Understand Relay passes: FuseOps, FoldConstant, ToMixedPrecision, QNN
- [ ] Know the difference between Graph Executor, AOT Executor, VM, and µTVM runtime
- [ ] Can explain MetaSchedule's search loop: sketch → mutate → measure → model
- [ ] Understand BYOC: annotation → partition → external codegen → runtime dispatch
- [ ] Know what Relax improves: symbolic shapes, mutable state, call_tir

### Ecosystem Knowledge (should know the landscape)

- [ ] Can explain MLIR's dialect + progressive lowering design
- [ ] Understand XLA/StableHLO's role in the JAX/TPU ecosystem
- [ ] Know ONNX Runtime's execution provider architecture
- [ ] Understand MLC-LLM's quantize → compile → deploy pipeline
- [ ] Can choose the right compiler tool for a given target/model/constraint

### Practical Skills (should have done at least once)

- [ ] Built TVM from source (or can follow the steps confidently)
- [ ] Compiled and run a model through Relay on at least one target
- [ ] Written a test using `tvm.testing` utilities
- [ ] Compared outputs across opt_level=0 vs opt_level=3
- [ ] Scored ≥ 7/10 on the self-check quiz above

---

## 6. Phase III → Phase IV Bridge

### What You Know Now

```
Phase III Gave You:
═══════════════════

  ✓ How ML compilers work internally (IR → passes → codegen)
  ✓ How to write and optimize compute kernels (TE + TIR + schedules)
  ✓ How auto-tuning finds fast schedules (MetaSchedule)
  ✓ How to deploy to diverse hardware (LLVM, CUDA, Metal, WebGPU)
  ✓ How the broader ecosystem fits together (MLIR, XLA, ORT)
  ✓ How to test compiler correctness (differential testing, fuzzing)
```

### What Phase IV Will Add

```
Phase IV Builds On This:
════════════════════════

  Day 50-51: Model Formats & ONNX
  └── How models are serialized, exchanged, and standardized

  Day 52-53: Quantization at Scale
  └── Post-training quantization, QAT, GPTQ, AWQ
      (builds on TVM QNN from Day 39, MLC-LLM from Day 46)

  Day 54-55: Inference Serving
  └── vLLM, TensorRT-LLM, serving systems
      (applies TVM/Triton kernels in production)

  Day 56: Edge Deployment
  └── TFLite, CoreML, µTVM in production
      (extends Day 34 and Day 40)

  You already have the foundation — Phase IV applies it.
```

---

## Key Takeaways

1. **Phase III covered the full TVM stack** — from Relay/Relax IR through TIR scheduling, auto-tuning, BYOC, quantization, edge deployment, and Relax
2. **The broader ecosystem (MLIR, XLA, ORT, Triton)** shares TVM's core ideas: progressive lowering, compute/schedule separation, fusion, hardware abstraction
3. **No single tool wins everywhere** — the decision matrix shows that target hardware, model type, and deployment constraints determine the right choice
4. **Compiler correctness is non-negotiable** — silent numerical bugs are worse than crashes; differential testing and fuzzing are essential
5. **Contributing to open-source compilers** is the fastest path to deep understanding and career-level expertise in the field

---

## Further Reading

- [The Deep Learning Compiler: A Comprehensive Survey (2020)](https://arxiv.org/abs/2002.03794) — maps the full landscape
- [MLC Course by Tianqi Chen](https://mlc.ai/) — best single resource for TVM
- [TVM: An Automated End-to-End Optimizing Compiler (OSDI 2018)](https://www.usenix.org/conference/osdi18/presentation/chen) — the original paper
- [Ansor: Generating High-Performance Tensor Programs (OSDI 2020)](https://www.usenix.org/conference/osdi20/presentation/zheng) — auto-scheduling
- [MLIR: Scaling Compiler Infrastructure (CGO 2021)](https://ieeexplore.ieee.org/document/9370308) — MLIR design paper
- [Triton: An Intermediate Language and Compiler (MAPL 2019)](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf) — Triton origins

---

## Next: Phase IV — Inference Optimization & Deployment

Day 50 begins **Phase IV** with model serialization formats. You'll deep-dive into ONNX — the specification, protobuf schema, operator semantics, and how to manually inspect and modify ONNX graphs. This is the bridge between training frameworks and the deployment pipeline you've been building.
