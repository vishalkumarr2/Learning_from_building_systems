# Day 70: Capstone Project — Part 3: Evaluation & What's Next

> **Phase V · Week 10 · Day 70 of 70 · 2.5 hours**

*"The measure of an engineer is not the code they write, but the rigour with which they prove it works — and the clarity with which they explain it to others."*

---

| ← Previous | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|
| [Day 69: Capstone Part 2](day-69-capstone-part2.md) | [Week 10: Distributed Training & Capstone](README.md) | [Phase V: Training at Scale](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Building a system is only half the work. **Evaluating** it rigorously — proving it's correct, measuring its performance, understanding its limitations — is what turns a prototype into a credible engineering artifact. Today you'll learn to write the kind of technical evaluation that gets accepted at systems conferences, that earns trust in code review, and that demonstrates mastery. Then we'll zoom out: what have you learned across 70 days, where does this field go next, and how do you continue growing?

---

## 1. Benchmarking Results Template

Every evaluation needs consistent presentation. Use this template:

```markdown
## Evaluation Results: [Project Name]

### Setup
- **Hardware**: NVIDIA A100 80GB, AMD EPYC 7763, 512 GB RAM
- **Software**: PyTorch 2.4, Triton 3.0, CUDA 12.4, Python 3.11
- **Models tested**: ResNet-50, GPT-2 (124M), ViT-B/16
- **Batch sizes**: 1, 8, 32, 64
- **Precision**: FP32, FP16, INT8
- **Iterations**: 100 measured (10 warmup)

### Latency (ms, lower is better)
| Model      | Baseline | Optimized | Speedup |
|------------|----------|-----------|---------|
| ResNet-50  | 12.4     | 8.1       | 1.53×   |
| GPT-2      | 45.2     | 31.7      | 1.43×   |
| ViT-B/16   | 18.9     | 13.2      | 1.43×   |

### Memory (MB, lower is better)
| Model      | Baseline | Optimized | Reduction |
|------------|----------|-----------|-----------|
| ResNet-50  | 1240     | 890       | 28.2%     |
| GPT-2      | 2100     | 1650      | 21.4%     |

### Correctness
| Model     | Max |Δ|    | Mean |Δ|   | Status |
|-----------|-------------|------------|--------|
| ResNet-50 | 2.4e-6      | 1.1e-7     | ✅ PASS |
| GPT-2     | 8.7e-5      | 3.2e-6     | ✅ PASS |
```

### Generating Results Programmatically

```python
# evaluation/benchmark.py
"""Automated benchmark runner with result formatting."""

import torch
import json
from pathlib import Path

def evaluate_project(
    original_model: torch.nn.Module,
    optimized_model: torch.nn.Module,
    test_inputs: list[dict],
    output_path: Path,
):
    """Run full evaluation and save results."""
    results = {"correctness": [], "latency": [], "memory": []}

    for test in test_inputs:
        name = test["name"]
        inp = test["input"].cuda()

        # --- Correctness ---
        with torch.no_grad():
            ref = original_model(inp)
            opt = optimized_model(inp)
        max_diff = (ref - opt).abs().max().item()
        mean_diff = (ref - opt).abs().mean().item()
        results["correctness"].append({
            "model": name, "max_diff": max_diff,
            "mean_diff": mean_diff, "pass": max_diff < 1e-4,
        })

        # --- Latency ---
        from evaluation.timing import benchmark_fn
        base_t = benchmark_fn(original_model, inp)
        opt_t = benchmark_fn(optimized_model, inp)
        results["latency"].append({
            "model": name,
            "baseline_ms": base_t["median_ms"],
            "optimized_ms": opt_t["median_ms"],
            "speedup": base_t["median_ms"] / opt_t["median_ms"],
        })

        # --- Memory ---
        from evaluation.memory import profile_memory
        base_m = profile_memory(original_model, inp)
        opt_m = profile_memory(optimized_model, inp)
        results["memory"].append({
            "model": name,
            "baseline_mb": base_m["peak_mb"],
            "optimized_mb": opt_m["peak_mb"],
            "reduction_pct": (1 - opt_m["peak_mb"] / base_m["peak_mb"]) * 100,
        })

    output_path.write_text(json.dumps(results, indent=2))
    return results
```

---

## 2. Performance Analysis Methodology

### Roofline Analysis

Place your optimized kernels on the roofline to understand if you've hit the hardware limit:

$$\text{Attainable FLOPS} = \min\left(\text{Peak FLOPS},\; \text{Bandwidth} \times \text{Arithmetic Intensity}\right)$$

```
Roofline Model for Your Capstone
═══════════════════════════════════════════════════════════════

  GFLOPS/s
  │
  │                          ┌── Peak Compute: 312 TFLOPS (A100 FP16)
  │                         ╱
  │                    ────╱── Your fused kernel (good!)
  │               ────╱
  │          ────╱
  │     ────╱
  │────╱                      Roofline
  │  ╱
  │╱  ↑ Unfused kernel (memory-bound → below roofline)
  │
  └────────────────────────── Arithmetic Intensity (FLOP/byte)
       1    4    16   64  256

  Key insight: Fusion increases arithmetic intensity
  by eliminating intermediate memory traffic.

  Before fusion:  matmul (high AI) → store → load → gelu (low AI) → store
                  Total AI = dominated by gelu's low AI

  After fusion:   matmul+gelu (combined high AI)
                  Total AI = matmul's AI (gelu computed in registers)
```

### Speedup Decomposition

Break down where the speedup comes from:

```python
# Analysis: where did the speedup come from?
def decompose_speedup(baseline_profile, optimized_profile):
    """Attribute speedup to specific optimizations."""
    total_speedup = baseline_profile["total_ms"] / optimized_profile["total_ms"]

    # Kernel fusion: fewer kernel launches
    launch_saving = (baseline_profile["num_kernels"] - optimized_profile["num_kernels"])
    launch_overhead = 0.005  # ~5μs per kernel launch

    # Memory bandwidth: fewer intermediate tensors
    mem_saving_gb = (baseline_profile["total_mem_traffic_gb"]
                     - optimized_profile["total_mem_traffic_gb"])
    bandwidth_gbps = 2039  # A100 HBM bandwidth

    # Compute: better utilization of tensor cores
    compute_util_delta = (optimized_profile["tensor_core_util"]
                          - baseline_profile["tensor_core_util"])

    print(f"Total speedup: {total_speedup:.2f}×")
    print(f"  Kernel launch reduction: {launch_saving} fewer launches "
          f"(~{launch_saving * launch_overhead:.1f} ms saved)")
    print(f"  Memory traffic reduction: {mem_saving_gb:.1f} GB "
          f"(~{mem_saving_gb / bandwidth_gbps * 1000:.1f} ms saved)")
    print(f"  Compute utilization: +{compute_util_delta:.1f}%")
```

---

## 3. Writing a Technical Report

Structure your capstone report like a systems paper:

```
Technical Report Structure
═══════════════════════════════════════════════════════════════

  1. Abstract (5 sentences)
     └─ Problem, approach, key result, significance

  2. Introduction (1 page)
     └─ Motivation, problem statement, contributions list

  3. Background (0.5 page)
     └─ Brief: FX graphs, Triton, relevant concepts

  4. Design (1 page)
     └─ Architecture diagram, key design decisions, trade-offs

  5. Implementation (1.5 pages)
     └─ Core algorithms, interesting engineering challenges

  6. Evaluation (2 pages)       ← Most important section
     └─ Setup, results tables, roofline analysis, ablation study

  7. Limitations & Future Work (0.5 page)
     └─ What doesn't work yet, what you'd do with more time

  8. Conclusion (3 sentences)

  Total: ~7 pages — sufficient for a workshop paper or blog post
```

### Ablation Study Template

Show which optimizations contribute how much:

```
Ablation Study: Contribution of Each Optimization
═══════════════════════════════════════════════════════════════

  Configuration              Latency (ms)  Speedup   Memory (MB)
  ──────────────────────────────────────────────────────────────
  Baseline (no optimization)     45.2       1.00×       2100
  + Fusion only                  38.1       1.19×       1850
  + Fusion + Memory planning     35.4       1.28×       1650
  + Fusion + Memory + Quant      28.9       1.56×       1050
  + All optimizations            27.3       1.66×        980

  Observation: Fusion provides the largest single-step improvement.
  Quantization has the biggest memory reduction but modest latency gain.
```

---

## 4. Curriculum Retrospective — Your 70-Day Journey

```
Your ML Compiler Journey
═══════════════════════════════════════════════════════════════

  Phase I: Foundations (Days 1–14)
  ════════════════════════════════
  Week 1: ML Frameworks          Week 2: Graph IRs
  ┌───────────────────┐          ┌───────────────────┐
  │ PyTorch internals │          │ Computation graphs│
  │ Tensor ops        │          │ SSA form          │
  │ Autograd engine   │          │ FX / MLIR / XLA   │
  │ Execution modes   │          │ IR design choices │
  └───────────────────┘          └───────────────────┘
         │                              │
         └──────────┐    ┌──────────────┘
                    ▼    ▼
  Phase II: Core Optimizations (Days 15–28)
  ═════════════════════════════════════════
  Week 3: Lowering & Tiling      Week 4: Operator Fusion
  ┌───────────────────┐          ┌───────────────────┐
  │ High→Low IR       │          │ Horizontal fusion │
  │ Loop tiling       │          │ Vertical fusion   │
  │ Polyhedral model  │          │ Pattern matching  │
  │ Memory hierarchy  │          │ Kernel generation │
  └───────────────────┘          └───────────────────┘
         │                              │
         └──────────┐    ┌──────────────┘
                    ▼    ▼
  Phase III: Hardware & Memory (Days 29–42)
  ═════════════════════════════════════════
  Week 5: Scheduling             Week 6: Memory Optimization
  ┌───────────────────┐          ┌───────────────────┐
  │ Instruction sched │          │ Liveness analysis │
  │ Pipeline parallel │          │ Buffer allocation │
  │ Register alloc    │          │ Activation ckpt   │
  │ Auto-tuning       │          │ Gradient ckpt     │
  └───────────────────┘          └───────────────────┘
         │                              │
         └──────────┐    ┌──────────────┘
                    ▼    ▼
  Phase IV: Production Systems (Days 43–56)
  ═════════════════════════════════════════
  Week 7: Hardware Backends      Week 8: Quantization
  ┌───────────────────┐          ┌───────────────────┐
  │ GPU architecture  │          │ INT8/INT4 quant   │
  │ Triton codegen    │          │ PTQ / QAT         │
  │ Tensor Cores      │          │ Mixed precision   │
  │ Vendor compilers  │          │ KV cache quant    │
  └───────────────────┘          └───────────────────┘
         │                              │
         └──────────┐    ┌──────────────┘
                    ▼    ▼
  Phase V: Training at Scale (Days 57–70)
  ════════════════════════════════════════
  Week 9: torch.compile Deep Dive  Week 10: Distributed & Capstone
  ┌───────────────────┐            ┌───────────────────┐
  │ Dynamo internals  │            │ Data/tensor/pipe  │
  │ Inductor backend  │            │ FSDP / GSPMD      │
  │ Graph breaks      │            │ Compiler for train│
  │ Custom backends   │            │ ★ CAPSTONE ★      │
  └───────────────────┘            └───────────────────┘

  Concepts that connect everything:
  ─────────────────────────────────
  Graphs → Passes → Lowering → Scheduling → Codegen → Hardware
     ↑                                                    │
     └──── Profiling / Benchmarking / Feedback ───────────┘
```

---

## 5. Concept Map — The Big Picture

Every concept you learned connects to others:

```
ML Compiler Concept Map
═══════════════════════════════════════════════════════════════

                    ┌─────────────┐
                    │ ML Model    │
                    │ (PyTorch)   │
                    └──────┬──────┘
                           │ torch.export / fx.trace
                    ┌──────▼──────┐
                    │  Graph IR   │◄──── MLIR, XLA HLO, TorchScript
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Analysis │ │ Fusion   │ │ Quantize │
        │          │ │          │ │          │
        │ shapes   │ │ elem-wise│ │ PTQ/QAT  │
        │ dtypes   │ │ matmul+  │ │ INT8/4   │
        │ FLOPs    │ │ attention│ │ calibrate│
        └──────────┘ └────┬─────┘ └────┬─────┘
                          │            │
                    ┌─────▼────────────▼─────┐
                    │   Memory Planning      │
                    │   liveness · reuse      │
                    │   activation ckpt       │
                    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │   Scheduling            │
                    │   tiling · vectorize     │
                    │   pipeline · autotune    │
                    └──────────┬──────────────┘
                               │
                    ┌──────────▼──────────────┐
                    │   Code Generation       │
                    │   Triton · CUDA · PTX    │
                    │   vendor libs (cuDNN)    │
                    └──────────┬──────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐   ┌──────────────┐  ┌──────────┐
        │  Single  │   │  Distributed │  │  Serving │
        │  GPU     │   │  Training    │  │  Infra   │
        │          │   │  FSDP/GSPMD  │  │  vLLM    │
        └──────────┘   └──────────────┘  └──────────┘
```

---

## 6. Career Paths in ML Systems

```
Career Paths After This Curriculum
═══════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────┐
  │               ML Compiler Engineer                  │
  │  Companies: Google (XLA), Meta (Glow/Inductor),     │
  │  NVIDIA (TensorRT), AMD (ROCm), Modular (Mojo)     │
  │  Focus: Graph optimization, code generation, IR     │
  └──────────────────────────┬──────────────────────────┘
                             │
  ┌──────────────────────────┼──────────────────────────┐
  │                          │                          │
  ┌▼─────────────────┐  ┌───▼───────────────┐  ┌───────▼────────┐
  │ ML Infrastructure│  │ GPU Kernel         │  │ ML Framework   │
  │ Engineer         │  │ Engineer           │  │ Developer      │
  │                  │  │                    │  │                │
  │ Scale training   │  │ Write CUDA/Triton  │  │ PyTorch/JAX    │
  │ Build serving    │  │ Optimize attention │  │ core team      │
  │ MLOps pipelines  │  │ Custom hardware    │  │ Autograd, JIT  │
  └──────────────────┘  └────────────────────┘  └────────────────┘

  Emerging roles:
  ───────────────
  • LLM Serving Engineer — vLLM, TGI, TensorRT-LLM optimization
  • AI Chip Compiler Engineer — TPU, Trainium, Gaudi compilers
  • ML Performance Engineer — profiling, roofline, bottleneck analysis
  • AI Systems Researcher — novel compiler techniques, papers at OSDI/MLSys
```

---

## 7. Open Problems in ML Compilers

These are active research areas — your next challenge after this curriculum:

```
Open Problems (2025–2030)
═══════════════════════════════════════════════════════════════

  1. DYNAMIC SHAPES
     Current compilers assume static shapes. Real workloads
     (variable-length text, ragged batches) need dynamic compilation
     without recompilation overhead.

  2. WHOLE-PROGRAM OPTIMIZATION FOR TRAINING
     XLA/GSPMD optimizes single-step. Optimizing across steps
     (learning rate schedules, curriculum learning) is unsolved.

  3. HETEROGENEOUS HARDWARE
     Training on GPU + CPU + TPU + custom accelerators requires
     unified IR and cost models that span architectures.

  4. COMPILER-HARDWARE CO-DESIGN
     Design hardware ISAs that are compiler-friendly, not just
     benchmark-friendly. Close the "hardware lottery" gap.

  5. VERIFIED COMPILATION
     Prove that compiler transformations preserve model semantics —
     essential for safety-critical ML (medical, autonomous driving).

  6. SPARSITY-AWARE COMPILATION
     Structured sparsity (2:4, block sparse) needs compiler support
     for pruning-aware scheduling and memory layout.

  7. ONLINE / JIT COMPILATION FOR AGENTS
     LLM agents generate variable compute graphs at runtime.
     Compilers must adapt in milliseconds, not seconds.
```

---

## 8. Recommended Next Steps

### Papers to Read

| Paper | Why | Year |
|-------|-----|------|
| Ansel et al., "PyTorch 2" | torch.compile architecture | 2024 |
| Zheng et al., "TVM: An Automated End-to-End Optimizing Compiler" | Foundational ML compiler | 2018 |
| Kwon et al., "Efficient Memory Management for LLM Serving with PagedAttention" | vLLM / memory innovation | 2023 |
| Shazeer, "Fast Transformer Decoding" | Multi-query attention origin | 2019 |
| Xu et al., "GSPMD" | Automated parallelism | 2021 |
| Tillet et al., "Triton: An Intermediate Language for Block-Structured Programs" | GPU codegen | 2019 |

### Conferences to Follow

- **MLSys** — top venue for ML systems research
- **OSDI / SOSP** — operating systems, including ML infra
- **ISCA / MICRO** — computer architecture, including AI accelerators
- **NeurIPS (Systems track)** — ML + systems intersection

### Open Source Projects to Contribute To

```
High-Impact Open Source Contributions
═══════════════════════════════════════════════════════════════

  Beginner-friendly:
  ├── pytorch/pytorch — Inductor backend (triton templates)
  ├── openai/triton — Triton language compiler
  └── vllm-project/vllm — LLM serving optimizations

  Intermediate:
  ├── llvm/torch-mlir — PyTorch ↔ MLIR bridge
  ├── onnx/onnx-mlir — ONNX model compiler
  └── apache/tvm — End-to-end ML compiler

  Advanced:
  ├── google/jax — XLA integration, custom partitioning
  ├── NVIDIA/TensorRT — inference optimization plugins
  └── modularml/mojo — next-gen ML language + compiler
```

---

## 🎉 Congratulations!

```
═══════════════════════════════════════════════════════════════

  ██████╗ ██████╗ ███╗   ██╗ ██████╗ ██████╗  █████╗ ████████╗███████╗██╗
 ██╔════╝██╔═══██╗████╗  ██║██╔════╝ ██╔══██╗██╔══██╗╚══██╔══╝██╔════╝██║
 ██║     ██║   ██║██╔██╗ ██║██║  ███╗██████╔╝███████║   ██║   ███████╗██║
 ██║     ██║   ██║██║╚██╗██║██║   ██║██╔══██╗██╔══██║   ██║   ╚════██║╚═╝
 ╚██████╗╚██████╔╝██║ ╚████║╚██████╔╝██║  ██║██║  ██║   ██║   ███████║██╗
  ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝

  You completed 70 days of ML Systems & Compilers.

  What you can now do:
  ✓ Read and write computation graph IRs (FX, MLIR, XLA HLO)
  ✓ Implement compiler passes: fusion, tiling, scheduling
  ✓ Write GPU kernels in Triton and understand CUDA codegen
  ✓ Apply quantization (INT8/INT4, PTQ, QAT) correctly
  ✓ Optimize memory with liveness analysis and checkpointing
  ✓ Use torch.compile effectively and build custom backends
  ✓ Understand distributed training: FSDP, tensor/pipeline parallel
  ✓ Benchmark ML systems rigorously with proper methodology
  ✓ Design and build end-to-end ML optimization tools

  What separates you from most ML engineers:
  • You understand what happens BELOW torch.compile
  • You can profile, diagnose, and fix performance bottlenecks
  • You know when to use each optimization and why
  • You can contribute to PyTorch, Triton, TVM, and XLA

═══════════════════════════════════════════════════════════════
```

The field of ML compilers is young, fast-moving, and desperately short of engineers who understand both the ML and the systems side. You now stand at that intersection. Every frontier model trained, every LLM served at scale, every AI chip that ships — they all depend on the compiler stack you've spent 70 days learning.

**Go build something extraordinary.**

---

## Hands-On Exercises

### Exercise 1: Complete Your Evaluation (45 min)

Run the full evaluation suite from Day 69's implementation. Fill in the benchmarking results template from Section 1 with your actual numbers. Include correctness, latency, and memory results for at least 2 models.

### Exercise 2: Write Your Technical Report (45 min)

Using the structure from Section 3, write a 2-page technical report covering:
- Your chosen project and design rationale
- Key implementation decisions
- Evaluation results with one ablation study
- Limitations and what you'd improve with more time

### Exercise 3: Concept Map Extension (30 min)

Take the concept map from Section 5 and extend it with:
- 3 concepts you found most challenging (mark in red)
- 3 connections between concepts that surprised you
- 1 area you want to explore deeper next

---

## Key Takeaways

1. **Rigorous evaluation** separates engineering from hacking — report median, p95, p99; show correctness bounds; decompose speedup sources
2. **Ablation studies** prove each optimization contributes — remove them one at a time and measure the delta
3. **Roofline analysis** reveals whether you've hit the hardware ceiling or have room to optimize further
4. **The field is wide open** — dynamic shapes, verified compilation, sparsity-aware scheduling, and agent-time JIT are all unsolved
5. **Your 70-day foundation** covers the full stack: IR → passes → lowering → scheduling → codegen → hardware → distributed → evaluation

---

## Further Reading

- Patterson & Hennessy, *Computer Architecture: A Quantitative Approach* — the systems performance bible
- Leiserson et al., "There's plenty of room at the Top" (Science, 2020) — why software performance matters
- Hooker, "The Hardware Lottery" (2021) — how hardware choices constrain algorithms
- MLSys Conference Proceedings (2020–2025) — the frontier of ML systems research
- Your own capstone report — the most important document you'll write this week
