# ML Systems & Compilers Curriculum

> **10 weeks · 70 daily lessons · From GPU kernels to serving pipelines**

## Overview

How ML models actually get compiled, optimized, and deployed — from graph IR to GPU kernels. Deep dive into Apache TVM and the broader ML optimization ecosystem including Triton, XLA, MLIR, TensorRT, vLLM, and DeepSpeed.

**Prerequisites:** Basic Python, some C/C++, familiarity with PyTorch (Phase I of LLM-to-VLA track is sufficient).

---

## Phase I: Hardware & Compute Foundations (Weeks 1–2, Days 1–14)

### Week 1: GPU Architecture & CUDA
| Day | Topic | Key Concepts |
|-----|-------|-------------|
| 1 | Why ML Needs Compilers | Software-hardware gap, compilation vs interpretation, performance walls |
| 2 | GPU Architecture Deep Dive | SMs, warps, memory hierarchy, occupancy |
| 3 | CUDA Programming Basics | Kernels, grids, blocks, thread indexing |
| 4 | Memory Coalescing & Shared Memory | Global vs shared vs register, bank conflicts |
| 5 | CUDA Profiling & Roofline Model | nsight, arithmetic intensity, memory-bound vs compute-bound |
| 6 | Matrix Multiply — Naive to Tiled | SGEMM evolution, loop tiling, register blocking |
| 7 | **Mini-Project**: Profile & Optimize a GEMM | Benchmark against cuBLAS, analyze bottlenecks |

### Week 2: PyTorch Internals & Profiling
| Day | Topic | Key Concepts |
|-----|-------|-------------|
| 8 | PyTorch Under the Hood | Dispatcher, ATen operators, autograd engine |
| 9 | Memory Management in PyTorch | Caching allocator, fragmentation, OOM strategies |
| 10 | Custom C++ Extensions & pybind11 | torch.utils.cpp_extension, JIT compilation |
| 11 | torch.profiler & Trace Analysis | Chrome trace, tensorboard profiler, flame graphs |
| 12 | Eager vs Graph Mode | torch.jit.trace, torch.jit.script, tradeoffs |
| 13 | Operator Fusion Fundamentals | Horizontal vs vertical fusion, elementwise fusions |
| 14 | **Stop & Reflect #1** | Review, self-assessment, key takeaways |

---

## Phase II: Compiler Infrastructure (Weeks 3–4, Days 15–28)

### Week 3: IR Design & Compiler Passes
| Day | Topic | Key Concepts |
|-----|-------|-------------|
| 15 | Compiler 101 for ML Engineers | Lexing/parsing → IR → optimization → codegen pipeline |
| 16 | LLVM Essentials | SSA form, basic blocks, LLVM IR, optimization passes |
| 17 | MLIR Fundamentals | Dialects, operations, regions, progressive lowering |
| 18 | MLIR Transformation Pipeline | Conversion patterns, canonicalization, dialect conversion |
| 19 | XLA Architecture | HLO IR, optimization passes, XLA-on-GPU, JAX compilation |
| 20 | torch.compile & TorchDynamo | Graph capture via frame evaluation, FX graphs |
| 21 | TorchInductor | FX graph → Triton kernels, codegen pipeline |

### Week 4: Triton & Kernel Engineering
| Day | Topic | Key Concepts |
|-----|-------|-------------|
| 22 | Triton Programming Model | Blocks, axes, pointer arithmetic, auto-vectorization |
| 23 | Triton Matmul | Block-level GEMM, accumulator tiles, L2 cache tiling |
| 24 | FlashAttention Internals | Online softmax, tiling, memory-efficient attention |
| 25 | FlashAttention 2 & 3 | Warp specialization, pipelining, FP8 support |
| 26 | Writing Custom Triton Kernels | Fused operations, activation functions, normalization |
| 27 | Kernel Autotuning | Search spaces, config selection, benchmarking |
| 28 | **Phase II Capstone**: Fused Attention Kernel | End-to-end custom kernel, benchmark vs PyTorch native |

---

## Phase III: Apache TVM Deep Dive (Weeks 5–7, Days 29–49)

### Week 5: TVM Foundations
| Day | Topic | Key Concepts |
|-----|-------|-------------|
| 29 | TVM Architecture Overview | Frontend → Relay → TIR → codegen pipeline |
| 30 | Relay IR — Graph-Level Representation | Functional IR, type system, pattern matching |
| 31 | Relay Optimization Passes | Fusion rules, layout transforms, constant folding |
| 32 | TensorIR (TIR) | Low-level tensor programs, loops, buffers |
| 33 | Schedule Primitives | split, reorder, vectorize, unroll, bind |
| 34 | Compute & Schedule Separation | Design philosophy, decoupling what from how |
| 35 | **Stop & Reflect #2** | TVM foundations review, hands-on practice |

### Week 6: TVM Tuning & Backends
| Day | Topic | Key Concepts |
|-----|-------|-------------|
| 36 | AutoTVM | Template-based tuning, cost model, XGBoost |
| 37 | MetaSchedule | Trace-based search, design space generation |
| 38 | Ansor / Auto-Scheduler | Sketch-based generation, evolutionary search |
| 39 | BYOC Framework | Bring Your Own Codegen, extern functions, annotation |
| 40 | TVM GPU Backends | CUDA, ROCm, Vulkan code generation |
| 41 | TVM CPU Backends | x86 AVX/SSE, ARM NEON, SIMD vectorization |
| 42 | microTVM & Edge Deployment | Bare-metal targets, AOT compilation, Arduino/Zephyr |

### Week 7: TVM Advanced & MLC Ecosystem
| Day | Topic | Key Concepts |
|-----|-------|-------------|
| 43 | Relax — Next-Gen Graph IR | Python-first transformations, dataflow blocks |
| 44 | Relax Transformations | FuseOps, LegalizeOps, LiftTransformParams |
| 45 | MLC LLM | TVM-based LLM deployment, quantized models |
| 46 | WebLLM | In-browser inference via WebGPU, model sharding |
| 47 | End-to-End: ONNX → TVM → Deploy | Import, optimize, compile, benchmark |
| 48 | TVM vs TensorRT vs ONNX Runtime | Head-to-head benchmarks, when to use what |
| 49 | **Phase III Capstone**: Compile & Deploy a Model | Full TVM pipeline, measure speedups |

---

## Phase IV: Inference Optimization (Weeks 8–9, Days 50–63)

### Week 8: Model Formats & Runtime Engines
| Day | Topic | Key Concepts |
|-----|-------|-------------|
| 50 | ONNX Format Deep Dive | Graph structure, opsets, shape inference, interoperability |
| 51 | ONNX Runtime Internals | Execution providers, graph optimization, session options |
| 52 | TensorRT Fundamentals | Builder API, layers, optimization profiles |
| 53 | TensorRT Advanced | Custom plugins, dynamic shapes, INT8 calibration |
| 54 | Quantization Deep Dive | PTQ, QAT, GPTQ, AWQ, SmoothQuant mechanics |
| 55 | Pruning & Distillation for Inference | Structured vs unstructured, knowledge distillation |
| 56 | **Stop & Reflect #3** | Inference engines review |

### Week 9: LLM Serving Systems
| Day | Topic | Key Concepts |
|-----|-------|-------------|
| 57 | KV Cache Internals | Memory layout, PagedAttention, block tables |
| 58 | vLLM Architecture | Scheduler, block manager, engine pipeline |
| 59 | Continuous Batching | Iteration-level scheduling, preemption |
| 60 | Speculative Decoding | Draft models, token trees, verification |
| 61 | SGLang & Structured Generation | RadixAttention, XGrammar, constrained decoding |
| 62 | Multi-GPU Inference | Tensor parallelism, pipeline parallelism, NVLink |
| 63 | **Phase IV Capstone**: Optimized LLM Serving | Build and benchmark a serving pipeline |

---

## Phase V: Training at Scale & Capstone (Week 10, Days 64–70)

### Week 10: Distributed Training & Final Project
| Day | Topic | Key Concepts |
|-----|-------|-------------|
| 64 | Mixed Precision Training | FP16, BF16, loss scaling, AMP |
| 65 | DeepSpeed ZeRO | Stages 1/2/3, optimizer/gradient/parameter partitioning |
| 66 | Megatron-LM Parallelism | Tensor/pipeline/sequence parallelism, 3D parallelism |
| 67 | PyTorch FSDP & Distributed | All-reduce, gradient compression, ring topology |
| 68 | Gradient Checkpointing | Activation recomputation, memory-compute tradeoff |
| 69 | **Final Capstone Day 1**: End-to-End Pipeline | Full optimization pipeline, profiling report |
| 70 | **Final Capstone Day 2**: Benchmark & Lessons | Final benchmarks, comparison, retrospective |

---

## How to Use This Curriculum

1. **One day = one lesson** (~2–3 hours of focused study)
2. **Each lesson** has: Theory → Code examples → Hands-on exercises → Key takeaways
3. **Capstone projects** integrate multiple days of learning
4. **Stop & Reflect** days are for consolidation and self-assessment
5. **Math**: Uses inline LaTeX — render with KaTeX-compatible viewers

## Folder Structure

```
learn/ml-compilers/
├── CURRICULUM.md          ← this file
├── study-notes/           ← phase summary notes
│   ├── 01-hardware-foundations.md
│   ├── 02-compiler-infrastructure.md
│   ├── 03-tvm-deep-dive.md
│   ├── 04-inference-optimization.md
│   └── 05-training-at-scale.md
└── weeks/
    ├── week-01/           ← GPU Architecture & CUDA
    ├── week-02/           ← PyTorch Internals
    ├── week-03/           ← IR & Compiler Passes
    ├── week-04/           ← Triton & Kernels
    ├── week-05/           ← TVM Foundations
    ├── week-06/           ← TVM Tuning & Backends
    ├── week-07/           ← TVM Advanced & MLC
    ├── week-08/           ← Model Formats & Runtimes
    ├── week-09/           ← LLM Serving Systems
    └── week-10/           ← Distributed Training & Capstone
```
