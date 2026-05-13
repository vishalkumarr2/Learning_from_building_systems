# Day 68: Capstone Project — Part 1: Design

> **Phase V · Week 10 · Day 68 of 70 · 2.5 hours**

*"Everyone can run torch.compile. The engineers who matter are the ones who can design a system that knows when, why, and how to apply every optimization in the stack."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 67: Compiler's Role in Training](day-67-compiler-role-in-training.md) | [Day 69: Capstone Part 2](day-69-capstone-part2.md) | [Week 10: Distributed Training & Capstone](README.md) | [Phase V: Training at Scale](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

You've spent 67 days learning individual techniques — operator fusion, memory planning, graph IR design, hardware backends, quantization, distributed training, and compiler-driven optimization. But real engineering requires **integrating** these into a coherent system. This capstone challenges you to design a complete project that combines concepts from across the curriculum. The design phase is critical: a well-architected system is easier to implement, test, debug, and extend. Today you'll practice the skill that separates junior engineers from senior ones — **thinking before coding**.

---

## 1. Choose Your Capstone Project

Select **one** of the three projects below. Each exercises different parts of the curriculum:

```
Capstone Project Options
═══════════════════════════════════════════════════════════════

  Project A: "mlopt" — End-to-End Model Optimizer CLI
  ────────────────────────────────────────────────────
  Weeks exercised: 2 (graph IR), 4 (fusion), 5 (scheduling),
                   6 (memory), 7 (hardware), 8 (quantization)
  Difficulty: ★★★☆☆
  Best for: Breadth — touches many topics at moderate depth

  Project B: Custom torch.compile Backend
  ────────────────────────────────────────
  Weeks exercised: 3 (lowering), 4 (fusion), 5 (scheduling),
                   7 (codegen), 9 (torch.compile internals)
  Difficulty: ★★★★☆
  Best for: Depth — deep into PyTorch compiler internals

  Project C: LLM Serving Benchmark Suite
  ──────────────────────────────────────
  Weeks exercised: 6 (memory), 7 (hardware), 8 (quantization),
                   9 (torch.compile), 10 (distributed)
  Difficulty: ★★★★★
  Best for: Systems — end-to-end performance engineering
```

---

## 2. Project A: "mlopt" — End-to-End Model Optimizer CLI

### Problem Statement

Build a command-line tool that takes a PyTorch model and applies a pipeline of optimizations: graph capture → analysis → fusion → quantization → code generation. The user specifies a target (latency, throughput, memory) and the tool selects and applies the appropriate optimizations.

### Architecture Design

```
mlopt Architecture
═══════════════════════════════════════════════════════════════

  User Input                  Output
  ──────────                  ──────
  model.pt + config.yaml  →  optimized_model.pt + report.json

  ┌─────────────────────────────────────────────────────────┐
  │                     CLI Frontend                        │
  │  argparse / click → config validation → pipeline setup  │
  └──────────────────────────┬──────────────────────────────┘
                             │
  ┌──────────────────────────▼──────────────────────────────┐
  │                   Graph Capture                         │
  │  torch.export / torch.fx.symbolic_trace → FX Graph      │
  └──────────────────────────┬──────────────────────────────┘
                             │
  ┌──────────────────────────▼──────────────────────────────┐
  │                  Analysis Pass                          │
  │  op_count · memory_estimate · bottleneck_detection      │
  │  compute_intensity (FLOPs/byte) per subgraph            │
  └──────────────────────────┬──────────────────────────────┘
                             │
  ┌──────────────────────────▼──────────────────────────────┐
  │               Optimization Pipeline                     │
  │  ┌─────────┐  ┌──────────┐  ┌────────────┐  ┌───────┐ │
  │  │ Fusion  │→ │ Quant    │→ │ Memory     │→ │Layout │ │
  │  │ Pass    │  │ Pass     │  │ Planning   │  │ Opt   │ │
  │  └─────────┘  └──────────┘  └────────────┘  └───────┘ │
  └──────────────────────────┬──────────────────────────────┘
                             │
  ┌──────────────────────────▼──────────────────────────────┐
  │                  Code Generation                        │
  │  Export optimized model + performance report             │
  └─────────────────────────────────────────────────────────┘
```

### Component Breakdown

```python
# Project A: Module structure
mlopt/
├── __init__.py
├── cli.py                  # CLI entry point (click/argparse)
├── config.py               # YAML config schema + validation
├── capture/
│   ├── __init__.py
│   ├── fx_capture.py       # torch.fx symbolic trace
│   └── export_capture.py   # torch.export for stricter capture
├── analysis/
│   ├── __init__.py
│   ├── op_counter.py       # Count ops by type
│   ├── memory_estimator.py # Estimate peak memory
│   └── bottleneck.py       # Identify compute vs memory bound ops
├── passes/
│   ├── __init__.py
│   ├── base_pass.py        # Abstract pass interface
│   ├── fusion.py           # Pattern-based op fusion
│   ├── quantization.py     # PTQ / dynamic quant
│   ├── memory.py           # In-place ops, buffer reuse
│   └── layout.py           # NCHW → NHWC conversion
├── codegen/
│   ├── __init__.py
│   └── exporter.py         # Save optimized model
├── report/
│   ├── __init__.py
│   └── generator.py        # JSON/HTML perf report
└── tests/
    ├── test_capture.py
    ├── test_passes.py
    └── test_e2e.py
```

### Key Design Decisions

```python
# Base pass interface — all optimization passes implement this
from abc import ABC, abstractmethod
from torch.fx import GraphModule

class OptimizationPass(ABC):
    """Base class for all optimization passes."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable pass name."""
        ...

    @abstractmethod
    def analyze(self, gm: GraphModule) -> dict:
        """Analyze graph, return metrics (no mutation)."""
        ...

    @abstractmethod
    def apply(self, gm: GraphModule) -> GraphModule:
        """Apply optimization, return new GraphModule."""
        ...

    def should_apply(self, gm: GraphModule, config: dict) -> bool:
        """Decide whether this pass is beneficial."""
        metrics = self.analyze(gm)
        return metrics.get("estimated_speedup", 1.0) > 1.05  # >5% gain


class PassPipeline:
    """Ordered sequence of optimization passes."""

    def __init__(self, passes: list[OptimizationPass]):
        self._passes = passes

    def run(self, gm: GraphModule, config: dict) -> tuple[GraphModule, list[dict]]:
        reports = []
        for p in self._passes:
            if p.should_apply(gm, config):
                before = p.analyze(gm)
                gm = p.apply(gm)
                after = p.analyze(gm)
                reports.append({
                    "pass": p.name(),
                    "before": before,
                    "after": after,
                })
        return gm, reports
```

---

## 3. Project B: Custom torch.compile Backend

### Problem Statement

Build a custom backend for `torch.compile` that targets a specific optimization goal — e.g., **maximum memory efficiency** by aggressively fusing operations and minimizing intermediate allocations. This backend hooks into PyTorch's compilation pipeline and generates optimized Triton kernels.

### Architecture Design

```
Custom torch.compile Backend Architecture
═══════════════════════════════════════════════════════════════

  User Code:
    @torch.compile(backend="memfuse")
    def model(x): ...

  ┌─────────────────────────────────────────────────────────┐
  │              Dynamo (Graph Capture)                     │
  │  Python bytecode → FX Graph (with guards)              │
  └──────────────────────────┬──────────────────────────────┘
                             │ FX GraphModule
  ┌──────────────────────────▼──────────────────────────────┐
  │            YOUR BACKEND: "memfuse"                      │
  │                                                         │
  │  1. Graph Analysis                                      │
  │     └─ Compute memory pressure per subgraph             │
  │  2. Aggressive Fusion                                   │
  │     └─ Fuse all element-wise chains + reductions        │
  │  3. Memory Planning                                     │
  │     └─ Assign buffers with liveness analysis            │
  │  4. Triton Codegen                                      │
  │     └─ Generate fused Triton kernels                    │
  │  5. Kernel Cache                                        │
  │     └─ Content-hash based caching                       │
  └──────────────────────────┬──────────────────────────────┘
                             │ Compiled callable
  ┌──────────────────────────▼──────────────────────────────┐
  │                   Execution                             │
  │  Cached kernel lookup → launch on CUDA stream           │
  └─────────────────────────────────────────────────────────┘
```

### Registration Point

```python
# Register custom backend with torch.compile
from torch._dynamo.backends.registry import register_backend
from torch.fx import GraphModule
import torch

@register_backend
def memfuse(gm: GraphModule, example_inputs: list[torch.Tensor]):
    """
    Custom torch.compile backend focused on memory-efficient fusion.

    This is the entry point that Dynamo calls with:
    - gm: The captured FX GraphModule
    - example_inputs: Concrete tensor inputs for shape specialization
    """
    # Phase 1: Analyze memory pressure
    mem_analysis = analyze_memory_pressure(gm, example_inputs)

    # Phase 2: Fuse operations aggressively
    fused_gm = aggressive_fusion_pass(gm, mem_analysis)

    # Phase 3: Plan memory allocation
    mem_plan = plan_memory(fused_gm, example_inputs)

    # Phase 4: Generate Triton code
    compiled_fn = generate_triton_kernels(fused_gm, mem_plan, example_inputs)

    return compiled_fn


# Usage
@torch.compile(backend="memfuse")
def transformer_block(x, weight, bias):
    h = x @ weight + bias
    h = torch.nn.functional.gelu(h)
    h = torch.nn.functional.layer_norm(h, [h.shape[-1]])
    return h
```

### Module Structure

```
memfuse/
├── __init__.py
├── backend.py           # register_backend entry point
├── analysis/
│   ├── __init__.py
│   ├── memory.py        # Peak memory estimation per fusion group
│   └── compute.py       # FLOP counting, arithmetic intensity
├── fusion/
│   ├── __init__.py
│   ├── patterns.py      # Fusion pattern matching (matmul+bias+gelu)
│   └── grouper.py       # Partition graph into fusible subgraphs
├── memory/
│   ├── __init__.py
│   ├── liveness.py      # Tensor liveness analysis
│   └── planner.py       # Buffer allocation / reuse
├── codegen/
│   ├── __init__.py
│   ├── triton_emitter.py  # FX nodes → Triton kernel source
│   └── cache.py           # Hash-based kernel cache
└── tests/
    ├── test_fusion.py
    ├── test_codegen.py
    └── test_e2e.py
```

---

## 4. Project C: LLM Serving Benchmark Suite

### Problem Statement

Build a comprehensive benchmark suite that measures LLM inference serving performance across multiple axes: latency (TTFT, TPOT, e2e), throughput (tokens/sec), memory usage, and quality (output consistency). Support multiple backends (vanilla PyTorch, torch.compile, vLLM/TensorRT-LLM stubs) and produce publication-ready reports.

### Architecture Design

```
LLM Serving Benchmark Suite Architecture
═══════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────┐
  │                  Benchmark Runner                    │
  │  load config → instantiate backends → run workloads  │
  └───────────────────────┬──────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
  ┌───────▼──────┐ ┌─────▼───────┐ ┌─────▼──────────┐
  │  Workload    │ │  Backend    │ │  Metric        │
  │  Generator   │ │  Manager    │ │  Collector     │
  │              │ │             │ │                │
  │  • chat      │ │  • eager    │ │  • TTFT        │
  │  • summarize │ │  • compiled │ │  • TPOT        │
  │  • code-gen  │ │  • quantized│ │  • throughput  │
  │  • long-ctx  │ │  • vllm     │ │  • memory      │
  └──────────────┘ └─────────────┘ └────────┬───────┘
                                            │
                                    ┌───────▼───────┐
                                    │  Report Gen   │
                                    │  • JSON       │
                                    │  • Markdown   │
                                    │  • Plots      │
                                    └───────────────┘

  Key metrics:
  ┌────────────────────────────────────────────────────┐
  │  TTFT = Time to First Token (prefill latency)      │
  │  TPOT = Time per Output Token (decode latency)     │
  │  E2E  = End-to-end latency = TTFT + N × TPOT      │
  │  Throughput = total tokens generated / wall time    │
  │  Peak Memory = max GPU memory during inference      │
  └────────────────────────────────────────────────────┘
```

### Measurement Methodology

Correct LLM benchmarking requires careful attention to warm-up, outlier handling, and statistical reporting:

$$\text{TPOT} = \frac{t_{\text{last\_token}} - t_{\text{first\_token}}}{N_{\text{output\_tokens}} - 1}$$

$$\text{Throughput}_{\text{effective}} = \frac{\sum_{i} N_{\text{tokens},i}}{t_{\text{wall}}}$$

```python
# Measurement design — key data class
from dataclasses import dataclass, field

@dataclass
class BenchmarkResult:
    backend: str
    workload: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float              # Time to first token
    tpot_ms: float              # Time per output token (median)
    e2e_latency_ms: float       # End-to-end
    throughput_tok_s: float     # Tokens per second
    peak_memory_mb: float       # Peak GPU memory
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    warmup_iterations: int = 3
    measurement_iterations: int = 10
```

---

## 5. Requirements Gathering Template

Regardless of which project you chose, fill out this requirements document:

```yaml
# capstone_requirements.yaml
project: "<A|B|C>"
title: "<your project title>"

functional_requirements:
  - id: FR-1
    description: "<what the system must do>"
    priority: must_have
  - id: FR-2
    description: "..."
    priority: should_have

non_functional_requirements:
  - id: NFR-1
    description: "Latency: optimization pipeline completes in < 60s for ResNet-50"
  - id: NFR-2
    description: "Memory: tool itself uses < 2 GB RAM"

constraints:
  - "Must run on single GPU (no multi-GPU requirement for the tool itself)"
  - "Python 3.10+, PyTorch 2.x"
  - "No proprietary dependencies"

test_strategy:
  unit_tests:
    - "Each pass independently produces valid FX graphs"
    - "Memory estimates within 20% of actual"
  integration_tests:
    - "Full pipeline on ResNet-50, GPT-2, ViT"
  benchmark_tests:
    - "Compare optimized vs unoptimized on standard models"

success_criteria:
  - "Measurable speedup (> 1.2×) on at least 2 model architectures"
  - "No correctness regression (max 1e-5 output deviation)"
  - "Clean CLI with --help, config files, JSON output"
```

---

## 6. Testing Strategy

### The Test Pyramid for ML Compiler Projects

```
Testing Pyramid for ML Systems
═══════════════════════════════════════════════════════════════

                        ╱╲
                       ╱  ╲
                      ╱ E2E╲         • Full pipeline correctness
                     ╱ Tests╲        • "Does ResNet-50 optimize?"
                    ╱────────╲       • Slow, run on CI nightly
                   ╱          ╲
                  ╱ Integration ╲    • Multi-pass interaction
                 ╱    Tests      ╲   • "Does fusion + quant compose?"
                ╱────────────────╲   • Medium speed, run on every PR
               ╱                  ╲
              ╱    Unit Tests      ╲  • Individual pass correctness
             ╱                      ╲ • "Does conv-bn fusion work?"
            ╱────────────────────────╲ • Fast, run on every commit
           ╱                          ╲
          ╱     Property Tests         ╲ • Invariants always hold
         ╱                              ╲• "Output shape preserved"
        ╱────────────────────────────────╲• Fast, run always

  Critical invariant tests for ML compilers:
  ──────────────────────────────────────────
  1. Output equivalence:  |opt(x) - ref(x)| < ε
  2. Shape preservation:  opt(x).shape == ref(x).shape
  3. Dtype preservation:  opt(x).dtype == ref(x).dtype
  4. Graph validity:      no dangling nodes, all inputs resolved
  5. Memory bound:        peak_mem(opt) ≤ peak_mem(ref)
```

### Correctness Checking Pattern

```python
import torch

def assert_model_equivalence(
    original: torch.nn.Module,
    optimized: torch.nn.Module,
    test_inputs: list[torch.Tensor],
    atol: float = 1e-5,
    rtol: float = 1e-4,
):
    """Verify optimized model matches original within tolerance."""
    original.eval()
    optimized.eval()

    for inp in test_inputs:
        with torch.no_grad():
            ref_out = original(inp)
            opt_out = optimized(inp)

        assert ref_out.shape == opt_out.shape, (
            f"Shape mismatch: {ref_out.shape} vs {opt_out.shape}"
        )
        assert ref_out.dtype == opt_out.dtype, (
            f"Dtype mismatch: {ref_out.dtype} vs {opt_out.dtype}"
        )
        if not torch.allclose(ref_out, opt_out, atol=atol, rtol=rtol):
            max_diff = (ref_out - opt_out).abs().max().item()
            raise AssertionError(
                f"Output divergence: max_diff={max_diff:.2e} "
                f"(atol={atol}, rtol={rtol})"
            )
```

---

## Hands-On Exercises

### Exercise 1: Write Your Requirements Document (45 min)

Choose your capstone project (A, B, or C). Fill out the `capstone_requirements.yaml` template with at least:
- 5 functional requirements (2 must-have, 2 should-have, 1 nice-to-have)
- 3 non-functional requirements
- 4 unit tests, 2 integration tests

### Exercise 2: Architecture Diagram (30 min)

Draw the component diagram for your chosen project (ASCII art or any tool). Identify:
- Data flow between components
- Which components map to which weeks of the curriculum
- External dependencies (PyTorch, Triton, CUDA)

### Exercise 3: Risk Register (15 min)

List 3 technical risks for your project and mitigation strategies:

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| torch.export fails on dynamic shapes | High | Medium | Fall back to `torch.fx.symbolic_trace` |
| Triton codegen produces wrong results | High | Low | Golden reference test on every kernel |
| Benchmark variance too high | Medium | High | 50 warmup iters, report p50/p95/p99 |

---

## Key Takeaways

1. **Design before code** — a clear architecture with defined interfaces makes implementation 3× faster and debugging 10× easier
2. **Pass pipeline pattern** — each optimization is an independent, composable pass with `analyze()` and `apply()` methods
3. **Correctness is non-negotiable** — every optimization must be tested against a reference implementation with tight numerical tolerances
4. **Choose your depth** — Project A covers breadth, B goes deep on compiler internals, C focuses on systems engineering
5. **Requirements first** — writing down what "done" looks like prevents scope creep and ensures you build the right thing

---

## Further Reading

- Lattner & Adve, "LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation" (2004) — the gold standard for pass-based compiler architecture
- Ansel et al., "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation" (2024)
- MLPerf Inference Benchmark Suite — industry-standard methodology for ML benchmarks
- Google's "How to Read a Paper" (Keshav, 2007) — useful for the capstone report writing

---

## Tomorrow's Teaser

Design is done. Tomorrow you start **building**. Day 69 provides skeleton code, implementation guidance, and the gotchas that will save you hours of debugging. Bring your architecture diagram — you're going to turn boxes into code.
