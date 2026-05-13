# Day 69: Capstone Project — Part 2: Implementation

> **Phase V · Week 10 · Day 69 of 70 · 2.5 hours**

*"The gap between knowing every optimization technique and shipping a tool that applies them correctly is where most engineers stall. Today you cross that gap."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 68: Capstone Part 1](day-68-capstone-part1.md) | [Day 70: Capstone Part 3](day-70-capstone-part3.md) | [Week 10: Distributed Training & Capstone](README.md) | [Phase V: Training at Scale](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Implementation is where theory meets reality. You'll discover that fusion patterns that look clean on paper interact in surprising ways with PyTorch's autograd, that memory estimation is harder than expected because of CUDA allocator fragmentation, and that Triton codegen has subtle tile-size constraints. These are the lessons that **only come from building**. Every pitfall you hit today is one you'll avoid in production.

---

## 1. Project A Implementation: "mlopt" CLI

### Skeleton: Graph Capture Module

```python
# mlopt/capture/fx_capture.py
"""Graph capture using torch.fx and torch.export."""

import torch
from torch.fx import symbolic_trace, GraphModule
from torch.export import export

def capture_fx(model: torch.nn.Module, sample_input: torch.Tensor) -> GraphModule:
    """Capture model graph via torch.fx symbolic trace.

    Pros: Works with most Python control flow
    Cons: Cannot handle data-dependent control flow
    """
    try:
        gm = symbolic_trace(model)
        # Validate by running a forward pass
        ref_out = model(sample_input)
        cap_out = gm(sample_input)
        assert torch.allclose(ref_out, cap_out, atol=1e-6), \
            "Capture divergence detected"
        return gm
    except Exception as e:
        raise RuntimeError(f"FX capture failed: {e}") from e


def capture_export(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
) -> GraphModule:
    """Capture via torch.export (stricter, but more optimizable)."""
    exported = export(model, (sample_input,))
    return exported.module()
```

### Skeleton: Fusion Pass

```python
# mlopt/passes/fusion.py
"""Pattern-based operator fusion pass."""

import torch
from torch.fx import GraphModule, Node
from torch.fx.passes.utils.fuser_utils import topo_sort
from .base_pass import OptimizationPass

# Fusible element-wise ops
ELEMENTWISE_OPS = {
    torch.relu, torch.sigmoid, torch.tanh,
    torch.nn.functional.gelu, torch.nn.functional.silu,
    torch.add, torch.mul,
}

class FusionPass(OptimizationPass):
    def name(self) -> str:
        return "operator_fusion"

    def analyze(self, gm: GraphModule) -> dict:
        fusible_groups = self._find_fusible_chains(gm)
        return {
            "num_fusible_groups": len(fusible_groups),
            "total_fusible_ops": sum(len(g) for g in fusible_groups),
            "estimated_speedup": 1 + 0.05 * len(fusible_groups),
        }

    def apply(self, gm: GraphModule) -> GraphModule:
        groups = self._find_fusible_chains(gm)
        for group in groups:
            self._fuse_group(gm, group)
        gm.graph.lint()
        gm.recompile()
        return gm

    def _find_fusible_chains(self, gm: GraphModule) -> list[list[Node]]:
        """Find chains of element-wise ops that can be fused."""
        chains = []
        visited = set()

        for node in gm.graph.nodes:
            if node in visited or node.op != "call_function":
                continue
            if node.target not in ELEMENTWISE_OPS:
                continue

            # Grow chain forward
            chain = [node]
            visited.add(node)
            current = node

            while True:
                users = list(current.users.keys())
                if (len(users) == 1
                        and users[0].op == "call_function"
                        and users[0].target in ELEMENTWISE_OPS
                        and users[0] not in visited):
                    current = users[0]
                    chain.append(current)
                    visited.add(current)
                else:
                    break

            if len(chain) >= 2:  # Only fuse chains of 2+
                chains.append(chain)

        return chains

    def _fuse_group(self, gm: GraphModule, group: list[Node]):
        """Replace chain with a single fused operation."""
        # Create fused function from the chain
        ops = [n.target for n in group]
        first_input = group[0].args[0]

        def make_fused(*args, _ops=ops):
            x = args[0]
            for op in _ops:
                x = op(x)
            return x

        # Replace in graph
        with gm.graph.inserting_after(group[-1]):
            fused_node = gm.graph.call_function(make_fused, (first_input,))
            group[-1].replace_all_uses_with(fused_node)

        # Remove old nodes (reverse order to respect dependencies)
        for node in reversed(group):
            gm.graph.erase_node(node)
```

### Skeleton: CLI Entry Point

```python
# mlopt/cli.py
"""Command-line interface for mlopt."""

import argparse
import json
import time
import torch
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        prog="mlopt",
        description="End-to-end PyTorch model optimizer",
    )
    parser.add_argument("model", type=Path, help="Path to model .pt file")
    parser.add_argument("--input-shape", type=str, default="1,3,224,224",
                        help="Comma-separated input shape")
    parser.add_argument("--passes", type=str, default="fusion,quantization",
                        help="Comma-separated passes to apply")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path for optimized model")
    parser.add_argument("--report", type=Path, default=None,
                        help="Output path for JSON report")
    args = parser.parse_args()

    # Load model
    model = torch.load(args.model, weights_only=False)
    shape = [int(x) for x in args.input_shape.split(",")]
    sample = torch.randn(shape)

    # Capture
    from mlopt.capture.fx_capture import capture_fx
    gm = capture_fx(model, sample)

    # Build pass pipeline
    from mlopt.passes import build_pipeline
    pipeline = build_pipeline(args.passes.split(","))

    # Run
    t0 = time.perf_counter()
    optimized, reports = pipeline.run(gm, {})
    elapsed = time.perf_counter() - t0

    # Verify correctness
    with torch.no_grad():
        ref = model(sample)
        opt = optimized(sample)
        max_diff = (ref - opt).abs().max().item()

    result = {
        "optimization_time_s": elapsed,
        "passes_applied": [r["pass"] for r in reports],
        "max_output_diff": max_diff,
        "pass_reports": reports,
    }

    print(json.dumps(result, indent=2))
    if args.report:
        args.report.write_text(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
```

---

## 2. Project B Implementation: Custom torch.compile Backend

### Skeleton: Backend Registration

```python
# memfuse/backend.py
"""Custom torch.compile backend: memory-efficient fusion."""

import torch
from torch.fx import GraphModule
from torch._dynamo.backends.registry import register_backend

from .analysis.memory import estimate_peak_memory
from .fusion.grouper import find_fusion_groups
from .fusion.patterns import apply_fusion_patterns
from .codegen.triton_emitter import compile_fusion_groups

@register_backend
def memfuse(gm: GraphModule, example_inputs: list[torch.Tensor]):
    """Memory-efficient fusion backend for torch.compile."""
    # Step 1: Analyze current memory pressure
    baseline_mem = estimate_peak_memory(gm, example_inputs)
    print(f"[memfuse] Baseline peak memory: {baseline_mem / 1e6:.1f} MB")

    # Step 2: Find fusible groups
    groups = find_fusion_groups(gm)
    print(f"[memfuse] Found {len(groups)} fusion groups")

    # Step 3: Apply pattern-based fusion
    fused_gm = apply_fusion_patterns(gm, groups)

    # Step 4: Estimate memory savings
    optimized_mem = estimate_peak_memory(fused_gm, example_inputs)
    savings = (baseline_mem - optimized_mem) / baseline_mem * 100
    print(f"[memfuse] Optimized peak memory: {optimized_mem / 1e6:.1f} MB "
          f"({savings:.1f}% reduction)")

    # Step 5: Compile fusion groups to Triton kernels
    compiled_fn = compile_fusion_groups(fused_gm, example_inputs)

    return compiled_fn
```

### Skeleton: Triton Kernel Emitter

```python
# memfuse/codegen/triton_emitter.py
"""Generate Triton kernels from fused FX subgraphs."""

import torch
import triton
import triton.language as tl
from torch.fx import GraphModule

# Pre-built Triton templates for common fusion patterns

@triton.jit
def fused_linear_gelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Fused linear + GeLU: out = GeLU(x @ w + b)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulate matmul in FP32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        a = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :])
        b = tl.load(w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :])
        acc += tl.dot(a, b)

    # Fuse bias add + GeLU (no intermediate tensor!)
    bias = tl.load(b_ptr + offs_n)
    acc = acc + bias[None, :]

    # GeLU approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    x = acc
    cdf = 0.5 * (1.0 + tl.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    result = x * cdf

    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], result.to(tl.float16))


def compile_fusion_groups(gm: GraphModule, example_inputs):
    """Compile fused graph into callable with Triton kernels."""
    # Map fusion patterns to pre-compiled kernels
    kernel_registry = {
        ("linear", "gelu"): fused_linear_gelu_kernel,
        # Add more patterns as needed
    }

    def compiled_fn(*args):
        # Route each fusion group to its kernel
        # Fallback to eager PyTorch for unfused ops
        return gm(*args)  # Placeholder — real impl dispatches to kernels

    return compiled_fn
```

### Common Pitfalls for Project B

```
torch.compile Backend Pitfalls
═══════════════════════════════════════════════════════════════

  1. GUARD FAILURES
     ─────────────
     Problem:  Backend compiled for shape (32, 512) but called
               with (16, 512) → Dynamo recompiles from scratch
     Fix:      Use dynamic shapes: torch._dynamo.mark_dynamic(x, 0)
               Or design kernels with dynamic grid dimensions

  2. TENSOR ALIASING
     ───────────────
     Problem:  In-place ops (x.add_(1)) create aliases that
               confuse liveness analysis
     Fix:      Treat in-place ops as copies in your IR

  3. AUTOGRAD INTERACTION
     ────────────────────
     Problem:  Custom kernels break autograd gradient flow
     Fix:      Register custom autograd.Function with forward/backward
               OR only optimize inference (simpler)

  4. TRITON TILE SIZE CONSTRAINTS
     ────────────────────────────
     Problem:  BLOCK_M must be power of 2 and ≤ 128 on most GPUs
     Fix:      Use triton.autotune to search valid tile sizes

  5. NUMERICAL PRECISION
     ───────────────────
     Problem:  Fused kernel accumulates in FP16, original in FP32
     Fix:      Always accumulate reductions in FP32, cast output
```

---

## 3. Project C Implementation: Benchmark Suite

### Skeleton: Backend Abstraction

```python
# llm_bench/backends/base.py
"""Abstract backend interface for LLM serving."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class GenerationConfig:
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_p: float = 1.0

@dataclass
class GenerationResult:
    output_text: str
    output_tokens: int
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    peak_memory_mb: float


class LLMBackend(ABC):
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def load_model(self, model_name: str, **kwargs) -> None: ...

    @abstractmethod
    def generate(
        self, prompt: str, config: GenerationConfig
    ) -> GenerationResult: ...

    @abstractmethod
    def cleanup(self) -> None: ...
```

### Skeleton: PyTorch Eager Backend

```python
# llm_bench/backends/eager.py
"""Vanilla PyTorch eager-mode backend."""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import LLMBackend, GenerationConfig, GenerationResult

class EagerBackend(LLMBackend):
    def name(self) -> str:
        return "pytorch_eager"

    def load_model(self, model_name: str, **kwargs):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="cuda",
        )
        self._model.eval()

    def generate(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda")
        torch.cuda.reset_peak_memory_stats()

        # Measure TTFT: time until first new token generated
        t_start = time.perf_counter()
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.temperature > 0,
            )
        t_end = time.perf_counter()

        output_ids = outputs[0][inputs["input_ids"].shape[1]:]
        n_tokens = len(output_ids)
        e2e_ms = (t_end - t_start) * 1000
        peak_mb = torch.cuda.max_memory_allocated() / 1e6

        return GenerationResult(
            output_text=self._tokenizer.decode(output_ids, skip_special_tokens=True),
            output_tokens=n_tokens,
            ttft_ms=e2e_ms / n_tokens,  # Approximation without streaming
            tpot_ms=e2e_ms / max(n_tokens - 1, 1),
            e2e_ms=e2e_ms,
            peak_memory_mb=peak_mb,
        )

    def cleanup(self):
        del self._model
        torch.cuda.empty_cache()
```

### Skeleton: Benchmark Runner

```python
# llm_bench/runner.py
"""Main benchmark orchestrator."""

import json
import statistics
from dataclasses import asdict
from .backends.base import LLMBackend, GenerationConfig, GenerationResult

WORKLOADS = {
    "chat_short": {"prompt": "Explain quicksort in two sentences.", "max_tokens": 64},
    "chat_long": {"prompt": "Write a detailed guide to backpropagation.", "max_tokens": 512},
    "code_gen": {"prompt": "Write a Python B-tree implementation.", "max_tokens": 256},
    "summarize": {"prompt": "Summarize: " + "word " * 500, "max_tokens": 128},
}

def run_benchmark(
    backend: LLMBackend,
    model_name: str,
    workloads: list[str] | None = None,
    warmup: int = 3,
    iterations: int = 10,
) -> dict:
    """Run full benchmark suite and return results."""
    backend.load_model(model_name)
    selected = workloads or list(WORKLOADS.keys())
    results = {}

    for wl_name in selected:
        wl = WORKLOADS[wl_name]
        config = GenerationConfig(max_new_tokens=wl["max_tokens"])

        # Warmup (discard results)
        for _ in range(warmup):
            backend.generate(wl["prompt"], config)

        # Measure
        measurements: list[GenerationResult] = []
        for _ in range(iterations):
            r = backend.generate(wl["prompt"], config)
            measurements.append(r)

        # Aggregate statistics
        e2e_vals = [m.e2e_ms for m in measurements]
        tpot_vals = [m.tpot_ms for m in measurements]

        results[wl_name] = {
            "backend": backend.name(),
            "e2e_p50_ms": statistics.median(e2e_vals),
            "e2e_p95_ms": sorted(e2e_vals)[int(0.95 * len(e2e_vals))],
            "e2e_p99_ms": sorted(e2e_vals)[int(0.99 * len(e2e_vals))],
            "tpot_median_ms": statistics.median(tpot_vals),
            "throughput_tok_s": sum(m.output_tokens for m in measurements)
                                / (sum(e2e_vals) / 1000),
            "peak_memory_mb": max(m.peak_memory_mb for m in measurements),
            "n_measurements": len(measurements),
        }

    backend.cleanup()
    return results
```

---

## 4. Implementation Techniques from the Curriculum

Map your implementation to concepts from prior weeks:

```
Curriculum Concept → Implementation Mapping
═══════════════════════════════════════════════════════════════

  Week 2 (Graph IR)
  └─ FX Graph as the shared IR across all passes
     gm.graph.nodes → iterate, pattern-match, rewrite

  Week 4 (Operator Fusion)
  └─ Chain detection: find_fusible_chains() in FusionPass
     Pattern matching: conv→bn→relu, linear→gelu, etc.

  Week 5 (Scheduling)
  └─ Topological sort for pass ordering
     Data dependency analysis for safe node removal

  Week 6 (Memory Planning)
  └─ Liveness analysis: when is each tensor last used?
     Buffer reuse: assign freed buffers to new tensors

  Week 7 (Hardware Backends)
  └─ Triton codegen: FX nodes → GPU kernels
     Tile size selection: match L2 cache / shared memory

  Week 8 (Quantization)
  └─ PTQ calibration: run representative inputs
     Per-channel vs per-tensor decision

  Week 9 (torch.compile)
  └─ Backend registration: @register_backend
     Dynamo guards: handle shape dynamism

  Week 10 (Distributed)
  └─ Communication overlap (Project C benchmarking)
     Multi-GPU memory profiling
```

---

## 5. Benchmarking Methodology

For any project, benchmarking must be rigorous:

```python
# Rigorous benchmarking template
import torch
import time
import gc

def benchmark_fn(fn, *args, warmup=10, iterations=100):
    """Benchmark with proper warmup and CUDA synchronization."""
    # Warmup
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # Collect timings
    timings = []
    for _ in range(iterations):
        gc.disable()  # Prevent GC jitter
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        fn(*args)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        gc.enable()
        timings.append((t1 - t0) * 1000)  # ms

    return {
        "median_ms": sorted(timings)[len(timings) // 2],
        "mean_ms": sum(timings) / len(timings),
        "std_ms": (sum((t - sum(timings)/len(timings))**2
                       for t in timings) / len(timings)) ** 0.5,
        "p95_ms": sorted(timings)[int(0.95 * len(timings))],
        "min_ms": min(timings),
        "max_ms": max(timings),
    }
```

### Memory Profiling

```python
def profile_memory(fn, *args):
    """Profile GPU memory usage of a function call."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    baseline = torch.cuda.memory_allocated()
    fn(*args)
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated()
    current = torch.cuda.memory_allocated()

    return {
        "baseline_mb": baseline / 1e6,
        "peak_mb": peak / 1e6,
        "current_mb": current / 1e6,
        "delta_mb": (peak - baseline) / 1e6,
    }
```

---

## 6. Testing Your Implementation

### Unit Test Example (Project A)

```python
# tests/test_fusion_pass.py
import torch
import torch.nn as nn
from torch.fx import symbolic_trace
from mlopt.passes.fusion import FusionPass

class TwoLayerMLP(nn.Module):
    def forward(self, x):
        x = torch.relu(x)
        x = torch.sigmoid(x)  # Should fuse with relu
        return x

def test_fusion_finds_chain():
    model = TwoLayerMLP()
    gm = symbolic_trace(model)
    fp = FusionPass()

    analysis = fp.analyze(gm)
    assert analysis["num_fusible_groups"] >= 1
    assert analysis["total_fusible_ops"] >= 2

def test_fusion_preserves_output():
    model = TwoLayerMLP()
    gm = symbolic_trace(model)
    fp = FusionPass()

    fused_gm = fp.apply(gm)

    x = torch.randn(4, 16)
    ref = model(x)
    opt = fused_gm(x)
    assert torch.allclose(ref, opt, atol=1e-6)

def test_fusion_reduces_node_count():
    model = TwoLayerMLP()
    gm = symbolic_trace(model)
    before = len([n for n in gm.graph.nodes if n.op == "call_function"])

    fp = FusionPass()
    fused = fp.apply(gm)
    after = len([n for n in fused.graph.nodes if n.op == "call_function"])

    assert after < before, f"Expected fewer ops: {after} >= {before}"
```

---

## Hands-On Exercises

### Exercise 1: Implement Your Core Module (60 min)

For your chosen project, implement the most critical module:
- **Project A**: `FusionPass` — detect and fuse 2+ element-wise chains
- **Project B**: `find_fusion_groups()` — partition FX graph into fusible subgraphs
- **Project C**: `EagerBackend.generate()` — with accurate TTFT measurement using streaming

### Exercise 2: Write 3 Unit Tests (30 min)

Write tests that verify:
1. Correctness: optimized output matches reference
2. Structure: graph transformation produces expected node count
3. Edge case: empty graph, single-op graph, already-fused graph

### Exercise 3: Run and Profile (30 min)

Run your implementation on ResNet-18 (Project A/B) or GPT-2 small (Project C). Collect:
- Wall-clock time for the optimization/benchmark
- Memory usage before and after
- Speedup vs baseline

---

## Key Takeaways

1. **Start with the interface** — define `OptimizationPass`, `LLMBackend`, or `@register_backend` first, then implement
2. **Correctness before performance** — always validate against a reference before measuring speedup
3. **FX graph manipulation** — `gm.graph.nodes`, `node.replace_all_uses_with()`, `graph.erase_node()`, `gm.recompile()` are your core tools
4. **CUDA sync matters** — always `torch.cuda.synchronize()` before timing, or your measurements are meaningless
5. **Fuse the chain, not the pair** — detecting maximal fusible chains gives 3-5× more benefit than pairwise fusion
6. **Test at every layer** — unit tests for individual passes, integration tests for the pipeline, e2e tests for the CLI

---

## Further Reading

- PyTorch FX documentation: "Writing Graph Transformations" — essential reference for node manipulation
- Triton documentation: "Fused Attention" tutorial — real-world example of multi-op kernel fusion
- `torch._inductor` source code — production implementation of the patterns you're building
- MLPerf Inference Rules v4.1 — official methodology for LLM inference benchmarking

---

## Tomorrow's Teaser

You've built it. Tomorrow you **evaluate** it. Day 70 covers benchmarking analysis, writing a technical report, and a full retrospective of your 70-day journey. Plus: career paths, open problems, and what's next after this curriculum.
