# Day 48: Compiler Testing & Verification

> **Phase III · Week 7 · Day 48 of 70 · 2.5 hours**

*"A compiler that's fast but wrong is worse than no compiler at all — you won't even know when it's lying to you."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 47: Contributing to TVM](day-47-contributing-to-tvm.md) | [Day 49: Stop & Reflect #4](day-49-stop-and-reflect-4.md) | [Week 7: TVM Advanced & MLC](README.md) | [Phase III: Apache TVM Deep Dive](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

ML compilers transform high-level models into low-level code through dozens of passes — operator fusion, constant folding, quantization, tiling, vectorization, memory planning. Each pass can introduce bugs: a fused kernel may silently drop a channel, a schedule may produce an out-of-bounds access that corrupts memory, or a quantization step may introduce catastrophic numerical drift. Unlike application bugs that crash visibly, **compiler bugs produce wrong numbers silently**. Your model runs, reports 92% accuracy, but it's actually 85% — and you don't discover the gap until production. This lesson covers the testing strategies that prevent this: numerical comparison, fuzzing, differential testing, and TVM's specific testing infrastructure.

---

## 1. The Correctness Problem in ML Compilers

### Why ML Compiler Bugs Are Especially Dangerous

```
ML Compiler Bug Taxonomy
════════════════════════

  Type 1: Crash Bugs (easy to detect)
  ├── Segfault in generated code
  ├── Shape mismatch assertion
  └── Out-of-memory during compilation
      Detection: automatic (program crashes)

  Type 2: Silent Numerical Bugs (DANGEROUS)
  ├── Fused kernel produces slightly wrong output
  ├── Quantization rounds in wrong direction
  ├── Memory layout mismatch (NCHW vs NHWC swap)
  └── Reduction over wrong axis
      Detection: requires explicit numerical validation

  Type 3: Performance Bugs (subtle)
  ├── Schedule produces correct but slow code
  ├── Missed fusion opportunity
  └── Suboptimal memory access pattern
      Detection: requires benchmarking against baseline

  The danger spectrum:
  ┌──────────────────────────────────────────────────────┐
  │  Easy to detect              Hard to detect          │
  │  ←──────────────────────────────────────────────→    │
  │  Crash    Shape error    1% drift    0.01% drift     │
  │    ✓         ✓             ✗              ✗✗         │
  │                                                      │
  │  A 0.01% per-layer drift across 100 layers =         │
  │  1 − 0.9999^100 ≈ 1% total error — enough to        │
  │  flip classification decisions silently               │
  └──────────────────────────────────────────────────────┘
```

### Numerical Error Sources

```
Sources of Numerical Divergence
═══════════════════════════════

  1. Floating-point non-associativity:
     (a + b) + c  ≠  a + (b + c)  in IEEE 754

     Example: sum([1e8, 1.0, -1e8])
       Left-to-right:  (1e8 + 1.0) + (-1e8) = 0.0  ← WRONG
       Right-to-left:  1e8 + (1.0 + (-1e8)) = 1.0  ← CORRECT

  2. Reduction order changes after tiling/parallelization:
     Sequential: s = x[0] + x[1] + x[2] + ... + x[N]
     Tiled:      s = (x[0]+...+x[31]) + (x[32]+...+x[63]) + ...
     → Different rounding at each partial sum

  3. FMA (fused multiply-add) availability:
     With FMA:  a*b + c  (single rounding)
     Without:   tmp = a*b (round), tmp + c (round again)
     → Results differ by up to 1 ULP

  4. Mixed precision accumulation:
     FP16 input × FP16 input → FP32 accumulator → FP16 output
     vs
     FP16 input × FP16 input → FP16 accumulator → FP16 output
     → Catastrophic difference for large reductions
```

The relative error from these sources follows:

$$\epsilon_{\text{total}} \leq n \cdot \epsilon_{\text{machine}} \cdot \kappa(A)$$

where $n$ is the number of operations, $\epsilon_{\text{machine}}$ is machine epsilon ($\approx 5.96 \times 10^{-8}$ for FP32, $\approx 9.77 \times 10^{-4}$ for FP16), and $\kappa(A)$ is the condition number of the computation.

---

## 2. Numerical Comparison: Getting `allclose` Right

### The `allclose` Function

The standard numerical comparison checks:

$$|a - b| \leq \text{atol} + \text{rtol} \cdot |b|$$

```python
import numpy as np

def smart_allclose(actual, expected, dtype="float32"):
    """Choose tolerances based on dtype and operation type."""
    tolerances = {
        # (atol, rtol)
        "float64": (1e-12, 1e-10),
        "float32": (1e-5,  1e-5),
        "float16": (1e-2,  1e-2),
        "bfloat16": (1e-1,  1e-1),   # BF16 has only 7-bit mantissa
        "int8":    (1,     0),        # Integer: exact or ±1
        "int32":   (0,     0),        # Integer: must be exact
    }
    atol, rtol = tolerances.get(dtype, (1e-5, 1e-5))
    return np.allclose(actual, expected, atol=atol, rtol=rtol)
```

### Tolerance Selection Guide

```
Tolerance Decision Matrix
═════════════════════════

  Operation          │ float32 atol │ float16 atol │ Why
  ───────────────────┼──────────────┼──────────────┼──────────────────
  Element-wise (relu)│ 1e-7         │ 1e-3         │ No accumulation
  MatMul (small K)   │ 1e-5         │ 5e-2         │ K additions
  MatMul (K=4096)    │ 1e-4         │ 5e-1         │ Many additions
  Convolution        │ 1e-4         │ 5e-2         │ K*R*S additions
  BatchNorm          │ 1e-4         │ 1e-1         │ Variance computation
  Softmax            │ 1e-5         │ 5e-2         │ exp + reduce
  LayerNorm          │ 1e-4         │ 1e-1         │ Mean + variance
  Full model (e2e)   │ 1e-3         │ 5e-1         │ Error compounds

  Rule of thumb for FP32:
  atol ≈ sqrt(K) × 1e-7  where K = number of accumulated terms

  For FP16 with FP32 accumulation:
  atol ≈ sqrt(K) × 1e-4
```

### Common Pitfalls

```python
# PITFALL 1: Using default tolerances for all dtypes
# BAD:
np.testing.assert_allclose(fp16_result, reference)  # atol=0, rtol=1e-7
# → Almost always fails for FP16!

# GOOD:
np.testing.assert_allclose(fp16_result, reference, atol=1e-2, rtol=1e-2)


# PITFALL 2: Comparing against wrong reference
# BAD: comparing TVM output against PyTorch output
#   (both are approximate — which is "correct"?)
# GOOD: compare against high-precision reference
ref_fp64 = compute_reference(inputs.astype("float64")).astype("float32")


# PITFALL 3: Ignoring output scale
# BAD: fixed atol for outputs of different magnitudes
# Consider: output values range from 1e-6 to 1e6
np.testing.assert_allclose(result, ref, atol=1e-5)
# → Passes for large values, fails for small values

# GOOD: use relative tolerance + small absolute tolerance
np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-7)
```

---

## 3. Testing Strategies

### The Testing Pyramid for ML Compilers

```
ML Compiler Testing Pyramid
════════════════════════════

              ┌──────────────┐
              │  Model-Level │  Full ResNet/BERT: slow but
              │   (E2E)      │  catches integration bugs
              └──────┬───────┘
                     │
           ┌─────────▼──────────┐
           │  Operator-Level     │  Single conv2d/matmul:
           │  (Integration)      │  catches numerical bugs
           └─────────┬──────────┘
                     │
      ┌──────────────▼──────────────┐
      │      Pass-Level (Unit)       │  Single pass on toy IR:
      │                              │  catches transform bugs
      └──────────────┬──────────────┘
                     │
  ┌──────────────────▼──────────────────┐
  │        IR Construction (Unit)        │  Build/validate IR nodes:
  │                                      │  catches API/type bugs
  └─────────────────────────────────────┘

  Coverage target per level:
  • IR construction:   >90% (fast, cheap)
  • Pass-level:        >80% per pass
  • Operator-level:    all ops × 2+ dtypes
  • Model-level:       top-10 models (ResNet, BERT, GPT-2, etc.)
```

### Unit Test: Pass Correctness

```python
import tvm
from tvm import relay

def test_fuse_ops_preserves_semantics():
    """FuseOps must not change computation results."""
    # Build a graph: conv2d → relu → conv2d → relu
    data = relay.var("data", shape=(1, 3, 32, 32), dtype="float32")
    w1 = relay.var("w1", shape=(16, 3, 3, 3), dtype="float32")
    w2 = relay.var("w2", shape=(32, 16, 3, 3), dtype="float32")

    conv1 = relay.nn.conv2d(data, w1, padding=(1, 1))
    relu1 = relay.nn.relu(conv1)
    conv2 = relay.nn.conv2d(relu1, w2, padding=(1, 1))
    relu2 = relay.nn.relu(conv2)

    mod_before = tvm.IRModule.from_expr(relu2)

    # Run FuseOps
    mod_after = relay.transform.FuseOps(fuse_opt_level=2)(mod_before)

    # Compile and run both versions
    np_data = np.random.uniform(-1, 1, (1, 3, 32, 32)).astype("float32")
    np_w1 = np.random.uniform(-0.1, 0.1, (16, 3, 3, 3)).astype("float32")
    np_w2 = np.random.uniform(-0.1, 0.1, (32, 16, 3, 3)).astype("float32")

    def run_mod(mod):
        with tvm.transform.PassContext(opt_level=0):  # no extra opts
            lib = relay.build(mod, target="llvm")
        rt = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.cpu()))
        rt.set_input("data", np_data)
        rt.set_input("w1", np_w1)
        rt.set_input("w2", np_w2)
        rt.run()
        return rt.get_output(0).numpy()

    out_before = run_mod(mod_before)
    out_after = run_mod(mod_after)

    # Must be bitwise identical (same dtype, same target)
    np.testing.assert_array_equal(out_before, out_after)
```

### Operator-Level Testing Pattern

```python
@tvm.testing.parametrize_targets("llvm", "cuda")
def test_matmul_against_numpy(target, dev):
    """Test matmul correctness across backends against NumPy reference."""
    M, K, N = 128, 256, 64

    a = relay.var("a", shape=(M, K), dtype="float32")
    b = relay.var("b", shape=(K, N), dtype="float32")
    out = relay.nn.dense(a, relay.op.transpose(b))

    mod = tvm.IRModule.from_expr(out)
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target)

    rt = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))

    a_np = np.random.uniform(-1, 1, (M, K)).astype("float32")
    b_np = np.random.uniform(-1, 1, (K, N)).astype("float32")

    rt.set_input("a", a_np)
    rt.set_input("b", b_np)
    rt.run()
    tvm_out = rt.get_output(0).numpy()

    # NumPy reference (uses higher-precision accumulation internally)
    ref = a_np @ b_np

    # Tolerance scales with K (number of accumulated terms)
    atol = np.sqrt(K) * 1e-6
    np.testing.assert_allclose(tvm_out, ref, atol=atol, rtol=1e-5)
```

---

## 4. Fuzzing ML Compilers

### What Fuzzing Catches

Fuzzing generates random (but valid) programs to find crashes and miscompilations:

```
Fuzzing Strategy for ML Compilers
══════════════════════════════════

  Graph Fuzzer: generates random Relay/Relax programs
  ┌────────────────────────────────────────────────────┐
  │  1. Pick random ops (conv2d, relu, add, reshape)   │
  │  2. Connect with valid shapes                      │
  │  3. Assign random dtypes                           │
  │  4. Compile with different opt_levels              │
  │  5. Compare outputs against reference              │
  └────────────────────────────────────────────────────┘

  Schedule Fuzzer: generates random TIR schedules
  ┌────────────────────────────────────────────────────┐
  │  1. Take known-correct TIR PrimFunc                │
  │  2. Apply random schedule transforms               │
  │     (split, reorder, vectorize, parallelize)       │
  │  3. Compile and run                                │
  │  4. Compare output against unscheduled version     │
  └────────────────────────────────────────────────────┘

  What fuzzing catches:
  ✓ Segfaults in generated code (bounds violations)
  ✓ Assertion failures in compilation passes
  ✓ Shape inference bugs (invalid intermediate shapes)
  ✓ Memory layout mismatches between passes
  ✓ Numerical divergence from reference
```

### Simple Graph Fuzzer

```python
import random
import tvm
from tvm import relay
import numpy as np

def generate_random_graph(depth=5, seed=42):
    """Generate a random but valid Relay graph for fuzz testing."""
    rng = random.Random(seed)
    shape = (1, rng.choice([3, 16, 32, 64]), 8, 8)
    x = relay.var("input", shape=shape, dtype="float32")
    current = x
    channels = shape[1]

    for _ in range(depth):
        op = rng.choice(["conv2d", "relu", "add_const", "pool"])
        if op == "conv2d":
            out_c = rng.choice([16, 32, 64])
            w = relay.var(f"w_{rng.randint(0,999)}", shape=(out_c, channels, 3, 3))
            current = relay.nn.conv2d(current, w, padding=(1, 1), channels=out_c)
            channels = out_c
        elif op == "relu":
            current = relay.nn.relu(current)
        elif op == "add_const":
            bias = relay.const(np.random.randn(1, channels, 1, 1).astype("float32"))
            current = relay.add(current, bias)
        elif op == "pool":
            current = relay.nn.avg_pool2d(current, pool_size=(2, 2),
                                           strides=(1, 1), padding=(0, 0))
    return relay.Function(relay.analysis.free_vars(current), current)


def fuzz_compile_and_check(n_programs=100):
    """Fuzz-test the compiler with random programs."""
    failures = []
    for seed in range(n_programs):
        try:
            func = generate_random_graph(depth=4, seed=seed)
            mod = tvm.IRModule.from_expr(func)
            # Compile at max optimization
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target="llvm")
            # Also compile at opt_level=0 and compare
            with tvm.transform.PassContext(opt_level=0):
                lib_ref = relay.build(mod, target="llvm")
            # If both compile, check outputs match
            # (exercise left to reader to fill in runtime comparison)
        except Exception as e:
            failures.append((seed, str(e)))

    print(f"Fuzzed {n_programs} programs, {len(failures)} failures")
    for seed, err in failures[:5]:
        print(f"  Seed {seed}: {err[:80]}")
```

---

## 5. Differential Testing Across Backends

### The Strategy

Differential testing compiles the **same program** for different targets and compares:

```
Differential Testing Architecture
══════════════════════════════════

  Input: Relay model + random test data
         │
    ┌────▼────────────────────────────────────────────┐
    │           Same Relay IR (frozen)                 │
    └────┬──────────┬──────────┬──────────┬───────────┘
         │          │          │          │
    ┌────▼───┐ ┌────▼───┐ ┌────▼───┐ ┌────▼───┐
    │  LLVM  │ │  CUDA  │ │ Vulkan │ │ OpenCL │
    │ (CPU)  │ │ (GPU)  │ │ (GPU)  │ │ (GPU)  │
    └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘
         │          │          │          │
    ┌────▼───┐ ┌────▼───┐ ┌────▼───┐ ┌────▼───┐
    │  out_0 │ │  out_1 │ │  out_2 │ │  out_3 │
    └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘
         │          │          │          │
         └──────────┴──────────┴──────────┘
                         │
                    ┌────▼────────────────────────┐
                    │  Pairwise allclose checks    │
                    │  with dtype-appropriate tol  │
                    │                              │
                    │  If ANY pair disagrees:      │
                    │  → Bug in one of the backends│
                    │  → Use CPU as reference      │
                    └─────────────────────────────┘
```

### Implementation

```python
import tvm
from tvm import relay
import numpy as np


def differential_test_model(mod, params, targets, test_inputs):
    """Run the same model on multiple backends and compare outputs."""
    outputs = {}
    for target_str in targets:
        target = tvm.target.Target(target_str)
        try:
            dev = tvm.device(target_str, 0)
            if not dev.exist:
                continue
        except Exception:
            continue

        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)

        rt = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
        for name, data in test_inputs.items():
            rt.set_input(name, data)
        rt.run()
        outputs[target_str] = rt.get_output(0).numpy()

    # Pairwise comparison (use CPU as reference)
    ref_target = "llvm"
    if ref_target not in outputs:
        return  # Can't test without CPU baseline

    ref = outputs[ref_target]
    for target_str, out in outputs.items():
        if target_str == ref_target:
            continue
        max_diff = np.max(np.abs(out - ref))
        rel_diff = np.max(np.abs(out - ref) / (np.abs(ref) + 1e-10))
        print(f"{ref_target} vs {target_str}: "
              f"max_abs_diff={max_diff:.2e}, max_rel_diff={rel_diff:.2e}")
        np.testing.assert_allclose(
            out, ref, atol=1e-4, rtol=1e-4,
            err_msg=f"Mismatch between {ref_target} and {target_str}"
        )


# Usage:
# differential_test_model(
#     mod, params,
#     targets=["llvm", "cuda", "vulkan"],
#     test_inputs={"data": np.random.randn(1, 3, 224, 224).astype("float32")}
# )
```

---

## 6. TVM's Testing Infrastructure

### Key Testing Utilities

```python
import tvm.testing

# 1. Parametrize across targets (skips unavailable ones)
@tvm.testing.parametrize_targets("llvm", "cuda", "vulkan")
def test_my_op(target, dev):
    pass  # dev is automatically created for target

# 2. Check numerical equality with good defaults
tvm.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

# 3. Known failure marking (for work-in-progress)
@tvm.testing.known_failing_targets("vulkan")
def test_broken_on_vulkan(target, dev):
    pass

# 4. Requires specific hardware
@tvm.testing.requires_cuda
def test_cuda_specific():
    pass

@tvm.testing.requires_gpu
def test_any_gpu():
    pass

# 5. Fixture for common model loading
@tvm.testing.parametrize_targets
def test_resnet(target, dev):
    mod, params = relay.testing.resnet.get_workload(
        num_layers=18, batch_size=1
    )
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target, params=params)
    # ... run and validate
```

### CI Pipeline Structure

```
TVM CI Matrix
═════════════

  Job                 │ What it tests                    │ Time
  ────────────────────┼──────────────────────────────────┼──────
  lint                │ clang-format, pylint, mypy       │ ~3 min
  build-cpu           │ cmake + ninja (CPU-only)         │ ~10 min
  build-gpu           │ cmake + ninja (CUDA enabled)     │ ~15 min
  test-cpu            │ pytest tests/ (no GPU required)  │ ~30 min
  test-gpu            │ pytest tests/ (CUDA + cuDNN)     │ ~45 min
  test-arm            │ Cross-compile + QEMU             │ ~20 min
  docs                │ Sphinx build + link check        │ ~10 min
  ────────────────────┼──────────────────────────────────┼──────
  Total (sequential)  │                                  │ ~2.5 hr
  Total (parallel CI) │                                  │ ~50 min
```

---

## Hands-On Exercises

### Exercise 1: Tolerance Experiment (20 min)

```python
import numpy as np

# Compute a large reduction in float32 and float16
K = 4096
a = np.random.randn(K).astype("float32")

sum_sequential = np.float32(0.0)
for x in a:
    sum_sequential += np.float32(x)

sum_numpy = np.sum(a)  # uses pairwise summation internally

print(f"Sequential sum:  {sum_sequential}")
print(f"NumPy sum:       {sum_numpy}")
print(f"Abs difference:  {abs(sum_sequential - sum_numpy)}")

# Repeat in float16 — observe much larger divergence
a16 = a.astype("float16")
sum16_seq = np.float16(0.0)
for x in a16:
    sum16_seq += np.float16(x)
sum16_np = np.sum(a16)
print(f"\nFP16 sequential: {sum16_seq}")
print(f"FP16 NumPy:      {sum16_np}")
print(f"FP16 abs diff:   {abs(float(sum16_seq) - float(sum16_np))}")
```

### Exercise 2: Write a Differential Test (30 min)

Take any Relay model (use `relay.testing.resnet.get_workload`) and:
1. Compile for `"llvm"` and `"llvm -mcpu=skylake-avx512"` (different instruction sets)
2. Run both with the same random input
3. Compare outputs — are they identical? Why or why not?

### Exercise 3: Build a Mini Fuzzer (30 min)

Extend the graph fuzzer from Section 4:
- Add `batch_norm`, `sigmoid`, and `concat` ops
- Track the maximum absolute difference between opt_level=0 and opt_level=3
- Report which random seeds produce the largest divergence

---

## Key Takeaways

1. **Silent numerical bugs** are the most dangerous compiler defects — the model runs fine but produces subtly wrong answers
2. **Tolerance selection is not trivial** — use $\text{atol} \approx \sqrt{K} \cdot \epsilon_{\text{machine}}$ where $K$ is the accumulation depth
3. **Differential testing** (same IR, different backends) catches backend-specific codegen bugs without needing a ground-truth oracle
4. **Fuzzing** generates random valid programs to find crashes and miscompilations that handwritten tests miss
5. **TVM's `tvm.testing` module** provides parametric target testing, hardware guards, and numerical comparison utilities out of the box
6. **Every optimization pass should be tested** both for correctness (output matches) and for idempotency (applying the pass twice gives the same result as once)

---

## Further Reading

- [TVM Testing Guide](https://tvm.apache.org/docs/contribute/ci.html) — CI and testing infrastructure
- [Csmith: A Randomized Compiler Testing Tool](https://embed.cs.utah.edu/csmith/) — seminal work in compiler fuzzing
- [Finding and Understanding Bugs in C Compilers (Yang et al., 2011)](https://dl.acm.org/doi/10.1145/1993316.1993532) — foundational differential testing paper
- [DeepTest (ICSE 2018)](https://arxiv.org/abs/1805.00089) — testing DNN systems
- [IEEE 754 Floating Point Standard](https://ieeexplore.ieee.org/document/8766229) — precision guarantees
- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) — essential reading

---

## Tomorrow: Stop & Reflect #4

Day 49 is a **consolidation day** marking the end of Phase III. You'll build a full concept map of the TVM stack, create a comparison matrix (TVM vs XLA vs Triton vs ORT vs MLIR ecosystem), take a self-assessment quiz, and verify you're ready for Phase IV's focus on inference optimization and deployment.
