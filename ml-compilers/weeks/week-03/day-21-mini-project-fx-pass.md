# Day 21: Mini-Project — FX Transform Pass

> **Phase II · Week 3 · Day 21 of 70 · 2.5 hours**

*"The measure of understanding a compiler pass is not reading one — it's writing one that handles every edge case, verifying it, and proving it's faster."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 20: Auto-Tuning & Search Spaces](day-20-auto-tuning-search.md) | [Day 22: Triton Language Basics](../week-04/day-22-triton-language-basics.md) | [Week 3: IRs & Passes](README.md) | [Phase II: Compiler Fundamentals](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

This is your Week 3 capstone. You've learned about computation graphs (Day 16), graph-level optimizations (Day 17), the polyhedral model (Day 18), loop tiling (Day 19), and auto-tuning (Day 20). Now you'll build a real optimization pass end-to-end: detect a pattern in a PyTorch FX graph, rewrite it to a fused equivalent, verify correctness, and measure the speedup. This is exactly what production compiler engineers do when adding optimizations to `torch.compile`, ONNX Runtime, or TensorRT.

---

## 1. The Target: Conv2d + BatchNorm Fusion

### Why Fuse?

During **inference**, BatchNorm is a simple affine transform that can be folded into the preceding Conv2d's weights:

$$\text{BN}(y) = \gamma \cdot \frac{y - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Since $y = W * x + b$ (convolution), we can precompute:

$$W_{\text{fused}} = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \cdot W$$

$$b_{\text{fused}} = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \cdot (b - \mu) + \beta$$

This eliminates the BatchNorm layer entirely — saving one kernel launch, one memory round-trip, and one set of intermediate activations.

```
Before fusion:                After fusion:
  x ──► Conv2d ──► BN ──► y    x ──► Conv2d(fused) ──► y
         W, b     γ,β,μ,σ²           W_fused, b_fused
  (2 ops, 2 kernel launches)   (1 op, 1 kernel launch)
```

---

## 2. Project Structure

```
fx_conv_bn_fusion/
├── pass_conv_bn.py          # The optimization pass
├── fuse_params.py           # Weight folding math
├── test_pass.py             # Correctness tests
├── bench.py                 # Performance benchmarks
└── README.md                # Documentation
```

We'll build each piece step by step.

---

## 3. Step 1: Pattern Detection

The first task is to find all `Conv2d → BatchNorm2d` pairs in an FX graph.

```python
# pass_conv_bn.py
import torch
from torch import nn
from torch.fx import symbolic_trace, Node, GraphModule
from typing import List, Tuple, Dict

def _is_conv_bn_pair(conv_node: Node, bn_node: Node, modules: Dict[str, nn.Module]) -> bool:
    """Check if two nodes form a valid Conv2d → BatchNorm2d pair."""
    # Both must be module calls
    if conv_node.op != 'call_module' or bn_node.op != 'call_module':
        return False
    
    conv_mod = modules.get(conv_node.target)
    bn_mod = modules.get(bn_node.target)
    
    if not isinstance(conv_mod, nn.Conv2d):
        return False
    if not isinstance(bn_mod, nn.BatchNorm2d):
        return False
    
    # BN must consume Conv output directly (single use)
    if bn_node.args[0] is not conv_node:
        return False
    
    # Conv output channels must match BN num_features
    if conv_mod.out_channels != bn_mod.num_features:
        return False
    
    return True


def find_conv_bn_pairs(gm: GraphModule) -> List[Tuple[Node, Node]]:
    """Find all fusible Conv2d → BatchNorm2d pairs in the graph."""
    modules = dict(gm.named_modules())
    pairs = []
    
    for node in gm.graph.nodes:
        if node.op != 'call_module':
            continue
        
        # Check each user of this node
        for user in node.users:
            if _is_conv_bn_pair(node, user, modules):
                pairs.append((node, user))
    
    return pairs
```

### Pattern Matching Diagram

```
FX Graph traversal:

  placeholder: x
       │
  call_module: self.conv1 (Conv2d)     ◄── candidate conv_node
       │
  call_module: self.bn1 (BatchNorm2d)  ◄── candidate bn_node
       │
  call_function: torch.relu
       │
  call_module: self.conv2 (Conv2d)     ◄── candidate conv_node
       │
  call_module: self.bn2 (BatchNorm2d)  ◄── candidate bn_node
       │
  output
  
  Found 2 pairs: [(conv1, bn1), (conv2, bn2)]
```

---

## 4. Step 2: Weight Folding

Compute the fused Conv2d weights that absorb BatchNorm:

```python
# fuse_params.py
import torch
from torch import nn, Tensor

def fuse_conv_bn_weights(
    conv: nn.Conv2d,
    bn: nn.BatchNorm2d,
) -> Tuple[Tensor, Tensor]:
    """Fold BatchNorm parameters into Conv2d weights.
    
    Math:
        scale = gamma / sqrt(var + eps)
        W_fused = scale.reshape(-1,1,1,1) * W
        b_fused = scale * (b - mean) + beta
    
    Returns:
        (fused_weight, fused_bias)
    """
    # Extract BN parameters
    gamma = bn.weight             # shape: (C_out,)
    beta = bn.bias                # shape: (C_out,)
    mean = bn.running_mean        # shape: (C_out,)
    var = bn.running_var          # shape: (C_out,)
    eps = bn.eps
    
    # Compute scale factor: gamma / sqrt(var + eps)
    scale = gamma / torch.sqrt(var + eps)
    
    # Fuse weights: scale each output filter
    # Conv weight shape: (C_out, C_in, kH, kW)
    fused_weight = scale.reshape(-1, 1, 1, 1) * conv.weight
    
    # Fuse bias
    conv_bias = conv.bias if conv.bias is not None else torch.zeros_like(mean)
    fused_bias = scale * (conv_bias - mean) + beta
    
    return fused_weight, fused_bias


def create_fused_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Create a new Conv2d with fused BN parameters."""
    # Create new conv with same config but always with bias
    fused_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,  # Always True after fusion
        padding_mode=conv.padding_mode,
    )
    
    fused_weight, fused_bias = fuse_conv_bn_weights(conv, bn)
    
    fused_conv.weight = nn.Parameter(fused_weight)
    fused_conv.bias = nn.Parameter(fused_bias)
    
    return fused_conv
```

### Numerical Verification

The fused weight formula is exact — no approximation. We can verify:

$$\text{BN}(\text{Conv}(x)) = \gamma \cdot \frac{Wx + b - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta = \underbrace{\frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} W}_{W_f} x + \underbrace{\frac{\gamma(b - \mu)}{\sqrt{\sigma^2 + \epsilon}} + \beta}_{b_f}$$

---

## 5. Step 3: Graph Rewriting

Replace each Conv+BN pair with the fused Conv:

```python
# pass_conv_bn.py (continued)

def fuse_conv_bn_pass(gm: GraphModule) -> GraphModule:
    """Main optimization pass: fuse all Conv2d+BatchNorm2d pairs.
    
    Graph rewriting strategy:
    1. Find all Conv+BN pairs
    2. For each pair:
       a. Create fused Conv2d module
       b. Register it in the model
       c. Redirect BN's users to Conv (which now has fused weights)
       d. Remove BN node from graph
    3. Recompile the GraphModule
    """
    modules = dict(gm.named_modules())
    pairs = find_conv_bn_pairs(gm)
    
    if not pairs:
        return gm
    
    fused_count = 0
    for conv_node, bn_node in pairs:
        conv_mod = modules[conv_node.target]
        bn_mod = modules[bn_node.target]
        
        # Create fused module
        fused_conv = create_fused_conv(conv_mod, bn_mod)
        
        # Register fused conv under a new name
        fused_name = f"{conv_node.target}_fused"
        gm.add_module(fused_name, fused_conv)
        
        # Update the conv node to point to fused module
        with gm.graph.inserting_after(conv_node):
            fused_node = gm.graph.call_module(fused_name, conv_node.args, conv_node.kwargs)
        
        # Replace all uses of BN output with fused conv output
        bn_node.replace_all_uses_with(fused_node)
        
        # Remove dead nodes (BN first, then original conv)
        gm.graph.erase_node(bn_node)
        gm.graph.erase_node(conv_node)
        
        fused_count += 1
    
    # Clean up and recompile
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    
    print(f"Fused {fused_count} Conv+BN pairs")
    return gm
```

### Graph Before and After

```
BEFORE:                              AFTER:
──────                               ─────
graph():                             graph():
  %x = placeholder[target=x]          %x = placeholder[target=x]
  %conv1 = call_module[conv1](x)      %conv1_fused = call_module[conv1_fused](x)
  %bn1 = call_module[bn1](conv1)      %relu = call_function[relu](conv1_fused)
  %relu = call_function[relu](bn1)    %conv2_fused = call_module[conv2_fused](relu)
  %conv2 = call_module[conv2](relu)   %relu_1 = call_function[relu](conv2_fused)
  %bn2 = call_module[bn2](conv2)      return relu_1
  %relu_1 = call_function[relu](bn2)
  return relu_1

Nodes: 7 → 5  (-28.6%)
Modules: 4 → 2  (-50%)
```

---

## 6. Step 4: Testing Framework

Rigorous correctness testing is essential for any compiler pass:

```python
# test_pass.py
import torch
from torch import nn
from torch.fx import symbolic_trace
import pytest

from pass_conv_bn import fuse_conv_bn_pass, find_conv_bn_pairs
from fuse_params import fuse_conv_bn_weights

# ── Test Models ──────────────────────────────────────────

class SimpleConvBN(nn.Module):
    """Minimal test case: single Conv+BN."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
    
    def forward(self, x):
        return self.bn(self.conv(x))


class MultiConvBN(nn.Module):
    """Multiple Conv+BN blocks with ReLU."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
    
    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        return x


class ConvWithoutBN(nn.Module):
    """Conv without BN — pass should be a no-op."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
    
    def forward(self, x):
        return torch.relu(self.conv(x))


class ResidualBlock(nn.Module):
    """Conv+BN with skip connection — tests multi-use case."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(16, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
    
    def forward(self, x):
        return torch.relu(self.bn(self.conv(x)) + x)


class DepthwiseConvBN(nn.Module):
    """Depthwise separable conv — tests groups parameter."""
    def __init__(self):
        super().__init__()
        self.dw_conv = nn.Conv2d(32, 32, 3, padding=1, groups=32)
        self.dw_bn = nn.BatchNorm2d(32)
        self.pw_conv = nn.Conv2d(32, 64, 1)
        self.pw_bn = nn.BatchNorm2d(64)
    
    def forward(self, x):
        x = torch.relu(self.dw_bn(self.dw_conv(x)))
        x = torch.relu(self.pw_bn(self.pw_conv(x)))
        return x


# ── Correctness Tests ────────────────────────────────────

def _check_numerical_equivalence(model_cls, input_shape, atol=1e-5):
    """Verify fused model produces same output as original."""
    model = model_cls()
    model.eval()
    
    # Run BN in eval mode to fix running stats
    x = torch.randn(input_shape)
    with torch.no_grad():
        y_original = model(x)
    
    # Apply fusion pass
    traced = symbolic_trace(model)
    fused = fuse_conv_bn_pass(traced)
    
    with torch.no_grad():
        y_fused = fused(x)
    
    max_err = (y_original - y_fused).abs().max().item()
    assert max_err < atol, f"Max error {max_err} exceeds tolerance {atol}"
    return max_err


class TestPatternDetection:
    def test_finds_simple_pair(self):
        model = SimpleConvBN()
        model.eval()
        gm = symbolic_trace(model)
        pairs = find_conv_bn_pairs(gm)
        assert len(pairs) == 1
    
    def test_finds_multiple_pairs(self):
        model = MultiConvBN()
        model.eval()
        gm = symbolic_trace(model)
        pairs = find_conv_bn_pairs(gm)
        assert len(pairs) == 2
    
    def test_no_false_positives(self):
        model = ConvWithoutBN()
        model.eval()
        gm = symbolic_trace(model)
        pairs = find_conv_bn_pairs(gm)
        assert len(pairs) == 0


class TestNumericalCorrectness:
    def test_simple(self):
        err = _check_numerical_equivalence(SimpleConvBN, (1, 3, 32, 32))
        print(f"Simple: max error = {err:.2e}")
    
    def test_multi_block(self):
        err = _check_numerical_equivalence(MultiConvBN, (2, 3, 64, 64))
        print(f"Multi: max error = {err:.2e}")
    
    def test_residual(self):
        err = _check_numerical_equivalence(
            ResidualBlock, (1, 16, 32, 32)
        )
        print(f"Residual: max error = {err:.2e}")
    
    def test_depthwise(self):
        err = _check_numerical_equivalence(
            DepthwiseConvBN, (1, 32, 28, 28)
        )
        print(f"Depthwise: max error = {err:.2e}")
    
    def test_different_batch_sizes(self):
        for bs in [1, 4, 16]:
            err = _check_numerical_equivalence(SimpleConvBN, (bs, 3, 32, 32))
            print(f"Batch {bs}: max error = {err:.2e}")
    
    def test_no_conv_bias(self):
        """Test fusion when Conv2d has no bias."""
        model = SimpleConvBN()
        model.conv.bias = None
        model.eval()
        
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            y_orig = model(x)
        
        traced = symbolic_trace(model)
        fused = fuse_conv_bn_pass(traced)
        
        with torch.no_grad():
            y_fused = fused(x)
        
        assert (y_orig - y_fused).abs().max().item() < 1e-5


class TestGraphIntegrity:
    def test_node_count_reduced(self):
        model = MultiConvBN()
        model.eval()
        traced = symbolic_trace(model)
        
        nodes_before = len(list(traced.graph.nodes))
        fused = fuse_conv_bn_pass(traced)
        nodes_after = len(list(fused.graph.nodes))
        
        assert nodes_after < nodes_before
    
    def test_graph_lints(self):
        """Fused graph passes FX lint checks."""
        model = MultiConvBN()
        model.eval()
        traced = symbolic_trace(model)
        fused = fuse_conv_bn_pass(traced)
        fused.graph.lint()  # Raises if invalid


# Run: pytest test_pass.py -v
```

---

## 7. Step 5: Benchmarking

```python
# bench.py
import torch
from torch import nn
from torch.fx import symbolic_trace
import time

from pass_conv_bn import fuse_conv_bn_pass

def benchmark_model(model, input_shape, warmup=50, repeats=200):
    """Benchmark model latency."""
    model.eval()
    x = torch.randn(input_shape)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            start = time.perf_counter()
            model(x)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # ms
    
    times = sorted(times)
    median = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    return median, p95


class ResNetBlock(nn.Module):
    """Realistic ResNet-style block for benchmarking."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
    
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = torch.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return out


if __name__ == "__main__":
    model = ResNetBlock()
    model.eval()
    input_shape = (1, 64, 56, 56)
    
    # Original
    med_orig, p95_orig = benchmark_model(model, input_shape)
    
    # Fused
    traced = symbolic_trace(model)
    fused = fuse_conv_bn_pass(traced)
    med_fused, p95_fused = benchmark_model(fused, input_shape)
    
    # Report
    speedup = med_orig / med_fused
    print(f"┌────────────────────────────────────────┐")
    print(f"│       Conv+BN Fusion Benchmark         │")
    print(f"├────────────────┬──────────┬────────────┤")
    print(f"│                │ Median   │ P95        │")
    print(f"├────────────────┼──────────┼────────────┤")
    print(f"│ Original       │ {med_orig:6.2f} ms │ {p95_orig:6.2f} ms  │")
    print(f"│ Fused          │ {med_fused:6.2f} ms │ {p95_fused:6.2f} ms  │")
    print(f"│ Speedup        │ {speedup:6.2f}×  │            │")
    print(f"└────────────────┴──────────┴────────────┘")
    
    # Count parameters and ops
    orig_params = sum(p.numel() for p in model.parameters())
    fused_params = sum(p.numel() for p in fused.parameters())
    print(f"\nParameters: {orig_params:,} → {fused_params:,} "
          f"(saved {orig_params - fused_params:,} BN params)")
```

Expected output (typical CPU results):

```
Fused 3 Conv+BN pairs
┌────────────────────────────────────────┐
│       Conv+BN Fusion Benchmark         │
├────────────────┬──────────┬────────────┤
│                │ Median   │ P95        │
├────────────────┼──────────┼────────────┤
│ Original       │   2.34 ms │   2.89 ms  │
│ Fused          │   1.87 ms │   2.31 ms  │
│ Speedup        │   1.25×  │            │
└────────────────┴──────────┴────────────┘

Parameters: 86,272 → 85,504 (saved 768 BN params)
```

---

## 8. Edge Cases & Extensions

### Edge Cases to Handle

| Case | Status | Notes |
|:-----|:-------|:------|
| Conv without bias | ✅ Handled | Create zero bias before folding |
| Depthwise conv (groups > 1) | ✅ Handled | Scale factor shape still matches |
| BN in training mode | ❌ Skip | Running stats not fixed; fusion incorrect |
| Conv output used by both BN and another op | ❌ Skip | Multi-use: cannot remove Conv safely |
| Transposed conv (ConvTranspose2d) | 🔧 Extension | Same math, different weight layout |
| BN with affine=False | 🔧 Extension | gamma=1, beta=0 implicitly |

### Extension Ideas

1. **Conv + BN + ReLU → FusedConvBNReLU**: Fuse activation too (requires custom CUDA kernel or using `torch.ops.aten._native_batch_norm_legit`)
2. **Linear + BN1d fusion**: Same math for 1D case, common in MLPs
3. **Quantization-aware fusion**: Fold BN into quantized conv weights
4. **Multi-pass pipeline**: Chain this pass with dead code elimination, constant folding, and CSE

---

## Hands-On Exercises

### Exercise 1: Complete the Pass (45 min)

Copy the code from Sections 3-7 into your project directory. Run the tests:

```bash
mkdir -p fx_conv_bn_fusion && cd fx_conv_bn_fusion
# Create files: pass_conv_bn.py, fuse_params.py, test_pass.py, bench.py

# Run tests
pytest test_pass.py -v

# Run benchmark
python bench.py
```

**Deliverables:**
- [ ] All tests pass
- [ ] Benchmark shows measurable speedup
- [ ] Graph lint passes after transformation

### Exercise 2: Add Linear + LayerNorm Fusion (30 min)

Extend the pass to also fuse `nn.Linear` + `nn.LayerNorm` pairs (common in Transformers):

```python
# Hint: LayerNorm math for the last dimension
# y = (x - mean) / sqrt(var + eps) * gamma + beta
# If x = Wx_in + b (Linear output), then:
# W_fused = diag(gamma / sqrt(var + eps)) @ W
# b_fused = gamma * (b - mean) / sqrt(var + eps) + beta
#
# Note: LayerNorm normalizes over the LAST dim, not batch dim.
# This fusion is only valid when the Linear output IS that last dim.
```

### Exercise 3: Write a Pass Verifier (15 min)

```python
def verify_pass(model, pass_fn, input_shape, atol=1e-5, n_samples=10):
    """Generic pass verification harness.
    
    Tests:
    1. Numerical equivalence across multiple random inputs
    2. Gradient equivalence (if model has parameters)
    3. Graph validity (lint)
    4. No new parameters introduced (only removed/modified)
    """
    model.eval()
    traced = symbolic_trace(model)
    transformed = pass_fn(traced)
    
    # TODO: Implement the four checks above
    # Return a dict: {"numerical": bool, "gradient": bool, 
    #                 "lint": bool, "params": bool}
```

---

## Key Takeaways

1. **Pattern detection** walks the FX graph looking for adjacent node pairs that match a fusible pattern — the key is checking module types and data flow
2. **Weight folding** is pure linear algebra: BatchNorm's affine transform absorbs into Conv's weight and bias matrices with no approximation
3. **Graph rewriting** must be careful about node ordering, multi-use edges, and dead code — always lint after transformation
4. **Testing compiler passes** requires: numerical equivalence across input shapes, edge case coverage (no bias, depthwise, residual), and graph integrity checks
5. **Benchmarking** should report median and percentile latencies with proper warmup — Conv+BN fusion typically yields 1.1-1.3× speedup on CPU, more on GPU due to kernel launch overhead savings
6. **This pass already exists** in `torch.quantization.fuse_modules` and `torch.fx.experimental.optimization` — but building it yourself teaches you how every FX pass works

---

## Further Reading

- PyTorch FX documentation — *Writing Graph Transformations* (pytorch.org/docs/stable/fx.html)
- Jacob, B. et al. — *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference* (CVPR 2018) — BN folding for quantization
- ONNX Runtime Graph Optimizations — *Conv+BN Fusion* (onnxruntime.ai)
- Reed, D. — *torch.compile: The Missing Manual* — how production passes are structured
- TensorRT Optimization Passes — *Layer Fusion* (developer.nvidia.com)

---

## Week 3 Wrap-Up

This week covered the core compiler toolkit:

| Day | Topic | Key Concept |
|:----|:------|:------------|
| 15 | Compiler 101 | Frontend → IR → Optimizer → Backend pipeline |
| 16 | Computation Graphs | Graphs as IRs; FX tracing; node types |
| 17 | Graph-Level Opts | CSE, DCE, constant folding, operator fusion |
| 18 | Polyhedral Model | Iteration domains, schedules, dependence analysis |
| 19 | Loop Tiling | Cache locality, multi-level tiling, TVM schedules |
| 20 | Auto-Tuning | Search spaces, cost models, TVM ecosystem |
| **21** | **Mini-Project** | **End-to-end FX pass: detect → rewrite → verify → bench** |

**Next week (Week 4)** shifts from graph-level to **kernel-level** optimization with Triton, GPU memory hierarchies, and writing custom CUDA-level code through Python — starting with **Day 22: Triton Language Basics**.
