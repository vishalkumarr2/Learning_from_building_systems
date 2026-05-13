# Day 51: Weight Compression & Pruning

> **Phase IV · Week 8 · Day 51 of 70 · 2.5 hours**

*"The art of pruning is knowing what to remove — the art of compression is knowing the network never needed it."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 50: Model Formats & ONNX](day-50-model-formats-onnx.md) | [Day 52: Knowledge Distillation](day-52-knowledge-distillation.md) | [Week 8: Model Formats & Runtimes](README.md) | [Phase IV: Inference & Deployment](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

A ResNet-50 has 25 million parameters. GPT-3 has 175 billion. LLaMA-3 70B has 70 billion. Moving these weights from memory to compute is **the** bottleneck in inference — not arithmetic. Pruning removes weights the model doesn't need; compression packs the remaining weights tighter. Together they can reduce model size by 10–100× while retaining 95–99% of accuracy. But not all pruning is equal: **unstructured pruning** zeroes individual weights (great compression, but needs sparse hardware), while **structured pruning** removes entire channels or heads (smaller dense model, runs anywhere). NVIDIA's A100 and H100 GPUs have dedicated **sparse tensor cores** that accelerate 2:4 sparsity patterns with zero overhead. Understanding these techniques — and how they interact with quantization — is critical for shipping models that fit in production constraints.

---

## 1. Pruning Taxonomy

```
                          Pruning Methods
  ════════════════════════════════════════════════════════
  
  By Granularity:
  ┌─────────────────┐  ┌────────────────┐  ┌─────────────┐
  │  Unstructured    │  │  Structured     │  │   N:M        │
  │  (individual w)  │  │  (channels/     │  │  Sparsity    │
  │                  │  │   heads/layers) │  │  (2:4, 4:8)  │
  │  ○ ● ○ ● ○      │  │  ██ ░░ ██ ░░   │  │  ● ● ○ ○    │
  │  ● ○ ○ ● ●      │  │  ██ ░░ ██ ░░   │  │  ○ ○ ● ●    │
  │  ○ ○ ● ○ ●      │  │  ██ ░░ ██ ░░   │  │  ● ○ ● ○    │
  │                  │  │                 │  │              │
  │  90%+ sparsity   │  │  50-70% pruned  │  │  50% exactly │
  │  Sparse HW only  │  │  Runs anywhere  │  │  Sparse cores│
  └─────────────────┘  └────────────────┘  └─────────────┘
  
  By Criterion:
  ┌─────────────────┐  ┌────────────────┐  ┌─────────────┐
  │  Magnitude       │  │  Movement       │  │  Gradient    │
  │  |w| < threshold │  │  Δw during      │  │  Fisher info │
  │  Simple, effective│  │  fine-tuning    │  │  or Taylor   │
  └─────────────────┘  └────────────────┘  └─────────────┘
```

---

## 2. Magnitude Pruning

The simplest and most widely used approach: **remove weights with the smallest absolute values**.

### 2.1 Global vs Local Thresholding

```python
import torch
import torch.nn.utils.prune as prune

model = torchvision.models.resnet50(pretrained=True)

# --- Local pruning: 30% of each layer independently ---
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name="weight", amount=0.3)

# --- Global pruning: 30% of ALL weights globally ---
parameters_to_prune = [
    (m, "weight") for m in model.modules() if isinstance(m, torch.nn.Conv2d)
]
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)

# Check sparsity
def sparsity(tensor):
    return (tensor == 0).sum().item() / tensor.numel() * 100

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        print(f"{name}: {sparsity(module.weight):.1f}% sparse")
```

### 2.2 The Lottery Ticket Hypothesis

> *"A randomly-initialized dense network contains a subnetwork (the 'winning ticket') that — when trained in isolation — can match the full network's accuracy."*  
> — Frankle & Carlin, 2019

$$\text{Winning ticket} = (m \odot \theta_0) \text{ where } m = \text{prune}(\theta_T)$$

The key insight: the **initialization** matters, not just the structure. This suggests that pruning discovers important structure present from the start.

### 2.3 Iterative Magnitude Pruning (IMP)

```
  Train to       Prune 20%     Rewind to        Train to       Prune 20%
  convergence    lowest |w|    init weights      convergence    lowest |w|
  ────────────▶ ──────────▶ ──────────────▶ ────────────▶ ──────────▶ ...
  100% weights    80%          80% (same mask)   80% weights    64%
       θ_T         mask m        m ⊙ θ_0           θ_T'         mask m'
```

---

## 3. Structured Pruning

Instead of individual weights, prune entire **filters, channels, or attention heads**:

```python
import torch.nn.utils.prune as prune

model = torchvision.models.resnet50(pretrained=True)

# Structured pruning: remove 40% of output channels (filters)
# based on L2 norm of the filter
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.ln_structured(module, name="weight", amount=0.4, n=2, dim=0)
        # dim=0 = output channels, n=2 = L2 norm criterion
```

### Channel Importance Scoring

For convolution layer with weight tensor $W \in \mathbb{R}^{C_{out} \times C_{in} \times k \times k}$:

$$\text{importance}(c) = \|W[c, :, :, :]\|_p = \left(\sum_{i,j,k} |W[c,i,j,k]|^p \right)^{1/p}$$

After removing channel $c$, the **downstream layers must also be adjusted**:

```
Structured Channel Pruning — Cascading Effect
══════════════════════════════════════════════

  Conv Layer N              Conv Layer N+1
  ┌───────────────┐         ┌───────────────┐
  │ Cout=64       │────────▶│ Cin=64        │
  │ Remove ch 17  │         │ Remove input  │
  │ Remove ch 42  │         │   ch 17, 42   │
  │ New: Cout=62  │────────▶│ New: Cin=62   │
  └───────────────┘         └───────────────┘
  
  Output shape changes ──▶ Next layer's input shape changes
  Both weight tensors shrink ──▶ True parameter reduction
```

---

## 4. N:M Sparsity — Hardware-Friendly Pruning

NVIDIA's Ampere (A100) and Hopper (H100) GPUs support **2:4 structured sparsity** natively in their Sparse Tensor Cores:

### 4.1 The 2:4 Pattern

In every group of 4 consecutive elements, exactly 2 must be zero:

$$\text{2:4 sparsity}: \quad [a, 0, b, 0], \quad [0, a, b, 0], \quad [a, 0, 0, b], \quad \ldots$$

There are $\binom{4}{2} = 6$ valid patterns per group.

```
 Dense weight row:     [0.5, 0.1, 0.8, 0.3, 0.2, 0.9, 0.4, 0.7]
                        ├─ group 1 ─┤  ├─ group 2 ─┤
 
 After 2:4 pruning:    [0.5, 0.0, 0.8, 0.0, 0.0, 0.9, 0.0, 0.7]
                        ├── 2 of 4 ──┤  ├── 2 of 4 ──┤
 
 Compressed storage:   values  = [0.5, 0.8, 0.9, 0.7]  (50% of original)
                        indices = [0,   2,   1,   3]     (2-bit per index)
```

### 4.2 Hardware Speedup

The Sparse Tensor Core **skips the zero multiplications**:

$$\text{Speedup} \approx 2\times \text{ on A100 Sparse Tensor Cores (FP16/INT8)}$$

```python
# PyTorch ASP (Automatic SParsity) for 2:4 pruning
from apex.contrib.sparsity import ASP

model = torchvision.models.resnet50(pretrained=True).cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Initialize sparse weights (prune to 2:4 pattern)
ASP.prune_trained_model(model, optimizer)

# Fine-tune with sparsity mask enforced
for epoch in range(5):
    for batch in train_loader:
        loss = criterion(model(batch["input"]), batch["target"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## 5. Sparse Tensor Storage Formats

Sparse weights need efficient storage to realize the size reduction:

```
Sparse Matrix Storage Formats
══════════════════════════════════════════════

  Dense (original):          CSR (Compressed Sparse Row):
  ┌─────────────────┐       values  = [5, 8, 3, 6, 2, 4]
  │ 5  0  0  8  0   │       col_idx = [0, 3, 1, 2, 0, 4]
  │ 0  3  6  0  0   │       row_ptr = [0, 2, 4, 6]
  │ 2  0  0  0  4   │
  └─────────────────┘       Storage: O(nnz) + O(rows)

  CSC (Compressed Sparse Column):    BSR (Block Sparse Row):
  values  = [5, 2, 3, 6, 8, 4]     ┌──────┐ ┌──────┐
  row_idx = [0, 2, 1, 1, 0, 2]     │ 5  0 │ │ 0  8 │  Block size: 2×2
  col_ptr = [0, 2, 3, 4, 5, 6]     │ 0  3 │ │ 6  0 │  Fewer index
  Storage: O(nnz) + O(cols)         └──────┘ └──────┘  overhead
```

| Format | Best For | Index Overhead | Hardware Support |
|--------|----------|----------------|------------------|
| CSR | Row-wise access, SpMV | O(rows + nnz) | CPU (MKL, SciPy) |
| CSC | Column-wise access | O(cols + nnz) | CPU (SuiteSparse) |
| BSR | Block-structured sparsity | O(blocks) | GPU (cuSPARSE) |
| 2:4 | Fine-grained structured | 2 bits per 4 elements | A100/H100 Sparse TC |

---

## 6. Combining Pruning with Quantization

The compression techniques **stack multiplicatively**:

$$\text{Compression} = \underbrace{\frac{1}{1 - p}}_{\text{pruning}} \times \underbrace{\frac{32}{\text{bits}}}_{\text{quantization}}$$

For 50% pruning + INT8 quantization: $2\times \times 4\times = 8\times$ compression.

```
Compression Pipeline — Prune → Quantize → Deploy
══════════════════════════════════════════════════════

  Full Model        Pruned Model       Quantized + Pruned
  (100 MB, FP32)    (50 MB, FP32)      (6.25 MB, INT8)
  ┌──────────┐      ┌──────────┐       ┌──────────┐
  │ ████████ │ ──▶  │ ██░░██░░ │  ──▶  │ █░█░     │
  │ ████████ │ 50%  │ ░░██░░██ │ INT8  │ ░█░█     │
  │ ████████ │prune │ ██░░██░░ │ quant │ █░█░     │
  └──────────┘      └──────────┘       └──────────┘
  25M params        12.5M params       12.5M × 1 byte
                                        = ~12 MB sparse
```

### SparseML — Unified Pruning + Quantization

```python
from sparseml.pytorch.optim import ScheduledModifierManager

recipe = """
version: 1.1.0
modifiers:
  - !GMPruningModifier
    params: ["re:.*conv.weight"]
    init_sparsity: 0.05
    final_sparsity: 0.80
    start_epoch: 0
    end_epoch: 30
    update_frequency: 1.0

  - !QuantizationModifier
    start_epoch: 30
    submodules: ["re:.*"]
    disable_quantization_observer_epoch: 35
    freeze_bn_stats_epoch: 35
"""

manager = ScheduledModifierManager.from_yaml(recipe)
optimizer = manager.modify(model, optimizer, steps_per_epoch=len(train_loader))
# Training loop applies pruning schedule + quantization-aware training
```

---

## Hands-On Exercises

### Exercise 1: Magnitude Pruning Sensitivity (25 min)
1. Load a pretrained ResNet-18 and evaluate baseline accuracy on CIFAR-10
2. Apply global unstructured pruning at 30%, 50%, 70%, 90% sparsity
3. Measure accuracy at each level — plot the accuracy vs. sparsity curve
4. Fine-tune the 70%-pruned model for 5 epochs — how much accuracy recovers?

### Exercise 2: Structured vs Unstructured (20 min)
1. Prune ResNet-18 to 50% using both unstructured (L1) and structured (L2 channel) pruning
2. Compare: parameter count, actual model file size, inference time on CPU
3. Why does unstructured pruning not reduce inference time on standard hardware?

### Exercise 3: 2:4 Sparsity Simulation (20 min)
```python
def apply_2_4_sparsity(tensor):
    """Zero out 2 smallest values in every group of 4."""
    t = tensor.clone().view(-1, 4)
    _, indices = t.abs().topk(2, dim=1, largest=False)
    t.scatter_(1, indices, 0.0)
    return t.view(tensor.shape)

# Apply to a conv layer's weights and measure:
# 1. What % of weights were actually changed?
# 2. How does output differ from dense forward pass?
```

### Exercise 4: Prune + Quantize Pipeline (15 min)
1. Start with a pretrained model, prune to 50% unstructured
2. Apply post-training quantization (INT8) on the pruned model
3. Measure final size and accuracy — compare against dense FP32 and INT8-only

---

## Key Takeaways

1. **Pruning removes redundancy** — neural networks are over-parameterized; 50–90% of weights can be removed with minimal accuracy loss
2. **Structured pruning gives real speedups** — removing entire channels/heads produces smaller dense models; unstructured pruning needs sparse hardware
3. **2:4 sparsity is the sweet spot** — exactly 50% sparsity with 2× speedup on A100/H100 Sparse Tensor Cores, near-zero accuracy loss
4. **Pruning + quantization stack** — 50% prune + INT8 = 8× compression; combine with distillation for maximum efficiency
5. **Lottery tickets are real** — the winning subnetwork exists at initialization, suggesting pruning discovers inherent structure
6. **Storage format matters** — CSR, BSR, or 2:4 compressed indices determine whether sparsity translates to actual memory savings

---

## Further Reading

- [Frankle & Carlin, "The Lottery Ticket Hypothesis" (ICLR 2019)](https://arxiv.org/abs/1803.03635)
- [Zhou et al., "Learning N:M Fine-grained Structured Sparse Neural Networks" (ICLR 2021)](https://arxiv.org/abs/2102.04010)
- [NVIDIA Ampere Sparsity Whitepaper](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/)
- [SparseML Documentation](https://github.com/neuralmagic/sparseml)
- [PyTorch Pruning Tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html)
- [Sanh et al., "Movement Pruning" (NeurIPS 2020)](https://arxiv.org/abs/2005.07683)

---

## Tomorrow's Preview

**Day 52: Knowledge Distillation** — Pruning removes weights; distillation transfers *knowledge*. You'll learn the teacher-student framework, Hinton's soft label trick, feature-based distillation, and how DistilBERT achieves 97% of BERT's accuracy at 60% the size — then combine distillation with pruning and quantization for maximum compression.
