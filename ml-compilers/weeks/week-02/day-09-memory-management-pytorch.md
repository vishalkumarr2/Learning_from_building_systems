# Day 9: Memory Management in PyTorch

> **Phase I · Week 2 · Day 9 of 70 · 2.5 hours**

*"OOM is not a memory problem — it's a fragmentation problem you haven't visualized yet."*

| Previous | Next | Week | Phase | Curriculum |
|----------|------|------|-------|------------|
| [Day 8: PyTorch Under the Hood](day-08-pytorch-under-the-hood.md) | [Day 10: Custom C++ Extensions](day-10-custom-cpp-extensions.md) | [Week 2: PyTorch Internals](README.md) | Phase I: Foundations | [Curriculum Home](../../STUDY-PLAN.md) |

---

## Why This Matters

A 7B parameter model in fp16 occupies ~14 GB. Training it with Adam optimizer states,
gradients, and activations can require 4–6× that. When you hit OOM on a 40 GB A100,
the bottleneck is rarely total memory — it's **fragmentation** in the caching allocator.
Understanding how PyTorch manages GPU memory lets you fit larger batches, debug OOM
errors in minutes instead of hours, and make informed tradeoffs with gradient
checkpointing.

---

## 1. The CUDA Caching Allocator

PyTorch does **not** call `cudaMalloc` / `cudaFree` for every tensor. These CUDA
calls are expensive (~1 ms each) and synchronize the device. Instead, PyTorch uses
a **caching allocator** that maintains a free-list of previously allocated blocks.

```
cudaMalloc lifecycle (without caching):
───────────────────────────────────────
  alloc 256MB → [cudaMalloc: 1.2ms + sync]
  free  256MB → [cudaFree:  0.8ms + sync]
  alloc 256MB → [cudaMalloc: 1.2ms + sync]   ← pays again!

PyTorch caching allocator:
──────────────────────────
  alloc 256MB → [cudaMalloc: 1.2ms]           ← first time only
  free  256MB → [moved to free list: ~0μs]
  alloc 256MB → [found in free list: ~0μs]    ← instant reuse!
```

### 1.1 Memory Pools

The allocator maintains two pools, separated by block size:

```
┌─────────────────────────────────────────────┐
│           CUDA Caching Allocator            │
├──────────────────┬──────────────────────────┤
│   Small Pool     │     Large Pool           │
│   (≤ 1 MB)       │     (> 1 MB)             │
│                  │                          │
│  ┌────┐ ┌────┐  │  ┌──────────┐ ┌────────┐ │
│  │512B│ │ 1K │  │  │  2 MB    │ │ 20 MB  │ │
│  └────┘ └────┘  │  └──────────┘ └────────┘ │
│  ┌────┐ ┌────┐  │  ┌──────────┐            │
│  │256B│ │768B│  │  │  512 MB  │  (segments │
│  └────┘ └────┘  │  └──────────┘   from     │
│                  │                cudaMalloc)│
├──────────────────┴──────────────────────────┤
│  Segment: large contiguous cudaMalloc chunk  │
│  Block:   sub-region within a segment        │
└─────────────────────────────────────────────┘
```

**Segments** are the actual `cudaMalloc` allocations (typically 2 MB or 20 MB+).
**Blocks** are carved out of segments to satisfy individual tensor requests.

---

## 2. Block Splitting and Merging

When a tensor is allocated, the allocator finds the smallest free block that fits.
If the block is significantly larger than needed, it **splits**:

```
Before allocation of 3 MB:
┌──────────────────────────────┐
│        Free: 8 MB            │
└──────────────────────────────┘

After split:
┌───────────┬──────────────────┐
│ Used: 3MB │   Free: 5 MB     │
└───────────┴──────────────────┘

After freeing the 3 MB tensor:
┌───────────┬──────────────────┐
│ Free: 3MB │   Free: 5 MB     │
└───────────┴──────────────────┘
        ↓ merge adjacent free blocks
┌──────────────────────────────┐
│        Free: 8 MB            │
└──────────────────────────────┘
```

### 2.1 When Merging Fails → Fragmentation

The problem arises when free blocks are **not adjacent**:

```
Fragmented state — 6 MB free but cannot allocate 4 MB:

┌──────┬──────┬──────┬──────┬──────┬──────┐
│Used  │Free  │Used  │Free  │Used  │Free  │
│ 2MB  │ 2MB  │ 2MB  │ 2MB  │ 2MB  │ 2MB  │
└──────┴──────┴──────┴──────┴──────┴──────┘
         ↑              ↑              ↑
    Can't merge — non-adjacent free blocks

Total free: 6 MB, but max contiguous: 2 MB → OOM for 4 MB request!
```

This is the **#1 cause** of unexpected OOM errors in training.

---

## 3. Inspecting Memory State

### 3.1 Basic Statistics

```python
import torch

x = torch.randn(1000, 1000, device='cuda')  # ~4 MB

print(f"Allocated:  {torch.cuda.memory_allocated() / 1e6:.1f} MB")
print(f"Reserved:   {torch.cuda.memory_reserved() / 1e6:.1f} MB")
print(f"Max alloc:  {torch.cuda.max_memory_allocated() / 1e6:.1f} MB")

# allocated = memory used by tensors
# reserved  = memory held by caching allocator (≥ allocated)
# The gap (reserved - allocated) = free blocks in the cache
```

### 3.2 Detailed Statistics

```python
stats = torch.cuda.memory_stats()

# Key fields:
print(f"Active blocks:     {stats['active.all.current']}")
print(f"Active bytes:      {stats['active_bytes.all.current'] / 1e6:.1f} MB")
print(f"Allocated segments:{stats['segment.all.current']}")
print(f"Num allocs:        {stats['allocation.all.current']}")
print(f"Num OOM retries:   {stats.get('num_oom_retries', 0)}")

# Fragmentation indicator:
reserved = torch.cuda.memory_reserved()
allocated = torch.cuda.memory_allocated()
frag_ratio = 1.0 - (allocated / reserved) if reserved > 0 else 0
print(f"Fragmentation:     {frag_ratio:.1%}")
# > 30% fragmentation = likely OOM risk
```

### 3.3 Memory Snapshot (PyTorch 2.1+)

```python
# Record a full memory snapshot for visualization
torch.cuda.memory._record_memory_history(max_entries=100000)

# ... run your training step ...
model(x).sum().backward()
optimizer.step()

# Save snapshot
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)  # stop

# Visualize at: https://pytorch.org/memory_viz
# Upload the .pickle file → interactive timeline
```

The snapshot shows every allocation/free event with Python stack traces,
letting you identify exactly which line of code caused fragmentation.

---

## 4. OOM Debugging Workflow

When you hit `CUDA out of memory`, follow this checklist:

```
Step 1: Is it a real capacity problem?
─────────────────────────────────────
  total_params = sum(p.numel() for p in model.parameters())
  param_bytes  = total_params * 4  # fp32
  # Add: gradients (1x), optimizer (2x for Adam)
  # Add: activations (batch_size × layers × hidden²)
  # Compare to GPU memory

Step 2: Check fragmentation
───────────────────────────
  torch.cuda.memory_stats()['num_alloc_retries']  # > 0 = fragmentation
  reserved - allocated  # large gap = fragmentation

Step 3: Try max_split_size_mb
─────────────────────────────
  # Prevents the allocator from splitting large blocks
  PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

Step 4: Nuclear option — empty cache
─────────────────────────────────────
  torch.cuda.empty_cache()  # returns all free blocks to CUDA
  # Only helps if fragmentation was the issue
  # Does NOT free tensors still referenced in Python

Step 5: Record memory snapshot
──────────────────────────────
  # Use the snapshot tool from §3.3 to find the culprit
```

### 4.1 The `max_split_size_mb` Knob

```
Without max_split_size_mb (default):
  Request 4 MB from a 20 MB block → split into [4 MB used][16 MB free]

With max_split_size_mb=128:
  Request 4 MB from a 200 MB block → DON'T split, waste 196 MB
  BUT: the 200 MB block stays contiguous for future large allocations!
```

Set via environment variable:

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# OR
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64,garbage_collection_threshold:0.8
```

**Tradeoff:** Higher values reduce fragmentation but waste more memory per allocation.

---

## 5. Gradient Checkpointing

Gradient checkpointing (activation checkpointing) is a **compute-memory tradeoff**:
instead of saving all intermediate activations for backward, recompute them on the fly.

### 5.1 Memory Analysis

For a transformer with $L$ layers, hidden size $H$, sequence length $S$, batch $B$:

$$M_{\text{activations}} \approx L \times B \times S \times H \times 4 \text{ bytes (fp32)}$$

For a 24-layer model with $H=1024$, $S=2048$, $B=8$:

$$M = 24 \times 8 \times 2048 \times 1024 \times 4 \approx 1.5 \text{ GB}$$

With checkpointing every $\sqrt{L}$ layers:

$$M_{\text{checkpoint}} \approx \sqrt{L} \times B \times S \times H \times 4 \approx 310 \text{ MB}$$

**Saves ~5× memory, costs ~33% extra compute** (one extra forward pass per segment).

### 5.2 Implementation

```python
import torch
from torch.utils.checkpoint import checkpoint

class CheckpointedTransformer(torch.nn.Module):
    def __init__(self, num_layers=24, d_model=1024):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(d_model, nhead=16)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            # Recompute this layer's activations during backward
            x = checkpoint(layer, x, use_reentrant=False)
        return x

# Memory comparison:
model_no_ckpt = CheckpointedTransformer().cuda()
model_ckpt    = CheckpointedTransformer().cuda()

x = torch.randn(8, 2048, 1024, device='cuda')
torch.cuda.reset_peak_memory_stats()

# Without checkpointing — stores all activations
y = model_no_ckpt(x).sum()
y.backward()
peak_no_ckpt = torch.cuda.max_memory_allocated()

torch.cuda.reset_peak_memory_stats()

# With checkpointing — recomputes activations
y = model_ckpt(x).sum()
y.backward()
peak_ckpt = torch.cuda.max_memory_allocated()

print(f"Without checkpointing: {peak_no_ckpt / 1e9:.2f} GB")
print(f"With checkpointing:    {peak_ckpt / 1e9:.2f} GB")
```

### 5.3 Selective Checkpointing

Not all layers benefit equally. Checkpoint layers with **large activation tensors**
but **cheap recomputation**:

```python
# Checkpoint every other layer (common heuristic)
def forward(self, x):
    for i, layer in enumerate(self.layers):
        if i % 2 == 0:  # checkpoint even layers
            x = checkpoint(layer, x, use_reentrant=False)
        else:
            x = layer(x)
    return x
```

---

## 6. Memory Timeline Diagram

A typical training step's memory timeline:

```
GPU Memory
 ▲
 │                    ┌─── backward starts
 │         ┌─ peak   │      gradients accumulate
 │         │         ▼      activations freed
 │    ┌────┤    ┌────────┐
 │    │act.│    │gradients│
 │    │    │    │ + recomp│
 │   ┌┤    │    │        │   ┌──── optimizer step
 │   ││    │    │        │   │  (momentary 2× for Adam states)
 │  ┌┤│    │    │        ├───┤
 │  │││    │    │        │   │
 │──┤│├────┤────┤────────┤───┼──── params + optimizer states (constant)
 │  │params│    │        │   │
 └──┴┴─────┴────┴────────┴───┴───▶ time
    fwd         bwd         opt
```

---

## Hands-On Exercises

### Exercise 1: Memory Profiling (30 min)

```python
# Profile peak memory of a ResNet-50 forward + backward pass
# at different batch sizes. Plot batch_size vs peak_memory.

import torch
import torchvision.models as models

model = models.resnet50().cuda()
results = []

for bs in [1, 2, 4, 8, 16, 32, 64]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    try:
        x = torch.randn(bs, 3, 224, 224, device='cuda')
        y = model(x).sum()
        y.backward()
        peak = torch.cuda.max_memory_allocated() / 1e9
        results.append((bs, peak))
        print(f"BS={bs:3d}: {peak:.2f} GB")
        del x, y
    except RuntimeError as e:
        print(f"BS={bs:3d}: OOM!")
        break

# QUESTION: Is peak memory linear in batch size? Why or why not?
```

### Exercise 2: Fragmentation Experiment (30 min)

```python
# Deliberately create fragmentation and measure its effect.

import torch

def create_fragmentation():
    """Allocate alternating tensors, free every other one."""
    tensors = []
    for i in range(100):
        t = torch.randn(1000, 1000, device='cuda')  # ~4 MB each
        tensors.append(t)

    # Free even-indexed tensors → creates holes
    for i in range(0, 100, 2):
        tensors[i] = None

    # Now try to allocate a single large tensor
    reserved = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()
    free_in_cache = reserved - allocated
    print(f"Free in cache: {free_in_cache / 1e6:.0f} MB")

    try:
        big = torch.randn(25000, 1000, device='cuda')  # ~100 MB
        print("Large allocation succeeded")
        del big
    except RuntimeError:
        print("Large allocation FAILED (fragmentation!)")

    return tensors

held = create_fragmentation()
```

### Exercise 3: Checkpoint vs No-Checkpoint (20 min)

```python
# Measure the compute-memory tradeoff of gradient checkpointing
# on a simple MLP with 20 layers.

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import time

class DeepMLP(nn.Module):
    def __init__(self, width=4096, depth=20, use_ckpt=False):
        super().__init__()
        self.use_ckpt = use_ckpt
        self.layers = nn.ModuleList(
            [nn.Linear(width, width) for _ in range(depth)]
        )

    def forward(self, x):
        for layer in self.layers:
            if self.use_ckpt:
                x = checkpoint(lambda l, inp: torch.relu(l(inp)),
                              layer, x, use_reentrant=False)
            else:
                x = torch.relu(layer(x))
        return x

for ckpt in [False, True]:
    model = DeepMLP(use_ckpt=ckpt).cuda()
    x = torch.randn(64, 4096, device='cuda')

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()

    y = model(x).sum()
    y.backward()

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9

    print(f"Checkpoint={ckpt!s:5s}: {peak:.2f} GB, {elapsed:.3f}s")
    del model, x, y
    torch.cuda.empty_cache()
```

---

## Key Takeaways

1. **Caching allocator** avoids expensive `cudaMalloc`/`cudaFree` calls by reusing
   freed blocks from an internal free-list
2. **Fragmentation**, not capacity, causes most OOM errors — free memory exists but
   isn't contiguous
3. **`torch.cuda.memory_stats()`** and **memory snapshots** are your primary debugging
   tools; `nvidia-smi` shows reserved, not allocated
4. **`max_split_size_mb`** reduces fragmentation by preventing excessive block splitting
5. **Gradient checkpointing** trades ~33% extra compute for ~$\sqrt{L}/L$ memory savings
   on activations

---

## Further Reading

- [PyTorch CUDA Memory Management docs](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [Understanding CUDA Memory Usage (PyTorch tutorial)](https://pytorch.org/docs/stable/torch_cuda_memory.html)
- [Gradient Checkpointing paper (Chen et al. 2016)](https://arxiv.org/abs/1604.06174)
- [PYTORCH_CUDA_ALLOC_CONF reference](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

---

## Tomorrow's Preview

**Day 10: Custom C++ Extensions & pybind11** — We'll write a custom CUDA kernel
(fused bias + GELU), bind it to Python using PyTorch's extension mechanism, register
it with the dispatcher, and implement the backward pass for autograd integration.
