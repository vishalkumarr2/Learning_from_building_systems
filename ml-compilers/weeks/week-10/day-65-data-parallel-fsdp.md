# Day 65: Data Parallel & FSDP

> **Phase V · Week 10 · Day 65 of 70 · 2.5 hours**

*"DDP replicates the model. FSDP shards everything. The difference is 4× memory savings — enough to turn 'impossible on 8 GPUs' into 'fits comfortably'."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 64: Distributed Training Basics](day-64-distributed-training-basics.md) | [Day 66: Tensor & Pipeline Parallelism](day-66-tensor-pipeline-parallel.md) | [Week 10: Distributed Training & Capstone](README.md) | [Phase V: Training at Scale](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Data parallelism is the most widely used distributed training strategy — it's conceptually simple, scales linearly, and introduces no model changes. **DDP** (DistributedDataParallel) is PyTorch's battle-tested implementation, handling gradient synchronization with bucket-based AllReduce overlapped with backward computation. But DDP replicates the entire model on every GPU, limiting you to models that fit in a single GPU's memory. **FSDP** (Fully Sharded Data Parallel) — PyTorch's implementation of Microsoft's ZeRO (Zero Redundancy Optimizer) — eliminates this redundancy by sharding parameters, gradients, and optimizer states across GPUs. Understanding DDP internals and FSDP's ZeRO stages is critical for training any model beyond 10B parameters.

---

## 1. DDP Internals: How DistributedDataParallel Works

### The Naive Approach (and Why It's Slow)

```
Naive Data Parallelism
═══════════════════════════════════════════════════════════════

  Step 1: Forward pass (independent per GPU)
  Step 2: Backward pass (independent per GPU)
  Step 3: AllReduce ALL gradients ← BLOCKING, NO OVERLAP
  Step 4: Optimizer step (identical per GPU)

  Problem: Step 3 waits for ALL gradients before starting.
           No overlap between compute and communication.
```

### DDP's Bucket-Based Gradient AllReduce

```
DDP: Overlapped Gradient Synchronization
═══════════════════════════════════════════════════════════════

  Backward pass computes gradients layer by layer (last → first):

  Time ──────────────────────────────────────────────────────►

  Backward:  [grad_L32] [grad_L31] [grad_L30] ... [grad_L1]
                │           │          │
  Buckets:    ┌─▼───────────▼──┐  ┌────▼────────────┐
              │  Bucket 2      │  │  Bucket 1        │  ...
              │  (25 MB)       │  │  (25 MB)         │
              └──────┬─────────┘  └──────┬───────────┘
                     │                   │
  AllReduce:    [AllReduce B2]      [AllReduce B1]
                     ▲                   ▲
                     │                   │
          Starts as soon as      Starts when B1 full
          bucket B2 is full      (overlaps with earlier
                                  backward compute)

  Key: Gradients are AllReduced while backward is STILL RUNNING
       for earlier layers. Communication is almost fully hidden.
```

### DDP Code

```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def train_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Create model and wrap with DDP
    model = MyTransformer().cuda(rank)
    model = DDP(model, device_ids=[rank])
    # DDP constructor:
    #   - Broadcasts parameters from rank 0 to all ranks
    #   - Registers backward hooks on each parameter
    #   - Creates gradient buckets (default 25 MB each)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # DistributedSampler ensures non-overlapping data per GPU
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, sampler=sampler, batch_size=32)

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Shuffle differently each epoch
        for batch in loader:
            inputs, labels = batch[0].cuda(rank), batch[1].cuda(rank)

            optimizer.zero_grad()
            outputs = model(inputs)       # Forward (independent)
            loss = criterion(outputs, labels)
            loss.backward()               # Backward + AllReduce (overlapped!)
            # After backward(): all ranks have identical averaged gradients

            optimizer.step()              # Identical update on all ranks

    dist.destroy_process_group()
```

### DDP Tuning Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `bucket_cap_mb` | 25 | Larger → better bandwidth utilization, worse overlap |
| `gradient_as_bucket_view` | False | Avoid gradient copy to bucket (saves memory) |
| `find_unused_parameters` | False | Handle models with conditional branches (slower) |
| `static_graph` | False | Enable optimizations for models with fixed structure |

---

## 2. The Memory Problem: Why DDP Isn't Enough

DDP replicates everything. For a model with $N$ parameters and $P$ GPUs:

```
DDP Memory Usage (per GPU) — REDUNDANT
═══════════════════════════════════════════════════════════════

  Per GPU              Total across P GPUs      Waste
  ──────────────────────────────────────────────────────────
  2N  (FP16 weights)   2NP                      (P-1)/P redundant
  2N  (FP16 grads)     2NP                      (P-1)/P redundant
  4N  (FP32 master)    4NP                      (P-1)/P redundant
  4N  (Adam m)         4NP                      (P-1)/P redundant
  4N  (Adam v)         4NP                      (P-1)/P redundant
  ──────────────────────────────────────────────────────────
  16N per GPU          16NP total

  With 8 GPUs, 7/8 = 87.5% of memory is WASTED on duplicates!
```

---

## 3. ZeRO: Zero Redundancy Optimizer

Microsoft's ZeRO (Rajbhandari et al., 2019) eliminates redundancy in three progressive stages:

```
ZeRO Stages — Progressive Sharding
═══════════════════════════════════════════════════════════════

  DDP (no ZeRO):  Each GPU holds ALL of: Params + Grads + Optimizer
  Memory per GPU: 2N + 2N + 12N = 16N bytes

  ZeRO Stage 1: Shard OPTIMIZER states only
  ────────────────────────────────────────────
  GPU 0: [Full Params] [Full Grads] [Optim shard 0]
  GPU 1: [Full Params] [Full Grads] [Optim shard 1]
  GPU 2: [Full Params] [Full Grads] [Optim shard 2]
  GPU 3: [Full Params] [Full Grads] [Optim shard 3]

  Memory per GPU: 2N + 2N + 12N/P = 4N + 12N/P
  8 GPUs: 4N + 1.5N = 5.5N   (2.9× reduction)

  ZeRO Stage 2: Shard OPTIMIZER + GRADIENTS
  ──────────────────────────────────────────
  GPU 0: [Full Params] [Grad shard 0] [Optim shard 0]
  GPU 1: [Full Params] [Grad shard 1] [Optim shard 1]

  Memory per GPU: 2N + 2N/P + 12N/P = 2N + 14N/P
  8 GPUs: 2N + 1.75N = 3.75N   (4.3× reduction)

  ZeRO Stage 3: Shard EVERYTHING (Params + Grads + Optimizer)
  ────────────────────────────────────────────────────────────
  GPU 0: [Param shard 0] [Grad shard 0] [Optim shard 0]
  GPU 1: [Param shard 1] [Grad shard 1] [Optim shard 1]

  Memory per GPU: 2N/P + 2N/P + 12N/P = 16N/P
  8 GPUs: 2N   (8× reduction!)

  ┌──────────────────────────────────────────────────┐
  │  7B model: DDP = 112 GB/GPU → ZeRO-3 = 14 GB/GPU │
  │  Fits on a single consumer GPU!                   │
  └──────────────────────────────────────────────────┘
```

### The Cost: Extra Communication

ZeRO Stage 3 trades memory for communication — parameters must be gathered before each forward/backward:

| Stage | Memory per GPU | Communication Volume |
|-------|---------------|---------------------|
| DDP | $16N$ | $2N$ (AllReduce grads) |
| ZeRO-1 | $4N + \frac{12N}{P}$ | $2N$ (AllReduce grads) |
| ZeRO-2 | $2N + \frac{14N}{P}$ | $2N$ (ReduceScatter grads) |
| ZeRO-3 | $\frac{16N}{P}$ | $3N$ (AllGather params fwd + AllGather bwd + ReduceScatter grads) |

---

## 4. FSDP in PyTorch: ZeRO Stage 3

FSDP is PyTorch's native implementation of ZeRO-3 with additional optimizations.

### FSDP Execution Flow

```
FSDP Forward + Backward with Parameter Gathering
═══════════════════════════════════════════════════════════════

  State: Each GPU holds 1/P of each FSDP unit's parameters

  FORWARD PASS:
  ─────────────
  For each FSDP unit (e.g., transformer layer):
    1. AllGather: Collect full parameters from all GPUs    ← communication
    2. Run forward computation with full parameters        ← compute
    3. Discard non-local parameter shards (free memory!)   ← memory saved
    4. Keep only local shard for backward

  BACKWARD PASS:
  ──────────────
  For each FSDP unit (reverse order):
    1. AllGather: Collect full parameters again             ← communication
    2. Compute gradients with full parameters               ← compute
    3. ReduceScatter: Each GPU gets its gradient shard      ← communication
    4. Discard full parameters and full gradients            ← memory saved

  Net effect: Only 1/P of params + 1/P of grads + 1/P of optim in memory!
```

### FSDP2 Code (PyTorch 2.x)

```python
import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

def train_fsdp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = LargeTransformer(num_layers=32, hidden=4096, heads=32)

    # Auto-wrap policy: wrap each TransformerBlock as an FSDP unit
    wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    # Mixed precision: compute in BF16, reduce in FP32, params stored as BF16
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,    # Gradient reduction in FP32 for stability
        buffer_dtype=torch.bfloat16,
    )

    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
        device_id=rank,
        use_orig_params=True,  # Required for torch.compile compatibility
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for batch in dataloader:
        optimizer.zero_grad()
        loss = model(batch).loss
        loss.backward()      # ReduceScatter gradients automatically
        optimizer.step()     # Each GPU updates only its shard
```

### Sharding Strategies

| Strategy | PyTorch Enum | Equivalent | Memory | Communication |
|----------|-------------|------------|--------|---------------|
| Full Shard | `FULL_SHARD` | ZeRO-3 | Best ($16N/P$) | Highest ($3N$) |
| Shard Grad Op | `SHARD_GRAD_OP` | ZeRO-2 | Good ($2N + 14N/P$) | Medium ($2N$) |
| No Shard | `NO_SHARD` | DDP | Worst ($16N$) | Lowest ($2N$) |
| Hybrid Shard | `HYBRID_SHARD` | ZeRO-3 intra-node, DDP inter-node | Balanced | Balanced |

### Hybrid Sharding: Best of Both Worlds

```
Hybrid Sharding (HYBRID_SHARD)
═══════════════════════════════════════════════════════════════

  Node 0 (8 GPUs, NVLink)         Node 1 (8 GPUs, NVLink)
  ┌─────────────────────────┐     ┌─────────────────────────┐
  │ FSDP shard across       │     │ FSDP shard across       │
  │ 8 GPUs (fast NVLink)    │     │ 8 GPUs (fast NVLink)    │
  │                         │     │                         │
  │ GPU0-GPU7 each hold     │     │ GPU0-GPU7 each hold     │
  │ 1/8 of params           │     │ 1/8 of params           │
  └────────┬────────────────┘     └────────┬────────────────┘
           │                                │
           └──── DDP AllReduce (slow IB) ───┘
                 Only gradients, not params!

  Benefit: AllGather uses fast NVLink (900 GB/s)
           Only gradient AllReduce goes over slow InfiniBand (50 GB/s)
```

---

## 5. Mixed Precision with FSDP

FSDP supports per-component precision control:

```
Mixed Precision Data Flow in FSDP
═══════════════════════════════════════════════════════════════

  Storage (sharded): BF16 weights  ← saves memory
        │
        ▼
  AllGather: gather full BF16 params
        │
        ▼
  Forward: compute in BF16          ← fast on Tensor Cores
        │
        ▼
  Backward: compute grads in BF16   ← fast
        │
        ▼
  ReduceScatter: accumulate in FP32 ← numerical stability!
        │
        ▼
  Optimizer: update in FP32         ← master weights
        │
        ▼
  Cast back to BF16 for storage     ← memory savings
```

---

## 6. FSDP vs DeepSpeed ZeRO

| Feature | PyTorch FSDP | DeepSpeed ZeRO |
|---------|-------------|----------------|
| Framework | Native PyTorch | Separate library |
| ZeRO Stages | 2, 3 (+ hybrid) | 1, 2, 3, 3+ (offload) |
| CPU Offload | Yes (basic) | Yes (advanced, NVMe) |
| torch.compile | Yes (with `use_orig_params`) | Limited |
| Activation Checkpointing | Via `checkpoint_wrapper` | Built-in |
| Config | Python API | JSON config + Python |
| Community | Growing | Mature, feature-rich |

### When to Choose What

- **FSDP**: Pure PyTorch, `torch.compile` integration needed, simpler setups
- **DeepSpeed**: CPU/NVMe offloading, very large models (100B+), MoE support, advanced pipeline parallelism

---

## Hands-On Exercises

### Exercise 1: DDP vs FSDP Memory Comparison (45 min)

Train a 1B-parameter model (GPT-2 XL) on 4 GPUs with both DDP and FSDP. Measure peak GPU memory for each. Verify that FSDP uses approximately $16N/P$ memory vs DDP's $16N$.

```python
# Memory measurement snippet
torch.cuda.reset_peak_memory_stats()
# ... training step ...
peak_mb = torch.cuda.max_memory_allocated() / 1024**2
print(f"[Rank {rank}] Peak memory: {peak_mb:.0f} MB")
```

### Exercise 2: Sharding Strategy Sweep (30 min)

Compare `FULL_SHARD`, `SHARD_GRAD_OP`, and `NO_SHARD` on the same model. Measure memory, throughput (samples/sec), and verify gradient equivalence across strategies.

### Exercise 3: FSDP + AMP Training Loop (45 min)

Write a complete FSDP training loop for a 7B model with mixed precision, gradient clipping, learning rate warmup, and checkpointing. Use `HYBRID_SHARD` if multi-node, `FULL_SHARD` if single-node.

---

## Key Takeaways

1. **DDP overlaps communication** — gradient AllReduce runs concurrently with backward pass via bucketing
2. **DDP wastes memory** — 87.5% redundancy across 8 GPUs (each stores full params + grads + optimizer)
3. **ZeRO-3 / FSDP shards everything** — memory per GPU drops from $16N$ to $16N/P$, making 70B models trainable on 8×80GB GPUs
4. **Extra communication cost** — FSDP adds AllGather for forward and backward ($+2N$), but overlaps with compute
5. **Hybrid sharding** — FSDP within a node (fast NVLink), DDP across nodes (slow IB) balances memory and bandwidth
6. **FSDP2 + torch.compile** — `use_orig_params=True` enables compilation, closing the performance gap with custom solutions

---

## Further Reading

- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2019)
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [PyTorch FSDP2 Design](https://pytorch.org/docs/stable/fsdp.html)
- Zhao et al., "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel" (2023)
- DeepSpeed ZeRO documentation and benchmarks

---

## Tomorrow's Teaser

FSDP handles models that are too large for one GPU's memory, but what about models with layers so large that a *single layer* exceeds GPU memory? Tomorrow we explore **tensor parallelism** (splitting individual matrix operations across GPUs) and **pipeline parallelism** (splitting the model depth-wise with micro-batch scheduling), completing the 3D parallelism picture used to train GPT-4 and LLaMA-3.
