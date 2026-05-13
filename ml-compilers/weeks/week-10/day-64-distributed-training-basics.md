# Day 64: Distributed Training Basics

> **Phase V · Week 10 · Day 64 of 70 · 2.5 hours**

*"A single GPU can train GPT-2 in a week. GPT-4 required 25,000 GPUs for 3 months. Distributed training isn't optional — it's the only way modern AI exists."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 63: Stop & Reflect #5](../week-09/day-63-stop-and-reflect-5.md) | [Day 65: Data Parallel & FSDP](day-65-data-parallel-fsdp.md) | [Week 10: Distributed Training & Capstone](README.md) | [Phase V: Training at Scale](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Training a 70B parameter model with FP16 requires 140 GB just for the weights — already exceeding any single GPU's memory. Add optimizer states (2× for Adam), gradients (1×), and activations, and you need **~1 TB** for training. Beyond memory, compute scaling laws (Kaplan et al., Chinchilla) show that optimal training requires compute budgets of $C \approx 6ND$ FLOPs, where $N$ is parameter count and $D$ is token count. For frontier models, this translates to $10^{24}$–$10^{25}$ FLOPs — physically impossible on a single device. Understanding distributed training fundamentals — parallelism strategies, collective communication, and their overhead — is essential for anyone working on training systems, compiler optimizations for training, or ML infrastructure.

---

## 1. Why Distributed Training: The Numbers

### Scaling Laws Drive Compute Requirements

The Chinchilla scaling law tells us the optimal compute budget:

$$C_{\text{optimal}} \approx 6 \times N \times D$$

where $N$ = parameters, $D$ = training tokens.

| Model | Params ($N$) | Tokens ($D$) | Compute ($C$) | A100 GPU-hours |
|-------|-------------|--------------|---------------|----------------|
| GPT-2 | 1.5B | 40B | $3.6 \times 10^{20}$ | ~460 |
| LLaMA-2 7B | 7B | 2T | $8.4 \times 10^{22}$ | ~180K |
| LLaMA-2 70B | 70B | 2T | $8.4 \times 10^{23}$ | ~1.7M |
| GPT-4 (est.) | 1.8T | 13T | $1.4 \times 10^{26}$ | ~200M |

### Memory Breakdown for Training

For a model with $N$ parameters in mixed precision:

```
Training Memory Breakdown (Mixed Precision + Adam)
═══════════════════════════════════════════════════

  Component               Size           Example (7B model)
  ───────────────────────────────────────────────────────────
  Weights (FP16)          2N bytes       14 GB
  Gradients (FP16)        2N bytes       14 GB
  Adam optimizer states:
    - FP32 weights copy   4N bytes       28 GB
    - First moment (m)    4N bytes       28 GB
    - Second moment (v)   4N bytes       28 GB
  ───────────────────────────────────────────────────────────
  Subtotal (no acts)      16N bytes      112 GB
  Activations             varies         50-200+ GB
  ───────────────────────────────────────────────────────────
  Total                                  162-312 GB

  Single A100 has 80 GB → Need at least 4 GPUs for 7B!
```

---

## 2. The Three Parallelism Strategies

```
Parallelism Taxonomy
═══════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────┐
  │              Full Training Iteration                    │
  │  Data × Model ──► Forward ──► Loss ──► Backward ──► Update│
  └─────────────────────────────────────────────────────────┘

  DATA PARALLELISM: Split the data, replicate the model
  ─────────────────────────────────────────────────────
  GPU 0: [Batch 0] × [Full Model] ──► grad₀ ─┐
  GPU 1: [Batch 1] × [Full Model] ──► grad₁ ─┤──AllReduce──► avg grad
  GPU 2: [Batch 2] × [Full Model] ──► grad₂ ─┤              → update all
  GPU 3: [Batch 3] × [Full Model] ──► grad₃ ─┘

  MODEL PARALLELISM (Tensor): Split layers across devices
  ─────────────────────────────────────────────────────────
  GPU 0: [Full Batch] × [Layer cols 0:H/2] ─┐
                                              ├─ combine ──► next layer
  GPU 1: [Full Batch] × [Layer cols H/2:H] ─┘

  PIPELINE PARALLELISM: Split layers sequentially
  ─────────────────────────────────────────────────
  GPU 0: [Layers 0-7]  ──output──► GPU 1: [Layers 8-15]
                                    ──output──► GPU 2: [Layers 16-23]
                                                 ──output──► GPU 3: [Layers 24-31]
```

### Comparison

| Strategy | Memory per GPU | Communication | Scaling | Best For |
|----------|---------------|---------------|---------|----------|
| Data Parallel | Full model | Gradients (AllReduce) | Batch size | Model fits in 1 GPU |
| Tensor Parallel | Partial layer | Activations (AllReduce per layer) | Per-layer | Large layers |
| Pipeline Parallel | Subset of layers | Activations (point-to-point) | Depth | Many layers |

---

## 3. Collective Communication Operations

The backbone of distributed training is **collective communication** — operations that move data between all participating GPUs.

### The Essential Collectives

```
Collective Operations (4 GPUs, each has a vector)
═══════════════════════════════════════════════════════════════

  BROADCAST: One-to-all
  ─────────────────────
  GPU 0: [A]     GPU 0: [A]
  GPU 1: [ ]  →  GPU 1: [A]
  GPU 2: [ ]     GPU 2: [A]
  GPU 3: [ ]     GPU 3: [A]

  REDUCE: All-to-one (sum/avg)
  ────────────────────────────
  GPU 0: [A₀]     GPU 0: [A₀+A₁+A₂+A₃]
  GPU 1: [A₁]  →  GPU 1: [ ]
  GPU 2: [A₂]     GPU 2: [ ]
  GPU 3: [A₃]     GPU 3: [ ]

  ALL-REDUCE: Reduce + Broadcast (every GPU gets result)
  ──────────────────────────────────────────────────────
  GPU 0: [A₀]     GPU 0: [A₀+A₁+A₂+A₃]
  GPU 1: [A₁]  →  GPU 1: [A₀+A₁+A₂+A₃]
  GPU 2: [A₂]     GPU 2: [A₀+A₁+A₂+A₃]
  GPU 3: [A₃]     GPU 3: [A₀+A₁+A₂+A₃]

  ALL-GATHER: Each sends its piece, all get everything
  ────────────────────────────────────────────────────
  GPU 0: [A₀]        GPU 0: [A₀|A₁|A₂|A₃]
  GPU 1: [A₁]     →  GPU 1: [A₀|A₁|A₂|A₃]
  GPU 2: [A₂]        GPU 2: [A₀|A₁|A₂|A₃]
  GPU 3: [A₃]        GPU 3: [A₀|A₁|A₂|A₃]

  REDUCE-SCATTER: Reduce + Scatter (each gets a shard of result)
  ──────────────────────────────────────────────────────────────
  GPU 0: [a₀|b₀|c₀|d₀]     GPU 0: [Σaᵢ]
  GPU 1: [a₁|b₁|c₁|d₁]  →  GPU 1: [Σbᵢ]
  GPU 2: [a₂|b₂|c₂|d₂]     GPU 2: [Σcᵢ]
  GPU 3: [a₃|b₃|c₃|d₃]     GPU 3: [Σdᵢ]
```

### Key Insight: AllReduce = ReduceScatter + AllGather

$$\text{AllReduce}(x) = \text{AllGather}(\text{ReduceScatter}(x))$$

This decomposition is why FSDP works — you can perform half the communication during backward (ReduceScatter gradients) and the other half just-in-time during forward (AllGather parameters).

---

## 4. Ring AllReduce Algorithm

The naïve AllReduce sends all data to one node and broadcasts back — $O(N \cdot P)$ traffic through the root. **Ring AllReduce** distributes the work evenly.

```
Ring AllReduce (4 GPUs, data split into 4 chunks)
═══════════════════════════════════════════════════════════════

  Phase 1: REDUCE-SCATTER (P-1 = 3 steps)
  Each GPU sends one chunk clockwise, accumulates received chunk

  Step 1:                Step 2:                Step 3:
  GPU0 ──c0──► GPU1      GPU0 ──c3──► GPU1      GPU0 ──c2──► GPU1
  GPU1 ──c1──► GPU2      GPU1 ──c0──► GPU2      GPU1 ──c3──► GPU2
  GPU2 ──c2──► GPU3      GPU2 ──c1──► GPU3      GPU2 ──c0──► GPU3
  GPU3 ──c3──► GPU0      GPU3 ──c2──► GPU0      GPU3 ──c1──► GPU0

  After ReduceScatter: Each GPU holds the fully reduced chunk i

  Phase 2: ALL-GATHER (P-1 = 3 steps)
  Same ring pattern, but forward received chunks without accumulating

  Total data transferred per GPU: 2 × (P-1)/P × N
  ────────────────────────────────────────────────
  For P=4, each GPU sends 1.5N total (75% of full data)
  For P=256, each GPU sends ~2N regardless of cluster size!
```

### Communication Cost Analysis

| Algorithm | Bandwidth Cost per GPU | Latency Cost |
|-----------|----------------------|--------------|
| Naïve (tree) | $O(N)$ | $O(P \cdot \alpha)$ |
| Ring AllReduce | $2 \cdot \frac{P-1}{P} \cdot N$ | $2(P-1) \cdot \alpha$ |
| Recursive Halving-Doubling | $\frac{P-1}{P} \cdot N$ | $2 \log_2(P) \cdot \alpha$ |

Where $N$ = message size, $P$ = number of GPUs, $\alpha$ = per-message latency.

**Key**: Ring AllReduce is **bandwidth-optimal** — total traffic is independent of $P$ for large messages. But latency grows linearly. For small messages, tree-based algorithms with $O(\log P)$ latency are better.

---

## 5. NCCL: The Communication Library

**NCCL** (NVIDIA Collective Communications Library) is the de facto standard for GPU-to-GPU communication.

```python
import torch
import torch.distributed as dist
import os

def setup_distributed():
    """Initialize distributed training with NCCL backend."""
    # Environment variables set by torchrun / SLURM
    rank = int(os.environ["RANK"])           # Global rank (0..world_size-1)
    local_rank = int(os.environ["LOCAL_RANK"])  # Rank within this node
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize process group — NCCL for GPU communication
    dist.init_process_group(
        backend="nccl",           # NCCL for GPU, gloo for CPU
        rank=rank,
        world_size=world_size,
    )

    # Bind this process to its GPU
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def demonstrate_collectives(rank, world_size):
    """Show basic collective operations."""
    device = torch.device(f"cuda:{rank}")

    # Each GPU creates its own tensor
    tensor = torch.tensor([rank * 10.0 + 1, rank * 10.0 + 2], device=device)
    print(f"[Rank {rank}] Before AllReduce: {tensor}")

    # AllReduce — sum across all GPUs (in-place)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"[Rank {rank}] After AllReduce(SUM): {tensor}")
    # All GPUs now have: [0*10+1 + 1*10+1 + ... , 0*10+2 + 1*10+2 + ...]

    # AllGather — collect tensors from all GPUs
    gathered = [torch.zeros(2, device=device) for _ in range(world_size)]
    my_tensor = torch.tensor([rank * 1.0, rank * 2.0], device=device)
    dist.all_gather(gathered, my_tensor)
    print(f"[Rank {rank}] AllGather result: {gathered}")

    # Broadcast — GPU 0 sends to all
    data = torch.tensor([42.0, 43.0], device=device) if rank == 0 \
           else torch.zeros(2, device=device)
    dist.broadcast(data, src=0)
    print(f"[Rank {rank}] After Broadcast from 0: {data}")


if __name__ == "__main__":
    rank, local_rank, world_size = setup_distributed()
    demonstrate_collectives(rank, world_size)
    dist.destroy_process_group()

# Launch with:  torchrun --nproc_per_node=4 this_script.py
```

### NCCL Topology Detection

NCCL automatically detects the GPU interconnect topology and chooses optimal algorithms:

```
GPU Interconnect Hierarchy (typical 8-GPU node)
═══════════════════════════════════════════════════════════════

  Within a node:
  ┌───────────────────────────────────────────────┐
  │  GPU 0 ══NVLink══ GPU 1    GPU 4 ══NVLink══ GPU 5  │
  │    ║                ║        ║                ║     │
  │  NVLink           NVLink   NVLink           NVLink  │
  │    ║                ║        ║                ║     │
  │  GPU 2 ══NVLink══ GPU 3    GPU 6 ══NVLink══ GPU 7  │
  │         └──NVSwitch (900 GB/s bisection)──┘         │
  └───────────────────────────────────────────────┘

  Across nodes:
  Node 0 ══InfiniBand (400 Gb/s)══ Node 1
         ══InfiniBand══ Node 2
         ══InfiniBand══ Node 3

  Bandwidth hierarchy:
  NVLink 4 (H100):  900 GB/s bidirectional
  PCIe Gen5:         64 GB/s per direction
  InfiniBand NDR:    50 GB/s (400 Gbps)
  Ethernet 100GbE:   12.5 GB/s
```

---

## 6. Communication Overhead Analysis

The key question: **how much time does communication add?**

### Overlap with Computation

```
Timeline: Data-Parallel Training Step
═══════════════════════════════════════════════════════════════

  WITHOUT overlap:
  |── Forward ──|── Backward ──|── AllReduce ──|── Update ──|
  |     T_f     |     T_b      |     T_c       |    T_u     |
  Total = T_f + T_b + T_c + T_u

  WITH overlap (DDP approach):
  |── Forward ──|──── Backward ─────────────|── Update ──|
                |  grad_N ──► AllReduce_N   |
                |  grad_N-1 ► AllReduce_N-1 |
                |  grad_N-2 ► AllReduce_N-2 |
                |  ...overlapped...          |
  Total ≈ T_f + max(T_b, T_c) + T_u    ← Communication hidden!
```

### When Communication Dominates

The **computation-to-communication ratio** determines scaling efficiency:

$$\text{Efficiency} = \frac{T_{\text{compute}}}{T_{\text{compute}} + T_{\text{comm}}}$$

For data parallelism with AllReduce on gradient size $G$:

$$T_{\text{comm}} \approx 2 \cdot \frac{P-1}{P} \cdot \frac{G}{\text{bandwidth}}$$

For a 7B model ($G = 14$ GB in FP16) across 8 GPUs on InfiniBand (50 GB/s):

$$T_{\text{comm}} \approx 2 \times \frac{7}{8} \times \frac{14}{50} \approx 0.49 \text{ seconds}$$

If one training step takes 2 seconds of compute, efficiency = $\frac{2.0}{2.0 + 0.49} \approx 80\%$.

With computation-communication overlap, efficiency approaches 95%+.

---

## Hands-On Exercises

### Exercise 1: Communication Primitives (30 min)

Write a script that demonstrates AllReduce, AllGather, ReduceScatter, and Broadcast using `torch.distributed`. Run with `torchrun --nproc_per_node=2`. Verify that AllReduce equals ReduceScatter followed by AllGather.

### Exercise 2: Ring AllReduce Simulation (45 min)

Implement ring AllReduce in pure Python (no GPU needed) to understand the algorithm:

```python
import numpy as np

def ring_allreduce(data_per_gpu: list[np.ndarray]) -> list[np.ndarray]:
    """
    Simulate ring AllReduce on CPU.

    Args:
        data_per_gpu: list of arrays, one per simulated GPU
    Returns:
        list of arrays, each containing the sum of all inputs
    """
    P = len(data_per_gpu)
    N = len(data_per_gpu[0])
    assert N % P == 0, "Data size must be divisible by GPU count"
    chunk_size = N // P

    # Split each GPU's data into P chunks
    chunks = [np.split(d, P) for d in data_per_gpu]

    # Phase 1: Reduce-Scatter (P-1 steps)
    for step in range(P - 1):
        for gpu in range(P):
            send_idx = (gpu - step) % P
            recv_idx = (gpu - step - 1) % P
            sender = (gpu - 1) % P
            # GPU receives chunk[recv_idx] from its left neighbor
            # and accumulates it
            # YOUR CODE: chunks[gpu][recv_idx] += chunks[sender][recv_idx]
            pass  # TODO

    # Phase 2: All-Gather (P-1 steps)
    for step in range(P - 1):
        for gpu in range(P):
            # Forward the fully-reduced chunk to the right
            # YOUR CODE
            pass  # TODO

    # Reconstruct full arrays
    return [np.concatenate(chunks[gpu]) for gpu in range(P)]

# Test
data = [np.array([1.0, 2.0, 3.0, 4.0]) * (i + 1) for i in range(4)]
result = ring_allreduce(data)
expected = sum(data)
for i, r in enumerate(result):
    assert np.allclose(r, expected), f"GPU {i}: {r} != {expected}"
    print(f"GPU {i}: {r}  ✓")
```

### Exercise 3: Scaling Analysis (30 min)

Calculate the communication-to-computation ratio for:
- 7B model on 8× A100 (NVLink, 900 GB/s)
- 70B model on 64× H100 across 8 nodes (InfiniBand, 400 Gbps)
- 1T model on 4096× H100 across 512 nodes

Determine where data parallelism alone breaks down and other strategies are needed.

---

## Key Takeaways

1. **Memory is the first bottleneck** — training a 7B model needs ~112 GB for states alone, exceeding any single GPU
2. **Three parallelism axes** — data (split batches), tensor (split layers), pipeline (split model depth) — and modern training uses all three
3. **AllReduce = ReduceScatter + AllGather** — this decomposition enables FSDP's memory savings
4. **Ring AllReduce is bandwidth-optimal** — each GPU transfers $2 \times \frac{P-1}{P} \times N$ bytes regardless of cluster size
5. **Overlap hides communication** — DDP overlaps gradient AllReduce with backward pass, approaching ideal scaling
6. **Bandwidth hierarchy matters** — NVLink (900 GB/s) vs InfiniBand (50 GB/s) means intra-node vs inter-node communication differs by 18×

---

## Further Reading

- [PyTorch Distributed Overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- Sergeev & Del Balso, "Horovod: fast and easy distributed deep learning" (2018)
- NCCL documentation and topology detection
- Kaplan et al., "Scaling Laws for Neural Language Models" (2020)
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)

---

## Tomorrow's Teaser

You've learned the communication primitives. Tomorrow we'll see how **DDP** and **FSDP** build on them — and how FSDP's ZeRO-style sharding cuts per-GPU memory from $16N$ bytes to as low as $\frac{16N}{P} + 2N$ bytes, making it possible to train 70B+ models on consumer-grade GPU clusters.
