# Day 66: Tensor & Pipeline Parallelism

> **Phase V · Week 10 · Day 66 of 70 · 2.5 hours**

*"Data parallelism scales the batch. Tensor parallelism scales the layer. Pipeline parallelism scales the depth. The art of training frontier models is combining all three."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 65: Data Parallel & FSDP](day-65-data-parallel-fsdp.md) | [Day 67: Compiler's Role in Training](day-67-compiler-role-in-training.md) | [Week 10: Distributed Training & Capstone](README.md) | [Phase V: Training at Scale](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Even with FSDP's memory sharding, two problems remain. First, a single forward pass through one transformer layer may be too slow — the matrix multiplications in a layer with hidden dimension 12,288 (GPT-3 scale) take significant time, and you want to parallelize *within* the layer. Second, with 96+ layers, the sequential forward-backward pipeline creates massive idle "bubble" time across GPUs. **Tensor parallelism** (Megatron-LM, Shoeybi et al. 2019) splits individual operations across GPUs, while **pipeline parallelism** (GPipe, Huang et al. 2019) splits the model into sequential stages. Together with data parallelism, these form **3D parallelism** — the actual strategy used to train models like GPT-4, PaLM, and LLaMA-3 405B.

---

## 1. Tensor Parallelism: Splitting Matrix Operations

### Column-Parallel Linear Layer

The key insight: a matrix multiplication $Y = XA$ can be split column-wise.

$$A = [A_1 | A_2], \quad Y = X[A_1 | A_2] = [XA_1 | XA_2] = [Y_1 | Y_2]$$

```
Column-Parallel Linear (TP=2)
═══════════════════════════════════════════════════════════════

  Input X (on both GPUs via broadcast or AllGather)

  GPU 0:                         GPU 1:
  X × A₁ = Y₁                   X × A₂ = Y₂
  [B,S,H] × [H,H/2] = [B,S,H/2]   [B,S,H] × [H,H/2] = [B,S,H/2]

  If followed by GeLU (element-wise):
  GeLU(Y₁)                      GeLU(Y₂)
  ← No communication needed! GeLU is element-wise ✓

  If output is needed as full tensor:
  AllGather([Y₁, Y₂]) → Y = [Y₁ | Y₂]    ← one AllGather
```

### Row-Parallel Linear Layer

For the second linear layer, split **row-wise** to avoid the AllGather:

$$A = \begin{bmatrix} A_1 \\ A_2 \end{bmatrix}, \quad Y = [X_1 | X_2] \begin{bmatrix} A_1 \\ A_2 \end{bmatrix} = X_1 A_1 + X_2 A_2$$

```
Row-Parallel Linear (TP=2)
═══════════════════════════════════════════════════════════════

  GPU 0: X₁ × A₁ = Z₁           GPU 1: X₂ × A₂ = Z₂
  [B,S,H/2] × [H/2,H] = [B,S,H]   [B,S,H/2] × [H/2,H] = [B,S,H]

  AllReduce(Z₁, Z₂) → Y = Z₁ + Z₂   ← one AllReduce

  Each GPU now has the full output Y
```

### Megatron-LM MLP: Column → Row Composition

```
Megatron MLP Block (TP=2)
═══════════════════════════════════════════════════════════════

  Input X ─── (identity, no comm) ───┐
                                      │
  GPU 0:                    GPU 1:    │
  ┌──────────────────┐     ┌──────────────────┐
  │ ColParallel:     │     │ ColParallel:     │
  │ Y₁ = X·A₁       │     │ Y₂ = X·A₂       │
  │ GeLU(Y₁)        │     │ GeLU(Y₂)        │
  │                  │     │                  │
  │ RowParallel:     │     │ RowParallel:     │
  │ Z₁ = GeLU(Y₁)·B₁│     │ Z₂ = GeLU(Y₂)·B₂│
  └────────┬─────────┘     └────────┬─────────┘
           │                        │
           └─── AllReduce(Z₁+Z₂) ──┘
                      │
                 Output = Z₁ + Z₂

  Communication: ONE AllReduce per MLP block (in forward)
                 ONE AllReduce per MLP block (in backward)
  Total: 2 AllReduce per transformer layer per MLP
```

---

## 2. Self-Attention Parallelism

Multi-head attention splits naturally across heads:

```
Tensor-Parallel Self-Attention (TP=2, H=32 heads)
═══════════════════════════════════════════════════════════════

  GPU 0: Heads 0-15                GPU 1: Heads 16-31

  Q₁ = X · W_Q₁  [B,S,H/2]       Q₂ = X · W_Q₂  [B,S,H/2]
  K₁ = X · W_K₁  [B,S,H/2]       K₂ = X · W_K₂  [B,S,H/2]
  V₁ = X · W_V₁  [B,S,H/2]       V₂ = X · W_V₂  [B,S,H/2]

  Attn₁ = softmax(Q₁K₁ᵀ/√d)V₁   Attn₂ = softmax(Q₂K₂ᵀ/√d)V₂

  O₁ = Attn₁ · W_O₁              O₂ = Attn₂ · W_O₂
       [B,S,H]                         [B,S,H]

       └────── AllReduce(O₁ + O₂) ──────┘
                      │
                    Output

  Key: Q,K,V projections are column-parallel
       Output projection is row-parallel
       → Same 1 AllReduce per attention block!
```

### Communication Cost of Tensor Parallelism

Per transformer layer (forward + backward):

$$\text{TP comm} = 4 \times \text{AllReduce}(\text{activation size}) = 4 \times 2 \times \frac{T-1}{T} \times B \times S \times H$$

where $T$ = tensor parallel degree, $B$ = batch, $S$ = sequence length, $H$ = hidden dim.

**Critical**: This is **per-layer** communication on the activation tensor, not the (much larger) gradient tensor. TP requires NVLink-speed interconnect — it's typically used within a single node.

---

## 3. Pipeline Parallelism: Splitting Model Depth

Pipeline parallelism divides the model's layers into sequential **stages** across GPUs.

### The Bubble Problem

```
Naive Pipeline (4 stages, 1 micro-batch)
═══════════════════════════════════════════════════════════════

  GPU 0: |█ Fwd █|                              |█ Bwd █|
  GPU 1:         |█ Fwd █|              |█ Bwd █|
  GPU 2:                 |█ Fwd █|█ Bwd █|
  GPU 3:                         |█ F+B █|

  █ = computation
  (blank) = idle (bubble!)

  Bubble fraction = (P-1)/P = 3/4 = 75% wasted!
```

### GPipe: Micro-Batch Pipelining

GPipe (Huang et al., 2019) splits the mini-batch into $M$ micro-batches:

```
GPipe Schedule (4 stages, 8 micro-batches)
═══════════════════════════════════════════════════════════════

  Time ─────────────────────────────────────────────────────►

  GPU 0: |F₁|F₂|F₃|F₄|F₅|F₆|F₇|F₈|  |B₈|B₇|B₆|B₅|B₄|B₃|B₂|B₁|
  GPU 1:    |F₁|F₂|F₃|F₄|F₅|F₆|F₇|F₈|  |B₈|B₇|B₆|B₅|B₄|B₃|B₂|B₁|
  GPU 2:       |F₁|F₂|F₃|F₄|F₅|F₆|F₇|F₈|  |B₈|B₇|B₆|B₅|B₄|B₃|B₂|B₁|
  GPU 3:          |F₁|F₂|F₃|F₄|F₅|F₆|F₇|F₈|  |B₈|B₇|B₆|B₅|B₄|B₃|B₂|B₁|

  Bubble fraction = (P-1) / (P-1+M) = 3/11 ≈ 27%

  Problem: Must store activations for ALL M micro-batches during
           forward before backward starts → high memory!
```

### 1F1B Schedule (One Forward One Backward)

The **1F1B** schedule (Narayanan et al., PipeDream) interleaves forward and backward passes:

```
1F1B Schedule (4 stages, 8 micro-batches)
═══════════════════════════════════════════════════════════════

  GPU 0: |F₁|F₂|F₃|F₄|B₁|F₅|B₂|F₆|B₃|F₇|B₄|F₈|B₅|B₆|B₇|B₈|
  GPU 1:    |F₁|F₂|F₃|B₁|F₄|B₂|F₅|B₃|F₆|B₄|F₇|B₅|F₈|B₆|B₇|B₈|
  GPU 2:       |F₁|F₂|B₁|F₃|B₂|F₄|B₃|F₅|B₄|F₆|B₅|F₇|B₆|F₈|B₇|B₈|
  GPU 3:          |F₁|B₁|F₂|B₂|F₃|B₃|F₄|B₄|F₅|B₅|F₆|B₆|F₇|B₇|F₈|B₈|

  Warmup ──► Steady state (1F1B) ──► Cooldown

  Advantages:
  - Same bubble fraction as GPipe: (P-1)/(P-1+M)
  - BUT peak activation memory = P micro-batches (not M!)
  - In steady state, each GPU holds at most P micro-batches of activations
```

### Interleaved 1F1B (Virtual Stages)

Assign **V virtual stages** per GPU to reduce the bubble further:

```
Interleaved Pipeline (4 GPUs, V=2 virtual stages each = 8 total stages)
═══════════════════════════════════════════════════════════════

  GPU 0: Stages {1, 5}   ←─ non-contiguous layers
  GPU 1: Stages {2, 6}
  GPU 2: Stages {3, 7}
  GPU 3: Stages {4, 8}

  Bubble fraction = (P-1) / (P-1 + V×M)

  With V=2, M=8: bubble = 3/(3+16) ≈ 16%  (vs 27% without interleaving)
  With V=4, M=8: bubble = 3/(3+32) ≈ 8.6%
```

---

## 4. 3D Parallelism: Combining All Three

Modern large model training uses all three dimensions simultaneously:

```
3D Parallelism (LLaMA-3 405B example)
═══════════════════════════════════════════════════════════════

  Total GPUs: 16,384 H100s

  ┌─────── Data Parallel (DP = 128) ─────────────────────┐
  │                                                       │
  │  DP replica 0         DP replica 1    ... DP replica 127
  │  ┌──────────────┐    ┌──────────────┐
  │  │ Pipeline      │    │ Pipeline      │
  │  │ Stage 0       │    │ Stage 0       │
  │  │ ┌──────────┐  │    │ ┌──────────┐  │
  │  │ │TP group  │  │    │ │TP group  │  │
  │  │ │GPU0..GPU7│  │    │ │GPU0..GPU7│  │
  │  │ └──────────┘  │    │ └──────────┘  │
  │  │ Stage 1       │    │ Stage 1       │
  │  │ ┌──────────┐  │    │ ┌──────────┐  │
  │  │ │TP group  │  │    │ │TP group  │  │
  │  │ │GPU8..15  │  │    │ │GPU8..15  │  │
  │  │ └──────────┘  │    │ └──────────┘  │
  │  │ ...           │    │ ...           │
  │  │ Stage P-1     │    │ Stage P-1     │
  │  └──────────────┘    └──────────────┘
  └─────────────────────────────────────────────────────────┘

  Mapping to hardware:
  TP = 8  → within a node (NVLink, 900 GB/s)
  PP = 16 → across nodes in a pod (InfiniBand, 400 Gbps)
  DP = 128 → across pods (AllReduce gradients only)

  Total: 8 × 16 × 128 = 16,384 GPUs
```

### Communication Patterns in 3D Parallelism

| Parallelism | Operation | Data Moved | Interconnect |
|------------|-----------|------------|--------------|
| TP (intra-layer) | AllReduce activations | $B \times S \times H$ per layer | NVLink (fast) |
| PP (inter-stage) | Point-to-point activations | $B \times S \times H$ between stages | InfiniBand |
| DP (across replicas) | AllReduce/ReduceScatter grads | $N/(\text{TP} \times \text{PP})$ params | InfiniBand |

---

## 5. Bubble Overhead Analysis

The pipeline bubble is the key efficiency bottleneck. Let's quantify:

$$\text{Bubble ratio} = \frac{(P - 1)}{M + P - 1}$$

where $P$ = pipeline stages, $M$ = micro-batches.

| Stages ($P$) | Micro-batches ($M$) | Bubble | Throughput Efficiency |
|------|-----|--------|-----|
| 4 | 4 | 42.8% | 57.2% |
| 4 | 16 | 15.8% | 84.2% |
| 4 | 64 | 4.5% | 95.5% |
| 16 | 16 | 48.4% | 51.6% |
| 16 | 64 | 19.0% | 81.0% |
| 16 | 256 | 5.5% | 94.5% |

**Rule of thumb**: $M \geq 4P$ for acceptable (<20%) bubble overhead.

### Memory vs Bubble Tradeoff

```
Pipeline Parallelism Tradeoffs
═══════════════════════════════════════════════════════════════

  More stages (larger P):
    ✓ Less memory per GPU (fewer layers per stage)
    ✗ More bubble (need more micro-batches to amortize)
    ✗ More pipeline flushes → lower throughput

  More micro-batches (larger M):
    ✓ Less bubble
    ✗ GPipe: more activation memory (stores all M)
    ✓ 1F1B: activation memory bounded by P (not M)
    ✗ Effective batch size = M × micro_batch_size (must be valid for training)
```

---

## 6. Sequence Parallelism and Beyond

For very long sequences, even TP doesn't help — LayerNorm and Dropout operate on the **full** hidden dimension and must be on a single GPU:

```
Sequence Parallelism (Korthikanti et al., 2022)
═══════════════════════════════════════════════════════════════

  Standard TP: LayerNorm/Dropout replicated on all TP ranks
               → memory waste for large sequences

  With Sequence Parallelism:
  ┌─────────────────────────────────────────────────┐
  │  GPU 0: LayerNorm(seq[0:S/2]) → TP attention    │
  │  GPU 1: LayerNorm(seq[S/2:S]) → TP attention    │
  │                                                   │
  │  AllGather before attention, ReduceScatter after  │
  │  → Saves S×H×B bytes per LayerNorm/Dropout       │
  └─────────────────────────────────────────────────┘
```

---

## Hands-On Exercises

### Exercise 1: Column/Row Parallel Linear (45 min)

Implement column-parallel and row-parallel linear layers using `torch.distributed`:

```python
class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        assert out_features % world_size == 0
        self.local_out = out_features // world_size
        self.weight = nn.Parameter(
            torch.randn(in_features, self.local_out) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(self.local_out))

    def forward(self, x):
        # x: [B, S, in_features] — same on all ranks
        return x @ self.weight + self.bias  # [B, S, local_out]


class RowParallelLinear(nn.Module):
    def __init__(self, in_features, out_features, world_size, rank):
        super().__init__()
        assert in_features % world_size == 0
        self.local_in = in_features // world_size
        self.weight = nn.Parameter(
            torch.randn(self.local_in, out_features) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # x: [B, S, local_in] — each rank has its partition
        local_out = x @ self.weight   # [B, S, out_features]
        # AllReduce to sum partial results
        dist.all_reduce(local_out, op=dist.ReduceOp.SUM)
        return local_out + self.bias
```

Verify that the distributed version produces identical results to a single-GPU linear layer.

### Exercise 2: Pipeline Bubble Calculator (30 min)

Build a pipeline simulator that visualizes the 1F1B schedule and computes bubble fraction for various $P$ and $M$ combinations. Output an ASCII Gantt chart like the diagrams above.

### Exercise 3: 3D Parallelism Configuration (30 min)

Given a 70B model, 64 H100 GPUs across 8 nodes (8 GPUs/node with NVLink), compute the optimal TP/PP/DP configuration. Consider:
- TP must be within a node (≤ 8)
- PP × TP must cover the model's memory requirements
- DP = total_GPUs / (TP × PP)

---

## Key Takeaways

1. **Tensor parallelism splits operations** — column-parallel + row-parallel composition requires only 1 AllReduce per transformer sublayer
2. **TP requires NVLink** — per-layer AllReduce on activations is latency-critical and must use fast intra-node interconnect
3. **Pipeline parallelism splits depth** — 1F1B schedule bounds activation memory to $P$ micro-batches, not $M$
4. **Bubble overhead** — pipeline efficiency requires $M \geq 4P$; interleaved schedules further reduce bubbles
5. **3D parallelism is the standard** — TP within node, PP across nearby nodes, DP across the cluster
6. **Communication hierarchy matches hardware** — fast NVLink for TP, InfiniBand for PP/DP, matching operation frequency to bandwidth

---

## Further Reading

- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019)
- Huang et al., "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism" (2019)
- Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM" (2021)
- Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models" (2022)

---

## Tomorrow's Teaser

We've covered *what* to parallelize and *how* to schedule it. But who decides the optimal parallelization strategy? Tomorrow we explore how **compilers** automate training optimization — from XLA's HLO partitioning for TPU training, to `torch.compile`'s backward pass fusion, to automated activation checkpointing and compute-communication overlap scheduling.
