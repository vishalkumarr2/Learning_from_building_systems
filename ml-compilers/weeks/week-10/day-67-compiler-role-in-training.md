# Day 67: Compiler's Role in Training

> **Phase V · Week 10 · Day 67 of 70 · 2.5 hours**

*"Manually fusing kernels, scheduling communication, and managing memory across 16,000 GPUs is humanly impossible. This is where compilers earn their keep."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 66: Tensor & Pipeline Parallelism](day-66-tensor-pipeline-parallel.md) | [Day 68: Capstone Part 1](day-68-capstone-part1.md) | [Week 10: Distributed Training & Capstone](README.md) | [Phase V: Training at Scale](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Training a frontier model involves thousands of operations — forward matmuls, activation functions, backward gradients, optimizer updates — across thousands of GPUs, with mixed precision, activation checkpointing, and overlapping communication. Manually optimizing this is a combinatorial nightmare. **ML compilers** automate the critical optimizations: fusing operations in the backward pass, inserting activation checkpoints to trade compute for memory, scheduling AllReduce to overlap with computation, and managing memory lifetimes so peak usage stays within GPU limits. Understanding how compilers optimize training — not just inference — reveals the full scope of what systems like XLA, `torch.compile`, and GSPMD do for the ML ecosystem.

---

## 1. The Training Compilation Challenge

Training is harder to compile than inference for three reasons:

```
Inference vs Training Compilation
═══════════════════════════════════════════════════════════════

  INFERENCE:
  ┌────────────────────────────────────────────────┐
  │  Forward pass only                             │
  │  Static graph (usually)                        │
  │  Small batch, latency-sensitive                │
  │  Known memory footprint                        │
  │  No gradient tracking needed                   │
  └────────────────────────────────────────────────┘

  TRAINING:
  ┌────────────────────────────────────────────────┐
  │  Forward + backward + optimizer                │
  │  Backward graph is auto-generated              │
  │  Large batch, throughput-sensitive              │
  │  Memory = activations + gradients + optimizer  │
  │  Activation checkpointing tradeoffs            │
  │  Communication must overlap with compute       │
  │  Mixed precision requires careful placement    │
  │  Dynamic shapes (variable sequence lengths)    │
  └────────────────────────────────────────────────┘

  Result: Training compilation must reason about:
  1. The joint forward-backward graph
  2. Memory pressure and lifetime analysis
  3. Distributed communication scheduling
  4. Numerical precision boundaries
```

---

## 2. Operator Fusion During Backward Pass

### Forward Fusion vs Backward Fusion

Fusing operations during inference is well-studied (Day 40). During training, the backward pass opens additional fusion opportunities:

```
Backward Pass Fusion Opportunities
═══════════════════════════════════════════════════════════════

  Forward:  y = LayerNorm(GeLU(x @ W + b))

  Backward (unfused):
    Step 1: d_layernorm = backward_layernorm(dy, y, mean, var)
    Step 2: d_gelu = backward_gelu(d_layernorm, x @ W + b)
    Step 3: d_bias = reduce_sum(d_gelu)
    Step 4: d_weight = x.T @ d_gelu
    Step 5: d_x = d_gelu @ W.T

  Fused backward:
    Step 1: d_gelu = fused_layernorm_gelu_backward(dy, y, gelu_input, mean, var)
           ← ONE kernel for steps 1+2, no intermediate d_layernorm materialized
    Step 2: d_bias = reduce_sum(d_gelu)    ← fused into step 3
    Step 3: d_weight = x.T @ d_gelu        ← with d_bias accumulation
    Step 4: d_x = d_gelu @ W.T

  Memory saved: don't materialize intermediate backward tensors
  Speed: fewer kernel launches, better data reuse
```

### torch.compile for Training

```python
import torch
from torch import nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model=4096, nhead=32, dim_ff=11008):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.ff(self.norm2(x))
        return x

model = TransformerBlock().cuda()

# Compile for training — optimizes BOTH forward and backward
compiled_model = torch.compile(model, mode="reduce-overhead")
# mode="reduce-overhead": CUDA graphs + aggressive fusion
# mode="max-autotune":    triton kernel autotuning + fusion

optimizer = torch.optim.AdamW(compiled_model.parameters(), lr=1e-4)

# First iteration: trace + compile (slow)
x = torch.randn(8, 2048, 4096, device="cuda")
loss = compiled_model(x).sum()
loss.backward()         # ← backward graph is ALSO compiled
optimizer.step()

# Subsequent iterations: run compiled kernels (fast)
```

### What torch.compile Fuses in Backward

```
torch.compile Backward Fusion Example
═══════════════════════════════════════════════════════════════

  Before compilation (backward):
  ┌────────────────────────────────────────────────────┐
  │ Kernel 1: mul(grad_out, gelu_derivative)           │
  │ Kernel 2: layernorm_backward(d_gelu, mean, rstd)   │
  │ Kernel 3: add(grad_input, residual_grad)            │
  │ Kernel 4: dropout_backward(grad, mask)              │
  │ 4 kernel launches, 3 intermediate tensors           │
  └────────────────────────────────────────────────────┘

  After compilation (fused):
  ┌────────────────────────────────────────────────────┐
  │ Triton kernel: fused_gelu_layernorm_dropout_bwd    │
  │   - Reads: grad_out, gelu_input, mean, rstd, mask │
  │   - Writes: grad_input (directly into residual)   │
  │   - 1 kernel launch, 0 intermediate tensors!       │
  └────────────────────────────────────────────────────┘

  Speedup: 1.3-2× for backward pass due to fewer memory round-trips
```

---

## 3. Activation Checkpointing via Compiler

Activation checkpointing (gradient checkpointing) trades compute for memory — instead of storing all activations for backward, recompute them:

### Manual vs Compiler-Driven Checkpointing

```
Activation Checkpointing Strategies
═══════════════════════════════════════════════════════════════

  NO checkpointing (32 layers):
  Memory: O(32 × B × S × H)  ← ALL activations stored
  Compute: 1× forward + 1× backward

  MANUAL checkpointing (every layer):
  Memory: O(1 × B × S × H)   ← only checkpoint boundaries
  Compute: 1× forward + 2× forward + 1× backward = ~1.33× overhead
  Developer must manually wrap each block

  COMPILER-DRIVEN checkpointing:
  Memory: O(√32 × B × S × H) ← compiler finds optimal boundaries
  Compute: minimal extra recomputation
  Developer writes: nothing (compiler decides!)
```

### How Compilers Choose What to Checkpoint

```python
# PyTorch selective activation checkpointing (compiler-integrated)
from torch.utils.checkpoint import checkpoint

# Manual (old way):
class ManualCheckpointedModel(nn.Module):
    def forward(self, x):
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        return x

# Compiler-driven (new way with torch.compile):
# The compiler analyzes the graph and inserts checkpoints optimally
from torch._functorch.config import activation_checkpointing

# Tell the compiler: checkpoint these ops (cheap to recompute)
# but NOT matmuls (expensive to recompute)
torch._dynamo.config.activation_checkpointing = True

# The compiler will:
# 1. Analyze memory pressure curve across the forward pass
# 2. Identify operations that are cheap to recompute (element-wise)
# 3. Avoid re-checkpointing expensive ops (matmul, convolution)
# 4. Insert checkpoint boundaries to minimize peak memory
```

### Compiler Activation Checkpointing Decision

```
Compiler Memory Planning with Checkpointing
═══════════════════════════════════════════════════════════════

  Forward graph analysis:

  Op 1: MatMul    [cost: HIGH, size: 4 GB]  → SAVE (expensive to recompute)
  Op 2: GeLU      [cost: LOW,  size: 2 GB]  → RECOMPUTE (cheap)
  Op 3: LayerNorm [cost: LOW,  size: 2 GB]  → RECOMPUTE (cheap)
  Op 4: Dropout   [cost: LOW,  size: 2 GB]  → SAVE (random mask needed!)
  Op 5: MatMul    [cost: HIGH, size: 4 GB]  → SAVE
  Op 6: Softmax   [cost: MED,  size: 3 GB]  → RECOMPUTE (moderate cost)

  Without checkpoint: Peak memory = 4+2+2+2+4+3 = 17 GB
  With selective:     Peak memory = 4+0+0+2+4+0 = 10 GB  (41% reduction)
  Extra compute:      Recompute GeLU + LayerNorm + Softmax = ~5% overhead

  ┌──────────────────────────────────────────────────────┐
  │  Optimal strategy: save expensive ops (matmuls),     │
  │  recompute cheap ops (element-wise), always save     │
  │  non-deterministic ops (dropout masks).              │
  └──────────────────────────────────────────────────────┘
```

---

## 4. Memory Planning and Lifetime Analysis

Compilers perform **liveness analysis** on tensors to determine when memory can be reused:

```
Tensor Lifetime Analysis
═══════════════════════════════════════════════════════════════

  Forward: x → a → b → c → d → loss
  Backward:           dc← db← da← dx

  Tensor Lifetimes (when each tensor must be in memory):
  ──────────────────────────────────────────────────────────
  x:  |████████████████████████████████████████████████████| (input)
  a:  |    ████████████████████████████████████████        | (fwd to bwd)
  b:  |        ████████████████████████████████            | (fwd to bwd)
  c:  |            ████████████████████████                | (fwd to bwd)
  d:  |                ████████████████                    | (fwd to bwd)
  da: |                                ████████            | (bwd only)
  db: |                            ████████████            | (bwd only)
  ──────────────────────────────────────────────────────────
       Forward ──────────────────► ◄───── Backward

  Peak memory occurs at the forward/backward boundary!

  Compiler optimization: in-place operations
  - If tensor a is only consumed by backward of the same layer,
    reuse a's memory for da immediately after recomputation
  - XLA's "rematerialization" automatically manages this
```

### Memory-Efficient Gradient Accumulation

```
Gradient Accumulation (effective batch = micro_batch × accum_steps)
═══════════════════════════════════════════════════════════════

  Without compiler optimization:
  Step 1: grads₁ = backward(loss₁)    ← allocate gradient buffers
  Step 2: grads₂ = backward(loss₂)    ← need SEPARATE buffers? No!
  Step 3: grads₃ = backward(loss₃)
  Step 4: grads₄ = backward(loss₄)
  optimizer.step(grads₁ + grads₂ + grads₃ + grads₄)

  With compiler: accumulate in-place, no extra memory
  grads = 0
  for i in range(4):
      grads += backward(loss_i)    ← compiler fuses accumulation
  optimizer.step(grads / 4)
```

---

## 5. Compute-Communication Overlap

The most impactful compiler optimization for distributed training: **hiding communication latency** behind computation.

```
Communication Overlap Scheduling
═══════════════════════════════════════════════════════════════

  WITHOUT overlap (naive):
  |── Compute layer N ──|── AllReduce grad_N ──|── Compute N-1 ──|

  WITH overlap (compiler-scheduled):
  |── Compute layer N ──|── Compute layer N-1 ──|── Compute N-2 ──|
                        |── AllReduce grad_N  ──|
                                                |── AllReduce N-1 ──|

  Compiler must:
  1. Identify which AllReduce depends on which backward ops
  2. Schedule AllReduce to START as early as possible
  3. Ensure AllReduce COMPLETES before optimizer needs the gradient
  4. Manage CUDA streams: compute stream + communication stream
```

### XLA's GSPMD Partitioner

XLA (used by JAX/TPU) takes a different approach — the compiler itself decides how to partition operations across devices:

```python
# JAX/XLA: Compiler-driven sharding
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding, Mesh

# Define device mesh
devices = jax.devices()  # e.g., 8 TPU cores
mesh = Mesh(devices.reshape(2, 4), axis_names=('dp', 'tp'))

# Annotate sharding — compiler handles the rest
def train_step(params, batch):
    def loss_fn(p):
        # Compiler infers which ops need AllReduce, AllGather, etc.
        logits = model.apply(p, batch['input'])
        return cross_entropy(logits, batch['label']).mean()

    loss, grads = jax.value_and_grad(loss_fn)(params)
    # Compiler automatically:
    # 1. Inserts AllReduce for DP gradient aggregation
    # 2. Inserts AllGather/ReduceScatter for TP
    # 3. Overlaps communication with computation
    # 4. Fuses backward ops
    params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
    return params, loss

# Shard params: TP over 'tp' axis, replicate over 'dp' axis
param_sharding = NamedSharding(mesh, P(None, 'tp'))
batch_sharding = NamedSharding(mesh, P('dp', None))

# JIT compile — XLA sees full graph and optimizes holistically
jitted_step = jax.jit(
    train_step,
    in_shardings=(param_sharding, batch_sharding),
    out_shardings=(param_sharding, None),
)
```

### What GSPMD Does Automatically

```
GSPMD Compiler Decisions
═══════════════════════════════════════════════════════════════

  Input: user-annotated sharding specs + computation graph

  Compiler propagates sharding through the graph:

  matmul(X[dp,·], W[·,tp])
    → X is replicated on tp, sharded on dp
    → W is sharded on tp, replicated on dp
    → Output is sharded on both dp AND tp
    → Insert AllReduce on dp for gradient aggregation
    → Insert AllGather on tp to reconstruct full output if needed

  Compiler cost model considers:
  - Communication volume for each sharding choice
  - Memory usage under each partitioning
  - Compute balance (avoid stragglers)
  - Hardware topology (NVLink vs IB bandwidth)

  Output: fully partitioned HLO with explicit communication ops
```

---

## 6. Mixed Precision via Compiler

Compilers manage precision boundaries to maximize performance while maintaining numerical stability:

```
Compiler Mixed Precision Decisions
═══════════════════════════════════════════════════════════════

  Rule-based precision assignment:

  Operation              Precision     Reason
  ──────────────────────────────────────────────────────────
  MatMul (forward)       BF16/FP16     Tensor Cores (2× speed)
  MatMul (backward)      BF16/FP16     Same as forward
  Softmax                FP32          Numerical stability (exp)
  LayerNorm              FP32          Reduction needs precision
  Loss computation       FP32          Small values, catastrophic cancel
  Gradient accumulation  FP32          Prevent underflow over many steps
  Optimizer update       FP32          Master weights must be precise
  AllReduce              FP32          Summation over many GPUs

  Compiler optimization: cast placement
  ──────────────────────────────────────
  Naive:   cast → matmul → cast → gelu → cast → matmul → cast
           (4 casts, 4 kernel launches)

  Fused:   cast+matmul → gelu → matmul+cast
           (0 standalone casts, fused into compute kernels)

  XLA: precision promotion is part of the HLO optimization pipeline
  torch.compile: AMP autocasting integrated into Inductor codegen
```

### torch.compile + AMP Integration

```python
# torch.compile seamlessly handles AMP
model = TransformerModel().cuda()
compiled_model = torch.compile(model)

scaler = torch.amp.GradScaler()  # Loss scaling for FP16

for batch in dataloader:
    optimizer.zero_grad()

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        # torch.compile fuses casts INTO the generated Triton kernels
        # No separate cast kernels needed
        loss = compiled_model(batch)

    # With BF16: no scaler needed (sufficient dynamic range)
    loss.backward()
    optimizer.step()

# What the compiler generates internally:
# @triton.jit
# def fused_mlp_bwd(grad_out_ptr, weight_ptr, act_input_ptr, ...):
#     # Load in BF16, compute in FP32 where needed, store in BF16
#     grad = tl.load(grad_out_ptr, ...).to(tl.float32)  # upcast
#     gelu_grad = gelu_backward(grad, ...)               # FP32 compute
#     d_input = (gelu_grad @ weight).to(tl.bfloat16)    # downcast
#     tl.store(d_input_ptr, d_input)
```

---

## Hands-On Exercises

### Exercise 1: Measure torch.compile Training Speedup (45 min)

Compare training throughput (samples/sec) for a GPT-2 model with and without `torch.compile`. Measure for `mode="default"`, `mode="reduce-overhead"`, and `mode="max-autotune"`. Profile with `torch.profiler` to identify which kernels were fused.

```python
import torch
from torch.profiler import profile, ProfilerActivity

model = GPT2Model().cuda()
compiled = torch.compile(model, mode="max-autotune")

# Warmup
for _ in range(3):
    loss = compiled(dummy_input).sum()
    loss.backward()

# Profile
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(10):
        loss = compiled(dummy_input).sum()
        loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

### Exercise 2: Activation Checkpointing Impact (30 min)

Train a 24-layer transformer with and without activation checkpointing. Measure:
- Peak GPU memory
- Training step time (expect ~33% overhead)
- Verify gradients are identical (no numerical difference)

### Exercise 3: Communication-Computation Overlap Analysis (45 min)

Using NVIDIA Nsight Systems, profile a DDP training step and identify:
1. Where gradient AllReduce starts relative to backward computation
2. How much overlap is achieved
3. Whether the communication stream is fully utilized or has gaps

Annotate the trace to mark compute vs communication time.

---

## Key Takeaways

1. **Training is harder to compile** — must optimize the joint forward-backward-optimizer graph with memory constraints and distributed communication
2. **Backward fusion matters** — fusing element-wise backward ops eliminates intermediate tensors and reduces kernel launches by 2-3×
3. **Compiler-driven checkpointing** — selectively recompute cheap ops (GeLU, LayerNorm) while saving expensive ones (matmuls) for optimal memory-compute tradeoff
4. **Memory planning** — liveness analysis determines when tensors can be freed or reused; peak memory occurs at the forward-backward boundary
5. **Overlap is key** — compilers schedule AllReduce on a separate CUDA stream overlapping with backward compute, hiding 80-95% of communication latency
6. **GSPMD goes furthest** — XLA's partitioner takes sharding annotations and automatically generates all communication ops, freeing users from manual parallelism

---

## Further Reading

- Ansel et al., "PyTorch 2: Faster Machine Learning Through Dynamic Python Bytecode Transformation and Graph Compilation" (2024)
- Xu et al., "GSPMD: General and Scalable Parallelization for ML Computation Graphs" (2021)
- Jia et al., "Beyond Data and Model Parallelism for Deep Neural Networks" (FlexFlow, 2019)
- Chen et al., "Training Deep Nets with Sublinear Memory Cost" (gradient checkpointing, 2016)
- NVIDIA, "Automatic Mixed Precision Training" documentation

---

## Tomorrow's Teaser

You've mastered the building blocks: parallelism strategies, communication primitives, and compiler optimizations. Starting tomorrow, you'll put it all together in a two-part **capstone project** — building an end-to-end training system that applies operator fusion, memory optimization, distributed training, and compiler techniques to train a real transformer model efficiently.
