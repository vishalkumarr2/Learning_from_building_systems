# Day 11: torch.profiler & Trace Analysis

> **Phase I · Week 2 · Day 11 of 70 · 2.5 hours**

*"Profiling without a hypothesis is stamp collecting. Profiling with one is engineering."*

| Previous | Next | Week | Phase | Curriculum |
|----------|------|------|-------|------------|
| [Day 10: Custom C++ Extensions](day-10-custom-cpp-extensions.md) | [Day 12: Eager vs Graph Mode](day-12-eager-vs-graph-mode.md) | [Week 2: PyTorch Internals](README.md) | Phase I: Foundations | [Curriculum Home](../../STUDY-PLAN.md) |

---

## Why This Matters

You've written a training loop that takes 400 ms per step. Is the GPU compute-bound
or memory-bound? Is the CPU feeding data fast enough? Is there dead time between kernel
launches? Without profiling, you're guessing. `torch.profiler` answers these questions
with microsecond-level traces that show exactly where time is spent — from Python-level
ops down to individual CUDA kernel launches. This is the tool that separates "I think
it's slow because..." from "it's slow because NCCL all-reduce overlaps 0% with compute."

---

## 1. torch.profiler API

### 1.1 Basic Usage

```python
import torch
from torch.profiler import profile, ProfilerActivity, schedule

model = torch.nn.Linear(4096, 4096).cuda()
x = torch.randn(64, 4096, device='cuda')

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for _ in range(5):
        y = model(x)
        y.sum().backward()

# Print a summary table
print(prof.key_averages().table(
    sort_by="cuda_time_total",
    row_limit=15,
))
```

**Output (simplified):**

```
Name                      CPU total   CUDA total   Calls
────────────────────────────────────────────────────────
aten::addmm               120.3 us    89.2 us      5
aten::sum                   18.1 us    12.4 us      5
AddmmBackward0             210.5 us   156.3 us      5
aten::mm                   198.2 us   142.1 us     10
...
Self CPU time total: 1.2 ms
Self CUDA time total: 0.8 ms
```

### 1.2 Key Parameters

| Parameter | Purpose |
|-----------|---------|
| `activities` | What to trace: `CPU`, `CUDA`, or both |
| `record_shapes` | Log tensor shapes (useful for identifying which call is which) |
| `profile_memory` | Track memory allocations/frees |
| `with_stack` | Capture Python call stacks (enables source-level attribution) |
| `with_flops` | Estimate FLOPs for matrix operations |
| `with_modules` | Annotate ops with their `nn.Module` name |

---

## 2. Scheduled Profiling for Training Loops

Profiling every step wastes resources. Use `schedule()` to skip warmup and
capture only a few representative steps:

```python
from torch.profiler import profile, schedule, tensorboard_trace_handler

# Profile steps 5-9 (after 5 warmup steps)
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(
        wait=1,       # skip 1 step (cold cache)
        warmup=4,     # warmup 4 steps (JIT, cache warming)
        active=5,     # actually record 5 steps
        repeat=1,     # do this once
    ),
    on_trace_ready=tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step in range(11):
        # Training step
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        prof.step()  # signal end of step
```

```
Timeline:
  Step:   0    1    2    3    4    5    6    7    8    9   10
        wait │──── warmup ────│──── active (recorded) ────│
```

---

## 3. Trace Visualization

### 3.1 Chrome Trace (JSON)

```python
# Export Chrome-compatible trace
prof.export_chrome_trace("trace.json")

# Open in:
#   chrome://tracing     (Chrome browser)
#   https://ui.perfetto.dev  (Perfetto — better for large traces)
```

The trace shows a **Gantt chart** of all operations:

```
CPU Thread 0:
├─ addmm ─────────┤          ├─ sum ──┤
│                  │          │        │
GPU Stream 0:      │          │        │
│      ├─ sgemm_128x128 ─────┤ ├─ reduce ──┤
│      │                      │ │           │
│  idle│      GPU active      │ │GPU active │
│ (gap)│                      │ │           │
└──────┴──────────────────────┴─┴───────────┘

Key observations:
  - Gap between CPU dispatch and GPU start = launch latency
  - Gap between GPU kernels = CPU overhead (Python, autograd)
  - Overlap of CPU prep with GPU execution = good pipelining
```

### 3.2 Perfetto UI

For large traces (>50 MB), Perfetto handles them much better than Chrome:

1. Go to `https://ui.perfetto.dev`
2. Click "Open trace file" → select your `trace.json`
3. Use `W`/`S` to zoom, `A`/`D` to pan
4. Click on any event to see duration, call stack, tensor shapes

### 3.3 TensorBoard Plugin

```bash
# Launch TensorBoard with the profiler plugin
tensorboard --logdir=./log/profiler

# Navigate to: http://localhost:6006/#pytorch_profiler
```

The TensorBoard plugin provides:
- **Overview tab** — GPU utilization %, step time breakdown
- **Operator view** — sorted by self/total time
- **Trace view** — interactive timeline (uses Perfetto under the hood)
- **Memory view** — allocation timeline with peak tracking
- **Module view** — time per `nn.Module` (requires `with_modules=True`)

---

## 4. Key Metrics to Extract

### 4.1 GPU Utilization and Idle Time

```python
events = prof.key_averages()

total_cuda = sum(e.cuda_time_total for e in events)
total_cpu = sum(e.cpu_time_total for e in events)

# Rough GPU utilization
# (better metrics come from the trace view)
print(f"Total CPU time: {total_cpu / 1e3:.1f} ms")
print(f"Total CUDA time: {total_cuda / 1e3:.1f} ms")
```

### 4.2 CPU-GPU Overlap Analysis

```
Ideal pipeline (high overlap):
CPU: [prep1][prep2][prep3][prep4]
GPU:    [kernel1][kernel2][kernel3][kernel4]
          ↑ GPU never idles

Bad pipeline (serial execution):
CPU: [prep1]          [prep2]          [prep3]
GPU:        [kernel1]          [kernel2]
                  ↑ GPU idles while CPU prepares
```

Detect overlap issues by looking at the trace:
- **Long gaps between GPU kernels** → CPU is the bottleneck (Python overhead,
  data loading, autograd graph construction)
- **GPU kernels much shorter than CPU ops** → operations are too small to
  saturate the GPU (consider kernel fusion or batching)

### 4.3 FLOPS Analysis

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_flops=True,
) as prof:
    y = model(x)

# Check FLOPS for each op
for event in prof.key_averages():
    if event.flops:
        gflops = event.flops / 1e9
        time_s = event.cuda_time_total / 1e6
        tflops = (gflops / time_s / 1e3) if time_s > 0 else 0
        print(f"{event.key:30s} {gflops:8.1f} GFLOP  {tflops:6.1f} TFLOP/s")

# Compare to theoretical peak:
# A100 FP32: 19.5 TFLOP/s
# A100 TF32: 156 TFLOP/s
# A100 FP16: 312 TFLOP/s (with tensor cores)
```

---

## 5. Memory Profiling with record_memory_history

For deep memory debugging, use the memory history recorder:

```python
import torch

# Start recording all allocations
torch.cuda.memory._record_memory_history(
    max_entries=100000,
    context="all",          # record Python stack traces
)

# Run your workload
model = torch.nn.TransformerEncoder(
    torch.nn.TransformerEncoderLayer(d_model=512, nhead=8),
    num_layers=6,
).cuda()
x = torch.randn(32, 128, 512, device='cuda')

y = model(x)
y.sum().backward()

# Save snapshot
torch.cuda.memory._dump_snapshot("mem_trace.pickle")
torch.cuda.memory._record_memory_history(enabled=None)

# Visualize at https://pytorch.org/memory_viz
# Upload mem_trace.pickle → see allocation timeline with stacks
```

### Memory Timeline Interpretation

```
Memory
  ▲
  │  ┌─── forward pass allocations (activations)
  │  │       ┌─── peak (all activations live)
  │  │       │          ┌─── backward: activations freed,
  │  │       │          │    gradients allocated
  │  ▼       ▼          ▼
  │ ┌──┬──┬──┬──┐  ┌──┬──┐
  │ │a1│a2│a3│a4│  │g4│g3│   ← activations freed LIFO
  ├─┤  │  │  │  ├──┤  │  │
  │ │params + opt │  │  │  │
  └─┴──┴──┴──┴──┴──┴──┴──┴──▶ time

Each block shows:
  - Which Python line allocated it (from stack trace)
  - Size in bytes
  - Lifetime (alloc time → free time)
```

---

## 6. Practical Profiling Workflow

A systematic approach to finding and fixing performance bottlenecks:

```
Step 1: Profile with torch.profiler
────────────────────────────────────
  Export trace.json → open in Perfetto

Step 2: Check GPU utilization
─────────────────────────────
  Is GPU busy >80% of the time?
  YES → compute-bound → look at kernel efficiency
  NO  → launch-bound or data-starved → find the gaps

Step 3: Identify top-N kernels by time
──────────────────────────────────────
  prof.key_averages().table(sort_by="cuda_time_total")
  Focus optimization on the top 3-5 kernels

Step 4: Check arithmetic intensity
───────────────────────────────────
  For each top kernel:
    FLOPS / bytes_accessed = arithmetic intensity
    Compare to GPU's ops:byte ratio (A100: ~200 for TF32)
    Low intensity → memory-bound → fuse ops, reduce data movement
    High intensity → compute-bound → use tensor cores, reduce precision

Step 5: Check CPU-GPU overlap
─────────────────────────────
  In trace view: are there gaps between GPU kernels?
  YES → CPU bottleneck → use CUDA graphs, reduce Python overhead
  NO  → GPU is fully utilized, optimize individual kernels

Step 6: Profile memory
──────────────────────
  Use record_memory_history for peak memory analysis
  Identify largest activation tensors
  Apply gradient checkpointing to the worst offenders
```

---

## Hands-On Exercises

### Exercise 1: Profile a ResNet Training Step (30 min)

```python
import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity

model = models.resnet50().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
x = torch.randn(32, 3, 224, 224, device='cuda')
target = torch.randint(1000, (32,), device='cuda')
criterion = torch.nn.CrossEntropyLoss()

# Warmup
for _ in range(3):
    optimizer.zero_grad()
    loss = criterion(model(x), target)
    loss.backward()
    optimizer.step()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=True,
    with_stack=True,
) as prof:
    optimizer.zero_grad()
    loss = criterion(model(x), target)
    loss.backward()
    optimizer.step()

# TASKS:
# 1. Print top 10 ops by CUDA time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# 2. Export and view Chrome trace
prof.export_chrome_trace("resnet_trace.json")

# 3. Answer: What fraction of CUDA time is spent in convolutions
#    vs batch norm vs other ops?
```

### Exercise 2: Find the Bottleneck (30 min)

```python
# This training loop has a deliberate performance bug. Find it.

import torch
import torch.nn as nn

class SlowModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(256, 256) for _ in range(10)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
            # Hint: what does .item() do to CPU-GPU sync?
            if x.max().item() > 100:
                x = x / 100
        return x

model = SlowModel().cuda()
x = torch.randn(64, 256, device='cuda')

# Profile this and identify why it's slow
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
) as prof:
    for _ in range(10):
        y = model(x).sum()
        y.backward()

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

# QUESTION: What is the fix? Implement it and re-profile.
```

### Exercise 3: Memory Profile a Transformer (20 min)

```python
import torch

torch.cuda.memory._record_memory_history(max_entries=100000)

model = torch.nn.TransformerEncoder(
    torch.nn.TransformerEncoderLayer(
        d_model=1024, nhead=16, dim_feedforward=4096, batch_first=True
    ),
    num_layers=12,
).cuda()

x = torch.randn(16, 256, 1024, device='cuda')

y = model(x)
y.sum().backward()

torch.cuda.memory._dump_snapshot("transformer_mem.pickle")
torch.cuda.memory._record_memory_history(enabled=None)

# TASKS:
# 1. Upload transformer_mem.pickle to https://pytorch.org/memory_viz
# 2. Identify the peak memory point
# 3. Which layer's activations consume the most memory?
# 4. How much memory could gradient checkpointing save?
peak = torch.cuda.max_memory_allocated()
print(f"Peak memory: {peak / 1e9:.2f} GB")
```

---

## Key Takeaways

1. **`torch.profiler`** captures CPU ops, CUDA kernels, memory events, and Python
   stacks in a single unified trace
2. **`schedule(wait, warmup, active)`** avoids profiling cold starts and limits
   overhead to representative steps only
3. **Chrome/Perfetto traces** reveal CPU-GPU overlap, kernel launch gaps, and idle
   time that summary tables hide
4. **GPU utilization < 80%** usually means CPU overhead, Python-level bottlenecks,
   or excessive synchronization (`.item()`, `.cpu()`, `print`)
5. **`record_memory_history`** with the PyTorch memory visualizer pinpoints exactly
   which line of code caused peak memory or fragmentation

---

## Further Reading

- [PyTorch Profiler tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [Perfetto UI](https://ui.perfetto.dev/) for trace visualization
- [Holistic Trace Analysis (HTA)](https://github.com/facebookresearch/HolisticTraceAnalysis) — automated trace analysis
- [PyTorch Memory Visualizer](https://pytorch.org/memory_viz)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) — system-level GPU profiling

---

## Tomorrow's Preview

**Day 12: Eager vs Graph Mode** — We'll compare PyTorch's default eager execution
with graph-based execution (`torch.jit.trace`, `torch.jit.script`, `torch.compile`),
understand why graph capture enables optimization, and see how `torch.compile`
bridges the gap with its Dynamo + Inductor pipeline.
