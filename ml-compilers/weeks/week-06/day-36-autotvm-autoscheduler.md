# Day 36: AutoTVM & AutoScheduler (Ansor)

> **Phase III · Week 6 · Day 36 of 70 · 2.5 hours**

*"When the search space has 10¹⁰ configurations, you don't hand-tune — you let the machine learn to compile itself."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 35: Mini-Project — End-to-End TVM Compilation](../week-05/day-35-mini-project-tvm-compile.md) | [Day 37: MetaSchedule](day-37-metaschedule.md) | [Week 6: TVM Tuning & Backends](README.md) | [Phase III: Apache TVM Deep Dive](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Last week you compiled a model end-to-end with TVM, but every schedule was TVM's default — generic loop nests with no hardware-specific tuning. On a real GPU, those defaults often leave 2–5× performance on the table. AutoTVM and AutoScheduler (Ansor) solve this by **searching** for optimal schedules automatically. AutoTVM uses hand-written templates with configurable knobs; Ansor generates schedules from scratch without templates. Together they're the reason TVM matches or beats vendor libraries on many workloads — and understanding how they work is essential for anyone deploying TVM in production.

---

## 1. The Auto-Tuning Problem

### Why Manual Scheduling Doesn't Scale

A single matrix multiplication on a modern GPU has a staggering number of schedule choices:

```
Tile sizes:       [1, 2, 4, 8, 16, 32, 64, 128] × 3 axes  →  ~500 combos
Unroll factors:   [1, 2, 4, 8, 16]                          →  ~5 choices
Vectorization:    [1, 2, 4, 8]                               →  ~4 choices
Thread binding:   blockIdx/threadIdx assignments             →  ~10 variants
Shared memory:    cache or not, tile sizes                   →  ~20 variants
───────────────────────────────────────────────────────────────
Total:            ~2,000,000 candidate schedules
```

For a model with 50+ operators, the combined space is effectively infinite. Auto-tuning uses **machine learning** to navigate this space efficiently.

### The Two Generations

```
┌───────────────────────────────────────────────────────────────┐
│                  TVM Auto-Tuning Evolution                    │
│                                                               │
│  Gen 1: AutoTVM (2018)           Gen 2: Ansor (2020)          │
│  ┌─────────────────────┐         ┌─────────────────────────┐  │
│  │ Human writes template│         │ No template needed      │  │
│  │ with ConfigSpace     │         │ Auto-generates sketches  │  │
│  │ knobs                │         │ from compute definition  │  │
│  │                      │         │                          │  │
│  │ Search over knob     │         │ Evolutionary search +    │  │
│  │ values only          │         │ cost model over full     │  │
│  │                      │         │ schedule space           │  │
│  └─────────────────────┘         └─────────────────────────┘  │
│  Pros: predictable,              Pros: no expert knowledge,   │
│        fast convergence                 better peak perf       │
│  Cons: needs expert,              Cons: slower search,         │
│        limited space                    larger search space     │
└───────────────────────────────────────────────────────────────┘
```

---

## 2. AutoTVM: Template-Based Tuning

AutoTVM requires a human expert to write a **schedule template** — a parameterized schedule with tunable knobs. The tuner then searches over knob values.

### ConfigSpace and Knobs

```python
import tvm
from tvm import te, autotvm
import numpy as np

# Define a tunable matrix multiplication
@autotvm.template("matmul")
def matmul_template(N, L, M, dtype="float32"):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)
    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

    s = te.create_schedule(C.op)

    # ---- ConfigSpace: define tunable knobs ----
    cfg = autotvm.get_config()

    # Tile sizes for the output axes
    cfg.define_split("tile_y", N, num_outputs=2)    # splits i
    cfg.define_split("tile_x", M, num_outputs=2)    # splits j
    cfg.define_split("tile_k", L, num_outputs=2)    # splits k

    # Apply the splits
    yo, yi = cfg["tile_y"].apply(s, C, C.op.axis[0])
    xo, xi = cfg["tile_x"].apply(s, C, C.op.axis[1])
    ko, ki = cfg["tile_k"].apply(s, C, k)

    # Reorder: outer loops → reduction → inner compute
    s[C].reorder(yo, xo, ko, yi, ki, xi)

    # Optional: unroll inner loops
    cfg.define_knob("auto_unroll_max_step", [0, 64, 512, 1024])
    s[C].pragma(yo, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)

    return s, [A, B, C]
```

### Key Knob Types

| Knob | Purpose | Example |
|:--|:--|:--|
| `define_split` | Tile a loop axis into factors | `tile_y: 1024 → (32, 32)` |
| `define_knob` | Choose from discrete values | `unroll: [0, 64, 512]` |
| `define_reorder` | Permute loop axes | `[i, j, k] → [j, i, k]` |
| `define_annotate` | Loop annotations | `vectorize`, `unroll`, `parallel` |

### Running AutoTVM Tuning

```python
# Define the task
N, L, M = 1024, 1024, 1024
task = autotvm.task.create("matmul", args=(N, L, M, "float32"),
                           target="llvm -mcpu=core-avx2")

print(f"ConfigSpace size: {len(task.config_space)}")
# → ConfigSpace size: 1728

# Configure the tuner
measure_option = autotvm.measure_option(
    builder="local",
    runner=autotvm.LocalRunner(number=5, repeat=3, timeout=10),
)

# XGBoost cost model + simulated annealing search
tuner = autotvm.tuner.XGBTuner(task, feature_type="curve")

# Run 200 trials
tuner.tune(
    n_trial=200,
    measure_option=measure_option,
    callbacks=[
        autotvm.callback.progress_bar(200),
        autotvm.callback.log_to_file("matmul_autotvm.json"),
    ],
)
```

### Cost Model Architecture

AutoTVM's XGBoost cost model predicts execution time from schedule features:

$$\hat{t} = f_{\text{XGB}}(\mathbf{x})$$

where $\mathbf{x}$ encodes loop structure (tile sizes, nesting order, annotations). The model is retrained after every batch of hardware measurements to improve predictions.

---

## 3. AutoScheduler (Ansor): Template-Free Tuning

Ansor eliminates templates entirely. Given only a TE compute definition, it:

1. **Generates sketches** — high-level schedule structures (e.g., "tile + fuse + parallelize")
2. **Annotates sketches** — fills in concrete tile sizes, unroll factors
3. **Evaluates** via cost model + selective hardware measurement
4. **Evolves** — mutates and recombines top-performing schedules

### Using Ansor

```python
from tvm import auto_scheduler

# Step 1: Define the computation (no template!)
@auto_scheduler.register_workload
def matmul_ansor(N, L, M, dtype="float32"):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)
    k = te.reduce_axis((0, L), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    return [A, B, C]

# Step 2: Create task
target = tvm.target.Target("llvm -mcpu=core-avx2")
task = auto_scheduler.SearchTask(
    func=matmul_ansor, args=(1024, 1024, 1024, "float32"), target=target
)

print(f"Compute DAG:\n{task.compute_dag}")

# Step 3: Configure tuning
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=1000,
    measure_callbacks=[auto_scheduler.RecordToFile("matmul_ansor.json")],
    verbose=2,
)

# Step 4: Run search
task.tune(tune_option)

# Step 5: Apply best schedule
sch, args = task.apply_best("matmul_ansor.json")
print(tvm.lower(sch, args, simple_mode=True))
```

### Sketch Generation

Ansor generates sketches by analyzing the compute DAG:

```
Compute: C[i,j] = sum(A[i,k] * B[k,j], axis=k)
           │
           ▼
┌──────────────────────────────────────────────┐
│ Sketch 1: Multi-level tiling                 │
│   tile(i, [t1, t2, t3]) → io, im, ii        │
│   tile(j, [t4, t5, t6]) → jo, jm, ji        │
│   tile(k, [t7, t8])     → ko, ki             │
│   reorder(io, jo, ko, im, jm, ki, ii, ji)    │
│   cache_write(C, "local")                     │
├──────────────────────────────────────────────┤
│ Sketch 2: Vectorized + parallel              │
│   tile(i, [t1, t2]) → io, ii                 │
│   tile(j, [t3, t4]) → jo, ji                 │
│   parallel(io)                                │
│   vectorize(ji)                               │
├──────────────────────────────────────────────┤
│ Sketch 3: Cache + reorder                    │
│   cache_read(A, "local")                      │
│   tile + fuse + bind to threads               │
└──────────────────────────────────────────────┘
```

Each sketch is then "annotated" — concrete values are assigned by evolutionary search.

---

## 4. Tuning a Full Model with Ansor

For real deployment, you tune all operators in a model together:

```python
import tvm.relay as relay
from tvm.contrib import graph_executor

# Import a model (e.g., ResNet-18 from ONNX)
import onnx
onnx_model = onnx.load("resnet18.onnx")
mod, params = relay.frontend.from_onnx(onnx_model, shape={"input": (1, 3, 224, 224)})

target = tvm.target.Target("cuda -arch=sm_80")

# Extract tuning tasks — one per unique operator
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

print(f"Extracted {len(tasks)} tuning tasks:")
for i, task in enumerate(tasks):
    print(f"  Task {i}: {task.desc}  (weight={task_weights[i]})")

# Tune all tasks with a shared cost model
tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=2000,           # total budget across all tasks
    runner=auto_scheduler.LocalRunner(timeout=30, repeat=3),
    measure_callbacks=[auto_scheduler.RecordToFile("resnet18_ansor.json")],
)
tuner.tune(tune_option)

# Compile with the best schedules
with auto_scheduler.ApplyHistoryBest("resnet18_ansor.json"):
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
```

### Task Scheduling Strategy

The `TaskScheduler` allocates measurement trials across operators using weights (proportional to their estimated runtime impact):

$$\text{trials}_i = \left\lfloor \frac{w_i}{\sum_j w_j} \cdot N_{\text{total}} \right\rfloor$$

This ensures the tuner spends the most time on the **hottest** operators (e.g., large convolutions) rather than wasting trials on cheap ops.

---

## 5. Analyzing Tuning Logs

### Speedup Progression

```python
import json

# Load tuning records
records = []
with open("resnet18_ansor.json", "r") as f:
    for line in f:
        records.append(json.loads(line))

# Plot performance over trials
import matplotlib.pyplot as plt

costs = [r["result"][0][0] if r["result"][0][0] < 1e8 else None for r in records]
valid = [(i, c) for i, c in enumerate(costs) if c is not None]
xs, ys = zip(*valid)

# Running best
best_so_far = []
current_best = float("inf")
for _, c in valid:
    current_best = min(current_best, c)
    best_so_far.append(current_best)

plt.figure(figsize=(10, 4))
plt.plot(xs, ys, ".", alpha=0.3, label="Individual trials")
plt.plot(xs, best_so_far, "r-", linewidth=2, label="Best so far")
plt.xlabel("Trial number")
plt.ylabel("Latency (s)")
plt.title("Ansor Tuning Progression")
plt.legend()
plt.yscale("log")
plt.savefig("tuning_progression.png")
```

### Typical Speedup Curve

```
Latency
  │
  │  ·  ·                                   Individual trials
  │ ···· ·  ·
  │  ·····  · ·   ·
  │   ·····  ··· · ·
  │────────────────────── Best so far
  │     ·   ·  ···  ·  ·  ·
  │         ────────────────────
  │              ···  ·    · ···
  │                ─────────────── Plateau (~optimal)
  │
  └──────────────────────────────────── Trial number
     0      200     500     1000    2000
```

The curve typically shows rapid improvement in the first 20% of trials, then diminishing returns.

---

## 6. AutoTVM vs Ansor: When to Use Which

| Criterion | AutoTVM | Ansor (AutoScheduler) |
|:--|:--|:--|
| **Template required** | Yes — expert writes it | No — automatic |
| **Search space** | Limited to template knobs | Full schedule space |
| **Peak performance** | Good (bounded by template) | Often better |
| **Convergence speed** | Faster (smaller space) | Slower (larger space) |
| **Custom hardware** | Easy to add template | May need new rules |
| **Maintenance** | Templates per operator | One framework |
| **Recommended for** | Known operators, quick wins | New models, best perf |

> **In practice**: Ansor is the recommended approach for TVM ≥ 0.8. AutoTVM templates remain useful for custom operators or when Ansor's sketch generation doesn't cover a pattern.

---

## Hands-On Exercises

### Exercise 1: Tune a Conv2D with AutoTVM (25 min)

Write an AutoTVM template for a 2D convolution (`N=1, C=64, H=W=56, kernel=3×3, out_channels=128`). Add knobs for:
- Tiling the output height/width
- Unroll factor for the reduction axis
- Whether to vectorize the innermost loop

Run 100 trials and compare the best result against TVM's default schedule.

### Exercise 2: Ansor on MobileNetV2 (30 min)

Use `auto_scheduler.extract_tasks()` to pull tuning tasks from MobileNetV2 (from last week's mini-project). Tune with 500 trials total. Questions:
1. How many unique tasks are extracted?
2. Which task gets the most trials? Why?
3. What's the end-to-end speedup vs the untuned compilation?

### Exercise 3: Compare Cost Models (20 min)

AutoTVM supports multiple tuners. Run the same task with:
- `XGBTuner` (XGBoost cost model)
- `GATuner` (genetic algorithm, no cost model)
- `RandomTuner` (baseline)

Plot the convergence curves. How many trials does each need to reach 90% of XGBTuner's best?

---

## Key Takeaways

1. **Auto-tuning is essential** — default TVM schedules leave 2–5× on the table compared to tuned ones
2. **AutoTVM** uses expert-written templates with configurable knobs; search is fast but bounded by template quality
3. **Ansor** generates schedules from scratch via sketch generation + evolutionary search — no templates needed
4. **Cost models** (XGBoost) dramatically reduce the number of hardware measurements needed
5. **Task schedulers** allocate tuning budget proportionally to operator runtime impact
6. **Tuning logs** are reusable — save them and apply with `ApplyHistoryBest` for reproducible compilation

---

## Further Reading

- [AutoTVM: Learning-based Compiler Optimization](https://arxiv.org/abs/1805.08166) — the original AutoTVM paper (Chen et al., 2018)
- [Ansor: Generating High-Performance Tensor Programs](https://arxiv.org/abs/2006.06762) — the AutoScheduler paper (Zheng et al., 2020)
- [TVM Auto-Tuning Tutorial](https://tvm.apache.org/docs/how_to/tune_with_autotvm/index.html) — official AutoTVM guide
- [Auto-Scheduling Tutorial](https://tvm.apache.org/docs/how_to/tune_with_autoscheduler/index.html) — official Ansor guide

---

## Tomorrow's Preview

AutoTVM and Ansor were groundbreaking, but they have limitations: AutoTVM needs templates, Ansor's search space can be too large, and neither integrates cleanly with TIR's `tvm.script`. **Day 37** introduces **MetaSchedule** — TVM's unified tuning framework that replaces both, built around trace-based scheduling and composable mutators.
