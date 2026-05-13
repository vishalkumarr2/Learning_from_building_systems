# Day 37: MetaSchedule

> **Phase III · Week 6 · Day 37 of 70 · 2.5 hours**

*"The best framework isn't the one with the most features — it's the one where every feature composes with every other."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 36: AutoTVM & AutoScheduler (Ansor)](day-36-autotvm-autoscheduler.md) | [Day 38: BYOC — Bring Your Own Codegen](day-38-byoc-custom-backends.md) | [Week 6: TVM Tuning & Backends](README.md) | [Phase III: Apache TVM Deep Dive](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Yesterday you saw two generations of TVM auto-tuning: AutoTVM (template-based) and Ansor (template-free). Both work, but they're separate systems with different APIs, cost models, and search strategies. **MetaSchedule** unifies them into a single framework built on TIR schedule traces — the same primitives you learned in Week 5. This means you can: (1) write custom schedule rules in the same language you already know, (2) replay and mutate schedules deterministically, and (3) share a single database of tuning results across projects. MetaSchedule is the future of TVM tuning, and as of TVM 0.12+ it's the recommended approach.

---

## 1. Why a Unified Framework?

### Problems with AutoTVM + Ansor

```
┌─────────────────────────────────────────────────────────────┐
│              Before MetaSchedule                             │
│                                                              │
│  AutoTVM                          Ansor                      │
│  ┌──────────────┐                 ┌──────────────┐           │
│  │ Template API  │                 │ Sketch API   │           │
│  │ ConfigSpace   │                 │ SearchPolicy │           │
│  │ XGBTuner      │                 │ CostModel    │           │
│  │ JSON logs     │                 │ JSON logs    │           │
│  └──────────────┘                 └──────────────┘           │
│        │                                │                    │
│        └─── Different APIs ─────────────┘                    │
│             Different search strategies                      │
│             Incompatible log formats                         │
│             Neither uses TIR schedule primitives directly    │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│              After MetaSchedule                              │
│                                                              │
│  ┌──────────────────────────────────────────────────┐        │
│  │              MetaSchedule                         │        │
│  │  ┌────────────┐ ┌───────────┐ ┌────────────────┐ │        │
│  │  │ Schedule   │ │ Mutators  │ │ Database       │ │        │
│  │  │ Rules      │ │           │ │ (JSON/SQLite)  │ │        │
│  │  └────────────┘ └───────────┘ └────────────────┘ │        │
│  │       Built on TIR Schedule Traces                │        │
│  └──────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### MetaSchedule Design Principles

| Principle | Description |
|:--|:--|
| **Trace-based** | Schedules are recorded as a sequence of TIR primitive calls |
| **Composable** | Rules and mutators compose freely — mix and match |
| **Replay-able** | Any schedule can be replayed deterministically from its trace |
| **Database-backed** | Results stored in JSON or SQLite for cross-session reuse |
| **Unified** | One API for both template-style and template-free tuning |

---

## 2. Schedule Traces

A **trace** records every schedule primitive call as a structured log. This is the key abstraction — it makes schedules inspectable, replayable, and mutable.

### What a Trace Looks Like

```python
import tvm
from tvm import tir
from tvm.script import tir as T

@T.prim_func
def matmul(a: T.handle, b: T.handle, c: T.handle):
    A = T.match_buffer(a, (1024, 1024), "float32")
    B = T.match_buffer(b, (1024, 1024), "float32")
    C = T.match_buffer(c, (1024, 1024), "float32")
    for i, j, k in T.grid(1024, 1024, 1024):
        with T.block("C"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = 0.0
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

# Create a schedule from the PrimFunc
sch = tir.Schedule(matmul)

# Apply primitives — each call is recorded in the trace
block = sch.get_block("C")
i, j, k = sch.get_loops(block)

# Tile the i-axis
io, ii = sch.split(i, factors=[32, 32])
# Tile the j-axis  
jo, ji = sch.split(j, factors=[32, 32])
# Tile the k-axis
ko, ki = sch.split(k, factors=[64, 16])

# Reorder
sch.reorder(io, jo, ko, ii, ki, ji)

# Print the trace — the recorded sequence of decisions
print(sch.trace)
```

### Trace Output

```
Instruction 0: GetBlock(name="C")
Instruction 1: GetLoops(block=b0)
Instruction 2: Split(loop=l0, factors=[32, 32])     # i → io, ii
Instruction 3: Split(loop=l1, factors=[32, 32])     # j → jo, ji
Instruction 4: Split(loop=l2, factors=[64, 16])     # k → ko, ki
Instruction 5: Reorder(order=[l3, l5, l7, l4, l8, l6])
```

The trace is **deterministic** — replaying these instructions on the same PrimFunc always produces the same schedule. This is what makes mutation-based search possible.

---

## 3. Schedule Rules and Space Generators

MetaSchedule uses **Schedule Rules** to define the search space. A rule is a function that takes a block and produces candidate schedule decisions.

### Built-in Schedule Rules

```python
from tvm.meta_schedule import schedule_rule

# Common rules shipped with TVM:
rules = [
    schedule_rule.AutoInline(            # inline trivial computations
        into_producer=False,
        into_consumer=True,
    ),
    schedule_rule.MultiLevelTiling(       # the workhorse: multi-level tiling
        structure="SSRSRS",              # Spatial-Spatial-Reduce-Spatial-Reduce-Spatial
        tile_binds=None,                 # for GPU: ["blockIdx.x", "threadIdx.x"]
        max_innermost_factor=64,
        vector_load_lens=None,
        reuse_read=None,
        reuse_write=None,
    ),
    schedule_rule.ParallelizeVectorizeUnroll(
        max_jobs_per_core=16,
        max_vectorize_extent=32,
        unroll_max_steps=[0, 16, 64, 512],
        unroll_explicit=True,
    ),
]
```

### Multi-Level Tiling Structure

The `"SSRSRS"` string encodes the tiling pattern:

```
S = Spatial loop level
R = Reduction loop level

SSRSRS for a matmul C[i,j] += A[i,k] * B[k,j]:

Level 1 (S): tile i → i0, remaining
Level 2 (S): tile j → j0, remaining  
Level 3 (R): tile k → k0, remaining
Level 4 (S): tile i_remaining → i1, i2
Level 5 (R): tile k_remaining → k1, k2
Level 6 (S): tile j_remaining → j1, j2

Result: reorder(i0, j0, k0, i1, k1, j1, i2, k2, j2)
```

For GPU targets, the tiling structure becomes `"SSSRRSRS"` with thread/block bindings:

```
Level 1 (S): blockIdx.x   ← coarsest spatial
Level 2 (S): vthread       ← virtual thread (bank conflict avoidance)
Level 3 (S): threadIdx.x   ← fine-grained spatial
Level 4 (R): reduction outer
Level 5 (R): reduction inner
Level 6 (S): register tile
Level 7 (R): reduction innermost
Level 8 (S): vectorize      ← innermost spatial
```

---

## 4. Mutators and Evolutionary Search

Instead of generating schedules from scratch each time, MetaSchedule **mutates** existing good schedules to explore neighboring configurations.

### How Mutators Work

```
Best schedule so far (trace):
  Split(i, [32, 32])
  Split(j, [32, 32])
  Split(k, [64, 16])
  Reorder(io, jo, ko, ii, ki, ji)
         │
         │  Mutator: MutateTileSize
         ▼
Candidate 1:                    Candidate 2:
  Split(i, [64, 16])  ← changed   Split(i, [32, 32])
  Split(j, [32, 32])              Split(j, [16, 64])  ← changed
  Split(k, [64, 16])              Split(k, [128, 8])  ← changed
  Reorder(...)                     Reorder(...)
```

### Built-in Mutators

| Mutator | What It Mutates |
|:--|:--|
| `MutateTileSize` | Changes tile factors within valid divisor sets |
| `MutateUnroll` | Changes unroll step from the allowed set |
| `MutateParallel` | Toggles parallelization on/off for outer loops |
| `MutateComputeLocation` | Moves `compute_at` to a different loop level |
| `MutateAutoInline` | Toggles inline decisions for intermediate stages |

### Search Loop

```python
from tvm import meta_schedule as ms

# The evolutionary search loop (conceptual)
population = [generate_initial_schedule(rules) for _ in range(64)]

for generation in range(num_generations):
    # 1. Score population with cost model
    scores = cost_model.predict(population)
    
    # 2. Select top-k (tournament selection)
    elites = select_top_k(population, scores, k=16)
    
    # 3. Mutate elites to create new candidates
    children = [mutator.mutate(random.choice(elites)) for _ in range(48)]
    
    # 4. Measure a few on hardware (to retrain cost model)
    measured = measure_on_hardware(children[:8])
    cost_model.update(measured)
    
    # 5. Next generation
    population = elites + children
```

---

## 5. Running MetaSchedule Tuning

### Single Operator Tuning

```python
from tvm import meta_schedule as ms

# Tune a single PrimFunc
database = ms.tune_tir(
    mod=tvm.IRModule({"main": matmul}),
    target="llvm -mcpu=core-avx2",
    max_trials_global=500,
    num_trials_per_iter=64,
    work_dir="./tune_matmul",
    # Uses default rules, mutators, and cost model
)

# Retrieve the best schedule
workload = database.commit_workload(tvm.IRModule({"main": matmul}))
top_record = database.get_top_k(workload, top_k=1)[0]
best_sch = top_record.as_schedule()
print(best_sch.mod.script())
```

### Full Model Tuning

```python
import tvm.relay as relay

# Load model
mod, params = relay.frontend.from_pytorch(traced_model, input_infos)

target = tvm.target.Target("cuda -arch=sm_80")

# Tune all operators via MetaSchedule
database = ms.relay_integration.tune_relay(
    mod=mod,
    params=params,
    target=target,
    work_dir="./tune_resnet",
    max_trials_global=3000,
    strategy="evolutionary",
)

# Compile with tuned schedules
lib = ms.relay_integration.compile_relay(
    database=database,
    mod=mod,
    params=params,
    target=target,
)
```

### Database Persistence

MetaSchedule stores results in a structured database (JSON lines or SQLite):

```python
# Re-use previous tuning results
database = ms.database.JSONDatabase(
    path_workload="./tune_resnet/database_workload.json",
    path_tuning_record="./tune_resnet/database_tuning_record.json",
)

# Check what's cached
workloads = database.get_all_tuning_records()
print(f"Database has {len(workloads)} tuned workloads")

# Apply cached schedules to a new compilation
lib = ms.relay_integration.compile_relay(
    database=database,
    mod=new_mod,     # works if operators match
    params=new_params,
    target=target,
)
```

---

## 6. Custom Space Generators and Mutators

### Writing a Custom Schedule Rule

```python
from tvm.meta_schedule import schedule_rule as sr

@sr._register_schedule_rule("my_custom_tiling")
class MyCustomTiling(sr.ScheduleRule):
    """Custom tiling that always uses power-of-2 factors."""
    
    def _initialize_with_tune_context(self, context):
        self.target = context.target
    
    def apply(self, sch, block):
        loops = sch.get_loops(block)
        candidates = []
        
        for loop in loops:
            extent = sch.get(loop).extent
            # Only allow power-of-2 tile sizes
            factors = [2**i for i in range(1, 10) if 2**i <= extent]
            if factors:
                for f in factors:
                    new_sch = sch.copy()
                    new_sch.split(loop, factors=[None, f])
                    candidates.append(new_sch)
        
        return candidates
```

### Writing a Custom Mutator

```python
from tvm.meta_schedule import mutator as mut
import random

@mut._register_mutator("my_tile_mutator")
class MyTileMutator(mut.Mutator):
    """Mutate tile sizes by doubling or halving."""
    
    def _initialize_with_tune_context(self, context):
        pass
    
    def apply(self, trace, _):
        # Find Split instructions in the trace
        for i, inst in enumerate(trace.insts):
            if inst.kind.name == "Split":
                # Randomly double or halve a factor
                factors = list(trace.decisions[inst])
                idx = random.randint(0, len(factors) - 1)
                if random.random() < 0.5 and factors[idx] > 1:
                    factors[idx] //= 2
                else:
                    factors[idx] *= 2
                # Create mutated trace
                new_trace = trace.copy()
                new_trace.decisions[inst] = factors
                return new_trace
        return None  # no valid mutation found
```

---

## Hands-On Exercises

### Exercise 1: MetaSchedule vs Ansor Comparison (30 min)

Tune a depthwise convolution (`N=1, C=128, H=W=56, kernel=3×3`) with both MetaSchedule and Ansor. Use 300 trials each. Compare:
1. Best latency achieved
2. Time to reach 90% of best latency
3. Number of hardware measurements required

### Exercise 2: Inspect a Trace (20 min)

After tuning a matmul with MetaSchedule, print the best schedule's trace. For each instruction:
1. What primitive does it call?
2. What were the decisions (tile sizes, annotations)?
3. Replay the trace on a fresh schedule to verify determinism.

### Exercise 3: Database Reuse (25 min)

Tune ResNet-18 for `cuda`. Save the database. Then load it and compile ResNet-34 — do any workloads hit the cache? Why or why not? (Hint: which operators have identical shapes?)

---

## Key Takeaways

1. **MetaSchedule unifies** AutoTVM and Ansor into a single trace-based framework
2. **Traces** record schedule decisions as replayable instruction sequences — the core abstraction
3. **Schedule Rules** define the initial search space; **Mutators** explore it via local perturbation
4. **Multi-level tiling** (`"SSRSRS"` / `"SSSRRSRS"`) is the most important schedule rule
5. **Database persistence** enables cross-session and cross-model schedule reuse
6. **Custom rules and mutators** extend MetaSchedule for novel hardware or operators

---

## Further Reading

- [MetaSchedule RFC](https://discuss.tvm.apache.org/t/rfc-meta-schedule-autotensorir/12539) — original design proposal
- [TVM Unity: MetaSchedule Tutorial](https://tvm.apache.org/docs/how_to/tune_with_meta_schedule/index.html) — official guide
- [TIR Schedule Primitives Reference](https://tvm.apache.org/docs/reference/api/python/tir.html#tvm.tir.Schedule) — the primitives that traces are built from
- [Tensor Program Optimization with Probabilistic Programs](https://arxiv.org/abs/2205.13603) — MetaSchedule's theoretical foundations

---

## Tomorrow's Preview

Even the best auto-tuner can't beat a vendor library hand-optimized over years for a specific operator on specific hardware. **Day 38** introduces **BYOC (Bring Your Own Codegen)** — TVM's framework for offloading subgraphs to external libraries like cuDNN, TensorRT, or DNNL, combining TVM's graph-level optimizations with vendor-tuned kernels.
