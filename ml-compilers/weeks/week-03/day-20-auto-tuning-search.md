# Day 20: Auto-Tuning & Search Spaces

> **Phase II · Week 3 · Day 20 of 70 · 2.5 hours**

*"The best schedule for a matmul on an A100 is not the best schedule on an M2 — and no human can keep both in their head simultaneously. Let the machine search."*

---

| ← Previous | Next → | 📅 Week | 🔷 Phase | 📚 Curriculum |
|:--|:--|:--|:--|:--|
| [Day 19: Loop Tiling & Scheduling](day-19-loop-tiling-scheduling.md) | [Day 21: Mini-Project — FX Transform Pass](day-21-mini-project-fx-pass.md) | [Week 3: IRs & Passes](README.md) | [Phase II: Compiler Fundamentals](../../STUDY-PLAN.md) | [ML Compilers](../../README.md) |

---

## Why This Matters

Yesterday we saw that a single matmul has millions of legal schedules — different tile sizes, loop orders, parallelization strategies, vectorization choices. Hand-tuning works for one operator on one GPU, but ML workloads have hundreds of operators across dozens of hardware targets. **Auto-tuning** is how modern ML compilers (TVM, Halide, XLA, Triton) find near-optimal schedules without human intervention. It also represents a fascinating case of **using ML to optimize ML**.

---

## 1. Why Hand-Tuning Fails at Scale

Consider the scheduling decisions for a single 2D convolution:

```
Decision                          Options          Count
──────────────────────────────────────────────────────────
Tile size for output height       4, 8, 16, 32, 64    5
Tile size for output width        4, 8, 16, 32, 64    5
Tile size for output channels     16, 32, 64, 128      4
Tile size for input channels      8, 16, 32, 64        4
Unroll factor for inner loop      1, 2, 4, 8           4
Vectorize width                   4, 8, 16             3
Parallelize axis                  n, co, ho            3
compute_at position               5 options             5
──────────────────────────────────────────────────────────
Total search space:               5×5×4×4×4×3×3×5 = 72,000
```

A ResNet-50 has ~50 unique convolution shapes. Across 5 hardware targets, that's **18 million schedule points** to explore. Even if each trial takes 1 ms, exhaustive search takes 5 hours — for **one** network on **five** GPUs.

```
┌──────────────────────────────────────────────────────┐
│              The Auto-Tuning Loop                    │
│                                                      │
│   Search Space ──► Sample Config ──► Build & Run     │
│       ▲                                  │           │
│       │                                  ▼           │
│   Update Model ◄── Measure Latency ◄── Execute      │
│                                                      │
│   Repeat until: budget exhausted / convergence       │
└──────────────────────────────────────────────────────┘
```

---

## 2. Search Space Definition

A search space is a structured description of all valid schedule configurations. Each **knob** defines a dimension:

### Tile Size Knobs

```python
# TVM-style search space definition
from tvm import autotvm

@autotvm.template("matmul")
def matmul_template(N, M, K):
    A = te.placeholder((N, K), name="A")
    B = te.placeholder((K, M), name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k))
    
    s = te.create_schedule(C.op)
    
    # Define search space knobs
    cfg = autotvm.get_config()
    
    # Tile sizes — each is a knob the tuner can adjust
    cfg.define_split("tile_i", N, num_outputs=3)  # 3-level split
    cfg.define_split("tile_j", M, num_outputs=3)
    cfg.define_split("tile_k", K, num_outputs=2)
    cfg.define_knob("unroll_kf", [1, 2, 4, 8])
    cfg.define_knob("vectorize", [4, 8, 16])
    
    # Apply configuration to schedule
    i, j = s[C].op.axis
    k_axis = s[C].op.reduce_axis[0]
    
    i_outer, i_mid, i_inner = cfg["tile_i"].apply(s, C, i)
    j_outer, j_mid, j_inner = cfg["tile_j"].apply(s, C, j)
    k_outer, k_inner = cfg["tile_k"].apply(s, C, k_axis)
    
    s[C].reorder(i_outer, j_outer, k_outer, i_mid, j_mid, k_inner, i_inner, j_inner)
    s[C].vectorize(j_inner)
    s[C].unroll(k_inner)
    s[C].parallel(i_outer)
    
    return s, [A, B, C]
```

### Knob Types

| Knob Type | Description | Example |
|:----------|:------------|:--------|
| `define_split` | Factorize a loop extent into $n$ parts | `split(1024, 3) → (8, 16, 8)` |
| `define_knob` | Choose from discrete values | `unroll ∈ {1, 2, 4, 8}` |
| `define_reorder` | Permutation of axes | `6! = 720` orderings |
| `define_annotate` | Apply annotation (vectorize, unroll, parallel) | Binary choice per axis |

---

## 3. Search Algorithms

### 3a. Grid Search

Exhaustive enumeration of all combinations. Only practical for tiny spaces.

$$\text{Trials} = \prod_{i=1}^{d} |K_i| \quad \text{(product of all knob sizes)}$$

```python
def grid_search(space, measure_func):
    """Brute-force: try every point."""
    best_config, best_time = None, float('inf')
    for config in itertools.product(*space.values()):
        t = measure_func(config)
        if t < best_time:
            best_config, best_time = config, t
    return best_config, best_time
```

**Verdict:** $O(|S|)$ trials. Unusable for spaces > ~1000 points.

### 3b. Random Search

Surprisingly effective baseline — often within 2× of optimal in 100-500 trials.

```python
def random_search(space, measure_func, n_trials=500):
    """Random sampling — strong baseline."""
    best_config, best_time = None, float('inf')
    for _ in range(n_trials):
        config = {k: random.choice(v) for k, v in space.items()}
        t = measure_func(config)
        if t < best_time:
            best_config, best_time = config, t
    return best_config, best_time
```

**Why it works:** In high-dimensional spaces, the top 1% of configurations are more likely to be found by random sampling than by systematic sweeps along each axis.

### 3c. Genetic Algorithm (GA)

Evolve a population of configurations through selection, crossover, and mutation:

```
Generation 0: [random configs]
    │
    ├─ Evaluate all → rank by latency
    ├─ Select top 20% as parents
    ├─ Crossover: combine knobs from two parents
    ├─ Mutation: randomly perturb 10% of knobs
    │
Generation 1: [evolved configs]
    │
    ... repeat for 50-100 generations ...
```

```python
def genetic_search(space, measure_func, pop_size=64, generations=50):
    """Genetic algorithm for schedule search."""
    keys = list(space.keys())
    
    # Initialize random population
    population = [
        {k: random.choice(space[k]) for k in keys}
        for _ in range(pop_size)
    ]
    
    for gen in range(generations):
        # Evaluate
        scores = [(cfg, measure_func(cfg)) for cfg in population]
        scores.sort(key=lambda x: x[1])
        
        # Select top 25%
        survivors = [cfg for cfg, _ in scores[:pop_size // 4]]
        
        # Breed next generation
        next_gen = list(survivors)  # Elitism: keep best
        while len(next_gen) < pop_size:
            p1, p2 = random.sample(survivors, 2)
            child = {}
            for k in keys:
                # Crossover: pick from either parent
                child[k] = random.choice([p1[k], p2[k]])
                # Mutation: 10% chance of random value
                if random.random() < 0.1:
                    child[k] = random.choice(space[k])
            next_gen.append(child)
        
        population = next_gen
        print(f"Gen {gen}: best = {scores[0][1]:.3f} ms")
    
    return scores[0]
```

### 3d. Simulated Annealing

Accept worse solutions with decreasing probability to escape local minima:

$$P(\text{accept worse}) = \exp\left(-\frac{\Delta t}{T}\right), \quad T = T_0 \cdot \alpha^{\text{step}}$$

### 3e. Cost-Model-Guided Search (XGBoost / Neural)

The key insight of TVM's AutoTVM: **train a model to predict performance without running on hardware**.

```
┌─────────────────────────────────────────────────┐
│         Cost-Model-Guided Search                │
│                                                 │
│  1. Sample N random configs, measure on HW      │
│  2. Train XGBoost: features → latency           │
│  3. Use model to score 10,000 candidates        │
│  4. Measure top-K on HW (ground truth)          │
│  5. Add results to training set                 │
│  6. Re-train model, goto 3                      │
│                                                 │
│  Features extracted from schedule config:        │
│  - Tile sizes, unroll factors                   │
│  - Memory access patterns (strides, reuse)      │
│  - Parallelism degree                           │
│  - Vectorization width                          │
│  - Estimated arithmetic intensity               │
└─────────────────────────────────────────────────┘
```

---

## 4. TVM's Tuning Ecosystem

TVM has three generations of auto-tuning infrastructure:

### Generation 1: AutoTVM (2018)

- **Template-based:** human writes schedule template with knobs
- **Search:** XGBoost cost model + simulated annealing
- **Limitation:** requires expert to write templates per operator

### Generation 2: AutoScheduler / Ansor (2020)

- **Template-free:** automatically generates schedule candidates from computation definition
- **Sketch-based:** generates high-level "sketches" (e.g., multi-level tiling + fusion), then tunes parameters
- **Search:** evolutionary + cost model

```python
from tvm import auto_scheduler

# No template needed — just the computation
@auto_scheduler.register_workload
def matmul(N, M, K):
    A = te.placeholder((N, K), name="A")
    B = te.placeholder((K, M), name="B")
    k = te.reduce_axis((0, K), name="k")
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")
    return [A, B, C]

target = tvm.target.Target("llvm -mcpu=skylake-avx512")
task = auto_scheduler.SearchTask(func=matmul, args=(1024, 1024, 1024), target=target)

# Run auto-tuning with 1000 measurement trials
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=1000,
    measure_callbacks=[auto_scheduler.RecordToFile("matmul_log.json")],
)
task.tune(tune_option)

# Apply best schedule
sch, args = task.apply_best("matmul_log.json")
```

### Generation 3: MetaSchedule (2022+)

- **Unified:** merges AutoTVM and AutoScheduler approaches
- **Design space:** expressed as `Schedule → Schedule` transformation rules
- **Database:** shares tuning results across models and hardware
- **Cost model:** gradient-boosted trees + transfer learning

```
 AutoTVM        Ansor         MetaSchedule
 (manual      (auto-         (unified,
  templates)   sketches)      rule-based)
    ▼             ▼               ▼
 ┌─────────────────────────────────────┐
 │     Measurement Infrastructure      │
 │  (RPC, cross-compilation, trackers) │
 └─────────────────────────────────────┘
```

---

## 5. Hardware-Aware Search

### Feature Extraction

A good cost model needs features that capture hardware interaction:

```python
def extract_features(schedule_config, hw_params):
    """Extract hardware-relevant features from a schedule."""
    features = {}
    
    # Memory hierarchy utilization
    working_set = compute_working_set(schedule_config)
    features["l1_fit"] = 1.0 if working_set <= hw_params["l1_size"] else 0.0
    features["l2_fit"] = 1.0 if working_set <= hw_params["l2_size"] else 0.0
    
    # Arithmetic intensity
    features["arith_intensity"] = (
        schedule_config["flops"] / schedule_config["bytes_transferred"]
    )
    
    # Parallelism efficiency
    total_threads = schedule_config["parallel_extent"]
    features["occupancy"] = min(total_threads / hw_params["max_threads"], 1.0)
    
    # Vectorization efficiency
    vec_width = schedule_config["vectorize_width"]
    features["vec_efficiency"] = vec_width / hw_params["simd_width"]
    
    # Memory coalescing (GPU)
    features["coalesced_ratio"] = schedule_config["coalesced_accesses"] / (
        schedule_config["total_accesses"] + 1e-10
    )
    
    return features
```

### Transfer Learning Across Hardware

Once you've tuned on GPU A, you can **warm-start** tuning on GPU B:

```
GPU A (A100):  tuned 2000 ops, cost model M_A
GPU B (H100):  transfer M_A as initialization
               → converge in 200 trials instead of 2000
               → 10× speedup in tuning time
```

---

## 6. Comparing Search Strategies

```
Performance vs. Tuning Budget (1024×1024 matmul, CPU)

Latency │                                      ┌── cuBLAS baseline
(ms)    │ ██                                    │
   20   │ ██                                    │
        │ ██ ██                                 │
   15   │ ██ ██                                 │
        │ ██ ██ ██                              │
   10   │ ██ ██ ██ ██                           │
        │ ██ ██ ██ ██                     ──────┤
    5   │ ██ ██ ██ ██ ██ ██ ██ ██         ──────┤
        │ ██ ██ ██ ██ ██ ██ ██ ██               │
    0   └─┬──┬──┬──┬──┬──┬──┬──┬────────────────
          10 50 100 200 500 1K 2K 5K
                Tuning Trials

  ██ Random    ██ GA    ██ SA    ██ Cost-Model
```

| Strategy | Trials to 90% optimal | Trials to 99% | Hardware cost |
|:---------|:---------------------|:--------------|:--------------|
| Grid | $|S|$ (all) | $|S|$ | Very high |
| Random | ~$10 \cdot d$ | ~$100 \cdot d$ | Moderate |
| GA | ~$50 \cdot \sqrt{d}$ | ~$200 \cdot \sqrt{d}$ | Moderate |
| SA | ~$100$ | ~$500$ | Moderate |
| Cost-model | ~$30$ | ~$200$ | Low (model evals are free) |

Where $d$ is the number of knob dimensions.

---

## Hands-On Exercises

### Exercise 1: Build a Micro-Autotuner (30 min)

Implement a simple autotuner for tiled matmul tile sizes:

```python
import numpy as np
import time
import random

def tiled_matmul(A, B, Ti, Tj, Tk):
    M, K = A.shape
    _, N = B.shape
    C = np.zeros((M, N), dtype=A.dtype)
    for io in range(0, M, Ti):
        for jo in range(0, N, Tj):
            for ko in range(0, K, Tk):
                C[io:io+Ti, jo:jo+Tj] += (
                    A[io:io+Ti, ko:ko+Tk] @ B[ko:ko+Tk, jo:jo+Tj]
                )
    return C

def measure(A, B, Ti, Tj, Tk, repeats=5):
    """Measure average latency."""
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        tiled_matmul(A, B, Ti, Tj, Tk)
        times.append(time.perf_counter() - start)
    return np.median(times)

# Search space
tile_options = [8, 16, 32, 64, 128, 256]
space = {"Ti": tile_options, "Tj": tile_options, "Tk": tile_options}

N = 512
A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)

# TODO: Implement three search strategies and compare
# 1. Random search (50 trials)
# 2. Genetic algorithm (pop=16, generations=10)
# 3. Hill climbing (start random, mutate one knob at a time)
#
# Report: best config and latency for each strategy
# Bonus: plot convergence curve (trial number vs. best latency so far)
```

### Exercise 2: Cost Model Training (20 min)

Train a simple cost model from tuning logs:

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import json

# Simulate tuning log data
def generate_tuning_data(n_samples=500):
    """Generate synthetic tuning data."""
    data = []
    for _ in range(n_samples):
        Ti = random.choice([8, 16, 32, 64, 128])
        Tj = random.choice([8, 16, 32, 64, 128])
        Tk = random.choice([8, 16, 32, 64, 128])
        
        # Synthetic latency model (simplified)
        working_set = Ti * Tk + Tk * Tj + Ti * Tj
        cache_penalty = max(0, working_set - 8192) * 0.001
        tile_overhead = (1024 / Ti) * (1024 / Tj) * (1024 / Tk) * 0.0001
        compute = 2 * 1024**3 / (Ti * Tj * Tk) * 1e-9
        latency = compute + cache_penalty + tile_overhead + random.gauss(0, 0.5)
        
        data.append({
            "Ti": Ti, "Tj": Tj, "Tk": Tk,
            "working_set": working_set,
            "arith_intensity": Ti * Tj * Tk / working_set,
            "latency_ms": max(0.1, latency)
        })
    return data

data = generate_tuning_data()

# TODO:
# 1. Split into train/test
# 2. Train GBR model on features → latency
# 3. Use model to rank 1000 unseen configs
# 4. Compare model's top-10 vs. actual top-10 (Kendall tau correlation)
```

### Exercise 3: Search Space Explosion Analysis (10 min)

Calculate the search space size for a real operator:

```python
def search_space_size(op_config):
    """Calculate total search space for an operator."""
    total = 1
    for knob_name, options in op_config.items():
        total *= len(options)
        print(f"  {knob_name}: {len(options)} options")
    print(f"  Total: {total:,} configurations")
    
    # Estimate tuning time
    ms_per_trial = 5  # typical for GPU measurement
    for budget in [100, 500, 1000, 5000]:
        coverage = budget / total * 100
        time_h = budget * ms_per_trial / 1000 / 3600
        print(f"  {budget} trials: {coverage:.4f}% coverage, {time_h:.2f} hours")

# ResNet-50 conv2d 3×3, 256→256
search_space_size({
    "tile_oh": [1, 2, 4, 7, 14],
    "tile_ow": [1, 2, 4, 7, 14],
    "tile_co": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    "tile_ci": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    "unroll_kh": [1, 3],
    "unroll_kw": [1, 3],
    "vectorize": [1, 4, 8, 16],
    "parallel_axis": ["n", "co", "oh"],
})
```

---

## Key Takeaways

1. **Hand-tuning doesn't scale** — even one operator has thousands of valid schedules; real networks have hundreds of operators across multiple targets
2. **Search spaces** are defined by knobs (tile sizes, unroll factors, vectorization widths) with exponential combinatorial growth
3. **Random search** is a surprisingly strong baseline — it beats grid search in high dimensions and requires no model
4. **Cost-model-guided search** (TVM's approach) trains a surrogate model to predict performance, reducing the number of expensive hardware measurements by 10×
5. **TVM's evolution:** AutoTVM (manual templates) → Ansor (auto-sketches) → MetaSchedule (rule-based, transferable)
6. **Hardware-aware features** (cache fit, occupancy, coalescing) are critical inputs to cost models — without them, the model cannot distinguish good schedules from bad

---

## Further Reading

- Zheng, L. et al. — *Ansor: Generating High-Performance Tensor Programs for Deep Learning* (OSDI 2020)
- Chen, T. et al. — *Learning to Optimize Tensor Programs* (NeurIPS 2018) — AutoTVM
- Shao, J. et al. — *Tensor Program Optimization with Probabilistic Programs* (NeurIPS 2022) — MetaSchedule
- Bergstra, J. & Bengio, Y. — *Random Search for Hyper-Parameter Optimization* (JMLR 2012) — why random beats grid
- Adams, A. et al. — *Learning to Optimize Halide with Tree Search and Random Programs* (SIGGRAPH 2019)

---

## Tomorrow's Preview

**Day 21** is the **Week 3 mini-project**: you'll build a complete FX optimization pass from scratch — detecting Conv+BatchNorm fusion opportunities in a traced graph, rewriting the graph to apply the fusion, verifying correctness, and benchmarking the speedup. This ties together everything from Days 15-20: IRs, graph rewrites, and the optimization mindset.
