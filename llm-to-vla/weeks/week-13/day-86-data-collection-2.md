# Day 86: Data Collection Part 2 — Scaling, Quality & Open X-Embodiment

> **Phase VI — Robot Learning: RL, Diffusion & Data | Week 13 | 2.5 hours**
> *"The most impactful decision in robot learning isn't the architecture — it's which data you train on and how you mix it." — Open X-Embodiment Team*

## Navigation
- **Previous:** [Day 85: Data Collection Day 1](day-85-data-collection-1.md)
- **Next:** [Day 87: Policy Evaluation](day-87-policy-evaluation.md)
- **Week:** [Week 13 Overview](README.md)
- **Phase:** [Phase VI: Robot Learning](../../study-notes/14-robot-data-eval.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 86.1 Open X-Embodiment: The ImageNet of Robotics

The Open X-Embodiment dataset aggregated data from **22 robot embodiments** across **21 institutions**:

```
Open X-Embodiment:
  ├── 527 skills across 160,266 tasks
  ├── 1M+ real robot episodes
  ├── 22 robot types (arms, hands, mobile, humanoid)
  ├── RLDS format on TensorFlow Datasets
  └── Enabled RT-X: one model, many robots
```

### 86.2 Data Mixing Strategies

When training on heterogeneous data, mixing proportions matter enormously:

| Strategy | Formula | When to Use |
|----------|---------|-------------|
| **Uniform** | Equal weight per dataset | Baseline |
| **Proportional** | Weight ∝ dataset size | Large datasets dominate |
| **Temperature-scaled** | $w_i \propto n_i^{1/T}$ | $T < 1$: upweight small sets |
| **Task-balanced** | Equal weight per task | Task diversity priority |
| **Quality-weighted** | Weight ∝ success rate × diversity | Best but needs metrics |

$$p_i = \frac{n_i^{1/T}}{\sum_j n_j^{1/T}}, \quad T = \text{temperature}$$

At $T = 1$: proportional. At $T \to 0$: uniform. At $T \to \infty$: largest dataset only.

### 86.3 Data Quality Metrics

| Metric | What It Measures | Threshold |
|--------|-----------------|-----------|
| **Success rate** | Task completion | > 80% of demos should succeed |
| **Action smoothness** | Jerk (3rd derivative) | Low jerk = smooth control |
| **Trajectory length variance** | Consistency | CV < 0.5 |
| **Action range utilization** | Coverage of action space | > 50% of range used |
| **Diversity index** | Variety of strategies | Multiple clusters in action space |

### 86.4 Data Augmentation for Robotics

| Augmentation | Applied To | Effect |
|--------------|-----------|--------|
| **Color jitter** | Images | Lighting robustness |
| **Random crop** | Images | Position invariance |
| **Camera pose perturbation** | Images + actions | Camera viewpoint robustness |
| **Action noise** | Actions | Smoothing, exploration |
| **Temporal subsampling** | Full trajectory | Speed invariance |
| **Goal relabeling** | Language | Task generalization |

### 86.5 Filtering Bad Data

```
Raw demos → Quality filter → Training set

Quality filter pipeline:
  1. Remove episodes with NaN/Inf values
  2. Remove episodes shorter than min_length
  3. Remove failed episodes (or keep some for recovery learning)
  4. Remove outlier trajectories (>3σ from mean length)
  5. Compute action smoothness; remove high-jerk episodes
  6. Manual spot-check: review random 5% visually
```

### 86.6 Scaling Laws for Robot Data

Emerging evidence suggests robot learning follows scaling laws similar to LLMs:

$$\text{Performance} \propto \left(\frac{N_\text{data}}{N_0}\right)^{\alpha_d} \cdot \left(\frac{N_\text{params}}{N_1}\right)^{\alpha_p}$$

But with crucial differences:
- **Exponents are steeper** — robot data is more information-dense
- **Diminishing returns come faster** — diversity matters more than volume
- **Cross-embodiment data helps** — but with negative transfer risk

---

## Implementation (60 min)

### Data Quality Pipeline

```python
import numpy as np
from pathlib import Path

class DataQualityPipeline:
    """Comprehensive data quality assessment and filtering."""

    def __init__(self, min_length=10, max_length=1000,
                 success_only=False, jerk_threshold=None):
        self.min_length = min_length
        self.max_length = max_length
        self.success_only = success_only
        self.jerk_threshold = jerk_threshold

    def compute_smoothness(self, actions):
        """Compute jerk (3rd derivative) as smoothness metric."""
        if len(actions) < 4:
            return 0.0
        velocity = np.diff(actions, axis=0)
        acceleration = np.diff(velocity, axis=0)
        jerk = np.diff(acceleration, axis=0)
        return np.mean(np.abs(jerk))

    def compute_diversity_index(self, episodes):
        """Measure diversity of strategies in the dataset."""
        # Use start/end state pairs as strategy fingerprints
        fingerprints = []
        for ep in episodes:
            obs = np.array(ep["observations"])
            fingerprints.append(np.concatenate([obs[0], obs[-1]]))

        fingerprints = np.array(fingerprints)
        # Pairwise distances
        from scipy.spatial.distance import pdist
        distances = pdist(fingerprints)
        return np.mean(distances), np.std(distances)

    def filter_dataset(self, episodes):
        """Filter episodes by quality criteria."""
        filtered = []
        removed_reasons = {}

        for i, ep in enumerate(episodes):
            reason = self._check_episode(ep)
            if reason is None:
                filtered.append(ep)
            else:
                removed_reasons[i] = reason

        print(f"Filtered: {len(filtered)}/{len(episodes)} episodes kept")
        if removed_reasons:
            reasons = {}
            for r in removed_reasons.values():
                reasons[r] = reasons.get(r, 0) + 1
            for r, c in sorted(reasons.items(), key=lambda x: -x[1]):
                print(f"  Removed {c} episodes: {r}")

        return filtered

    def _check_episode(self, episode):
        length = len(episode["actions"])
        if length < self.min_length:
            return f"too_short ({length})"
        if length > self.max_length:
            return f"too_long ({length})"

        if self.success_only and not episode.get("metadata", {}).get("success", True):
            return "failed_episode"

        actions = np.array(episode["actions"])
        if np.any(np.isnan(actions)) or np.any(np.isinf(actions)):
            return "nan_or_inf"

        if self.jerk_threshold is not None:
            jerk = self.compute_smoothness(actions)
            if jerk > self.jerk_threshold:
                return f"high_jerk ({jerk:.4f})"

        return None

    def dataset_report(self, episodes):
        """Generate comprehensive dataset quality report."""
        print("=" * 60)
        print("DATASET QUALITY REPORT")
        print("=" * 60)

        lengths = [len(ep["actions"]) for ep in episodes]
        successes = sum(1 for ep in episodes
                       if ep.get("metadata", {}).get("success", True))

        print(f"\nBasic Stats:")
        print(f"  Episodes: {len(episodes)}")
        print(f"  Success rate: {successes/len(episodes):.1%}")
        print(f"  Length: {np.mean(lengths):.0f} ± {np.std(lengths):.0f} "
              f"[{min(lengths)}, {max(lengths)}]")
        print(f"  Total transitions: {sum(lengths):,}")

        # Action statistics
        all_actions = np.concatenate([np.array(ep["actions"]) for ep in episodes])
        print(f"\nAction Statistics ({all_actions.shape[1]} dimensions):")
        for d in range(all_actions.shape[1]):
            col = all_actions[:, d]
            util = (col.max() - col.min()) / (2 * max(abs(col.max()), abs(col.min())) + 1e-8)
            print(f"  dim {d}: μ={col.mean():.4f} σ={col.std():.4f} "
                  f"range=[{col.min():.4f}, {col.max():.4f}] util={util:.1%}")

        # Smoothness
        jerks = [self.compute_smoothness(np.array(ep["actions"])) for ep in episodes]
        print(f"\nSmoothness (jerk):")
        print(f"  Mean: {np.mean(jerks):.6f}")
        print(f"  Max:  {np.max(jerks):.6f}")
        print(f"  Episodes with high jerk (>2σ): "
              f"{sum(1 for j in jerks if j > np.mean(jerks) + 2*np.std(jerks))}")

        print("=" * 60)

# --- Data Mixing ---
class DataMixer:
    """Mix datasets with temperature-scaled proportions."""

    @staticmethod
    def compute_weights(dataset_sizes, temperature=1.0):
        sizes = np.array(dataset_sizes, dtype=float)
        weights = sizes ** (1.0 / temperature)
        return weights / weights.sum()

    @staticmethod
    def sample_batch(datasets, weights, batch_size):
        """Sample a batch from mixed datasets."""
        batch = []
        counts = np.random.multinomial(batch_size, weights)
        for ds, n in zip(datasets, counts):
            if n > 0:
                indices = np.random.choice(len(ds), min(n, len(ds)), replace=False)
                batch.extend([ds[i] for i in indices])
        np.random.shuffle(batch)
        return batch

# Example mixing
sizes = [100000, 5000, 1000, 500]  # 4 datasets of different sizes
names = ["Bridge", "RT-1", "ALOHA", "Custom"]
for T in [0.3, 0.5, 1.0, 2.0]:
    w = DataMixer.compute_weights(sizes, T)
    print(f"T={T}: " + " | ".join(f"{n}: {p:.1%}" for n, p in zip(names, w)))
```

---

## Exercise (45 min)

1. **Mixing temperature:** Plot mixing weights for 4 datasets (sizes: 100K, 10K, 1K, 100) across temperatures $T \in [0.1, 5.0]$. What temperature gives reasonable balance?

2. **Quality pipeline:** Apply `DataQualityPipeline` to a collected dataset. Generate the report. Identify and remove the worst 10% of episodes.

3. **Augmentation ablation:** Train BC with and without image augmentation (color jitter + random crop). Compare generalization to new lighting.

4. **Cross-embodiment analysis:** Download two datasets from Open X-Embodiment. Compare action distributions. Can you identify where negative transfer might occur?

---

## Key Takeaways

1. **Open X-Embodiment** is the largest robot learning dataset — 1M+ episodes, 22 robots
2. **Temperature-scaled mixing** balances large and small datasets ($T \approx 0.5$ typical)
3. **Data quality > quantity** — filter aggressively, validate thoroughly
4. **Image augmentation** is the single most impactful augmentation for robot learning
5. **Diversity matters more** than volume for generalization

---

## Connection to the Thread

> *We now have data. Tomorrow: how to know if the policy we trained on that data is actually good. Policy evaluation in robotics is fundamentally different from NLP/vision — there's no BLEU score for robot manipulation.*

---

## Further Reading

- Open X-Embodiment (2023), "Open X-Embodiment: Robotic Learning Datasets and RT-X Models"
- Padalkar et al. (2023), "An Open X-Embodiment Robotic Learning Datasets and RT-X Models" (technical report)
- HuggingFace LeRobot Dataset Format documentation
