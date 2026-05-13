# Day 109: Deployment — Fleet Management & Continuous Learning

> **Phase VII — VLAs: Architecture to Deployment | Week 16 | 2.5 hours**
> *"One robot is a research project. A fleet is a product. Fleet management is where VLAs meet operations." — Fleet-Scale VLAs*

## Navigation
- **Previous:** [Day 108: Deployment — Safety](day-108-deployment-2.md)
- **Next:** [Day 110: Final Capstone — Day 1](day-110-final-capstone-1.md)
- **Week:** [Week 16 Overview](README.md)
- **Phase:** [Phase VII: VLAs](../../study-notes/16-deployment-hybrid.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (60 min)

### 109.1 Fleet Architecture

```
┌─────────────────────────────────────────────────────────┐
│              FLEET VLA ARCHITECTURE                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Cloud Layer:                                            │
│  ┌──────────────────────────────────────────────┐       │
│  │  Model Registry    Training Pipeline          │       │
│  │  ┌──────┐         ┌──────────────────┐       │       │
│  │  │v1.0  │         │ Aggregate data   │       │       │
│  │  │v1.1  │         │ from all robots  │       │       │
│  │  │v1.2* │←────────│ → retrain weekly │       │       │
│  │  └──────┘         └──────────────────┘       │       │
│  │                                                │       │
│  │  Fleet Dashboard   A/B Test Manager           │       │
│  │  ┌──────┐         ┌──────────────────┐       │       │
│  │  │All   │         │ 50% robots: v1.1 │       │       │
│  │  │robots│         │ 50% robots: v1.2 │       │       │
│  │  │status│         │ → compare metrics│       │       │
│  │  └──────┘         └──────────────────┘       │       │
│  └──────────────────────────────────────────────┘       │
│                          ↕ OTA update                    │
│  Edge Layer (per robot):                                 │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐       │
│  │Robot 1 │  │Robot 2 │  │Robot 3 │  │  ...   │       │
│  │VLA v1.2│  │VLA v1.1│  │VLA v1.2│  │        │       │
│  │Local   │  │Local   │  │Local   │  │        │       │
│  │adapt.  │  │adapt.  │  │adapt.  │  │        │       │
│  └────────┘  └────────┘  └────────┘  └────────┘       │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 109.2 Continuous Learning Pipeline

```
Data Flywheel:
  1. Robots execute tasks (VLA inference)
  2. Log all (observation, action, outcome) tuples
  3. Human operators correct failures → store corrections
  4. Aggregate data centrally
  5. Retrain VLA on accumulated dataset
  6. A/B test new model vs current
  7. If improved: roll out to fleet
  8. Repeat

Key constraint: catastrophic forgetting
  - New data must not degrade old capabilities
  - Solution: replay buffer of representative old data
  - Mix: 70% old data + 30% new corrections
```

### 109.3 A/B Testing for VLAs

```python
# A/B testing framework for VLA fleet deployment

class ABTest:
    def __init__(self, model_a, model_b, split=0.5):
        self.model_a = model_a  # Control (current production)
        self.model_b = model_b  # Treatment (candidate)
        self.split = split
        self.results_a = []
        self.results_b = []

    def assign_robot(self, robot_id):
        """Deterministic assignment based on robot ID."""
        # Hash-based assignment for consistency
        import hashlib
        h = int(hashlib.md5(str(robot_id).encode()).hexdigest(), 16)
        return "B" if (h % 100) < (self.split * 100) else "A"

    def record(self, robot_id, success, latency):
        group = self.assign_robot(robot_id)
        if group == "A":
            self.results_a.append({"success": success, "latency": latency})
        else:
            self.results_b.append({"success": success, "latency": latency})

    def evaluate(self, min_samples=100):
        """Statistical test for difference."""
        if len(self.results_a) < min_samples or len(self.results_b) < min_samples:
            return {"status": "insufficient_data"}

        rate_a = np.mean([r["success"] for r in self.results_a])
        rate_b = np.mean([r["success"] for r in self.results_b])
        n_a = len(self.results_a)
        n_b = len(self.results_b)

        # Wilson confidence intervals
        def wilson_ci(successes, n, z=1.96):
            p = successes / n
            denom = 1 + z**2/n
            center = (p + z**2/(2*n)) / denom
            margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
            return center - margin, center + margin

        ci_a = wilson_ci(rate_a * n_a, n_a)
        ci_b = wilson_ci(rate_b * n_b, n_b)

        # Non-overlapping CIs → significant difference
        significant = ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0]

        return {
            "rate_a": rate_a, "rate_b": rate_b,
            "ci_a": ci_a, "ci_b": ci_b,
            "significant": significant,
            "winner": "B" if rate_b > rate_a else "A",
            "n_a": n_a, "n_b": n_b,
        }
```

### 109.4 Model Versioning

```
Version strategy:
  v1.0.0   — Major (architecture change)
  v1.1.0   — Minor (retrained with new data)
  v1.1.1   — Patch (fine-tuned for specific task)

Rollout strategy:
  Stage 1: Deploy to 5% of fleet (canary)
  Stage 2: Monitor for 24h, compare metrics
  Stage 3: If ≥ parity: expand to 25%
  Stage 4: If ≥ improvement: expand to 100%
  Stage 5: If degradation at any stage: rollback

Rollback protocol:
  - Every robot stores previous model version
  - Rollback is a config change (no re-download)
  - Time to rollback: < 1 minute
```

### 109.5 Federated Learning for Fleet VLAs

```
Standard: Centralize all data → train one model
  + Simple, consistent
  - Privacy concerns, bandwidth, data silos

Federated: Train locally → share gradients → aggregate
  + Data stays on-device
  + Lower bandwidth
  - Heterogeneous data
  - Slower convergence

Practical hybrid:
  1. Each robot fine-tunes locally (LoRA)
  2. Periodically upload LoRA weights (small: ~10MB)
  3. Server merges LoRA adapters
  4. Distribute merged adapter to fleet
```

---

## Implementation (60 min)

```python
import torch
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

class FleetDataAggregator:
    """Collect and aggregate data from robot fleet."""

    def __init__(self, max_buffer_per_robot=10000):
        self.buffers = defaultdict(list)
        self.max_buffer = max_buffer_per_robot
        self.correction_buffer = []

    def add_episode(self, robot_id: str, observations, actions, success: bool):
        """Add episode from a robot."""
        episode = {
            "observations": observations,
            "actions": actions,
            "success": success,
            "robot_id": robot_id,
        }
        self.buffers[robot_id].append(episode)
        if len(self.buffers[robot_id]) > self.max_buffer:
            self.buffers[robot_id].pop(0)

    def add_correction(self, robot_id: str, observation, original_action,
                       corrected_action):
        """Human correction — high value data."""
        self.correction_buffer.append({
            "observation": observation,
            "original": original_action,
            "corrected": corrected_action,
            "robot_id": robot_id,
        })

    def build_training_set(self, old_data_ratio=0.7):
        """Build retraining dataset with replay."""
        all_episodes = []
        for robot_id, episodes in self.buffers.items():
            all_episodes.extend(episodes)

        # Prioritize corrections (10× weight)
        corrections_weight = 10
        n_corrections = len(self.correction_buffer) * corrections_weight

        # Mix old data (replay) with new
        n_old = int(len(all_episodes) * old_data_ratio)
        n_new = len(all_episodes) - n_old

        return {
            "total_episodes": len(all_episodes),
            "corrections": len(self.correction_buffer),
            "effective_corrections": n_corrections,
            "replay_ratio": old_data_ratio,
            "robots_contributing": len(self.buffers),
        }

class CanaryDeployment:
    """Canary rollout manager."""

    def __init__(self, fleet_size: int):
        self.fleet_size = fleet_size
        self.stages = [
            {"name": "canary", "pct": 0.05, "duration_hours": 24},
            {"name": "early", "pct": 0.25, "duration_hours": 48},
            {"name": "wide", "pct": 0.50, "duration_hours": 24},
            {"name": "full", "pct": 1.00, "duration_hours": 0},
        ]
        self.current_stage = 0
        self.metrics = defaultdict(list)

    def current_rollout(self):
        stage = self.stages[self.current_stage]
        n_robots = int(self.fleet_size * stage["pct"])
        return {
            "stage": stage["name"],
            "robots_on_new_model": n_robots,
            "robots_on_old_model": self.fleet_size - n_robots,
        }

    def record_metrics(self, model_version: str, success_rate: float):
        self.metrics[model_version].append(success_rate)

    def should_advance(self, new_version: str, old_version: str,
                       min_samples: int = 50):
        """Check if we should advance to next stage."""
        new_rates = self.metrics.get(new_version, [])
        old_rates = self.metrics.get(old_version, [])

        if len(new_rates) < min_samples or len(old_rates) < min_samples:
            return False, "Insufficient data"

        new_mean = np.mean(new_rates)
        old_mean = np.mean(old_rates)

        if new_mean >= old_mean - 0.02:  # Allow 2% margin
            return True, f"New ({new_mean:.1%}) >= Old ({old_mean:.1%})"
        return False, f"New ({new_mean:.1%}) < Old ({old_mean:.1%})"

    def advance(self):
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            return self.current_rollout()
        return {"stage": "complete", "robots_on_new_model": self.fleet_size}

    def rollback(self):
        self.current_stage = 0
        return {"stage": "rolled_back", "robots_on_new_model": 0}

class LoRAFederatedMerger:
    """Merge LoRA adapters from fleet robots."""

    def __init__(self, base_dim=256, lora_rank=8):
        self.base_dim = base_dim
        self.lora_rank = lora_rank

    def create_lora(self) -> dict:
        """Create a LoRA adapter (A and B matrices)."""
        return {
            "A": torch.randn(self.base_dim, self.lora_rank) * 0.01,
            "B": torch.randn(self.lora_rank, self.base_dim) * 0.01,
        }

    def merge_adapters(self, adapters: List[dict],
                       weights: List[float] = None) -> dict:
        """Weighted average of LoRA adapters."""
        if weights is None:
            weights = [1.0 / len(adapters)] * len(adapters)

        merged_A = sum(w * a["A"] for w, a in zip(weights, adapters))
        merged_B = sum(w * a["B"] for w, a in zip(weights, adapters))

        return {"A": merged_A, "B": merged_B}

    def compute_adapter_delta(self, lora: dict) -> torch.Tensor:
        """Full weight delta from LoRA: ΔW = BA."""
        return lora["B"] @ lora["A"]  # (base_dim, base_dim)? No: (rank, dim) @ (dim, rank)
        # Actually: A is (base_dim, rank), B is (rank, base_dim)
        # ΔW = A @ B gives (base_dim, base_dim)

# Demo
print("=== Fleet Data Aggregation ===")
aggregator = FleetDataAggregator()
for i in range(5):
    for j in range(100):
        aggregator.add_episode(
            f"robot_{i}", None, None, np.random.random() > 0.1
        )
for i in range(20):
    aggregator.add_correction(f"robot_{i%5}", None, None, None)

stats = aggregator.build_training_set()
print(f"Total episodes: {stats['total_episodes']}")
print(f"Corrections: {stats['corrections']}")
print(f"Robots contributing: {stats['robots_contributing']}")

print("\n=== Canary Deployment ===")
canary = CanaryDeployment(fleet_size=100)
print(f"Stage: {canary.current_rollout()}")

# Simulate good performance
for _ in range(60):
    canary.record_metrics("v1.2", np.random.normal(0.90, 0.05))
    canary.record_metrics("v1.1", np.random.normal(0.87, 0.05))

should, reason = canary.should_advance("v1.2", "v1.1")
print(f"Advance? {should}: {reason}")
if should:
    print(f"Advanced to: {canary.advance()}")

print("\n=== Federated LoRA Merge ===")
merger = LoRAFederatedMerger()
adapters = [merger.create_lora() for _ in range(5)]
merged = merger.merge_adapters(adapters)
delta = merger.compute_adapter_delta(merged)
print(f"Merged LoRA delta shape: A={merged['A'].shape}, B={merged['B'].shape}")
print(f"Adapter size: {merged['A'].numel() + merged['B'].numel()} params")
print(f"vs full layer: {256*256} params")
print(f"Compression: {256*256 / (merged['A'].numel() + merged['B'].numel()):.0f}×")
```

---

## Exercise (45 min)

1. **A/B test simulation:** Simulate a fleet of 50 robots. Model A has 85% success, Model B has 88%. How many episodes per robot are needed to detect the difference with 95% confidence?

2. **Canary rollout:** Simulate a bad model update (success drops from 87% to 75%). How quickly does the canary system detect and rollback?

3. **Data flywheel:** Start with a VLA at 80% success. Each week, add corrections from failures. Simulate 10 weeks. Plot the success rate trajectory. How quickly does the flywheel compound?

4. **Federated vs centralized:** Compare federated LoRA merging vs centralized retraining with the same data. Measure final model quality and communication cost (bytes transferred).

---

## Key Takeaways

1. **Fleet architecture** separates cloud (training, A/B testing) from edge (inference)
2. **Continuous learning flywheel** compounds corrections into better models over time
3. **Canary deployments** protect the fleet from bad model updates
4. **A/B testing with Wilson CIs** provides statistical rigor for model comparisons
5. **Federated LoRA merging** enables fleet learning without centralizing data

---

## Connection to the Thread

> *You've covered the complete deployment stack: compute optimization, safety, monitoring, and fleet management. Now: the final capstone. Days 110-112, three sessions to design, build, and evaluate a complete VLA system from scratch.*

---

## Further Reading

- Levine et al. (2020), "Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems" (data flywheel)
- McMahan et al. (2017), "Communication-Efficient Learning of Deep Networks from Decentralized Data" (federated learning)
- Open X-Embodiment (2024) — fleet-scale data collection
