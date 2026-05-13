# Day 100: VLA Training Recipes

> **Phase VII — VLAs: Architecture to Deployment | Week 15 | 2.5 hours**
> *"Architecture is 20% of the result. The other 80% is data, training schedule, and hyperparameters." — Distilled VLA training wisdom*

## Navigation
- **Previous:** [Day 99: GR-2 Deep Dive](day-99-gr2.md)
- **Next:** [Day 101: Sim-to-Real Transfer — Day 1](day-101-sim-to-real-1.md)
- **Week:** [Week 15 Overview](README.md)
- **Phase:** [Phase VII: VLAs](../../study-notes/15-vla-architectures.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (60 min)

### 100.1 The VLA Training Stack

```
┌──────────────────────────────────────────────────────┐
│                VLA TRAINING RECIPE                    │
├──────────────────────────────────────────────────────┤
│                                                       │
│  1. DATA RECIPE                                       │
│     ├── Source composition (web + robot ratio)         │
│     ├── Multi-task mixing (task weighting)             │
│     ├── Data quality filtering                        │
│     └── Augmentation strategy                         │
│                                                       │
│  2. TRAINING SCHEDULE                                 │
│     ├── Stage 1: VLM pre-training (frozen)            │
│     ├── Stage 2: Co-fine-tuning (VLM + action)        │
│     ├── Stage 3: Task-specific fine-tuning            │
│     └── Learning rate schedule                        │
│                                                       │
│  3. HYPERPARAMETERS                                   │
│     ├── Action representation (bins, chunks)          │
│     ├── Observation history (context length)          │
│     ├── Batch size and gradient accumulation          │
│     └── Regularization (weight decay, dropout)        │
│                                                       │
│  4. EVALUATION PROTOCOL                               │
│     ├── Offline metrics (loss, action accuracy)       │
│     ├── Online metrics (success rate, Wilson CI)      │
│     └── Generalization splits (seen/unseen)           │
│                                                       │
└──────────────────────────────────────────────────────┘
```

### 100.2 Data Recipes Across VLAs

| Model | Robot Data | Web Data | Ratio | Key Insight |
|-------|-----------|----------|-------|-------------|
| RT-1 | 130K eps | 0 | 100% robot | Enough real data → no web needed |
| RT-2 | 130K eps | ~1B img-txt | 50/50 | Web data prevents forgetting |
| OpenVLA | 970K eps | VLM pretrain | 50/50 | Multi-embodiment helps |
| Octo | 800K eps | 0 | 100% robot | Small model → robot-only works |
| π₀ | ~1M eps | VLM pretrain | Staged | Freeze VLM, train expert only |

### 100.3 Multi-Task Data Mixing

**The problem:** some tasks have 50K demos, others have 500. Uniform sampling under-trains rare tasks.

**Solutions:**

```python
# Strategy 1: Temperature sampling
# p(task_i) ∝ n_i^(1/T), T > 1 upweights rare tasks
import numpy as np

task_counts = {"pick": 50000, "place": 30000, "stack": 2000, "pour": 500}
T = 2.0  # Temperature
counts = np.array(list(task_counts.values()))
probs = counts ** (1/T)
probs /= probs.sum()
# Result: pick=0.40, place=0.35, stack=0.15, pour=0.10
# vs uniform data ratio: pick=0.61, place=0.36, stack=0.02, pour=0.006

# Strategy 2: Per-task loss weighting
# Weight loss inversely proportional to task frequency
weights = 1.0 / counts
weights /= weights.sum()

# Strategy 3: Curriculum (easy → hard)
# Start with simple tasks, progressively add complex ones
curriculum = {
    "epoch_0_100": ["pick", "place"],
    "epoch_100_200": ["pick", "place", "stack"],
    "epoch_200_300": ["pick", "place", "stack", "pour"],
}
```

### 100.4 Training Schedule

**Three-stage recipe (most VLAs):**

```
Stage 1: VLM alignment (1-5% of training)
  - Freeze vision encoder
  - Train projection layer only
  - LR: 1e-3 (high, projection only)
  - Purpose: align visual tokens with LM space

Stage 2: Co-fine-tuning (80% of training)
  - Unfreeze VLM backbone
  - Add robot data alongside VLM data
  - LR: 2e-5 (low, full model)
  - Purpose: learn action prediction while retaining VLM

Stage 3: Task-specific fine-tuning (15% of training)
  - Freeze VLM backbone (optional)
  - Train on target robot data only
  - LR: 1e-5 (very low) or LoRA
  - Purpose: specialize to deployment scenario
```

### 100.5 Critical Hyperparameters

```
The 10 hyperparameters that matter most (ranked):

1. Data quality (filter bad demos)        → 20% impact
2. Action representation (delta EE, bins) → 15% impact
3. Chunk size (4, 8, 16, 50)              → 10% impact
4. Learning rate schedule                  → 10% impact
5. Multi-task mixing temperature           → 8% impact
6. Web/robot data ratio                    → 8% impact
7. Vision encoder (freeze vs fine-tune)    → 7% impact
8. Observation history length              → 7% impact
9. Batch size                              → 5% impact
10. Weight decay                            → 5% impact
```

### 100.6 Common Training Failures

| Failure | Symptom | Fix |
|---------|---------|-----|
| Mode collapse | All predictions identical | Lower LR, add noise |
| Catastrophic forgetting | VLM reasoning degrades | Keep web data, use LoRA |
| Action jitter | Robot vibrates in place | Smooth actions, larger chunks |
| Overfitting | Low train loss, high eval loss | More data, augmentation |
| Gradient explosion | NaN losses | Gradient clipping, lower LR |
| Task imbalance | Good at common, bad at rare | Temperature sampling |

---

## Implementation (60 min)

### Training Recipe Framework

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

class VLATrainingRecipe:
    """Configurable VLA training recipe."""

    def __init__(self, model, config):
        self.model = model
        self.config = config

    def build_sampler(self, dataset):
        """Temperature-based task sampling."""
        task_counts = {}
        for item in dataset:
            task = item.get("task", "default")
            task_counts[task] = task_counts.get(task, 0) + 1

        T = self.config.get("sampling_temperature", 2.0)
        weights = []
        for item in dataset:
            task = item.get("task", "default")
            count = task_counts[task]
            weights.append(count ** (1.0 / T - 1.0))

        return WeightedRandomSampler(weights, len(weights))

    def build_optimizer(self, stage):
        """Stage-specific optimizer configuration."""
        if stage == 1:
            # Projection only
            params = [p for n, p in self.model.named_parameters()
                      if "projection" in n or "action_head" in n]
            return torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
        elif stage == 2:
            # Full model, low LR
            return torch.optim.AdamW(
                self.model.parameters(), lr=2e-5, weight_decay=0.01
            )
        else:
            # Task-specific, very low LR
            return torch.optim.AdamW(
                self.model.parameters(), lr=1e-5, weight_decay=0.01
            )

    def build_scheduler(self, optimizer, total_steps):
        """Cosine schedule with warmup."""
        warmup_steps = int(0.03 * total_steps)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def train_epoch(self, loader, optimizer, scheduler, stage,
                    web_loader=None, web_ratio=0.5):
        """One training epoch with optional web data mixing."""
        self.model.train()
        total_loss = 0
        n_batches = 0

        web_iter = iter(web_loader) if web_loader and stage == 2 else None

        for batch in loader:
            # Robot data
            robot_loss = self.model.compute_loss(batch)

            # Web data (stage 2 only)
            if web_iter is not None:
                try:
                    web_batch = next(web_iter)
                except StopIteration:
                    web_iter = iter(web_loader)
                    web_batch = next(web_iter)
                web_loss = self.model.compute_vlm_loss(web_batch)
                loss = (1 - web_ratio) * robot_loss + web_ratio * web_loss
            else:
                loss = robot_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def train_full(self, robot_dataset, web_dataset=None, n_epochs_per_stage=None):
        """Full 3-stage training."""
        if n_epochs_per_stage is None:
            n_epochs_per_stage = {1: 5, 2: 80, 3: 15}

        results = {}

        for stage in [1, 2, 3]:
            print(f"\n{'='*40}")
            print(f"Stage {stage}: {['', 'Projection Alignment',
                  'Co-Fine-Tuning', 'Task Specialization'][stage]}")
            print(f"{'='*40}")

            # Freeze/unfreeze based on stage
            if stage == 1:
                for name, param in self.model.named_parameters():
                    param.requires_grad = "projection" in name or "action" in name
            elif stage == 2:
                for param in self.model.parameters():
                    param.requires_grad = True
            # Stage 3: keep all trainable (or apply LoRA)

            optimizer = self.build_optimizer(stage)
            sampler = self.build_sampler(robot_dataset)
            loader = DataLoader(robot_dataset, batch_size=32, sampler=sampler)
            web_loader = DataLoader(web_dataset, batch_size=32, shuffle=True) \
                         if web_dataset and stage == 2 else None

            total_steps = n_epochs_per_stage[stage] * len(loader)
            scheduler = self.build_scheduler(optimizer, total_steps)

            for epoch in range(n_epochs_per_stage[stage]):
                loss = self.train_epoch(loader, optimizer, scheduler, stage, web_loader)
                if epoch % 10 == 0:
                    print(f"  Epoch {epoch}: loss = {loss:.4f}")

            results[f"stage_{stage}_final_loss"] = loss

        return results

# Usage example
print("VLA Training Recipe Framework")
print("Stages: Align → Co-fine-tune → Specialize")
print("Key: temperature sampling, cosine LR, gradient clipping")
```

---

## Exercise (45 min)

1. **Data mixing ablation:** Train a VLA with robot-only vs 50/50 robot+web vs 80/20 robot+web. Measure both action accuracy and language understanding retention.

2. **Sampling temperature sweep:** Try $T \in \{0.5, 1.0, 2.0, 5.0, \infty\}$. Plot per-task success rate. Find the temperature that best balances common and rare tasks.

3. **Stage ablation:** Compare 3-stage training vs 1-stage (everything at once). Measure final performance and training stability.

4. **Hyperparameter sensitivity:** Pick 3 hyperparameters from the top-10 list. Vary each by 2× up and down. Which has the largest impact on success rate?

---

## Key Takeaways

1. **Data quality > architecture** — filtering bad demos matters more than model size
2. **Three-stage training** (align → co-fine-tune → specialize) is the standard recipe
3. **Temperature sampling** balances multi-task training across uneven datasets
4. **Keep web data during fine-tuning** to prevent catastrophic forgetting
5. **The top 10 hyperparameters** account for >95% of training outcome variance

---

## Connection to the Thread

> *Training recipes solve "how to train." But real deployment requires bridging the sim-to-real gap. Tomorrow and the next day: two sessions on sim-to-real transfer — domain randomization, system identification, progressive transfer, and the techniques that make simulation-trained VLAs work on physical robots.*

---

## Further Reading

- Review training details in: RT-2, OpenVLA, Octo, π₀ papers
- Ghosh et al. (2024), "Scaling Data for Robot Learning: OpenX-Embodiment"
- Liu et al. (2024), "LIBERO: Benchmarking VLA Training Recipes"
