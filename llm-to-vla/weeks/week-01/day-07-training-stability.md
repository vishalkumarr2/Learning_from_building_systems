# Day 7: Training Stability Cookbook

> **Phase I — DL Foundations & Information Theory | Week 1 | 2.5 hours**
> *"The difference between a model that trains and one that doesn't is rarely the architecture — it's the cookbook of stability tricks you apply."*

## Navigation
- **Previous:** [Day 6: Embeddings & Representation Learning](day-06-embeddings-representations.md)
- **Next:** [Day 8: Phase I Mini-Project](../week-02/day-08-phase1-miniproject.md)
- **Week:** [Week 1 Overview](../README.md)
- **Phase:** [Phase I: DL Foundations](../../study-notes/01-dl-foundations.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### Why Training Stability Matters

Every VLA paper you'll read later uses *all* of these techniques. RT-2, Octo, π₀ — none of them train without normalization, careful initialization, learning rate schedules, and gradient clipping. These aren't optional add-ons; they're **non-negotiable infrastructure**.

The core problem: deep networks are chaotic dynamical systems. Small perturbations in early layers amplify exponentially through depth. Stability techniques tame this chaos.

### 1. Batch Normalization

**Idea:** Normalize activations across the batch dimension, then learn scale ($\gamma$) and shift ($\beta$).

For a mini-batch $\mathcal{B} = \{x_1, \ldots, x_m\}$:

$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}}$$

$$y_i = \gamma \hat{x}_i + \beta$$

where $\mu_\mathcal{B} = \frac{1}{m}\sum_{i=1}^{m} x_i$ and $\sigma^2_\mathcal{B} = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_\mathcal{B})^2$.

**Why it works (three perspectives):**
1. **Internal covariate shift** (original paper): each layer sees a more stable input distribution
2. **Smoothing the loss landscape** (Santurkar et al., 2018): BN makes the loss surface smoother, enabling larger learning rates
3. **Regularization effect**: batch statistics inject noise, acting as implicit regularization

**The catch:** BN introduces *batch dependency*. The statistics $\mu_\mathcal{B}$, $\sigma^2_\mathcal{B}$ change with different batches, which causes problems for:
- Small batch sizes (noisy statistics)
- Sequence models (variable lengths within a batch)
- Inference (must use running averages)

```
BatchNorm operation (channels-first, batch of images):

Input shape: [B, C, H, W]
                │
   Compute mean/var across B, H, W dimensions (per channel)
                │
        Normalize → scale (γ) → shift (β)
                │
Output shape: [B, C, H, W]

Key: statistics are computed PER CHANNEL, ACROSS the batch
```

### 2. Layer Normalization

**Idea:** Normalize across the *feature* dimension for each sample independently. No batch dependency.

For a single sample with features $\{a_1, \ldots, a_H\}$:

$$\hat{a}_i = \frac{a_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad \mu = \frac{1}{H}\sum_{i=1}^{H} a_i, \quad \sigma^2 = \frac{1}{H}\sum_{i=1}^{H}(a_i - \mu)^2$$

```
LayerNorm vs BatchNorm (for a tensor [B, C, H, W]):

BatchNorm:  normalize across  B, H, W  → statistics per C
LayerNorm:  normalize across  C, H, W  → statistics per B (per sample)

                Batch dimension →
              ┌───┬───┬───┬───┐
Feature       │   │   │   │   │  BatchNorm: normalize ↓ (across batch)
dimension  ↓  │   │   │   │   │  LayerNorm: normalize → (across features)
              │   │   │   │   │
              └───┴───┴───┴───┘
```

**Why transformers use LayerNorm:**
- No batch dependency → works with any batch size
- Consistent behavior during training and inference
- Works naturally with variable-length sequences (each token normalized independently)

### 3. Weight Initialization

Bad initialization = gradients that vanish or explode on the *very first forward pass*.

**Goal:** maintain variance of activations across layers. If $\text{Var}(\text{output}) \approx \text{Var}(\text{input})$ at every layer, signals propagate cleanly.

**Xavier/Glorot initialization** (for sigmoid/tanh):
$$W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right) \quad \text{or} \quad W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}\right)$$

**He/Kaiming initialization** (for ReLU):
$$W \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)$$

ReLU zeroes out ~half the activations, so the variance halves each layer. He init compensates by doubling the initial variance (the factor of 2 in the numerator).

**Rule of thumb:**
| Activation | Init |
|-----------|------|
| sigmoid / tanh | Xavier |
| ReLU / variants | He (Kaiming) |
| GELU (transformers) | Xavier or He — both work with LayerNorm present |

### 4. Learning Rate Schedules

The learning rate is the single most important hyperparameter. A constant LR almost never works at scale.

**Warmup:** Start with a tiny LR, increase linearly over $T_w$ steps to the target LR. This prevents early instability when the model is randomly initialized and gradients are noisy.

$$\text{lr}(t) = \text{lr}_{\max} \cdot \frac{t}{T_w} \quad \text{for } t < T_w$$

**Cosine decay:** After warmup, decay the LR following a cosine curve to near zero:

$$\text{lr}(t) = \text{lr}_{\min} + \frac{1}{2}(\text{lr}_{\max} - \text{lr}_{\min})\left(1 + \cos\left(\frac{t - T_w}{T - T_w}\pi\right)\right)$$

```
Learning Rate: Warmup + Cosine Decay

lr_max ─ ─ ─ ─ ─ ─ ┐
                    │╲
           ╱        │  ╲
          ╱         │    ╲
         ╱          │      ╲
        ╱           │        ╲
       ╱            │          ╲
lr_min ─────────────┴────────────╲───
      0         T_w              T
      ├─warmup──┤├──cosine decay──┤
```

**Why warmup matters for transformers:** The Adam optimizer accumulates gradient statistics. Early in training these estimates are wildly inaccurate. Warmup gives Adam time to calibrate before taking large steps.

### 5. Gradient Clipping

When gradients explode, a single bad batch can destroy your model. Gradient clipping caps the gradient norm:

$$\mathbf{g} \leftarrow \frac{\text{max\_norm}}{\max(\|\mathbf{g}\|, \text{max\_norm})} \cdot \mathbf{g}$$

If $\|\mathbf{g}\| \leq \text{max\_norm}$, gradients pass through unchanged. Otherwise, they're scaled down to have norm = max_norm. Direction is preserved; magnitude is capped.

**Typical values:** max_norm = 1.0 for transformers, 5.0 for RNNs.

---

## Implementation (60 min)

### The Training Stability Cookbook

Build a reusable training loop with *all* stability techniques and gradient monitoring.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import math

# --- Model with configurable normalization ---
class StableNet(nn.Module):
    """A 4-layer CNN with configurable normalization."""
    def __init__(self, norm_type='none'):
        super().__init__()
        self.norm_type = norm_type

        def make_norm(channels):
            if norm_type == 'batch':
                return nn.BatchNorm2d(channels)
            elif norm_type == 'layer':
                return nn.GroupNorm(1, channels)  # GroupNorm(1, C) == LayerNorm
            else:
                return nn.Identity()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            make_norm(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            make_norm(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            make_norm(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, 10)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# --- Cosine schedule with warmup ---
class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, lr_min=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_min = lr_min
        self.lr_max = optimizer.param_groups[0]['lr']
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            lr = self.lr_max * self.step_count / self.warmup_steps
        else:
            progress = (self.step_count - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1 + math.cos(math.pi * progress)
            )
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr


# --- Gradient monitoring ---
def log_gradient_norms(model):
    """Returns a dict of gradient norms per layer."""
    norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.norm().item()
    return norms


# --- Full training loop with all stability tricks ---
def train_with_cookbook(
    norm_type='batch',
    use_warmup=True,
    use_clipping=True,
    max_norm=1.0,
    epochs=5,
    lr=1e-3,
    batch_size=128,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Model (He init is applied inside)
    model = StableNet(norm_type=norm_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Schedule
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader) if use_warmup else 0
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)

    history = {'loss': [], 'grad_norms': [], 'lr': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Gradient clipping
            if use_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # Log gradient norms BEFORE optimizer step
            if batch_idx % 50 == 0:
                gnorms = log_gradient_norms(model)
                total_norm = sum(v**2 for v in gnorms.values()) ** 0.5
                history['grad_norms'].append(total_norm)

            optimizer.step()
            lr_now = scheduler.step()
            history['lr'].append(lr_now)
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        history['loss'].append(avg_loss)
        print(f"[{norm_type}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | LR: {lr_now:.6f}")

    return history
```

### Running the Comparison

```python
# Run all four configurations
configs = {
    'no_tricks': dict(norm_type='none', use_warmup=False, use_clipping=False),
    'batchnorm': dict(norm_type='batch', use_warmup=False, use_clipping=False),
    'layernorm': dict(norm_type='layer', use_warmup=False, use_clipping=False),
    'all_tricks': dict(norm_type='batch', use_warmup=True, use_clipping=True),
}

results = {}
for name, cfg in configs.items():
    print(f"\n{'='*50}")
    print(f"Training with: {name}")
    print(f"{'='*50}")
    results[name] = train_with_cookbook(**cfg)
```

---

## Exercise (45 min)

### Experiment: Comparing Stability Techniques

1. **Run all four configurations** from the implementation above. Plot loss curves on the same axes.

2. **Answer these questions:**
   - Which configuration converges fastest in the first epoch?
   - Which achieves the lowest final loss?
   - Look at the gradient norms for "no_tricks" vs "all_tricks". What do you observe?

3. **Breaking things intentionally:** Try setting `lr=0.1` (100× too high) with each configuration. Which ones still converge? This reveals the *safety margin* each technique provides.

4. **Gradient norm histogram:** For the "all_tricks" run, collect gradient norms across all layers at step 100, 500, and 2000. Are they roughly uniform across layers, or do early/late layers differ?

5. **The warmup experiment:** Train with cosine decay but *no* warmup (warmup_steps=0). Compare to warmup. At what model size does warmup become critical?

### Expected Results

| Config | Epoch 1 Loss | Final Loss | Grad Norm (avg) |
|--------|-------------|------------|-----------------|
| no_tricks | ~0.8 | ~0.15 | Highly variable |
| batchnorm | ~0.3 | ~0.05 | Moderate |
| layernorm | ~0.4 | ~0.08 | Stable |
| all_tricks | ~0.25 | ~0.04 | Very stable |

**Key insight:** Normalization has the *single biggest impact*. Warmup and clipping are safety nets — they prevent catastrophic failures more than they improve the average case.

---

## Key Takeaways

- **BatchNorm** smooths the loss landscape and enables higher learning rates, but introduces batch dependency
- **LayerNorm** gives per-sample normalization — essential for transformers and variable-length sequences
- **Weight initialization** (Xavier/He) prevents signal explosion/vanishing on the first forward pass
- **LR warmup** gives optimizers time to calibrate their statistics; critical for Adam at scale
- **Gradient clipping** is your safety net — it prevents catastrophic updates from bad batches
- All five techniques are used together in every modern VLA; omitting any one is asking for trouble

## Connection to the Thread

The stability cookbook is about **preserving information flow** through deep networks. Each technique addresses a specific way information can be destroyed:
- Normalization prevents activation magnitudes from drifting → maintains the *signal*
- Proper initialization starts with the right signal-to-noise ratio
- LR schedules control how much the compression function changes per step
- Gradient clipping prevents catastrophic information loss from a single bad update

When we build transformers (Day 10+) and eventually VLAs, every single one of these techniques will be present. The transformer architecture was *designed* with LayerNorm as a first-class citizen. Training a 7B-parameter VLA like RT-2 without this cookbook is impossible.

## Further Reading

- [Batch Normalization: Accelerating Deep Network Training](https://arxiv.org/abs/1502.03167) — Ioffe & Szegedy, 2015
- [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604) — Santurkar et al., 2018 (the "smoothing" perspective)
- [Layer Normalization](https://arxiv.org/abs/1607.06450) — Ba, Kiros & Hinton, 2016
- [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) — He et al., 2015 (Kaiming init)
- [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265) — RAdam: why warmup works
