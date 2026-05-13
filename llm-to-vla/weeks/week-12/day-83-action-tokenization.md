# Day 83: Action Tokenization — Discretizing Robot Actions

> **Phase VI — Robot Learning: RL, Diffusion & Data | Week 12 | 2.5 hours**
> *"RT-2's trick: turn continuous robot actions into tokens, then let the language model do what language models do — predict the next token." — Brohan et al., 2023*

## Navigation
- **Previous:** [Day 82: Action Representations](day-82-action-representations.md)
- **Next:** [Day 84: Stop & Reflect #5](day-84-stop-reflect-5.md)
- **Week:** [Week 12 Overview](README.md)
- **Phase:** [Phase VI: Robot Learning](../../study-notes/13-imitation-learning.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 83.1 Why Tokenize Actions?

Language models output discrete tokens via softmax. Robot actions are continuous. To use a VLM as a VLA, we need to **discretize actions into tokens**.

```
Continuous:  a = [0.0312, -0.0157, 0.0089, 0.872, 0.001, -0.003, 1.0]
                   ↓ discretize (256 bins per dimension)
Discrete:    tokens = [132, 118, 130, 223, 128, 127, 255]
                   ↓ add to vocabulary
Text tokens: ["pick", "up", "the", "mug", 132, 118, 130, 223, 128, 127, 255]
```

### 83.2 Uniform Discretization (RT-2 Approach)

RT-2 uses the simplest approach — uniform bins:

1. Normalize each action dimension to $[0, 1]$
2. Discretize into $N$ bins (RT-2 uses 256)
3. Map to token indices starting after the text vocabulary

$$\text{bin}(a_i) = \text{clamp}\left(\lfloor a_i \cdot N \rfloor, 0, N-1\right)$$

**Example with $N = 256$:**

| Action Dimension | Range | Value | Bin |
|-----------------|-------|-------|-----|
| $\Delta x$ | $[-0.05, 0.05]$ | 0.012 | 159 |
| $\Delta y$ | $[-0.05, 0.05]$ | -0.008 | 118 |
| $\Delta z$ | $[-0.05, 0.05]$ | 0.003 | 140 |
| $\Delta\text{yaw}$ | $[-\pi/4, \pi/4]$ | 0.01 | 130 |
| gripper | $[0, 1]$ | 1.0 | 255 |

### 83.3 Resolution vs Vocabulary Size

| Bins | Resolution ($\Delta x$ in 10cm range) | Vocab Tokens | Accuracy |
|------|--------------------------------------|-------------|----------|
| 16 | 6.25mm | 16 × 7 = 112 | Poor |
| 64 | 1.56mm | 64 × 7 = 448 | Acceptable |
| 256 | 0.39mm | 256 × 7 = 1792 | Good |
| 1024 | 0.10mm | 1024 × 7 = 7168 | Excellent but slow |

**Trade-off:** more bins = finer control but larger softmax = slower inference.

### 83.4 VQ-VAE for Action Tokenization

Instead of uniform bins, learn a **codebook** that captures the structure of the action distribution:

```
Continuous actions  →  VQ-VAE Encoder  →  Codebook lookup  →  Discrete tokens
                                              │
                                    Codebook: K learned centroids
                                    Each centroid represents a
                                    common action pattern
```

$$z_q = \text{argmin}_{e_k \in \mathcal{C}} \| z_e - e_k \|_2$$

**Advantages over uniform bins:**
- Allocates more codes to high-density action regions
- Can tokenize entire action chunks (not just individual dimensions)
- Learned representations capture temporal structure

### 83.5 Autoregressive vs Parallel Action Decoding

**Autoregressive (RT-2 style):**
```
[image] [text] → [a₁] → [a₂] → [a₃] → ... → [a₇]
                  ↑ previous tokens condition each prediction
Time: 7 × forward pass
```

**Parallel (Octo/OpenVLA style):**
```
[image] [text] → Diffusion/Flow head → [a₁, a₂, ..., a₇] simultaneously
Time: 1 × forward pass (plus denoising steps)
```

| Method | Speed | Expressiveness | Architecture |
|--------|-------|---------------|-------------|
| Autoregressive | Slow (7× forward) | Can model inter-dim dependencies | Pure transformer |
| Parallel | Fast (1× forward) | Independent dimensions | Diffusion/flow head |

---

## Implementation (60 min)

### Uniform Action Tokenization

```python
import torch
import torch.nn as nn
import numpy as np

class UniformActionTokenizer:
    """RT-2 style uniform binning."""
    def __init__(self, action_ranges, n_bins=256, vocab_offset=32000):
        """
        action_ranges: list of (min, max) per dimension
        n_bins: number of bins per dimension
        vocab_offset: start of action tokens in vocabulary
        """
        self.ranges = np.array(action_ranges)
        self.n_bins = n_bins
        self.offset = vocab_offset
        self.action_dim = len(action_ranges)

    def encode(self, actions):
        """Continuous actions → discrete tokens."""
        normalized = (actions - self.ranges[:, 0]) / (self.ranges[:, 1] - self.ranges[:, 0])
        normalized = np.clip(normalized, 0, 1 - 1e-6)
        bins = (normalized * self.n_bins).astype(int)
        tokens = bins + self.offset
        return tokens

    def decode(self, tokens):
        """Discrete tokens → continuous actions (bin centers)."""
        bins = tokens - self.offset
        normalized = (bins + 0.5) / self.n_bins
        actions = normalized * (self.ranges[:, 1] - self.ranges[:, 0]) + self.ranges[:, 0]
        return actions

    @property
    def vocab_size(self):
        return self.n_bins  # Shared across dimensions

# Usage
tokenizer = UniformActionTokenizer(
    action_ranges=[
        (-0.05, 0.05),   # delta_x
        (-0.05, 0.05),   # delta_y
        (-0.05, 0.05),   # delta_z
        (-0.25, 0.25),   # delta_yaw
        (-0.25, 0.25),   # delta_pitch
        (-0.25, 0.25),   # delta_roll
        (0.0, 1.0),      # gripper
    ],
    n_bins=256,
)

action = np.array([0.01, -0.02, 0.005, 0.1, 0.0, -0.05, 1.0])
tokens = tokenizer.encode(action)
reconstructed = tokenizer.decode(tokens)
print(f"Original:      {action}")
print(f"Tokens:        {tokens}")
print(f"Reconstructed: {reconstructed}")
print(f"Max error:     {np.abs(action - reconstructed).max():.6f}")
```

### VQ-VAE Action Tokenizer

```python
class VQVAEActionTokenizer(nn.Module):
    """Learned codebook for action tokenization."""
    def __init__(self, action_dim, codebook_size=512, code_dim=32, chunk_size=4):
        super().__init__()
        self.chunk_size = chunk_size
        self.encoder = nn.Sequential(
            nn.Linear(action_dim * chunk_size, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, code_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, action_dim * chunk_size),
        )
        # Learnable codebook
        self.codebook = nn.Embedding(codebook_size, code_dim)
        self.codebook.weight.data.uniform_(-1/codebook_size, 1/codebook_size)

    def quantize(self, z_e):
        """Find nearest codebook entry."""
        distances = torch.cdist(z_e.unsqueeze(1), self.codebook.weight.unsqueeze(0))
        indices = distances.argmin(dim=-1).squeeze(1)
        z_q = self.codebook(indices)
        return z_q, indices

    def forward(self, action_chunks):
        """action_chunks: (B, chunk_size, action_dim)"""
        flat = action_chunks.flatten(1)
        z_e = self.encoder(flat)
        z_q, indices = self.quantize(z_e)

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        reconstructed = self.decoder(z_q_st).view_as(action_chunks)

        # Losses
        recon_loss = ((reconstructed - action_chunks)**2).mean()
        commitment_loss = ((z_e.detach() - z_q)**2).mean()
        codebook_loss = ((z_e - z_q.detach())**2).mean()

        return reconstructed, indices, recon_loss + 0.25 * commitment_loss + codebook_loss
```

---

## Exercise (45 min)

1. **Resolution analysis:** Compare 64, 128, 256, 512 bins. Measure reconstruction error and training loss of a BC policy using tokenized actions.

2. **VQ-VAE codebook utilization:** Train the VQ-VAE tokenizer. What fraction of codebook entries are actually used? Implement codebook reset for unused entries.

3. **Autoregressive action generation:** Build a tiny transformer that predicts action tokens autoregressively. Measure if inter-dimension conditioning helps accuracy.

4. **RT-2 vocabulary design:** Given text vocab size 32,000 and 256 action bins for 7 dimensions, design the complete token vocabulary. How does the action token fraction affect training?

---

## Key Takeaways

1. **Uniform binning** is simple and works well (RT-2, OpenVLA)
2. **256 bins ≈ 0.4mm resolution** for typical manipulation ranges
3. **VQ-VAE** learns task-aware codebooks but adds complexity
4. **Autoregressive decoding** is slow but captures inter-dimension dependencies
5. **Action tokenization bridges** continuous robotics and discrete language models

---

## Connection to the Thread

> *With action tokenization, we've completed the bridge between language models and robot control. A VLM can now output text tokens AND action tokens from the same vocabulary. Tomorrow's reflection day connects all of Phase VI: from RL fundamentals through diffusion to action tokenization — the full toolkit for building VLAs.*

---

## Further Reading

- Brohan et al. (2023), "RT-2: Vision-Language-Action Models" — action tokenization design
- van den Oord et al. (2017), "Neural Discrete Representation Learning" (VQ-VAE)
- Lee et al. (2024), "Behavior Generation with Latent Actions" (VQ-BeT)
