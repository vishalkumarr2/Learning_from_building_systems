# Day 8: Phase I Mini-Project — The Autoencoder

> **Phase I — DL Foundations & Information Theory | Week 2 | 2.5 hours**
> *"If you can compress it, you understand it. An autoencoder is your first compression algorithm — and the ancestor of every generative model in robotics."*

## Navigation
- **Previous:** [Day 7: Training Stability Cookbook](../week-01/day-07-training-stability.md)
- **Next:** [Day 9: Phase I Checkpoint](day-09-phase1-checkpoint.md)
- **Week:** [Week 2 Overview](../README.md)
- **Phase:** [Phase I: DL Foundations](../../study-notes/01-dl-foundations.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### The Autoencoder: Compression Made Tangible

Everything from Days 1–7 converges here. An autoencoder is a neural network trained to reconstruct its input through a **bottleneck**. It must learn to compress the input into a low-dimensional latent vector, then decompress it back.

```
Input x ──→ [Encoder] ──→ z (latent) ──→ [Decoder] ──→ x̂ (reconstruction)
  28×28         ↓            ↓                ↓
  784 dims    compress    d dims          784 dims
                         (d << 784)
```

The bottleneck dimension $d$ controls how much information passes through. This is literally the **rate** in rate-distortion theory (Day 5):

- **Small $d$**: aggressive compression → lossy reconstructions → low rate, high distortion
- **Large $d$**: mild compression → near-perfect reconstructions → high rate, low distortion

### Why This Matters for VLAs

The autoencoder architecture is the foundation of:
- **VAE** (Variational Autoencoder): adds probabilistic structure to the latent space → used in latent diffusion models
- **VQ-VAE**: discretizes the latent space → used in image tokenizers for multimodal LLMs
- **Image tokenizers in VLAs**: RT-2 and Octo use pretrained encoders that are conceptually autoencoders trained on massive datasets

### The Information-Theoretic Question

From Day 5, we know MNIST digits have an entropy of roughly $H \approx$ 2–3 bits per image (there are only 10 classes with relatively low intra-class variation). So in theory, you need only ~3 dimensions to capture the essential information. But the pixel-level reconstruction requires more — the *perceptual* information is higher than the *semantic* information.

$$\text{Rate-Distortion: } R(D) = \min_{p(\hat{x}|x): \mathbb{E}[d(x,\hat{x})] \leq D} I(X; \hat{X})$$

Your autoencoder will empirically discover this tradeoff.

### Convolutional Autoencoder Architecture

Why convolutional? Images have spatial structure. Fully-connected autoencoders ignore this and need far more parameters.

```
ENCODER                              DECODER
┌──────────────────────┐             ┌──────────────────────┐
│ Input: 1×28×28       │             │ Latent: d dims       │
│         ↓            │             │         ↓            │
│ Conv2d(1→32, 3, s=2) │             │ Linear(d→128×7×7)    │
│ + ReLU + BN          │             │ Reshape to 128×7×7   │
│ → 32×14×14           │             │         ↓            │
│         ↓            │             │ ConvT(128→64, 3, s=2)│
│ Conv2d(32→64, 3, s=2)│             │ + ReLU + BN          │
│ + ReLU + BN          │             │ → 64×14×14           │
│ → 64×7×7             │             │         ↓            │
│         ↓            │             │ ConvT(64→32, 3, s=2) │
│ Conv2d(64→128, 3, p) │             │ + ReLU + BN          │
│ + ReLU + BN          │             │ → 32×28×28           │
│ → 128×7×7            │             │         ↓            │
│         ↓            │             │ Conv2d(32→1, 3, p=1) │
│ Flatten → Linear(d)  │             │ + Sigmoid            │
│ → d dims (latent)    │             │ → 1×28×28            │
└──────────────────────┘             └──────────────────────┘
```

**Loss function:** MSE (mean squared error) between input and reconstruction:

$$\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} \|x_i - \hat{x}_i\|^2$$

This is equivalent to maximizing a Gaussian log-likelihood — connecting back to our cross-entropy discussion from Day 5.

---

## Implementation (60 min)

### Building the Convolutional Autoencoder

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # 28→14
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14→7
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), # 7→7
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 7 * 7, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 7 * 7)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), # 7→14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 14→28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),  # 28→28
            nn.Sigmoid(),  # pixel values in [0, 1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 128, 7, 7)
        return self.deconv(x)


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
```

### Training Loop (With Stability Cookbook!)

```python
def train_autoencoder(latent_dim, epochs=15, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        # No normalization — Sigmoid output expects [0, 1] targets
    ])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    model = ConvAutoencoder(latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Cosine schedule — from Day 7's cookbook
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, _ in train_loader:  # labels unused — unsupervised!
            data = data.to(device)
            optimizer.zero_grad()
            x_hat, z = model(data)
            loss = nn.functional.mse_loss(x_hat, data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step()
        history.append(avg_loss)
        print(f"[d={latent_dim}] Epoch {epoch+1}/{epochs} | MSE: {avg_loss:.6f}")

    return model, history


# Train with different bottleneck sizes
bottleneck_sizes = [2, 8, 32, 128]
models = {}
histories = {}

for d in bottleneck_sizes:
    print(f"\n{'='*50}")
    print(f"Training autoencoder with latent_dim = {d}")
    print(f"{'='*50}")
    models[d], histories[d] = train_autoencoder(d, epochs=15)
```

### Visualizing Results

```python
def visualize_reconstructions(models, test_loader, device, n=8):
    """Show originals and reconstructions for each bottleneck size."""
    test_batch, _ = next(iter(test_loader))
    test_batch = test_batch[:n].to(device)

    fig, axes = plt.subplots(len(models) + 1, n, figsize=(n * 1.5, (len(models) + 1) * 1.5))

    # Original images
    for i in range(n):
        axes[0, i].imshow(test_batch[i, 0].cpu(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=10)

    # Reconstructions for each bottleneck
    for row, (d, model) in enumerate(sorted(models.items()), 1):
        model.eval()
        with torch.no_grad():
            x_hat, _ = model(test_batch)
        for i in range(n):
            axes[row, i].imshow(x_hat[i, 0].cpu(), cmap='gray')
            axes[row, i].axis('off')
            if i == 0:
                axes[row, i].set_ylabel(f'd={d}', fontsize=10)

    plt.suptitle('Reconstruction Quality vs Bottleneck Size', fontsize=14)
    plt.tight_layout()
    plt.savefig('autoencoder_reconstructions.png', dpi=150)
    plt.show()


def visualize_latent_space(model_2d, test_loader, device):
    """Plot the 2D latent space colored by digit class."""
    model_2d.eval()
    zs, labels = [], []
    with torch.no_grad():
        for data, label in test_loader:
            _, z = model_2d(data.to(device))
            zs.append(z.cpu())
            labels.append(label)

    zs = torch.cat(zs).numpy()
    labels = torch.cat(labels).numpy()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(zs[:, 0], zs[:, 1], c=labels, cmap='tab10',
                          alpha=0.5, s=1)
    plt.colorbar(scatter, label='Digit')
    plt.xlabel('z₁')
    plt.ylabel('z₂')
    plt.title('2D Latent Space of MNIST Autoencoder')
    plt.savefig('latent_space_2d.png', dpi=150)
    plt.show()
```

### Computing the Compression Ratio

```python
def compression_analysis(bottleneck_sizes, histories):
    """Quantify the rate-distortion tradeoff."""
    print(f"\n{'Latent Dim':>10} | {'Compression':>12} | {'Final MSE':>10} | {'PSNR (dB)':>10}")
    print("-" * 50)
    for d in bottleneck_sizes:
        compression_ratio = 784 / d  # input dims / latent dims
        final_mse = histories[d][-1]
        # PSNR: peak signal-to-noise ratio (higher = better)
        psnr = 10 * np.log10(1.0 / final_mse) if final_mse > 0 else float('inf')
        print(f"{d:>10} | {compression_ratio:>10.1f}× | {final_mse:>10.6f} | {psnr:>10.2f}")

compression_analysis(bottleneck_sizes, histories)
```

Expected output:
```
 Latent Dim | Compression |  Final MSE |  PSNR (dB)
--------------------------------------------------
         2 |      392.0× |   0.042000 |      13.77
         8 |       98.0× |   0.018000 |      17.45
        32 |       24.5× |   0.006500 |      21.87
       128 |        6.1× |   0.002100 |      26.78
```

---

## Exercise (45 min)

### 1. The Information-Theoretic Question

At what bottleneck size does reconstruction become "good enough"?

- Define "good enough" as PSNR > 20 dB (reconstructions look crisp to the human eye)
- Where does your curve cross this threshold?
- Connect to Day 5: the entropy of MNIST determines the minimum information needed. The class entropy is $\log_2(10) \approx 3.3$ bits, but pixel-level reconstruction requires encoding *style* too — stroke thickness, slant, size

### 2. Latent Space Exploration (d=2 only)

With the 2D model:
- Are the digit clusters well-separated? Which digits overlap most? (4/9, 3/5, 7/1 are typical)
- Sample a grid of points in the 2D latent space and decode them. What do you see between clusters?
- This "interpolation" between digits is a preview of *generation* — the key idea behind diffusion models

### 3. The Overcomplete Autoencoder

What happens if $d > 784$ (latent dim *larger* than input)?
- The model can learn the identity function — no compression at all
- Try $d = 1024$. Is MSE lower? Is the latent space meaningful?
- This is why VAEs add a KL penalty: to prevent trivial solutions

### 4. Connecting Everything

Fill in this table linking each Day 1-7 concept to where it appears in your autoencoder:

| Concept | Where in Autoencoder |
|---------|---------------------|
| Backpropagation (Day 1) | ? |
| CNN spatial features (Day 2) | ? |
| Residual connections (Day 3) | ? |
| Seq2seq bottleneck (Day 4) | ? |
| Compression = prediction (Day 5) | ? |
| Learned embeddings (Day 6) | ? |
| Training stability (Day 7) | ? |

---

## Key Takeaways

- An autoencoder learns to **compress** and **reconstruct** — making the rate-distortion tradeoff tangible
- The bottleneck dimension controls the information capacity: too small → blurry, too large → trivial
- The 2D latent space reveals how the network *organizes* knowledge — similar digits cluster together
- Convolutional architecture exploits spatial structure, requiring far fewer parameters than FC
- This architecture is the direct ancestor of VAEs, VQ-VAEs, and the image tokenizers used in VLAs
- Every stability technique from Day 7 (BatchNorm, Kaiming init, cosine schedule, gradient clipping) is used here

## Connection to the Thread

The autoencoder is **compression made literal**. You're building a function that maps 784 dimensions to $d$ dimensions and back. The quality of reconstruction tells you how much information was preserved — and how much was discarded.

This is the same operation that VLAs perform: compress a high-dimensional camera image into a latent representation, then use that representation to predict actions. The autoencoder bottleneck is the prototype of every "visual encoder" in robotics.

The thread: compression = prediction = intelligence. Today you built the compressor.

## Further Reading

- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Kingma & Welling, 2013 (the VAE paper — next evolution)
- [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — van den Oord et al., 2017 (VQ-VAE)
- [An Introduction to Autoencoders](https://www.deeplearningbook.org/contents/autoencoders.html) — Goodfellow et al., Deep Learning textbook Ch. 14
- [Tutorial: Autoencoders](https://lilianweng.github.io/posts/2018-08-12-vae/) — Lilian Weng's excellent walkthrough
