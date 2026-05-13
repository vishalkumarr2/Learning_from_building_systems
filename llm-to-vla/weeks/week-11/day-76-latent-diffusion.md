# Day 76: Diffusion Day 3 — Latent Diffusion & Stable Diffusion

> **Phase VI — Robot Learning: RL, Diffusion & Data | Week 11 | 2.5 hours**
> *"Why diffuse in 512×512 pixel space when a 64×64 latent captures all the information?" — Rombach et al., 2022*

## Navigation
- **Previous:** [Day 75: Diffusion Day 2 — DDIM + CFG](day-75-ddim-cfg.md)
- **Next:** [Day 77: Flow Matching](day-77-flow-matching.md)
- **Week:** [Week 11 Overview](README.md)
- **Phase:** [Phase VI: Robot Learning](../../study-notes/12-diffusion-flow.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 76.1 The Pixel-Space Problem

DDPM operates on full-resolution images. For 512×512×3 images:
- Each denoising step processes 786,432 values
- 1000 steps × U-Net forward pass = enormous compute
- Training is slow, sampling is slower

**Insight:** most pixels are redundant. A pretrained VAE can compress images to a much smaller latent space while preserving semantics.

### 76.2 Latent Diffusion Architecture

```
Encoding:                    Diffusion:                    Decoding:
Image (512×512×3)    →    Latent (64×64×4)    →    Image (512×512×3)
     │                         │                         │
  VAE Encoder E(x)      U-Net denoising            VAE Decoder D(z)
  (frozen, pretrained)   (this is trained)         (frozen, pretrained)
                               │
                        Text Conditioning
                        via cross-attention
                        (CLIP text encoder)
```

**Three components:**
1. **VAE** (pretrained, frozen): compresses images 8× spatially → 64× fewer pixels
2. **U-Net**: denoises in latent space (much smaller tensors)
3. **Text encoder** (CLIP/T5): converts text prompts to conditioning embeddings

### 76.3 VAE: Compressing to Latent Space

The VAE learns a compressed representation:

$$z = E(x) \in \mathbb{R}^{h/f \times w/f \times c}$$

$$\hat{x} = D(z) \approx x$$

where $f = 8$ is the downsampling factor. KL regularization keeps the latent space smooth:

$$\mathcal{L}_\text{VAE} = \|x - D(E(x))\|^2 + \beta \cdot D_\text{KL}(q(z|x) \| \mathcal{N}(0, I))$$

### 76.4 Cross-Attention for Text Conditioning

At each U-Net layer, text embeddings condition the denoising via cross-attention:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

where:
- $Q = W_Q \cdot \phi_\text{spatial}$ (from U-Net spatial features)
- $K = W_K \cdot \tau_\theta(\text{text})$ (from text encoder)
- $V = W_V \cdot \tau_\theta(\text{text})$ (from text encoder)

### 76.5 Stable Diffusion = Latent Diffusion + Scale

| Component | Stable Diffusion 1.5 | SDXL |
|-----------|---------------------|------|
| VAE | KL-VAE, f=8, z∈ℝ^{64×64×4} | Same |
| U-Net | ~860M params | ~2.6B params |
| Text encoder | CLIP ViT-L/14 | CLIP ViT-L + OpenCLIP ViT-bigG |
| Training data | LAION-5B (filtered) | LAION-5B (higher quality) |
| Resolution | 512×512 | 1024×1024 |

### 76.6 Why This Matters for Robotics

Latent diffusion's architecture transfers directly to robot actions:

| Image Diffusion | Robot Action Diffusion |
|----------------|----------------------|
| VAE encodes images | Encoder compresses observation |
| U-Net denoises latent pixels | U-Net denoises latent actions |
| Text conditioning | Task instruction conditioning |
| Cross-attention on text | Cross-attention on observation features |

---

## Implementation (60 min)

### Using Stable Diffusion via Diffusers

```python
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch

# Load pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Generate
image = pipe(
    prompt="a robot arm picking up a red mug from a table, photorealistic",
    num_inference_steps=50,
    guidance_scale=7.5,
).images[0]
image.save("robot_mug.png")
```

### Inspect the VAE Latent Space

```python
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float16
).to("cuda")

# Encode an image
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

img = Image.open("robot_mug.png")
x = transform(img).unsqueeze(0).half().to("cuda")

# Encode to latent
with torch.no_grad():
    latent = vae.encode(x).latent_dist.sample()
print(f"Image shape: {x.shape}")       # [1, 3, 512, 512]
print(f"Latent shape: {latent.shape}")  # [1, 4, 64, 64]
print(f"Compression: {x.numel() / latent.numel():.0f}×")  # ~48×

# Visualize latent channels
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i in range(4):
    axes[i].imshow(latent[0, i].cpu().float(), cmap="viridis")
    axes[i].set_title(f"Latent channel {i}")
    axes[i].axis("off")
plt.tight_layout()
plt.show()

# Decode back
with torch.no_grad():
    reconstructed = vae.decode(latent).sample
# Compare original vs reconstructed
```

### Build a Minimal Latent Diffusion

```python
class LatentDiffusion:
    """Minimal latent diffusion wrapper."""
    def __init__(self, vae, denoiser, scheduler):
        self.vae = vae
        self.denoiser = denoiser
        self.scheduler = scheduler

    def encode(self, x):
        with torch.no_grad():
            return self.vae.encode(x).latent_dist.sample() * 0.18215

    def decode(self, z):
        with torch.no_grad():
            return self.vae.decode(z / 0.18215).sample

    def train_step(self, x, condition):
        z = self.encode(x)
        noise = torch.randn_like(z)
        t = torch.randint(0, len(self.scheduler), (z.shape[0],))
        z_noisy = self.scheduler.add_noise(z, noise, t)
        pred_noise = self.denoiser(z_noisy, t, condition)
        return ((pred_noise - noise) ** 2).mean()
```

---

## Exercise (45 min)

1. **Latent interpolation:** Encode two images to latents $z_1, z_2$. Generate images at interpolation points $z = (1-\alpha)z_1 + \alpha z_2$ for $\alpha \in \{0, 0.25, 0.5, 0.75, 1\}$. Does interpolation produce meaningful intermediate images?

2. **Guidance scale experiment:** Generate the same prompt at $w \in \{1, 3, 7, 12, 20\}$. At what point does quality degrade?

3. **Latent space arithmetic:** Can you do "concept algebra" in latent space, similar to word embedding arithmetic? Try encoding images of different objects and doing vector operations.

4. **Compute comparison:** Time the forward pass of a U-Net on 512×512×3 vs 64×64×4 inputs. How much faster is latent space?

---

## Key Takeaways

1. **Latent diffusion = VAE compression + diffusion in latent space** — 48× fewer values to process
2. **Three frozen components** (VAE encoder, VAE decoder, text encoder) + one trained (U-Net)
3. **Cross-attention** bridges text semantics to spatial features
4. **The same architecture** applies to robot actions: compress observations, denoise action sequences
5. **Stable Diffusion** = latent diffusion at scale with CLIP text conditioning

---

## Connection to the Thread

> *Latent diffusion teaches us that you don't need to work in raw data space — compress first, then generate. Tomorrow, flow matching provides an even simpler framework: straight-line paths in latent space, no noise schedule, no complicated math. π₀ (Day 96) will use flow matching for robot actions. The progression of generative models mirrors the progression of robot action generation: from explicit (RL) → implicit (diffusion) → elegant (flow matching).*

---

## Further Reading

- Rombach et al. (2022), "High-Resolution Image Synthesis with Latent Diffusion Models"
- Esser et al. (2024), "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (SD3)
- Podell et al. (2023), "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis"
