# Day 60: CLIP Internals + SigLIP

> **Phase V — Vision-Language Models | Week 9 | 2.5 hours**
> *"The temperature parameter isn't just a hyperparameter — it's a learned measure of how confident the model should be about its alignments." — On learned temperature*

## Navigation
- **Previous:** [Day 59: CLIP](day-59-clip.md)
- **Next:** [Day 61: Flamingo & BLIP-2](day-61-flamingo-blip2.md)
- **Week:** [Week 9 Overview](README.md)
- **Phase:** [Phase V: Vision-Language Models](../../study-notes/10-vision-language-models.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### CLIP's Learned Temperature

The temperature $\tau$ in CLIP's contrastive loss is **learned** (initialized to 0.07):

$$\text{logits}_{ij} = \frac{\mathbf{v}_i \cdot \mathbf{t}_j}{\tau}$$

Why learn it?
- **High $\tau$** → soft distribution → model explores, considers multiple matches
- **Low $\tau$** → sharp distribution → model is confident, focuses on best match
- Optimal $\tau$ changes during training: starts soft, becomes sharper

In practice, CLIP parameterizes it as $\tau = \exp(\log\tau)$ where $\log\tau$ is learnable, clamped to $[-\log 100, \log 100]$.

### Vision-Language Alignment Deep Dive

CLIP's alignment has subtle but important properties:

```
Batch similarity matrix (N×N):
              text₁  text₂  text₃  ...  textₙ
image₁    [  0.95   0.12   0.05  ...  0.03  ]  ← diagonal = matched
image₂    [  0.08   0.91   0.15  ...  0.04  ]
image₃    [  0.03   0.11   0.88  ...  0.07  ]
  ...     [  ...    ...    ...   ...  ...   ]
imageₙ   [  0.02   0.06   0.03  ...  0.93  ]

Goal: maximize diagonal, minimize off-diagonal
This is symmetric: image→text AND text→image
```

**Hard negatives:** Within a batch, the most similar non-matching pairs are the hardest to separate. Larger batch sizes = more hard negatives = better alignment. CLIP uses batch size **32,768**.

### SigLIP: Sigmoid Loss for Language-Image Pretraining

SigLIP (2023) replaces CLIP's softmax-based loss with **pairwise sigmoid loss**:

$$\mathcal{L}_{\text{SigLIP}} = -\frac{1}{N^2} \sum_{i,j} \log \sigma\left(z_{ij}(-1)^{y_{ij}} (\mathbf{v}_i \cdot \mathbf{t}_j - b)\right)$$

where:
- $y_{ij} = 1$ if $(i, j)$ is a matched pair, $-1$ otherwise
- $b$ is a learned bias
- $z_{ij}$ is a per-pair weight

Key differences from CLIP:

| Property | CLIP (Softmax) | SigLIP (Sigmoid) |
|----------|---------------|-----------------|
| Loss type | N-way classification | Binary per-pair |
| Normalization | Across all negatives | Independent per pair |
| Batch dependency | Softmax couples all items | Each pair independent |
| Large batch scaling | Requires distributed softmax | Naturally parallel |
| Performance | Baseline | +2% on ImageNet zero-shot |

### Why Sigmoid > Softmax for Scaling

Softmax requires computing a normalizer over the entire batch:

$$\text{softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)} \quad \leftarrow \text{requires all-gather across GPUs}$$

Sigmoid treats each pair independently:

$$\sigma(x_{ij}) = \frac{1}{1 + \exp(-x_{ij})} \quad \leftarrow \text{no cross-GPU communication}$$

This makes SigLIP much easier to scale to massive batch sizes across many GPUs.

### Embedding Space Geometry

CLIP/SigLIP create a **hypersphere** where semantically similar items are close:

```
Embedding space (projected to 2D):

    "a golden retriever"  •──────•  [photo of golden retriever]
                           \
                            •  [photo of labrador]
                           /
    "a labrador dog"      •

    "a red sports car"    •──────•  [photo of Ferrari]
                                    
    "a pickup truck"      •──────•  [photo of F-150]

Distance between clusters >> distance within clusters
```

---

## Implementation (60 min)

### Implementing Contrastive Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPLoss(nn.Module):
    """Standard CLIP contrastive loss with learned temperature."""
    
    def __init__(self, init_temperature=0.07):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(init_temperature)))
    
    def forward(self, image_embeds, text_embeds):
        """
        Args:
            image_embeds: (B, D) normalized image embeddings
            text_embeds: (B, D) normalized text embeddings
        """
        temperature = self.log_temperature.exp().clamp(max=100.0)
        
        # Similarity matrix
        logits = (image_embeds @ text_embeds.T) / temperature  # (B, B)
        
        # Labels: diagonal entries are correct matches
        labels = torch.arange(len(logits), device=logits.device)
        
        # Symmetric loss
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        return (loss_i2t + loss_t2i) / 2


class SigLIPLoss(nn.Module):
    """SigLIP pairwise sigmoid loss — no softmax normalization."""
    
    def __init__(self, init_temperature=10.0, init_bias=-10.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temperature))
        self.bias = nn.Parameter(torch.tensor(init_bias))
    
    def forward(self, image_embeds, text_embeds):
        """
        Args:
            image_embeds: (B, D) normalized image embeddings
            text_embeds: (B, D) normalized text embeddings
        """
        # Pairwise similarities
        logits = (image_embeds @ text_embeds.T) * self.temperature + self.bias  # (B, B)
        
        # Labels: +1 for matched, -1 for unmatched
        B = logits.shape[0]
        labels = 2 * torch.eye(B, device=logits.device) - 1  # +1 diagonal, -1 elsewhere
        
        # Binary cross-entropy per pair
        loss = -F.logsigmoid(labels * logits).mean()
        
        return loss


# Compare the two losses
B, D = 32, 512
img_emb = F.normalize(torch.randn(B, D), dim=-1)
txt_emb = F.normalize(torch.randn(B, D), dim=-1)

clip_loss = CLIPLoss()
siglip_loss = SigLIPLoss()

print(f"CLIP loss:   {clip_loss(img_emb, txt_emb).item():.4f}")
print(f"SigLIP loss: {siglip_loss(img_emb, txt_emb).item():.4f}")
print(f"Temperature (CLIP):  {clip_loss.log_temperature.exp().item():.4f}")
print(f"Temperature (SigLIP): {siglip_loss.temperature.item():.4f}")
```

### Embedding Space Analysis

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def analyze_clip_embeddings(image_paths, text_descriptions, model_name="openai/clip-vit-base-patch32"):
    """Visualize the geometry of CLIP's embedding space."""
    from transformers import CLIPModel, CLIPProcessor
    
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    
    # Encode images
    images = [Image.open(p).convert("RGB") for p in image_paths]
    img_inputs = processor(images=images, return_tensors="pt", padding=True)
    with torch.no_grad():
        img_embeds = F.normalize(model.get_image_features(**img_inputs), dim=-1)
    
    # Encode text
    txt_inputs = processor(text=text_descriptions, return_tensors="pt", padding=True)
    with torch.no_grad():
        txt_embeds = F.normalize(model.get_text_features(**txt_inputs), dim=-1)
    
    # Combine and project with t-SNE
    all_embeds = torch.cat([img_embeds, txt_embeds], dim=0).numpy()
    n_images = len(image_paths)
    
    tsne = TSNE(n_components=2, perplexity=min(5, len(all_embeds) - 1), random_state=42)
    projected = tsne.fit_transform(all_embeds)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(projected[:n_images, 0], projected[:n_images, 1],
               c='blue', marker='o', s=100, label='Images')
    ax.scatter(projected[n_images:, 0], projected[n_images:, 1],
               c='red', marker='x', s=100, label='Text')
    
    # Draw lines between matched pairs
    for i in range(min(n_images, len(text_descriptions))):
        ax.plot([projected[i, 0], projected[n_images + i, 0]],
                [projected[i, 1], projected[n_images + i, 1]],
                'k--', alpha=0.3)
    
    ax.legend()
    ax.set_title('CLIP Embedding Space (t-SNE)')
    plt.tight_layout()
    plt.savefig('clip_embedding_space.png', dpi=150)
```

---

## Exercise (45 min)

1. **Temperature ablation:** Train a small CLIP model (using CIFAR-10 + synthetic captions) with fixed temperatures {0.01, 0.07, 0.5, 1.0} vs learned temperature. Plot validation loss curves. What temperature does the learnable version converge to?

2. **SigLIP vs CLIP:** Implement both losses and train identical models. Compare convergence speed and final zero-shot accuracy. Does SigLIP's independence per pair help?

3. **Embedding analysis:** Encode 100 images and their captions with CLIP. Compute the full similarity matrix. What fraction of images have their correct caption as the top-1 retrieval? What about top-5?

---

## Key Takeaways

1. **Learned temperature.** Adapts confidence automatically — starts exploratory, becomes confident
2. **SigLIP improves CLIP.** Pairwise sigmoid removes softmax bottleneck, enables easier scaling
3. **Batch size matters.** Larger batches = more hard negatives = better alignment
4. **Embedding geometry.** Matched image-text pairs cluster in the shared space
5. **Foundation model.** CLIP/SigLIP vision encoders power most modern VLMs

## Connection to the Thread

> *CLIP aligns vision and language in a shared space, but it can't generate text. Tomorrow: Flamingo and BLIP-2, which connect CLIP-style vision encoders to frozen LLMs for visual understanding and generation.*

---

## Further Reading

- [SigLIP (Zhai et al., 2023)](https://arxiv.org/abs/2303.15343)
- [Scaling Up Visual and Vision-Language Representation Learning (ALIGN, 2021)](https://arxiv.org/abs/2102.05918)
- [OpenCLIP training recipes](https://github.com/mlfoundations/open_clip)
- [EVA-CLIP (Sun et al., 2023)](https://arxiv.org/abs/2303.15389)
