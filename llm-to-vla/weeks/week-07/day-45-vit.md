# Day 45: ViT — An Image Is Worth 16×16 Words

> **Phase IV — Vision: ViT, 3D, Video | Week 7 | 2.5 hours**
> *"An image is worth 16×16 words — and the transformer doesn't care which kind of token it processes." — Dosovitskiy et al., 2020*

## Navigation
- **Previous:** [Day 44: Phase III Capstone Day 3](../week-06/day-44-phase3-capstone-3.md)
- **Next:** [Day 46: Training ViT + DeiT](day-46-training-vit-deit.md)
- **Week:** [Week 7 Overview](README.md)
- **Phase:** [Phase IV: Vision](../../study-notes/08-vision-transformers.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### The Core Insight

CNNs process images with convolutions — sliding filters with inductive biases for locality and translation invariance. ViT asks: what if we **throw all that away** and treat an image as a sequence of patches, exactly like words?

```
Image (224 × 224 × 3)
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  Split into 16×16 patches → 196 patches of 768 dims │
│                                                      │
│  [CLS] P₁ P₂ P₃ ... P₁₉₆  ← sequence of tokens    │
│    +    +  +  +       +                              │
│  E_pos E₁ E₂ E₃ ... E₁₉₆  ← position embeddings   │
│                                                      │
│  ┌─────────────────────────────────────┐             │
│  │   Transformer Encoder × L layers   │             │
│  │   (same as GPT/BERT encoder)       │             │
│  └─────────────────────────────────────┘             │
│                                                      │
│  [CLS] output → classification head                  │
└──────────────────────────────────────────────────────┘
```

### Patch Embedding

Each 16×16×3 patch is flattened to a 768-dim vector via a linear projection:

$$\mathbf{z}_0 = [\mathbf{x}_{\text{class}}; \; \mathbf{x}_p^1 \mathbf{E}; \; \mathbf{x}_p^2 \mathbf{E}; \; \ldots; \; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{\text{pos}}$$

where:
- $\mathbf{x}_p^i \in \mathbb{R}^{P^2 \cdot C}$ is the flattened $i$-th patch
- $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$ is the linear projection
- $\mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ are learned position embeddings
- $\mathbf{x}_{\text{class}}$ is the learnable [CLS] token

### ViT vs CNN Comparison

| Property | CNN | ViT |
|----------|-----|-----|
| Inductive bias | Locality, translation invariance | Almost none — learned from data |
| Data efficiency | Good with small data | Needs large data (or tricks like DeiT) |
| Receptive field | Grows layer by layer | Global from layer 1 |
| Scalability | Saturates at large scale | Scales well with data + compute |
| Position info | Implicit from convolution stride | Explicit position embeddings |

### Why [CLS] Token?

The [CLS] token is a learnable embedding prepended to the sequence. After passing through the transformer, its output is used for classification — it serves as a "summary" of the entire image, analogous to BERT's [CLS] for sentences.

### Position Embeddings

ViT uses **learned 1D positional embeddings** — surprisingly, 2D-aware embeddings don't help much. The model learns spatial structure from data. When you visualize them:

```
Learned position embedding similarity matrix:

  The embeddings for spatially nearby patches
  are similar → the model discovers 2D layout
  from 1D positions automatically!
```

---

## Implementation (60 min)

### Build ViT from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbedding(nn.Module):
    """Convert image to sequence of patch embeddings."""
    
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        # Equivalent to Conv2d with kernel_size=stride=patch_size
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # x: (B, C, H, W) → (B, embed_dim, H/P, W/P)
        x = self.projection(x)
        # (B, embed_dim, H/P, W/P) → (B, n_patches, embed_dim)
        x = rearrange(x, 'b e h w -> b (h w) e')
        return x


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention (reused from Phase II)."""
    
    def __init__(self, embed_dim=768, n_heads=12, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, n_heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    """Vision Transformer (ViT-Base/16 configuration)."""
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        n_classes=1000,
        embed_dim=768,
        depth=12,
        n_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # Learnable [CLS] token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer encoder
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, n_classes)
        
        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, n_patches + 1, embed_dim)
        
        # Add position embeddings
        x = self.pos_drop(x + self.pos_embed)
        
        # Transformer encoder
        x = self.blocks(x)
        x = self.norm(x)
        
        # Classify from [CLS] token
        return self.head(x[:, 0])


# Test
model = ViT(img_size=224, patch_size=16, n_classes=10, embed_dim=384, depth=6, n_heads=6)
dummy = torch.randn(2, 3, 224, 224)
out = model(dummy)
print(f"Output shape: {out.shape}")  # (2, 10)
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
```

### Visualize Position Embeddings

```python
import matplotlib.pyplot as plt

def visualize_pos_embeddings(model):
    """Show that learned 1D pos embeddings discover 2D structure."""
    pos = model.pos_embed[0, 1:].detach()  # skip CLS
    n = int(pos.shape[0] ** 0.5)
    
    # Cosine similarity between all position pairs
    sim = F.cosine_similarity(pos.unsqueeze(0), pos.unsqueeze(1), dim=-1)
    
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        idx = i * (n * n // 16)
        if idx < n * n:
            ax.imshow(sim[idx].reshape(n, n).numpy(), cmap='viridis')
            ax.set_title(f'Patch {idx}')
        ax.axis('off')
    plt.suptitle('Position Embedding Similarity (learns 2D structure!)')
    plt.tight_layout()
    plt.savefig('pos_embed_similarity.png', dpi=150)
```

---

## Exercise (45 min)

1. **Parameter counting:** For ViT-Base/16 (embed_dim=768, depth=12, n_heads=12, img_size=224), calculate the total parameter count by hand. Verify with code.

2. **Patch size ablation:** Modify the ViT to use patch_size=8 (instead of 16). How does this change:
   - Number of tokens?
   - Computational cost ($O(n^2)$ for attention)?
   - What does this mean for high-resolution images?

3. **No [CLS] token:** Replace the [CLS]-based classification with **global average pooling** over all patch tokens. Compare the two approaches — which is more common in modern ViTs?

---

## Key Takeaways

1. **Images are sequences.** ViT treats images as 16×16 patch sequences — no convolutions needed
2. **Same transformer.** The encoder is identical to BERT/GPT — attention, FFN, LayerNorm
3. **Position matters.** Learned 1D embeddings spontaneously discover 2D spatial structure
4. **Data hunger.** ViT needs large-scale pretraining (JFT-300M); DeiT (tomorrow) fixes this
5. **[CLS] as summary.** A special token aggregates global information, like BERT

## Connection to the Thread

> *You've now applied the same self-attention mechanism to text tokens (Phase II) and image patch tokens (Phase IV). The transformer is a universal sequence processor. Next: making it train efficiently on ImageNet-scale data.*

---

## Further Reading

- [An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929)
- [How to Train Your ViT (Steiner et al., 2021)](https://arxiv.org/abs/2106.10270)
- Phil Wang's `vit-pytorch` implementation: github.com/lucidrains/vit-pytorch
- Lilian Weng's blog: "Generalized Visual Language Models"
