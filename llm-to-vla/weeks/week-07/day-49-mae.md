# Day 49: MAE — Masked Autoencoders

> **Phase IV — Vision: ViT, 3D, Video | Week 7 | 2.5 hours**
> *"Mask 75% of patches, reconstruct pixels. Simple, scalable, and it works spectacularly." — He et al., 2022*

## Navigation
- **Previous:** [Day 48: DINO & Self-Supervised Vision](day-48-dino-self-supervised.md)
- **Next:** [Day 50: Stop & Reflect #3](../week-08/day-50-stop-reflect-3.md)
- **Week:** [Week 7 Overview](README.md)
- **Phase:** [Phase IV: Vision](../../study-notes/08-vision-transformers.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### MAE = Visual BERT

BERT masks 15% of tokens and predicts them. MAE masks **75%** of image patches and reconstructs pixels. Why so aggressive? Because images have much higher spatial redundancy than text — a missing patch can often be inferred from its neighbors.

```
Original image:                After 75% masking:
┌──┬──┬──┬──┐                 ┌──┬──┬──┬──┐
│P1│P2│P3│P4│                 │  │P2│  │  │
├──┼──┼──┼──┤                 ├──┼──┼──┼──┤
│P5│P6│P7│P8│    mask 75%     │  │  │P7│  │
├──┼──┼──┼──┤  ──────────►    ├──┼──┼──┼──┤
│P9│10│11│12│                 │P9│  │  │12│
├──┼──┼──┼──┤                 ├──┼──┼──┼──┤
│13│14│15│16│                 │  │  │  │  │
└──┴──┴──┴──┘                 └──┴──┴──┴──┘

Encoder: processes ONLY visible patches (25%)
Decoder: reconstructs ALL patches from encoded visible + mask tokens
```

### Architecture: Asymmetric Encoder-Decoder

```
┌─────────────────────────────────────────────────────────────┐
│                         MAE Pipeline                         │
│                                                              │
│  1. Patch + embed all patches                               │
│  2. Random mask 75% → keep 25% visible                     │
│                                                              │
│  3. Encoder (large ViT):                                    │
│     - Processes ONLY visible patches (25%)                  │
│     - Full self-attention but on reduced sequence            │
│     - This is where 75% compute savings come from!          │
│                                                              │
│  4. Insert mask tokens at masked positions                   │
│     - Mask token = shared learnable vector                   │
│     - Add positional embeddings to all tokens               │
│                                                              │
│  5. Decoder (small transformer):                            │
│     - Processes full set (visible encoded + mask tokens)    │
│     - Only used during pretraining → can be lightweight     │
│                                                              │
│  6. Loss: MSE on masked patches only                        │
│     $\mathcal{L} = \frac{1}{|M|} \sum_{i \in M} \|         │
│      \hat{x}_i - x_i \|^2$                                  │
│     (optionally: per-patch normalized pixels)                │
└─────────────────────────────────────────────────────────────┘
```

### Why 75% Masking Works

| Masking ratio | Training speed | Quality |
|---------------|---------------|---------|
| 15% (BERT-like) | Slow (process 85% of patches) | Lower — too easy, neighbor copying |
| 50% | Medium | Good |
| **75%** | **Fast (only 25% through encoder)** | **Best** — forces semantic understanding |
| 90% | Very fast | Degrades — too hard |

At 75%, the model can't just copy neighbors — it must understand **what** the object is to fill in distant patches. This forces learning of high-level semantics.

### Connection to BERT and Compression

The same principle underlies both:
- **BERT:** Mask words → model learns language semantics by predicting them
- **MAE:** Mask patches → model learns visual semantics by reconstructing them
- **Key insight:** What you can predict from partial information reveals what you **understand**

### MAE vs DINO

| Property | MAE | DINO |
|----------|-----|------|
| Task | Pixel reconstruction | Feature matching |
| Supervision | Self (reconstruct masked) | Self (student-teacher) |
| What's learned | Low-level + high-level | Primarily high-level |
| Best for | Fine-tuning → classification | Zero-shot / k-NN transfer |
| Training speed | 3× faster (75% mask) | Slower (EMA, multi-crop) |

---

## Implementation (60 min)

### Build MAE from Scratch

```python
import torch
import torch.nn as nn
from einops import rearrange, repeat


class MAE(nn.Module):
    """Masked Autoencoder for self-supervised ViT pretraining."""
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        encoder_depth=12,
        encoder_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_heads=16,
        mask_ratio=0.75,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        n_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size ** 2
        
        # --- Encoder ---
        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        
        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, encoder_heads)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # --- Decoder ---
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, n_patches + 1, decoder_embed_dim)
        )
        
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_heads)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_dim)
        
        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
    
    def patchify(self, imgs):
        """Convert images to patch sequences."""
        p = self.patch_size
        x = rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        return x
    
    def random_masking(self, x, mask_ratio):
        """Random masking: keep (1-mask_ratio) patches, mask the rest."""
        B, N, D = x.shape
        keep = int(N * (1 - mask_ratio))
        
        # Random permutation per sample
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        
        # Keep first `keep` indices
        ids_keep = ids_shuffle[:, :keep]
        x_visible = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        # Binary mask: 0 = keep, 1 = masked
        mask = torch.ones(B, N, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_visible, mask, ids_restore
    
    def forward_encoder(self, x):
        # Patchify and embed
        patches = self.patchify(x)
        x = self.patch_embed(patches)
        x = x + self.pos_embed[:, 1:]  # no CLS position yet
        
        # Mask
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        
        # Prepend CLS
        cls = self.cls_token + self.pos_embed[:, :1]
        cls = cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        
        # Encode (only visible patches!)
        for block in self.encoder_blocks:
            x = block(x)
        x = self.encoder_norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        # Project to decoder dimension
        x = self.decoder_embed(x)
        
        # Insert mask tokens
        B = x.shape[0]
        mask_tokens = self.mask_token.repeat(
            B, ids_restore.shape[1] + 1 - x.shape[1], 1
        )
        
        # Unshuffle: put visible and mask tokens back in original order
        x_no_cls = torch.cat([x[:, 1:], mask_tokens], dim=1)
        x_no_cls = torch.gather(
            x_no_cls, dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2])
        )
        x = torch.cat([x[:, :1], x_no_cls], dim=1)  # re-add CLS
        
        # Add decoder positional embeddings
        x = x + self.decoder_pos_embed
        
        # Decode
        for block in self.decoder_blocks:
            x = block(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        
        return x[:, 1:]  # remove CLS
    
    def forward(self, imgs):
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        target = self.patchify(imgs)
        
        # MSE loss on masked patches only
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # per-patch MSE
        loss = (loss * mask).sum() / mask.sum()  # masked patches only
        
        return loss, pred, mask
```

---

## Exercise (45 min)

1. **Masking ratio experiment:** Train a small MAE (embed_dim=256, depth=4) on CIFAR-10 with mask ratios of 25%, 50%, 75%, and 90%. Compare reconstruction quality and downstream linear probe accuracy.

2. **Reconstruction visualization:** Train MAE for 50 epochs on a small dataset. Visualize: original → masked → reconstructed for 8 images. How well does the model fill in missing patches?

3. **MAE vs DINO downstream:** Compare MAE and DINOv2 features on the same downstream task (k-NN on CIFAR-10 or linear probe). Which produces better features for transfer? Why?

---

## Key Takeaways

1. **75% masking.** Most of the image is thrown away — forcing semantic understanding
2. **Asymmetric design.** Large encoder on 25% of patches; small decoder for reconstruction
3. **3× speedup.** Encoder only processes visible patches — enormous compute savings
4. **Pixel reconstruction.** MSE on raw pixels works — no contrastive loss or negative samples
5. **Visual BERT.** The masked prediction paradigm unifies language and vision pretraining

## Connection to the Thread

> *MAE completes the self-supervised vision picture. You've now seen two paradigms: matching-based (DINO) and reconstruction-based (MAE). Both produce features without labels. Next week: extending vision to 3D, depth, and video.*

---

## Further Reading

- [Masked Autoencoders Are Scalable Vision Learners (He et al., 2022)](https://arxiv.org/abs/2111.06377)
- [BEiT: BERT Pre-Training of Image Transformers (Bao et al., 2021)](https://arxiv.org/abs/2106.08254)
- [VideoMAE (Tong et al., 2022)](https://arxiv.org/abs/2203.12602)
- Official MAE: github.com/facebookresearch/mae
