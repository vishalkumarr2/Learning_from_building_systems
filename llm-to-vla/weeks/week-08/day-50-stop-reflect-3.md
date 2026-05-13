# Day 50: Stop & Reflect #3 — Images ARE Sequences

> **Phase IV — Vision: ViT, 3D, Video | Week 8 | 2.5 hours**
> *"The transformer doesn't care what tokens are — text, patches, audio frames, point clouds. It just processes sequences." — The universal lesson*

## Navigation
- **Previous:** [Day 49: MAE](../week-07/day-49-mae.md)
- **Next:** [Day 51: 3D Vision & Depth](day-51-3d-vision-depth.md)
- **Week:** [Week 8 Overview](README.md)
- **Phase:** [Phase IV: Vision](../../study-notes/08-vision-transformers.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## The Big Picture (30 min)

### Universal Tokenization

The most important insight from Phase IV so far is that **the transformer architecture is modality-agnostic**. The same self-attention mechanism that processes words also processes image patches:

```
Text (Phase II):     "The cat sat" → [tok₁, tok₂, tok₃] → Transformer → prediction
                      ↓
Images (Phase IV):   [photo]       → [patch₁, ..., patch₁₉₆] → Transformer → classification
                      ↓
Audio:               [waveform]    → [frame₁, ..., frame_N] → Transformer → transcription
                      ↓
Video:               [clip]        → [frame₁×patch₁, ..., frame_T×patch_N] → Transformer → understanding
                      ↓
3D:                  [point cloud] → [point₁, ..., point_K] → Transformer → segmentation
                      ↓
Actions (Phase VI):  [trajectory]  → [action₁, ..., action_T] → Transformer → control
```

The transformer is a **universal sequence processor**. The innovation is always in **how you tokenize the input**, not in the architecture itself.

### Reflection Questions

Spend 15 minutes writing answers (without looking at notes):

1. **Explain ViT to a colleague in 3 sentences.** Include: patch embedding, position embedding, [CLS] token.

2. **Why does Swin Transformer exist?** What problem does it solve that ViT doesn't?

3. **DINO vs MAE:** Both are self-supervised. What's the key difference in their learning signal? When would you choose one over the other?

4. **The masking insight:** Why can MAE mask 75% of patches (vs BERT's 15%)? What does this tell you about visual vs linguistic redundancy?

5. **Position embeddings:** ViT uses learned 1D positions. Swin uses relative position bias. Why does the 1D approach work at all for a 2D image?

---

## Connections Map (30 min)

### Draw the Architecture Family Tree

```
Self-Attention (Day 9)
  │
  ├── Encoder-only: BERT (Day 21) ──── masked prediction
  │                                         │
  │                          ┌───────────────┤
  │                          │               │
  │                     MAE (Day 49)    BEiT (visual tokens)
  │
  ├── Decoder-only: GPT (Day 23) ──── autoregressive prediction
  │
  ├── Enc-Dec: T5 (Day 28) ──── seq2seq
  │
  └── Vision:
       ├── ViT (Day 45) ──── global attention, [CLS] classification
       │    │
       │    ├── DeiT (Day 46) ──── training recipe, distillation token
       │    │
       │    └── DINO (Day 48) ──── self-distillation, emergent segmentation
       │
       └── Swin (Day 47) ──── windowed attention, hierarchical features
```

### Key Parallels

| NLP Concept | Vision Equivalent | Why It Works |
|-------------|-------------------|--------------|
| Word token | Image patch | Both are discrete input units |
| Position embedding | Position embedding / relative bias | Sequence needs order information |
| BERT masking | MAE masking | Predict hidden parts → learn semantics |
| Knowledge distillation | DINO self-distillation | Soft targets carry richer signal |
| Sentence → [CLS] | Image → [CLS] | Special token for global summary |

---

## Hands-on Review (60 min)

### Quick Implementation Check

Without looking at code, implement these from memory, then verify:

```python
# 1. Patch embedding: image → sequence of patches
def patch_embed(image, patch_size=16):
    """Convert (B, C, H, W) image to (B, N, D) patch sequence."""
    B, C, H, W = image.shape
    P = patch_size
    N = (H // P) * (W // P)
    
    # Reshape into patches
    patches = image.unfold(2, P, P).unfold(3, P, P)  # (B, C, H/P, W/P, P, P)
    patches = patches.contiguous().view(B, C, N, P * P)
    patches = patches.permute(0, 2, 1, 3).reshape(B, N, C * P * P)
    
    return patches  # (B, N, patch_dim)

# 2. Random masking for MAE
def random_mask(x, mask_ratio=0.75):
    """Mask patches randomly, return visible subset."""
    B, N, D = x.shape
    keep = int(N * (1 - mask_ratio))
    
    noise = torch.rand(B, N, device=x.device)
    ids = noise.argsort(dim=1)
    ids_keep = ids[:, :keep]
    
    visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
    return visible, ids

# 3. Windowed attention partition
def window_partition(x, window_size=7):
    """(B, H, W, C) → (B*nW, M, M, C)"""
    B, H, W, C = x.shape
    M = window_size
    x = x.view(B, H//M, M, W//M, M, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, M, M, C)
```

### Feature Comparison Experiment

```python
# Compare representations: supervised ViT vs DINO vs MAE
import timm
import torch

# Load three models with different pretraining
models = {
    'supervised': timm.create_model('vit_small_patch16_224', pretrained=True),
    'dino': torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14'),
    # MAE requires separate loading
}

# Extract features on same images → compare with CKA or linear probe
# Which features are most transferable?
```

---

## Looking Ahead (15 min)

The rest of Phase IV extends vision beyond 2D images:

| Remaining Days | What's New | Why It Matters for VLAs |
|----------------|-----------|------------------------|
| Day 51: 3D & Depth | Monocular depth estimation | Robots perceive 3D world |
| Day 52: Point Clouds | PointNet, 3D transformers | Manipulation needs 3D reasoning |
| Day 53-54: Video | Temporal attention | Actions unfold over time |
| Day 55: DETR + SAM | Object detection + segmentation | Robots must identify objects |
| Day 56: VL Bridge | How to connect vision→language | The foundation for VLMs/VLAs |

> *You've mastered image understanding with transformers. Now: the real world is 3D, dynamic, and requires detecting individual objects — not just classifying scenes.*

---

## Key Takeaways

1. **Tokenization is the innovation.** The transformer stays the same; how you create tokens changes
2. **Self-supervised works.** DINO and MAE prove that vision features rival supervised ones
3. **Efficiency matters.** Swin's $O(n)$ windows and MAE's 75% masking make large-scale training feasible
4. **The thread is continuous.** Attention → text transformers → image transformers → soon: action transformers

## Connection to the Thread

> *Phase IV has shown that images are just another token type. The rest of the phase extends this to depth, 3D, and video. Then Phase V connects vision to language — the direct precursor to VLAs.*

---

## Further Reading

- Review: [A Survey on Vision Transformer (Han et al., 2022)](https://arxiv.org/abs/2012.12556)
- Yannic Kilcher's ViT video explanation
- Jay Alammar's "The Illustrated ViT"
