# 08 — Vision Transformers

> Phase IV · Days 45–50 · ~15 hours
> Prerequisites: 03-transformer-architecture, 04-transformer-variants
> Learning Objectives: Understand how transformers process images, self-supervised vision, why this matters for robot perception

---

## The Big Idea

Everything we learned about transformers for text? It works for images too.

An image is just a sequence of patches, exactly like a sentence is a sequence of tokens.
The same attention mechanism that learns "the cat sat on the mat" can learn
"this patch of fur is next to this patch of whiskers."

This is not an analogy. It is literally the same architecture.

---

## 1. ViT — An Image is Worth 16×16 Words (Day 45)

### 1.1 The Core Insight

In 2020, Dosovitskiy et al. asked a simple question:
**What if we just treat an image as a sequence of tokens and feed it to a standard transformer?**

No convolutions. No pooling. No CNN inductive biases.
Just patches → tokens → transformer → classification.

It worked. Spectacularly.

### 1.2 Patch Embedding: Image → Token Sequence

The key operation that makes ViT possible:

```
Input Image: H × W × C  (e.g., 224 × 224 × 3)
                ↓
Split into patches: N patches of size P × P × C
  - P = 16 (the "16×16 words" from the paper title)
  - N = (H/P) × (W/P) = (224/16) × (224/16) = 14 × 14 = 196 patches
                ↓
Flatten each patch: P × P × C = 16 × 16 × 3 = 768 values per patch
                ↓
Linear projection: 768 → D (model dimension, e.g., 768)
                ↓
Result: sequence of 196 token embeddings, each of dimension D
```

This is directly analogous to:
- **Text**: word → lookup in embedding table → vector of dimension D
- **Vision**: patch → flatten → linear projection → vector of dimension D

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """Convert image into a sequence of patch embeddings."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2  # 196 for 224/16

        # Single conv layer does both: extract patch + project to embed_dim
        # Equivalent to: flatten patch → linear projection
        # But more efficient as a strided convolution
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size  # non-overlapping patches
        )

    def forward(self, x):
        # x: (batch, channels, height, width) = (B, 3, 224, 224)
        x = self.projection(x)      # (B, embed_dim, 14, 14)
        x = x.flatten(2)            # (B, embed_dim, 196)
        x = x.transpose(1, 2)       # (B, 196, embed_dim) — sequence of tokens!
        return x
```

**Why Conv2d instead of manual flatten + Linear?**
They're mathematically equivalent, but Conv2d is faster on GPU because
it avoids the explicit reshape operations.

### 1.3 Position Embeddings

Patches lose their spatial information when flattened into a sequence.
We need position embeddings, just like in text transformers.

```python
class ViTEmbedding(nn.Module):
    """Patch embedding + [CLS] token + position embeddings."""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = (img_size // patch_size) ** 2

        # [CLS] token — learnable, prepended to the sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Position embeddings: one for [CLS] + one for each patch
        # Learned (not sinusoidal) — works better for images
        self.position_embedding = nn.Parameter(
            torch.randn(1, n_patches + 1, embed_dim)
        )

    def forward(self, x):
        B = x.shape[0]
        patches = self.patch_embed(x)                          # (B, 196, 768)
        cls_tokens = self.cls_token.expand(B, -1, -1)          # (B, 1, 768)
        x = torch.cat([cls_tokens, patches], dim=1)            # (B, 197, 768)
        x = x + self.position_embedding                        # Add position info
        return x
```

**Key difference from text transformers:**
- Text uses 1D positions (token 0, 1, 2, ...)
- ViT also uses 1D positions (patch 0, 1, 2, ...) — even though patches have 2D spatial structure
- The model LEARNS the 2D structure from the 1D positions
- 2D-aware position embeddings exist (e.g., in Swin) but aren't necessary

### 1.4 The [CLS] Token

Borrowed directly from BERT:

```
Input sequence:  [CLS]  patch_1  patch_2  ...  patch_196
                   ↓       ↓        ↓             ↓
Transformer:     attend to ALL patches via self-attention
                   ↓
Output:         [CLS]_out  (aggregated representation of entire image)
                   ↓
Classification:  Linear(embed_dim → num_classes)
```

The [CLS] token has no image content — it's purely a learnable aggregation token.
Through self-attention, it learns to gather information from all patches.

### 1.5 Complete ViT Architecture

```
Image (224 × 224 × 3)
        ↓
Patch Embedding (196 patches × 768)
        ↓
Prepend [CLS] token → 197 × 768
        ↓
Add Position Embeddings
        ↓
Transformer Encoder × L layers
  (same as BERT: multi-head self-attention + FFN + LayerNorm)
        ↓
Take [CLS] output → 768
        ↓
Classification Head (Linear 768 → num_classes)
        ↓
Class prediction
```

ViT model sizes (from the paper):

| Model    | Layers | Hidden | Heads | Params |
|----------|--------|--------|-------|--------|
| ViT-Base | 12     | 768    | 12    | 86M    |
| ViT-Large| 24     | 1024   | 16    | 307M   |
| ViT-Huge | 32     | 1280   | 16    | 632M   |

### 1.6 ViT vs CNN: What Changed

| Aspect            | CNN (ResNet)                        | ViT                                  |
|-------------------|-------------------------------------|--------------------------------------|
| Local structure   | Built-in (conv kernel = local)      | Learned via attention                |
| Global context    | Only at deep layers                 | From layer 1 (attention is global)   |
| Translation equiv.| Built-in (weight sharing)           | Learned from data + position embed   |
| Data efficiency   | Better with small data              | Needs large data (or good pretrain)  |
| Scalability       | Saturates with more data            | Keeps improving with more data       |
| Inductive bias    | Strong (locality, translation)      | Weak (must learn from data)          |

**The key finding from the paper:**
- With small data (ImageNet-1K only): CNN wins
- With large data (JFT-300M, ImageNet-21K): ViT wins, and keeps scaling
- ViT + large pretraining → THEN fine-tune on small dataset = best of both worlds

### 1.7 Why Patches Are Visual "Tokens"

This deserves emphasis because it's the conceptual bridge:

```
Text:    "The  robot  picked  up  the  box"
          ↓      ↓       ↓     ↓    ↓    ↓
Tokens:  [t1]  [t2]    [t3]  [t4] [t5] [t6]

Image:   [patch_00] [patch_01] ... [patch_13]
         [patch_14] [patch_15] ... [patch_27]
          ...
         [patch_182] ... [patch_195]
          ↓           ↓              ↓
Tokens:  [t1]        [t2]    ...   [t196]
```

Both are sequences. Both go through the SAME transformer architecture.
The attention mechanism doesn't "know" if it's processing text or image patches.
It just processes a sequence of vectors.

---

## 2. ViT Variants & Swin Transformer (Day 46)

### 2.1 DeiT — Data-Efficient Image Transformers

The problem with vanilla ViT: needs massive pretraining data (JFT-300M).

DeiT (Touvron et al., 2021) fixes this with:

1. **Better training recipe**: strong augmentation, regularization, longer training
2. **Distillation token**: learn from a CNN teacher
3. **Result**: ViT-quality on ImageNet-1K alone (no JFT pretraining needed)

```
Standard ViT:  [CLS]  patch_1  patch_2  ...  patch_N
                 ↓
DeiT:          [CLS]  [DIST]  patch_1  patch_2  ...  patch_N
                 ↓      ↓
              class   distill
              loss     loss (match CNN teacher predictions)
```

The distillation token learns from a CNN teacher, giving the transformer
CNN-like inductive biases without hard-coding them into the architecture.

### 2.2 The Quadratic Problem

Recall from study note 03: self-attention is O(N²) in sequence length.

For ViT with 224×224 images and 16×16 patches:
- N = 196 patches → 196² = 38,416 attention entries. Fine.

But what about higher resolution or smaller patches?
- 384×384 with 16×16: N = 576 → 331,776 entries
- 512×512 with 16×16: N = 1,024 → 1,048,576 entries
- 1024×1024 with 16×16: N = 4,096 → 16.7M entries 🔥

Dense prediction tasks (detection, segmentation) need high-resolution features.
Global self-attention at high resolution is prohibitive.

### 2.3 Swin Transformer: Shifted Windows

Swin Transformer (Liu et al., 2021) solves this with two ideas:

**Idea 1: Window Attention**
Instead of global attention over all patches, compute attention within
local windows (e.g., 7×7 patches per window).

```
Global attention (ViT):        Window attention (Swin):
┌─────────────────────┐        ┌─────┬─────┬─────┐
│ every patch attends  │        │win 1│win 2│win 3│
│ to every other patch │        ├─────┼─────┼─────┤
│ O(N²) where N=H×W   │        │win 4│win 5│win 6│
│                      │        ├─────┼─────┼─────┤
│                      │        │win 7│win 8│win 9│
└─────────────────────┘        └─────┴─────┴─────┘
                                Each window: O(M²) where M=49
                                Total: O(N × M)  ← LINEAR in N!
```

**Idea 2: Shifted Windows**
Problem: window attention has no cross-window communication.
Solution: shift the window partition between consecutive layers.

```
Layer L (regular windows):     Layer L+1 (shifted windows):
┌─────┬─────┬─────┐           ┌──┬──────┬──────┬──┐
│  1  │  2  │  3  │           │  │      │      │  │
├─────┼─────┼─────┤    →      ├──┼──────┼──────┼──┤
│  4  │  5  │  6  │           │  │      │      │  │
├─────┼─────┼─────┤           ├──┼──────┼──────┼──┤
│  7  │  8  │  9  │           │  │      │      │  │
└─────┴─────┴─────┘           └──┴──────┴──────┴──┘

Shift by (M/2, M/2) pixels → windows now straddle old boundaries
→ information flows across the original window boundaries
```

**Idea 3: Hierarchical Features**
Swin builds a feature pyramid, like CNNs do:

```
Stage 1: 56×56 patches, dim=96    (1/4 resolution)
    ↓ patch merging (2×2 → 1, doubles channels)
Stage 2: 28×28 patches, dim=192   (1/8 resolution)
    ↓ patch merging
Stage 3: 14×14 patches, dim=384   (1/16 resolution)
    ↓ patch merging
Stage 4: 7×7 patches, dim=768     (1/32 resolution)
```

This gives multi-scale features, essential for:
- Object detection (objects at different sizes)
- Segmentation (need fine + coarse features)
- Robot perception (near objects = large, far objects = small)

### 2.4 Window vs Global Attention — Summary

| Property            | ViT (Global)          | Swin (Window)                |
|---------------------|-----------------------|------------------------------|
| Attention scope     | All patches           | Within 7×7 windows           |
| Complexity          | O(N²)                 | O(N × M), M=49               |
| High-res support    | Poor (quadratic)      | Excellent (linear)            |
| Cross-patch comms   | Always                | Via shifted windows           |
| Feature hierarchy   | Single scale          | Multi-scale pyramid           |
| Best for            | Classification        | Detection, segmentation       |
| Robot relevance     | Feature extraction    | Dense perception tasks        |

### 2.5 Why Swin Matters for Robotics

Robots need **dense prediction**, not just classification:
- "What's in the scene?" → classification (ViT is fine)
- "WHERE is each object?" → detection (need multi-scale features)
- "Which pixels are floor vs obstacle?" → segmentation (need high-res features)
- "How far is each point?" → depth estimation (need spatial features)

Swin gives you the multi-scale, high-resolution features that dense tasks demand,
while keeping the transformer's ability to model long-range relationships.

---

## 3. DINO — Self-Supervised Vision (Day 47)

### 3.1 The Label Problem

ImageNet has 1.28M labeled images across 1,000 classes.
Creating labels is expensive, slow, and limited.

The internet has **billions** of unlabeled images.
Can we learn visual representations WITHOUT labels?

DINO (Caron et al., 2021): **Yes, and the representations are better.**

### 3.2 Self-Distillation Framework

DINO = "**Di**stillation with **No** labels"

```
                    Image x
                   /       \
            augment_1     augment_2
               ↓              ↓
         Student ViT      Teacher ViT
         (updated by      (updated by EMA
          gradient)        of student weights)
               ↓              ↓
           student_out    teacher_out
               ↓              ↓
            softmax(τ_s)  softmax(τ_t)   ← different temperatures!
               ↓              ↓
           ┌───────────────────┐
           │  Cross-entropy    │
           │  loss: match      │
           │  student to       │
           │  teacher          │
           └───────────────────┘
```

**Key components:**

1. **Two networks**: student and teacher (both ViTs, same architecture)
2. **Different augmentations**: student sees local crops, teacher sees global crops
3. **Teacher = EMA of student**: `θ_teacher ← m * θ_teacher + (1-m) * θ_student`
   - m starts at 0.996, increases to 1.0 during training
4. **Centering + sharpening**: prevents mode collapse (all outputs becoming the same)

```python
# Simplified DINO training loop (pseudocode)
def dino_loss(student_output, teacher_output, center, tau_s=0.1, tau_t=0.04):
    """Cross-entropy between student and teacher softmax outputs."""
    student_probs = F.softmax(student_output / tau_s, dim=-1)
    teacher_probs = F.softmax((teacher_output - center) / tau_t, dim=-1)
    loss = -torch.sum(teacher_probs * torch.log(student_probs), dim=-1)
    return loss.mean()

# Training step
for images in dataloader:
    # Create multiple augmented views
    global_views = [augment_global(images) for _ in range(2)]
    local_views = [augment_local(images) for _ in range(6)]

    # Teacher only sees global views
    teacher_output = [teacher(v) for v in global_views]

    # Student sees all views
    student_output = [student(v) for v in global_views + local_views]

    # Loss: student must match teacher on ALL view combinations
    loss = compute_cross_view_loss(student_output, teacher_output)

    # Update student by gradient
    loss.backward()
    optimizer.step()

    # Update teacher by EMA (no gradients!)
    with torch.no_grad():
        for p_t, p_s in zip(teacher.parameters(), student.parameters()):
            p_t.data = momentum * p_t.data + (1 - momentum) * p_s.data
```

### 3.3 Why DINO Works: Multi-Crop Strategy

The student sees **local** crops (small patches of the image).
The teacher sees **global** crops (most of the image).

The student must predict the teacher's output from only a small patch.
This forces it to learn: "what does a small region tell me about the whole image?"

```
Original image:       Teacher sees:         Student sees:
┌───────────────┐    ┌───────────────┐    ┌──────┐
│               │    │ ████████████ │    │ ████ │  (small crop)
│    🤖 robot   │    │ ████████████ │    │ ████ │
│    on floor   │    │ ████████████ │    └──────┘
│    in aisle   │    │              │
│               │    └───────────────┘
└───────────────┘

Student must infer "robot in aisle" from just seeing "part of robot"
→ Forces learning of semantic understanding
```

### 3.4 Emergent Object Segmentation

The most striking result: **DINO attention maps automatically segment objects.**

Nobody trained it to segment. Nobody gave it segmentation labels.
The attention heads in the last layer naturally focus on semantic objects.

```python
# Visualizing DINO attention maps
import torch
from torchvision import transforms
from PIL import Image

def visualize_dino_attention(model, image_path, patch_size=14):
    """Extract and visualize self-attention from last layer."""
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize(518),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(Image.open(image_path)).unsqueeze(0)

    # Get attention from last layer
    with torch.no_grad():
        attentions = model.get_last_selfattention(img)

    # attentions shape: (1, n_heads, N+1, N+1)
    # Take attention from [CLS] token to all patches
    nh = attentions.shape[1]  # number of heads
    cls_attention = attentions[0, :, 0, 1:]  # (n_heads, N_patches)

    # Reshape to spatial grid
    w_feat = h_feat = int(cls_attention.shape[-1] ** 0.5)
    attention_maps = cls_attention.reshape(nh, h_feat, w_feat)

    return attention_maps  # Each head segments different things!
```

**What different attention heads learn:**
- Head 3: focuses on the main object
- Head 5: focuses on the background
- Head 8: focuses on edges/boundaries
- Combined: clean semantic segmentation, zero labels!

### 3.5 DINOv2: The Universal Visual Feature Extractor

DINOv2 (Oquab et al., 2023) scales DINO with:
- Larger models (ViT-g with 1.1B parameters)
- More data (142M curated images, LVD-142M dataset)
- Better training (iBOT + DINO objectives combined)
- Frozen features work: no fine-tuning needed for many tasks

**DINOv2 as a universal feature extractor:**

```python
import torch
from transformers import AutoModel, AutoImageProcessor

# Load DINOv2
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")

def extract_features(image):
    """Extract universal visual features — works for ANYTHING."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    # CLS token: global image representation (768-dim)
    cls_features = outputs.last_hidden_state[:, 0]
    # Patch tokens: spatial features (N × 768)
    patch_features = outputs.last_hidden_state[:, 1:]
    return cls_features, patch_features

# These features work for:
# - Classification (linear probe on CLS)
# - Detection (use patch features)
# - Segmentation (upsample patch features)
# - Depth estimation (patch features → depth decoder)
# - Retrieval (CLS similarity)
# ALL without fine-tuning the backbone!
```

### 3.6 Why DINO Matters for Robotics

1. **No labels needed**: robots encounter novel objects constantly
2. **Universal features**: one backbone for detection, grasping, navigation
3. **Emergent segmentation**: object understanding without segmentation training
4. **Transfer**: features learned on internet images work on robot camera images
5. **Efficiency**: extract once, use for multiple downstream tasks

---

## 4. MAE — Masked Autoencoders (Day 48)

### 4.1 The Idea: BERT for Images

BERT masks 15% of tokens, predicts them from context → learns language.

MAE (He et al., 2022) masks **75%** of patches, reconstructs them → learns vision.

```
Original:           Masked (75%!):        Reconstructed:
┌──┬──┬──┬──┐      ┌──┬──┬──┬──┐        ┌──┬──┬──┬──┐
│▓▓│▓▓│▓▓│▓▓│      │  │▓▓│  │  │        │▓▓│▓▓│▓▓│▓▓│
├──┼──┼──┼──┤      ├──┼──┼──┼──┤        ├──┼──┼──┼──┤
│▓▓│▓▓│▓▓│▓▓│  →   │  │  │▓▓│  │   →    │▓▓│▓▓│▓▓│▓▓│
├──┼──┼──┼──┤      ├──┼──┼──┼──┤        ├──┼──┼──┼──┤
│▓▓│▓▓│▓▓│▓▓│      │▓▓│  │  │▓▓│        │▓▓│▓▓│▓▓│▓▓│
├──┼──┼──┼──┤      ├──┼──┼──┼──┤        ├──┼──┼──┼──┤
│▓▓│▓▓│▓▓│▓▓│      │  │▓▓│  │  │        │▓▓│▓▓│▓▓│▓▓│
└──┴──┴──┴──┘      └──┴──┴──┴──┘        └──┴──┴──┴──┘
16 patches          4 visible (25%)       All 16 reconstructed
```

### 4.2 Why 75%?

BERT masks 15% of tokens. MAE masks 75% of patches. Why the difference?

**Images are more redundant than text.**
- Missing one word in "The ___ sat on the mat" → many possibilities
- Missing one patch from a blue sky → easy, it's probably more blue sky

To make the task challenging enough to force learning,
you need to remove MOST of the image.

At 75% masking, the model can't just interpolate from neighbors —
it must understand the scene structure to reconstruct.

### 4.3 Asymmetric Encoder-Decoder

MAE's key efficiency insight: **only encode the visible patches.**

```
┌────────────────────────────────────────────────────────────┐
│                     MAE Architecture                        │
│                                                             │
│  Visible patches (25%)     Mask tokens (75%)                │
│       ↓                                                     │
│  ┌──────────────┐                                           │
│  │   Encoder    │  ← HEAVY: ViT-Large, processes only 25%  │
│  │  (ViT-L)    │     of patches → 4× speedup!              │
│  └──────────────┘                                           │
│       ↓                                                     │
│  Insert mask tokens at masked positions                     │
│       ↓                                                     │
│  ┌──────────────┐                                           │
│  │   Decoder    │  ← LIGHTWEIGHT: small transformer         │
│  │  (small)     │     processes all 100% of tokens          │
│  └──────────────┘                                           │
│       ↓                                                     │
│  Pixel reconstruction of masked patches                     │
│  Loss: MSE on masked patches only                           │
└────────────────────────────────────────────────────────────┘
```

**Training efficiency:**
- Encoder only processes 25% of patches (4× faster than full ViT)
- Decoder is lightweight (8 layers vs encoder's 24)
- Total: ~3× faster training than contrastive methods like DINO

```python
class MAE(nn.Module):
    """Simplified Masked Autoencoder."""

    def __init__(self, encoder, decoder, mask_ratio=0.75):
        super().__init__()
        self.encoder = encoder      # ViT-Large
        self.decoder = decoder      # Small ViT (8 layers)
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder.embed_dim))

    def random_masking(self, x, mask_ratio):
        """Random masking: keep (1-mask_ratio) of patches."""
        B, N, D = x.shape
        keep = int(N * (1 - mask_ratio))

        # Random permutation → take first 'keep' as visible
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep only visible patches
        ids_keep = ids_shuffle[:, :keep]
        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Binary mask: 0 = keep, 1 = masked
        mask = torch.ones(B, N, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return x_visible, mask, ids_restore

    def forward(self, x):
        # Encode only visible patches
        patches = self.encoder.patch_embed(x)
        patches, mask, ids_restore = self.random_masking(patches, self.mask_ratio)
        encoded = self.encoder(patches)       # Only 25% of patches!

        # Decode: insert mask tokens, reconstruct all patches
        mask_tokens = self.mask_token.expand(encoded.shape[0], ids_restore.shape[1] - encoded.shape[1], -1)
        full_sequence = torch.cat([encoded, mask_tokens], dim=1)
        full_sequence = torch.gather(full_sequence, 1, ids_restore.unsqueeze(-1).expand(-1, -1, full_sequence.shape[-1]))
        decoded = self.decoder(full_sequence)

        # Loss: MSE only on masked patches
        loss = F.mse_loss(decoded[mask == 1], target_pixels[mask == 1])
        return loss
```

### 4.4 What MAE Learns

By reconstructing masked patches, MAE learns:
- **Local texture**: what does this surface look like?
- **Object structure**: if I see wheels, there's probably a car body above
- **Spatial relationships**: objects follow consistent spatial patterns
- **Scene layout**: floors are below, ceilings above, walls on sides

```
Example: robot in warehouse aisle

Visible (25%):           What MAE must infer:
- Part of robot wheel    → "There's a robot here"
- Shelf edge             → "This is a warehouse aisle"
- Floor tile             → "Floor continues, shelves on sides"
- Ceiling light          → "Indoor, well-lit environment"

From 25% of pixels, MAE reconstructs the entire scene.
It must UNDERSTAND scene structure to do this.
```

### 4.5 MAE vs DINO: Different Self-Supervised Objectives

| Aspect           | DINO                              | MAE                                |
|------------------|-----------------------------------|------------------------------------|
| Objective        | Match teacher output              | Reconstruct masked patches         |
| Architecture     | Student-teacher                   | Encoder-decoder                    |
| Loss             | Cross-entropy on features         | MSE on pixels                      |
| Mask ratio       | N/A (uses crops instead)          | 75%                                |
| Training speed   | Slower (two forward passes)       | Faster (sparse encoding)           |
| Feature quality  | Better for detection/segmentation | Better for pretraining efficiency   |
| Best for         | Universal features, few-shot      | Efficient pretraining, fine-tuning  |

**In practice, both work well. DINOv2 combines ideas from both.**

---

## 5. 🛑 STOP AND REFLECT #3 (Day 50)

### The Thread So Far

```
Phase I:   Text = sequence of tokens
Phase II:  Attention compresses sequences by learning what to focus on
Phase III: Transformers scale attention to any sequence length
Phase IV:  Images ARE sequences of patches → same transformer works!
```

### The Core Insight

**An image is a sequence of patches, just like text is a sequence of tokens.**

The SAME attention mechanism processes both.

ViT proved it: take a standard transformer, feed it image patches, get
state-of-the-art image classification.

DINO proved something even more remarkable: you don't even need labels.
The transformer learns to understand images just by looking at them —
no human annotation required.

MAE showed that masking works for images too, just like it works for text.
The learning principle is universal: **predict the missing parts from context.**

### The Transformer Doesn't Care

```
Input type:        Tokenization:           Transformer sees:
─────────────────────────────────────────────────────────────
Text               word → embedding         Sequence of vectors
Image              patch → projection       Sequence of vectors
Audio              spectrogram → frames     Sequence of vectors
Video              spatial + temporal       Sequence of vectors
Point cloud        point → embedding        Sequence of vectors
Robot actions      action → embedding       Sequence of vectors
```

The transformer is a **universal sequence processor**.
It doesn't know or care what the vectors represent.
It just finds patterns in sequences.

### Why This Matters for Robots

A robot needs to:
1. **See** the world (image patches → visual tokens)
2. **Understand** language commands (words → text tokens)
3. **Act** in the world (actions → action tokens)

If all three are just sequences of tokens...
**can one transformer do all three?**

That's Phase VI (Vision-Language-Action models).
But we're building the foundation right now.

### Questions to Sit With

1. If an image is "just" 196 tokens, what information is lost in the patch embedding?
2. Why does DINO's self-supervised training produce BETTER features than supervised training for some tasks?
3. What happens when you feed video to a transformer — is it just a very long sequence of patches?
4. If the transformer doesn't need to know what kind of data it's processing, what does that imply about the nature of "understanding"?

---

## 6. Comparison Table

| Model  | Year | Supervision  | Key Innovation                    | Complexity | Best For                     |
|--------|------|-------------|-----------------------------------|------------|------------------------------|
| ViT    | 2020 | Supervised   | Image patches as tokens           | O(N²)      | Classification, features     |
| DeiT   | 2021 | Supervised   | Data-efficient ViT training       | O(N²)      | When limited data available  |
| Swin   | 2021 | Supervised   | Shifted windows, hierarchical     | O(N·M)     | Detection, segmentation      |
| DINO   | 2021 | Self-sup     | Self-distillation, no labels      | O(N²)      | Universal features, few-shot |
| MAE    | 2022 | Self-sup     | Masked reconstruction, 75%        | O(N²)*     | Efficient pretraining        |
| DINOv2 | 2023 | Self-sup     | Scaled DINO + iBOT               | O(N²)      | Universal frozen features    |

*MAE encoder processes only 25% of patches during training, so effective cost is much lower.

### Which to use?

```
Need classification only?           → ViT or DeiT
Need detection/segmentation?        → Swin Transformer
Need frozen features (no fine-tune)? → DINOv2
Need efficient pretraining?         → MAE then fine-tune
Need everything?                    → DINOv2 backbone + task-specific heads
```

---

## 7. Key Takeaways

### Technical
1. **ViT** splits images into patches and feeds them to a standard transformer — no convolutions needed
2. **Swin** introduces windowed attention for O(N) complexity and hierarchical features
3. **DINO** learns visual features through self-distillation without any labels
4. **MAE** masks 75% of patches and reconstructs them, forcing scene understanding
5. **DINOv2** combines ideas from DINO and MAE to create universal visual features

### Conceptual
1. The boundary between "language models" and "vision models" is dissolving
2. Self-supervised learning (DINO, MAE) often outperforms supervised learning
3. The transformer is a universal sequence processor — data modality doesn't matter
4. Patch embedding is the visual equivalent of tokenization

### For Robotics
1. Frozen DINOv2 features can power multiple perception tasks simultaneously
2. Self-supervised pretraining means robots can learn from unlabeled camera data
3. Swin-style hierarchical features are essential for spatial reasoning
4. The same backbone that processes images will eventually process actions too

---

## 8. Paper References

| Paper | Year | Key Contribution |
|-------|------|------------------|
| [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) | 2020 | ViT: patches as tokens |
| [Training data-efficient image transformers](https://arxiv.org/abs/2012.12877) | 2021 | DeiT: distillation for ViT |
| [Swin Transformer](https://arxiv.org/abs/2103.14030) | 2021 | Shifted windows, hierarchical |
| [Emerging Properties in Self-Supervised ViTs](https://arxiv.org/abs/2104.14294) | 2021 | DINO: self-distillation |
| [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) | 2022 | MAE: masked reconstruction |
| [DINOv2: Learning Robust Visual Features](https://arxiv.org/abs/2304.07193) | 2023 | Universal visual features |

---

## 9. Connection to the Thread

### The Compression Story Continues

```
Phase I:   "apple" → [2481]                    (word → token ID)
Phase II:  "The cat sat..." → attention weights (sequence → weighted focus)
Phase III: GPT compresses ALL of text into weights (training = compression)
Phase IV:  Image → 196 patch tokens → attention  (vision = same compression!)
```

**The insight:** image patches ARE tokens. The transformer compresses visual
information the same way it compresses textual information — by learning
which parts of the sequence are relevant to which other parts.

A patch of robot wheel is related to patches of robot body.
A word "picked" is related to words "up" and "box."
The attention mechanism captures both relationships identically.

### What's Next

Now we can process text AND images as token sequences.
Phase V will combine them: **Vision-Language Models**.
The transformer will process BOTH modalities in the same sequence.

```
Phase V preview:
  [image_patch_1] ... [image_patch_196] [text_token_1] ... [text_token_N]
  ↓
  Single transformer processes the WHOLE sequence
  ↓
  "The robot is picking up a red box from shelf B3"
```

The modality barrier is about to break completely.

---

*Next: [09-3d-video-detection.md](./09-3d-video-detection.md) — 3D vision, depth, video, and detection*
