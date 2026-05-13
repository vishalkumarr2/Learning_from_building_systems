# Day 46: Training ViT + DeiT

> **Phase IV — Vision: ViT, 3D, Video | Week 7 | 2.5 hours**
> *"DeiT showed that you don't need 300 million images — you need better training recipes and a teacher." — Touvron et al., 2021*

## Navigation
- **Previous:** [Day 45: ViT](day-45-vit.md)
- **Next:** [Day 47: Swin Transformer](day-47-swin-transformer.md)
- **Week:** [Week 7 Overview](README.md)
- **Phase:** [Phase IV: Vision](../../study-notes/08-vision-transformers.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### The Data Efficiency Problem

ViT-Base trained on ImageNet-1k from scratch: **~76%** top-1 accuracy.
ViT-Base pretrained on JFT-300M → fine-tuned on ImageNet: **~85%**.

The gap is enormous. DeiT (Data-efficient Image Transformers) closes it with **training recipe + knowledge distillation**, achieving 83%+ on ImageNet-1k alone.

### DeiT Training Recipe

```
┌─────────────────────────────────────────────────────────┐
│              DeiT Training Ingredients                   │
│                                                          │
│  1. Data augmentation                                    │
│     ├─ RandAugment (random augmentation policy)         │
│     ├─ CutMix (cut-paste regions between images)        │
│     ├─ MixUp (linear interpolation of image pairs)      │
│     └─ Random Erasing (mask random rectangles)          │
│                                                          │
│  2. Regularization                                       │
│     ├─ Stochastic depth (drop entire layers randomly)   │
│     ├─ Label smoothing (ε = 0.1)                        │
│     └─ Repeated augmentation                            │
│                                                          │
│  3. Knowledge distillation                               │
│     ├─ Teacher: RegNet-Y-16GF (CNN)                     │
│     ├─ Hard distillation (teacher labels)               │
│     └─ Distillation token (like a second [CLS])        │
│                                                          │
│  4. Optimizer: AdamW, lr=5e-4, cosine schedule          │
│     └─ 300 epochs (vs ViT's 90 on JFT)                 │
└─────────────────────────────────────────────────────────┘
```

### Data Augmentation Deep Dive

**CutMix:** Cut a random rectangle from image B and paste it onto image A. Labels are mixed proportionally to area:

$$\tilde{y} = \lambda \cdot y_A + (1 - \lambda) \cdot y_B$$

**MixUp:** Linearly blend two images and their labels:

$$\tilde{x} = \lambda \cdot x_A + (1 - \lambda) \cdot x_B$$

**RandAugment:** Sample $N$ augmentations from a pool (rotation, color jitter, sharpness, etc.) at magnitude $M$.

### Knowledge Distillation

DeiT introduces a **distillation token** alongside the [CLS] token:

$$\text{tokens} = [\text{CLS}, \; \text{DIST}, \; p_1, p_2, \ldots, p_N]$$

- **[CLS]** is trained with the true label loss
- **[DIST]** is trained to match the teacher's output (hard or soft labels)
- At inference, predictions from both tokens are averaged

**Hard distillation** (using teacher's argmax) works better than soft distillation for ViT — this is surprising and suggests the CNN teacher provides useful inductive bias about locality.

### Stochastic Depth

Randomly drop entire transformer blocks during training (like dropout for layers):

$$\text{output} = \begin{cases} x + f(x) & \text{with probability } p \\ x & \text{with probability } 1-p \end{cases}$$

Survival probability typically decreases linearly from 1.0 (first layer) to 0.8 (last layer).

---

## Implementation (60 min)

### Train ViT on CIFAR-10 with DeiT Augmentations

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from timm.data import create_transform
from timm.data.mixup import Mixup


def get_deit_transforms(img_size=224):
    """DeiT-style data augmentation pipeline."""
    train_transform = create_transform(
        input_size=img_size,
        is_training=True,
        auto_augment='rand-m9-mstd0.5-inc1',  # RandAugment
        re_prob=0.25,        # Random erasing probability
        re_mode='pixel',
        re_count=1,
        interpolation='bicubic',
    )
    
    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_transform


def setup_mixup():
    """Configure CutMix + MixUp."""
    return Mixup(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,     # 50% chance of CutMix vs MixUp
        mode='batch',
        label_smoothing=0.1,
        num_classes=10,
    )


# Build data loaders
train_tf, val_tf = get_deit_transforms(img_size=32)  # CIFAR is 32x32
train_ds = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_tf)
val_ds = datasets.CIFAR10(root='./data', train=False, download=True, transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=4)
```

### Implement Stochastic Depth

```python
class StochasticDepth(nn.Module):
    """Drop entire residual branch with given probability."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        
        keep_prob = 1 - self.drop_prob
        # Random tensor: (B, 1, 1) — same decision for all tokens in a sample
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device) < keep_prob
        return x * mask / keep_prob


class DeiTBlock(nn.Module):
    """Transformer block with stochastic depth."""
    
    def __init__(self, embed_dim, n_heads, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, n_heads)
        self.drop_path = StochasticDepth(drop_path)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
```

### Training Loop with Label Smoothing

```python
def train_vit_cifar(model, train_loader, val_loader, epochs=50, lr=5e-4):
    """Full training loop with DeiT augmentation strategy."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Label smoothing loss
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    mixup_fn = setup_mixup()
    
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            
            # Apply MixUp/CutMix
            images, targets = mixup_fn(images, targets)
            
            logits = model(images)
            loss = criterion(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                preds = model(images).argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        
        acc = correct / total
        best_acc = max(best_acc, acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, "
                  f"val_acc={acc:.4f}, best={best_acc:.4f}")
    
    return best_acc
```

---

## Exercise (45 min)

1. **Augmentation ablation:** Train the same ViT model with:
   - (a) No augmentation
   - (b) Only RandAugment
   - (c) RandAugment + CutMix + MixUp
   - (d) Full DeiT recipe (+ stochastic depth + label smoothing)
   
   Plot validation accuracy curves. How much does each component contribute?

2. **Knowledge distillation:** Use a pretrained ResNet-50 as teacher. Train a student ViT with:
   - Soft distillation: KL divergence on softmax outputs
   - Hard distillation: cross-entropy with teacher's argmax
   
   Which works better? (Hint: DeiT found hard distillation wins for ViTs.)

3. **CIFAR-10 target:** Can you reach **95%+ accuracy** on CIFAR-10 with a ViT-Tiny (embed_dim=192, depth=12, heads=3) using the full DeiT recipe?

---

## Key Takeaways

1. **Training recipe > architecture.** DeiT matches ViT-JFT with ImageNet-1k alone
2. **Augmentation is critical.** CutMix, MixUp, RandAugment — each contributes meaningfully
3. **Distillation from CNNs works.** The CNN teacher injects useful locality bias
4. **Stochastic depth.** Regularization for deep transformers — drop entire blocks during training
5. **Label smoothing.** Prevents overconfident predictions; helps calibration

## Connection to the Thread

> *DeiT proved that the transformer architecture is not the bottleneck — training methodology is. Tomorrow's Swin Transformer tackles the other bottleneck: computational efficiency for high-resolution images.*

---

## Further Reading

- [Training Data-efficient Image Transformers (Touvron et al., 2021)](https://arxiv.org/abs/2012.12877)
- [CutMix (Yun et al., 2019)](https://arxiv.org/abs/1905.04899)
- [RandAugment (Cubuk et al., 2020)](https://arxiv.org/abs/1909.13719)
- timm library documentation: huggingface.co/docs/timm
