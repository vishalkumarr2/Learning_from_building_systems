# Day 48: DINO & Self-Supervised Vision

> **Phase IV — Vision: ViT, 3D, Video | Week 7 | 2.5 hours**
> *"No labels needed. DINO's attention maps discover objects and parts — segmentation for free." — Caron et al., 2021*

## Navigation
- **Previous:** [Day 47: Swin Transformer](day-47-swin-transformer.md)
- **Next:** [Day 49: MAE](day-49-mae.md)
- **Week:** [Week 7 Overview](README.md)
- **Phase:** [Phase IV: Vision](../../study-notes/08-vision-transformers.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### Self-Supervised Learning for Vision

Supervised learning needs labeled data. Self-supervised learning creates its own supervision signal from the data itself. DINO (self-DIstillation with NO labels) trains a ViT using **student-teacher self-distillation**:

```
┌──────────────────────────────────────────────────────────┐
│                     DINO Framework                        │
│                                                          │
│   Image x                                                │
│     │                                                    │
│     ├──► Global crops (224×224)  ──► Teacher network     │
│     │    + local crops (96×96)       (EMA of student)    │
│     │                                    │               │
│     └──► Global + local crops   ──► Student network      │
│                                      │                   │
│                                      ▼                   │
│                  Loss: cross-entropy(student, teacher)    │
│                  Teacher output is sharpened + centered   │
│                                                          │
│   Key: teacher = exponential moving average of student   │
│   No labels, no contrastive pairs, no negative samples!  │
└──────────────────────────────────────────────────────────┘
```

### Multi-Crop Strategy

DINO uses asymmetric crops:
- **Teacher sees only global crops** (2 crops, 224×224, covering >50% of image)
- **Student sees global + local crops** (6-8 local crops, 96×96)

The student must predict the teacher's output for global views from its local views — forcing it to learn about the whole image from partial glimpses.

### Self-Distillation Mechanics

**Teacher update** — exponential moving average (no gradient):

$$\theta_t \leftarrow m \cdot \theta_t + (1 - m) \cdot \theta_s$$

with momentum $m$ starting at 0.996 and annealing to 1.0 (teacher becomes frozen).

**Centering and sharpening** prevent mode collapse:

$$P_t(x) = \text{softmax}\left(\frac{g_t(x) - c}{\tau_t}\right), \quad c \leftarrow m_c \cdot c + (1 - m_c) \cdot \bar{g}_t$$

where $\tau_t = 0.04$ (sharp teacher) and $\tau_s = 0.1$ (softer student).

### The Magic: Emergent Segmentation

DINO's self-attention maps in the last layer spontaneously learn to segment objects — **without any segmentation labels**:

```
Input image: [photo of a dog on grass]
Attention map (head 0): highlights the dog body
Attention map (head 1): highlights the dog face
Attention map (head 2): highlights the grass background

→ Combining heads = free object segmentation!
```

This happens because the [CLS] token must attend to semantically meaningful regions to predict global views from local crops.

### DINOv2: Scaling Up

DINOv2 (2023) combines DINO + iBOT (masked image modeling) at massive scale:
- Trained on LVD-142M curated dataset
- ViT-Giant (1.1B params)
- Produces features that work across tasks **without fine-tuning**: classification, segmentation, depth, matching

---

## Implementation (60 min)

### Using DINOv2 Pretrained Features

```python
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np


def load_dinov2(model_name='dinov2_vits14'):
    """Load DINOv2 pretrained model from torch hub."""
    model = torch.hub.load('facebookresearch/dinov2', model_name)
    model.eval()
    return model


def extract_attention_maps(model, image_tensor):
    """Extract self-attention maps from the last layer."""
    # Register hook to capture attention weights
    attention_maps = []
    
    def hook_fn(module, input, output):
        # output is the attention output; we need the attention weights
        # For DINOv2, we use the forward method with attention output
        pass
    
    # Use model's built-in method if available
    with torch.no_grad():
        # Get intermediate features with attention
        features = model.get_intermediate_layers(image_tensor, n=1, return_class_token=True)
    
    return features


def visualize_dino_features(model, image_path, patch_size=14):
    """Visualize DINO attention maps showing emergent segmentation."""
    transform = transforms.Compose([
        transforms.Resize((518, 518)),  # DINOv2 default
        transforms.CenterCrop((518, 518)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        # Get patch tokens (exclude CLS)
        features = model.forward_features(img_tensor)
        patch_tokens = features['x_norm_patchtokens']  # (1, N, D)
    
    # PCA visualization of patch features
    N, D = patch_tokens.shape[1], patch_tokens.shape[2]
    h = w = int(N ** 0.5)
    
    # Reduce to 3 dims with PCA for RGB visualization
    tokens = patch_tokens[0].cpu().numpy()
    mean = tokens.mean(axis=0)
    centered = tokens - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pca_features = centered @ Vt[:3].T
    
    # Normalize to [0, 1] for display
    pca_features = (pca_features - pca_features.min(0)) / (pca_features.max(0) - pca_features.min(0))
    pca_image = pca_features.reshape(h, w, 3)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(pca_image)
    axes[1].set_title('DINO Feature PCA (no labels used!)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('dino_features.png', dpi=150)
    print("Saved dino_features.png")
```

### k-NN Classification with DINO Features

```python
from torchvision import datasets
from torch.utils.data import DataLoader


def build_knn_classifier(model, train_loader, val_loader, k=20):
    """k-NN classification using frozen DINO features — no training needed."""
    device = next(model.parameters()).device
    
    # Extract features
    def extract_features(loader):
        all_features = []
        all_labels = []
        with torch.no_grad():
            for images, labels in loader:
                features = model(images.to(device))
                features = F.normalize(features, dim=-1)
                all_features.append(features.cpu())
                all_labels.append(labels)
        return torch.cat(all_features), torch.cat(all_labels)
    
    train_features, train_labels = extract_features(train_loader)
    val_features, val_labels = extract_features(val_loader)
    
    # k-NN: for each val sample, find k nearest train samples
    similarity = val_features @ train_features.T  # (N_val, N_train)
    topk_sim, topk_idx = similarity.topk(k, dim=-1)
    
    # Vote
    topk_labels = train_labels[topk_idx]  # (N_val, k)
    preds = topk_labels.mode(dim=-1).values
    
    accuracy = (preds == val_labels).float().mean().item()
    print(f"k-NN accuracy (k={k}): {accuracy:.4f}")
    return accuracy
```

---

## Exercise (45 min)

1. **Feature quality comparison:** Extract features from DINOv2 and a supervised ViT (from timm) on the same images. Run k-NN on CIFAR-10 with both. Which features are better? Why?

2. **Attention head diversity:** Visualize attention maps from all heads in the last layer of DINOv2. Do different heads attend to different semantic concepts?

3. **Linear probing:** Add a single linear layer on top of frozen DINOv2 features. Train on CIFAR-10 for 10 epochs. How does this compare to k-NN?

---

## Key Takeaways

1. **No labels needed.** DINO learns powerful visual features through self-distillation alone
2. **Emergent segmentation.** Attention maps discover object boundaries without supervision
3. **Student-teacher framework.** Teacher = EMA of student; centering + sharpening prevent collapse
4. **Multi-crop asymmetry.** Student predicts global views from local crops → learns holistic features
5. **DINOv2 is a foundation.** Its features transfer across tasks without fine-tuning

## Connection to the Thread

> *DINO shows that transformers can learn visual structure without supervision — the same way LLMs learn language structure from unlabeled text. Tomorrow: MAE takes the complementary approach — mask patches and reconstruct, just like BERT masks words.*

---

## Further Reading

- [Emerging Properties in Self-Supervised ViTs (Caron et al., 2021)](https://arxiv.org/abs/2104.14294)
- [DINOv2 (Oquab et al., 2023)](https://arxiv.org/abs/2304.07193)
- Official DINOv2: github.com/facebookresearch/dinov2
- Blog: "Self-Supervised Learning — DINO" by Lilian Weng
