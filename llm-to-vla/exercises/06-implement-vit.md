# Exercise 06 — Implement ViT from Scratch + DINO Features

> Phase IV · Days 45–50 · ~8 hours
> Prerequisites: Study note 08-vision-transformers, Exercise 01 (ResNet baseline)
> Goal: Build ViT from scratch, train it, then explore self-supervised features

---

## Setup

```python
# Environment setup
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# We'll need these later
# pip install transformers timm open-clip-torch
```

---

## Exercise 1: Patch Embedding (45 min)

### 1.1 Manual Patch Extraction

Understand what patch embedding does by doing it manually first.

```python
def extract_patches_manual(image, patch_size=16):
    """
    Extract patches from an image manually.

    Args:
        image: tensor of shape (C, H, W)
        patch_size: size of each square patch

    Returns:
        patches: tensor of shape (N_patches, patch_size * patch_size * C)
        grid_size: (H_patches, W_patches)

    TODO: Implement this function
    Steps:
    1. Get image dimensions
    2. Calculate number of patches in each dimension
    3. Use unfold to extract patches
    4. Reshape to (N_patches, patch_size * patch_size * C)
    """
    pass

# Test with a sample image
sample_image = torch.randn(3, 224, 224)
patches, grid_size = extract_patches_manual(sample_image, patch_size=16)

# Verify
assert patches.shape == (196, 768), f"Expected (196, 768), got {patches.shape}"
assert grid_size == (14, 14), f"Expected (14, 14), got {grid_size}"
print(f"✓ Extracted {patches.shape[0]} patches of dimension {patches.shape[1]}")
print(f"  Grid: {grid_size[0]} × {grid_size[1]}")
```

### 1.2 Visualize Patches

```python
def visualize_patches(image, patch_size=16, n_show=16):
    """
    Visualize the first n_show patches from an image.

    TODO: Implement visualization
    1. Extract patches manually
    2. Reshape each patch back to (patch_size, patch_size, C)
    3. Plot in a grid using matplotlib
    4. Draw grid lines on original image showing patch boundaries
    """
    pass

# Load a real image and visualize
# Use any image — ideally a robot/warehouse scene
def get_sample_image(size=(224, 224)):
    """Download a sample image or generate a synthetic one as fallback."""
    import urllib.request, os
    url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
    path = "test_image.jpg"
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(url, path)
        except Exception:
            # Fallback: use a random CIFAR-10 image
            ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
            img = ds[0][0].resize(size)
            img.save(path)
    return Image.open(path).convert("RGB").resize(size)

img = get_sample_image((224, 224))
img_tensor = transforms.ToTensor()(img)
visualize_patches(img_tensor)
```

### 1.3 Conv2d Patch Embedding

```python
class PatchEmbedding(nn.Module):
    """
    Implement patch embedding using Conv2d.

    This should be equivalent to: flatten patches → linear projection
    But uses a strided convolution for efficiency.

    TODO: Implement __init__ and forward
    """

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        # TODO: Calculate n_patches
        # TODO: Create Conv2d with correct kernel_size and stride
        pass

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, N_patches, embed_dim)
        """
        # TODO: Apply conv, flatten spatial dims, transpose
        pass

# Test
patch_embed = PatchEmbedding()
x = torch.randn(2, 3, 224, 224)
out = patch_embed(x)
assert out.shape == (2, 196, 768), f"Expected (2, 196, 768), got {out.shape}"
print(f"✓ PatchEmbedding output shape: {out.shape}")
```

### 1.4 Verify Equivalence

```python
def verify_patch_methods_equivalent():
    """
    Verify that manual patch extraction + linear projection
    gives the same result as Conv2d patch embedding.

    TODO:
    1. Create a PatchEmbedding module
    2. Extract patches manually
    3. Apply the Conv2d weights as a linear layer to manual patches
    4. Compare outputs — they should be identical (within floating point)
    """
    pass

verify_patch_methods_equivalent()
```

---

## Exercise 2: ViT from Scratch (2–3 hours)

### 2.1 Core Components

```python
class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention (same as in transformer, but applied to patch tokens).

    TODO: Implement using what you learned in Exercise 03.
    """

    def __init__(self, embed_dim=768, num_heads=12, dropout=0.0):
        super().__init__()
        # TODO: qkv projection, output projection, etc.
        pass

    def forward(self, x):
        """
        Args:
            x: (B, N, embed_dim) — N includes [CLS] token
        Returns:
            (B, N, embed_dim)
        """
        pass


class TransformerBlock(nn.Module):
    """
    Single transformer block: LayerNorm → MHSA → residual → LayerNorm → FFN → residual

    Note: ViT uses PRE-norm (LayerNorm before attention), not POST-norm.
    """

    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        # TODO: LayerNorm, MHSA, FFN (with GELU activation)
        pass

    def forward(self, x):
        # TODO: Pre-norm residual connections
        pass
```

### 2.2 Complete ViT

```python
class VisionTransformer(nn.Module):
    """
    Complete Vision Transformer (ViT).

    Architecture:
        Image → PatchEmbedding → [CLS] + position embedding
        → TransformerBlock × num_layers
        → LayerNorm → [CLS] output → Classification head

    TODO: Implement the full ViT
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()
        # TODO: Implement all components:
        # 1. PatchEmbedding
        # 2. [CLS] token (learnable parameter)
        # 3. Position embeddings (learnable parameter)
        # 4. Dropout
        # 5. Stack of TransformerBlocks
        # 6. Final LayerNorm
        # 7. Classification head (Linear)
        pass

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) — batch of images
        Returns:
            (B, num_classes) — class logits
        """
        # TODO: Full forward pass
        pass

# Create a "tiny" ViT for CIFAR-10 training
vit_tiny = VisionTransformer(
    img_size=32,        # CIFAR-10 is 32×32
    patch_size=4,       # 4×4 patches → 8×8 = 64 patches
    in_channels=3,
    num_classes=10,
    embed_dim=256,      # Smaller for CIFAR
    num_layers=6,
    num_heads=8,
    mlp_ratio=4.0,
    dropout=0.1,
)

# Count parameters
n_params = sum(p.numel() for p in vit_tiny.parameters())
print(f"ViT-Tiny parameters: {n_params:,}")

# Test forward pass
x = torch.randn(4, 3, 32, 32)
logits = vit_tiny(x)
assert logits.shape == (4, 10), f"Expected (4, 10), got {logits.shape}"
print(f"✓ Forward pass: input {x.shape} → output {logits.shape}")
```

### 2.3 Train on CIFAR-10

```python
def train_vit_cifar10(model, epochs=50, lr=1e-3, batch_size=128):
    """
    Train ViT on CIFAR-10.

    TODO: Implement training loop
    1. Load CIFAR-10 with appropriate transforms
       - RandomCrop, RandomHorizontalFlip, Normalize
       - For ViT: consider RandAugment or Mixup for better performance
    2. Use AdamW optimizer with cosine learning rate schedule
    3. Train for `epochs` epochs
    4. Track and plot training/validation loss and accuracy
    5. Return final accuracy

    Tips for ViT on CIFAR-10:
    - Warmup the learning rate for first 5 epochs
    - Use weight decay (0.05)
    - Label smoothing helps (0.1)
    - Don't expect ImageNet-level accuracy — CIFAR-10 is small
    - Target: ~85-90% accuracy with tiny ViT
    """
    pass

# Train!
train_acc, val_acc = train_vit_cifar10(vit_tiny)
print(f"Final validation accuracy: {val_acc:.1f}%")
```

### 2.4 Compare with ResNet

```python
def compare_vit_vs_resnet():
    """
    Compare your ViT with a ResNet of similar parameter count on CIFAR-10.

    TODO:
    1. Load or train a ResNet-18 on CIFAR-10 (from Exercise 01 or torchvision)
    2. Compare: accuracy, parameters, inference speed (FPS), training time
    3. Create a comparison table
    4. Discuss: why does one outperform the other on small data?

    Expected finding: ResNet likely wins on CIFAR-10 due to:
    - CNN inductive bias (locality, translation invariance)
    - CIFAR-10 is small — ViT needs more data
    """
    pass

compare_vit_vs_resnet()
```

---

## Exercise 3: DINO Feature Extraction (1.5–2 hours)

### 3.1 Load DINOv2 and Extract Features

```python
def load_dinov2():
    """
    Load DINOv2-base from Hugging Face and prepare for feature extraction.

    TODO:
    1. Load model and processor from "facebook/dinov2-base"
    2. Set model to eval mode
    3. Return model, processor
    """
    pass

def extract_dinov2_features(model, processor, image):
    """
    Extract CLS and patch features from an image using DINOv2.

    Args:
        model: DINOv2 model
        processor: DINOv2 image processor
        image: PIL Image

    Returns:
        cls_features: (768,) — global image representation
        patch_features: (N, 768) — per-patch features

    TODO: Implement feature extraction
    """
    pass

# Test
model, processor = load_dinov2()
img = get_sample_image()  # Uses helper from Exercise 1
cls_feat, patch_feat = extract_dinov2_features(model, processor, img)
print(f"CLS features: {cls_feat.shape}")
print(f"Patch features: {patch_feat.shape}")
```

### 3.2 Visualize Attention Maps

```python
def visualize_dino_attention(model, image, patch_size=14):
    """
    Visualize self-attention maps from DINOv2's last layer.

    This reveals DINO's emergent object segmentation ability.

    TODO:
    1. Get attention weights from the last transformer layer
       - Access model.encoder.layer[-1].attention
       - Hook into the attention computation to get the attention matrix
    2. Extract [CLS] → patch attention for each head
    3. Reshape to spatial grid (H_patches × W_patches)
    4. Visualize: original image + attention map for each head
    5. Show the mean attention across all heads

    Expected: different heads attend to different semantic parts
    (one head for object, one for background, one for edges, etc.)
    """
    pass

# Visualize
img = get_sample_image()  # Uses helper from Exercise 1
visualize_dino_attention(model, img)
```

### 3.3 k-NN Classification with DINO Features (No Fine-Tuning!)

```python
def dino_knn_classification(model, processor, train_dataset, test_dataset, k=20):
    """
    Classify images using DINO features + k-nearest neighbors.
    NO fine-tuning — just frozen features + simple distance metric.

    TODO:
    1. Extract DINOv2 CLS features for ALL training images
    2. Extract DINOv2 CLS features for ALL test images
    3. For each test image: find k nearest training images by cosine similarity
    4. Majority vote among k neighbors → predicted class
    5. Report accuracy

    This demonstrates how powerful self-supervised features are:
    - No training on this dataset
    - No fine-tuning
    - Just "find the most similar images" using frozen features
    - Yet achieves surprisingly high accuracy!

    Try on CIFAR-10 or a custom dataset with robot images.
    """
    pass

# Load CIFAR-10
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

accuracy = dino_knn_classification(model, processor, train_data, test_data, k=20)
print(f"DINO k-NN accuracy on CIFAR-10: {accuracy:.1f}%")
# Expected: ~75-85% with NO training on CIFAR-10!
```

### 3.4 Feature Similarity Visualization

```python
def visualize_feature_similarity(model, processor, images, labels):
    """
    Visualize how DINO groups images by semantic similarity.

    TODO:
    1. Extract CLS features for a batch of images
    2. Compute pairwise cosine similarity matrix
    3. Plot as heatmap (sorted by class)
    4. Apply t-SNE or UMAP to features, plot 2D scatter colored by class

    Expected: images of the same category cluster together,
    even though DINO was never trained with category labels!
    """
    pass

# Collect images from different categories
# visualize_feature_similarity(model, processor, images, labels)
```

---

## Exercise 4: MAE Visualization (1 hour)

### 4.1 Masked Patch Reconstruction

```python
def visualize_mae_reconstruction(image_path, mask_ratio=0.75):
    """
    Visualize MAE's mask-and-reconstruct process.

    TODO:
    1. Load a pretrained MAE model (e.g., "facebook/vit-mae-base" from HF)
    2. Process image into patches
    3. Randomly mask `mask_ratio` fraction of patches
    4. Run through MAE (encode visible → decode all)
    5. Visualize side-by-side:
       a. Original image
       b. Masked image (visible patches only, rest grayed out)
       c. MAE reconstruction
       d. Reconstruction error (difference between original and reconstruction)

    Tips:
    - Use torchvision or HuggingFace model
    - The model's pixel_values should be the full image
    - You may need to handle the masking yourself or use the model's built-in masking
    """
    pass

# Test with different images and mask ratios
for ratio in [0.5, 0.75, 0.9]:
    print(f"\n=== Mask ratio: {ratio} ===")
    visualize_mae_reconstruction(get_sample_image(), mask_ratio=ratio)
```

### 4.2 What Does MAE Reconstruct?

```python
def analyze_mae_reconstruction_quality(model, images, mask_ratio=0.75, n_trials=10):
    """
    Analyze what MAE reconstructs well vs poorly.

    TODO:
    1. For each image, run MAE multiple times with different random masks
    2. Compute per-patch reconstruction error
    3. Identify: which patches are easiest/hardest to reconstruct?
    4. Visualize: error heatmap overlaid on original image

    Expected findings:
    - Uniform regions (sky, floor) → easy to reconstruct
    - Edges and textures → moderate difficulty
    - Unique features (text, faces, small objects) → hardest
    - This reveals what the model "understands" about image structure
    """
    pass
```

---

## Self-Check

After completing all exercises, verify:

- [ ] **Patch embedding**: Can extract patches manually AND with Conv2d, results match
- [ ] **ViT architecture**: Built complete ViT from scratch, all shapes correct
- [ ] **Training**: ViT trained on CIFAR-10, achieves reasonable accuracy (~85%+)
- [ ] **ViT vs ResNet**: Understood why CNN has advantage on small data
- [ ] **DINO features**: Extracted features, visualized attention maps
- [ ] **k-NN classification**: Achieved good accuracy with frozen DINO features (no training!)
- [ ] **MAE visualization**: Understand mask → reconstruct process

### Key Questions

1. Why does ViT need more data than ResNet to achieve good performance?
2. How does DINO achieve object segmentation without segmentation labels?
3. Why does MAE mask 75% while BERT only masks 15%?
4. If you had to choose ONE model for robot perception, which would it be and why?

---

## Stretch Goals

1. **Implement Swin window attention**: Replace global attention in your ViT with windowed attention. Compare complexity and accuracy.

2. **DINO feature visualization on robot data**: If you have access to robot camera images, extract and visualize DINO attention maps. Do they segment robot-relevant objects (boxes, shelves, obstacles)?

3. **Fine-tune DINOv2 on a custom dataset**: Take DINOv2 features, add a linear head, fine-tune on a small custom dataset. Compare: k-NN vs linear probe vs full fine-tune.

4. **Attention distance analysis**: For your trained ViT, measure the average attention distance (how far apart patches attend to each other) at each layer. Do early layers attend locally and deep layers attend globally?

5. **Position embedding visualization**: Extract and visualize the learned position embeddings from your ViT. Do nearby patches have similar position embeddings? Does a 2D structure emerge?

---

*Next: [07-3d-depth-video.md](./07-3d-depth-video.md) — Depth estimation, video understanding, and detection*
