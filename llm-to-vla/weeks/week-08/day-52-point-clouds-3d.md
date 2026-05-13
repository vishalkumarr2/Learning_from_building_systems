# Day 52: Point Clouds & 3D Scenes

> **Phase IV — Vision: ViT, 3D, Video | Week 8 | 2.5 hours**
> *"Point clouds are the native language of 3D perception — unordered, sparse, and directly representing the physical world." — Qi et al., 2017*

## Navigation
- **Previous:** [Day 51: 3D Vision & Depth](day-51-3d-vision-depth.md)
- **Next:** [Day 53: Video Understanding Day 1](day-53-video-understanding-1.md)
- **Week:** [Week 8 Overview](README.md)
- **Phase:** [Phase IV: Vision](../../study-notes/09-3d-video-detection.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### What Are Point Clouds?

A point cloud is a set of 3D points $\{(x_i, y_i, z_i)\}_{i=1}^N$, often with additional features (color, normals). Unlike images (regular grids), point clouds are:
- **Unordered** — no canonical ordering of points
- **Sparse** — unevenly sampled in 3D space
- **Permutation invariant** — $f(\{p_1, p_2, p_3\}) = f(\{p_3, p_1, p_2\})$

### PointNet: The Foundation

PointNet (2017) solves the permutation invariance problem with a simple architecture:

```
Input: N × 3 points
    │
    ▼
Per-point MLP: shared MLP applied independently to each point
    │         (64 → 128 → 1024 dims)
    ▼
Max Pooling: aggregate across all points → global feature
    │         (permutation invariant!)
    ▼
Classification / Segmentation head
```

Key insight: **max pooling over points** is permutation invariant. The shared MLP maps each point to a high-dimensional feature, and max pooling extracts the most "activated" feature per dimension.

$$f(\{p_1, \ldots, p_N\}) = g\left(\max_{i=1}^N h(p_i)\right)$$

### PointNet++: Hierarchical Features

PointNet processes all points globally — it misses local structure. PointNet++ adds hierarchy:

```
Stage 1: Sample 1024 centroids → group neighbors (radius 0.1) → PointNet per group
    │
Stage 2: Sample 256 centroids → group neighbors (radius 0.2) → PointNet per group
    │
Stage 3: Sample 64 centroids → group neighbors (radius 0.4) → PointNet per group
    │
Global features → classification/segmentation
```

This is analogous to CNN's progressive receptive field growth, but for unordered 3D data.

### Point Cloud Transformers

Modern approaches apply self-attention to point clouds:

**Point Transformer (2021):** Vector self-attention with position encoding:

$$y_i = \sum_{j \in \mathcal{N}(i)} \text{softmax}\left(\varphi(x_i) - \psi(x_j) + \delta_{ij}\right) \odot (\alpha(x_j) + \delta_{ij})$$

where $\delta_{ij}$ encodes the relative 3D position between points $i$ and $j$.

### 3D Scene Representations for Robotics

| Representation | Pros | Cons | Robotics Use |
|---------------|------|------|--------------|
| Point cloud | Direct from sensors, sparse | Unordered, variable size | Grasping, obstacle detection |
| Voxel grid | Regular, CNN-friendly | Memory-hungry ($O(n^3)$) | Occupancy mapping |
| Mesh | Surface topology | Hard to learn | Simulation |
| NeRF/3DGS | Photorealistic | Slow, implicit | Scene understanding |
| Truncated SDF | Continuous surface | Requires fusion | SLAM, reconstruction |

---

## Implementation (60 min)

### PointNet from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    """PointNet feature extractor."""
    
    def __init__(self, in_channels=3, feature_dim=1024):
        super().__init__()
        # Shared MLPs (applied per-point)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, feature_dim, 1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, N, 3) point cloud
        Returns:
            global_feat: (B, feature_dim) global feature
            point_feat: (B, N, feature_dim) per-point features
        """
        x = x.transpose(1, 2)  # (B, 3, N)
        point_feat = self.mlp1(x)  # (B, feature_dim, N)
        
        # Max pooling → permutation invariant global feature
        global_feat = point_feat.max(dim=-1)[0]  # (B, feature_dim)
        
        return global_feat, point_feat.transpose(1, 2)


class PointNetClassifier(nn.Module):
    """PointNet for 3D shape classification."""
    
    def __init__(self, n_classes=40, in_channels=3):
        super().__init__()
        self.encoder = PointNetEncoder(in_channels, feature_dim=1024)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )
    
    def forward(self, x):
        global_feat, _ = self.encoder(x)
        return self.classifier(global_feat)


class PointNetSegmenter(nn.Module):
    """PointNet for per-point segmentation."""
    
    def __init__(self, n_classes=50, in_channels=3):
        super().__init__()
        self.encoder = PointNetEncoder(in_channels, feature_dim=1024)
        
        # Per-point classifier using local + global features
        self.seg_head = nn.Sequential(
            nn.Conv1d(1024 + 1024, 512, 1),  # concat local + global
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, n_classes, 1),
        )
    
    def forward(self, x):
        B, N, _ = x.shape
        global_feat, point_feat = self.encoder(x)
        
        # Broadcast global feature to each point
        global_expanded = global_feat.unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([point_feat, global_expanded], dim=-1)
        
        out = self.seg_head(combined.transpose(1, 2))
        return out.transpose(1, 2)  # (B, N, n_classes)


# Test
model = PointNetClassifier(n_classes=10)
points = torch.randn(4, 1024, 3)  # 4 samples, 1024 points each
logits = model(points)
print(f"Classification output: {logits.shape}")  # (4, 10)

seg_model = PointNetSegmenter(n_classes=5)
seg_out = seg_model(points)
print(f"Segmentation output: {seg_out.shape}")  # (4, 1024, 5)
```

### Robotics Application: Grasp Point Detection

```python
def find_grasp_candidates(points, normals=None, n_candidates=10):
    """Simple heuristic grasp point detection from point cloud.
    
    For a top-down grasp, find points where:
    1. Surface normal points upward (graspable from above)
    2. Local geometry is relatively flat (stable grasp)
    3. Points are not on the ground plane
    """
    # Filter ground plane (z > threshold)
    height_mask = points[:, 2] > 0.02  # 2cm above ground
    candidates = points[height_mask]
    
    if normals is not None:
        surface_normals = normals[height_mask]
        # Upward-facing normals (dot product with z-axis > 0.8)
        up = torch.tensor([0.0, 0.0, 1.0])
        upward_mask = (surface_normals @ up) > 0.8
        candidates = candidates[upward_mask]
    
    # Sample n_candidates from remaining points
    if len(candidates) > n_candidates:
        idx = torch.randperm(len(candidates))[:n_candidates]
        candidates = candidates[idx]
    
    return candidates
```

---

## Exercise (45 min)

1. **Permutation invariance proof:** Verify empirically that PointNet gives the same output regardless of point ordering. Shuffle the input points and check.

2. **ModelNet10 classification:** Download ModelNet10 (10 shape categories). Train PointNet for classification. Report accuracy vs number of input points (256, 512, 1024, 2048).

3. **Point cloud from depth:** Using yesterday's depth estimation, generate a point cloud from an image. Then run PointNet feature extraction on it. Can you cluster objects?

---

## Key Takeaways

1. **Permutation invariance.** Max pooling makes point cloud processing order-independent
2. **Shared MLPs.** Same network applied to every point independently → scalable
3. **Hierarchy helps.** PointNet++ adds local structure that PointNet misses
4. **3D for robotics.** Grasping, navigation, and scene understanding all rely on 3D perception
5. **Transformers extend.** Point cloud transformers use relative 3D positions in attention

## Connection to the Thread

> *You've now seen transformers process 1D sequences (text), 2D grids (images), and unordered 3D sets (point clouds). The same attention mechanism, three modalities. Next: video — adding the temporal dimension.*

---

## Further Reading

- [PointNet (Qi et al., 2017)](https://arxiv.org/abs/1612.00593)
- [PointNet++ (Qi et al., 2017)](https://arxiv.org/abs/1706.02413)
- [Point Transformer (Zhao et al., 2021)](https://arxiv.org/abs/2012.09164)
- [Contact-GraspNet (Sundermeyer et al., 2021)](https://arxiv.org/abs/2103.14127) — neural grasp detection
