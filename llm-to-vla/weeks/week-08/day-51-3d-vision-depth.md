# Day 51: 3D Vision & Depth

> **Phase IV — Vision: ViT, 3D, Video | Week 8 | 2.5 hours**
> *"A robot doesn't just see pixels — it perceives a 3D world. Depth is the bridge between images and physical interaction." — Ranftl et al., 2021*

## Navigation
- **Previous:** [Day 50: Stop & Reflect #3](day-50-stop-reflect-3.md)
- **Next:** [Day 52: Point Clouds & 3D Scenes](day-52-point-clouds-3d.md)
- **Week:** [Week 8 Overview](README.md)
- **Phase:** [Phase IV: Vision](../../study-notes/09-3d-video-detection.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### Why Depth Matters for Robotics

A 2D image loses the third dimension. For manipulation and navigation, robots need to know:
- How far away objects are
- Which surfaces are graspable
- Where obstacles lie in 3D space

```
RGB Image          Depth Map              3D Point Cloud
┌──────────┐      ┌──────────┐           .  .  .  .
│  🤖  📦  │  →   │ ░░░ ▓▓▓ │    →      .   .   .
│   table  │      │ ░░░░░░░ │          ..........
│  floor   │      │ ░░░░░░░ │          ...........
└──────────┘      └──────────┘

near=dark        far=light             x, y, z per pixel
```

### Approaches to Depth Estimation

| Method | Input | How | Accuracy | Use Case |
|--------|-------|-----|----------|----------|
| **Stereo vision** | 2 cameras | Triangulation from disparity | High | Industrial robots |
| **Structured light** | Projector + camera | Decode projected patterns | Very high | RGB-D sensors (RealSense) |
| **LiDAR** | Laser scanner | Time-of-flight | Very high | Autonomous vehicles |
| **Monocular depth** | 1 camera | Neural network | Medium | Any camera → depth |

### Monocular Depth Estimation with ViTs

The key insight: ViTs with global attention can reason about depth cues that span the entire image — vanishing points, relative sizes, occlusions, atmospheric perspective.

**MiDaS (2020):** Multi-dataset training for robust monocular depth:

$$d = f_\theta(\text{image}) \quad \text{where } d \in \mathbb{R}^{H \times W}$$

Trained on a mixture of datasets with different depth representations (metric, relative, stereo) using **scale-and-shift invariant loss**:

$$\mathcal{L} = \frac{1}{n} \sum_i \left( \frac{d_i - \text{median}(d)}{\text{MAD}(d)} - \frac{d_i^* - \text{median}(d^*)}{\text{MAD}(d^*)} \right)^2$$

**Depth Anything (2024):** Scales monocular depth to 62M unlabeled images via self-training:
1. Train teacher on labeled data
2. Generate pseudo-labels for unlabeled data
3. Train student on labeled + pseudo-labeled data
4. Student surpasses teacher

### DPT: Dense Prediction Transformer

DPT adapts ViT for dense prediction (depth, segmentation) by reassembling multi-scale features:

```
ViT Encoder (layer outputs at L/4, L/2, 3L/4, L)
       │
       ▼
Reassemble: project tokens back to spatial maps
       │
       ▼
Fusion: progressive upsampling with skip connections
       │
       ▼
Head: per-pixel depth prediction at full resolution
```

---

## Implementation (60 min)

### Using MiDaS / Depth Anything

```python
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


def estimate_depth_midas(image_path):
    """Monocular depth estimation with MiDaS v3."""
    # Load model
    model = torch.hub.load('intel-isl/MiDaS', 'DPT_Large')
    model.eval()
    
    # Load transforms
    midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
    transform = midas_transforms.dpt_transform
    
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(np.array(img)).unsqueeze(0)
    
    with torch.no_grad():
        depth = model(input_tensor)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=img.size[::-1],
            mode='bicubic',
            align_corners=False,
        ).squeeze()
    
    return depth.numpy()


def estimate_depth_anything(image_path):
    """Monocular depth with Depth Anything V2."""
    from transformers import pipeline
    
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
    )
    
    image = Image.open(image_path)
    result = pipe(image)
    depth = np.array(result["depth"])
    
    return depth


def visualize_depth(image_path, depth, title="Depth Estimation"):
    """Side-by-side visualization of RGB and depth."""
    img = Image.open(image_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(img)
    axes[0].set_title('RGB Image')
    axes[0].axis('off')
    
    im = axes[1].imshow(depth, cmap='inferno')
    axes[1].set_title(title)
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('depth_estimation.png', dpi=150)
```

### RGB-D to Point Cloud

```python
def depth_to_pointcloud(rgb, depth, fx=500, fy=500, cx=None, cy=None):
    """Convert RGB + depth map to 3D point cloud.
    
    Args:
        rgb: (H, W, 3) uint8 image
        depth: (H, W) depth map in meters
        fx, fy: focal lengths
        cx, cy: principal point (defaults to image center)
    
    Returns:
        points: (N, 3) xyz coordinates
        colors: (N, 3) rgb colors normalized to [0, 1]
    """
    H, W = depth.shape
    if cx is None:
        cx = W / 2
    if cy is None:
        cy = H / 2
    
    # Create pixel coordinate grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Back-project to 3D
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Filter invalid depths
    valid = (z > 0) & (z < 10.0)  # reasonable range
    
    points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    colors = rgb[valid].astype(np.float32) / 255.0
    
    return points, colors


def save_pointcloud_ply(points, colors, filename='scene.ply'):
    """Save point cloud as PLY file."""
    N = points.shape[0]
    header = (
        f"ply\nformat ascii 1.0\n"
        f"element vertex {N}\n"
        f"property float x\nproperty float y\nproperty float z\n"
        f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"end_header\n"
    )
    
    colors_uint8 = (colors * 255).astype(np.uint8)
    with open(filename, 'w') as f:
        f.write(header)
        for i in range(N):
            f.write(f"{points[i,0]:.4f} {points[i,1]:.4f} {points[i,2]:.4f} "
                    f"{colors_uint8[i,0]} {colors_uint8[i,1]} {colors_uint8[i,2]}\n")
    print(f"Saved {N} points to {filename}")
```

---

## Exercise (45 min)

1. **Depth estimation comparison:** Run both MiDaS and Depth Anything on the same 5 images. Compare depth maps visually. Which handles edges better? Which is faster?

2. **Indoor scene reconstruction:** Take an RGB image of your room. Estimate depth → create point cloud → save as PLY → view in MeshLab or Open3D. How accurate does the 3D reconstruction look?

3. **Depth for obstacle avoidance:** Given a depth map from a robot's camera, write a function that returns the closest obstacle distance and its (x, y) position in the image. This is the foundation for reactive navigation.

---

## Key Takeaways

1. **Monocular depth from ViTs.** Global attention enables reasoning about depth cues across the whole image
2. **Scale-invariant training.** Multi-dataset training with normalized loss enables robust generalization
3. **Depth → 3D.** With camera intrinsics, depth maps become point clouds for robotic reasoning
4. **Depth Anything scales.** Self-training on 62M unlabeled images pushes quality boundaries
5. **Robotics pipeline.** RGB → depth → point cloud → 3D reasoning is a core robot perception stack

## Connection to the Thread

> *Depth estimation adds the third dimension to our vision pipeline. Tomorrow: processing 3D data directly with point cloud transformers — the representation robots actually use for manipulation.*

---

## Further Reading

- [MiDaS v3: DPT (Ranftl et al., 2021)](https://arxiv.org/abs/2103.13413)
- [Depth Anything (Yang et al., 2024)](https://arxiv.org/abs/2401.10891)
- [Depth Anything V2 (Yang et al., 2024)](https://arxiv.org/abs/2406.09414)
- Open3D documentation: open3d.org
