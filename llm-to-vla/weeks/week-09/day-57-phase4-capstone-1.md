# Day 57: Phase IV Capstone — Day 1

> **Phase IV — Vision: ViT, 3D, Video | Week 9 | 2.5 hours**
> *"A robot needs to see in 2D, estimate depth, detect objects, and track them — today you build that perception stack." — Capstone project*

## Navigation
- **Previous:** [Day 56: Vision-Language Bridge](../week-08/day-56-vision-language-bridge.md)
- **Next:** [Day 58: Phase IV Capstone Day 2](day-58-phase4-capstone-2.md)
- **Week:** [Week 9 Overview](README.md)
- **Phase:** [Phase IV: Vision](../../study-notes/08-vision-transformers.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Project Overview

**Goal:** Build a multi-modal feature extractor that combines image classification, depth estimation, and object detection into a unified perception pipeline — the kind of stack a robot would use.

```
┌─────────────────────────────────────────────────────────────┐
│              Multi-Modal Perception Pipeline                 │
│                                                              │
│  Input: RGB Image                                           │
│    │                                                         │
│    ├──► ViT/DINOv2 Encoder ──► Global scene features        │
│    │                                                         │
│    ├──► Depth Anything ──► Depth map → 3D point cloud       │
│    │                                                         │
│    ├──► DETR / Florence-2 ──► Object bounding boxes         │
│    │                                                         │
│    └──► SAM 2 ──► Object masks (from DETR boxes)            │
│                                                              │
│  Output: Structured scene representation                     │
│    {                                                         │
│      "scene_embedding": tensor(1024),                       │
│      "depth_map": tensor(H, W),                             │
│      "objects": [                                            │
│        {"class": "cup", "box": [x1,y1,x2,y2],             │
│         "mask": tensor(H,W), "depth": 0.45,                │
│         "centroid_3d": [x, y, z]},                          │
│        ...                                                   │
│      ]                                                       │
│    }                                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: Scene Feature Extraction (30 min)

```python
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


class SceneEncoder:
    """Extract global scene features using DINOv2."""
    
    def __init__(self, model_name='dinov2_vits14'):
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.CenterCrop((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    
    def encode(self, image: Image.Image) -> dict:
        """Extract scene-level and patch-level features."""
        img_tensor = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            features = self.model.forward_features(img_tensor)
        
        return {
            'cls_token': features['x_norm_clstoken'][0],    # (D,)
            'patch_tokens': features['x_norm_patchtokens'][0],  # (N, D)
        }
```

## Step 2: Depth Estimation (30 min)

```python
class DepthEstimator:
    """Monocular depth estimation with Depth Anything V2."""
    
    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
        )
    
    def estimate(self, image: Image.Image) -> np.ndarray:
        """Return depth map as numpy array."""
        result = self.pipe(image)
        return np.array(result["depth"]).astype(np.float32)
    
    def to_pointcloud(self, rgb: np.ndarray, depth: np.ndarray,
                       fx=500, fy=500) -> np.ndarray:
        """Convert depth map to 3D point cloud."""
        H, W = depth.shape
        cx, cy = W / 2, H / 2
        
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        points = np.stack([x, y, z], axis=-1)  # (H, W, 3)
        return points
```

## Step 3: Object Detection + Segmentation (40 min)

```python
class ObjectDetector:
    """DETR-based object detection with SAM segmentation."""
    
    def __init__(self, det_threshold=0.7):
        from transformers import DetrForObjectDetection, DetrImageProcessor
        
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.model.eval()
        self.threshold = det_threshold
    
    def detect(self, image: Image.Image) -> list:
        """Detect objects and return bounding boxes."""
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold
        )[0]
        
        objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            objects.append({
                'class': self.model.config.id2label[label.item()],
                'score': score.item(),
                'box': box.tolist(),  # [x1, y1, x2, y2]
            })
        
        return objects


class PerceptionPipeline:
    """Full multi-modal perception pipeline."""
    
    def __init__(self):
        self.scene_encoder = SceneEncoder()
        self.depth_estimator = DepthEstimator()
        self.object_detector = ObjectDetector()
    
    def process(self, image_path: str) -> dict:
        """Run full perception pipeline on an image."""
        image = Image.open(image_path).convert("RGB")
        rgb = np.array(image)
        
        # 1. Scene features
        scene = self.scene_encoder.encode(image)
        
        # 2. Depth estimation
        depth = self.depth_estimator.estimate(image)
        points_3d = self.depth_estimator.to_pointcloud(rgb, depth)
        
        # 3. Object detection
        objects = self.object_detector.detect(image)
        
        # 4. Enrich objects with depth information
        for obj in objects:
            x1, y1, x2, y2 = [int(c) for c in obj['box']]
            # Average depth within bounding box
            obj_depth = depth[y1:y2, x1:x2]
            obj['median_depth'] = float(np.median(obj_depth))
            
            # 3D centroid
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if 0 <= cy < points_3d.shape[0] and 0 <= cx < points_3d.shape[1]:
                obj['centroid_3d'] = points_3d[cy, cx].tolist()
        
        return {
            'scene_embedding': scene['cls_token'].numpy(),
            'patch_features': scene['patch_tokens'].numpy(),
            'depth_map': depth,
            'point_cloud': points_3d,
            'objects': objects,
        }
```

## Step 4: Visualization (20 min)

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_perception(image_path, result):
    """Visualize the full perception pipeline output."""
    image = Image.open(image_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # RGB with detections
    axes[0].imshow(image)
    for obj in result['objects']:
        x1, y1, x2, y2 = obj['box']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                  linewidth=2, edgecolor='lime', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(x1, y1-5, f"{obj['class']}: {obj['score']:.2f}",
                     color='white', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))
    axes[0].set_title(f"Detection ({len(result['objects'])} objects)")
    axes[0].axis('off')
    
    # Depth map
    axes[1].imshow(result['depth_map'], cmap='inferno')
    axes[1].set_title('Depth Estimation')
    axes[1].axis('off')
    
    # Scene embedding PCA (first 3 components as RGB)
    patches_feat = result['patch_features']
    h = w = int(patches_feat.shape[0] ** 0.5)
    centered = patches_feat - patches_feat.mean(0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pca = centered @ Vt[:3].T
    pca = (pca - pca.min(0)) / (pca.max(0) - pca.min(0) + 1e-8)
    axes[2].imshow(pca.reshape(h, w, 3))
    axes[2].set_title('DINOv2 Feature PCA')
    axes[2].axis('off')
    
    plt.suptitle('Multi-Modal Perception Pipeline', fontsize=14)
    plt.tight_layout()
    plt.savefig('perception_pipeline.png', dpi=150)
    print("Saved perception_pipeline.png")
```

---

## Deliverables

By end of Day 57:
- [ ] Working `PerceptionPipeline` class with all 3 modules
- [ ] Tested on at least 5 images
- [ ] Visualization output saved
- [ ] Measured inference time per image

Tomorrow: evaluation, ablation, and Phase IV checkpoint questions.

---

## Key Takeaways

1. **Modular perception.** Each module (scene, depth, detection) runs independently but combines into rich scene understanding
2. **Depth enriches detection.** Adding median depth per object creates 3D-aware perception
3. **Feature reuse.** DINOv2 patch features serve both classification and spatial understanding
4. **Pipeline design.** Clean interfaces between modules enable swapping (e.g., DETR → Florence-2)

## Connection to the Thread

> *This perception pipeline is exactly what a VLA needs to see the world. Tomorrow you'll evaluate it and answer Phase IV checkpoint questions before moving to Vision-Language Models.*
