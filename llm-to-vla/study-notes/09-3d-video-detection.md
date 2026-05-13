# 09 — 3D Vision, Video Understanding & Detection

> Phase IV · Days 51–58 · ~20 hours
> Prerequisites: 08-vision-transformers
> Learning Objectives: Understand depth estimation, point clouds, video models, modern detection, why robots need 3D+temporal vision

---

## Why This Matters

A robot doesn't live in a 2D photograph. It lives in a 3D world that changes over time.

Study note 08 showed us how transformers process single images.
Now we add the dimensions that robots actually need:
- **Depth** — how far is everything?
- **3D structure** — what's the shape of the world?
- **Time** — what's happening? What happened before? What will happen next?
- **Detection** — where exactly is each object?

These capabilities turn a "camera that understands pictures" into a
"perception system that understands the world."

---

## 1. Depth Estimation (Day 51)

### 1.1 Why Robots NEED Depth

A 2D image tells you WHAT is in the scene. Depth tells you WHERE.

```
Without depth:                    With depth:
"There's a box"                   "There's a box 1.2m ahead, 0.3m to the right"
"There's an obstacle"             "Obstacle 0.5m away — STOP!"
"There's a shelf"                 "Shelf at arm's reach — can grasp from here"
```

Robot tasks that require depth:
- **Navigation**: obstacle distance → path planning
- **Manipulation**: object distance → gripper control
- **Collision avoidance**: nearest obstacle distance → emergency stop
- **Mapping**: depth + pose → 3D map of environment

### 1.2 How Depth is Obtained

```
Method 1: Hardware (direct measurement)
─────────────────────────────────────────
Stereo cameras    → triangulation from two viewpoints
LiDAR             → time-of-flight of laser pulses
Structured light  → project pattern, measure distortion (Intel RealSense)
ToF cameras       → time-of-flight of IR light

Method 2: Software (estimation from single image)
─────────────────────────────────────────
Monocular depth   → neural network predicts depth from ONE image
                    No special hardware needed!
                    Works with any camera!
```

### 1.3 Monocular Depth Estimation

The ill-posed problem: from a single 2D image, estimate the 3D depth at every pixel.

Humans do this constantly — we judge distance from:
- Object size (smaller = farther)
- Perspective lines (converging = farther)
- Occlusion (behind = farther)
- Texture gradient (finer = farther)
- Atmospheric haze (hazier = farther)

Neural networks learn the same cues from data.

### 1.4 Depth Anything v2

State-of-the-art monocular depth estimation (Yang et al., 2024).

**Architecture:**
```
Input Image (H × W × 3)
        ↓
DINOv2 Encoder (frozen or fine-tuned)
  → multi-scale features from 4 stages
        ↓
DPT Decoder (Dense Prediction Transformer)
  → fuse multi-scale features
  → progressively upsample
        ↓
Depth Map (H × W × 1)
  → relative depth (not metric) or metric with scale head
```

**Key innovations:**
1. Uses DINOv2 as backbone → benefits from self-supervised pretraining
2. Trained on massive synthetic + real data
3. Two variants:
   - **Relative depth**: correct ordering, no metric scale
   - **Metric depth**: actual distances in meters

```python
import torch
from transformers import pipeline

# Simple usage with Hugging Face
depth_estimator = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf")

result = depth_estimator("warehouse_scene.jpg")
depth_map = result["depth"]  # PIL Image with depth values

# For robotics — metric depth is critical
# depth_map[y, x] = distance in meters from camera to that point
```

### 1.5 MiDaS (Older but Foundational)

MiDaS (Ranftl et al., 2020) established the paradigm:
- Train on MULTIPLE depth datasets with different scales
- Learn relative depth that generalizes across domains
- "Zero-shot" depth estimation on any image

**MiDaS → DPT → Depth Anything → Depth Anything v2**
Each builds on the previous, with better backbones and more data.

### 1.6 Stereo Depth

When you have two cameras (like human eyes):

```
Left camera image        Right camera image
┌──────────────┐        ┌──────────────┐
│      🤖      │        │    🤖        │
│              │        │              │
└──────────────┘        └──────────────┘
  Object at position x₁    Object at position x₂

Disparity: d = x₁ - x₂
Depth: Z = (focal_length × baseline) / d

Larger disparity → closer object
Smaller disparity → farther object
```

Stereo is more accurate than monocular for absolute distances,
but requires calibrated camera pairs.

### 1.7 Depth → Point Cloud

Depth maps are 2D arrays of distances. To get 3D coordinates:

```python
import numpy as np

def depth_to_pointcloud(depth_map, camera_intrinsics):
    """Convert depth map to 3D point cloud.

    Camera intrinsics: fx, fy (focal lengths), cx, cy (principal point)
    """
    fx, fy, cx, cy = camera_intrinsics
    H, W = depth_map.shape

    # Create pixel coordinate grids
    u = np.arange(W)  # column indices
    v = np.arange(H)  # row indices
    u, v = np.meshgrid(u, v)

    # Back-project to 3D
    # (u, v, depth) → (X, Y, Z) in camera frame
    Z = depth_map
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    # Stack into point cloud: (N, 3)
    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    # Filter invalid depths
    valid = Z.reshape(-1) > 0
    points = points[valid]

    return points  # Each row is (X, Y, Z) in meters
```

**Visualization:**
```python
import open3d as o3d

def visualize_pointcloud(points, colors=None):
    """Visualize point cloud in 3D."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    o3d.visualization.draw_geometries([pcd])
```

---

## 2. Point Clouds & 3D Representations (Day 52)

### 2.1 Point Cloud Basics

A point cloud is a set of 3D points: `{(x₁,y₁,z₁), (x₂,y₂,z₂), ..., (xₙ,yₙ,zₙ)}`

Key properties:
- **Unordered**: no inherent sequence (unlike images or text)
- **Irregular**: points are not on a grid
- **Sparse**: mostly empty space
- **Variable size**: different scenes have different numbers of points

### 2.2 PointNet: Deep Learning Directly on Point Sets

The challenge: how do you apply neural networks to unordered sets?

**PointNet (Qi et al., 2017) solution:**

```
Input: N points × 3 (x, y, z)
        ↓
Per-point MLP: shared MLP applied to EACH point independently
  (3 → 64 → 128 → 1024)
        ↓
N points × 1024 features
        ↓
Max Pooling across all points → 1 × 1024 global feature
        ↓
Classification / Segmentation head
```

**Key insight**: max pooling over point features creates a permutation-invariant
global feature. The output doesn't change if you reorder the input points.

```python
import torch
import torch.nn as nn

class PointNet(nn.Module):
    """Simplified PointNet for point cloud classification."""

    def __init__(self, num_classes=10):
        super().__init__()
        # Shared MLP (applied to each point independently)
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 1024), nn.ReLU()
        )
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: (B, N, 3) — batch of point clouds
        point_features = self.mlp1(x)       # (B, N, 1024)
        global_feature = point_features.max(dim=1)[0]  # (B, 1024) — permutation invariant!
        return self.classifier(global_feature)
```

### 2.3 PointNet++: Hierarchical Point Set Learning

PointNet treats all points globally — it misses local structure.

PointNet++ (Qi et al., 2017) adds hierarchy:

```
Input: N points
    ↓
Set Abstraction Layer 1:
  - Sample M₁ centroids (farthest point sampling)
  - For each centroid: gather nearby points within radius r₁
  - Apply PointNet to each local group → M₁ features
    ↓
Set Abstraction Layer 2:
  - Sample M₂ centroids from M₁
  - Gather local neighborhoods, apply PointNet
  - → M₂ features (more abstract)
    ↓
... repeat ...
    ↓
Global features for classification/segmentation
```

This is analogous to CNN's pooling hierarchy:
- Layer 1: local features (edges, surfaces)
- Layer 2: mid-level features (object parts)
- Layer 3: global features (whole objects)

### 2.4 NeRF: Neural Radiance Fields (Conceptual)

NeRF (Mildenhall et al., 2020) represents 3D scenes as a neural network:

```
Input: 3D position (x, y, z) + viewing direction (θ, φ)
        ↓
    MLP network
        ↓
Output: color (r, g, b) + density (σ)

Train by: rendering images from the NeRF and comparing with real photos
Result: implicit 3D representation that can render novel viewpoints
```

**Relevance to robotics:**
- 3D scene reconstruction from a few images
- Novel view synthesis for planning
- Scene understanding without explicit 3D reconstruction

### 2.5 Other 3D Representations

| Representation | Description | Pros | Cons |
|---------------|-------------|------|------|
| Point cloud | Set of (x,y,z) points | Simple, direct from sensors | Sparse, unstructured |
| Voxel grid | 3D grid of occupied/empty cells | Regular structure, easy conv | Memory-heavy (O(N³)) |
| Mesh | Vertices + triangles | Efficient surface repr. | Complex topology |
| Occupancy net | Neural network: (x,y,z) → occupied? | Continuous, memory-efficient | Slow inference |
| NeRF | Neural network: (x,y,z,θ,φ) → (r,g,b,σ) | Photorealistic rendering | Training-intensive |
| Gaussian splats | 3D Gaussians with color/opacity | Fast rendering, good quality | Large storage |

### 2.6 Why 3D Matters for Robots

```
2D perception:    "There's a box on the shelf"
3D perception:    "There's a box (20×15×10 cm) at position (1.2, 0.3, 0.8)m
                   with orientation 15° from horizontal.
                   The gripper can approach from above or from the right side.
                   Clearance to neighboring objects: 5cm left, 8cm right."
```

3D perception enables:
- **Grasp planning**: where and how to grip an object
- **Motion planning**: collision-free paths through 3D space
- **Place planning**: where an object fits on a surface
- **Scene understanding**: spatial relationships between objects

---

## 3. Video Understanding — Day 1 (Day 53)

### 3.1 Video = Temporal Sequence of Images

```
Image:    Single frame → spatial understanding
Video:    T frames    → spatial + TEMPORAL understanding

Frame t-2    Frame t-1    Frame t      Frame t+1    Frame t+2
┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐
│ 🤖→  │    │  🤖→ │    │   🤖→│    │    🤖│→   │     🤖│
│      │    │      │    │      │    │  📦  │    │  📦  │
└──────┘    └──────┘    └──────┘    └──────┘    └──────┘
                              "Robot approaching the box"
```

Understanding video means understanding:
- **What** is happening (action recognition)
- **When** it's happening (temporal localization)
- **Why** it's happening (causal reasoning)
- **What** will happen next (prediction)

### 3.2 Naive Approach: Process Each Frame Independently

```
Frame 1 → ViT → features_1    "Robot at position A"
Frame 2 → ViT → features_2    "Robot at position B"
Frame 3 → ViT → features_3    "Robot at position C"
...

Problem: No temporal reasoning! Can't understand motion or actions.
Each frame is processed in isolation.
```

### 3.3 TimeSformer: Factorized Space-Time Attention

TimeSformer (Bertasius et al., 2021) extends ViT to video with **divided attention**:

```
Video input: T frames × H × W → T × N patches per frame

Option 1: Joint Space-Time Attention (expensive)
  Every patch attends to EVERY other patch across ALL frames
  Complexity: O((T × N)²) — prohibitive!

Option 2: Divided Space-Time Attention (TimeSformer's approach)
  Step 1 — Temporal attention: each patch attends to same-position patches in OTHER frames
  Step 2 — Spatial attention: each patch attends to other patches in SAME frame
  Complexity: O(T × N² + N × T²) — much better!
```

```
Frame 1:  [p₁₁] [p₁₂] [p₁₃] ... [p₁ₙ]
Frame 2:  [p₂₁] [p₂₂] [p₂₃] ... [p₂ₙ]
Frame 3:  [p₃₁] [p₃₂] [p₃₃] ... [p₃ₙ]

Temporal attention (vertical): p₁₁ ↔ p₂₁ ↔ p₃₁  (same position, different times)
Spatial attention (horizontal): p₁₁ ↔ p₁₂ ↔ p₁₃  (same time, different positions)
```

### 3.4 Video MAE: Self-Supervised Video Understanding

VideoMAE (Tong et al., 2022) extends MAE to video:

```
Video: T frames × H × W
        ↓
Split into spatiotemporal "tubes" (patches across time)
        ↓
Mask 90% of tubes (!!)
        ↓
Encode visible 10%
        ↓
Decode → reconstruct masked tubes
        ↓
Loss: MSE on masked spatiotemporal patches
```

**Why 90% masking (vs MAE's 75%)?**
Video is even MORE redundant than images:
- Consecutive frames are nearly identical
- 90% masking forces the model to truly understand temporal dynamics

```python
# Conceptual VideoMAE masking
def create_tube_mask(T, H_patches, W_patches, mask_ratio=0.9):
    """Create consistent mask across frames (tube masking)."""
    n_spatial = H_patches * W_patches
    n_keep = int(n_spatial * (1 - mask_ratio))

    # Same spatial positions masked across ALL frames
    # This is "tube masking" — temporal consistency
    spatial_indices = torch.randperm(n_spatial)[:n_keep]

    mask = torch.zeros(T, H_patches, W_patches)
    for idx in spatial_indices:
        h, w = idx // W_patches, idx % W_patches
        mask[:, h, w] = 1  # Keep this position in ALL frames

    return mask
```

### 3.5 Video Transformers: The Scaling Challenge

Video = many frames × many patches per frame = VERY long sequences.

```
Single image (ViT): 196 tokens
8-frame video:      8 × 196 = 1,568 tokens
32-frame video:     32 × 196 = 6,272 tokens
1-second @ 30fps:   30 × 196 = 5,880 tokens

Self-attention on 6,272 tokens: 6,272² = ~39M operations
→ Need efficient attention (divided, windowed, sparse)
```

This is why factorized attention (TimeSformer) and tube masking (VideoMAE) matter.

---

## 4. Video Understanding — Day 2 (Day 54)

### 4.1 Optical Flow

Optical flow estimates per-pixel motion between consecutive frames:

```
Frame t:           Frame t+1:         Optical Flow:
┌──────────┐      ┌──────────┐      ┌──────────┐
│    🤖    │      │      🤖  │      │    →→    │  (arrows show motion)
│  📦      │      │  📦      │      │          │  (box is stationary)
└──────────┘      └──────────┘      └──────────┘

Flow(x, y) = (dx, dy) — how much pixel (x,y) moved between frames
```

**Classic methods:** Lucas-Kanade, Horn-Schunck, Farneback
**Neural methods:** RAFT (Teed & Deng, 2020) — state-of-the-art

```python
# Using RAFT for optical flow
import torch
from torchvision.models.optical_flow import raft_large

model = raft_large(pretrained=True)
model.eval()

# frame1, frame2: (1, 3, H, W) tensors
with torch.no_grad():
    flow_predictions = model(frame1, frame2)
    flow = flow_predictions[-1]  # (1, 2, H, W) — dx, dy per pixel
```

### 4.2 Action Recognition

Classifying what action is happening in a video:
- "Robot picking up object"
- "Robot navigating aisle"
- "Person walking across path"
- "Forklift turning corner"

**Approaches:**
1. **Frame-level features + temporal aggregation**: ViT per frame → pool/attention across frames
2. **Video transformers**: TimeSformer, ViViT — end-to-end space-time attention
3. **Two-stream**: RGB stream + optical flow stream → fuse predictions

### 4.3 VideoMAE v2

VideoMAE v2 (Wang et al., 2023) scales VideoMAE with:
- Dual masking: mask differently for encoder and decoder
- Larger models (ViT-g with 1B parameters)
- More video data
- State-of-the-art on action recognition benchmarks

### 4.4 Temporal Reasoning

Beyond "what action?" — understanding temporal structure:

```
"The robot picked up the box, then navigated to shelf B3, then placed it."

Temporal reasoning requires understanding:
1. Event ordering (pick → navigate → place)
2. Causality (picked up BECAUSE it was told to)
3. Duration (navigation takes longer than grasping)
4. State changes (box: on floor → in gripper → on shelf)
```

Current video models are good at action recognition but still struggle
with fine-grained temporal reasoning. This is an active research area.

### 4.5 Video Understanding for Robots

Why robots need video understanding (not just single frames):

```
Single frame:  "Robot is next to a box"
Two frames:    "Robot is moving toward a box"
Many frames:   "Robot approached, grasped, lifted, and is now carrying the box"
Full video:    "Robot completed a pick-and-place task with one retry after initial grasp failed"
```

Robot-specific video tasks:
- **Activity recognition**: what task is the robot performing?
- **Failure detection**: did the grasp fail? Did it collide with something?
- **Progress monitoring**: how far along is the current task?
- **Anomaly detection**: is something unusual happening?
- **Imitation learning**: learn actions from video demonstrations

---

## 5. Detection: DETR + Florence-2 (Day 55)

### 5.1 Object Detection: The Problem

```
Input:  Image of warehouse scene
Output: For each object:
        - Bounding box: (x, y, width, height)
        - Class label: "box", "shelf", "robot", "person"
        - Confidence: 0.95

┌─────────────────────────────────────┐
│                                     │
│    ┌─────────┐   ┌──────┐          │
│    │  robot   │   │ box  │          │
│    │  0.97    │   │ 0.94 │          │
│    └─────────┘   └──────┘          │
│                      ┌────────────┐ │
│                      │   shelf    │ │
│                      │   0.89    │ │
│                      └────────────┘ │
└─────────────────────────────────────┘
```

### 5.2 Traditional Detection Pipeline (Pre-DETR)

```
Image → Backbone (ResNet) → Feature Pyramid Network
    → Region Proposal Network → thousands of proposals
    → ROI pooling → per-proposal classification + bbox regression
    → Non-Maximum Suppression (NMS) → remove duplicates
    → Final detections

Problems:
- NMS is hand-designed, not learned
- Anchor boxes are hyperparameters that need tuning
- Two-stage pipeline is complex
- Many post-processing steps
```

### 5.3 DETR: End-to-End Object Detection

DETR (Carion et al., 2020) revolutionizes detection with transformers:

**No anchors. No NMS. No post-processing. Just a transformer.**

```
Image → CNN backbone → flattened features (sequence of tokens)
    → Transformer Encoder (self-attention on image features)
    → Transformer Decoder with N "object queries"
    → N predictions (class + bounding box)
    → Bipartite matching with ground truth
    → Hungarian loss
```

**Key innovation: Object Queries**

```
Object queries:  [q₁] [q₂] [q₃] ... [q₁₀₀]
                   ↓     ↓     ↓          ↓
Decoder:        Cross-attend to image features
                   ↓     ↓     ↓          ↓
Output:         [pred₁] [pred₂] [pred₃] ... [pred₁₀₀]

Each prediction: {class, bbox} or {∅ (no object)}
```

Object queries are **learned embeddings** that each learn to "look for" different things:
- Query 1 might specialize in large objects at image center
- Query 2 might specialize in small objects at image edges
- etc.

The decoder uses cross-attention: queries attend to image features
to "find" their objects.

```python
# Simplified DETR detection
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

image = Image.open("warehouse.jpg")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# Post-process: filter by confidence threshold
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(
    outputs, target_sizes=target_sizes, threshold=0.7
)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = box.tolist()
    print(f"Detected {model.config.id2label[label.item()]} "
          f"at [{box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f}] "
          f"with confidence {score:.2f}")
```

### 5.4 Florence-2: Unified Vision Foundation Model

Florence-2 (Xiao et al., 2024) is a single model that does EVERYTHING:

```
Florence-2 can:
- Object detection       ("Where are the objects?")
- Image captioning       ("Describe this image")
- Visual grounding       ("Where is the red box?")
- OCR                    ("What text is visible?")
- Segmentation           ("Segment the robot")
- Region description     ("What's in this box?")

All with the SAME model, SAME weights!
Just change the text prompt.
```

**Architecture:**
```
Image → DaViT encoder → image embeddings
                              ↓
Text prompt → tokenizer → text embeddings
                              ↓
                    Multimodal decoder
                              ↓
                    Text output (structured)

Example:
  Image: warehouse scene
  Prompt: "<OD>"  (object detection)
  Output: "robot<loc_102><loc_234><loc_456><loc_678>
           box<loc_500><loc_300><loc_600><loc_400>"
```

```python
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

image = Image.open("warehouse.jpg")

# Object detection
prompt = "<OD>"
inputs = processor(text=prompt, images=image, return_tensors="pt")
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
results = processor.batch_decode(generated_ids, skip_special_tokens=False)
parsed = processor.post_process_generation(results[0], task="<OD>", image_size=image.size)
# parsed contains bounding boxes and labels

# Image captioning
prompt = "<CAPTION>"
# ... same process, different prompt

# Visual grounding
prompt = "<CAPTION_TO_PHRASE_GROUNDING>The robot near the shelf"
# ... returns bounding boxes for "robot" and "shelf"
```

### 5.5 SAM 2: Segment Anything in Videos

SAM 2 (Ravi et al., 2024) extends the Segment Anything Model to video:

```
Single image (SAM 1):
  Point/box/text prompt → segment that object in ONE frame

Video (SAM 2):
  Point/box prompt on ONE frame → track and segment that object
  through ALL frames of the video

  Frame 1: Click on robot → segment robot
  Frame 2-100: SAM 2 automatically tracks and segments the robot
               even as it moves, rotates, gets partially occluded
```

**Relevance for robots:**
- Click once on an object → track it through the entire operation
- Segment objects for manipulation planning
- Track multiple objects simultaneously
- Handle occlusions (object goes behind shelf, reappears)

### 5.6 Detection vs Segmentation vs Grounding

```
Detection:      "Where are objects?"     → bounding boxes + classes
Segmentation:   "Which pixels = object?" → pixel-level masks
Grounding:      "Where is THE red box?"  → bbox for specific object described in text

┌─────────────────────┐
│ Detection:           │
│  ┌──────┐ ┌───┐    │  Bounding boxes
│  │robot │ │box│    │
│  └──────┘ └───┘    │
├─────────────────────┤
│ Segmentation:        │
│  ███████  ████      │  Pixel masks
│  ███████  ████      │
│  █████              │
├─────────────────────┤
│ Grounding:           │
│  "red box" → ┌───┐  │  Language-guided
│              │ ▓ │  │
│              └───┘  │
└─────────────────────┘
```

---

## 6. Phase IV Capstone: Robot Perception Pipeline (Days 56–58)

### 6.1 Putting It All Together

A complete robot perception pipeline combines everything from Phase IV:

```
Camera Feed (RGB frames)
        ↓
┌──────────────────────────────────────────────────────┐
│                  Perception Pipeline                   │
│                                                        │
│  1. Depth Estimation (Depth Anything v2)               │
│     → depth_map: per-pixel depth                       │
│                                                        │
│  2. Object Detection (Florence-2 or DETR)              │
│     → bounding boxes + class labels                    │
│                                                        │
│  3. Feature Extraction (DINOv2)                        │
│     → rich visual features for each object             │
│                                                        │
│  4. 3D Localization (depth + detection + intrinsics)   │
│     → 3D position (x, y, z) for each detected object  │
│                                                        │
│  5. Tracking (across frames)                           │
│     → consistent object IDs over time                  │
│                                                        │
└──────────────────────────────────────────────────────┘
        ↓
Output: For each object:
  - Class: "box", "shelf", "robot", "person"
  - 3D position: (x, y, z) in world frame
  - Velocity: (vx, vy, vz) from tracking
  - Features: DINOv2 embedding for downstream tasks
```

### 6.2 Pipeline Implementation Sketch

```python
class RobotPerceptionPipeline:
    """Combine depth, detection, and tracking for robot perception."""

    def __init__(self):
        self.depth_model = load_depth_anything_v2()
        self.detector = load_florence2()
        self.feature_extractor = load_dinov2()
        self.tracker = SimpleTracker()
        self.camera_intrinsics = load_camera_calibration()

    def process_frame(self, rgb_frame):
        """Process single frame → 3D object detections."""
        # 1. Depth
        depth_map = self.depth_model(rgb_frame)

        # 2. Detection
        detections = self.detector(rgb_frame)  # list of {bbox, class, score}

        # 3. 3D localization
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            depth = depth_map[int(cy), int(cx)]
            det['position_3d'] = pixel_to_3d(cx, cy, depth, self.camera_intrinsics)

        # 4. Features
        for det in detections:
            crop = rgb_frame[y1:y2, x1:x2]
            det['features'] = self.feature_extractor(crop)

        # 5. Track across frames
        tracked = self.tracker.update(detections)

        return tracked

    def process_video(self, frames):
        """Process video → tracked 3D objects over time."""
        all_tracked = []
        for frame in frames:
            tracked = self.process_frame(frame)
            all_tracked.append(tracked)
        return all_tracked
```

### 6.3 Evaluation Metrics

How to measure perception pipeline quality:

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| mAP@0.5 | Detection accuracy (IoU > 0.5) | > 0.70 |
| Depth RMSE | Depth estimation error (meters) | < 0.2m |
| Depth δ₁ | % of pixels with relative error < 1.25 | > 0.95 |
| MOTA | Multi-object tracking accuracy | > 0.60 |
| FPS | Processing speed | > 10 fps |
| Latency | Frame-to-output delay | < 100ms |

### 6.4 Real-World Challenges

```
Challenge                         Solution approach
─────────────────────────────────────────────────────────
Varying lighting                  → DINOv2 (robust features)
Reflective surfaces               → Multi-sensor fusion
Transparent objects               → Specialized depth models
Cluttered scenes                  → Florence-2 (handles clutter)
Moving camera (on robot)          → Ego-motion compensation
Real-time constraint              → Model optimization, TensorRT
Metric depth accuracy             → Depth Anything v2 metric model
Novel objects (never seen)        → DINO features + open-vocab detection
Partial occlusion                 → SAM 2 tracking through occlusion
```

---

## 7. Key Takeaways

### Technical
1. **Depth estimation** has become remarkably accurate with Depth Anything v2, powered by DINOv2 features
2. **Point clouds** give robots true 3D understanding — PointNet processes them as unordered sets
3. **Video transformers** use factorized attention to handle the quadratic cost of space-time sequences
4. **DETR** eliminates all hand-designed detection components with a pure transformer approach
5. **Florence-2** unifies detection, captioning, grounding, and segmentation in one model
6. **SAM 2** enables click-and-track segmentation through video

### Conceptual
1. Robots need **3D + temporal** vision, not just 2D image understanding
2. **Depth + detection + tracking** is the minimal viable perception pipeline
3. Video understanding is essentially **attention across space AND time**
4. Foundation models (Florence-2, SAM 2) are making robot perception more accessible

### For Robotics
1. **Depth Anything v2** gives metric depth from a single camera — no LiDAR needed for many tasks
2. **DINOv2 features** are the best starting point for any robot vision task
3. **Florence-2** can handle diverse perception needs with a single model
4. The perception pipeline is becoming modular: swap components as better models appear

---

## 8. Paper References

| Paper | Year | Key Contribution |
|-------|------|------------------|
| [PointNet](https://arxiv.org/abs/1612.00593) | 2017 | Deep learning on point sets |
| [PointNet++](https://arxiv.org/abs/1706.02413) | 2017 | Hierarchical point learning |
| [NeRF](https://arxiv.org/abs/2003.08934) | 2020 | Neural radiance fields |
| [RAFT](https://arxiv.org/abs/2003.12039) | 2020 | Recurrent optical flow |
| [DETR](https://arxiv.org/abs/2005.12872) | 2020 | End-to-end detection with transformers |
| [TimeSformer](https://arxiv.org/abs/2102.05095) | 2021 | Divided space-time attention |
| [VideoMAE](https://arxiv.org/abs/2203.12602) | 2022 | Masked autoencoders for video |
| [MiDaS v3.1](https://arxiv.org/abs/2307.14460) | 2023 | Robust monocular depth |
| [Depth Anything](https://arxiv.org/abs/2401.10891) | 2024 | Foundation model for depth |
| [Depth Anything v2](https://arxiv.org/abs/2406.09414) | 2024 | Improved depth estimation |
| [Florence-2](https://arxiv.org/abs/2311.06242) | 2024 | Unified vision foundation model |
| [SAM 2](https://arxiv.org/abs/2408.00714) | 2024 | Segment anything in videos |

---

## 9. Connection to the Thread

### Adding Dimensions

```
Phase I:    1D — text tokens in a sequence
Phase II:   1D — attention over token sequences
Phase III:  1D — scale attention with transformers
Phase IV-a: 2D — image patches as token sequences (study note 08)
Phase IV-b: 3D — depth adds the Z dimension
Phase IV-c: 4D — video adds the time dimension
```

### The Compression Continues

Robots need to compress **massive** sensory streams into actionable representations:

```
Raw camera: 640×480×3 @ 30fps = 27.6 MB/s
            → must compress to: "box at (1.2, 0.3, 0.8) moving left at 0.1 m/s"

That's a compression ratio of roughly 1,000,000:1
From millions of pixel values to a handful of meaningful numbers.

The perception pipeline IS this compression.
Depth estimation compresses appearance → distance.
Detection compresses pixels → bounding boxes.
Tracking compresses frame sequences → object trajectories.
```

### What's Next

We can now process:
- Text (Phases I-III)
- Images (Phase IV-a, study note 08)
- 3D + video (Phase IV-b, this study note)

Phase V combines text + vision: **Vision-Language Models**.
The robot will understand "pick up the red box" by jointly processing
the language instruction and the visual scene.

---

*Next: [10-vision-language-models.md](./10-vision-language-models.md) — Combining vision and language in one model*
