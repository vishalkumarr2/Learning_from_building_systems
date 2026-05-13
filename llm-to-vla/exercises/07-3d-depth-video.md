# Exercise 07 — 3D Vision, Depth Estimation & Video Understanding

> Phase IV · Days 51–55 · ~6 hours
> Prerequisites: Study note 09-3d-video-detection, Exercise 06 (ViT + DINO)
> Goal: Run depth estimation, create point clouds, explore video models, use Florence-2

---

## Setup

```python
# Core dependencies
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Additional packages needed:
# pip install transformers timm open3d
# pip install depth-anything-v2  # or use HF transformers pipeline
```

---

## Exercise 1: Depth Estimation (1.5 hours)

### 1.1 Run Depth Anything v2

```python
def run_depth_estimation(image_path):
    """
    Run Depth Anything v2 on an image and visualize the depth map.

    TODO:
    1. Load the Depth Anything v2 model from Hugging Face
       - Use: "depth-anything/Depth-Anything-V2-Small-hf" (or Base)
       - Or use the transformers depth-estimation pipeline
    2. Run inference on the input image
    3. Get the depth map as a numpy array
    4. Visualize side-by-side:
       a. Original RGB image
       b. Depth map (use 'plasma' or 'inferno' colormap)
       c. Depth histogram (distribution of depth values)
    5. Print depth statistics: min, max, mean, median

    Returns:
        depth_map: numpy array (H, W) with depth values
    """
    pass

# Test with different images
# Helper: download a sample warehouse/indoor image if not present
def get_warehouse_image(path="warehouse_scene.jpg"):
    """Download a sample indoor scene image or generate a placeholder."""
    import urllib.request, os
    if not os.path.exists(path):
        # Download a Creative Commons indoor scene image
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9c/Golden_Gate_bridge.jpg/800px-Golden_Gate_bridge.jpg"
        try:
            urllib.request.urlretrieve(url, path)
            print(f"Downloaded sample image → {path}")
        except Exception:
            # Fallback: generate a synthetic indoor-like image
            from PIL import Image
            import numpy as np
            img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            img.save(path)
            print(f"Generated synthetic placeholder → {path}")
    return path

depth_map = run_depth_estimation(get_warehouse_image())
# Try: indoor scene, outdoor scene, close-up, wide shot
```

### 1.2 Depth Map Analysis

```python
def analyze_depth_regions(image_path, depth_map):
    """
    Analyze depth at specific regions of interest.

    TODO:
    1. Display the image and let user click points (or define ROIs manually)
    2. For each ROI: compute mean depth, std depth
    3. Create a "depth profile" along a horizontal/vertical line through the image
       - Plot: pixel position (x-axis) vs depth (y-axis)
    4. Identify: nearest point, farthest point, depth discontinuities (edges)

    This mimics how a robot would use depth:
    - "How far is the nearest obstacle?"
    - "What's the depth of the object I want to grasp?"
    - "Is the path ahead clear?"
    """
    pass
```

### 1.3 Depth Map → Point Cloud

```python
def depth_to_pointcloud(rgb_image, depth_map, fx=500, fy=500, cx=None, cy=None):
    """
    Convert RGB image + depth map into a colored 3D point cloud.

    Args:
        rgb_image: numpy array (H, W, 3), values 0-255
        depth_map: numpy array (H, W), depth in relative units
        fx, fy: focal lengths (use approximate values if unknown)
        cx, cy: principal point (default: image center)

    Returns:
        points: numpy array (N, 3) — 3D coordinates
        colors: numpy array (N, 3) — RGB colors normalized to [0, 1]

    TODO:
    1. Create meshgrid of pixel coordinates (u, v)
    2. Back-project each pixel to 3D using:
       X = (u - cx) * depth / fx
       Y = (v - cy) * depth / fy
       Z = depth
    3. Stack into (N, 3) array
    4. Filter out invalid depths (zero, inf, nan)
    5. Get corresponding RGB colors for each valid point
    """
    pass

# Create point cloud
img = np.array(Image.open(get_warehouse_image()))
points, colors = depth_to_pointcloud(img, depth_map)
print(f"Point cloud: {points.shape[0]} points")
```

### 1.4 Visualize Point Cloud

```python
def visualize_pointcloud_matplotlib(points, colors, subsample=10):
    """
    Visualize point cloud using matplotlib's 3D scatter plot.

    TODO:
    1. Subsample points for faster rendering (every Nth point)
    2. Create 3D scatter plot
    3. Color each point by its RGB value
    4. Set axis labels (X, Y, Z in meters)
    5. Try different viewing angles

    Bonus: Use Open3D for interactive visualization if available
    """
    pass

visualize_pointcloud_matplotlib(points, colors)
```

```python
def visualize_pointcloud_open3d(points, colors):
    """
    Interactive 3D visualization using Open3D.

    TODO:
    1. Create Open3D PointCloud object
    2. Set points and colors
    3. Optionally: estimate normals for better rendering
    4. Visualize with draw_geometries
    """
    try:
        import open3d as o3d
        # TODO: implement
        pass
    except ImportError:
        print("Open3D not installed. Use: pip install open3d")
        print("Falling back to matplotlib visualization")
        visualize_pointcloud_matplotlib(points, colors)

visualize_pointcloud_open3d(points, colors)
```

### 1.5 Compare Depth Methods

```python
def compare_depth_models(image_path):
    """
    Compare different depth estimation approaches on the same image.

    TODO:
    1. Run Depth Anything v2 (Small and Base)
    2. Run MiDaS v3.1 (if available)
    3. Visualize all depth maps side-by-side
    4. Compute correlation between different models' depth maps
    5. Discuss: where do they agree? Where do they disagree?
       - Reflective surfaces? Transparent objects? Far distances?
    """
    pass

compare_depth_models(get_warehouse_image())
```

---

## Exercise 2: Video Understanding (1.5 hours)

### Video Setup

> **Note**: Exercises below reference video files. Use this helper to generate a synthetic video
> if you don't have real robot footage. Requires `opencv-python` (`pip install opencv-python`).

```python
import cv2

def get_sample_video(filename="sample_video.mp4", n_frames=30):
    """Generate a synthetic video for exercise purposes."""
    import cv2
    import numpy as np
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 10, (320, 240))
    for i in range(n_frames):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Moving circle simulates a robot arm
        cx = int(160 + 100 * np.sin(2 * np.pi * i / n_frames))
        cy = int(120 + 60 * np.cos(2 * np.pi * i / n_frames))
        cv2.circle(frame, (cx, cy), 20, (0, 255, 0), -1)
        out.write(frame)
    out.release()
    return filename
```

### 2.1 Frame Extraction and Temporal Analysis

```python
def extract_and_analyze_frames(video_path, sample_rate=5):
    """
    Extract frames from a video and analyze temporal changes.

    TODO:
    1. Load video using OpenCV (cv2.VideoCapture)
    2. Extract every `sample_rate`-th frame
    3. For each consecutive pair of frames:
       a. Compute frame difference (absolute pixel difference)
       b. Compute mean intensity change
    4. Plot: frame index vs mean change → identifies "events" in the video
    5. Visualize the top-3 highest-change frame pairs (moments of action)

    This is a simple but effective way to detect "when something happens"
    without any deep learning.
    """
    pass

# extract_and_analyze_frames(get_sample_video("robot_operation.mp4"))
```

### 2.2 Simple Motion Detection

```python
def motion_detection(video_path, threshold=30):
    """
    Implement simple frame-difference motion detection.

    TODO:
    1. Read consecutive frame pairs
    2. Convert to grayscale
    3. Compute absolute difference
    4. Threshold to get binary motion mask
    5. Find contours of moving regions
    6. Draw bounding boxes around moving objects
    7. Visualize: original frame with motion bounding boxes

    This is the simplest "video understanding" — detect where motion occurs.
    Compare with what a transformer-based approach would give you.
    """
    pass

# motion_detection(get_sample_video("robot_operation.mp4"))
```

### 2.3 DINOv2 Temporal Feature Analysis

```python
def temporal_feature_analysis(frames, model, processor):
    """
    Analyze how DINOv2 features change over time in a video.

    TODO:
    1. Extract DINOv2 CLS features for each frame
    2. Compute cosine similarity between consecutive frames' features
    3. Plot: frame index vs feature similarity
       - High similarity = little change between frames
       - Low similarity = something significant changed
    4. Compare with simple pixel-level frame difference (Exercise 2.1)

    Question: Does feature-level change detection find different
    "events" than pixel-level change detection? Why?
    """
    pass
```

### 2.4 VideoMAE Exploration (Optional — requires GPU)

```python
def run_video_mae(video_path, n_frames=16):
    """
    Run VideoMAE for action recognition on a short video clip.

    TODO:
    1. Load VideoMAE model from Hugging Face
       - "MCG-NJU/videomae-base-finetuned-kinetics"
    2. Sample n_frames uniformly from the video
    3. Preprocess and run inference
    4. Get top-5 predicted action labels with confidence scores
    5. Discuss: do the predictions make sense for robot operations?

    Note: VideoMAE expects Kinetics-400 action classes, which may not
    include robot-specific actions. The features are still useful
    even if the class labels don't match.
    """
    pass

# run_video_mae(get_sample_video("robot_picking.mp4", n_frames=16))
```

---

## Exercise 3: Object Detection with Florence-2 (2 hours)

### 3.1 Basic Object Detection

```python
def florence2_detect(image_path):
    """
    Run Florence-2 object detection on a warehouse scene.

    TODO:
    1. Load Florence-2 model and processor
       - "microsoft/Florence-2-base" or "microsoft/Florence-2-large"
    2. Run object detection with prompt "<OD>"
    3. Parse the output: extract bounding boxes and labels
    4. Draw bounding boxes on the image with labels and confidence
    5. Print detection summary: count of each object type

    Returns:
        detections: list of {label, bbox, score}
    """
    pass

detections = florence2_detect(get_warehouse_image())
for d in detections:
    print(f"  {d['label']}: bbox={d['bbox']}, score={d['score']:.2f}")
```

### 3.2 Compare Florence-2 Modes

```python
def florence2_multi_task(image_path):
    """
    Run Florence-2 in multiple modes on the same image.

    TODO: For each mode, run inference and visualize results:
    1. "<OD>" — Object Detection → bounding boxes
    2. "<CAPTION>" — Image Captioning → text description
    3. "<DETAILED_CAPTION>" — Detailed Caption → longer description
    4. "<MORE_DETAILED_CAPTION>" — Very detailed description
    5. "<OCR>" — Text recognition (if text visible)
    6. "<CAPTION_TO_PHRASE_GROUNDING>the robot" — Visual grounding

    Create a summary figure showing all results for one image.
    Discuss: how do different modes complement each other for robot perception?
    """
    pass

florence2_multi_task(get_warehouse_image())
```

### 3.3 Visual Grounding — Language-Guided Detection

```python
def florence2_grounding(image_path, queries):
    """
    Use Florence-2 to find specific objects described in natural language.

    Args:
        image_path: path to image
        queries: list of text descriptions, e.g.:
            ["the robot", "the red box", "the top shelf", "the nearest obstacle"]

    TODO:
    1. For each query, run Florence-2 with CAPTION_TO_PHRASE_GROUNDING
    2. Parse the output to get bounding box for the described object
    3. Visualize all grounded objects on the same image with different colors
    4. Report: which queries succeed? Which fail?

    This is powerful for robotics:
    "Pick up THE RED BOX on THE TOP SHELF"
    → Florence-2 grounds "red box" and "top shelf" to specific locations
    """
    pass

queries = [
    "the robot",
    "the nearest box",
    "the shelf on the left",
    "the floor",
]
florence2_grounding(get_warehouse_image(), queries)
```

### 3.4 Detection Accuracy Measurement

```python
def measure_detection_accuracy(image_paths, ground_truth):
    """
    Measure Florence-2 detection accuracy on a small test set.

    TODO:
    1. For each image, run Florence-2 detection
    2. Compare with ground truth bounding boxes
    3. Compute IoU (Intersection over Union) for each match
    4. Compute:
       - Precision: of detected objects, how many are correct?
       - Recall: of real objects, how many were detected?
       - mAP@0.5: mean Average Precision at IoU threshold 0.5
    5. Identify: what types of objects are hardest to detect?

    If no ground truth available:
    - Manually annotate 5-10 images (draw boxes)
    - Or use this as a qualitative assessment
    """
    pass
```

---

## Self-Check

After completing all exercises, verify:

- [ ] **Depth estimation**: Successfully ran Depth Anything v2, visualized depth maps
- [ ] **Point cloud**: Converted depth → 3D point cloud, visualized in 3D
- [ ] **Video analysis**: Extracted frames, detected motion, analyzed temporal features
- [ ] **Florence-2 detection**: Ran object detection on warehouse scenes
- [ ] **Multi-task**: Explored detection, captioning, and grounding modes
- [ ] **Grounding**: Used natural language to locate specific objects

### Key Questions

1. Why is monocular depth estimation called an "ill-posed problem"? What cues does the model use?
2. How would you combine depth + detection to get 3D object positions for a robot?
3. What's the advantage of Florence-2's unified approach vs separate models for each task?
4. For a real robot, which is more important: depth accuracy or detection accuracy? Why?

---

## Stretch Goals

1. **Depth + Detection fusion**: Combine your depth map with Florence-2 detections to estimate 3D positions for each detected object. Visualize as labeled points in the point cloud.

2. **Video object tracking**: Use DINOv2 features to track an object across video frames (match features frame-to-frame). Compare with simple optical flow tracking.

3. **Real-time pipeline**: Build a pipeline that processes webcam frames in real-time: depth estimation + detection + display. Measure FPS. What's the bottleneck?

4. **Multi-camera fusion**: If you have images from two different viewpoints, use depth maps from both to create a combined 3D scene. How does multi-view improve coverage?

5. **SAM 2 exploration**: If GPU resources allow, try SAM 2 for video segmentation. Click on an object in frame 1, track it through the video. How robust is it to occlusion?

---

*Next: Project 04 — [Robot Perception Pipeline](../projects/04-robot-perception/README.md)*
