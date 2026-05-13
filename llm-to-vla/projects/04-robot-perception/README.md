# Project 04 — Robot Perception Pipeline

> Phase IV Capstone · Days 56–58 · ~10 hours
> Prerequisites: Exercises 06 (ViT + DINO) and 07 (Depth + Detection)

---

## Objective

Build an end-to-end perception pipeline for a warehouse robot that combines
depth estimation, object detection, and simple tracking to produce
**3D object locations** from a camera feed.

This is the minimal viable perception system a robot needs to operate.

---

## Pipeline Architecture

```
Camera Frame (RGB)
       ↓
┌─────────────────────────────────────────┐
│  1. Depth Estimation (Depth Anything v2) │
│     → per-pixel depth map               │
├─────────────────────────────────────────┤
│  2. Object Detection (Florence-2)        │
│     → 2D bounding boxes + class labels   │
├─────────────────────────────────────────┤
│  3. 3D Localization (depth + detection)  │
│     → (x, y, z) position per object      │
├─────────────────────────────────────────┤
│  4. Simple Tracking (across frames)      │
│     → consistent object IDs over time    │
└─────────────────────────────────────────┘
       ↓
Output: List of tracked 3D objects
  { id, class, position_3d, velocity, confidence }
```

---

## Input

- **Minimum**: a set of static images (warehouse, robot lab, or any indoor scene)
- **Preferred**: a short video clip from a robot camera or a webcam walkthrough
- **Ideal**: ROS bag file with camera topic (if available from OKS attachments)

If no robot data is available, use:
- Indoor scene videos from YouTube
- KITTI or NYU Depth v2 dataset samples
- Your own webcam recording of an indoor space

---

## Deliverables

### 1. Pipeline Code (`pipeline.py`)

Single Python file (or module) implementing `RobotPerceptionPipeline`:

```python
class RobotPerceptionPipeline:
    def __init__(self, config):
        """Load depth, detection, and feature models."""

    def process_frame(self, rgb_frame) -> list[Detection3D]:
        """Process one frame → list of 3D detections."""

    def process_video(self, frames) -> list[list[TrackedObject]]:
        """Process video → tracked 3D objects over time."""
```

### 2. Evaluation Results (`evaluation.md`)

Measure and report:

| Metric | Description | Your Result |
|--------|-------------|-------------|
| Detection count accuracy | Correct object count per frame | |
| Depth consistency | Std dev of depth for static objects across frames | |
| Tracking continuity | % of objects tracked without ID switch | |
| Processing latency | ms per frame (total and per-component) | |
| FPS | Frames per second end-to-end | |

### 3. Visualization (`visualize.py`)

Generate visual outputs:
- Annotated frames: bounding boxes + depth coloring + object IDs
- 3D scatter plot: detected objects in 3D space
- Timeline: tracked object positions over time
- Component timing: bar chart of latency per pipeline stage

### 4. Analysis Write-Up (`analysis.md`)

Short document (~1 page) covering:
- What works well and what doesn't
- Bottleneck analysis (which component is slowest?)
- Failure modes (what types of objects/scenes cause errors?)
- Potential improvements for a real robot deployment

---

## Evaluation Criteria

| Criterion | Weight | Pass Threshold |
|-----------|--------|----------------|
| Pipeline runs end-to-end | 30% | Processes frames without crashing |
| Detection quality | 20% | Detects most visible objects |
| Depth integration | 20% | 3D positions are plausible |
| Tracking works | 15% | Objects maintain IDs for ≥5 frames |
| Visualization quality | 15% | Clear, informative output |

---

## Suggested Timeline

| Day | Focus | Output |
|-----|-------|--------|
| 56 | Build pipeline: depth + detection + 3D projection | `pipeline.py` working on single frames |
| 57 | Add tracking, process video, generate visualizations | `visualize.py`, video output |
| 58 | Evaluate, analyze failures, write up results | `evaluation.md`, `analysis.md` |

---

## Tips

- Start with single-frame processing before adding tracking
- Use the smallest model variants first (speed > accuracy for prototyping)
- Camera intrinsics: if unknown, use approximate values (fx=fy=500, cx=W/2, cy=H/2)
- For tracking: start with simple IoU-based matching between frames (no need for DeepSORT)
- Profile each component separately to identify the bottleneck
- Relative depth is fine — you don't need metric depth for the project to work

---

## Connection to the Curriculum

This project proves that **a single camera + foundation models** gives a robot
meaningful 3D understanding of its environment. The same DINOv2 features that
classify images (Exercise 06) power the depth estimation. The same transformer
architecture from Phase III drives the detection model.

In Phase V, we'll add language: "pick up the red box" will be grounded
into the 3D detections this pipeline produces.

---

*Next phase: [Phase V — Vision-Language Models](../../study-notes/10-vision-language-models.md)*
