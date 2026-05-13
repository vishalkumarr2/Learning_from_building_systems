# Day 65: Spatial Grounding

> **Phase V — Vision-Language Models | Week 10 | 2.5 hours**
> *"A VLM that can say 'the cup is on the table' is useful. A VLM that can point to the cup's exact location is essential for robotics." — Visual grounding*

## Navigation
- **Previous:** [Day 64: Open VLM Landscape](day-64-open-vlm-landscape.md)
- **Next:** [Day 66: Stop & Reflect 4](day-66-stop-reflect-4.md)
- **Week:** [Week 10 Overview](README.md)
- **Phase:** [Phase V: Vision-Language Models](../../study-notes/10-vision-language-models.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### From Understanding to Pointing

Standard VLMs answer questions about images. Grounded VLMs **locate** what they describe:

```
Standard VLM:
  Q: "What is on the table?"
  A: "A red cup and a screwdriver."

Grounded VLM:
  Q: "What is on the table?"
  A: "A red cup <box>(0.2, 0.3, 0.4, 0.6)</box> and 
      a screwdriver <box>(0.5, 0.4, 0.8, 0.7)</box>."
```

### Visual Grounding Tasks

| Task | Input | Output |
|------|-------|--------|
| **Referring Expression Comprehension** | Image + "the red cup" | Bounding box |
| **Referring Expression Generation** | Image + box | "the red cup on the left" |
| **Phrase Grounding** | Image + sentence | Box per phrase |
| **Pointing** | Image + point (x,y) | Object description |
| **Region Captioning** | Image + box | Description of region |

### How VLMs Learn to Ground

**Approach 1: Coordinate tokens (text-based)**

PaLI and Qwen-VL tokenize coordinates as special text tokens:

```
Vocabulary expansion:
  <loc_000>, <loc_001>, ..., <loc_999>  (1000 position tokens)

Training example:
  Input:  "detect cup" + [image]
  Output: "<loc_200><loc_300><loc_400><loc_600>"  (x1, y1, x2, y2)

Coordinates normalized to [0, 999] → model learns to generate them
```

**Approach 2: Special tokens (Florence-2)**

Florence-2 uses `<loc_X>` tokens within natural language:

```
Input:  "Locate the coffee cup in the image"
Output: "The coffee cup<loc_210><loc_350><loc_445><loc_620> is on the table"
```

**Approach 3: Pointing (Ferret, Shikra)**

Support both input and output coordinates:

```
Input:  "What is at position (0.5, 0.3)?" (user clicks a point)
Output: "That's a screwdriver, specifically a Phillips head."

Input:  "Where is the power button?"
Output: "The power button is at (0.85, 0.12)."
```

### Coordinate Representations

Different models encode spatial information differently:

$$\text{Normalized box:} \quad (x_1, y_1, x_2, y_2) \in [0, 1]^4$$

$$\text{Discretized:} \quad \text{round}(x \times N_{\text{bins}}) \quad \text{where } N_{\text{bins}} \in \{100, 1000\}$$

$$\text{Center + size:} \quad (c_x, c_y, w, h) \quad \text{alternative parameterization}$$

### Why Grounding Matters for VLAs

A robot needs to ground language to physical locations:

```
"Pick up the red cup" → WHERE is the red cup?
  → VLM grounds "red cup" to pixel coordinates
  → Depth map converts pixels to 3D coordinates  
  → Robot controller moves to that 3D position

Without grounding:
  VLM: "I see a red cup" ← helpful but not actionable
  
With grounding:
  VLM: "red cup at (0.35, 0.42)" → camera_to_world() → [0.2m, -0.1m, 0.3m]
  → Robot can act!
```

---

## Implementation (60 min)

### Florence-2 Grounding

```python
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw
import torch


class GroundedVLM:
    """Florence-2 based visual grounding."""
    
    def __init__(self, model_name="microsoft/Florence-2-base"):
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, trust_remote_code=True
        )
        self.model.eval()
    
    def ground(self, image_path, text_query):
        """Find objects matching the text query."""
        image = Image.open(image_path).convert("RGB")
        prompt = f"<OPEN_VOCABULARY_DETECTION> {text_query}"
        
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            torch.float16
        )
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=1024,
                num_beams=3,
            )
        
        result = self.processor.batch_decode(output, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            result, task="<OPEN_VOCABULARY_DETECTION>", image_size=image.size
        )
        
        return parsed
    
    def caption_region(self, image_path, box):
        """Generate caption for a specific region."""
        image = Image.open(image_path).convert("RGB")
        prompt = f"<REGION_TO_DESCRIPTION>"
        
        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(torch.float16)
        
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=256)
        
        result = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        return result
    
    def visualize(self, image_path, detections):
        """Draw bounding boxes on image."""
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        
        if 'bboxes' in detections and 'labels' in detections:
            for box, label in zip(detections['bboxes'], detections['labels']):
                x1, y1, x2, y2 = box
                draw.rectangle([x1, y1, x2, y2], outline='lime', width=3)
                draw.text((x1, y1 - 15), label, fill='lime')
        
        image.save('grounded_output.png')
        print("Saved grounded_output.png")
        return image
```

### Coordinate Tokenization

```python
class CoordinateTokenizer:
    """Convert between pixel coordinates and text tokens."""
    
    def __init__(self, n_bins=1000):
        self.n_bins = n_bins
    
    def normalize_box(self, box, image_size):
        """Normalize pixel box to [0, 1]."""
        W, H = image_size
        x1, y1, x2, y2 = box
        return [x1/W, y1/H, x2/W, y2/H]
    
    def discretize(self, normalized_coords):
        """Convert [0,1] coords to bin indices."""
        return [int(round(c * (self.n_bins - 1))) for c in normalized_coords]
    
    def to_tokens(self, box, image_size):
        """Convert pixel box to location tokens."""
        normed = self.normalize_box(box, image_size)
        bins = self.discretize(normed)
        return "".join([f"<loc_{b:03d}>" for b in bins])
    
    def from_tokens(self, token_string, image_size):
        """Parse location tokens back to pixel coordinates."""
        import re
        bins = [int(x) for x in re.findall(r'<loc_(\d+)>', token_string)]
        
        if len(bins) != 4:
            return None
        
        W, H = image_size
        coords = [b / (self.n_bins - 1) for b in bins]
        return [
            coords[0] * W,  # x1
            coords[1] * H,  # y1
            coords[2] * W,  # x2
            coords[3] * H,  # y2
        ]


# Example
tokenizer = CoordinateTokenizer(n_bins=1000)
box = [100, 150, 300, 400]  # pixels
image_size = (640, 480)

tokens = tokenizer.to_tokens(box, image_size)
print(f"Box {box} → {tokens}")

reconstructed = tokenizer.from_tokens(tokens, image_size)
print(f"Tokens → {reconstructed}")
```

### Grounding for Robot Manipulation

```python
import numpy as np


class RobotGrounding:
    """Convert VLM grounding to 3D robot coordinates."""
    
    def __init__(self, camera_matrix, depth_estimator):
        self.K = camera_matrix  # 3×3 intrinsic matrix
        self.depth_est = depth_estimator
    
    def pixel_to_3d(self, u, v, depth):
        """Convert pixel (u, v) + depth to 3D camera coordinates."""
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        
        return np.array([x, y, z])
    
    def ground_object_3d(self, image_path, object_description, depth_map=None):
        """Ground a text description to 3D coordinates."""
        # Step 1: VLM grounding → pixel bounding box
        vlm = GroundedVLM()
        detections = vlm.ground(image_path, object_description)
        
        if not detections.get('bboxes'):
            print(f"Object '{object_description}' not found")
            return None
        
        # Step 2: Box center → pixel coordinates
        box = detections['bboxes'][0]
        u_center = (box[0] + box[2]) / 2
        v_center = (box[1] + box[3]) / 2
        
        # Step 3: Depth at center point
        if depth_map is None:
            from PIL import Image
            image = Image.open(image_path)
            depth_map = self.depth_est.estimate(image)
        
        u_int, v_int = int(u_center), int(v_center)
        depth = depth_map[v_int, u_int]
        
        # Step 4: Pixel + depth → 3D
        point_3d = self.pixel_to_3d(u_center, v_center, depth)
        
        print(f"'{object_description}' → pixel ({u_center:.0f}, {v_center:.0f})")
        print(f"  depth: {depth:.3f}m")
        print(f"  3D position: ({point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f})m")
        
        return {
            'label': detections['labels'][0],
            'box_2d': box,
            'center_pixel': (u_center, v_center),
            'depth': depth,
            'position_3d': point_3d,
        }
```

---

## Exercise (45 min)

1. **Grounding evaluation:** Use Florence-2 to ground 10 referring expressions (e.g., "the leftmost chair", "the red object nearest to the camera"). Compute IoU between predicted and manually annotated boxes.

2. **Coordinate precision:** Test how coordinate discretization affects localization accuracy. Compare n_bins = {100, 500, 1000}. What's the minimum bin count for sub-centimeter robot manipulation?

3. **End-to-end pipeline:** Combine grounding + depth estimation to compute 3D coordinates for 5 objects. Verify spatial consistency (e.g., object A should be further than object B if A's depth is greater).

---

## Key Takeaways

1. **Grounding = actionable understanding.** Without spatial grounding, VLMs describe but can't locate
2. **Coordinate tokens.** Discretizing coordinates into special tokens lets LLMs predict locations
3. **Pixel → 3D pipeline.** VLM grounding + depth = 3D coordinates for robot manipulation
4. **Resolution of grounding.** 1000 bins ≈ 0.1% precision ≈ sub-pixel accuracy at 1000px
5. **Bridge to VLAs.** Grounding is the link between "see and describe" and "see and act"

## Connection to the Thread

> *Spatial grounding completes the perception-to-action pipeline: VLM finds the object in pixels, depth converts to 3D, and the robot moves. Tomorrow: reflecting on the VLM journey before the capstone.*

---

## Further Reading

- [Florence-2 (Xiao et al., 2024)](https://arxiv.org/abs/2311.06242)
- [Ferret: Refer and Ground (You et al., 2023)](https://arxiv.org/abs/2310.07704)
- [Shikra: Unleashing Multimodal LLM's Referential Dialogue Magic (Chen et al., 2023)](https://arxiv.org/abs/2306.15195)
- [Kosmos-2 (Peng et al., 2023)](https://arxiv.org/abs/2306.14824)
