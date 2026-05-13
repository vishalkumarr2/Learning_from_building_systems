# Day 55: DETR + Florence-2 + SAM 2

> **Phase IV — Vision: ViT, 3D, Video | Week 8 | 2.5 hours**
> *"Object detection as set prediction — no anchors, no NMS, just a transformer predicting a set of objects." — Carion et al., 2020*

## Navigation
- **Previous:** [Day 54: Video Understanding Day 2](day-54-video-understanding-2.md)
- **Next:** [Day 56: Vision-Language Bridge](day-56-vision-language-bridge.md)
- **Week:** [Week 8 Overview](README.md)
- **Phase:** [Phase IV: Vision](../../study-notes/09-3d-video-detection.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### DETR: Detection as Set Prediction

Traditional detectors (Faster R-CNN, YOLO) use:
- **Anchors:** Predefined boxes at every position
- **NMS:** Post-processing to remove duplicate detections
- **Hand-designed components:** Region proposals, IoU thresholds

DETR replaces all of this with a transformer:

```
Image → CNN backbone → Transformer Encoder → Transformer Decoder → Set of predictions
                                                    ↑
                                              Object queries
                                              (N learnable embeddings)

Each query independently predicts:
  - Class label (or "no object")
  - Bounding box (cx, cy, w, h)

Loss: Hungarian matching → bipartite assignment of predictions to GT
```

### Hungarian Matching

DETR uses $N = 100$ object queries. The loss matches predictions to ground truth via **optimal bipartite matching**:

$$\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_{i=1}^N \mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)})$$

where $\mathcal{L}_{\text{match}}$ combines classification probability and box L1 + GIoU distance. Unmatched predictions must predict "no object."

### Florence-2: Open-Vocabulary Detection

Florence-2 (2024) unifies multiple vision tasks under a single sequence-to-sequence framework:

```
Input: Image + text prompt ("Detect all objects")
Output: "<loc_x1><loc_y1><loc_x2><loc_y2> cat, <loc_...> dog, ..."

Tasks Florence-2 handles:
├── Object detection (open vocabulary)
├── Dense region captioning
├── Visual grounding ("find the red ball")
├── OCR
├── Image captioning
└── Referring expression segmentation
```

The key: location tokens are **discretized coordinates** added to the text vocabulary. Detection becomes text generation.

### SAM 2: Segment Anything in Images and Video

SAM 2 (2024) extends the original Segment Anything Model to video:

```
┌─────────────────────────────────────────────────┐
│                    SAM 2                         │
│                                                  │
│  Prompt types:                                   │
│    - Point (click on object)                    │
│    - Box (bounding box)                         │
│    - Mask (initial mask)                        │
│    - Text (coming soon)                         │
│                                                  │
│  Image mode: segment any object from a prompt   │
│  Video mode: track object across frames          │
│    └── Memory bank stores object representations │
│    └── Occlusion handling                        │
│    └── Re-identification after disappearance     │
└─────────────────────────────────────────────────┘
```

### Why These Matter for Robotics

| Model | Robotics Application |
|-------|---------------------|
| DETR | Identify objects for manipulation |
| Florence-2 | "Find the screwdriver" → bounding box |
| SAM 2 | Precise segmentation for grasping; object tracking during manipulation |

---

## Implementation (60 min)

### DETR Inference

```python
from transformers import DetrForObjectDetection, DetrImageProcessor
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def detect_objects_detr(image_path, threshold=0.7):
    """Run DETR object detection."""
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    model.eval()
    
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process: convert to boxes and scores
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]
    
    # Visualize
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                  linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        label_name = model.config.id2label[label.item()]
        ax.text(x1, y1-5, f"{label_name}: {score:.2f}",
                color='white', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('detr_detection.png', dpi=150)
    print(f"Detected {len(results['scores'])} objects")
    return results


def detect_florence2(image_path, prompt="<OD>"):
    """Open-vocabulary detection with Florence-2."""
    from transformers import AutoProcessor, AutoModelForCausalLM
    
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base", trust_remote_code=True
    )
    model.eval()
    
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
        )
    
    result = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(result, task=prompt, image_size=image.size)
    
    print(f"Florence-2 result: {parsed}")
    return parsed
```

### SAM 2 Segmentation

```python
def segment_with_sam2(image_path, point_coords=None, box=None):
    """Segment objects using SAM 2."""
    from transformers import SamModel, SamProcessor
    
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    model = SamModel.from_pretrained("facebook/sam-vit-huge")
    model.eval()
    
    image = Image.open(image_path).convert("RGB")
    
    # Prepare prompts
    inputs = processor(
        image,
        input_points=[point_coords] if point_coords else None,
        input_boxes=[box] if box else None,
        return_tensors="pt",
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    
    scores = outputs.iou_scores
    
    # Visualize best mask
    best_idx = scores.argmax()
    mask = masks[0][0][best_idx].numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[1].imshow(image)
    axes[1].imshow(mask, alpha=0.5, cmap='jet')
    axes[1].set_title(f'SAM Segmentation (score: {scores[0, 0, best_idx]:.3f})')
    
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('sam_segmentation.png', dpi=150)
    
    return mask, scores
```

---

## Exercise (45 min)

1. **DETR vs YOLO:** Run both DETR and YOLOv8 on the same 5 images. Compare detection quality and inference speed. When would you prefer each?

2. **Florence-2 tasks:** Test Florence-2 with different prompts on the same image:
   - `<OD>` — object detection
   - `<CAPTION>` — image caption
   - `<DENSE_REGION_CAPTION>` — region descriptions
   - `<OCR>` — text recognition

3. **SAM for grasping:** Given an image of objects on a table, use SAM to segment each object. From the masks, estimate object centroids and areas — these would be grasp candidates for a robot.

---

## Key Takeaways

1. **Set prediction.** DETR eliminates anchors and NMS — transformer predicts object sets directly
2. **Hungarian matching.** Optimal assignment loss enables end-to-end training without hand-designed rules
3. **Unified vision models.** Florence-2 handles detection, captioning, grounding, and OCR in one model
4. **Segment anything.** SAM 2 segments any object from a point/box prompt, even in video
5. **Robotics stack.** Detection → segmentation → grasping is the core manipulation pipeline

## Connection to the Thread

> *You now have the full vision perception stack: classification (ViT), depth (MiDaS), 3D (PointNet), video (TimeSformer), detection (DETR), and segmentation (SAM). Tomorrow: how to connect all of this to language models.*

---

## Further Reading

- [DETR (Carion et al., 2020)](https://arxiv.org/abs/2005.12872)
- [Florence-2 (Xiao et al., 2024)](https://arxiv.org/abs/2311.06242)
- [SAM 2 (Ravi et al., 2024)](https://arxiv.org/abs/2408.00714)
- [Grounding DINO (Liu et al., 2023)](https://arxiv.org/abs/2303.05499) — open-set detection
