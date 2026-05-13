# Day 68: Phase V Capstone — Day 2

> **Phase V — Vision-Language Models | Week 10 | 2.5 hours**
> *"Before you fine-tune VLMs, prove you understand how they work." — Phase V checkpoint*

## Navigation
- **Previous:** [Day 67: Phase V Capstone Day 1](day-67-phase5-capstone-1.md)
- **Next:** [Day 69: VLM Fine-tuning Day 1](day-69-vlm-finetuning-1.md)
- **Week:** [Week 10 Overview](README.md)
- **Phase:** [Phase V: Vision-Language Models](../../study-notes/10-vision-language-models.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Pipeline Evaluation (60 min)

### Evaluation Metrics

```python
import json
from collections import defaultdict


def evaluate_pipeline(pipeline, test_samples):
    """Evaluate the VLM pipeline on a test set.
    
    test_samples: list of {
        'image': path, 'question': str, 'gt_answer': str,
        'gt_objects': [{'name': str, 'box': [x1,y1,x2,y2]}],
        'gt_scene': str
    }
    """
    metrics = defaultdict(list)
    
    for sample in test_samples:
        result = pipeline.process(
            sample['image'],
            question=sample.get('question'),
            ground_objects=[o['name'] for o in sample.get('gt_objects', [])],
        )
        
        # Scene classification accuracy
        if 'gt_scene' in sample:
            correct = result['scene_type'] == sample['gt_scene']
            metrics['scene_accuracy'].append(int(correct))
        
        # VQA - simple string match
        if 'gt_answer' in sample and 'answer' in result:
            gt = sample['gt_answer'].lower().strip()
            pred = result['answer'].lower().strip()
            metrics['vqa_exact_match'].append(int(gt in pred or pred in gt))
        
        # Object detection recall
        if 'gt_objects' in sample:
            gt_names = {o['name'].lower() for o in sample['gt_objects']}
            pred_names = {o['name'].lower() for o in result['objects']}
            if gt_names:
                recall = len(gt_names & pred_names) / len(gt_names)
                metrics['detection_recall'].append(recall)
        
        # Grounding IoU
        if 'gt_objects' in sample and 'grounded' in result:
            for gt_obj in sample['gt_objects']:
                name = gt_obj['name']
                if name in result['grounded'] and result['grounded'][name]['found']:
                    pred_box = result['grounded'][name]['boxes'][0]
                    gt_box = gt_obj['box']
                    iou = compute_iou(pred_box, gt_box)
                    metrics['grounding_iou'].append(iou)
    
    # Summarize
    summary = {}
    for k, v in metrics.items():
        summary[k] = {
            'mean': sum(v) / len(v) if v else 0,
            'count': len(v),
        }
    
    return summary


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0
```

### Error Analysis

```python
def error_analysis(pipeline, test_samples):
    """Categorize failure modes."""
    failures = {
        'hallucination': [],    # Model describes objects not present
        'missed_object': [],    # Model misses visible objects
        'wrong_location': [],   # Grounding error (low IoU)
        'wrong_answer': [],     # VQA error
        'wrong_scene': [],      # Scene misclassification
    }
    
    for sample in test_samples:
        result = pipeline.process(
            sample['image'],
            question=sample.get('question'),
            ground_objects=[o['name'] for o in sample.get('gt_objects', [])],
        )
        
        # Check for hallucinations
        gt_names = {o['name'].lower() for o in sample.get('gt_objects', [])}
        pred_names = {o['name'].lower() for o in result['objects']}
        hallucinated = pred_names - gt_names
        if hallucinated:
            failures['hallucination'].append({
                'image': sample['image'],
                'hallucinated': list(hallucinated),
            })
        
        # Check for missed objects
        missed = gt_names - pred_names
        if missed:
            failures['missed_object'].append({
                'image': sample['image'],
                'missed': list(missed),
            })
    
    print("\n=== Error Analysis ===")
    for category, items in failures.items():
        print(f"{category}: {len(items)} failures")
        for item in items[:3]:
            print(f"  {item}")
    
    return failures
```

---

## Phase V Checkpoint (60 min)

**Answer these 6 questions without looking at notes.**

### 1. Contrastive Loss

> *Write the CLIP contrastive loss. Why is it symmetric? What role does the temperature play?*

_Key: InfoNCE loss applied both image→text and text→image. Symmetric ensures both modalities align. Temperature controls distribution sharpness — learned, starts soft, becomes sharp._

### 2. SigLIP vs CLIP

> *How does SigLIP improve on CLIP's training? Why does sigmoid scale better than softmax?*

_Key: Pairwise sigmoid treats each (image, text) pair independently. No all-gather for softmax normalization across GPUs. Easier to scale to massive batch sizes._

### 3. Bridge Architectures

> *Compare MLP (LLaVA), Q-Former (BLIP-2), and Perceiver (Flamingo). Give one advantage of each.*

_Key: MLP — simplest, fewest params, fast. Q-Former — compresses to 32 tokens, 3-stage training. Perceiver — variable input, gated insertion preserves LLM._

### 4. Visual Instruction Tuning

> *What is LLaVA's key innovation? How does it generate training data?*

_Key: GPT-4 generates instruction-following conversations from image captions. Two-stage: align projector, then fine-tune LLM. Simple MLP bridge + quality data > complex architecture._

### 5. Spatial Grounding

> *How do VLMs output bounding box coordinates? Describe the coordinate tokenization approach.*

_Key: Normalize coordinates to [0, 1], discretize to N bins (e.g., 1000), map to special `<loc_XXX>` tokens. Model generates these tokens autoregressively as part of text output._

### 6. From VLM to VLA

> *What additional capability does a VLA need that a VLM lacks? How does the transition work?*

_Key: Action prediction — motor commands, joint angles, gripper states. Tokenize actions like coordinates. VLA = VLM + action decoder. Same architecture, new output vocabulary._

---

## Self-Assessment

| Score | Meaning |
|-------|---------|
| 6/6 | Ready for VLM fine-tuning and Phase VI |
| 4-5/6 | Strong — review gaps, then continue |
| 2-3/6 | Revisit CLIP (Day 59) through grounding (Day 65) |
| 0-1/6 | Re-do Phase V before proceeding |

---

## Phase V Summary Table

| Day | Topic | Key Concept |
|-----|-------|-------------|
| 59 | CLIP | Contrastive VL alignment, zero-shot transfer |
| 60 | CLIP + SigLIP | Learned temperature, sigmoid > softmax |
| 61 | Flamingo + BLIP-2 | Perceiver, gated cross-attention, Q-Former |
| 62 | LLaVA | MLP bridge, visual instruction tuning |
| 63 | PaLI + CoCa | Scale, dual objectives, unified text interface |
| 64 | Open VLMs | InternVL2, Qwen2-VL, Phi-3-Vision landscape |
| 65 | Grounding | Coordinate tokens, pixel→3D pipeline |
| 66 | Reflection | Bridge to VLAs, what's missing |
| 67-68 | Capstone | Unified pipeline, evaluation, checkpoint |

---

## Key Takeaways

1. **Pipeline works end-to-end.** CLIP + BLIP-2 + Florence-2 creates comprehensive scene understanding
2. **Error analysis reveals patterns.** Hallucination, missed objects, and wrong grounding are distinct failure modes
3. **Phase V knowledge verified.** 6 checkpoint questions cover the essential concepts
4. **Ready for fine-tuning.** Days 69-70 will teach you to adapt VLMs to your own data

## Connection to the Thread

> *Phase V capstone complete. Next: hands-on VLM fine-tuning with LoRA — adapting a pretrained VLM to a custom domain.*
