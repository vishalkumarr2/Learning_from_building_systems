# Day 67: Phase V Capstone — Day 1

> **Phase V — Vision-Language Models | Week 10 | 2.5 hours**
> *"Build a VLM inference pipeline that takes an image, answers questions, grounds objects, and outputs structured scene descriptions." — Capstone project*

## Navigation
- **Previous:** [Day 66: Stop & Reflect 4](day-66-stop-reflect-4.md)
- **Next:** [Day 68: Phase V Capstone Day 2](day-68-phase5-capstone-2.md)
- **Week:** [Week 10 Overview](README.md)
- **Phase:** [Phase V: Vision-Language Models](../../study-notes/10-vision-language-models.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Project Overview

Build a **unified VLM pipeline** that demonstrates every VLM capability from Phase V:

```
┌──────────────────────────────────────────────────────────────┐
│                  Unified VLM Pipeline                         │
│                                                               │
│  Input: RGB Image + Natural Language Query                    │
│                                                               │
│  Capabilities:                                                │
│  1. Image captioning    → "A workspace with tools"           │
│  2. Visual QA           → Q: "What is on the table?"         │
│  3. Zero-shot classify  → CLIP similarity to categories      │
│  4. Object grounding    → "cup" → [x1, y1, x2, y2]          │
│  5. Scene description   → Structured JSON output             │
│                                                               │
│  Output: Structured scene representation                      │
│  {                                                            │
│    "caption": "...",                                          │
│    "objects": [{"name": "cup", "box": [...], "confidence": 0.9}],│
│    "vqa_answer": "...",                                       │
│    "scene_type": "workshop",                                  │
│    "clip_scores": {"indoor": 0.85, "outdoor": 0.15}          │
│  }                                                            │
└──────────────────────────────────────────────────────────────┘
```

---

## Step 1: CLIP-based Classification (30 min)

```python
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image


class CLIPClassifier:
    """CLIP zero-shot classification with prompt ensembling."""
    
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.model.eval()
        
        self.templates = [
            "a photo of a {}",
            "an image showing a {}",
            "a {} scene",
        ]
    
    def classify(self, image, categories):
        """Zero-shot classification with prompt ensemble."""
        all_prompts = []
        for cat in categories:
            for template in self.templates:
                all_prompts.append(template.format(cat))
        
        inputs = self.processor(
            text=all_prompts, images=image, return_tensors="pt", padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            img_embed = F.normalize(outputs.image_embeds, dim=-1)
            txt_embeds = F.normalize(outputs.text_embeds, dim=-1)
        
        # Average over templates per category
        n_templates = len(self.templates)
        similarities = (img_embed @ txt_embeds.T).squeeze(0)
        
        scores = {}
        for i, cat in enumerate(categories):
            cat_sims = similarities[i * n_templates:(i + 1) * n_templates]
            scores[cat] = cat_sims.mean().item()
        
        # Normalize to probabilities
        total = sum(max(0, v) for v in scores.values()) + 1e-8
        scores = {k: max(0, v) / total for k, v in scores.items()}
        
        return scores
```

## Step 2: Visual QA + Captioning (30 min)

```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class VQAEngine:
    """Visual QA and captioning using BLIP-2."""
    
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b"):
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
        self.model.eval()
    
    def caption(self, image):
        """Generate image caption."""
        inputs = self.processor(images=image, return_tensors="pt").to(torch.float16)
        
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=50)
        
        return self.processor.decode(ids[0], skip_special_tokens=True).strip()
    
    def answer(self, image, question):
        """Answer a question about the image."""
        prompt = f"Question: {question} Answer:"
        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(torch.float16)
        
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=50)
        
        full_text = self.processor.decode(ids[0], skip_special_tokens=True).strip()
        # Extract answer after "Answer:"
        if "Answer:" in full_text:
            return full_text.split("Answer:")[-1].strip()
        return full_text
```

## Step 3: Object Grounding (30 min)

```python
from transformers import AutoProcessor, AutoModelForCausalLM


class ObjectGrounder:
    """Florence-2 based object grounding."""
    
    def __init__(self, model_name="microsoft/Florence-2-base"):
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, trust_remote_code=True
        )
        self.model.eval()
    
    def detect_all(self, image):
        """Detect all objects in the image."""
        prompt = "<OD>"
        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(torch.float16)
        
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=1024)
        
        result = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            result, task="<OD>", image_size=image.size
        )
        return parsed
    
    def ground_phrase(self, image, phrase):
        """Ground a specific phrase to bounding boxes."""
        prompt = f"<OPEN_VOCABULARY_DETECTION> {phrase}"
        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(torch.float16)
        
        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=1024, num_beams=3)
        
        result = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            result, task="<OPEN_VOCABULARY_DETECTION>", image_size=image.size
        )
        return parsed
```

## Step 4: Unified Pipeline (30 min)

```python
import json


class UnifiedVLMPipeline:
    """Combines all VLM capabilities into one pipeline."""
    
    def __init__(self):
        self.classifier = CLIPClassifier()
        self.vqa = VQAEngine()
        self.grounder = ObjectGrounder()
    
    def process(self, image_path, question=None, ground_objects=None,
                scene_categories=None):
        """Run the full VLM pipeline."""
        image = Image.open(image_path).convert("RGB")
        result = {'image': image_path}
        
        # 1. Caption
        result['caption'] = self.vqa.caption(image)
        
        # 2. Scene classification
        if scene_categories is None:
            scene_categories = ['indoor', 'outdoor', 'workshop', 'kitchen',
                                'office', 'warehouse', 'laboratory']
        result['scene_scores'] = self.classifier.classify(image, scene_categories)
        result['scene_type'] = max(result['scene_scores'], key=result['scene_scores'].get)
        
        # 3. Visual QA
        if question:
            result['question'] = question
            result['answer'] = self.vqa.answer(image, question)
        
        # 4. Object detection
        detections = self.grounder.detect_all(image)
        result['objects'] = []
        if detections.get('bboxes'):
            for box, label in zip(detections['bboxes'], detections['labels']):
                result['objects'].append({
                    'name': label,
                    'box': [round(c, 1) for c in box],
                })
        
        # 5. Specific object grounding
        if ground_objects:
            result['grounded'] = {}
            for obj_name in ground_objects:
                grounding = self.grounder.ground_phrase(image, obj_name)
                if grounding.get('bboxes'):
                    result['grounded'][obj_name] = {
                        'found': True,
                        'boxes': [[round(c, 1) for c in b] for b in grounding['bboxes']],
                    }
                else:
                    result['grounded'][obj_name] = {'found': False}
        
        return result
    
    def pretty_print(self, result):
        """Display pipeline results."""
        print(f"\n{'='*60}")
        print(f"Image: {result['image']}")
        print(f"Caption: {result['caption']}")
        print(f"Scene: {result['scene_type']} ({result['scene_scores'][result['scene_type']]:.1%})")
        
        if 'answer' in result:
            print(f"Q: {result['question']}")
            print(f"A: {result['answer']}")
        
        print(f"Objects ({len(result['objects'])}):")
        for obj in result['objects']:
            print(f"  - {obj['name']}: {obj['box']}")
        
        if 'grounded' in result:
            print("Grounded queries:")
            for name, info in result['grounded'].items():
                status = f"found at {info['boxes']}" if info['found'] else "NOT FOUND"
                print(f"  - '{name}': {status}")
        print(f"{'='*60}")


# Usage
# pipeline = UnifiedVLMPipeline()
# result = pipeline.process(
#     "workspace.jpg",
#     question="How many tools are on the table?",
#     ground_objects=["screwdriver", "red cup"],
# )
# pipeline.pretty_print(result)
```

---

## Deliverables

By end of Day 67:
- [ ] Working `UnifiedVLMPipeline` with all 4 modules
- [ ] Tested on at least 5 different images
- [ ] Structured JSON output for each image
- [ ] Timing measurements for each module

Tomorrow: evaluation, Phase V checkpoint questions, and reflection.

---

## Key Takeaways

1. **Modular VLM pipeline.** CLIP + BLIP-2 + Florence-2 combine into comprehensive scene understanding
2. **Each model has strengths.** CLIP for classification, BLIP-2 for QA, Florence-2 for grounding
3. **Structured output.** Converting free-form VLM outputs to structured JSON enables downstream use
4. **Pipeline composition.** Real systems chain multiple VLMs for different capabilities

## Connection to the Thread

> *This pipeline demonstrates everything Phase V covered. Tomorrow: evaluation and Phase V checkpoint to verify your understanding before VLM fine-tuning.*
