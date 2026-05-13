# Exercise 08 — VLM Experiments

> **Phase V · Days 59–68 · ~15 hours**
> Companion to: [study-notes/10-vision-language-models.md](../study-notes/10-vision-language-models.md)
> Prerequisites: Exercises 06 (ViT) and 07 (detection/segmentation)

---

## Setup

### Environment

```bash
# Create/activate environment
conda activate llm-to-vla  # or your env name

# Core dependencies
pip install torch torchvision transformers accelerate
pip install open_clip_torch  # OpenCLIP implementation
pip install Pillow matplotlib scikit-learn

# For Exercise 3-4
pip install supervision  # Roboflow's visualization library
pip install gradio       # Quick UI for demos

# Optional: for Florence-2
pip install timm einops
```

### Hardware Requirements

| Exercise | GPU Memory | Estimated Time |
|----------|-----------|---------------|
| Ex 1: CLIP from Scratch | 8GB+ | 3–4 hours |
| Ex 2: Mini-LLaVA | 16GB+ (or use LoRA on 8GB) | 4–5 hours |
| Ex 3: Grounding | 8GB+ | 2–3 hours |
| Ex 4: Scene Understanding | 8GB+ | 2–3 hours |

### Test Images

Prepare a small set of robot workspace images (or use any tabletop scenes):

```python
# Download sample images if you don't have robot workspace photos
import urllib.request
from pathlib import Path

img_dir = Path("data/workspace_images")
img_dir.mkdir(parents=True, exist_ok=True)

# Use your own robot workspace images or any tabletop scenes
# Place 10-20 images in data/workspace_images/
print(f"Place test images in: {img_dir.resolve()}")
```

---

## Exercise 1: CLIP from Scratch (Days 59–60)

### 1.1 Implement the Contrastive Loss

```python
"""
Exercise 1.1: Implement CLIP's InfoNCE contrastive loss from scratch.
Do NOT look at the OpenCLIP source — implement from the paper's equations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPContrastiveLoss(nn.Module):
    """
    CLIP-style symmetric contrastive loss (InfoNCE).
    
    Given a batch of N image-text pairs:
    - Each image should match its paired text (diagonal of similarity matrix)
    - All other pairings are negatives
    
    Loss = 0.5 * (image_to_text_loss + text_to_image_loss)
    """
    
    def __init__(self, temperature_init=0.07, learnable_temperature=True):
        super().__init__()
        # TODO: Initialize temperature parameter
        # CLIP uses a learnable log-temperature (log scale for numerical stability)
        # Initialize so that exp(log_temp) = 1/temperature_init
        pass
    
    def forward(self, image_embeddings, text_embeddings):
        """
        Args:
            image_embeddings: (N, D) L2-normalized image features
            text_embeddings:  (N, D) L2-normalized text features
        
        Returns:
            loss: scalar contrastive loss
            
        Steps:
            1. Compute cosine similarity matrix (N x N)
            2. Scale by temperature
            3. Compute cross-entropy loss treating diagonal as labels
            4. Average image→text and text→image losses
        """
        # TODO: Implement
        pass


# --- Test your implementation ---
def test_contrastive_loss():
    torch.manual_seed(42)
    loss_fn = CLIPContrastiveLoss()
    
    N, D = 8, 64
    img_emb = F.normalize(torch.randn(N, D), dim=-1)
    txt_emb = F.normalize(torch.randn(N, D), dim=-1)
    
    loss = loss_fn(img_emb, txt_emb)
    print(f"Loss with random embeddings: {loss.item():.4f}")
    # Expected: ~log(N) ≈ 2.08 for N=8 (random → uniform over negatives)
    
    # Perfect alignment: loss should be near 0
    loss_perfect = loss_fn(img_emb, img_emb)
    print(f"Loss with perfect alignment: {loss_perfect.item():.4f}")
    # Expected: close to 0
    
    assert loss.item() > loss_perfect.item(), "Random should have higher loss than aligned"
    print("✓ Basic sanity checks passed")

test_contrastive_loss()
```

### 1.2 Build a Mini-CLIP Model

```python
"""
Exercise 1.2: Build a small CLIP model with ViT image encoder + text transformer.
Train on a small image-text dataset.
"""

import torch
import torch.nn as nn
from torchvision import transforms


class MiniCLIP(nn.Module):
    """
    Small CLIP model for educational purposes.
    
    Image encoder: Small ViT (from Exercise 06 or timm)
    Text encoder:  Small transformer encoder
    """
    
    def __init__(self, embed_dim=256, image_size=224, patch_size=16,
                 num_layers=6, num_heads=8, vocab_size=30522, max_text_len=77):
        super().__init__()
        
        # TODO: Build image encoder
        # Option A: Use your ViT from Exercise 06
        # Option B: Use timm.create_model('vit_tiny_patch16_224', pretrained=False)
        # The output should be a single [CLS] token embedding
        self.image_encoder = None  # TODO
        
        # TODO: Build text encoder
        # Simple transformer encoder with:
        # - Token embedding + positional embedding
        # - num_layers transformer encoder layers
        # - Take [CLS] token (position 0) as text representation
        self.text_encoder = None  # TODO
        
        # TODO: Projection heads to shared embedding space
        # image features → embed_dim
        # text features → embed_dim
        self.image_projection = None  # TODO
        self.text_projection = None  # TODO
        
        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))
    
    def encode_image(self, images):
        """Encode images to normalized embeddings."""
        # TODO: image → image_encoder → projection → L2 normalize
        pass
    
    def encode_text(self, input_ids, attention_mask=None):
        """Encode text to normalized embeddings."""
        # TODO: tokens → text_encoder → projection → L2 normalize
        pass
    
    def forward(self, images, input_ids, attention_mask=None):
        """
        Returns:
            image_embeddings: (N, embed_dim) normalized
            text_embeddings:  (N, embed_dim) normalized
            logit_scale: scalar temperature
        """
        image_emb = self.encode_image(images)
        text_emb = self.encode_text(input_ids, attention_mask)
        return image_emb, text_emb, self.logit_scale.exp()
```

### 1.3 Train and Evaluate

```python
"""
Exercise 1.3: Training loop and zero-shot evaluation.
"""

def train_mini_clip(model, dataloader, num_epochs=10, lr=1e-4):
    """
    Train the mini-CLIP model.
    
    TODO:
    1. Set up optimizer (AdamW, weight_decay=0.1)
    2. Set up learning rate schedule (cosine with warmup)
    3. Training loop:
       - Forward pass → image_emb, text_emb
       - Compute contrastive loss
       - Backward + optimizer step
    4. Log loss every 100 steps
    """
    pass


def zero_shot_classify(model, image, class_names, tokenizer):
    """
    Zero-shot classification using CLIP.
    
    TODO:
    1. Create text prompts: "a photo of a {class_name}" for each class
    2. Encode all text prompts → text_embeddings (C, D)
    3. Encode the image → image_embedding (1, D)
    4. Compute similarities → pick highest
    
    Returns: (predicted_class_name, confidence_scores)
    """
    pass


def visualize_embedding_space(model, dataset, tokenizer, num_samples=500):
    """
    Visualize CLIP embedding space with t-SNE.
    
    TODO:
    1. Encode num_samples images and their captions
    2. Concatenate all embeddings (images + texts)
    3. Apply t-SNE to reduce to 2D
    4. Plot: images as one color, texts as another
    5. Draw lines between matching image-text pairs
    6. Observation: matching pairs should cluster together
    """
    pass
```

### 1.4 Analysis Questions

After training, answer:

1. **What happens when you increase the batch size?** (Try 32, 64, 128, 256)
   - Plot loss curves for each. Why does larger batch help?

2. **What happens when temperature is too low vs too high?**
   - Try fixed τ = 0.01, 0.07, 0.5, 1.0. How does each affect training?

3. **Zero-shot on novel categories**: Test on categories NOT in training data.
   - Does it work? Why or why not?
   - How does prompt engineering ("a photo of a {}" vs "a {}" vs "an image containing {}") affect accuracy?

---

## Exercise 2: Visual Instruction Tuning Pipeline (Days 61–63)

### 2.1 Build a Mini-LLaVA

```python
"""
Exercise 2.1: Build a simplified LLaVA architecture.

Architecture:
  Image → Pretrained ViT → Linear Projection → Small LLM
  
Keep everything small for educational purposes:
- ViT: clip-vit-base-patch16 (frozen)
- Projection: 2-layer MLP
- LLM: GPT-2 small (124M) or TinyLlama (1.1B with LoRA)
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoModelForCausalLM, AutoTokenizer


class MiniLLaVA(nn.Module):
    """
    Simplified LLaVA for visual instruction tuning.
    """
    
    def __init__(self, vision_model_name="openai/clip-vit-base-patch16-224",
                 llm_name="gpt2",  # or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                 freeze_vision=True):
        super().__init__()
        
        # TODO: Load vision encoder
        # Use CLIPVisionModel, extract patch features (not just [CLS])
        # Shape: (B, num_patches, vision_dim)
        self.vision_encoder = None  # TODO
        self.vision_dim = None  # TODO: get from model config
        
        # TODO: Load LLM
        self.llm = None  # TODO
        self.llm_dim = None  # TODO: get from model config
        self.tokenizer = None  # TODO
        
        # TODO: Build MLP projection
        # vision_dim → llm_dim (2-layer MLP with GELU)
        self.projection = nn.Sequential(
            # TODO
        )
        
        # Freeze vision encoder
        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
    
    def encode_image(self, pixel_values):
        """
        Encode image to visual tokens in LLM's embedding space.
        
        Args:
            pixel_values: (B, 3, H, W)
        Returns:
            visual_tokens: (B, num_patches, llm_dim)
        """
        # TODO: vision_encoder → patch features → projection
        pass
    
    def prepare_inputs(self, pixel_values, input_ids, attention_mask, labels=None):
        """
        Interleave visual and text tokens for LLM input.
        
        TODO:
        1. Encode image → visual_tokens (B, M, D)
        2. Embed text → text_tokens (B, N, D)
        3. Find <image> placeholder token position
        4. Replace placeholder with visual_tokens
        5. Adjust attention_mask and labels accordingly
        
        Returns: combined embeddings, combined attention mask, combined labels
        """
        pass
    
    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        """
        Forward pass: image + text → next token prediction loss.
        """
        inputs_embeds, combined_mask, combined_labels = self.prepare_inputs(
            pixel_values, input_ids, attention_mask, labels
        )
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
        )
        return outputs
```

### 2.2 Two-Stage Training

```python
"""
Exercise 2.2: Implement two-stage training for mini-LLaVA.
"""

def stage1_alignment(model, caption_dataloader, num_epochs=1, lr=1e-3):
    """
    Stage 1: Feature alignment pretraining.
    
    Freeze: vision encoder + LLM
    Train: projection layer only
    Data: image-caption pairs
    
    TODO:
    1. Freeze LLM parameters
    2. Only enable gradients for model.projection
    3. Train with next-token prediction on captions
    4. Use format: "<image> Caption: {caption}"
    """
    # Freeze LLM
    for param in model.llm.parameters():
        param.requires_grad = False
    
    # Only train projection
    optimizer = torch.optim.AdamW(model.projection.parameters(), lr=lr)
    
    # TODO: Training loop
    pass


def stage2_instruction_tuning(model, instruction_dataloader, num_epochs=1, lr=2e-5):
    """
    Stage 2: Visual instruction tuning.
    
    Freeze: vision encoder
    Train: projection + LLM (full fine-tune or LoRA)
    Data: visual instruction-response pairs
    
    TODO:
    1. Unfreeze LLM (or apply LoRA)
    2. Enable gradients for projection + LLM
    3. Train with next-token prediction on instruction-response pairs
    4. Use format: "<image> [INST] {question} [/INST] {answer}"
    5. Only compute loss on answer tokens (mask instruction tokens)
    """
    # Unfreeze LLM
    for param in model.llm.parameters():
        param.requires_grad = True
    
    # TODO: Training loop with instruction masking
    pass
```

### 2.3 Ablation: With vs Without Stage 1

```python
"""
Exercise 2.3: Compare models with and without alignment pretraining.
"""

def ablation_study():
    """
    Train two models:
    A) Full pipeline: Stage 1 → Stage 2
    B) Stage 2 only (skip alignment)
    
    Compare on evaluation set:
    - Caption quality (BLEU, CIDEr if available, or manual inspection)
    - Instruction following accuracy
    - Hallucination rate (does the model describe things not in the image?)
    
    TODO:
    1. Train model A with both stages
    2. Train model B with stage 2 only (random projection)
    3. Evaluate both on 50 test images
    4. Create comparison table
    """
    # Record observations:
    # - Does alignment pretraining help?
    # - How much does it help? 
    # - What specific behaviors differ?
    pass
```

---

## Exercise 3: Grounding Experiment (Days 64–65)

### 3.1 Run Grounding Models

```python
"""
Exercise 3.1: Language-conditioned object detection with grounding models.
"""

from PIL import Image
import torch


def grounding_with_florence2():
    """
    Use Florence-2 for grounded detection on robot workspace images.
    
    TODO:
    1. Load Florence-2 model
    2. For each test image, run these tasks:
       a) "<CAPTION>" → detailed caption
       b) "<DETAILED_CAPTION>" → very detailed caption
       c) "<OD>" → open-vocabulary object detection
       d) "<CAPTION_TO_PHRASE_GROUNDING>" + "the red box" → grounded detection
    3. Visualize results with bounding boxes overlaid
    """
    from transformers import AutoProcessor, AutoModelForCausalLM
    
    model_id = "microsoft/Florence-2-large"
    # TODO: Load model and processor
    # TODO: Run detection on workspace images
    # TODO: Visualize with supervision library
    pass


def grounding_with_grounding_dino():
    """
    Alternative: Use Grounding DINO for text-conditioned detection.
    
    TODO:
    1. Load Grounding DINO model
    2. Run detection with text prompts:
       - "red box"
       - "cup on the table"
       - "robot gripper"
    3. Visualize results and compare with Florence-2
    """
    pass
```

### 3.2 Language-Conditioned Detection Benchmark

```python
"""
Exercise 3.2: Systematic evaluation of grounding accuracy.
"""

def grounding_benchmark():
    """
    Create a mini benchmark for grounding on robot workspace scenes.
    
    TODO:
    1. Annotate 20 images with:
       - Object names + bounding boxes (ground truth)
       - Natural language descriptions for each object
    
    2. Test queries at three difficulty levels:
       Level 1 (easy):   "the cup"
       Level 2 (medium): "the red cup near the plate"  
       Level 3 (hard):   "the object the robot should pick up next"
    
    3. Measure per-level:
       - Precision: of detected boxes, how many are correct? (IoU > 0.5)
       - Recall: of ground truth objects, how many are found?
       - mAP at IoU thresholds [0.25, 0.5, 0.75]
    
    4. Compare Florence-2 vs Grounding DINO vs standard YOLO detector
    
    Returns: comparison table and analysis
    """
    pass


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes [x1, y1, x2, y2].
    
    TODO: Implement intersection-over-union.
    """
    pass
```

### 3.3 Spatial Relationship Evaluation

```python
"""
Exercise 3.3: Can grounding models understand spatial language?
"""

def spatial_grounding_eval():
    """
    Test if grounding models understand spatial descriptions:
    
    Prompts:
    - "the cup to the LEFT of the plate"
    - "the object BETWEEN the two boxes"
    - "the item CLOSEST to the robot arm"
    - "the TALLEST object on the table"
    
    TODO:
    1. Create 10 scenes with clear spatial relationships
    2. For each scene, write 3-5 spatial queries
    3. Run grounding model on each query
    4. Evaluate: does the model select the spatially correct object?
    5. Categorize failures: wrong object? right object but wrong box? no detection?
    
    Hypothesis: grounding models handle simple spatial words (left, right, above)
    but fail on relative/comparative spatial reasoning.
    """
    pass
```

---

## Exercise 4: VLM for Robot Scene Understanding (Days 66–68)

### 4.1 Scene Description Pipeline

```python
"""
Exercise 4.1: Use VLMs to describe robot workspace scenes.
"""

from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image


def scene_description_pipeline(image_path):
    """
    Run multiple VLMs on a robot workspace image and compare outputs.
    
    TODO:
    1. Load at least 2 VLM backends:
       - LLaVA (llava-hf/llava-v1.6-mistral-7b-hf)
       - InternVL or Qwen-VL (pick one available model)
    
    2. For each model, ask:
       a) "Describe what you see in this robot workspace."
       b) "List all objects and their approximate positions."
       c) "What task might a robot be performing here?"
    
    3. Compare outputs:
       - Factual accuracy (objects actually present?)
       - Spatial accuracy (positions correct?)
       - Hallucinations (objects described but not present?)
       - Usefulness for robotics (actionable information?)
    
    Returns: structured comparison of model outputs
    """
    pass


def batch_scene_analysis(image_dir, model):
    """
    Run scene analysis on all images in a directory.
    
    TODO:
    1. Load all images from image_dir
    2. For each, generate: description, object list, spatial relations
    3. Save structured output as JSON:
       {
           "image": "scene_01.jpg",
           "description": "...",
           "objects": [{"name": "cup", "position": "left side", "confidence": 0.9}],
           "spatial_relations": [{"subject": "cup", "relation": "left_of", "object": "plate"}]
       }
    """
    pass
```

### 4.2 Spatial Reasoning Evaluation

```python
"""
Exercise 4.2: Evaluate VLM spatial reasoning capabilities.
"""

def spatial_reasoning_benchmark():
    """
    Test VLM spatial understanding with structured questions.
    
    Create a benchmark with:
    
    Category 1: Absolute Position
    - "Is the cup on the left or right side of the table?"
    - "Which shelf is the box on — top, middle, or bottom?"
    
    Category 2: Relative Position  
    - "Is the fork to the left or right of the knife?"
    - "Which object is closer to the robot: the cup or the plate?"
    
    Category 3: Spatial Reasoning
    - "Is there enough space between the objects to place a new box?"
    - "If the robot approaches from the right, which object can it reach first?"
    
    Category 4: Counting
    - "How many cups are on the table?"
    - "Are there more plates or bowls?"
    
    TODO:
    1. Create 10 scenes with known ground truth answers
    2. Write 4 questions per scene (one per category)
    3. Run VLM, extract answer, compare to ground truth
    4. Report accuracy per category
    
    Expected findings:
    - Category 1: ~80% (VLMs handle absolute position OK)
    - Category 2: ~60% (relative position is harder)
    - Category 3: ~40% (reasoning requires understanding scale)
    - Category 4: ~50% (counting is a known VLM weakness)
    """
    pass
```

### 4.3 Affordance and Action Proposal

```python
"""
Exercise 4.3: VLM-based affordance detection and action proposals.
"""

def affordance_analysis(image_path, vlm):
    """
    Ask VLM about manipulation affordances.
    
    Prompts:
    1. "Where should the robot grasp the mug?" 
    2. "What is the safest way to pick up this object?"
    3. "Can this object be stacked on top of the other?"
    4. "Is this object fragile? How should the robot handle it?"
    
    TODO:
    1. Run each prompt on 10 different scenes
    2. Manually evaluate responses (1-5 scale):
       - Correctness: Is the suggestion physically valid?
       - Specificity: Does it give actionable information?
       - Safety: Does it consider damage/spills/breakage?
    3. Create evaluation report
    """
    pass


def task_decomposition(image_path, task_instruction, vlm):
    """
    Ask VLM to decompose a high-level task into steps.
    
    Example:
    Task: "Set the table for dinner"
    Image: Kitchen counter with plates, cups, and utensils
    
    Expected output:
    1. Pick up the plate and place it in the center
    2. Place the fork to the left of the plate
    3. Place the knife to the right of the plate
    ...
    
    TODO:
    1. Test with 5 different tasks:
       - "Clean up the workspace"
       - "Sort objects by color"
       - "Stack all the boxes"
       - "Move all fragile items to the safe zone"
       - "Prepare the workspace for assembly"
    2. Evaluate: are the steps logical, complete, and feasible?
    3. Identify failure modes: missing steps, wrong order, impossible actions
    """
    pass
```

---

## Self-Check Rubric

### Exercise 1: CLIP
- [ ] Contrastive loss matches expected values (random ≈ log(N), aligned ≈ 0)
- [ ] Mini-CLIP trains and loss decreases
- [ ] Zero-shot transfer works on at least some novel categories
- [ ] t-SNE shows matching pairs clustering together
- [ ] Can explain why batch size and temperature matter

### Exercise 2: Mini-LLaVA
- [ ] Model architecture is correct (frozen ViT → projection → LLM)
- [ ] Stage 1 trains projection without touching LLM
- [ ] Stage 2 unfreezes LLM and trains on instruction data
- [ ] Ablation shows alignment pretraining helps
- [ ] Model generates coherent responses to image questions

### Exercise 3: Grounding
- [ ] Florence-2 or Grounding DINO runs on test images
- [ ] Bounding box visualizations are correct
- [ ] IoU computation is correct
- [ ] Benchmark shows quantitative comparison
- [ ] Spatial queries show failure modes

### Exercise 4: Scene Understanding
- [ ] At least 2 VLM backends compared
- [ ] Spatial reasoning benchmark shows per-category accuracy
- [ ] Affordance analysis identifies useful vs vague responses
- [ ] Task decomposition evaluation completed
- [ ] Can articulate why VLMs are necessary but insufficient for robot action

---

## Stretch Goals

### S1: SigLIP Implementation
Replace CLIP's InfoNCE loss with SigLIP's sigmoid loss. Compare convergence speed and final accuracy.

### S2: Dynamic Resolution
Implement LLaVA-NeXT's AnyRes approach: split a high-res image into tiles, encode each tile, concatenate visual tokens. Compare with single-resolution on fine-grained tasks.

### S3: Multi-Image VLM
Extend your mini-LLaVA to handle **multiple images**:
- "What changed between these two images?"
- "Which image shows a cleaner workspace?"
- This is relevant for robot temporal understanding.

### S4: VLM + Grounding Pipeline
Chain a VLM with a grounding model:
1. VLM identifies objects: "I see a red cup and a blue plate"
2. Grounding model localizes each: "red cup" → [x1,y1,x2,y2]
3. Combine into a structured scene graph with spatial coordinates

### S5: Real-Time VLM Demo
Build a Gradio interface:
- Upload robot camera image (or use webcam)
- Ask questions in natural language
- Show VLM answer + grounded bounding boxes
- Useful for demonstrating capabilities to non-technical collaborators

---

*Previous: [07-detection-segmentation.md](07-detection-segmentation.md) · Next: [09-robot-learning.md](09-robot-learning.md)*
