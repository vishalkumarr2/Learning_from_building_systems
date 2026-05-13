# Day 59: CLIP — Contrastive Vision-Language Learning

> **Phase V — Vision-Language Models | Week 9 | 2.5 hours**
> *"400 million image-text pairs. Contrastive loss. Zero-shot transfer to any visual task. CLIP changed everything." — Radford et al., 2021*

## Navigation
- **Previous:** [Day 58: Phase IV Capstone Day 2](day-58-phase4-capstone-2.md)
- **Next:** [Day 60: CLIP Internals + SigLIP](day-60-clip-internals-siglip.md)
- **Week:** [Week 9 Overview](README.md)
- **Phase:** [Phase V: Vision-Language Models](../../study-notes/10-vision-language-models.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### The CLIP Revolution

Before CLIP, vision models were trained on fixed label sets (ImageNet's 1000 classes). CLIP learns from **natural language supervision** — any text description paired with an image:

```
Training: 400M image-text pairs from the internet

┌─────────────┐          ┌─────────────┐
│ Image       │          │ Text        │
│ Encoder     │          │ Encoder     │
│ (ViT/ResNet)│          │ (Transformer)│
└──────┬──────┘          └──────┬──────┘
       │                        │
       ▼                        ▼
    v = f(img)              t = g(text)
       │                        │
       └────── similarity ──────┘
              = v · t / (|v| · |t|)

Loss: contrastive — matched pairs should have high similarity,
      unmatched pairs should have low similarity
```

### Contrastive Loss (InfoNCE)

For a batch of $N$ image-text pairs $(I_i, T_i)$:

$$\mathcal{L}_{\text{image}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\text{sim}(v_i, t_i) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(v_i, t_j) / \tau)}$$

$$\mathcal{L}_{\text{text}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\text{sim}(t_i, v_i) / \tau)}{\sum_{j=1}^N \exp(\text{sim}(t_i, v_j) / \tau)}$$

$$\mathcal{L} = \frac{1}{2}(\mathcal{L}_{\text{image}} + \mathcal{L}_{\text{text}})$$

where $\tau$ is a learned temperature parameter and $\text{sim}$ is cosine similarity.

### Zero-Shot Transfer

CLIP enables classification on **any** categories without training:

```
Given: image of a dog, categories = ["cat", "dog", "bird", "car"]

1. Encode image: v = image_encoder(image)
2. Create text prompts: "a photo of a cat", "a photo of a dog", ...
3. Encode each prompt: t_i = text_encoder(prompt_i)
4. Predict: argmax_i sim(v, t_i)

Result: "dog" — without ever training on a dog classifier!
```

### Why CLIP Matters for VLAs

CLIP creates a **shared embedding space** where images and text are directly comparable:
- "Pick up the red cup" → text embedding → find matching image region
- Robot camera view → image embedding → match to task descriptions
- Zero-shot object recognition → no need to retrain for new objects

### Architecture Details

| Component | CLIP ViT-B/32 | CLIP ViT-L/14 |
|-----------|--------------|---------------|
| Image encoder | ViT-B/32 (86M) | ViT-L/14 (304M) |
| Text encoder | 12-layer, 512-dim (63M) | 12-layer, 768-dim (123M) |
| Embedding dim | 512 | 768 |
| Training data | 400M pairs (WIT) | 400M pairs |
| Batch size | 32,768 | 32,768 |

---

## Implementation (60 min)

### CLIP Inference from Scratch

```python
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor
from PIL import Image


def clip_zero_shot(image_path, categories, template="a photo of a {}"):
    """Zero-shot classification with CLIP."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    
    image = Image.open(image_path).convert("RGB")
    prompts = [template.format(cat) for cat in categories]
    
    inputs = processor(
        text=prompts,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds  # (1, D)
        text_embeds = outputs.text_embeds    # (N_cats, D)
    
    # Cosine similarity
    similarity = F.cosine_similarity(
        image_embeds.unsqueeze(1),  # (1, 1, D)
        text_embeds.unsqueeze(0),   # (1, N_cats, D)
        dim=-1
    ).squeeze(0)  # (N_cats,)
    
    probs = (similarity * 100).softmax(dim=0)
    
    for cat, prob in sorted(zip(categories, probs.tolist()), key=lambda x: -x[1]):
        print(f"  {cat}: {prob:.1%}")
    
    return categories[probs.argmax().item()]


# Example
categories = ["cat", "dog", "bird", "fish", "robot", "car"]
# result = clip_zero_shot("test_image.jpg", categories)
```

### Image-Text Retrieval

```python
def clip_retrieval(image_paths, text_queries, top_k=3):
    """Cross-modal retrieval: find images matching text queries."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    
    # Encode all images
    images = [Image.open(p).convert("RGB") for p in image_paths]
    image_inputs = processor(images=images, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        image_embeds = model.get_image_features(**image_inputs)
        image_embeds = F.normalize(image_embeds, dim=-1)
    
    # Encode all text queries
    text_inputs = processor(text=text_queries, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        text_embeds = model.get_text_features(**text_inputs)
        text_embeds = F.normalize(text_embeds, dim=-1)
    
    # Similarity matrix: (n_queries, n_images)
    similarity = text_embeds @ image_embeds.T
    
    # Retrieve top-k images per query
    for i, query in enumerate(text_queries):
        topk_idx = similarity[i].topk(top_k).indices
        print(f"\nQuery: '{query}'")
        for rank, idx in enumerate(topk_idx):
            print(f"  #{rank+1}: {image_paths[idx]} (sim={similarity[i, idx]:.3f})")
    
    return similarity


def build_clip_embedding_index(image_dir, model_name="openai/clip-vit-base-patch32"):
    """Build a CLIP embedding index for fast retrieval."""
    import os
    
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    
    embeddings = []
    paths = []
    
    for fname in sorted(os.listdir(image_dir)):
        if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            path = os.path.join(image_dir, fname)
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                embed = model.get_image_features(**inputs)
                embed = F.normalize(embed, dim=-1)
            
            embeddings.append(embed)
            paths.append(path)
    
    index = torch.cat(embeddings, dim=0)  # (N_images, D)
    print(f"Built index: {index.shape[0]} images, {index.shape[1]}-dim embeddings")
    
    return index, paths
```

### Prompt Engineering for CLIP

```python
def clip_prompt_ensemble(image_path, category, templates=None):
    """Use multiple prompt templates for robust zero-shot classification."""
    if templates is None:
        templates = [
            "a photo of a {}.",
            "a blurry photo of a {}.",
            "a close-up photo of a {}.",
            "a bright photo of a {}.",
            "a dark photo of a {}.",
            "a photo of many {}.",
            "a photo of a small {}.",
            "a photo of a large {}.",
            "a photo of the {}.",
            "an image of a {}.",
        ]
    
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    
    image = Image.open(image_path).convert("RGB")
    prompts = [t.format(category) for t in templates]
    
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Average text embeddings across templates
        text_embed = F.normalize(outputs.text_embeds.mean(dim=0, keepdim=True), dim=-1)
        image_embed = F.normalize(outputs.image_embeds, dim=-1)
    
    similarity = (image_embed @ text_embed.T).item()
    return similarity
```

---

## Exercise (45 min)

1. **Zero-shot benchmark:** Run CLIP zero-shot on CIFAR-10 test set. Report accuracy. How does it compare to a ViT trained on CIFAR-10 from Day 46?

2. **Prompt sensitivity:** Test 5 different prompt templates for the same image and category. How much does accuracy vary? Does ensembling help?

3. **Failure analysis:** Find 5 images where CLIP's zero-shot classification fails. What types of images confuse CLIP? (Hint: try abstract concepts, counting, spatial relationships.)

---

## Key Takeaways

1. **Natural language supervision.** CLIP learns from image-text pairs, not fixed label sets
2. **Zero-shot transfer.** Classify any category by comparing image and text embeddings — no retraining
3. **Shared embedding space.** Images and text live in the same vector space — enables cross-modal search
4. **Prompt engineering matters.** Template choice significantly affects zero-shot accuracy
5. **Foundation for VLMs.** CLIP's visual encoder is the starting point for LLaVA, BLIP-2, and VLAs

## Connection to the Thread

> *CLIP created the shared vision-language space that all VLMs build on. Tomorrow: diving into CLIP's internals — learned temperature, the alignment mechanism, and SigLIP's improvement.*

---

## Further Reading

- [Learning Transferable Visual Models (Radford et al., 2021)](https://arxiv.org/abs/2103.00020)
- [OpenCLIP: open-source reproduction](https://github.com/mlfoundations/open_clip)
- [CLIP benchmark on various datasets](https://github.com/LAION-AI/CLIP_benchmark)
- Lilian Weng: "Contrastive Representation Learning"
