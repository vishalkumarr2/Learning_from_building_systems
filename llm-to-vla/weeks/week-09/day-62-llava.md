# Day 62: LLaVA — Visual Instruction Tuning

> **Phase V — Vision-Language Models | Week 9 | 2.5 hours**
> *"An MLP. That's it. LLaVA connects CLIP to LLaMA with a 2-layer MLP and visual instruction tuning does the rest." — Liu et al., 2023*

## Navigation
- **Previous:** [Day 61: Flamingo & BLIP-2](day-61-flamingo-blip2.md)
- **Next:** [Day 63: PaLI & CoCa](day-63-pali-coca.md)
- **Week:** [Week 9 Overview](README.md)
- **Phase:** [Phase V: Vision-Language Models](../../study-notes/10-vision-language-models.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### LLaVA's Simplicity

LLaVA (Large Language and Vision Assistant) demonstrates that **simple architecture + good data >> complex architecture**:

```
┌──────────────────────────────────────────────────────────┐
│                      LLaVA Architecture                   │
│                                                           │
│  Image ──► CLIP ViT-L/14 ──► 576 patch tokens (1024-d)  │
│                   (frozen)          │                      │
│                                     │                      │
│                              MLP Projector                │
│                              (2-layer, GELU)              │
│                                     │                      │
│                                     ▼                      │
│  Text  ──► Tokenizer ──►  [visual tokens, text tokens]   │
│                                     │                      │
│                              LLaMA/Vicuna                 │
│                              (fine-tuned)                  │
│                                     │                      │
│                              Generated response           │
└──────────────────────────────────────────────────────────┘
```

That's it. No Perceiver, no Q-Former, no gated cross-attention. Just:
1. CLIP vision encoder (frozen)
2. 2-layer MLP (trained)
3. LLM (fine-tuned with LoRA or full)

### Two-Stage Training

**Stage 1: Feature Alignment (pretrain)**
- Freeze: vision encoder + LLM
- Train: only the MLP projector
- Data: 558K image-caption pairs (CC3M filtered)
- Purpose: learn to project visual features into LLM's embedding space

**Stage 2: Visual Instruction Tuning**
- Freeze: vision encoder
- Train: MLP projector + LLM (full fine-tuning or LoRA)
- Data: 150K instruction-following conversations about images
- Purpose: teach the model to follow visual instructions

### Visual Instruction Data

The key innovation: GPT-4 generates instruction-following data from image captions:

```
Input to GPT-4:
  Caption: "A brown dog is sitting on a red couch in a living room"
  Bounding boxes: dog (0.2, 0.3, 0.6, 0.8), couch (0.1, 0.4, 0.9, 0.95)

GPT-4 generates conversations:
  Q: "What is the dog doing in the image?"
  A: "The dog is sitting on a red couch in what appears to be a living room."
  
  Q: "What color is the couch?"
  A: "The couch is red."
  
  Q: "Describe the scene in detail."
  A: "A brown dog sits comfortably on a red couch in a cozy living room..."
```

Three types of instruction data:
1. **Conversation:** Multi-turn Q&A about images
2. **Detailed description:** Rich, paragraph-length image descriptions
3. **Complex reasoning:** Questions requiring spatial reasoning or inference

### LLaVA-1.5 Improvements

| Change | LLaVA | LLaVA-1.5 |
|--------|-------|-----------|
| Vision encoder | CLIP ViT-L/14 (224px) | CLIP ViT-L/14 (336px) |
| Projector | Linear | 2-layer MLP |
| LLM | LLaMA-7B | Vicuna-7B/13B |
| Training data | 150K | 665K |
| Resolution | 224×224 | 336×336 |

Higher resolution = more visual detail = better performance on tasks requiring fine-grained understanding.

---

## Implementation (60 min)

### LLaVA-style Architecture

```python
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, AutoTokenizer, AutoModelForCausalLM


class LLaVAProjector(nn.Module):
    """2-layer MLP projector from vision to language space."""
    
    def __init__(self, vision_dim=1024, llm_dim=4096):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
    
    def forward(self, vision_features):
        return self.projector(vision_features)


class SimpleLLaVA(nn.Module):
    """Simplified LLaVA for understanding the architecture."""
    
    def __init__(self, vision_model_name="openai/clip-vit-large-patch14",
                 llm_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        super().__init__()
        
        # Vision encoder (frozen)
        self.vision_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        
        vision_dim = self.vision_encoder.config.hidden_size
        
        # LLM
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        llm_dim = self.llm.config.hidden_size
        
        # Projector (the only new component!)
        self.projector = LLaVAProjector(vision_dim, llm_dim)
    
    def encode_image(self, pixel_values):
        """Extract and project visual features."""
        with torch.no_grad():
            vision_out = self.vision_encoder(pixel_values)
            # Use all patch tokens (skip CLS)
            patch_features = vision_out.last_hidden_state[:, 1:]  # (B, N, vision_dim)
        
        visual_tokens = self.projector(patch_features)  # (B, N, llm_dim)
        return visual_tokens
    
    def prepare_inputs(self, pixel_values, input_ids, attention_mask):
        """Concatenate visual tokens with text tokens."""
        visual_tokens = self.encode_image(pixel_values)
        
        # Get text embeddings from LLM
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # Concatenate: [visual_tokens, text_tokens]
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
        
        # Extend attention mask for visual tokens
        visual_mask = torch.ones(
            visual_tokens.shape[:2], device=attention_mask.device, dtype=attention_mask.dtype
        )
        extended_mask = torch.cat([visual_mask, attention_mask], dim=1)
        
        return inputs_embeds, extended_mask
    
    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        inputs_embeds, attention_mask = self.prepare_inputs(
            pixel_values, input_ids, attention_mask
        )
        
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return outputs
```

### Using Pretrained LLaVA

```python
from transformers import LlavaForConditionalGeneration, AutoProcessor
from PIL import Image


def llava_chat(image_path, question, model_name="llava-hf/llava-1.5-7b-hf"):
    """Visual question answering with LLaVA."""
    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    
    image = Image.open(image_path).convert("RGB")
    
    prompt = f"USER: <image>\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(
        model.device, torch.float16
    )
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256)
    
    response = processor.decode(output_ids[0], skip_special_tokens=True)
    # Extract assistant response
    assistant_response = response.split("ASSISTANT:")[-1].strip()
    
    print(f"Q: {question}")
    print(f"A: {assistant_response}")
    return assistant_response
```

### Instruction Tuning Data Format

```python
def create_llava_training_sample(image_path, conversation):
    """Format a training sample for LLaVA instruction tuning.
    
    Args:
        image_path: path to image
        conversation: list of {"role": "user"|"assistant", "content": str}
    
    Returns:
        formatted training dict
    """
    return {
        "image": image_path,
        "conversations": [
            {
                "from": turn["role"],
                "value": ("<image>\n" + turn["content"]) if i == 0 and turn["role"] == "user"
                         else turn["content"],
            }
            for i, turn in enumerate(conversation)
        ],
    }


# Example training sample
sample = create_llava_training_sample(
    "robot_workspace.jpg",
    [
        {"role": "user", "content": "What objects are on the table?"},
        {"role": "assistant", "content": "I can see a red cup, a screwdriver, and a circuit board on the table."},
        {"role": "user", "content": "Which object is closest to the edge?"},
        {"role": "assistant", "content": "The red cup appears to be closest to the edge of the table."},
    ]
)
```

---

## Exercise (45 min)

1. **Architecture comparison:** Count trainable parameters for: (a) LLaVA MLP projector, (b) BLIP-2 Q-Former, (c) Flamingo Perceiver + gated cross-attention. Which is most parameter-efficient?

2. **LLaVA inference:** Use a pretrained LLaVA model to answer questions about 5 images. Test with questions about: color, counting, spatial relations, text reading, and complex reasoning. Where does it succeed/fail?

3. **Instruction data quality:** Write 3 high-quality conversation samples about robot workspace images. Follow the LLaVA format. What makes a training sample "high quality" for visual instruction tuning?

---

## Key Takeaways

1. **MLP is enough.** A 2-layer MLP matches Q-Former/Perceiver for vision→language bridging
2. **Instruction tuning > architecture.** Quality of training data matters more than bridge complexity
3. **Two-stage training.** Alignment first (projector only), then instruction tuning (projector + LLM)
4. **Visual conversations.** Multi-turn dialogue about images requires instruction-following capability
5. **GPT-4 as data engine.** Synthetic instruction data generated by a capable LLM bootstraps training

## Connection to the Thread

> *LLaVA demonstrated the "less is more" principle for VLMs. Tomorrow: PaLI and CoCa, which scale up the approach with joint contrastive + captioning objectives.*

---

## Further Reading

- [LLaVA (Liu et al., 2023)](https://arxiv.org/abs/2304.08485)
- [LLaVA-1.5 (Liu et al., 2023)](https://arxiv.org/abs/2310.03744)
- [Visual Instruction Tuning blog](https://llava-vl.github.io/)
- [LLaVA-NeXT (Liu et al., 2024)](https://arxiv.org/abs/2406.16860)
