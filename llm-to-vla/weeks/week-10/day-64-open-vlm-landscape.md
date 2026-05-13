# Day 64: Open VLM Landscape

> **Phase V — Vision-Language Models | Week 10 | 2.5 hours**
> *"InternVL, Qwen-VL, Phi-3-Vision, Idefics2 — the open VLM ecosystem has exploded. Knowing the landscape is knowing what to build on." — 2024 VLM survey*

## Navigation
- **Previous:** [Day 63: PaLI & CoCa](../week-09/day-63-pali-coca.md)
- **Next:** [Day 65: Spatial Grounding](day-65-spatial-grounding.md)
- **Week:** [Week 10 Overview](README.md)
- **Phase:** [Phase V: Vision-Language Models](../../study-notes/10-vision-language-models.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### The Open VLM Ecosystem (2024)

The VLM landscape has expanded dramatically since LLaVA. Each model makes different architectural and data decisions:

```
                      Open VLM Family Tree
                      
CLIP (2021) ──────────────────────────────────────────
    │                                                  
    ├── LLaVA (2023)  ── LLaVA-1.5 ── LLaVA-NeXT     
    │                                                  
    ├── BLIP-2 (2023) ── InstructBLIP                  
    │                                                  
    ├── InternVL (2023) ── InternVL-1.5 ── InternVL2  
    │                                                  
    ├── Qwen-VL (2023) ── Qwen-VL-Chat ── Qwen2-VL   
    │                                                  
    ├── Phi-3-Vision (2024) ── Phi-3.5-Vision          
    │                                                  
    ├── Idefics2 (2024) ── Idefics3                    
    │                                                  
    └── Cambrian-1 (2024) — any-vision-encoder         
```

### Architecture Comparison

| Model | Vision Encoder | Language Model | Bridge | Resolution | Params |
|-------|---------------|----------------|--------|------------|--------|
| LLaVA-NeXT | CLIP ViT-L | Vicuna/LLaMA3 | MLP | 672×672 | 7-34B |
| InternVL2 | InternViT-6B | InternLM2 | PixelShuffle+MLP | 448×448 | 2-76B |
| Qwen2-VL | ViT (native) | Qwen2 | Cross-attn | Dynamic | 2-72B |
| Phi-3-Vision | CLIP ViT | Phi-3 | MLP | 1344×1344 | 4.2B |
| Idefics2 | SigLIP | Mistral | Perceiver | 980×980 | 8B |
| Cambrian-1 | Any combination | LLaMA-3 | Spatial tokens | Varies | 8-34B |

### Key Design Decisions

**1. Vision Encoder Choice:**
- **InternViT** (InternVL): custom-trained 6B ViT — largest open vision encoder
- **SigLIP** (Idefics2, PaLI-3): superior to CLIP for VLM tasks
- **Dynamic resolution** (Qwen2-VL): process images at native resolution, variable token count

**2. Resolution Strategy:**

```
Fixed resolution:     Resize everything to 336×336 → always 576 tokens
                      ✗ Loses detail in large images

Tiled resolution:     Split image into tiles, encode each
(LLaVA-NeXT)         ┌──────┬──────┐
                      │ tile1│ tile2│  → 576 × 4 = 2304 tokens
                      ├──────┼──────┤
                      │ tile3│ tile4│
                      └──────┴──────┘
                      ✓ Preserves detail  ✗ More tokens

Dynamic resolution:   Encode at native aspect ratio
(Qwen2-VL)           Variable token count based on image size
                      ✓ Most flexible  ✗ Variable compute
```

**3. Multi-Image Support:**
- Most VLMs handle single images
- Qwen2-VL, InternVL2, and Idefics2 support interleaved multi-image inputs
- Critical for robotics (multiple camera views, temporal sequences)

### Benchmark Landscape

| Benchmark | Tests | Why It Matters |
|-----------|-------|---------------|
| MMBench | Multi-choice VQA | General visual understanding |
| MMMU | University-level QA | Expert-level reasoning |
| MathVista | Math + vision | Quantitative reasoning |
| OCRBench | Text recognition | Reading text in images |
| RealWorldQA | Real-world photos | Practical visual QA |
| HallusionBench | Hallucination detection | Factual accuracy |

### Which VLM Should You Use?

```
Decision tree:
                                          
Need < 5B params? ─── Yes ──► Phi-3-Vision (4.2B)
         │                     
         No                    
         │                     
Need multi-image? ── Yes ──► Qwen2-VL or InternVL2
         │                     
         No                    
         │                     
Need OCR? ────────── Yes ──► Qwen2-VL (best at OCR)
         │                     
         No                    
         │                     
Best overall? ────────────► InternVL2-Pro or Qwen2-VL-72B
```

---

## Implementation (60 min)

### Comparing VLMs on the Same Task

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
import time


class VLMBenchmark:
    """Compare multiple VLMs on the same questions."""
    
    def __init__(self):
        self.models = {}
        self.results = []
    
    def load_model(self, name, model_id):
        """Load a VLM for comparison."""
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto",
            trust_remote_code=True
        )
        model.eval()
        self.models[name] = (processor, model)
        print(f"Loaded {name}: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params")
    
    def ask(self, name, image_path, question):
        """Ask a question to a specific model."""
        processor, model = self.models[name]
        image = Image.open(image_path).convert("RGB")
        
        prompt = f"<image>\n{question}"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(
            model.device, torch.float16
        )
        
        start = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=128)
        elapsed = time.time() - start
        
        response = processor.decode(output[0], skip_special_tokens=True)
        return response, elapsed
    
    def compare(self, image_path, questions):
        """Run comparison across all loaded models."""
        for q in questions:
            print(f"\n{'='*60}")
            print(f"Q: {q}")
            print(f"{'='*60}")
            for name in self.models:
                response, elapsed = self.ask(name, image_path, q)
                print(f"  [{name}] ({elapsed:.1f}s): {response}")
                self.results.append({
                    'model': name,
                    'question': q,
                    'response': response,
                    'time': elapsed,
                })


# Usage
# bench = VLMBenchmark()
# bench.load_model("phi3v", "microsoft/Phi-3-vision-128k-instruct")
# bench.load_model("llava", "llava-hf/llava-1.5-7b-hf")
# bench.compare("test.jpg", ["What objects are visible?", "Count the items."])
```

### Qwen2-VL Dynamic Resolution

```python
def qwen2vl_inference(image_path, question):
    """Qwen2-VL with dynamic resolution — adapts to image size."""
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    image = Image.open(image_path)
    print(f"Image size: {image.size} → dynamic token count")
    
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question},
        ]}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)
    
    response = processor.batch_decode(output, skip_special_tokens=True)[0]
    return response
```

### Model Selection Helper

```python
def recommend_vlm(requirements):
    """Recommend a VLM based on requirements."""
    models = {
        'phi3v': {'params': 4.2, 'multi_image': False, 'ocr': 'good', 'speed': 'fast'},
        'llava_next': {'params': 7, 'multi_image': True, 'ocr': 'ok', 'speed': 'medium'},
        'qwen2vl_2b': {'params': 2, 'multi_image': True, 'ocr': 'excellent', 'speed': 'fast'},
        'qwen2vl_7b': {'params': 7, 'multi_image': True, 'ocr': 'excellent', 'speed': 'medium'},
        'internvl2_8b': {'params': 8, 'multi_image': True, 'ocr': 'good', 'speed': 'medium'},
        'idefics2': {'params': 8, 'multi_image': True, 'ocr': 'good', 'speed': 'medium'},
    }
    
    candidates = list(models.keys())
    
    if requirements.get('max_params'):
        candidates = [m for m in candidates if models[m]['params'] <= requirements['max_params']]
    
    if requirements.get('multi_image'):
        candidates = [m for m in candidates if models[m]['multi_image']]
    
    if requirements.get('need_ocr'):
        candidates = sorted(candidates, key=lambda m: {'excellent': 0, 'good': 1, 'ok': 2}[models[m]['ocr']])
    
    print("Recommended VLMs:")
    for m in candidates[:3]:
        info = models[m]
        print(f"  {m}: {info['params']}B params, OCR={info['ocr']}, multi_image={info['multi_image']}")
    
    return candidates


# For a robot perception task:
# recommend_vlm({'max_params': 8, 'multi_image': True, 'need_ocr': True})
```

---

## Exercise (45 min)

1. **Model comparison:** Pick 2 VLMs (e.g., Phi-3-Vision and Qwen2-VL-2B). Test both on 5 images with questions about counting, OCR, spatial relations, and complex reasoning. Which model is better at which task?

2. **Resolution impact:** Take one image at different resolutions (224, 336, 512, 1024). Run the same VLM on each. At what resolution does performance plateau? Does going higher help for OCR vs scene understanding?

3. **Multi-image test:** Using a model that supports multi-image (Qwen2-VL or InternVL2), test with 2-3 related images (e.g., before/after, multiple camera views). How well does the model reason across images?

---

## Key Takeaways

1. **Rich ecosystem.** 6+ competitive open VLMs with different strengths
2. **Resolution matters.** Dynamic/tiled resolution dramatically improves fine-grained understanding
3. **SigLIP > CLIP.** Newer VLMs are migrating from CLIP to SigLIP vision encoders
4. **Multi-image is critical.** Robotics needs models that handle multiple views/frames
5. **Size vs capability.** Phi-3-Vision (4.2B) is competitive with 7B models on many tasks

## Connection to the Thread

> *You now know the VLM landscape. Tomorrow: spatial grounding — teaching VLMs to point at specific regions, which is essential for robotic manipulation.*

---

## Further Reading

- [InternVL2 (Chen et al., 2024)](https://arxiv.org/abs/2404.16821)
- [Qwen2-VL (Wang et al., 2024)](https://arxiv.org/abs/2409.12191)
- [Phi-3 Technical Report (2024)](https://arxiv.org/abs/2404.14219)
- [Idefics2 (Laurençon et al., 2024)](https://arxiv.org/abs/2405.02246)
- [Cambrian-1 (Tong et al., 2024)](https://arxiv.org/abs/2406.16860)
- [Open VLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)
