# Day 70: VLM Fine-tuning — Day 2

> **Phase V — Vision-Language Models | Week 10 | 2.5 hours**
> *"The base model sees a world. Your fine-tuned model sees YOUR world. Today you measure the difference." — VLM evaluation*

## Navigation
- **Previous:** [Day 69: VLM Fine-tuning Day 1](day-69-vlm-finetuning-1.md)
- **Next:** Phase VI — Vision-Language-Action Models (coming soon)
- **Week:** [Week 10 Overview](README.md)
- **Phase:** [Phase V: Vision-Language Models](../../study-notes/10-vision-language-models.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Step 1: Load Fine-tuned Model (15 min)

```python
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image


def load_finetuned_vlm(base_model_name, adapter_path):
    """Load base model with LoRA adapter."""
    # Load base model
    base_model = LlavaForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    
    processor = AutoProcessor.from_pretrained(base_model_name)
    
    # Count adapter parameters
    adapter_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Adapter: {adapter_params:,} params ({100*adapter_params/total_params:.2f}%)")
    
    return model, processor


def merge_and_save(model, output_path):
    """Merge LoRA weights into base model for faster inference."""
    merged = model.merge_and_unload()
    merged.save_pretrained(output_path)
    print(f"Merged model saved to {output_path}")
    return merged
```

## Step 2: Side-by-side Comparison (45 min)

```python
import json
import time


class VLMComparator:
    """Compare base vs fine-tuned VLM on the same questions."""
    
    def __init__(self, base_model_name, adapter_path):
        # Load base model
        self.base_model = LlavaForConditionalGeneration.from_pretrained(
            base_model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.base_model.eval()
        
        # Load fine-tuned model
        base_for_ft = LlavaForConditionalGeneration.from_pretrained(
            base_model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.ft_model = PeftModel.from_pretrained(base_for_ft, adapter_path)
        self.ft_model.eval()
        
        self.processor = AutoProcessor.from_pretrained(base_model_name)
    
    def generate(self, model, image, question, max_tokens=256):
        """Generate response from a model."""
        prompt = f"USER: <image>\n{question}\nASSISTANT:"
        inputs = self.processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(model.device, torch.float16)
        
        start = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_tokens)
        elapsed = time.time() - start
        
        text = self.processor.decode(output[0], skip_special_tokens=True)
        response = text.split("ASSISTANT:")[-1].strip()
        
        return response, elapsed
    
    def compare(self, image_path, questions):
        """Run side-by-side comparison."""
        image = Image.open(image_path).convert("RGB")
        results = []
        
        for q in questions:
            base_resp, base_time = self.generate(self.base_model, image, q)
            ft_resp, ft_time = self.generate(self.ft_model, image, q)
            
            print(f"\n{'='*60}")
            print(f"Q: {q}")
            print(f"Base ({base_time:.1f}s): {base_resp}")
            print(f"Fine-tuned ({ft_time:.1f}s): {ft_resp}")
            
            results.append({
                'question': q,
                'base_response': base_resp,
                'ft_response': ft_resp,
                'base_time': base_time,
                'ft_time': ft_time,
            })
        
        return results
    
    def score_comparison(self, results, gt_answers):
        """Score both models against ground truth."""
        scores = {'base': [], 'ft': []}
        
        for result, gt in zip(results, gt_answers):
            gt_lower = gt.lower()
            base_match = int(gt_lower in result['base_response'].lower())
            ft_match = int(gt_lower in result['ft_response'].lower())
            
            scores['base'].append(base_match)
            scores['ft'].append(ft_match)
        
        base_acc = sum(scores['base']) / len(scores['base'])
        ft_acc = sum(scores['ft']) / len(scores['ft'])
        
        print(f"\nBase accuracy: {base_acc:.1%}")
        print(f"Fine-tuned accuracy: {ft_acc:.1%}")
        print(f"Improvement: {ft_acc - base_acc:+.1%}")
        
        return scores
```

## Step 3: Quantitative Evaluation (45 min)

### Evaluation Suite

```python
from collections import defaultdict


class VLMEvaluator:
    """Comprehensive VLM evaluation."""
    
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
    
    def evaluate_dataset(self, test_data):
        """Run evaluation on a test dataset."""
        metrics = defaultdict(list)
        
        for sample in test_data:
            image = Image.open(sample['image']).convert("RGB")
            question = sample['question']
            gt_answer = sample['gt_answer']
            
            # Generate response
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            inputs = self.processor(
                text=prompt, images=image, return_tensors="pt"
            ).to(self.model.device, torch.float16)
            
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=128)
            
            pred = self.processor.decode(output[0], skip_special_tokens=True)
            pred = pred.split("ASSISTANT:")[-1].strip()
            
            # Exact match
            metrics['exact_match'].append(
                int(gt_answer.lower().strip() == pred.lower().strip())
            )
            
            # Contains match
            metrics['contains_match'].append(
                int(gt_answer.lower() in pred.lower() or pred.lower() in gt_answer.lower())
            )
            
            # Response length
            metrics['response_length'].append(len(pred.split()))
            
            # Category-specific
            if 'category' in sample:
                cat = sample['category']
                metrics[f'accuracy_{cat}'].append(
                    int(gt_answer.lower() in pred.lower())
                )
        
        # Summarize
        summary = {}
        for k, v in metrics.items():
            summary[k] = round(sum(v) / len(v), 3) if v else 0
        
        return summary
    
    def hallucination_check(self, test_data):
        """Check for hallucinated content in responses."""
        hallucinations = []
        
        for sample in test_data:
            image = Image.open(sample['image']).convert("RGB")
            
            prompt = "USER: <image>\nList every object you see. Only mention objects that are actually visible.\nASSISTANT:"
            inputs = self.processor(
                text=prompt, images=image, return_tensors="pt"
            ).to(self.model.device, torch.float16)
            
            with torch.no_grad():
                output = self.model.generate(**inputs, max_new_tokens=200)
            
            pred = self.processor.decode(output[0], skip_special_tokens=True)
            pred = pred.split("ASSISTANT:")[-1].strip()
            
            # Check mentioned objects against ground truth
            gt_objects = set(o.lower() for o in sample.get('gt_objects', []))
            mentioned = set(pred.lower().split())  # simplified — use NER in practice
            
            suspicious = mentioned - gt_objects
            if len(suspicious) > 5:  # heuristic threshold
                hallucinations.append({
                    'image': sample['image'],
                    'response': pred,
                    'suspicious_words': list(suspicious)[:10],
                })
        
        return hallucinations
```

### Visualization

```python
import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(base_metrics, ft_metrics, title="Base vs Fine-tuned"):
    """Visualize base vs fine-tuned model performance."""
    categories = ['exact_match', 'contains_match']
    available = [c for c in categories if c in base_metrics and c in ft_metrics]
    
    base_scores = [base_metrics[c] for c in available]
    ft_scores = [ft_metrics[c] for c in available]
    
    x = np.arange(len(available))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, base_scores, width, label='Base', color='steelblue')
    bars2 = ax.bar(x + width/2, ft_scores, width, label='Fine-tuned', color='coral')
    
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(available, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.0)
    
    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.1%}', ha='center', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.1%}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('vlm_comparison.png', dpi=150)
    print("Saved vlm_comparison.png")


def plot_training_curve(training_log_path):
    """Plot training loss curve from trainer logs."""
    with open(training_log_path) as f:
        logs = json.load(f)
    
    steps = [l['step'] for l in logs if 'loss' in l]
    losses = [l['loss'] for l in logs if 'loss' in l]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, losses, 'b-', alpha=0.3, label='Raw')
    
    # Smoothed
    window = min(20, len(losses) // 5)
    if window > 1:
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(steps[window-1:], smoothed, 'b-', linewidth=2, label='Smoothed')
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('VLM Fine-tuning Loss')
    ax.legend()
    plt.tight_layout()
    plt.savefig('training_curve.png', dpi=150)
```

---

## Step 4: Adapter Management (15 min)

```python
import os
from pathlib import Path


def list_adapters(adapters_dir):
    """List all saved LoRA adapters."""
    for adapter in sorted(Path(adapters_dir).iterdir()):
        if adapter.is_dir() and (adapter / "adapter_config.json").exists():
            config_path = adapter / "adapter_config.json"
            with open(config_path) as f:
                config = json.load(f)
            
            size_mb = sum(
                f.stat().st_size for f in adapter.rglob("*") if f.is_file()
            ) / 1e6
            
            print(f"  {adapter.name}: r={config.get('r')}, "
                  f"targets={config.get('target_modules')}, "
                  f"size={size_mb:.1f}MB")


def swap_adapter(base_model, new_adapter_path):
    """Hot-swap LoRA adapter without reloading base model."""
    # Unload current adapter
    base_model = base_model.unload()
    
    # Load new adapter
    model = PeftModel.from_pretrained(base_model, new_adapter_path)
    model.eval()
    
    return model
```

---

## Exercise (30 min)

1. **Full pipeline:** Run the complete fine-tuning pipeline: prepare data (Day 69) → train → evaluate (today). Report base vs fine-tuned accuracy on your test set.

2. **Rank ablation:** Train adapters with r={4, 8, 16, 32, 64}. Plot test accuracy vs adapter size. What rank gives the best accuracy/size tradeoff?

3. **Domain adaptation:** Fine-tune a VLM on 50 images from a specific domain (e.g., electronics workbench). Test on 10 held-out images. Does the model learn domain-specific vocabulary?

---

## Phase V Complete Summary

```
Phase V Journey:
  Day 59: CLIP — contrastive vision-language alignment
  Day 60: SigLIP — sigmoid improvement, embedding geometry
  Day 61: Flamingo/BLIP-2 — bridge architectures
  Day 62: LLaVA — simple MLP + instruction tuning
  Day 63: PaLI/CoCa — scale + dual objectives
  Day 64: Open VLMs — ecosystem survey
  Day 65: Grounding — coordinate tokens, pixel→3D
  Day 66: Reflection — connecting the thread
  Day 67-68: Capstone — unified pipeline + evaluation
  Day 69-70: Fine-tuning — LoRA adaptation

You can now:
  ✓ Understand CLIP/SigLIP contrastive learning
  ✓ Compare bridge architectures (MLP, Q-Former, Perceiver)
  ✓ Use pretrained VLMs for QA, captioning, grounding
  ✓ Fine-tune VLMs with LoRA/QLoRA on custom data
  ✓ Evaluate and compare VLM performance

Next: Phase VI — Vision-Language-ACTION Models
  Where models learn to see, understand, and ACT.
```

---

## Key Takeaways

1. **Comparison is essential.** Always evaluate base vs fine-tuned on the same test set
2. **Small adapters, big impact.** LoRA r=16 (~10MB) can significantly improve domain performance
3. **Hallucination monitoring.** Fine-tuning can increase hallucination — always check
4. **Adapter swapping.** Keep multiple adapters for different domains, hot-swap at inference
5. **Phase V complete.** You understand VLMs end-to-end: architecture, training, evaluation, fine-tuning

## Connection to the Thread

> *Phase V is complete. You've mastered vision-language models — from CLIP's contrastive alignment to LoRA fine-tuning. Phase VI adds the final piece: ACTION. VLAs will take everything you've built and add the ability to control a robot.*

---

## Further Reading

- [LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
- [LLaVA fine-tuning recipes](https://github.com/haotian-liu/LLaVA)
- [PEFT: Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft)
- [VLM Evaluation Survey (Li et al., 2024)](https://arxiv.org/abs/2402.09996)
