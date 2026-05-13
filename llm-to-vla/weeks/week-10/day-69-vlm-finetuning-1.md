# Day 69: VLM Fine-tuning — Day 1

> **Phase V — Vision-Language Models | Week 10 | 2.5 hours**
> *"A pretrained VLM knows about the world. LoRA fine-tuning teaches it about YOUR world — your objects, your workspace, your robot." — Practical VLM adaptation*

## Navigation
- **Previous:** [Day 68: Phase V Capstone Day 2](day-68-phase5-capstone-2.md)
- **Next:** [Day 70: VLM Fine-tuning Day 2](day-70-vlm-finetuning-2.md)
- **Week:** [Week 10 Overview](README.md)
- **Phase:** [Phase V: Vision-Language Models](../../study-notes/10-vision-language-models.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### Why Fine-tune a VLM?

Pretrained VLMs are general-purpose. Fine-tuning adapts them to:
- **Domain-specific objects:** industrial parts, medical images, robot tools
- **Custom vocabulary:** your team's terminology for workspace items
- **Specific output format:** structured JSON instead of free-form text
- **Task specialization:** counting objects, reading gauges, estimating distances

### LoRA for VLMs

LoRA (Low-Rank Adaptation) adds small trainable matrices to frozen model weights:

$$W' = W + \Delta W = W + BA$$

where $W \in \mathbb{R}^{d \times k}$, $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$.

For a VLM, you apply LoRA to specific components:

```
Which layers to LoRA-adapt?

Vision Encoder (usually frozen):
  ✗ Keep frozen — pretrained features are good

MLP Projector:
  ✓ Fine-tune fully (small, fast)

LLM layers:
  ✓ LoRA on q_proj, v_proj (attention)
  ○ Optionally: k_proj, o_proj, gate_proj, up_proj, down_proj
  
Total trainable params: ~0.5-2% of full model
```

### QLoRA for Memory Efficiency

QLoRA quantizes the base model to 4-bit while keeping LoRA adapters in 16-bit:

$$\text{Memory} = \underbrace{\frac{|\theta|}{2}}_{\text{4-bit base}} + \underbrace{2r \cdot |\theta_{\text{target}}|}_{\text{16-bit LoRA}} \approx 4\text{GB for 7B model}$$

This makes fine-tuning a 7B VLM possible on a single consumer GPU (24GB).

### Data Preparation

Fine-tuning data follows the instruction format:

```json
[
  {
    "image": "workspace_001.jpg",
    "conversations": [
      {"from": "user", "value": "<image>\nWhat tools are on the workbench?"},
      {"from": "assistant", "value": "I can see a Phillips screwdriver, wire strippers, and a multimeter on the workbench."}
    ]
  },
  {
    "image": "workspace_002.jpg",
    "conversations": [
      {"from": "user", "value": "<image>\nIs the soldering iron turned on?"},
      {"from": "assistant", "value": "Yes, the LED indicator on the soldering station is green, indicating it's powered on and at operating temperature."}
    ]
  }
]
```

---

## Implementation (75 min)

### Step 1: Dataset Preparation

```python
import json
import os
from PIL import Image
from torch.utils.data import Dataset


class VLMFineTuneDataset(Dataset):
    """Dataset for VLM fine-tuning with image-conversation pairs."""
    
    def __init__(self, data_path, image_dir, processor, max_length=2048):
        with open(data_path) as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        image_path = os.path.join(self.image_dir, item['image'])
        image = Image.open(image_path).convert("RGB")
        
        # Format conversation
        conversation = item['conversations']
        user_msg = conversation[0]['value']
        assistant_msg = conversation[1]['value']
        
        prompt = f"USER: {user_msg}\nASSISTANT: {assistant_msg}"
        
        # Process
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        
        # Create labels (mask user tokens, only train on assistant response)
        labels = inputs['input_ids'].clone()
        # Find where assistant response starts
        assistant_token = self.processor.tokenizer.encode("ASSISTANT:", add_special_tokens=False)
        input_ids_list = inputs['input_ids'][0].tolist()
        
        # Mask everything before assistant response
        for i in range(len(input_ids_list)):
            if input_ids_list[i:i+len(assistant_token)] == assistant_token:
                labels[0, :i+len(assistant_token)] = -100
                break
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': labels.squeeze(0),
        }


def create_training_data(image_dir, output_path, n_samples=100):
    """Create synthetic training data for VLM fine-tuning.
    
    In practice, you'd annotate real images. This generates templates.
    """
    templates = [
        {
            'question': "What objects do you see in this image?",
            'answer_template': "I can see {objects} in the image.",
        },
        {
            'question': "Describe the workspace layout.",
            'answer_template': "The workspace contains {layout_description}.",
        },
        {
            'question': "Is there anything that looks unsafe?",
            'answer_template': "{safety_assessment}",
        },
    ]
    
    data = []
    for i, fname in enumerate(sorted(os.listdir(image_dir))[:n_samples]):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        
        template = templates[i % len(templates)]
        data.append({
            'image': fname,
            'conversations': [
                {'from': 'user', 'value': f"<image>\n{template['question']}"},
                {'from': 'assistant', 'value': f"[ANNOTATE: {template['answer_template']}]"},
            ]
        })
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Created {len(data)} training samples at {output_path}")
    return data
```

### Step 2: LoRA Configuration

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import torch


def setup_qlora_vlm(model_name="llava-hf/llava-1.5-7b-hf"):
    """Set up QLoRA fine-tuning for a VLM."""
    
    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model in 4-bit
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config — target LLM attention layers only
    lora_config = LoraConfig(
        r=16,                        # rank
        lora_alpha=32,               # scaling factor
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "v_proj",      # attention projections
            "k_proj", "o_proj",      # optionally more
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    return model, processor


def count_lora_params(model):
    """Count LoRA parameters by module."""
    lora_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            module = name.split('.')[0]
            if module not in lora_params:
                lora_params[module] = 0
            lora_params[module] += param.numel()
    
    for module, count in sorted(lora_params.items(), key=lambda x: -x[1]):
        print(f"  {module}: {count:,} params")
    
    return lora_params
```

### Step 3: Training Loop

```python
from transformers import TrainingArguments, Trainer


def train_vlm_lora(model, processor, train_dataset, eval_dataset=None,
                    output_dir="./vlm-lora-output", epochs=3, batch_size=4):
    """Fine-tune VLM with LoRA."""
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_dataset else "no",
        fp16=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    # Train
    result = trainer.train()
    
    # Save LoRA adapter (small — only the trained weights)
    model.save_pretrained(output_dir)
    print(f"\nLoRA adapter saved to {output_dir}")
    print(f"Adapter size: {sum(f.stat().st_size for f in Path(output_dir).rglob('*') if f.is_file()) / 1e6:.1f} MB")
    
    return result
```

---

## Exercise (30 min)

1. **Data preparation:** Create 20 training samples for a domain you're interested in (e.g., robot workstation, kitchen, lab equipment). Write realistic Q&A conversations for each image.

2. **LoRA config exploration:** Compare these LoRA configs on training loss:
   - r=4 vs r=16 vs r=64
   - target_modules = ["q_proj", "v_proj"] vs all attention layers
   
   Which configuration converges fastest? Which gives the best final loss?

3. **Memory budget:** Calculate the memory required for: (a) full fine-tuning of LLaVA-7B, (b) LoRA r=16, (c) QLoRA r=16. Can each fit on your GPU?

---

## Key Takeaways

1. **LoRA is practical.** Fine-tune 0.5-2% of parameters, keep 98%+ frozen
2. **QLoRA enables consumer GPUs.** 4-bit base + 16-bit adapters → 7B VLM in 4-6GB
3. **Data quality > quantity.** 500 high-quality samples can match 50K noisy ones
4. **Freeze vision encoder.** Only adapt the LLM and projector — vision features are robust
5. **Labels matter.** Mask user tokens in loss — only train on assistant responses

## Connection to the Thread

> *You can now configure LoRA fine-tuning for any VLM. Tomorrow: running the training, evaluating the fine-tuned model, and comparing with the base model.*

---

## Further Reading

- [LoRA (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
- [QLoRA (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
- [LLaVA fine-tuning guide](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md)
- [PEFT library documentation](https://huggingface.co/docs/peft)
