# Day 32: Supervised Fine-Tuning

> **Phase III — LLMs: Training & Alignment | Week 5 | 2.5 hours**
> *"A pretrained model is a jack of all trades — SFT makes it a helpful assistant." — Jason Wei*

## Navigation
- **Previous:** [Day 31: The Modern LLM Recipe](day-31-modern-llm-recipe.md)
- **Next:** [Day 33: RLHF](day-33-rlhf.md)
- **Week:** [Week 5 Overview](README.md)
- **Phase:** [Phase III: LLM Training & Alignment](../../study-notes/06-llm-training-alignment.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 32.1 Why SFT? The Gap Between Prediction and Following

A pretrained model can *complete* text but can't *follow instructions*:

```
Pretrained model:
  Input:  "What is the capital of France?"
  Output: "What is the capital of Germany? What is the capital of Spain?..."
  
After SFT:
  Input:  "What is the capital of France?"
  Output: "The capital of France is Paris."
```

**SFT bridges the format gap:** the model already *knows* the answer from pretraining — SFT teaches it to *output* the answer in the right format.

### 32.2 Instruction Datasets

Key datasets that shaped the field:

| Dataset | Size | Method | Key Innovation |
|---------|------|--------|----------------|
| FLAN (2022) | 1.8K tasks, millions of examples | Aggregated NLP benchmarks | Task diversity |
| Alpaca (2023) | 52K | GPT-3.5 distillation | Self-instruct: LLM generates training data |
| ShareGPT | ~90K | User conversations | Real user distribution |
| OpenAssistant | 161K | Human-written | Multilingual, multi-turn |
| UltraChat | 1.5M | GPT-4 generated | Scale + diversity |
| Orca | 5M | GPT-4 explanations | Chain-of-thought reasoning |

**Self-Instruct pipeline (Alpaca):**

```
Seed instructions (175 examples)
        │
        ▼
LLM generates new instructions  ──→  Filter low quality
        │
        ▼
LLM generates input-output pairs
        │
        ▼
52K instruction-following examples
```

### 32.3 Chat Templates

Models use specific templates to separate roles in conversation:

```
# ChatML format (OpenAI / Qwen):
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
2+2 equals 4.<|im_end|>

# Llama 2 format:
[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>
What is 2+2? [/INST] 2+2 equals 4.

# Llama 3 format:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful assistant.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
What is 2+2?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
2+2 equals 4.<|eot_id|>
```

**Critical:** During SFT, we only compute loss on the **assistant tokens** — the model isn't penalized for not predicting the user's messages.

$$
\mathcal{L}_{\text{SFT}} = -\sum_{t \in \text{assistant tokens}} \log P_\theta(x_t \mid x_{<t})
$$

### 32.4 SFT Training Dynamics

```
Typical SFT hyperparameters:
────────────────────────────
Learning rate:    1e-5 to 2e-5 (10-100x lower than pretraining)
Epochs:           2-3 (overfitting is easy with small datasets)
Batch size:       128-256 (effective, with gradient accumulation)
Warmup:           3-10% of steps
Weight decay:     0.0 to 0.1
Max seq length:   2048-4096
Scheduler:        Cosine decay to 0

Rule of thumb: If loss stops decreasing after 1 epoch,
your dataset is too small or too easy.
```

**The alignment tax:** SFT slightly degrades base model performance on benchmarks (1-3%) because it narrows the output distribution. This is expected and acceptable.

---

## Implementation (60 min)

### SFT with LoRA on TinyLlama Using HuggingFace + TRL

```python
"""
Day 32 Implementation: Supervised Fine-Tuning with LoRA on TinyLlama.
Uses TRL's SFTTrainer for instruction-tuning.
"""
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# --- 1. Load base model ---
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)

# --- 2. Prepare instruction dataset ---
def create_instruction_dataset() -> Dataset:
    """Create a small robotics instruction-following dataset."""
    examples = [
        {
            "instruction": "Move the robot forward 2 meters",
            "response": "Executing linear motion: distance=2.0m, speed=0.5m/s. "
                        "Publishing cmd_vel with linear.x=0.5 for 4.0 seconds.",
        },
        {
            "instruction": "What is the robot's current battery level?",
            "response": "Current battery status: voltage=25.2V, "
                        "charge=78%, estimated runtime=3.2 hours.",
        },
        {
            "instruction": "Navigate to the charging station",
            "response": "Planning path to charging station at coordinates "
                        "(12.5, 3.2). Distance: 15.3m. Estimated time: 45s. "
                        "Initiating autonomous navigation.",
        },
        {
            "instruction": "Rotate 90 degrees clockwise",
            "response": "Executing rotation: angle=-90°(clockwise), "
                        "angular_speed=0.5rad/s. Publishing cmd_vel with "
                        "angular.z=-0.5 for π seconds.",
        },
        {
            "instruction": "Report any obstacles detected",
            "response": "LiDAR scan results: 3 obstacles detected. "
                        "Obstacle 1: 2.1m at 15° (wall). "
                        "Obstacle 2: 0.8m at 340° (box). "
                        "Obstacle 3: 5.2m at 90° (shelf). "
                        "Closest obstacle at 0.8m — within safety threshold.",
        },
    ]
    return Dataset.from_list(examples)

def format_chat(example: dict) -> str:
    """Format as ChatML for SFT."""
    return (
        "<|im_start|>system\n"
        "You are a warehouse robot controller. Execute commands precisely "
        "and report status clearly.<|im_end|>\n"
        f"<|im_start|>user\n{example['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['response']}<|im_end|>"
    )

dataset = create_instruction_dataset()

# --- 3. Configure LoRA ---
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # rank of the update matrices
    lora_alpha=32,           # scaling factor
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# → trainable params: ~4M / 1.1B total → 0.36%

# --- 4. Train with SFTTrainer ---
training_args = TrainingArguments(
    output_dir="./sft-tinyllama-robot",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,          # higher LR for LoRA
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    formatting_func=format_chat,
    max_seq_length=512,
    tokenizer=tokenizer,
)

trainer.train()

# --- 5. Inference test ---
model.eval()
prompt = (
    "<|im_start|>system\n"
    "You are a warehouse robot controller.<|im_end|>\n"
    "<|im_start|>user\nStop immediately<|im_end|>\n"
    "<|im_start|>assistant\n"
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
    )
print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```

---

## Exercise (45 min)

### E32.1 — Chat Template Masking (20 min)
Implement a custom data collator that masks user tokens from the loss:
1. Tokenize a multi-turn conversation
2. Create a `labels` tensor where user/system tokens are set to `-100` (ignored)
3. Verify that only assistant tokens contribute to the loss

### E32.2 — Dataset Quality Analysis (25 min)
Download the Alpaca dataset (52K examples) and analyze:
1. Distribution of instruction lengths (histogram)
2. Most common verbs in instructions (what tasks dominate?)
3. Average response length by task type
4. Find 5 examples that seem low-quality — what patterns do you notice?

---

## Key Takeaways

1. **SFT teaches format, not knowledge** — the model already knows answers from pretraining
2. **Only compute loss on assistant tokens** — this is the critical implementation detail
3. **Chat templates** vary by model family — using the wrong template breaks the model
4. **Small, high-quality datasets** (10-100K) are sufficient for SFT
5. **Self-Instruct** (LLM generates training data) bootstraps instruction datasets cheaply

---

## Connection to the Thread

SFT is exactly how robot foundation models learn to follow commands. After pretraining on internet-scale data, models like RT-2 are fine-tuned on robot demonstration data — instruction-action pairs that are the robotic equivalent of instruction-response pairs. The chat template structure (system/user/assistant) maps directly to (robot context/user command/robot action).

---

## Further Reading

- [Training Language Models to Follow Instructions (InstructGPT)](https://arxiv.org/abs/2203.02155) — OpenAI, 2022
- [Self-Instruct: Aligning LLMs with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) — Alpaca's foundation
- [FLAN Collection: Designing Data and Methods for Effective Instruction Tuning](https://arxiv.org/abs/2301.13688) — Google, 2023
- [Lima: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — 1K examples suffice
- [TRL Documentation](https://huggingface.co/docs/trl) — HuggingFace training library
