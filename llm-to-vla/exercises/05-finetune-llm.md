# Exercise 05 — Fine-Tune an LLM with LoRA

> Phase III · Days 31–36 · ~8 hours hands-on
> Prerequisites: Study notes 06 (LLM Training & Alignment)
> Hardware: GPU with ≥16 GB VRAM (or use QLoRA with ≥8 GB)

---

## Setup

### Environment

```bash
# Create environment
conda create -n llm-finetune python=3.11 -y
conda activate llm-finetune

# Core packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# HuggingFace ecosystem
pip install transformers datasets accelerate

# PEFT (Parameter-Efficient Fine-Tuning)
pip install peft

# Quantization
pip install bitsandbytes

# Training
pip install trl  # Transformer Reinforcement Learning (SFT + DPO trainers)

# Evaluation
pip install evaluate lm-eval

# Monitoring
pip install wandb
wandb login
```

### Verify Setup

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

import transformers, peft, trl, bitsandbytes
print(f"Transformers: {transformers.__version__}")
print(f"PEFT: {peft.__version__}")
print(f"TRL: {trl.__version__}")
print("✓ All packages installed")
```

---

## Exercise 1: Inspect a Base Model

> Goal: Understand what a base model does (and doesn't do)

### 1.1 Load a Base Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Ungated model - no HF approval needed. For Llama-3.2-1B, request access at huggingface.co/meta-llama

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Set pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

### 1.2 Test Base Model on Instructions

```python
def generate(model, tokenizer, prompt, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test: instruction following
prompts = [
    "What is the capital of France?",
    "Explain how a robot navigates using SLAM in 3 sentences.",
    "Write a Python function to compute the Fibonacci sequence.",
    "### Instruction:\nTranslate 'hello' to Spanish.\n\n### Response:\n",
]

print("=" * 60)
print("BASE MODEL OUTPUTS (before fine-tuning)")
print("=" * 60)
for prompt in prompts:
    print(f"\n--- Prompt: {prompt[:50]}... ---")
    output = generate(model, tokenizer, prompt)
    print(output)
    print()
```

**Expected:** The base model will continue the text pattern rather than answer
the question. It may generate more questions, random text, or partial answers.

### 1.3 Count Parameters

```python
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"Total parameters:     {total:>15,}")
    print(f"Trainable parameters: {trainable:>15,}")
    print(f"Frozen parameters:    {frozen:>15,}")
    print(f"Trainable %:          {100 * trainable / total:.4f}%")
    print(f"Model size (FP16):    {total * 2 / 1e9:.2f} GB")
    return total, trainable

total_params, trainable_params = count_parameters(model)
```

### 1.4 Questions

- [ ] Does the base model follow instructions? Why or why not?
- [ ] How many parameters does the model have?
- [ ] How much GPU memory does the model use? Check with `nvidia-smi`.
- [ ] What would full fine-tuning cost in memory (params × 12 bytes for Adam)?

---

## Exercise 2: SFT with LoRA

> Goal: Fine-tune the base model to follow instructions using LoRA

### 2.1 Prepare the Dataset

```python
from datasets import load_dataset

# Load an instruction-following dataset
dataset = load_dataset("tatsu-lab/alpaca", split="train")
print(f"Dataset size: {len(dataset)}")
print(f"Example: {dataset[0]}")

# Format into chat template
def format_alpaca(example):
    if example["input"]:
        text = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Input:\n{example['input']}\n\n"
            f"### Response:\n{example['output']}"
        )
    else:
        text = (
            f"### Instruction:\n{example['instruction']}\n\n"
            f"### Response:\n{example['output']}"
        )
    return {"text": text}

dataset = dataset.map(format_alpaca)

# Use a subset for faster training (full dataset for better results)
train_dataset = dataset.select(range(5000))
print(f"Training examples: {len(train_dataset)}")
print(f"Sample:\n{train_dataset[0]['text'][:300]}")
```

### 2.2 Apply LoRA

```python
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # Rank
    lora_alpha=32,                 # Scaling factor
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "v_proj",       # Attention projections
        "k_proj", "o_proj",       # More attention projections
    ],
    bias="none",
)

model = get_peft_model(model, lora_config)
print("\n--- After LoRA ---")
count_parameters(model)
model.print_trainable_parameters()
```

**Expected Output:**
```
trainable params: ~1.3M || all params: ~1,236M || trainable%: 0.10%
```

### 2.3 Train with SFTTrainer

```python
from trl import SFTTrainer, SFTConfig

training_args = SFTConfig(
    output_dir="./sft-lora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    max_seq_length=512,
    report_to="wandb",        # or "none" to disable
    run_name="sft-lora-r16",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train
trainer.train()

# Save the LoRA adapter
model.save_pretrained("./sft-lora-adapter")
tokenizer.save_pretrained("./sft-lora-adapter")
```

### 2.4 Compare Base vs SFT

```python
from peft import PeftModel

# Load base + LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)
sft_model = PeftModel.from_pretrained(base_model, "./sft-lora-adapter")

# Compare
print("=" * 60)
print("COMPARISON: Base Model vs SFT Model")
print("=" * 60)

test_prompts = [
    "### Instruction:\nWhat is the capital of France?\n\n### Response:\n",
    "### Instruction:\nExplain SLAM in 2 sentences.\n\n### Response:\n",
    "### Instruction:\nWrite a haiku about robots.\n\n### Response:\n",
]

for prompt in test_prompts:
    print(f"\n--- Prompt: {prompt[:50]}... ---")
    print(f"[BASE]  {generate(base_model, tokenizer, prompt, 100)}")
    print(f"[SFT]   {generate(sft_model, tokenizer, prompt, 100)}")
```

### 2.5 Questions

- [ ] How many trainable parameters does LoRA add? What percentage of total?
- [ ] Does the SFT model follow instructions better than the base model?
- [ ] What happens if you increase rank from 16 to 64? (try it)
- [ ] How does training loss change over epochs? (check wandb)

---

## Exercise 3: DPO Alignment

> Goal: Align the SFT model using Direct Preference Optimization

### 3.1 Create Preference Pairs

> **Note**: This is a mechanics demonstration with toy data. Real DPO training requires thousands of preference pairs. The goal here is to understand the training loop, not to achieve meaningful alignment.

```python
# Create synthetic preference pairs
# In practice, use human-annotated data or AI-generated preferences

preference_data = []
prompts_for_dpo = [
    "What is machine learning?",
    "How do robots navigate?",
    "Explain gradient descent.",
    "What is a transformer?",
    "How does SLAM work?",
]

# Generate pairs from SFT model (in practice, use human preferences)
for prompt in prompts_for_dpo:
    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
    
    # Generate multiple responses
    responses = []
    for _ in range(4):
        resp = generate(sft_model, tokenizer, formatted, max_new_tokens=150)
        # Extract just the response part
        resp = resp.split("### Response:\n")[-1].strip()
        responses.append(resp)
    
    # Simple heuristic: longer, more detailed = preferred (crude proxy)
    # In real DPO, use human annotators
    responses.sort(key=len, reverse=True)
    
    preference_data.append({
        "prompt": formatted,
        "chosen": responses[0],      # Preferred (more detailed)
        "rejected": responses[-1],    # Rejected (less detailed)
    })

from datasets import Dataset
dpo_dataset = Dataset.from_list(preference_data)
print(f"DPO dataset size: {len(dpo_dataset)}")
print(f"Example chosen: {dpo_dataset[0]['chosen'][:100]}...")
print(f"Example rejected: {dpo_dataset[0]['rejected'][:100]}...")
```

### 3.2 Train DPO

```python
from trl import DPOTrainer, DPOConfig

# Reload fresh SFT model as starting point
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)
sft_model = PeftModel.from_pretrained(base_model, "./sft-lora-adapter")

dpo_config = DPOConfig(
    output_dir="./dpo-output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-7,           # Very low LR for DPO
    beta=0.1,                      # KL penalty strength
    logging_steps=5,
    bf16=True,
    max_length=512,
    max_prompt_length=256,
    report_to="wandb",
    run_name="dpo-alignment",
)

# Note: DPO requires a reference model (automatically handled by DPOTrainer)
dpo_trainer = DPOTrainer(
    model=sft_model,
    args=dpo_config,
    train_dataset=dpo_dataset,
    processing_class=tokenizer,
)

dpo_trainer.train()
sft_model.save_pretrained("./dpo-lora-adapter")
```

### 3.3 Compare SFT vs DPO

```python
# Load DPO model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)
dpo_model = PeftModel.from_pretrained(base_model, "./dpo-lora-adapter")

print("=" * 60)
print("COMPARISON: SFT vs DPO")
print("=" * 60)

for prompt in test_prompts:
    print(f"\n--- Prompt: {prompt[:50]}... ---")
    # Reload SFT model for fair comparison
    sft_base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
    )
    sft_m = PeftModel.from_pretrained(sft_base, "./sft-lora-adapter")
    print(f"[SFT]  {generate(sft_m, tokenizer, prompt, 150)}")
    print(f"[DPO]  {generate(dpo_model, tokenizer, prompt, 150)}")
    del sft_base, sft_m
```

### 3.4 Questions

- [ ] How does DPO change the outputs compared to SFT?
- [ ] What happens if you increase beta (e.g., 0.5)? Decrease it (e.g., 0.01)?
- [ ] Why is the learning rate much lower for DPO than SFT?
- [ ] What are the limitations of using length as a preference proxy?

---

## Exercise 4: Quantization Comparison

> Goal: Compare inference quality, speed, and memory across precisions

### 4.1 Load Model at Different Precisions

```python
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Ungated model - no HF approval needed. For Llama-3.2-1B, request access at huggingface.co/meta-llama
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def load_model(precision):
    """Load model at specified precision."""
    if precision == "fp16":
        return AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
        )
    elif precision == "bf16":
        return AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
        )
    elif precision == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        return AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb_config, device_map="auto"
        )
    elif precision == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb_config, device_map="auto"
        )
    else:
        raise ValueError(f"Unknown precision: {precision}")
```

### 4.2 Benchmark

```python
def benchmark_model(model, tokenizer, prompt, n_tokens=100, n_runs=5):
    """Benchmark inference speed and measure GPU memory."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Warmup
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10)
    
    # Measure
    latencies = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=n_tokens)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)
    
    avg_latency = sum(latencies) / len(latencies)
    tokens_per_sec = n_tokens / avg_latency
    gpu_mem = torch.cuda.max_memory_allocated() / 1e9
    
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {
        "latency_s": avg_latency,
        "tokens_per_sec": tokens_per_sec,
        "gpu_memory_gb": gpu_mem,
        "output": output_text,
    }

# Run benchmarks
prompt = "### Instruction:\nExplain how a robot uses SLAM for navigation.\n\n### Response:\n"
results = {}

for precision in ["fp16", "bf16", "int8", "int4"]:
    print(f"\n{'='*40}")
    print(f"Loading model in {precision}...")
    torch.cuda.reset_peak_memory_stats()
    
    model = load_model(precision)
    result = benchmark_model(model, tokenizer, prompt)
    results[precision] = result
    
    print(f"  Latency:    {result['latency_s']:.2f}s")
    print(f"  Tokens/sec: {result['tokens_per_sec']:.1f}")
    print(f"  GPU Memory: {result['gpu_memory_gb']:.2f} GB")
    print(f"  Output:     {result['output'][:100]}...")
    
    del model
    torch.cuda.empty_cache()

# Summary table
print("\n" + "=" * 70)
print(f"{'Precision':<10} {'Latency (s)':<15} {'Tokens/s':<12} {'GPU Mem (GB)':<15}")
print("-" * 70)
for p, r in results.items():
    print(f"{p:<10} {r['latency_s']:<15.2f} {r['tokens_per_sec']:<12.1f} {r['gpu_memory_gb']:<15.2f}")
```

### 4.3 Questions

- [ ] Which precision gives the best speed/quality tradeoff?
- [ ] How much memory does INT4 save compared to FP16?
- [ ] Can you notice quality differences in the outputs?
- [ ] What precision would you choose for a Jetson Oort (8 GB RAM)?

---

## Exercise 5: Robotics LoRA Fine-tuning

> Goal: Create a domain-specific robotics assistant

### 5.1 Prepare Robotics Data

```python
import json

# Create robotics QA dataset from domain knowledge
robotics_data = [
    {
        "instruction": "What is SLAM in robotics?",
        "output": "SLAM (Simultaneous Localization and Mapping) is a technique where a robot builds a map of an unknown environment while simultaneously tracking its location within that map. It uses sensor data (lidar, cameras, IMU) to create and update the map in real-time."
    },
    {
        "instruction": "Explain the difference between global and local path planning.",
        "output": "Global path planning computes an optimal route from start to goal using a known map (e.g., A*, Dijkstra). Local path planning handles real-time obstacle avoidance along the global path using sensor data (e.g., DWA, TEB). Global plans are recomputed infrequently; local plans update at control frequency."
    },
    {
        "instruction": "What causes a robot to report NAV_ESTIMATED_STATE_NOT_FINITE?",
        "output": "NAV_ESTIMATED_STATE_NOT_FINITE occurs when the navigation estimator computes NaN or Inf values in the robot's state estimate (position, velocity, or orientation). Common causes include: sensor data gaps (sensorbar disconnect), encoder failures, sudden large jumps in odometry, or numerical instability in the Kalman filter."
    },
    # Add 50+ more robotics-specific QA pairs
    # Topics: ROS2, navigation, sensors, motor control, error codes, etc.
    {
        "instruction": "What is the sensorbar on an OKS AMR?",
        "output": "The sensorbar is a downward-facing optical sensor array on OKS AMR robots that reads floor-mounted QR code tiles for localization. It provides absolute position corrections to the navigation estimator, complementing wheel odometry and IMU data. Sensorbar failures can cause localization drift."
    },
    {
        "instruction": "Explain ROS 2 topics vs services vs actions.",
        "output": "ROS 2 Topics are pub/sub channels for streaming data (e.g., sensor readings) — one-to-many, no response expected. Services are synchronous request/response pairs for quick queries. Actions are for long-running tasks with feedback — they combine a goal, periodic feedback, and a result, with the ability to cancel."
    },
]

# Format for training
def format_robotics_qa(item):
    return {
        "text": f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
    }

formatted = [format_robotics_qa(item) for item in robotics_data]
robotics_dataset = Dataset.from_list(formatted)
print(f"Robotics dataset: {len(robotics_dataset)} examples")
```

### 5.2 Fine-tune on Robotics Data

```python
# Load base model with QLoRA for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# Apply LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,                          # Higher rank for domain knowledge
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
)

model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# Train
training_args = SFTConfig(
    output_dir="./robotics-lora-output",
    num_train_epochs=5,            # More epochs for small dataset
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=5,
    save_strategy="epoch",
    bf16=True,
    max_seq_length=512,
    report_to="wandb",
    run_name="robotics-qlora",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=robotics_dataset,
)

trainer.train()
model.save_pretrained("./robotics-lora-adapter")
tokenizer.save_pretrained("./robotics-lora-adapter")
```

### 5.3 Evaluate

```python
# Load robotics-tuned model
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
robotics_model = PeftModel.from_pretrained(base_model, "./robotics-lora-adapter")

# Test questions
test_questions = [
    "What causes sensorbar stiction on a warehouse robot?",
    "How does the navigation estimator fuse odometry and sensorbar data?",
    "What is the difference between ROS 2 Humble and Iron?",
    "Explain PID control in simple terms.",
    "What should you check when a robot reports MOTOR_OVERCURRENT?",
]

print("=" * 60)
print("ROBOTICS DOMAIN EVALUATION")
print("=" * 60)

for q in test_questions:
    prompt = f"### Instruction:\n{q}\n\n### Response:\n"
    print(f"\n--- Q: {q} ---")
    response = generate(robotics_model, tokenizer, prompt, max_new_tokens=200)
    response_only = response.split("### Response:\n")[-1].strip()
    print(f"A: {response_only}")
```

### 5.4 Questions

- [ ] Does the robotics LoRA model answer domain questions better?
- [ ] Where does it still fail? (knowledge boundaries)
- [ ] How would you improve the training data?
- [ ] What rank gave the best results for this domain?

---

## Self-Check

Before moving on, verify you can:

- [ ] Load a base model and test its (lack of) instruction following
- [ ] Apply LoRA to a model and count trainable parameters
- [ ] Fine-tune with SFTTrainer on instruction data
- [ ] Run DPO training on preference pairs
- [ ] Load models at different quantization levels (FP16, INT8, INT4)
- [ ] Benchmark inference speed and memory across precisions
- [ ] Fine-tune a domain-specific LoRA adapter
- [ ] Compare base vs fine-tuned model outputs qualitatively
- [ ] Merge LoRA weights into base model and verify equivalence
- [ ] Profile peak memory during training and identify the bottleneck

---

## Exercise 6: LoRA Merge and Memory Profiling

**Goal**: Understand LoRA's math deeply by implementing merge/unmerge,
and profile GPU memory to know exactly where your VRAM goes during training.

### 6.1 — LoRA Merge Implementation

```python
"""LoRA weight merging: convert adapter back to full-rank weights.

The math:
  During training:  output = Wx + BAx     (A ∈ R^{r×d_in}, B ∈ R^{d_out×r})
  After merge:      W_merged = W + B @ A  (full-rank weight, no runtime overhead)
  
Why merge?
  - Inference speed: no extra computation from adapter
  - Deployment: single model file, works with any inference engine
  - Can quantize the merged model for even more efficiency
"""

import torch
import torch.nn as nn
import copy


class LoRALinear(nn.Module):
    """Linear layer with LoRA adapter for manual experimentation."""
    
    def __init__(self, in_features: int, out_features: int, 
                 rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Scaling factor
        self.scaling = alpha / rank
        
        # Freeze base weights
        self.linear.weight.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.linear(x)
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_output + lora_output
    
    def merge_weights(self) -> nn.Linear:
        """Merge LoRA into base weights. Returns a standard nn.Linear.
        
        W_merged = W + scaling * B @ A
        """
        merged = nn.Linear(
            self.linear.in_features, self.linear.out_features, bias=False
        )
        merged.weight.data = (
            self.linear.weight.data + self.scaling * self.lora_B @ self.lora_A
        )
        return merged
    
    def unmerge_weights(self, merged_linear: nn.Linear) -> None:
        """Extract LoRA from a merged model (for continued training).
        
        Warning: This only works if you know the original base weights!
        ΔW = W_merged - W_base, then approximate with SVD: ΔW ≈ U[:,:r] @ S[:r] @ V[:r,:]
        """
        delta_W = merged_linear.weight.data - self.linear.weight.data
        
        # Low-rank approximation via SVD
        U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
        rank = self.lora_A.shape[0]
        
        # Reconstruct LoRA factors from top-r singular values
        self.lora_B.data = U[:, :rank] * S[:rank].sqrt()
        self.lora_A.data = (Vh[:rank, :].T * S[:rank].sqrt()).T
        # Adjust for scaling
        self.lora_B.data /= (self.scaling ** 0.5)
        self.lora_A.data /= (self.scaling ** 0.5)


# TODO 6a: Create a LoRALinear(768, 768, rank=16)
# 1. Initialize with random base weights
# 2. Do a few training steps to update LoRA
# 3. Merge weights
# 4. Verify: merged_output == original_output for the same input (within fp tolerance)

# TODO 6b: Measure the approximation error of unmerge_weights()
# 1. Create a LoRA layer, train it
# 2. Merge it
# 3. Unmerge it back
# 4. Compare original LoRA outputs vs reconstructed LoRA outputs
# 5. What's the maximum rank where SVD reconstruction is nearly perfect?
```

### 6.2 — GPU Memory Profiling

```python
"""Profile where GPU memory goes during LLM fine-tuning.

Memory breakdown for training a 7B model:
  - Model parameters: 7B × 2 bytes (fp16) = 14 GB
  - Gradients: 7B × 2 bytes = 14 GB  (or 0 for frozen params!)
  - Optimizer states (Adam): 7B × 8 bytes = 56 GB  (fp32 copy + m + v)
  - Activations: varies with batch size and seq length
  - LoRA advantage: only store grads + optimizer for rank*2*n_layers params
  
With LoRA (rank=16, on 7B model):
  - Model: 14 GB (frozen, no gradients needed)
  - LoRA params: ~16M × 2 bytes = 32 MB
  - LoRA grads: 32 MB
  - LoRA optimizer: ~128 MB (Adam states for 16M params)
  - Activations: ~2-4 GB (still need these for backward!)
  Total: ~18 GB vs 84+ GB for full fine-tuning
"""


def profile_training_memory(model: nn.Module, batch_size: int = 4, 
                           seq_len: int = 512) -> dict:
    """Profile memory usage during a training step.
    
    Requires GPU. Measures:
    1. Model loaded (no gradients)
    2. After optimizer initialization
    3. Peak during forward pass
    4. Peak during backward pass
    5. After optimizer step
    """
    if not torch.cuda.is_available():
        print("GPU required for memory profiling. Showing analytical estimates.")
        param_count = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            "total_params": param_count,
            "trainable_params": trainable_count,
            "model_memory_mb": param_count * 2 / 1e6,  # fp16
            "gradient_memory_mb": trainable_count * 2 / 1e6,
            "optimizer_memory_mb": trainable_count * 8 / 1e6,  # Adam states
            "estimated_activation_mb": batch_size * seq_len * 768 * 12 * 2 / 1e6,
        }
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Baseline
    model = model.cuda()
    mem_model = torch.cuda.memory_allocated() / 1e6
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4
    )
    mem_optimizer = torch.cuda.memory_allocated() / 1e6
    
    # Forward pass
    dummy_input = torch.randint(0, 1000, (batch_size, seq_len)).cuda()
    output = model(dummy_input)
    mem_forward = torch.cuda.max_memory_allocated() / 1e6
    
    # Backward pass
    loss = output.sum()
    loss.backward()
    mem_backward = torch.cuda.max_memory_allocated() / 1e6
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    mem_step = torch.cuda.memory_allocated() / 1e6
    
    return {
        "model_loaded_mb": mem_model,
        "with_optimizer_mb": mem_optimizer,
        "peak_forward_mb": mem_forward,
        "peak_backward_mb": mem_backward,
        "after_step_mb": mem_step,
        "activation_memory_mb": mem_forward - mem_optimizer,
    }


# TODO 6c: Profile a small transformer (6 layers, 512 dim) with full fine-tuning vs LoRA
# Print a comparison table showing memory at each stage

# TODO 6d: Experiment: how does batch_size affect activation memory?
# Profile at batch sizes [1, 2, 4, 8, 16] and plot memory vs batch_size
# Is the relationship linear? (It should be!)

# TODO 6e: Calculate the maximum batch size that fits in 24GB for:
# - 1B model, full fine-tuning
# - 7B model, LoRA rank 16
# - 7B model, QLoRA (4-bit base + LoRA)
```

### 6.3 — Exercises

| Task | Description | Difficulty |
|------|-------------|------------|
| 6a | Implement and verify LoRA merge | ★☆☆ |
| 6b | Test unmerge via SVD approximation | ★★☆ |
| 6c | Profile memory: full vs LoRA | ★★☆ |
| 6d | Batch size vs memory plot | ★☆☆ |
| 6e | Max batch size calculations | ★★★ |

---

## Stretch Goals

### S1: Multi-LoRA Serving
Load the base model once, then swap between SFT LoRA and robotics LoRA adapters.
Measure adapter switching latency.

### S2: LoRA Rank Ablation
Train LoRA adapters with rank {4, 8, 16, 32, 64, 128}. Plot:
- Training loss vs rank
- Eval quality vs rank
- Trainable parameters vs rank
Find the "knee" where more rank stops helping.

### S3: Merge and Export
Merge the LoRA adapter into the base model weights. Export to GGUF format.
Run inference with llama.cpp on CPU.

```bash
# Convert to GGUF (requires llama.cpp)
python convert_hf_to_gguf.py ./merged-model --outtype f16
./llama-quantize ./merged-model.gguf ./merged-model-Q4_K_M.gguf Q4_K_M
./llama-cli -m ./merged-model-Q4_K_M.gguf -p "### Instruction:\nWhat is SLAM?\n\n### Response:\n"
```

### S4: Evaluate with lm-eval-harness
Run standard benchmarks on your fine-tuned model:
```bash
lm_eval --model hf \
    --model_args pretrained=./merged-model \
    --tasks mmlu,hellaswag,arc_easy \
    --batch_size 8
```

### S5: Full Robotics Assistant (→ Capstone)
Expand the robotics dataset to 500+ QA pairs. Include:
- OKS error codes and troubleshooting
- ROS 2 concepts and commands
- Navigation algorithms
- Sensor fusion explanations
This feeds into the Phase III capstone project.

---

## Notes

*This exercise is part of the LLM-to-VLA curriculum, Phase III.*
*Study notes: [06 — LLM Training & Alignment](../study-notes/06-llm-training-alignment.md)*
*Capstone: [03 — Robotics Assistant](../projects/03-robotics-assistant/README.md)*
