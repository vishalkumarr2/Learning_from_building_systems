# 06 — LLM Training & Alignment

> Phase III · Days 31–36 · ~15 hours
> Prerequisites: 05-gpt-scaling
> Learning Objectives: Understand the full LLM pipeline: pretrain → SFT → RLHF/DPO; fine-tune with LoRA

---

## Table of Contents

1. [Pretraining Recap](#1-pretraining-recap)
2. [Supervised Fine-Tuning (SFT)](#2-supervised-fine-tuning-sft)
3. [RLHF — Reinforcement Learning from Human Feedback](#3-rlhf--reinforcement-learning-from-human-feedback)
4. [DPO — Direct Preference Optimization](#4-dpo--direct-preference-optimization)
5. [LoRA — Low-Rank Adaptation](#5-lora--low-rank-adaptation)
6. [LLM Evaluation](#6-llm-evaluation)
7. [Synthesis & Phase Gate](#7-synthesis--phase-gate)
8. [Key Equations & Training Pipeline Diagram](#8-key-equations--training-pipeline-diagram)
9. [Key Takeaways & Paper References](#9-key-takeaways--paper-references)
10. [Connection to Thread](#10-connection-to-thread)

---

## 1. Pretraining Recap

### 1.1 Data Curation

Pretraining data quality determines model capability more than any other factor.

**Major Pretraining Datasets:**

| Dataset | Size | Key Properties |
|---------|------|---------------|
| The Pile | 825 GB | 22 diverse subsets (academic, code, web) |
| RedPajama | 1.2T tokens | Open reproduction of LLaMA training data |
| FineWeb | 15T tokens | Deduplicated, quality-filtered CommonCrawl |
| RefinedWeb | 5T tokens | Strict dedup + quality heuristics |
| StarCoder | 783 GB | Code-focused, 80+ languages |

**Data Pipeline:**

```
Raw Web Crawl
    ↓
Language Detection (fastText)
    ↓
Quality Filtering (perplexity, heuristics)
    ↓
Deduplication (MinHash, exact substring)
    ↓
PII Removal
    ↓
Toxicity Filtering
    ↓
Domain Mixing (web + books + code + academic)
    ↓
Tokenized Training Shards
```

**Key Insight:** Data quality > data quantity. FineWeb showed that aggressive filtering
of a larger dataset beats using everything. The Chinchilla paper showed that most
models were trained on too little data for their parameter count.

### 1.2 Tokenization at Scale

**BPE (Byte Pair Encoding) Review:**
- Start with byte-level vocabulary
- Iteratively merge the most frequent pair
- Vocabulary size: typically 32K–128K tokens

**Tokenizer Choices:**

| Model | Tokenizer | Vocab Size |
|-------|-----------|-----------|
| GPT-4 | cl100k_base | 100,256 |
| LLaMA 2 | SentencePiece BPE | 32,000 |
| LLaMA 3 | tiktoken-based BPE | 128,256 |
| Mistral | SentencePiece BPE | 32,000 |

**Why Tokenizer Matters:**
- Fertility (tokens per word) affects effective context length
- Multilingual coverage requires larger vocabularies
- Code tokenization needs special handling (whitespace-sensitive)
- A 128K vocabulary compresses English ~20% better than 32K

### 1.3 Distributed Training

**The Parallelism Zoo:**

```
┌─────────────────────────────────────────────┐
│           Distributed Training              │
├──────────────┬──────────────┬───────────────┤
│ Data Parallel│ Tensor       │ Pipeline      │
│ (DP/DDP/FSDP)│ Parallel (TP)│ Parallel (PP) │
├──────────────┼──────────────┼───────────────┤
│ Same model   │ Split layers │ Split layers  │
│ on each GPU, │ across GPUs  │ sequentially  │
│ split data   │ within a     │ across GPUs   │
│              │ single layer │               │
└──────────────┴──────────────┴───────────────┘
```

**FSDP (Fully Sharded Data Parallel):**
- Shards optimizer states, gradients, AND parameters across GPUs
- Each GPU holds only 1/N of the full model state
- All-gather before forward pass, reduce-scatter after backward
- Memory: $O(\text{params}/N)$ per GPU vs $O(\text{params})$ for DDP

**DeepSpeed ZeRO Stages:**

| Stage | What's Sharded | Memory Savings |
|-------|---------------|----------------|
| ZeRO-1 | Optimizer states | ~4x |
| ZeRO-2 | + Gradients | ~8x |
| ZeRO-3 | + Parameters | ~N×  (N = #GPUs) |

**Mixed Precision Training:**
- Forward pass: FP16 or BF16 (half memory, faster compute)
- Master weights: FP32 (for numerical stability)
- Loss scaling: prevent gradient underflow in FP16
- BF16 preferred: same range as FP32, no loss scaling needed

### 1.4 Compute Budgeting

**Chinchilla Scaling Law:**

$$C \approx 6ND$$

where:
- $C$ = compute (FLOPs)
- $N$ = parameters
- $D$ = training tokens

**Chinchilla-Optimal Ratio:** $D \approx 20N$

| Model | Params (N) | Optimal Tokens (20N) | Actual Tokens |
|-------|-----------|---------------------|---------------|
| LLaMA 2 7B | 7B | 140B | 2T (over-trained) |
| LLaMA 2 70B | 70B | 1.4T | 2T |
| Chinchilla | 70B | 1.4T | 1.4T (optimal) |

**Over-training Rationale:** Train smaller models on more data → better inference
efficiency. A 7B model trained on 2T tokens is cheaper to serve than a 70B model
trained on 1.4T tokens, even if the latter is compute-optimal for training.

---

## 2. Supervised Fine-Tuning (SFT)

> Day 31 · ~3 hours

### 2.1 From Base Model to Assistant

**The Problem:** A pretrained model is a text completion engine. It predicts the
next token given context. It does NOT follow instructions.

```
Base Model Input:  "What is the capital of France?"
Base Model Output: "What is the capital of Germany? What is the capital of Italy?"
                   (continues the pattern of questions)

SFT Model Input:  "What is the capital of France?"
SFT Model Output: "The capital of France is Paris."
                   (follows the instruction)
```

**SFT transforms the model from:**
- A document completer → An instruction follower
- P(next token | document) → P(response | instruction)

### 2.2 Instruction Tuning

**The Key Idea:** Fine-tune on (instruction, response) pairs where the response
demonstrates the desired behavior.

**Training Format:**

```
### Instruction:
{user instruction}

### Input:
{optional additional context}

### Response:
{desired model response}
```

**Loss Computation:**
- Compute cross-entropy loss ONLY on the response tokens
- Instruction tokens are included in context but masked from loss
- This teaches the model to generate responses, not repeat instructions

$$\mathcal{L}_{SFT} = -\sum_{t \in \text{response}} \log P_\theta(x_t | x_{<t})$$

### 2.3 Chat Templates

Modern SFT uses structured chat templates:

**ChatML Format (used by many models):**
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
The capital of France is Paris.<|im_end|>
```

**LLaMA Chat Format:**
```
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

What is the capital of France? [/INST] The capital of France is Paris. </s>
```

**Why Templates Matter:**
- Consistent format helps the model learn role boundaries
- Special tokens delineate turns
- System prompts enable behavioral steering
- Multi-turn conversations require proper turn management

### 2.4 SFT Datasets

| Dataset | Size | Source | Style |
|---------|------|--------|-------|
| Alpaca | 52K | GPT-3.5 generated | Single-turn instructions |
| Dolly | 15K | Human-written | Diverse instructions |
| OpenAssistant | 161K | Human conversations | Multi-turn dialogue |
| ShareGPT | ~90K | User-shared ChatGPT convos | Real user interactions |
| UltraChat | 1.47M | LLM-generated | Diverse, multi-turn |
| FLAN Collection | 1.8K tasks | Academic NLP tasks | Task-focused |

**Quality vs Quantity:**
- LIMA showed that 1,000 high-quality examples can produce strong results
- Quality of data > quantity for SFT
- Diversity of instructions matters more than volume

### 2.5 SFT Training Details

**Hyperparameters (typical):**
```python
{
    "learning_rate": 2e-5,        # Lower than pretraining
    "epochs": 3,                   # Few epochs to avoid overfitting
    "batch_size": 128,             # Effective batch size
    "max_seq_length": 2048,        # Or model's context length
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "lr_scheduler": "cosine",
}
```

**Common Pitfalls:**
1. **Overfitting:** SFT datasets are small → train for few epochs
2. **Catastrophic forgetting:** Model loses pretraining knowledge
3. **Mode collapse:** Model gives same response to different inputs
4. **Loss on instruction tokens:** Accidentally training on inputs too

---

## 3. RLHF — Reinforcement Learning from Human Feedback

> Day 32 · ~3 hours

### 3.1 Why RLHF?

**SFT Limitation:** SFT teaches the model to imitate demonstrations, but:
- Human demonstrations vary in quality
- Hard to demonstrate "don't do X" (negative examples)
- Easier to judge quality than to produce it

**RLHF Insight:** Humans can compare two responses more reliably than they can
write the perfect response. Use comparison data to train alignment.

### 3.2 The RLHF Pipeline

```
┌──────────────────────────────────────────────────────┐
│                   RLHF Pipeline                       │
│                                                       │
│  Step 1: SFT Model                                   │
│  ┌─────────────┐                                     │
│  │ Pretrained   │──→ SFT on demonstrations ──→ π_SFT │
│  │ Base Model   │                                     │
│  └─────────────┘                                     │
│                                                       │
│  Step 2: Reward Model                                │
│  ┌─────────────┐     ┌──────────────┐                │
│  │ π_SFT       │──→  │ Generate y_1 │──→ Human       │
│  │ (generates  │     │ Generate y_2 │    ranks        │
│  │  responses) │     └──────────────┘    y_1 > y_2    │
│  └─────────────┘                              │       │
│                                               ↓       │
│                                     Train Reward      │
│                                     Model R(x,y)      │
│                                                       │
│  Step 3: PPO Optimization                            │
│  ┌─────────────┐                                     │
│  │ π_SFT       │──→ PPO with reward R(x,y) ──→ π_RL │
│  │ (init)      │    + KL penalty to π_SFT            │
│  └─────────────┘                                     │
└──────────────────────────────────────────────────────┘
```

### 3.3 Reward Model Training

**Data Collection:**
1. Sample prompt $x$ from a dataset
2. Generate two responses $y_1, y_2$ from the SFT model
3. Human annotator labels preference: $y_w \succ y_l$ (winner vs loser)

**Bradley-Terry Model:**
The reward model learns a scalar reward $r_\theta(x, y)$ such that:

$$P(y_w \succ y_l | x) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

where $\sigma$ is the sigmoid function.

**Reward Model Loss:**

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))\right]$$

**Architecture:**
- Typically: same architecture as the LLM, with a scalar head replacing the LM head
- Initialize from the SFT model checkpoint
- Output: single scalar per (prompt, response) pair

### 3.4 PPO for LLM Alignment

**Objective:**

$$\max_{\pi_\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)}\left[r_\phi(x, y)\right] - \beta \cdot \text{KL}\left[\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)\right]$$

**Components:**
- $r_\phi(x, y)$: learned reward model score
- $\pi_\theta$: the policy (LLM being optimized)
- $\pi_{\text{ref}}$: reference policy (usually the SFT model)
- $\beta$: KL penalty coefficient
- KL term prevents the model from drifting too far from the SFT model

**PPO Update (simplified for LLMs):**

$$\mathcal{L}^{PPO} = -\min\left(\frac{\pi_\theta(y|x)}{\pi_{\text{old}}(y|x)} A, \text{clip}\left(\frac{\pi_\theta(y|x)}{\pi_{\text{old}}(y|x)}, 1-\epsilon, 1+\epsilon\right) A\right)$$

where $A$ is the advantage estimated using the reward model.

### 3.5 KL Penalty and Reward Hacking

**Problem: Reward Hacking**
Without the KL penalty, the model finds adversarial outputs that get high reward
scores but are actually low quality (exploiting reward model weaknesses).

**Example of Reward Hacking:**
```
Prompt: "Write a poem about nature"

Without KL penalty (hacked):
"Beautiful beautiful beautiful nature nature nature trees trees trees
 wonderful wonderful wonderful amazing amazing amazing perfect!!!"
 → Reward model gives high score (positive sentiment words)
 → But obviously terrible output

With KL penalty:
"Beneath the oak, the river hums a silver song..."
 → Stays close to natural language distribution
```

**KL Penalty Effect:**

$$\text{KL}[\pi_\theta \| \pi_{\text{ref}}] = \sum_y \pi_\theta(y|x) \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

Higher $\beta$ → more conservative (stays closer to SFT model)
Lower $\beta$ → more optimization (higher reward but risk of reward hacking)

### 3.6 RLHF Challenges

1. **Reward model quality:** Garbage in, garbage out
2. **Reward hacking:** Model exploits reward model weaknesses
3. **Training instability:** PPO is notoriously hard to tune
4. **Computational cost:** 4 models in memory (policy, ref, reward, value)
5. **Human labeling:** Expensive, inconsistent, biased
6. **KL-reward tradeoff:** Hard to find the right $\beta$

---

## 4. DPO — Direct Preference Optimization

> Day 33 · ~3 hours

### 4.1 Motivation

**RLHF is Complex:**
- Need to train a separate reward model
- Need to run PPO (unstable, hyperparameter-sensitive)
- 4 models in memory simultaneously
- Reward hacking requires careful KL tuning

**DPO Insight:** The optimal policy under the RLHF objective has a closed-form
solution. We can reparameterize the reward model in terms of the policy itself
and optimize directly.

### 4.2 Mathematical Derivation

**Starting Point:** The RLHF objective:

$$\max_\pi \mathbb{E}_{x,y \sim \pi}[r(x,y)] - \beta \text{KL}[\pi \| \pi_{\text{ref}}]$$

**Closed-form optimal policy:**

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$$

**Rearranging for the reward:**

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

**Key insight:** We can express the reward entirely in terms of the policy ratio.
The partition function $Z(x)$ cancels when we compute preference probabilities.

### 4.3 The DPO Loss

$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

**In words:** Increase the log-probability ratio of preferred responses relative
to dispreferred responses, compared to the reference model.

**Gradient Intuition:**

$$\nabla_\theta \mathcal{L}_{DPO} \propto -\beta \sigma(\hat{r}_l - \hat{r}_w)\left[\nabla_\theta \log \pi_\theta(y_w|x) - \nabla_\theta \log \pi_\theta(y_l|x)\right]$$

where $\hat{r} = \beta \log \frac{\pi_\theta(y)}{\pi_{\text{ref}}(y)}$ is the implicit reward.

**Interpretation:**
- When the model wrongly prefers the losing response → large gradient update
- When it correctly prefers the winner → small update
- The implicit reward automatically prevents reward hacking via the reference ratio

### 4.4 DPO vs RLHF vs KTO

| Aspect | RLHF (PPO) | DPO | KTO |
|--------|-----------|-----|-----|
| **Reward Model** | Separate model needed | Implicit (in policy) | Implicit |
| **Training** | PPO (unstable) | Standard supervised | Standard supervised |
| **Data** | Preference pairs | Preference pairs | Binary feedback (good/bad) |
| **Models in memory** | 4 (policy, ref, reward, value) | 2 (policy, ref) | 2 (policy, ref) |
| **Hyperparameters** | Many (PPO + KL) | Few (β) | Few (β) |
| **Stability** | Fragile | Robust | Robust |
| **Performance** | Gold standard | Competitive | Competitive |
| **Compute** | Very expensive | Moderate | Moderate |

**KTO (Kahneman-Tversky Optimization):**
- Doesn't need paired preferences, just "thumbs up/down"
- Based on prospect theory (losses loom larger than gains)
- Even simpler data requirements than DPO

### 4.5 DPO Training Details

**Hyperparameters:**
```python
{
    "beta": 0.1,                   # KL penalty strength
    "learning_rate": 5e-7,         # Very low LR
    "epochs": 1-3,                 # Few epochs
    "batch_size": 64,
    "max_length": 1024,
    "max_prompt_length": 512,
    "loss_type": "sigmoid",        # or "hinge", "ipo"
}
```

**DPO Variants:**
- **IPO (Identity Preference Optimization):** Addresses DPO overfitting
- **cDPO:** Label-smoothed DPO for noisy preferences
- **ORPO:** Combines SFT and preference optimization in one step
- **SimPO:** Reference-free, uses length-normalized log-probability

### 4.6 Practical Considerations

**When to Use DPO over RLHF:**
- Limited compute budget
- Smaller models (< 13B)
- Preference data is available
- Training stability is important

**When RLHF Still Wins:**
- Very large models with dedicated infra
- Need to iterate on reward signal
- Complex reward functions (not just A vs B)
- Frontier model training (GPT-4, Claude)

---

## 5. LoRA — Low-Rank Adaptation

> Day 34 · ~3 hours

### 5.1 The Full Fine-Tuning Problem

**Cost of Full Fine-Tuning:**

| Model | Params | FP16 Memory | Optimizer (Adam) | Total |
|-------|--------|------------|-----------------|-------|
| 7B | 7B | 14 GB | 56 GB | ~70 GB |
| 13B | 13B | 26 GB | 104 GB | ~130 GB |
| 70B | 70B | 140 GB | 560 GB | ~700 GB |

Full fine-tuning requires storing:
- Model parameters (2 bytes/param in FP16)
- Gradients (2 bytes/param)
- Optimizer states (8 bytes/param for Adam: momentum + variance + FP32 copy)

**This is prohibitively expensive for most practitioners.**

### 5.2 LoRA Core Idea

**Hypothesis:** Fine-tuning changes to weight matrices are low-rank.

For a pretrained weight matrix $W_0 \in \mathbb{R}^{d \times d}$:

$$W = W_0 + \Delta W = W_0 + BA$$

where:
- $B \in \mathbb{R}^{d \times r}$ (down-projection)
- $A \in \mathbb{R}^{r \times d}$ (up-projection)
- $r \ll d$ (rank, typically 4-64)

**Visualization:**

```
Input x (dim d)
    │
    ├──────────────────────────┐
    │                          │
    ↓                          ↓
┌────────┐              ┌──────────┐
│  W₀    │              │  A (r×d) │  ← Trainable
│ (d×d)  │              └──────────┘
│ Frozen │                    │
└────────┘                    ↓
    │                   ┌──────────┐
    │                   │  B (d×r) │  ← Trainable
    │                   └──────────┘
    │                          │
    ↓                          ↓
    └───────── + ──────────────┘
              │
              ↓
         Output (dim d)
```

### 5.3 Parameter Efficiency

**Trainable Parameters:**

$$\text{LoRA params} = r \times (d_{\text{in}} + d_{\text{out}})$$

**Example for LLaMA 7B:**
- Hidden dim $d = 4096$
- Full attention weight: $4096 \times 4096 = 16.7M$ parameters
- LoRA rank 16: $16 \times (4096 + 4096) = 131K$ parameters
- **Reduction: 128×**

**Typical LoRA Configuration:**

| Model Size | Rank | Alpha | Trainable % |
|-----------|------|-------|------------|
| 7B | 16 | 32 | 0.1-0.5% |
| 13B | 16 | 32 | 0.05-0.3% |
| 70B | 64 | 128 | 0.05-0.2% |

### 5.4 LoRA Details

**Initialization:**
- $A$: Gaussian initialization
- $B$: Zero initialization
- At start: $\Delta W = BA = 0$ (model starts at pretrained weights)

**Scaling Factor:**

$$h = W_0 x + \frac{\alpha}{r} B A x$$

where $\alpha$ is a scaling hyperparameter (often set to $2r$).

**Which Layers to Apply LoRA:**
- Query and Value projections in attention (most common)
- All attention projections (Q, K, V, O)
- Attention + MLP (more expensive, often better)
- The choice affects both cost and quality

```python
# Typical target modules for LLaMA
target_modules = [
    "q_proj", "v_proj",           # Minimum (cheapest)
    "k_proj", "o_proj",           # + key and output
    "gate_proj", "up_proj",       # + MLP
    "down_proj",                   # Full LoRA
]
```

### 5.5 QLoRA — Quantized LoRA

**Innovation:** Combine 4-bit quantization of the base model with LoRA adapters.

**QLoRA Components:**
1. **4-bit NormalFloat (NF4):** Information-theoretically optimal quantization
2. **Double Quantization:** Quantize the quantization constants
3. **Paged Optimizers:** Handle memory spikes via CPU offloading

**Memory Comparison:**

| Method | 7B Model | 13B Model | 70B Model |
|--------|---------|----------|----------|
| Full FT (FP16) | ~70 GB | ~130 GB | ~700 GB |
| LoRA (FP16) | ~15 GB | ~28 GB | ~145 GB |
| QLoRA (4-bit) | ~5 GB | ~9 GB | ~40 GB |

**QLoRA makes fine-tuning a 70B model possible on a single 48GB GPU.**

### 5.6 LoRA Merging and Serving

**Merging LoRA back into base weights:**

$$W_{\text{merged}} = W_0 + \frac{\alpha}{r} B A$$

After merging:
- No additional inference latency
- No additional memory for adapters
- Can't easily switch between tasks

**Multi-LoRA Serving:**
- Keep base model in memory
- Load/swap LoRA adapters per request
- Enables multi-tenant serving with different specializations

---

## 6. LLM Evaluation

> Day 35 · ~3 hours

### 6.1 The Evaluation Challenge

LLM evaluation is fundamentally hard because:
- Open-ended generation has no single correct answer
- Quality is subjective and context-dependent
- Models can game benchmarks while failing in real use
- Human evaluation is expensive and inconsistent

### 6.2 Automatic Benchmarks

**Perplexity:**

$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(x_i | x_{<i})\right)$$

- Lower = better
- Measures how "surprised" the model is by text
- Only works for comparing models with same tokenizer
- Doesn't measure instruction following or helpfulness

**MMLU (Massive Multitask Language Understanding):**
- 57 subjects: STEM, humanities, social sciences, etc.
- 4-way multiple choice
- Tests factual knowledge and reasoning
- Scores: GPT-4 ~87%, LLaMA 3 70B ~82%, Random ~25%

**HumanEval (Code Generation):**
- 164 Python programming problems
- Pass@k metric: probability of passing in k samples
- Tests: function signature → complete implementation
- Scores: GPT-4 ~87%, Claude 3.5 ~92%

**Other Key Benchmarks:**

| Benchmark | What It Measures | Format |
|-----------|-----------------|--------|
| HellaSwag | Common sense reasoning | Multiple choice |
| ARC | Science reasoning | Multiple choice |
| GSM8K | Math word problems | Numerical answer |
| TruthfulQA | Truthfulness | Multiple choice |
| BBH | Hard reasoning | Mixed |
| MATH | Competition mathematics | Proof/answer |

### 6.3 Chat/Instruction Evaluation

**MT-Bench:**
- 80 multi-turn questions across 8 categories
- GPT-4 judges model responses on 1-10 scale
- Categories: writing, roleplay, reasoning, math, coding, extraction, STEM, humanities
- Correlates well with human preferences

**Chatbot Arena (LMSYS):**
- Crowdsourced blind A/B testing
- Users chat with two anonymous models
- Vote for preferred response
- Elo rating system (like chess)
- Most reliable ranking of chat models

**AlpacaEval:**
- 805 instruction-following tasks
- GPT-4 judges if model output beats reference
- Win-rate metric
- Fast and cheap proxy for Chatbot Arena

### 6.4 LLM-as-Judge

Using strong LLMs (GPT-4) to evaluate weaker models:

**Advantages:**
- Scalable and reproducible
- Consistent (no inter-annotator disagreement)
- Cheap compared to human evaluation
- Can evaluate many dimensions at once

**Limitations:**
- Biases: verbosity bias, position bias, self-preference bias
- Can't evaluate beyond judge model's capability
- May miss subtle quality differences
- Circular: using AI to evaluate AI

### 6.5 Evaluation Best Practices

1. **Use multiple benchmarks** — no single metric captures everything
2. **Report confidence intervals** — small benchmark → high variance
3. **Watch for contamination** — test data in training data
4. **Separate capabilities** — factual knowledge ≠ reasoning ≠ instruction following
5. **Real-world testing** — benchmarks are proxies, not ground truth

---

## 7. Synthesis & Phase Gate

> Day 36 · ~2 hours

### 7.1 Checkpoint Questions

Test your understanding of the full LLM pipeline:

**Pretraining:**
1. Why does data quality matter more than quantity for LLM pretraining?
2. Explain the Chinchilla scaling law and why many models are "over-trained."
3. What are the three types of parallelism in distributed training?

**SFT:**
4. Why does a base model fail at instruction following?
5. What's the key difference between SFT loss and pretraining loss?
6. Why do we mask instruction tokens from the loss?

**RLHF:**
7. Draw the 3-step RLHF pipeline from memory.
8. What is reward hacking? How does the KL penalty prevent it?
9. Why is RLHF preferred over just doing more SFT?

**DPO:**
10. Write the DPO loss from memory. Explain each term.
11. Why is DPO simpler than RLHF? What does it eliminate?
12. When might RLHF still be preferred over DPO?

**LoRA:**
13. Write the LoRA decomposition: $W = W_0 + BA$. What are the dimensions?
14. Calculate the parameter savings for rank 16 on a 4096×4096 weight matrix.
15. How does QLoRA further reduce memory requirements?

**Evaluation:**
16. Why is LLM evaluation fundamentally difficult?
17. What are the strengths and weaknesses of using GPT-4 as a judge?
18. Compare MMLU, Chatbot Arena, and MT-Bench: what does each measure?

### 7.2 Concept Map

```
                    Pretraining
                   (data + compute)
                        │
                        ↓
                   Base Model
                   (next-token predictor)
                        │
                   ┌────┴────┐
                   │   SFT   │ ← instruction-response pairs
                   └────┬────┘
                        │
                   SFT Model
                   (instruction follower)
                        │
              ┌─────────┼─────────┐
              │         │         │
         ┌────┴───┐ ┌───┴──┐ ┌───┴──┐
         │  RLHF  │ │ DPO  │ │ KTO  │
         └────┬───┘ └───┬──┘ └───┬──┘
              │         │        │
              └─────────┼────────┘
                        │
                  Aligned Model
                  (helpful, harmless, honest)
                        │
                  ┌─────┴─────┐
                  │   LoRA    │ ← task-specific adaptation
                  └─────┬─────┘
                        │
                  Specialized Model
                  (domain expert)
```

---

## 8. Key Equations & Training Pipeline Diagram

### 8.1 Equation Summary

**SFT Loss:**
$$\mathcal{L}_{SFT} = -\sum_{t \in \text{response}} \log P_\theta(x_t | x_{<t})$$

**Reward Model (Bradley-Terry):**
$$\mathcal{L}_{RM} = -\mathbb{E}\left[\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))\right]$$

**RLHF Objective:**
$$\max_\pi \mathbb{E}[r(x,y)] - \beta \text{KL}[\pi \| \pi_{\text{ref}}]$$

**DPO Loss:**
$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

**LoRA:**
$$W = W_0 + \frac{\alpha}{r} BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times d}$$

**Chinchilla Scaling:**
$$C \approx 6ND, \quad D_{\text{optimal}} \approx 20N$$

### 8.2 Full Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Training Pipeline                         │
│                                                                  │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐                │
│  │ Raw Data │────→│ Tokenize │────→│ Pretrain │                 │
│  │ (web,    │     │ (BPE)    │     │ (next-   │                 │
│  │  books,  │     └──────────┘     │  token)  │                 │
│  │  code)   │                      └────┬─────┘                 │
│  └──────────┘                           │                        │
│                                    Base Model                    │
│                                         │                        │
│                                    ┌────┴────┐                   │
│                                    │   SFT   │← (instruction,   │
│                                    │         │   response) pairs │
│                                    └────┬────┘                   │
│                                         │                        │
│                                    SFT Model                     │
│                                         │                        │
│                              ┌──────────┼──────────┐            │
│                              │          │          │             │
│                         ┌────┴───┐ ┌────┴───┐ ┌───┴───┐        │
│                         │  RLHF  │ │  DPO   │ │  KTO  │        │
│                         │(reward │ │(direct │ │(binary│        │
│                         │ model  │ │ optim) │ │ fdbk) │        │
│                         │ + PPO) │ │        │ │       │        │
│                         └────┬───┘ └────┬───┘ └───┬───┘        │
│                              │          │         │              │
│                              └──────────┼─────────┘             │
│                                         │                        │
│                                   Aligned Model                  │
│                                         │                        │
│                              ┌──────────┼──────────┐            │
│                              │          │          │             │
│                         ┌────┴───┐ ┌────┴───┐ ┌───┴───┐        │
│                         │ LoRA   │ │ QLoRA  │ │ Full  │        │
│                         │ (rank  │ │(4-bit  │ │  FT   │        │
│                         │  16)   │ │+LoRA)  │ │       │        │
│                         └────┬───┘ └────┬───┘ └───┬───┘        │
│                              │          │         │              │
│                              └──────────┼─────────┘             │
│                                         │                        │
│                                Specialized Model                 │
│                                         │                        │
│                              ┌──────────┼──────────┐            │
│                              │          │          │             │
│                         ┌────┴───┐ ┌────┴───┐ ┌───┴───┐        │
│                         │Quantize│ │ Merge  │ │ Serve │        │
│                         │(INT4/8)│ │ LoRA   │ │(vLLM) │        │
│                         └────────┘ └────────┘ └───────┘        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Key Takeaways & Paper References

### 9.1 Key Takeaways

1. **LLM training is a pipeline, not a single step:** pretrain → SFT → alignment → specialization
2. **SFT transforms completers into assistants** using (instruction, response) pairs
3. **RLHF optimizes for human preferences** using a learned reward model + PPO
4. **DPO eliminates the reward model** by optimizing preferences directly — simpler, cheaper, competitive
5. **The KL penalty is critical** — without it, both RLHF and DPO degenerate via reward hacking
6. **LoRA makes fine-tuning accessible** — train 0.1% of parameters, get 90%+ of full fine-tuning performance
7. **QLoRA further democratizes** — fine-tune 70B models on a single consumer GPU
8. **LLM evaluation is an open problem** — use multiple benchmarks and real-world testing
9. **Data quality dominates** at every stage: pretraining data, SFT data, preference data

### 9.2 Paper References

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| **InstructGPT** (Ouyang et al.) | 2022 | Introduced the SFT → RLHF pipeline for alignment |
| **Training language models to follow instructions with human feedback** | 2022 | Detailed InstructGPT methodology |
| **DPO: Direct Preference Optimization** (Rafailov et al.) | 2023 | Eliminated reward model from RLHF |
| **LoRA: Low-Rank Adaptation** (Hu et al.) | 2021 | Parameter-efficient fine-tuning |
| **QLoRA** (Dettmers et al.) | 2023 | 4-bit quantization + LoRA |
| **Chinchilla** (Hoffmann et al.) | 2022 | Compute-optimal scaling laws |
| **LIMA** (Zhou et al.) | 2023 | Less Is More for Alignment (1K examples suffice) |
| **KTO** (Ethayarajh et al.) | 2024 | Binary feedback alignment (no pairs needed) |
| **ORPO** (Hong et al.) | 2024 | Combined SFT + preference optimization |
| **Self-Play Fine-Tuning (SPIN)** | 2024 | Self-play without human preferences |
| **Constitutional AI** (Bai et al.) | 2022 | AI-generated feedback for alignment |

### 9.3 Suggested Reading Order

1. **Start:** InstructGPT (the original RLHF paper — clear and complete)
2. **Then:** LoRA (elegant and practical)
3. **Then:** DPO (the math is beautiful once you see the derivation)
4. **Then:** QLoRA (enables practical experiments)
5. **Optional:** Chinchilla, LIMA, KTO (for depth)

---

## 10. Connection to Thread

### The Compression Lens

**Alignment as Compression Direction:**
- Pretraining compresses all of human knowledge into weights
- SFT compresses the format of helpful responses
- RLHF/DPO compress what humans actually prefer
- Alignment = teaching the model what to compress FOR (helpfulness vs. harmfulness)

**LoRA as Efficient Compression:**
- Full fine-tuning: learn $\Delta W \in \mathbb{R}^{d \times d}$ = $d^2$ parameters
- LoRA: learn $\Delta W = BA$ with rank $r$ = $r(d+d)$ parameters
- LoRA works because task-specific changes are low-rank
- Low-rank = high compressibility = the task is simpler than the full model

**The Path Forward:**
- LLMs compress language and reasoning
- But they operate in token space — they can't SEE or ACT
- Vision models compress visual information
- Action models compress motor behavior
- **VLAs will compress all three into one model**
- The alignment techniques learned here (SFT, DPO, LoRA) will be reused for VLA training

### Why This Matters for Robotics

```
LLM (language) ← SFT + RLHF/DPO for alignment
    +
Vision Encoder (perception) ← CLIP-style pretraining
    +
Action Head (motor control) ← Robot demonstration data
    =
VLA (Vision-Language-Action) ← All alignment techniques combined
```

**LoRA for VLAs:** When we get to VLAs (Phase V), we'll see that LoRA is the
primary fine-tuning method because:
- VLAs are huge (7B+ parameters)
- Robot data is scarce (thousands, not millions of examples)
- Different robots need different specializations
- LoRA enables multi-robot, multi-task adaptation

---

## Notes

*These notes are part of the LLM-to-VLA curriculum.*
*Next: [07 — LLM Engineering & Robotics Connection](07-llm-engineering.md)*
