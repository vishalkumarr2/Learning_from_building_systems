# 05 — GPT, Scaling Laws & The Bitter Lesson

> **Phase II · Days 23–30 · ~20 hours**
> Prerequisites: [04-transformer-variants](04-transformer-variants.md)
> Learning Objectives: Understand decoder-only LMs, run nanoGPT ablations, deeply grasp scaling laws, understand T5

---

## Table of Contents

1. [GPT & Decoder-Only Architecture (Day 23)](#1-gpt--decoder-only-architecture-day-23)
2. [nanoGPT Ablation Laboratory (Day 24)](#2-nanogpt-ablation-laboratory-day-24)
3. [Scaling Laws & Emergence (Day 25)](#3-scaling-laws--emergence-day-25)
4. [🛑 STOP AND REFLECT #2 (Day 26)](#4--stop-and-reflect-2-day-26)
5. [Sampling & Generation (Day 27)](#5-sampling--generation-day-27)
6. [T5 & Encoder-Decoder LMs (Day 28)](#6-t5--encoder-decoder-lms-day-28)
7. [Phase II Synthesis (Days 29-30)](#7-phase-ii-synthesis-days-29-30)
8. [Key Equations & Scaling Law Formulas](#8-key-equations--scaling-law-formulas)
9. [Key Takeaways & Paper References](#9-key-takeaways--paper-references)
10. [Connection to Thread](#10-connection-to-thread)

---

## 1. GPT & Decoder-Only Architecture (Day 23)

### 1.1 What is an Autoregressive Language Model?

An **autoregressive LM** models the probability of a sequence by factoring it as a product of conditional distributions:

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})$$

At each step, the model predicts the *next token* given all previous tokens. This is fundamentally different from masked LMs (BERT) which predict *masked* tokens given bidirectional context.

💡 **Key Insight**: Autoregressive generation is *causal* — you can only look backward. This maps naturally to sequential decision-making in robotics: given history of observations, predict the next action.

### 1.2 The Causal Mask

The decoder-only Transformer enforces autoregression through a **causal attention mask**:

```
Causal Mask (lower-triangular):

        t1  t2  t3  t4  t5
  t1  [  1   0   0   0   0 ]
  t2  [  1   1   0   0   0 ]
  t3  [  1   1   1   0   0 ]
  t4  [  1   1   1   1   0 ]
  t5  [  1   1   1   1   1 ]

0 = masked (set to -∞ before softmax)
1 = attend
```

In code (from nanoGPT):
```python
# Register causal mask as buffer
self.register_buffer(
    "bias",
    torch.tril(torch.ones(block_size, block_size))
    .view(1, 1, block_size, block_size)
)

# Apply during attention
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
att = F.softmax(att, dim=-1)
```

⚠️ **Common Confusion**: The causal mask doesn't *remove* positions — it sets their attention weights to zero via $-\infty$ before softmax. The computation still touches all positions; the gradient just doesn't flow through masked ones.

### 1.3 GPT-1 → GPT-2 → GPT-3 Progression

```
┌─────────────┬───────────┬────────────┬───────────────┬──────────────────┐
│   Model     │ Parameters│   Layers   │   d_model     │  Key Innovation  │
├─────────────┼───────────┼────────────┼───────────────┼──────────────────┤
│ GPT-1       │   117M    │    12      │     768       │ Pretrain+finetune│
│ GPT-2       │   1.5B    │    48      │    1600       │ Zero-shot via    │
│             │           │            │               │ prompt formatting│
│ GPT-3       │   175B    │    96      │   12288       │ Few-shot ICL,    │
│             │           │            │               │ no fine-tuning   │
└─────────────┴───────────┴────────────┴───────────────┴──────────────────┘
```

**GPT-1 (2018)** — "Improving Language Understanding by Generative Pre-Training"
- Showed that *generative* pre-training (not just discriminative) transfers well
- Two-stage: unsupervised pre-training → supervised fine-tuning
- Architecture: 12-layer decoder-only Transformer
- Training data: BooksCorpus (~7K unpublished books)

**GPT-2 (2019)** — "Language Models are Unsupervised Multitask Learners"
- Key insight: a sufficiently large LM can perform tasks *without* fine-tuning
- Task specification via natural language prompt (zero-shot)
- Training data: WebText (8M web pages, ~40GB)
- Moved layer norm before attention (Pre-LN)

**GPT-3 (2020)** — "Language Models are Few-Shot Learners"
- Demonstrated **in-context learning (ICL)**: provide examples in the prompt
- No gradient updates needed — the model "learns" from the prompt alone
- 175B parameters, trained on 300B tokens
- Showed emergent abilities at scale

### 1.4 Why Decoder-Only Won

The ML community debated three architectures:

```
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Encoder-Only    │  │  Encoder-Decoder  │  │  Decoder-Only    │
│    (BERT)        │  │    (T5, BART)     │  │  (GPT, LLaMA)   │
│                  │  │                  │  │                  │
│  Bidirectional   │  │  Bidir → Causal  │  │  Causal only     │
│  No generation   │  │  Seq2Seq tasks   │  │  Universal       │
│  Task-specific   │  │  Fixed input/out │  │  Scales better   │
│  heads needed    │  │  split           │  │  Simpler         │
└──────────────────┘  └──────────────────┘  └──────────────────┘
         ↓                     ↓                     ↓
   Classification         Translation           Everything
   NER, QA              Summarization           (with scale)
```

**Why decoder-only dominates today:**

1. **Simplicity**: One architecture for everything — classification, generation, reasoning, code
2. **Scaling efficiency**: All parameters contribute to the same objective (next-token prediction)
3. **Emergent versatility**: At sufficient scale, decoder-only models do everything encoder-decoder models can, plus more
4. **Training efficiency**: Causal LM objective uses *every* token as both context and target (no masking waste)
5. **KV-cache friendly**: Autoregressive generation with KV-cache is straightforward

💡 **The Simplicity Argument**: Decoder-only is the simplest architecture that captures the full generative distribution. By the bitter lesson (Sutskever), the simplest approach that scales wins.

### 1.5 nanoGPT Architecture Walkthrough

Karpathy's nanoGPT distills GPT-2 into ~300 lines of PyTorch. The core architecture:

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),  # position embeddings
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: output projection shares weights with token embedding
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, device=idx.device)

        tok_emb = self.transformer.wte(idx)      # (B, T, C)
        pos_emb = self.transformer.wpe(pos)       # (T, C)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)              # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # Inference: only compute logits for last position
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            loss = None

        return logits, loss
```

**Key design decisions in nanoGPT:**

| Decision | Choice | Why |
|----------|--------|-----|
| Position encoding | Learned absolute | Simple, GPT-2 style |
| Normalization | Pre-LN (before attn) | More stable training |
| Activation | GELU | GPT-2 default |
| Weight tying | wte = lm_head | Reduces params, improves perf |
| Bias | Optional (config) | Modern trend: no bias |
| Dropout | In attn + residual | Regularization |

```
nanoGPT Block:
┌───────────────────────────────┐
│         Input x               │
├───────────────────────────────┤
│    LayerNorm(x)               │  ← Pre-LN
│    CausalSelfAttention(...)   │
│    + x (residual)             │
├───────────────────────────────┤
│    LayerNorm(x)               │  ← Pre-LN
│    MLP(...)                   │
│    + x (residual)             │
├───────────────────────────────┤
│         Output                │
└───────────────────────────────┘

MLP:
  Linear(C → 4C) → GELU → Linear(4C → C) → Dropout
```

### 1.6 Weight Tying Deep Dive

**Weight tying** shares the token embedding matrix $W_E \in \mathbb{R}^{V \times d}$ with the output projection:

$$P(x_t = v \mid h_t) = \text{softmax}(h_t \cdot W_E^T)$$

where $h_t \in \mathbb{R}^d$ is the hidden state and $W_E$ maps vocabulary → embedding space.

**Why it works:**
- The embedding maps tokens to a semantic space
- The output projection maps hidden states *back* to that same space
- Forcing them to share weights means "understanding a token" (embedding) and "predicting a token" (output) use the same representation
- Reduces parameters by $V \times d$ (significant for large vocabularies)

⚠️ **Subtle point**: Weight tying means the embedding dimension *must* equal the model dimension. In architectures where they differ, you need a projection layer.

---

## 2. nanoGPT Ablation Laboratory (Day 24)

### 2.1 What is an Ablation Study?

An **ablation study** systematically removes or varies one component at a time to measure its contribution. Named after surgical ablation — removing tissue to understand function.

```
Scientific Method for Neural Networks:
1. Start with a baseline configuration
2. Change EXACTLY ONE variable
3. Train with IDENTICAL everything else (seed, data, steps)
4. Measure the outcome (loss, perplexity)
5. Attribute the difference to the changed variable
```

💡 **Golden Rule of Ablations**: Change one thing at a time. If you change two things and performance improves, you don't know which one helped (or if they interacted).

### 2.2 Baseline Configuration

For all ablation experiments, use this nanoGPT baseline:

```python
# Baseline config for ablations
baseline = {
    'n_layer': 6,
    'n_head': 6,
    'n_embd': 384,
    'block_size': 256,
    'batch_size': 64,
    'learning_rate': 6e-4,
    'max_iters': 5000,
    'lr_decay_iters': 5000,
    'warmup_iters': 100,
    'dropout': 0.2,
    'dataset': 'shakespeare_char',
}
# Total params: ~10.7M
# Training time: ~5 min on single GPU
```

### 2.3 Ablation Experiments

#### Experiment 1: Depth (n_layers)

```
┌──────────┬────────┬─────────────┬──────────┐
│ n_layers │ Params │  Val Loss   │  Notes   │
├──────────┼────────┼─────────────┼──────────┤
│     2    │  3.6M  │   ~1.65     │ Underfit │
│     4    │  7.2M  │   ~1.52     │          │
│     6    │ 10.7M  │   ~1.47     │ Baseline │
│     8    │ 14.3M  │   ~1.44     │          │
│    12    │ 21.4M  │   ~1.42     │ Diminish │
└──────────┴────────┴─────────────┴──────────┘
```

**What to observe**: Loss improves sublinearly with depth. Deeper networks learn more abstract representations in later layers but with diminishing returns per layer.

#### Experiment 2: Attention Heads

```
┌─────────┬──────────────────┬─────────────┐
│ n_heads │  Head dim (d/h)  │  Val Loss   │
├─────────┼──────────────────┼─────────────┤
│    1    │      384         │   ~1.55     │
│    2    │      192         │   ~1.50     │
│    4    │       96         │   ~1.48     │
│    6    │       64         │   ~1.47     │
│    8    │       48         │   ~1.47     │
└─────────┴──────────────────┴─────────────┘
```

**What to observe**: Multiple heads help significantly over single-head, but returns diminish. Each head learns to attend to different linguistic patterns (positional, syntactic, semantic).

#### Experiment 3: Model Width (d_model)

```
┌─────────┬────────┬─────────────┐
│ d_model │ Params │  Val Loss   │
├─────────┼────────┼─────────────┤
│    64   │  0.3M  │   ~2.10     │
│   128   │  1.0M  │   ~1.75     │
│   256   │  3.5M  │   ~1.55     │
│   384   │ 10.7M  │   ~1.47     │
│   512   │ 18.8M  │   ~1.43     │
└─────────┴────────┴─────────────┘
```

**What to observe**: Width has a strong effect on capacity. The relationship between params and loss follows an approximate power law — your first empirical scaling curve!

#### Experiment 4: Activation Functions

```python
# Swap activation in MLP
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        # Choose activation
        if config.activation == 'relu':
            self.act = nn.ReLU()
        elif config.activation == 'gelu':
            self.act = nn.GELU()
        elif config.activation == 'swiglu':
            # SwiGLU: needs different architecture
            self.gate = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.act = lambda x: F.silu(self.gate(x)) * x
```

```
┌────────────┬─────────────┬────────────────────────┐
│ Activation │  Val Loss   │  Notes                 │
├────────────┼─────────────┼────────────────────────┤
│ ReLU       │   ~1.50     │ Dead neurons possible  │
│ GELU       │   ~1.47     │ Baseline, smooth       │
│ SwiGLU     │   ~1.45     │ Best, but more params  │
└────────────┴─────────────┴────────────────────────┘
```

#### Experiment 5: Normalization

```
┌──────────────┬─────────────┬──────────────────────────┐
│ Norm Type    │  Val Loss   │  Notes                   │
├──────────────┼─────────────┼──────────────────────────┤
│ Post-LN      │   ~1.52     │ Training less stable     │
│ Pre-LN       │   ~1.47     │ Baseline, stable         │
│ RMSNorm      │   ~1.46     │ Slightly better, cheaper │
└──────────────┴─────────────┴──────────────────────────┘
```

### 2.4 How to Plot Scaling Curves from Ablations

```python
import matplotlib.pyplot as plt
import numpy as np

# Collect results: (params, best_val_loss)
results = [
    (0.3e6, 2.10),   # d_model=64
    (1.0e6, 1.75),   # d_model=128
    (3.5e6, 1.55),   # d_model=256
    (10.7e6, 1.47),  # d_model=384
    (18.8e6, 1.43),  # d_model=512
]

params, losses = zip(*results)
params, losses = np.array(params), np.array(losses)

# Fit power law: L = a * N^(-alpha)
# In log space: log(L) = log(a) - alpha * log(N)
log_params = np.log(params)
log_losses = np.log(losses)
alpha, log_a = np.polyfit(log_params, log_losses, 1)
a = np.exp(log_a)

print(f"Scaling exponent α = {-alpha:.4f}")
print(f"L(N) ≈ {a:.4f} · N^({alpha:.4f})")

# Plot
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.scatter(params, losses, s=80, zorder=5)
N_fit = np.logspace(np.log10(params.min()), np.log10(params.max()), 100)
ax.plot(N_fit, a * N_fit**alpha, 'r--', label=f'$L = {a:.2f} \\cdot N^{{{alpha:.3f}}}$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Parameters (N)')
ax.set_ylabel('Validation Loss')
ax.set_title('Your First Scaling Curve')
ax.legend()
plt.tight_layout()
```

💡 **What you'll find**: The exponent $\alpha$ from your nanoGPT ablations will be roughly $\alpha \approx 0.07$–$0.10$, which is close to the published Kaplan value of $\alpha_N \approx 0.076$ for parameter scaling. You're reproducing a key scientific result!

---

## 3. Scaling Laws & Emergence (Day 25)

### 3.1 Kaplan Scaling Laws (2020)

The landmark paper "Scaling Laws for Neural Language Models" (Kaplan et al., 2020) discovered that LM loss follows **smooth power laws** in three quantities:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095$$

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050$$

Where:
- $N$ = number of parameters (non-embedding)
- $D$ = number of training tokens (dataset size)
- $C$ = compute budget in FLOPs
- $N_c, D_c, C_c$ = constants
- $\alpha$ = scaling exponents

```
Log-Log Scaling Curves (schematic):

  Loss
  (log)
    │
  2.0│\
    │  \
  1.5│   \.
    │     \..
  1.0│       \....
    │           \........
  0.5│                   \..........
    │
    └───────────────────────────────── Compute (log)
     10⁸     10¹²    10¹⁶    10²⁰

  Key observation: STRAIGHT LINE on log-log plot
  = power law relationship
```

### 3.2 Key Findings from Kaplan

1. **Performance depends strongly on scale, weakly on shape**: A 1B parameter model with depth 24 performs similarly to 1B with depth 48, but very differently from a 100M model
2. **Larger models are more sample-efficient**: Big models reach a given loss with *fewer* tokens than small models
3. **Optimal compute allocation**: For a fixed compute budget, it's better to train a *larger* model for *fewer* steps than a smaller model for more steps

```
Kaplan's Compute-Optimal Allocation:
  If compute grows 10×:
    → Parameters should grow ~5.5×
    → Training tokens should grow ~1.8×
  
  "Spend most of your compute budget on a bigger model"
```

⚠️ **This was WRONG** — Chinchilla corrected this allocation. But the power law observation was solid.

### 3.3 Chinchilla Optimal Ratio (2022)

Hoffmann et al. ("Training Compute-Optimal Large Language Models") showed Kaplan's allocation was suboptimal:

$$N_{\text{opt}} \propto C^{0.50}, \quad D_{\text{opt}} \propto C^{0.50}$$

**Chinchilla Rule of Thumb**: Parameters and tokens should scale *equally* with compute. For every doubling of model size, also double the training data.

```
┌───────────────────┬─────────────────────┬───────────────────┐
│                   │   Kaplan (2020)     │  Chinchilla (2022)│
├───────────────────┼─────────────────────┼───────────────────┤
│ If C grows 10×    │ N grows 5.5×        │ N grows ~3.2×     │
│                   │ D grows 1.8×        │ D grows ~3.2×     │
├───────────────────┼─────────────────────┼───────────────────┤
│ Optimal tokens    │ ~few B per model    │ ~20× parameters   │
│ for 70B model     │                     │ = 1.4T tokens     │
├───────────────────┼─────────────────────┼───────────────────┤
│ Practical impact  │ Undertrained models │ Properly trained  │
│                   │ (GPT-3: 300B tok)   │ (Chinchilla: 1.4T)│
└───────────────────┴─────────────────────┴───────────────────┘
```

**Chinchilla (70B) matched GPT-3 (175B) using 4× less compute** — simply by training on the right amount of data.

💡 **Practical implication**: Most LLMs before Chinchilla were *undertrained*. This shifted the field toward "smaller but well-trained" models (LLaMA, Mistral).

### 3.4 The Compression Interpretation (Sutskever)

Ilya Sutskever's profound insight: **next-token prediction is compression, and compression is understanding**.

Consider: to predict the next token well, the model must:
- Understand grammar (syntax compression)
- Understand semantics (meaning compression)
- Understand world knowledge (factual compression)
- Understand reasoning patterns (logical compression)

$$\text{Cross-entropy loss} = H(p, q) = -\sum_{x} p(x) \log q(x)$$

This is *literally* the number of bits needed to encode the data using the model's distribution $q$ instead of the true distribution $p$. **Better prediction = better compression = more understanding.**

```
Compression Hierarchy:

  Loss 2.5  → Memorized character frequencies
  Loss 2.0  → Learned common words
  Loss 1.5  → Learned grammar & syntax
  Loss 1.2  → Learned semantics & context
  Loss 1.0  → Learned world knowledge
  Loss 0.8  → Learned reasoning patterns
  Loss 0.5  → Approaching human-level prediction

  Each level requires exponentially more compute
  but represents genuinely deeper understanding.
```

⚠️ **The controversial part**: Does compression *equal* understanding, or just *correlate* with it? Chinese Room argument applies. But empirically, models that compress better also perform better on downstream tasks — the correlation is remarkably strong.

### 3.5 Emergence

**Emergent abilities** are capabilities that appear suddenly at scale — absent in smaller models, present in larger ones:

```
Performance vs Scale (schematic):

  Task
  Accuracy
    │
  100%│                    ╭────────
    │                   ╱
   50%│                  │   ← "Phase transition"
    │                  │
    0%│──────────────────╯
    └──────────────────────────── Scale (log)
                        ↑
              "Emergence threshold"
```

**Examples of emergent abilities:**
- **In-context learning**: GPT-3 (175B) can learn from examples in the prompt; GPT-2 (1.5B) cannot
- **Chain-of-thought reasoning**: Only works above ~100B parameters
- **Code generation**: Requires scale + code in training data
- **Multilingual transfer**: Model trained on English answers in French

💡 **The Emergence Debate**: Schaeffer et al. (2023) argued that emergence is a "mirage" caused by nonlinear metrics. When you use linear metrics, the improvement is smooth, not sudden. The debate continues — but the practical fact remains: some abilities only appear at sufficient scale.

### 3.6 In-Context Learning as Implicit Bayesian Inference

A fascinating theoretical perspective: ICL can be viewed as **implicit Bayesian inference**.

Given a prompt with examples $(x_1, y_1), (x_2, y_2), \ldots, (x_k, y_k)$, the model's prediction for a new $x$ is:

$$P(y \mid x, \text{examples}) \approx \int P(y \mid x, f) P(f \mid \text{examples}) df$$

Where $f$ ranges over functions the model has learned during pre-training.

**Intuition**: The model has internalized many input-output mappings during pre-training. The few-shot examples in the prompt act as evidence to *select* the right mapping — not to *learn* a new one.

```
Pre-training learns:                  ICL selects:
┌─────────────────────┐              ┌─────────────┐
│ f₁: sentiment       │              │ examples → │
│ f₂: translation     │ ──────────→  │ "this looks │
│ f₃: summarization   │              │  like f₁"   │
│ f₄: QA              │              │ → apply f₁  │
│ ...                  │              └─────────────┘
│ fₙ: arbitrary       │
└─────────────────────┘
```

This explains why ICL performance improves with:
- More examples (more evidence for Bayesian update)
- Better examples (higher-quality evidence)
- Larger models (more functions internalized)

---

## 4. 🛑 STOP AND REFLECT #2 (Day 26)

> **"The same architecture, more compute, keeps getting better. What does this tell us about learning?"**

Take a full day to sit with these questions. Write at least 500 words total.

### Reflection Prompt 1: Scaling = Understanding?

If compression = intelligence (Sutskever's thesis), then:
- Scaling compute = scaling intelligence
- A bigger model that predicts the next word better *genuinely understands more*
- There's no "trick" — just more computation applied to the same objective

**Write your response**: Do you believe this? What would falsify it? Is there a ceiling? If a model achieves perfect next-token prediction on internet text, does it "understand everything"?

### Reflection Prompt 2: The Bitter Lesson

Rich Sutton's "Bitter Lesson" (2019):
> "The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective."

Every time researchers tried to hand-engineer knowledge (expert systems, grammar rules, feature engineering), the approach that just *scaled compute with general methods* eventually won.

- Chess: Deep Blue (hand-crafted) → AlphaZero (general learning + compute)
- Computer vision: SIFT+SVM (hand-crafted) → CNNs (general learning + compute)
- NLP: Parse trees → Word2Vec → Transformers (increasingly general, more compute)

**Write your response**: If the bitter lesson is true, what does it mean for robotics? Should we stop engineering robot controllers and just scale up general models with robot data?

### Reflection Prompt 3: Scaling Laws for Robot Actions

The scaling laws we studied are for *language*. Do they transfer to robotics?

Consider:
- RT-2 showed that VLMs can output robot actions — is this "scaling" for robotics?
- OpenVLA trains on the Open X-Embodiment dataset — more data, better performance?
- But robots have continuous actions, not discrete tokens — does the power law still hold?
- The action space is constrained by physics — does this change the scaling dynamics?

**Write your response**: Predict what scaling laws for robot foundation models will look like. Will the exponents be the same? Larger? Will there be a different bottleneck (data? embodiment diversity? sim-to-real gap)?

### Reflection Prompt 4: Connect Your Learning

Trace the thread from Study Notes 01 through 05:

```
Backpropagation (01) → enables training deep networks
    ↓
Word2Vec (02) → learned representations > hand-crafted features
    ↓
Seq2Seq + Attention (03) → attention mechanism discovers alignment
    ↓
Transformers (04) → parallel attention replaces recurrence
    ↓
GPT + Scaling (05) → same architecture + more compute = more capability
    ↓
??? (future) → same architecture + robot data = robot intelligence?
```

**Write your response**: What's the *minimal set of ideas* that got us from perceptrons to GPT-3? Is it just (a) backprop, (b) attention, (c) scale?

---

## 5. Sampling & Generation (Day 27)

### 5.1 The Generation Problem

Given a trained autoregressive model $P(x_t \mid x_{<t})$, how do we generate text?

The model outputs a probability distribution over the vocabulary at each step. We need a **sampling strategy** to select the next token.

### 5.2 Greedy Decoding

Always pick the most likely token:

$$x_t = \arg\max_v P(v \mid x_{<t})$$

```python
def greedy_decode(model, prompt_ids, max_new_tokens):
    idx = prompt_ids
    for _ in range(max_new_tokens):
        logits = model(idx)[:, -1, :]  # (B, vocab_size)
        next_id = logits.argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=-1)
    return idx
```

**Problem**: Greedy decoding is *repetitive* and *boring*. It tends to get stuck in loops because the most probable continuation is often a repetition of recent text.

### 5.3 Temperature Scaling

Scale logits before softmax to control randomness:

$$P(v) = \frac{\exp(z_v / \tau)}{\sum_{v'} \exp(z_{v'} / \tau)}$$

Where $\tau$ is the temperature:

```
Temperature Effects:

  τ → 0:  Approaches greedy (argmax)
           Distribution becomes a spike
           
  τ = 1:  Original distribution
           Model's natural confidence
           
  τ → ∞:  Approaches uniform random
           All tokens equally likely

  ┌─────────────────────────┐
  │  P(v)                   │
  │  ▲                      │
  │  █ τ=0.1 (sharp)        │
  │  █░                     │
  │  █░░ τ=0.7              │
  │  █░░░░                  │
  │  █░░░░░░░ τ=1.0         │
  │  ▓▓▓▓▓▓▓▓▓ τ=2.0 (flat)│
  │──────────────────────▶ v│
  └─────────────────────────┘
```

```python
def sample_with_temperature(logits, temperature=1.0):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### 5.4 Top-k Sampling

Only consider the $k$ most probable tokens:

```python
def top_k_sample(logits, k=50, temperature=1.0):
    logits = logits / temperature
    # Zero out everything below top-k
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
    # Create mask
    logits[logits < top_k_logits[..., -1:]] = float('-inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

**Problem**: Fixed $k$ doesn't adapt to the model's confidence. Sometimes the model is very confident (top 3 tokens have 95% mass) — $k=50$ wastes probability on garbage. Sometimes it's uncertain — $k=50$ might cut off valid options.

### 5.5 Top-p (Nucleus) Sampling

Dynamically select the smallest set of tokens whose cumulative probability exceeds $p$:

$$V_p = \min \{V' \subseteq V : \sum_{v \in V'} P(v) \geq p\}$$

```python
def top_p_sample(logits, p=0.9, temperature=1.0):
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Keep the first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = sorted_indices_to_remove.scatter(
        -1, sorted_indices, sorted_indices_to_remove
    )
    logits[indices_to_remove] = float('-inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

💡 **Why top-p is usually better than top-k**: It adapts to the model's confidence. When the model is certain, it considers few tokens (like greedy). When uncertain, it considers many (like random). This produces more natural text.

### 5.6 Repetition Penalty

Reduce the probability of tokens that have already appeared:

$$z_v' = \begin{cases} z_v / \theta & \text{if } v \in \text{generated tokens and } z_v > 0 \\ z_v \cdot \theta & \text{if } v \in \text{generated tokens and } z_v < 0 \end{cases}$$

Where $\theta > 1$ is the repetition penalty factor.

```python
def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    for token_id in set(generated_ids.tolist()):
        if logits[token_id] > 0:
            logits[token_id] /= penalty
        else:
            logits[token_id] *= penalty
    return logits
```

### 5.7 Constrained Decoding

Force the model to output valid structured text (e.g., JSON):

```python
# Simple JSON-constrained decoding
class JSONConstrainer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.state = 'START'  # FSM state

    def get_allowed_tokens(self):
        """Return set of allowed token IDs given current state."""
        if self.state == 'START':
            return self._tokens_starting_with('{')
        elif self.state == 'KEY':
            return self._tokens_starting_with('"')
        elif self.state == 'COLON':
            return self._tokens_starting_with(':')
        # ... etc

    def mask_logits(self, logits):
        allowed = self.get_allowed_tokens()
        mask = torch.full_like(logits, float('-inf'))
        mask[allowed] = 0
        return logits + mask
```

💡 **For robotics**: Constrained decoding is how VLAs output valid action tokens. The action space defines constraints on what tokens are valid at each step. This connects sampling strategies directly to robot control.

### 5.8 Sampling Strategy Comparison

```
┌──────────────┬─────────────────────┬─────────────────────┐
│   Strategy   │     Best For        │    Parameters       │
├──────────────┼─────────────────────┼─────────────────────┤
│ Greedy       │ Factual answers     │ None                │
│ Beam Search  │ Translation, summ.  │ beam_width          │
│ Temperature  │ Creative writing    │ τ ∈ (0, 2]          │
│ Top-k        │ General generation  │ k ∈ [10, 100]       │
│ Top-p        │ Best default        │ p ∈ [0.9, 0.95]     │
│ Top-p + temp │ Fine control        │ p=0.9, τ=0.8        │
│ Constrained  │ Structured output   │ Grammar/FSM         │
└──────────────┴─────────────────────┴─────────────────────┘
```

---

## 6. T5 & Encoder-Decoder LMs (Day 28)

### 6.1 The Text-to-Text Framework

T5 ("Text-to-Text Transfer Transformer", Raffel et al., 2020) unified all NLP tasks into a single format:

```
Every task is: text → text

Sentiment:    "sst2 sentence: This movie is great"  → "positive"
Translation:  "translate English to French: Hello"   → "Bonjour"
Summarization: "summarize: <long article>"           → "<summary>"
QA:           "question: What is 2+2? context: ..."  → "4"
```

💡 **The key insight**: By casting everything as text-to-text, you can use the *same* model, *same* loss function, and *same* training pipeline for every task. The task is specified by a text prefix.

### 6.2 T5 Architecture

T5 uses the **original encoder-decoder** Transformer architecture (not decoder-only):

```
Input: "translate English to German: The house is wonderful"

┌──────────────────────────────────────────────┐
│                 ENCODER                       │
│  ┌──────┐  ┌──────┐       ┌──────┐          │
│  │ Self │  │ Self │  ...  │ Self │           │
│  │ Attn │  │ Attn │       │ Attn │           │
│  │(bidi)│  │(bidi)│       │(bidi)│           │
│  └──────┘  └──────┘       └──────┘           │
│  Bidirectional attention (no causal mask)     │
└──────────────────────┬───────────────────────┘
                       │ encoder hidden states
                       ▼
┌──────────────────────────────────────────────┐
│                 DECODER                       │
│  ┌──────┐  ┌───────────┐  ┌──────┐          │
│  │Causal│  │  Cross-   │  │ FFN  │  × L     │
│  │ Self │→ │ Attention │→ │      │           │
│  │ Attn │  │(enc→dec)  │  │      │           │
│  └──────┘  └───────────┘  └──────┘           │
│  Causal self-attention + cross-attention      │
└──────────────────────────────────────────────┘

Output: "Das Haus ist wunderbar"
```

**Key differences from GPT:**
- Encoder processes input with *bidirectional* attention (like BERT)
- Decoder uses *causal* self-attention + *cross-attention* to encoder
- Separate input and output sequences
- Relative position encodings instead of absolute

### 6.3 T5 Pre-training: Span Corruption

T5 uses a **span corruption** objective instead of next-token prediction:

```
Original:  "The cute dog walked in the park on a sunny day"
Corrupted: "The <X> walked in <Y> sunny day"
Target:    "<X> cute dog <Y> the park on a <Z>"

Where <X>, <Y>, <Z> are sentinel tokens marking corrupted spans
```

- Randomly mask 15% of tokens as contiguous spans
- Model must predict the missing spans
- This is more challenging than BERT's single-token masking

### 6.4 When Encoder-Decoder Beats Decoder-Only

```
┌────────────────────┬─────────────────────────────────────┐
│  Use Enc-Dec When  │  Use Decoder-Only When              │
├────────────────────┼─────────────────────────────────────┤
│ Fixed input/output │ Open-ended generation               │
│ Translation        │ Dialogue, chat                      │
│ Summarization      │ Code generation                     │
│ Structured extract │ Reasoning chains                    │
│ Input >> Output    │ General-purpose                     │
│ Compute-efficient  │ Scale is the priority               │
│ for specific tasks │ Versatility matters most            │
└────────────────────┴─────────────────────────────────────┘
```

**The efficiency argument**: For seq2seq tasks, encoder-decoder is more compute-efficient because the encoder processes the input *once* with bidirectional attention (better representation), and the decoder only needs to generate the output. In a decoder-only model, the entire input is reprocessed at every generation step (mitigated by KV-cache, but still less efficient).

### 6.5 Architecture Comparison Table

```
┌───────────────────┬──────────────┬────────────────┬──────────────┐
│  Property         │  BERT        │  GPT           │  T5          │
│                   │ (Enc-only)   │ (Dec-only)     │ (Enc-Dec)    │
├───────────────────┼──────────────┼────────────────┼──────────────┤
│ Attention         │ Bidirectional│ Causal (left)  │ Bidir + Caus │
│ Pre-training      │ MLM + NSP    │ Next-token     │ Span corrupt │
│ Generation        │ ✗            │ ✓ (native)     │ ✓ (decoder)  │
│ Classification    │ ✓ (CLS)      │ ✓ (last tok)   │ ✓ (text→text)│
│ Seq2Seq           │ ✗            │ ✓ (w/ prompt)  │ ✓ (native)   │
│ Param efficiency  │ Medium       │ Low (at small) │ High         │
│ At scale          │ Abandoned    │ Dominant       │ Niche        │
│ Modern examples   │ DeBERTa      │ GPT-4, LLaMA   │ Flan-T5, UL2│
│ Best use case     │ Embed/class  │ Everything     │ Seq2seq tasks│
└───────────────────┴──────────────┴────────────────┴──────────────┘
```

### 6.6 T5's Systematic Study

The T5 paper is valuable not just for the model but for its **systematic comparison** of design choices:

1. **Architectures**: Encoder-decoder, decoder-only, prefix-LM
2. **Pre-training objectives**: LM, prefix LM, MLM, span corruption, deshuffling
3. **Training strategies**: Fine-tune all, adapters, gradual unfreezing
4. **Scale**: 60M to 11B parameters
5. **Data**: C4 (Colossal Clean Crawled Corpus)

**Conclusion**: Encoder-decoder with span corruption worked best at the time, but this was before the GPT-3 scaling breakthrough. At sufficient scale, decoder-only with next-token prediction catches up and surpasses.

---

## 7. Phase II Synthesis (Days 29-30)

### 7.1 Capstone Project: Mini-LM Ablations

See [Project 02: Mini-LM Ablations](../projects/02-mini-lm-ablations/README.md)

**Goal**: Train a mini-LM on Shakespeare, systematically ablate architectural choices, and produce a scaling curve.

### 7.2 Phase II Checkpoint Questions

Answer these before moving to Phase III:

**Architecture Understanding:**
1. What is the difference between causal and bidirectional attention, and when do you use each?
2. Draw the full forward pass of a GPT block from input IDs to output logits.
3. Why does weight tying work? What constraint does it impose?
4. In nanoGPT, what happens if you remove the positional encoding?

**Scaling:**
5. State Kaplan's scaling laws in mathematical form.
6. How did Chinchilla correct Kaplan's compute allocation?
7. Explain the compression interpretation of language modeling.
8. What is the bitter lesson and why is it relevant to LLMs?

**Generation:**
9. Compare top-k and top-p sampling. When is each appropriate?
10. How does constrained decoding work and why is it relevant for VLAs?

**Architectures:**
11. When would you choose T5 over GPT?
12. What is span corruption and how does it differ from MLM?

**Integration:**
13. If you had to build a robot foundation model, would you use encoder-decoder or decoder-only? Justify.
14. Do you expect scaling laws to hold for robot action prediction? What might be different?

### 7.3 Phase II Summary

```
Phase II: Language Models — What You Learned
═══════════════════════════════════════════════

Day 15-22 (Study Note 04):
  ✓ Transformer architecture from scratch
  ✓ Multi-head attention, positional encodings
  ✓ BERT, ViT, pre-training objectives
  ✓ Implementation in PyTorch

Day 23-30 (Study Note 05 — this file):
  ✓ GPT and decoder-only architecture
  ✓ nanoGPT code walkthrough and ablations
  ✓ Scaling laws (Kaplan, Chinchilla)
  ✓ Compression = understanding thesis
  ✓ Sampling strategies for generation
  ✓ T5 and encoder-decoder comparison

Key Thread: Scaling laws prove the bitter lesson —
general methods + compute > specialized engineering.
This same principle will drive robot foundation models.
```

---

## 8. Key Equations & Scaling Law Formulas

### Autoregressive Language Modeling

$$P(x_1, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})$$

### Cross-Entropy Loss

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})$$

### Perplexity

$$\text{PPL} = \exp(\mathcal{L}) = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})\right)$$

Perplexity = "how many tokens is the model choosing between on average." PPL of 10 means the model is as uncertain as choosing uniformly from 10 tokens.

### Kaplan Scaling Laws

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095$$

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050$$

### Chinchilla Optimal Allocation

$$N_{\text{opt}} \approx 0.0556 \cdot C^{0.50}$$

$$D_{\text{opt}} \approx 0.2948 \cdot C^{0.50}$$

Rule of thumb: $D_{\text{opt}} \approx 20 \times N$

### Compute Estimation

$$C \approx 6 N D \quad \text{(FLOPs for a training run)}$$

Where $N$ = parameters, $D$ = training tokens, and the factor 6 accounts for forward + backward pass.

### Temperature-Scaled Softmax

$$P(v \mid z, \tau) = \frac{\exp(z_v / \tau)}{\sum_{v'} \exp(z_{v'} / \tau)}$$

### Top-p (Nucleus) Sampling

$$V_p = \arg\min_{V' \subseteq V} |V'| \quad \text{s.t.} \quad \sum_{v \in V'} P(v) \geq p$$

### Shannon Entropy (Information Content)

$$H(X) = -\sum_{x} P(x) \log_2 P(x) \quad \text{bits}$$

The theoretical minimum bits to encode samples from distribution $P$. A language model with loss $\mathcal{L}$ uses $\mathcal{L} / \ln 2$ bits per token.

---

## 9. Key Takeaways & Paper References

### Key Takeaways

1. **Decoder-only architectures won** because they're simpler, scale better, and at sufficient scale do everything encoder-decoder models can do
2. **nanoGPT ablations teach you to think empirically** — change one variable, measure the effect, build intuition through data
3. **Scaling laws are the most important discovery in modern AI** — they tell you that performance is predictable and that more compute always helps
4. **Compression = understanding** is a deep thesis — a model that predicts text better genuinely understands more
5. **The bitter lesson applies to robotics** — hand-engineering robot behaviors will likely lose to scaling general models with robot data
6. **Sampling strategies matter** — the same model produces wildly different outputs depending on temperature, top-k/p, and constraints
7. **T5 showed the power of text-to-text framing** — casting everything as the same format simplifies the entire pipeline (VLAs do this for robot actions)

### Paper References

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| Radford et al., "Improving Language Understanding by Generative Pre-Training" (GPT-1) | 2018 | Generative pre-training transfers |
| Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2) | 2019 | Zero-shot task performance |
| Brown et al., "Language Models are Few-Shot Learners" (GPT-3) | 2020 | In-context learning, scaling |
| Kaplan et al., "Scaling Laws for Neural Language Models" | 2020 | Power law scaling in N, D, C |
| Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla) | 2022 | Optimal N/D allocation |
| Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5) | 2020 | Text-to-text framework, systematic study |
| Sutton, "The Bitter Lesson" | 2019 | General methods + compute > engineering |
| Holtzman et al., "The Curious Case of Neural Text Degeneration" | 2020 | Nucleus (top-p) sampling |
| Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" | 2016 | BPE tokenization |
| Schaeffer et al., "Are Emergent Abilities of Large Language Models a Mirage?" | 2023 | Emergence debate |
| Xie et al., "An Explanation of In-context Learning as Implicit Bayesian Inference" | 2022 | ICL theory |

### Recommended Code

| Resource | Link | Purpose |
|----------|------|---------|
| nanoGPT (Karpathy) | github.com/karpathy/nanoGPT | GPT-2 training from scratch |
| minbpe (Karpathy) | github.com/karpathy/minbpe | BPE tokenizer from scratch |
| nanoGPT lecture | youtube.com/@AndrejKarpathy | 2-hour code walkthrough |
| Scaling Laws notebook | github.com/... | Reproducing Kaplan results |

---

## 10. Connection to Thread

### The Scaling Laws → VLA Pipeline

```
Study Note 01: Backprop enables learning from data
Study Note 02: Learned representations > hand-crafted features
Study Note 03: Attention discovers what's relevant in a sequence
Study Note 04: Transformers parallelize attention → enables scale
Study Note 05: Scaling laws guarantee improvement with compute ← YOU ARE HERE
    ↓
Study Note 06: RLHF/alignment make models follow instructions
Study Note 07: Vision encoders extend LMs to images
Study Note 08: VLAs extend VLMs to robot actions
```

### Scaling Laws ARE Compression Efficiency Curves

The central insight connecting this note to the VLA thread:

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha}$$

This power law says: **as you invest more compute, the model compresses the data better**. In language, this means understanding grammar → semantics → world knowledge → reasoning.

For robotics (VLAs):
- The "data" is observation-action sequences
- Better compression = better understanding of how actions relate to observations
- Scaling compute *should* yield smoother, more capable robot policies
- The open question is whether the exponent $\alpha$ is as favorable for actions as for language

### The Bitter Lesson for Robotics

If the bitter lesson holds for robotics:
- Classical control theory → will be replaced by learned control
- Hand-designed motion planners → will be replaced by general models
- Task-specific reward engineering → will be replaced by scaling general pre-training
- This is *exactly* the thesis behind VLAs

⚠️ **But**: Robotics has harder constraints than language:
- Safety: a language model hallucinating is annoying; a robot hallucinating is dangerous
- Physics: language is arbitrary; physics is constrained
- Data: internet text is abundant; robot data is expensive
- Real-time: language generation can be slow; robot control must be fast

**The question isn't whether scaling will win — it's how much domain engineering you need *on top of* scaling.** That's what we'll explore in the VLA notes.

---

*Next: [06-llm-training-alignment.md](06-llm-training-alignment.md) — RLHF, instruction tuning, and making models follow human intent*
