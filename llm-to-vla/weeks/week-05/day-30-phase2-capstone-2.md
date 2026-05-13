# Day 30: Phase II Capstone — Day 2 + Checkpoint

> **Phase II — Attention, Transformers & Scaling | Week 5 | 2.5 hours**
> *"A model is not done when it trains. It's done when you understand why it works."*

## Navigation
- **Previous:** [Day 29: Phase II Capstone Day 1](day-29-phase2-capstone-1.md)
- **Next:** [Day 31: The Modern LLM Recipe](day-31-modern-llm-recipe.md)
- **Week:** [Week 5 Overview](../README.md)
- **Phase:** [Phase II: Attention & Transformers](../../study-notes/05-gpt-scaling.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Part 1: Complete Training & Analysis (75 min)

### 1.1 Training Curve Analysis

Your model should have finished training (or be near completion). Plot the full training history:

```python
import matplotlib.pyplot as plt
import numpy as np


def plot_training_report(metrics: dict, config_name: str = "capstone"):
    """Generate comprehensive training report plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Training loss (smoothed)
    ax = axes[0, 0]
    train_loss = np.array(metrics["train_losses"])
    # Exponential moving average for smoothing
    alpha = 0.01
    smoothed = np.zeros_like(train_loss)
    smoothed[0] = train_loss[0]
    for i in range(1, len(train_loss)):
        smoothed[i] = alpha * train_loss[i] + (1 - alpha) * smoothed[i-1]
    
    ax.plot(train_loss, alpha=0.2, color="blue", label="Raw")
    ax.plot(smoothed, color="blue", linewidth=2, label="Smoothed")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Validation loss
    ax = axes[0, 1]
    val_steps = np.arange(len(metrics["val_losses"])) * metrics.get("eval_interval", 250)
    ax.plot(val_steps, metrics["val_losses"], 'o-', color="orange", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Val Loss")
    ax.set_title(f"Validation Loss (best: {min(metrics['val_losses']):.4f})")
    ax.grid(True, alpha=0.3)
    
    # 3. Learning rate schedule
    ax = axes[1, 0]
    ax.plot(metrics["lrs"], color="green", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("LR Schedule (Cosine + Warmup)")
    ax.grid(True, alpha=0.3)
    
    # 4. Gradient norms
    ax = axes[1, 1]
    grad_norms = np.array(metrics["grad_norms"])
    ax.plot(grad_norms, alpha=0.3, color="red")
    # Rolling mean
    window = 100
    if len(grad_norms) > window:
        rolling = np.convolve(grad_norms, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(grad_norms)), rolling, 
                color="red", linewidth=2, label=f"Rolling {window}")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Norms (clipped)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Training Report: {config_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"training_report_{config_name}.png", dpi=150)
    plt.show()
```

### 1.2 Generation Quality Assessment

Generate samples at different training checkpoints and temperatures:

```python
def assess_generation_quality(model, decode_fn, device):
    """Generate samples with various settings to assess model quality."""
    import torch
    
    model.eval()
    
    prompts = [
        "The ",
        "To be or ",
        "In the beginning ",
    ]
    
    temperatures = [0.3, 0.7, 1.0, 1.5]
    
    for prompt_text in prompts:
        print(f"\n{'='*60}")
        print(f"Prompt: '{prompt_text}'")
        print('='*60)
        
        for temp in temperatures:
            # Encode prompt (character-level example)
            context = torch.tensor(
                [ord(c) for c in prompt_text],  # adjust for your tokenizer
                dtype=torch.long, device=device
            ).unsqueeze(0)
            
            with torch.no_grad():
                generated = model.generate(
                    context, max_new_tokens=150, temperature=temp
                )
            
            text = decode_fn(generated[0].tolist())
            print(f"\n  T={temp}: {text[:200]}")
```

---

## Part 2: Ablation Report (45 min)

### 2.1 Reuse Day 24 Results

If you ran ablations on Day 24, compile your results here. If not, run a subset now (pick the 2 most interesting ablation axes).

### Ablation Report Template

```markdown
## Ablation Report: Mini-LM Capstone

### Baseline Configuration
- n_layers: 8
- n_heads: 8  
- d_model: 512
- d_ff: 2048
- Parameters: ~10M
- Vocab: [char/BPE] with size [N]
- Training: 10K steps, batch 64, cosine LR 3e-4→3e-5

### Results Summary

| Ablation | Config | Val Loss | Δ from baseline | Wall time |
|----------|--------|----------|-----------------|-----------|
| Baseline | — | X.XXX | — | XXs |
| Depth | 2L | X.XXX | +X.XXX | XXs |
| Depth | 4L | X.XXX | +X.XXX | XXs |
| Depth | 12L | X.XXX | -X.XXX | XXs |
| Width | 128d | X.XXX | +X.XXX | XXs |
| Width | 256d | X.XXX | +X.XXX | XXs |
| Activation | ReLU | X.XXX | +X.XXX | XXs |
| Activation | SwiGLU | X.XXX | -X.XXX | XXs |
| Norm | Post-LN | X.XXX | +X.XXX | XXs |

### Key Findings
1. ...
2. ...
3. ...
```

### 2.2 Scaling Analysis

Plot your own loss-vs-parameters scaling curve from the width and depth ablations:

```python
def scaling_analysis(ablation_results: list):
    """Fit and plot scaling law from your ablation experiments."""
    params = [r["n_params"] for r in ablation_results]
    losses = [r["final_val_loss"] for r in ablation_results]
    
    # Fit: L(N) = a * N^(-alpha)
    log_N = np.log(params)
    log_L = np.log(losses)
    alpha, log_a = np.polyfit(log_N, log_L, 1)
    
    print(f"Your scaling exponent: α = {-alpha:.4f}")
    print(f"Kaplan et al. found:   α ≈ 0.076")
    print(f"Ratio: your α is {-alpha/0.076:.1f}x Kaplan's")
    
    # The exponent will differ because:
    # - You're training on much less data
    # - Your models are much smaller
    # - You may not be in the scaling regime yet
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(params, losses, s=80, zorder=5, label="Your experiments")
    
    N_fit = np.logspace(np.log10(min(params)*0.5), np.log10(max(params)*2), 100)
    L_fit = np.exp(log_a) * N_fit ** alpha
    ax.plot(N_fit, L_fit, '--r', label=f"Fit: α = {-alpha:.4f}")
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Your Scaling Curve")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    
    plt.tight_layout()
    plt.savefig("capstone_scaling_curve.png", dpi=150)
    plt.show()
```

---

## Part 3: Phase II Checkpoint (45 min)

**Rules:** Answer each question from memory first. No peeking. Then verify and note your weak spots.

### Q1: The Full Attention Equation

Write the complete attention equation from memory. Label each term.

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

**Check yourself:**
- $Q = XW_Q$, $K = XW_K$, $V = XW_V$ — linear projections of input
- $QK^\top$ — dot product similarity between queries and keys, shape $(n, n)$
- $\sqrt{d_k}$ — scaling factor to prevent softmax saturation
- $\text{softmax}$ — converts scores to probabilities (rows sum to 1)
- Multiply by $V$ — weighted sum of values based on attention weights

---

### Q2: Why Scale by $\sqrt{d_k}$?

Without scaling, for large $d_k$:
- Dot products $q \cdot k$ have variance proportional to $d_k$ (if $q_i, k_i \sim \mathcal{N}(0, 1)$, then $\text{Var}(q \cdot k) = d_k$)
- Large dot products → softmax saturates → gradients vanish
- Dividing by $\sqrt{d_k}$ normalizes variance to 1

**Without scaling:** $\text{softmax}([100, 1, 1, 1]) \approx [1.0, 0.0, 0.0, 0.0]$ — no gradient flows through non-attended positions.

---

### Q3: Causal Mask in Decoder Self-Attention

The causal mask ensures position $i$ can only attend to positions $\leq i$:

$$M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

Applied before softmax: $\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V$

```
Causal mask for sequence of length 4:

      k₁  k₂  k₃  k₄
q₁ [  0  -∞  -∞  -∞ ]    position 1 sees only itself
q₂ [  0   0  -∞  -∞ ]    position 2 sees 1, 2
q₃ [  0   0   0  -∞ ]    position 3 sees 1, 2, 3
q₄ [  0   0   0   0 ]    position 4 sees all
```

This prevents the model from "cheating" by looking at future tokens during training, while enabling parallel computation of all positions simultaneously.

---

### Q4: KV Cache

**What is cached:** The key ($K$) and value ($V$) matrices for all previously generated tokens across all layers.

**Why:** During autoregressive generation, each new token only needs its own $Q$ vector. The $K$ and $V$ for previous positions don't change. Without cache, we'd recompute $K, V$ for all previous tokens at every step — $O(n^2)$ total work becomes $O(n)$ per step.

**Memory cost:** For each layer: $2 \times \text{batch} \times \text{seq\_len} \times d_{\text{model}}$ floats.

For a model with $L$ layers, sequence length $n$, and $d$ dimensions:

$$\text{KV cache memory} = 2 \times L \times n \times d \times \text{sizeof(float)}$$

---

### Q5: Chinchilla Scaling Laws

**Key finding:** The optimal number of training tokens is ~20× the number of parameters.

**Implication:** GPT-3 (175B params, 300B tokens) was undertrained. Chinchilla (70B params, 1.4T tokens) matched GPT-3's quality with 2.5× fewer parameters.

**Rule of thumb:** $D_{\text{opt}} \approx 20N$, where $D$ = training tokens and $N$ = parameters.

**Modern practice:** Over-train smaller models (LLaMA 3: 8B params, 15T tokens = 1875 tokens/param) because inference cost dominates deployment cost.

---

### Q6: BPE Tokenization Step by Step

1. Start with character-level vocabulary: `{a, b, c, ..., z, space, ...}`
2. Count all adjacent character pairs in corpus
3. Find most frequent pair (e.g., `t` + `h` → 2,847 occurrences)
4. Merge that pair into a new token: `th`
5. Update corpus with new token, re-count pairs
6. Repeat steps 2-5 for desired vocabulary size (e.g., 50,000 merges)

Each merge reduces total token count. The resulting vocabulary contains common subwords (`ing`, `tion`, `the`, etc.) that balance between character-level (too many tokens per word) and word-level (too many rare words) tokenization.

---

### Q7: GPT vs BERT vs T5

| | GPT | BERT | T5 |
|--|-----|------|-----|
| Architecture | Decoder-only | Encoder-only | Encoder-decoder |
| Pre-training | Autoregressive (next token) | Masked LM (fill blanks) | Span corruption |
| Context | Causal (left-to-right) | Bidirectional | Encoder: bidirectional; Decoder: causal |
| Best for | Generation, few-shot, scaling | Classification, NLU | Translation, summarization, seq2seq |
| Weakness | No bidirectional context | Can't generate text | More parameters for same performance |

---

### Q8: Multi-Head Attention Pseudocode

```
function MultiHeadAttention(X, n_heads, d_model):
    d_k = d_model / n_heads
    
    # Linear projections
    Q = X @ W_Q    # (seq_len, d_model)
    K = X @ W_K
    V = X @ W_V
    
    # Split into heads
    Q = reshape(Q, (seq_len, n_heads, d_k))  # then transpose
    K = reshape(K, (seq_len, n_heads, d_k))
    V = reshape(V, (seq_len, n_heads, d_k))
    
    # Scaled dot-product attention per head (parallelized)
    scores = Q @ K.T / sqrt(d_k)       # (n_heads, seq_len, seq_len)
    scores = scores + causal_mask       # -inf for future positions
    weights = softmax(scores, dim=-1)   # (n_heads, seq_len, seq_len)
    output = weights @ V                # (n_heads, seq_len, d_k)
    
    # Concatenate heads
    output = reshape(output, (seq_len, d_model))
    
    # Final projection
    return output @ W_O
```

---

## Deliverables Checklist

By the end of Day 30, confirm you have:

- [ ] **Trained model checkpoint** (`capstone_best.pt`)
- [ ] **Training curves** — loss, LR schedule, gradient norms plotted
- [ ] **Generated text samples** — at multiple temperatures, showing model learned the corpus
- [ ] **Ablation report** — at least 3 ablation axes with plots and analysis
- [ ] **Scaling curve** — loss vs parameters with fitted power law exponent
- [ ] **Checkpoint answers** — all 8 questions answered from memory, weak spots identified

---

## Key Takeaways

1. **You built a language model from scratch** — from raw text to generated output
2. **Training dynamics are universal** — the same loss curves, stability tricks, and scaling patterns you saw apply to models 1000× larger
3. **Ablation is essential** — without it, you're guessing which components matter
4. **Scaling laws hold even at small scale** — your power-law exponent may differ from Kaplan's, but the pattern is there
5. **Phase II is complete** — you understand attention, transformers, tokenization, scaling, and generation

## Connection to the Thread

Phase II built the engine. You can now explain *how* a transformer works, *why* it scales, and *what* each component contributes. Phase III starts tomorrow with the question: "You have a pretrained LM — now what?" The answer is the modern LLM pipeline: SFT → RLHF/DPO → deployment. This is the same pipeline that turns a pretrained vision-language model into a VLA.

## Further Reading

- Karpathy, "The spelled-out intro to neural networks and backpropagation" (YouTube)
- Karpathy, nanoGPT repository — compare your implementation against his
- Zhang et al., "OPT: Open Pre-trained Transformer Language Models" (2022) — training log of a large model, showing the same dynamics at scale
- Biderman et al., "Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling" (2023)
