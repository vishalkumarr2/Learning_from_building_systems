# Day 29: Phase II Capstone — Day 1

> **Phase II — Attention, Transformers & Scaling | Week 5 | 2.5 hours**
> *"Theory without practice is empty. Practice without theory is blind. Today you build." — paraphrasing Kant*

## Navigation
- **Previous:** [Day 28: T5 & Encoder-Decoder LMs](../week-04/day-28-t5-encoder-decoder.md)
- **Next:** [Day 30: Phase II Capstone Day 2](day-30-phase2-capstone-2.md)
- **Week:** [Week 5 Overview](../README.md)
- **Phase:** [Phase II: Attention & Transformers](../../study-notes/05-gpt-scaling.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Project Overview

**Goal:** Train a mini-LM (~10M parameters) from scratch using your nanoGPT setup. This consolidates everything from Phase II: attention, transformers, positional encoding, normalization, optimization, tokenization, and scaling.

```
Phase II Capstone Architecture:

  ┌──────────────────────────────────────────────────────────┐
  │                                                            │
  │  Day 29: Setup & Train         Day 30: Analyze & Report   │
  │                                                            │
  │  1. Choose corpus              5. Complete training        │
  │  2. Tokenize data              6. Run ablations (Day 24)  │
  │  3. Configure model            7. Scaling analysis         │
  │  4. Start training             8. Phase II Checkpoint      │
  │     └─ wandb logging                                      │
  │     └─ stability tricks                                    │
  │                                                            │
  │  Deliverables:                                             │
  │  ✅ Trained model checkpoint                              │
  │  ✅ Training curves (loss, lr, grad norm)                 │
  │  ✅ Ablation report with plots                            │
  │  ✅ Scaling analysis                                       │
  │  ✅ Phase II checkpoint answers                           │
  │                                                            │
  └──────────────────────────────────────────────────────────┘
```

---

## Step 1: Choose Your Domain Corpus (15 min)

Pick one corpus that interests you. Each has different characteristics:

| Corpus | Size | Characteristics | Why Interesting |
|--------|------|----------------|-----------------|
| **Shakespeare** | ~1MB | Archaic English, poetry, dialogue | Classic nanoGPT benchmark |
| **Python code** | ~5-50MB | Structured, syntax rules, indentation | Model must learn grammar |
| **Robotics papers** | ~5-20MB | Technical, equations, domain-specific | Directly relevant to VLA goal |
| **Wikipedia subset** | ~50MB | Factual, diverse topics | Tests general knowledge compression |

For robotics papers, you can scrape abstracts from arXiv using the `robotics` category:

```python
import urllib.request
import json

def fetch_arxiv_abstracts(category="cs.RO", max_results=5000):
    """Fetch robotics paper abstracts from arXiv API."""
    base_url = "http://export.arxiv.org/api/query"
    all_text = []
    batch_size = 100
    
    for start in range(0, max_results, batch_size):
        query = (
            f"search_query=cat:{category}"
            f"&start={start}&max_results={batch_size}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
        url = f"{base_url}?{query}"
        
        # Use urllib to avoid extra dependencies
        with urllib.request.urlopen(url) as response:
            data = response.read().decode()
        
        # Simple XML parsing for abstracts
        import re
        abstracts = re.findall(r'<summary>(.*?)</summary>', data, re.DOTALL)
        all_text.extend([a.strip() for a in abstracts])
        
        if len(abstracts) < batch_size:
            break
    
    return "\n\n".join(all_text)

# corpus = fetch_arxiv_abstracts()
# with open("robotics_corpus.txt", "w") as f:
#     f.write(corpus)
```

---

## Step 2: Tokenize the Data (20 min)

Use BPE tokenization (from Day 22) or character-level for simplicity:

```python
import os
import numpy as np


def prepare_data(corpus_path: str, tokenizer_type: str = "char"):
    """Prepare training and validation data."""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Corpus size: {len(text):,} characters")
    
    if tokenizer_type == "char":
        # Character-level tokenization
        chars = sorted(list(set(text)))
        vocab_size = len(chars)
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
        
    elif tokenizer_type == "bpe":
        # Use tiktoken for GPT-2 BPE
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        vocab_size = enc.n_vocab
        
        encode = enc.encode
        decode = enc.decode
    
    # Encode entire corpus
    data = np.array(encode(text), dtype=np.uint16)
    print(f"Total tokens: {len(data):,}")
    print(f"Vocab size: {vocab_size}")
    
    # Split: 90% train, 10% val
    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Train tokens: {len(train_data):,}")
    print(f"Val tokens: {len(val_data):,}")
    
    return train_data, val_data, vocab_size, encode, decode
```

---

## Step 3: Configure the Model (30 min)

Design a ~10M parameter model. Use the parameter counting formula:

$$N \approx 12 \cdot n_{\text{layers}} \cdot d_{\text{model}}^2 + V \cdot d_{\text{model}}$$

where the first term is the transformer blocks and the second is the embedding matrix.

```python
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class CapstoneConfig:
    """~10M parameter GPT configuration."""
    # Architecture (tuned for ~10M params)
    n_layers: int = 8
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048         # 4 * d_model
    block_size: int = 256    # context length
    vocab_size: int = 65     # updated after tokenization
    dropout: float = 0.1
    
    # Normalization (use lessons from Day 24 ablation)
    norm_type: str = "rmsnorm"
    norm_position: str = "pre"  # Pre-LN for training stability
    
    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    max_steps: int = 10000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # Logging
    eval_interval: int = 250
    log_interval: int = 50
    seed: int = 42


def count_parameters(config: CapstoneConfig) -> int:
    """Estimate parameter count before building."""
    d = config.d_model
    L = config.n_layers
    V = config.vocab_size
    
    # Embedding: V * d
    emb = V * d
    # Position embedding: block_size * d
    pos = config.block_size * d
    # Per layer: 4*d*d (attn) + 2*d*d*4 (ffn) + norms
    per_layer = 4 * d * d + 2 * 4 * d * d + 2 * d  # Q,K,V,O + FFN up,down + norms
    # Total
    total = emb + pos + L * per_layer + d  # final norm
    
    return total


config = CapstoneConfig()
estimated = count_parameters(config)
print(f"Estimated parameters: {estimated / 1e6:.1f}M")
```

---

## Step 4: Training with All Stability Techniques (60 min)

Apply everything from Day 7's training stability cookbook:

```python
import math


def get_lr(step: int, config: CapstoneConfig) -> float:
    """Cosine learning rate schedule with warmup."""
    # Warmup phase
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    
    # Cosine decay phase
    decay_steps = config.max_steps - config.warmup_steps
    progress = (step - config.warmup_steps) / decay_steps
    progress = min(progress, 1.0)
    
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def train_capstone(model, train_data, val_data, config, decode_fn):
    """Full training loop with stability techniques and logging."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Optimizer: AdamW with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.95),       # GPT-3 betas
        weight_decay=config.weight_decay,
    )
    
    # Training state
    train_losses = []
    val_losses = []
    grad_norms = []
    lrs = []
    best_val_loss = float('inf')
    
    for step in range(config.max_steps):
        model.train()
        
        # Get batch
        ix = torch.randint(len(train_data) - config.block_size, (config.batch_size,))
        x = torch.stack([
            torch.from_numpy(train_data[i:i+config.block_size].astype(int))
            for i in ix
        ]).to(device)
        y = torch.stack([
            torch.from_numpy(train_data[i+1:i+1+config.block_size].astype(int))
            for i in ix
        ]).to(device)
        
        # Forward pass
        logits = model(x)
        loss = nn.functional.cross_entropy(
            logits.view(-1, config.vocab_size),
            y.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (Day 7)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.grad_clip
        )
        grad_norms.append(grad_norm.item())
        
        # Learning rate schedule
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        lrs.append(lr)
        
        # Step
        optimizer.step()
        train_losses.append(loss.item())
        
        # Logging
        if step % config.log_interval == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | "
                  f"lr {lr:.2e} | grad_norm {grad_norm:.2f}")
        
        # Evaluation
        if step % config.eval_interval == 0:
            model.eval()
            with torch.no_grad():
                val_ix = torch.randint(len(val_data) - config.block_size, (config.batch_size,))
                vx = torch.stack([
                    torch.from_numpy(val_data[i:i+config.block_size].astype(int))
                    for i in val_ix
                ]).to(device)
                vy = torch.stack([
                    torch.from_numpy(val_data[i+1:i+1+config.block_size].astype(int))
                    for i in val_ix
                ]).to(device)
                
                val_logits = model(vx)
                val_loss = nn.functional.cross_entropy(
                    val_logits.view(-1, config.vocab_size),
                    vy.view(-1)
                )
                val_losses.append(val_loss.item())
                
                print(f"  → val_loss: {val_loss.item():.4f}")
                
                # Save best model
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    torch.save(model.state_dict(), "capstone_best.pt")
            
            # Generate sample
            if step % (config.eval_interval * 2) == 0:
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                generated = model.generate(context, max_new_tokens=100)
                print(f"  → sample: {decode_fn(generated[0].tolist())[:200]}")
    
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "grad_norms": grad_norms,
        "lrs": lrs,
        "best_val_loss": best_val_loss,
    }
```

### Wandb Logging (Optional but Recommended)

```python
# If wandb is available, add structured logging
try:
    import wandb
    
    wandb.init(
        project="llm-to-vla-capstone",
        config=config.__dict__,
        name=f"capstone-{config.n_layers}L-{config.d_model}d",
    )
    
    # Inside training loop, add:
    # wandb.log({
    #     "train/loss": loss.item(),
    #     "train/lr": lr,
    #     "train/grad_norm": grad_norm.item(),
    #     "val/loss": val_loss.item(),
    # }, step=step)
    
except ImportError:
    print("wandb not installed — logging to stdout only")
```

---

## Checklist for Today

By the end of Day 29, you should have:

- [ ] **Corpus selected and downloaded** — know its size in characters and tokens
- [ ] **Tokenizer configured** — char-level or BPE, vocabulary inspected
- [ ] **Model configured** — ~10M params, Pre-LN, appropriate dimensions
- [ ] **Training started** — loss is decreasing, no NaN, grad norms are stable
- [ ] **Logging active** — can see loss curves, learning rate schedule, grad norms
- [ ] **First samples generated** — model produces something (even if gibberish early on)

Training will continue overnight or into Day 30. If you have a GPU, aim for 5,000–10,000 steps. On CPU, reduce to 2,000 steps with a smaller model.

---

## Key Takeaways

1. **A 10M param model is small but instructive** — you'll see all the training dynamics that larger models exhibit
2. **Pre-LN + cosine schedule + gradient clipping** are non-negotiable (Day 7 revisited)
3. **Data preparation matters** — tokenization choice affects model quality and training speed
4. **Logging everything** enables the ablation analysis you'll do tomorrow
5. **Start training early** — don't wait for perfection; iterate on a running system

## Connection to the Thread

This capstone proves you can go from raw text to a working language model. Every VLA starts here — the language modeling backbone (RT-2 uses PaLM, OpenVLA uses LLaMA). By training your own, you understand exactly what's happening inside. Tomorrow, you'll analyze what you've built and connect it to scaling laws.

## Further Reading

- Karpathy, "nanoGPT" (GitHub) — the reference implementation
- Karpathy, "Let's build GPT: from scratch, in code, spelled out" (YouTube, 2023)
- Goyal et al., "Accurate, Large Minibatch SGD" (2017) — learning rate warmup
- Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2019) — AdamW
