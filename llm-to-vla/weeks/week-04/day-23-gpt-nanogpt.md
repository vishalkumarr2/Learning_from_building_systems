# Day 23: GPT & nanoGPT — Decoder-Only Transformers (Ablation Lab Day 1)

> **Phase II — Attention, Transformers & Scaling | Week 4 | 2.5 hours**
> *"The unreasonable effectiveness of predicting the next token." — Ilya Sutskever*

## Navigation
- **Previous:** [Day 22: Tokenization Deep Dive](day-22-tokenization.md)
- **Next:** [Day 24: nanoGPT Ablations Day 2](day-24-nanogpt-ablations.md)
- **Week:** [Week 4 Overview](README.md)
- **Phase:** [Phase II: Attention & Transformers](../../study-notes/03-transformers.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 23.1 The Decoder-Only Architecture

GPT uses **only** the transformer decoder — no encoder, no cross-attention. Just causal self-attention stacked deep.

```
Full Transformer (Vaswani 2017):         GPT (Decoder-Only):
┌──────────────┐ ┌──────────────┐       ┌──────────────┐
│   Encoder    │ │   Decoder    │       │   Decoder    │
│              │ │              │       │              │
│ Self-Attn    │ │ Masked       │       │ Causal       │
│ (bidir.)     │→│ Self-Attn    │       │ Self-Attn    │
│              │ │              │       │              │
│ FFN          │ │ Cross-Attn   │       │ FFN          │
│              │ │ (to encoder) │       │              │
│              │ │              │       │ × N layers   │
│              │ │ FFN          │       │              │
│              │ │              │       │              │
│              │ │ × N layers   │       │              │
└──────────────┘ └──────────────┘       └──────────────┘
     ↑ source         ↑ target                ↑ everything
     
Needs paired data    Needs paired data     Just needs text!
(src → tgt)          (src → tgt)           Next-token prediction
```

**Why decoder-only won:**
1. **Simplicity** — One architecture, one training objective, one data type
2. **Data efficiency** — Only needs raw text (unlimited on the internet)
3. **Generality** — Any task can be framed as text completion
4. **Emergent abilities** — Scale unlocks capabilities not present in smaller versions

### 23.2 GPT Evolution: 1 → 2 → 3 → 4

```
GPT-1 (2018)          GPT-2 (2019)         GPT-3 (2020)          GPT-4 (2023)
117M params           1.5B params          175B params           ~1.8T params (est.)
12 layers             48 layers            96 layers             ~120 layers (est.)
768 dim               1600 dim             12288 dim             ~?
BooksCorpus           WebText (40GB)       CommonCrawl+          ~?
                                           (570GB)
                                           
Key idea:             Key idea:            Key idea:             Key idea:
Transfer learning     Zero-shot via        In-context            Multimodal +
(pretrain + finetune) task description     learning (few-shot)   RLHF + MoE (est.)

"Fine-tuning          "Language models      "Language models     "The architecture
 works for NLP"        are unsupervised     are few-shot         works for everything
                       multitask learners"   learners"            (text, vision, code)"
```

**The progression:**
1. **GPT-1:** Proved that transformer pre-training + fine-tuning works for NLP
2. **GPT-2:** Showed that scale alone enables zero-shot task performance
3. **GPT-3:** Demonstrated in-context learning — no weight updates needed
4. **GPT-4:** Extended to multimodal, achieved near-human performance on benchmarks

### 23.3 The Causal Language Modeling Objective

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_1, x_2, \ldots, x_{t-1}; \theta)$$

Every position predicts the next token, supervised by the actual next token:

```
Input:    [BOS]  The   cat   sat   on    the
Target:    The   cat   sat   on    the   mat

Position 0: P("The" | [BOS])           ← predict "The"
Position 1: P("cat" | [BOS], The)      ← predict "cat"  
Position 2: P("sat" | [BOS], The, cat) ← predict "sat"
...

Every token is BOTH a prediction target AND context for later tokens.
Training is maximally efficient — every position contributes to the loss.
```

### 23.4 The GPT Block (nanoGPT Style)

```
Input tokens → Token Embedding + Position Embedding
                        ↓
              ┌─────────────────────┐
              │ LayerNorm           │ ← Pre-LN
              │ Causal Self-Attn    │
              │ + Residual          │
              │                     │
              │ LayerNorm           │ ← Pre-LN
              │ FFN (GELU)          │
              │ + Residual          │
              └─────────────────────┘
                   × N layers
                        ↓
              LayerNorm (final)
                        ↓
              Linear → logits (vocab_size)
                        ↓
              softmax → P(next token)
```

**GPT-2 specifics:**
- Pre-LN (not Post-LN like original transformer)
- GELU activation (not ReLU)
- Learned positional embeddings (not sinusoidal)
- No bias in attention projections (GPT-3+/LLaMA)
- Weight tying: token embedding = output projection (transposed)

### 23.5 What Makes This Different from Day 14?

On Day 14, you built a full encoder-decoder transformer for translation. nanoGPT is different:

| Day 14 Transformer | nanoGPT |
|---------------------|---------|
| Encoder + Decoder | Decoder only |
| Cross-attention | No cross-attention |
| Sinusoidal positions | Learned positions |
| Post-LN | Pre-LN |
| ReLU FFN | GELU FFN |
| Translation task | Language modeling |
| Small vocab (~10K) | Large vocab (50K+) |
| Short sequences (~50) | Medium sequences (1024) |
| Trained from scratch on toy data | Can reproduce GPT-2 |

---

## Implementation (60 min)

### 23.6 Read and Annotate nanoGPT

The core of nanoGPT is ~300 lines. Here's the annotated architecture:

```python
"""
nanoGPT — annotated core architecture.
Original: https://github.com/karpathy/nanoGPT
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, n_embd, n_head, block_size, dropout=0.0, bias=False):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

        # Key, Query, Value projections — all in one matrix for efficiency
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask — registered as buffer (not a parameter)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
            .view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Calculate Q, K, V in one shot
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # Reshape: (B, T, C) → (B, n_head, T, head_dim)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Attention: (B, n_head, T, T)
        # Use PyTorch's SDPA which auto-selects Flash Attention when possible
        att = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,  # ← this enables the causal mask!
        )

        # Re-assemble heads: (B, n_head, T, head_dim) → (B, T, C)
        att = att.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        return self.resid_dropout(self.c_proj(att))


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, n_embd, dropout=0.0, bias=False):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    """Transformer block: Pre-LN attention + FFN with residuals."""

    def __init__(self, n_embd, n_head, block_size, dropout=0.0, bias=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, bias)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x):
        # Pre-LN: normalize BEFORE sublayer
        x = x + self.attn(self.ln_1(x))   # residual + attention
        x = x + self.mlp(self.ln_2(x))    # residual + FFN
        return x


class GPT(nn.Module):
    """The full GPT language model."""

    def __init__(self, vocab_size, block_size=1024, n_layer=12,
                 n_head=12, n_embd=768, dropout=0.0, bias=False):
        super().__init__()
        self.block_size = block_size

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(vocab_size, n_embd),          # token embeddings
            wpe=nn.Embedding(block_size, n_embd),          # position embeddings
            drop=nn.Dropout(dropout),
            h=nn.ModuleList([
                Block(n_embd, n_head, block_size, dropout, bias)
                for _ in range(n_layer)
            ]),
            ln_f=nn.LayerNorm(n_embd),                    # final layer norm
        ))
        # Language model head — shares weights with token embedding!
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        # Initialize weights
        self.apply(self._init_weights)
        # Special scaled init for residual projections (GPT-2 recipe)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"GPT model: {n_params/1e6:.1f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        idx: (B, T) token indices
        targets: (B, T) target token indices (shifted by 1)
        """
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence {T} > block_size {self.block_size}"

        # Token + position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)     # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)     # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            # Training: compute loss
            logits = self.lm_head(x)            # (B, T, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # Inference: only compute last position
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressive generation."""
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
```

### 23.7 Train on Shakespeare

```python
import os
import requests

# Download Shakespeare dataset
data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
data_path = "shakespeare.txt"
if not os.path.exists(data_path):
    text = requests.get(data_url).text
    with open(data_path, 'w') as f:
        f.write(text)
else:
    with open(data_path, 'r') as f:
        text = f.read()

print(f"Dataset: {len(text)} characters")

# Character-level tokenizer (simple for this exercise)
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(f"Vocab size: {vocab_size}")
print(f"Chars: {''.join(chars[:50])}...")

# Train/val split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split, batch_size=64, block_size=256):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x, y


# Create a small GPT
model = GPT(
    vocab_size=vocab_size,
    block_size=256,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.2,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Training loop
for step in range(5000):
    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)

    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 500 == 0:
        model.eval()
        val_x, val_y = get_batch('val')
        val_x, val_y = val_x.to(device), val_y.to(device)
        _, val_loss = model(val_x, val_y)
        model.train()

        print(f"Step {step}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")

# Generate!
model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=50)
print("\n--- Generated Shakespeare ---")
print(decode(generated[0].tolist()))
```

---

## Exercise (45 min)

### E23.1 Architecture Comparison

List EVERY difference between your Day 14 transformer and this nanoGPT:

| Feature | Day 14 Transformer | nanoGPT |
|---------|-------------------|---------|
| Architecture type | ? | ? |
| Normalization | ? | ? |
| Activation | ? | ? |
| Positional encoding | ? | ? |
| Weight tying | ? | ? |
| Cross-attention | ? | ? |
| Initialization | ? | ? |

### E23.2 Parameter Count Verification

For the Shakespeare model (6 layers, 6 heads, 384 dim, vocab=65):

Calculate by hand:
1. Token embedding: `vocab_size × n_embd` = ?
2. Position embedding: `block_size × n_embd` = ?
3. Per block attention: `4 × n_embd²` = ? (Q, K, V, Out projections)
4. Per block FFN: `2 × n_embd × 4*n_embd` = ?
5. Per block LayerNorm: `2 × 2 × n_embd` = ?
6. Total = ?

Compare with `sum(p.numel() for p in model.parameters())`. Do they match?

### E23.3 Generation Quality vs Training

Generate text at steps 0, 500, 1000, 2000, 5000:

```python
checkpoints = [0, 500, 1000, 2000, 5000]
# At each checkpoint, generate 200 characters and save
# Observe: random → character patterns → word patterns → Shakespeare-like

# Also: try temperature = [0.1, 0.5, 0.8, 1.0, 1.5, 2.0]
# What happens at each extreme?
```

---

## Key Takeaways

1. **Decoder-only transformers** use causal self-attention — each token can only attend to previous tokens
2. **GPT's training objective is simple:** predict the next token — this scales remarkably well
3. **Weight tying** shares the token embedding with the output projection, reducing parameters
4. **nanoGPT is ~300 lines** — the entire architecture fits in your head, yet it reproduces GPT-2
5. **GPT-1→2→3→4** shows that scale + simplicity beats architectural complexity
6. **Character-level Shakespeare** is a perfect sandbox — fast training, immediate visual feedback

## Connection to the Thread

Today you built the architecture that dominates modern AI. GPT's decoder-only design is what powers ChatGPT, Claude, LLaMA, and eventually the language backbone of VLA models. Tomorrow, you'll use this codebase as an **ablation laboratory** — systematically varying architecture choices (heads, layers, dim, activation, normalization) to understand what matters and why. This is how real ML research works.

## Further Reading

- **[nanoGPT Repository](https://github.com/karpathy/nanoGPT)** — Karpathy, full code + training scripts
- **[Karpathy's "Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY)** — 2-hour video, essential
- **[GPT-1 Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)** — Radford et al., 2018
- **[GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)** — Radford et al., 2019
- **[GPT-3 Paper](https://arxiv.org/abs/2005.14165)** — Brown et al., 2020
- **[Weight Tying](https://arxiv.org/abs/1608.05859)** — Press & Wolf, 2017
