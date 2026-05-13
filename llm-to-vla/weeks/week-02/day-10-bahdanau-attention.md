# Day 10: Bahdanau Attention — "Looking Back at the Source"

> **Phase II — Attention, Transformers & Scaling | Week 2 | 2.5 hours**
> *"The most important development in neural machine translation was allowing the decoder to look back."*
> — Dzmitry Bahdanau, 2014

## Navigation
- **Previous:** [Day 9: Phase I Checkpoint](day-09-phase1-checkpoint.md)
- **Next:** [Day 11: Scaled Dot-Product Attention](day-11-scaled-dot-product-attention.md)
- **Week:** [Week 2 Overview](../README.md)
- **Phase:** [Phase II: Attention & Transformers](../../study-notes/02-attention-mechanism.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 10.1 The Bottleneck Problem

Recall from Day 7 the seq2seq model: an encoder reads the source sentence and compresses it into a **single fixed-length vector** $c$, which the decoder must use to generate the entire target sentence.

```
Source:  x₁ ─→ h₁ ─→ h₂ ─→ h₃ ─→ h₄ ──→ [c]   ← entire sentence in one vector!
                                              │
Target:                                      [c] ─→ s₁ ─→ s₂ ─→ s₃
                                                     y₁    y₂    y₃
```

This is the **information bottleneck**. No matter how long the source sentence is — 5 words or 50 — everything must be squeezed through a vector of dimension $d$. Performance degrades dramatically on long sentences (Cho et al., 2014).

**The question:** Can the decoder look at *all* encoder hidden states, not just the last one?

### 10.2 Bahdanau Attention (2014)

Bahdanau, Cho, and Bengio proposed: let the decoder attend to different parts of the source for each target word.

**Key insight:** Attention is a **soft dictionary lookup**.

- **Query** ($q$): What the decoder is looking for — the current decoder state $s_{i-1}$
- **Keys/Values** ($k, v$): What the encoder offers — all encoder hidden states $h_1, \ldots, h_T$

For each decoder step $i$, compute a **context vector** $c_i$ as a weighted sum of encoder states:

$$c_i = \sum_{j=1}^{T} \alpha_{ij} \, h_j$$

where $\alpha_{ij}$ are **attention weights** (soft alignment):

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T} \exp(e_{ik})}$$

The **energy function** (alignment model) is a learnable feedforward network:

$$e_{ij} = v^T \tanh(W_s \, s_{i-1} + W_h \, h_j)$$

```
                   Encoder hidden states
                   h₁    h₂    h₃    h₄
                    │     │     │     │
                    ▼     ▼     ▼     ▼
   s_{i-1} ───→ [energy] [energy] [energy] [energy]
                    │     │     │     │
                    ▼     ▼     ▼     ▼
                  e_{i1} e_{i2} e_{i3} e_{i4}
                    │     │     │     │
                    └─────┴─────┴─────┘
                          │
                       softmax
                          │
                    ┌─────┴─────┐
                    ▼     ▼     ▼     ▼
                  α_{i1} α_{i2} α_{i3} α_{i4}
                    │     │     │     │
                    ▼     ▼     ▼     ▼
                  α·h₁ + α·h₂ + α·h₃ + α·h₄ = c_i
```

### 10.3 Why This Is Called "Additive" Attention

The energy function uses **addition** inside the tanh:

$$e_{ij} = v^T \tanh(\underbrace{W_s \, s_{i-1}}_{\text{query term}} + \underbrace{W_h \, h_j}_{\text{key term}})$$

This is why Bahdanau attention is classified as **additive attention**, in contrast to the **dot-product attention** we'll see in Day 11.

**Parameters to learn:**
- $W_s \in \mathbb{R}^{d_a \times d_s}$ — projects decoder state
- $W_h \in \mathbb{R}^{d_a \times d_h}$ — projects encoder state
- $v \in \mathbb{R}^{d_a}$ — weight vector that produces a scalar energy

### 10.4 Attention as Soft Alignment

In traditional machine translation, **alignment** is a hard mapping between source and target words. Attention replaces this with a *soft*, differentiable version:

| Hard alignment | Soft attention |
|---|---|
| Each target word → exactly one source word | Each target word → weighted combination |
| Discrete, non-differentiable | Continuous, differentiable |
| Must be computed separately | Learned end-to-end |

The attention weight $\alpha_{ij}$ tells us: "When generating target word $i$, how much should we look at source word $j$?"

### 10.5 The Decoder Update

With attention, the decoder GRU update becomes:

$$s_i = \text{GRU}(s_{i-1}, [y_{i-1}; c_i])$$

where $[y_{i-1}; c_i]$ is the concatenation of the previous target embedding and the context vector. The context changes at every step — the decoder sees a *different* view of the source for each word it generates.

---

## Implementation (60 min)

### 10.6 Building a Seq2Seq Model with Bahdanau Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention mechanism."""
    
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()
        self.W_h = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.W_s = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)
    
    def forward(
        self,
        decoder_state: torch.Tensor,   # (batch, decoder_dim)
        encoder_outputs: torch.Tensor,  # (batch, src_len, encoder_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Project encoder outputs: (batch, src_len, attn_dim)
        keys = self.W_h(encoder_outputs)
        
        # Project decoder state: (batch, 1, attn_dim)
        query = self.W_s(decoder_state).unsqueeze(1)
        
        # Energy: (batch, src_len, 1) → squeeze → (batch, src_len)
        energy = self.v(torch.tanh(keys + query)).squeeze(-1)
        
        # Attention weights: (batch, src_len)
        alpha = F.softmax(energy, dim=-1)
        
        # Context vector: (batch, encoder_dim)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, alpha


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, src: torch.Tensor):
        # src: (batch, src_len)
        embedded = self.embedding(src)                # (batch, src_len, embed_dim)
        outputs, hidden = self.rnn(embedded)          # outputs: (batch, src_len, 2*hidden)
        # Combine forward/backward final states for decoder init
        hidden = torch.tanh(self.fc(
            torch.cat([hidden[-2], hidden[-1]], dim=-1)
        ))
        return outputs, hidden


class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, encoder_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(encoder_dim, hidden_dim, hidden_dim)
        self.rnn = nn.GRU(embed_dim + encoder_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim + encoder_dim + embed_dim, vocab_size)
    
    def forward(self, token, decoder_state, encoder_outputs):
        # token: (batch,) — single time step
        embedded = self.embedding(token)              # (batch, embed_dim)
        context, alpha = self.attention(decoder_state, encoder_outputs)
        
        rnn_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)
        output, hidden = self.rnn(rnn_input, decoder_state.unsqueeze(0))
        
        prediction = self.fc_out(
            torch.cat([output.squeeze(1), context, embedded], dim=-1)
        )
        return prediction, hidden.squeeze(0), alpha
```

### 10.7 Visualizing Attention Weights

```python
import matplotlib.pyplot as plt
import numpy as np


def plot_attention(attention_weights: np.ndarray, src_tokens: list[str], tgt_tokens: list[str]):
    """Plot attention heatmap showing soft alignment between source and target."""
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(attention_weights, cmap='viridis')
    fig.colorbar(cax)
    
    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45, ha='left')
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens)
    
    ax.set_xlabel('Source')
    ax.set_ylabel('Target')
    ax.set_title('Bahdanau Attention Weights')
    plt.tight_layout()
    plt.savefig('attention_heatmap.png', dpi=150)
    plt.show()


# Example: after training, collect attention weights during inference
# attention_matrix[i, j] = α_{ij} = how much target word i attends to source word j
# Expect near-diagonal for monotonic languages, crossed patterns for reordering
```

---

## Exercise (45 min)

### E10.1 Attention Computation by Hand

Given encoder states $h_1 = [1, 0]$, $h_2 = [0, 1]$, $h_3 = [1, 1]$ and decoder state $s_0 = [0.5, 0.5]$:

1. Assuming $W_s = W_h = I_{2 \times 2}$ (identity) and $v = [1, 1]^T$, compute energies $e_{01}, e_{02}, e_{03}$
2. Compute attention weights $\alpha_{01}, \alpha_{02}, \alpha_{03}$
3. Compute the context vector $c_0$
4. Which encoder state gets the most attention? Why?

### E10.2 Training Comparison

Train two seq2seq models on Multi30k EN→DE:
1. Vanilla seq2seq (Day 7 style) with final-state-only context
2. Seq2seq with Bahdanau attention

Compare:
- BLEU scores on test set
- Training curves (loss vs epoch)
- Performance on short (< 10 tokens) vs long (> 20 tokens) sentences

### E10.3 Attention Heatmap Analysis

Using your trained attention model:
1. Generate translations for 5 sentences of varying length
2. Plot attention heatmaps for each
3. Find examples of: monotonic alignment, reordering, many-to-one alignment
4. Does the model learn word order differences between EN and DE?

---

## Key Takeaways

- The **information bottleneck** of fixed-context seq2seq is solved by letting the decoder look at all encoder states
- **Bahdanau attention** learns a soft alignment: $c_i = \sum_j \alpha_{ij} h_j$
- The energy function $e_{ij} = v^T \tanh(W_s s_{i-1} + W_h h_j)$ is a small learned network
- Attention weights are **interpretable** — they show which source words influence each target word
- This is **additive** attention (addition inside tanh); tomorrow we'll see **multiplicative** (dot-product) attention

## Connection to the Thread

Attention solves a compression problem. Fixed-context seq2seq tries to compress the entire source into one vector — a lossy compression that loses information for long sequences. Attention is **adaptive decompression**: at each step, the decoder selects which parts of the source to decompress. This is the beginning of the transformer's power: *selective access to information beats wholesale compression*.

## Further Reading

- Bahdanau, Cho, Bengio. **"Neural Machine Translation by Jointly Learning to Align and Translate"** (2014). [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
- Luong, Pham, Manning. **"Effective Approaches to Attention-based Neural Machine Translation"** (2015). [arXiv:1508.04025](https://arxiv.org/abs/1508.04025)
- Olah & Carter. **"Attention and Augmented Recurrent Neural Networks"** (Distill, 2016). [distill.pub](https://distill.pub/2016/augmented-rnns/)
- Jay Alammar. **"Visualizing A Neural Machine Translation Model"**. [jalammar.github.io](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
