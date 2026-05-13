# Day 21: BERT & Masked Language Modeling — The Encoder Paradigm

> **Phase II — Attention, Transformers & Scaling | Week 3 | 2.5 hours**
> *"To understand a word, you must understand its context — and context means looking in both directions." — Devlin et al.*

## Navigation
- **Previous:** [Day 20: Mixture of Experts](day-20-mixture-of-experts.md)
- **Next:** [Day 22: Tokenization Deep Dive](../week-04/day-22-tokenization.md)
- **Week:** [Week 3 Overview](README.md)
- **Phase:** [Phase II: Attention & Transformers](../../study-notes/03-transformers.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 21.1 Two Philosophies of Language Modeling

Until now, you've focused on **autoregressive** (left-to-right) models — the GPT lineage. BERT represents the other branch: **bidirectional encoders**.

```
Autoregressive (GPT):                 Bidirectional (BERT):
"The cat sat on the ___"              "The cat [MASK] on the mat"

Can only see LEFT context:            Sees ALL context:
  The → cat → sat → on → the → ?       The ← cat → [MASK] ← on ← the ← mat
  ──────────────────────→                ←──────────────────────────────────→

Causal mask:                          No mask (full attention):
  ■ · · · · ·                           ■ ■ ■ ■ ■ ■
  ■ ■ · · · ·                           ■ ■ ■ ■ ■ ■
  ■ ■ ■ · · ·                           ■ ■ ■ ■ ■ ■
  ■ ■ ■ ■ · ·                           ■ ■ ■ ■ ■ ■
  ■ ■ ■ ■ ■ ·                           ■ ■ ■ ■ ■ ■
  ■ ■ ■ ■ ■ ■                           ■ ■ ■ ■ ■ ■
```

**The key trade-off:**
- **GPT (decoder-only):** Can generate text (each token depends only on past) but has limited "understanding" — can't look ahead
- **BERT (encoder-only):** Rich understanding (every token sees every other token) but **cannot generate** — there's no causal structure to sample from

### 21.2 Masked Language Modeling (MLM)

Since BERT sees everything, you can't train it by predicting the next token (it would be trivially visible). Instead:

**Training objective:** Randomly mask 15% of tokens. Predict them.

```
Input:   "The  cat  [MASK]  on   the  [MASK]"
Target:  " -    -    sat     -    -    mat"

MLM Loss = CrossEntropy(predicted_token, actual_token) averaged over masked positions
```

**The 15% masking strategy (to reduce train-test mismatch):**
- 80% of selected tokens → replace with `[MASK]`
- 10% → replace with a random token
- 10% → keep original

Why the 80/10/10 split? At fine-tuning time, there are no `[MASK]` tokens. If the model only ever sees `[MASK]` during training, it won't know what to do with real tokens. The 10% random + 10% keep forces the model to maintain representations for all tokens, not just masked ones.

### 21.3 BERT Architecture

```
Input:   [CLS] The  cat  sat  on  the  mat  [SEP]
          ↓     ↓    ↓    ↓    ↓    ↓    ↓    ↓
        ┌──────────────────────────────────────────┐
        │  Token Embeddings + Position Embeddings  │
        │  + Segment Embeddings (sentence A/B)     │
        ├──────────────────────────────────────────┤
        │  Transformer Encoder × 12 (base)         │
        │  or × 24 (large)                         │
        │  Full bidirectional attention (no mask)   │
        ├──────────────────────────────────────────┤
        │  Output: contextualized embeddings       │
        └──────────────────────────────────────────┘
          ↓     ↓    ↓    ↓    ↓    ↓    ↓    ↓
        [CLS]  h₁   h₂   h₃   h₄   h₅   h₆  [SEP]
          ↓                                    
     Classification                           
     head (NSP, etc.)  

BERT-base:  L=12, H=768,  A=12 → 110M params
BERT-large: L=24, H=1024, A=16 → 340M params
```

**Special tokens:**
- `[CLS]` — Classification token. Its final hidden state aggregates the whole sequence's meaning. Used for classification tasks.
- `[SEP]` — Separator between sentences (for sentence-pair tasks like NLI).
- `[MASK]` — Placeholder for masked tokens during pre-training.

### 21.4 The Fine-Tuning Paradigm

BERT established the **pretrain → fine-tune** paradigm:

```
Phase 1: Pre-training (expensive, done once)
┌─────────────────────────────────┐
│ Massive unlabeled text corpus   │  Books + Wikipedia
│ → MLM objective                 │  ~4 days on 16 TPUs
│ → Learn general representations │
└─────────────────────────────────┘
                ↓
Phase 2: Fine-tuning (cheap, done per task)
┌─────────────────────────────────┐
│ Small labeled dataset (~1K-50K) │
│ + Add task-specific head        │
│ → Train ~3 epochs               │  ~1 hour on 1 GPU
│ → State-of-the-art on NLU tasks │
└─────────────────────────────────┘

Task-specific heads:
┌──────────────┬──────────────────────┬─────────────────┐
│ Task         │ Input                │ Head             │
├──────────────┼──────────────────────┼─────────────────┤
│ Sentiment    │ [CLS] text [SEP]     │ Linear([CLS]→2) │
│ NER          │ [CLS] text [SEP]     │ Linear(h_i→tags)│
│ QA (SQuAD)   │ [CLS] Q [SEP] P     │ Linear(h_i→2)   │
│              │                      │ (start, end pos) │
│ NLI          │ [CLS] S1 [SEP] S2   │ Linear([CLS]→3) │
└──────────────┴──────────────────────┴─────────────────┘
```

### 21.5 BERT vs GPT: When Is Each Better?

| Dimension | BERT (Encoder) | GPT (Decoder) |
|-----------|----------------|---------------|
| Attention | Bidirectional | Causal (left-to-right) |
| Pre-training | MLM (predict masked) | Next-token prediction |
| Generation | Cannot generate | Natural generation |
| Understanding | Excellent — sees full context | Good but limited to left context |
| Few-shot learning | Weak (needs fine-tuning) | Strong (in-context learning) |
| Classification | Superior with fine-tuning | Competitive with prompting at scale |
| Token representations | Deeply bidirectional | Only left-contextual |
| Scaling trajectory | BERT→RoBERTa→DeBERTa | GPT-1→2→3→4 |
| Current status | Dominant for NLU tasks | Dominant for everything at scale |

**The irony:** BERT's fine-tuning paradigm was revolutionary in 2019, but GPT-3 showed that scale + in-context learning can match or exceed BERT without ANY fine-tuning. The "pretrain + prompt" paradigm replaced "pretrain + fine-tune" for most tasks.

**But BERT is not dead:** For production classification, NER, search ranking, and embedding tasks, fine-tuned BERT models remain faster, cheaper, and often more accurate than prompting a giant LLM.

### 21.6 The BERT Legacy

BERT spawned an entire family:
- **RoBERTa** (2019): Better training recipe (more data, longer training, no NSP)
- **ALBERT** (2019): Parameter sharing across layers → smaller model
- **DeBERTa** (2021): Disentangled attention (separate content + position attention)
- **Sentence-BERT** (2019): BERT for semantic similarity via siamese networks
- **E5, BGE** (2023-24): Modern embedding models for RAG, still BERT-based

---

## Implementation (60 min)

### 21.7 Fine-Tune BERT on SST-2

```python
import torch
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
from datasets import load_dataset
import matplotlib.pyplot as plt
from tqdm import tqdm


# Load SST-2 (Stanford Sentiment Treebank — binary sentiment)
dataset = load_dataset("glue", "sst2")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_fn(examples):
    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )

tokenized = dataset.map(tokenize_fn, batched=True)
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(tokenized["train"], batch_size=32, shuffle=True)
val_loader = DataLoader(tokenized["validation"], batch_size=64)

# Load pre-trained BERT with classification head
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer with weight decay (AdamW)
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
n_epochs = 3
total_steps = len(train_loader) * n_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
)


def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"],
        )
        loss = outputs.loss
        logits = outputs.logits

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == batch["label"]).sum().item()
        total += len(batch["label"])

    return total_loss / len(loader), correct / total


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == batch["label"]).sum().item()
            total += len(batch["label"])
    return correct / total


# Training loop
train_losses = []
train_accs = []
val_accs = []

for epoch in range(n_epochs):
    loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler)
    val_acc = evaluate(model, val_loader)
    train_losses.append(loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    print(f"Epoch {epoch+1}: loss={loss:.4f}, train_acc={train_acc:.4f}, "
          f"val_acc={val_acc:.4f}")
```

Expected output:
```
Epoch 1: loss=0.2314, train_acc=0.9082, val_acc=0.9163
Epoch 2: loss=0.1087, train_acc=0.9621, val_acc=0.9243
Epoch 3: loss=0.0534, train_acc=0.9834, val_acc=0.9312
```

### 21.8 Visualize BERT Attention

```python
from transformers import BertModel

bert = BertModel.from_pretrained("bert-base-uncased", output_attentions=True)
bert.eval()

text = "The bank by the river had a nice view"
inputs = tokenizer(text, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

with torch.no_grad():
    outputs = bert(**inputs)

# outputs.attentions is a tuple of (n_layers,) each with shape
# (batch, n_heads, seq_len, seq_len)
attentions = torch.stack(outputs.attentions)  # (12, 1, 12, 10, 10)

# Visualize Layer 6, Head 0
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (layer, head) in enumerate([(0, 0), (5, 0), (11, 0)]):
    attn = attentions[layer, 0, head].numpy()
    im = axes[idx].imshow(attn, cmap='Blues', vmin=0, vmax=0.5)
    axes[idx].set_xticks(range(len(tokens)))
    axes[idx].set_yticks(range(len(tokens)))
    axes[idx].set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    axes[idx].set_yticklabels(tokens, fontsize=8)
    axes[idx].set_title(f'Layer {layer+1}, Head {head+1}')
    plt.colorbar(im, ax=axes[idx], fraction=0.046)

plt.suptitle('BERT Attention Patterns — "The bank by the river had a nice view"')
plt.tight_layout()
plt.savefig('bert_attention.png', dpi=150)
plt.show()

# What to look for:
# - Layer 1: mostly local attention (adjacent tokens)
# - Layer 6: "bank" attending to "river" (disambiguation!)
# - Layer 12: [CLS] attending to key content words
```

---

## Exercise (45 min)

### E21.1 BERT vs GPT-2 Sentiment Classification

Compare fine-tuned BERT with GPT-2 zero-shot prompting on SST-2:

```python
from transformers import pipeline

# GPT-2 zero-shot (via text generation + prompting)
gpt2_classifier = pipeline(
    "text-classification",
    model="distilgpt2",  # smaller for speed
    # Note: GPT-2 is not naturally a classifier — this uses
    # it as a feature extractor with a classification head
)

# Alternative: zero-shot prompting
from transformers import GPT2LMHeadModel, GPT2Tokenizer

gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
gpt2_tok = GPT2Tokenizer.from_pretrained("gpt2")

def gpt2_sentiment(text):
    """Zero-shot sentiment via probability of positive/negative continuation."""
    prompt_pos = f'"{text}" The sentiment of this text is positive.'
    prompt_neg = f'"{text}" The sentiment of this text is negative.'

    ids_pos = gpt2_tok.encode(prompt_pos, return_tensors="pt")
    ids_neg = gpt2_tok.encode(prompt_neg, return_tensors="pt")

    with torch.no_grad():
        loss_pos = gpt2(ids_pos, labels=ids_pos).loss
        loss_neg = gpt2(ids_neg, labels=ids_neg).loss

    return "positive" if loss_pos < loss_neg else "negative"

# Test on 100 validation examples
# Compare accuracy: BERT fine-tuned vs GPT-2 zero-shot
```

**Questions:**
1. What accuracy does each achieve?
2. Why is BERT better at this task?
3. At what model scale might GPT surpass fine-tuned BERT?

### E21.2 Masking Strategy Experiment

BERT masks 15% of tokens with the 80/10/10 strategy. What if you change it?

```python
# Try: 100/0/0 (always [MASK])
# Try: 0/0/100 (never replace, just predict)
# Try: 50/50/0 ([MASK] or random, never keep)
# Compare MLM validation loss after 1000 steps
```

### E21.3 [CLS] Token Analysis

Extract [CLS] embeddings for 200 positive and 200 negative SST-2 sentences from the fine-tuned model. Reduce to 2D with t-SNE. Do they cluster by sentiment?

```python
from sklearn.manifold import TSNE
import numpy as np

# Extract [CLS] embeddings
cls_embeddings = []
labels = []
# ... collect from val set ...

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
reduced = tsne.fit_transform(np.array(cls_embeddings))

plt.scatter(reduced[labels==0, 0], reduced[labels==0, 1], alpha=0.5, label='Negative')
plt.scatter(reduced[labels==1, 0], reduced[labels==1, 1], alpha=0.5, label='Positive')
plt.legend()
plt.title('[CLS] Embeddings — Fine-tuned BERT')
plt.show()
```

---

## Key Takeaways

1. **BERT is a bidirectional encoder** — every token sees every other token, giving rich contextual representations
2. **MLM masks 15% of tokens** and predicts them, forcing the model to learn deep language understanding
3. **Fine-tuning adds a task head** on top of pre-trained representations — cheap and effective for NLU tasks
4. **BERT cannot generate text** — no causal structure means no autoregressive sampling
5. **The paradigm shifted:** BERT's pretrain+fine-tune was replaced by GPT's pretrain+prompt at scale, but BERT-style models remain dominant for classification and embedding tasks
6. **[CLS] token** aggregates sequence meaning for classification — a learned summary vector

## Connection to the Thread

This lesson bridges encoder-only (BERT) and decoder-only (GPT) architectures. You've now seen both branches of the transformer family tree. Tomorrow, you'll start Week 4 by diving into tokenization — the preprocessing step that both architectures depend on but that introduces surprising failure modes. Then you'll build GPT from scratch with nanoGPT (Days 23-24), and eventually encounter encoder-decoder models (T5, Day 28) that combine both paradigms. In Phase V, you'll see how BERT-style encoders return for vision (ViT) and how the encoder-decoder pattern enables vision-language models.

## Further Reading

- **[BERT Paper](https://arxiv.org/abs/1810.04805)** — Devlin et al., 2019
- **[RoBERTa](https://arxiv.org/abs/1907.11692)** — Liu et al., 2019 (better training recipe)
- **[DeBERTa](https://arxiv.org/abs/2006.03654)** — He et al., 2021 (disentangled attention)
- **[The Illustrated BERT](https://jalammar.github.io/illustrated-bert/)** — Jay Alammar
- **[BERT Fine-Tuning Tutorial](https://huggingface.co/docs/transformers/training)** — HuggingFace docs
- **[Sentence-BERT](https://arxiv.org/abs/1908.10084)** — Reimers & Gurevych, 2019
