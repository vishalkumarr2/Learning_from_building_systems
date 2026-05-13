# Day 6: Embeddings & Representation Learning

> **Phase I — DL Foundations & Information Theory | Week 1 | 2.5 hours**
> *"You shall know a word by the company it keeps." — J.R. Firth (1957)*

## Navigation
- **Previous:** [Day 5: Information Theory & Compression](day-05-information-theory-compression.md)
- **Next:** [Day 7: Training Stability Cookbook](day-07-training-stability.md)
- **Week:** [Week 1: DL Foundations](../README.md)
- **Phase:** [Phase I: DL Foundations](../../study-notes/01-dl-foundations.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 6.1 The Problem with One-Hot Encoding

The naive way to represent tokens: assign each a unique index and encode it as a
one-hot vector.

For vocabulary size $V = 50{,}000$:

$$\text{"cat"} \rightarrow [0, 0, \ldots, 0, \underbrace{1}_{\text{index 4217}}, 0, \ldots, 0] \in \mathbb{R}^{50000}$$

**Three fatal problems:**

| Problem | Explanation |
|---|---|
| **Dimensionality** | Each vector has 50,000 dimensions — wasteful |
| **No similarity** | $\cos(\text{one-hot}(\text{"cat"}), \text{one-hot}(\text{"dog"})) = 0$ — all words are equidistant |
| **No generalization** | Learning something about "cat" teaches nothing about "dog" |

```
 One-hot space (V = 5):

   "cat"   → [1, 0, 0, 0, 0]    All vectors are orthogonal.
   "dog"   → [0, 1, 0, 0, 0]    distance(cat, dog) = distance(cat, quantum) = √2
   "kitten"→ [0, 0, 1, 0, 0]    No structure. No similarity. No transfer.
   "puppy" → [0, 0, 0, 1, 0]
   "quantum"→ [0, 0, 0, 0, 1]
```

### 6.2 Dense Embeddings — Distributed Representations

The solution: map each token to a **dense, low-dimensional** vector where geometry
encodes meaning.

$$e: \mathcal{V} \rightarrow \mathbb{R}^d \quad \text{where } d \ll V$$

Typically $d \in \{64, 128, 256, 512, 768, 1024\}$.

```
 Embedding space (d = 2, simplified):

              "kitten"
                 •
                  \
                   "cat"
                    •
                                  "quantum"
                                     •
                 •
               "puppy"
                 \
                  "dog"
                   •

   Similar words → nearby vectors
   Dissimilar words → distant vectors
```

**Key insight:** In this space, similarity is **geometric**:

$$\text{sim}(\text{"cat"}, \text{"kitten"}) = \cos(\mathbf{e}_\text{cat}, \mathbf{e}_\text{kitten}) \approx 0.9$$

The embedding is a **lookup table** — a matrix $E \in \mathbb{R}^{V \times d}$ where
row $i$ is the embedding for token $i$:

$$\mathbf{e}_w = E[w] = E^\top \cdot \text{one-hot}(w)$$

In PyTorch, `nn.Embedding(V, d)` is exactly this matrix. The one-hot multiplication
is optimized away into an index lookup.

### 6.3 Word2Vec — Learning Embeddings from Context

Mikolov et al. (2013) showed that useful embeddings can be learned from raw text
with a simple objective: **predict context from a word** (or vice versa).

**Skip-gram:** Given a center word, predict surrounding context words.

$$\max \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} | w_t)$$

where $c$ is the context window size and:

$$P(w_o | w_i) = \frac{\exp(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_i})}{\sum_{w=1}^{V} \exp(\mathbf{u}_w^\top \mathbf{v}_{w_i})}$$

Here $\mathbf{v}_w$ is the "input" embedding and $\mathbf{u}_w$ is the "output"
embedding. In practice, **negative sampling** avoids the expensive softmax over $V$:

$$\mathcal{L} = \log \sigma(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_i}) + \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n} [\log \sigma(-\mathbf{u}_{w_k}^\top \mathbf{v}_{w_i})]$$

**CBOW (Continuous Bag of Words):** The reverse — predict the center word from its
context. Faster to train, slightly worse on rare words.

```
 Skip-gram:                          CBOW:

 Context ← Center → Context         Context → Center ← Context

 "the"  ← "cat" → "sat"             "the" + "sat" → "cat"
 "cat"  ← "sat" → "on"              "cat" + "on"  → "sat"
```

### 6.4 The Magic of Linear Structure — Analogies

The most famous result: trained Word2Vec embeddings exhibit **linear algebraic
structure**:

$$\mathbf{v}_\text{king} - \mathbf{v}_\text{man} + \mathbf{v}_\text{woman} \approx \mathbf{v}_\text{queen}$$

This works because the "gender direction" is a consistent vector offset in embedding
space. Other analogies:

| Relationship | Equation |
|---|---|
| Gender | king − man + woman ≈ queen |
| Country-capital | Paris − France + Japan ≈ Tokyo |
| Tense | walking − walk + swim ≈ swimming |
| Comparative | bigger − big + small ≈ smaller |

**Why does this happen?** The training objective forces co-occurrence statistics into
the geometry. Words used in similar contexts get similar vectors. Systematic
relationships become linear directions because the model is linear (dot product).

### 6.5 Subword Embeddings (Preview)

A vocabulary of whole words has a problem: what about "unbreakable", "COVID-19",
or "transformerification"? Out-of-vocabulary (OOV) words get no embedding.

**Solution:** Break words into subword tokens.

| Method | How it works | Example |
|---|---|---|
| **BPE** (Byte-Pair Encoding) | Merge most frequent character pairs iteratively | "un" + "break" + "able" |
| **WordPiece** | Merge pairs that maximize likelihood | "un" + "##break" + "##able" |
| **SentencePiece** | Treat input as raw bytes, no pre-tokenization | "▁un" + "break" + "able" |

We will dive deep into BPE tokenization on Day 22 (Phase II). For now, know that
modern transformers do NOT embed whole words — they embed **subword tokens**.

### 6.6 Embeddings in Transformers and Beyond

In a transformer, `nn.Embedding` is the **first layer**. But the concept extends far
beyond text:

```
 ┌──────────────────────────────────────────────────────────┐
 │              WHAT GETS EMBEDDED WHERE                     │
 │                                                          │
 │  Text tokens   ──▶ nn.Embedding(vocab, d)     (lookup)  │
 │  Image patches ──▶ nn.Conv2d(3, d, 16, 16)    (linear)  │
 │  Audio frames  ──▶ nn.Linear(mel_bins, d)      (linear)  │
 │  Action tokens ──▶ nn.Embedding(num_actions, d) (lookup) │
 │  Positions     ──▶ nn.Embedding(max_len, d)    (lookup)  │
 │                                                          │
 │  ALL produce vectors in R^d — the same space!            │
 └──────────────────────────────────────────────────────────┘
```

**VLAs** embed text tokens, image patches, AND action tokens into a **shared** space.
The transformer then processes them uniformly. This "everything is a token" paradigm
is the foundation of multimodal AI (Phase III+).

**Image patch embeddings (ViT preview):**

A 224×224 image is split into 16×16 patches (= 196 patches). Each patch is flattened
to a vector of $16 \times 16 \times 3 = 768$ values, then linearly projected to $d$:

$$\mathbf{z}_i = W_p \cdot \text{flatten}(\text{patch}_i) + \mathbf{b}_p$$

This is identical in spirit to word embeddings — a linear map from raw representation
to a shared latent space.

---

## Implementation (60 min)

### 6.7 Hands-On: Train Word2Vec from Scratch

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import Counter
import random

# ── 6.7.1 Prepare a small corpus ────────────────────────────
corpus = """
the king ruled the kingdom with wisdom
the queen ruled the kingdom with grace
the man worked in the field every day
the woman worked in the market every day
the prince was the son of the king
the princess was the daughter of the queen
the boy played in the field with joy
the girl played in the garden with joy
a cat sat on the mat near the door
a dog sat on the rug near the window
the king and queen had a royal feast
the man and woman walked to the town
""".strip().lower().split('\n')

# Tokenize
sentences = [line.split() for line in corpus]
all_words = [w for s in sentences for w in s]
vocab = sorted(set(all_words))
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
V = len(vocab)
print(f"Vocabulary size: {V}")

# ── 6.7.2 Generate skip-gram pairs ──────────────────────────
def get_skipgram_pairs(sentences, window=2):
    pairs = []
    for sentence in sentences:
        indices = [word2idx[w] for w in sentence]
        for i, center in enumerate(indices):
            for j in range(max(0, i - window), min(len(indices), i + window + 1)):
                if j != i:
                    pairs.append((center, indices[j]))
    return pairs

pairs = get_skipgram_pairs(sentences, window=2)
print(f"Training pairs: {len(pairs)}")

# ── 6.7.3 Skip-gram model with negative sampling ────────────
EMBED_DIM = 32
NUM_NEG = 5

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)
        # Initialize small
        nn.init.uniform_(self.center_embeddings.weight, -0.5/embed_dim, 0.5/embed_dim)
        nn.init.zeros_(self.context_embeddings.weight)

    def forward(self, center, context, neg_samples):
        # center: (batch,), context: (batch,), neg_samples: (batch, num_neg)
        c = self.center_embeddings(center)         # (batch, d)
        o = self.context_embeddings(context)       # (batch, d)
        n = self.context_embeddings(neg_samples)   # (batch, num_neg, d)

        # Positive score
        pos_score = torch.sum(c * o, dim=1)                    # (batch,)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-10)

        # Negative scores
        neg_score = torch.bmm(n, c.unsqueeze(2)).squeeze(2)    # (batch, num_neg)
        neg_loss = -torch.log(torch.sigmoid(-neg_score) + 1e-10).sum(dim=1)

        return (pos_loss + neg_loss).mean()

model = SkipGram(V, EMBED_DIM)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Word frequency for negative sampling distribution
word_freq = Counter(all_words)
freq_array = np.array([word_freq[idx2word[i]] for i in range(V)], dtype=np.float64)
freq_array = freq_array ** 0.75   # Smooth distribution (Mikolov's trick)
freq_array /= freq_array.sum()

# ── 6.7.4 Train ─────────────────────────────────────────────
EPOCHS = 100
BATCH_SIZE = 64

for epoch in range(EPOCHS):
    random.shuffle(pairs)
    total_loss = 0
    num_batches = 0

    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i:i + BATCH_SIZE]
        centers = torch.tensor([p[0] for p in batch])
        contexts = torch.tensor([p[1] for p in batch])
        neg = torch.tensor(
            np.random.choice(V, size=(len(batch), NUM_NEG), p=freq_array)
        )

        loss = model(centers, contexts, neg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/num_batches:.4f}")

# ── 6.7.5 Explore the learned embeddings ────────────────────
def most_similar(word, topk=5):
    """Find the most similar words by cosine similarity."""
    if word not in word2idx:
        print(f"'{word}' not in vocabulary")
        return
    idx = word2idx[word]
    emb = model.center_embeddings.weight.data
    word_vec = emb[idx]

    # Cosine similarity with all words
    sims = torch.cosine_similarity(word_vec.unsqueeze(0), emb)
    sims[idx] = -1  # Exclude self
    top_indices = sims.argsort(descending=True)[:topk]

    print(f"\nMost similar to '{word}':")
    for i in top_indices:
        print(f"  {idx2word[i.item()]:12s}  cos={sims[i]:.3f}")

most_similar("king")
most_similar("woman")
most_similar("field")

# ── 6.7.6 Analogy test ──────────────────────────────────────
def analogy(a, b, c, topk=3):
    """a is to b as c is to ???"""
    emb = model.center_embeddings.weight.data
    vec = emb[word2idx[b]] - emb[word2idx[a]] + emb[word2idx[c]]

    sims = torch.cosine_similarity(vec.unsqueeze(0), emb)
    for w in [a, b, c]:
        sims[word2idx[w]] = -1  # Exclude input words

    top_indices = sims.argsort(descending=True)[:topk]
    print(f"\n'{a}' is to '{b}' as '{c}' is to:")
    for i in top_indices:
        print(f"  {idx2word[i.item()]:12s}  cos={sims[i]:.3f}")

analogy("king", "queen", "man")     # Expect: woman
analogy("king", "prince", "queen")  # Expect: princess
```

### 6.8 Visualize with t-SNE

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

embeddings = model.center_embeddings.weight.data.numpy()
tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, V-1))
coords = tsne.fit_transform(embeddings)

plt.figure(figsize=(12, 8))
for i, word in idx2word.items():
    plt.scatter(coords[i, 0], coords[i, 1], c='steelblue', s=40)
    plt.annotate(word, (coords[i, 0] + 0.3, coords[i, 1] + 0.3), fontsize=9)

plt.title("Word2Vec Embeddings (t-SNE projection)")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.tight_layout()
plt.savefig("word2vec_tsne.png", dpi=150)
plt.show()
print("Saved: word2vec_tsne.png")
```

### 6.9 Embedding Layer Parameter Count

```python
# ── How many parameters in an embedding layer? ───────────────
vocab_size = 50_000
embed_dim = 512

embedding = nn.Embedding(vocab_size, embed_dim)
num_params = sum(p.numel() for p in embedding.parameters())
print(f"\nnn.Embedding({vocab_size}, {embed_dim})")
print(f"  Parameters: {num_params:,} = {num_params/1e6:.1f}M")

# Compare with a 3-layer MLP
mlp = nn.Sequential(
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
)
mlp_params = sum(p.numel() for p in mlp.parameters())
print(f"\n3-layer MLP (512→512→512→512)")
print(f"  Parameters: {mlp_params:,} = {mlp_params/1e6:.1f}M")
print(f"\nEmbedding has {num_params/mlp_params:.1f}x more parameters than the MLP!")
print("Embeddings are the LARGEST component of small language models.")
```

---

## Exercise (45 min)

1. **Parameter calculation:** An embedding layer for vocabulary $V = 32{,}000$ with
   $d = 768$ has how many parameters? Express in millions. Now consider GPT-2's full
   model at 117M parameters — what percentage is the embedding layer alone?

2. **Cosine vs. Euclidean:** For the Word2Vec model you trained, compute both cosine
   similarity and Euclidean distance for 5 word pairs. Do they rank pairs in the same
   order? Why might cosine be preferred? (Hint: think about vector magnitude vs.
   direction.)

3. **Embedding arithmetic:** Beyond the standard analogies, try to find 3 novel
   relationships in your trained embeddings. Not all will work on a small corpus —
   explain why the corpus size matters for embedding quality.

4. **Image patch embeddings:** A ViT processes 224×224 RGB images with 16×16 patches,
   projecting to $d = 768$.
   - How many patches are there?
   - What is the "raw" dimension of each patch?
   - How many parameters does the patch embedding layer (a linear projection) have?
   - How does this compare to a text embedding layer with $V = 32{,}000$?

---

## Key Takeaways

- **One-hot** vectors are high-dimensional, sparse, and encode no similarity
- **Dense embeddings** map tokens to $\mathbb{R}^d$ where geometry = meaning
- **Word2Vec** learns embeddings by predicting context (skip-gram) or center (CBOW)
- **Linear structure** emerges: king − man + woman ≈ queen
- **`nn.Embedding`** is a learnable lookup table — and often the largest layer
- **Everything is embeddable:** text tokens, image patches, action tokens, positions
- **The "everything is a token" paradigm** unifies modalities in shared embedding spaces

## Connection to the Thread

Embeddings connect directly to the compression thread:

- **One-hot to dense** is a form of **dimensionality compression**: 50,000 dims → 512
  dims, with structure preserved
- **Word2Vec's training objective** is implicitly a compression objective: predict
  context = compress co-occurrence statistics into dense vectors
- **Shared embedding spaces** for VLAs compress heterogeneous inputs (text, vision,
  action) into a common representation — the transformer doesn't care where the
  vectors came from, only their geometry
- **Better embeddings = better compression = more useful representations**

Tomorrow (Day 7) we address a practical prerequisite: how to actually train these
models without them exploding or collapsing.

## Further Reading

- Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in
  Vector Space." *ICLR Workshop 2013.* — The original Word2Vec paper.
- Mikolov, T., et al. (2013). "Distributed Representations of Words and Phrases and
  their Compositionality." *NeurIPS 2013.* — Negative sampling, phrase vectors.
- Pennington, J., Socher, R., & Manning, C. D. (2014). "GloVe: Global Vectors for
  Word Representation." *EMNLP 2014.* — Matrix factorization perspective.
- Dosovitskiy, A., et al. (2021). "An Image is Worth 16x16 Words." *ICLR 2021.* —
  ViT patch embeddings.
- Sennrich, R., Haddow, B., & Birch, A. (2016). "Neural Machine Translation of Rare
  Words with Subword Units." *ACL 2016.* — BPE for NMT.
