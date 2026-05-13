# Day 31: The Modern LLM Recipe

> **Phase III — LLMs: Training & Alignment | Week 5 | 2.5 hours**
> *"Data is the new code. Curation is the new programming." — Andrej Karpathy*

## Navigation
- **Previous:** [Day 30: Phase II Capstone Day 2](day-30-phase2-capstone-2.md)
- **Next:** [Day 32: Supervised Fine-Tuning](day-32-supervised-finetuning.md)
- **Week:** [Week 5 Overview](README.md)
- **Phase:** [Phase III: LLM Training & Alignment](../../study-notes/06-llm-training-alignment.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 31.1 The 3-Stage Pipeline

Every modern LLM follows the same fundamental recipe:

```
Stage 1: PRETRAINING          Stage 2: SFT                Stage 3: ALIGNMENT
━━━━━━━━━━━━━━━━━━━          ━━━━━━━━━━━                 ━━━━━━━━━━━━━━━
Internet text (TB)            Instruction pairs (10-100K)  Human preferences (50-200K)
Next-token prediction         Input→Output format          RLHF / DPO / KTO
1000s of GPU-hours            10s of GPU-hours             10s of GPU-hours
                                                           
"Learn language"              "Learn to follow"            "Learn values"
Broad knowledge               Task formatting              Safety + helpfulness
                                                           
Loss: cross-entropy           Loss: cross-entropy          Loss: preference-based
on next token                 on response tokens            (reward model or direct)
```

**Why 3 stages?** Each stage optimizes a different objective:

$$
\underbrace{\mathcal{L}_{\text{pretrain}} = -\sum_{t} \log P(x_t | x_{<t})}_{\text{Stage 1: predict next token}}
\quad \rightarrow \quad
\underbrace{\mathcal{L}_{\text{SFT}} = -\sum_{t \in \text{response}} \log P(x_t | x_{<t})}_{\text{Stage 2: predict response only}}
\quad \rightarrow \quad
\underbrace{\mathcal{L}_{\text{align}}}_{\text{Stage 3: preference}}
$$

### 31.2 Stage 1: Pretraining — The Foundation

Pretraining is the most expensive stage (~99% of total compute). The model learns:
- Syntax, grammar, semantics
- World knowledge and facts
- Reasoning patterns
- Code understanding

**Data sources and their properties:**

| Source | Size | Quality | Diversity | Typical % |
|--------|------|---------|-----------|-----------|
| Common Crawl | PB-scale | Low (needs filtering) | Very high | 60-70% |
| Books | ~100GB | High | Medium | 5-10% |
| Wikipedia | ~20GB | Very high | Medium | 3-5% |
| Code (GitHub) | ~300GB | Variable | High | 10-15% |
| Scientific papers | ~100GB | Very high | Low | 3-5% |
| Curated web | Variable | High | High | 5-10% |

### 31.3 Data Quality: The Secret Weapon

The revolution in LLM quality came not from architecture but from **data curation**.

**Deduplication pipeline:**

```
Raw crawl (100TB+)
    │
    ▼
URL dedup (exact match)           → removes ~30%
    │
    ▼
Document-level MinHash dedup     → removes ~40%
    │
    ▼
Paragraph-level dedup            → removes ~10%
    │
    ▼
Quality filtering (perplexity,   → removes ~50%
  classifier, heuristics)
    │
    ▼
Safety filtering                 → removes ~5%
    │
    ▼
Clean corpus (~1-5TB)
```

**Key insight (Llama 2 paper):** Training on 2T tokens of *high-quality* data beats training on 10T tokens of unfiltered data.

### 31.4 Data Mix: The Art of Ratios

The proportion of each data source significantly affects model capabilities:

```python
# Llama 2 approximate data mix
DATA_MIX = {
    "common_crawl": 0.67,    # General knowledge
    "c4":           0.15,    # Cleaned web text
    "github":       0.04,    # Code understanding
    "wikipedia":    0.04,    # Factual knowledge
    "books":        0.04,    # Long-form reasoning
    "arxiv":        0.03,    # Scientific knowledge
    "stackexchange": 0.03,   # Q&A format
}

# Key trade-offs:
# More code    → better reasoning, worse conversation
# More books   → better long-form, slower learning
# More web     → more knowledge, more noise
```

### 31.5 The Chinchilla Connection

Recall from Day 25 — **compute-optimal training** dictates:

$$
N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}
$$

For a given compute budget $C$:
- **Chinchilla approach:** Balance model size and data (20 tokens per parameter)
- **LLaMA approach:** Over-train smaller model on more data (200+ tokens/param) → cheaper inference

```
Chinchilla (70B, 1.4T tokens) ≈ Gopher (280B, 300B tokens)
LLaMA-7B (7B, 1T tokens)     → undertrained by Chinchilla, but fast at inference
LLaMA-2 (7B, 2T tokens)      → even more over-trained, even better at inference
```

---

## Implementation (60 min)

### Build a Mini Data Curation Pipeline

```python
"""
Day 31 Implementation: Data curation pipeline for LLM pretraining.
Demonstrates quality filtering, dedup, and data mixing.
"""
import hashlib
import re
from collections import Counter
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

@dataclass
class Document:
    text: str
    source: str
    url: str = ""
    quality_score: float = 0.0

class QualityFilter:
    """Heuristic quality filter inspired by CCNet and RefinedWeb."""

    def __init__(self, min_words: int = 50, max_words: int = 100_000):
        self.min_words = min_words
        self.max_words = max_words

    def score(self, doc: Document) -> float:
        """Score document quality (0-1). Higher = better."""
        text = doc.text
        words = text.split()
        n_words = len(words)

        if n_words < self.min_words or n_words > self.max_words:
            return 0.0

        score = 1.0

        # Penalty: too many short lines (boilerplate/menus)
        lines = text.split('\n')
        short_lines = sum(1 for l in lines if len(l.split()) < 3)
        if lines:
            score *= max(0.0, 1.0 - short_lines / len(lines))

        # Penalty: excessive repetition
        word_counts = Counter(words)
        if word_counts:
            most_common_frac = word_counts.most_common(1)[0][1] / n_words
            if most_common_frac > 0.1:
                score *= 0.5

        # Penalty: too many special characters
        alpha_chars = sum(c.isalpha() for c in text)
        if len(text) > 0:
            alpha_ratio = alpha_chars / len(text)
            if alpha_ratio < 0.5:
                score *= 0.3

        # Bonus: paragraph structure
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) > 2:
            score *= 1.1

        return min(1.0, score)


class MinHashDeduplicator:
    """Approximate near-duplicate detection using MinHash signatures."""

    def __init__(self, num_hashes: int = 128, ngram_size: int = 5):
        self.num_hashes = num_hashes
        self.ngram_size = ngram_size
        self.seen_signatures: list[tuple] = []

    def _get_ngrams(self, text: str) -> set[str]:
        words = text.lower().split()
        return {' '.join(words[i:i+self.ngram_size])
                for i in range(len(words) - self.ngram_size + 1)}

    def _minhash(self, ngrams: set[str]) -> tuple:
        if not ngrams:
            return tuple([float('inf')] * self.num_hashes)
        signatures = []
        for seed in range(self.num_hashes):
            min_hash = min(
                int(hashlib.md5(f"{seed}_{ng}".encode()).hexdigest(), 16)
                for ng in ngrams
            )
            signatures.append(min_hash)
        return tuple(signatures)

    def _jaccard_from_minhash(self, sig1: tuple, sig2: tuple) -> float:
        return sum(a == b for a, b in zip(sig1, sig2)) / len(sig1)

    def is_duplicate(self, text: str, threshold: float = 0.5) -> bool:
        ngrams = self._get_ngrams(text)
        sig = self._minhash(ngrams)
        for existing_sig in self.seen_signatures:
            if self._jaccard_from_minhash(sig, existing_sig) > threshold:
                return True
        self.seen_signatures.append(sig)
        return False


class DataMixer:
    """Mix data sources according to specified ratios."""

    def __init__(self, mix_ratios: dict[str, float]):
        total = sum(mix_ratios.values())
        self.ratios = {k: v / total for k, v in mix_ratios.items()}

    def create_batch(
        self,
        sources: dict[str, list[Document]],
        batch_size: int = 1000,
    ) -> list[Document]:
        batch = []
        for source, ratio in self.ratios.items():
            n = int(batch_size * ratio)
            docs = sources.get(source, [])
            batch.extend(docs[:n])
        return batch[:batch_size]


# --- Demo pipeline ---
if __name__ == "__main__":
    qf = QualityFilter()
    dedup = MinHashDeduplicator(num_hashes=64, ngram_size=3)

    sample_docs = [
        Document("This is a well-written article about machine learning. " * 20,
                 source="web"),
        Document("Buy now! Click here! Free! " * 50, source="web"),
        Document("This is a well-written article about machine learning. " * 20,
                 source="web"),  # near-duplicate
    ]

    for i, doc in enumerate(sample_docs):
        score = qf.score(doc)
        is_dup = dedup.is_duplicate(doc.text)
        status = "KEEP" if score > 0.3 and not is_dup else "DROP"
        print(f"Doc {i}: quality={score:.2f}, dup={is_dup} → {status}")
```

---

## Exercise (45 min)

### E31.1 — Data Mix Ablation (20 min)
Design an experiment to test how different data mixes affect downstream task performance. Create 3 different mix ratios emphasizing: (a) code, (b) books, (c) web. Predict which mix wins on each benchmark (MMLU, HumanEval, HellaSwag).

### E31.2 — Quality Classifier (25 min)
Extend the `QualityFilter` class with a learned component:
1. Create 20 manually-labeled examples (10 high-quality, 10 low-quality)
2. Extract features: avg word length, vocabulary diversity, sentence count, punctuation ratio
3. Train a simple logistic regression to predict quality
4. Compare with heuristic scoring

---

## Key Takeaways

1. **3-stage pipeline** (pretrain → SFT → alignment) is the universal recipe for modern LLMs
2. **Data quality beats data quantity** — careful curation is the biggest lever
3. **Data mix ratios** directly shape model capabilities and trade-offs
4. **Deduplication** at multiple levels (URL, document, paragraph) is essential
5. **Over-training** smaller models on more data trades compute for cheaper inference (LLaMA strategy)

---

## Connection to the Thread

We've spent Phase II understanding *what* transformers compute (attention, scaling, generation). Now we learn *how to teach them*. The 3-stage pipeline is also exactly how VLAs are trained: pretrain a vision-language model, fine-tune on robot demonstrations, then align with preference data. **Mastering LLM training is mastering robot training.**

---

## Further Reading

- [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) — Meta, 2023
- [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116) — Data curation at scale
- [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) — What to do when data runs out
- [Chinchilla: Training Compute-Optimal LLMs](https://arxiv.org/abs/2203.15556) — DeepMind, 2022
- [FineWeb: 15T Tokens of Quality Web Data](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) — HuggingFace, 2024
