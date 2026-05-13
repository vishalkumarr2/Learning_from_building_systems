# Day 44: Phase III Capstone Day 3 + Checkpoint

> **Phase III — LLMs: Training & Alignment | Week 7 | 2.5 hours**
> *"Before moving forward, prove you understand what's behind you."*

## Navigation
- **Previous:** [Day 43: Phase III Capstone Day 2](day-43-phase3-capstone-2.md)
- **Next:** [Day 45: ViT — Image as Tokens](day-45-vit.md)
- **Week:** [Week 7 Overview](README.md)
- **Phase:** [Phase III: LLM Training & Alignment](../../study-notes/06-llm-training-alignment.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Capstone Deliverable (90 min)

### Final Report Structure

Compile your Phase III capstone into a structured deliverable:

```
Phase III Capstone: Robotics Assistant
======================================

1. Architecture Overview
   - System diagram (LLM + LoRA + RAG + tools)
   - Component responsibilities and data flow
   - Design decisions and trade-offs

2. Training Summary
   - Dataset: N examples, M categories
   - LoRA config: r=?, α=?, target modules
   - Training: epochs, lr, final loss
   - Trainable params: X / Y total (Z%)

3. RAG Configuration
   - Knowledge base: N documents, chunking strategy
   - Embedding model and dimensions
   - Retrieval: top-k=?, similarity threshold
   - Vector store implementation

4. Evaluation Results
   ┌──────────────────┬──────────┬──────────┬──────────┐
   │ Metric           │ Base     │ LoRA     │ LoRA+RAG │
   ├──────────────────┼──────────┼──────────┼──────────┤
   │ Knowledge recall │          │          │          │
   │ Diagnosis recall │          │          │          │
   │ Command accuracy │          │          │          │
   │ Hallucination %  │          │          │          │
   │ Avg latency (ms) │          │          │          │
   └──────────────────┴──────────┴──────────┴──────────┘

5. Error Analysis
   - Top failure modes with examples
   - Which component (LoRA vs RAG) addresses each failure
   - Remaining gaps and proposed solutions

6. Lessons Learned
   - What surprised you?
   - What would you do differently?
   - How does this connect to VLA training?
```

### Implementation: Generate the Report

```python
"""
Day 44 Capstone: Generate final report and deliverable.
"""
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CapstoneReport:
    title: str = "Phase III Capstone: Robotics Assistant"
    date: str = ""
    architecture: str = ""
    training_summary: dict = None
    rag_config: dict = None
    eval_results: dict = None
    error_analysis: list = None
    lessons: list = None

    def __post_init__(self):
        self.date = datetime.now().strftime("%Y-%m-%d")
        if self.training_summary is None:
            self.training_summary = {}
        if self.rag_config is None:
            self.rag_config = {}
        if self.eval_results is None:
            self.eval_results = {}
        if self.error_analysis is None:
            self.error_analysis = []
        if self.lessons is None:
            self.lessons = []

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            f"*Generated: {self.date}*\n",
            "## 1. Architecture",
            self.architecture or "*[Fill in system diagram]*\n",
            "## 2. Training Summary",
        ]

        for key, value in self.training_summary.items():
            lines.append(f"- **{key}:** {value}")

        lines.append("\n## 3. RAG Configuration")
        for key, value in self.rag_config.items():
            lines.append(f"- **{key}:** {value}")

        lines.append("\n## 4. Evaluation Results")
        if self.eval_results:
            configs = list(self.eval_results.keys())
            metrics = set()
            for config_scores in self.eval_results.values():
                metrics.update(config_scores.keys())

            header = "| Metric | " + " | ".join(configs) + " |"
            sep = "|" + "---|" * (len(configs) + 1)
            lines.extend([header, sep])
            for metric in sorted(metrics):
                row = f"| {metric} |"
                for config in configs:
                    val = self.eval_results[config].get(metric, "—")
                    if isinstance(val, float):
                        row += f" {val:.1%} |"
                    else:
                        row += f" {val} |"
                lines.append(row)

        lines.append("\n## 5. Error Analysis")
        for i, err in enumerate(self.error_analysis, 1):
            lines.append(f"{i}. {err}")

        lines.append("\n## 6. Lessons Learned")
        for lesson in self.lessons:
            lines.append(f"- {lesson}")

        return "\n".join(lines)


# Example report
if __name__ == "__main__":
    report = CapstoneReport(
        architecture="LLM (TinyLlama 1.1B) + LoRA adapter (r=16) + "
                     "TF-IDF RAG over 5 technical documents + "
                     "rule-based command parser with safety validator.",
        training_summary={
            "Base model": "TinyLlama 1.1B Chat",
            "Dataset": "6 robotics instruction pairs",
            "LoRA config": "r=16, α=32, target=q/k/v/o_proj",
            "Training": "3 epochs, lr=2e-4, cosine schedule",
            "Trainable params": "~4M / 1.1B (0.36%)",
        },
        rag_config={
            "Documents": "5 technical spec documents",
            "Chunking": "Full document (small docs)",
            "Embedding": "TF-IDF bag-of-words",
            "Retrieval": "Top-3, cosine similarity",
        },
        eval_results={
            "Base": {"knowledge": 0.30, "diagnosis": 0.20, "command": 0.50},
            "LoRA": {"knowledge": 0.55, "diagnosis": 0.50, "command": 0.70},
            "LoRA+RAG": {"knowledge": 0.80, "diagnosis": 0.65, "command": 0.75},
        },
        error_analysis=[
            "Reasoning questions remain weak across all configs — need CoT",
            "RAG retrieval misses when question phrasing differs from docs",
            "Command parser fails on ambiguous multi-step instructions",
        ],
        lessons=[
            "Data quality >> quantity for SFT",
            "RAG fixes knowledge gaps that fine-tuning can't address cheaply",
            "Safety validation layer is non-negotiable for robotics",
            "Evaluation design is as important as model training",
        ],
    )
    print(report.to_markdown())
```

---

## Phase III Checkpoint (60 min)

Answer each question in 3-5 sentences with equations or code where appropriate. **Score yourself honestly:** each question is worth 1 point, minimum 4/6 to proceed.

### Checkpoint Question 1: The 3-Stage Pipeline

**Describe the 3-stage modern LLM training pipeline. For each stage, state: (a) the training objective, (b) the data type and typical size, (c) what capability it provides.**

<details>
<summary>Expected Answer</summary>

**Stage 1 — Pretraining:** Next-token prediction ($\mathcal{L} = -\sum_t \log P(x_t | x_{<t})$) on internet-scale text (1-15T tokens). Provides broad language understanding, world knowledge, and reasoning patterns.

**Stage 2 — SFT:** Cross-entropy on response tokens only ($\mathcal{L}_{\text{SFT}} = -\sum_{t \in \text{response}} \log P(x_t | x_{<t})$) on instruction-response pairs (10K-1M examples). Teaches the model to follow instructions and use the right output format.

**Stage 3 — Alignment:** Preference-based loss (RLHF reward + KL penalty, or DPO loss) on human preference data (50-200K comparisons). Teaches the model human values — helpfulness, harmlessness, honesty.

</details>

---

### Checkpoint Question 2: LoRA Equation

**Write the LoRA weight update equation. Explain each term, state typical values for rank $r$ and scaling $\alpha$, and calculate the parameter savings for a 4096×4096 weight matrix with $r=16$.**

<details>
<summary>Expected Answer</summary>

$$W' = W_0 + \frac{\alpha}{r} \cdot BA$$

- $W_0 \in \mathbb{R}^{d \times k}$: frozen pretrained weights
- $B \in \mathbb{R}^{d \times r}$: down-projection (initialized to zero)
- $A \in \mathbb{R}^{r \times k}$: up-projection (initialized randomly)
- $\alpha$: scaling factor (typically $\alpha = 2r$, e.g., 32)
- $r$: rank (typically 8-64, sweet spot around 16)

Parameter savings for $d=k=4096$, $r=16$:
- Full: $4096 \times 4096 = 16,777,216$ parameters
- LoRA: $4096 \times 16 + 16 \times 4096 = 131,072$ parameters
- Savings: $\frac{16.8M - 131K}{16.8M} = 99.2\%$ reduction

</details>

---

### Checkpoint Question 3: DPO vs RLHF

**Compare DPO and RLHF. Write the DPO loss function, explain why it doesn't need a reward model, and state when you would choose RLHF over DPO.**

<details>
<summary>Expected Answer</summary>

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

DPO derives from the RLHF objective by solving for the optimal policy in closed form. The reward is implicitly represented as the log-ratio between policy and reference model — so no explicit reward model is needed.

Choose **RLHF over DPO** when: (1) you need a reusable reward model for multiple policies, (2) you want to filter/reject generations at inference time using the reward model, (3) you're doing online RLHF where the model generates new data during training.

</details>

---

### Checkpoint Question 4: In-Context Learning & Compression

**Explain how in-context learning works. Why do even random labels help? How does ICL relate to data compression?**

<details>
<summary>Expected Answer</summary>

ICL lets LLMs learn new tasks from examples in the prompt without parameter updates. The model implements something like implicit Bayesian inference: $P(y|x, \mathcal{D}) = \sum_c P(y|x,c) \cdot P(c|\mathcal{D})$, where the examples $\mathcal{D}$ sharpen the posterior over latent concepts $c$.

Random labels still help because examples provide **format specification** (input→output structure) and **input distribution** information, even without correct mappings. Min et al. (2022) showed label correctness adds only ~7% accuracy on top of the ~13% boost from format alone.

**Compression connection:** ICL examples compress the task description. Without examples, "classify sentiment" is ambiguous (many bits needed). With 3 labeled examples, the task is unambiguous (few bits). ICL = providing a compressed program specification.

</details>

---

### Checkpoint Question 5: Quantization

**Explain quantization from FP16 to INT4. What is the absmax quantization formula? What is GPTQ's key innovation? Why does AWQ outperform GPTQ?**

<details>
<summary>Expected Answer</summary>

Quantization reduces weight precision: FP16 (2 bytes) → INT4 (0.5 bytes) = 4× compression.

Absmax: $x_q = \text{round}\left(\frac{x}{\max(|x|)} \cdot 7\right)$, dequantize: $\hat{x} = x_q \cdot \frac{\max(|x|)}{7}$

**GPTQ** uses second-order information (Hessian of the reconstruction error) to quantize weights column-by-column. After quantizing each column, it adjusts remaining columns to compensate for the error — a form of optimal brain damage.

**AWQ** outperforms GPTQ because it identifies the ~1% of weights that correspond to large activations and protects them by scaling up before quantization (then scaling down the activations). This preserves the most important weight values while allowing aggressive quantization of less important ones.

</details>

---

### Checkpoint Question 6: Speculative Decoding

**Explain speculative decoding. Why does it produce outputs identical to the large model? Under what conditions does it not provide speedup?**

<details>
<summary>Expected Answer</summary>

Speculative decoding uses a small "draft" model to generate K candidate tokens quickly, then the large model verifies all K tokens in a single forward pass. Accepted tokens are kept; at the first rejection, generation continues from that point using the large model's distribution.

**Identical outputs** because verification uses rejection sampling: a draft token is accepted with probability $\min(1, P_{\text{large}}(t) / P_{\text{draft}}(t))$. Rejected tokens are resampled from an adjusted distribution. This guarantees the output distribution is exactly the large model's distribution.

**No speedup when:** (1) the draft model is too different from the large model (most tokens rejected → overhead of draft + verify > just running the large model), (2) K is too large (verification cost grows), (3) the task has high entropy (many valid continuations → hard for draft model to guess right).

</details>

---

## Scoring

| Score | Assessment | Action |
|-------|-----------|--------|
| 6/6 | Excellent — ready for Phase IV | Proceed to Vision |
| 5/6 | Strong — minor gaps | Review the weak topic, then proceed |
| 4/6 | Adequate — some gaps | Spend 30 min reviewing weak areas, then proceed |
| 3/6 or below | Needs review | Re-read Days 31-42, redo exercises before proceeding |

---

## Key Takeaways

1. **Phase III mastery** requires understanding *why* each stage exists, not just *how* it works
2. **The equations matter** — LoRA, DPO, and quantization formulas reveal the core insights
3. **Trade-offs are everywhere** — DPO vs RLHF, full vs LoRA, RAG vs fine-tuning
4. **Everything connects to robotics** — every LLM technique has a direct analog in robot learning

---

## Connection to the Thread

Phase III taught us to teach LLMs. Phase IV will teach us to give them eyes. Vision Transformers (ViT) take the same architecture we've mastered — attention, transformers, scaling — and apply it to images. The key insight: an image is just a sequence of patches, exactly like a sentence is a sequence of tokens. **Same architecture, different modality.** This is the path to VLAs.

---

## What's Next

**Day 45: ViT — Image as Tokens** begins Phase IV: Vision. We'll learn how to split images into patches, embed them as tokens, and process them through the same transformer architecture we've been studying. The multimodal journey begins.
