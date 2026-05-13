# Day 36: LLM Evaluation

> **Phase III — LLMs: Training & Alignment | Week 6 | 2.5 hours**
> *"If you can't measure it, you can't improve it." — Peter Drucker (and every ML researcher)*

## Navigation
- **Previous:** [Day 35: LoRA & Efficient Fine-Tuning](../week-05/day-35-lora-finetuning.md)
- **Next:** [Day 37: Quantization & Inference](day-37-quantization-inference.md)
- **Week:** [Week 6 Overview](README.md)
- **Phase:** [Phase III: LLM Engineering](../../study-notes/07-llm-engineering.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 36.1 Why Evaluation Is Hard

LLMs are general-purpose — they can translate, code, reason, chat, write. No single metric captures "how good" a model is. Evaluation must be multi-dimensional.

```
                    ┌─────────────────────────────┐
                    │     LLM Evaluation Space     │
                    ├─────────────────────────────┤
                    │                             │
     Automated      │  Perplexity ← Language      │  Benchmark
     metrics        │  BLEU/ROUGE ← Translation   │  suites
                    │  pass@k ← Code              │
                    │                             │
     Knowledge      │  MMLU ← Academic            │  Human
     benchmarks     │  ARC ← Science reasoning    │  evaluation
                    │  GSM8K ← Math               │
                    │  TruthfulQA ← Factuality    │
                    │                             │
     Open-ended     │  MT-Bench ← Multi-turn      │  Preference
     evaluation     │  AlpacaEval ← Instruction   │  ranking
                    │  Chatbot Arena ← ELO        │
                    └─────────────────────────────┘
```

### 36.2 Perplexity

The most fundamental language model metric — how surprised is the model by the test data?

$$
\text{PPL}(X) = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P_\theta(x_i \mid x_{<i})\right)
$$

**Interpretation:**
- PPL = 1: perfect prediction (knows exactly what comes next)
- PPL = 10: model is "choosing between 10 equally likely tokens" on average
- PPL = 50,000: random guess over vocab (useless model)

```
Typical perplexity values:
  GPT-2 (1.5B) on WikiText: ~22
  GPT-3 (175B) on WikiText: ~10
  Llama-2 (70B) on WikiText: ~4
  
Lower is better, but:
  ⚠️ PPL is dataset-specific — can't compare across different test sets
  ⚠️ Low PPL ≠ useful model (a parrot has low PPL but no reasoning)
```

### 36.3 Knowledge Benchmarks

| Benchmark | Tasks | Format | What It Tests |
|-----------|-------|--------|---------------|
| **MMLU** | 57 subjects | 4-choice MC | Broad knowledge (STEM, humanities, social sciences) |
| **ARC** | Science questions | MC | Grade-school reasoning |
| **HellaSwag** | Sentence completion | MC | Common sense |
| **GSM8K** | 8.5K math problems | Free-form | Multi-step arithmetic reasoning |
| **HumanEval** | 164 coding problems | Code generation | Python programming |
| **TruthfulQA** | 817 questions | MC + open | Resistance to common misconceptions |
| **MATH** | Competition math | Free-form | Advanced mathematical reasoning |
| **WinoGrande** | Pronoun resolution | Binary | Coreference understanding |

**MMLU scoring:**

$$
\text{MMLU} = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \text{Accuracy}_s
$$

where $\mathcal{S}$ is the set of 57 subjects.

### 36.4 LLM-as-Judge

Use a strong model (GPT-4, Claude) to evaluate weaker models:

```
Prompt to judge:
"Rate the following response on a scale of 1-10 for helpfulness,
accuracy, and safety. Explain your reasoning."

Advantages:
  ✅ Scales to any number of samples
  ✅ Captures nuanced quality
  ✅ Cheaper than human evaluation
  
Disadvantages:
  ⚠️ Judge model has its own biases
  ⚠️ Prefers verbose, confident responses
  ⚠️ Struggles with factual verification
  ⚠️ Self-bias: models rate their own outputs higher
```

**MT-Bench** uses GPT-4 as judge on 80 multi-turn questions across 8 categories.

### 36.5 Chatbot Arena & ELO Ratings

The gold standard for open-ended evaluation:

```
1. User submits prompt → two anonymous models respond
2. User picks winner (or declares tie)
3. Update ELO ratings using Bradley-Terry model
4. Thousands of users, millions of votes → robust rankings

ELO update:
  E_A = 1 / (1 + 10^((R_B - R_A) / 400))
  R_A' = R_A + K(S - E_A)
  
  where S = 1 (win), 0.5 (tie), 0 (loss)
```

### 36.6 Benchmark Contamination

**The biggest threat to evaluation:** if training data contains benchmark questions, scores are meaningless.

```
Contamination detection methods:
1. N-gram overlap between training data and benchmark
2. Canary strings inserted in benchmarks
3. Performance gap on seen vs. unseen variants
4. Temporal analysis (performance on questions written after training cutoff)

Example: GPT-4 scores 86% on MMLU, but some questions may have
appeared in its training data. True ability could be lower.
```

---

## Implementation (60 min)

### Build a Mini Evaluation Harness

```python
"""
Day 36 Implementation: LLM evaluation harness.
Implements perplexity, multiple-choice accuracy, and LLM-as-judge.
"""
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class MCQuestion:
    question: str
    choices: list[str]
    correct: int  # 0-indexed


def compute_perplexity(
    model, tokenizer, text: str, stride: int = 512,
) -> float:
    """Compute perplexity of text using sliding window."""
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)
    seq_len = input_ids.size(1)
    max_length = getattr(model.config, "max_position_embeddings", 2048)

    nlls = []
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        target_len = end - begin

        ids = input_ids[:, begin:end]
        target_ids = ids.clone()
        # Only compute loss on the stride portion (avoid double-counting)
        if begin > 0:
            target_ids[:, :-stride] = -100

        with torch.no_grad():
            outputs = model(ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * stride

        nlls.append(neg_log_likelihood)

        if end == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / seq_len)
    return ppl.item()


def evaluate_multiple_choice(
    model, tokenizer, questions: list[MCQuestion],
) -> dict:
    """Evaluate model on multiple-choice questions using log-likelihood."""
    correct = 0

    for q in questions:
        choice_logprobs = []

        for choice in q.choices:
            prompt = f"Question: {q.question}\nAnswer: {choice}"
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            input_ids = input_ids.to(model.device)

            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits[:, :-1, :]
                targets = input_ids[:, 1:]
                logprobs = F.log_softmax(logits, dim=-1)
                token_lps = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

            # Average log-prob (length-normalized)
            avg_lp = token_lps.mean().item()
            choice_logprobs.append(avg_lp)

        predicted = max(range(len(choice_logprobs)), key=lambda i: choice_logprobs[i])
        if predicted == q.correct:
            correct += 1

    accuracy = correct / len(questions) if questions else 0
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": len(questions),
    }


def llm_as_judge_prompt(question: str, response: str) -> str:
    """Generate an LLM-as-judge evaluation prompt."""
    return (
        "You are an expert evaluator. Rate the following response on a "
        "scale of 1-10 for each criterion.\n\n"
        f"## Question\n{question}\n\n"
        f"## Response\n{response}\n\n"
        "## Evaluation Criteria\n"
        "1. **Helpfulness** (1-10): Does it answer the question?\n"
        "2. **Accuracy** (1-10): Is the information correct?\n"
        "3. **Clarity** (1-10): Is it well-organized and clear?\n\n"
        "Output JSON: {\"helpfulness\": X, \"accuracy\": X, \"clarity\": X, "
        "\"reasoning\": \"...\"}"
    )


# --- Demo ---
if __name__ == "__main__":
    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="auto",
    )

    # 1. Perplexity
    test_text = (
        "The transformer architecture uses self-attention to process "
        "sequences in parallel. Each attention head computes queries, "
        "keys, and values from the input representations."
    )
    ppl = compute_perplexity(model, tokenizer, test_text)
    print(f"Perplexity: {ppl:.2f}")

    # 2. Multiple choice
    questions = [
        MCQuestion(
            "What does SLAM stand for in robotics?",
            [
                "Simultaneous Localization and Mapping",
                "Signal Level Analysis Module",
                "System Load Allocation Method",
                "Sequential Learning with Attention Mechanism",
            ],
            correct=0,
        ),
        MCQuestion(
            "Which optimizer is most common for LLM training?",
            ["SGD", "AdamW", "RMSprop", "Adagrad"],
            correct=1,
        ),
    ]
    results = evaluate_multiple_choice(model, tokenizer, questions)
    print(f"MC Accuracy: {results['accuracy']:.1%} "
          f"({results['correct']}/{results['total']})")

    # 3. Judge prompt (for use with GPT-4/Claude)
    prompt = llm_as_judge_prompt(
        "How does LoRA reduce training costs?",
        "LoRA decomposes weight updates into low-rank matrices B and A, "
        "where W' = W0 + BA. Since r << d, this reduces trainable "
        "parameters by 100x while maintaining 95%+ of full fine-tune quality.",
    )
    print(f"\nJudge prompt ({len(prompt)} chars) ready for evaluation.")
```

---

## Exercise (45 min)

### E36.1 — Benchmark Contamination Detector (25 min)
Build a simple contamination checker:
1. Take 10 MMLU questions and compute their 8-gram fingerprints
2. Check if any 8-gram appears in a sample of training data (e.g., C4 subset)
3. Report contamination rate and discuss implications

### E36.2 — ELO Simulation (20 min)
Simulate a Chatbot Arena:
1. Create 5 "models" with different true quality levels
2. Simulate 1000 pairwise comparisons with noise
3. Compute ELO ratings — do they converge to the true ranking?
4. How many comparisons are needed for reliable rankings?

---

## Key Takeaways

1. **No single metric captures LLM quality** — use multiple complementary evaluations
2. **Perplexity** measures language modeling but not usefulness
3. **Benchmarks** (MMLU, HumanEval) test specific skills but are subject to contamination
4. **LLM-as-judge** scales well but inherits the judge model's biases
5. **Chatbot Arena** is the closest to ground truth but requires massive user participation

---

## Connection to the Thread

Evaluation is the unsolved core problem in robotics too. How do you measure if a robot's grasp is "good"? Success rate alone misses nuance — speed, gentleness, generalization. The same multi-dimensional evaluation challenge from LLMs (knowledge × helpfulness × safety) maps to robots (success × efficiency × safety × generalization).

---

## Further Reading

- [MMLU: Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300) — Hendrycks et al., 2021
- [Chatbot Arena: An Open Platform for Evaluating LLMs](https://arxiv.org/abs/2403.04132) — Zheng et al., 2024
- [Judging LLM-as-a-Judge](https://arxiv.org/abs/2306.05685) — MT-Bench paper
- [HumanEval: Evaluating Large Language Models on Code](https://arxiv.org/abs/2107.03374) — Chen et al., 2021
- [Contamination in LLM Benchmarks](https://arxiv.org/abs/2311.01964) — Survey of the problem
