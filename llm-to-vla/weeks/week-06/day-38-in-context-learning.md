# Day 38: In-Context Learning

> **Phase III — LLMs: Training & Alignment | Week 6 | 2.5 hours**
> *"The most surprising capability of LLMs: learning from examples in the prompt without any gradient updates." — Sewon Min*

## Navigation
- **Previous:** [Day 37: Quantization & Inference](day-37-quantization-inference.md)
- **Next:** [Day 39: Long Context & Reasoning](day-39-long-context-reasoning.md)
- **Week:** [Week 6 Overview](README.md)
- **Phase:** [Phase III: LLM Engineering](../../study-notes/07-llm-engineering.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 38.1 What Is In-Context Learning?

In-context learning (ICL) is the ability to learn new tasks from a few examples provided in the prompt — with **no parameter updates**.

```
Zero-shot (no examples):
  "Translate English to French: Hello → "
  
One-shot (1 example):
  "Translate English to French:
   Good morning → Bonjour
   Hello → "
   
Few-shot (2-5 examples):
  "Translate English to French:
   Good morning → Bonjour
   Thank you → Merci
   Goodbye → Au revoir
   Hello → "
```

**Why this is remarkable:** The model was never explicitly trained to "learn from examples." It emerged from scale — GPT-2 (1.5B) barely does ICL, GPT-3 (175B) does it reliably.

### 38.2 ICL as Implicit Bayesian Inference

One theory: ICL implements implicit Bayesian inference over a latent concept space.

$$
P(y \mid x, \mathcal{D}_{\text{context}}) = \sum_c P(y \mid x, c) \cdot P(c \mid \mathcal{D}_{\text{context}})
$$

where $c$ is a latent "concept" (e.g., "translation task") and $\mathcal{D}_{\text{context}}$ are the few-shot examples.

```
Before examples:    P(c | ∅) is spread over many concepts
                    (translation? sentiment? QA? code?)

After 3 examples:   P(c | D) concentrates on "translation"
                    → model behaves as if it "knows" the task
```

### 38.3 ICL as Mesa-Optimization

**Mesa-optimization** hypothesis: during pretraining, the model learns an internal optimizer that runs at inference time.

```
Pretraining loss:
  L = E[−log P(next token | context)]
  
The "context" during pretraining naturally contains patterns like:
  "In this document, X means Y. Therefore, Z means..."
  
The model learns to:
  1. Extract the pattern from context (like gradient descent on examples)
  2. Apply the pattern to new inputs (like inference with learned weights)
  
→ Attention layers implement a form of gradient descent!
```

**Formal connection (Akyürek et al., 2023):**

For linear regression, transformer attention computes:

$$
\hat{y} = x^T \hat{w}, \quad \text{where } \hat{w} = (X^T X)^{-1} X^T Y
$$

This is exactly the ordinary least squares solution — the transformer learns to do regression *in its forward pass*.

### 38.4 ICL as Compression

**Connection to Kolmogorov complexity:** ICL works because the examples compress the description of the task.

```
Without examples:
  "Do the classification task" → ambiguous, many possible tasks
  Description length: high → many bits needed
  
With 3 examples:
  "positive: 'great movie' | negative: 'terrible film' | positive: 'loved it'"
  → unambiguous, exactly one task
  Description length: low → few bits suffice
  
ICL = providing a compressed program specification via examples
```

### 38.5 What Makes ICL Work (and Fail)

**What matters:**

| Factor | Impact | Evidence |
|--------|--------|----------|
| Label space | Critical | Providing the right labels matters more than correct pairings |
| Input distribution | High | Examples should match the test distribution |
| Format consistency | High | Consistent formatting → better performance |
| Label correctness | Moderate | Random labels still help! (shows the format) |
| Example ordering | Moderate | Order affects results (recency bias) |
| Number of examples | Diminishing returns | 4-8 often sufficient, more rarely helps |

**Surprising finding (Min et al., 2022):** ICL works even with *random labels*! The examples provide format and distribution information, not just input-output mapping.

```
Correct labels:     "great movie" → positive    accuracy: 89%
Random labels:      "great movie" → negative    accuracy: 82%
Zero-shot:                                      accuracy: 69%

→ Much of ICL's benefit comes from format, not label correctness
→ But correct labels still provide meaningful lift
```

### 38.6 Prompt Engineering Principles

```
1. Format consistency:
   ✅ "Input: X\nOutput: Y"  (always the same format)
   ❌ "X → Y" then "X: Y"    (inconsistent)

2. Representative examples:
   ✅ Cover the diversity of expected inputs
   ❌ All examples from one narrow distribution

3. Order matters:
   ✅ Similar example last (recency bias helps)
   ❌ Random ordering

4. Label balance:
   ✅ Equal representation of each class
   ❌ All examples of one class

5. Instruction clarity:
   ✅ "Classify the sentiment as 'positive' or 'negative'."
   ❌ "Do the thing."
```

---

## Implementation (60 min)

### ICL Experiments: Testing the Theory

```python
"""
Day 38 Implementation: In-context learning experiments.
Test how example count, label correctness, and ordering affect ICL.
"""
import random
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

@dataclass
class ICLExample:
    text: str
    label: str


# Sentiment classification dataset
EXAMPLES = [
    ICLExample("This movie was absolutely fantastic!", "positive"),
    ICLExample("Terrible waste of time, awful acting.", "negative"),
    ICLExample("I loved every minute of this film.", "positive"),
    ICLExample("Boring, predictable, and poorly written.", "negative"),
    ICLExample("A masterpiece of modern cinema.", "positive"),
    ICLExample("I couldn't even finish watching it.", "negative"),
    ICLExample("Brilliant performances all around.", "positive"),
    ICLExample("The worst movie I've seen this year.", "negative"),
]

TEST_CASES = [
    ICLExample("An incredible journey that moved me to tears.", "positive"),
    ICLExample("Disappointing sequel that misses the mark.", "negative"),
    ICLExample("Charming and heartwarming from start to finish.", "positive"),
    ICLExample("Dull and unimaginative, skip this one.", "negative"),
]


def build_icl_prompt(
    examples: list[ICLExample],
    test_input: str,
    randomize_labels: bool = False,
) -> str:
    """Build an ICL prompt for sentiment classification."""
    prompt = "Classify the sentiment as 'positive' or 'negative'.\n\n"

    for ex in examples:
        label = ex.label
        if randomize_labels:
            label = random.choice(["positive", "negative"])
        prompt += f"Text: {ex.text}\nSentiment: {label}\n\n"

    prompt += f"Text: {test_input}\nSentiment:"
    return prompt


def run_icl_experiment(
    pipe,
    n_shots: int,
    randomize_labels: bool = False,
    reverse_order: bool = False,
) -> dict:
    """Run ICL experiment with given configuration."""
    selected = EXAMPLES[:n_shots]
    if reverse_order:
        selected = list(reversed(selected))

    correct = 0
    total = len(TEST_CASES)

    for test in TEST_CASES:
        prompt = build_icl_prompt(
            selected, test.text, randomize_labels=randomize_labels,
        )
        output = pipe(prompt, max_new_tokens=5, return_full_text=False)
        prediction = output[0]["generated_text"].strip().lower()

        # Extract first word as prediction
        pred_label = "positive" if "positive" in prediction else "negative"
        if pred_label == test.label:
            correct += 1

    return {
        "n_shots": n_shots,
        "randomize_labels": randomize_labels,
        "reverse_order": reverse_order,
        "accuracy": correct / total,
        "correct": correct,
        "total": total,
    }


def icl_as_regression():
    """Demonstrate ICL as implicit regression in a simple setting."""
    import torch
    import torch.nn.functional as F

    # Simulate: transformer attention implements least squares
    # Given examples (x_i, y_i), predict y for new x

    # Training examples (in-context)
    X_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    Y_train = torch.tensor([[2.1], [3.9], [6.1], [8.0]])  # y ≈ 2x

    # Test point
    x_test = torch.tensor([[5.0]])

    # OLS solution (what the transformer approximates)
    # w = (X^T X)^{-1} X^T Y
    XtX_inv = torch.inverse(X_train.T @ X_train)
    w = XtX_inv @ X_train.T @ Y_train
    y_pred = x_test @ w

    print("ICL as Implicit Regression:")
    print(f"  Learned weight: {w.item():.3f} (true: 2.0)")
    print(f"  Prediction for x=5: {y_pred.item():.3f} (expected: ~10.0)")

    # Show how adding more examples improves the estimate
    for n in [2, 3, 4]:
        X = X_train[:n]
        Y = Y_train[:n]
        w_n = torch.inverse(X.T @ X) @ X.T @ Y
        y_n = x_test @ w_n
        print(f"  With {n} examples: w={w_n.item():.3f}, "
              f"pred={y_n.item():.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("ICL as Implicit Regression")
    print("=" * 60)
    icl_as_regression()

    print("\n" + "=" * 60)
    print("ICL Experiments (requires model)")
    print("=" * 60)

    try:
        pipe = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            torch_dtype="auto",
            device_map="auto",
        )

        configs = [
            {"n_shots": 0, "randomize_labels": False},
            {"n_shots": 2, "randomize_labels": False},
            {"n_shots": 4, "randomize_labels": False},
            {"n_shots": 8, "randomize_labels": False},
            {"n_shots": 4, "randomize_labels": True},   # random labels
            {"n_shots": 4, "reverse_order": True},       # reversed order
        ]

        for cfg in configs:
            result = run_icl_experiment(pipe, **cfg)
            label = f"{result['n_shots']}-shot"
            if cfg.get("randomize_labels"):
                label += " (random labels)"
            if cfg.get("reverse_order"):
                label += " (reversed)"
            print(f"  {label}: {result['accuracy']:.0%} "
                  f"({result['correct']}/{result['total']})")
    except Exception as e:
        print(f"  Skipped model experiments: {e}")
```

---

## Exercise (45 min)

### E38.1 — Example Selection Strategy (25 min)
Implement and compare three strategies for selecting ICL examples:
1. **Random:** Pick examples randomly
2. **Diverse:** Maximize coverage across label space
3. **Similar:** Pick examples most similar to the test input (cosine similarity of embeddings)
Which strategy gives the best accuracy across 20 test cases?

### E38.2 — Chain-of-Thought ICL (20 min)
Extend the ICL framework with chain-of-thought:
1. Add reasoning traces to examples: `"Text: great movie → The word 'great' is positive → Sentiment: positive"`
2. Compare accuracy with and without chain-of-thought
3. Does CoT help more for "hard" examples (ambiguous sentiment)?

---

## Key Takeaways

1. **ICL is emergent** — it arises from scale, not explicit training
2. **Format matters more than labels** — even random labels improve over zero-shot
3. **Implicit Bayesian inference** and **mesa-optimization** are competing explanations for why ICL works
4. **Attention implements regression** — transformers can solve least squares in their forward pass
5. **Practical tips:** consistent format, representative examples, balanced labels, similar example last

---

## Connection to the Thread

ICL has profound implications for robotics. Instead of fine-tuning a robot model for each new task, you can provide a few demonstration examples in the prompt: "Here are 3 examples of picking up a cup. Now pick up this bottle." This is the foundation of **prompt-based robot learning** (SayCan, Code as Policies) — which we'll explore in Day 41.

---

## Further Reading

- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — Brown et al., 2020
- [Rethinking the Role of Demonstrations](https://arxiv.org/abs/2202.12837) — Min et al., 2022
- [Transformers Learn In-Context by Gradient Descent](https://arxiv.org/abs/2212.07677) — von Oswald et al., 2023
- [What Learning Algorithm is In-Context Learning?](https://arxiv.org/abs/2211.15661) — Akyürek et al., 2023
- [In-Context Learning as Implicit Bayesian Inference](https://arxiv.org/abs/2111.02080) — Xie et al., 2022
