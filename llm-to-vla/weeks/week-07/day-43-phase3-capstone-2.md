# Day 43: Phase III Capstone Day 2 — Evaluation & Comparison

> **Phase III — LLMs: Training & Alignment | Week 7 | 2.5 hours**
> *"A model without evaluation is a hypothesis without evidence."*

## Navigation
- **Previous:** [Day 42: Phase III Capstone Day 1](../week-06/day-42-phase3-capstone-1.md)
- **Next:** [Day 44: Phase III Capstone Day 3 + Checkpoint](day-44-phase3-capstone-3.md)
- **Week:** [Week 7 Overview](README.md)
- **Phase:** [Phase III: LLM Training & Alignment](../../study-notes/06-llm-training-alignment.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Today's Goal

Rigorously evaluate the robotics assistant built yesterday across three configurations:

```
Config A: Base model (no fine-tuning, no RAG)
Config B: LoRA fine-tuned model (no RAG)
Config C: LoRA fine-tuned model + RAG (full pipeline)

Compare on:
  1. Knowledge accuracy (keyword recall on domain questions)
  2. Diagnosis quality (correct root cause identification)
  3. Command parsing accuracy (NL → structured command)
  4. Response relevance (human-like judgment score)
  5. Latency (tokens per second)
  6. Hallucination rate (stated facts not in training/retrieval data)
```

---

## Implementation (120 min)

### Evaluation Framework

```python
"""
Day 43 Capstone: Evaluation framework for robotics assistant.
Compares base, LoRA-tuned, and LoRA+RAG configurations.
"""
from dataclasses import dataclass, field
import time
import json


@dataclass
class EvalResult:
    question: str
    category: str
    config: str
    response: str
    keyword_recall: float = 0.0
    relevance_score: float = 0.0
    latency_ms: float = 0.0
    hallucination_detected: bool = False
    notes: str = ""


@dataclass
class BenchmarkSuite:
    """Comprehensive evaluation benchmark for robotics assistant."""

    questions: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if not self.questions:
            self.questions = self._default_questions()

    def _default_questions(self) -> list[dict]:
        return [
            # --- Knowledge questions ---
            {
                "question": "What sensor fusion algorithm does the OKS "
                            "navigation estimator use?",
                "expected": ["EKF", "Extended Kalman", "IMU", "wheel encoder",
                             "LiDAR"],
                "category": "knowledge",
                "difficulty": "easy",
            },
            {
                "question": "What is the maximum linear velocity of the "
                            "OKS robot?",
                "expected": ["1.5", "m/s"],
                "category": "knowledge",
                "difficulty": "easy",
            },
            {
                "question": "Explain the difference between the global planner "
                            "and local planner in robot navigation.",
                "expected": ["global", "local", "A*", "DWA", "obstacle",
                             "path"],
                "category": "knowledge",
                "difficulty": "medium",
            },
            {
                "question": "How does the Guardian node decide when to trigger "
                            "an emergency stop?",
                "expected": ["threshold", "watchdog", "temperature", "battery",
                             "obstacle"],
                "category": "knowledge",
                "difficulty": "hard",
            },
            # --- Diagnosis questions ---
            {
                "question": "Robot OKS-15 shows 'NAV_ESTIMATED_STATE_NOT_FINITE' "
                            "error. What could cause this?",
                "expected": ["NaN", "IMU", "encoder", "sensorbar", "SPI"],
                "category": "diagnosis",
                "difficulty": "medium",
            },
            {
                "question": "A robot keeps stopping with 'sensorbar SPI timeout' "
                            "in Zone C. Three robots in Zone C are affected, "
                            "but Zone A robots are fine. What's the likely cause?",
                "expected": ["environmental", "floor", "debris", "zone",
                             "interference"],
                "category": "diagnosis",
                "difficulty": "hard",
            },
            # --- Command questions ---
            {
                "question": "Send robot OKS-42 to the charging station.",
                "expected": ["navigate", "charging", "OKS-42"],
                "category": "command",
                "difficulty": "easy",
            },
            {
                "question": "Emergency stop all robots in Zone B!",
                "expected": ["stop", "emergency", "zone B"],
                "category": "command",
                "difficulty": "medium",
            },
            # --- Reasoning questions ---
            {
                "question": "We have 12 robots and 3 charging stations. "
                            "Each robot needs 20 minutes of charging every "
                            "3 hours. Can we maintain 100% fleet availability?",
                "expected": ["no", "charging", "capacity", "schedule",
                             "downtime"],
                "category": "reasoning",
                "difficulty": "hard",
            },
            {
                "question": "If the sensorbar update rate drops from 50Hz "
                            "to 10Hz, how does this affect the navigation "
                            "estimator's performance?",
                "expected": ["accuracy", "drift", "latency", "EKF",
                             "prediction"],
                "category": "reasoning",
                "difficulty": "hard",
            },
        ]


def evaluate_keyword_recall(
    response: str, expected_keywords: list[str],
) -> float:
    """Compute keyword recall: fraction of expected keywords found."""
    response_lower = response.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in response_lower)
    return found / len(expected_keywords) if expected_keywords else 0.0


def detect_hallucination(
    response: str, knowledge_base: list[str],
) -> bool:
    """Basic hallucination detection: check for suspicious specifics."""
    # Heuristic: specific numbers/values not in knowledge base
    import re
    numbers_in_response = re.findall(r'\b\d+\.?\d*\b', response)
    all_kb_text = " ".join(knowledge_base).lower()

    suspicious = 0
    for num in numbers_in_response:
        if num not in all_kb_text and float(num) > 1:
            suspicious += 1

    # If more than 30% of specific numbers aren't in KB, flag it
    return suspicious > len(numbers_in_response) * 0.3 if numbers_in_response else False


def compute_aggregate_scores(results: list[EvalResult]) -> dict:
    """Compute aggregate scores per configuration and category."""
    configs = set(r.config for r in results)
    categories = set(r.category for r in results)

    scores = {}
    for config in configs:
        config_results = [r for r in results if r.config == config]
        scores[config] = {
            "overall_recall": sum(r.keyword_recall for r in config_results) / len(config_results),
            "avg_latency_ms": sum(r.latency_ms for r in config_results) / len(config_results),
            "hallucination_rate": sum(r.hallucination_detected for r in config_results) / len(config_results),
        }

        for cat in categories:
            cat_results = [r for r in config_results if r.category == cat]
            if cat_results:
                scores[config][f"{cat}_recall"] = (
                    sum(r.keyword_recall for r in cat_results) / len(cat_results)
                )

    return scores


def print_comparison_table(scores: dict):
    """Print a formatted comparison table."""
    configs = sorted(scores.keys())
    metrics = ["overall_recall", "knowledge_recall", "diagnosis_recall",
               "command_recall", "avg_latency_ms", "hallucination_rate"]

    print(f"\n{'Metric':<25}", end="")
    for config in configs:
        print(f"{config:>15}", end="")
    print()
    print("-" * (25 + 15 * len(configs)))

    for metric in metrics:
        print(f"{metric:<25}", end="")
        for config in configs:
            val = scores[config].get(metric, 0)
            if "latency" in metric:
                print(f"{val:>14.0f}ms", end="")
            elif "rate" in metric:
                print(f"{val:>14.1%}", end="")
            else:
                print(f"{val:>14.1%}", end="")
        print()


# --- Demo with simulated results ---
if __name__ == "__main__":
    bench = BenchmarkSuite()
    print(f"Benchmark: {len(bench.questions)} questions")
    print(f"Categories: {set(q['category'] for q in bench.questions)}")
    print(f"Difficulties: {set(q['difficulty'] for q in bench.questions)}")

    # Simulated results for demonstration
    simulated_results = []
    for q in bench.questions:
        for config, base_recall in [("Base", 0.3), ("LoRA", 0.6), ("LoRA+RAG", 0.8)]:
            import random
            recall = min(1.0, base_recall + random.uniform(-0.15, 0.15))
            simulated_results.append(EvalResult(
                question=q["question"],
                category=q["category"],
                config=config,
                response=f"[simulated {config} response]",
                keyword_recall=recall,
                latency_ms=random.uniform(50, 300),
                hallucination_detected=random.random() < (0.3 if config == "Base" else 0.1),
            ))

    scores = compute_aggregate_scores(simulated_results)
    print_comparison_table(scores)
```

---

## Exercise (Remaining time)

### E43.1 — Run Full Evaluation
If you have the trained models from Day 42:
1. Run all 10 benchmark questions through each configuration
2. Record actual keyword recall, latency, and hallucination rates
3. Identify the **best and worst question** for each configuration

### E43.2 — Error Analysis
For each wrong or weak answer:
1. Classify the failure mode: hallucination, omission, wrong focus, or formatting
2. Which failures does RAG fix? Which does LoRA fix? Which remain?
3. What additional training data would address the remaining failures?

### E43.3 — Ablation Study
Test the impact of individual components:
1. LoRA rank: compare r=4 vs r=16 vs r=64
2. RAG top-k: compare k=1 vs k=3 vs k=5
3. Which hyperparameter has the largest effect on final quality?

---

## Key Takeaways

1. **RAG provides the largest improvement** on factual/knowledge questions (grounds answers in real docs)
2. **LoRA improves format and domain style** — responses sound more like a robotics expert
3. **Neither fixes reasoning failures** — multi-step logic requires stronger base model or CoT
4. **Hallucination detection** is critical for safety-relevant robotics applications
5. **Latency matters** — quantization (Day 37) would be needed for production deployment

---

## Connection to the Thread

Evaluation methodology carries directly to robot systems. In Phase VII (VLAs), we'll evaluate robot policies on success rate, efficiency, safety, and generalization — the same multi-dimensional evaluation framework we built here for language outputs. Learning to evaluate rigorously now prevents false confidence later.

---

## Further Reading

- [RAGAS: Automated Evaluation of RAG](https://arxiv.org/abs/2309.15217) — Evaluation metrics
- [DeepEval: LLM Evaluation Framework](https://github.com/confident-ai/deepeval) — Open-source tool
- [Holistic Evaluation of Language Models (HELM)](https://arxiv.org/abs/2211.09110) — Stanford, 2022
