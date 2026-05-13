# Day 87: Policy Evaluation — How Do You Know It Works?

> **Phase VI — Robot Learning: RL, Diffusion & Data | Week 13 | 2.5 hours**
> *"In NLP, you compute BLEU. In vision, you compute FID. In robotics, you put the robot on a table and see if it breaks things." — Sergey Levine*

## Navigation
- **Previous:** [Day 86: Data Collection Part 2](day-86-data-collection-2.md)
- **Next:** [Day 88: Policy Debugging & Failure Analysis](day-88-policy-debugging.md)
- **Week:** [Week 13 Overview](README.md)
- **Phase:** [Phase VI: Robot Learning](../../study-notes/14-robot-data-eval.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 87.1 The Evaluation Problem

| Domain | Evaluation | Cost | Speed |
|--------|-----------|------|-------|
| Language | Benchmark (MMLU, etc.) | Free | Seconds |
| Vision | FID, IS, CLIP score | Free | Seconds |
| Robotics | **Physical rollouts** | $$$, risk of damage | Hours |

Robot evaluation is **expensive, slow, and potentially destructive**.

### 87.2 Offline vs Online Metrics

**Offline (cheap, fast, incomplete):**

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| MSE loss | $\|a - \hat{a}\|^2$ | Action prediction accuracy |
| Action likelihood | $\log p(a \mid o)$ | How well model explains data |
| FVD | Feature Video Distance | Visual trajectory similarity |
| Trajectory similarity | DTW distance | Shape of predicted trajectory |

**Online (expensive, slow, definitive):**

| Metric | How to Measure | What It Means |
|--------|---------------|---------------|
| **Success rate** | $N_\text{success}/N_\text{trials}$ | Primary metric |
| **Completion fraction** | Subtask progress (0-1) | Partial credit |
| **Cycle time** | Time to complete task | Efficiency |
| **Collision rate** | Contact sensor triggers | Safety |
| **Generalization** | Test on unseen objects/poses | Robustness |

### 87.3 The Offline-Online Gap

Low training loss ≠ high success rate. Why?

```
Training loss ↓                    Success rate ?

Reasons for the gap:
  1. Compounding errors (not visible in single-step loss)
  2. Distribution shift (test poses differ from training)
  3. Mode averaging (loss is low but policy grabs air)
  4. Observation processing (lighting, camera angle changes)
  5. Timing (control frequency different from training)
```

### 87.4 Evaluation Protocol Design

A rigorous evaluation protocol specifies:

```
Protocol: "Push T evaluation" (Diffusion Policy benchmark)
  ├── Task: Push T-shaped block to goal pose
  ├── Trials: 50 rollouts per checkpoint
  ├── Randomization:
  │   ├── Initial block pose: sampled from distribution D₁
  │   ├── Goal pose: sampled from distribution D₂
  │   └── Distractors: 2 random objects on table
  ├── Success criteria:
  │   ├── Block within 5mm of goal position
  │   └── Block within 5° of goal orientation
  ├── Max steps: 300 (at 10 Hz = 30 seconds)
  └── Reporting: mean ± std over 3 seeds, 50 trials each
```

### 87.5 Simulation vs Real Evaluation

| Aspect | Simulation | Real |
|--------|-----------|------|
| Speed | 1000× faster | Real-time only |
| Cost | ~$0 | Robot time, operator |
| Fidelity | Approximate physics | Ground truth |
| Diversity | Easy to randomize | Limited |
| Safety | No damage | Risk of damage |

**The sim-to-real gap:** policies that work in simulation often fail on real robots due to:
- Physics differences (friction, deformation, contact)
- Visual differences (textures, lighting, reflections)
- Sensor noise (not modeled in sim)

### 87.6 Statistical Rigor

**How many trials do you need?**

For binomial success rate with confidence interval:

$$n = \frac{z^2 \cdot p(1-p)}{\epsilon^2}$$

where $z = 1.96$ for 95% CI, $p$ = expected success rate, $\epsilon$ = margin of error.

| Expected Rate | Margin ±5% | Margin ±10% |
|--------------|-----------|------------|
| 50% | 385 trials | 97 trials |
| 80% | 246 trials | 62 trials |
| 95% | 73 trials | 19 trials |

**In practice:** most papers use 20-50 trials (statistically weak but practically necessary).

---

## Implementation (60 min)

### Evaluation Framework

```python
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EvalResult:
    success: bool
    completion: float  # 0-1 partial progress
    steps: int
    collision: bool = False
    info: dict = field(default_factory=dict)

class PolicyEvaluator:
    """Systematic policy evaluation with statistical reporting."""

    def __init__(self, env, policy, max_steps=300):
        self.env = env
        self.policy = policy
        self.max_steps = max_steps

    def run_single_trial(self, seed=None) -> EvalResult:
        """Run a single evaluation trial."""
        if seed is not None:
            obs, _ = self.env.reset(seed=seed)
        else:
            obs, _ = self.env.reset()

        total_reward = 0
        for step in range(self.max_steps):
            action = self.policy.predict(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        return EvalResult(
            success=total_reward > 0,
            completion=min(1.0, max(0.0, total_reward)),
            steps=step + 1,
            info=info,
        )

    def evaluate(self, n_trials=50, seeds=None):
        """Run full evaluation with statistics."""
        if seeds is None:
            seeds = list(range(n_trials))

        results = []
        for i, seed in enumerate(seeds[:n_trials]):
            result = self.run_single_trial(seed=seed)
            results.append(result)
            if (i + 1) % 10 == 0:
                sr = sum(r.success for r in results) / len(results)
                print(f"  Trial {i+1}/{n_trials}: running SR = {sr:.1%}")

        return self._compute_statistics(results)

    def _compute_statistics(self, results):
        """Compute evaluation statistics with confidence intervals."""
        n = len(results)
        successes = sum(r.success for r in results)
        sr = successes / n

        # Wilson score confidence interval (better than Wald for small n)
        z = 1.96  # 95% CI
        denominator = 1 + z**2 / n
        center = (sr + z**2 / (2*n)) / denominator
        spread = z * np.sqrt((sr*(1-sr) + z**2/(4*n)) / n) / denominator
        ci_low = max(0, center - spread)
        ci_high = min(1, center + spread)

        completions = [r.completion for r in results]
        steps = [r.steps for r in results]

        stats = {
            "n_trials": n,
            "success_rate": sr,
            "ci_95": (ci_low, ci_high),
            "completion_mean": np.mean(completions),
            "completion_std": np.std(completions),
            "steps_mean": np.mean(steps),
            "steps_std": np.std(steps),
            "collision_rate": sum(r.collision for r in results) / n,
        }

        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Success Rate: {sr:.1%} ({successes}/{n})")
        print(f"95% CI: [{ci_low:.1%}, {ci_high:.1%}]")
        print(f"Completion: {stats['completion_mean']:.2f} ± {stats['completion_std']:.2f}")
        print(f"Steps: {stats['steps_mean']:.0f} ± {stats['steps_std']:.0f}")
        print("="*50)

        return stats

    @staticmethod
    def compare_policies(stats_a, stats_b, name_a="A", name_b="B"):
        """Compare two policies with statistical significance."""
        sr_a, n_a = stats_a["success_rate"], stats_a["n_trials"]
        sr_b, n_b = stats_b["success_rate"], stats_b["n_trials"]

        # Two-proportion z-test
        p_pool = (sr_a * n_a + sr_b * n_b) / (n_a + n_b)
        se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
        z_stat = (sr_a - sr_b) / (se + 1e-10)
        significant = abs(z_stat) > 1.96

        print(f"\n{name_a}: {sr_a:.1%} vs {name_b}: {sr_b:.1%}")
        print(f"Difference: {(sr_a-sr_b):.1%}")
        print(f"z-statistic: {z_stat:.2f}")
        print(f"Significant (p<0.05): {'YES' if significant else 'NO'}")
```

---

## Exercise (45 min)

1. **Confidence interval experiment:** Simulate a policy with 80% true success rate. Run 10, 20, 50, 100, 200 trials. Plot CI width vs trial count.

2. **Offline-online correlation:** Train BC with different dataset sizes. Plot MSE loss vs simulated success rate. Is there a correlation? When does it break down?

3. **Evaluation protocol:** Write a complete evaluation protocol specification for a task of your choice (push, pick-place, or navigation). Include randomization, success criteria, and trial count justification.

4. **Policy comparison:** Train two variants of a policy (e.g., chunk_size=4 vs 16). Use `compare_policies` to determine if the difference is statistically significant with 50 trials.

---

## Key Takeaways

1. **Success rate is the primary metric** — offline losses are insufficient
2. **Wilson score CI** is better than simple binomial CI for small sample sizes
3. **50 trials is the minimum** for credible evaluation; 100+ is better
4. **Sim evaluation is necessary** but not sufficient — real-world testing is essential
5. **Report confidence intervals**, not just point estimates

---

## Connection to the Thread

> *Evaluation tells you what's wrong. Tomorrow: how to diagnose why it's wrong. Policy debugging is more art than science — but structured hypothesis testing helps. Then Days 89-91 are the Phase VI capstone: integrating RL, diffusion, data, and evaluation into a complete robot learning pipeline.*

---

## Further Reading

- Brown et al. (2001), "Interval Estimation for a Binomial Proportion" (Wilson CI)
- Chi et al. (2023), "Diffusion Policy" — evaluation protocol appendix
- RT-2 evaluation: Brohan et al. (2023), extensive real-world eval section
