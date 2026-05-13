# Day 88: Policy Debugging & Failure Analysis

> **Phase VI — Robot Learning: RL, Diffusion & Data | Week 13 | 2.5 hours**
> *"A policy that fails 30% of the time is useless in production. But the failure modes tell you exactly what to fix." — Suraj Nair*

## Navigation
- **Previous:** [Day 87: Policy Evaluation](day-87-policy-evaluation.md)
- **Next:** [Day 89: Phase VI Capstone Day 1](day-89-phase6-capstone-1.md)
- **Week:** [Week 13 Overview](README.md)
- **Phase:** [Phase VI: Robot Learning](../../study-notes/14-robot-data-eval.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 88.1 Failure Taxonomy

| Category | Symptoms | Root Cause | Fix |
|----------|----------|-----------|-----|
| **Perception** | Wrong grasp point, misaligned | Visual encoder fails | More image augmentation, better cameras |
| **Prediction** | Jerky motion, oscillation | Mode averaging | Use diffusion/CVAE instead of MSE |
| **Timing** | Too slow, misses target | Control frequency too low | Action chunking, faster inference |
| **Generalization** | Works on trained objects only | Overfitting | More diverse training data |
| **Compounding** | Starts OK, drifts | No recovery data | DAgger, longer chunks |
| **Physical** | Slips, collides | Physics not in training | Domain randomization, compliance |

### 88.2 Systematic Debugging Protocol

```
Step 1: CATEGORIZE failure mode
  └── Watch 10 failures. What's the dominant pattern?

Step 2: ISOLATE the component
  ├── Is perception correct? (overlay predicted pose)
  ├── Is action prediction reasonable? (plot predicted vs actual)
  └── Is execution accurate? (log commanded vs executed)

Step 3: HYPOTHESIZE root cause
  └── Form 2-3 hypotheses. Design test for each.

Step 4: TEST hypothesis
  └── Minimal intervention: change ONE thing.

Step 5: VALIDATE fix
  └── Re-run full evaluation. Is success rate up?
```

### 88.3 Visualization Tools

| Tool | What to Plot | Diagnostic Value |
|------|-------------|-----------------|
| **Action trajectory** | Predicted vs actual over time | Reveals oscillation, drift |
| **Attention maps** | Where the model "looks" | Perception debugging |
| **Latent space** | t-SNE of CVAE latent | Mode coverage |
| **Loss per timestep** | Loss at each step in trajectory | When errors appear |
| **Action histogram** | Distribution of predicted actions | Mode collapse |
| **Gripper timing** | Open/close signal over time | Grasp timing issues |

### 88.4 Common Failure Patterns & Fixes

**Pattern 1: Mode Averaging**
```
Symptom: Robot reaches between two objects
Diagnosis: MSE loss averages two valid grasps
Fix: Switch to diffusion policy or GMM loss
Test: Plot action histogram — bimodal → unimodal?
```

**Pattern 2: Compounding Error**
```
Symptom: Policy works for 5 steps, then fails
Diagnosis: Never saw off-distribution states
Fix: Collect DAgger corrections OR increase chunk size
Test: Compare success vs trajectory length
```

**Pattern 3: Visual Overfitting**
```
Symptom: Works under training lighting only
Diagnosis: No visual augmentation
Fix: Color jitter, random crop, background randomization
Test: Evaluate under novel lighting conditions
```

**Pattern 4: Action Delay**
```
Symptom: Robot moves correctly but 0.5s late
Diagnosis: Observation-action time misalignment in data
Fix: Check timestamp alignment, add latency compensation
Test: Plot cross-correlation of obs and actions
```

### 88.5 Ablation Studies

Structured ablation reveals what matters:

```
Full model:     chunk=16, CVAE, ResNet-18, 100 demos → 85% success

Ablations:
  chunk=1:      -25% (60% success) → chunking is critical
  No CVAE:      -15% (70% success) → CVAE helps multimodality
  ResNet-50:    +2%  (87% success) → bigger backbone not worth it
  50 demos:     -10% (75% success) → data quantity matters
  No img aug:   -20% (65% success) → augmentation is critical
```

---

## Implementation (60 min)

### Debugging Toolkit

```python
import numpy as np
import json
from pathlib import Path

class PolicyDebugger:
    """Tools for diagnosing policy failures."""

    def __init__(self, env, policy, n_debug_episodes=20):
        self.env = env
        self.policy = policy
        self.n_debug = n_debug_episodes

    def collect_debug_trajectories(self):
        """Collect trajectories with full diagnostic info."""
        trajectories = []
        for i in range(self.n_debug):
            obs, _ = self.env.reset()
            traj = {
                "observations": [],
                "predicted_actions": [],
                "executed_actions": [],
                "rewards": [],
                "success": False,
            }

            for step in range(300):
                pred_action = self.policy.predict(obs)
                traj["observations"].append(obs.tolist())
                traj["predicted_actions"].append(pred_action.tolist())

                obs, reward, term, trunc, info = self.env.step(pred_action)
                traj["executed_actions"].append(pred_action.tolist())
                traj["rewards"].append(float(reward))

                if term or trunc:
                    traj["success"] = bool(reward > 0)
                    break

            trajectories.append(traj)

        successes = sum(t["success"] for t in trajectories)
        print(f"Debug rollouts: {successes}/{len(trajectories)} successful")
        return trajectories

    def analyze_failures(self, trajectories):
        """Analyze failure patterns."""
        failures = [t for t in trajectories if not t["success"]]
        successes = [t for t in trajectories if t["success"]]

        if not failures:
            print("No failures to analyze!")
            return

        print(f"\n{'='*50}")
        print(f"FAILURE ANALYSIS ({len(failures)} failures)")
        print(f"{'='*50}")

        # 1. Failure timing
        fail_lengths = [len(t["rewards"]) for t in failures]
        succ_lengths = [len(t["rewards"]) for t in successes] if successes else [0]
        print(f"\nFailure timing:")
        print(f"  Failure episode length: {np.mean(fail_lengths):.0f} ± {np.std(fail_lengths):.0f}")
        print(f"  Success episode length: {np.mean(succ_lengths):.0f} ± {np.std(succ_lengths):.0f}")

        early_failures = sum(1 for l in fail_lengths if l < np.mean(succ_lengths) * 0.3)
        print(f"  Early failures (<30% of success length): {early_failures}/{len(failures)}")

        # 2. Action statistics comparison
        fail_actions = np.concatenate([np.array(t["predicted_actions"]) for t in failures])
        succ_actions = np.concatenate([np.array(t["predicted_actions"]) for t in successes]) if successes else fail_actions

        print(f"\nAction distribution comparison:")
        for d in range(fail_actions.shape[1]):
            f_mean, f_std = fail_actions[:, d].mean(), fail_actions[:, d].std()
            s_mean, s_std = succ_actions[:, d].mean(), succ_actions[:, d].std()
            print(f"  dim {d}: fail μ={f_mean:.4f} σ={f_std:.4f} | "
                  f"succ μ={s_mean:.4f} σ={s_std:.4f}")

        # 3. Action smoothness
        fail_jerks = []
        succ_jerks = []
        for t in failures:
            acts = np.array(t["predicted_actions"])
            if len(acts) > 3:
                jerk = np.mean(np.abs(np.diff(acts, n=3, axis=0)))
                fail_jerks.append(jerk)
        for t in successes:
            acts = np.array(t["predicted_actions"])
            if len(acts) > 3:
                jerk = np.mean(np.abs(np.diff(acts, n=3, axis=0)))
                succ_jerks.append(jerk)

        if fail_jerks and succ_jerks:
            print(f"\nAction smoothness (jerk):")
            print(f"  Failures: {np.mean(fail_jerks):.6f}")
            print(f"  Successes: {np.mean(succ_jerks):.6f}")
            if np.mean(fail_jerks) > 2 * np.mean(succ_jerks):
                print(f"  ⚠️ Failures are significantly jerkier → possible mode averaging")

        # 4. Diagnose pattern
        print(f"\n{'='*50}")
        print("DIAGNOSIS:")
        if early_failures > len(failures) * 0.5:
            print("  → Perception failure likely (fails before reaching target)")
        elif np.mean(fail_jerks) > 2 * np.mean(succ_jerks) if succ_jerks else False:
            print("  → Mode averaging likely (jerky actions in failures)")
        elif np.mean(fail_lengths) > np.mean(succ_lengths) * 1.5:
            print("  → Compounding error likely (reaches target but drifts)")
        else:
            print("  → Mixed failure modes. Manual video review recommended.")

    def run_ablation(self, configs, n_trials=20):
        """Run systematic ablation study."""
        results = {}
        for name, config_fn in configs.items():
            print(f"\nAblation: {name}")
            config_fn()  # Apply config change
            trajectories = self.collect_debug_trajectories()
            sr = sum(t["success"] for t in trajectories) / len(trajectories)
            results[name] = sr

        # Print sorted results
        print(f"\n{'='*50}")
        print("ABLATION RESULTS")
        print(f"{'='*50}")
        for name, sr in sorted(results.items(), key=lambda x: -x[1]):
            bar = "█" * int(sr * 40)
            print(f"  {name:30s} {sr:6.1%} {bar}")

        return results
```

---

## Exercise (45 min)

1. **Failure taxonomy:** Collect 20 rollouts of a trained policy. Manually classify each failure into the categories from §88.1. What's the dominant mode?

2. **Debug a policy:** Use `PolicyDebugger.analyze_failures()`. Based on the output, form a hypothesis and propose a fix. Implement the fix and re-evaluate.

3. **Ablation study:** Take your best policy. Run 5 ablations (remove augmentation, reduce data, smaller model, no chunking, different learning rate). Present results as a table.

4. **Action visualization:** Plot predicted action trajectories for 5 successful and 5 failed episodes on the same axes. What visual patterns distinguish success from failure?

---

## Key Takeaways

1. **Categorize first** — don't guess; watch failures and classify them
2. **Isolate the component** — perception, prediction, or execution?
3. **Change one thing at a time** — systematic ablation, not shotgun debugging
4. **Visual analysis is essential** — overlay predictions, plot distributions
5. **Most failures are data problems** — more diverse demos often fix more than architecture changes

---

## Connection to the Thread

> *With debugging tools in hand, we're ready for the Phase VI capstone. Days 89-91: build a complete robot learning pipeline from scratch — collect data, train a policy (BC, diffusion, or flow), evaluate systematically, debug failures, iterate. This consolidates everything from RL through diffusion to deployment.*

---

## Further Reading

- Mandlekar et al. (2021), "What Matters in Learning from Offline Human Demonstrations" — systematic ablation study
- Zhao et al. (2023), ACT paper — excellent failure analysis section
- Chi et al. (2023), Diffusion Policy — comprehensive ablation appendix
