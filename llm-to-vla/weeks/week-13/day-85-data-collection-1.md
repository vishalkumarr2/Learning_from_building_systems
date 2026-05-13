# Day 85: Data Collection for Robot Learning — Part 1

> **Phase VI — Robot Learning: RL, Diffusion & Data | Week 13 | 2.5 hours**
> *"Data is the new algorithm. The best policy architecture with bad data loses to the simplest BC with excellent demonstrations."*

## Navigation
- **Previous:** [Day 84: Stop & Reflect #5](../week-12/day-84-stop-reflect-5.md)
- **Next:** [Day 86: Data Collection Part 2 — Scaling & Quality](day-86-data-collection-2.md)
- **Week:** [Week 13 Overview](README.md)
- **Phase:** [Phase VI: Robot Learning](../../study-notes/14-robot-data-eval.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 85.1 The Robot Data Problem

| Domain | Dataset Size | Cost per Sample | Diversity |
|--------|-------------|-----------------|-----------|
| NLP | Trillions of tokens | ~$0 (web scrape) | Extreme |
| Vision | Billions of images | ~$0 (web scrape) | High |
| Robotics | Thousands of demos | $5-50 per demo | Low |

The gap: ~1,000,000× less data than language models.

### 85.2 Teleoperation Methods

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  VR headset  │     │   Leader-    │     │   Phone/     │
│  + haptics   │     │   follower   │     │   keyboard   │
│              │     │   arms       │     │              │
│  High DOF    │     │  Bilateral   │     │  Low DOF     │
│  Intuitive   │     │  force       │     │  Cheap       │
│  Expensive   │     │  feedback    │     │  Imprecise   │
└──────────────┘     └──────────────┘     └──────────────┘
     Meta Quest           ALOHA             SpaceMouse
     Apple Vision         Gello
```

| Method | Cost | DOF | Intuitiveness | Used By |
|--------|------|-----|---------------|---------|
| VR controller | $300-3500 | 6-7 | High | Some labs |
| Leader-follower | $5K-20K | Full arm DOF | Very high | ACT/ALOHA |
| SpaceMouse | $200 | 6 | Medium | robomimic |
| Keyboard | $0 | 2-3 | Low | Prototyping |
| Kinesthetic teaching | $0 | Full DOF | High | Small datasets |

### 85.3 What to Record

Every demonstration should capture:

```
Timestep t:
  ├── Observation
  │   ├── images: {camera_1: RGB, camera_2: RGB, wrist: RGB}
  │   ├── depth: {camera_1: depth_map}  (optional)
  │   ├── robot_state: {joint_positions, joint_velocities, ee_pose}
  │   └── gripper_state: {width, force}
  ├── Action
  │   ├── commanded: {joint_positions or ee_delta}
  │   └── executed: {actual joint positions}
  ├── Metadata
  │   ├── timestamp
  │   ├── task_id
  │   ├── episode_id
  │   └── success: bool
  └── Language
      └── instruction: "pick up the red mug and place it on the coaster"
```

### 85.4 Data Formats

| Format | Used By | Strengths |
|--------|---------|-----------|
| **LeRobot** (HF) | OpenVLA, community | Standard, sharable |
| **RLDS** (TFRecords) | RT-1, RT-2, Octo | Open X-Embodiment |
| **robomimic** (HDF5) | ACT, many labs | Compact, fast I/O |
| **rosbag** | ROS robots | Raw, complete |
| **custom pickle** | Quick prototypes | Don't do this |

### 85.5 Common Data Collection Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| Camera calibration drift | Features misalign | Re-calibrate regularly |
| Operator fatigue | Demo quality drops | 30-min sessions max |
| Missing failure recovery | Policy can't recover | Intentionally include recovery demos |
| Single operator | Style bias | Multiple operators |
| Fixed lighting | Not robust | Vary lighting conditions |
| Clean workspace only | Not robust | Add distractors |

---

## Implementation (60 min)

### Data Collection Pipeline

```python
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

class DemoCollector:
    """Minimal demonstration collection system."""

    def __init__(self, task_name, save_dir="./demos", hz=10):
        self.task_name = task_name
        self.save_dir = Path(save_dir) / task_name
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.hz = hz
        self.dt = 1.0 / hz

    def collect_episode(self, env, teleop_fn, language_instruction):
        """
        Collect a single demonstration episode.

        Args:
            env: gymnasium environment
            teleop_fn: function that returns action given observation
            language_instruction: text description of the task
        """
        obs, _ = env.reset()
        episode = {
            "metadata": {
                "task": self.task_name,
                "instruction": language_instruction,
                "timestamp": datetime.now().isoformat(),
                "hz": self.hz,
            },
            "observations": [],
            "actions": [],
            "rewards": [],
        }

        done = False
        step = 0
        while not done:
            t_start = time.time()

            # Get teleop action
            action = teleop_fn(obs)

            # Record
            episode["observations"].append(obs.tolist() if hasattr(obs, 'tolist') else obs)
            episode["actions"].append(action.tolist() if hasattr(action, 'tolist') else action)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode["rewards"].append(float(reward))
            done = terminated or truncated
            step += 1

            # Maintain control frequency
            elapsed = time.time() - t_start
            if elapsed < self.dt:
                time.sleep(self.dt - elapsed)

        # Record success
        episode["metadata"]["success"] = bool(sum(episode["rewards"]) > 0)
        episode["metadata"]["length"] = step

        return episode

    def save_episode(self, episode, episode_id=None):
        """Save episode to disk."""
        if episode_id is None:
            existing = list(self.save_dir.glob("episode_*.json"))
            episode_id = len(existing)

        path = self.save_dir / f"episode_{episode_id:04d}.json"
        with open(path, 'w') as f:
            json.dump(episode, f)
        print(f"Saved: {path} ({episode['metadata']['length']} steps, "
              f"success={episode['metadata']['success']})")
        return path

    def collect_dataset(self, env, teleop_fn, instruction, n_episodes=50):
        """Collect a full dataset."""
        episodes = []
        successes = 0
        for i in range(n_episodes):
            print(f"\n--- Episode {i+1}/{n_episodes} ---")
            ep = self.collect_episode(env, teleop_fn, instruction)
            self.save_episode(ep, episode_id=i)
            episodes.append(ep)
            if ep["metadata"]["success"]:
                successes += 1
            print(f"Running success rate: {successes}/{i+1} = {successes/(i+1):.1%}")

        print(f"\nDataset complete: {len(episodes)} episodes, "
              f"{successes}/{len(episodes)} successful")
        return episodes

# --- Data Quality Validation ---
class DataValidator:
    """Validate collected demonstrations."""

    @staticmethod
    def check_episode(episode):
        issues = []

        # Check lengths match
        n_obs = len(episode["observations"])
        n_act = len(episode["actions"])
        if n_obs != n_act:
            issues.append(f"Length mismatch: {n_obs} obs vs {n_act} actions")

        # Check for NaN/Inf
        for i, obs in enumerate(episode["observations"]):
            arr = np.array(obs)
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                issues.append(f"NaN/Inf in observation at step {i}")

        for i, act in enumerate(episode["actions"]):
            arr = np.array(act)
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                issues.append(f"NaN/Inf in action at step {i}")

        # Check action bounds
        actions = np.array(episode["actions"])
        if np.any(np.abs(actions) > 10.0):
            issues.append(f"Actions exceed bounds: max={np.abs(actions).max():.2f}")

        # Check episode length
        if len(episode["actions"]) < 5:
            issues.append(f"Episode too short: {len(episode['actions'])} steps")

        return issues

    @staticmethod
    def dataset_statistics(episodes):
        """Compute dataset-level statistics."""
        lengths = [ep["metadata"]["length"] for ep in episodes]
        successes = sum(1 for ep in episodes if ep["metadata"]["success"])

        all_actions = np.concatenate(
            [np.array(ep["actions"]) for ep in episodes]
        )

        print(f"Episodes: {len(episodes)}")
        print(f"Success rate: {successes}/{len(episodes)} = {successes/len(episodes):.1%}")
        print(f"Length: mean={np.mean(lengths):.0f}, "
              f"std={np.std(lengths):.0f}, "
              f"range=[{min(lengths)}, {max(lengths)}]")
        print(f"Action stats per dimension:")
        for d in range(all_actions.shape[1]):
            print(f"  dim {d}: mean={all_actions[:,d].mean():.4f}, "
                  f"std={all_actions[:,d].std():.4f}, "
                  f"range=[{all_actions[:,d].min():.4f}, {all_actions[:,d].max():.4f}]")
```

---

## Exercise (45 min)

1. **Collect and validate:** Use `DemoCollector` with CartPole or Pendulum (keyboard teleop). Collect 20 episodes. Run `DataValidator` on the dataset.

2. **Operator variance:** Collect 10 episodes with "aggressive" control and 10 with "gentle" control. Train BC on each. Compare policies.

3. **Data augmentation:** Implement observation noise injection, action perturbation, and temporal subsampling. Measure impact on BC training.

4. **LeRobot format:** Convert your collected dataset to HuggingFace LeRobot format. Push to the Hub.

---

## Key Takeaways

1. **Robot data is 1,000,000× scarcer** than language data — quality over quantity
2. **Leader-follower teleoperation** gives the best demonstrations for manipulation
3. **Record everything:** images, proprioception, actions, language, metadata
4. **Validate data** before training — NaNs, action bounds, episode lengths
5. **Data quality > data quantity** — 50 perfect demos beats 500 sloppy ones

---

## Connection to the Thread

> *Tomorrow: scaling data collection, quality metrics, data mixing strategies, and how Open X-Embodiment aggregated data from 22 robots into one dataset. Then on Day 87, we tackle the harder question: how do you know if your trained policy is actually good?*

---

## Further Reading

- Mandlekar et al. (2021), "What Matters in Learning from Offline Human Demonstrations"
- HuggingFace LeRobot documentation
- Open X-Embodiment Collaboration (2023), "Open X-Embodiment: Robotic Learning Datasets and RT-X Models"
