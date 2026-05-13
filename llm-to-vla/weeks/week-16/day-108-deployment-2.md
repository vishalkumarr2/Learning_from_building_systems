# Day 108: Deployment — Safety, Monitoring & Human Oversight

> **Phase VII — VLAs: Architecture to Deployment | Week 16 | 2.5 hours**
> *"An autonomous robot that can't detect its own failures is a hazard. Safety is not a feature — it's a constraint." — Deployment Safety*

## Navigation
- **Previous:** [Day 107: Deployment — Latency](day-107-deployment-1.md)
- **Next:** [Day 109: Deployment — Fleet Management](day-109-deployment-3.md)
- **Week:** [Week 16 Overview](README.md)
- **Phase:** [Phase VII: VLAs](../../study-notes/16-deployment-hybrid.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (60 min)

### 108.1 Safety Layers

```
Layer 5: Task-level safety
  "Don't pour liquid on electronics"
  ← VLM reasoning about consequences

Layer 4: Behavior-level safety
  "Stop if confidence drops below threshold"
  ← Action distribution analysis

Layer 3: Motion-level safety
  "Limit velocity to 0.5 m/s near humans"
  ← Trajectory filtering

Layer 2: Hardware-level safety
  "Emergency stop if joint torque exceeds limit"
  ← Real-time hardware watchdog

Layer 1: Physical safety
  "Compliant joints, soft covers, force limiting"
  ← Mechanical design
```

### 108.2 Failure Detection

```python
# Types of failure a VLA can experience:

failure_modes = {
    "prediction_failure": {
        "symptom": "Actions are erratic or nonsensical",
        "detection": "Action entropy > threshold",
        "response": "Pause, request human input",
    },
    "execution_failure": {
        "symptom": "Robot can't reach commanded position",
        "detection": "Joint error > threshold for N steps",
        "response": "Retry with alternative approach",
    },
    "task_failure": {
        "symptom": "Object dropped, wrong item picked",
        "detection": "Vision-based task success estimator",
        "response": "Re-plan from current state",
    },
    "environment_change": {
        "symptom": "Scene differs from expectation",
        "detection": "Image similarity to training distribution",
        "response": "Re-observe, update plan",
    },
    "hardware_failure": {
        "symptom": "Sensor noise, motor stall, communication loss",
        "detection": "Hardware health monitoring",
        "response": "Emergency stop, alert operator",
    },
}
```

### 108.3 Out-of-Distribution Detection

VLAs encounter novel situations in deployment. Detecting OOD is critical:

$$\text{OOD score} = -\log p(x | \theta_\text{train})$$

Methods:
```
1. Embedding distance:
   d = ||encoder(x) - nearest_training_embedding||
   If d > threshold → OOD

2. Ensemble disagreement:
   σ = std([model_1(x), model_2(x), ..., model_K(x)])
   If σ > threshold → OOD

3. Reconstruction error:
   e = ||decoder(encoder(x)) - x||
   If e > threshold → OOD (x doesn't fit learned distribution)

4. Action entropy:
   H = -Σ p(a|x) log p(a|x)
   If H > threshold → VLA is uncertain
```

### 108.4 Human-in-the-Loop

```
Autonomy Levels:
  
  Level 0: Full teleoperation (human controls everything)
  Level 1: Shared control (human high-level, VLA low-level)
  Level 2: Supervised autonomy (VLA acts, human monitors)
  Level 3: Conditional autonomy (VLA asks when uncertain)
  Level 4: Full autonomy (VLA handles everything)

Most practical VLA deployments: Level 2-3

Level 3 Protocol:
  1. VLA generates action plan
  2. If confidence > 0.9: execute autonomously
  3. If 0.5 < confidence < 0.9: show plan to human, wait for approval
  4. If confidence < 0.5: pause, request human demonstration
  5. Learn from human corrections (online adaptation)
```

### 108.5 Monitoring Dashboard

```
┌─────────────────────────────────────────────────────────┐
│  VLA DEPLOYMENT DASHBOARD                                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  REAL-TIME                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Success  │  │ Latency  │  │ OOD      │              │
│  │ Rate     │  │ p95      │  │ Events   │              │
│  │  87.3%   │  │  42ms    │  │  3/hr    │              │
│  └──────────┘  └──────────┘  └──────────┘              │
│                                                          │
│  TRENDS (last 24h)                                       │
│  Success: ██████████████████████░░░░  87% (↑2%)         │
│  Latency: ████████░░░░░░░░░░░░░░░░░  42ms (stable)     │
│  OOD:     ██░░░░░░░░░░░░░░░░░░░░░░░  3/hr (↓1)        │
│  Errors:  █░░░░░░░░░░░░░░░░░░░░░░░░  2/day (stable)   │
│                                                          │
│  ALERTS                                                  │
│  ⚠ Success rate dropped below 85% at 14:32              │
│  ✓ Recovered after human correction at 14:35            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Implementation (60 min)

```python
import torch
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
import time

@dataclass
class SafetyConfig:
    max_joint_velocity: float = 1.0      # rad/s
    max_cartesian_velocity: float = 0.5  # m/s
    max_joint_torque: float = 50.0       # Nm
    confidence_threshold: float = 0.6
    ood_threshold: float = 3.0
    max_consecutive_failures: int = 3

class FailureDetector:
    """Multi-modal failure detection."""

    def __init__(self, config: SafetyConfig):
        self.config = config
        self.consecutive_failures = 0
        self.training_embeddings = None  # Set from training data

    def set_training_distribution(self, embeddings: torch.Tensor):
        """Store training embeddings for OOD detection."""
        self.training_embeddings = embeddings

    def check_action_confidence(self, action_dist):
        """Check if the model is confident about its action."""
        # For discrete: entropy
        if hasattr(action_dist, 'entropy'):
            entropy = action_dist.entropy().mean().item()
            return entropy < np.log(self.config.confidence_threshold)

        # For continuous: variance
        if hasattr(action_dist, 'variance'):
            var = action_dist.variance.mean().item()
            return var < 0.1

        return True  # Default: assume confident

    def check_ood(self, embedding: torch.Tensor) -> dict:
        """Out-of-distribution detection."""
        if self.training_embeddings is None:
            return {"is_ood": False, "score": 0}

        # Nearest neighbor distance
        distances = torch.cdist(
            embedding.unsqueeze(0),
            self.training_embeddings
        )
        min_dist = distances.min().item()

        return {
            "is_ood": min_dist > self.config.ood_threshold,
            "score": min_dist,
        }

    def check_execution(self, commanded, actual, threshold=0.1):
        """Check if robot executed the commanded action."""
        error = np.abs(commanded - actual).max()
        succeeded = error < threshold

        if not succeeded:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0

        return {
            "succeeded": succeeded,
            "error": error,
            "consecutive_failures": self.consecutive_failures,
            "should_stop": self.consecutive_failures >= self.config.max_consecutive_failures,
        }

class MotionSafetyFilter:
    """Real-time motion safety filtering."""

    def __init__(self, config: SafetyConfig, dt=0.02):
        self.config = config
        self.dt = dt
        self.prev_action = None

    def filter(self, action: np.ndarray) -> tuple:
        """Apply safety limits to action. Returns (safe_action, was_clipped)."""
        safe = action.copy()
        clipped = False

        # Velocity limiting
        if self.prev_action is not None:
            velocity = (safe - self.prev_action) / self.dt
            speed = np.linalg.norm(velocity)

            if speed > self.config.max_joint_velocity:
                scale = self.config.max_joint_velocity / speed
                safe = self.prev_action + velocity * scale * self.dt
                clipped = True

        self.prev_action = safe.copy()
        return safe, clipped

class DeploymentMonitor:
    """Track and report deployment metrics."""

    def __init__(self, window_size=1000):
        self.successes = deque(maxlen=window_size)
        self.latencies = deque(maxlen=window_size)
        self.ood_events = deque(maxlen=window_size)
        self.safety_clips = deque(maxlen=window_size)
        self.alerts = []

    def record_step(self, success: bool, latency_ms: float,
                    ood_score: float, was_clipped: bool):
        self.successes.append(1 if success else 0)
        self.latencies.append(latency_ms)
        self.ood_events.append(1 if ood_score > 3.0 else 0)
        self.safety_clips.append(1 if was_clipped else 0)

        # Alert conditions
        if len(self.successes) >= 100:
            rate = np.mean(list(self.successes)[-100:])
            if rate < 0.85:
                self.alerts.append(
                    f"[{time.strftime('%H:%M')}] Success rate dropped: "
                    f"{rate:.1%}"
                )

    def get_dashboard(self) -> dict:
        return {
            "success_rate": np.mean(list(self.successes)) if self.successes else 0,
            "latency_p50": np.percentile(list(self.latencies), 50) if self.latencies else 0,
            "latency_p95": np.percentile(list(self.latencies), 95) if self.latencies else 0,
            "ood_rate": np.mean(list(self.ood_events)) if self.ood_events else 0,
            "clip_rate": np.mean(list(self.safety_clips)) if self.safety_clips else 0,
            "recent_alerts": self.alerts[-5:],
        }

    def print_dashboard(self):
        d = self.get_dashboard()
        print(f"\n{'='*50}")
        print(f"  VLA DEPLOYMENT DASHBOARD")
        print(f"{'='*50}")
        print(f"  Success Rate:    {d['success_rate']:.1%}")
        print(f"  Latency (p50):   {d['latency_p50']:.1f} ms")
        print(f"  Latency (p95):   {d['latency_p95']:.1f} ms")
        print(f"  OOD Event Rate:  {d['ood_rate']:.1%}")
        print(f"  Safety Clip Rate:{d['clip_rate']:.1%}")
        if d['recent_alerts']:
            print(f"\n  ALERTS:")
            for a in d['recent_alerts']:
                print(f"  ⚠ {a}")
        print(f"{'='*50}")

# Demo
config = SafetyConfig()
detector = FailureDetector(config)
safety_filter = MotionSafetyFilter(config)
monitor = DeploymentMonitor()

# Simulate deployment
np.random.seed(42)
for step in range(200):
    # Simulated VLA output
    action = np.random.randn(7) * 0.5
    latency = np.random.normal(35, 5)
    ood_score = np.random.exponential(1.0)
    success = np.random.random() > 0.15

    # Safety filter
    safe_action, clipped = safety_filter.filter(action)

    # Execution check
    actual = safe_action + np.random.randn(7) * 0.05
    exec_result = detector.check_execution(safe_action, actual)

    # Monitor
    monitor.record_step(success, latency, ood_score, clipped)

monitor.print_dashboard()
```

---

## Exercise (45 min)

1. **OOD calibration:** Collect embeddings from 1000 training observations. Deploy on 200 in-distribution and 200 OOD observations. Plot the ROC curve. Find the threshold that gives 95% true positive rate.

2. **Safety filter impact:** Run 1000 episodes with and without the safety filter. Measure success rate and safety incidents. Does the safety filter reduce task performance?

3. **Failure recovery:** Implement a simple recovery policy: when confidence drops, switch to a conservative "retract" behavior. Compare with just pausing.

4. **Dashboard stress test:** Inject a gradual performance degradation (success rate drops 1% per hour). How quickly does the monitoring system detect it?

---

## Key Takeaways

1. **Five safety layers** from physical to task-level protect against different failure modes
2. **OOD detection** prevents the VLA from acting on unfamiliar inputs
3. **Motion safety filtering** enforces velocity/torque limits in real-time
4. **Human-in-the-loop** (Level 3 autonomy) is the practical deployment sweet spot
5. **Monitoring dashboards** track deployment health and trigger alerts

---

## Connection to the Thread

> *Single-robot safety is Day 108's lesson. Tomorrow: fleet-scale deployment — when you have 10, 100, or 1000 robots running the same VLA, fleet management, A/B testing, and continuous learning become critical.*

---

## Further Reading

- Haddadin et al. (2017), "Robot Collisions: A Survey on Detection, Isolation, and Identification"
- ISO 15066: Robots and robotic devices — Collaborative robots
- Amodei et al. (2016), "Concrete Problems in AI Safety"
