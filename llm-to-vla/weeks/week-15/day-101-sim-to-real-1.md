# Day 101: Sim-to-Real Transfer — Day 1: Fundamentals

> **Phase VII — VLAs: Architecture to Deployment | Week 15 | 2.5 hours**
> *"Simulation is where you get unlimited data. Reality is where data costs $100/hour. Bridging the gap is the engineering challenge." — Sim-to-Real Transfer*

## Navigation
- **Previous:** [Day 100: VLA Training Recipes](day-100-training-recipes.md)
- **Next:** [Day 102: Sim-to-Real Transfer — Day 2](day-102-sim-to-real-2.md)
- **Week:** [Week 15 Overview](README.md)
- **Phase:** [Phase VII: VLAs](../../study-notes/15-vla-architectures.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (60 min)

### 101.1 The Sim-to-Real Gap

```
Simulation:                     Reality:
  ├── Perfect physics              ├── Imperfect, complex physics
  ├── Clean observations           ├── Noisy sensors
  ├── Exact state                  ├── Partial observability
  ├── Instant reset                ├── Manual reset (minutes)
  ├── Parallel environments        ├── One robot at a time
  ├── Free data                    ├── $50-150/hour
  └── No safety concerns           └── Breaking things costs $$
```

### 101.2 Types of Sim-to-Real Gap

| Gap Type | Example | Transfer Difficulty |
|----------|---------|-------------------|
| **Visual** | Rendered vs real images | Medium (DR helps) |
| **Dynamics** | Joint friction, contact | Hard (SysID needed) |
| **Sensor** | Depth noise, latency | Medium (noise injection) |
| **Geometric** | Object shapes/sizes | Easy (mesh randomization) |
| **Task** | Success criteria mismatch | Easy (redefine reward) |

### 101.3 Domain Randomization (DR)

Make the simulation deliberately imperfect in random ways:

$$\pi^* = \arg\max_\pi \mathbb{E}_{\xi \sim P(\Xi)} \left[ J(\pi, \xi) \right]$$

where $\xi$ are randomized environment parameters.

```python
# Domain Randomization Parameters
randomization_params = {
    # Visual randomization
    "lighting": {
        "intensity": (0.3, 3.0),    # Ambient light
        "color_temp": (3000, 8000),  # Warm to cool
        "shadow_softness": (0, 1),
    },
    "camera": {
        "position_noise": 0.02,      # meters
        "orientation_noise": 5,       # degrees
        "fov": (55, 75),             # field of view
    },
    "texture": {
        "table_color": "random_rgb",
        "object_color": "random_rgb",
        "background": "random_image",
    },
    # Dynamics randomization
    "physics": {
        "friction_coeff": (0.5, 1.5),
        "mass_scale": (0.8, 1.2),
        "joint_damping": (0.9, 1.1),
        "action_delay": (0, 3),       # frames
    },
    # Sensor randomization
    "observation": {
        "image_noise_std": 0.02,
        "depth_noise_std": 0.005,
        "proprioception_noise_std": 0.01,
    },
}
```

### 101.4 System Identification (SysID)

Instead of randomizing, measure the real system and match simulation:

```
Real robot measurement:
  ├── Drop test → estimate friction, restitution
  ├── Free motion → estimate joint damping
  ├── Force/torque → estimate inertia
  └── Camera calibration → exact intrinsics/extrinsics

Sim parameter fitting:
  sim_params = argmin ||sim_trajectory(params) - real_trajectory||²
```

### 101.5 Progressive Transfer

Don't jump from sim to real. Transfer gradually:

```
Level 1: Simple sim (MuJoCo, basic rendering)
  ↓ Train base policy
Level 2: Realistic sim (Isaac Sim, PBR rendering)
  ↓ Fine-tune with visual realism
Level 3: Sim + real demos (mixed dataset)
  ↓ Co-train on both
Level 4: Real world (fine-tune with 50-100 demos)
  ↓ Final adaptation
Level 5: Deployed
```

---

## Implementation (60 min)

### Domain Randomization Framework

```python
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

@dataclass
class DomainRandomizationConfig:
    # Visual
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    hue_shift_range: Tuple[float, float] = (-0.1, 0.1)
    noise_std: float = 0.02

    # Dynamics
    friction_range: Tuple[float, float] = (0.5, 1.5)
    mass_scale_range: Tuple[float, float] = (0.8, 1.2)
    action_noise_std: float = 0.01
    action_delay_range: Tuple[int, int] = (0, 3)

    # Geometry
    object_scale_range: Tuple[float, float] = (0.9, 1.1)
    position_noise: float = 0.01

class VisualRandomizer:
    """Randomize image observations."""

    def __init__(self, config: DomainRandomizationConfig):
        self.config = config

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply visual domain randomization."""
        img = image.clone()

        # Brightness
        brightness = np.random.uniform(*self.config.brightness_range)
        img = img * brightness

        # Contrast
        contrast = np.random.uniform(*self.config.contrast_range)
        mean = img.mean()
        img = (img - mean) * contrast + mean

        # Additive noise
        noise = torch.randn_like(img) * self.config.noise_std
        img = img + noise

        # Color jitter (simplified)
        hue_shift = np.random.uniform(*self.config.hue_shift_range)
        img[:1] = img[:1] + hue_shift  # Shift first channel

        return torch.clamp(img, 0, 1)

class DynamicsRandomizer:
    """Randomize physics parameters."""

    def __init__(self, config: DomainRandomizationConfig):
        self.config = config
        self._action_buffer = []

    def randomize_physics(self):
        """Sample new physics parameters."""
        return {
            "friction": np.random.uniform(*self.config.friction_range),
            "mass_scale": np.random.uniform(*self.config.mass_scale_range),
            "action_delay": np.random.randint(*self.config.action_delay_range),
        }

    def apply_action_noise(self, action: np.ndarray) -> np.ndarray:
        """Add noise to action execution."""
        noise = np.random.normal(0, self.config.action_noise_std, action.shape)
        return action + noise

    def apply_action_delay(self, action: np.ndarray, delay: int) -> np.ndarray:
        """Simulate action execution delay."""
        self._action_buffer.append(action)
        if len(self._action_buffer) > delay:
            return self._action_buffer.pop(0)
        return np.zeros_like(action)  # No action until buffer fills

class SimToRealTrainer:
    """Training loop with domain randomization."""

    def __init__(self, model, dr_config=None):
        self.model = model
        self.config = dr_config or DomainRandomizationConfig()
        self.visual_dr = VisualRandomizer(self.config)
        self.dynamics_dr = DynamicsRandomizer(self.config)

    def augment_batch(self, batch):
        """Apply domain randomization to a batch."""
        images = batch["images"].clone()

        # Apply visual randomization per sample
        for i in range(images.shape[0]):
            images[i] = self.visual_dr(images[i])

        # Apply action noise
        actions = batch["actions"].clone()
        noise = torch.randn_like(actions) * self.config.action_noise_std
        actions = actions + noise

        return {"images": images, "actions": actions, **{
            k: v for k, v in batch.items() if k not in ("images", "actions")
        }}

    def train_step(self, batch):
        """Single training step with DR."""
        augmented = self.augment_batch(batch)
        loss = self.model.compute_loss(augmented)
        return loss

    def progressive_transfer(self, sim_data, real_data, n_stages=4):
        """Progressive sim-to-real transfer."""
        stages = [
            {"name": "Sim only",     "sim_ratio": 1.0, "dr_strength": 0.5},
            {"name": "Strong DR",    "sim_ratio": 1.0, "dr_strength": 1.0},
            {"name": "Mixed",        "sim_ratio": 0.7, "dr_strength": 0.8},
            {"name": "Real focused", "sim_ratio": 0.2, "dr_strength": 0.3},
        ]

        for stage in stages[:n_stages]:
            print(f"\nStage: {stage['name']}")
            print(f"  Sim ratio: {stage['sim_ratio']:.0%}")
            print(f"  DR strength: {stage['dr_strength']:.0%}")
            # In practice: train for N epochs with these settings

# Demo
config = DomainRandomizationConfig()
vis_dr = VisualRandomizer(config)

img = torch.rand(3, 64, 64)  # Simulated image
augmented = vis_dr(img)
print(f"Original range: [{img.min():.3f}, {img.max():.3f}]")
print(f"Augmented range: [{augmented.min():.3f}, {augmented.max():.3f}]")

dyn_dr = DynamicsRandomizer(config)
physics = dyn_dr.randomize_physics()
print(f"\nRandomized physics: {physics}")

action = np.array([0.1, -0.2, 0.05])
noisy_action = dyn_dr.apply_action_noise(action)
print(f"Original action: {action}")
print(f"Noisy action: {noisy_action}")
```

---

## Exercise (45 min)

1. **DR sweep:** Train a policy with no DR, mild DR, strong DR. Evaluate in a "real" environment (simulation with fixed realistic parameters). Plot success rate vs DR strength.

2. **Gap analysis:** Create a "real" simulation with specific friction=0.8, mass_scale=1.1, camera_noise=0.03. Train in "sim" with friction=1.0, mass_scale=1.0, no noise. Measure the performance drop. Then add DR and measure recovery.

3. **Visual vs dynamics DR:** Apply visual-only DR vs dynamics-only DR vs both. Which gap is harder to close?

4. **SysID simulation:** Measure the "real" simulation's physics parameters by running diagnostic trajectories. Set sim parameters to match. Compare SysID vs DR approaches.

---

## Key Takeaways

1. **Sim-to-real gap** has visual, dynamics, sensor, and geometric components
2. **Domain randomization** makes policies robust by training on diverse environments
3. **System identification** matches simulation to reality (complementary to DR)
4. **Progressive transfer** bridges the gap gradually instead of in one jump
5. **Visual DR** is easier than dynamics DR — cameras are easier to randomize than physics

---

## Connection to the Thread

> *Today covered the fundamentals: DR, SysID, progressive transfer. Tomorrow: advanced sim-to-real techniques — real-to-sim adaptation (NeRF-based), teacher-student distillation, and the specific approaches that RT-2, Octo, and π₀ use for real-world deployment.*

---

## Further Reading

- Tobin et al. (2017), "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World"
- Peng et al. (2018), "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization"
- James et al. (2019), "Sim-to-Real via Sim-to-Sim"
