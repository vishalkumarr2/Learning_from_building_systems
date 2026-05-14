# Exercise 13 — Sim-to-Real Transfer & Domain Adaptation
> Phase VII · Days 100-102 · ~8 hours · **Robotics-specific**

[← Exercise 12: Robot Manipulation](12-robot-manipulation-mujoco.md) | [Projects →](../projects/)

---

## Objectives

By completing this exercise you will:
- Understand the sim-to-real gap and why it matters for learned policies
- Implement visual and dynamics domain randomization
- Build a visual domain adaptation pipeline (simulated → "real-looking" images)
- Evaluate transfer robustness with systematic ablation
- Apply sim-to-real strategies used by RT-1, RT-2, and OpenVLA

## Prerequisites
- Exercise 12 (MuJoCo manipulation — custom envs, RL/BC training)
- Study Note 14 (Robot Data & Evaluation)
- Study Note 15 (VLA Architectures) — for context on real-world deployment

## Setup

```bash
pip install mujoco gymnasium torch torchvision numpy matplotlib \
    Pillow opencv-python albumentations scikit-image
```

---

## Part 1: Understanding the Sim-to-Real Gap (~1.5 hours)

### 1.1 — Measuring the Gap

The sim-to-real gap has two components:
1. **Visual gap**: simulated images look different from real camera feeds
2. **Dynamics gap**: simulated physics doesn't perfectly match real robots

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ── Simulated vs "Real" dynamics comparison ──
# We simulate the "real world" as a different MuJoCo config with:
# - Different friction coefficients
# - Different object masses
# - Sensor noise
# - Actuator delay

class SimEnv:
    """Idealized simulation: perfect physics, no noise."""
    def __init__(self):
        self.friction = 1.0
        self.mass = 0.05
        self.noise_std = 0.0
        self.actuator_delay = 0  # steps
    
    def step(self, pos, action, dt=0.01):
        """Simple 2D point mass dynamics."""
        force = action * self.friction
        accel = force / self.mass
        new_pos = pos + accel * dt
        # Add observation noise
        obs_pos = new_pos + np.random.normal(0, self.noise_std, size=2)
        return new_pos, obs_pos


class RealEnv:
    """'Real world' with different dynamics and noise."""
    def __init__(self):
        self.friction = 0.7     # lower friction than sim
        self.mass = 0.08        # heavier than sim
        self.noise_std = 0.005  # sensor noise
        self.actuator_delay = 2  # 2-step delay
        self._action_buffer = []
    
    def step(self, pos, action, dt=0.01):
        """Same interface, different dynamics."""
        self._action_buffer.append(action.copy())
        # Apply delayed action
        if len(self._action_buffer) > self.actuator_delay:
            effective_action = self._action_buffer.pop(0)
        else:
            effective_action = np.zeros(2)
        
        force = effective_action * self.friction
        accel = force / self.mass
        new_pos = pos + accel * dt
        obs_pos = new_pos + np.random.normal(0, self.noise_std, size=2)
        return new_pos, obs_pos


def run_trajectory(env, policy_fn, goal, n_steps=200):
    """Run a policy and record trajectory."""
    pos = np.array([0.0, 0.0])
    trajectory = [pos.copy()]
    
    for step in range(n_steps):
        obs_pos = pos + np.random.normal(0, getattr(env, 'noise_std', 0), size=2)
        action = policy_fn(obs_pos, goal)
        pos, _ = env.step(pos, action)
        trajectory.append(pos.copy())
    
    return np.array(trajectory)


# P-controller tuned for sim
def sim_tuned_policy(obs, goal, kp=5.0):
    """P-controller tuned on simulator."""
    error = goal - obs
    return np.clip(error * kp, -1, 1)


goal = np.array([1.0, 0.5])

# Run same policy in both environments
sim_traj = run_trajectory(SimEnv(), sim_tuned_policy, goal)
real_traj = run_trajectory(RealEnv(), sim_tuned_policy, goal)

# Measure gap
sim_final_error = np.linalg.norm(sim_traj[-1] - goal)
real_final_error = np.linalg.norm(real_traj[-1] - goal)

print(f"=== Sim-to-Real Gap ===")
print(f"Sim  final error: {sim_final_error:.4f}")
print(f"Real final error: {real_final_error:.4f}")
print(f"Gap (ratio):      {real_final_error / max(sim_final_error, 1e-6):.1f}x worse")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(sim_traj[:, 0], sim_traj[:, 1], 'b-', label='Sim trajectory')
axes[0].plot(real_traj[:, 0], real_traj[:, 1], 'r-', label='Real trajectory')
axes[0].scatter(*goal, c='green', s=100, marker='*', label='Goal')
axes[0].set_title('Trajectories: Sim vs Real')
axes[0].legend()
axes[0].set_aspect('equal')

# Error over time
sim_errors = np.linalg.norm(sim_traj - goal, axis=1)
real_errors = np.linalg.norm(real_traj - goal, axis=1)
axes[1].plot(sim_errors, 'b-', label='Sim error')
axes[1].plot(real_errors, 'r-', label='Real error')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('Distance to goal')
axes[1].set_title('Tracking Error Over Time')
axes[1].legend()

plt.tight_layout()
plt.savefig('sim_to_real_gap.png', dpi=150)
plt.show()
print("Saved: sim_to_real_gap.png")
```

### 1.2 — Categorizing Sim-to-Real Challenges

```python
# ── YOUR TASK ──
# Fill in this table from your understanding of the sim-to-real problem:

sim2real_challenges = """
| Challenge          | Description                              | Mitigation Strategy        |
|-------------------|------------------------------------------|---------------------------|
| Visual appearance | Textures, lighting, shadows differ       | ___                       |
| Object properties | Mass, friction, deformability            | ___                       |
| Sensor noise      | Camera noise, depth errors, latency      | ___                       |
| Actuator dynamics | Motor delays, backlash, compliance       | ___                       |
| Contact physics   | Rigid vs soft contact, friction cones    | ___                       |
| Calibration       | Camera extrinsics, kinematic errors      | ___                       |

Common strategies:
1. Domain Randomization (DR): ___  (describe in 1 sentence)
2. System Identification (SysID): ___
3. Domain Adaptation (DA): ___
4. Sim-to-Real Fine-tuning: ___
"""
print(sim2real_challenges)
```

---

## Part 2: Visual Domain Randomization (~2 hours)

### 2.1 — Image-Level Augmentations

```python
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def render_sim_image(width=224, height=224):
    """
    Generate a synthetic 'simulated' robot scene.
    Uses simple shapes to represent a robot arm workspace.
    """
    img = np.ones((height, width, 3), dtype=np.uint8) * 200  # gray background
    
    # Table surface
    img[140:160, 20:204] = [139, 119, 101]  # brown table
    
    # Robot arm (simplified as rectangles)
    import cv2
    # Base
    cv2.rectangle(img, (90, 60), (134, 140), (80, 80, 80), -1)
    # Gripper
    cv2.rectangle(img, (100, 40), (124, 65), (100, 100, 100), -1)
    # Gripper fingers
    cv2.line(img, (102, 40), (102, 30), (120, 120, 120), 2)
    cv2.line(img, (122, 40), (122, 30), (120, 120, 120), 2)
    
    # Object (red cube)
    cv2.rectangle(img, (150, 130), (170, 150), (0, 0, 220), -1)
    
    # Goal marker (green circle)
    cv2.circle(img, (60, 100), 10, (0, 200, 0), -1)
    
    return img


def create_visual_randomization_pipeline():
    """
    Visual domain randomization transforms.
    These simulate the visual gap between sim and real.
    """
    return T.Compose([
        T.RandomApply([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1)
        ], p=0.8),
        T.RandomApply([
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.3),
        T.RandomGrayscale(p=0.05),
        T.RandomApply([
            T.RandomAffine(degrees=3, translate=(0.05, 0.05), scale=(0.95, 1.05))
        ], p=0.5),
        T.RandomErasing(p=0.1, scale=(0.02, 0.1)),
    ])


# Generate and augment
sim_img = render_sim_image()
pil_img = Image.fromarray(sim_img)
augment = create_visual_randomization_pipeline()

# Show original + 8 augmented versions
fig, axes = plt.subplots(3, 3, figsize=(12, 12))
axes[0, 0].imshow(sim_img)
axes[0, 0].set_title("Original (Sim)")

for i in range(1, 9):
    ax = axes[i // 3, i % 3]
    aug_tensor = augment(T.ToTensor()(pil_img))
    aug_img = (aug_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    ax.imshow(aug_img)
    ax.set_title(f"Augmented #{i}")

for ax in axes.flat:
    ax.axis('off')

plt.suptitle("Visual Domain Randomization", fontsize=14)
plt.tight_layout()
plt.savefig('visual_domain_randomization.png', dpi=150)
plt.show()
print("Saved: visual_domain_randomization.png")
```

### 2.2 — Advanced Augmentation with Albumentations

```python
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("Install albumentations for advanced augmentation: pip install albumentations")

if HAS_ALBUMENTATIONS:
    # Robotics-specific augmentation pipeline
    robot_augment = A.Compose([
        # Lighting variation (simulates different warehouse lighting)
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        
        # Color shifts (simulates different cameras)
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        A.ChannelShuffle(p=0.05),  # rare but tests color robustness
        
        # Sensor noise
        A.GaussNoise(var_limit=(5, 30), p=0.5),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=0.3),
        
        # Optical effects
        A.MotionBlur(blur_limit=5, p=0.2),  # camera shake
        A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.15, p=0.1),
        
        # Geometric (small — simulates camera mounting variations)
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=3, p=0.3),
        
        # Occlusions (simulates partial object blocking)
        A.CoarseDropout(max_holes=3, max_height=20, max_width=20,
                        fill_value=128, p=0.2),
    ])
    
    # Apply and visualize
    sim_img = render_sim_image()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(sim_img)
    axes[0, 0].set_title("Original")
    
    for i in range(1, 8):
        augmented = robot_augment(image=sim_img)["image"]
        ax = axes[i // 4, i % 4]
        ax.imshow(augmented)
        ax.set_title(f"Aug #{i}")
    
    for ax in axes.flat:
        ax.axis('off')
    plt.suptitle("Robotics-Specific Visual Augmentations", fontsize=14)
    plt.tight_layout()
    plt.savefig('robot_augmentations.png', dpi=150)
    plt.show()
```

---

## Part 3: Dynamics Domain Randomization (~1.5 hours)

### 3.1 — Systematic Parameter Randomization

```python
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field

@dataclass
class DynamicsParams:
    """Dynamics parameters that can be randomized."""
    mass: float = 0.05           # object mass (kg)
    friction: float = 1.0        # friction coefficient
    damping: float = 0.1         # joint damping
    actuator_noise: float = 0.0  # action noise std
    obs_noise: float = 0.0       # observation noise std
    latency_steps: int = 0       # actuator delay (steps)
    gravity_z: float = -9.81     # gravity


def sample_randomized_params(rng=None):
    """Sample randomized dynamics for training robustness."""
    if rng is None:
        rng = np.random.default_rng()
    
    return DynamicsParams(
        mass=rng.uniform(0.02, 0.15),          # 0.4x to 3x default
        friction=rng.uniform(0.3, 2.0),         # wide range
        damping=rng.uniform(0.05, 0.3),
        actuator_noise=rng.uniform(0.0, 0.05),  # 0-5% noise
        obs_noise=rng.uniform(0.0, 0.01),       # 0-1% noise
        latency_steps=rng.integers(0, 4),       # 0-3 steps delay
        gravity_z=rng.uniform(-10.5, -9.0),     # ±8% gravity variation
    )


class ParameterizedPointMassEnv:
    """2D point mass with configurable dynamics."""
    
    def __init__(self, params=None):
        self.params = params or DynamicsParams()
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self.goal = np.array([1.0, 0.5])
        self._action_buffer = []
    
    def reset(self, goal=None):
        self.pos = np.zeros(2)
        self.vel = np.zeros(2)
        self._action_buffer = []
        if goal is not None:
            self.goal = goal.copy()
        return self._get_obs()
    
    def _get_obs(self):
        noisy_pos = self.pos + np.random.normal(0, self.params.obs_noise, size=2)
        noisy_vel = self.vel + np.random.normal(0, self.params.obs_noise, size=2)
        return np.concatenate([noisy_pos, noisy_vel, self.goal])
    
    def step(self, action, dt=0.02):
        # Add actuator noise
        noisy_action = action + np.random.normal(0, self.params.actuator_noise, size=2)
        noisy_action = np.clip(noisy_action, -1, 1)
        
        # Handle latency
        self._action_buffer.append(noisy_action.copy())
        if len(self._action_buffer) > self.params.latency_steps:
            effective_action = self._action_buffer.pop(0)
        else:
            effective_action = np.zeros(2)
        
        # Physics
        force = effective_action * self.params.friction
        accel = force / self.params.mass
        accel[1] += self.params.gravity_z * 0.001  # small gravity effect in 2D
        
        self.vel = self.vel * (1 - self.params.damping) + accel * dt
        self.pos = self.pos + self.vel * dt
        
        dist = np.linalg.norm(self.pos - self.goal)
        reward = -dist
        done = dist < 0.05
        
        return self._get_obs(), reward, done, {"distance": dist}


# ── Training with Domain Randomization ──

class SimplePolicy(nn.Module):
    def __init__(self, obs_dim=6, act_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )
    
    def forward(self, obs):
        return self.net(obs)


def train_bc_with_randomization(use_randomization=False, n_demos=300, n_epochs=100):
    """Train BC policy with optional dynamics randomization."""
    # Collect demonstrations across randomized dynamics
    all_obs, all_acts = [], []
    
    for _ in range(n_demos):
        if use_randomization:
            params = sample_randomized_params()
        else:
            params = DynamicsParams()  # default
        
        env = ParameterizedPointMassEnv(params)
        goal = np.array([np.random.uniform(0.5, 1.5), np.random.uniform(-0.5, 0.5)])
        obs = env.reset(goal)
        
        for step in range(100):
            # Expert: P-controller (adapts to any dynamics)
            error = obs[4:6] - obs[:2]  # goal - pos
            action = np.clip(error * 3.0, -1, 1)
            
            all_obs.append(obs.copy())
            all_acts.append(action.copy())
            
            obs, _, done, _ = env.step(action)
            if done:
                break
    
    # Train
    obs_t = torch.FloatTensor(np.array(all_obs))
    act_t = torch.FloatTensor(np.array(all_acts))
    
    policy = SimplePolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    
    for epoch in range(n_epochs):
        idx = torch.randperm(len(obs_t))[:512]
        pred = policy(obs_t[idx])
        loss = nn.functional.mse_loss(pred, act_t[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return policy


def evaluate_transfer(policy, test_params_list, n_episodes=20):
    """Evaluate policy across different dynamics configurations."""
    results = []
    for name, params in test_params_list:
        successes = 0
        total_dist = 0
        for _ in range(n_episodes):
            env = ParameterizedPointMassEnv(params)
            goal = np.array([1.0, 0.5])
            obs = env.reset(goal)
            
            for step in range(200):
                with torch.no_grad():
                    action = policy(torch.FloatTensor(obs).unsqueeze(0)).squeeze().numpy()
                obs, _, done, info = env.step(action)
                if done:
                    break
            
            successes += done
            total_dist += info["distance"]
        
        avg_dist = total_dist / n_episodes
        results.append((name, successes / n_episodes, avg_dist))
        print(f"  {name:25s}: success={successes/n_episodes:.0%}, avg_dist={avg_dist:.4f}")
    
    return results


# Train two policies
print("Training WITHOUT domain randomization...")
policy_no_dr = train_bc_with_randomization(use_randomization=False)
print("Training WITH domain randomization...")
policy_with_dr = train_bc_with_randomization(use_randomization=True)

# Define test configurations
test_configs = [
    ("Default (same as train)", DynamicsParams()),
    ("Heavy object (3x)",       DynamicsParams(mass=0.15)),
    ("Low friction (0.3x)",     DynamicsParams(friction=0.3)),
    ("Noisy sensors",           DynamicsParams(obs_noise=0.01, actuator_noise=0.05)),
    ("High latency (3 steps)",  DynamicsParams(latency_steps=3)),
    ("Combined perturbation",   DynamicsParams(mass=0.1, friction=0.5,
                                               obs_noise=0.008, latency_steps=2)),
]

print("\n=== Policy WITHOUT Domain Randomization ===")
results_no_dr = evaluate_transfer(policy_no_dr, test_configs)

print("\n=== Policy WITH Domain Randomization ===")
results_with_dr = evaluate_transfer(policy_with_dr, test_configs)

# Compare
print("\n=== Transfer Gap Summary ===")
print(f"{'Config':25s} | {'No DR':>8s} | {'With DR':>8s} | {'Δ':>8s}")
print("-" * 60)
for (name, s1, _), (_, s2, _) in zip(results_no_dr, results_with_dr):
    delta = s2 - s1
    print(f"{name:25s} | {s1:7.0%} | {s2:7.0%} | {delta:+7.0%}")
```

---

## Part 4: Visual Domain Adaptation (~1.5 hours)

### 4.1 — Feature-Level Alignment

```python
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np

class SimpleFeatureExtractor(nn.Module):
    """Shared vision backbone for domain adaptation."""
    def __init__(self, feature_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, feature_dim)
    
    def forward(self, x):
        features = self.conv(x).flatten(1)
        return self.fc(features)


class DomainDiscriminator(nn.Module):
    """Predicts whether features come from sim or real domain."""
    def __init__(self, feature_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
    
    def forward(self, features):
        return self.net(features)


class GradientReversal(torch.autograd.Function):
    """Gradient Reversal Layer (GRL) for adversarial domain adaptation."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def generate_sim_batch(batch_size=32, img_size=64):
    """Generate batch of simulated images (simple shapes on gray bg)."""
    images = []
    for _ in range(batch_size):
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 180
        # Random colored rectangle (object)
        x1, y1 = np.random.randint(10, 40, size=2)
        x2, y2 = x1 + np.random.randint(8, 20), y1 + np.random.randint(8, 20)
        color = np.random.randint(50, 200, size=3)
        img[y1:y2, x1:x2] = color
        images.append(img)
    return torch.FloatTensor(np.array(images)).permute(0, 3, 1, 2) / 255.0


def generate_real_batch(batch_size=32, img_size=64):
    """Generate batch of 'real' images (same shapes but with noise + lighting)."""
    images = generate_sim_batch(batch_size, img_size)
    # Add realistic perturbations
    noise = torch.randn_like(images) * 0.05
    brightness = torch.rand(batch_size, 1, 1, 1) * 0.4 + 0.8
    images = (images * brightness + noise).clamp(0, 1)
    return images


# ── Domain-Adversarial Training ──
feature_extractor = SimpleFeatureExtractor()
domain_disc = DomainDiscriminator()
task_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))  # predict object position

opt_features = torch.optim.Adam(feature_extractor.parameters(), lr=1e-3)
opt_disc = torch.optim.Adam(domain_disc.parameters(), lr=1e-3)
opt_task = torch.optim.Adam(task_head.parameters(), lr=1e-3)

for epoch in range(100):
    # Generate data
    sim_images = generate_sim_batch(32)
    real_images = generate_real_batch(32)
    
    # Fake labels (object position from image center — simplified)
    sim_labels = torch.rand(32, 2) * 0.5 + 0.25  # random positions
    
    # Extract features
    sim_features = feature_extractor(sim_images)
    real_features = feature_extractor(real_images)
    
    # Task loss (only on sim — we have labels there)
    task_pred = task_head(sim_features)
    task_loss = nn.functional.mse_loss(task_pred, sim_labels)
    
    # Domain discrimination loss
    sim_domain = domain_disc(sim_features.detach())
    real_domain = domain_disc(real_features.detach())
    disc_loss = (
        nn.functional.binary_cross_entropy(sim_domain, torch.zeros_like(sim_domain)) +
        nn.functional.binary_cross_entropy(real_domain, torch.ones_like(real_domain))
    ) / 2
    
    # Adversarial loss (confuse the discriminator via GRL)
    alpha = min(1.0, epoch / 50)  # ramp up adversarial signal
    sim_features_grl = GradientReversal.apply(sim_features, alpha)
    real_features_grl = GradientReversal.apply(real_features, alpha)
    adv_sim = domain_disc(sim_features_grl)
    adv_real = domain_disc(real_features_grl)
    adv_loss = (
        nn.functional.binary_cross_entropy(adv_sim, torch.zeros_like(adv_sim)) +
        nn.functional.binary_cross_entropy(adv_real, torch.ones_like(adv_real))
    ) / 2
    
    # Combined update
    total_loss = task_loss + 0.1 * adv_loss
    
    opt_features.zero_grad()
    opt_task.zero_grad()
    total_loss.backward()
    opt_features.step()
    opt_task.step()
    
    # Update discriminator separately
    opt_disc.zero_grad()
    disc_loss.backward()
    opt_disc.step()
    
    if (epoch + 1) % 20 == 0:
        # Check domain confusion
        with torch.no_grad():
            sim_f = feature_extractor(generate_sim_batch(100))
            real_f = feature_extractor(generate_real_batch(100))
            sim_d = domain_disc(sim_f).mean().item()
            real_d = domain_disc(real_f).mean().item()
        
        print(f"Epoch {epoch+1:3d} | task_loss={task_loss:.4f} | "
              f"disc_loss={disc_loss:.4f} | "
              f"sim_conf={sim_d:.2f} | real_conf={real_d:.2f}")
        # When sim_conf ≈ real_conf ≈ 0.5, features are domain-invariant

print("\n✓ Domain-invariant features: discriminator should be near chance (0.5)")
```

### 4.2 — Measuring Feature Alignment

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Visualize feature alignment with PCA
with torch.no_grad():
    sim_features = feature_extractor(generate_sim_batch(200)).numpy()
    real_features = feature_extractor(generate_real_batch(200)).numpy()

all_features = np.concatenate([sim_features, real_features])
pca = PCA(n_components=2)
projected = pca.fit_transform(all_features)

plt.figure(figsize=(8, 6))
plt.scatter(projected[:200, 0], projected[:200, 1], c='blue', alpha=0.5, label='Sim')
plt.scatter(projected[200:, 0], projected[200:, 1], c='red', alpha=0.5, label='Real')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Feature Space: Sim vs Real (After Domain Adaptation)')
plt.legend()
plt.savefig('domain_adaptation_features.png', dpi=150)
plt.show()

# ── YOUR TASK ──
# 1. Train the feature extractor WITHOUT adversarial loss — visualize features
# 2. Train WITH adversarial loss — visualize features
# 3. The features should overlap more after adversarial training
# 4. Compute MMD (Maximum Mean Discrepancy) between domains as a metric:

def compute_mmd(x, y, kernel='rbf', sigma=1.0):
    """Maximum Mean Discrepancy between two distributions."""
    def rbf_kernel(a, b, sigma):
        dist = torch.cdist(a, b)
        return torch.exp(-dist ** 2 / (2 * sigma ** 2))
    
    x, y = torch.FloatTensor(x), torch.FloatTensor(y)
    xx = rbf_kernel(x, x, sigma).mean()
    yy = rbf_kernel(y, y, sigma).mean()
    xy = rbf_kernel(x, y, sigma).mean()
    return (xx + yy - 2 * xy).item()

mmd = compute_mmd(sim_features, real_features)
print(f"MMD between sim and real features: {mmd:.6f}")
print("Lower MMD = better domain alignment")
```

---

## Part 5: Sim-to-Real Strategies Used by VLAs (~1.5 hours)

### 5.1 — How Real VLA Systems Handle Sim-to-Real

```python
# ── YOUR TASK ──
# Research and fill in how each system handles sim-to-real:

vla_sim2real = """
╔═══════════════╦══════════════════════════════════════════════════════╗
║ System        ║ Sim-to-Real Strategy                                ║
╠═══════════════╬══════════════════════════════════════════════════════╣
║ RT-1          ║ No sim — trained on 130K REAL demos only.           ║
║               ║ Real data diversity acts as implicit randomization. ║
╠═══════════════╬══════════════════════════════════════════════════════╣
║ RT-2          ║ ___  (How does web-scale pretraining help?)        ║
╠═══════════════╬══════════════════════════════════════════════════════╣
║ Octo          ║ ___  (Open X-Embodiment — multi-robot data)       ║
╠═══════════════╬══════════════════════════════════════════════════════╣
║ OpenVLA       ║ ___  (Prismatic VLM backbone + Open X)            ║
╠═══════════════╬══════════════════════════════════════════════════════╣
║ π₀            ║ ___  (Flow matching — how does this help?)        ║
╠═══════════════╬══════════════════════════════════════════════════════╣
║ Diffusion     ║ ___  (Chi et al. 2023 — what sim envs did they    ║
║ Policy        ║       use, and how did they transfer?)             ║
╠═══════════════╬══════════════════════════════════════════════════════╣
║ GR00T N1      ║ ___  (NVIDIA — Isaac Sim + real data)             ║
╚═══════════════╩══════════════════════════════════════════════════════╝
"""
print(vla_sim2real)
```

### 5.2 — Build Your Own Transfer Pipeline

```python
import torch
import torch.nn as nn
import numpy as np

class SimToRealPipeline:
    """
    Complete sim-to-real transfer pipeline combining:
    1. Visual domain randomization (image augmentation)
    2. Dynamics randomization (physics variation)  
    3. Feature alignment (optional domain adaptation)
    """
    
    def __init__(self, visual_dr=True, dynamics_dr=True, domain_adapt=False):
        self.visual_dr = visual_dr
        self.dynamics_dr = dynamics_dr
        self.domain_adapt = domain_adapt
        
        if visual_dr:
            import torchvision.transforms as T
            self.augment = T.Compose([
                T.ColorJitter(0.3, 0.3, 0.2, 0.1),
                T.GaussianBlur(3, sigma=(0.1, 1.5)),
                T.RandomErasing(p=0.1),
            ])
    
    def randomize_dynamics(self):
        """Return randomized dynamics parameters."""
        return sample_randomized_params()
    
    def augment_observation(self, obs_image):
        """Apply visual randomization to observation image."""
        if self.visual_dr and obs_image.dim() == 3:
            return self.augment(obs_image)
        return obs_image
    
    def collect_training_data(self, expert_fn, n_episodes=500, img_size=64):
        """Collect demonstrations with randomization."""
        all_obs, all_acts, all_imgs = [], [], []
        
        for ep in range(n_episodes):
            if self.dynamics_dr:
                params = self.randomize_dynamics()
            else:
                params = DynamicsParams()
            
            env = ParameterizedPointMassEnv(params)
            goal = np.array([
                np.random.uniform(0.5, 1.5),
                np.random.uniform(-0.5, 0.5)
            ])
            obs = env.reset(goal)
            
            for step in range(100):
                action = expert_fn(obs)
                all_obs.append(obs.copy())
                all_acts.append(action.copy())
                
                # Generate synthetic image observation
                img = self._state_to_image(obs, img_size)
                if self.visual_dr:
                    img = self.augment(img)
                all_imgs.append(img)
                
                obs, _, done, _ = env.step(action)
                if done:
                    break
        
        return {
            "states": np.array(all_obs),
            "actions": np.array(all_acts),
            "images": torch.stack(all_imgs),
        }
    
    def _state_to_image(self, obs, size=64):
        """Convert state to a simple image representation."""
        img = torch.ones(3, size, size) * 0.7
        # Draw position as a dot
        px = int(np.clip(obs[0] / 2 * size, 0, size - 1))
        py = int(np.clip((obs[1] + 1) / 2 * size, 0, size - 1))
        img[0, max(0,py-2):py+3, max(0,px-2):px+3] = 0.0  # red dot
        img[1, max(0,py-2):py+3, max(0,px-2):px+3] = 0.0
        # Draw goal
        gx = int(np.clip(obs[4] / 2 * size, 0, size - 1))
        gy = int(np.clip((obs[5] + 1) / 2 * size, 0, size - 1))
        img[1, max(0,gy-2):gy+3, max(0,gx-2):gx+3] = 1.0  # green dot
        img[0, max(0,gy-2):gy+3, max(0,gx-2):gx+3] = 0.0
        img[2, max(0,gy-2):gy+3, max(0,gx-2):gx+3] = 0.0
        return img


# ── YOUR TASK ──
# Run the full ablation:

def expert_policy(obs):
    error = obs[4:6] - obs[:2]
    return np.clip(error * 3.0, -1, 1).astype(np.float32)

configs = [
    ("No DR",            SimToRealPipeline(visual_dr=False, dynamics_dr=False)),
    ("Visual DR only",   SimToRealPipeline(visual_dr=True,  dynamics_dr=False)),
    ("Dynamics DR only", SimToRealPipeline(visual_dr=False, dynamics_dr=True)),
    ("Full DR",          SimToRealPipeline(visual_dr=True,  dynamics_dr=True)),
]

# For each config:
# 1. Collect training data
# 2. Train a BC policy
# 3. Evaluate on "real" dynamics (heavy object, low friction, noisy)
# 4. Record success rate

# Expected ranking: Full DR > Dynamics DR > Visual DR > No DR
# (for this task, dynamics matters more than visuals since it's state-based)

print("Run the ablation and fill in the results table:")
print(f"{'Config':20s} | {'Train Success':>14s} | {'Transfer Success':>16s}")
print("-" * 60)
for name, pipeline in configs:
    print(f"{name:20s} |     ___        |      ___")
```

---

## Checklist

- [ ] Measured sim-to-real gap with P-controller (dynamics mismatch)
- [ ] Filled in sim-to-real challenges table
- [ ] Applied visual domain randomization (torchvision + albumentations)
- [ ] Implemented dynamics domain randomization with parameter sampling
- [ ] Trained BC with and without dynamics randomization
- [ ] Evaluated transfer across 6 dynamics configurations
- [ ] Implemented adversarial domain adaptation (GRL + discriminator)
- [ ] Visualized feature alignment with PCA (before/after DA)
- [ ] Computed MMD metric for domain alignment
- [ ] Filled in VLA sim-to-real strategies table
- [ ] Built complete SimToRealPipeline with visual + dynamics DR
- [ ] Ran ablation: No DR vs Visual vs Dynamics vs Full

## Expected Results
- Sim-tuned policy: >95% success in sim, <50% in "real"
- Dynamics DR: recovers to >70% in "real"
- Full DR (visual + dynamics): >75% in "real"
- Domain adaptation: MMD drops 50%+ after adversarial training
- VLA strategy table: shows shift from sim-heavy (Diffusion Policy) to real-data-heavy (RT-1/2)

---

> **Connection to the Compression Thread**: Sim-to-real transfer asks: can a policy
> compress sim experience into knowledge that generalizes to reality? Domain
> randomization works because it forces the policy to compress *invariant* features —
> the underlying task structure — rather than memorizing specific dynamics. This is
> exactly the "compression = generalization" principle from Note 05: a model that
> truly compresses its training distribution will generalize to nearby distributions.

[← Exercise 12: Robot Manipulation](12-robot-manipulation-mujoco.md) | [Projects →](../projects/)
