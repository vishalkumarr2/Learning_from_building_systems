# Project 06 — Diffusion Policy for Manipulation

> Phase VI Capstone · Days 90–91 · ~10 hours
> Prerequisites: Study Notes 11–14, Exercises 09–10

---

## Objective

Train and evaluate a Diffusion Policy on a manipulation task using the LeRobot library.
Compare with behavioral cloning (BC) and ACT baselines. Produce a quantitative comparison
report with visualizations of the action generation process.

---

## Task

Choose ONE of the following manipulation tasks from LeRobot:

| Task | Dataset | Difficulty | Action Dim |
|------|---------|-----------|-----------|
| **PushT** (recommended) | `lerobot/pusht` | Easy | 2 |
| ALOHA Transfer Cube | `lerobot/aloha_sim_transfer_cube_human` | Medium | 14 |
| xArm Lift | `lerobot/xarm_lift_medium` | Medium | 4 |

**PushT is recommended** for the first attempt — it's 2D, fast to train, and provides
clear visualizations.

---

## Deliverables

### 1. Trained Policies (3 models)

| Model | Architecture | Training |
|-------|-------------|---------|
| **BC baseline** | MLP, single-step | MSE on (obs, action) pairs |
| **ACT** | CVAE + Transformer/MLP | ELBO loss, chunk_size=16 |
| **Diffusion Policy** | Conditional U-Net/MLP | Noise prediction, K=50 steps |

### 2. Quantitative Comparison (100 evaluation episodes each)

| Metric | BC | ACT | Diffusion Policy |
|--------|----|----|-----------------|
| Success rate (%) | | | |
| Mean final distance to target | | | |
| Trajectory smoothness (jerk) | | | |
| Inference latency (ms/action) | | | |

### 3. Visualizations

- **Learning curves**: training loss vs steps for all three methods
- **Action generation process**: for Diffusion Policy, show denoising steps
  (noise → intermediate → final action) for a single observation
- **Trajectory comparison**: overlay trajectories from all three methods on the same
  task instance
- **Multimodality test**: for the same observation, sample 20 actions from ACT and
  Diffusion Policy — do they capture different modes?

### 4. Comparison Report (1–2 pages)

Write a short report covering:
1. Task description and data
2. Architecture and training details for each method
3. Quantitative results table
4. Qualitative analysis: when does each method succeed/fail?
5. Key insight: which method and why?

---

## Implementation Guide

### Step 1: Data Loading

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("lerobot/pusht")

# Explore the dataset
print(f"Episodes: {dataset.num_episodes}")
print(f"Frames: {dataset.num_frames}")
print(f"FPS: {dataset.fps}")
print(f"Features: {list(dataset.features.keys())}")

# Access a sample
sample = dataset[0]
print(f"Observation shape: {sample['observation.image'].shape}")
print(f"State shape: {sample['observation.state'].shape}")
print(f"Action shape: {sample['action'].shape}")
```

### Step 2: Train BC Baseline

```python
# Use LeRobot's built-in training or implement your own
# If implementing from scratch:
# 1. Create DataLoader from dataset
# 2. Train MLP: observation.state → action (MSE loss)
# 3. For image-based: observation.image → CNN → MLP → action
```

### Step 3: Train ACT

```python
# Use LeRobot: policy="act"
# Or implement simplified version from Exercise 10
# Key hyperparameters:
#   chunk_size: 16 (for PushT, try 50-100)
#   z_dim: 32
#   beta: 0.01
#   hidden_dim: 256
```

### Step 4: Train Diffusion Policy

```python
# Use LeRobot: policy="diffusion"
# Or implement from Exercise 10
# Key hyperparameters:
#   num_diffusion_steps: 50-100
#   num_inference_steps: 10 (DDIM)
#   action_horizon: 16
#   observation_horizon: 2
```

### Step 5: Evaluate

```python
# For each trained policy:
# 1. Run 100 episodes in the environment
# 2. Record: success/failure, final distance, trajectory
# 3. Compute statistics: mean, std error, 95% CI

results = {}
for name, policy in [("BC", bc_model), ("ACT", act_model), ("DP", dp_model)]:
    successes, distances, trajectories = [], [], []
    for ep in range(100):
        # Run episode
        # Record metrics
        pass
    results[name] = {
        "success_rate": np.mean(successes),
        "success_se": np.std(successes) / np.sqrt(len(successes)),
        "mean_distance": np.mean(distances),
    }
```

### Step 6: Visualize Denoising Process

```python
# For Diffusion Policy: show the iterative denoising
# Save intermediate action predictions at steps K, K*0.75, K*0.5, K*0.25, 0
# Visualize how the random noise converges to a valid action trajectory

# This is the key visualization for understanding diffusion:
# Random noise → structured motion plan in ~10 denoising steps
```

---

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Working implementations** | 40% | All three models train and produce actions |
| **Quantitative evaluation** | 25% | Proper 100-episode eval with statistics |
| **Visualizations** | 20% | Clear, informative plots |
| **Report quality** | 15% | Concise analysis of results |

---

## Expected Results (PushT)

Based on published results and LeRobot benchmarks:

| Method | Expected Success Rate |
|--------|---------------------|
| BC (state) | ~50-70% |
| ACT | ~80-90% |
| Diffusion Policy | ~85-95% |

If your results differ significantly, check:
- Data loading: are observations and actions aligned?
- Normalization: are inputs/outputs normalized?
- Chunk size: appropriate for the task?
- Training steps: enough for convergence?

---

## Tips

1. **Start with PushT state-based** (not image-based) — faster training, easier debugging
2. **Use LeRobot's training scripts** for the first run — then customize
3. **Compare at similar training compute** — don't compare 10K-step BC with 200K-step DP
4. **Visualize early** — plot predictions after 1K steps to catch bugs early
5. **Save checkpoints** — training crashes happen, especially with diffusion

---

## Stretch Goals

- **Image-based**: use `observation.image` instead of `observation.state`
- **DDIM vs DDPM**: compare sampling quality at different step counts
- **Flow matching**: implement a flow matching variant and compare
- **Generalization**: train on one set of initial conditions, test on different ones
- **Real-time**: measure and optimize inference latency for each method

---

## Resources

- [LeRobot GitHub](https://github.com/huggingface/lerobot)
- [Diffusion Policy Paper](https://arxiv.org/abs/2303.04137)
- [ACT Paper](https://arxiv.org/abs/2304.13705)
- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
