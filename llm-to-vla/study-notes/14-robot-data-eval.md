# 14 — Robot Data, Simulation & Evaluation

> Phase VI · Days 85–91 · ~18 hours
> Prerequisites: 13-imitation-learning
> Learning Objectives: Understand robot data collection, simulation, evaluation metrics, DROID/Bridge/Open X-Embodiment

---

## Navigation

| Previous | [13-imitation-learning](13-imitation-learning.md) | Next | [15-vla-architectures](15-vla-architectures.md) |
|----------|--------------------------------------------------|------|------------------------------------------------|

---

## Overview

Models are only as good as their data. In NLP, internet text provides virtually unlimited
training data. In computer vision, billions of images are available online. But for robot
learning, data is scarce, expensive, and fragmented across different robots, environments,
and task definitions. This note covers the **data infrastructure** that enables modern
robot learning: collection methods, open datasets, simulation, evaluation metrics, and
the LeRobot library.

**The data bottleneck**: while LLMs train on trillions of tokens and vision models train
on billions of images, the largest robot datasets have only hundreds of thousands of
demonstrations. This scarcity drives the VLA approach: transfer knowledge from internet-scale
pretraining to compensate for limited robot data.

---

## Day 85 — Data Collection for Robotics

### Teleoperation Methods

| Method | Equipment | Quality | Speed | Embodiment |
|--------|-----------|---------|-------|-----------|
| **VR controller** | Meta Quest, HTC Vive | High | Moderate | Any arm |
| **Spacemouse** | 3DConnexion device | Good | Slow | Any arm |
| **Joint teaching** | Physical robot guidance | Good | Slow | Specific robot |
| **Puppet arm** | Leader-follower arms | **Best** | **Fast** | Matched kinematics |
| **Mobile phone** | Smartphone + IMU | Moderate | Fast | Limited |
| **Eye tracking** | Gaze + gesture | Low | Moderate | Any arm |

**Puppet arm** (used in ALOHA, ACT): a leader arm physically identical to the follower.
The human moves the leader → the follower mirrors in real time → record the follower's
trajectory. This gives the highest quality demonstrations because the kinematic mapping
is exact.

### Data Format

A typical robot demonstration contains:

```
Demonstration d = {
    observations: {
        images: [cam1_t0, cam1_t1, ..., cam1_tT,     # wrist camera
                 cam2_t0, cam2_t1, ..., cam2_tT],     # external camera
        robot_state: [q_t0, q_t1, ..., q_tT],         # joint positions
        ee_pose: [p_t0, p_t1, ..., p_tT],             # end-effector pose
        gripper: [g_t0, g_t1, ..., g_tT],             # gripper state
    },
    actions: [a_t0, a_t1, ..., a_tT],                 # commanded actions
    language: "pick up the red cup",                    # task description
    metadata: {
        robot: "Franka Panda",
        fps: 10,
        scene: "kitchen_counter_1",
        success: true,
    }
}
```

### Why Data is the Bottleneck

| Domain | Dataset Size | Collection Cost | Diversity |
|--------|-------------|----------------|-----------|
| NLP | ~15T tokens | Free (internet) | Very high |
| Vision | ~5B images | Free (internet) | Very high |
| **Robotics** | **~2M demos** (total) | **$10-100/demo** | **Low** |

Each robot demonstration requires:
- Physical setup time (minutes)
- Human teleoperator (skilled)
- Task-specific environment
- Real-time recording infrastructure
- Quality filtering (remove failed demos)

**Cost estimate**: collecting 1000 demonstrations ≈ $10K–$100K depending on complexity.

### Data Quality Matters

| Quality Factor | Effect on Learning | Mitigation |
|---------------|-------------------|-----------|
| **Demonstration quality** | Bad demos → bad policy | Filter failures, use skilled operators |
| **Camera placement** | Missing info → wrong actions | Multiple cameras, wrist + external |
| **Action recording rate** | Low fps → jerky behavior | 10-30 Hz typical |
| **Scene diversity** | Low diversity → poor generalization | Vary objects, backgrounds, lighting |
| **Language annotation** | Vague instructions → confusion | Structured, specific descriptions |

### Key Concepts Checklist — Day 85

- [ ] Know the main teleoperation methods and their tradeoffs
- [ ] Understand the standard robot data format
- [ ] Can explain why robot data is expensive and scarce
- [ ] Know the quality factors that affect learned policies

---

## Day 86 — Open Datasets

### Open X-Embodiment (2023)

The largest cross-embodiment robot dataset initiative:

| Metric | Value |
|--------|-------|
| **Robots** | 60+ different robot platforms |
| **Skills** | 527 distinct skills |
| **Demonstrations** | ~2M episodes |
| **Institutions** | 20+ labs worldwide |
| **Format** | RLDS (Reinforcement Learning Datasets) |

**Key insight**: training on data from many different robots improves generalization,
even to new robots not seen during training (cross-embodiment transfer).

**Embodiment diversity**:
- Franka Panda, Sawyer, UR5, KUKA iiwa (arms)
- Google Robot, Bridge Robot (custom platforms)
- Stretch RE-1, Hello Robot (mobile manipulation)
- Bimanual setups (ALOHA, dual arms)

### Bridge V2 (2023)

| Metric | Value |
|--------|-------|
| **Robot** | WidowX 250 6DOF arm |
| **Demonstrations** | 60,096 across 13 skills |
| **Environments** | 24 different environments |
| **Language labels** | ✓ (for all demonstrations) |
| **Camera** | Third-person + wrist camera |
| **Cost** | ~$1K robot, community-collected |

**Why Bridge matters**: low-cost robot → community data collection at scale.

### DROID (2024)

| Metric | Value |
|--------|-------|
| **Demonstrations** | 76,000 |
| **Scenes** | 564 unique scenes |
| **Buildings** | 52 buildings |
| **Cities** | Multiple cities |
| **Robot** | Franka Panda |
| **Cameras** | 3 cameras (2 external + 1 wrist) |
| **Language** | Crowd-sourced annotations |

**Why DROID matters**: unprecedented scene diversity. Most datasets are collected in a
single lab — DROID spans hundreds of real-world environments.

### RT-2-X Cross-Embodiment Results

Training on data from multiple robots (using Open X-Embodiment):

| Model | Single Robot Data | Multi-Robot Data | Improvement |
|-------|------------------|-----------------|-------------|
| RT-1 | 70% success | - | baseline |
| RT-2-X | - | 76% success | +6% absolute |
| Octo | 45% (single) | 52% (multi) | +7% absolute |

**Key finding**: cross-embodiment training helps even when the target robot has
substantial data of its own. Different robots in different environments provide
complementary learning signals.

### The Data Scaling Challenge

```
Current (2024):   ~2M robot demonstrations total (across all datasets)
LLM equivalent:   ~15T text tokens
Vision equivalent: ~5B images

Gap: ~1000x fewer data points in robotics
```

**Strategies to bridge the gap**:
1. **VLM pretraining**: leverage internet-scale vision-language data
2. **Simulation**: generate unlimited synthetic demonstrations
3. **Cross-embodiment**: share data across different robots
4. **Foundation models**: one model for many tasks (amortize data cost)
5. **Data augmentation**: geometric transforms, color jitter, etc.
6. **Active data collection**: only collect demos where the policy struggles

### Dataset Comparison

| Dataset | Robot(s) | Demos | Scenes | Language | Year |
|---------|---------|-------|--------|----------|------|
| RoboNet | 7 robots | 162K | 113 | ✗ | 2020 |
| Bridge V1 | WidowX | 7,200 | 10 | ✓ | 2022 |
| **Bridge V2** | WidowX | 60K | 24 | ✓ | 2023 |
| RT-1 Dataset | Google Robot | 130K | Lab | ✓ | 2023 |
| **Open X-Emb.** | 60+ robots | ~2M | Many | Partial | 2023 |
| **DROID** | Franka | 76K | 564 | ✓ | 2024 |
| RH20T | Various | 110K | 100+ | ✓ | 2023 |

### Key Concepts Checklist — Day 86

- [ ] Know the major open robot datasets and their characteristics
- [ ] Understand cross-embodiment transfer and why it helps
- [ ] Can explain the data scaling challenge in robotics vs NLP/vision
- [ ] Know the strategies for bridging the robot data gap

---

## Day 87 — Simulation Environments

### Major Simulation Platforms

| Simulator | Physics Engine | GPU Parallel | Rendering | Robotics Focus |
|-----------|---------------|-------------|-----------|---------------|
| **MuJoCo** | MuJoCo | ✓ (MJX) | Basic → improving | Manipulation, locomotion |
| **Isaac Gym/Lab** | PhysX | **✓✓✓** | High quality | All robotics |
| **SAPIEN** | PhysX | ✓ | **Excellent** | Manipulation |
| **PyBullet** | Bullet | ✗ | Basic | Manipulation, locomotion |
| **Gazebo** | Multiple | ✗ | Good | ROS ecosystem |
| **RLBench** | CoppeliaSim | ✗ | Good | Manipulation benchmarks |

### Simulation for Robot Learning

**Advantages**:
- Unlimited, free data collection
- Parallel environments (1000s simultaneously)
- Automatic success labeling
- Safe exploration
- Perfect state information available
- Instant reset

**Disadvantages**:
- Sim-to-real gap (physics, rendering, sensors)
- Modeling real-world complexity is hard
- Objects, textures, lighting differ from reality
- Contact dynamics (friction, deformation) are approximate

### Sim-to-Real Transfer Techniques

**1. Domain Randomization**:
Randomize simulation parameters during training so the policy becomes robust:

| Parameter | Randomization Range | Effect |
|-----------|-------------------|--------|
| Object mass | 0.5× – 2× | Robust to weight variation |
| Friction | 0.3 – 1.0 | Handles different surfaces |
| Camera position | ±5cm, ±10° | Robust to camera mounting |
| Lighting | Random color/intensity | Works under different lights |
| Object texture | Random textures | Appearance robustness |
| Action delay | 0–2 steps | Handles control latency |

**Philosophy**: if the policy works across thousands of simulated variations, the real
world is "just another variation."

**2. System Identification**:
Measure real-world physics parameters and match them in simulation:
- Friction coefficients (measure with force sensors)
- Object masses and inertias
- Camera intrinsics and extrinsics
- Control delays and actuator dynamics

**3. Progressive Transfer**:
```
1. Train in simulation with domain randomization
2. Evaluate on real robot with frozen policy
3. Collect real-world correction data
4. Fine-tune with combined sim + real data
```

### Simulation for Data Generation

**Generating demonstrations in simulation**:
- Script expert policies for simple tasks
- Use motion planning for complex tasks
- Human teleoperation in VR → simulation
- RL-trained policies as "experts" in sim

**For VLA pretraining**: simulation provides unlimited paired (language, image, action)
data, which can be mixed with real data during training.

### Key Concepts Checklist — Day 87

- [ ] Know the major simulation platforms and when to use each
- [ ] Understand the sim-to-real gap and three approaches to address it
- [ ] Can explain domain randomization strategy
- [ ] Know the advantages and limitations of simulated data

---

## Day 88 — Evaluation Metrics

### Task Success Rate

The primary metric: did the robot complete the task?

$$\text{Success Rate} = \frac{\text{# successful episodes}}{\text{total episodes}} \times 100\%$$

**Practical considerations**:
- Evaluate over ≥100 episodes (statistical significance)
- Report mean ± standard error
- Define "success" precisely (e.g., object lifted >10cm for >2s)
- Use automated success detection when possible

### Generalization Levels

| Level | Description | Example | Difficulty |
|-------|------------|---------|-----------|
| **In-distribution** | Same objects, same scene | Train and test on kitchen A | Easy |
| **Novel objects** | Unseen objects, same category | New mugs, trained on other mugs | Medium |
| **Novel arrangements** | Same objects, different positions | Objects in new configurations | Medium |
| **Novel scenes** | Completely new environments | Different kitchen entirely | Hard |
| **Novel instructions** | Unseen language commands | "Stack the blue one on the red one" | Hard |
| **Cross-embodiment** | Different robot | Train on Franka, test on UR5 | **Very hard** |

### Additional Metrics

| Metric | What It Measures | When Important |
|--------|-----------------|---------------|
| **Completion rate** | Partial task progress | Multi-step tasks |
| **Path efficiency** | Trajectory length / optimal | Navigation tasks |
| **Smoothness** | Jerk in action trajectory | Real-world deployment |
| **Inference latency** | Time per action prediction | Real-time control |
| **Safety violations** | Collisions, joint limits hit | Always |
| **Grasp stability** | Object held through task | Manipulation |
| **Language grounding** | Correct object selected | Language-conditioned tasks |

### The Benchmark Problem

There is no universally accepted benchmark for robot manipulation:

| Benchmark | Tasks | Real/Sim | Standardized? |
|-----------|-------|----------|--------------|
| RLBench | 100 tasks | Sim | ✓ |
| MetaWorld | 50 tasks | Sim | ✓ |
| CALVIN | 34 tasks, language | Sim | ✓ |
| SIMPLER | 4 Google Robot tasks | Sim (→ real) | ✓ |
| Real-world | Varies | Real | **✗** |

**The core issue**: success on simulated benchmarks doesn't guarantee real-world performance.
But real-world evaluation is expensive and not reproducible across labs.

### Evaluation Protocol Best Practices

```
1. Define success criteria precisely (height, duration, orientation)
2. Randomize initial conditions (object positions, robot start pose)
3. Use ≥100 evaluation episodes
4. Report: mean success rate, std error, 95% confidence interval
5. Separate in-distribution from generalization evaluation
6. Include failure analysis (common failure modes)
7. Video recording for all episodes (for human verification)
```

### Key Concepts Checklist — Day 88

- [ ] Know the primary metric (task success rate) and how to compute it properly
- [ ] Understand the generalization hierarchy (in-distribution → cross-embodiment)
- [ ] Can list additional metrics beyond success rate
- [ ] Know the benchmark landscape and its limitations
- [ ] Understand evaluation best practices (sample size, randomization, reporting)

---

## Day 89 — LeRobot Library

### What is LeRobot?

[LeRobot](https://github.com/huggingface/lerobot) is HuggingFace's open-source library
for robot learning:

| Feature | Description |
|---------|------------|
| **Datasets** | Standardized format, hosted on HuggingFace Hub |
| **Pretrained policies** | ACT, Diffusion Policy, TDMPC |
| **Training scripts** | End-to-end training pipelines |
| **Simulation** | Integration with gym environments |
| **Real robot support** | ALOHA, Koch v1.1, SO-100, Moss |
| **Visualization** | Dataset exploration, trajectory plotting |

### Dataset Format

LeRobot uses a standardized format based on HuggingFace Datasets:

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Load a dataset
dataset = LeRobotDataset("lerobot/pusht")

# Each item contains:
item = dataset[0]
# item["observation.image"]     → camera image (C, H, W)
# item["observation.state"]     → robot state vector
# item["action"]                → action vector
# item["episode_index"]         → which episode this belongs to
# item["frame_index"]           → timestep within episode
# item["timestamp"]             → wall-clock time
```

### Available Datasets

| Dataset | Robot | Tasks | Episodes | Source |
|---------|-------|-------|---------|--------|
| `lerobot/pusht` | 2D sim | Push T-shape | 206 | LeRobot |
| `lerobot/aloha_sim_*` | ALOHA sim | Bimanual tasks | Varies | ACT |
| `lerobot/xarm_*` | xArm sim | Manipulation | Varies | Community |
| `lerobot/umi_cup_in_the_wild` | UMI | Cup manipulation | 1000+ | UMI |
| Community datasets | Various | Various | Varies | HF Hub |

### Training a Policy

```python
# Train ACT on PushT
from lerobot.scripts.train import train

train(
    dataset_repo_id="lerobot/pusht",
    policy="act",
    env="pusht",
    training_steps=100_000,
    batch_size=64,
    lr=1e-4,
    action_chunk_size=100,
    # ...
)
```

### Training Diffusion Policy

```python
# Train Diffusion Policy on PushT
train(
    dataset_repo_id="lerobot/pusht",
    policy="diffusion",
    env="pusht",
    training_steps=200_000,
    batch_size=64,
    lr=1e-4,
    num_diffusion_steps=100,
    num_inference_steps=10,    # DDIM steps at inference
    # ...
)
```

### Evaluation

```python
from lerobot.scripts.eval import eval_policy

results = eval_policy(
    policy_path="outputs/pusht_act/checkpoints/last",
    env="pusht",
    n_episodes=100,
)
# results["success_rate"], results["mean_reward"], ...
```

### Using LeRobot for Your Own Data

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Create a new dataset
dataset = LeRobotDataset.create(
    repo_id="your-username/your-dataset",
    fps=10,
    robot_type="koch",
    features={
        "observation.image": {"dtype": "video", "shape": (3, 480, 640)},
        "observation.state": {"dtype": "float32", "shape": (6,)},
        "action": {"dtype": "float32", "shape": (6,)},
    },
)

# Add episodes from teleoperation
for episode_data in recorded_episodes:
    for frame in episode_data:
        dataset.add_frame(frame)
    dataset.save_episode()

# Push to Hub
dataset.push_to_hub()
```

### Key Concepts Checklist — Day 89

- [ ] Can install and use LeRobot
- [ ] Know the dataset format and how to load existing datasets
- [ ] Can train ACT and Diffusion Policy using LeRobot
- [ ] Understand how to evaluate policies
- [ ] Know how to create and share your own datasets

---

## Days 90–91 — Phase VI Capstone

### Capstone Project: Train and Evaluate Diffusion Policy

See [Project 06](../projects/06-diffusion-policy/README.md) for the full capstone.

**Summary**:
1. Select a manipulation task from LeRobot datasets
2. Train BC baseline, ACT, and Diffusion Policy
3. Evaluate all three on 100 episodes
4. Compare: success rate, trajectory quality, multimodality
5. Write comparison report with visualizations

### What You Should Know After Phase VI

| Topic | Minimum Competency | Target Competency |
|-------|-------------------|------------------|
| RL foundations | Explain MDP, PPO | Implement REINFORCE, use PPO library |
| Diffusion models | Explain DDPM training/sampling | Implement simple DDPM |
| Flow matching | Explain the concept | Compare with diffusion |
| Behavioral cloning | Implement BC | Analyze failure modes |
| Action chunking | Explain why it helps | Implement and compare chunk sizes |
| ACT | Explain architecture | Implement simplified version |
| Diffusion Policy | Explain architecture | **Train on real dataset** |
| Robot datasets | Know major datasets | Use LeRobot to load and train |
| Evaluation | Know metrics | Run proper evaluation protocol |

---

## Key Takeaways

1. **Robot data is the bottleneck**: orders of magnitude less data than NLP/vision.
   This scarcity drives the VLA approach of leveraging pretrained VLMs.

2. **Open datasets** (Open X-Embodiment, DROID, Bridge V2) are enabling cross-embodiment
   learning, where data from different robots contributes to a shared model.

3. **Simulation** provides unlimited data but with a reality gap. Domain randomization
   and progressive transfer help bridge this gap.

4. **Evaluation** in robotics is harder than in NLP/vision: no standard benchmarks,
   success criteria vary by task, and real-world evaluation is expensive.

5. **LeRobot** provides a practical, standardized framework for training and evaluating
   robot policies on open datasets.

---

## Paper References

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| Open X-Embodiment Collaboration | 2023 | Cross-embodiment dataset and transfer |
| Khazatsky et al., "DROID" | 2024 | Large-scale diverse robot data |
| Walke et al., "Bridge V2" | 2023 | Low-cost community data collection |
| Brohan et al., "RT-2-X" | 2024 | Cross-embodiment VLA evaluation |
| Cadene et al., "LeRobot" | 2024 | Open-source robot learning library |
| Todorov et al., "MuJoCo" | 2012 | Physics simulation for robotics |
| Makoviychuk et al., "Isaac Gym" | 2021 | GPU-parallel robot simulation |
| James et al., "RLBench" | 2020 | Standardized manipulation benchmarks |
| Tobin et al., "Domain Randomization" | 2017 | Sim-to-real via randomization |
| Li et al., "SIMPLER" | 2024 | Simulated evaluation for real policies |

---

## Connection to the Thread

> **More diverse data = better compression of the robot-world interaction.** Each
> demonstration teaches the model about objects, physics, and task structure. Cross-embodiment
> data teaches it that the underlying skills (grasping, placing, pushing) are shared across
> different robot bodies — the model compresses what is invariant across embodiments.
>
> **The data challenge explains the VLA architecture**: since robot data is scarce, we
> compensate by pretraining on internet-scale language and vision data. The VLM backbone
> compresses billions of images and text documents into a world model that only needs a
> few thousand demonstrations to connect to physical action.
>
> **Cross-embodiment = compressing knowledge that transfers across robot bodies.** This is
> the ultimate compression: what's universal about manipulation, regardless of whether you
> have a 6-DOF arm or a humanoid hand?

---

## What's Next

Phase VI is complete! You now understand:
- RL foundations and why pure RL struggles in robotics
- Diffusion models and flow matching for generative action prediction
- Imitation learning: BC, ACT, Diffusion Policy
- Robot data ecosystems and evaluation

In **Phase VII** ([Study Note 15](15-vla-architectures.md)), we'll finally study the VLA
architectures that unify all of these components: RT-2, Octo, π₀, and OpenVLA. You'll see
how everything we've learned — transformers, vision encoders, VLMs, diffusion, imitation
learning — comes together into a single model that reads language, sees images, and outputs
robot actions.
