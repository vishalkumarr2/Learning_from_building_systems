# 13 — Imitation Learning for Robotics

> Phase VI · Days 78–84 · ~18 hours
> Prerequisites: 11-rl-foundations, 12-diffusion-flow
> Learning Objectives: Understand BC, learn action representations, ACT, diffusion policy implementation

---

## Navigation

| Previous | [12-diffusion-flow](12-diffusion-flow.md) | Next | [14-robot-data-eval](14-robot-data-eval.md) |
|----------|-------------------------------------------|------|---------------------------------------------|

---

## Overview

Imitation learning (IL) is the dominant paradigm for teaching robots new skills.
Instead of engineering reward functions and running millions of RL episodes, we
collect a few hundred expert demonstrations and learn to mimic them. IL turns robot
learning from a reinforcement problem into a supervised learning problem — and we
already know that supervised learning scales beautifully with data and compute.

**The IL hierarchy for this curriculum**:
1. **Behavioral Cloning (BC)** — simplest: supervised learning on (obs, action) pairs
2. **DAgger** — fix distribution shift by iteratively collecting more data
3. **ACT** — action chunking with a CVAE for multimodal, temporally coherent actions
4. **Diffusion Policy** — diffusion-based action generation for complex distributions
5. **Language-conditioned IL** — condition policies on natural language instructions

This note culminates in **STOP AND REFLECT #5**: the realization that a VLA is simply
a unified transformer that processes text, image, and action tokens together.

---

## Day 78 — Behavioral Cloning

### The Simplest Approach: Supervised Learning

Given a dataset of expert demonstrations:

$$\mathcal{D} = \{(o_1, a_1), (o_2, a_2), \ldots, (o_N, a_N)\}$$

where $o_i$ is an observation (image + robot state) and $a_i$ is the expert's action.

**Behavioral Cloning (BC)**: train a policy $\pi_\theta(a|o)$ to minimize:

$$L_{\text{BC}} = \mathbb{E}_{(o,a) \sim \mathcal{D}}\left[\|\pi_\theta(o) - a\|^2\right]$$

This is just regression. For continuous actions, MSE loss. For discrete actions, cross-entropy.

**Advantages**:
- Simplest possible approach
- No environment interaction needed during training
- Standard supervised learning tools apply
- Works surprisingly well with enough data

### The Covariate Shift Problem

**The core issue**: during training, the policy sees observations from the expert's
trajectory distribution $d_{\text{expert}}(o)$. During deployment, small errors accumulate,
pushing the robot into states the expert never visited: $d_{\pi_\theta}(o) \neq d_{\text{expert}}(o)$.

**Compounding errors**: a small per-step error $\epsilon$ leads to:
- After $T$ steps: error grows as $O(\epsilon T^2)$ in the worst case
- The policy encounters unfamiliar observations → makes larger errors → diverges further

**Visualization**:
```
Expert trajectory:     ─────────────────────→ ✓ goal
                           ↗ small error
BC trajectory:         ───── ─ ─ ─ ─ → → → → ✗ off course
                                          (never seen these states)
```

### Mitigations for Covariate Shift

| Strategy | How It Works | Practical? |
|----------|-------------|-----------|
| **More data** | Cover more of state space | Sometimes (expensive) |
| **Data augmentation** | Add noise to observations | Yes, very effective |
| **Action chunking** | Predict multiple steps (fewer decision points) | Yes |
| **DAgger** | Iteratively collect on-policy data | Yes, but needs expert |
| **History conditioning** | Give policy past observations (context) | Yes |
| **Trajectory-level loss** | Evaluate on full rollouts, not single steps | Hard to differentiate |

### DAgger: Dataset Aggregation

**DAgger** (Ross et al., 2011) fixes covariate shift by iteratively collecting data
under the learned policy's own distribution:

```
Algorithm: DAgger
─────────────────
1. Collect initial dataset D₁ using expert π*
2. Train policy π₁ on D₁
3. For i = 1, 2, ..., N:
   a. Roll out πᵢ in the environment
   b. At each state, query expert π* for the action label
   c. Aggregate: Dᵢ₊₁ = Dᵢ ∪ {(o_πᵢ, a_π*)}
   d. Train πᵢ₊₁ on Dᵢ₊₁
```

**Key insight**: now the training data includes states the policy actually visits,
not just states the expert visits.

**Practical issues**:
- Needs an interactive expert (or scripted oracle in simulation)
- Expert must label states they would never naturally visit
- Data collection is iterative → slow
- In practice: **data augmentation + action chunking often work better**

### Key Concepts Checklist — Day 78

- [ ] Can implement basic BC (supervised learning on observation-action pairs)
- [ ] Understand the covariate shift problem and compounding errors
- [ ] Can explain DAgger and when it helps
- [ ] Know practical mitigations (augmentation, action chunking, history)

---

## Day 79 — Action Representations

### How Do We Represent Robot Actions?

The choice of action representation dramatically affects learning difficulty.

| Representation | Dimensions | Properties |
|---------------|-----------|-----------|
| **Joint positions** | $n_{\text{joints}}$ (typically 6-7) | Direct, but ignores task space |
| **Joint velocities** | $n_{\text{joints}}$ | Smoother, but needs integration |
| **End-effector pose** | 6-7 (xyz + rotation) | Task-relevant, but needs IK |
| **Delta end-effector** | 6-7 | Small values, easy to learn |
| **End-effector + gripper** | 7-8 | Most common for manipulation |

**For manipulation tasks**: delta end-effector pose + gripper binary is most common:

$$a = [\Delta x, \Delta y, \Delta z, \Delta \text{roll}, \Delta \text{pitch}, \Delta \text{yaw}, \text{gripper}]$$

### Continuous vs Discrete Actions

| Approach | Encoding | Loss | Use Case |
|----------|---------|------|----------|
| **Continuous** | Direct float values | MSE / L1 | Most IL methods |
| **Discretized** | Bin each dimension | Cross-entropy | RT-2, some VLAs |
| **Tokenized** | Map to vocabulary | Cross-entropy | Language model-based VLAs |

**VLA insight**: RT-2 and similar models discretize actions into tokens, treating action
prediction identically to language generation. This unification is powerful but loses
precision (bin resolution).

### Action Normalization

Always normalize actions before training:

```python
# Per-dimension normalization
a_normalized = (a - a_mean) / a_std

# Or min-max to [-1, 1]
a_normalized = 2 * (a - a_min) / (a_max - a_min) - 1
```

**Why**: different action dimensions have different scales (meters vs radians vs binary).
Without normalization, the MSE loss is dominated by high-magnitude dimensions.

### Action Chunking: The Key Idea

Instead of predicting one action $a_t$, predict a chunk of $H$ future actions:

$$\text{chunk}_t = [a_t, a_{t+1}, \ldots, a_{t+H-1}]$$

**Why action chunking works**:

1. **Fewer decision points**: instead of $T$ decisions, make $T/H$ decisions → fewer
   opportunities for compounding errors

2. **Temporal coherence**: the chunk is generated all at once, so actions within a chunk
   are automatically consistent with each other

3. **Reduced effective horizon**: the policy only needs to "look ahead" $H$ steps at a
   time, not the entire trajectory

4. **Multimodal commitment**: once the policy "decides" on a mode (left vs right), the
   entire chunk follows that mode

**Receding horizon execution**: execute first $H_e$ actions from the chunk, then re-plan:
```
while not done:
    chunk = policy(observation)           # predict H actions
    for i in range(H_e):                  # execute first H_e < H
        env.step(chunk[i])
    observation = env.observe()           # get new observation
```

Typical values: $H = 10-100$, $H_e = 1-10$.

### Rotation Representations

Representing 3D rotations is subtle — naive representations have discontinuities:

| Representation | Dimensions | Continuous? | Common Use |
|---------------|-----------|------------|-----------|
| Euler angles | 3 | ✗ (gimbal lock) | Avoid |
| Quaternion | 4 (unit constraint) | ✗ (antipodal) | Common but tricky |
| Rotation matrix | 9 (orthogonality constraint) | ✓ | Theoretical |
| 6D rotation | 6 | ✓ | **Recommended** |
| Axis-angle | 3-4 | ✗ (discontinuous) | Sometimes |

**6D rotation representation** (Zhou et al., 2019): use the first two columns of the
rotation matrix (6 values), then reconstruct the full rotation via Gram-Schmidt.
This is the only low-dimensional representation that is continuous.

### Key Concepts Checklist — Day 79

- [ ] Know the common action representations and their tradeoffs
- [ ] Understand why action chunking reduces compounding errors
- [ ] Can explain receding horizon execution
- [ ] Know why action normalization is important
- [ ] Understand the 6D rotation representation and why it's preferred

---

## Day 80 — ACT: Action Chunking with Transformers

### ACT Architecture (Zhao et al., 2023)

**ACT** = Action Chunking with Transformers, a CVAE-based approach that generates
temporally coherent action chunks while handling multimodality.

**Key components**:

1. **CVAE Encoder** (training only): takes the action chunk + observations, outputs
   a style variable $z$ that captures the "how" of execution
2. **Decoder**: takes observations + $z$, generates the action chunk
3. **Action chunking**: predicts $H$ future actions at once

### CVAE Framework

**Conditional Variational Autoencoder** structure:

**Encoder** $q_\phi(z | a_{1:H}, o)$:
- Input: ground truth action chunk + observations
- Output: parameters of $q(z) = \mathcal{N}(\mu_\phi, \sigma_\phi^2)$
- Purpose: compress the "style" of execution into $z$

**Decoder** $p_\theta(a_{1:H} | z, o)$:
- Input: sampled $z$ + observations
- Output: predicted action chunk
- Purpose: generate actions given the execution style

**Training loss** (ELBO):

$$L = \underbrace{\|a_{1:H} - \hat{a}_{1:H}\|^2}_{\text{reconstruction}} + \beta \underbrace{D_{\text{KL}}(q_\phi(z|a,o) \| \mathcal{N}(0,I))}_{\text{regularization}}$$

where $\beta$ controls the information bottleneck (typically $\beta = 10^{-2}$).

### Why the CVAE Structure?

The latent variable $z$ captures the **multimodality** of demonstrations:

- Expert 1 reaches from the left → $z_1$
- Expert 2 reaches from the right → $z_2$
- Same observation, different $z$ → different valid action chunks

Without $z$ (plain BC): the model averages the two trajectories → bad action.
With $z$: each sample of $z$ produces a complete, coherent trajectory.

### ACT Transformer Architecture

```
Observations (images + state)
        ↓
   Vision Encoder (ResNet/ViT)
        ↓
   [CLS] [obs_1] [obs_2] ... [obs_M]     ← observation tokens
        ↓
   Transformer Encoder
        ↓
   [z]  [obs_1'] [obs_2'] ... [obs_M']    ← z prepended
        ↓
   Transformer Decoder
        ↓
   [a_1] [a_2] [a_3] ... [a_H]           ← action chunk
```

**Encoder** (training only):
- Takes observation tokens + action tokens
- Outputs mean and variance for $z$

**Decoder** (training and inference):
- Takes observation tokens + $z$ (sampled)
- Autoregressively generates action chunk (or parallel with masking)

### Training Procedure

```python
# Training loop
for obs, actions in dataloader:
    # Encode observations
    obs_tokens = vision_encoder(obs)

    # Encoder: get z distribution (training only)
    mu, logvar = cvae_encoder(obs_tokens, actions)
    z = reparameterize(mu, logvar)

    # Decoder: predict actions
    pred_actions = decoder(obs_tokens, z)

    # Loss
    recon_loss = mse(pred_actions, actions)
    kl_loss = kl_divergence(mu, logvar)
    loss = recon_loss + beta * kl_loss
    loss.backward()
```

### Inference

```python
# Inference: no encoder needed
obs_tokens = vision_encoder(current_obs)
z = torch.randn(z_dim)                    # sample from prior N(0,I)
pred_actions = decoder(obs_tokens, z)      # generate action chunk
execute(pred_actions[:H_e])                # execute first H_e actions
```

### Temporal Ensembling

ACT introduces **temporal ensembling** to smooth out inter-chunk transitions:

```
Time:     t   t+1  t+2  t+3  t+4  t+5  t+6
Chunk 1:  a₁  a₂   a₃   a₄
Chunk 2:       a₁'  a₂'  a₃'  a₄'
Chunk 3:            a₁'' a₂'' a₃'' a₄''

At time t+2: execute weighted average of a₃, a₂', a₁''
Weights: exponential decay by age
```

This eliminates jerkiness at chunk boundaries.

### Key Concepts Checklist — Day 80

- [ ] Understand the ACT architecture: CVAE encoder + transformer decoder
- [ ] Can explain the role of the latent variable $z$ for multimodality
- [ ] Know the ELBO loss (reconstruction + KL regularization)
- [ ] Understand temporal ensembling for smooth execution
- [ ] Can describe the difference between training (encoder needed) and inference (prior sampling)

---

## Days 81–82 — Diffusion Policy Implementation

### Diffusion Policy Recap

From [Study Note 12](12-diffusion-flow.md), Diffusion Policy models actions as a
denoising process:

$$\epsilon_\theta(a^k, k, O) \approx \epsilon$$

**Training**: predict the noise added to action chunks.
**Inference**: iteratively denoise random noise → valid action chunk.

### Implementation: Key Components

**1. Observation Encoder**:
```python
class ObservationEncoder(nn.Module):
    def __init__(self):
        self.image_encoder = ResNet18(pretrained=True)  # or ViT
        self.state_encoder = MLP(state_dim, 256)

    def forward(self, images, state):
        img_feat = self.image_encoder(images)
        state_feat = self.state_encoder(state)
        return torch.cat([img_feat, state_feat], dim=-1)
```

**2. Noise Prediction Network** (1D Temporal U-Net):
```python
class ConditionalUNet1D(nn.Module):
    """
    Input: noisy action chunk (B, H, action_dim)
    Condition: observation features + diffusion timestep
    Output: predicted noise (B, H, action_dim)
    """
    def __init__(self, action_dim, obs_dim, diffusion_steps):
        # Sinusoidal time embedding
        self.time_embed = SinusoidalPosEmb(128)
        # FiLM conditioning from observations
        self.obs_encoder = MLP(obs_dim, 256)
        # 1D U-Net: down → mid → up with skip connections
        self.down_blocks = nn.ModuleList([...])
        self.mid_block = ...
        self.up_blocks = nn.ModuleList([...])
```

**3. Training Loop**:
```python
for obs, action_chunk in dataloader:
    obs_feat = obs_encoder(obs)

    # Sample noise level and noise
    k = torch.randint(0, K, (batch_size,))
    noise = torch.randn_like(action_chunk)

    # Create noisy actions
    noisy_actions = sqrt_alpha_bar[k] * action_chunk + sqrt_one_minus_alpha_bar[k] * noise

    # Predict noise
    pred_noise = unet(noisy_actions, k, obs_feat)

    # Loss
    loss = F.mse_loss(pred_noise, noise)
    loss.backward()
```

**4. Inference Loop** (DDIM):
```python
def generate_actions(obs):
    obs_feat = obs_encoder(obs)
    actions = torch.randn(1, H, action_dim)  # start from noise

    for k in reversed(timestep_schedule):     # e.g., [100, 80, 60, 40, 20, 0]
        pred_noise = unet(actions, k, obs_feat)
        actions = ddim_step(actions, pred_noise, k)

    return actions
```

### ACT vs Diffusion Policy Comparison

| Aspect | ACT | Diffusion Policy |
|--------|-----|-----------------|
| Generative model | CVAE | DDPM/DDIM |
| Multimodality | Via latent $z$ | Via diffusion sampling |
| Training loss | ELBO (recon + KL) | MSE (noise prediction) |
| Inference steps | 1 forward pass | 10-100 denoising steps |
| Inference speed | **Fast** | Slower (but DDIM helps) |
| Action quality | Good | **Slightly better on complex tasks** |
| Implementation | Moderate | Moderate |
| Hyperparameters | $\beta$, $z$-dim | Noise schedule, steps, $K$ |

**When to use which**:
- **ACT**: real-time control needed, lower latency budget
- **Diffusion Policy**: complex multimodal tasks, higher quality needed
- **Both work well**: the choice is often empirical for a given task

### Key Concepts Checklist — Days 81-82

- [ ] Can implement the Diffusion Policy training loop
- [ ] Understand the role of each component (obs encoder, noise predictor, sampler)
- [ ] Can implement DDIM sampling for faster inference
- [ ] Know the practical differences between ACT and Diffusion Policy

---

## Day 83 — Advanced IL Techniques

### Residual Learning

Instead of learning the full action, learn a correction on top of a base controller:

$$a = a_{\text{base}}(o) + \pi_\theta(o)$$

**When to use**: when you have a decent but imperfect controller (e.g., from classical
control or a pretrained policy). The learned residual handles the hard parts.

### HG-DAgger: Human-Gated DAgger

**Practical DAgger**: instead of labeling every state, the human only intervenes when the
robot is about to fail:

```
1. Robot executes its own policy
2. Human watches and only takes over when needed
3. Interventions are added to the dataset
4. Retrain → fewer interventions needed → cycle continues
```

This is much more practical than standard DAgger because it minimizes expert effort.

### Multi-Task Imitation Learning

Train one policy that can perform multiple tasks:

$$\pi_\theta(a | o, \text{task})$$

**Task conditioning options**:
- One-hot task ID (fixed set of tasks)
- Language instruction (open-ended)
- Goal image (visual specification)
- Task embedding (learned)

**Language conditioning** is most flexible and connects directly to VLAs:

$$\pi_\theta(a | o, \text{``pick up the red block''})$$

### Language-Conditioned IL

**Architecture pattern**:
```
Language instruction → Language Encoder (CLIP/BERT) → text embedding
Image observation   → Vision Encoder (ViT/ResNet)   → visual embedding
Robot state         → State Encoder (MLP)            → state embedding
                                    ↓
                    Fusion (cross-attention / concatenation)
                                    ↓
                           Action Head (MLP / Diffusion / ACT)
                                    ↓
                               Action chunk
```

This is essentially a simplified VLA! The key VLA contribution is using a pretrained
VLM as the fusion backbone, bringing internet-scale knowledge.

### The IL → RL Spectrum

| Method | Data Source | Training Signal | Sample Efficiency | Performance Ceiling |
|--------|-----------|----------------|-------------------|-------------------|
| BC | Expert demos only | Supervised loss | **Best** | Limited by demos |
| DAgger | Expert demos + corrections | Supervised loss | Very good | Expert-level |
| IRL | Expert demos | Inferred reward | Good | Expert-level |
| RL fine-tuning | IL init + environment | Reward | Moderate | **Superhuman** |
| Pure RL | Environment only | Reward | Worst | Superhuman |

**The practical recipe**:
1. Start with BC or Diffusion Policy (fast, data-efficient)
2. Use DAgger or HG-DAgger for improvement
3. If superhuman performance needed: RL fine-tuning with the IL policy as initialization

### Key Concepts Checklist — Day 83

- [ ] Understand residual learning and when to apply it
- [ ] Can explain HG-DAgger and why it's more practical than standard DAgger
- [ ] Know how to condition a policy on language instructions
- [ ] Understand the IL → RL spectrum and the practical recipe

---

## Day 84 — 🛑 STOP AND REFLECT #5

### The Grand Unification

Take a moment to appreciate what we've built across Phases I through VI:

| Phase | Capability | Key Mechanism |
|-------|-----------|---------------|
| I — Deep Learning | Optimize any differentiable function | Gradient descent + backpropagation |
| II — Transformers | Process sequences with attention | Self-attention: $\text{softmax}(QK^T/\sqrt{d})V$ |
| III — LLMs | Understand and generate language | Autoregressive token prediction |
| IV — Vision | See and understand images | ViT: images as token sequences |
| V — VLMs | Connect vision and language | Align image and text embeddings |
| **VI — IL** | **Generate robot actions** | **Diffusion / ACT on demonstrations** |

### The VLA Insight

**A Vision-Language-Action model simply UNIFIES all these into one transformer**:

```
Input tokens:  [language tokens] [image patch tokens] [robot state tokens]
                      ↓                  ↓                    ↓
              ┌─────────────────────────────────────────────────────┐
              │           Pretrained VLM Backbone                    │
              │         (Transformer with attention)                 │
              │                                                      │
              │   Same architecture handles ALL modalities           │
              │   because everything is just token sequences         │
              └─────────────────────────────────────────────────────┘
                                     ↓
Output:                      [action tokens]
                    (decoded via diffusion, ACT, or autoregressive)
```

**The architecture is the same — only the tokenization changes.**

### What Each Component Contributes

| Component | What It Provides to the VLA |
|-----------|---------------------------|
| **Transformer backbone** | General-purpose sequence processing |
| **Language pretraining** | World knowledge, reasoning, instruction following |
| **Vision encoder** | Visual understanding, object recognition |
| **VLM alignment** | Grounding language in visual perception |
| **Action head (diffusion/ACT)** | Multimodal, temporally coherent actions |
| **Demonstrations** | Grounding all of the above in physical interaction |

### The Compression Perspective

Each training stage compresses a different aspect of the world:

1. **Language pretraining**: compress the structure of human knowledge (text)
2. **Vision pretraining**: compress the structure of visual perception (images)
3. **VLM alignment**: compress the mapping between seeing and describing
4. **IL on demonstrations**: compress the mapping between perceiving and acting

A VLA is the **most compressed representation of "how to act in the world given language
instructions and visual observations"**.

### Why This Path (Not Starting from Scratch)

Starting from an already-pretrained VLM means:
- The model already understands objects, spatial relationships, physics
- Language instructions are already grounded in visual concepts
- Only the action generation needs to be learned from robot data
- This is **dramatically more sample-efficient** than learning everything from scratch

**Analogy**: teaching a person who can already see and understand language to perform
a new manipulation task (show them a few times) vs teaching a newborn (years of development).

### Remaining Questions (to be answered in Phase VII)

1. How exactly do current VLAs tokenize actions?
2. How is the pretrained VLM adapted for action prediction?
3. What is the training recipe: which stages, which data, which losses?
4. How do you handle different robot embodiments?
5. What are the current limitations and open research directions?

### Self-Assessment

At this point, you should be able to:

- [ ] Explain the complete path from raw data to robot actions
- [ ] Draw the architecture of a VLA from memory
- [ ] Explain why pretraining matters for robot learning
- [ ] Compare BC, ACT, and Diffusion Policy with specific tradeoffs
- [ ] Articulate the "everything is token prediction" unification
- [ ] Explain why demonstrations are more efficient than RL for robotics

---

## Key Equations Reference

### Behavioral Cloning

$$L_{\text{BC}} = \mathbb{E}_{(o,a) \sim \mathcal{D}}\left[\|\pi_\theta(o) - a\|^2\right]$$

### CVAE (ACT)

$$L_{\text{ACT}} = \|a - \hat{a}\|^2 + \beta \cdot D_{\text{KL}}(q_\phi(z|a,o) \| \mathcal{N}(0,I))$$

### Diffusion Policy

$$L_{\text{DP}} = \mathbb{E}_{k, a_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(a^k, k, O)\|^2\right]$$

### DAgger

$$\mathcal{D}_{i+1} = \mathcal{D}_i \cup \{(o \sim d_{\pi_i}, \; a^* = \pi^*(o))\}$$

---

## Method Comparison Table

| Method | Multimodal | Temporal Coherence | Inference Speed | Data Efficiency | Complexity |
|--------|-----------|-------------------|----------------|----------------|-----------|
| BC (MLP) | ✗ | ✗ (single step) | **Fastest** | Good | Simplest |
| BC + chunking | ✗ | ✓ | Fast | Good | Simple |
| DAgger | ✗ | Depends | Fast | **Better** (iterative) | Moderate |
| **ACT** | **✓** (via z) | **✓** (chunks) | Fast | Good | Moderate |
| **Diffusion Policy** | **✓** (via sampling) | **✓** (chunks) | Moderate | Good | Moderate |
| Flow Matching Policy | ✓ | ✓ | **Fast** (fewer steps) | Good | Moderate |

---

## Key Takeaways

1. **Behavioral cloning** is surprisingly effective with enough data and the right tricks
   (action chunking, augmentation, history conditioning), but fails on distribution shift.

2. **Action chunking** is one of the most important practical improvements: predict multiple
   steps at once for temporal coherence and reduced compounding errors.

3. **ACT** uses a CVAE to handle multimodality: the latent variable $z$ captures the "style"
   of execution (left vs right, fast vs slow, different grasps).

4. **Diffusion Policy** models the full action distribution via denoising, naturally handling
   multimodality without explicit latent variables.

5. **The VLA unification**: language understanding (transformers) + visual perception (ViT) +
   action generation (diffusion/ACT) = a single model that processes text + image + action tokens.
   The architecture is the same throughout — only the tokenization changes.

---

## Paper References

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| Pomerleau, "ALVINN" | 1989 | First behavioral cloning for driving |
| Ross et al., "A Reduction of IL to No-Regret Online Learning" | 2011 | DAgger algorithm |
| Zhao et al., "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware" | 2023 | **ACT** — CVAE + action chunking |
| Chi et al., "Diffusion Policy" | 2023 | **Diffusion models for action generation** |
| Zhou et al., "On the Continuity of Rotation Representations" | 2019 | 6D rotation representation |
| Lynch et al., "Language Conditioned IL" | 2023 | Multi-task language-conditioned policies |
| Kelly et al., "HG-DAgger" | 2019 | Human-gated interventions |

---

## Connection to the Thread

> **Imitation learning = compressing expert demonstrations into a policy**. A few hundred
> demonstrations contain enough information to specify a task that would require millions
> of RL interactions to discover.
>
> **Action chunking = temporal compression**. Instead of deciding at every timestep, compress
> temporal behavior into chunks — fewer decisions, more coherent behavior.
>
> **The VLA insight: it's ALL token prediction.** A transformer doesn't care whether its
> input tokens represent words, image patches, or robot states. It doesn't care whether
> its output tokens represent the next word or the next action. The same attention mechanism
> processes everything. The key insight of VLAs is that by unifying all modalities in a
> single token-prediction framework, knowledge transfers freely between language, vision,
> and action.
>
> We've now completed the building blocks. In Phase VII, we'll see exactly how models like
> RT-2, Octo, and π₀ put these pieces together into working VLA systems.

---

## What's Next

In [Study Note 14](14-robot-data-eval.md), we'll study the **data infrastructure** that
makes robot learning possible: open datasets (Open X-Embodiment, DROID, Bridge V2),
simulation environments, evaluation metrics, and the LeRobot library. Then in the Phase VI
capstone, you'll train your own Diffusion Policy on real robot data. After that, we're
ready for the grand finale: Phase VII — actual VLA architectures.
