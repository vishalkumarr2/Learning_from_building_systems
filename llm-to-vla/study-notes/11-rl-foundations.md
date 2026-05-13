# 11 — RL Foundations for Robotics

> Phase VI · Days 71–73 · ~8 hours
> Prerequisites: 01-dl-foundations
> Learning Objectives: Understand RL fundamentals, policy gradient, PPO, why RL alone isn't enough for robotics

---

## Navigation

| Previous | [10-vision-language-models](10-vision-language-models.md) | Next | [12-diffusion-flow](12-diffusion-flow.md) |
|----------|---------------------------------------------|------|-------------------------------------------|

---

## Overview

Reinforcement learning is the mathematical framework for learning optimal behavior from
interaction. An agent takes actions in an environment, receives rewards, and learns a policy
that maximizes cumulative reward. While RL has produced spectacular results in games (AlphaGo,
Atari), its application to real robotics faces fundamental challenges that motivate imitation
learning and VLA approaches.

**Why study RL if we're heading toward imitation learning?**
1. RL provides the *theoretical foundation* for sequential decision-making
2. Many VLA training pipelines use RL fine-tuning after imitation pretraining
3. Understanding RL failures in robotics clarifies why demonstrations are so valuable
4. PPO is the backbone of RLHF used in LLM alignment (connects to Phase III)

---

## Day 71 — RL Basics for Our Purposes

### The Markov Decision Process (MDP)

An MDP is defined by the tuple $(S, A, P, R, \gamma)$:

| Symbol | Meaning | Robotics Example |
|--------|---------|------------------|
| $S$ | State space | Joint angles, end-effector pose, camera image |
| $A$ | Action space | Joint torques, velocity commands, gripper open/close |
| $P(s'|s,a)$ | Transition dynamics | Physics of the robot + environment |
| $R(s,a,s')$ | Reward function | +1 for grasping object, -0.01 per timestep |
| $\gamma$ | Discount factor | Typically 0.99 — how much we value future rewards |

The agent's goal: find a policy $\pi(a|s)$ that maximizes the expected cumulative discounted
reward:

$$J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{T} \gamma^t R(s_t, a_t, s_{t+1})\right]$$

where $\tau = (s_0, a_0, s_1, a_1, \ldots)$ is a trajectory sampled under policy $\pi$.

### Value Functions

**State value function** — expected return starting from state $s$, following policy $\pi$:

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_t \mid s_0 = s\right]$$

**Action-value (Q) function** — expected return starting from state $s$, taking action $a$,
then following policy $\pi$:

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_t \mid s_0 = s, a_0 = a\right]$$

**Advantage function** — how much better is action $a$ than average?

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

The advantage function is crucial for policy gradient methods — it tells us which actions
are better than expected, not just which states are good.

### Bellman Equations

The recursive structure of value functions:

$$V^\pi(s) = \mathbb{E}_{a \sim \pi}\left[R(s,a) + \gamma \mathbb{E}_{s' \sim P}[V^\pi(s')]\right]$$

$$Q^\pi(s,a) = R(s,a) + \gamma \mathbb{E}_{s' \sim P}\left[\mathbb{E}_{a' \sim \pi}[Q^\pi(s', a')]\right]$$

**Optimal Bellman equation** (for $Q^*$):

$$Q^*(s,a) = R(s,a) + \gamma \mathbb{E}_{s' \sim P}\left[\max_{a'} Q^*(s', a')\right]$$

### Policy-Based vs Value-Based Methods

| Aspect | Value-Based (DQN) | Policy-Based (PG) |
|--------|-------------------|-------------------|
| What is learned | $Q(s,a)$ → derive policy | $\pi_\theta(a|s)$ directly |
| Action space | Discrete (enumerate all actions) | Continuous or discrete |
| Stochastic policies | Hard (ε-greedy hack) | Natural |
| Convergence | Can be unstable | Guaranteed local convergence |
| Sample efficiency | Better (off-policy possible) | Worse (on-policy) |
| Robotics fit | Poor (continuous actions) | **Better** |

**For robotics**: policy gradient methods dominate because:
- Actions are continuous (joint torques, velocities)
- We want stochastic policies (for exploration and multimodal behavior)
- Direct policy optimization is more natural for control

### Exploration vs Exploitation

The fundamental dilemma:
- **Exploit**: use current best policy to maximize reward
- **Explore**: try new actions to discover potentially better strategies

In robotics, exploration is dangerous and expensive:
- Random exploration can damage the robot or environment
- Each real-world interaction takes real time (no fast-forward)
- The cost of a bad action in the real world is high

This is a key motivation for **imitation learning**: instead of exploring randomly, start
from expert demonstrations.

### Key Concepts Checklist — Day 71

- [ ] Can define all components of an MDP for a robot manipulation task
- [ ] Understand the difference between $V(s)$ and $Q(s,a)$
- [ ] Know why advantage functions reduce variance
- [ ] Can explain why policy methods are preferred for continuous control
- [ ] Understand why exploration is problematic in robotics

---

## Day 72 — Policy Gradient & PPO

### REINFORCE: The Simplest Policy Gradient

Start with a parameterized policy $\pi_\theta(a|s)$ (e.g., a neural network outputting
action distribution). We want to maximize:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_t \gamma^t R_t\right]$$

**The Policy Gradient Theorem** (Williams, 1992):

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

where $G_t = \sum_{k=t}^{T} \gamma^{k-t} R_k$ is the return from timestep $t$.

**Intuition**: increase the probability of actions that led to high returns, decrease the
probability of actions that led to low returns.

**REINFORCE algorithm**:
```
for each episode:
    collect trajectory τ = (s₀, a₀, r₀, s₁, a₁, r₁, ...)
    for each timestep t:
        compute return Gₜ = Σ γᵏ rₜ₊ₖ
        θ ← θ + α · ∇log π_θ(aₜ|sₜ) · Gₜ
```

**Problem**: high variance! The return $G_t$ varies wildly between episodes, making
gradient estimates noisy.

### Variance Reduction with Baselines

Replace $G_t$ with the **advantage** $A_t = G_t - b(s_t)$ where $b(s_t)$ is a baseline
(typically the value function $V(s_t)$):

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A^\pi(s_t, a_t)\right]$$

**Why this works**: subtracting the baseline doesn't change the expected gradient (unbiased)
but dramatically reduces variance. The advantage tells us: "was this action better or worse
than average from this state?"

### Generalized Advantage Estimation (GAE)

GAE (Schulman et al., 2016) balances bias and variance with parameter $\lambda$:

$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD residual.

- $\lambda = 0$: low variance, high bias (1-step TD)
- $\lambda = 1$: high variance, low bias (Monte Carlo)
- $\lambda = 0.95$: typical sweet spot

### PPO: Proximal Policy Optimization

PPO (Schulman et al., 2017) is the workhorse of modern RL. Key insight: prevent
destructively large policy updates.

**The clipped surrogate objective**:

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ is the
  probability ratio
- $\epsilon$ is the clip parameter (typically 0.2)
- $\hat{A}_t$ is the estimated advantage

**How the clip works**:

| Advantage | Ratio too high ($r > 1+\epsilon$) | Ratio too low ($r < 1-\epsilon$) |
|-----------|-----------------------------------|----------------------------------|
| $A > 0$ (good action) | Clip: don't increase probability too much | Allow: increase probability |
| $A < 0$ (bad action) | Allow: decrease probability | Clip: don't decrease too much |

**PPO algorithm**:
```
for each iteration:
    collect trajectories with current policy π_θ_old
    compute advantages Â using GAE
    for each mini-batch epoch (typically 3-10):
        compute ratio r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
        compute clipped objective L^CLIP
        update θ via gradient ascent on L^CLIP
    θ_old ← θ
```

### Why PPO Dominates

1. **Robust**: works across many environments with same hyperparameters
2. **Simple**: much easier to implement than TRPO (trust region methods)
3. **Stable**: clipping prevents catastrophic policy updates
4. **Scalable**: easily parallelized (collect trajectories across many workers)
5. **Foundation for RLHF**: PPO is used to fine-tune LLMs with human feedback

### PPO Hyperparameters

| Hyperparameter | Typical Value | Effect |
|----------------|--------------|--------|
| Clip $\epsilon$ | 0.2 | Constrains policy update size |
| GAE $\lambda$ | 0.95 | Bias-variance tradeoff for advantages |
| Learning rate | 3e-4 | Adam optimizer step size |
| Epochs per iteration | 3-10 | How many passes over collected data |
| Batch size | 2048+ | Transitions per update |
| Discount $\gamma$ | 0.99 | How far ahead to look |

### Key Concepts Checklist — Day 72

- [ ] Can derive the REINFORCE update rule
- [ ] Understand why baselines reduce variance without introducing bias
- [ ] Can explain the PPO clipped objective and why it prevents large updates
- [ ] Know the connection between PPO and RLHF in LLMs
- [ ] Understand GAE as a bias-variance tradeoff

---

## Day 73 — RL for Robotics: Challenges

### The Sim-to-Real Gap

Training RL in simulation is fast but transfer to real robots is hard:

| Simulation | Reality |
|-----------|---------|
| Perfect physics | Friction, deformation, contact dynamics |
| Perfect sensing | Noisy cameras, encoder drift |
| Instant reset | Manual reset takes minutes |
| Unlimited episodes | Robot wear, safety concerns |
| Deterministic | Non-deterministic, non-stationary |

**Approaches to bridge the gap**:
- **Domain randomization**: randomize physics parameters during sim training
- **System identification**: measure real-world parameters, match in sim
- **Sim-to-real transfer**: progressive adaptation from sim to real
- **Real-world fine-tuning**: sim pretrain → real fine-tune (few-shot)

### The Sample Efficiency Disaster

RL algorithms need enormous amounts of data:

| Algorithm | Samples for Basic Manipulation | Wall-Clock Time (Real Robot) |
|-----------|-------------------------------|------------------------------|
| PPO | ~10M timesteps | ~3000 hours (125 days!) |
| SAC | ~1M timesteps | ~300 hours (12 days) |
| Model-based | ~100K timesteps | ~30 hours |
| BC (imitation) | ~100 demonstrations | ~1 hour of demos |

**This is the fundamental argument for imitation learning**: a human demonstrator
compresses millions of timesteps of experience into a few hundred demonstrations.

### Reward Engineering

Designing reward functions is surprisingly hard:

**Example: "pick up the red block"**
- Sparse reward: +1 when block is grasped → agent never discovers it randomly
- Dense reward: distance to block → agent hovers hand near block, never grasps
- Shaped reward: distance + height + grasp → agent finds unintended shortcuts

**Reward hacking**: the agent optimizes the reward function, not the intended task.
The reward function is a lossy compression of human intent.

**Comparison with imitation learning**: demonstrations ARE the reward signal.
The expert implicitly encodes the task objective through their behavior.

### Safety Constraints

Real robots can:
- Damage themselves (joint limits, collisions)
- Damage the environment (breaking objects, scratching surfaces)
- Harm humans (contact forces, unexpected movements)

RL exploration involves taking random actions → unsafe by default.

**Safe RL approaches**:
- Constrained MDPs with safety constraints
- Action space limits and velocity bounds
- Human oversight during training
- Sim-only exploration + real-world deployment

### Sparse Rewards and Credit Assignment

In robotics, success/failure is often binary and delayed:
- "Did the robot successfully assemble the furniture?" → one bit of feedback after
  thousands of actions
- Which actions were responsible for success or failure?
- **Credit assignment problem**: distributing the final reward across the trajectory

**Hindsight Experience Replay (HER)**: reinterpret failed trajectories as successful
for a different goal. "I didn't reach the target, but I reached *somewhere* — that's
a valid demonstration for reaching that location."

### Why Pure RL Rarely Works on Real Robots

| Challenge | Severity | RL Solution | Better Alternative |
|-----------|----------|-------------|-------------------|
| Sample efficiency | Critical | Model-based RL | Imitation learning |
| Reward design | High | Inverse RL | Demonstrations |
| Safety | Critical | Constrained RL | Supervised learning from safe demos |
| Sim-to-real | High | Domain randomization | Real demonstrations |
| Sparse rewards | High | HER, curiosity | Dense supervision from demos |

**The emerging paradigm**:
1. **Pretrain** with imitation learning (demonstrations)
2. **Fine-tune** with RL (for optimization beyond human performance)
3. This is exactly the LLM pattern: pretrain on data, RLHF for alignment

### Key Concepts Checklist — Day 73

- [ ] Can explain the sim-to-real gap and three approaches to address it
- [ ] Understand why sample efficiency makes pure RL impractical for real robots
- [ ] Can give an example of reward hacking
- [ ] Know why safety constraints make RL exploration dangerous
- [ ] Understand the pretrain-with-IL + fine-tune-with-RL paradigm

---

## Key Equations Reference

### MDP & Value Functions

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_t \mid s_0 = s\right]$$

$$Q^\pi(s, a) = \mathbb{E}_\pi\left[\sum_{t=0}^{\infty} \gamma^t R_t \mid s_0 = s, a_0 = a\right]$$

$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$

### Policy Gradient

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A^\pi(s_t, a_t)\right]$$

### GAE

$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

### PPO Clipped Objective

$$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$

---

## RL Algorithm Comparison

| Algorithm | Type | On/Off Policy | Action Space | Sample Eff. | Stability | Robotics Use |
|-----------|------|---------------|-------------|-------------|-----------|-------------|
| REINFORCE | PG | On | Continuous | Very low | Low | Rare |
| PPO | PG | On | Continuous | Low | **High** | **Common** |
| TRPO | PG | On | Continuous | Low | High | Moderate |
| SAC | Actor-Critic | **Off** | Continuous | **Medium** | High | **Common** |
| TD3 | Actor-Critic | Off | Continuous | Medium | Medium | Common |
| DQN | Value | Off | Discrete only | Medium | Medium | Rare |
| DDPG | Actor-Critic | Off | Continuous | Medium | Low | Declining |
| DrQ-v2 | Actor-Critic | Off | Continuous | **High** | Medium | Growing |

**For robotics, the winners are**: PPO (on-policy, stable) and SAC (off-policy, sample-efficient).

---

## Key Takeaways

1. **RL provides the theoretical foundation** for sequential decision-making, but pure RL
   is impractical for real robotics due to sample efficiency, safety, and reward design
   challenges.

2. **Policy gradient methods** (especially PPO) are preferred for robotics because they
   handle continuous action spaces naturally and are robust to hyperparameter choices.

3. **PPO's clipped objective** prevents catastrophically large policy updates — the same
   principle of "controlled updates" appears in LLM fine-tuning (RLHF).

4. **The sample efficiency gap** between RL (~millions of steps) and imitation learning
   (~hundreds of demonstrations) is the primary motivation for IL in robotics.

5. **The emerging paradigm**: pretrain with demonstrations (imitation), fine-tune with RL.
   This mirrors the LLM paradigm of pretraining + RLHF.

---

## Paper References

| Paper | Year | Key Contribution |
|-------|------|-----------------|
| Sutton & Barto, "RL: An Introduction" | 2018 | Textbook — comprehensive RL foundations |
| Williams, "Simple Statistical Gradient-Following" | 1992 | REINFORCE algorithm |
| Schulman et al., "High-Dimensional Continuous Control Using GAE" | 2016 | Generalized Advantage Estimation |
| Schulman et al., "Proximal Policy Optimization" | 2017 | PPO — the standard RL algorithm |
| Haarnoja et al., "Soft Actor-Critic" | 2018 | SAC — off-policy alternative to PPO |
| Tobin et al., "Domain Randomization for Sim-to-Real" | 2017 | Bridging the simulation gap |
| Andrychowicz et al., "Hindsight Experience Replay" | 2017 | Learning from failed episodes |

---

## Connection to the Thread

> **RL tries to learn optimal behavior from rewards alone** — but for robotics, this
> requires millions of interactions and careful reward engineering. **Demonstrations
> provide a much richer compression of expert knowledge**: a single demonstration
> implicitly encodes the reward function, the task structure, and safe behavior patterns.
>
> The VLA approach takes this further: by pretraining on internet-scale language and
> vision data, the model starts with a compressed world model before seeing a single
> robot demonstration. RL then serves as a fine-tuning tool, not the primary learning
> mechanism.
>
> **Pattern**: unsupervised pretraining (compress the world) → supervised fine-tuning
> (compress demonstrations) → RL fine-tuning (optimize beyond human performance).
> This is the recipe for both LLMs and VLAs.

---

## What's Next

In [Study Note 12](12-diffusion-flow.md), we'll explore **diffusion models** — a powerful
generative framework that turns noise into structured predictions. Diffusion policies can
model multimodal action distributions (multiple valid ways to do a task), which is crucial
for robot learning. Think of diffusion as iterative decompression: starting from maximum
entropy (noise) and progressively adding structure until a valid action emerges.
