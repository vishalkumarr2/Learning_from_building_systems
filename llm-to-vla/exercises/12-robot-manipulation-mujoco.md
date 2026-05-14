# Exercise 12 — Robot Manipulation in MuJoCo
> Phase VI · Days 84-86 · ~8 hours · **Robotics-specific**

[← Exercise 11: VLA Experiments](11-vla-experiments.md) | [Exercise 13: Sim-to-Real →](13-sim-to-real-transfer.md)

---

## Objectives

By completing this exercise you will:
- Build custom MuJoCo environments for manipulation tasks
- Understand robot action spaces (joint vs task space, absolute vs delta)
- Implement forward kinematics and Jacobian-based control
- Engineer reward functions for contact-rich manipulation
- Train policies with RL and behavioral cloning in simulation
- Evaluate policy robustness with object/pose randomization

## Prerequisites
- Exercise 09 (RL fundamentals — REINFORCE, PPO)
- Exercise 10 (Imitation learning — BC, action chunking)
- Study Note 11 (RL Foundations) and Note 13 (Imitation Learning)

## Setup

```bash
pip install mujoco gymnasium robotics-gymnasium numpy torch matplotlib
```

> **Note**: `gymnasium-robotics` provides the Fetch and Adroit manipulation environments.
> MuJoCo is now open-source — no license needed since v2.1.2.

---

## Part 1: Understanding Robot State & Action Spaces (~1.5 hours)

### 1.1 — Explore a Manipulation Environment

```python
import gymnasium as gym
import numpy as np
import mujoco
import mujoco.viewer

# Fetch environments: robot arm + gripper + object on table
env = gym.make("FetchReach-v3", render_mode="human", max_episode_steps=50)
obs, info = env.reset()

print("=== Observation Space ===")
print(f"Type: {type(obs)}")
for key, val in obs.items():
    print(f"  {key}: shape={val.shape}, range=[{val.min():.3f}, {val.max():.3f}]")

print(f"\n=== Action Space ===")
print(f"Shape: {env.action_space.shape}")
print(f"Low:   {env.action_space.low}")
print(f"High:  {env.action_space.high}")
# Actions are 4D: (dx, dy, dz, gripper) — delta end-effector position + gripper

# Run a random episode
obs, info = env.reset()
total_reward = 0
for step in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"\nEpisode: {step+1} steps, total reward: {total_reward:.2f}")
print(f"Success: {info.get('is_success', False)}")
env.close()
```

**Questions to answer:**
1. What does each component of the observation represent?
2. Why are actions in *delta* (relative) rather than *absolute* coordinates?
3. What is the `achieved_goal` vs `desired_goal` in the observation?

### 1.2 — Joint Space vs Task Space

```python
import gymnasium as gym
import numpy as np

env = gym.make("FetchReach-v3", max_episode_steps=50)
obs, _ = env.reset()

# Access the underlying MuJoCo model
model = env.unwrapped.model
data = env.unwrapped.data

print("=== Robot Structure ===")
print(f"Number of joints (nq): {model.nq}")
print(f"Number of DoF (nv):    {model.nv}")
print(f"Number of actuators:   {model.nu}")

# Joint names and positions
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    qpos_addr = model.jnt_qposadr[i]
    print(f"  Joint {i}: {name:20s}  q={data.qpos[qpos_addr]:.4f}")

# End-effector position (task space)
ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "robot0:grip")
ee_pos = data.site_xpos[ee_site_id].copy()
print(f"\nEnd-effector position (task space): {ee_pos}")

# ── YOUR TASK ──
# Compute the full Jacobian of the end-effector
# The Jacobian maps joint velocities to end-effector velocities: v_ee = J @ dq

def compute_jacobian(model, data, site_id):
    """Compute the 6×nv Jacobian for a site (3 position + 3 orientation rows)."""
    jacp = np.zeros((3, model.nv))  # position Jacobian
    jacr = np.zeros((3, model.nv))  # rotation Jacobian
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    return np.vstack([jacp, jacr])  # 6 × nv

J = compute_jacobian(model, data, ee_site_id)
print(f"\nJacobian shape: {J.shape}")
print(f"Jacobian (position rows):\n{J[:3, :7]}")  # First 7 cols = arm joints

# Verify: small joint perturbation → predicted vs actual EE displacement
dq = np.zeros(model.nv)
dq[0] = 0.01  # perturb first joint
predicted_dx = J[:3, :] @ dq
print(f"\nPredicted EE displacement from dq[0]=0.01: {predicted_dx}")
```

### 1.3 — Action Space Comparison

```python
import gymnasium as gym
import numpy as np
import torch

env = gym.make("FetchReach-v3", max_episode_steps=50)

def run_episode_with_controller(env, controller_fn, name=""):
    """Run one episode, return trajectory."""
    obs, _ = env.reset()
    trajectory = {"ee_pos": [], "actions": [], "rewards": []}
    
    for step in range(50):
        action = controller_fn(obs, step)
        trajectory["ee_pos"].append(obs["achieved_goal"].copy())
        trajectory["actions"].append(action.copy())
        
        obs, reward, terminated, truncated, info = env.step(action)
        trajectory["rewards"].append(reward)
        if terminated or truncated:
            break
    
    success = info.get("is_success", False)
    print(f"{name}: {step+1} steps, reward={sum(trajectory['rewards']):.1f}, success={success}")
    return trajectory, success

# Controller 1: Move toward goal with delta actions
def proportional_controller(obs, step):
    """P-controller in task space — delta actions toward goal."""
    error = obs["desired_goal"] - obs["achieved_goal"]
    # Scale: clip large movements, add gripper=0 (no grasp needed for reach)
    action = np.clip(error * 5.0, -1.0, 1.0)
    return np.append(action, 0.0)  # [dx, dy, dz, gripper]

# Controller 2: Random noise (baseline)
def random_controller(obs, step):
    return env.action_space.sample()

# ── YOUR TASK ──
# Controller 3: Implement a bang-bang controller that moves at max speed
#                along the axis with the largest error first
def bang_bang_controller(obs, step):
    """Bang-bang controller: move max speed along largest error axis."""
    error = obs["desired_goal"] - obs["achieved_goal"]
    action = np.zeros(4)
    # ___  # YOUR CODE: identify largest error axis, set that to +1 or -1
    return action

# Compare all three
print("=== Controller Comparison ===")
for name, ctrl in [("P-control", proportional_controller),
                    ("Random", random_controller),
                    ("Bang-bang", bang_bang_controller)]:
    successes = sum(run_episode_with_controller(env, ctrl, name)[1] for _ in range(10))
    print(f"  {name}: {successes}/10 successes\n")

env.close()
```

> **Key insight**: The action space defines the *interface* between the policy and the robot.
> Delta task-space actions (dx, dy, dz) are much easier for learning than joint-space
> torques, because the policy doesn't need to learn inverse kinematics internally.
> This is why most VLAs use delta EE actions or discretized action tokens.

---

## Part 2: Building a Custom Manipulation Environment (~2 hours)

### 2.1 — Custom Pick-and-Place with Reward Engineering

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

# We'll build a simplified pick-and-place on top of Gymnasium
# Key concepts: reward shaping, contact detection, curriculum

PICK_PLACE_XML = """
<mujoco model="pick_place">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  
  <worldbody>
    <!-- Table -->
    <body name="table" pos="0.5 0 0.4">
      <geom type="box" size="0.3 0.3 0.02" rgba="0.5 0.4 0.3 1" condim="3"/>
    </body>
    
    <!-- Object to pick -->
    <body name="object" pos="0.5 0 0.45">
      <joint name="obj_x" type="slide" axis="1 0 0" damping="0.01"/>
      <joint name="obj_y" type="slide" axis="0 1 0" damping="0.01"/>
      <joint name="obj_z" type="slide" axis="0 0 1" damping="0.01"/>
      <geom name="object_geom" type="box" size="0.02 0.02 0.02" 
            rgba="1 0 0 1" mass="0.05" condim="3" friction="1.5 0.005 0.0001"/>
    </body>
    
    <!-- Goal indicator (visual only) -->
    <body name="goal" pos="0.6 0.1 0.5">
      <geom type="sphere" size="0.02" rgba="0 1 0 0.3" contype="0" conaffinity="0"/>
    </body>
    
    <!-- Simple 3-DoF "arm" (point mass that can grip) -->
    <body name="gripper" pos="0.5 0 0.6">
      <joint name="grip_x" type="slide" axis="1 0 0" range="-0.3 0.3" damping="1"/>
      <joint name="grip_y" type="slide" axis="0 1 0" range="-0.3 0.3" damping="1"/>
      <joint name="grip_z" type="slide" axis="0 0 1" range="0.42 0.7" damping="1"/>
      <geom name="gripper_geom" type="sphere" size="0.025" rgba="0 0 1 0.8" mass="0.5"/>
    </body>
    
    <light pos="0.5 0 1.5" dir="0 0 -1"/>
  </worldbody>
  
  <actuator>
    <position joint="grip_x" kp="50" ctrlrange="-0.3 0.3"/>
    <position joint="grip_y" kp="50" ctrlrange="-0.3 0.3"/>
    <position joint="grip_z" kp="50" ctrlrange="0.42 0.7"/>
  </actuator>
</mujoco>
"""


class SimplePickPlaceEnv(gym.Env):
    """
    Minimal pick-and-place:
    - 3D gripper position control
    - Gripper "grips" when close to object (magnetic grip)
    - Goal: move object to target position
    """
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, render_mode=None, reward_type="dense"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_string(PICK_PLACE_XML)
        self.data = mujoco.MjData(self.model)
        self.reward_type = reward_type
        self.render_mode = render_mode
        
        # Action: target (x, y, z) for gripper
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        
        # Observation: gripper_pos(3) + object_pos(3) + goal_pos(3) + grip_state(1)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(10,), dtype=np.float32)
        
        self.goal = np.array([0.6, 0.1, 0.5])
        self.gripping = False
        self.max_steps = 100
        self.step_count = 0
        
        if render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None
    
    def _get_obs(self):
        grip_pos = self.data.qpos[:3].copy()   # gripper x,y,z
        obj_pos = self.data.qpos[3:6].copy()    # object x,y,z
        return np.concatenate([grip_pos, obj_pos, self.goal, [float(self.gripping)]])
    
    def _grip_check(self):
        """Magnetic grip: attach object if gripper is close enough."""
        grip_pos = self.data.qpos[:3]
        obj_pos = self.data.qpos[3:6]
        dist = np.linalg.norm(grip_pos - obj_pos)
        if dist < 0.04:  # close enough to grip
            self.gripping = True
        if self.gripping:
            # Move object to gripper position (magnetic attachment)
            self.data.qpos[3:6] = self.data.qpos[:3].copy()
            self.data.qpos[5] -= 0.03  # offset below gripper
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Randomize object position on table
        self.data.qpos[3] = 0.5 + self.np_random.uniform(-0.1, 0.1)  # obj_x
        self.data.qpos[4] = self.np_random.uniform(-0.1, 0.1)         # obj_y
        self.data.qpos[5] = 0.45                                       # obj_z (on table)
        
        # Randomize goal
        self.goal = np.array([
            0.5 + self.np_random.uniform(-0.15, 0.15),
            self.np_random.uniform(-0.15, 0.15),
            0.45 + self.np_random.uniform(0.05, 0.2),  # always above table
        ])
        
        self.gripping = False
        self.step_count = 0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}
    
    def step(self, action):
        # Scale action to control range
        self.data.ctrl[:3] = np.clip(action * 0.3, -0.3, 0.3)
        
        # Step simulation
        for _ in range(10):  # 10 substeps per action
            mujoco.mj_step(self.model, self.data)
        self._grip_check()
        
        if self.viewer:
            self.viewer.sync()
        
        obs = self._get_obs()
        obj_pos = obs[3:6]
        
        # Reward computation
        dist_to_goal = np.linalg.norm(obj_pos - self.goal)
        grip_to_obj = np.linalg.norm(obs[:3] - obj_pos)
        
        if self.reward_type == "dense":
            # ── YOUR TASK: Design a staged reward function ──
            # Phase 1: Approach object (reward for closing distance to object)
            # Phase 2: Grip (bonus when gripping succeeds)
            # Phase 3: Lift & carry to goal (reward for object→goal distance)
            
            if not self.gripping:
                reward = -grip_to_obj  # approach reward
            else:
                reward = -dist_to_goal + 0.5  # carry reward + grip bonus
        else:
            reward = -(dist_to_goal > 0.05).astype(np.float32)  # sparse: -1 or 0
        
        success = dist_to_goal < 0.05
        self.step_count += 1
        terminated = success
        truncated = self.step_count >= self.max_steps
        
        return obs, float(reward), terminated, truncated, {"is_success": success}
    
    def close(self):
        if self.viewer:
            self.viewer.close()


# Test the environment
env = SimplePickPlaceEnv(reward_type="dense")
obs, _ = env.reset()
print(f"Observation shape: {obs.shape}")
print(f"Initial obs: grip={obs[:3]}, obj={obs[3:6]}, goal={obs[6:9]}, grip={obs[9]}")

# Run with a simple scripted policy
total_rewards = []
for ep in range(5):
    obs, _ = env.reset()
    ep_reward = 0
    for step in range(100):
        # Simple strategy: go to object first, then go to goal
        if obs[9] < 0.5:  # not gripping
            action = (obs[3:6] - obs[:3]) * 5  # move toward object
        else:
            action = (obs[6:9] - obs[:3]) * 5  # move toward goal
        action = np.clip(action, -1, 1).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        if terminated or truncated:
            break
    total_rewards.append(ep_reward)
    print(f"Episode {ep+1}: reward={ep_reward:.2f}, success={info['is_success']}")

env.close()
print(f"\nMean reward: {np.mean(total_rewards):.2f}")
```

### 2.2 — Reward Engineering Experiments

```python
import numpy as np
import matplotlib.pyplot as plt

# ── YOUR TASK ──
# Implement 3 different reward functions and compare their training curves

def reward_sparse(obs, gripping, goal):
    """Binary: 0 if object at goal, -1 otherwise."""
    obj_pos = obs[3:6]
    return 0.0 if np.linalg.norm(obj_pos - goal) < 0.05 else -1.0

def reward_dense_v1(obs, gripping, goal):
    """Dense: negative distance from object to goal."""
    obj_pos = obs[3:6]
    return -np.linalg.norm(obj_pos - goal)

def reward_dense_v2(obs, gripping, goal):
    """
    Staged dense reward:
    Stage 1 (not gripping): reward for gripper→object distance  
    Stage 2 (gripping): reward for object→goal distance + grip bonus
    Stage 3 (at goal): large success bonus
    """
    grip_pos = obs[:3]
    obj_pos = obs[3:6]
    grip_to_obj = np.linalg.norm(grip_pos - obj_pos)
    obj_to_goal = np.linalg.norm(obj_pos - goal)
    
    reward = 0.0
    if not gripping:
        reward = -grip_to_obj * 2.0            # approach
    else:
        reward = -obj_to_goal + 1.0            # carry + grip bonus
    
    if obj_to_goal < 0.05:
        reward += 10.0                          # success bonus
    
    return reward

def reward_hindsight(obs, gripping, goal):
    """
    Hindsight-inspired: pretend the current object position IS the goal.
    This provides signal even when the object never reaches the real goal.
    (Simplified — real HER replays full episodes.)
    """
    obj_pos = obs[3:6]
    grip_pos = obs[:3]
    # Primary: distance to real goal
    primary = -np.linalg.norm(obj_pos - goal)
    # Auxiliary: reward for any controlled object movement
    auxiliary = -np.linalg.norm(grip_pos - obj_pos) * 0.5
    return primary + auxiliary

# ── YOUR TASK ──
# Train with each reward function for N episodes using the P-controller
# or a simple MLP policy. Plot mean episode reward over training.
# Which reward function leads to fastest learning?

# Expected ranking (for this task): dense_v2 > dense_v1 > hindsight >> sparse

print("Implement the training loop and plot comparison curves.")
print("Hint: use stable-baselines3 PPO with each reward variant.")
```

> **Key insight**: Reward engineering is critical for RL-based manipulation. Sparse
> rewards (success/fail) are hard to learn from. Dense staged rewards accelerate
> learning but can create reward hacking. This is why imitation learning (Phase VI)
> often works better for manipulation — the demonstrations implicitly encode the
> "how" that reward functions struggle to specify.

---

## Part 3: Training Policies for Manipulation (~2.5 hours)

### 3.1 — RL Training with PPO

```python
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from collections import deque

# Use our custom environment or Gymnasium's FetchReach
# For faster iteration, start with FetchReach (simpler)

class ManipulationPolicyMLP(nn.Module):
    """Actor-Critic MLP for manipulation."""
    def __init__(self, obs_dim=10, act_dim=3, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden, act_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(hidden, 1)
    
    def forward(self, obs):
        features = self.shared(obs)
        return self.actor_mean(features), self.critic(features)
    
    def get_action(self, obs, deterministic=False):
        mean, value = self.forward(obs)
        if deterministic:
            return mean, value
        std = self.actor_logstd.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value


def collect_rollout(env, policy, n_steps=2048):
    """Collect experience for PPO update."""
    obs_buf, act_buf, rew_buf, val_buf, logp_buf, done_buf = [], [], [], [], [], []
    obs, _ = env.reset()
    
    for _ in range(n_steps):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value = policy.get_action(obs_t)
        
        action_np = action.squeeze(0).numpy()
        action_np = np.clip(action_np, -1, 1)
        
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        
        obs_buf.append(obs)
        act_buf.append(action_np)
        rew_buf.append(reward)
        val_buf.append(value.item())
        logp_buf.append(log_prob.item())
        done_buf.append(terminated or truncated)
        
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    
    return {
        "obs": np.array(obs_buf, dtype=np.float32),
        "actions": np.array(act_buf, dtype=np.float32),
        "rewards": np.array(rew_buf, dtype=np.float32),
        "values": np.array(val_buf, dtype=np.float32),
        "log_probs": np.array(logp_buf, dtype=np.float32),
        "dones": np.array(done_buf, dtype=np.float32),
    }


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Generalized Advantage Estimation."""
    advantages = np.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
    returns = advantages + values
    return advantages, returns


def ppo_update(policy, optimizer, rollout, epochs=10, clip_eps=0.2, batch_size=64):
    """PPO clipped objective update."""
    obs = torch.FloatTensor(rollout["obs"])
    actions = torch.FloatTensor(rollout["actions"])
    old_log_probs = torch.FloatTensor(rollout["log_probs"])
    
    advantages, returns = compute_gae(
        rollout["rewards"], rollout["values"], rollout["dones"]
    )
    advantages = torch.FloatTensor(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = torch.FloatTensor(returns)
    
    total_loss = 0
    n_updates = 0
    
    for _ in range(epochs):
        indices = torch.randperm(len(obs))
        for start in range(0, len(obs), batch_size):
            idx = indices[start:start + batch_size]
            
            mean, value = policy(obs[idx])
            std = policy.actor_logstd.exp()
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions[idx]).sum(-1)
            
            # PPO clipped objective
            ratio = (new_log_probs - old_log_probs[idx]).exp()
            clip_adv = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[idx]
            policy_loss = -torch.min(ratio * advantages[idx], clip_adv).mean()
            
            # Value loss
            value_loss = 0.5 * (returns[idx] - value.squeeze(-1)).pow(2).mean()
            
            # Entropy bonus (exploration)
            entropy = dist.entropy().sum(-1).mean()
            
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            n_updates += 1
    
    return total_loss / max(n_updates, 1)


# ── TRAINING LOOP ──
env = SimplePickPlaceEnv(reward_type="dense")
policy = ManipulationPolicyMLP(obs_dim=10, act_dim=3)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

success_history = deque(maxlen=50)

for iteration in range(100):
    rollout = collect_rollout(env, policy, n_steps=2048)
    loss = ppo_update(policy, optimizer, rollout)
    
    # Evaluate
    eval_successes = 0
    for _ in range(10):
        obs, _ = env.reset()
        for step in range(100):
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, _ = policy.get_action(obs_t, deterministic=True)
            obs, _, terminated, truncated, info = env.step(
                action.squeeze(0).numpy().clip(-1, 1)
            )
            if terminated or truncated:
                break
        eval_successes += info.get("is_success", False)
    
    success_rate = eval_successes / 10
    success_history.append(success_rate)
    
    if (iteration + 1) % 10 == 0:
        print(f"Iter {iteration+1:3d} | Loss: {loss:.4f} | "
              f"Success: {success_rate:.0%} | "
              f"Avg(50): {np.mean(success_history):.0%}")

env.close()
print(f"\nFinal success rate (last 50): {np.mean(success_history):.0%}")
```

### 3.2 — Behavioral Cloning from Scripted Demonstrations

```python
import torch
import torch.nn as nn
import numpy as np

def collect_demonstrations(env, n_demos=200):
    """Collect expert demonstrations using the scripted P-controller."""
    demos = {"obs": [], "actions": []}
    successes = 0
    
    for _ in range(n_demos):
        obs, _ = env.reset()
        for step in range(100):
            # Expert policy: P-controller
            if obs[9] < 0.5:  # not gripping
                action = (obs[3:6] - obs[:3]) * 5.0
            else:
                action = (obs[6:9] - obs[:3]) * 5.0
            action = np.clip(action, -1, 1).astype(np.float32)
            
            demos["obs"].append(obs.copy())
            demos["actions"].append(action.copy())
            
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                successes += info.get("is_success", False)
                break
    
    print(f"Collected {len(demos['obs'])} transitions from {n_demos} demos "
          f"({successes}/{n_demos} successful)")
    
    return {
        "obs": np.array(demos["obs"], dtype=np.float32),
        "actions": np.array(demos["actions"], dtype=np.float32),
    }


class BCPolicy(nn.Module):
    """Simple MLP for behavioral cloning."""
    def __init__(self, obs_dim=10, act_dim=3, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),  # actions in [-1, 1]
        )
    
    def forward(self, obs):
        return self.net(obs)


# Collect data
env = SimplePickPlaceEnv(reward_type="dense")
demos = collect_demonstrations(env, n_demos=500)

# Train BC
bc_policy = BCPolicy()
optimizer = torch.optim.Adam(bc_policy.parameters(), lr=1e-3)

obs_tensor = torch.FloatTensor(demos["obs"])
act_tensor = torch.FloatTensor(demos["actions"])
dataset = torch.utils.data.TensorDataset(obs_tensor, act_tensor)
loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

for epoch in range(50):
    total_loss = 0
    for obs_batch, act_batch in loader:
        pred = bc_policy(obs_batch)
        loss = nn.functional.mse_loss(pred, act_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1:3d} | BC Loss: {avg_loss:.6f}")

# Evaluate
successes = 0
for _ in range(50):
    obs, _ = env.reset()
    for step in range(100):
        with torch.no_grad():
            action = bc_policy(torch.FloatTensor(obs).unsqueeze(0)).squeeze(0).numpy()
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    successes += info.get("is_success", False)

print(f"\nBC Success rate: {successes}/50 = {successes/50:.0%}")

# ── YOUR TASK ──
# 1. Compare BC success rate vs PPO success rate — which learns faster?
# 2. Add Gaussian noise to BC actions during evaluation. At what noise level
#    does performance collapse? This demonstrates covariate shift.
# 3. Try BC with only 20 demonstrations vs 500. How does data efficiency compare?

env.close()
```

---

## Part 4: Object & Pose Randomization (~1 hour)

### 4.1 — Domain Randomization for Robustness

```python
import numpy as np

class RandomizedPickPlaceEnv(SimplePickPlaceEnv):
    """Pick-and-place with visual and dynamics randomization."""
    
    def __init__(self, randomize_dynamics=True, randomize_visuals=True, **kwargs):
        super().__init__(**kwargs)
        self.randomize_dynamics = randomize_dynamics
        self.randomize_visuals = randomize_visuals
        self._default_friction = self.model.geom_friction.copy()
        self._default_mass = self.model.body_mass.copy()
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        if self.randomize_dynamics:
            # Randomize object mass: 0.5x to 2x
            obj_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "object"
            )
            scale = self.np_random.uniform(0.5, 2.0)
            self.model.body_mass[obj_body_id] = self._default_mass[obj_body_id] * scale
            
            # Randomize friction: 0.5x to 1.5x
            obj_geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom"
            )
            self.model.geom_friction[obj_geom_id] = (
                self._default_friction[obj_geom_id] 
                * self.np_random.uniform(0.5, 1.5, size=3)
            )
        
        if self.randomize_visuals:
            # Randomize object color
            obj_geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom"
            )
            self.model.geom_rgba[obj_geom_id, :3] = self.np_random.uniform(0.2, 1.0, size=3)
        
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), info


# ── YOUR TASK ──
# 1. Train a policy with NO randomization → evaluate on randomized env
# 2. Train a policy WITH randomization → evaluate on randomized env
# 3. Compare success rates — how much does randomization help generalization?

# Expected: policy trained with randomization should maintain ~70%+ success
# on randomized eval, while the non-randomized policy may drop to <30%.

env_standard = SimplePickPlaceEnv(reward_type="dense")
env_random = RandomizedPickPlaceEnv(reward_type="dense")

print("Train on standard, evaluate on randomized — expect performance drop")
print("Train on randomized, evaluate on randomized — expect robustness")
```

---

## Part 5: Analysis & Connection to VLAs (~1 hour)

### 5.1 — Action Space Trade-offs

```python
# ── YOUR TASK ──
# Fill in this comparison table based on what you learned:

action_space_comparison = """
| Action Space     | Dimensionality | Learning Ease | Precision | Safety   | Used By          |
|-----------------|----------------|---------------|-----------|----------|------------------|
| Joint torques   | nq (e.g. 7)   | Hard          | ___       | ___      | Classic RL       |
| Joint positions | nq             | Medium        | High      | ___      | RT-1, Octo       |
| Delta EE pos    | 3-6            | ___           | Medium    | ___      | Fetch envs, ACT  |
| Abs EE pos      | 3-6            | Easy          | ___       | Medium   | Some BC methods  |
| Action tokens   | Discrete (256) | ___           | ___       | High     | RT-2, OpenVLA    |
"""
print(action_space_comparison)

# ── YOUR TASK ──
# Write a 200-word analysis: "Why do VLAs prefer discretized action tokens
# over continuous actions? What are the trade-offs?"
```

### 5.2 — From This Exercise to VLAs

```
┌──────────────────────────────────────────────────────────────────┐
│                  What You Built (This Exercise)                  │
│                                                                  │
│  obs (10D) ──→ MLP Policy ──→ action (3D) ──→ MuJoCo           │
│                                                                  │
│                  What VLAs Do (Exercise 11)                       │
│                                                                  │
│  image + "pick the red block" ──→ VLA (7B) ──→ action tokens    │
│                                     │             ──→ detokenize │
│                              ┌──────┴──────┐       ──→ robot    │
│                              │ Vision encoder│                    │
│                              │ Language model │                    │
│                              │ Action decoder │                    │
│                              └───────────────┘                    │
│                                                                  │
│  Same fundamentals:                                              │
│  ✓ Action spaces (you compared 5 types)                         │
│  ✓ Reward vs demonstration (RL vs BC — you trained both)        │
│  ✓ Randomization for robustness (you measured the gap)          │
│  ✓ Contact-rich manipulation (gripping, carrying)               │
│                                                                  │
│  What VLAs add:                                                  │
│  + Vision instead of state vectors                               │
│  + Language conditioning instead of fixed goal vectors           │
│  + Massive pretraining (billions of tokens)                      │
│  + Action tokenization instead of continuous output              │
└──────────────────────────────────────────────────────────────────┘
```

---

## Checklist

- [ ] Explored FetchReach observation/action spaces
- [ ] Computed end-effector Jacobian
- [ ] Compared P-controller, random, and bang-bang controllers
- [ ] Built custom pick-and-place MuJoCo environment
- [ ] Implemented and compared 3+ reward functions
- [ ] Trained PPO policy for manipulation
- [ ] Trained BC policy from scripted demonstrations
- [ ] Compared RL vs BC learning efficiency
- [ ] Measured BC sensitivity to noise (covariate shift demo)
- [ ] Trained with domain randomization and measured robustness gap
- [ ] Completed action space trade-off table
- [ ] Wrote VLA action tokenization analysis (200 words)

## Expected Results
- P-controller: ~90% success on FetchReach
- PPO after 100 iterations: >60% success on pick-and-place
- BC with 500 demos: >70% success on pick-and-place
- BC with noise σ=0.3: success drops to <20% (covariate shift)
- Domain randomization: maintains >60% on randomized eval vs <30% without

---

> **Connection to the Compression Thread**: A policy is a *compressed representation*
> of the mapping from states to actions. RL compresses reward signals over many
> episodes; BC compresses demonstration trajectories. VLAs compress all of
> language + vision + action into a single transformer — the ultimate compression
> of robotic behavior into next-token prediction.

[← Exercise 11: VLA Experiments](11-vla-experiments.md) | [Exercise 13: Sim-to-Real →](13-sim-to-real-transfer.md)
