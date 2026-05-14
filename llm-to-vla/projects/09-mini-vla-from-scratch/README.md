# Project 09 — Mini VLA from Scratch: Train, Evaluate & Deploy
> Phase VII · Days 108-112 · ~15 hours · **Integration capstone**

---

## Overview

Build a **complete, working VLA** from scratch — small enough to train on a single GPU
in under an hour, but architecturally faithful to production systems (RT-2, Octo, OpenVLA).

You will:
1. Collect demonstrations (reusing Exercise 14's scripted policy)
2. Build a vision encoder + language conditioning + action decoder
3. Train end-to-end on language-conditioned manipulation
4. Evaluate systematically with the safety pipeline from Exercise 16
5. Deploy in MuJoCo with real-time inference

**This is the "tie everything together" project** — every exercise and study note converges here.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Mini VLA Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Camera Image ──→ [Vision Encoder] ──→ visual_tokens [4 x 128]     │
│       (64x64 RGB)     (Tiny ViT)                                    │
│                                                                     │
│  Language Cmd ──→ [Text Encoder] ──→ lang_token [1 x 128]          │
│  ("pick red")     (CLIP text or learned)                            │
│                                                                     │
│  Proprioception ──→ [MLP] ──→ proprio_token [1 x 128]             │
│   (joint angles)                                                    │
│                                                                     │
│  [visual_tokens | lang_token | proprio_token] ──→ [Transformer]    │
│       (6 tokens total)                              (2 layers)      │
│                                                           │         │
│                                                           ▼         │
│                                              [Action Head (MLP)]    │
│                                                           │         │
│                                                           ▼         │
│                                              action [7D: Δxyz +    │
│                                               Δrpy + gripper]       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Exercise/Project | What it provides |
|-----------------|------------------|
| Study Note 08 (ViT) | Vision encoder architecture |
| Study Note 10 (VLMs) | Vision-language fusion |
| Study Note 13 (Imitation Learning) | Action prediction, BC loss |
| Exercise 14 (Data Collection) | Demo generation pipeline |
| Exercise 15 (Kinematics) | Action space understanding |
| Exercise 16 (Safety) | Deployment safety monitors |
| Project 08 (ROS2 Pipeline) | Deployment architecture |

---

## Setup

```bash
pip install torch torchvision numpy mujoco gymnasium h5py matplotlib tqdm
```

Hardware: Single GPU with ≥4GB VRAM (or CPU, slower but works).

---

## Part 1: Data Generation (~2 hours)

### 1.1 — Environment: Language-Conditioned Pick-and-Place

```python
"""
MuJoCo environment with colored blocks for language-conditioned manipulation.
Supports 4 commands: "pick red", "pick blue", "pick green", "place on target"
"""
import numpy as np
import mujoco
import mujoco.viewer
from dataclasses import dataclass
from pathlib import Path


# Minimal MJCF with colored blocks
ENV_XML = """
<mujoco model="mini_vla_env">
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <asset>
    <texture type="2d" name="grid" builtin="checker" width="512" height="512"
             rgb1="0.9 0.9 0.9" rgb2="0.7 0.7 0.7"/>
    <material name="grid_mat" texture="grid" texrepeat="4 4"/>
    <material name="red_mat" rgba="0.9 0.1 0.1 1"/>
    <material name="blue_mat" rgba="0.1 0.1 0.9 1"/>
    <material name="green_mat" rgba="0.1 0.8 0.1 1"/>
    <material name="target_mat" rgba="0.8 0.8 0.0 0.5"/>
    <material name="arm_mat" rgba="0.4 0.4 0.4 1"/>
  </asset>

  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.01" material="grid_mat"/>

    <!-- 3-DOF Planar Arm + Gripper -->
    <body name="base" pos="0 0 0.05">
      <joint name="joint0" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="0.5"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.2 0 0" material="arm_mat"/>
      <body name="link1" pos="0.2 0 0">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-2.5 2.5" damping="0.5"/>
        <geom type="capsule" size="0.015" fromto="0 0 0 0.15 0 0" material="arm_mat"/>
        <body name="link2" pos="0.15 0 0">
          <joint name="joint2" type="hinge" axis="0 0 1" range="-2.5 2.5" damping="0.3"/>
          <geom type="capsule" size="0.01" fromto="0 0 0 0.08 0 0" material="arm_mat"/>
          <!-- Gripper (simplified as actuated slide) -->
          <body name="gripper" pos="0.08 0 0">
            <joint name="gripper_joint" type="slide" axis="0 1 0" range="0 0.03" damping="0.1"/>
            <geom type="box" size="0.01 0.005 0.015" pos="0 0.015 0" material="arm_mat"/>
            <geom type="box" size="0.01 0.005 0.015" pos="0 -0.015 0" material="arm_mat"/>
          </body>
        </body>
      </body>
    </body>

    <!-- Objects -->
    <body name="red_block" pos="0.25 0.1 0.02">
      <joint type="free"/>
      <geom type="box" size="0.02 0.02 0.02" material="red_mat" mass="0.05"/>
    </body>
    <body name="blue_block" pos="0.2 -0.15 0.02">
      <joint type="free"/>
      <geom type="box" size="0.02 0.02 0.02" material="blue_mat" mass="0.05"/>
    </body>
    <body name="green_block" pos="0.3 -0.05 0.02">
      <joint type="free"/>
      <geom type="box" size="0.02 0.02 0.02" material="green_mat" mass="0.05"/>
    </body>

    <!-- Target zone -->
    <body name="target" pos="-0.15 0.2 0.01">
      <geom type="cylinder" size="0.04 0.005" material="target_mat"/>
    </body>

    <!-- Camera -->
    <camera name="overhead" pos="0 0 0.8" quat="1 0 0 0" fovy="60"/>
  </worldbody>

  <actuator>
    <motor joint="joint0" ctrlrange="-1 1" gear="50"/>
    <motor joint="joint1" ctrlrange="-1 1" gear="50"/>
    <motor joint="joint2" ctrlrange="-1 1" gear="30"/>
    <motor joint="gripper_joint" ctrlrange="-1 1" gear="10"/>
  </actuator>
</mujoco>
"""


LANGUAGE_COMMANDS = {
    "pick red": {"target_object": "red_block", "action": "pick"},
    "pick blue": {"target_object": "blue_block", "action": "pick"},
    "pick green": {"target_object": "green_block", "action": "pick"},
    "place on target": {"target_object": "target", "action": "place"},
}


@dataclass
class Observation:
    """Single observation from the environment."""
    image: np.ndarray          # [64, 64, 3] RGB
    joint_angles: np.ndarray   # [3] (arm joints)
    gripper_state: float       # 0=closed, 1=open
    ee_position: np.ndarray    # [2] (x, y) of end-effector
    language_command: str       # e.g., "pick red"


class MiniVLAEnv:
    """
    Language-conditioned manipulation environment.
    Provides image observations and accepts delta EE actions.
    """

    def __init__(self, image_size: int = 64, render: bool = False):
        self.image_size = image_size
        self.model = mujoco.MjModel.from_xml_string(ENV_XML)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=image_size, width=image_size)

        # Object positions (reset will randomize these)
        self.object_positions = {}
        self.current_command = ""

    def reset(self, command: str = None) -> Observation:
        """Reset environment with optional language command."""
        mujoco.mj_resetData(self.model, self.data)

        # Randomize object positions slightly
        for name in ["red_block", "blue_block", "green_block"]:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            jnt_id = self.model.body_jntadr[body_id]
            qpos_adr = self.model.jnt_qposadr[jnt_id]
            # Add small random offset to initial positions
            self.data.qpos[qpos_adr:qpos_adr+2] += np.random.uniform(-0.03, 0.03, size=2)

        # Random command if not specified
        if command is None:
            command = np.random.choice(list(LANGUAGE_COMMANDS.keys()))
        self.current_command = command

        mujoco.mj_forward(self.model, self.data)
        return self._get_observation()

    def step(self, action: np.ndarray) -> tuple[Observation, float, bool, dict]:
        """
        Execute action and return next observation.

        Args:
            action: [4] = [Δjoint0, Δjoint1, Δjoint2, gripper_command]
                    gripper_command: >0 = open, <0 = close

        Returns:
            (observation, reward, done, info)
        """
        # Apply action as position targets
        for i in range(3):
            current = self.data.qpos[i]
            target = current + action[i] * 0.1  # scale delta
            self.data.ctrl[i] = np.clip(target, -3.14, 3.14)

        # Gripper
        self.data.ctrl[3] = 1.0 if action[3] > 0 else -1.0

        # Step physics
        for _ in range(10):  # 10 substeps per action
            mujoco.mj_step(self.model, self.data)

        obs = self._get_observation()
        reward = self._compute_reward()
        done = self._check_done()

        return obs, reward, done, {"command": self.current_command}

    def _get_observation(self) -> Observation:
        """Render image and collect proprioception."""
        # Render overhead camera
        self.renderer.update_scene(self.data, camera="overhead")
        image = self.renderer.render()

        # Joint angles
        joint_angles = self.data.qpos[:3].copy()
        gripper_state = self.data.qpos[3] / 0.03  # normalize to [0, 1]

        # EE position (approximate from FK)
        ee_pos = self._get_ee_position()

        return Observation(
            image=image,
            joint_angles=joint_angles,
            gripper_state=gripper_state,
            ee_position=ee_pos,
            language_command=self.current_command,
        )

    def _get_ee_position(self) -> np.ndarray:
        """Compute end-effector position from joint angles."""
        q = self.data.qpos[:3]
        l1, l2, l3 = 0.2, 0.15, 0.08
        x = l1 * np.cos(q[0]) + l2 * np.cos(q[0]+q[1]) + l3 * np.cos(q[0]+q[1]+q[2])
        y = l1 * np.sin(q[0]) + l2 * np.sin(q[0]+q[1]) + l3 * np.sin(q[0]+q[1]+q[2])
        return np.array([x, y])

    def _compute_reward(self) -> float:
        """Sparse reward: 1.0 if task completed, 0.0 otherwise."""
        # Simplified — check if target object near target zone
        return 0.0  # Students implement

    def _check_done(self) -> bool:
        """Check if episode should end."""
        return False  # Students implement with success/timeout


def get_object_position(env: MiniVLAEnv, object_name: str) -> np.ndarray:
    """Get the XY position of a named object."""
    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    return env.data.xpos[body_id][:2].copy()
```

### 1.2 — Scripted Expert Policy for Data Collection

```python
"""
Scripted expert that generates demonstrations for each language command.
Reuses principles from Exercise 14 but now language-conditioned.
"""
import numpy as np
import h5py
from enum import Enum, auto


class PickPhase(Enum):
    REACH = auto()
    DESCEND = auto()
    GRASP = auto()
    LIFT = auto()
    MOVE_TO_TARGET = auto()
    PLACE = auto()
    DONE = auto()


class ScriptedExpert:
    """
    Language-conditioned scripted policy for demonstration generation.
    Given a command ("pick red", "pick blue", etc.), executes the task.
    """

    def __init__(self, env: MiniVLAEnv):
        self.env = env
        self.phase = PickPhase.REACH
        self.grasp_counter = 0

    def get_action(self, obs: Observation) -> np.ndarray:
        """
        Compute expert action based on current phase and command.

        Returns:
            [4] action: [Δj0, Δj1, Δj2, gripper]
        """
        command = obs.language_command
        task_info = LANGUAGE_COMMANDS[command]
        target_obj = task_info["target_object"]

        # Get positions
        obj_pos = get_object_position(self.env, target_obj)
        ee_pos = obs.ee_position
        target_pos = get_object_position(self.env, "target")

        # Simple proportional control toward target
        if self.phase == PickPhase.REACH:
            error = obj_pos - ee_pos
            if np.linalg.norm(error) < 0.02:
                self.phase = PickPhase.GRASP
            action = self._position_to_joint_delta(error * 2.0, obs)
            action[3] = 1.0  # open gripper

        elif self.phase == PickPhase.GRASP:
            self.grasp_counter += 1
            action = np.zeros(4)
            action[3] = -1.0  # close gripper
            if self.grasp_counter > 10:
                self.phase = PickPhase.MOVE_TO_TARGET

        elif self.phase == PickPhase.MOVE_TO_TARGET:
            error = target_pos - ee_pos
            if np.linalg.norm(error) < 0.03:
                self.phase = PickPhase.PLACE
            action = self._position_to_joint_delta(error * 1.5, obs)
            action[3] = -1.0  # keep closed

        elif self.phase == PickPhase.PLACE:
            action = np.zeros(4)
            action[3] = 1.0  # open gripper
            self.phase = PickPhase.DONE

        else:
            action = np.zeros(4)

        # Add small noise for diversity
        action[:3] += np.random.normal(0, 0.02, size=3)
        return np.clip(action, -1.0, 1.0)

    def _position_to_joint_delta(self, ee_delta: np.ndarray, obs: Observation) -> np.ndarray:
        """Approximate inverse: EE delta → joint delta (simplified Jacobian)."""
        # For a planar arm, approximate with finite differences
        q = obs.joint_angles
        J = np.zeros((2, 3))
        l1, l2, l3 = 0.2, 0.15, 0.08

        # Analytical Jacobian for planar 3-link
        s01 = np.sin(q[0] + q[1])
        c01 = np.cos(q[0] + q[1])
        s012 = np.sin(q[0] + q[1] + q[2])
        c012 = np.cos(q[0] + q[1] + q[2])

        J[0, 0] = -l1*np.sin(q[0]) - l2*s01 - l3*s012
        J[0, 1] = -l2*s01 - l3*s012
        J[0, 2] = -l3*s012
        J[1, 0] = l1*np.cos(q[0]) + l2*c01 + l3*c012
        J[1, 1] = l2*c01 + l3*c012
        J[1, 2] = l3*c012

        # Damped pseudoinverse
        JJT = J @ J.T + 0.01 * np.eye(2)
        dq = J.T @ np.linalg.solve(JJT, ee_delta)

        action = np.zeros(4)
        action[:3] = dq
        return action

    def reset(self):
        """Reset expert state for new episode."""
        self.phase = PickPhase.REACH
        self.grasp_counter = 0


def generate_dataset(
    n_episodes: int = 200,
    max_steps: int = 100,
    save_path: str = "mini_vla_demos.hdf5",
):
    """
    Generate language-conditioned demonstration dataset.

    Saves in HDF5 format compatible with Part 2 training.
    """
    env = MiniVLAEnv(image_size=64)
    expert = ScriptedExpert(env)

    commands = list(LANGUAGE_COMMANDS.keys())

    with h5py.File(save_path, 'w') as f:
        f.attrs['n_episodes'] = n_episodes
        f.attrs['image_size'] = 64
        f.attrs['action_dim'] = 4
        f.attrs['commands'] = [c.encode() for c in commands]

        for ep_idx in range(n_episodes):
            command = commands[ep_idx % len(commands)]
            obs = env.reset(command=command)
            expert.reset()

            images = []
            actions = []
            joint_angles = []

            for step in range(max_steps):
                action = expert.get_action(obs)

                images.append(obs.image)
                actions.append(action)
                joint_angles.append(obs.joint_angles)

                obs, reward, done, info = env.step(action)
                if done or expert.phase == PickPhase.DONE:
                    break

            # Save episode
            ep_grp = f.create_group(f"episode_{ep_idx:04d}")
            ep_grp.create_dataset("images", data=np.array(images), compression="gzip")
            ep_grp.create_dataset("actions", data=np.array(actions))
            ep_grp.create_dataset("joint_angles", data=np.array(joint_angles))
            ep_grp.attrs['command'] = command
            ep_grp.attrs['length'] = len(images)
            ep_grp.attrs['success'] = (expert.phase == PickPhase.DONE)

            if (ep_idx + 1) % 50 == 0:
                print(f"Generated {ep_idx + 1}/{n_episodes} episodes")

    print(f"Dataset saved: {save_path} ({n_episodes} episodes)")
```

---

## Part 2: Model Architecture (~3 hours)

### 2.1 — Vision Encoder (Tiny ViT)

```python
"""
Mini VLA model: Vision encoder + Language conditioning + Action decoder.
Small enough to train on a single GPU in <1 hour.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TinyViTEncoder(nn.Module):
    """
    Minimal Vision Transformer for 64x64 images.

    Architecture:
    - Patch size: 8x8 → 64 patches
    - Embedding dim: 128
    - 2 transformer layers
    - Output: 4 visual tokens (via learned pooling)
    """

    def __init__(self, image_size: int = 64, patch_size: int = 8,
                 embed_dim: int = 128, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2  # 64
        self.embed_dim = embed_dim

        # Patch embedding: Conv2d that converts patches to embeddings
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, embed_dim) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1, batch_first=True,
            )
            for _ in range(n_layers)
        ])

        # Learned pooling: 64 patches → 4 visual tokens
        self.n_output_tokens = 4
        self.pool_queries = nn.Parameter(torch.randn(1, self.n_output_tokens, embed_dim) * 0.02)
        self.pool_attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, 64, 64] normalized RGB

        Returns:
            visual_tokens: [B, 4, 128] — 4 visual tokens per image
        """
        B = images.shape[0]

        # Patch embedding
        x = self.patch_embed(images)  # [B, embed_dim, 8, 8]
        x = x.flatten(2).transpose(1, 2)  # [B, 64, embed_dim]

        # Add position embedding
        x = x + self.pos_embed

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Learned pooling: cross-attention from queries to patches
        queries = self.pool_queries.expand(B, -1, -1)
        visual_tokens, _ = self.pool_attention(queries, x, x)

        return visual_tokens  # [B, 4, 128]


class LanguageEncoder(nn.Module):
    """
    Simple language encoder for a small fixed vocabulary.

    In production: use CLIP text encoder or T5.
    Here: learned embeddings for our 4 commands (sufficient for the task).
    """

    def __init__(self, vocab_size: int = 20, embed_dim: int = 128, max_tokens: int = 5):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_tokens = max_tokens

        # Simple token embedding + positional
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_tokens, embed_dim) * 0.02)

        # Pool to single language token
        self.pool = nn.Sequential(
            nn.Linear(embed_dim * max_tokens, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Build vocabulary
        self.vocab = self._build_vocab()

    def _build_vocab(self) -> dict:
        """Simple word-level tokenizer."""
        words = ["<pad>", "pick", "place", "on", "red", "blue", "green", "target",
                 "the", "block", "put", "move", "to", "grab", "drop"]
        return {word: idx for idx, word in enumerate(words)}

    def tokenize(self, commands: list[str]) -> torch.Tensor:
        """Convert string commands to token indices."""
        batch_tokens = []
        for cmd in commands:
            words = cmd.lower().split()
            tokens = [self.vocab.get(w, 0) for w in words[:self.max_tokens]]
            # Pad to max_tokens
            tokens += [0] * (self.max_tokens - len(tokens))
            batch_tokens.append(tokens)
        return torch.tensor(batch_tokens, dtype=torch.long)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [B, max_tokens] token indices

        Returns:
            lang_token: [B, 1, 128] — single language embedding
        """
        B = token_ids.shape[0]

        x = self.token_embed(token_ids)  # [B, max_tokens, embed_dim]
        x = x + self.pos_embed

        # Pool to single token
        x_flat = x.reshape(B, -1)  # [B, max_tokens * embed_dim]
        lang_token = self.pool(x_flat)  # [B, embed_dim]

        return lang_token.unsqueeze(1)  # [B, 1, embed_dim]


class ProprioceptionEncoder(nn.Module):
    """Encode proprioceptive state (joint angles + gripper)."""

    def __init__(self, proprio_dim: int = 4, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proprio_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proprio: [B, 4] (3 joints + gripper)

        Returns:
            proprio_token: [B, 1, 128]
        """
        return self.net(proprio).unsqueeze(1)


class ActionDecoder(nn.Module):
    """
    Decode fused tokens into action prediction.

    Options (increasing complexity):
    1. MLP head (simplest — we use this)
    2. Gaussian mixture model (multi-modal actions)
    3. Diffusion head (best for complex manipulation)
    """

    def __init__(self, embed_dim: int = 128, n_tokens: int = 6,
                 action_dim: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim

        # Pool all tokens → single vector → action
        self.pool = nn.Sequential(
            nn.Linear(embed_dim * n_tokens, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Actions in [-1, 1]
        )

    def forward(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_tokens: [B, n_tokens, embed_dim] — concatenated tokens

        Returns:
            actions: [B, action_dim] predicted actions
        """
        B = fused_tokens.shape[0]
        x = fused_tokens.reshape(B, -1)  # Flatten all tokens
        return self.pool(x)


class MiniVLA(nn.Module):
    """
    Complete Mini VLA: vision + language + proprioception → action.

    Total parameters: ~500K (trainable on a laptop GPU).
    """

    def __init__(self, image_size: int = 64, action_dim: int = 4, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

        # Encoders
        self.vision_encoder = TinyViTEncoder(image_size=image_size, embed_dim=embed_dim)
        self.language_encoder = LanguageEncoder(embed_dim=embed_dim)
        self.proprio_encoder = ProprioceptionEncoder(embed_dim=embed_dim)

        # Fusion transformer (cross-modal attention)
        self.fusion = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=4,
            dim_feedforward=embed_dim * 4,
            dropout=0.1, batch_first=True,
        )

        # Action decoder
        # 4 visual + 1 language + 1 proprio = 6 tokens
        self.action_decoder = ActionDecoder(
            embed_dim=embed_dim, n_tokens=6, action_dim=action_dim
        )

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass.

        Args:
            images: [B, 3, 64, 64] normalized images
            token_ids: [B, max_tokens] language token IDs
            proprio: [B, 4] proprioception (joints + gripper)

        Returns:
            actions: [B, 4] predicted actions
        """
        # Encode each modality
        visual_tokens = self.vision_encoder(images)      # [B, 4, 128]
        lang_token = self.language_encoder(token_ids)     # [B, 1, 128]
        proprio_token = self.proprio_encoder(proprio)     # [B, 1, 128]

        # Concatenate tokens
        all_tokens = torch.cat([visual_tokens, lang_token, proprio_token], dim=1)  # [B, 6, 128]

        # Cross-modal fusion
        fused = self.fusion(all_tokens)  # [B, 6, 128]

        # Decode action
        action = self.action_decoder(fused)  # [B, 4]

        return action

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

### 2.2 — Model Verification

```python
# Verify the model works
model = MiniVLA(image_size=64, action_dim=4, embed_dim=128)
print(f"Mini VLA parameters: {model.count_parameters():,}")
# Expected: ~500K-800K parameters

# Test forward pass
B = 4
dummy_images = torch.randn(B, 3, 64, 64)
dummy_tokens = torch.randint(0, 15, (B, 5))
dummy_proprio = torch.randn(B, 4)

actions = model(dummy_images, dummy_tokens, dummy_proprio)
print(f"Output shape: {actions.shape}")  # [4, 4]
assert actions.shape == (B, 4)
assert torch.all(actions >= -1) and torch.all(actions <= 1)
print("✓ Model verification passed")
```

---

## Part 3: Training (~4 hours)

### 3.1 — Dataset & DataLoader

```python
"""
Training pipeline for Mini VLA.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np


class VLADataset(Dataset):
    """
    PyTorch dataset for language-conditioned manipulation demonstrations.
    Loads from HDF5 generated in Part 1.
    """

    def __init__(self, hdf5_path: str, transform=None):
        self.hdf5_path = hdf5_path
        self.transform = transform

        # Index all timesteps across episodes
        self.samples = []
        with h5py.File(hdf5_path, 'r') as f:
            for ep_name in sorted(f.keys()):
                if not ep_name.startswith("episode_"):
                    continue
                ep = f[ep_name]
                length = ep.attrs['length']
                command = ep.attrs['command']
                for t in range(length):
                    self.samples.append((ep_name, t, command))

        print(f"VLADataset: {len(self.samples)} samples from {hdf5_path}")

        # Build tokenizer (matches LanguageEncoder)
        self.vocab = {"<pad>": 0, "pick": 1, "place": 2, "on": 3,
                      "red": 4, "blue": 5, "green": 6, "target": 7,
                      "the": 8, "block": 9, "put": 10, "move": 11,
                      "to": 12, "grab": 13, "drop": 14}
        self.max_tokens = 5

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_name, t, command = self.samples[idx]

        with h5py.File(self.hdf5_path, 'r') as f:
            ep = f[ep_name]
            image = ep['images'][t]           # [64, 64, 3] uint8
            action = ep['actions'][t]          # [4] float
            joints = ep['joint_angles'][t]     # [3] float

        # Normalize image: [0, 255] → [0, 1] → [-1, 1]
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = torch.from_numpy(image).permute(2, 0, 1)  # [3, 64, 64]

        # Tokenize command
        words = command.lower().split()
        tokens = [self.vocab.get(w, 0) for w in words[:self.max_tokens]]
        tokens += [0] * (self.max_tokens - len(tokens))
        tokens = torch.tensor(tokens, dtype=torch.long)

        # Proprioception: joints + gripper (assume 0 gripper opening)
        proprio = np.concatenate([joints, [0.0]])
        proprio = torch.from_numpy(proprio.astype(np.float32))

        action = torch.from_numpy(action.astype(np.float32))

        return image, tokens, proprio, action
```

### 3.2 — Training Loop

```python
def train_mini_vla(
    dataset_path: str = "mini_vla_demos.hdf5",
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: str = "mini_vla_trained.pt",
):
    """
    Train the Mini VLA with behavioral cloning.

    Loss: MSE between predicted and expert actions.
    """
    print(f"Training on: {device}")

    # Data
    dataset = VLADataset(dataset_path)
    n_val = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Model
    model = MiniVLA(image_size=64, action_dim=4, embed_dim=128).to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Training
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for images, tokens, proprio, actions in train_loader:
            images = images.to(device)
            tokens = tokens.to(device)
            proprio = proprio.to(device)
            actions = actions.to(device)

            # Forward
            pred_actions = model(images, tokens, proprio)

            # MSE loss
            loss = F.mse_loss(pred_actions, actions)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for images, tokens, proprio, actions in val_loader:
                images = images.to(device)
                tokens = tokens.to(device)
                proprio = proprio.to(device)
                actions = actions.to(device)

                pred_actions = model(images, tokens, proprio)
                loss = F.mse_loss(pred_actions, actions)
                val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss / max(n_val_batches, 1)
        val_losses.append(avg_val_loss)

        scheduler.step()

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': best_val_loss,
            }, save_path)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | "
                  f"Best: {best_val_loss:.6f}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Model saved: {save_path}")

    return train_losses, val_losses
```

---

## Part 4: Evaluation (~3 hours)

### 4.1 — Closed-Loop Evaluation

```python
"""
Evaluate the trained Mini VLA in closed-loop MuJoCo simulation.
"""
import torch
import numpy as np


class VLAEvaluator:
    """
    Run trained VLA in the environment and measure performance.
    """

    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        self.model = MiniVLA(image_size=64, action_dim=4, embed_dim=128)

        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(device)

        # Language tokenizer
        self.vocab = {"<pad>": 0, "pick": 1, "place": 2, "on": 3,
                      "red": 4, "blue": 5, "green": 6, "target": 7}
        self.max_tokens = 5

    def tokenize(self, command: str) -> torch.Tensor:
        """Tokenize a single command."""
        words = command.lower().split()
        tokens = [self.vocab.get(w, 0) for w in words[:self.max_tokens]]
        tokens += [0] * (self.max_tokens - len(tokens))
        return torch.tensor([tokens], dtype=torch.long)

    @torch.no_grad()
    def get_action(self, obs: Observation) -> np.ndarray:
        """Run inference for a single observation."""
        # Preprocess image
        image = obs.image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)

        # Tokenize command
        tokens = self.tokenize(obs.language_command).to(self.device)

        # Proprioception
        proprio = np.concatenate([obs.joint_angles, [obs.gripper_state]])
        proprio = torch.from_numpy(proprio.astype(np.float32)).unsqueeze(0).to(self.device)

        # Forward pass
        action = self.model(image, tokens, proprio)
        return action.cpu().numpy()[0]

    def evaluate(
        self,
        n_episodes: int = 50,
        max_steps: int = 100,
        commands: list[str] = None,
    ) -> dict:
        """
        Run evaluation episodes and compute metrics.

        Returns:
            {
                'success_rate': float,
                'per_command_success': dict,
                'avg_episode_length': float,
                'avg_action_magnitude': float,
            }
        """
        if commands is None:
            commands = list(LANGUAGE_COMMANDS.keys())

        env = MiniVLAEnv(image_size=64)
        results = {cmd: [] for cmd in commands}

        for ep in range(n_episodes):
            command = commands[ep % len(commands)]
            obs = env.reset(command=command)

            episode_length = 0
            action_magnitudes = []

            for step in range(max_steps):
                action = self.get_action(obs)
                action_magnitudes.append(np.linalg.norm(action[:3]))

                obs, reward, done, info = env.step(action)
                episode_length += 1

                if done:
                    break

            success = reward > 0.5  # Sparse reward = 1 on success
            results[command].append({
                'success': success,
                'length': episode_length,
                'mean_action_mag': np.mean(action_magnitudes),
            })

        # Aggregate
        all_successes = [r['success'] for cmd_results in results.values() for r in cmd_results]
        all_lengths = [r['length'] for cmd_results in results.values() for r in cmd_results]

        per_command_success = {
            cmd: np.mean([r['success'] for r in cmd_results])
            for cmd, cmd_results in results.items()
        }

        return {
            'success_rate': np.mean(all_successes),
            'per_command_success': per_command_success,
            'avg_episode_length': np.mean(all_lengths),
            'n_episodes': n_episodes,
        }


def evaluate_with_safety(
    evaluator: VLAEvaluator,
    n_episodes: int = 20,
):
    """
    Evaluate VLA with the full safety pipeline from Exercise 16.
    Shows how OOD detection and safety monitors work in practice.
    """
    from exercises.exercise_16 import (  # noqa
        SafetySupervisor, MahalanobisOODDetector, FallbackManager
    )

    env = MiniVLAEnv(image_size=64)
    safety = SafetySupervisor()
    fallback = FallbackManager(home_position=np.zeros(3))

    safety_events = []

    for ep in range(n_episodes):
        command = np.random.choice(list(LANGUAGE_COMMANDS.keys()))
        obs = env.reset(command=command)

        for step in range(100):
            # Get VLA action
            action = evaluator.get_action(obs)

            # Safety check
            ee_pos_3d = np.array([obs.ee_position[0], obs.ee_position[1], 0.05])
            level, violation = safety.check_all(
                ee_position=ee_pos_3d,
                joint_angles=obs.joint_angles,
            )

            # Fallback decision
            safe_action, fb_level = fallback.get_action(
                vla_action=action,
                current_position=obs.joint_angles,
                safety_level=level,
                uncertainty=0.0,  # Would come from MC Dropout
            )

            if violation:
                safety_events.append({
                    'episode': ep, 'step': step,
                    'level': level.value, 'monitor': violation.monitor_name,
                })

            obs, reward, done, _ = env.step(safe_action)
            if done:
                break

    print(f"\nSafety evaluation over {n_episodes} episodes:")
    print(f"  Total safety events: {len(safety_events)}")
    print(f"  Events by level: {safety.summary()}")
```

### 4.2 — Visualization & Analysis

```python
"""Visualization tools for VLA evaluation."""
import matplotlib.pyplot as plt
import numpy as np


def plot_training_curves(train_losses: list, val_losses: list):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(train_losses, label='Train Loss', color='steelblue')
    ax.plot(val_losses, label='Val Loss', color='coral')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Mini VLA Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.show()


def plot_evaluation_results(results: dict):
    """Plot per-command success rates."""
    commands = list(results['per_command_success'].keys())
    rates = [results['per_command_success'][c] for c in commands]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bars = ax.bar(commands, rates, color='steelblue', edgecolor='navy', alpha=0.8)
    ax.axhline(y=results['success_rate'], color='red', linestyle='--',
               label=f"Overall: {results['success_rate']:.1%}")
    ax.set_ylabel('Success Rate')
    ax.set_title('Mini VLA — Per-Command Performance')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.0%}', ha='center', fontweight='bold')

    plt.tight_layout()
    plt.show()


def visualize_policy_rollout(evaluator: VLAEvaluator, command: str, save_frames: bool = False):
    """Visualize a single rollout with annotated actions."""
    env = MiniVLAEnv(image_size=64)
    obs = env.reset(command=command)

    frames = [obs.image.copy()]
    actions = []
    ee_positions = [obs.ee_position.copy()]

    for step in range(100):
        action = evaluator.get_action(obs)
        actions.append(action)

        obs, reward, done, _ = env.step(action)
        frames.append(obs.image.copy())
        ee_positions.append(obs.ee_position.copy())

        if done:
            break

    # Plot trajectory
    ee_positions = np.array(ee_positions)
    actions = np.array(actions)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # EE trajectory
    axes[0].plot(ee_positions[:, 0], ee_positions[:, 1], 'b-', linewidth=2)
    axes[0].plot(ee_positions[0, 0], ee_positions[0, 1], 'go', markersize=10, label='Start')
    axes[0].plot(ee_positions[-1, 0], ee_positions[-1, 1], 'r*', markersize=15, label='End')
    axes[0].set_title(f'EE Trajectory: "{command}"')
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].legend()
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)

    # Actions over time
    for i in range(3):
        axes[1].plot(actions[:, i], label=f'Joint {i}')
    axes[1].plot(actions[:, 3], label='Gripper', linestyle='--')
    axes[1].set_title('Actions Over Time')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Action Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Key frames
    n_frames = min(5, len(frames))
    frame_indices = np.linspace(0, len(frames)-1, n_frames, dtype=int)
    for i, fi in enumerate(frame_indices):
        ax_inset = axes[2].inset_axes([i/n_frames, 0, 1/n_frames, 1])
        ax_inset.imshow(frames[fi])
        ax_inset.set_title(f't={fi}', fontsize=8)
        ax_inset.axis('off')
    axes[2].axis('off')
    axes[2].set_title('Key Frames')

    plt.suptitle(f'Mini VLA Rollout: "{command}"', fontsize=14)
    plt.tight_layout()
    plt.show()
```

---

## Part 5: Ablations & Extensions (~3 hours)

### 5.1 — Ablation Studies

```python
# TODO: Run these ablation experiments

# 1. VISION ABLATION
#    - Remove vision encoder (replace with zeros) → train & eval
#    - Expected: success drops to near-random (can't see objects)

# 2. LANGUAGE ABLATION
#    - Remove language conditioning (replace with fixed embedding) → train & eval
#    - Expected: robot picks the SAME object regardless of command

# 3. PROPRIOCEPTION ABLATION
#    - Remove proprioception → train & eval
#    - Expected: larger actions, less precision

# 4. DATA SIZE ABLATION
#    - Train with 50, 100, 200, 500 episodes
#    - Plot: success rate vs dataset size
#    - Question: how much data does this tiny VLA need?

# 5. MODEL SIZE ABLATION
#    - embed_dim: 64, 128, 256
#    - n_layers: 1, 2, 4
#    - Plot: success rate vs parameters
#    - Question: where are the diminishing returns?

# YOUR CODE HERE — implement at least ablations 1-3
```

### 5.2 — Extension: Diffusion Action Head

```python
"""
BONUS: Replace the MLP action decoder with a simple diffusion head.
This is what Diffusion Policy (Chi et al., 2023) does.
"""
import torch
import torch.nn as nn


class SimpleDiffusionHead(nn.Module):
    """
    Minimal DDPM-style action decoder.

    Instead of directly predicting actions, learns to denoise actions:
    - Training: corrupt expert actions with noise, predict the noise
    - Inference: start from pure noise, iteratively denoise

    Why: handles multi-modal action distributions
    (when multiple valid actions exist for the same observation).
    """

    def __init__(self, action_dim: int = 4, cond_dim: int = 768,
                 hidden_dim: int = 256, n_diffusion_steps: int = 20):
        super().__init__()
        self.action_dim = action_dim
        self.n_steps = n_diffusion_steps

        # Noise prediction network
        self.noise_pred = nn.Sequential(
            nn.Linear(action_dim + cond_dim + 1, hidden_dim),  # +1 for timestep
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Noise schedule (linear beta schedule)
        betas = torch.linspace(1e-4, 0.02, n_diffusion_steps)
        alphas = 1.0 - betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))

    def forward(self, actions: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Training forward: add noise to actions, predict the noise.

        Args:
            actions: [B, action_dim] expert actions
            condition: [B, cond_dim] conditioning (fused tokens flattened)

        Returns:
            loss: MSE between predicted and actual noise
        """
        B = actions.shape[0]

        # Random timestep
        t = torch.randint(0, self.n_steps, (B,), device=actions.device)
        t_normalized = t.float() / self.n_steps  # [0, 1]

        # Add noise
        noise = torch.randn_like(actions)
        alpha_t = self.alphas_cumprod[t].unsqueeze(1)
        noisy_actions = torch.sqrt(alpha_t) * actions + torch.sqrt(1 - alpha_t) * noise

        # Predict noise
        x = torch.cat([noisy_actions, condition, t_normalized.unsqueeze(1)], dim=1)
        noise_pred = self.noise_pred(x)

        # Loss
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def sample(self, condition: torch.Tensor) -> torch.Tensor:
        """
        Inference: iteratively denoise from random noise.

        Args:
            condition: [B, cond_dim]

        Returns:
            actions: [B, action_dim] denoised actions
        """
        B = condition.shape[0]
        x = torch.randn(B, self.action_dim, device=condition.device)

        for t in reversed(range(self.n_steps)):
            t_tensor = torch.full((B, 1), t / self.n_steps, device=condition.device)
            inp = torch.cat([x, condition, t_tensor], dim=1)
            noise_pred = self.noise_pred(inp)

            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)

            # DDPM update step
            x = (1 / torch.sqrt(alpha_t / alpha_prev)) * (
                x - (1 - alpha_t / alpha_prev) / torch.sqrt(1 - alpha_t) * noise_pred
            )

            # Add noise (except last step)
            if t > 0:
                x += 0.01 * torch.randn_like(x)

        return torch.tanh(x)  # Clip to [-1, 1]


# TODO:
# 1. Replace ActionDecoder with SimpleDiffusionHead in MiniVLA
# 2. Modify training loop: loss = diffusion_head(expert_actions, fused_features)
# 3. Modify inference: action = diffusion_head.sample(fused_features)
# 4. Train & compare: MLP head vs Diffusion head
# 5. Question: when does diffusion help? (Hint: multi-modal actions)
```

### 5.3 — Final Reflection

```python
# Answer these questions in your notebook:
#
# 1. ARCHITECTURE: What's the minimum model size that achieves >50% success?
#    How does this compare to production VLAs (RT-2: 55B, Octo: 93M, OpenVLA: 7B)?
#
# 2. DATA: How many demonstrations did you need for reasonable performance?
#    Production VLAs use 100K-1M+ episodes. What's the scaling relationship?
#
# 3. GENERALIZATION: Does your VLA generalize to:
#    a) Slightly different block positions? (Should yes)
#    b) Different colored blocks? (Probably not)
#    c) Different table textures? (Probably not)
#    What would you need for (b) and (c)?
#
# 4. SAFETY: How many times did the safety system intervene?
#    Were they true positives or false positives?
#
# 5. LATENCY: What's the inference time per step?
#    Is it fast enough for 10Hz control? For 100Hz?
#    What would you change for real-time deployment?
#
# 6. CONNECTION: Map each component of your Mini VLA to the equivalent
#    in a production system:
#    - Your TinyViT ↔ ??? in RT-2
#    - Your LanguageEncoder ↔ ??? in OpenVLA
#    - Your ActionDecoder ↔ ??? in Octo
#    - Your ScriptedExpert ↔ ??? in Open X-Embodiment
```

---

## Summary

| Component | What You Built | Production Equivalent |
|-----------|---------------|----------------------|
| Vision Encoder | 2-layer ViT, 64x64, 4 tokens | SigLIP/DINOv2, 224x224, 256 tokens |
| Language Encoder | Learned embeddings, 15 words | CLIP/T5, full vocabulary |
| Fusion | 1 transformer layer | 12-32 layers, cross-attention |
| Action Decoder | MLP + tanh | Diffusion head, action tokens |
| Data | 200 scripted episodes | 1M+ teleoperated episodes |
| Training | 50 epochs, 1 GPU, <1 hour | 100K+ steps, 64+ TPUs, days |
| Evaluation | Single task, 4 commands | 100+ tasks, 600+ environments |

**The architecture is the same. The scale is different.** Understanding the small version deeply prepares you to work with the large version.

---

## Checklist

- [ ] Built MuJoCo environment with colored blocks and overhead camera
- [ ] Implemented scripted expert for language-conditioned pick-and-place
- [ ] Generated 200+ demonstration episodes in HDF5 format
- [ ] Implemented TinyViT vision encoder with learned pooling
- [ ] Implemented language encoder with tokenization
- [ ] Implemented proprioception encoder
- [ ] Built full MiniVLA model (~500K parameters)
- [ ] Verified forward pass produces correct shapes
- [ ] Trained model to convergence (<1 hour)
- [ ] Plotted training/validation loss curves
- [ ] Evaluated in closed-loop (per-command success rates)
- [ ] Visualized rollouts with EE trajectories and key frames
- [ ] Ran at least 3 ablation experiments
- [ ] (BONUS) Implemented diffusion action head
- [ ] Connected findings to production VLA systems
- [ ] Evaluated with safety pipeline from Exercise 16
