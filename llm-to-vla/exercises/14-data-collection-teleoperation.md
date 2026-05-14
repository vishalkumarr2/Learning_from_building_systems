# Exercise 14 — Robot Data Collection & Teleoperation
> Phase VI–VII · Days 90-92 · ~8 hours · **Robotics-specific**

[← Exercise 13: Sim-to-Real](13-sim-to-real-transfer.md) | [Exercise 15: Kinematics & Trajectory →](15-kinematics-trajectory.md)

---

## Objectives

By completing this exercise you will:
- Build a teleoperation interface for controlling a simulated robot
- Record demonstration episodes in standard formats (HDF5, RLDS-like)
- Implement data quality filtering and validation pipelines
- Generate demonstrations from scripted policies for sim environments
- Compute dataset statistics and visualize episode distributions
- Understand the data requirements for training VLA/BC/ACT policies

## Prerequisites
- Exercise 12 (MuJoCo manipulation — environment basics)
- Study Note 13 (Imitation Learning — BC, DAgger, ACT data requirements)
- Study Note 14 (Robot Data & Evaluation)

## Setup

```bash
pip install mujoco gymnasium numpy torch h5py matplotlib
pip install opencv-python pynput  # for keyboard teleop
```

> **Note**: For real hardware teleoperation you'd use spacemouse (`pyspacemouse`) or
> VR controllers. This exercise uses keyboard + scripted policies to keep setup simple.

---

## Part 1: Keyboard Teleoperation Interface (~1.5 hours)

### 1.1 — Build a Teleop Controller

The most common way to collect demonstrations is teleoperation — a human controls
the robot while sensor data and actions are recorded.

```python
"""
Keyboard teleoperation for a MuJoCo robot arm.
Maps WASD + QE keys to end-effector delta commands.
"""
import numpy as np
import gymnasium as gym
import mujoco
import mujoco.viewer
import threading
import time
from pynput import keyboard


class KeyboardTeleop:
    """Maps keyboard input to robot end-effector velocity commands."""

    def __init__(self, translation_speed: float = 0.02, rotation_speed: float = 0.05):
        self.translation_speed = translation_speed
        self.rotation_speed = rotation_speed

        # Current velocity command [dx, dy, dz, drx, dry, drz, gripper]
        self.command = np.zeros(7)
        self._keys_pressed = set()
        self._running = True

        # Key mappings: key → (index, direction)
        self.key_map = {
            'w': (0, +1),   # +X (forward)
            's': (0, -1),   # -X (backward)
            'a': (1, +1),   # +Y (left)
            'd': (1, -1),   # -Y (right)
            'q': (2, +1),   # +Z (up)
            'e': (2, -1),   # -Z (down)
            'j': (3, +1),   # +Rx
            'l': (3, -1),   # -Rx
            'i': (4, +1),   # +Ry
            'k': (4, -1),   # -Ry
            'u': (5, +1),   # +Rz
            'o': (5, -1),   # -Rz
        }

        # Start keyboard listener in background
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self._listener.start()

    def _on_press(self, key):
        try:
            self._keys_pressed.add(key.char)
        except AttributeError:
            pass

    def _on_release(self, key):
        try:
            self._keys_pressed.discard(key.char)
        except AttributeError:
            pass
        if key == keyboard.Key.esc:
            self._running = False

    def get_action(self) -> np.ndarray:
        """Compute action from currently pressed keys."""
        self.command[:] = 0.0

        for k in list(self._keys_pressed):
            if k in self.key_map:
                idx, direction = self.key_map[k]
                speed = self.translation_speed if idx < 3 else self.rotation_speed
                self.command[idx] = direction * speed

        # Gripper: space = close, shift = open
        if ' ' in self._keys_pressed:
            self.command[6] = 1.0   # close
        else:
            self.command[6] = -1.0  # open

        return self.command.copy()

    @property
    def running(self) -> bool:
        return self._running

    def stop(self):
        self._running = False
        self._listener.stop()


# ---------- Quick test ----------
if __name__ == "__main__":
    teleop = KeyboardTeleop()
    print("Controls: WASD=XY, QE=Z, IJKL=rotation, SPACE=close gripper, ESC=quit")

    while teleop.running:
        action = teleop.get_action()
        if np.any(action != 0):
            print(f"Action: {action}")
        time.sleep(0.05)  # 20Hz

    teleop.stop()
```

### 1.2 — Integrate Teleop with a Simulated Environment

```python
"""
Teleoperation loop: human controls robot, observations are rendered.
"""
import mujoco
import mujoco.viewer
import numpy as np


class TeleopEnvironment:
    """Wraps a MuJoCo model for keyboard teleoperation."""

    SIMPLE_ARM_XML = """
    <mujoco model="teleop_arm">
      <option gravity="0 0 -9.81" timestep="0.002"/>
      <worldbody>
        <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
        <geom type="plane" size="1 1 0.1" rgba="0.8 0.8 0.8 1"/>
        <body name="base" pos="0 0 0.1">
          <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.3 1"/>
          <body name="link1" pos="0 0 0.1">
            <joint name="j1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.3" size="0.03" rgba="0.2 0.6 0.8 1"/>
            <body name="link2" pos="0 0 0.3">
              <joint name="j2" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.25" size="0.025" rgba="0.8 0.4 0.2 1"/>
              <body name="link3" pos="0 0 0.25">
                <joint name="j3" type="hinge" axis="0 1 0" range="-2.5 2.5"/>
                <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02" rgba="0.2 0.8 0.4 1"/>
                <body name="ee" pos="0 0 0.2">
                  <site name="ee_site" size="0.02" rgba="1 0 0 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
        <!-- Target object -->
        <body name="target" pos="0.3 0.2 0.15">
          <geom type="sphere" size="0.03" rgba="1 0.2 0.2 1" contype="0" conaffinity="0"/>
        </body>
      </worldbody>
      <actuator>
        <velocity name="a1" joint="j1" kv="50"/>
        <velocity name="a2" joint="j2" kv="50"/>
        <velocity name="a3" joint="j3" kv="50"/>
      </actuator>
    </mujoco>
    """

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(self.SIMPLE_ARM_XML)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def get_ee_position(self) -> np.ndarray:
        """Get end-effector position from site."""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        return self.data.site_xpos[site_id].copy()

    def get_jacobian(self) -> np.ndarray:
        """Compute end-effector Jacobian (position only)."""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        nv = self.model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
        return jacp

    def teleop_step(self, ee_velocity: np.ndarray):
        """
        Convert end-effector velocity command to joint velocities via Jacobian.
        Uses damped least-squares inverse.
        """
        J = self.get_jacobian()
        damping = 0.01
        # J^T (J J^T + λ²I)^{-1} * dx
        JJT = J @ J.T + damping**2 * np.eye(3)
        joint_vel = J.T @ np.linalg.solve(JJT, ee_velocity[:3])

        # Apply as velocity actuator commands
        self.data.ctrl[:] = joint_vel

        # Step simulation
        for _ in range(10):  # 10 substeps at 0.002s = 20ms per teleop step
            mujoco.mj_step(self.model, self.data)

    def get_observation(self) -> dict:
        """Collect full observation for recording."""
        return {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ee_pos": self.get_ee_position(),
            "timestamp": self.data.time,
        }
```

### 1.3 — Exercise: Run Teleoperation

```python
# Run in a script (not notebook — needs keyboard access)
# 1. Instantiate TeleopEnvironment
# 2. Instantiate KeyboardTeleop
# 3. Loop: get_action() → teleop_step() → render
# 4. Print EE position every 0.5s

# YOUR CODE HERE:
# env = TeleopEnvironment()
# teleop = KeyboardTeleop()
# with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
#     while teleop.running and viewer.is_running():
#         action = teleop.get_action()
#         env.teleop_step(action)
#         viewer.sync()
#         time.sleep(0.05)
```

**Reflection**: What makes teleoperation hard? Consider: latency, workspace limits,
lack of force feedback, cognitive load of mapping 2D input to 3D motion.

---

## Part 2: Episode Recording in HDF5 Format (~2 hours)

### 2.1 — Design the Episode Data Structure

VLA training requires episodes structured as sequences of (observation, action) pairs.
The standard format used by robomimic, LeRobot, and RLDS:

```python
"""
Episode recorder — saves demonstrations in HDF5 format.
Compatible with robomimic / LeRobot data loading conventions.
"""
import h5py
import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Timestep:
    """Single timestep in a demonstration."""
    observation: dict        # sensor readings
    action: np.ndarray       # commanded action
    reward: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)


class EpisodeRecorder:
    """Records and saves demonstration episodes to HDF5."""

    def __init__(self, save_dir: str, task_name: str = "manipulation"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.task_name = task_name

        self._current_episode: list[Timestep] = []
        self._episode_count = self._count_existing()
        self._recording = False

    def _count_existing(self) -> int:
        """Count existing episodes in save directory."""
        return len(list(self.save_dir.glob("episode_*.hdf5")))

    def start_episode(self):
        """Begin recording a new episode."""
        self._current_episode = []
        self._recording = True
        print(f"🔴 Recording episode {self._episode_count}...")

    def record_step(self, observation: dict, action: np.ndarray,
                    reward: float = 0.0, done: bool = False, info: dict = None):
        """Record a single timestep."""
        if not self._recording:
            return

        self._current_episode.append(Timestep(
            observation=observation,
            action=action.copy(),
            reward=reward,
            done=done,
            info=info or {}
        ))

    def end_episode(self, success: bool = False, discard: bool = False) -> Optional[str]:
        """
        End current episode. Save to disk unless discarded.

        Returns:
            Path to saved file, or None if discarded.
        """
        self._recording = False

        if discard or len(self._current_episode) < 5:
            print(f"⏭️  Episode discarded ({len(self._current_episode)} steps)")
            self._current_episode = []
            return None

        # Save to HDF5
        filepath = self.save_dir / f"episode_{self._episode_count:05d}.hdf5"
        self._save_hdf5(filepath, success)
        print(f"✅ Saved episode {self._episode_count}: "
              f"{len(self._current_episode)} steps, success={success}")

        self._episode_count += 1
        self._current_episode = []
        return str(filepath)

    def _save_hdf5(self, filepath: Path, success: bool):
        """Save episode data in robomimic-compatible HDF5 format."""
        with h5py.File(filepath, 'w') as f:
            # Metadata
            f.attrs["task"] = self.task_name
            f.attrs["num_steps"] = len(self._current_episode)
            f.attrs["success"] = success
            f.attrs["timestamp"] = time.time()

            # Create data group
            data = f.create_group("data")

            # Stack observations by key
            obs_keys = self._current_episode[0].observation.keys()
            obs_group = data.create_group("observations")
            for key in obs_keys:
                values = np.array([t.observation[key] for t in self._current_episode])
                obs_group.create_dataset(key, data=values, compression="gzip")

            # Actions
            actions = np.array([t.action for t in self._current_episode])
            data.create_dataset("actions", data=actions, compression="gzip")

            # Rewards
            rewards = np.array([t.reward for t in self._current_episode])
            data.create_dataset("rewards", data=rewards)

            # Dones
            dones = np.array([t.done for t in self._current_episode])
            data.create_dataset("dones", data=dones)


# ---------- Usage Example ----------
def demo_recording():
    """Example: record a scripted episode."""
    recorder = EpisodeRecorder(save_dir="./demo_data", task_name="reach_target")
    recorder.start_episode()

    # Simulate 50 steps of a reaching motion
    for t in range(50):
        obs = {
            "qpos": np.random.randn(3) * 0.1,
            "qvel": np.random.randn(3) * 0.01,
            "ee_pos": np.array([0.1 * t / 50, 0.05, 0.3]),
        }
        action = np.array([0.01, 0.005, -0.002])  # constant velocity
        recorder.record_step(obs, action, reward=0.0, done=(t == 49))

    filepath = recorder.end_episode(success=True)
    print(f"Saved to: {filepath}")


if __name__ == "__main__":
    demo_recording()
```

### 2.2 — Read and Validate Recorded Data

```python
"""
Dataset reader — load and inspect recorded demonstrations.
"""
import h5py
import numpy as np
from pathlib import Path


class DemoDataset:
    """Load and inspect a directory of HDF5 demonstration episodes."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.episode_files = sorted(self.data_dir.glob("episode_*.hdf5"))
        print(f"Found {len(self.episode_files)} episodes in {data_dir}")

    def load_episode(self, idx: int) -> dict:
        """Load a single episode by index."""
        with h5py.File(self.episode_files[idx], 'r') as f:
            episode = {
                "task": f.attrs["task"],
                "num_steps": f.attrs["num_steps"],
                "success": f.attrs["success"],
                "observations": {},
                "actions": f["data/actions"][:],
                "rewards": f["data/rewards"][:],
            }
            for key in f["data/observations"].keys():
                episode["observations"][key] = f[f"data/observations/{key}"][:]
        return episode

    def get_statistics(self) -> dict:
        """Compute dataset-level statistics."""
        all_actions = []
        all_lengths = []
        success_count = 0

        for ep_file in self.episode_files:
            with h5py.File(ep_file, 'r') as f:
                all_actions.append(f["data/actions"][:])
                all_lengths.append(f.attrs["num_steps"])
                if f.attrs["success"]:
                    success_count += 1

        actions = np.concatenate(all_actions, axis=0)

        return {
            "num_episodes": len(self.episode_files),
            "success_rate": success_count / max(len(self.episode_files), 1),
            "episode_lengths": {
                "mean": np.mean(all_lengths),
                "std": np.std(all_lengths),
                "min": np.min(all_lengths),
                "max": np.max(all_lengths),
            },
            "action_statistics": {
                "mean": actions.mean(axis=0),
                "std": actions.std(axis=0),
                "min": actions.min(axis=0),
                "max": actions.max(axis=0),
            },
        }

    def validate(self) -> list[str]:
        """
        Validate dataset integrity.
        Returns list of issues found.
        """
        issues = []

        for i, ep_file in enumerate(self.episode_files):
            try:
                with h5py.File(ep_file, 'r') as f:
                    n = f.attrs["num_steps"]
                    actions = f["data/actions"]
                    if actions.shape[0] != n:
                        issues.append(f"Episode {i}: action length {actions.shape[0]} != {n}")

                    # Check for NaN
                    if np.any(np.isnan(actions[:])):
                        issues.append(f"Episode {i}: NaN in actions")

                    # Check for constant actions (stuck teleop)
                    if actions.shape[0] > 10:
                        action_std = np.std(actions[:], axis=0)
                        if np.all(action_std < 1e-6):
                            issues.append(f"Episode {i}: constant actions (stuck?)")

            except Exception as e:
                issues.append(f"Episode {i}: failed to read — {e}")

        return issues
```

### 2.3 — Exercise: Record and Validate

```python
# TODO: Complete these tasks
# 1. Record 5 episodes using the scripted demo (modify demo_recording)
# 2. Load the dataset and compute statistics
# 3. Run validation — are there any issues?
# 4. Plot: action distributions (histogram per dimension)
# 5. Plot: episode length distribution

import matplotlib.pyplot as plt

# YOUR CODE HERE
```

---

## Part 3: Scripted Policy Data Generation (~2 hours)

### 3.1 — Why Scripted Demos?

For sim-to-real pipelines, you often need thousands of demonstrations. Human
teleoperation is slow (~50 demos/hour). Scripted policies can generate unlimited data:

| Method | Speed | Quality | Diversity |
|--------|-------|---------|-----------|
| Human teleop | ~1 demo/min | High (natural) | Medium |
| Scripted policy | ~100 demos/min | Medium (robotic) | Low without randomization |
| RL policy demos | ~50 demos/min | High (optimal) | Medium |
| Mixed (scripted + noise) | ~100 demos/min | Medium-High | High |

### 3.2 — Build a Scripted Reach-and-Grasp Policy

```python
"""
Scripted policies for generating demonstrations at scale.
Uses simple motion primitives: move_to, grasp, lift, place.
"""
import numpy as np
from enum import Enum, auto


class Phase(Enum):
    APPROACH = auto()
    DESCEND = auto()
    GRASP = auto()
    LIFT = auto()
    MOVE_TO_PLACE = auto()
    PLACE = auto()
    DONE = auto()


class ScriptedPickPlace:
    """
    Scripted pick-and-place policy using waypoint following.

    Generates delta end-effector actions like a teleoperated demonstration.
    """

    def __init__(
        self,
        approach_height: float = 0.15,
        grasp_height: float = 0.02,
        lift_height: float = 0.20,
        position_threshold: float = 0.01,
        max_speed: float = 0.03,
    ):
        self.approach_height = approach_height
        self.grasp_height = grasp_height
        self.lift_height = lift_height
        self.threshold = position_threshold
        self.max_speed = max_speed
        self.phase = Phase.APPROACH
        self.grasp_counter = 0

    def reset(self, pick_pos: np.ndarray, place_pos: np.ndarray):
        """Reset for a new episode."""
        self.pick_pos = pick_pos.copy()
        self.place_pos = place_pos.copy()
        self.phase = Phase.APPROACH
        self.grasp_counter = 0

        # Compute waypoints
        self._waypoints = {
            Phase.APPROACH: np.array([pick_pos[0], pick_pos[1], self.approach_height]),
            Phase.DESCEND: np.array([pick_pos[0], pick_pos[1], self.grasp_height]),
            Phase.LIFT: np.array([pick_pos[0], pick_pos[1], self.lift_height]),
            Phase.MOVE_TO_PLACE: np.array([place_pos[0], place_pos[1], self.lift_height]),
            Phase.PLACE: np.array([place_pos[0], place_pos[1], self.grasp_height + 0.02]),
        }

    def get_action(self, ee_pos: np.ndarray) -> tuple[np.ndarray, bool]:
        """
        Compute next action given current end-effector position.

        Returns:
            (action[7], done): action is [dx, dy, dz, 0, 0, 0, gripper]
        """
        if self.phase == Phase.DONE:
            return np.zeros(7), True

        if self.phase == Phase.GRASP:
            # Wait a few steps with gripper closed
            self.grasp_counter += 1
            action = np.zeros(7)
            action[6] = 1.0  # close gripper
            if self.grasp_counter >= 5:
                self.phase = Phase.LIFT
            return action, False

        # Move toward current waypoint
        target = self._waypoints[self.phase]
        delta = target - ee_pos
        distance = np.linalg.norm(delta)

        if distance < self.threshold:
            # Advance to next phase
            self.phase = self._next_phase(self.phase)
            return self.get_action(ee_pos)  # recurse for next phase

        # Clip to max speed
        direction = delta / (distance + 1e-8)
        step = direction * min(distance, self.max_speed)

        action = np.zeros(7)
        action[:3] = step
        # Gripper: open during approach/descend, closed after grasp
        action[6] = 1.0 if self.phase.value > Phase.GRASP.value else -1.0

        return action, False

    def _next_phase(self, current: Phase) -> Phase:
        transitions = {
            Phase.APPROACH: Phase.DESCEND,
            Phase.DESCEND: Phase.GRASP,
            Phase.GRASP: Phase.LIFT,
            Phase.LIFT: Phase.MOVE_TO_PLACE,
            Phase.MOVE_TO_PLACE: Phase.PLACE,
            Phase.PLACE: Phase.DONE,
        }
        return transitions.get(current, Phase.DONE)


# ---------- Generate demonstrations at scale ----------
def generate_demonstrations(
    num_episodes: int = 100,
    workspace_bounds: tuple = ((-0.3, 0.3), (-0.3, 0.3)),
    noise_std: float = 0.002,
) -> list[dict]:
    """
    Generate scripted demonstrations with randomized pick/place positions.

    Args:
        num_episodes: Number of demonstrations to generate
        workspace_bounds: ((x_min, x_max), (y_min, y_max))
        noise_std: Gaussian noise added to actions for diversity

    Returns:
        List of episode dicts with observations and actions
    """
    policy = ScriptedPickPlace()
    episodes = []

    for ep in range(num_episodes):
        # Randomize pick and place positions
        pick_pos = np.array([
            np.random.uniform(*workspace_bounds[0]),
            np.random.uniform(*workspace_bounds[1]),
            0.02,  # table height
        ])
        place_pos = np.array([
            np.random.uniform(*workspace_bounds[0]),
            np.random.uniform(*workspace_bounds[1]),
            0.02,
        ])

        # Ensure pick != place
        while np.linalg.norm(pick_pos[:2] - place_pos[:2]) < 0.1:
            place_pos[:2] = np.random.uniform(
                [workspace_bounds[0][0], workspace_bounds[1][0]],
                [workspace_bounds[0][1], workspace_bounds[1][1]]
            )

        policy.reset(pick_pos, place_pos)

        # Simulate episode
        ee_pos = np.array([0.0, 0.0, 0.3])  # start position
        observations = []
        actions = []

        for step in range(200):  # max 200 steps
            action, done = policy.get_action(ee_pos)

            # Add noise for diversity (mimics human imprecision)
            if noise_std > 0:
                action[:3] += np.random.randn(3) * noise_std

            observations.append({
                "ee_pos": ee_pos.copy(),
                "pick_pos": pick_pos.copy(),
                "place_pos": place_pos.copy(),
            })
            actions.append(action.copy())

            # Simple kinematics: position += action
            ee_pos += action[:3]

            if done:
                break

        episodes.append({
            "observations": observations,
            "actions": np.array(actions),
            "num_steps": len(actions),
            "success": done,  # reached DONE phase
            "pick_pos": pick_pos,
            "place_pos": place_pos,
        })

    success_rate = sum(ep["success"] for ep in episodes) / num_episodes
    print(f"Generated {num_episodes} episodes, success rate: {success_rate:.1%}")
    return episodes
```

### 3.3 — Exercise: Generate and Analyze a Dataset

```python
# TODO:
# 1. Generate 200 demonstrations with noise_std=0.003
# 2. Save them using EpisodeRecorder
# 3. Compute dataset statistics
# 4. Visualize:
#    a) Top-down view of pick/place positions (scatter plot)
#    b) Action magnitude over time (mean across episodes)
#    c) Gripper state transitions
# 5. How many demos have > 150 steps? Filter those out.
# 6. What's the action dimension distribution after filtering?

# YOUR CODE HERE
```

---

## Part 4: Data Quality Pipeline (~1.5 hours)

### 4.1 — Filtering Bad Demonstrations

Not all demonstrations are useful for training. Common issues:

```python
"""
Data quality filters for demonstration datasets.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class QualityMetrics:
    """Quality metrics for a single episode."""
    episode_idx: int
    length: int
    success: bool
    action_smoothness: float     # lower = smoother
    progress_score: float        # 0-1, how much task progress
    idle_fraction: float         # fraction of near-zero actions
    has_nan: bool
    has_collision: bool
    is_valid: bool               # passes all checks


class DataQualityFilter:
    """Filter demonstrations based on quality criteria."""

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 300,
        max_idle_fraction: float = 0.5,
        min_progress: float = 0.3,
        max_jerk: float = 0.1,
        require_success: bool = False,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.max_idle_fraction = max_idle_fraction
        self.min_progress = min_progress
        self.max_jerk = max_jerk
        self.require_success = require_success

    def evaluate_episode(self, episode: dict, idx: int = 0) -> QualityMetrics:
        """Compute quality metrics for a single episode."""
        actions = episode["actions"]
        n = len(actions)

        # Basic checks
        has_nan = np.any(np.isnan(actions))

        # Smoothness: mean jerk (third derivative approximation)
        if n > 3:
            velocities = np.diff(actions[:, :3], axis=0)
            accelerations = np.diff(velocities, axis=0)
            jerk = np.diff(accelerations, axis=0)
            smoothness = np.mean(np.abs(jerk))
        else:
            smoothness = 0.0

        # Idle fraction: steps with near-zero action
        action_magnitudes = np.linalg.norm(actions[:, :3], axis=1)
        idle_steps = np.sum(action_magnitudes < 1e-4)
        idle_fraction = idle_steps / max(n, 1)

        # Progress: ratio of distance covered to straight-line distance
        if "observations" in episode and len(episode["observations"]) > 1:
            start_pos = episode["observations"][0]["ee_pos"]
            end_pos = episode["observations"][-1]["ee_pos"]
            total_distance = np.sum(action_magnitudes)
            straight_distance = np.linalg.norm(end_pos - start_pos)
            progress = straight_distance / max(total_distance, 1e-6)
        else:
            progress = 0.5  # unknown

        # Validity check
        is_valid = (
            not has_nan
            and self.min_length <= n <= self.max_length
            and idle_fraction <= self.max_idle_fraction
            and progress >= self.min_progress
            and smoothness <= self.max_jerk
            and (not self.require_success or episode.get("success", False))
        )

        return QualityMetrics(
            episode_idx=idx,
            length=n,
            success=episode.get("success", False),
            action_smoothness=smoothness,
            progress_score=progress,
            idle_fraction=idle_fraction,
            has_nan=has_nan,
            has_collision=False,  # would need sim to check
            is_valid=is_valid,
        )

    def filter_dataset(self, episodes: list[dict]) -> tuple[list[dict], list[QualityMetrics]]:
        """
        Filter dataset, returning valid episodes and all metrics.
        """
        metrics = [self.evaluate_episode(ep, i) for i, ep in enumerate(episodes)]
        valid_episodes = [ep for ep, m in zip(episodes, metrics) if m.is_valid]

        n_total = len(episodes)
        n_valid = len(valid_episodes)
        print(f"Quality filter: {n_valid}/{n_total} episodes passed ({n_valid/n_total:.1%})")

        # Print rejection reasons
        rejected = [m for m in metrics if not m.is_valid]
        if rejected:
            reasons = {
                "too_short": sum(1 for m in rejected if m.length < self.min_length),
                "too_long": sum(1 for m in rejected if m.length > self.max_length),
                "too_idle": sum(1 for m in rejected if m.idle_fraction > self.max_idle_fraction),
                "low_progress": sum(1 for m in rejected if m.progress_score < self.min_progress),
                "too_jerky": sum(1 for m in rejected if m.action_smoothness > self.max_jerk),
                "has_nan": sum(1 for m in rejected if m.has_nan),
            }
            for reason, count in reasons.items():
                if count > 0:
                    print(f"  Rejected ({reason}): {count}")

        return valid_episodes, metrics
```

### 4.2 — Exercise: Build Your Quality Pipeline

```python
# TODO:
# 1. Generate 500 demonstrations with high noise (noise_std=0.01)
# 2. Apply DataQualityFilter with default parameters
# 3. How many pass? Adjust thresholds to keep ~80% of data
# 4. Plot: quality metric distributions (histograms)
#    - action_smoothness, idle_fraction, progress_score
# 5. Compare training a simple BC policy on:
#    a) All 500 raw demos
#    b) Only filtered demos
#    Which produces better policy performance?

# YOUR CODE HERE
```

---

## Part 5: Dataset Visualization & Analysis (~1 hour)

### 5.1 — Visualize Demonstration Trajectories

```python
"""
Visualization tools for demonstration datasets.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D


def plot_trajectories_3d(episodes: list[dict], max_episodes: int = 20):
    """Plot 3D end-effector trajectories."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, ep in enumerate(episodes[:max_episodes]):
        positions = np.array([obs["ee_pos"] for obs in ep["observations"]])
        color = 'green' if ep.get("success") else 'red'
        alpha = 0.6
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                color=color, alpha=alpha, linewidth=0.8)
        # Mark start and end
        ax.scatter(*positions[0], color='blue', s=20, zorder=5)
        ax.scatter(*positions[-1], color=color, s=40, marker='*', zorder=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'EE Trajectories ({max_episodes} episodes)')
    plt.tight_layout()
    plt.show()


def plot_action_statistics(episodes: list[dict]):
    """Plot per-dimension action statistics over time."""
    # Align episodes (pad shorter ones)
    max_len = max(len(ep["actions"]) for ep in episodes)
    action_dim = episodes[0]["actions"].shape[1]

    # Compute mean and std at each timestep
    all_actions = np.full((len(episodes), max_len, action_dim), np.nan)
    for i, ep in enumerate(episodes):
        n = len(ep["actions"])
        all_actions[i, :n, :] = ep["actions"]

    mean_actions = np.nanmean(all_actions, axis=0)
    std_actions = np.nanstd(all_actions, axis=0)

    fig, axes = plt.subplots(action_dim, 1, figsize=(12, 2 * action_dim), sharex=True)
    dim_names = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'gripper']

    for d in range(min(action_dim, len(dim_names))):
        ax = axes[d] if action_dim > 1 else axes
        t = np.arange(max_len)
        ax.plot(t, mean_actions[:, d], 'b-', linewidth=1)
        ax.fill_between(t,
                        mean_actions[:, d] - std_actions[:, d],
                        mean_actions[:, d] + std_actions[:, d],
                        alpha=0.3)
        ax.set_ylabel(dim_names[d])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Timestep')
    fig.suptitle('Action Statistics Over Time (mean ± std)')
    plt.tight_layout()
    plt.show()


def plot_dataset_summary(metrics: list) -> None:
    """Plot dataset quality summary from QualityMetrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    lengths = [m.length for m in metrics]
    smoothness = [m.action_smoothness for m in metrics]
    idle = [m.idle_fraction for m in metrics]
    progress = [m.progress_score for m in metrics]

    axes[0, 0].hist(lengths, bins=30, color='steelblue', edgecolor='white')
    axes[0, 0].set_title('Episode Length Distribution')
    axes[0, 0].axvline(np.median(lengths), color='red', linestyle='--', label='median')
    axes[0, 0].legend()

    axes[0, 1].hist(smoothness, bins=30, color='coral', edgecolor='white')
    axes[0, 1].set_title('Action Smoothness (lower = better)')

    axes[1, 0].hist(idle, bins=30, color='mediumpurple', edgecolor='white')
    axes[1, 0].set_title('Idle Fraction')

    axes[1, 1].hist(progress, bins=30, color='seagreen', edgecolor='white')
    axes[1, 1].set_title('Progress Score')

    # Color by valid/invalid
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
```

### 5.2 — Exercise: Full Analysis Pipeline

```python
# TODO: Put it all together
# 1. Generate 300 demonstrations (mix of noise levels: 0, 0.003, 0.01)
# 2. Record all to HDF5 using EpisodeRecorder
# 3. Load back using DemoDataset, compute statistics
# 4. Run quality filter
# 5. Visualize:
#    - 3D trajectories (color by success)
#    - Action statistics over time
#    - Dataset quality summary
# 6. Final report: how many usable demos? What's the effective data rate?
#    (usable_demos / generation_time)

# YOUR CODE HERE
```

---

## Summary & Key Takeaways

| Concept | Why It Matters for VLAs |
|---------|------------------------|
| Teleoperation interface | All real-world VLA training data comes from teleop |
| HDF5 episode format | Standard format for robomimic, LeRobot, RLDS |
| Scripted policies | Scale up sim demonstrations 100x |
| Data quality filtering | Bad demos corrupt policy training |
| Action noise injection | Increases diversity, prevents overfitting to scripted paths |
| Dataset statistics | Normalization parameters for training |

### Connection to VLA Training

The data you collect and curate here feeds directly into:
- **Exercise 10** (Imitation Learning): BC and ACT require clean demonstrations
- **Exercise 12** (MuJoCo): Training policies from demos in simulation
- **Project 06** (Diffusion Policy): Diffusion policies trained on these episodes
- **Project 07** (VLA Capstone): End-to-end VLA pipeline needs curated datasets

### Further Reading
- [robomimic data format](https://robomimic.github.io/docs/datasets/overview.html)
- [RLDS (RL Dataset Schema)](https://github.com/google-research/rlds)
- [LeRobot data format](https://github.com/huggingface/lerobot)
- [Mandlekar et al. "What Matters in Learning from Offline Human Demonstrations" (CoRL 2021)](https://arxiv.org/abs/2108.03298)

---

## Checklist

- [ ] Built keyboard teleoperation interface
- [ ] Recorded episodes in HDF5 format
- [ ] Loaded and validated recorded data
- [ ] Generated scripted demonstrations at scale
- [ ] Implemented data quality filtering pipeline
- [ ] Visualized trajectories and action statistics
- [ ] Compared filtered vs. unfiltered training performance
- [ ] Understand how data quality impacts policy learning
