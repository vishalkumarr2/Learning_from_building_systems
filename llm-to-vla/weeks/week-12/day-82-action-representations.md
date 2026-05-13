# Day 82: Action Representations — How Robots Speak in Actions

> **Phase VI — Robot Learning: RL, Diffusion & Data | Week 12 | 2.5 hours**
> *"The way you represent actions determines what your policy can learn. Get this wrong and no architecture will save you."*

## Navigation
- **Previous:** [Day 81: Diffusion Policy](day-81-diffusion-policy.md)
- **Next:** [Day 83: Action Tokenization](day-83-action-tokenization.md)
- **Week:** [Week 12 Overview](README.md)
- **Phase:** [Phase VI: Robot Learning](../../study-notes/13-imitation-learning.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 82.1 Action Space Taxonomy

| Representation | Example | Pros | Cons |
|----------------|---------|------|------|
| **Joint positions** (absolute) | $q = [q_1, \ldots, q_7]$ | Precise, reproducible | Task-specific, not transferable |
| **Joint velocities** | $\dot{q} = [\dot{q}_1, \ldots, \dot{q}_7]$ | Smooth motion | Drift over time |
| **Joint position deltas** | $\Delta q = q_{t+1} - q_t$ | Small values, stable | Still joint-specific |
| **End-effector pose** (abs) | $[x,y,z,\text{quat}]$ | Task-relevant | Needs IK, singularities |
| **End-effector delta** | $[\Delta x, \Delta y, \Delta z, \Delta\text{rot}]$ | Transferable | Cumulative error |
| **Discrete tokens** | bin index: $[128, 64, 200]$ | Language model compatible | Resolution limited |

### 82.2 Absolute vs Delta Actions

```
Absolute:  "Move to position (0.5, 0.3, 0.1)"
  + Repeatable trajectories
  + No drift
  - Must learn full pose from scratch
  - Distribution differs per task/setup

Delta:     "Move +0.01 in x, -0.02 in y"
  + Small values → easier to learn
  + Transferable across setups
  - Error accumulates over time
  - Sensitive to control frequency
```

**In practice:** delta end-effector actions with a compliance controller are the most common choice for manipulation VLAs.

### 82.3 End-Effector vs Joint Space

```
End-Effector space:                Joint space:
  "Move gripper to (x,y,z)"         "Set joints to (q1,...,q7)"
      │                                  │
      ▼                                  ▼
  Inverse Kinematics (IK)            Direct motor commands
      │                                  │
      ▼                                  ▼
  Joint commands                      Motor execution
```

| Factor | End-Effector | Joint |
|--------|-------------|-------|
| **Transferability** | High (across robot morphologies) | Low (robot-specific) |
| **Precision** | Lower (IK errors) | Higher (direct control) |
| **VLA compatibility** | Better (observation-aligned) | Harder (abstract) |
| **Action dimension** | 6-7 (pose + gripper) | 6-12+ (per joint) |

### 82.4 Rotation Representations

Representing 3D rotations is surprisingly tricky:

| Representation | Dim | Continuity | Used By |
|----------------|-----|------------|---------|
| Euler angles | 3 | Discontinuous (gimbal lock) | Avoid for learning |
| Quaternion | 4 | Double cover (q = -q) | RT-1, Octo |
| Axis-angle | 3 | Singularity at 0 | Some, with care |
| Rotation matrix | 9 | Redundant but continuous | Preferred for learning |
| 6D representation | 6 | Continuous, minimal+ | Zhou et al. (2019) |

**The 6D rotation** from Zhou et al. (2019) extracts the first two columns of the rotation matrix, then reconstructs the third via cross product:

$$R = [\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_1 \times \mathbf{a}_2], \quad \text{after Gram-Schmidt}$$

### 82.5 Gripper Actions

| Method | Description | Trade-offs |
|--------|-------------|------------|
| Continuous | Gripper width: 0.0 (closed) to 1.0 (open) | Smooth but mode-averaging |
| Binary | 0 (close) or 1 (open) | Simple, natural for pick-place |
| Velocity | Open/close speed | Responsive but drifts |

### 82.6 Action Frequency and Control

| Frequency | Latency Budget | Suitable For |
|-----------|---------------|--------------|
| 50 Hz | 20ms | Low-level joint control |
| 10 Hz | 100ms | End-effector delta control |
| 5 Hz | 200ms | VLA inference (current limit) |
| 2 Hz | 500ms | High-level planning |

**Action chunking mitigates low frequency:** predict 10 actions at 5 Hz inference = 50 Hz effective control.

---

## Implementation (60 min)

### Action Space Comparison

```python
import numpy as np
import torch

class ActionConverter:
    """Convert between action representations."""

    @staticmethod
    def absolute_to_delta(actions):
        """Convert absolute positions to deltas."""
        deltas = np.diff(actions, axis=0)
        return deltas

    @staticmethod
    def delta_to_absolute(deltas, start_pos):
        """Convert deltas back to absolute positions."""
        positions = [start_pos]
        for d in deltas:
            positions.append(positions[-1] + d)
        return np.array(positions)

    @staticmethod
    def euler_to_rotation_matrix(euler):
        """Convert euler angles (roll, pitch, yaw) to rotation matrix."""
        r, p, y = euler
        Rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
        Ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
        Rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
        return Rz @ Ry @ Rx

    @staticmethod
    def rotation_matrix_to_6d(R):
        """Extract 6D continuous rotation representation."""
        return R[:, :2].T.flatten()  # First two columns, flattened

    @staticmethod
    def sixd_to_rotation_matrix(sixd):
        """Reconstruct rotation matrix from 6D representation."""
        a1 = sixd[:3]
        a2 = sixd[3:]
        # Gram-Schmidt
        b1 = a1 / np.linalg.norm(a1)
        b2 = a2 - np.dot(b1, a2) * b1
        b2 = b2 / np.linalg.norm(b2)
        b3 = np.cross(b1, b2)
        return np.column_stack([b1, b2, b3])

# --- Experiment: delta accumulation error ---
def simulate_delta_drift(n_steps=200, noise_std=0.001):
    """Show how small errors in delta actions accumulate."""
    true_trajectory = np.sin(np.linspace(0, 4*np.pi, n_steps)).reshape(-1, 1)
    true_trajectory = np.hstack([true_trajectory, np.cos(np.linspace(0, 4*np.pi, n_steps)).reshape(-1, 1)])

    # Perfect deltas
    deltas = np.diff(true_trajectory, axis=0)

    # Add noise to deltas
    noisy_deltas = deltas + np.random.randn(*deltas.shape) * noise_std

    # Reconstruct
    noisy_trajectory = ActionConverter.delta_to_absolute(noisy_deltas, true_trajectory[0])

    # Measure drift
    errors = np.linalg.norm(true_trajectory - noisy_trajectory, axis=1)
    print(f"Final drift: {errors[-1]:.4f}")
    print(f"Max drift: {errors.max():.4f}")
    return true_trajectory, noisy_trajectory, errors

true_traj, noisy_traj, errors = simulate_delta_drift()
```

### Normalize Action Spaces

```python
class ActionNormalizer:
    """Normalize actions to [-1, 1] for stable learning."""
    def __init__(self, low, high):
        self.low = np.array(low, dtype=np.float32)
        self.high = np.array(high, dtype=np.float32)
        self.range = self.high - self.low

    def normalize(self, actions):
        return 2.0 * (actions - self.low) / self.range - 1.0

    def denormalize(self, normalized):
        return (normalized + 1.0) * self.range / 2.0 + self.low

# Example: 7-DOF robot arm
normalizer = ActionNormalizer(
    low=[-0.05, -0.05, -0.05, -0.1, -0.1, -0.1, 0.0],   # delta xyz, delta rot, gripper
    high=[0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 1.0],
)
```

---

## Exercise (45 min)

1. **Drift experiment:** Generate a trajectory with `simulate_delta_drift` for noise levels $\sigma \in \{0.0001, 0.001, 0.01\}$. Plot drift over time. At what noise level does delta representation become impractical?

2. **Rotation continuity:** Sample 1000 random rotations. Convert to Euler, quaternion, and 6D. Perturb slightly and convert back. Which representation is most stable (smallest reconstruction error)?

3. **Action space for OKS:** Your OKS AMR robot has $(v, \omega)$ velocity commands. Design the action representation: continuous delta? Discrete bins? What chunk size? Write a specification.

4. **Normalization impact:** Train a BC policy on unnormalized vs normalized actions. Compare learning curves.

---

## Key Takeaways

1. **Delta end-effector** is the most common VLA action representation
2. **6D rotation** is continuous and avoids gimbal lock/quaternion ambiguity
3. **Action normalization** to $[-1, 1]$ is critical for stable training
4. **Chunking compensates** for low-frequency VLA inference
5. **Binary gripper** actions are simpler and often more effective than continuous

---

## Connection to the Thread

> *Continuous actions work for diffusion/flow-based policies. But what if you want to use a language model to generate actions? Tomorrow we tokenize: discretize continuous actions into a finite vocabulary so transformers can predict them with cross-entropy loss — exactly how RT-2 turns a VLM into a VLA.*

---

## Further Reading

- Zhou et al. (2019), "On the Continuity of Rotation Representations in Neural Networks"
- Mandlekar et al. (2021), "What Matters in Learning from Offline Human Demonstrations for Robot Manipulation"
- Chi et al. (2023), "Diffusion Policy" — action space analysis in appendix
