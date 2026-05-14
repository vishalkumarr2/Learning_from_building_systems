# Exercise 16 — Safety, Monitoring & Failure Recovery for Learned Policies
> Phase VII · Days 105-107 · ~8 hours · **Deployment-critical**

[← Exercise 15: Kinematics & Trajectory](15-kinematics-trajectory.md)

---

## Objectives

By completing this exercise you will:
- Detect out-of-distribution (OOD) inputs that may cause policy failures
- Estimate uncertainty/confidence of learned policy predictions
- Implement runtime safety monitors with workspace, velocity, and force limits
- Design graceful degradation and fallback behaviors
- Build a complete safety pipeline from policy output to robot command
- Understand why safety engineering is non-negotiable for real deployments

## Prerequisites
- Exercise 15 (Kinematics & velocity limits)
- Exercise 12 (MuJoCo manipulation environment)
- Study Note 16 (Deployment & hybrid control)
- Project 08 (Safety gate architecture)

## Setup

```bash
pip install numpy torch matplotlib scikit-learn scipy
```

---

## Part 1: Out-of-Distribution Detection (~2 hours)

### 1.1 — Why OOD Detection Matters

A VLA trained on "pick red blocks from white tables" will confidently output
garbage when it sees a green block on a dark table. OOD detection catches this
*before* the robot acts.

### 1.2 — Feature-Space OOD Detection

```python
"""
Out-of-Distribution detection for VLA inputs.
Detects when the policy sees something it wasn't trained on.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OODStats:
    """Statistics for OOD scoring."""
    is_ood: bool
    score: float           # Higher = more OOD
    threshold: float
    method: str
    details: dict = field(default_factory=dict)


class MahalanobisOODDetector:
    """
    Mahalanobis distance-based OOD detector.

    Fits a multivariate Gaussian to in-distribution feature embeddings,
    then flags inputs whose features are far from the distribution.

    This is what many deployed robotics systems use because it's:
    - Fast (single matrix multiply at inference)
    - Interpretable (distance in feature space)
    - Trainable on only in-distribution data (no OOD examples needed)
    """

    def __init__(self, threshold_percentile: float = 95.0):
        self.threshold_percentile = threshold_percentile
        self.mean: Optional[np.ndarray] = None
        self.cov_inv: Optional[np.ndarray] = None
        self.threshold: float = 0.0
        self._fitted = False

    def fit(self, features: np.ndarray):
        """
        Fit the detector on in-distribution feature embeddings.

        Args:
            features: [N x D] array of feature vectors from training data
        """
        self.mean = np.mean(features, axis=0)
        cov = np.cov(features, rowvar=False)

        # Regularize for numerical stability
        cov += 1e-6 * np.eye(cov.shape[0])
        self.cov_inv = np.linalg.inv(cov)

        # Compute threshold from training data distances
        distances = self._compute_distances(features)
        self.threshold = np.percentile(distances, self.threshold_percentile)
        self._fitted = True

        print(f"OOD Detector fitted: D={features.shape[1]}, N={features.shape[0]}")
        print(f"  Threshold (p{self.threshold_percentile}): {self.threshold:.4f}")
        print(f"  Mean distance: {np.mean(distances):.4f}")

    def _compute_distances(self, features: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distances for a batch."""
        diff = features - self.mean
        # d² = (x - μ)ᵀ Σ⁻¹ (x - μ)
        left = diff @ self.cov_inv
        distances = np.sqrt(np.sum(left * diff, axis=1))
        return distances

    def score(self, feature: np.ndarray) -> OODStats:
        """
        Score a single input feature vector.

        Args:
            feature: [D] feature vector

        Returns:
            OODStats with detection result
        """
        assert self._fitted, "Must call fit() first"

        distance = self._compute_distances(feature.reshape(1, -1))[0]

        return OODStats(
            is_ood=distance > self.threshold,
            score=distance,
            threshold=self.threshold,
            method="mahalanobis",
            details={"distance": distance, "ratio": distance / self.threshold},
        )

    def score_batch(self, features: np.ndarray) -> list[OODStats]:
        """Score a batch of features."""
        distances = self._compute_distances(features)
        return [
            OODStats(
                is_ood=d > self.threshold,
                score=d,
                threshold=self.threshold,
                method="mahalanobis",
                details={"distance": d, "ratio": d / self.threshold},
            )
            for d in distances
        ]


class EnsembleDisagreementDetector:
    """
    OOD detection via ensemble disagreement.

    Train K policy heads on the same data. At inference, if they disagree
    significantly, the input is likely OOD.

    Used by: RT-2, Octo (implicitly via action token sampling diversity)
    """

    def __init__(self, n_heads: int = 5, disagreement_threshold: float = 0.5):
        self.n_heads = n_heads
        self.disagreement_threshold = disagreement_threshold
        self.heads: list = []  # Would be neural network heads in practice

    def compute_disagreement(self, predictions: np.ndarray) -> OODStats:
        """
        Measure disagreement across ensemble predictions.

        Args:
            predictions: [n_heads x action_dim] — predictions from each head

        Returns:
            OODStats based on disagreement magnitude
        """
        # Standard deviation across heads for each action dimension
        std_per_dim = np.std(predictions, axis=0)
        mean_std = np.mean(std_per_dim)
        max_std = np.max(std_per_dim)

        # Coefficient of variation (normalized disagreement)
        mean_pred = np.mean(predictions, axis=0)
        cv = mean_std / (np.linalg.norm(mean_pred) + 1e-8)

        return OODStats(
            is_ood=cv > self.disagreement_threshold,
            score=cv,
            threshold=self.disagreement_threshold,
            method="ensemble_disagreement",
            details={
                "mean_std": mean_std,
                "max_std": max_std,
                "coefficient_of_variation": cv,
                "per_dim_std": std_per_dim.tolist(),
            },
        )


class TemporalOODDetector:
    """
    Detect OOD based on temporal consistency.

    If the policy output changes drastically between timesteps (high jerk),
    the policy may be hallucinating.
    """

    def __init__(self, window_size: int = 10, jerk_threshold: float = 2.0):
        self.window_size = window_size
        self.jerk_threshold = jerk_threshold
        self.history: list[np.ndarray] = []

    def update(self, action: np.ndarray) -> OODStats:
        """
        Update with new action and check temporal consistency.

        Args:
            action: Latest policy output [action_dim]

        Returns:
            OODStats — flags if temporal pattern is abnormal
        """
        self.history.append(action.copy())

        if len(self.history) < 3:
            return OODStats(
                is_ood=False, score=0.0,
                threshold=self.jerk_threshold,
                method="temporal_consistency",
            )

        # Keep window
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size:]

        # Compute jerk (3rd derivative) from recent actions
        actions = np.array(self.history[-4:])
        if len(actions) >= 4:
            vel = np.diff(actions, axis=0)
            acc = np.diff(vel, axis=0)
            jerk = np.diff(acc, axis=0)
            jerk_magnitude = np.linalg.norm(jerk[-1])
        else:
            jerk_magnitude = 0.0

        # Also check: sudden large action after a period of small actions
        recent_magnitudes = [np.linalg.norm(a) for a in self.history[-5:]]
        if len(recent_magnitudes) >= 3:
            mean_magnitude = np.mean(recent_magnitudes[:-1])
            current_magnitude = recent_magnitudes[-1]
            spike_ratio = current_magnitude / (mean_magnitude + 1e-8)
        else:
            spike_ratio = 1.0

        score = max(jerk_magnitude, spike_ratio - 1.0)

        return OODStats(
            is_ood=score > self.jerk_threshold,
            score=score,
            threshold=self.jerk_threshold,
            method="temporal_consistency",
            details={
                "jerk_magnitude": jerk_magnitude,
                "spike_ratio": spike_ratio,
                "history_length": len(self.history),
            },
        )

    def reset(self):
        """Reset history (e.g., at start of new episode)."""
        self.history.clear()
```

### 1.3 — Exercise: Build an OOD Pipeline

```python
# TODO:
# 1. Generate synthetic "in-distribution" features:
#    - 1000 samples from N(μ=[1,2,3,4], Σ=random positive definite)
# 2. Generate "OOD" features:
#    - 100 samples from N(μ=[5,6,7,8], different Σ)
#    - 100 samples from uniform [-10, 10]^4
# 3. Fit the MahalanobisOODDetector on in-distribution data
# 4. Score both ID and OOD data
# 5. Plot: histogram of scores for ID vs OOD (they should separate)
# 6. Compute: precision, recall, F1 for detection
# 7. Test the TemporalOODDetector:
#    - Feed a smooth sinusoidal action sequence → should NOT flag
#    - Insert a random spike at step 50 → should flag
# 8. Design question: what should the robot DO when OOD is detected?
#    Write 3 strategies ordered from conservative to aggressive.

# YOUR CODE HERE
```

---

## Part 2: Uncertainty Estimation (~1.5 hours)

### 2.1 — Monte Carlo Dropout for Policy Uncertainty

```python
"""
Uncertainty estimation for learned policies.
Use MC Dropout to get calibrated uncertainty without changing the model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyWithUncertainty(nn.Module):
    """
    Simple policy network with dropout for MC uncertainty estimation.

    At inference: run forward pass K times with dropout enabled.
    The variance across passes = epistemic uncertainty.
    """

    def __init__(self, obs_dim: int = 32, action_dim: int = 7,
                 hidden_dim: int = 256, dropout_rate: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, action_dim),
        )
        self.dropout_rate = dropout_rate

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.net(obs)

    @torch.no_grad()
    def predict_with_uncertainty(
        self, obs: torch.Tensor, n_samples: int = 20
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        MC Dropout uncertainty estimation.

        Args:
            obs: [batch x obs_dim] observation
            n_samples: Number of stochastic forward passes

        Returns:
            (mean_action, std_action, all_samples)
            - mean_action: [batch x action_dim] — use this as the action
            - std_action: [batch x action_dim] — per-dimension uncertainty
            - all_samples: [n_samples x batch x action_dim] — raw predictions
        """
        self.train()  # Enable dropout even at inference

        samples = []
        for _ in range(n_samples):
            action = self.forward(obs)
            samples.append(action)

        samples = torch.stack(samples)  # [n_samples x batch x action_dim]
        mean_action = samples.mean(dim=0)
        std_action = samples.std(dim=0)

        self.eval()  # Restore eval mode
        return mean_action, std_action, samples


class UncertaintyGate:
    """
    Gate that decides whether to trust the policy based on uncertainty.

    Strategies:
    - EXECUTE: uncertainty low → execute action normally
    - SLOW: uncertainty medium → execute at reduced speed
    - PAUSE: uncertainty high → stop and wait for human input
    """

    def __init__(
        self,
        low_threshold: float = 0.1,
        high_threshold: float = 0.5,
        slow_factor: float = 0.3,
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.slow_factor = slow_factor
        self.consecutive_high: int = 0
        self.max_consecutive_high: int = 5

    def decide(self, mean_action: np.ndarray, std_action: np.ndarray) -> dict:
        """
        Make a safety decision based on uncertainty.

        Returns:
            {
                'decision': 'execute' | 'slow' | 'pause' | 'abort',
                'action': modified action (or None for pause/abort),
                'uncertainty': scalar summary,
                'reason': str,
            }
        """
        # Summary uncertainty: mean of per-dim std
        uncertainty = float(np.mean(std_action))

        if uncertainty < self.low_threshold:
            self.consecutive_high = 0
            return {
                'decision': 'execute',
                'action': mean_action,
                'uncertainty': uncertainty,
                'reason': f'Low uncertainty ({uncertainty:.4f} < {self.low_threshold})',
            }

        elif uncertainty < self.high_threshold:
            self.consecutive_high = 0
            # Execute at reduced speed
            scaled_action = mean_action * self.slow_factor
            return {
                'decision': 'slow',
                'action': scaled_action,
                'uncertainty': uncertainty,
                'reason': f'Medium uncertainty ({uncertainty:.4f}), reducing speed to {self.slow_factor:.0%}',
            }

        else:
            self.consecutive_high += 1
            if self.consecutive_high >= self.max_consecutive_high:
                return {
                    'decision': 'abort',
                    'action': None,
                    'uncertainty': uncertainty,
                    'reason': f'High uncertainty for {self.consecutive_high} steps — aborting episode',
                }
            return {
                'decision': 'pause',
                'action': None,
                'uncertainty': uncertainty,
                'reason': f'High uncertainty ({uncertainty:.4f} > {self.high_threshold}), pausing',
            }
```

### 2.2 — Exercise: Uncertainty in Action

```python
# TODO:
# 1. Create a PolicyWithUncertainty (obs_dim=10, action_dim=4)
# 2. Generate "familiar" observations: torch.randn(100, 10) * 0.5
#    Generate "unfamiliar" observations: torch.randn(100, 10) * 5.0
# 3. For each, compute predict_with_uncertainty(n_samples=30)
# 4. Compare: mean uncertainty for familiar vs unfamiliar inputs
# 5. Plot: uncertainty histogram for both sets
# 6. Wire up UncertaintyGate — feed the predictions through it
# 7. Count: how many familiar inputs get 'execute' vs 'slow' vs 'pause'?
#    How about unfamiliar inputs?
# 8. Tune thresholds until: >90% of familiar → 'execute',
#    >80% of unfamiliar → 'pause' or 'slow'

# YOUR CODE HERE
```

---

## Part 3: Runtime Safety Monitors (~2 hours)

### 3.1 — The Safety Monitor Stack

Real robot deployments need multiple independent safety layers:

```
VLA Policy → OOD Check → Uncertainty Gate → Safety Monitor → Trajectory Limits → Robot
                ↓              ↓                   ↓                  ↓
            [BLOCK]        [SLOW/PAUSE]        [CLAMP/STOP]      [TIME-SCALE]
```

### 3.2 — Implementation

```python
"""
Runtime safety monitors for VLA-controlled robots.
Multiple independent monitors run in parallel — ANY can stop the robot.
"""
import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SafetyLevel(Enum):
    """Safety decision levels (ordered by severity)."""
    NOMINAL = "nominal"          # All good
    WARNING = "warning"          # Log but continue
    LIMIT = "limit"              # Clamp action to safe range
    SLOW = "slow"                # Reduce speed significantly
    STOP = "stop"                # Zero velocity, hold position
    EMERGENCY = "emergency"      # Power off actuators


@dataclass
class SafetyViolation:
    """A detected safety violation."""
    monitor_name: str
    level: SafetyLevel
    message: str
    value: float            # The violating value
    limit: float            # The limit that was exceeded
    action_override: Optional[np.ndarray] = None


class WorkspaceLimitsMonitor:
    """
    Prevents the robot from leaving its safe workspace.

    Checks:
    - Cartesian position bounds (box or sphere)
    - Joint angle limits
    - Height limits (don't go through the table!)
    """

    def __init__(
        self,
        pos_min: np.ndarray = np.array([-0.5, -0.5, 0.0]),
        pos_max: np.ndarray = np.array([0.5, 0.5, 0.6]),
        joint_limits: list[tuple] = None,
    ):
        self.pos_min = pos_min
        self.pos_max = pos_max
        self.joint_limits = joint_limits

    def check(self, ee_position: np.ndarray, joint_angles: np.ndarray = None) -> Optional[SafetyViolation]:
        """Check if current state violates workspace limits."""

        # Cartesian bounds
        for i, (name, val, lo, hi) in enumerate(zip(
            ['x', 'y', 'z'], ee_position, self.pos_min, self.pos_max
        )):
            margin = 0.02  # 2cm warning margin
            if val < lo:
                return SafetyViolation(
                    monitor_name="workspace_limits",
                    level=SafetyLevel.STOP,
                    message=f"EE {name}={val:.3f} below minimum {lo:.3f}",
                    value=val, limit=lo,
                )
            elif val < lo + margin:
                return SafetyViolation(
                    monitor_name="workspace_limits",
                    level=SafetyLevel.WARNING,
                    message=f"EE {name}={val:.3f} near minimum {lo:.3f}",
                    value=val, limit=lo,
                )
            elif val > hi:
                return SafetyViolation(
                    monitor_name="workspace_limits",
                    level=SafetyLevel.STOP,
                    message=f"EE {name}={val:.3f} above maximum {hi:.3f}",
                    value=val, limit=hi,
                )
            elif val > hi - margin:
                return SafetyViolation(
                    monitor_name="workspace_limits",
                    level=SafetyLevel.WARNING,
                    message=f"EE {name}={val:.3f} near maximum {hi:.3f}",
                    value=val, limit=hi,
                )

        # Joint limits
        if joint_angles is not None and self.joint_limits:
            for i, (q, (lo, hi)) in enumerate(zip(joint_angles, self.joint_limits)):
                if q < lo or q > hi:
                    return SafetyViolation(
                        monitor_name="workspace_limits",
                        level=SafetyLevel.LIMIT,
                        message=f"Joint {i}: q={q:.3f} outside [{lo:.3f}, {hi:.3f}]",
                        value=q, limit=lo if q < lo else hi,
                    )

        return None


class VelocityLimitsMonitor:
    """
    Prevents excessive joint or Cartesian velocities.

    Critical for: preventing whiplash motions, hardware damage,
    and unsafe interactions with humans.
    """

    def __init__(
        self,
        max_joint_velocity: np.ndarray = None,
        max_ee_velocity: float = 1.0,        # m/s
        max_ee_angular_velocity: float = 2.0,  # rad/s
    ):
        self.max_joint_velocity = max_joint_velocity
        self.max_ee_velocity = max_ee_velocity
        self.max_ee_angular_velocity = max_ee_angular_velocity
        self.prev_joint_angles: Optional[np.ndarray] = None
        self.prev_ee_position: Optional[np.ndarray] = None
        self.dt: float = 0.01  # 100 Hz control

    def check(self, joint_angles: np.ndarray, ee_position: np.ndarray,
              dt: float = None) -> Optional[SafetyViolation]:
        """Check velocity limits."""
        if dt:
            self.dt = dt

        if self.prev_joint_angles is not None:
            # Joint velocity
            joint_vel = (joint_angles - self.prev_joint_angles) / self.dt

            if self.max_joint_velocity is not None:
                violations = np.abs(joint_vel) > self.max_joint_velocity
                if np.any(violations):
                    worst_joint = np.argmax(np.abs(joint_vel) / self.max_joint_velocity)
                    self.prev_joint_angles = joint_angles
                    self.prev_ee_position = ee_position
                    return SafetyViolation(
                        monitor_name="velocity_limits",
                        level=SafetyLevel.LIMIT,
                        message=f"Joint {worst_joint} velocity {joint_vel[worst_joint]:.3f} "
                                f"exceeds limit {self.max_joint_velocity[worst_joint]:.3f} rad/s",
                        value=float(np.abs(joint_vel[worst_joint])),
                        limit=float(self.max_joint_velocity[worst_joint]),
                    )

            # EE velocity
            if self.prev_ee_position is not None:
                ee_vel = np.linalg.norm(ee_position - self.prev_ee_position) / self.dt
                if ee_vel > self.max_ee_velocity:
                    self.prev_joint_angles = joint_angles
                    self.prev_ee_position = ee_position
                    return SafetyViolation(
                        monitor_name="velocity_limits",
                        level=SafetyLevel.LIMIT,
                        message=f"EE velocity {ee_vel:.3f} m/s exceeds {self.max_ee_velocity:.3f} m/s",
                        value=ee_vel, limit=self.max_ee_velocity,
                    )

        self.prev_joint_angles = joint_angles.copy()
        self.prev_ee_position = ee_position.copy()
        return None


class ForceMonitor:
    """
    Monitors contact forces and torques.

    Prevents: crushing objects, damaging the robot, injuring humans.
    Requires: force/torque sensor data (simulated here).
    """

    def __init__(
        self,
        max_force: float = 20.0,       # Newtons
        max_torque: float = 5.0,        # Nm
        sustained_force_limit: float = 10.0,
        sustained_duration: float = 0.5,  # seconds
    ):
        self.max_force = max_force
        self.max_torque = max_torque
        self.sustained_force_limit = sustained_force_limit
        self.sustained_duration = sustained_duration
        self.force_history: list[tuple[float, float]] = []  # (timestamp, force_magnitude)

    def check(self, force: np.ndarray, torque: np.ndarray,
              timestamp: float) -> Optional[SafetyViolation]:
        """Check force/torque limits."""
        force_mag = np.linalg.norm(force)
        torque_mag = np.linalg.norm(torque)

        # Instantaneous force limit
        if force_mag > self.max_force:
            return SafetyViolation(
                monitor_name="force_limits",
                level=SafetyLevel.EMERGENCY,
                message=f"Force {force_mag:.1f}N exceeds emergency limit {self.max_force:.1f}N",
                value=force_mag, limit=self.max_force,
            )

        # Instantaneous torque limit
        if torque_mag > self.max_torque:
            return SafetyViolation(
                monitor_name="force_limits",
                level=SafetyLevel.STOP,
                message=f"Torque {torque_mag:.2f}Nm exceeds limit {self.max_torque:.2f}Nm",
                value=torque_mag, limit=self.max_torque,
            )

        # Sustained force check
        self.force_history.append((timestamp, force_mag))
        # Trim old entries
        cutoff = timestamp - self.sustained_duration
        self.force_history = [(t, f) for t, f in self.force_history if t > cutoff]

        if len(self.force_history) > 5:
            avg_force = np.mean([f for _, f in self.force_history])
            if avg_force > self.sustained_force_limit:
                return SafetyViolation(
                    monitor_name="force_limits",
                    level=SafetyLevel.SLOW,
                    message=f"Sustained force {avg_force:.1f}N > {self.sustained_force_limit:.1f}N "
                            f"over {self.sustained_duration}s",
                    value=avg_force, limit=self.sustained_force_limit,
                )

        return None


class ProgressMonitor:
    """
    Detects if the policy is stuck (no progress toward goal).

    Symptoms: oscillation, repeated small motions, circular paths.
    """

    def __init__(self, stall_threshold: float = 0.005, stall_window: int = 50):
        self.stall_threshold = stall_threshold  # meters
        self.stall_window = stall_window        # timesteps
        self.position_history: list[np.ndarray] = []

    def check(self, ee_position: np.ndarray) -> Optional[SafetyViolation]:
        """Check if the robot is making progress."""
        self.position_history.append(ee_position.copy())

        if len(self.position_history) < self.stall_window:
            return None

        # Keep window
        self.position_history = self.position_history[-self.stall_window:]

        # Total displacement over window
        start = self.position_history[0]
        end = self.position_history[-1]
        displacement = np.linalg.norm(end - start)

        # Path length over window (how much it moved total)
        path_length = sum(
            np.linalg.norm(self.position_history[i+1] - self.position_history[i])
            for i in range(len(self.position_history) - 1)
        )

        if displacement < self.stall_threshold and path_length > 0.01:
            # Moving but not going anywhere → oscillating
            return SafetyViolation(
                monitor_name="progress",
                level=SafetyLevel.WARNING,
                message=f"Possible oscillation: displacement={displacement:.4f}m "
                        f"but path_length={path_length:.4f}m over {self.stall_window} steps",
                value=displacement, limit=self.stall_threshold,
            )
        elif displacement < self.stall_threshold and path_length < self.stall_threshold:
            # Not moving at all
            return SafetyViolation(
                monitor_name="progress",
                level=SafetyLevel.WARNING,
                message=f"Robot stalled: no motion for {self.stall_window} steps",
                value=displacement, limit=self.stall_threshold,
            )

        return None
```

### 3.3 — The Safety Supervisor

```python
class SafetySupervisor:
    """
    Orchestrates all safety monitors and makes final decisions.

    Architecture: multiple independent monitors → worst-case aggregation → decision.
    This is the same pattern used in real autonomous systems (aviation, nuclear, robotics).
    """

    def __init__(self):
        self.monitors = {
            'workspace': WorkspaceLimitsMonitor(),
            'velocity': VelocityLimitsMonitor(
                max_joint_velocity=np.array([2.0, 2.0, 2.0, 3.0, 3.0, 3.0]),
                max_ee_velocity=1.0,
            ),
            'force': ForceMonitor(),
            'progress': ProgressMonitor(),
        }
        self.violation_log: list[SafetyViolation] = []
        self.step_count: int = 0

    def check_all(
        self,
        ee_position: np.ndarray,
        joint_angles: np.ndarray,
        force: np.ndarray = None,
        torque: np.ndarray = None,
        timestamp: float = None,
    ) -> tuple[SafetyLevel, Optional[SafetyViolation]]:
        """
        Run all monitors and return the worst safety level.

        Returns:
            (worst_level, worst_violation) or (NOMINAL, None)
        """
        self.step_count += 1
        violations = []

        # Workspace check
        v = self.monitors['workspace'].check(ee_position, joint_angles)
        if v:
            violations.append(v)

        # Velocity check
        v = self.monitors['velocity'].check(joint_angles, ee_position)
        if v:
            violations.append(v)

        # Force check (if sensors available)
        if force is not None and torque is not None:
            if timestamp is None:
                timestamp = self.step_count * 0.01
            v = self.monitors['force'].check(force, torque, timestamp)
            if v:
                violations.append(v)

        # Progress check
        v = self.monitors['progress'].check(ee_position)
        if v:
            violations.append(v)

        if not violations:
            return SafetyLevel.NOMINAL, None

        # Return worst violation
        severity_order = [
            SafetyLevel.EMERGENCY,
            SafetyLevel.STOP,
            SafetyLevel.LIMIT,
            SafetyLevel.SLOW,
            SafetyLevel.WARNING,
        ]

        worst = violations[0]
        for v in violations[1:]:
            if severity_order.index(v.level) < severity_order.index(worst.level):
                worst = v

        self.violation_log.append(worst)
        return worst.level, worst

    def get_safe_action(
        self,
        proposed_action: np.ndarray,
        level: SafetyLevel,
        current_position: np.ndarray,
    ) -> np.ndarray:
        """
        Modify action based on safety level.

        Args:
            proposed_action: The action the policy wants to execute
            level: Current safety level
            current_position: Current joint configuration

        Returns:
            Safe action to execute
        """
        if level == SafetyLevel.NOMINAL or level == SafetyLevel.WARNING:
            return proposed_action

        elif level == SafetyLevel.SLOW:
            # Reduce to 20% speed
            return proposed_action * 0.2

        elif level == SafetyLevel.LIMIT:
            # Clamp magnitude
            max_magnitude = 0.01  # very small step
            magnitude = np.linalg.norm(proposed_action)
            if magnitude > max_magnitude:
                return proposed_action * (max_magnitude / magnitude)
            return proposed_action

        elif level in (SafetyLevel.STOP, SafetyLevel.EMERGENCY):
            # Zero action — stay put
            return np.zeros_like(proposed_action)

        return proposed_action

    def summary(self) -> dict:
        """Get summary statistics of safety events."""
        if not self.violation_log:
            return {"total_violations": 0, "worst_level": "nominal"}

        level_counts = {}
        for v in self.violation_log:
            level_counts[v.level.value] = level_counts.get(v.level.value, 0) + 1

        return {
            "total_violations": len(self.violation_log),
            "total_steps": self.step_count,
            "violation_rate": len(self.violation_log) / max(self.step_count, 1),
            "level_counts": level_counts,
            "worst_level": self.violation_log[-1].level.value,
        }
```

### 3.4 — Exercise: Safety Monitor Integration

```python
# TODO:
# 1. Create a SafetySupervisor
# 2. Simulate a robot trajectory:
#    - 100 steps of normal motion (sinusoidal joint angles)
#    - 20 steps where EE goes out of bounds
#    - 30 steps of excessive velocity
#    - 10 steps of high force
# 3. Run check_all() at each step
# 4. Plot: safety level over time (color-coded)
# 5. Count violations by type and severity
# 6. Test get_safe_action: show that dangerous actions are clamped/zeroed
# 7. Design question: In a real deployment, what happens when EMERGENCY fires?
#    Write the shutdown sequence (hint: think about gravity, gripper state,
#    communication with the fleet manager).

# YOUR CODE HERE
```

---

## Part 4: Graceful Degradation & Fallback Behaviors (~1.5 hours)

### 4.1 — Fallback Strategy Pattern

```python
"""
Fallback behaviors when the VLA policy fails or is uncertain.
Implements the "policy stack" pattern: try policy → fallback → safe stop.
"""
import numpy as np
from enum import Enum
from typing import Callable, Optional


class FallbackLevel(Enum):
    """Hierarchy of fallback behaviors (ordered by autonomy)."""
    VLA_POLICY = 0        # Full VLA — highest autonomy
    REDUCED_SPEED = 1     # Same policy, slower execution
    SCRIPTED_RECOVERY = 2 # Pre-programmed recovery motion
    SAFE_RETRACT = 3      # Move to known safe configuration
    HOLD_POSITION = 4     # Stop all motion, maintain current pose
    COMPLIANCE_MODE = 5   # Gravity compensation only (soft)
    POWER_OFF = 6         # Cut power to actuators (last resort)


class FallbackManager:
    """
    Manages transitions between autonomy levels.

    When the VLA fails, don't just stop — degrade gracefully:
    1. Reduce speed first
    2. Try scripted recovery
    3. If all else fails, retract to home
    """

    def __init__(self, home_position: np.ndarray):
        self.home_position = home_position
        self.current_level = FallbackLevel.VLA_POLICY
        self.level_history: list[tuple[float, FallbackLevel]] = []
        self.recovery_attempts: int = 0
        self.max_recovery_attempts: int = 3
        self.time: float = 0.0

    def escalate(self, reason: str) -> FallbackLevel:
        """Escalate to next fallback level."""
        next_level = FallbackLevel(min(self.current_level.value + 1, 6))
        print(f"[FALLBACK] Escalating: {self.current_level.name} → {next_level.name}")
        print(f"  Reason: {reason}")
        self.current_level = next_level
        self.level_history.append((self.time, next_level))
        return next_level

    def de_escalate(self) -> FallbackLevel:
        """Return to higher autonomy after recovery."""
        if self.current_level.value > 0:
            prev_level = FallbackLevel(self.current_level.value - 1)
            print(f"[FALLBACK] De-escalating: {self.current_level.name} → {prev_level.name}")
            self.current_level = prev_level
            self.level_history.append((self.time, prev_level))
        return self.current_level

    def get_action(
        self,
        vla_action: np.ndarray,
        current_position: np.ndarray,
        safety_level: SafetyLevel,
        uncertainty: float,
    ) -> tuple[np.ndarray, FallbackLevel]:
        """
        Get the appropriate action given current conditions.

        Args:
            vla_action: Raw VLA policy output
            current_position: Current joint angles
            safety_level: From SafetySupervisor
            uncertainty: From UncertaintyGate

        Returns:
            (action_to_execute, active_fallback_level)
        """
        self.time += 0.01

        # Decision logic
        if safety_level == SafetyLevel.EMERGENCY:
            self.current_level = FallbackLevel.POWER_OFF
            return np.zeros_like(vla_action), self.current_level

        elif safety_level == SafetyLevel.STOP:
            if self.current_level.value < FallbackLevel.HOLD_POSITION.value:
                self.escalate("Safety STOP triggered")
            return np.zeros_like(vla_action), self.current_level

        elif safety_level == SafetyLevel.LIMIT or uncertainty > 0.5:
            if self.current_level == FallbackLevel.VLA_POLICY:
                self.escalate("High uncertainty or safety limit")
            return self._scripted_recovery(current_position), self.current_level

        elif safety_level == SafetyLevel.SLOW or uncertainty > 0.2:
            if self.current_level == FallbackLevel.VLA_POLICY:
                self.current_level = FallbackLevel.REDUCED_SPEED
            return vla_action * 0.3, self.current_level

        else:
            # Everything normal — try to de-escalate if we were in fallback
            if self.current_level != FallbackLevel.VLA_POLICY:
                self.recovery_attempts += 1
                if self.recovery_attempts >= 10:  # 10 consecutive good steps
                    self.de_escalate()
                    self.recovery_attempts = 0
            return vla_action, self.current_level

    def _scripted_recovery(self, current_position: np.ndarray) -> np.ndarray:
        """
        Move toward home position at reduced speed.

        Uses proportional control: action = gain * (home - current)
        """
        gain = 0.1  # Slow movement toward home
        error = self.home_position - current_position
        action = gain * error

        # Clip to very conservative limits
        max_step = 0.01  # rad per step
        action = np.clip(action, -max_step, max_step)
        return action


class TaskTimeoutMonitor:
    """
    Detects when a task is taking too long.

    VLA tasks should complete within a reasonable time window.
    If not, the policy is likely stuck or confused.
    """

    def __init__(self, max_duration: float = 30.0, warning_ratio: float = 0.8):
        self.max_duration = max_duration
        self.warning_ratio = warning_ratio
        self.start_time: Optional[float] = None
        self.task_name: str = ""

    def start_task(self, name: str, max_duration: float = None):
        """Begin timing a task."""
        self.task_name = name
        self.start_time = 0.0
        if max_duration:
            self.max_duration = max_duration

    def tick(self, dt: float = 0.01) -> Optional[SafetyViolation]:
        """Update timer and check for timeout."""
        if self.start_time is None:
            return None

        self.start_time += dt
        elapsed = self.start_time

        if elapsed > self.max_duration:
            return SafetyViolation(
                monitor_name="task_timeout",
                level=SafetyLevel.STOP,
                message=f"Task '{self.task_name}' exceeded {self.max_duration}s timeout",
                value=elapsed,
                limit=self.max_duration,
            )
        elif elapsed > self.max_duration * self.warning_ratio:
            return SafetyViolation(
                monitor_name="task_timeout",
                level=SafetyLevel.WARNING,
                message=f"Task '{self.task_name}' at {elapsed/self.max_duration:.0%} of timeout",
                value=elapsed,
                limit=self.max_duration,
            )

        return None

    def complete_task(self):
        """Mark task as completed."""
        elapsed = self.start_time if self.start_time else 0
        print(f"Task '{self.task_name}' completed in {elapsed:.2f}s")
        self.start_time = None
```

### 4.2 — Exercise: Full Safety Pipeline

```python
# TODO:
# 1. Build the complete pipeline:
#    OOD Detector → Uncertainty Gate → Safety Supervisor → Fallback Manager
# 2. Simulate 500-step episode with a "policy" that:
#    - Steps 0-200: normal actions (small smooth motions)
#    - Steps 200-250: OOD input (policy outputs large random actions)
#    - Steps 250-350: recovery (policy returns to normal)
#    - Steps 350-400: workspace violation (drifts out of bounds)
#    - Steps 400-500: normal again
# 3. Plot timeline showing:
#    - Row 1: action magnitudes
#    - Row 2: OOD score
#    - Row 3: uncertainty
#    - Row 4: safety level (color-coded)
#    - Row 5: fallback level (escalation/de-escalation)
# 4. Count: how many timesteps was the robot in each fallback level?
# 5. Verify: the robot NEVER executes a dangerous action when safety fires
# 6. Design extension: add a "human notification" module that would alert
#    an operator when fallback level >= SAFE_RETRACT for > 5 seconds.
#    (This connects to OKS AMR's RequestManualAssistance pattern!)

# YOUR CODE HERE
```

---

## Part 5: Evaluation & Metrics (~1 hour)

### 5.1 — Safety Metrics for VLA Deployments

```python
"""
Metrics for evaluating safety system effectiveness.
These are what you'd report in a VLA deployment paper or safety review.
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class SafetyMetrics:
    """Comprehensive safety evaluation metrics."""

    # Detection metrics
    true_positive_rate: float    # Caught real dangers
    false_positive_rate: float   # False alarms (annoying but safe)
    detection_latency_ms: float  # Time from danger onset to detection

    # Intervention metrics
    total_interventions: int     # How many times safety fired
    intervention_rate: float     # interventions / total_steps
    mean_intervention_duration: float  # How long each intervention lasts

    # Task metrics
    task_success_rate: float     # Did the robot complete the task?
    task_time_overhead: float    # How much slower with safety vs without

    # Severity metrics
    max_violation_force: float   # Worst force ever applied
    max_violation_velocity: float # Fastest speed ever reached
    workspace_violation_count: int  # Times EE left bounds (should be 0!)


def evaluate_safety_system(
    actions_without_safety: list[np.ndarray],
    actions_with_safety: list[np.ndarray],
    violations_detected: list[SafetyViolation],
    ground_truth_dangers: list[int],  # timesteps that SHOULD be caught
    task_completed: bool,
    task_duration_safe: float,
    task_duration_baseline: float,
) -> SafetyMetrics:
    """
    Compare system performance with and without safety monitors.

    Args:
        actions_without_safety: Raw policy outputs (unchecked)
        actions_with_safety: Actual executed actions (after safety)
        violations_detected: All violations flagged by monitors
        ground_truth_dangers: Timesteps where danger actually existed
        task_completed: Whether the task finished successfully
        task_duration_safe: Duration with safety enabled
        task_duration_baseline: Duration without safety

    Returns:
        Comprehensive safety metrics
    """
    total_steps = len(actions_without_safety)
    detected_steps = set(i for i, v in enumerate(violations_detected) if v is not None)
    danger_steps = set(ground_truth_dangers)

    # Detection metrics
    true_positives = len(detected_steps & danger_steps)
    false_positives = len(detected_steps - danger_steps)
    false_negatives = len(danger_steps - detected_steps)

    tpr = true_positives / max(len(danger_steps), 1)
    fpr = false_positives / max(total_steps - len(danger_steps), 1)

    # Latency: how quickly after danger starts is it detected?
    latencies = []
    for danger_step in sorted(danger_steps):
        for detect_step in sorted(detected_steps):
            if detect_step >= danger_step:
                latencies.append((detect_step - danger_step) * 10)  # ms at 100Hz
                break

    # Intervention analysis
    interventions = [v for v in violations_detected if v is not None
                     and v.level.value >= SafetyLevel.SLOW.value]

    # Force/velocity analysis
    action_magnitudes = [np.linalg.norm(a) for a in actions_without_safety]

    return SafetyMetrics(
        true_positive_rate=tpr,
        false_positive_rate=fpr,
        detection_latency_ms=np.mean(latencies) if latencies else 0.0,
        total_interventions=len(interventions),
        intervention_rate=len(interventions) / max(total_steps, 1),
        mean_intervention_duration=0.0,  # Would compute from consecutive violations
        task_success_rate=1.0 if task_completed else 0.0,
        task_time_overhead=(task_duration_safe - task_duration_baseline) / task_duration_baseline,
        max_violation_force=0.0,  # From force monitor logs
        max_violation_velocity=max(action_magnitudes) if action_magnitudes else 0.0,
        workspace_violation_count=sum(
            1 for v in violations_detected
            if v and v.monitor_name == "workspace_limits"
            and v.level.value >= SafetyLevel.STOP.value
        ),
    )


def print_safety_report(metrics: SafetyMetrics):
    """Print a formatted safety evaluation report."""
    print("=" * 60)
    print("        SAFETY SYSTEM EVALUATION REPORT")
    print("=" * 60)
    print(f"\n--- Detection Performance ---")
    print(f"  True Positive Rate:  {metrics.true_positive_rate:.1%}")
    print(f"  False Positive Rate: {metrics.false_positive_rate:.1%}")
    print(f"  Detection Latency:   {metrics.detection_latency_ms:.1f} ms")
    print(f"\n--- Intervention Stats ---")
    print(f"  Total Interventions: {metrics.total_interventions}")
    print(f"  Intervention Rate:   {metrics.intervention_rate:.2%}")
    print(f"\n--- Task Performance ---")
    print(f"  Task Success Rate:   {metrics.task_success_rate:.0%}")
    print(f"  Time Overhead:       {metrics.task_time_overhead:+.1%}")
    print(f"\n--- Worst Cases ---")
    print(f"  Max Velocity:        {metrics.max_violation_velocity:.3f}")
    print(f"  Workspace Violations:{metrics.workspace_violation_count}")
    print("=" * 60)

    # Pass/Fail criteria
    passed = (
        metrics.true_positive_rate > 0.95
        and metrics.workspace_violation_count == 0
        and metrics.false_positive_rate < 0.10
    )
    print(f"\n  VERDICT: {'✅ PASS' if passed else '❌ FAIL'}")
    if not passed:
        if metrics.true_positive_rate <= 0.95:
            print(f"    ⚠ TPR {metrics.true_positive_rate:.1%} < 95% required")
        if metrics.workspace_violation_count > 0:
            print(f"    ⚠ {metrics.workspace_violation_count} workspace violations (must be 0)")
        if metrics.false_positive_rate >= 0.10:
            print(f"    ⚠ FPR {metrics.false_positive_rate:.1%} ≥ 10% threshold")
```

### 5.2 — Exercise: Run Full Evaluation

```python
# TODO:
# 1. Simulate 1000-step episode with known danger regions
#    (steps 200-250 and 600-650 are "dangerous")
# 2. Run the full safety pipeline
# 3. Compute SafetyMetrics
# 4. Print the safety report
# 5. Tune thresholds until: TPR > 95%, FPR < 10%, latency < 50ms
# 6. Plot ROC curve: vary OOD threshold, compute TPR vs FPR
# 7. Final reflection questions:
#    a) What's the cost of a false positive? (Robot pauses unnecessarily)
#    b) What's the cost of a false negative? (Robot damages something)
#    c) For a warehouse robot, which error is MORE acceptable?
#    d) How does this relate to OKS AMR's safety philosophy?

# YOUR CODE HERE
```

---

## Summary & Key Takeaways

| Component | Purpose | Failure Mode Without It |
|-----------|---------|------------------------|
| OOD Detection | Catch unseen inputs | Policy outputs garbage, robot acts randomly |
| Uncertainty Estimation | Know when policy is confused | Overconfident wrong actions |
| Workspace Limits | Prevent out-of-bounds | Collision with environment/humans |
| Velocity Limits | Prevent fast motions | Hardware damage, whiplash |
| Force Limits | Prevent crushing | Object/human injury |
| Progress Monitor | Detect stuck policies | Infinite loops, oscillation |
| Fallback Manager | Graceful degradation | Binary "works or crashes" |
| Task Timeout | Catch silent failures | Robot stands forever |

### The Key Lesson

**Safety systems are NOT optional for VLA deployment.** They are the difference between a research demo and a production system. Every deployed VLA needs:
1. Multiple independent safety layers (defense in depth)
2. Graceful degradation (don't just stop — recover)
3. Quantitative metrics (prove it works before deployment)
4. Logging and monitoring (learn from every near-miss)

### Connection to Other Exercises
- **Exercise 15**: Velocity/acceleration limits come from kinematics
- **Project 08**: Safety gate implements these monitors in ROS 2
- **Exercise 14**: Data quality filtering is a training-time safety measure
- **Real OKS**: Guardian node, e-stop, RequestManualAssistance — same architecture!

---

## Checklist

- [ ] Implemented Mahalanobis distance OOD detector
- [ ] Implemented temporal OOD detection
- [ ] Built ensemble disagreement detector
- [ ] Implemented MC Dropout uncertainty estimation
- [ ] Built UncertaintyGate with execute/slow/pause decisions
- [ ] Implemented workspace limits monitor
- [ ] Implemented velocity limits monitor
- [ ] Implemented force/torque monitor
- [ ] Implemented progress (stall/oscillation) monitor
- [ ] Built SafetySupervisor aggregating all monitors
- [ ] Implemented FallbackManager with escalation/de-escalation
- [ ] Built task timeout detection
- [ ] Ran full safety pipeline simulation
- [ ] Computed safety metrics (TPR, FPR, latency)
- [ ] Tuned thresholds for >95% TPR, <10% FPR
- [ ] Connected to real deployment considerations (OKS AMR patterns)
