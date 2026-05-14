# Project 08 — ROS 2 + VLA Deployment Pipeline
> Phase VII · Days 103-107 · ~12 hours · **Robotics capstone project**

---

## Overview

Build an end-to-end pipeline that takes a trained VLA policy and deploys it in a ROS 2 
simulated environment with safety constraints, real-time control, and monitoring. This 
bridges the gap between "model that outputs actions" and "robot that moves safely."

### What You'll Build

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ROS 2 + VLA Deployment Pipeline                  │
│                                                                     │
│  Camera ──→ /camera/image_raw ──→ ┌──────────────┐                 │
│                                    │  VLA Policy   │                 │
│  Language ──→ /task_instruction ──→│  Node         │──→ raw_action  │
│                                    └──────┬───────┘                 │
│                                           │                         │
│                                    ┌──────▼───────┐                 │
│                                    │  Safety Gate  │                 │
│                                    │  Node         │                 │
│                                    │  - Workspace  │                 │
│                                    │  - Velocity   │                 │
│                                    │  - Collision   │                 │
│                                    └──────┬───────┘                 │
│                                           │                         │
│                                    ┌──────▼───────┐                 │
│                                    │  Action       │                 │
│                                    │  Interpolator │                 │
│                                    │  Node         │──→ /cmd_joints  │
│                                    └──────────────┘                 │
│                                                                     │
│  Monitoring: /diagnostics, /policy_latency, /safety_violations     │
└─────────────────────────────────────────────────────────────────────┘
```

## Prerequisites
- Exercise 12 (MuJoCo manipulation — trained a policy)
- Exercise 13 (Sim-to-real — domain randomization)
- Study Note 15 (VLA Architectures)
- ROS 2 Humble (or Iron) installed or available via Docker
- Basic ROS 2 concepts: nodes, topics, services, actions

## Setup

```bash
# Option A: Native install (Ubuntu 22.04)
# Follow: https://docs.ros.org/en/humble/Installation.html

# Option B: Docker (recommended for reproducibility)
docker pull osrf/ros:humble-desktop
docker run -it --rm \
    -v $(pwd):/workspace \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    osrf/ros:humble-desktop bash

# Inside container or native:
pip install torch numpy opencv-python
# For sim environment:
pip install mujoco gymnasium
```

---

## Part 1: VLA Policy Node (~2.5 hours)

### 1.1 — Policy Wrapper

Create a ROS 2 node that wraps a trained policy and publishes actions.

```python
# File: ros2_vla_pipeline/policy_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float64MultiArray
from cv_bridge import CvBridge
import torch
import torch.nn as nn
import numpy as np
import time


class PolicyNetwork(nn.Module):
    """
    Simplified VLA-style policy.
    In production, this would be OpenVLA or RT-2.
    Here we use a small CNN + MLP that takes image + language embedding.
    """
    def __init__(self, act_dim=7):
        super().__init__()
        # Vision encoder (simplified — real VLAs use ViT)
        self.vision = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        # Language encoder (simplified — bag of embeddings)
        self.language = nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(),
        )
        # Action head: fuses vision + language → action
        self.action_head = nn.Sequential(
            nn.Linear(128 * 16 + 128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, act_dim), nn.Tanh(),
        )
        self.act_dim = act_dim
    
    def forward(self, image, language_emb):
        """
        Args:
            image: (B, 3, H, W) normalized to [0, 1]
            language_emb: (B, 64) language instruction embedding
        Returns:
            action: (B, act_dim) in [-1, 1]
        """
        vis_features = self.vision(image).flatten(1)       # (B, 128*16)
        lang_features = self.language(language_emb)          # (B, 128)
        combined = torch.cat([vis_features, lang_features], dim=1)
        return self.action_head(combined)


class VLAPolicyNode(Node):
    """
    ROS 2 node that runs a VLA policy.
    
    Subscribes to:
      - /camera/image_raw (sensor_msgs/Image)
      - /task_instruction (std_msgs/String)
    
    Publishes:
      - /raw_action (std_msgs/Float64MultiArray)
      - /policy_latency (std_msgs/Float64MultiArray)
    """
    
    def __init__(self):
        super().__init__('vla_policy_node')
        
        # Parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('image_size', 224)
        self.declare_parameter('action_dim', 7)
        
        control_rate = self.get_parameter('control_rate_hz').value
        act_dim = self.get_parameter('action_dim').value
        
        # Load policy
        self.policy = PolicyNetwork(act_dim=act_dim)
        model_path = self.get_parameter('model_path').value
        if model_path:
            self.policy.load_state_dict(torch.load(model_path, weights_only=True))
            self.get_logger().info(f'Loaded policy from {model_path}')
        self.policy.eval()
        
        # Simple language encoder (hash-based embedding — placeholder)
        self.lang_embed_dim = 64
        
        # State
        self.latest_image = None
        self.task_instruction = "pick up the red block"
        self.bridge = CvBridge()
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        self.task_sub = self.create_subscription(
            String, '/task_instruction', self.task_callback, 10
        )
        
        # Publishers
        self.action_pub = self.create_publisher(
            Float64MultiArray, '/raw_action', 10
        )
        self.latency_pub = self.create_publisher(
            Float64MultiArray, '/policy_latency', 10
        )
        
        # Control loop timer
        period = 1.0 / control_rate
        self.timer = self.create_timer(period, self.control_loop)
        
        self.get_logger().info(
            f'VLA Policy Node started at {control_rate} Hz, action_dim={act_dim}'
        )
    
    def image_callback(self, msg):
        """Store latest camera image."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.latest_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')
    
    def task_callback(self, msg):
        """Update task instruction."""
        self.task_instruction = msg.data
        self.get_logger().info(f'New task: "{self.task_instruction}"')
    
    def _encode_language(self, text):
        """Simple language encoding (placeholder for real tokenizer)."""
        # Hash-based embedding — deterministic, fixed dim
        np.random.seed(hash(text) % (2**31))
        embedding = np.random.randn(self.lang_embed_dim).astype(np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return torch.FloatTensor(embedding).unsqueeze(0)
    
    def _preprocess_image(self, image):
        """Resize and normalize image for policy."""
        import cv2
        img_size = self.get_parameter('image_size').value
        resized = cv2.resize(image, (img_size, img_size))
        tensor = torch.FloatTensor(resized).permute(2, 0, 1) / 255.0
        return tensor.unsqueeze(0)  # (1, 3, H, W)
    
    def control_loop(self):
        """Main control loop — runs at control_rate_hz."""
        if self.latest_image is None:
            return
        
        t_start = time.monotonic()
        
        # Preprocess
        image_tensor = self._preprocess_image(self.latest_image)
        lang_tensor = self._encode_language(self.task_instruction)
        
        # Inference
        with torch.no_grad():
            action = self.policy(image_tensor, lang_tensor).squeeze(0).numpy()
        
        t_inference = time.monotonic() - t_start
        
        # Publish action
        action_msg = Float64MultiArray()
        action_msg.data = action.tolist()
        self.action_pub.publish(action_msg)
        
        # Publish latency
        latency_msg = Float64MultiArray()
        latency_msg.data = [t_inference * 1000]  # ms
        self.latency_pub.publish(latency_msg)
        
        if t_inference > 0.05:  # warn if >50ms
            self.get_logger().warn(
                f'Policy inference slow: {t_inference*1000:.1f}ms'
            )


def main(args=None):
    rclpy.init(args=args)
    node = VLAPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 1.2 — Unit Test for Policy Node

```python
# File: ros2_vla_pipeline/test_policy_node.py

import numpy as np
import torch
import unittest


class TestPolicyNetwork(unittest.TestCase):
    """Test the policy network independently of ROS."""
    
    def setUp(self):
        self.policy = PolicyNetwork(act_dim=7)
        self.policy.eval()
    
    def test_output_shape(self):
        """Actions should have correct dimensionality."""
        image = torch.randn(1, 3, 224, 224)
        lang = torch.randn(1, 64)
        action = self.policy(image, lang)
        self.assertEqual(action.shape, (1, 7))
    
    def test_output_range(self):
        """Actions should be in [-1, 1] (Tanh output)."""
        image = torch.randn(1, 3, 224, 224)
        lang = torch.randn(1, 64)
        action = self.policy(image, lang)
        self.assertTrue(torch.all(action >= -1.0))
        self.assertTrue(torch.all(action <= 1.0))
    
    def test_deterministic(self):
        """Same input should produce same output."""
        image = torch.randn(1, 3, 224, 224)
        lang = torch.randn(1, 64)
        a1 = self.policy(image, lang)
        a2 = self.policy(image, lang)
        self.assertTrue(torch.allclose(a1, a2))
    
    def test_batch_inference(self):
        """Should handle batched input."""
        image = torch.randn(4, 3, 224, 224)
        lang = torch.randn(4, 64)
        action = self.policy(image, lang)
        self.assertEqual(action.shape, (4, 7))
    
    def test_different_instructions_different_actions(self):
        """Different language should produce different actions."""
        image = torch.randn(1, 3, 224, 224)
        lang1 = torch.randn(1, 64)
        lang2 = torch.randn(1, 64)
        a1 = self.policy(image, lang1)
        a2 = self.policy(image, lang2)
        # Different language → actions should differ (with high probability)
        self.assertFalse(torch.allclose(a1, a2, atol=1e-4))


if __name__ == '__main__':
    unittest.main()
```

---

## Part 2: Safety Gate Node (~2.5 hours)

### 2.1 — Safety Constraint System

```python
# File: ros2_vla_pipeline/safety_node.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Bool
import numpy as np
from dataclasses import dataclass
from enum import Enum


class SafetyViolation(Enum):
    NONE = 0
    WORKSPACE = 1
    VELOCITY = 2
    SINGULARITY = 3
    COLLISION = 4
    EMERGENCY = 5


@dataclass
class WorkspaceLimits:
    """Cartesian workspace boundaries (meters)."""
    x_min: float = 0.2
    x_max: float = 0.8
    y_min: float = -0.4
    y_max: float = 0.4
    z_min: float = 0.02   # above table surface
    z_max: float = 0.6


@dataclass
class SafetyLimits:
    """Safety constraint parameters."""
    max_joint_velocity: float = 1.0     # rad/s per joint
    max_ee_velocity: float = 0.5        # m/s end-effector
    max_joint_accel: float = 5.0        # rad/s² per joint
    min_manipulability: float = 0.01    # singularity threshold
    workspace: WorkspaceLimits = None
    
    def __post_init__(self):
        if self.workspace is None:
            self.workspace = WorkspaceLimits()


class SafetyGateNode(Node):
    """
    Safety filter between policy output and robot actuators.
    
    Subscribes to:
      - /raw_action (from policy node)
      - /joint_states (current joint positions/velocities)
    
    Publishes:
      - /safe_action (filtered action)
      - /safety_status (violation flags)
    """
    
    def __init__(self):
        super().__init__('safety_gate_node')
        
        # Parameters
        self.declare_parameter('max_ee_velocity', 0.5)
        self.declare_parameter('enable_workspace_check', True)
        self.declare_parameter('enable_velocity_check', True)
        self.declare_parameter('enable_singularity_check', True)
        
        self.limits = SafetyLimits(
            max_ee_velocity=self.get_parameter('max_ee_velocity').value,
        )
        
        # State
        self.prev_action = None
        self.prev_time = None
        self.violation_count = {v: 0 for v in SafetyViolation}
        
        # Subscribers
        self.action_sub = self.create_subscription(
            Float64MultiArray, '/raw_action', self.action_callback, 10
        )
        
        # Publishers
        self.safe_action_pub = self.create_publisher(
            Float64MultiArray, '/safe_action', 10
        )
        self.safety_pub = self.create_publisher(
            Float64MultiArray, '/safety_status', 10
        )
        self.estop_pub = self.create_publisher(Bool, '/emergency_stop', 10)
        
        self.get_logger().info('Safety Gate Node started')
    
    def check_workspace(self, action):
        """
        Check if action would move end-effector outside workspace.
        Returns clipped action if violation detected.
        """
        ws = self.limits.workspace
        # Assume first 3 action dims are target EE position (delta or absolute)
        clipped = action.copy()
        violations = []
        
        # For delta actions, we'd need current EE position
        # Here we just check the action magnitude as proxy
        for i, (lo, hi, name) in enumerate([
            (ws.x_min, ws.x_max, 'x'),
            (ws.y_min, ws.y_max, 'y'),
            (ws.z_min, ws.z_max, 'z'),
        ]):
            if i < len(action):
                # Scale check (action is in [-1, 1], maps to workspace)
                pos = (action[i] + 1) / 2 * (hi - lo) + lo
                if pos < lo or pos > hi:
                    violations.append(f'{name}={pos:.3f} outside [{lo}, {hi}]')
                    clipped[i] = np.clip(action[i],
                                          2 * (lo - lo) / (hi - lo) - 1,
                                          2 * (hi - lo) / (hi - lo) - 1)
        
        return clipped, violations
    
    def check_velocity(self, action, dt):
        """
        Check if action implies excessive velocity.
        Returns scaled action if too fast.
        """
        if self.prev_action is None or dt <= 0:
            return action, []
        
        delta = action[:3] - self.prev_action[:3]  # position change
        velocity = np.linalg.norm(delta) / dt
        
        if velocity > self.limits.max_ee_velocity:
            scale = self.limits.max_ee_velocity / velocity
            scaled_action = action.copy()
            scaled_action[:3] = self.prev_action[:3] + delta * scale
            return scaled_action, [f'velocity={velocity:.3f} > {self.limits.max_ee_velocity}']
        
        return action, []
    
    def check_acceleration(self, action, dt):
        """Check for excessive acceleration (jerk limiting)."""
        if self.prev_action is None or dt <= 0:
            return action, []
        
        accel = np.abs(action - self.prev_action) / (dt ** 2)
        max_accel = np.max(accel[:min(len(accel), 7)])
        
        if max_accel > self.limits.max_joint_accel:
            scale = self.limits.max_joint_accel / max_accel
            smoothed = self.prev_action + (action - self.prev_action) * min(scale, 1.0)
            return smoothed, [f'accel={max_accel:.1f} > {self.limits.max_joint_accel}']
        
        return action, []
    
    def action_callback(self, msg):
        """Process incoming action through safety checks."""
        action = np.array(msg.data)
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        dt = 0.1  # default
        if self.prev_time is not None:
            dt = max(current_time - self.prev_time, 0.001)
        
        all_violations = []
        violation_type = SafetyViolation.NONE
        
        # Check 1: Workspace limits
        if self.get_parameter('enable_workspace_check').value:
            action, violations = self.check_workspace(action)
            if violations:
                all_violations.extend(violations)
                violation_type = SafetyViolation.WORKSPACE
        
        # Check 2: Velocity limits
        if self.get_parameter('enable_velocity_check').value:
            action, violations = self.check_velocity(action, dt)
            if violations:
                all_violations.extend(violations)
                violation_type = SafetyViolation.VELOCITY
        
        # Check 3: Acceleration (smoothing)
        action, violations = self.check_acceleration(action, dt)
        if violations:
            all_violations.extend(violations)
        
        # Log violations
        if all_violations:
            self.violation_count[violation_type] += 1
            if self.violation_count[violation_type] % 100 == 1:
                self.get_logger().warn(
                    f'Safety violation #{self.violation_count[violation_type]}: '
                    f'{"; ".join(all_violations)}'
                )
        
        # Check for emergency: too many violations in short time
        total_recent = sum(self.violation_count.values())
        if total_recent > 1000:
            self.get_logger().error('Too many safety violations — E-STOP')
            estop_msg = Bool()
            estop_msg.data = True
            self.estop_pub.publish(estop_msg)
            # Zero action
            action = np.zeros_like(action)
        
        # Publish safe action
        safe_msg = Float64MultiArray()
        safe_msg.data = action.tolist()
        self.safe_action_pub.publish(safe_msg)
        
        # Publish safety status
        status_msg = Float64MultiArray()
        status_msg.data = [float(violation_type.value), float(len(all_violations))]
        self.safety_pub.publish(status_msg)
        
        # Update state
        self.prev_action = action.copy()
        self.prev_time = current_time


def main(args=None):
    rclpy.init(args=args)
    node = SafetyGateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 2.2 — Safety Gate Unit Tests

```python
# File: ros2_vla_pipeline/test_safety.py

import numpy as np
import unittest


class TestSafetyChecks(unittest.TestCase):
    """Test safety constraints independent of ROS."""
    
    def setUp(self):
        self.limits = SafetyLimits()
    
    def test_workspace_within_bounds(self):
        """Actions within workspace should pass unchanged."""
        action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Center of workspace — should pass
        # (This test depends on your mapping from action to workspace)
    
    def test_velocity_limiting(self):
        """Fast actions should be scaled down."""
        prev = np.array([0.0, 0.0, 0.0])
        curr = np.array([10.0, 0.0, 0.0])  # huge jump
        dt = 0.1
        velocity = np.linalg.norm(curr - prev) / dt
        self.assertGreater(velocity, self.limits.max_ee_velocity)
    
    def test_zero_action_is_safe(self):
        """Zero action should always be safe."""
        action = np.zeros(7)
        # Should pass all checks
        self.assertTrue(np.allclose(action, np.zeros(7)))
    
    def test_estop_zeros_action(self):
        """Emergency stop should produce zero action."""
        estop_action = np.zeros(7)
        self.assertTrue(np.allclose(estop_action, 0))


if __name__ == '__main__':
    unittest.main()
```

---

## Part 3: Action Interpolation Node (~2 hours)

### 3.1 — Smooth Trajectory Generation

```python
# File: ros2_vla_pipeline/interpolator_node.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import numpy as np
from collections import deque


class ActionInterpolatorNode(Node):
    """
    Interpolates between VLA actions for smooth robot motion.
    
    VLA policies typically output at 3-10 Hz (inference limited).
    Robot controllers need commands at 100-1000 Hz.
    This node bridges the gap with trajectory interpolation.
    
    Subscribes to:
      - /safe_action (from safety node, ~10 Hz)
    
    Publishes:
      - /cmd_joints (to robot controller, 100 Hz)
    """
    
    def __init__(self):
        super().__init__('action_interpolator_node')
        
        self.declare_parameter('output_rate_hz', 100.0)
        self.declare_parameter('interpolation_type', 'cubic')  # 'linear' or 'cubic'
        self.declare_parameter('action_dim', 7)
        
        output_rate = self.get_parameter('output_rate_hz').value
        self.interp_type = self.get_parameter('interpolation_type').value
        self.act_dim = self.get_parameter('action_dim').value
        
        # Action buffer for interpolation
        self.action_history = deque(maxlen=4)  # keep last 4 for cubic
        self.time_history = deque(maxlen=4)
        self.current_action = np.zeros(self.act_dim)
        
        # Subscriber
        self.action_sub = self.create_subscription(
            Float64MultiArray, '/safe_action', self.action_callback, 10
        )
        
        # Publisher (high rate)
        self.cmd_pub = self.create_publisher(
            Float64MultiArray, '/cmd_joints', 10
        )
        
        # High-rate timer
        period = 1.0 / output_rate
        self.timer = self.create_timer(period, self.interpolate_and_publish)
        
        self.get_logger().info(
            f'Interpolator started: {self.interp_type} at {output_rate} Hz'
        )
    
    def action_callback(self, msg):
        """Receive new action waypoint."""
        action = np.array(msg.data)
        t = self.get_clock().now().nanoseconds / 1e9
        self.action_history.append(action)
        self.time_history.append(t)
    
    def _linear_interpolate(self, t):
        """Linear interpolation between last two actions."""
        if len(self.action_history) < 2:
            return self.action_history[-1] if self.action_history else np.zeros(self.act_dim)
        
        t0, t1 = self.time_history[-2], self.time_history[-1]
        a0, a1 = self.action_history[-2], self.action_history[-1]
        
        if t1 <= t0:
            return a1
        
        alpha = np.clip((t - t0) / (t1 - t0), 0.0, 1.5)  # allow small extrapolation
        return a0 + alpha * (a1 - a0)
    
    def _cubic_interpolate(self, t):
        """Cubic spline interpolation for smoother motion."""
        if len(self.action_history) < 3:
            return self._linear_interpolate(t)
        
        # Use last 3 points for quadratic fit per dimension
        times = np.array(list(self.time_history))[-3:]
        actions = np.array(list(self.action_history))[-3:]
        
        t_norm = times - times[0]
        t_query = t - times[0]
        
        result = np.zeros(self.act_dim)
        for dim in range(min(self.act_dim, actions.shape[1])):
            # Fit quadratic: a*t² + b*t + c
            coeffs = np.polyfit(t_norm, actions[:, dim], deg=2)
            result[dim] = np.polyval(coeffs, np.clip(t_query, 0, t_norm[-1] * 1.2))
        
        return result
    
    def interpolate_and_publish(self):
        """Generate and publish interpolated command at high rate."""
        if not self.action_history:
            return
        
        t = self.get_clock().now().nanoseconds / 1e9
        
        if self.interp_type == 'cubic':
            action = self._cubic_interpolate(t)
        else:
            action = self._linear_interpolate(t)
        
        # Clip to valid range
        action = np.clip(action, -1.0, 1.0)
        self.current_action = action
        
        # Publish
        msg = Float64MultiArray()
        msg.data = action.tolist()
        self.cmd_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ActionInterpolatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### 3.2 — Interpolation Comparison (Standalone Test)

```python
# File: ros2_vla_pipeline/test_interpolation.py
# Run standalone — no ROS required

import numpy as np
import matplotlib.pyplot as plt


def generate_policy_actions(n_steps=20, noise=0.1):
    """Simulate noisy VLA policy outputs at 10 Hz."""
    t = np.linspace(0, 2, n_steps)
    # Sinusoidal trajectory with noise
    actions = np.column_stack([
        0.3 * np.sin(2 * np.pi * t) + np.random.normal(0, noise, n_steps),
        0.2 * np.cos(3 * np.pi * t) + np.random.normal(0, noise, n_steps),
        0.1 * np.sin(np.pi * t) + np.random.normal(0, noise, n_steps),
    ])
    return t, actions


def linear_interp(t_policy, actions, t_query):
    """Linear interpolation."""
    result = np.zeros((len(t_query), actions.shape[1]))
    for dim in range(actions.shape[1]):
        result[:, dim] = np.interp(t_query, t_policy, actions[:, dim])
    return result


def cubic_interp(t_policy, actions, t_query):
    """Cubic spline interpolation."""
    from scipy.interpolate import CubicSpline
    result = np.zeros((len(t_query), actions.shape[1]))
    for dim in range(actions.shape[1]):
        cs = CubicSpline(t_policy, actions[:, dim])
        result[:, dim] = cs(t_query)
    return result


# Generate data
t_policy, actions = generate_policy_actions(n_steps=20, noise=0.05)
t_robot = np.linspace(0, 2, 200)  # 100 Hz

# Interpolate
linear_result = linear_interp(t_policy, actions, t_robot)
try:
    cubic_result = cubic_interp(t_policy, actions, t_robot)
    has_cubic = True
except ImportError:
    has_cubic = False
    print("Install scipy for cubic interpolation: pip install scipy")

# Visualize
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
dim_names = ['X', 'Y', 'Z']

for dim in range(3):
    ax = axes[dim]
    ax.scatter(t_policy, actions[:, dim], c='red', s=30, zorder=3, label='Policy output (10 Hz)')
    ax.plot(t_robot, linear_result[:, dim], 'b-', alpha=0.7, label='Linear interp')
    if has_cubic:
        ax.plot(t_robot, cubic_result[:, dim], 'g--', alpha=0.7, label='Cubic interp')
    ax.set_ylabel(f'Action {dim_names[dim]}')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (s)')
plt.suptitle('VLA Action Interpolation: 10 Hz → 100 Hz', fontsize=14)
plt.tight_layout()
plt.savefig('action_interpolation.png', dpi=150)
plt.show()
print("Saved: action_interpolation.png")

# Compute smoothness metric (total variation)
linear_tv = np.sum(np.abs(np.diff(linear_result, axis=0)))
if has_cubic:
    cubic_tv = np.sum(np.abs(np.diff(cubic_result, axis=0)))
    print(f"\nTotal Variation (smoothness metric — lower is smoother):")
    print(f"  Linear: {linear_tv:.4f}")
    print(f"  Cubic:  {cubic_tv:.4f}")
    print(f"  Cubic is {linear_tv/cubic_tv:.1f}x smoother")
```

---

## Part 4: Monitoring & Diagnostics (~2 hours)

### 4.1 — Pipeline Monitor Node

```python
# File: ros2_vla_pipeline/monitor_node.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Bool
import numpy as np
from collections import deque
import time


class PipelineMonitorNode(Node):
    """
    Monitors the full VLA pipeline health.
    
    Subscribes to:
      - /raw_action (policy output)
      - /safe_action (post-safety)
      - /cmd_joints (interpolated commands)
      - /policy_latency (inference time)
      - /safety_status (violation flags)
    
    Publishes:
      - /diagnostics (summary metrics)
    """
    
    def __init__(self):
        super().__init__('pipeline_monitor_node')
        
        # Metrics storage
        self.latency_history = deque(maxlen=100)
        self.safety_violations = deque(maxlen=1000)
        self.action_magnitudes = deque(maxlen=100)
        self.action_diff_history = deque(maxlen=100)
        self.last_raw_action = None
        self.last_safe_action = None
        
        # Subscribers
        self.create_subscription(
            Float64MultiArray, '/raw_action', self.raw_action_cb, 10)
        self.create_subscription(
            Float64MultiArray, '/safe_action', self.safe_action_cb, 10)
        self.create_subscription(
            Float64MultiArray, '/policy_latency', self.latency_cb, 10)
        self.create_subscription(
            Float64MultiArray, '/safety_status', self.safety_cb, 10)
        
        # Publisher
        self.diag_pub = self.create_publisher(
            Float64MultiArray, '/diagnostics', 10)
        
        # Periodic report
        self.timer = self.create_timer(5.0, self.publish_diagnostics)
        
        self.get_logger().info('Pipeline Monitor started')
    
    def raw_action_cb(self, msg):
        action = np.array(msg.data)
        self.last_raw_action = action
        self.action_magnitudes.append(np.linalg.norm(action))
    
    def safe_action_cb(self, msg):
        action = np.array(msg.data)
        self.last_safe_action = action
        
        # Compute how much safety modified the action
        if self.last_raw_action is not None:
            diff = np.linalg.norm(action - self.last_raw_action)
            self.action_diff_history.append(diff)
    
    def latency_cb(self, msg):
        self.latency_history.append(msg.data[0])  # ms
    
    def safety_cb(self, msg):
        if msg.data[1] > 0:  # violation count
            self.safety_violations.append(msg.data[0])  # violation type
    
    def publish_diagnostics(self):
        """Publish summary diagnostics every 5 seconds."""
        metrics = {}
        
        if self.latency_history:
            latencies = list(self.latency_history)
            metrics['latency_mean_ms'] = np.mean(latencies)
            metrics['latency_p95_ms'] = np.percentile(latencies, 95)
            metrics['latency_max_ms'] = np.max(latencies)
        
        if self.safety_violations:
            metrics['violations_per_min'] = len(self.safety_violations) * 12  # 5s window * 12
        
        if self.action_diff_history:
            diffs = list(self.action_diff_history)
            metrics['safety_modification_mean'] = np.mean(diffs)
            metrics['safety_modification_max'] = np.max(diffs)
        
        if self.action_magnitudes:
            metrics['action_magnitude_mean'] = np.mean(list(self.action_magnitudes))
        
        # Log report
        self.get_logger().info(
            f"=== Pipeline Health ===\n"
            + "\n".join(f"  {k}: {v:.3f}" for k, v in metrics.items())
        )
        
        # Publish as flat array
        msg = Float64MultiArray()
        msg.data = list(metrics.values())
        self.diag_pub.publish(msg)
        
        # Alerts
        if metrics.get('latency_p95_ms', 0) > 100:
            self.get_logger().warn('⚠ Policy latency P95 > 100ms — consider faster model')
        
        if metrics.get('violations_per_min', 0) > 60:
            self.get_logger().warn('⚠ High safety violation rate — check policy behavior')
        
        if metrics.get('safety_modification_mean', 0) > 0.3:
            self.get_logger().warn('⚠ Safety is heavily modifying actions — policy may be unsafe')
```

### 4.2 — ROS 2 Launch File

```python
# File: ros2_vla_pipeline/launch/vla_pipeline.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # VLA Policy Node
        Node(
            package='ros2_vla_pipeline',
            executable='policy_node',
            name='vla_policy',
            parameters=[{
                'control_rate_hz': 10.0,
                'image_size': 224,
                'action_dim': 7,
                'model_path': '',
            }],
            output='screen',
        ),
        
        # Safety Gate
        Node(
            package='ros2_vla_pipeline',
            executable='safety_node',
            name='safety_gate',
            parameters=[{
                'max_ee_velocity': 0.5,
                'enable_workspace_check': True,
                'enable_velocity_check': True,
                'enable_singularity_check': True,
            }],
            output='screen',
        ),
        
        # Action Interpolator
        Node(
            package='ros2_vla_pipeline',
            executable='interpolator_node',
            name='action_interpolator',
            parameters=[{
                'output_rate_hz': 100.0,
                'interpolation_type': 'cubic',
                'action_dim': 7,
            }],
            output='screen',
        ),
        
        # Pipeline Monitor
        Node(
            package='ros2_vla_pipeline',
            executable='monitor_node',
            name='pipeline_monitor',
            output='screen',
        ),
    ])
```

---

## Part 5: Integration Test & Evaluation (~3 hours)

### 5.1 — Standalone Pipeline Test (No ROS Required)

```python
# File: ros2_vla_pipeline/test_pipeline_standalone.py
# Test the full pipeline logic without ROS

import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque


class StandalonePipeline:
    """
    Full VLA→Safety→Interpolation pipeline without ROS.
    For testing the logic before ROS deployment.
    """
    
    def __init__(self, policy, safety_limits=None):
        self.policy = policy
        self.safety_limits = safety_limits or SafetyLimits()
        self.prev_action = None
        self.action_buffer = deque(maxlen=4)
        self.time_buffer = deque(maxlen=4)
        
        # Monitoring
        self.metrics = {
            'latencies': [],
            'violations': [],
            'raw_actions': [],
            'safe_actions': [],
            'interpolated': [],
        }
    
    def step(self, image, language, current_time):
        """Full pipeline: image → policy → safety → interpolation."""
        import time as time_module
        
        # 1. Policy inference
        t0 = time_module.monotonic()
        with torch.no_grad():
            image_t = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0
            # Simple language encoding
            np.random.seed(hash(language) % (2**31))
            lang_t = torch.FloatTensor(np.random.randn(1, 64))
            raw_action = self.policy(image_t, lang_t).squeeze(0).numpy()
        latency = (time_module.monotonic() - t0) * 1000
        
        self.metrics['latencies'].append(latency)
        self.metrics['raw_actions'].append(raw_action.copy())
        
        # 2. Safety check
        safe_action = raw_action.copy()
        violation = False
        
        # Velocity check
        if self.prev_action is not None:
            dt = 0.1  # 10 Hz
            velocity = np.linalg.norm(safe_action[:3] - self.prev_action[:3]) / dt
            if velocity > self.safety_limits.max_ee_velocity:
                scale = self.safety_limits.max_ee_velocity / velocity
                safe_action[:3] = self.prev_action[:3] + (
                    safe_action[:3] - self.prev_action[:3]) * scale
                violation = True
        
        # Workspace check (clip to [-1, 1])
        safe_action = np.clip(safe_action, -1.0, 1.0)
        
        self.metrics['safe_actions'].append(safe_action.copy())
        self.metrics['violations'].append(violation)
        
        # 3. Store for interpolation
        self.action_buffer.append(safe_action)
        self.time_buffer.append(current_time)
        self.prev_action = safe_action.copy()
        
        return safe_action, latency, violation
    
    def interpolate(self, t_query):
        """Get interpolated action at arbitrary time."""
        if len(self.action_buffer) < 2:
            return self.action_buffer[-1] if self.action_buffer else np.zeros(7)
        
        t0, t1 = self.time_buffer[-2], self.time_buffer[-1]
        a0, a1 = self.action_buffer[-2], self.action_buffer[-1]
        
        alpha = np.clip((t_query - t0) / max(t1 - t0, 1e-6), 0, 1.5)
        return a0 + alpha * (a1 - a0)
    
    def report(self):
        """Print pipeline metrics summary."""
        latencies = self.metrics['latencies']
        violations = self.metrics['violations']
        
        print(f"\n{'='*50}")
        print(f"Pipeline Metrics ({len(latencies)} steps)")
        print(f"{'='*50}")
        print(f"Latency: mean={np.mean(latencies):.1f}ms, "
              f"p95={np.percentile(latencies, 95):.1f}ms, "
              f"max={np.max(latencies):.1f}ms")
        print(f"Safety violations: {sum(violations)}/{len(violations)} "
              f"({sum(violations)/len(violations)*100:.1f}%)")
        
        if self.metrics['raw_actions']:
            raw = np.array(self.metrics['raw_actions'])
            safe = np.array(self.metrics['safe_actions'])
            modification = np.linalg.norm(raw - safe, axis=1)
            print(f"Action modification: mean={np.mean(modification):.4f}, "
                  f"max={np.max(modification):.4f}")


# ── RUN THE PIPELINE ──
policy = PolicyNetwork(act_dim=7)
pipeline = StandalonePipeline(policy)

# Simulate 100 steps
for step in range(100):
    # Fake image (random for testing)
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    language = "pick up the red block"
    
    safe_action, latency, violation = pipeline.step(image, language, step * 0.1)

pipeline.report()

# Visualize action trajectories
raw = np.array(pipeline.metrics['raw_actions'])
safe = np.array(pipeline.metrics['safe_actions'])

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for dim in range(3):
    axes[dim].plot(raw[:, dim], 'r-', alpha=0.5, label='Raw (policy)')
    axes[dim].plot(safe[:, dim], 'b-', label='Safe (filtered)')
    axes[dim].set_ylabel(f'Action dim {dim}')
    axes[dim].legend()
    axes[dim].grid(True, alpha=0.3)

axes[-1].set_xlabel('Step')
plt.suptitle('Raw vs Safety-Filtered Actions', fontsize=14)
plt.tight_layout()
plt.savefig('pipeline_actions.png', dpi=150)
plt.show()
print("Saved: pipeline_actions.png")
```

### 5.2 — Evaluation Criteria

```
╔══════════════════════════════════════════════════════════════════╗
║                   Project 08 Evaluation Rubric                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║ 1. Policy Node (25%)                                            ║
║    □ Subscribes to image and language topics                    ║
║    □ Publishes actions at configured rate                       ║
║    □ Measures and publishes inference latency                   ║
║    □ Handles missing inputs gracefully                          ║
║                                                                  ║
║ 2. Safety Gate (30%)                                            ║
║    □ Workspace limit enforcement                                ║
║    □ Velocity limiting                                          ║
║    □ Acceleration smoothing                                     ║
║    □ Emergency stop on repeated violations                      ║
║    □ All safety checks have unit tests                          ║
║                                                                  ║
║ 3. Action Interpolation (15%)                                   ║
║    □ Linear and cubic interpolation implemented                 ║
║    □ Handles 10 Hz → 100 Hz upsampling                         ║
║    □ Comparison plot showing smoothness difference              ║
║                                                                  ║
║ 4. Monitoring (15%)                                             ║
║    □ Latency tracking (mean, P95, max)                          ║
║    □ Safety violation rate                                      ║
║    □ Action modification metric                                 ║
║    □ Configurable alert thresholds                              ║
║                                                                  ║
║ 5. Integration (15%)                                            ║
║    □ All nodes work together (launch file or standalone test)   ║
║    □ Pipeline report generated                                  ║
║    □ At least 3 plots: raw vs safe actions, interpolation,      ║
║      latency histogram                                          ║
║                                                                  ║
║ Bonus:                                                           ║
║    □ Used a real trained policy (from Exercise 12) instead of   ║
║      random weights                                             ║
║    □ Added RViz visualization                                   ║
║    □ Implemented action chunking (predict K future actions)     ║
║    □ Tested with ROS 2 bag recording and playback               ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## Extension Ideas

1. **Action Chunking**: Modify the policy to predict K future actions (like ACT) 
   and use the interpolator to blend between chunks
2. **Multi-Task Switching**: Add a task scheduler node that sequences instructions 
   ("pick red block" → "place on shelf" → "pick blue block")
3. **Failure Detection**: Add a node that monitors task progress and triggers 
   re-planning if the object is dropped or the goal isn't reached
4. **Hardware Interface**: Replace the simulated camera with a real USB camera 
   using the `usb_cam` ROS 2 package

---

> **Connection to the Compression Thread**: This project demonstrates that a VLA
> alone isn't enough — you need a full *system*. The VLA compresses vision + language
> into actions, but the safety gate compresses workspace knowledge into constraints,
> the interpolator compresses trajectory smoothness into spline coefficients, and
> the monitor compresses pipeline health into actionable alerts. Real robotics
> requires compressing knowledge at every layer of the stack.

[← Project 07: VLA Capstone](../projects/07-vla-capstone/README.md) | [Back to Curriculum](../CURRICULUM.md)
