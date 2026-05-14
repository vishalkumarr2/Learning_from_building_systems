# Exercise 17 — VLA Systems Engineering

> **Prerequisites**: Exercises 11 (VLA Experiments), 12 (Robot Manipulation), 15 (Kinematics & Trajectory), 16 (Safety)  
> **Time**: 6–8 hours  
> **Theme**: Taking a VLA from "works in a notebook" to "runs a real robot at 10 Hz"

---

## Learning Objectives

After this exercise you will be able to:

1. Profile end-to-end VLA inference latency and identify bottlenecks
2. Implement action chunking with temporal ensembling in a real-time control loop
3. Design a complete inference pipeline with image preprocessing, model inference, and action execution
4. Handle failure modes: network latency, prediction stalls, safety stops
5. Implement model serving patterns (batching, quantization, TensorRT)

---

## Exercise 1: Inference Pipeline Profiling

**Goal**: Measure where time goes in a VLA prediction cycle and determine
if your system can meet real-time deadlines.

### 1.1 — The Real-Time Budget

```python
"""
VLA real-time control budget analysis.

A robot arm typically needs new commands at:
  - 10 Hz (100ms budget) — minimum for smooth motion
  - 20 Hz (50ms budget) — good for dynamic tasks
  - 50 Hz (20ms budget) — high-precision manipulation

Where does time go in a VLA inference step?
  1. Image capture & preprocessing: 5-15ms
  2. Image encoding (ViT): 10-30ms
  3. Language encoding (cached after first step): 0ms (amortized)
  4. Action generation (LLM decode): 20-80ms
  5. Action post-processing & safety checks: 1-5ms
  6. Communication to robot controller: 1-5ms
  
Total: 37-135ms → only fits 10Hz for smaller models!
"""

import time
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class LatencyProfile:
    """Timing breakdown for one VLA inference step."""
    image_preprocess_ms: float
    vision_encode_ms: float
    language_encode_ms: float
    action_decode_ms: float
    postprocess_ms: float
    communication_ms: float
    
    @property
    def total_ms(self) -> float:
        return (self.image_preprocess_ms + self.vision_encode_ms + 
                self.language_encode_ms + self.action_decode_ms + 
                self.postprocess_ms + self.communication_ms)
    
    @property
    def fits_10hz(self) -> bool:
        return self.total_ms < 100.0
    
    @property
    def max_control_rate_hz(self) -> float:
        return 1000.0 / self.total_ms if self.total_ms > 0 else float('inf')
    
    def bottleneck(self) -> str:
        times = {
            "image_preprocess": self.image_preprocess_ms,
            "vision_encode": self.vision_encode_ms,
            "language_encode": self.language_encode_ms,
            "action_decode": self.action_decode_ms,
            "postprocess": self.postprocess_ms,
            "communication": self.communication_ms,
        }
        return max(times, key=times.get)


class VLAInferencePipeline:
    """Complete VLA inference pipeline with profiling."""
    
    def __init__(self, vision_encoder, language_encoder, action_decoder,
                 image_size: tuple = (224, 224), device: str = "cuda"):
        self.vision_encoder = vision_encoder
        self.language_encoder = language_encoder
        self.action_decoder = action_decoder
        self.image_size = image_size
        self.device = device
        
        # Cached language features (recomputed only when instruction changes)
        self._cached_lang_features: Optional[torch.Tensor] = None
        self._cached_instruction: Optional[str] = None
    
    def preprocess_image(self, raw_image: np.ndarray) -> torch.Tensor:
        """Resize, normalize, and move to GPU."""
        # Simulate realistic preprocessing
        import torchvision.transforms.functional as TF
        from PIL import Image
        
        img = Image.fromarray(raw_image)
        img = TF.resize(img, self.image_size)
        tensor = TF.to_tensor(img)
        tensor = TF.normalize(tensor, mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
        return tensor.unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def predict_action(self, raw_image: np.ndarray, instruction: str,
                       profile: bool = True) -> tuple[np.ndarray, Optional[LatencyProfile]]:
        """Full prediction pipeline with optional timing.
        
        Returns:
            action: (action_dim,) numpy array
            profile: LatencyProfile if profiling enabled
        """
        timings = {}
        
        # 1. Image preprocessing
        t0 = time.perf_counter()
        image_tensor = self.preprocess_image(raw_image)
        timings["image_preprocess"] = (time.perf_counter() - t0) * 1000
        
        # 2. Vision encoding
        t0 = time.perf_counter()
        if self.device == "cuda":
            torch.cuda.synchronize()
        vision_features = self.vision_encoder(image_tensor)
        if self.device == "cuda":
            torch.cuda.synchronize()
        timings["vision_encode"] = (time.perf_counter() - t0) * 1000
        
        # 3. Language encoding (with caching)
        t0 = time.perf_counter()
        if instruction != self._cached_instruction:
            # Only re-encode if instruction changed
            self._cached_lang_features = self.language_encoder(instruction)
            self._cached_instruction = instruction
            timings["language_encode"] = (time.perf_counter() - t0) * 1000
        else:
            timings["language_encode"] = 0.0
        
        # 4. Action decoding
        t0 = time.perf_counter()
        if self.device == "cuda":
            torch.cuda.synchronize()
        action_tokens = self.action_decoder(vision_features, self._cached_lang_features)
        if self.device == "cuda":
            torch.cuda.synchronize()
        timings["action_decode"] = (time.perf_counter() - t0) * 1000
        
        # 5. Post-processing (detokenize, safety clamp)
        t0 = time.perf_counter()
        action = self._postprocess(action_tokens)
        timings["postprocess"] = (time.perf_counter() - t0) * 1000
        
        # 6. Communication overhead (simulated)
        timings["communication"] = 1.0  # Typical USB/ethernet latency
        
        latency = LatencyProfile(**{f"{k}_ms": v for k, v in timings.items()})
        return action, latency if profile else None
    
    def _postprocess(self, action_tokens: torch.Tensor) -> np.ndarray:
        """Detokenize and apply safety limits."""
        action = action_tokens.cpu().numpy().flatten()
        # Clamp to joint limits
        action = np.clip(action, -1.0, 1.0)
        return action


# TODO 1a: Create mock vision_encoder, language_encoder, action_decoder
# with realistic sizes (ViT-B: 86M params, Action head: small MLP)
# Run 100 inference steps and collect LatencyProfile statistics

# TODO 1b: Create a latency histogram plot showing the distribution
# of total inference time. Is it consistent or does it have long-tail spikes?

# TODO 1c: Identify the bottleneck. What optimization would help most?
# Options: quantization, smaller ViT, action caching, TensorRT compilation
```

### 1.2 — Exercises

1. Build the mock pipeline and profile 100 steps — report mean, p50, p95, p99 latencies
2. Add warmup: the first inference is always slow (CUDA kernel compilation). How many warmup steps needed?
3. Compare: run with `torch.compile()` vs without. What's the speedup?
4. **Trade-off analysis**: smaller ViT (ViT-Tiny: 5M params) vs larger (ViT-L: 300M params). How does each affect total latency?

---

## Exercise 2: Action Chunking Control Loop

**Goal**: Implement a production-quality control loop that predicts action chunks
and executes them with temporal ensembling for smooth robot motion.

### 2.1 — Theory

```
Standard approach: predict 1 action per VLA call → need 10 Hz inference
Action chunking: predict K actions at once → need only 10/K Hz inference!

Example with K=16, robot at 10 Hz:
  - VLA called every 16 steps (every 1.6 seconds)
  - But overlap by K/2 for smoother transitions
  - Temporal ensembling averages overlapping predictions
  
This is how ACT (Action Chunking with Transformers) works:
  - Predict chunks of 16–100 actions
  - Execute with exponential-weighted temporal ensembling
  - New prediction every K/2 steps (overlap for smooth blending)
```

### 2.2 — Implementation

```python
"""Production action chunking control loop with temporal ensembling."""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Callable
import time


@dataclass
class ChunkConfig:
    """Configuration for action chunking."""
    chunk_size: int = 16          # Actions predicted per VLA call
    overlap: int = 8              # Overlap between consecutive chunks
    ensemble_lambda: float = 0.01  # Temporal ensembling decay (lower = more smoothing)
    control_rate_hz: float = 10.0  # Robot command rate
    max_action_dim: int = 7       # Robot DoF


class ActionChunkController:
    """Real-time controller that manages action chunk execution.
    
    Lifecycle:
    1. Request new chunk from VLA when buffer running low
    2. Execute actions from buffer at control_rate_hz
    3. Blend overlapping chunks via temporal ensembling
    4. Handle stalls (VLA too slow) with graceful degradation
    """
    
    def __init__(self, config: ChunkConfig, vla_predict_fn: Callable):
        self.config = config
        self.vla_predict_fn = vla_predict_fn
        
        # Action buffer: stores blended actions ready for execution
        self.action_buffer: deque = deque(maxlen=config.chunk_size * 2)
        
        # Ensemble accumulator: weighted sum of overlapping predictions
        self._ensemble_weights = np.zeros(config.chunk_size)
        self._ensemble_actions = np.zeros((config.chunk_size, config.max_action_dim))
        self._ensemble_position = 0  # Current position in ensemble buffer
        
        # Timing
        self._last_predict_time = 0.0
        self._step_count = 0
        self._stall_count = 0
    
    def _request_chunk(self, observation: dict) -> np.ndarray:
        """Get a new action chunk from the VLA model.
        
        Args:
            observation: dict with 'image' and 'instruction'
            
        Returns:
            actions: (chunk_size, action_dim) array
        """
        t0 = time.perf_counter()
        actions = self.vla_predict_fn(observation)
        latency = time.perf_counter() - t0
        self._last_predict_time = latency
        
        assert actions.shape == (self.config.chunk_size, self.config.max_action_dim), \
            f"Expected ({self.config.chunk_size}, {self.config.max_action_dim}), got {actions.shape}"
        
        return actions
    
    def _blend_chunk(self, new_actions: np.ndarray):
        """Blend new chunk into ensemble buffer using exponential weighting.
        
        For each timestep t in the overlap region:
          blended[t] = (w_old * old[t] + w_new * new[t]) / (w_old + w_new)
        where w = exp(-lambda * age)
        """
        lam = self.config.ensemble_lambda
        chunk_size = self.config.chunk_size
        overlap = self.config.overlap
        
        for i in range(chunk_size):
            # Weight for this new prediction (freshest = highest weight)
            w_new = np.exp(-lam * 0)  # Age 0 for new prediction
            
            buffer_idx = self._ensemble_position + i
            if buffer_idx < len(self._ensemble_actions):
                # Existing prediction — blend
                w_old = self._ensemble_weights[buffer_idx % chunk_size]
                total_w = w_old + w_new
                self._ensemble_actions[buffer_idx % chunk_size] = (
                    w_old * self._ensemble_actions[buffer_idx % chunk_size] + 
                    w_new * new_actions[i]
                ) / total_w
                self._ensemble_weights[buffer_idx % chunk_size] = total_w
            else:
                # No overlap — just store
                self._ensemble_actions[i % chunk_size] = new_actions[i]
                self._ensemble_weights[i % chunk_size] = w_new
        
        # Push blended actions to execution buffer
        for i in range(chunk_size - overlap):
            idx = (self._ensemble_position + i) % chunk_size
            self.action_buffer.append(self._ensemble_actions[idx].copy())
        
        self._ensemble_position += chunk_size - overlap
    
    def step(self, observation: dict) -> np.ndarray:
        """Get the next action to execute.
        
        Called at control_rate_hz. Manages chunk requests and blending.
        
        Returns:
            action: (action_dim,) array to send to robot
        """
        self._step_count += 1
        
        # Request new chunk if buffer is running low
        trigger_threshold = self.config.overlap
        if len(self.action_buffer) <= trigger_threshold:
            try:
                new_chunk = self._request_chunk(observation)
                self._blend_chunk(new_chunk)
            except Exception as e:
                self._stall_count += 1
                print(f"WARNING: VLA prediction failed ({e}), using last action")
        
        # Execute from buffer
        if len(self.action_buffer) > 0:
            return self.action_buffer.popleft()
        else:
            # Stall: no actions available, hold position
            self._stall_count += 1
            return np.zeros(self.config.max_action_dim)
    
    @property
    def diagnostics(self) -> dict:
        return {
            "buffer_size": len(self.action_buffer),
            "step_count": self._step_count,
            "stall_count": self._stall_count,
            "stall_rate": self._stall_count / max(1, self._step_count),
            "last_predict_ms": self._last_predict_time * 1000,
        }


def run_control_loop(controller: ActionChunkController, 
                     env, instruction: str, max_steps: int = 300):
    """Run the action chunking control loop in an environment.
    
    Simulates real-time execution with timing constraints.
    """
    obs = env.reset()
    total_reward = 0
    actions_executed = []
    latencies = []
    
    dt = 1.0 / controller.config.control_rate_hz
    
    for step in range(max_steps):
        loop_start = time.perf_counter()
        
        observation = {"image": obs["image"], "instruction": instruction}
        action = controller.step(observation)
        
        # Execute in environment
        obs, reward, done, info = env.step(action)
        total_reward += reward
        actions_executed.append(action.copy())
        
        loop_time = time.perf_counter() - loop_start
        latencies.append(loop_time * 1000)
        
        # Real-time enforcement (in real deployment, would sleep)
        if loop_time < dt:
            pass  # In simulation, no need to wait
        else:
            print(f"WARNING: Step {step} overran budget: {loop_time*1000:.1f}ms > {dt*1000:.1f}ms")
        
        if done:
            break
    
    return {
        "total_reward": total_reward,
        "steps": len(actions_executed),
        "actions": np.array(actions_executed),
        "latencies": latencies,
        "diagnostics": controller.diagnostics,
    }


# TODO 2a: Implement a mock VLA that returns sinusoidal action chunks
# Run the control loop for 300 steps, plot:
#   - Executed actions over time (should be smooth due to ensembling)
#   - Buffer occupancy over time
#   - Histogram of per-step latencies

# TODO 2b: Ablation — compare chunk_size={4, 8, 16, 32, 64}
# For each, measure:
#   - Action smoothness (std of consecutive action differences)
#   - Average buffer occupancy
#   - Number of stalls
#   - "Freshness": average age of executed actions

# TODO 2c: Simulate VLA latency variability
# Make vla_predict_fn take random time (50-200ms, occasionally 500ms)
# Does the controller handle it gracefully? At what stall rate?

# TODO 2d: Compare execution strategies:
# Strategy A: No chunking (predict 1 action per step) 
# Strategy B: Chunk=16, execute first action only (discard rest)
# Strategy C: Chunk=16, execute all sequentially (no overlap)
# Strategy D: Chunk=16, overlap=8, temporal ensembling (our controller)
# Plot smoothness and tracking error for each
```

---

## Exercise 3: Model Serving and Optimization

**Goal**: Learn production techniques for deploying VLA models efficiently.

### 3.1 — Quantization for Deployment

```python
"""Model quantization techniques for VLA inference speed."""

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic


def compare_quantization_levels(model: nn.Module, sample_input: torch.Tensor):
    """Compare FP32, FP16, INT8, and INT4 inference.
    
    Measures:
    - Inference latency
    - Model size on disk
    - Output deviation from FP32 reference
    """
    results = {}
    
    # FP32 reference
    with torch.no_grad():
        ref_output = model(sample_input)
    
    # FP16
    model_fp16 = model.half()
    input_fp16 = sample_input.half()
    with torch.no_grad():
        t0 = time.perf_counter()
        out_fp16 = model_fp16(input_fp16)
        results["fp16_ms"] = (time.perf_counter() - t0) * 1000
        results["fp16_deviation"] = (out_fp16.float() - ref_output).abs().mean().item()
    
    # Dynamic INT8 quantization
    model_int8 = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    with torch.no_grad():
        t0 = time.perf_counter()
        out_int8 = model_int8(sample_input)
        results["int8_ms"] = (time.perf_counter() - t0) * 1000
        results["int8_deviation"] = (out_int8 - ref_output).abs().mean().item()
    
    return results


# TODO 3a: Quantize your Exercise 11 VLA model at all precision levels
# Report: latency speedup, accuracy degradation, model size reduction

# TODO 3b: Research and summarize TensorRT optimization:
# 1. What does TensorRT do? (layer fusion, kernel auto-tuning, precision calibration)
# 2. How much speedup is typical for transformer models? (2-4×)
# 3. What's the workflow? (export ONNX → trtexec → load TRT engine)

# TODO 3c: torch.compile() experiment
# Apply torch.compile(mode="reduce-overhead") to your model
# Measure: first-call overhead vs steady-state speedup
```

### 3.2 — Batched Inference for Fleet Deployment

```python
"""
When serving VLAs for a fleet of robots, batch inference saves compute.

Single robot: 1 inference per timestep
Fleet of N robots: batch N images → single GPU call → N action predictions

This is how cloud-based VLA serving works:
  - Multiple robots send observations to a central GPU server
  - Server batches them and runs inference once
  - Trade-off: higher latency (batching delay) but better GPU utilization
"""


class VLABatchServer:
    """Batched VLA inference server for multi-robot deployment."""
    
    def __init__(self, model: nn.Module, max_batch_size: int = 8, 
                 max_wait_ms: float = 20.0):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        
        self._request_queue = []
        self._batch_start_time = None
    
    def submit_request(self, observation: dict) -> int:
        """Submit an observation for batched inference. Returns request ID."""
        request_id = len(self._request_queue)
        self._request_queue.append(observation)
        
        if self._batch_start_time is None:
            self._batch_start_time = time.perf_counter()
        
        return request_id
    
    def should_flush(self) -> bool:
        """Check if batch should be processed (full or timeout)."""
        if len(self._request_queue) >= self.max_batch_size:
            return True
        if self._batch_start_time is not None:
            elapsed = (time.perf_counter() - self._batch_start_time) * 1000
            if elapsed >= self.max_wait_ms:
                return True
        return False
    
    @torch.no_grad()
    def flush(self) -> list[np.ndarray]:
        """Process the current batch and return actions for each request."""
        if not self._request_queue:
            return []
        
        # Collate observations into batch tensors
        images = torch.stack([obs["image"] for obs in self._request_queue])
        
        # Single batched forward pass
        actions = self.model(images)
        
        # Split back to individual results
        results = [actions[i].cpu().numpy() for i in range(len(self._request_queue))]
        
        # Reset queue
        self._request_queue.clear()
        self._batch_start_time = None
        
        return results


# TODO 3d: Simulate a fleet of 4 robots sending requests
# Compare: individual inference vs batched
# Measure: GPU utilization, per-robot latency, throughput (actions/second)
```

---

## Exercise 4: Failure Recovery Engineering

**Goal**: Handle the things that go wrong in production VLA deployments.

### 4.1 — Failure Modes and Recovery

```python
"""Production VLA failure modes and recovery strategies.

Common failures:
  1. Model stall: inference takes too long
  2. Prediction NaN/Inf: numerical instability
  3. Unsafe action: violates joint limits or workspace bounds
  4. Communication drop: network to GPU server fails
  5. Camera failure: black/frozen image
  6. Instruction ambiguity: model outputs random actions

Each needs a specific recovery strategy!
"""

from enum import Enum, auto
from typing import Optional


class FailureType(Enum):
    MODEL_STALL = auto()
    INVALID_ACTION = auto()
    UNSAFE_ACTION = auto()
    COMMUNICATION_DROP = auto()
    SENSOR_FAILURE = auto()
    LOW_CONFIDENCE = auto()


class RecoveryStrategy(Enum):
    HOLD_POSITION = auto()       # Stay where you are
    REPLAY_LAST = auto()         # Repeat last valid action
    SLOW_STOP = auto()           # Decelerate to zero velocity
    EMERGENCY_STOP = auto()      # Immediate halt
    RETRY = auto()               # Try prediction again
    FALLBACK_POLICY = auto()     # Switch to a simpler policy


class RobustVLAController:
    """VLA controller with comprehensive failure handling."""
    
    def __init__(self, vla_pipeline: 'VLAInferencePipeline',
                 joint_limits: np.ndarray,
                 workspace_bounds: np.ndarray,
                 max_velocity: float = 1.0,
                 max_acceleration: float = 5.0,
                 stall_timeout_ms: float = 200.0):
        self.vla = vla_pipeline
        self.joint_limits = joint_limits     # (n_joints, 2): [min, max]
        self.workspace_bounds = workspace_bounds
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.stall_timeout_ms = stall_timeout_ms
        
        # State tracking
        self._last_valid_action: Optional[np.ndarray] = None
        self._last_action_time: float = 0.0
        self._consecutive_failures: int = 0
        self._failure_log: list = []
    
    def _validate_action(self, action: np.ndarray) -> tuple[bool, Optional[FailureType]]:
        """Check action validity before execution."""
        # Check for NaN/Inf
        if not np.all(np.isfinite(action)):
            return False, FailureType.INVALID_ACTION
        
        # Check joint limits
        if np.any(action < self.joint_limits[:, 0]) or \
           np.any(action > self.joint_limits[:, 1]):
            return False, FailureType.UNSAFE_ACTION
        
        # Check velocity limit (action change rate)
        if self._last_valid_action is not None:
            velocity = np.abs(action - self._last_valid_action) * 10.0  # at 10 Hz
            if np.any(velocity > self.max_velocity):
                return False, FailureType.UNSAFE_ACTION
        
        return True, None
    
    def _recover(self, failure: FailureType) -> tuple[np.ndarray, RecoveryStrategy]:
        """Select and execute recovery strategy based on failure type."""
        self._consecutive_failures += 1
        self._failure_log.append({
            "type": failure, 
            "time": time.time(),
            "consecutive": self._consecutive_failures
        })
        
        # Escalation: more consecutive failures → more aggressive recovery
        if self._consecutive_failures > 10:
            return np.zeros_like(self._last_valid_action or np.zeros(7)), \
                   RecoveryStrategy.EMERGENCY_STOP
        
        if failure == FailureType.MODEL_STALL:
            if self._last_valid_action is not None:
                # Gradually decelerate
                decay = 0.9 ** self._consecutive_failures
                return self._last_valid_action * decay, RecoveryStrategy.SLOW_STOP
            return np.zeros(7), RecoveryStrategy.HOLD_POSITION
        
        elif failure == FailureType.INVALID_ACTION:
            return self._last_valid_action if self._last_valid_action is not None \
                else np.zeros(7), RecoveryStrategy.REPLAY_LAST
        
        elif failure == FailureType.UNSAFE_ACTION:
            # Clamp to safe limits
            if self._last_valid_action is not None:
                return np.clip(self._last_valid_action, 
                              self.joint_limits[:, 0], 
                              self.joint_limits[:, 1]), RecoveryStrategy.HOLD_POSITION
            return np.zeros(7), RecoveryStrategy.EMERGENCY_STOP
        
        else:
            return np.zeros(7), RecoveryStrategy.EMERGENCY_STOP
    
    def get_action(self, observation: dict) -> tuple[np.ndarray, dict]:
        """Get a safe action, with recovery if needed.
        
        Returns:
            action: safe action to execute
            info: diagnostic information
        """
        info = {"recovery": None, "failure": None}
        
        # Attempt VLA prediction with timeout
        t0 = time.perf_counter()
        try:
            action, profile = self.vla.predict_action(
                observation["image"], observation["instruction"]
            )
            latency = (time.perf_counter() - t0) * 1000
            
            if latency > self.stall_timeout_ms:
                action, strategy = self._recover(FailureType.MODEL_STALL)
                info["recovery"] = strategy
                info["failure"] = FailureType.MODEL_STALL
                return action, info
            
        except Exception as e:
            action, strategy = self._recover(FailureType.MODEL_STALL)
            info["recovery"] = strategy
            info["failure"] = FailureType.MODEL_STALL
            info["error"] = str(e)
            return action, info
        
        # Validate action
        is_valid, failure_type = self._validate_action(action)
        if not is_valid:
            action, strategy = self._recover(failure_type)
            info["recovery"] = strategy
            info["failure"] = failure_type
            return action, info
        
        # Success — reset failure counter
        self._consecutive_failures = 0
        self._last_valid_action = action.copy()
        self._last_action_time = time.perf_counter()
        
        return action, info


# TODO 4a: Simulate 1000 control steps where the VLA:
# - Returns NaN 5% of the time
# - Exceeds joint limits 3% of the time
# - Stalls (>200ms) 2% of the time
# Measure: how many emergency stops, how smooth is the trajectory?

# TODO 4b: Plot the failure recovery timeline:
# X-axis: timestep
# Y-axis: action values
# Color code: green=normal, yellow=recovery, red=emergency_stop

# TODO 4c: Implement a "confidence estimator" that detects when
# the VLA is uncertain (e.g., high action variance across chunk steps)
# and triggers preemptive slowdown before a failure occurs
```

---

## Exercise 5: End-to-End System Integration

**Goal**: Wire everything together into a complete, deployable VLA system.

### 5.1 — System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    VLA Control System                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────┐    ┌───────────────┐    ┌─────────────────┐  │
│  │  Camera  │───▶│ Image Preproc │───▶│                 │  │
│  │  (30fps) │    │ (resize/norm) │    │   VLA Model     │  │
│  └──────────┘    └───────────────┘    │  (7B params)    │  │
│                                        │                 │  │
│  ┌──────────┐    ┌───────────────┐    │  ViT encoder    │  │
│  │  Task    │───▶│ Text Tokenize │───▶│  + LLM decoder  │  │
│  │  Instr.  │    │  (cached)     │    │  + Action head  │  │
│  └──────────┘    └───────────────┘    └────────┬────────┘  │
│                                                 │            │
│                                     ┌───────────▼──────┐    │
│  ┌──────────┐    ┌─────────────┐   │ Action Chunking  │    │
│  │  Robot   │◀───│ Safety Gate │◀──│ + Temporal Ens.  │    │
│  │  Joints  │    │ (validate)  │   │  (buffer=16)     │    │
│  └──────────┘    └─────────────┘   └──────────────────┘    │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Monitors: Latency | Buffer | Stalls | Safety Stops   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 — Integration Test

```python
"""Full system integration test."""


def integration_test():
    """Test the complete VLA pipeline end-to-end.
    
    Checks:
    1. Image → model → action → robot (correctness)
    2. Latency stays within budget (performance)
    3. Failure recovery works (robustness)
    4. No memory leaks over 1000 steps (stability)
    """
    
    # TODO 5a: Set up complete pipeline
    # 1. Create mock environment (gym-like)
    # 2. Create VLA inference pipeline
    # 3. Wrap in ActionChunkController
    # 4. Wrap in RobustVLAController
    # 5. Run 1000 steps
    
    # TODO 5b: Verify key metrics
    # assert mean_latency < 100  # ms
    # assert stall_rate < 0.05   # 5%
    # assert emergency_stops == 0
    # assert memory_growth < 10  # MB over 1000 steps
    
    # TODO 5c: Generate system report
    # - Latency breakdown pie chart
    # - Action trajectory plot
    # - Buffer occupancy timeline
    # - Failure recovery events marked
    pass
```

---

## Self-Check

Before moving on, verify:

- [ ] You can profile VLA inference and identify the bottleneck
- [ ] You understand action chunking's trade-off: latency vs freshness
- [ ] You can implement temporal ensembling in a real-time control loop
- [ ] You know the recovery strategies for each failure mode
- [ ] You can calculate whether a model fits a real-time budget
- [ ] You understand batched serving for multi-robot deployment
- [ ] You can list 3 ways to speed up VLA inference (quantization, compile, smaller model)

---

## Stretch Goals

### S1: Async Pipeline
Implement truly asynchronous inference where image capture, model inference,
and action execution run on separate threads. Measure the throughput improvement.

### S2: Model Distillation
Distill a 7B VLA into a 1B student model. Compare:
- Accuracy on benchmark tasks
- Inference latency
- Does the smaller model fit real-time constraints?

### S3: Adaptive Control Rate
Implement a controller that dynamically adjusts control rate based on:
- Task phase (fast during reaching, slow during precision grasping)
- Model confidence (slow down when uncertain)
- Buffer occupancy (speed up when buffer is full)

### S4: A/B Testing Framework
Build a framework to compare two VLA policies online:
- Randomly assign episodes to policy A vs B
- Track success rate, completion time, smoothness
- Compute confidence intervals for which policy is better

---

*This exercise connects directly to production deployment of VLA systems like RT-2, OpenVLA, and π₀.*
