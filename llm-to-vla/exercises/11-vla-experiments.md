# Exercise 11 — VLA Experiments

> Phase VII · Days 92–106
> Prerequisites: exercises/08-vlm-experiments, exercises/10-imitation-learning

---

## Overview

Vision-Language-Action (VLA) models unify perception, language understanding, and motor
control into a single transformer. In this exercise set you will run inference with
OpenVLA, explore action tokenization strategies, fine-tune with LoRA, build a hybrid
architecture, and analyse cross-modal attention patterns.

**Learning objectives**

| # | Objective | Exercises |
|---|-----------|-----------|
| 1 | Run a pretrained VLA end-to-end | 1 |
| 2 | Understand action tokenization trade-offs | 2 |
| 3 | Parameter-efficient fine-tuning of a VLA | 3 |
| 4 | Integrate classical control with a VLA | 4 |
| 5 | Interpret and debug VLA behaviour | 5 |

---

## Environment Setup

```bash
# Create a dedicated environment
conda create -n vla python=3.10 -y
conda activate vla

# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.40 accelerate>=0.30
pip install peft>=0.11 bitsandbytes>=0.43

# VLA / robotics specific
# NOTE: There is no "openvla" pip package. OpenVLA loads via transformers:
#   AutoModelForVision2Seq.from_pretrained("openvla/openvla-7b")
pip install lerobot            # Hugging Face LeRobot datasets & envs
pip install gymnasium>=0.29
pip install wandb matplotlib seaborn tqdm

# Optional — for Exercise 4 hybrid controller
pip install simple-pid
```

```python
# Quick sanity check
import torch, transformers, peft
print(f"torch  : {torch.__version__}  CUDA: {torch.cuda.is_available()}")
print(f"transformers : {transformers.__version__}")
print(f"peft         : {peft.__version__}")

# Verify OpenVLA loads from HuggingFace Hub (downloads ~14 GB on first run)
from transformers import AutoModelForVision2Seq, AutoProcessor
print("OpenVLA model ID: openvla/openvla-7b  (loads via transformers, not a separate pip package)")
```

> **GPU note**: Exercises 1 and 5 can run on a single T4 (16 GB) with 4-bit
> quantization. Exercise 3 fine-tuning benefits from an A100 (40 GB) or two T4s
> with gradient checkpointing.

---

## Exercise 1 — OpenVLA Inference

### 1.1 Goal

Load a pretrained VLA, feed it a camera image and a natural-language instruction,
and observe the predicted robot action.

### 1.2 Background

OpenVLA-7B maps `(image, instruction) → action` where the action is a 7-DoF
vector (Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper). The model is a Llama-2
backbone with a vision encoder (SigLIP) fused via cross-attention.

### 1.3 Starter Code

```python
"""Exercise 1 — OpenVLA Inference

Run: python 11_ex1_openvla_inference.py --image scene.png --instruction "pick up the red block"
"""
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# TODO 1a: Import the OpenVLA model and processor from transformers
# Hint: AutoModelForVision2Seq, AutoProcessor
# from transformers import ???


def load_model(quantize: bool = True):
    """Load OpenVLA-7B (optionally 4-bit quantized)."""
    model_id = "openvla/openvla-7b"

    # TODO 1b: Configure quantization if requested
    # Hint: Use BitsAndBytesConfig with load_in_4bit=True
    quant_config = None
    if quantize:
        from transformers import BitsAndBytesConfig
        # quant_config = BitsAndBytesConfig(???)
        pass

    # TODO 1c: Load model and processor
    # processor = AutoProcessor.from_pretrained(model_id)
    # model = AutoModelForVision2Seq.from_pretrained(
    #     model_id, quantization_config=quant_config, device_map="auto",
    #     torch_dtype=torch.bfloat16
    # )
    processor, model = None, None  # Replace

    return processor, model


def predict_action(processor, model, image: Image.Image, instruction: str):
    """Predict a 7-DoF action from image + instruction.

    Returns:
        np.ndarray of shape (7,) — [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """
    # TODO 1d: Tokenize inputs
    # inputs = processor(images=image, text=instruction, return_tensors="pt").to(model.device)

    # TODO 1e: Forward pass (no grad)
    # with torch.no_grad():
    #     outputs = model.generate(**inputs, max_new_tokens=256)

    # TODO 1f: Decode action tokens into a 7-DoF numpy array
    # IMPORTANT: There is NO `processor.decode_action()` method.
    # OpenVLA outputs action tokens as text; you must decode manually:
    #
    # 1. Decode generated token IDs to text:
    #    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    #
    # 2. Parse the 7 action token strings from the generated text.
    #    OpenVLA generates tokens like "<action_0>", "<action_1>", ...
    #    Each maps to a bin index (0-255) in a 256-bin uniform discretization.
    #
    # 3. Convert bin indices to continuous values using the dataset statistics:
    #    action_i = dataset_mean[i] + dataset_std[i] * (2 * bin_index / 255 - 1)
    #
    # For the default Bridge V2 normalization:
    #    action_mean = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    #    action_std  = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    #
    # Skeleton:
    # generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    # # extract the 7 bin indices from generated_text (implementation depends on tokenizer)
    # bin_indices = ???  # list of 7 ints in [0, 255]
    # action = np.array([(2 * b / 255 - 1) for b in bin_indices])  # normalized
    action = np.zeros(7)  # Replace with your decoding

    return action


def visualize_action(image: Image.Image, action: np.ndarray, instruction: str):
    """Overlay predicted action on the image."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: image with instruction
    axes[0].imshow(image)
    axes[0].set_title(f"Instruction: {instruction}", fontsize=10)
    axes[0].axis("off")

    # Right: bar chart of action dimensions
    labels = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]
    colors = ["#2196F3"] * 6 + ["#FF5722"]
    axes[1].barh(labels, action, color=colors)
    axes[1].set_xlabel("Predicted Value")
    axes[1].set_title("Predicted Action")
    axes[1].axvline(x=0, color="gray", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("ex1_action_prediction.png", dpi=150)
    plt.show()
    print("Saved → ex1_action_prediction.png")


# ──────────────────────────────────────────────────────────────
# 1.4  Generalization probes
# ──────────────────────────────────────────────────────────────

def generalization_test(processor, model, image: Image.Image):
    """Test the same image with multiple instructions."""
    instructions = [
        "pick up the red block",
        "push the blue cube to the left",
        "stack the green block on top of the red one",
        "move the arm to the home position",
        "open the gripper",
    ]
    results = {}
    for inst in instructions:
        action = predict_action(processor, model, image, inst)
        results[inst] = action
        print(f"  [{inst:.<50s}] → {action.round(3)}")
    return results


def plot_generalization(results: dict):
    """Compare action vectors across instructions."""
    labels = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.15

    for i, (inst, action) in enumerate(results.items()):
        offset = (i - len(results) / 2) * width
        ax.bar(x + offset, action, width, label=inst[:30])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Predicted Value")
    ax.set_title("VLA Generalization — Same Image, Different Instructions")
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig("ex1_generalization.png", dpi=150)
    plt.show()


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="scene.png")
    parser.add_argument("--instruction", type=str, default="pick up the red block")
    parser.add_argument("--no-quantize", action="store_true")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    processor, model = load_model(quantize=not args.no_quantize)

    # Single prediction
    action = predict_action(processor, model, image, args.instruction)
    print(f"\nPredicted action: {action.round(4)}")
    visualize_action(image, action, args.instruction)

    # Generalization sweep
    print("\n=== Generalization Test ===")
    results = generalization_test(processor, model, image)
    plot_generalization(results)
```

### 1.4 Tasks

| Task | Description | Difficulty |
|------|-------------|------------|
| 1a–1f | Complete all TODO items in starter code | ★★☆ |
| 1g | Run with 3 different scene images and compare | ★☆☆ |
| 1h | Measure inference latency (ms) with and without quantization | ★★☆ |
| 1i | Find an instruction that causes a nonsensical action — explain why | ★★★ |

### 1.5 Expected Output

- `ex1_action_prediction.png` — annotated image + bar chart
- `ex1_generalization.png` — grouped bar chart across instructions
- Console table of actions for 5 instructions
- Latency report: fp16 vs 4-bit (aim for < 200 ms on T4 with quant)

---

## Exercise 2 — Action Tokenization Comparison

### 2.1 Goal

Understand how continuous robot actions are converted to discrete tokens and back.
Compare **uniform binning** (OpenVLA style) with **direct regression** heads.

### 2.2 Background

OpenVLA discretizes each action dimension into 256 bins uniformly spanning the
action range. Each bin is a token in the LLM vocabulary. This converts a
regression problem into a classification problem — which transformers are naturally
good at. The trade-off is quantization error.

### 2.3 Starter Code

```python
"""Exercise 2 — Action Tokenization Comparison

Compare discrete bins vs direct regression for robot actions.
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Tuple


# ──────────────────────────────────────────────────────────────
# 2A  Discrete Binning Tokenizer
# ──────────────────────────────────────────────────────────────

class DiscreteActionTokenizer:
    """Uniform binning for continuous actions (OpenVLA style)."""

    def __init__(self, n_bins: int = 256, action_range: Tuple[float, float] = (-1.0, 1.0)):
        self.n_bins = n_bins
        self.lo, self.hi = action_range
        # TODO 2a: Compute bin edges and bin centers
        # self.edges = np.linspace(???)
        # self.centers = ???
        self.edges = np.linspace(self.lo, self.hi, n_bins + 1)
        self.centers = (self.edges[:-1] + self.edges[1:]) / 2.0

    def encode(self, actions: np.ndarray) -> np.ndarray:
        """Continuous actions → bin indices.

        Args:
            actions: shape (batch, action_dim), values in [lo, hi]
        Returns:
            indices: shape (batch, action_dim), int in [0, n_bins)
        """
        # TODO 2b: Clip and digitize
        # Hint: np.clip, then np.digitize, then clamp to [0, n_bins-1]
        clipped = np.clip(actions, self.lo, self.hi)
        indices = np.digitize(clipped, self.edges[1:-1])
        return indices

    def decode(self, indices: np.ndarray) -> np.ndarray:
        """Bin indices → reconstructed continuous actions."""
        # TODO 2c: Map indices to bin centers
        return self.centers[indices]

    def reconstruction_error(self, actions: np.ndarray) -> float:
        """Round-trip encode→decode and measure MSE."""
        indices = self.encode(actions)
        reconstructed = self.decode(indices)
        return float(np.mean((actions - reconstructed) ** 2))


# ──────────────────────────────────────────────────────────────
# 2B  Direct Regression Head
# ──────────────────────────────────────────────────────────────

class RegressionActionHead(nn.Module):
    """MLP head that directly predicts continuous action values."""

    def __init__(self, input_dim: int = 768, action_dim: int = 7, hidden_dim: int = 256):
        super().__init__()
        # TODO 2d: Build a 2-layer MLP with ReLU
        # self.net = nn.Sequential(???)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # constrain to [-1, 1]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


# ──────────────────────────────────────────────────────────────
# 2C  Comparison Experiment
# ──────────────────────────────────────────────────────────────

def compare_tokenization_strategies():
    """Compare discrete bins vs regression on synthetic data."""
    np.random.seed(42)

    # Synthetic ground-truth actions: 1000 samples × 7 DoF
    gt_actions = np.random.uniform(-1.0, 1.0, size=(1000, 7))

    # ── Discrete binning across different bin counts ──
    bin_counts = [8, 16, 32, 64, 128, 256, 512, 1024]
    discrete_errors = []

    for n_bins in bin_counts:
        tokenizer = DiscreteActionTokenizer(n_bins=n_bins)
        mse = tokenizer.reconstruction_error(gt_actions)
        discrete_errors.append(mse)
        print(f"  Bins={n_bins:>5d}  →  MSE = {mse:.6f}")

    # ── Theoretical lower bound for uniform quantization ──
    # For uniform quantization on [-1, 1]: MSE = (2/n_bins)^2 / 12
    theoretical = [(2.0 / n) ** 2 / 12.0 for n in bin_counts]

    # TODO 2e: Plot the comparison
    # X-axis: number of bins (log scale)
    # Y-axis: MSE (log scale)
    # Two curves: empirical discrete error, theoretical bound
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(bin_counts, discrete_errors, "o-", label="Empirical Discrete MSE", linewidth=2)
    ax.plot(bin_counts, theoretical, "s--", label="Theoretical Bound", linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Number of Bins")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Action Tokenization: Discretization Error vs Bin Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("ex2_tokenization_comparison.png", dpi=150)
    plt.show()

    return bin_counts, discrete_errors, theoretical


def training_stability_experiment():
    """Compare training stability of classification (discrete) vs regression heads."""
    torch.manual_seed(42)

    input_dim = 768
    action_dim = 7
    n_samples = 500
    n_epochs = 100
    lr = 1e-3

    # Synthetic features and targets
    features = torch.randn(n_samples, input_dim)
    targets = torch.rand(n_samples, action_dim) * 2 - 1  # [-1, 1]

    # ── Regression head ──
    reg_model = RegressionActionHead(input_dim, action_dim)
    reg_opt = torch.optim.Adam(reg_model.parameters(), lr=lr)
    reg_losses = []

    for epoch in range(n_epochs):
        pred = reg_model(features)
        loss = nn.functional.mse_loss(pred, targets)
        reg_opt.zero_grad()
        loss.backward()
        reg_opt.step()
        reg_losses.append(loss.item())

    # ── Classification head (discrete tokens) ──
    n_bins = 256
    tokenizer = DiscreteActionTokenizer(n_bins=n_bins)
    target_indices = torch.from_numpy(
        tokenizer.encode(targets.numpy())
    ).long()  # (n_samples, action_dim)

    cls_model = nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Linear(256, n_bins * action_dim),
    )
    cls_opt = torch.optim.Adam(cls_model.parameters(), lr=lr)
    cls_losses = []

    for epoch in range(n_epochs):
        logits = cls_model(features).view(n_samples, action_dim, n_bins)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, n_bins), target_indices.reshape(-1)
        )
        cls_opt.zero_grad()
        loss.backward()
        cls_opt.step()
        cls_losses.append(loss.item())

    # TODO 2f: Plot training curves side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(reg_losses, label="Regression (MSE)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Regression Head Training")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(cls_losses, label="Classification (CE)", color="orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Discrete Token Head Training")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ex2_training_stability.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    print("=== Tokenization Comparison ===")
    compare_tokenization_strategies()

    print("\n=== Training Stability ===")
    training_stability_experiment()
```

### 2.4 Tasks

| Task | Description | Difficulty |
|------|-------------|------------|
| 2a–2c | Complete the DiscreteActionTokenizer | ★☆☆ |
| 2d | Build the regression head MLP | ★☆☆ |
| 2e–2f | Generate both plots | ★★☆ |
| 2g | Add a **non-uniform** binning strategy (e.g., log-spaced near zero) and compare | ★★★ |
| 2h | Measure wall-clock time for encode/decode at batch size 1024 | ★★☆ |

### 2.5 Expected Output

- `ex2_tokenization_comparison.png` — log-log MSE vs bins with theoretical curve
- `ex2_training_stability.png` — two training loss curves side by side
- Console printout of MSE for each bin count

### 2.6 Reflection Questions

1. Why does OpenVLA use 256 bins? What would go wrong with 16 bins? With 4096?
2. Regression heads avoid quantization error — so why do VLAs prefer discrete tokens?
3. How does the training stability differ? Which converges faster?

### 2.7 Advanced: μ-Law and Learned Tokenization

```python
"""Advanced action tokenization: non-uniform and learned approaches."""

import torch
import torch.nn as nn
import numpy as np


class MuLawActionTokenizer:
    """μ-law companding tokenizer (used in audio, applicable to actions).
    
    Key idea: Actions near zero need finer resolution than actions at extremes.
    μ-law compression concentrates bins near zero.
    
    Formula: F(x) = sign(x) * ln(1 + μ|x|) / ln(1 + μ)
    """
    
    def __init__(self, n_bins: int = 256, mu: float = 255.0, 
                 action_range: tuple = (-1.0, 1.0)):
        self.n_bins = n_bins
        self.mu = mu
        self.lo, self.hi = action_range
    
    def _compress(self, x: np.ndarray) -> np.ndarray:
        """Apply μ-law compression: maps [-1,1] → [-1,1] non-linearly."""
        return np.sign(x) * np.log1p(self.mu * np.abs(x)) / np.log1p(self.mu)
    
    def _expand(self, y: np.ndarray) -> np.ndarray:
        """Inverse μ-law: decompress back to linear."""
        return np.sign(y) * (np.exp(np.abs(y) * np.log1p(self.mu)) - 1) / self.mu
    
    def encode(self, actions: np.ndarray) -> np.ndarray:
        """Encode actions → token indices using μ-law compression."""
        # Normalize to [-1, 1]
        normalized = 2.0 * (actions - self.lo) / (self.hi - self.lo) - 1.0
        # Compress
        compressed = self._compress(normalized)
        # Quantize compressed values to bins
        indices = ((compressed + 1.0) / 2.0 * (self.n_bins - 1)).astype(int)
        return np.clip(indices, 0, self.n_bins - 1)
    
    def decode(self, indices: np.ndarray) -> np.ndarray:
        """Decode token indices → actions."""
        # Indices → compressed values
        compressed = indices.astype(float) / (self.n_bins - 1) * 2.0 - 1.0
        # Expand
        normalized = self._expand(compressed)
        # Denormalize
        return (normalized + 1.0) / 2.0 * (self.hi - self.lo) + self.lo


class LearnedVQActionTokenizer(nn.Module):
    """Learned action tokenizer via Vector Quantization (VQ-VAE style).
    
    Instead of fixed bins, learn a codebook of action prototypes.
    The encoder maps continuous actions to the nearest codebook entry.
    """
    
    def __init__(self, action_dim: int = 7, codebook_size: int = 512,
                 embedding_dim: int = 32):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        
        # Encoder: action → embedding
        self.encoder = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )
        
        # Codebook: learnable embeddings
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)
        
        # Decoder: embedding → action
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )
    
    def encode(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode actions → nearest codebook indices.
        
        Returns:
            indices: (batch,) codebook indices
            quantized: (batch, embedding_dim) quantized embeddings
        """
        z_e = self.encoder(actions)  # (batch, embedding_dim)
        
        # Find nearest codebook entry
        distances = torch.cdist(z_e.unsqueeze(0), 
                                self.codebook.weight.unsqueeze(0)).squeeze(0)
        indices = distances.argmin(dim=-1)  # (batch,)
        
        # Look up quantized embeddings
        quantized = self.codebook(indices)  # (batch, embedding_dim)
        
        return indices, quantized
    
    def decode(self, quantized: torch.Tensor) -> torch.Tensor:
        """Decode quantized embeddings → reconstructed actions."""
        return self.decoder(quantized)
    
    def forward(self, actions: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Full forward pass with VQ loss.
        
        Returns reconstructed actions and loss dict.
        """
        z_e = self.encoder(actions)
        indices, z_q = self.encode(actions)
        
        # Straight-through estimator: gradient flows through z_q as if it were z_e
        z_q_st = z_e + (z_q - z_e).detach()
        
        reconstructed = self.decoder(z_q_st)
        
        # Losses
        commitment_loss = torch.mean((z_e.detach() - z_q) ** 2)
        codebook_loss = torch.mean((z_e - z_q.detach()) ** 2)
        recon_loss = torch.mean((actions - reconstructed) ** 2)
        
        losses = {
            "recon": recon_loss,
            "codebook": commitment_loss,
            "commitment": codebook_loss,
            "total": recon_loss + commitment_loss + 0.25 * codebook_loss,
        }
        
        return reconstructed, losses


# TODO: Train the VQ tokenizer on a dataset of robot actions
# 1. Generate 10000 random trajectories (action sequences)
# 2. Train VQ-VAE for 100 epochs
# 3. Compare reconstruction error with uniform binning (256 bins)
# 4. Visualize the learned codebook: do clusters correspond to action types?
# 5. Measure codebook utilization: how many entries are actually used?

# TODO: Compare all three on manipulation-specific action distributions:
# actions = sample_realistic_robot_actions()  # bimodal, clustered near zero
# errors = {
#     "Uniform 256": DiscreteActionTokenizer(256).reconstruction_error(actions),
#     "μ-law 256": MuLawActionTokenizer(256).reconstruction_error(actions),
#     "VQ 512": vq_model_error(actions),
# }
```

**Exercises**:
1. Implement `MuLawActionTokenizer` — compare MSE with uniform binning at 64 and 256 bins
2. Train `LearnedVQActionTokenizer` on synthetic 7-DoF actions — visualize codebook usage
3. **Key experiment**: Create a realistic action distribution (mostly near zero with occasional large motions) — which tokenizer has lowest error?
4. **π₀ connection**: The π₀ paper uses flow matching instead of discrete tokens. What's the advantage? What's lost?

### 3.1 Goal

Apply LoRA adapters to OpenVLA's language-model backbone and fine-tune on a
custom manipulation task from LeRobot.

### 3.2 Background

Full fine-tuning of a 7B model requires ≥80 GB VRAM. LoRA (rank 16) adds
<0.5 % trainable parameters, making fine-tuning feasible on a single GPU.
We freeze the vision encoder and apply LoRA only to the LLM's Q/K/V/O
projection matrices.

### 3.3 Starter Code

```python
"""Exercise 3 — VLA Fine-tuning with LoRA

Fine-tune OpenVLA on a LeRobot task using parameter-efficient adaptation.

Usage:
    python 11_ex3_vla_finetune.py --task pusht --epochs 10 --lr 2e-4
"""
import argparse
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# TODO 3a: Import required classes
# from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
# from peft import LoraConfig, get_peft_model, TaskType


# ──────────────────────────────────────────────────────────────
# 3A  Data Pipeline
# ──────────────────────────────────────────────────────────────

def build_dataloader(task: str, batch_size: int = 8, split: str = "train"):
    """Load a LeRobot dataset and return a DataLoader.

    Each sample contains:
        - "observation.image": (C, H, W) tensor
        - "action": (action_dim,) tensor
        - "language_instruction": str
    """
    # TODO 3b: Load dataset from LeRobot hub
    # from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    # dataset = LeRobotDataset(f"lerobot/{task}")
    #
    # def collate_fn(batch):
    #     images = torch.stack([s["observation.image"] for s in batch])
    #     actions = torch.stack([s["action"] for s in batch])
    #     instructions = [s.get("language_instruction", f"perform {task}") for s in batch]
    #     return {"images": images, "actions": actions, "instructions": instructions}
    #
    # return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"),
    #                   collate_fn=collate_fn, num_workers=4, pin_memory=True)
    pass  # Replace with above


# ──────────────────────────────────────────────────────────────
# 3B  Model with LoRA
# ──────────────────────────────────────────────────────────────

def load_model_with_lora(rank: int = 16, alpha: int = 32):
    """Load OpenVLA with 4-bit quant and attach LoRA adapters."""
    model_id = "openvla/openvla-7b"

    # TODO 3c: Quantization config
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    #     bnb_4bit_use_double_quant=True,
    # )

    # TODO 3d: Load base model
    # processor = AutoProcessor.from_pretrained(model_id)
    # model = AutoModelForVision2Seq.from_pretrained(
    #     model_id, quantization_config=bnb_config,
    #     device_map="auto", torch_dtype=torch.bfloat16
    # )

    # TODO 3e: LoRA config — target Q, K, V, O projection layers
    # lora_config = LoraConfig(
    #     r=rank,
    #     lora_alpha=alpha,
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type=TaskType.CAUSAL_LM,
    # )
    # model = get_peft_model(model, lora_config)
    # model.print_trainable_parameters()

    processor, model = None, None  # Replace
    return processor, model


# ──────────────────────────────────────────────────────────────
# 3C  Training Loop
# ──────────────────────────────────────────────────────────────

def train(
    processor,
    model,
    train_loader,
    val_loader,
    epochs: int = 10,
    lr: float = 2e-4,
    grad_accum: int = 4,
    log_wandb: bool = True,
):
    """Fine-tune the LoRA-adapted VLA."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if log_wandb:
        wandb.init(project="vla-finetune", config={
            "epochs": epochs, "lr": lr, "grad_accum": grad_accum,
        })

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            # TODO 3f: Forward pass
            # inputs = processor(
            #     images=batch["images"], text=batch["instructions"],
            #     return_tensors="pt"
            # ).to(model.device)
            # outputs = model(**inputs, labels=???)  # Action token labels
            # loss = outputs.loss / grad_accum

            loss = torch.tensor(0.0)  # Replace

            loss.backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * grad_accum

        avg_train_loss = epoch_loss / len(train_loader)
        scheduler.step()

        # ── Validation ──
        val_loss = evaluate(processor, model, val_loader)

        print(f"  Epoch {epoch+1}: train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}")

        if log_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "lr": scheduler.get_last_lr()[0],
            })

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained("checkpoints/best_lora_adapter")
            print(f"  ✓ Saved best adapter (val_loss={val_loss:.4f})")

    if log_wandb:
        wandb.finish()


def evaluate(processor, model, val_loader) -> float:
    """Compute average validation loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            # TODO 3g: Mirror the forward pass from training
            loss = torch.tensor(0.0)  # Replace
            total_loss += loss.item()
    return total_loss / max(len(val_loader), 1)


# ──────────────────────────────────────────────────────────────
# 3D  Success Rate Evaluation
# ──────────────────────────────────────────────────────────────

def measure_success_rate(processor, model, task: str, n_episodes: int = 50):
    """Roll out the fine-tuned VLA in the LeRobot environment.

    Returns:
        success_rate: float in [0, 1]
        episode_returns: list of floats
    """
    import gymnasium as gym

    # TODO 3h: Create environment, run episodes, track successes
    # env = gym.make(f"lerobot/{task}")
    # successes = 0
    # returns = []
    # for ep in range(n_episodes):
    #     obs, info = env.reset()
    #     done = False
    #     ep_return = 0.0
    #     while not done:
    #         image = obs["image"]
    #         instruction = info.get("instruction", f"perform {task}")
    #         action = predict_action(processor, model, image, instruction)
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         done = terminated or truncated
    #         ep_return += reward
    #     if info.get("success", False):
    #         successes += 1
    #     returns.append(ep_return)
    # return successes / n_episodes, returns

    return 0.0, []  # Replace


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="pusht")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    print(f"=== Fine-tuning OpenVLA on '{args.task}' with LoRA (rank={args.lora_rank}) ===")

    processor, model = load_model_with_lora(rank=args.lora_rank)
    train_loader = build_dataloader(args.task, batch_size=args.batch_size, split="train")
    val_loader = build_dataloader(args.task, batch_size=args.batch_size, split="val")

    train(processor, model, train_loader, val_loader,
          epochs=args.epochs, lr=args.lr, log_wandb=not args.no_wandb)

    print("\n=== Evaluating ===")
    sr_base, _ = measure_success_rate(processor, model, args.task)
    print(f"  Base success rate:      {sr_base:.2%}")

    # Load fine-tuned adapter
    # model.load_adapter("checkpoints/best_lora_adapter")
    sr_ft, _ = measure_success_rate(processor, model, args.task)
    print(f"  Fine-tuned success rate: {sr_ft:.2%}")
    print(f"  Improvement:             {sr_ft - sr_base:+.2%}")
```

### 3.4 Tasks

| Task | Description | Difficulty |
|------|-------------|------------|
| 3a–3e | Complete model loading and LoRA setup | ★★☆ |
| 3f–3g | Implement training and validation forward passes | ★★★ |
| 3h | Implement rollout-based success rate evaluation | ★★★ |
| 3i | Compare rank 4 vs 16 vs 64 — plot success rate vs trainable params | ★★★ |
| 3j | Enable gradient checkpointing for memory savings | ★★☆ |

### 3.5 Expected Output

- wandb dashboard with train/val loss curves, LR schedule
- `checkpoints/best_lora_adapter/` — saved LoRA weights
- Console: base vs fine-tuned success rate comparison
- If doing 3i: bar chart of success rate vs LoRA rank

---

## Exercise 4 — Hybrid VLA + Classical Controller

### 4.1 Goal

Build a system where the VLA generates high-level waypoints or sub-goals, and
a classical PID controller executes the low-level motion. Compare this hybrid
approach against pure end-to-end VLA control.

### 4.2 Background

Pure VLA control at 10 Hz is ambitious. A common pattern is:
- **VLA** runs at 2–5 Hz → produces waypoint / sub-goal (x, y, z, gripper_state)
- **PID** runs at 50–100 Hz → smoothly tracks the waypoint

This hybrid gives you VLA's language-conditioned reasoning with the smoothness
and safety of classical control.

### 4.3 Starter Code

```python
"""Exercise 4 — Hybrid VLA + PID Controller

Run: python 11_ex4_hybrid_controller.py --mode hybrid
"""
import argparse
import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from simple_pid import PID


@dataclass
class Waypoint:
    """A sub-goal produced by the VLA."""
    position: np.ndarray   # (3,) xyz
    orientation: np.ndarray # (3,) rpy
    gripper: float          # 0=open, 1=closed
    timestamp: float


class HybridController:
    """VLA for high-level planning + PID for low-level execution."""

    def __init__(self, vla_hz: float = 3.0, pid_hz: float = 50.0):
        self.vla_hz = vla_hz
        self.pid_hz = pid_hz
        self.waypoint_queue = deque(maxlen=10)

        # TODO 4a: Create PID controllers for each axis (x, y, z)
        # Tune Kp, Ki, Kd for smooth motion
        # self.pid_x = PID(Kp=2.0, Ki=0.1, Kd=0.5, setpoint=0, output_limits=(-1, 1))
        # self.pid_y = PID(???)
        # self.pid_z = PID(???)
        self.pid_x = PID(2.0, 0.1, 0.5, setpoint=0, output_limits=(-1, 1))
        self.pid_y = PID(2.0, 0.1, 0.5, setpoint=0, output_limits=(-1, 1))
        self.pid_z = PID(2.0, 0.1, 0.5, setpoint=0, output_limits=(-1, 1))

        self.current_waypoint = None
        self.metrics = {"smoothness": [], "tracking_error": [], "latencies": []}

    def update_waypoint(self, waypoint: Waypoint):
        """Called at VLA frequency with new sub-goal."""
        self.current_waypoint = waypoint
        self.pid_x.setpoint = waypoint.position[0]
        self.pid_y.setpoint = waypoint.position[1]
        self.pid_z.setpoint = waypoint.position[2]

    def compute_action(self, current_pos: np.ndarray) -> np.ndarray:
        """Called at PID frequency. Returns low-level action.

        Args:
            current_pos: (3,) current end-effector position
        Returns:
            action: (7,) [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        if self.current_waypoint is None:
            return np.zeros(7)

        # TODO 4b: Compute PID output for each axis
        dx = self.pid_x(current_pos[0])
        dy = self.pid_y(current_pos[1])
        dz = self.pid_z(current_pos[2])

        # Orientation — passthrough from VLA for now
        orientation = self.current_waypoint.orientation
        gripper = self.current_waypoint.gripper

        action = np.array([dx, dy, dz, *orientation, gripper])

        # Track metrics
        error = np.linalg.norm(self.current_waypoint.position - current_pos)
        self.metrics["tracking_error"].append(error)

        return action

    def compute_smoothness(self, action_history: list) -> float:
        """Compute jerk-based smoothness metric."""
        if len(action_history) < 3:
            return 0.0
        actions = np.array(action_history)
        # TODO 4c: Compute smoothness as negative mean absolute jerk
        # velocity = np.diff(actions, axis=0)
        # acceleration = np.diff(velocity, axis=0)
        # jerk = np.diff(acceleration, axis=0)
        # smoothness = -np.mean(np.abs(jerk))
        velocity = np.diff(actions, axis=0)
        acceleration = np.diff(velocity, axis=0)
        jerk = np.diff(acceleration, axis=0)
        return -float(np.mean(np.abs(jerk)))


class PureVLAController:
    """Baseline: VLA directly outputs low-level actions at control frequency."""

    def __init__(self, control_hz: float = 10.0):
        self.control_hz = control_hz
        self.metrics = {"smoothness": [], "tracking_error": [], "latencies": []}

    def compute_action(self, processor, model, image, instruction) -> np.ndarray:
        """Run VLA inference at every control step."""
        t0 = time.time()
        # action = predict_action(processor, model, image, instruction)  # from Ex1
        action = np.random.uniform(-0.1, 0.1, size=7)  # Placeholder
        latency = time.time() - t0
        self.metrics["latencies"].append(latency)
        return action


def compare_controllers():
    """Run simulated comparison of hybrid vs pure VLA."""
    import matplotlib.pyplot as plt

    np.random.seed(42)
    n_steps = 500

    # ── Simulate hybrid controller ──
    hybrid = HybridController(vla_hz=3.0, pid_hz=50.0)
    hybrid_actions = []
    pos = np.zeros(3)

    for step in range(n_steps):
        # VLA produces a new waypoint every ~17 steps (50/3 ≈ 17)
        if step % 17 == 0:
            target = np.random.uniform(-0.5, 0.5, size=3)
            wp = Waypoint(target, np.zeros(3), 0.0, step / 50.0)
            hybrid.update_waypoint(wp)

        action = hybrid.compute_action(pos)
        hybrid_actions.append(action[:3])
        pos += action[:3] * 0.02  # simple Euler integration

    # ── Simulate pure VLA controller ──
    pure_actions = []
    pos = np.zeros(3)
    for step in range(n_steps):
        # Pure VLA at 10 Hz — noisier, less smooth
        target = np.random.uniform(-0.5, 0.5, size=3) if step % 5 == 0 else target
        noise = np.random.normal(0, 0.05, size=3)  # simulate VLA jitter
        action = (target - pos) * 0.3 + noise
        pure_actions.append(action)
        pos += action * 0.02

    # ── Plot comparison ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Trajectories
    h = np.array(hybrid_actions)
    p = np.array(pure_actions)

    for dim, label in enumerate(["X", "Y", "Z"]):
        axes[0, 0].plot(h[:, dim], alpha=0.7, label=f"Hybrid {label}")
    axes[0, 0].set_title("Hybrid Controller — Actions")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    for dim, label in enumerate(["X", "Y", "Z"]):
        axes[0, 1].plot(p[:, dim], alpha=0.7, label=f"Pure VLA {label}")
    axes[0, 1].set_title("Pure VLA — Actions")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Smoothness histogram
    h_smooth = hybrid.compute_smoothness(hybrid_actions)
    p_smooth = hybrid.compute_smoothness(pure_actions)  # reuse method
    axes[1, 0].bar(["Hybrid", "Pure VLA"], [h_smooth, p_smooth],
                   color=["#2196F3", "#FF5722"])
    axes[1, 0].set_ylabel("Smoothness (↑ better)")
    axes[1, 0].set_title("Motion Smoothness Comparison")

    # Action magnitude over time
    axes[1, 1].plot(np.linalg.norm(h, axis=1), alpha=0.7, label="Hybrid")
    axes[1, 1].plot(np.linalg.norm(p, axis=1), alpha=0.7, label="Pure VLA")
    axes[1, 1].set_xlabel("Time Step")
    axes[1, 1].set_ylabel("Action Magnitude")
    axes[1, 1].set_title("Action Magnitude Over Time")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ex4_hybrid_vs_pure.png", dpi=150)
    plt.show()
    print(f"Smoothness — Hybrid: {h_smooth:.4f}  Pure VLA: {p_smooth:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hybrid", "pure", "compare"], default="compare")
    args = parser.parse_args()

    if args.mode == "compare":
        compare_controllers()
```

### 4.4 Tasks

| Task | Description | Difficulty |
|------|-------------|------------|
| 4a | Tune PID gains for each axis | ★★☆ |
| 4b–4c | Complete PID action computation and smoothness metric | ★★☆ |
| 4d | Add a **safety clamp** — limit max velocity and acceleration | ★★☆ |
| 4e | Integrate with the real VLA model from Exercise 1 as the waypoint source | ★★★ |
| 4f | Add collision detection — stop if action magnitude exceeds threshold | ★★★ |

### 4.5 Expected Output

- `ex4_hybrid_vs_pure.png` — 2×2 comparison plot
- Console: smoothness scores for both controllers
- Discussion: when does hybrid outperform pure VLA?

---

## Exercise 5 — VLA Attention Analysis

### 5.1 Goal

Visualize cross-modal attention in a VLA to understand what the model "looks at"
when generating each action token.

### 5.2 Background

VLAs process both image tokens (from the vision encoder) and language tokens (from
the instruction). By extracting attention weights from the final layers, we can see:
- Which image regions influence each action dimension
- Whether the model attends to task-relevant objects
- How attention shifts across different instructions

### 5.3 Starter Code

```python
"""Exercise 5 — VLA Attention Analysis

Visualize where the VLA looks when generating actions.

Usage:
    python 11_ex5_attention_analysis.py --image scene.png --instruction "pick up the red block"
"""
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple


def extract_attention_maps(
    processor, model, image: Image.Image, instruction: str
) -> Dict[str, np.ndarray]:
    """Run forward pass and extract cross-attention maps.

    Returns:
        Dictionary with keys:
            "image_attention": (n_layers, n_heads, n_action_tokens, n_image_patches)
            "text_attention":  (n_layers, n_heads, n_action_tokens, n_text_tokens)
    """
    # TODO 5a: Prepare inputs
    # inputs = processor(images=image, text=instruction, return_tensors="pt").to(model.device)

    # TODO 5b: Forward pass with attention output
    # with torch.no_grad():
    #     outputs = model(**inputs, output_attentions=True)
    # attentions = outputs.attentions  # list of (batch, n_heads, seq_len, seq_len)

    # TODO 5c: Separate image-to-action and text-to-action attention
    # Determine which token positions are image, text, and action tokens
    # n_image_patches = processor.image_seq_length  # e.g., 256 for 16×16 grid
    # n_text_tokens = len(processor.tokenizer(instruction)["input_ids"])
    # action_start = n_image_patches + n_text_tokens

    # TODO 5d: Extract and average
    # For the last 4 layers, extract attention from action tokens to image patches
    # image_attn = ...
    # text_attn = ...

    return {
        "image_attention": np.random.rand(4, 16, 7, 256),  # placeholder
        "text_attention": np.random.rand(4, 16, 7, 20),     # placeholder
    }


def visualize_image_attention(
    image: Image.Image,
    attention: np.ndarray,
    patch_grid: Tuple[int, int] = (16, 16),
    action_labels: List[str] = None,
):
    """Overlay attention heatmap on the image for each action dimension."""
    if action_labels is None:
        action_labels = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]

    # Average over layers and heads
    attn_avg = attention.mean(axis=(0, 1))  # (n_action_tokens, n_patches)
    n_actions = min(attn_avg.shape[0], 7)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    h, w = patch_grid
    img_array = np.array(image.resize((w * 14, h * 14)))  # match patch resolution

    for i in range(n_actions):
        ax = axes[i + 1]
        attn_map = attn_avg[i].reshape(h, w)

        # Upsample attention to image size
        from scipy.ndimage import zoom
        attn_upsampled = zoom(attn_map, (img_array.shape[0] / h, img_array.shape[1] / w))

        ax.imshow(img_array)
        ax.imshow(attn_upsampled, alpha=0.5, cmap="jet")
        ax.set_title(f"Action: {action_labels[i]}", fontsize=11)
        ax.axis("off")

    plt.suptitle("VLA Image Attention per Action Dimension", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("ex5_image_attention.png", dpi=150, bbox_inches="tight")
    plt.show()


def compare_attention_across_instructions(
    processor, model, image: Image.Image, instructions: List[str]
):
    """Show how attention shifts with different instructions on the same image."""
    fig, axes = plt.subplots(len(instructions), 8, figsize=(24, 4 * len(instructions)))

    action_labels = ["Δx", "Δy", "Δz", "Δroll", "Δpitch", "Δyaw", "gripper"]

    for row, inst in enumerate(instructions):
        maps = extract_attention_maps(processor, model, image, inst)
        attn = maps["image_attention"].mean(axis=(0, 1))  # (n_actions, n_patches)

        # Instruction label
        axes[row, 0].text(0.5, 0.5, inst, ha="center", va="center",
                          fontsize=9, wrap=True, transform=axes[row, 0].transAxes)
        axes[row, 0].axis("off")

        for col in range(min(7, attn.shape[0])):
            attn_map = attn[col].reshape(16, 16)
            axes[row, col + 1].imshow(attn_map, cmap="viridis")
            if row == 0:
                axes[row, col + 1].set_title(action_labels[col], fontsize=10)
            axes[row, col + 1].axis("off")

    plt.suptitle("Attention Comparison: Same Image, Different Instructions", fontsize=14)
    plt.tight_layout()
    plt.savefig("ex5_attention_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


def error_analysis(
    processor, model, dataset, n_samples: int = 50
) -> Dict[str, list]:
    """Categorize VLA failures by error type.

    Error categories:
        - wrong_object: attends to wrong object
        - wrong_action: correct object, wrong manipulation
        - spatial_error: correct intent, wrong direction/magnitude
        - gripper_error: everything right except gripper timing
    """
    results = {
        "wrong_object": [],
        "wrong_action": [],
        "spatial_error": [],
        "gripper_error": [],
        "success": [],
    }

    # TODO 5e: For each sample, run inference, compare to ground truth,
    # and categorize the error type based on attention analysis
    # For each failure:
    #   1. Check if attention peaks on the correct object → wrong_object
    #   2. Check if action direction matches instruction → wrong_action
    #   3. Check spatial magnitude → spatial_error
    #   4. Check gripper state → gripper_error

    print("\n=== Error Analysis Summary ===")
    for category, items in results.items():
        print(f"  {category}: {len(items)} samples")

    return results


def plot_error_distribution(results: Dict[str, list]):
    """Pie chart + bar chart of error categories."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts = {k: len(v) for k, v in results.items()}
    labels = list(counts.keys())
    values = list(counts.values())
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#9b59b6", "#2ecc71"]

    # Pie chart
    axes[0].pie(values, labels=labels, colors=colors, autopct="%1.0f%%", startangle=140)
    axes[0].set_title("Error Distribution")

    # Bar chart
    axes[1].barh(labels, values, color=colors)
    axes[1].set_xlabel("Count")
    axes[1].set_title("Error Categories")

    plt.tight_layout()
    plt.savefig("ex5_error_analysis.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="scene.png")
    parser.add_argument("--instruction", type=str, default="pick up the red block")
    args = parser.parse_args()

    # Placeholder — replace with real model loading
    processor, model = None, None

    image = Image.open(args.image).convert("RGB")

    print("=== Extracting Attention Maps ===")
    maps = extract_attention_maps(processor, model, image, args.instruction)
    visualize_image_attention(image, maps["image_attention"])

    print("\n=== Cross-Instruction Comparison ===")
    instructions = [
        "pick up the red block",
        "push the blue cube to the left",
        "stack blocks",
    ]
    compare_attention_across_instructions(processor, model, image, instructions)
```

### 5.4 Tasks

| Task | Description | Difficulty |
|------|-------------|------------|
| 5a–5d | Complete attention extraction from the VLA | ★★★ |
| 5e | Implement error categorization based on attention | ★★★ |
| 5f | Generate all three visualizations | ★★☆ |
| 5g | Find a case where high attention ≠ correct action — explain | ★★★ |
| 5h | Compare attention patterns of base vs fine-tuned model (from Ex3) | ★★★ |

### 5.5 Expected Output

- `ex5_image_attention.png` — 2×4 grid: original image + 7 action attention heatmaps
- `ex5_attention_comparison.png` — multi-row grid comparing instructions
- `ex5_error_analysis.png` — pie + bar chart of error categories

### 5.6 Reflection Questions

1. Does the VLA consistently attend to the correct object for each instruction?
2. How does attention differ between positional actions (Δx, Δy, Δz) and gripper?
3. When does attention analysis mislead you? (Hint: attention ≠ attribution)

---

## Self-Check Rubric

| Criterion | Meets Expectations | Needs Work |
|-----------|-------------------|------------|
| **Ex1**: VLA produces sensible actions for tabletop scenes | Actions correlate with instruction semantics; generalization plot shows differentiation | Same action regardless of instruction |
| **Ex2**: Tokenization comparison is complete | Log-log plot matches theoretical bound; both training curves converge | Missing theoretical curve or plot not log-scale |
| **Ex3**: LoRA fine-tuning improves success rate | ≥10 pp improvement over base; wandb shows smooth loss descent | Training diverges or no improvement |
| **Ex4**: Hybrid controller is smoother than pure VLA | Jerk metric is ≥2× better; trajectory plots visibly smoother | No measurable smoothness difference |
| **Ex5**: Attention maps reveal task-relevant patterns | Heatmaps focus on correct objects; error analysis categorizes ≥3 failure modes | Uniform attention or no error categorization |

### Completion Checklist

- [ ] Exercise 1: Inference runs, generalization plot saved
- [ ] Exercise 2: Both tokenization plots generated
- [ ] Exercise 3: LoRA fine-tuning completes, success rate measured
- [ ] Exercise 4: Hybrid vs pure comparison plot saved
- [ ] Exercise 5: Attention heatmaps and error analysis generated
- [ ] All 14 saved figures present

---

## Stretch Goals

### S1 — World Model Integration

Add a learned world model that predicts the next observation given current state
and VLA action. Use the prediction to:
- Detect impossible actions before execution
- Plan multi-step ahead via model-predictive control (MPC)
- Score action candidates by predicted outcome

### S2 — Multi-Task VLA

Fine-tune a single LoRA adapter on 3+ tasks simultaneously:
- Shared adapter vs per-task adapters
- Measure negative transfer between dissimilar tasks
- Plot per-task success rate vs training steps

### S3 — Real-Time Safety Monitor

Build a safety layer that:
- Monitors VLA action magnitude and rate of change
- Enforces workspace boundaries (joint limits, Cartesian bounds)
- Falls back to a safe stop if anomaly score > threshold
- Logs all interventions for post-hoc analysis

### S4 — VLA Distillation

Distill the 7B VLA into a 1B student model:
- Train student on teacher's soft action distributions
- Compare success rate, latency, and model size
- Find the Pareto frontier: performance vs inference speed

---

## References

1. Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model," 2024.
2. Brohan et al., "RT-2: Vision-Language-Action Models," 2023.
3. Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," ICLR 2022.
4. Cadene et al., "LeRobot: State-of-the-art Machine Learning for Robotics," 2024.
5. O'Neill et al., "Neural Network Controllers for Smooth Robot Motions," 2024.

---

*End of Exercise 11 — VLA Experiments*
