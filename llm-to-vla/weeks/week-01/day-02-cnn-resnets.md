# Day 2: CNNs & ResNets

> **Phase I · Week 1 · Day 2 of 112 · 2.5 hours**

*"The residual connection is the single most important architectural idea in deep learning. Without it, nothing deeper than ~20 layers would train. Transformers, ViTs, VLAs — all use it."*

---

## Navigation

| | |
|---|---|
| **Previous** | [← Day 1: Computation Graphs & Backprop](day-01-computation-graphs-backprop.md) |
| **Next** | [Day 3: RNN/LSTM Essentials →](day-03-rnn-lstm-essentials.md) |
| **Week** | [Week 1: DL Foundations](README.md) |
| **Phase** | [Phase I: DL Foundations](../../study-notes/01-dl-foundations.md) |
| **Curriculum** | [Full Curriculum](../../CURRICULUM.md) |

---

## Why This Matters

Yesterday you built the gradient engine. Today you'll see why naively stacking layers *breaks* it. CNNs introduced feature hierarchies — the idea that early layers detect edges, middle layers detect parts, and deep layers detect objects. But going deeper than ~20 layers caused *degradation* — deeper networks performed *worse* than shallower ones, even on training data.

ResNets solved this with a shockingly simple idea: skip connections. This one trick enabled networks of 100+ layers and became the DNA of every modern architecture. When you study Vision Transformers (Day 45), you'll see the same residual connections. When you study VLAs (Day 92+), every action prediction head uses them. Residuals are non-negotiable.

---

## 1. Theory: Convolutional Neural Networks

### 1.1 Convolution as Learnable Feature Extraction

A convolution slides a learnable **kernel** (filter) across the input, computing dot products:

$$\text{output}(i, j) = \sum_{m}\sum_{n} \text{input}(i+m, j+n) \cdot \text{kernel}(m, n) + \text{bias}$$

```
Input (5×5)          Kernel (3×3)         Output (3×3)
┌─────────────┐      ┌─────────┐         ┌─────────┐
│ 1  0  1  0  1│      │ 1  0  1 │         │ ?  ?  ? │
│ 0  1  0  1  0│      │ 0  1  0 │  ──▶    │ ?  ?  ? │
│ 1  0  1  0  1│      │ 1  0  1 │         │ ?  ?  ? │
│ 0  1  0  1  0│      └─────────┘         └─────────┘
│ 1  0  1  0  1│
└─────────────┘
```

**Key properties**:
- **Parameter sharing**: Same kernel across all spatial positions → translation equivariance
- **Locality**: Each output depends only on a local patch → inductive bias for spatial data
- **Depth**: Stack layers → grow the **receptive field** (area of input that affects each output)

### 1.2 Key Hyperparameters

| Parameter | Effect | Formula |
|-----------|--------|---------|
| Kernel size $k$ | Local patch size | — |
| Stride $s$ | Step size | Reduces spatial dims |
| Padding $p$ | Border handling | Preserves spatial dims if $p = \lfloor k/2 \rfloor$ |
| Dilation $d$ | Gaps in kernel | Larger receptive field without more params |

**Output size formula**:

$$H_{\text{out}} = \left\lfloor \frac{H_{\text{in}} + 2p - d(k - 1) - 1}{s} + 1 \right\rfloor$$

### 1.3 The Feature Hierarchy

CNNs learn a hierarchy of increasingly abstract features:

```
Layer 1:   Edges, gradients, colors
              ╱  ╲
Layer 2:   Corners, textures, simple shapes
              ╱  ╲
Layer 3:   Parts (eyes, wheels, handles)
              ╱  ╲
Layer 4:   Objects (faces, cars, cups)
              ╱  ╲
Layer 5:   Scenes, contexts, relationships
```

This was first visualized by Zeiler & Fergus (2013) — each layer learns progressively more complex features by *composing* simpler ones from the layer below.

---

## 2. Theory: The Depth Problem

### 2.1 The Degradation Problem

Before ResNets (2015), practitioners observed a paradox:

| Network Depth | Training Error | Expected | Actual |
|--------------|----------------|----------|--------|
| 20 layers | Low | — | — |
| 56 layers | Higher | Lower (more capacity) | **Higher** (degradation!) |

This is **not** overfitting (training error goes up, not just test error). A 56-layer network should be able to represent a 20-layer network by setting the extra layers to identity — but SGD *can't find* that solution.

### 2.2 Why Plain Deep Networks Fail

The gradient of a deep network is a product of many Jacobians:

$$\frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial x_L} \prod_{\ell=1}^{L-1} \frac{\partial x_{\ell+1}}{\partial x_\ell}$$

If each factor's spectral norm is:
- **< 1**: gradients vanish exponentially → early layers don't learn
- **> 1**: gradients explode exponentially → training diverges
- **= 1**: perfect, but impossible to maintain through nonlinearities

---

## 3. Theory: Residual Connections

### 3.1 The ResNet Idea

Instead of learning $H(x)$ directly, learn the **residual** $F(x) = H(x) - x$:

$$y = F(x) + x$$

```
        x ─────────────────────┐
        │                      │ (skip / shortcut)
        ▼                      │
   ┌─────────┐                 │
   │ Conv 3×3 │                │
   │ BN + ReLU│                │
   └────┬─────┘                │
        │                      │
   ┌────▼─────┐                │
   │ Conv 3×3 │                │
   │   BN     │                │
   └────┬─────┘                │
        │                      │
        ▼                      ▼
       (+) ◀───────────────────┘
        │
      ReLU
        │
        ▼
     output
```

### 3.2 Three Perspectives on Why Residuals Work

**Perspective 1: Gradient Highway**

The gradient through a residual block:

$$\frac{\partial y}{\partial x} = \frac{\partial F(x)}{\partial x} + I$$

The identity term $I$ creates a **gradient highway** — gradients can flow directly to early layers regardless of how deep the network is. Even if $\frac{\partial F}{\partial x} \approx 0$, the gradient is at least $I$.

Through $L$ residual blocks:

$$\frac{\partial L}{\partial x_0} = \frac{\partial L}{\partial x_L} \prod_{\ell=0}^{L-1}\left(I + \frac{\partial F_\ell}{\partial x_\ell}\right)$$

Expanding this product yields $2^L$ terms — gradients flow through **exponentially many paths** of different lengths.

**Perspective 2: Ensemble of Shallow Networks**

Veit et al. (2016) showed that a ResNet of depth $L$ behaves like an **ensemble of $2^L$ shallow networks** of varying depths. Removing any single layer barely hurts performance, unlike a plain network where removing a layer is catastrophic.

**Perspective 3: Making Identity Easy**

If the optimal mapping is close to identity (common in deep networks), learning $F(x) \approx 0$ is much easier than learning $H(x) \approx x$ from scratch. The skip connection provides the right **prior**.

### 3.3 ResNet Block Variants

```python
import torch.nn as nn

class BasicBlock(nn.Module):
    """ResNet basic block (used in ResNet-18, ResNet-34)."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut for dimension mismatch
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity  # THE residual connection
        out = nn.functional.relu(out)
        return out
```

---

## 4. Implementation: ResNet-18 from Scratch

Build a complete ResNet-18 for CIFAR-10 — no `torchvision.models`.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Initial conv (CIFAR-10: 32×32, so use 3×3 instead of 7×7)
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 4 groups of residual blocks
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))  # 32×32
        x = self.layer1(x)                    # 32×32
        x = self.layer2(x)                    # 16×16
        x = self.layer3(x)                    # 8×8
        x = self.layer4(x)                    # 4×4
        x = self.avg_pool(x)                  # 1×1
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Count parameters
model = ResNet18()
n_params = sum(p.numel() for p in model.parameters())
print(f"ResNet-18 parameters: {n_params:,}")  # ~11.2M
```

### 4.1 Training on CIFAR-10

```python
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

transform_train = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True,
                                         transform=transform_train)
testset = torchvision.datasets.CIFAR10('./data', train=False,
                                        transform=transform_test)

train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                            weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
criterion = nn.CrossEntropyLoss()

for epoch in range(200):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    scheduler.step()
    
    if (epoch + 1) % 20 == 0:
        acc = 100. * correct / total
        print(f"Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.3f} | "
              f"Acc: {acc:.1f}%")
```

### 4.2 Comparison: Plain Network vs ResNet

```python
class PlainNet(nn.Module):
    """Same architecture but WITHOUT skip connections."""
    
    def __init__(self, num_classes=10, depth=18):
        super().__init__()
        layers = [nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU()]
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

---

## 5. Exercises

### Exercise 1: Gradient Norm Experiment

Compare gradient norms in a 50-layer plain network vs a 50-layer ResNet:

```python
def measure_gradient_norms(model, x, y):
    """Return per-layer gradient norms after one forward+backward pass."""
    out = model(x)
    loss = F.cross_entropy(out, y)
    loss.backward()
    
    norms = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.grad is not None and param.dim() >= 2:
            norms.append((name, param.grad.norm().item()))
    return norms
```

**Expected observation**: Plain network gradient norms decay exponentially from output to input. ResNet gradient norms stay relatively stable across all layers.

### Exercise 2: Parameter Counting

Count parameters in ResNet-18 layer by layer. Fill in:

| Component | Shape | Parameters |
|-----------|-------|------------|
| conv1 | 3→64, 3×3 | ? |
| layer1 block0 conv1 | 64→64, 3×3 | ? |
| layer1 block0 conv2 | 64→64, 3×3 | ? |
| ... | ... | ... |
| fc | 512→10 | ? |
| **Total** | | **?** |

### Exercise 3: Receptive Field Calculation

For ResNet-18 on CIFAR-10, what is the **theoretical receptive field** at the output of each layer group? Use the formula:

$$r_\ell = r_{\ell-1} + (k_\ell - 1) \times \prod_{i=1}^{\ell-1} s_i$$

Does it cover the entire 32×32 input?

### Exercise 4: Visualization

After training, visualize the first-layer convolutional filters (64 filters of shape 3×3×3). What patterns do you see? Compare with known Gabor-like edge detectors.

---

## 6. Key Takeaways

- CNNs exploit **spatial locality** and **translation equivariance** through shared kernels
- Deeper networks learn more abstract feature hierarchies, but **degradation** prevents plain networks from going beyond ~20 layers
- The **residual connection** $y = F(x) + x$ solves degradation by creating gradient highways
- Gradient through a residual block: $\frac{\partial y}{\partial x} = \frac{\partial F}{\partial x} + I$ — the identity term guarantees gradient flow
- ResNets behave as **ensembles of $2^L$ shallow networks** — robust to layer removal
- Batch normalization stabilizes training by normalizing activations
- Skip connections appear in *every* modern architecture: Transformers, ViTs, U-Nets, VLAs

---

## 7. The Thread: Compression = Prediction = Intelligence

Residual connections encode a powerful **prior**: most of the information should pass through unchanged. The network only needs to learn the *delta* — what to add or subtract. This is a form of **compression**: instead of re-encoding all information at every layer, you only encode the *new* information.

This is exactly what differential coding does in signal processing, what residual learning does in video compression (P-frames), and what LoRA will do for efficient fine-tuning (Day 35). The pattern repeats: **compress the change, not the whole signal.**

---

## 8. Further Reading

- **He et al., "Deep Residual Learning for Image Recognition"** (2015) — The original ResNet paper
- **Veit et al., "Residual Networks Behave Like Ensembles of Relatively Shallow Networks"** (2016) — The ensemble interpretation
- **He et al., "Identity Mappings in Deep Residual Networks"** (2016) — Pre-activation ResNets
- **Zeiler & Fergus, "Visualizing and Understanding Convolutional Networks"** (2013) — Feature visualization
- **Kaiming initialization** — `torch.nn.init.kaiming_normal_` — essential for training deep networks
