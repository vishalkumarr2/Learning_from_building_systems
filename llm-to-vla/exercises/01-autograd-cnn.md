# Exercise 01 — Autograd, CNNs & Training Stability
> Phase I · Days 1–7
> Prerequisites: study-notes/01-dl-foundations.md
> Tools: PyTorch, matplotlib, numpy

---

## Setup

```bash
pip install torch torchvision matplotlib numpy
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
```

---

## Exercise 1: Micrograd — Build Your Own Autograd (Day 1)

### Goal
Build a minimal autograd engine from scratch. Understand how gradients flow backward through a computation graph.

### Step 1: The Value Class

```python
class Value:
    """A scalar value with automatic differentiation."""

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # TODO: implement gradient for addition
            # Hint: d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += ___  # fill in
            other.grad += ___  # fill in
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # TODO: implement gradient for multiplication
            # Hint: d(a*b)/da = b, d(a*b)/db = a
            self.grad += ___  # fill in
            other.grad += ___  # fill in
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (np.exp(2*x) - 1) / (np.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            # TODO: derivative of tanh
            # Hint: d(tanh(x))/dx = 1 - tanh(x)^2
            self.grad += ___  # fill in
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')

        def _backward():
            # TODO: derivative of relu
            # Hint: 0 if x <= 0, else 1
            self.grad += ___  # fill in
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(np.exp(x), (self,), 'exp')

        def _backward():
            # TODO: derivative of exp
            # Hint: d(exp(x))/dx = exp(x)
            self.grad += ___  # fill in
        out._backward = _backward
        return out

    def backward(self):
        """Reverse-mode autodiff via topological sort."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    # Convenience operators
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
```

### Step 2: Build a Small MLP

```python
import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)

    def __call__(self, x):
        # TODO: compute w·x + b, then apply tanh
        act = ___  # sum of w[i]*x[i] + b
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

### Step 3: Train on XOR

```python
# XOR dataset
xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [-1.0, 1.0, 1.0, -1.0]  # using -1/1 for tanh

# TODO: Create MLP(2, [4, 4, 1]) and train for 200 steps
# Use MSE loss: sum((pred - target)^2 for each sample)
# Learning rate: 0.05
# Zero gradients before each forward pass!

model = MLP(2, [4, 4, 1])

for step in range(200):
    # Forward pass
    ypred = [model(x) for x in xs]
    loss = ___  # TODO: compute MSE loss

    # Backward pass
    for p in model.parameters():
        p.grad = 0.0  # zero grad
    loss.backward()

    # Update
    for p in model.parameters():
        p.data -= 0.05 * p.grad

    if step % 20 == 0:
        print(f"Step {step}: loss = {loss.data:.4f}")
```

### Step 4: Verify Against PyTorch

```python
def verify_gradients():
    """Compare your autograd with PyTorch."""
    # Your engine
    a = Value(2.0)
    b = Value(-3.0)
    c = a * b
    d = c + Value(10.0)
    e = d.tanh()
    e.backward()

    # PyTorch
    a_pt = torch.tensor(2.0, requires_grad=True)
    b_pt = torch.tensor(-3.0, requires_grad=True)
    c_pt = a_pt * b_pt
    d_pt = c_pt + 10.0
    e_pt = d_pt.tanh()
    e_pt.backward()

    print(f"a.grad: yours={a.grad:.6f}, pytorch={a_pt.grad.item():.6f}")
    print(f"b.grad: yours={b.grad:.6f}, pytorch={b_pt.grad.item():.6f}")
    assert abs(a.grad - a_pt.grad.item()) < 1e-6, "Gradient mismatch!"
    assert abs(b.grad - b_pt.grad.item()) < 1e-6, "Gradient mismatch!"
    print("✓ All gradients match to 6 decimal places!")

verify_gradients()
```

**Expected output:**
```
a.grad: yours=-2.999999, pytorch=-2.999999
b.grad: yours= 1.999999, pytorch= 1.999999
✓ All gradients match to 6 decimal places!
```

---

## Exercise 2: ResNet from Scratch (Day 2)

### Goal
Implement ResNet-18 and prove that skip connections solve the degradation problem.

### Step 1: Residual Block

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # TODO: implement two conv layers with batch norm
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # TODO: implement F(x) + x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += ___  # shortcut connection
        out = F.relu(out)
        return out
```

### Step 2: Build ResNet-18

```python
class ResNet(nn.Module):
    def __init__(self, num_classes=10, use_residuals=True):
        super().__init__()
        self.use_residuals = use_residuals
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            if self.use_residuals:
                layers.append(ResidualBlock(self.in_channels, out_channels, s))
            else:
                # Plain block without skip connections
                layers.append(self._plain_block(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _plain_block(self, in_ch, out_ch, stride):
        """Block WITHOUT residual connection for comparison."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
```

### Step 3: Train and Compare

```python
def train_resnet(use_residuals=True, epochs=20):
    """Train on CIFAR-10 and track gradient norms."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)

    model = ResNet(use_residuals=use_residuals).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    grad_norms = []

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Track gradient norms per layer
            norms = []
            for name, param in model.named_parameters():
                if param.grad is not None and 'conv' in name:
                    norms.append(param.grad.norm().item())
            grad_norms.append(np.mean(norms))

            optimizer.step()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        acc = 100. * correct / total
        print(f"Epoch {epoch}: Loss={total_loss/len(trainloader):.3f}, "
              f"Acc={acc:.1f}%")

    return grad_norms, acc

# Run experiment
print("=== With Residual Connections ===")
norms_res, acc_res = train_resnet(use_residuals=True)

print("\n=== Without Residual Connections ===")
norms_plain, acc_plain = train_resnet(use_residuals=False)

# Plot gradient norms
plt.figure(figsize=(10, 4))
plt.semilogy(norms_res[:500], label=f'ResNet (acc={acc_res:.1f}%)', alpha=0.7)
plt.semilogy(norms_plain[:500], label=f'PlainNet (acc={acc_plain:.1f}%)', alpha=0.7)
plt.xlabel('Training Step')
plt.ylabel('Mean Gradient Norm (log scale)')
plt.title('Gradient Flow: Residual vs Plain Networks')
plt.legend()
plt.savefig('gradient_norms_resnet.png', dpi=150)
plt.show()
```

**Expected output:**
```
=== With Residual Connections ===
Epoch 19: Loss=0.121, Acc=91.2%

=== Without Residual Connections ===
Epoch 19: Loss=0.482, Acc=72.4%
```

---

## Exercise 3: RNN vs LSTM Gradient Flow (Day 3)

### Goal
Prove that vanilla RNNs suffer from vanishing gradients and LSTMs solve it.

### Step 1: Character-Level Language Model

```python
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, rnn_type='rnn'):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)

        if rnn_type == 'rnn':
            self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        self.fc = nn.Linear(hidden_size, vocab_size)
        self.rnn_type = rnn_type

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden
```

### Step 2: Measure Gradient Norms Through Time

```python
def measure_gradient_flow(model, sequence, vocab_size):
    """
    Measure how gradients flow backward through time.
    Returns gradient norm at each timestep.
    """
    model.zero_grad()
    x = sequence[:, :-1]
    y = sequence[:, 1:]

    output, _ = model(x)
    loss = F.cross_entropy(output.reshape(-1, vocab_size), y.reshape(-1))
    loss.backward()

    # Extract gradient norms from the hidden-to-hidden weight
    grad_norms = []
    for name, param in model.named_parameters():
        if 'weight_hh' in name:
            # This approximates gradient flow through time
            grad_norms.append(param.grad.norm().item())

    return grad_norms, loss.item()


def experiment_sequence_lengths(text, seq_lengths=[50, 100, 200, 500]):
    """Compare RNN vs LSTM across sequence lengths."""
    # Prepare dataset
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    vocab_size = len(chars)
    encoded = torch.tensor([stoi[c] for c in text])

    results = {'rnn': {}, 'lstm': {}}

    for rnn_type in ['rnn', 'lstm']:
        for seq_len in seq_lengths:
            model = CharRNN(vocab_size, 128, rnn_type)

            # Train for 100 steps
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            losses = []

            for step in range(100):
                # Random batch
                ix = torch.randint(0, len(encoded) - seq_len - 1, (32,))
                batch = torch.stack([encoded[i:i+seq_len+1] for i in ix])

                x, y = batch[:, :-1], batch[:, 1:]
                output, _ = model(x)
                loss = F.cross_entropy(output.reshape(-1, vocab_size),
                                       y.reshape(-1))
                optimizer.zero_grad()
                loss.backward()

                # Clip gradients for RNN stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(loss.item())

            results[rnn_type][seq_len] = np.mean(losses[-20:])
            print(f"{rnn_type.upper()} seq_len={seq_len}: "
                  f"final loss={results[rnn_type][seq_len]:.3f}")

    # Plot
    plt.figure(figsize=(8, 5))
    for rnn_type in ['rnn', 'lstm']:
        plt.plot(seq_lengths, [results[rnn_type][s] for s in seq_lengths],
                 'o-', label=rnn_type.upper())
    plt.xlabel('Sequence Length')
    plt.ylabel('Training Loss (lower = better)')
    plt.title('RNN vs LSTM: Performance vs Sequence Length')
    plt.legend()
    plt.savefig('rnn_vs_lstm_seqlen.png', dpi=150)
    plt.show()

# Load Shakespeare (or any text)
import urllib.request, os
shakespeare_path = 'shakespeare.txt'
if not os.path.exists(shakespeare_path):
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    urllib.request.urlretrieve(url, shakespeare_path)
    print(f'Downloaded → {shakespeare_path}')
text = open(shakespeare_path).read()
# experiment_sequence_lengths(text)
```

**Expected output:**
```
RNN  seq_len=50:  final loss=2.12
RNN  seq_len=100: final loss=2.45
RNN  seq_len=200: final loss=2.89
RNN  seq_len=500: final loss=3.21
LSTM seq_len=50:  final loss=1.85
LSTM seq_len=100: final loss=1.91
LSTM seq_len=200: final loss=1.98
LSTM seq_len=500: final loss=2.15
```

---

## Exercise 4: Seq2Seq Bottleneck Visualization (Day 4)

### Goal
See the information bottleneck in fixed-size encoder-decoder architectures.

### Step 1: Number-to-Text Dataset

```python
# Dataset: "123" → "one hundred twenty three"
ONES = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
        'eight', 'nine', 'ten', 'eleven', 'twelve', 'thirteen',
        'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty',
        'sixty', 'seventy', 'eighty', 'ninety']

def number_to_words(n):
    """Convert integer to English words (0-9999)."""
    if n == 0:
        return 'zero'
    parts = []
    if n >= 1000:
        parts.append(ONES[n // 1000] + ' thousand')
        n %= 1000
    if n >= 100:
        parts.append(ONES[n // 100] + ' hundred')
        n %= 100
    if n >= 20:
        parts.append(TENS[n // 10])
        n %= 10
    if n > 0:
        parts.append(ONES[n])
    return ' '.join(parts).strip()


def create_dataset(max_digits=4, n_samples=10000):
    """Generate (number_string, word_string) pairs."""
    pairs = []
    for _ in range(n_samples):
        num = random.randint(0, 10**max_digits - 1)
        src = str(num)
        tgt = number_to_words(num)
        pairs.append((src, tgt))
    return pairs
```

### Step 2: Encoder-Decoder Model

```python
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        embedded = self.embed(x)
        _, hidden = self.rnn(embedded)
        return hidden  # This is the "bottleneck"


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embed(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.encoder = Encoder(src_vocab, embed_dim, hidden_dim)
        self.decoder = Decoder(tgt_vocab, embed_dim, hidden_dim)

    def forward(self, src, tgt):
        hidden = self.encoder(src)
        output, _ = self.decoder(tgt, hidden)
        return output
```

### Step 3: Measure Bottleneck Effect

```python
def measure_bottleneck(hidden_dim=128):
    """
    Train seq2seq and measure accuracy vs input length.
    """
    # TODO: Train the model and evaluate on inputs of varying length
    # Group test samples by input length
    # Measure: exact-match accuracy per length bucket

    results = {}
    for max_digits in [1, 2, 3, 4]:
        # Train on numbers up to max_digits
        # Evaluate exact match accuracy
        # Record hidden state norms at bottleneck
        pass  # implement training loop

    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    # plt.plot(lengths, accuracies)
    plt.xlabel('Input Length (digits)')
    plt.ylabel('Exact Match Accuracy')
    plt.title('Seq2Seq: Accuracy Degrades with Length')

    plt.subplot(1, 2, 2)
    # plt.plot(lengths, hidden_norms)
    plt.xlabel('Input Length (digits)')
    plt.ylabel('Hidden State Norm at Bottleneck')
    plt.title('Information Saturation in Fixed-Size Vector')

    plt.tight_layout()
    plt.savefig('seq2seq_bottleneck.png', dpi=150)
    plt.show()
```

**Expected output:**
```
1 digit:  accuracy=99.2%, hidden_norm=1.23
2 digits: accuracy=94.5%, hidden_norm=2.87
3 digits: accuracy=71.3%, hidden_norm=4.52
4 digits: accuracy=43.8%, hidden_norm=5.91
→ Quality degrades sharply for inputs longer than ~20 output tokens
```

---

## Exercise 5: Information Theory Lab (Day 5)

### Goal
Connect language modeling to information theory: compression, entropy, cross-entropy.

### Step 1: Compute Corpus Entropy

```python
from collections import Counter
import math

def char_entropy(text):
    """Compute empirical character-level entropy (bits)."""
    counts = Counter(text)
    total = len(text)
    entropy = 0.0
    for char, count in counts.items():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy

def bigram_entropy(text):
    """Compute bigram entropy (bits per character)."""
    bigrams = [text[i:i+2] for i in range(len(text)-1)]
    counts = Counter(bigrams)
    total = len(bigrams)
    entropy = 0.0
    for bigram, count in counts.items():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy / 2  # per character

# Test on different corpora
corpora = {
    'random': ''.join(random.choices('abcdefghijklmnopqrstuvwxyz ', k=10000)),
    'english': open(shakespeare_path).read()[:10000] if os.path.exists(shakespeare_path) else "to be or not to be " * 500,
    'repetitive': 'abcabc' * 1667,
}

print("Corpus Entropy (bits per character):")
print("-" * 40)
for name, text in corpora.items():
    h1 = char_entropy(text)
    h2 = bigram_entropy(text)
    print(f"{name:12s}: H1={h1:.3f}, H2={h2:.3f}")
```

### Step 2: Train Char-LM and Track BPC

```python
def train_char_lm_bpc(text, hidden_size=256, epochs=50):
    """
    Train character-level LM and measure bits-per-character over training.
    BPC = cross_entropy_loss / ln(2)
    """
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    vocab_size = len(chars)

    # Encode
    data = torch.tensor([stoi[c] for c in text])

    model = nn.Sequential(
        nn.Embedding(vocab_size, hidden_size),
        nn.LSTM(hidden_size, hidden_size, batch_first=True),
    )
    # Wrap for proper output
    class CharLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_size)
            self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, vocab_size)

        def forward(self, x):
            x = self.embed(x)
            out, _ = self.lstm(x)
            return self.fc(out)

    model = CharLM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    bpc_history = []
    seq_len = 100

    for epoch in range(epochs):
        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(data) - seq_len - 1, seq_len):
            x = data[i:i+seq_len].unsqueeze(0)
            y = data[i+1:i+seq_len+1].unsqueeze(0)

            output = model(x)
            loss = F.cross_entropy(output.squeeze(0), y.squeeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        bpc = avg_loss / math.log(2)  # Convert nats to bits
        bpc_history.append(bpc)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: BPC = {bpc:.3f}")

    # Plot BPC over training
    plt.figure(figsize=(8, 4))
    plt.plot(bpc_history)
    plt.axhline(y=1.5, color='r', linestyle='--', label='English entropy ~1.5 bpc')
    plt.xlabel('Epoch')
    plt.ylabel('Bits Per Character (BPC)')
    plt.title('Learning = Compression: BPC Decreases Over Training')
    plt.legend()
    plt.savefig('bpc_training.png', dpi=150)
    plt.show()

    return bpc_history
```

### Step 3: KL Divergence

```python
def kl_divergence(p, q):
    """
    Compute KL(P || Q) for discrete distributions.
    p, q are dicts mapping symbols to probabilities.
    """
    kl = 0.0
    for symbol in p:
        if p[symbol] > 0 and q.get(symbol, 0) > 0:
            kl += p[symbol] * math.log2(p[symbol] / q[symbol])
        elif p[symbol] > 0:
            return float('inf')  # Q assigns zero prob where P is nonzero
    return kl

# Example: compare English character distribution vs uniform
english_dist = Counter("to be or not to be that is the question")
total = sum(english_dist.values())
p = {c: count/total for c, count in english_dist.items()}
q = {c: 1/27 for c in 'abcdefghijklmnopqrstuvwxyz '}  # uniform

print(f"KL(English || Uniform) = {kl_divergence(p, q):.3f} bits")
print(f"KL(Uniform || English) = {kl_divergence(q, p):.3f} bits")
print("Note: KL divergence is asymmetric!")
```

**Expected output:**
```
Epoch 0:  BPC = 4.523
Epoch 10: BPC = 2.341
Epoch 20: BPC = 1.876
Epoch 30: BPC = 1.643
Epoch 40: BPC = 1.521
→ BPC converges toward ~1.5 for English text (Shannon's estimate)
```

---

## Exercise 6: Training Stability Experiments (Day 7)

### Goal
Deliberately break training in 3 ways, then build a monitoring/prevention system.

### Step 1: Training Monitor

```python
class TrainingMonitor:
    """Track training health metrics."""

    def __init__(self):
        self.grad_norms = []
        self.losses = []
        self.lrs = []
        self.weight_norms = []
        self.alerts = []

    def log_step(self, model, loss, lr):
        self.losses.append(loss)
        self.lrs.append(lr)

        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.grad_norms.append(total_norm)

        w_norm = sum(p.data.norm(2).item() ** 2
                     for p in model.parameters()) ** 0.5
        self.weight_norms.append(w_norm)

        # Alert conditions
        if total_norm > 100:
            self.alerts.append(('GRAD_EXPLOSION', len(self.losses)))
        if math.isnan(loss) or math.isinf(loss):
            self.alerts.append(('NAN_LOSS', len(self.losses)))
        if len(self.losses) > 10 and loss > 2 * np.mean(self.losses[-10:]):
            self.alerts.append(('LOSS_SPIKE', len(self.losses)))

    def plot(self, title='Training Health'):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(title)

        axes[0, 0].semilogy(self.grad_norms)
        axes[0, 0].set_title('Gradient Norms')
        axes[0, 0].set_xlabel('Step')

        axes[0, 1].plot(self.losses)
        axes[0, 1].set_title('Loss')
        axes[0, 1].set_xlabel('Step')

        axes[1, 0].plot(self.lrs)
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Step')

        axes[1, 1].plot(self.weight_norms)
        axes[1, 1].set_title('Weight Norms')
        axes[1, 1].set_xlabel('Step')

        for alert_type, step in self.alerts:
            for ax in axes.flat:
                ax.axvline(x=step, color='r', alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(f'training_monitor_{title.replace(" ", "_")}.png', dpi=150)
        plt.show()
```

### Step 2: Failure Mode 1 — Learning Rate Too High

```python
def experiment_lr_explosion():
    """Demonstrate gradient explosion from high learning rate."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(use_residuals=True).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=10.0)  # Way too high!
    criterion = nn.CrossEntropyLoss()
    monitor = TrainingMonitor()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

    for i, (inputs, targets) in enumerate(loader):
        if i >= 50:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if torch.isnan(loss):
            print(f"Step {i}: NaN loss detected! Training diverged.")
            break

        loss.backward()
        monitor.log_step(model, loss.item(), 10.0)
        optimizer.step()

    monitor.plot('Failure: LR Too High')
    return monitor
```

### Step 3: Failure Mode 2 — No Warmup on Transformer

```python
def experiment_no_warmup():
    """Show loss spike without learning rate warmup."""
    # Simple transformer
    model = nn.Transformer(d_model=128, nhead=4, num_encoder_layers=4,
                           num_decoder_layers=4, dim_feedforward=512)

    # High constant LR without warmup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    monitor = TrainingMonitor()

    for step in range(200):
        src = torch.randn(10, 32, 128)  # (seq_len, batch, d_model)
        tgt = torch.randn(10, 32, 128)
        output = model(src, tgt)
        loss = F.mse_loss(output, torch.zeros_like(output))

        optimizer.zero_grad()
        loss.backward()
        monitor.log_step(model, loss.item(), 1e-3)
        optimizer.step()

    monitor.plot('Failure: No Warmup')
    return monitor
```

### Step 4: Failure Mode 3 — FP16 Without Loss Scaling

```python
def experiment_fp16_no_scaling():
    """Demonstrate NaN from FP16 without proper loss scaling."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(use_residuals=True).to(device).half()  # FP16
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    monitor = TrainingMonitor()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

    nan_step = None
    for i, (inputs, targets) in enumerate(loader):
        if i >= 100:
            break
        inputs = inputs.to(device).half()
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # No loss scaling — gradients can underflow in FP16
        loss.backward()

        if any(torch.isnan(p.grad).any() for p in model.parameters()
               if p.grad is not None):
            print(f"Step {i}: NaN gradients — FP16 underflow!")
            nan_step = i
            break

        monitor.log_step(model, loss.item(), 0.1)
        optimizer.step()

    monitor.plot('Failure: FP16 No Loss Scaling')
    print(f"\nFix: Use torch.amp.GradScaler('cuda')")
    return monitor
```

### Step 5: Implement Proper Cosine Schedule with Warmup

```python
def cosine_lr_with_warmup(optimizer, step, total_steps,
                          warmup_steps=1000, max_lr=1e-3, min_lr=1e-5):
    """Cosine decay with linear warmup."""
    if step < warmup_steps:
        # Linear warmup
        lr = max_lr * step / warmup_steps
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def experiment_proper_training():
    """Demonstrate stable training with all fixes applied."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(use_residuals=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')  # FP16 loss scaling
    criterion = nn.CrossEntropyLoss()
    monitor = TrainingMonitor()

    total_steps = 1000
    warmup_steps = 100

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

    step = 0
    for inputs, targets in loader:
        if step >= total_steps:
            break

        lr = cosine_lr_with_warmup(optimizer, step, total_steps,
                                   warmup_steps=warmup_steps)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        monitor.log_step(model, loss.item(), lr)
        step += 1

    monitor.plot('Stable Training (All Fixes)')
    print("\n✓ Training completed without NaN or divergence")
    return monitor
```

### Training Stability Checklist

```
□ Learning rate warmup (linear, 5-10% of total steps)
□ Cosine or linear decay schedule
□ Gradient clipping (max_norm=1.0)
□ Loss scaling for FP16 (torch.amp.GradScaler)
□ Weight decay (AdamW, not Adam with L2)
□ Gradient accumulation for effective large batch
□ Monitor gradient norms every N steps
□ Alert on: NaN loss, grad norm > 100x mean, loss spike > 2x
```

---

## Self-Check Questions

Before moving on, verify you can answer these:

1. **Can you write a training loop from memory?**
   - Forward pass, loss computation, backward, optimizer step, zero grad

2. **Can you explain backprop through a computation graph?**
   - Chain rule, topological sort, accumulate gradients at each node

3. **Why do residual connections help?**
   - Gradient flows directly through skip connection (gradient highway)
   - Loss surface becomes smoother, easier to optimize

4. **What does lower BPC mean?**
   - Better compression = model captures more structure
   - Theoretical minimum ≈ true entropy of the source

5. **What are the three most common training instabilities?**
   - Learning rate too high → gradient explosion
   - No warmup → early loss spikes
   - FP16 without scaling → gradient underflow

---

## Stretch Goals

### A: Implement GRU Cell from Scratch

```python
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        # TODO: Implement update gate, reset gate, candidate hidden state
        # z_t = σ(W_z·[h_{t-1}, x_t])
        # r_t = σ(W_r·[h_{t-1}, x_t])
        # h̃_t = tanh(W·[r_t * h_{t-1}, x_t])
        # h_t = (1 - z_t) * h_{t-1} + z_t * h̃_t
        pass

    def forward(self, x, h_prev):
        pass
```

### B: Add Dropout and BatchNorm to ResNet

```python
# Modify ResidualBlock to include:
# - Dropout after each ReLU (p=0.1)
# - Compare: with vs without dropout on CIFAR-10
# - Observe: does dropout help more on smaller datasets?
```

### C: Implement Your Own BPE Tokenizer

```python
def train_bpe(text, vocab_size=1000):
    """
    Byte-Pair Encoding from scratch.
    1. Start with character-level vocabulary
    2. Find most frequent adjacent pair
    3. Merge that pair into a new token
    4. Repeat until vocab_size reached
    """
    # TODO: implement the BPE merge loop
    # Track: how does vocab size affect BPC?
    pass
```

---

## Summary

| Exercise | Key Insight | Metric |
|----------|-------------|--------|
| 1. Micrograd | Backprop = chain rule + topo sort | Gradient match < 1e-6 |
| 2. ResNet | Skip connections = gradient highways | +20% accuracy on deep nets |
| 3. RNN/LSTM | Gates prevent vanishing gradients | LSTM stable at 500 steps |
| 4. Seq2Seq | Fixed-size bottleneck limits capacity | Accuracy drops at len > 20 |
| 5. Info Theory | Learning = compression | BPC → 1.5 for English |
| 6. Stability | Monitor + warmup + scaling = stable | Zero NaN, smooth curves |

**Next:** → [Exercise 02: Attention & Transformers](02-attention-transformers.md)
