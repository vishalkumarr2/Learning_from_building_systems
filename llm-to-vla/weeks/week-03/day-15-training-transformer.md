# Day 15: Training a Transformer

> **Phase II — Attention, Transformers & Scaling | Week 3 | 2.5 hours**
> *"The transformer is easy to define but tricky to train. The details of optimization matter enormously."*

## Navigation
- **Previous:** [Day 14: The Full Transformer](../week-02/day-14-full-transformer.md)
- **Next:** [Day 16: Stop & Reflect #1](day-16-stop-reflect-1.md)
- **Week:** [Week 3 Overview](../README.md)
- **Phase:** [Phase II: Attention & Transformers](../../study-notes/02-attention-mechanism.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (45 min)

### 15.1 Why Transformers Are Hard to Train

The transformer's power comes from attention, but attention creates optimization challenges:

1. **Softmax saturation:** Without careful initialization and scaling, attention weights can become nearly one-hot, killing gradients
2. **Loss landscape curvature:** The loss surface has sharp cliffs and flat valleys; large learning rates cause divergence
3. **Gradient magnification:** In Post-LN transformers, gradients at early layers can explode during the first few steps

The result: transformers require **learning rate warmup**, **careful initialization**, and often **label smoothing** to train well.

### 15.2 The Noam Learning Rate Schedule

The original transformer paper introduced a specific schedule with linear warmup followed by inverse-square-root decay:

$$\text{lr}(t) = d_{\text{model}}^{-0.5} \cdot \min\!\left(t^{-0.5}, \; t \cdot t_{\text{warmup}}^{-1.5}\right)$$

where $t$ is the step number and $t_{\text{warmup}}$ is the number of warmup steps (typically 4000).

```
Learning rate
    │
    │           ╱╲
    │          ╱  ╲
    │         ╱    ╲
    │        ╱      ╲────────────
    │       ╱                     ╲───────
    │      ╱                               ╲───
    │     ╱
    │    ╱
    │   ╱
    │──╱
    └──────────────────────────────────────── step
       ↑                    ↑
    warmup            peak (at t_warmup)
```

**Why warmup?** At initialization, attention weights are near-uniform (good), but without warmup the optimizer takes large steps that push attention into sharp distributions, causing training instability. Warmup lets the model find a stable region before increasing the learning rate.

### 15.3 Modern Schedules: Warmup + Cosine Decay

Modern practice (GPT-3, LLaMA, Chinchilla) uses:

$$\text{lr}(t) = \begin{cases} \text{lr}_{\max} \cdot \frac{t}{t_{\text{warmup}}} & t \leq t_{\text{warmup}} \\ \text{lr}_{\min} + \frac{1}{2}(\text{lr}_{\max} - \text{lr}_{\min})(1 + \cos(\pi \cdot \frac{t - t_{\text{warmup}}}{T - t_{\text{warmup}}})) & t > t_{\text{warmup}} \end{cases}$$

Typical values:
- $\text{lr}_{\max} = 3 \times 10^{-4}$ (for AdamW with $d_{\text{model}} = 512$)
- $\text{lr}_{\min} = \text{lr}_{\max} / 10$
- $t_{\text{warmup}} = 2000$ steps
- $T$ = total training steps

### 15.4 Label Smoothing

Standard cross-entropy trains the model to output a one-hot distribution: $P(\text{correct}) = 1.0$. This can cause **overconfidence**.

Label smoothing replaces the one-hot target with:

$$y_{\text{smooth}}(k) = \begin{cases} 1 - \epsilon & k = \text{correct class} \\ \frac{\epsilon}{K - 1} & k \neq \text{correct class} \end{cases}$$

where $\epsilon$ is typically 0.1 and $K$ is the vocabulary size.

**Effect:** The model learns to be slightly uncertain, which:
- Improves generalization
- Better calibrates probabilities
- Prevents the model from putting infinite logit mass on one token

### 15.5 Masked Loss for Padding

Batched sequences have different lengths, so we pad shorter sequences. We must **not** compute loss on padding tokens:

```
Tokens:  [BOS] The cat sat [EOS] [PAD] [PAD]
Mask:      1    1   1   1    1     0     0
Loss:      ✓    ✓   ✓   ✓    ✓     ✗     ✗
```

### 15.6 Adam vs AdamW

The transformer paper uses Adam with $\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-9}$.

Modern practice uses **AdamW** (decoupled weight decay), which separates the L2 regularization from the adaptive gradient:

$$\theta_{t+1} = \theta_t - \text{lr} \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$

Key difference: AdamW applies weight decay *directly* to the parameters, not through the gradient. With Adam, weight decay interacts poorly with the adaptive learning rate; AdamW fixes this.

---

## Implementation (60 min)

### 15.7 Learning Rate Schedulers

```python
import torch
import torch.optim as optim
import math


class NoamScheduler:
    """Original transformer learning rate schedule."""
    
    def __init__(self, optimizer: optim.Optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


class WarmupCosineScheduler:
    """Modern warmup + cosine decay schedule."""
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        lr_max: float = 3e-4,
        lr_min: float = 3e-5,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup_steps:
            lr = self.lr_max * self.step_num / self.warmup_steps
        else:
            progress = (self.step_num - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
```

### 15.8 Label Smoothing Loss

```python
class LabelSmoothingLoss(torch.nn.Module):
    """Cross-entropy with label smoothing and padding mask."""
    
    def __init__(self, vocab_size: int, padding_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: (batch * seq_len, vocab_size)
        target: (batch * seq_len,)
        """
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Smooth targets
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.sum(dim=-1) / self.vocab_size
        
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        # Mask out padding
        mask = target != self.padding_idx
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
```

### 15.9 Full Training Loop

```python
def train_transformer(
    model,
    train_loader,
    val_loader,
    epochs: int = 30,
    warmup_steps: int = 4000,
    lr_max: float = 3e-4,
    label_smoothing: float = 0.1,
    pad_idx: int = 0,
):
    """Train a transformer on a translation task."""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr_max,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=0.01,
    )
    
    total_steps = epochs * len(train_loader)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, lr_max)
    criterion = LabelSmoothingLoss(model.output_proj.out_features, pad_idx, label_smoothing)
    
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (src, tgt) in enumerate(train_loader):
            tgt_input = tgt[:, :-1]   # shift right: teacher forcing
            tgt_output = tgt[:, 1:]   # expected output
            
            # Create causal mask
            tgt_len = tgt_input.size(1)
            causal_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
            causal_mask = causal_mask.to(src.device)
            
            logits = model(src, tgt_input, tgt_mask=causal_mask)
            
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            lr = scheduler.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        history['lr'].append(lr)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                tgt_len = tgt_input.size(1)
                causal_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool()
                
                logits = model(src, tgt_input, tgt_mask=causal_mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
                val_loss += loss.item()
        
        avg_val = val_loss / len(val_loader)
        history['val_loss'].append(avg_val)
        
        print(f"Epoch {epoch+1}/{epochs} | Train: {avg_loss:.4f} | Val: {avg_val:.4f} | LR: {lr:.2e}")
    
    return history
```

### 15.10 Plotting Training Curves

```python
import matplotlib.pyplot as plt


def plot_training(history: dict):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    
    axes[1].plot(history['lr'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()
```

---

## Exercise (45 min)

### E15.1 Schedule Comparison

Implement and plot three schedules for 100,000 steps:
1. **Noam** (warmup=4000, $d_{\text{model}}=512$)
2. **Warmup + cosine** (warmup=2000, $\text{lr}_{\max}=3 \times 10^{-4}$)
3. **Warmup + linear decay**

Plot them on the same axes. Which reaches the highest peak? Which decays fastest?

### E15.2 Transformer vs Seq2Seq Comparison

Using Multi30k EN→DE:
1. Train your Day 14 transformer (3 layers, $d=256$, $h=4$)
2. Compare with Day 4 seq2seq (same hidden dim)
3. Report: final BLEU, convergence speed (epochs to 90% of best BLEU), total training time
4. Plot attention heatmaps from both models for the same sentence

### E15.3 Ablation Studies

Starting from a working transformer:
1. **Remove warmup** — train with constant learning rate. What happens?
2. **Remove label smoothing** — use hard targets. Compare calibration (reliability diagrams)
3. **Vary number of heads**: $h \in \{1, 4, 8\}$ with $d=256$. Which works best?
4. **Vary depth**: $N \in \{1, 2, 4, 6\}$ layers. Plot BLEU vs parameter count

---

## Key Takeaways

- Transformers require **warmup** to avoid early training instability from attention saturation
- The **Noam schedule** ($d^{-0.5} \cdot \min(t^{-0.5}, t \cdot t_w^{-1.5})$) is the classic; cosine decay is the modern standard
- **Label smoothing** ($\epsilon = 0.1$) prevents overconfidence and improves generalization
- **Masked loss** ensures padding tokens don't pollute the gradient signal
- **AdamW** with $\beta_1=0.9, \beta_2=0.98$ and gradient clipping ($\text{max\_norm}=1.0$) is the standard optimizer setup

## Connection to the Thread

Training a transformer is an exercise in **controlling information flow through the optimization landscape**. Warmup controls the *rate* of learning — starting slow to avoid catastrophic early steps. Label smoothing controls the *precision* of targets — accepting that language is inherently ambiguous, exact one-hot targets are a worse compression of the true conditional distribution $P(\text{next token} \mid \text{context})$. These tricks are not hacks — they reflect the fundamental nature of language: uncertain, contextual, and gradual to learn.

## Further Reading

- Vaswani et al. **"Attention Is All You Need"** (2017), Section 5.3 (Optimizer). [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Loshchilov & Hutter. **"Decoupled Weight Decay Regularization"** (2019). [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) — AdamW
- Szegedy et al. **"Rethinking the Inception Architecture"** (2016) — original label smoothing paper
- Liu et al. **"On the Variance of the Adaptive Learning Rate and Beyond"** (2020). [arXiv:1908.03265](https://arxiv.org/abs/1908.03265) — RAdam, warmup analysis
- The Annotated Transformer (Harvard NLP). [nlp.seas.harvard.edu](https://nlp.seas.harvard.edu/annotated-transformer/) — reference implementation
