# Manual Calculations — Pen-and-Paper Exercises

> **Purpose**: Build deep mathematical intuition before (or alongside) coding.
> Each section maps to a specific exercise. Do these with pen, paper, and a calculator.
> Then verify your results match the code.

---

## Phase I: Deep Learning Foundations

### MC-1: Backpropagation Through a Mini-Network (Exercise 01)

**Setup**: A 2-layer network with weights:
```
Input x = [2.0, -1.0]
W1 = [[0.5, -0.3],    b1 = [0.1, 0.0]
       [0.8,  0.2]]
W2 = [[0.4, -0.6]]    b2 = [0.2]
Activation: ReLU
Loss: MSE with target y = 1.0
```

**Tasks**:
1. Compute the forward pass by hand:
   - $h_1^{pre} = W_1 x + b_1 = ?$
   - $h_1 = \text{ReLU}(h_1^{pre}) = ?$
   - $\hat{y} = W_2 h_1 + b_2 = ?$
   - $L = \frac{1}{2}(\hat{y} - y)^2 = ?$

2. Compute all gradients backward:
   - $\frac{\partial L}{\partial \hat{y}} = ?$
   - $\frac{\partial L}{\partial W_2} = ?$ (use chain rule)
   - $\frac{\partial L}{\partial h_1} = ?$
   - $\frac{\partial L}{\partial h_1^{pre}} = ?$ (ReLU derivative!)
   - $\frac{\partial L}{\partial W_1} = ?$ (outer product)
   - $\frac{\partial L}{\partial x} = ?$ (why does this matter?)

3. Apply one SGD step with $\eta = 0.01$. Write the new weights.

**Verification**: Code this in PyTorch with `requires_grad=True`, call `.backward()`, compare every gradient to your hand calculation. They must match exactly (float precision aside).

---

### MC-2: Softmax Numerical Stability (Exercise 02)

**Setup**: Logits $z = [1000, 1001, 1002]$

**Tasks**:
1. Compute $\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ directly:
   - $e^{1000} = ?$ → What happens? (overflow)
   
2. Apply the max-subtract trick: $z' = z - \max(z) = [0, 1, 2]$
   - $\text{softmax}(z')_0 = \frac{e^0}{e^0 + e^1 + e^2} = \frac{1}{1 + 2.718 + 7.389} = ?$
   - Compute all three values.

3. Prove mathematically that $\text{softmax}(z) = \text{softmax}(z - c)$ for any constant $c$.

4. **Log-sum-exp trick**: Show that $\log \sum_i e^{z_i} = \max(z) + \log \sum_i e^{z_i - \max(z)}$

**Verification**: In Python, compute `torch.softmax(torch.tensor([1000., 1001., 1002.]), dim=0)` and confirm it uses this trick internally (check that it doesn't produce `nan`).

---

## Phase II: Attention & Transformers

### MC-3: Attention Weights by Hand (Exercise 02)

**Setup**: Single-head attention with $d_k = 2$:
```
Q = [[1, 0],    (2 queries)
     [0, 1]]

K = [[1, 0],    (3 keys)
     [0, 1],
     [1, 1]]

V = [[1, 0, 0],  (3 values, d_v = 3)
     [0, 1, 0],
     [0, 0, 1]]
```

**Tasks**:
1. Compute $QK^T$:
   $$QK^T = \begin{bmatrix} ? & ? & ? \\ ? & ? & ? \end{bmatrix}$$

2. Scale by $\frac{1}{\sqrt{d_k}} = \frac{1}{\sqrt{2}} \approx 0.707$:
   - Scaled scores = ?

3. Apply softmax row-wise:
   - Row 1: $\text{softmax}([0.707, 0, 0.707]) = ?$
   - Row 2: $\text{softmax}([0, 0.707, 0.707]) = ?$

4. Compute output $= \text{softmax}(\frac{QK^T}{\sqrt{d_k}}) \cdot V$:
   - Output row 1 = weighted sum of values = ?
   - Output row 2 = ?

5. **Interpretation**: What does query 1 "attend to"? Why?

**Verification**: Implement with `torch.nn.functional.scaled_dot_product_attention` and compare.

---

### MC-4: Positional Encoding Values (Exercise 03)

**Setup**: Sinusoidal positional encoding with $d_{model} = 4$

$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**Tasks**:
1. Compute $PE(0, :)$: (pos=0, all dimensions)
   - $PE(0,0) = \sin(0/10000^{0/4}) = \sin(0) = 0$
   - $PE(0,1) = \cos(0/10000^{0/4}) = \cos(0) = 1$
   - $PE(0,2) = \sin(0/10000^{2/4}) = ?$
   - $PE(0,3) = \cos(0/10000^{2/4}) = ?$

2. Compute $PE(1, :)$ and $PE(100, :)$

3. Compute the dot product $PE(1) \cdot PE(2)$ and $PE(1) \cdot PE(100)$. Which is larger? Why does this encode relative position?

4. Show that there exists a linear transformation $M$ such that $PE(pos+k) = M \cdot PE(pos)$ for fixed $k$. (Hint: rotation matrix in 2D subspaces.)

---

### MC-5: Multi-Head Attention Parameter Count (Exercise 03)

**Setup**: Transformer with $d_{model} = 512$, $h = 8$ heads, $d_k = d_v = 64$

**Tasks**:
1. Count parameters in one attention head:
   - $W_Q^i$: $512 \times 64 = ?$ params
   - $W_K^i$: ? params
   - $W_V^i$: ? params
   - Total per head: ?

2. Count total multi-head attention parameters:
   - 8 heads × params_per_head = ?
   - Output projection $W_O$: $512 \times 512 = ?$
   - **Total MHA**: ?

3. Count parameters in the FFN (with $d_{ff} = 2048$):
   - $W_1$: $512 \times 2048 = ?$
   - $W_2$: $2048 \times 512 = ?$
   - **Total FFN**: ?

4. One transformer layer = MHA + FFN = ? parameters
5. GPT-2 small (12 layers, $d_{model}=768$, $h=12$): estimate total params. Compare to published 117M.

---

## Phase III: LLMs

### MC-6: LoRA Rank Decomposition (Exercise 05)

**Setup**: Original weight matrix $W \in \mathbb{R}^{4 \times 4}$, LoRA rank $r = 2$:
```
W = [[1.0, 0.5, -0.3, 0.8],
     [0.2, 1.1, 0.4, -0.6],
     [-0.5, 0.3, 0.9, 0.1],
     [0.7, -0.2, 0.6, 1.2]]

A = [[0.1, -0.2],     # shape (4, 2)
     [0.3, 0.1],
     [-0.1, 0.4],
     [0.2, -0.3]]

B = [[0.5, 0.1, -0.3, 0.2],   # shape (2, 4)
     [-0.1, 0.4, 0.2, -0.5]]
```

**Tasks**:
1. Compute $\Delta W = A \cdot B$ (4×4 matrix):
   - Row 1 of $\Delta W$: $[0.1 \times 0.5 + (-0.2)(-0.1), ...]$ = ?
   - Complete all rows.

2. Compute updated weight: $W_{new} = W + \alpha \cdot \Delta W$ with $\alpha = 1.0$

3. How many parameters in LoRA vs full fine-tuning?
   - LoRA: $4 \times 2 + 2 \times 4 = ?$
   - Full: $4 \times 4 = 16$
   - Savings: ?%

4. What is `rank(ΔW)`? Why can't LoRA capture full-rank updates?

5. For GPT-2 (768×768 attention weights): LoRA-8 params vs full? Savings?

---

### MC-7: BPE Merge Steps (Exercise 04)

**Setup**: Vocabulary starts with characters: `{a, b, c, d, ' '}`. Corpus:
```
"abab cdcd abab cdcd abab"
```

**Tasks**:
1. Count all bigram frequencies:
   - `ab`: ? times
   - `ba`: ? times
   - `cd`: ? times
   - `dc`: ? times
   - (include space bigrams)

2. Merge the most frequent bigram. New vocab = ?
3. Recount bigrams with the merged token. Next merge = ?
4. After 4 merges, what's the vocabulary? What's the tokenized corpus?

**Verification**: Compare with your BPE implementation from Exercise 04.

---

## Phase IV–V: Vision

### MC-8: ViT Patch Embedding Dimensions (Exercise 06)

**Setup**: Image size 224×224, patch size 16×16, $d_{model} = 768$

**Tasks**:
1. Number of patches: $(224/16)^2 = ?$
2. Each patch flattened: $16 \times 16 \times 3 = ?$ pixels
3. Patch embedding projection: $? \times 768$ params
4. Total sequence length (with CLS token): $? + 1 = ?$
5. Positional embedding params: $? \times 768$
6. Self-attention FLOPs for one layer: $4 \times n^2 \times d + 2 \times n \times d^2 \approx ?$
   (where $n$ = seq_len, $d = 768$)
7. How does this compare to ResNet-50's ~4 GFLOPs?

---

## Phase VI: Robot Learning

### MC-9: Diffusion Forward Process (Exercise 09)

**Setup**: $x_0 = [0.5, 0.3]$ (a 2D action), linear schedule with $\beta_1=0.001, \beta_2=0.002, \beta_3=0.003$

**Tasks**:
1. Compute $\alpha_t = 1 - \beta_t$ for $t=1,2,3$:
   - $\alpha_1 = 0.999$, $\alpha_2 = ?$, $\alpha_3 = ?$

2. Compute $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$:
   - $\bar{\alpha}_1 = 0.999$
   - $\bar{\alpha}_2 = 0.999 \times 0.998 = ?$
   - $\bar{\alpha}_3 = ?$

3. Forward process (closed form): Given noise $\epsilon = [0.1, -0.2]$:
   $$x_3 = \sqrt{\bar{\alpha}_3} \cdot x_0 + \sqrt{1 - \bar{\alpha}_3} \cdot \epsilon = ?$$
   Compute each component.

4. **Reverse step** (given perfect noise prediction):
   The DDPM reverse formula:
   $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t)\right) + \sigma_t z$$
   
   Given $x_3$ from above and assuming $\epsilon_\theta = \epsilon$ (perfect prediction), $z = [0.05, -0.1]$, $\sigma_3 = \sqrt{\beta_3}$:
   - Compute $x_2$ from the formula.

5. Why does the reverse process need $T$ steps (slow) while forward is one-shot?

---

### MC-10: Jacobian for a 2-Link Planar Arm (Exercise 15)

**Setup**: Link lengths $L_1 = 1.0$, $L_2 = 0.5$, joint angles $\theta_1 = \pi/4$, $\theta_2 = \pi/3$

**Tasks**:
1. Forward kinematics:
   - $x = L_1\cos\theta_1 + L_2\cos(\theta_1 + \theta_2) = ?$
   - $y = L_1\sin\theta_1 + L_2\sin(\theta_1 + \theta_2) = ?$
   - Compute numerically.

2. Jacobian (2×2):
   $$J = \begin{bmatrix} 
   \frac{\partial x}{\partial \theta_1} & \frac{\partial x}{\partial \theta_2} \\
   \frac{\partial y}{\partial \theta_1} & \frac{\partial y}{\partial \theta_2}
   \end{bmatrix}$$
   
   - $\frac{\partial x}{\partial \theta_1} = -L_1\sin\theta_1 - L_2\sin(\theta_1+\theta_2) = ?$
   - $\frac{\partial x}{\partial \theta_2} = -L_2\sin(\theta_1+\theta_2) = ?$
   - $\frac{\partial y}{\partial \theta_1} = L_1\cos\theta_1 + L_2\cos(\theta_1+\theta_2) = ?$
   - $\frac{\partial y}{\partial \theta_2} = L_2\cos(\theta_1+\theta_2) = ?$

3. Pseudoinverse: $J^+ = J^T(JJ^T)^{-1}$
   - Compute $JJ^T$ (2×2 matrix)
   - Compute $(JJ^T)^{-1}$ using $\frac{1}{ad-bc}\begin{bmatrix}d & -b \\ -c & a\end{bmatrix}$
   - Compute $J^+ = J^T(JJ^T)^{-1}$

4. **IK step**: Target displacement $\Delta p = [0.1, 0.05]$
   $$\Delta \theta = J^+ \Delta p = ?$$
   New joint angles after one step: $\theta_{new} = \theta + \Delta\theta$

5. Compute new FK with $\theta_{new}$. How close are you to the target? Why might you need multiple iterations?

---

### MC-11: Action Chunking Math (Exercise 10/11)

**Setup**: Action chunk size $H = 4$, query frequency = 10 Hz, exponential temporal ensembling with $\lambda = 0.1$

At timestep $t$, you have overlapping predictions from chunks started at $t-3, t-2, t-1, t$:
```
Chunk from t-3: predicts a_{t|t-3} = [0.12, 0.08, -0.03]  (4th prediction in its chunk)
Chunk from t-2: predicts a_{t|t-2} = [0.10, 0.09, -0.02]  (3rd prediction)
Chunk from t-1: predicts a_{t|t-1} = [0.11, 0.07, -0.04]  (2nd prediction)
Chunk from t:   predicts a_{t|t}   = [0.13, 0.06, -0.01]  (1st prediction)
```

**Tasks**:
1. Compute exponential weights: $w_k = \exp(-\lambda \cdot k)$ where $k$ = age of chunk:
   - $w_0 = \exp(0) = 1.0$ (current chunk)
   - $w_1 = \exp(-0.1) = ?$
   - $w_2 = \exp(-0.2) = ?$
   - $w_3 = \exp(-0.3) = ?$

2. Normalize weights: $\hat{w}_k = w_k / \sum w_k$. Compute all $\hat{w}_k$.

3. Compute ensembled action:
   $$a_t = \sum_k \hat{w}_k \cdot a_{t|t-k}$$
   Compute each dimension.

4. What happens with $\lambda = 0$ (uniform)? With $\lambda = 10$ (nearly no ensembling)?

5. **Smoothness**: Compute jerk (third derivative) if you used only the latest chunk vs. ensembled. Why does ensembling reduce jerk?

---

## Phase VII: VLA Deployment

### MC-12: Inference Latency Budget (Exercise 17)

**Setup**: VLA with:
- Vision encoder: SigLIP (400M params, 224×224 input)
- Language model: Llama-2 7B backbone  
- Action head: 7 tokens autoregressively decoded
- Target: 10 Hz control (100ms budget)

**Tasks**:
1. Estimate vision encoder FLOPs:
   - Patches: $(224/14)^2 = 256$ tokens
   - 24 layers, $d=1024$, self-attention + FFN
   - Per layer: $\approx 4n^2d + 8nd^2$ FLOPs (where $n=256, d=1024$)
   - Total vision: ?

2. Estimate language model FLOPs for 7 action tokens:
   - Context: 256 (vision) + ~20 (text) = 276 tokens  
   - Prefill: $2 \times 7B \times 276 \approx ?$ FLOPs
   - 7 autoregressive steps: $2 \times 7B \times 7 \approx ?$ FLOPs

3. Given A100 at 312 TFLOPS (FP16):
   - Vision time: total_flops / 312e12 = ? ms
   - LM prefill time: ? ms
   - Action decode time: ? ms
   - **Total**: ? ms — does it fit in 100ms?

4. What if you use **action chunking** (predict 4 actions at once, execute at 10 Hz)?
   - Effective inference frequency = ? Hz
   - Time budget per inference = ? ms
   - Does the budget now work?

5. **KV-cache**: Without cache, each new action token reprocesses all previous tokens.
   - With cache: only the new token goes through attention. Speedup for 7-token generation = ?

---

### MC-13: KL Divergence and ELBO (Diffusion Foundation)

**Setup**: True posterior $q(z|x) = \mathcal{N}(\mu_q, \sigma_q^2)$ with $\mu_q = 2.0, \sigma_q = 0.5$  
Approximate posterior $p(z) = \mathcal{N}(\mu_p, \sigma_p^2)$ with $\mu_p = 0, \sigma_p = 1.0$

**Tasks**:
1. Compute KL divergence analytically:
   $$D_{KL}(q \| p) = \log\frac{\sigma_p}{\sigma_q} + \frac{\sigma_q^2 + (\mu_q - \mu_p)^2}{2\sigma_p^2} - \frac{1}{2}$$
   - $= \log\frac{1.0}{0.5} + \frac{0.25 + 4.0}{2.0} - 0.5 = ?$

2. VAE loss with reconstruction term $\mathbb{E}[\log p(x|z)] = -3.2$:
   - ELBO = reconstruction - KL = ?
   - Loss = -ELBO = ?

3. Why can't KL be negative? What does $D_{KL} = 0$ mean?

4. In diffusion models, the variational lower bound sums KL terms over all timesteps:
   $$L_{VLB} = \sum_{t=1}^T D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))$$
   
   If each term averages $\approx 0.01$ nats and $T = 1000$:
   - Total $L_{VLB} \approx ?$
   - Why does the simplified loss $\|\epsilon - \epsilon_\theta\|^2$ work better in practice?

---

## How to Use These

1. **Before coding**: Do the calculation for the relevant exercise first
2. **After coding**: Verify your code produces the same numbers
3. **During debugging**: If code gives wrong results, re-derive by hand to find the bug
4. **For interviews**: These are exactly the kind of derivations asked in ML research interviews

### Answer Key Protocol

After completing each calculation:
```python
# Verify MC-1 (example)
import torch

x = torch.tensor([2.0, -1.0], requires_grad=False)
W1 = torch.tensor([[0.5, -0.3], [0.8, 0.2]], requires_grad=True)
b1 = torch.tensor([0.1, 0.0], requires_grad=True)
W2 = torch.tensor([[0.4, -0.6]], requires_grad=True)
b2 = torch.tensor([0.2], requires_grad=True)

h1_pre = W1 @ x + b1
h1 = torch.relu(h1_pre)
y_hat = W2 @ h1 + b2
loss = 0.5 * (y_hat - 1.0) ** 2
loss.backward()

print(f"h1_pre: {h1_pre.data}")  # Compare to your hand calculation
print(f"h1:     {h1.data}")
print(f"y_hat:  {y_hat.data}")
print(f"loss:   {loss.data}")
print(f"dL/dW2: {W2.grad}")      # Must match your derivation
print(f"dL/dW1: {W1.grad}")
```
