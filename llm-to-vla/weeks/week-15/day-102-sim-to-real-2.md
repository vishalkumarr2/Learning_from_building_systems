# Day 102: Sim-to-Real Transfer — Day 2: Advanced Techniques

> **Phase VII — VLAs: Architecture to Deployment | Week 15 | 2.5 hours**
> *"The best sim-to-real pipelines don't just randomize — they reconstruct reality in simulation, then distill robust policies." — Advanced Transfer*

## Navigation
- **Previous:** [Day 101: Sim-to-Real Fundamentals](day-101-sim-to-real-1.md)
- **Next:** [Day 103: Hybrid VLA Architectures — Day 1](day-103-hybrid-1.md)
- **Week:** [Week 15 Overview](README.md)
- **Phase:** [Phase VII: VLAs](../../study-notes/15-vla-architectures.md)
- **Curriculum:** [Full Curriculum](../../CURRICULUM.md)

---

## Theory (60 min)

### 102.1 Real-to-Sim: Bringing Reality into Simulation

Instead of randomizing simulation → match it to reality:

```
Approach 1: NeRF-based reconstruction
  Real scene → 3D scan → NeRF → photo-realistic sim rendering
  + Perfect visual match
  - Expensive scanning process
  - Static scenes only

Approach 2: Digital twins
  CAD models + measured physics → accurate sim environment
  + Dynamics + visual match
  - Requires engineering effort
  - Must maintain as real changes

Approach 3: Image translation (CycleGAN)
  Sim images → style transfer → "real-looking" images
  + Cheap and automatic
  - May introduce artifacts
  - Doesn't fix dynamics gap
```

### 102.2 Teacher-Student Distillation

```
┌─────────────────────────────────────────────────────────┐
│            TEACHER-STUDENT TRANSFER                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Simulation (privileged access):                         │
│    Teacher policy sees: full state, contact forces,      │
│    object poses, velocities, friction coefficients       │
│    → Learns optimal policy with perfect information      │
│                                                          │
│  Distillation:                                           │
│    Student policy sees: only camera images + proprio     │
│    → Trained to mimic teacher's actions from limited obs │
│                                                          │
│  Real world:                                             │
│    Deploy student (only needs camera + proprio)           │
│    → Robust because teacher provided "oracle" targets    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

$$\mathcal{L}_\text{distill} = \text{KL}\left(\pi_\text{student}(a|o_\text{partial}) \| \pi_\text{teacher}(a|s_\text{full})\right)$$

### 102.3 Automatic Domain Randomization (ADR)

Instead of hand-tuning DR ranges, learn them:

```python
# ADR algorithm (simplified)
for iteration in range(1000):
    # 1. Train policy with current DR ranges
    policy = train(env, dr_params=current_ranges)

    # 2. Evaluate in reference environments
    perf = evaluate(policy, reference_envs)

    # 3. If performance is good → expand DR ranges
    if perf > threshold:
        for param in current_ranges:
            current_ranges[param] = expand(current_ranges[param], delta=0.05)
        print(f"Expanding DR: {current_ranges}")

    # 4. If performance drops → contract DR ranges
    else:
        for param in current_ranges:
            current_ranges[param] = contract(current_ranges[param], delta=0.02)
        print(f"Contracting DR: {current_ranges}")
```

### 102.4 How VLAs Handle Sim-to-Real

| VLA | Transfer Strategy | Key Technique |
|-----|------------------|---------------|
| RT-1 | No sim (all real) | 130K real demos |
| RT-2 | VLM pre-training | Web images bridge visual gap |
| Octo | Mixed real data | Multi-embodiment diversity |
| OpenVLA | LoRA fine-tuning | Quick adaptation to real |
| π₀ | Expert layers | Freeze VLM, train action head |
| GR-2 | Video pre-training | Internet videos teach physics |

### 102.5 Representation Transfer

The most practical VLA approach: transfer **representations**, not policies.

```
Strategy: Feature-level transfer

1. Train VLM on web data → learns general visual features
2. Features encode:
   - Object identity (what)
   - Spatial relationships (where)
   - Physical properties (how heavy, how soft)
3. Fine-tune only action head on real robot data
4. Visual gap is small because VLM already sees reality

Why this works:
  VLM sees billions of real images during pre-training
  → Visual representations are already "real-world"
  → Only action mapping needs sim-to-real transfer
  → 50-100 real demos sufficient for action fine-tuning
```

---

## Implementation (60 min)

### Teacher-Student Pipeline

```python
import torch
import torch.nn as nn
import torch.distributions as D

class PrivilegedTeacher(nn.Module):
    """Teacher with full state access (sim only)."""
    def __init__(self, state_dim=20, action_dim=7, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, full_state):
        h = self.net(full_state)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -5, 2)
        return D.Normal(mean, log_std.exp())

class VisionStudent(nn.Module):
    """Student with only camera + proprio (deployable)."""
    def __init__(self, img_dim=512, proprio_dim=7, action_dim=7, hidden=256):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64*16, img_dim),
        )
        self.policy = nn.Sequential(
            nn.Linear(img_dim + proprio_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, image, proprio):
        img_feat = self.image_encoder(image)
        h = self.policy(torch.cat([img_feat, proprio], dim=-1))
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), -5, 2)
        return D.Normal(mean, log_std.exp())

def distill_teacher_to_student(teacher, student, sim_env_fn,
                                n_epochs=100, n_rollouts_per_epoch=50):
    """Distill teacher knowledge into student."""
    optimizer = torch.optim.Adam(student.parameters(), lr=3e-4)

    for epoch in range(n_epochs):
        total_loss = 0

        for _ in range(n_rollouts_per_epoch):
            # Simulate rollout (pseudo-code)
            # full_state, image, proprio = sim_env.reset()
            full_state = torch.randn(1, 20)
            image = torch.randn(1, 3, 64, 64)
            proprio = torch.randn(1, 7)

            # Teacher action distribution
            with torch.no_grad():
                teacher_dist = teacher(full_state)

            # Student action distribution
            student_dist = student(image, proprio)

            # KL divergence loss
            kl = D.kl_divergence(student_dist, teacher_dist).sum(dim=-1).mean()

            # Behavior cloning loss (match teacher actions)
            teacher_action = teacher_dist.mean
            bc_loss = ((student_dist.mean - teacher_action)**2).sum(dim=-1).mean()

            loss = kl + bc_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 20 == 0:
            avg_loss = total_loss / n_rollouts_per_epoch
            print(f"Epoch {epoch}: distillation loss = {avg_loss:.4f}")

    return student

# Adaptive Domain Randomization
class ADR:
    """Automatic Domain Randomization."""
    def __init__(self, initial_ranges, performance_threshold=0.7):
        self.ranges = {k: list(v) for k, v in initial_ranges.items()}
        self.threshold = performance_threshold
        self.expand_delta = 0.05
        self.contract_delta = 0.02

    def sample(self):
        """Sample randomized parameters."""
        return {k: np.random.uniform(v[0], v[1])
                for k, v in self.ranges.items()}

    def update(self, performance):
        """Expand or contract ranges based on performance."""
        if performance >= self.threshold:
            for k in self.ranges:
                spread = self.ranges[k][1] - self.ranges[k][0]
                self.ranges[k][0] -= self.expand_delta * spread
                self.ranges[k][1] += self.expand_delta * spread
            return "expanded"
        else:
            for k in self.ranges:
                spread = self.ranges[k][1] - self.ranges[k][0]
                center = (self.ranges[k][0] + self.ranges[k][1]) / 2
                self.ranges[k][0] = center - (spread/2) * (1 - self.contract_delta)
                self.ranges[k][1] = center + (spread/2) * (1 - self.contract_delta)
            return "contracted"

# Demo
teacher = PrivilegedTeacher()
student = VisionStudent()

print("Teacher params:", sum(p.numel() for p in teacher.parameters()))
print("Student params:", sum(p.numel() for p in student.parameters()))

# ADR demo
import numpy as np
adr = ADR({"friction": [0.8, 1.2], "mass": [0.9, 1.1]})
for i in range(5):
    perf = 0.8 - i * 0.05  # Simulated performance
    action = adr.update(perf)
    print(f"Step {i}: perf={perf:.2f}, {action}, ranges={adr.ranges}")
```

---

## Exercise (45 min)

1. **Distillation quality:** Train a teacher with full state → train student via distillation → compare student with direct BC from images. Does distillation help? By how much?

2. **ADR convergence:** Run ADR for 100 iterations. Plot the DR ranges over time. Do they converge? What's the final range compared to hand-tuned?

3. **Transfer approach comparison:** Compare (A) DR only, (B) SysID only, (C) teacher-student, (D) progressive transfer on the same task. Rank by real-world performance.

4. **Representation analysis:** Freeze a pre-trained VLM vision encoder. Train only an action head on 50 real demos. Compare with training vision encoder from scratch on 50 demos. Quantify the benefit of pre-trained representations.

---

## Key Takeaways

1. **Real-to-sim** (NeRF, digital twins) complements domain randomization
2. **Teacher-student distillation** transfers privileged knowledge to deployable policies
3. **ADR** automates the tedious process of tuning randomization ranges
4. **VLM pre-training is the best sim-to-real technique** — web images are already real
5. **Representation transfer + action fine-tuning** is the practical VLA approach

---

## Connection to the Thread

> *Sim-to-real is solved (enough) for VLAs thanks to VLM pre-training. Tomorrow: hybrid VLA architectures that combine the best ideas — tokenized actions AND diffusion, planning AND reactive control, large VLMs AND small action experts. The field is converging on these hybrid designs.*

---

## Further Reading

- Chen et al. (2021), "Teacher-Student Sim-to-Real Transfer"
- Mehta et al. (2020), "Active Domain Randomization"
- Akkaya et al. (2019), "Solving Rubik's Cube with a Robot Hand" (massive DR at OpenAI)
