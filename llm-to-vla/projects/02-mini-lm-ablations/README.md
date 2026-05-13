# Project 02 — Mini-LM Ablations

> **Phase II Capstone · Days 29–30 · ~8 hours**
> Prerequisites: Study Note [05-gpt-scaling](../../study-notes/05-gpt-scaling.md), Exercise [04-nanogpt-ablations](../../exercises/04-nanogpt-ablations.md)

---

## Objective

Train a mini language model on Shakespeare using nanoGPT, run **systematic ablation experiments** varying architectural choices, and produce a **scaling curve** with power-law fit. Synthesize findings in a written ablation report.

This project validates that you can:
1. Train a Transformer LM from scratch
2. Design and execute controlled experiments
3. Fit and interpret scaling laws
4. Communicate empirical results clearly

---

## Requirements

### R1: Trained Baseline Model
- Train nanoGPT on `shakespeare_char` with the baseline config (6 layers, 384 dim, 6 heads)
- Achieve validation loss ≤ 1.50
- Save checkpoint

### R2: Ablation Experiments (minimum 15 runs)
Run at least **3 experiments per category**:

| Category | Variables | Min Runs |
|----------|-----------|----------|
| Depth | n_layer ∈ {2, 4, 6, 8, 12} | 5 |
| Width | n_embd ∈ {64, 128, 256, 384, 512} | 5 |
| Heads | n_head ∈ {1, 2, 4, 6, 8} | 5 |
| Activation | {ReLU, GELU, SwiGLU} | 3 |
| Normalization | {Pre-LN, Post-LN, RMSNorm} | 3 |

Control: change **exactly one** variable per experiment. Everything else (seed, data, steps, LR) must match baseline.

### R3: Scaling Curve
- Plot validation loss vs. parameter count (log-log)
- Fit power law: $L(N) = a \cdot N^{-\alpha}$
- Report $\alpha$ and $R^2$
- Compare your $\alpha$ with published Kaplan value ($\alpha_N \approx 0.076$)

### R4: Ablation Report
Written report (Jupyter notebook or Markdown) containing:
- Methodology: baseline config, what was varied, training setup
- Results table: all experiments with params, train loss, val loss
- Plots: scaling curve, ablation bar charts, training curves
- Analysis: which choices matter most, diminishing returns, surprises
- Comparison with published scaling laws
- Discussion: implications for larger models

---

## Deliverables

```
projects/02-mini-lm-ablations/
├── README.md                    ← this file
├── configs/
│   ├── baseline.py              ← baseline config
│   └── ablations/               ← one config per experiment
├── results/
│   ├── ablation_table.csv       ← all results in CSV
│   └── checkpoints/             ← saved model checkpoints
├── plots/
│   ├── scaling_curve.png        ← loss vs params (log-log)
│   ├── ablation_bars.png        ← grouped bar chart
│   └── training_curves.png      ← loss vs iteration overlay
├── ablation_report.ipynb        ← full report notebook
└── scripts/
    ├── run_all_ablations.sh     ← automated experiment runner
    └── plot_results.py          ← plotting script
```

---

## Evaluation Rubric

| Criterion | Points | Requirements |
|-----------|--------|--------------|
| **Trained model** | 15 | Baseline achieves val loss ≤ 1.50 |
| **Ablation completeness** | 25 | ≥15 runs across all 5 categories, controlled methodology |
| **Scaling curve** | 20 | Log-log plot with power-law fit, $R^2$ reported, comparison with Kaplan |
| **Ablation plots** | 15 | Clear, labeled bar charts and training curves |
| **Written analysis** | 20 | Methodology, findings, interpretation, implications |
| **Code quality** | 5 | Reproducible scripts, clean configs, documented |
| **Total** | **100** | |

**Pass threshold**: ≥ 70 points

---

## Getting Started

```bash
# 1. Clone nanoGPT (if not already)
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT

# 2. Prepare data
python data/shakespeare_char/prepare.py

# 3. Copy baseline config
cp config/train_shakespeare_char.py configs/baseline.py
# Edit to match the ablation baseline from Exercise 04

# 4. Run baseline
python train.py configs/baseline.py --out_dir=results/checkpoints/baseline

# 5. Start ablations (see Exercise 04 for full commands)
# ...

# 6. Collect results and plot
python scripts/plot_results.py
```

---

## Tips

- **Seed matters**: Use `--seed=42` for all runs to reduce variance
- **Don't over-train**: 5000 iterations is enough for ablation comparisons
- **Log everything**: Save train/val loss at every eval interval
- **Parallel runs**: If you have multiple GPUs, run experiments simultaneously
- **Git tag each run**: Makes it easy to reproduce specific results

---

*Part of the [LLM-to-VLA Curriculum](../../CURRICULUM.md) · Phase II: Language Models*
