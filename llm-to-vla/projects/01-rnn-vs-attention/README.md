# Project 01 — RNN vs Attention: Character-Level Language Model Comparison
> Phase I Capstone · Days 8–9 · ~5 hours

## Overview
Build and compare two character-level language models to empirically demonstrate WHY attention is needed and validate the "compression = prediction" insight from Phase I.

## Learning Objectives
- Apply all Phase I concepts in a single comparison project
- Empirically demonstrate the limitations of recurrent models
- Measure compression efficiency (bits-per-character) as the key metric
- Build intuition for WHY attention was invented

## Requirements

### Model A: LSTM Character-LM
- 2-layer LSTM with hidden size 256
- Character-level tokenization
- Train on Shakespeare (or corpus of choice)
- Use your Day 7 training monitor

### Model B: Simple Self-Attention Character-LM
- Single self-attention layer + FFN (preview of transformer)
- Same hidden size (256) for fair comparison
- Causal mask for autoregressive generation
- Same training setup

### Experiments (mandatory)
1. Training curves: loss over steps for both models
2. Bits-per-character comparison (Day 5 metric)
3. Sample quality: generate 500 characters from each model
4. Training stability: gradient norm plots from both models
5. Sequence length scaling: measure BPC at length 50, 100, 200, 500

## Evaluation Rubric
| Criterion | Points |
|-----------|--------|
| Both models train successfully | 20 |
| BPC comparison with analysis | 20 |
| Training stability plots (grad norms, loss curves) | 20 |
| Generated samples with quality discussion | 15 |
| Sequence length scaling experiment | 15 |
| Written analysis: "Why attention wins" (1 paragraph) | 10 |

## Deliverables
- [ ] Jupyter notebook with all code, plots, and analysis
- [ ] Training stability diagnostics for both models
- [ ] Generated text samples (500 chars each)
- [ ] 1-paragraph "Why attention wins" analysis

## Pass Threshold
To pass this project, you must achieve **all** of the following:
- Both models train to convergence (loss plateaus)
- BPC for attention model is lower than LSTM on at least 3/4 sequence lengths
- Generated samples are recognizable English (not random characters)
- Written analysis correctly identifies at least 2 reasons attention outperforms RNNs
- Total rubric score ≥ 70/100

## Directory Structure
```
01-rnn-vs-attention/
├── README.md              ← this file
├── rnn_vs_attention.ipynb ← main notebook (all code + plots + analysis)
├── models/
│   ├── lstm_charlm.py     ← optional: factor out model definitions
│   └── attention_charlm.py
├── data/
│   └── shakespeare.txt    ← auto-downloaded by notebook
├── checkpoints/           ← saved model weights
└── figures/               ← exported plots for report
```

## Getting Started
```bash
# From the project directory
mkdir -p models data checkpoints figures

# Download Shakespeare corpus (same as Exercise 01)
python -c "
import urllib.request
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
urllib.request.urlretrieve(url, 'data/shakespeare.txt')
print(f'Downloaded {len(open(\"data/shakespeare.txt\").read()):,} characters')
"
```

## Suggested Approach
1. Start from your Day 3 LSTM code
2. Build a minimal self-attention block (just Q, K, V with causal mask)
3. Use identical training setup for fair comparison
4. Run all 5 experiments
5. Write analysis

## Tips
- Use the **same** character vocabulary for both models — don't let tokenization differences confound results
- Log gradient norms per layer with `torch.nn.utils.clip_grad_norm_` (set max_norm high to observe, not clip)
- For BPC: `bpc = loss / math.log(2)` — this is bits per character
- The attention model should clearly win on longer sequences (200+) — if it doesn't, check your causal mask
- Keep batch size and learning rate identical; only the model architecture should differ

## Stretch Goals
- Add a GRU model as Model C
- Test on code (Python) vs natural language — does the gap change?
- Implement beam search decoding for both models
- Plot attention weights to visualize what the attention model learns
