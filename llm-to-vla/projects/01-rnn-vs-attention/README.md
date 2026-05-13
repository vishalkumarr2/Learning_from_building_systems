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

## Suggested Approach
1. Start from your Day 3 LSTM code
2. Build a minimal self-attention block (just Q, K, V with causal mask)
3. Use identical training setup for fair comparison
4. Run all 5 experiments
5. Write analysis

## Stretch Goals
- Add a GRU model as Model C
- Test on code (Python) vs natural language — does the gap change?
- Implement beam search decoding for both models
