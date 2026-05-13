# Week 1: DL Foundations — Backprop to Information Theory

> **Phase I · Days 1–7 · 17.5 hours**

This week builds the foundation everything else rests on. You'll revisit backpropagation through computation graphs, understand *why* CNNs and RNNs were the dominant paradigms (and their limitations), and discover the information-theoretic thread — compression = prediction = intelligence — that unifies the entire curriculum.

## Daily Lessons

| Day | Topic | Focus |
|-----|-------|-------|
| 1 | [Computation Graphs & Backprop](day-01-computation-graphs-backprop.md) | How gradients actually flow |
| 2 | [CNN & ResNets](day-02-cnn-resnets.md) | Spatial hierarchies & residual revolution |
| 3 | [RNN/LSTM Essentials](day-03-rnn-lstm-essentials.md) | Sequential processing & vanishing gradients |
| 4 | [Seq2Seq & The Bottleneck](day-04-seq2seq-bottleneck.md) | The fixed-vector bottleneck that demands attention |
| 5 | [Information Theory & Compression](day-05-information-theory-compression.md) | Cross-entropy, KL divergence, compression = intelligence |
| 6 | [Embeddings & Representation Learning](day-06-embeddings-representations.md) | How neural nets learn meaning |
| 7 | [Training Stability Cookbook](day-07-training-stability.md) | Practical recipes for stable training |

## Key Concepts

- Reverse-mode AD (backprop) computes all gradients in one backward pass
- Residual connections create gradient highways — non-negotiable at scale
- The vanishing gradient problem in RNNs directly motivates attention
- The seq2seq bottleneck is the "last straw" that forced the invention of attention
- Cross-entropy loss = negative log-likelihood = compression efficiency
- Better prediction = better compression = more understanding

## Study Notes Reference

For detailed chapter-level coverage of all Week 1 topics, see:
[01 — DL Foundations & Information Theory](../../study-notes/01-dl-foundations.md)
