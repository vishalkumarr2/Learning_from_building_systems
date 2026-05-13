# Week 6: LLM Engineering

> **Days 36–42 · 17.5 hours**

This week covers the engineering side of LLMs: evaluation, quantization, in-context learning, long context, RAG, tool use, and applying LLMs to robotics. Culminates in Phase III Capstone Day 1.

## Daily Lessons

| Day | Topic | Focus |
|-----|-------|-------|
| 36 | [LLM Evaluation](day-36-llm-evaluation.md) | Perplexity, MMLU, HumanEval, LLM-as-judge |
| 37 | [Quantization & Inference](day-37-quantization-inference.md) | INT4/INT8, GPTQ, AWQ, vLLM |
| 38 | [In-Context Learning](day-38-in-context-learning.md) | Zero/few-shot, mesa-optimization |
| 39 | [Long Context & Reasoning](day-39-long-context-reasoning.md) | RoPE scaling, ring attention, o1-style |
| 40 | [RAG & Tool Use](day-40-rag-tool-use.md) | Retrieval-augmented generation, function calling |
| 41 | [LLM for Robotics](day-41-llm-for-robotics.md) | SayCan, Code as Policies, fleet planning |
| 42 | [Phase III Capstone Day 1](day-42-phase3-capstone-1.md) | Fine-tune robotics assistant + RAG |

## Key Concepts

- **Evaluation:** How to measure if an LLM is actually good — benchmarks, contamination, Chatbot Arena
- **Quantization:** Compress 7B models to run on consumer hardware with minimal quality loss
- **In-context learning:** The most surprising emergent ability — learning from examples in the prompt
- **Long context & reasoning:** Scaling context windows and chain-of-thought for complex tasks
- **RAG:** Augment LLMs with external knowledge without fine-tuning
- **LLMs for robotics:** From language understanding to physical world actions

## Study Notes References

- [06 — LLM Training & Alignment](../../study-notes/06-llm-training-alignment.md)
- [07 — LLM Engineering](../../study-notes/07-llm-engineering.md)
