# Logic & Critical Reasoning — Learning Plan
### From argument analysis to formal proof — 9 lessons + exercises

**Source:** *A Concise Introduction to Logic* (14th ed.), Patrick J. Hurley & Lori Watson

**Why this matters for engineering:**
Every debugging session is an argument: you gather evidence (premises) and draw a conclusion (root cause). Every design review is a chain of reasoning. Every incident report claims that X caused Y. Logic teaches you to evaluate these arguments rigorously — to spot when reasoning is valid, when evidence is weak, and when conclusions don't follow from premises.

---

## Track Structure

| # | Lesson | Book Chapters | Key Skills |
|---|--------|---------------|------------|
| 01 | Arguments, Premises & Conclusions | Ch 1 (§1.1–1.5) | Identify arguments, premises, conclusions; deduction vs induction; validity vs soundness |
| 02 | Language, Meaning & Definition | Ch 2 (§2.1–2.4) | Cognitive vs emotive meaning; ambiguity; vagueness; types of definitions |
| 03 | Informal Fallacies | Ch 3 (§3.1–3.5) | Recognize 25+ named fallacies: relevance, weak induction, presumption, ambiguity |
| 04 | Categorical Logic | Ch 4–5 (§4.1–5.7) | Standard-form propositions; Venn diagrams; categorical syllogisms; rules and fallacies |
| 05 | Propositional Logic | Ch 6 (§6.1–6.6) | Logical connectives; truth tables; validity testing; argument forms |
| 06 | Natural Deduction in Propositional Logic | Ch 7 (§7.1–7.7) | Rules of inference; replacement rules; conditional & indirect proof |
| 07 | Predicate Logic | Ch 8 (§8.1–8.7) | Quantifiers; translation; universal/existential instantiation; proofs with quantifiers |
| 08 | Inductive Reasoning | Ch 9–11 (§9.1–11.4) | Analogy; legal/moral reasoning; causality; Mill's methods; probability |
| 09 | Statistical & Scientific Reasoning | Ch 12–14 (§12.1–14.4) | Samples; averages; percentages; hypothetical reasoning; science vs pseudoscience |

---

## How to Use This Track

**For each lesson:**
1. Read the lesson notes (the lesson file)
2. Work through the inline examples — try to solve them *before* reading the explanation
3. Complete the matching exercise set in `exercises/`
4. Check your answers using the collapsible solutions

**Estimated time:** 2–3 hours per lesson, including exercises. Total: ~20–25 hours.

**Suggested pace:**
- Weeks 1–2: Lessons 01–03 (informal logic foundation)
- Weeks 3–4: Lessons 04–05 (categorical + propositional logic)
- Weeks 5–6: Lessons 06–07 (formal proofs)
- Weeks 7–8: Lessons 08–09 (inductive and scientific reasoning)

---

## Prerequisites

None. This track assumes no prior logic background. If you can read and understand ordinary English sentences, you have everything you need.

---

## Connection to Engineering Work

| Logic Concept | Engineering Application |
|---|---|
| Argument identification | Parsing bug reports: "What is actually being claimed? What's the evidence?" |
| Deductive validity | "If the premises of this RCA are true, must the conclusion be true?" |
| Inductive strength | "How strong is this failure pattern based on 5 similar incidents?" |
| Informal fallacies | Spotting bad reasoning in meetings: ad hominem, false dilemma, hasty generalization |
| Propositional logic | Boolean conditions in code, truth tables for complex if-else chains |
| Categorical syllogisms | "All OKS robots have sensorbar. This device has no sensorbar. Therefore this is not an OKS robot." |
| Predicate logic | "For all robots in fleet X, if battery < 20% then dock" — formalizing fleet policies |
| Mill's methods | Systematic root cause isolation: method of difference, concomitant variation |
| Scientific reasoning | Hypothesis formation and testing in debugging; distinguishing correlation from causation |
