# 09 — Statistical & Scientific Reasoning
### Samples, averages, hypotheses, and science vs. pseudoscience

**Covers:** Chapters 12–14 — Statistical Reasoning, Hypothetical/Scientific Reasoning, Science and Superstition (§12.1–14.4)
**Prerequisite:** Lesson 08 (Inductive Reasoning)
**Unlocks:** You've completed the logic track!

---

## Why Should I Care?

Every data-driven decision in engineering involves statistics: fleet failure rates, A/B testing firmware versions, calculating mean-time-between-failures. And every investigation is a miniature scientific inquiry: hypothesize, predict, test, refine. This lesson teaches you the reasoning patterns behind both — and the traps that lead to wrong conclusions.

---

# PART 1 — STATISTICAL REASONING

## 1.1 Populations and Samples

- **Population:** The complete group you want to know about (e.g., all 200 robots in the fleet).
- **Sample:** A subset you actually observe (e.g., 30 robots you tested).

The goal is to draw conclusions about the population from the sample. This only works if the sample is **representative**.

**Random sample:** Every member of the population has an equal chance of being selected. This minimizes bias.

**Biased sample:** Some members are more likely to be selected. Common biases:
- **Selection bias:** Testing only the robots in building A (they might have different conditions than building B).
- **Survivorship bias:** Only analyzing robots that are still operational (ignoring ones that failed catastrophically and were removed).
- **Convenience sampling:** Testing whichever robots are nearest to your desk.

## 1.2 Measures of Central Tendency

Three types of "average":

**Mean (arithmetic average):**
$$\bar{x} = \frac{\sum x_i}{n}$$

Sensitive to outliers. If 9 robots have 0 failures and 1 has 90 failures, the mean is 9 — misleading.

**Median:** The middle value when sorted. Less sensitive to outliers.
- {0, 0, 0, 0, 0, 0, 0, 0, 0, 90} → median = 0

**Mode:** The most frequent value.
- {0, 0, 0, 0, 0, 0, 0, 0, 0, 90} → mode = 0

**Which to use:**
- Symmetric data → mean is fine.
- Skewed data (common in failure analysis) → use median.
- Always report which measure you're using!

## 1.3 Measures of Dispersion

**Range:** max - min. Simple but sensitive to outliers.

**Standard deviation (σ):** Average distance from the mean.
$$\sigma = \sqrt{\frac{\sum (x_i - \bar{x})^2}{n}}$$

Small σ → data is clustered near the mean (consistent).
Large σ → data is spread out (variable).

**In engineering:** "The mean motor temperature is 55°C with σ = 3°C" tells you most motors run between 52–58°C. "Mean 55°C with σ = 15°C" tells you motor temperatures vary wildly.

## 1.4 Percentage Pitfalls

**Ambiguous percentages:** "Sales increased by 50%." 50% of what? From 2 to 3? From 2 million to 3 million?

**Percentage vs. percentage points:**
- "Failure rate went from 2% to 4%." That's a 2 **percentage point** increase, but a 100% **relative** increase.
- Both descriptions are technically correct but convey very different impressions.

**Small base fallacy:** "Failures increased by 200%!" Sounds alarming — but if it went from 1 to 3 out of 10,000 robots, the absolute increase is negligible.

**Always report both absolute and relative numbers.** "Failure rate increased from 0.01% to 0.03% (a 200% relative increase, representing 2 additional failures out of 10,000 units)."

## 1.5 Graphs and Visual Deception

**Truncated Y-axis:** Starting the Y-axis at a non-zero value makes small changes look dramatic.
```
Misleading:        Honest:
│    /             │
│   /              │         /
│  /               │        /
│_/____            │_______/____
  2%  4%           0%   2%   4%
```

**Distorted scales:** Non-linear scales, missing labels, cherry-picked time ranges.

**Rule:** Always check the axes. Ask "What would this look like with a zero baseline and the full time range?"

---

# PART 2 — HYPOTHETICAL & SCIENTIFIC REASONING

## 2.1 The Structure of Hypothetical Reasoning

Scientific and engineering reasoning follows this pattern:

1. **Observe** a phenomenon (the robot keeps delocalizating in aisle 7).
2. **Hypothesize** an explanation (the floor in aisle 7 causes encoder slip).
3. **Derive predictions** from the hypothesis (if encoder slip, then we'll see velocity estimate > actual velocity during turns in aisle 7).
4. **Test** the predictions (instrument the estimator, drive through aisle 7, compare).
5. **Evaluate:**
   - Prediction confirmed → hypothesis is *supported* (not proven).
   - Prediction fails → hypothesis is *disconfirmed* (or needs revision).

**Critical asymmetry:** A failed prediction strongly disconfirms a hypothesis (via Modus Tollens). A confirmed prediction only weakly supports it — there might be other hypotheses that predict the same thing.

## 2.2 Criteria for Evaluating Hypotheses

When multiple hypotheses can explain the same evidence, prefer the one that:

**1. Has greater explanatory power** — explains more of the observed facts.
- Hypothesis A explains the delocalization but not the motor current spikes.
- Hypothesis B explains both.
- Prefer B.

**2. Is more testable/falsifiable** — makes specific, risky predictions.
- "Something went wrong" → not testable.
- "The SPI clock jitter exceeds 5ns when CPU temperature > 70°C" → testable.

**3. Is simpler (Occam's Razor)** — don't multiply causes beyond necessity.
- If one firmware bug explains all the symptoms, don't hypothesize three different hardware failures.
- But don't oversimplify either — sometimes there really are multiple causes.

**4. Is more conservative** — fits with established knowledge.
- A hypothesis that requires rewriting the laws of physics is less plausible than one that identifies a known failure mode.

**5. Has greater predictive power** — predicts NEW observations, not just explains old ones.
- A hypothesis that says "if we look at robot-87, we'll see the same pattern" and it's confirmed → much stronger support.

## 2.3 The Hypothetico-Deductive Method

Formal version of scientific reasoning:

```
1. H → O         (If hypothesis H is true, observation O will occur)
2. O occurs       (We observe O)
3. Therefore H    ← INVALID! (Affirming the consequent)

1. H → O         (If hypothesis H is true, observation O will occur)
2. O does NOT occur
3. Therefore ~H   ← VALID (Modus Tollens)
```

**This is why Karl Popper emphasized falsification.** You can never *prove* a hypothesis by confirming predictions (that's affirming the consequent). You can only *disprove* it by finding predictions that fail (Modus Tollens).

In practice, strong evidence comes from:
1. Many diverse confirmed predictions (though none is conclusive alone).
2. Surviving rigorous attempts at falsification.
3. No viable alternative hypotheses remaining.

## 2.4 Ad Hoc Modifications

When evidence contradicts a hypothesis, it's tempting to add **ad hoc** modifications to save it:

> Original: "The SPI bug only occurs in firmware v1.18."
> Contradicted by: A robot running v1.20 also exhibits the symptoms.
> Ad hoc: "The SPI bug only occurs in firmware v1.18, except when the hardware is Rev C, in which case it also occurs in v1.20."

Each ad hoc modification makes the hypothesis less falsifiable and less elegant. Too many modifications signal that the hypothesis is wrong and you need a fundamentally different explanation.

**Warning sign:** If you keep adding exceptions and special cases to your root cause theory, step back and reconsider.

---

# PART 3 — SCIENCE VS. PSEUDOSCIENCE

## 3.1 Distinguishing Features

| Science | Pseudoscience |
|---|---|
| Testable, falsifiable hypotheses | Vague claims that can't be tested |
| Seeks disconfirmation | Seeks only confirmation |
| Self-correcting (updates with evidence) | Ignores contradictory evidence |
| Precise predictions | Post hoc explanations |
| Peer review and replication | Authority-based claims |
| Acknowledges uncertainty | Claims certainty |

## 3.2 Cognitive Biases in Technical Work

**Confirmation bias:** Seeking only evidence that supports your hypothesis.
> You suspect firmware. You only look at firmware logs. You find something suspicious. You declare firmware is the cause — without checking hardware, environment, or configuration.

**Anchoring:** Over-relying on the first piece of information.
> The first person to look at the incident says "it's probably the sensorbar." Every subsequent investigator focuses on the sensorbar, even when evidence points elsewhere.

**Availability heuristic:** Overweighting recent or memorable examples.
> The last three incidents were all SPI issues. When a new incident comes in, you assume SPI without checking — but this one is actually a power supply problem.

**Groupthink:** The team converges on a hypothesis and stops questioning it.
> In the RCA meeting, the senior engineer says "firmware bug." Nobody pushes back because they don't want to disagree with the senior engineer.

## 3.3 Applying Scientific Rigor to Debugging

1. **Form explicit hypotheses** before investigating. Write them down.
2. **Design tests that can falsify** your hypothesis, not just confirm it.
3. **Track ruled-out hypotheses** alongside supported ones.
4. **Seek disconfirmation actively.** Ask: "What evidence would prove me wrong?"
5. **Avoid premature closure.** Don't stop at the first explanation that fits.
6. **Document negative results.** "We tested X and it was NOT the cause" is valuable information.
7. **Challenge anchoring.** When someone proposes a cause, ask: "What else could explain this?"

---

## Quick Reference: Statistical Fallacies

| Fallacy | Example | Fix |
|---|---|---|
| Biased sample | Testing only the newest robots | Random sampling across the fleet |
| Misleading average | Mean failure rate hides one robot with 10x more failures | Report median + distribution |
| Small base | "200% increase!" (from 1 to 3 incidents) | Report absolute numbers |
| Truncated axis | Graph makes 2% → 4% look like 10x increase | Start Y-axis at 0 |
| Cherry-picked timeframe | Choosing a period where data looks favorable | Report full history |
| Ignoring base rates | "Test is 99% accurate → 99% certain" | Use Bayes' Theorem |

---

## Key Takeaways

1. **Representative samples are essential.** Random sampling beats convenience sampling. Small biased samples produce wrong conclusions.
2. **Report absolute + relative numbers** for percentages. "200% increase" means nothing without the base.
3. **Hypotheses can be disconfirmed but never conclusively confirmed.** Design tests that can falsify, not just verify.
4. **Occam's Razor:** Prefer the simplest explanation that covers all the evidence. But don't ignore evidence to keep things simple.
5. **Guard against cognitive biases:** confirmation bias, anchoring, and groupthink are the biggest threats to sound technical reasoning.
6. **Scientific debugging = explicit hypotheses + falsifiable predictions + systematic testing + documented results.** This is the methodology of every good RCA.
