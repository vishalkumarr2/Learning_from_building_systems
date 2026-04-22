# 08 — Inductive Reasoning
### Analogy, causality, and probability

**Covers:** Chapters 9–11 — Analogy, Causality and Mill's Methods, Probability (§9.1–11.4)
**Prerequisite:** Lessons 01–07
**Unlocks:** Lesson 09 (Statistical & Scientific Reasoning)

---

## Why Should I Care?

Most real-world engineering reasoning is inductive, not deductive. You rarely have perfect information. Instead, you reason from:
- **Analogies:** "This robot has the same symptoms as the last one — probably the same root cause."
- **Causal inference:** "The failure started after the firmware update — maybe the update caused it."
- **Probability:** "95% of robots with this symptom had a sensorbar fault."

This lesson teaches you to do inductive reasoning *well* — and to recognize when it's done poorly.

---

# PART 1 — ARGUMENT BY ANALOGY

## 1.1 Structure

An argument by analogy has this form:
1. Entity A has properties p, q, r, and also property z.
2. Entity B has properties p, q, r.
3. Therefore, entity B probably also has property z.

**Example:**
- Robot-42 (A) had stale sensorbar data, estimator divergence, and stopped with DELOCALIZED. The root cause was an SPI clock jitter issue.
- Robot-87 (B) has stale sensorbar data and estimator divergence.
- Therefore, Robot-87 probably has an SPI clock jitter issue.

## 1.2 Six Criteria for Evaluating Analogies

An analogy is stronger when:

**1. More similarities** between A and B.
- Robot-42 and Robot-87 have the same hardware rev, same firmware, same environment → strong.
- They only share the symptom → weak.

**2. Fewer dissimilarities.**
- If Robot-87 has a different hardware revision, that's a significant dissimilarity.

**3. More diverse instances** supporting the analogy.
- If robots 42, 53, 67, and 91 all had the same symptom→cause pattern → stronger.
- If only robot 42 had it → weaker.

**4. More relevant similarities.**
- Same SPI bus configuration is highly relevant. Same paint color is not.

**5. Weaker conclusion** (less claimed).
- "Probably the same cause" → reasonable.
- "Definitely the same cause and no need to investigate" → too strong.

**6. Fewer entities** in the conclusion.
- Claiming it applies to Robot-87 → reasonable.
- Claiming it applies to all 200 robots in the fleet → needs more evidence.

## 1.3 Refuting Analogies

To refute an analogy, you can:
1. **Point out relevant dissimilarities** — different hardware rev, different environment.
2. **Present a counteranalogy** — "Robot-99 had identical symptoms but the cause was a loose connector, not SPI jitter."
3. **Show the similarities are superficial** — the symptoms look the same but the underlying data differs.

---

# PART 2 — CAUSALITY AND MILL'S METHODS

## 2.1 Necessary and Sufficient Conditions

| Type | Definition | Example |
|---|---|---|
| **Necessary condition** | X is necessary for Y if Y cannot occur without X | Oxygen is necessary for fire |
| **Sufficient condition** | X is sufficient for Y if X's occurrence guarantees Y | Decapitation is sufficient for death |
| **Necessary and sufficient** | X if and only if Y | Being a triangle ↔ having exactly 3 sides |

**Engineering examples:**
- Power is a **necessary** condition for the robot to run (no power → no run). But power alone isn't **sufficient** (you also need firmware, sensors, etc.).
- A 15A current spike through a 10A-rated fuse is **sufficient** for the fuse to blow. But it's not **necessary** — a 12A sustained current might also blow it eventually.

## 2.2 Mill's Five Methods

John Stuart Mill formalized five methods for identifying causal relationships. These are the foundation of systematic debugging.

### Method of Agreement
If two or more instances of the effect have only one factor in common, that factor is probably the cause.

**Application:**
| Instance | Firmware | Hardware Rev | Environment | Floor Type | Failure? |
|---|---|---|---|---|---|
| Robot-42 | v1.18 | Rev C | Warehouse A | Polished | Yes |
| Robot-53 | v1.18 | Rev D | Warehouse B | Concrete | Yes |
| Robot-67 | v1.18 | Rev C | Warehouse C | Tile | Yes |

The only common factor is **firmware v1.18** → probable cause.

### Method of Difference
If an instance where the effect occurs and one where it doesn't differ in only one factor, that factor is probably the cause.

**Application:**
| Instance | Firmware | Everything Else | Failure? |
|---|---|---|---|
| Robot-42 | v1.18 | Same | Yes |
| Robot-42 | v1.24 | Same | No |

Only firmware changed → firmware is probably the cause.

**This is the most powerful method** and the basis of controlled testing. Change one variable, keep everything else the same.

### Joint Method of Agreement and Difference
Combine both methods for stronger evidence.

### Method of Residues
If part of an effect is already explained by known causes, the remaining part is due to the remaining factors.

**Application:** "We know the 2ms latency in the control loop comes from sensor polling (0.5ms), computation (0.8ms), and actuator command (0.3ms). That's 1.6ms. The remaining 0.4ms must be from the bus communication overhead."

### Method of Concomitant Variation
If a factor varies in correlation with the effect, it's probably related causally.

**Application:** "As ambient temperature increases from 20°C to 45°C, the motor failure rate increases proportionally. Temperature is probably a causal factor."

**Warning:** Correlation ≠ causation. Both might be caused by a third factor (confounding variable). Use this method to generate hypotheses, then confirm with the method of difference.

## 2.3 Debugging Framework Using Mill's Methods

1. **Observe the failure** — what happened, when, where.
2. **Collect instances** — when does it happen? When doesn't it?
3. **Method of Agreement** — what do failing instances have in common?
4. **Method of Difference** — what's different between failing and working instances?
5. **Formulate hypothesis** — "Factor X causes the failure."
6. **Test** — change only factor X and observe. Does the failure stop? (Method of Difference, controlled)
7. **Verify** — reintroduce factor X. Does the failure return?

---

# PART 3 — PROBABILITY

## 3.1 Basic Probability Theory

**Probability** measures the likelihood of an event, from 0 (impossible) to 1 (certain).

**Classical theory:** P(E) = favorable outcomes / total outcomes
- P(rolling a 6) = 1/6
- P(picking a defective unit from 3 defective out of 100) = 3/100 = 0.03

**Relative frequency theory:** P(E) = number of times E occurred / number of trials
- If a robot crashed 7 times out of 500 operating hours → P(crash per hour) ≈ 0.014

## 3.2 Rules of Probability

### Negation Rule
P(~A) = 1 - P(A)

If the probability of the firmware update succeeding is 0.95, the probability of it failing is 0.05.

### Restricted Conjunction (Independent Events)
P(A and B) = P(A) × P(B) — when A and B are independent.

If the probability of a left motor failure is 0.01 and right motor failure is 0.01 (independent), then:
P(both motors fail) = 0.01 × 0.01 = 0.0001

### General Conjunction (Dependent Events)
P(A and B) = P(A) × P(B|A)

P(B|A) = probability of B *given* A has occurred.

### Restricted Disjunction (Mutually Exclusive)
P(A or B) = P(A) + P(B) — when A and B cannot both happen.

### General Disjunction (Not Mutually Exclusive)
P(A or B) = P(A) + P(B) - P(A and B)

If P(hardware fault) = 0.05 and P(firmware fault) = 0.10, and P(both) = 0.01:
P(hardware OR firmware fault) = 0.05 + 0.10 - 0.01 = 0.14

## 3.3 Bayes' Theorem

The most important probability formula for diagnostics:

$$P(H|E) = \frac{P(E|H) \times P(H)}{P(E)}$$

Where:
- P(H|E) = probability of hypothesis H given evidence E (posterior)
- P(E|H) = probability of seeing evidence E if H is true (likelihood)
- P(H) = prior probability of H (before seeing evidence)
- P(E) = total probability of seeing evidence E

**Example — Diagnostic reasoning:**

A sensorbar stale-data alarm fires. What's the probability it's an actual SPI bus failure vs. a software glitch?

- P(SPI failure) = 0.02 (2% of robots have hardware SPI issues)
- P(software glitch) = 0.05 (5% of robots have software-related stale data)
- P(alarm | SPI failure) = 0.95 (SPI failure almost always triggers alarm)
- P(alarm | software glitch) = 0.30 (software glitch sometimes triggers alarm)
- P(alarm | neither) = 0.01 (false alarm rate)

P(alarm) = P(alarm|SPI)×P(SPI) + P(alarm|SW)×P(SW) + P(alarm|neither)×P(neither)
= 0.95×0.02 + 0.30×0.05 + 0.01×0.93
= 0.019 + 0.015 + 0.0093 = 0.0433

P(SPI failure | alarm) = (0.95 × 0.02) / 0.0433 ≈ 0.44

Even though the alarm is strongly associated with SPI failure, the prior probability of SPI failure is low. The posterior probability is 44% — significant, but not enough to skip investigating the software glitch hypothesis.

## 3.4 Base Rate Fallacy

Ignoring prior probabilities (base rates) when interpreting evidence.

> "The test for defective sensors is 99% accurate. The test says this sensor is defective. So it's 99% likely defective."

Wrong! If only 1% of sensors are actually defective:
- P(defective) = 0.01
- P(positive | defective) = 0.99
- P(positive | not defective) = 0.01

P(defective | positive) = (0.99 × 0.01) / (0.99×0.01 + 0.01×0.99) = 0.50

The test is accurate, but with a low base rate, a positive result only gives 50% confidence! You need a second test or additional evidence.

---

## Key Takeaways

1. **Analogies are the starting point of most debugging.** Evaluate them using the six criteria — especially relevance of similarities.
2. **Mill's Method of Difference is the most powerful debugging tool.** Change one variable, keep everything else constant.
3. **Method of Agreement** finds common factors across failures. Method of Difference confirms which one is causal.
4. **Correlation ≠ causation.** Use concomitant variation to generate hypotheses, then test with controlled experiments.
5. **Bayes' Theorem** is the correct way to update beliefs with evidence. Don't ignore base rates.
6. **The base rate fallacy** is extremely common in diagnostics. A "99% accurate" test can still be wrong half the time if the condition is rare.
