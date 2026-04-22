# Exercise 8 — Inductive Reasoning

**Covers:** Lesson 08
**Difficulty:** Intermediate

---

## Problem 1: Evaluate the Analogy

Rate each analogy as strong or weak. Explain which criteria (from the six) support your rating.

**A.** "Robot-42 (Rev C, firmware v1.18, Warehouse A, polished floor) had stale sensorbar data and the root cause was SPI clock jitter. Robot-87 (Rev C, firmware v1.18, Warehouse B, concrete floor) also has stale sensorbar data. Therefore, Robot-87 probably has SPI clock jitter too."

**B.** "My laptop crashed after a software update. A robot crashed after a firmware update. Therefore, the robot crash was probably caused by the firmware update, just like my laptop."

**C.** "In the last 8 incidents involving estimator divergence at Hamamatsu, 7 were caused by sensorbar cable issues. This is another estimator divergence at Hamamatsu. Therefore, it's probably a sensorbar cable issue."

---

## Problem 2: Mill's Methods

Identify which of Mill's methods is being applied, and state the conclusion.

**A.**
| Robot | Firmware | HW Rev | Floor | Temperature | Failure? |
|---|---|---|---|---|---|
| R-42 | v1.18 | C | Polished | 25°C | Yes |
| R-53 | v1.20 | D | Concrete | 30°C | Yes |
| R-67 | v1.18 | C | Tile | 22°C | Yes |
| R-71 | v1.20 | D | Polished | 28°C | Yes |

All failing robots are in Warehouse A (not shown in table but true). What method? What conclusion?

**B.**
| Condition | Robot-42 (fails) | Robot-42 (works) |
|---|---|---|
| Firmware | v1.18 | v1.24 |
| Hardware | Rev C | Rev C |
| Environment | Same | Same |
| Floor | Same | Same |

**C.** A robot's motor current draw increases proportionally as ambient temperature increases from 20°C to 45°C, with a consistent linear relationship observed across 50 measurements.

**D.** Total system latency is 12ms. Network latency accounts for 3ms, sensor polling for 4ms, and computation for 2ms. Therefore, the remaining 3ms is attributable to actuator command transmission.

---

## Problem 3: Probability Calculation

**A.** A fleet has 200 robots. 15 have hardware Rev A, 85 have Rev B, and 100 have Rev C. If a robot is randomly selected, what's the probability it's Rev C?

**B.** P(firmware bug) = 0.10, P(hardware fault) = 0.05, P(both) = 0.01. What's P(firmware bug OR hardware fault)?

**C.** Two independent robots each have a 0.02 probability of failure per shift. What's the probability that both fail in the same shift?

**D.** A diagnostic test for SPI failures has: sensitivity = 0.95 (P(positive | SPI failure) = 0.95), specificity = 0.90 (P(negative | no SPI failure) = 0.90). The base rate of SPI failures is 3%. If the test is positive, what's the probability the robot actually has an SPI failure? (Use Bayes' theorem.)

---

## Problem 4: Identify the Causal Fallacy

Each scenario commits a causal reasoning error. Name it and explain.

**A.** "We deployed firmware v1.24 on Monday. On Tuesday, a completely different robot (that wasn't updated) crashed. Therefore, the deployment process somehow destabilized the fleet."

**B.** "Ice cream sales and drowning rates both increase in summer. Therefore, eating ice cream causes drowning."

**C.** "Robot-42 crashed 3 times at 3 AM. 3 AM must be a problematic time for robots."

---

## Problem 5: Experimental Design

A fleet of robots is experiencing intermittent localization failures. You suspect either (a) a firmware timing bug, (b) floor surface changes, or (c) temperature-related sensor drift.

Design a testing plan using Mill's methods to isolate the cause. Specify:
1. What variables you would control.
2. What you would vary.
3. How you would apply the method of difference.
4. What observations would confirm/disconfirm each hypothesis.

---

## Problem 6: Bayes in Practice

You're investigating a robot failure. Initially, you consider three hypotheses:
- H1: SPI bus failure (prior: 20%)
- H2: Software crash (prior: 50%)
- H3: Power supply issue (prior: 30%)

Evidence E1: The error log shows "DMA timeout."
- P(E1 | H1) = 0.90 (SPI failure almost always causes DMA timeout)
- P(E1 | H2) = 0.15 (some software crashes cause DMA timeout)
- P(E1 | H3) = 0.05 (power issues rarely cause DMA timeout)

Update your beliefs using Bayes' theorem. Which hypothesis is now most likely?

---

## Solutions

<details>
<summary>Click to reveal solutions</summary>

**Problem 1:**

**A.** **Strong.** High number of relevant similarities (same hardware rev, same firmware — these are causally relevant to SPI behavior). The one dissimilarity (floor type) is probably not relevant to SPI issues. Conclusion is appropriately hedged ("probably").

**B.** **Weak.** The similarities are superficial: a laptop and a robot are very different systems. A "software update" and a "firmware update" have different characteristics. The only similarity is "device crashed after update," which is too vague to be compelling.

**C.** **Strong.** Based on 8 diverse instances with 87.5% agreement. The pattern is specific (same symptom, same location, same cause in 7/8 cases). Appropriately probabilistic conclusion.

**Problem 2:**

**A.** Method of Agreement — All failing instances share one common factor (Warehouse A). Conclusion: Something about Warehouse A is probably causing the failures.

**B.** Method of Difference — Same robot, same everything except firmware version. When firmware changes from v1.18 to v1.24, failure stops. Conclusion: The firmware (specifically the difference between v1.18 and v1.24) is probably the cause.

**C.** Method of Concomitant Variation — As temperature increases, current draw increases proportionally. Conclusion: Temperature is probably a causal factor in motor current draw.

**D.** Method of Residues — Known causes account for 9ms of 12ms. The remaining 3ms is attributed to the remaining factor (actuator command transmission).

**Problem 3:**

**A.** P(Rev C) = 100/200 = 0.50

**B.** P(firmware OR hardware) = P(F) + P(H) - P(both) = 0.10 + 0.05 - 0.01 = **0.14**

**C.** P(both fail) = 0.02 × 0.02 = **0.0004** (independent events)

**D.** Bayes' theorem:
- P(positive | no SPI failure) = 1 - specificity = 0.10
- P(SPI failure) = 0.03
- P(no SPI failure) = 0.97

P(positive) = P(pos|SPI) × P(SPI) + P(pos|no SPI) × P(no SPI)
= 0.95 × 0.03 + 0.10 × 0.97
= 0.0285 + 0.097 = 0.1255

P(SPI failure | positive) = (0.95 × 0.03) / 0.1255 = 0.0285 / 0.1255 ≈ **0.227**

Only ~23%! Despite the test's 95% sensitivity, the low base rate (3%) means a positive result is still more likely to be a false positive. You need additional evidence before concluding SPI failure.

**Problem 4:**

**A.** False cause (non causa pro causa) — the crashed robot wasn't even updated. There's no causal mechanism connecting the deployment of one robot's firmware to another robot's crash. This is coincidence, not causation.

**B.** Confounding variable — both ice cream sales and drowning increase because of a third factor (hot weather). This is the classic "correlation ≠ causation" example.

**C.** Insufficient data + possible confounding — 3 crashes is too few to establish a pattern, and "3 AM" might correlate with other factors (shift change, different traffic patterns, lower staffing, temperature changes, scheduled maintenance activities).

**Problem 5:**

**Testing plan:**

1. **Variables to control:** Hardware revision, robot model, payload weight, route, operator procedures.

2. **Variables to vary (one at a time):**
   - Firmware version (v1.18 vs v1.24) — tests hypothesis (a)
   - Floor surface (original vs. new wax/coating) — tests hypothesis (b)
   - Temperature (morning 20°C vs afternoon 35°C, or use climate control) — tests hypothesis (c)

3. **Method of Difference application:**
   - Take one robot. Run it under identical conditions EXCEPT change one variable.
   - Test 1: Same robot, same conditions, firmware v1.18 → observe failure/no failure. Then firmware v1.24 → observe.
   - Test 2: Same robot, same firmware, original floor → observe. Then modified floor → observe.
   - Test 3: Same robot, same firmware, same floor, controlled temperature 20°C → observe. Then 35°C → observe.

4. **Observations:**
   - If failure only occurs with v1.18 and not v1.24 → firmware (a) confirmed.
   - If failure only occurs on new floor surface → floor (b) confirmed.
   - If failure correlates with temperature → sensor drift (c) confirmed.
   - If failure persists regardless → a different cause or combination.

**Problem 6:**

P(E1) = P(E1|H1)×P(H1) + P(E1|H2)×P(H2) + P(E1|H3)×P(H3)
= 0.90×0.20 + 0.15×0.50 + 0.05×0.30
= 0.18 + 0.075 + 0.015 = 0.27

Updated probabilities:
- P(H1|E1) = (0.90 × 0.20) / 0.27 = 0.18 / 0.27 ≈ **0.667 (66.7%)**
- P(H2|E1) = (0.15 × 0.50) / 0.27 = 0.075 / 0.27 ≈ **0.278 (27.8%)**
- P(H3|E1) = (0.05 × 0.30) / 0.27 = 0.015 / 0.27 ≈ **0.056 (5.6%)**

**SPI bus failure (H1)** is now the most likely hypothesis at 66.7%, up from 20%. Software crash dropped from 50% to 28%. Power supply is nearly ruled out at 5.6%.

The DMA timeout evidence dramatically shifted the probability toward SPI failure because DMA timeouts are highly associated with SPI issues and rarely caused by other factors.

</details>
