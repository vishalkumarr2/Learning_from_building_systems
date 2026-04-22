# 02 — Language, Meaning & Definition
### Precision in communication — the foundation of clear reasoning

**Covers:** Chapter 2 — Language: Meaning and Definition (§2.1–2.4)
**Prerequisite:** Lesson 01 (Arguments, Premises & Conclusions)
**Unlocks:** Lesson 03 (Informal Fallacies)

---

## Why Should I Care?

Ambiguous language is the #1 source of miscommunication in engineering. Consider:

> Bug report: "The robot is slow."

Slow how? Slow velocity? Slow response time? Slow to boot? Slow compared to what? If you don't pin down what words mean, you can't even formulate a proper argument — let alone evaluate one.

Logic requires precise language. This lesson teaches you to spot and fix the language problems that poison reasoning before it starts.

---

# PART 1 — VARIETIES OF MEANING

## 1.1 Cognitive vs. Emotive Meaning

Every word or phrase can carry two kinds of meaning:

**Cognitive meaning** — the factual, informational content. What the word refers to or describes.
- "The motor temperature is 85°C" — pure cognitive meaning.

**Emotive meaning** — the feelings or attitudes the word evokes.
- "The motor is dangerously overheating" — same fact, but "dangerously" adds alarm.

In engineering, you want to maximize cognitive meaning and minimize emotive meaning. Compare:

| Emotive (bad for reports) | Cognitive (good for reports) |
|---|---|
| "The firmware is a disaster" | "Firmware v1.18 has 3 critical bugs in the SPI driver" |
| "The robot keeps crashing constantly" | "The robot stopped 7 times in an 8-hour shift" |
| "Management's ridiculous deadline" | "The deadline requires 40% more throughput than current capacity" |

**Why this matters for arguments:** Emotive language can make a weak argument *feel* strong. "This catastrophic firmware failure will obviously destroy our fleet reliability" *feels* more persuasive than "This firmware bug affects 3% of robots under specific conditions." But the second is more useful for decision-making.

## 1.2 Intension vs. Extension

Two ways to specify what a word means:

**Extension (denotation):** The set of all things the word refers to.
- Extension of "OKS robot" = {oksbot-1, oksbot-2, ..., oksbot-N} — every actual OKS robot.
- Extension of "prime number" = {2, 3, 5, 7, 11, 13, ...}

**Intension (connotation):** The properties or attributes that define membership.
- Intension of "OKS robot" = an autonomous mobile robot manufactured by Rapyuta Robotics using the OKS platform.
- Intension of "prime number" = a natural number greater than 1 that has no positive divisors other than 1 and itself.

**Why this matters:** Two words can have the same extension but different intension.
- "Creatures with a heart" and "creatures with a kidney" have (nearly) the same extension — the same set of animals. But the intension is completely different.
- In practice: "robots that failed last Tuesday" and "robots running firmware v1.18" might have the same extension (same set of robots). But the intension is different — one describes a time-based event, the other a configuration state. Confusing the two leads to wrong conclusions about causation.

## 1.3 Levels of Meaning

Words have a hierarchy of intension:
1. **Conventional intension:** The common dictionary definition.
2. **Stipulative intension:** A special meaning assigned for a specific context.

In engineering, we constantly use stipulative definitions:
- "DELOCALIZED" doesn't mean what a dictionary says — it means the robot's estimated position diverged from ground truth beyond a threshold.
- "STALL" in our context means the sensorbar sequence counter didn't increment for N cycles.

**Rule:** When using technical terms in arguments, always make sure everyone agrees on the stipulative definition. Many debugging debates are actually definition disputes.

---

# PART 2 — LANGUAGE PITFALLS

## 2.1 Ambiguity

A word or expression is **ambiguous** if it has two or more distinct meanings in the context used.

**Three types:**

### Semantic Ambiguity
A single word has multiple meanings:
- "The port is not responding." — Network port? Serial port? USB port?
- "The bus is locked." — SPI bus? CAN bus? System bus?
- "Check the log." — ROS log? System log? Application log? Physical logbook?

### Syntactic (Grammatical) Ambiguity
The sentence structure allows multiple interpretations:
- "The robot hit the wall going 2 m/s." — Was the robot going 2 m/s, or was the wall going 2 m/s?
- "We need to test robots with sensors." — Do we need robots that have sensors, or do we need to use sensors to test robots?

### Cross-Reference Ambiguity
Pronouns or references are unclear:
- "The estimator gets data from the sensorbar and the IMU. It sometimes fails." — What fails? The estimator, the sensorbar, or the IMU?

**Fix:** In technical writing, eliminate ambiguity by being specific:
- ~~"The port is not responding"~~ → "TCP port 8080 is not responding to health checks"
- ~~"It sometimes fails"~~ → "The estimator occasionally diverges when sensorbar data is stale"

## 2.2 Vagueness

A word is **vague** when it has a fuzzy boundary — there's no clear line between what it applies to and what it doesn't.

- "The robot is going **fast**." — How fast? 0.3 m/s? 0.5 m/s? 1.0 m/s?
- "The battery is **low**." — 20%? 10%? 5%?
- "The temperature is **high**." — 60°C? 80°C? 100°C?
- "The fix was tested **a lot**." — 5 times? 50 times? 500 times?

**Ambiguity vs. Vagueness:**
- Ambiguous = multiple distinct meanings (which one do you mean?)
- Vague = one meaning with fuzzy boundaries (where exactly is the cutoff?)

**Engineering solution:** Replace vague terms with precise values or thresholds:
- ~~"high temperature"~~ → "above 75°C (the thermal protection threshold)"
- ~~"tested a lot"~~ → "tested on 20 robots across 3 sites over 48 hours"
- ~~"low battery"~~ → "battery below 15% SoC"

## 2.3 Disputes: Verbal vs. Genuine

When two engineers disagree, the first question to ask is: **Is this a genuine disagreement about facts, or a verbal dispute about definitions?**

**Verbal dispute (apparent disagreement):**
> A: "The robot delocalized." B: "No it didn't — it just had high covariance for 2 seconds."

Are they disagreeing about what happened, or about what counts as "delocalized"? If the disagreement is about the *threshold* for calling something delocalization (>0.5m error? >1.0m error? any covariance spike?), it's a verbal dispute. Resolve it by agreeing on the definition.

**Genuine dispute:**
> A: "The firmware update caused the failure." B: "No, it was a hardware defect."

This is a factual disagreement. No amount of redefining words will resolve it — you need evidence.

**Merely verbal disputes waste enormous debugging time.** Before arguing about a conclusion, make sure you agree on what the key terms mean.

---

# PART 3 — TYPES OF DEFINITIONS

## 3.1 Five Types

| Type | Purpose | Example |
|---|---|---|
| **Stipulative** | Assigns a new meaning to a new or existing term | "For this RCA, 'STALL' means sensorbar sequence unchanged for ≥5 cycles" |
| **Lexical** | Reports existing standard meaning | "A motor is a machine that converts electrical energy into mechanical motion" |
| **Précising** | Reduces vagueness of an existing term | "'High temperature' for this system means above 75°C at the motor driver" |
| **Theoretical** | Provides meaning within a theory or framework | "'Delocalization' means the Mahalanobis distance between estimated and true pose exceeds 3σ" |
| **Persuasive** | Assigns meaning to influence attitudes (use with caution) | "'Real engineering' means using formal verification" — smuggles in a value judgment |

## 3.2 Techniques of Definition

**Extensional (denotative)** — list members or point to examples:
- "Communication protocols include UART, SPI, I2C, and CAN."
- Limitation: you can't list everything; the list might be incomplete or misleading.

**Intensional (connotative)** — state the properties:
- "A communication protocol is a set of rules governing the format, timing, and error handling of data exchange between devices."
- Better for precision; captures *all* members, even ones you haven't encountered.

**Genus and Difference** (the gold standard):
1. Name the broader category (genus)
2. State what distinguishes this term from others in that category (difference)

Example: "A **sensorbar** (genus: sensor module) is a linear array of optical encoders that measures wheel displacement by detecting alternating stripes on a code strip, providing incremental position feedback to the navigation estimator."

## 3.3 Rules for Good Definitions

1. **Not too broad:** "A robot is a machine" — too broad (a toaster is also a machine).
2. **Not too narrow:** "A robot is a wheeled autonomous warehouse vehicle" — too narrow (excludes drones, arms, legged robots).
3. **Not circular:** "Logic is the study of logical reasoning" — tells you nothing.
4. **Not negative when positive is possible:** "A robot is a machine that is not a computer" — unhelpful. Say what it *is*.
5. **Not figurative or obscure:** "Logic is the music of reason" — poetic but useless as a definition.
6. **Not emotive:** "A bug is a stupid mistake in code" — "stupid" adds nothing factual.

---

## Key Takeaways

1. **Separate cognitive from emotive meaning** in technical communication. Facts, not feelings.
2. **Eliminate ambiguity** by being specific — name the exact component, measurement, threshold.
3. **Eliminate vagueness** by quantifying — numbers, thresholds, counts.
4. **Check for verbal disputes** before debugging disagreements. Do you actually disagree, or are you using the same word differently?
5. **Define key terms** at the start of any investigation. Stipulative and précising definitions prevent confusion.
6. **Use genus-and-difference** for technical definitions — it's the clearest format.
