# 01 — Arguments, Premises & Conclusions
### The building blocks of all reasoning

**Covers:** Chapter 1 — Basic Concepts (§1.1–1.5)
**Prerequisite:** None
**Unlocks:** All subsequent lessons

---

## Why Should I Care?

Every time you debug a system, write an RCA report, or argue for a design decision, you're constructing arguments. The question is whether you're constructing *good* ones.

Consider this from a real incident report:
> "The robot stopped because the sensorbar reported stale data. The sensorbar data was stale because the SPI bus locked up. The SPI bus locked up because of a firmware timing conflict."

That's a chain of three arguments. Each link claims that one thing follows from another. If any link is broken — if the SPI bus lockup *doesn't actually cause* stale data, or if the stale data wasn't what stopped the robot — the whole chain collapses. Logic gives you the tools to evaluate each link.

---

# PART 1 — WHAT IS AN ARGUMENT?

## 1.1 Statements

A **statement** (or **proposition**) is a sentence that is either true or false.

**Statements:**
- "The battery voltage is 12.4V" (true or false — you can measure it)
- "All OKS robots use differential drive" (true or false — you can check)
- "Python is faster than C++" (false, but it's still a statement — it has a truth value)

**Not statements:**
- "Restart the robot" (command — neither true nor false)
- "Is the sensorbar working?" (question)
- "Wow, that crash was spectacular!" (exclamation)
- "Let's try firmware v1.24" (proposal)

**Key insight:** Only statements can be premises or conclusions of arguments. If someone says "We should deploy firmware v1.24" — that's a proposal, not an argument. But "We should deploy firmware v1.24 *because* it fixes the SPI timing bug and passes all regression tests" — *that's* an argument. It has two premises and a conclusion.

## 1.2 Arguments Defined

An **argument** is a group of statements where one or more (the **premises**) are claimed to provide support for another (the **conclusion**).

```
Premises ──(claimed support)──→ Conclusion
(evidence)                      (what follows)
```

Every argument has exactly one conclusion but can have any number of premises.

**Example — Bug triage:**
- P1: The crash log shows a segfault in `estimator_update()`.
- P2: The last code change modified `estimator_update()` to use an uninitialized pointer.
- P3: Uninitialized pointers cause segfaults.
- **∴ C:** The last code change caused the crash.

(∴ means "therefore")

## 1.3 Identifying Premises and Conclusions

**Conclusion indicators** — words that signal what follows is the conclusion:
- therefore, hence, thus, consequently, so
- it follows that, we may conclude, which means that, accordingly

**Premise indicators** — words that signal what follows is a premise:
- since, because, for, as, given that
- seeing that, inasmuch as, for the reason that, owing to

**Example:** "Since the motor current spiked to 15A (*premise*), the H-bridge must have had a short (*conclusion*)."

**Example:** "The firmware must be reflashed (*conclusion*), because the checksum doesn't match (*premise*)."

**Trick:** Sometimes there are no indicator words. You have to ask: "What is being *claimed*? What is the *evidence* for that claim?"

> "The robot delocalized on the north corridor. The floor there was just waxed. The encoders slip on waxed floors."

No indicator words, but the structure is:
- P1: The floor in the north corridor was just waxed.
- P2: Encoders slip on waxed floors.
- C: The robot delocalized on the north corridor [because of encoder slip on waxed floor].

## 1.4 What Is NOT an Argument

Not every group of statements is an argument. You need to distinguish:

**1. Reports / Descriptions** — statements about a situation with no claim of inference:
> "At 03:42, the robot entered tile B7. Battery was at 45%. Speed was 0.3 m/s."

This is a sequence of facts. No statement is offered as a *reason* for another.

**2. Illustrations** — examples used to clarify, not to prove:
> "Robots can have many types of sensors. For example, LIDAR, cameras, ultrasonic."

"For example" is an illustration indicator, not a premise indicator.

**3. Explanations** — saying *why* something happened (assumed to be true), not *that* it happened:
> "The motor overheated because the ambient temperature was 45°C."

**Explanation vs. argument confusion:** This is subtle. An *explanation* takes the fact as given and says *why*. An *argument* tries to *establish* the fact. Context matters:
- If everyone agrees the motor overheated and you're explaining why → explanation
- If you're trying to convince someone the motor will overheat under these conditions → argument

**4. Conditional statements** — "if...then..." by itself is not an argument:
> "If the battery drops below 10%, the robot will dock."

This is a single conditional statement, not an argument. But:
> "If the battery drops below 10%, the robot will dock. The battery is at 8%. Therefore, the robot will dock."

*That's* an argument (and a valid one — this is called *modus ponens*, which we'll study in Lesson 05).

---

# PART 2 — DEDUCTION AND INDUCTION

## 2.1 Two Types of Reasoning

All arguments fall into one of two categories:

**Deductive arguments** claim that if the premises are true, the conclusion *must* be true. The conclusion follows with **necessity**.

**Inductive arguments** claim that if the premises are true, the conclusion is *probably* true. The conclusion follows with **probability**.

| | Deductive | Inductive |
|---|---|---|
| Claim | Conclusion *must* follow | Conclusion *probably* follows |
| Based on | Logic, math, definitions | Experience, patterns, sampling |
| Strength | Valid or invalid | Strong or weak |

**Deductive example:**
- P1: All firmware versions before v1.20 have the SPI timing bug.
- P2: This robot is running firmware v1.18.
- C: This robot has the SPI timing bug. ✓ (must be true if premises are true)

**Inductive example:**
- P1: The last 12 robots we tested with firmware v1.24 passed the sensorbar regression.
- C: Firmware v1.24 fixes the sensorbar regression issue. (probably true, but the 13th robot might fail)

## 2.2 How to Tell Them Apart

Five indicators that an argument is **deductive:**
1. **Argument from math:** "The circumference equals 2πr, so..."
2. **Argument from definition:** "A bachelor is an unmarried man, so..."
3. **Categorical syllogism:** "All X are Y. All Y are Z. So all X are Z."
4. **Hypothetical syllogism:** "If A then B. If B then C. So if A then C."
5. **Disjunctive syllogism:** "Either A or B. Not A. So B."

Five indicators that an argument is **inductive:**
1. **Generalization:** "Every robot we tested had this bug, so all robots probably do."
2. **Prediction:** "It failed in the last 5 tests, so it will probably fail again."
3. **Argument from analogy:** "Robot A had this symptom and the cause was X. Robot B has the same symptom, so probably the same cause."
4. **Argument from authority:** "The firmware lead says this is safe, so it's safe."
5. **Causal inference:** "The crash started after the firmware update, so the update probably caused it."

---

# PART 3 — EVALUATING ARGUMENTS

## 3.1 Deductive Arguments: Validity and Soundness

A deductive argument is **valid** if, assuming the premises are true, the conclusion *must* be true. Validity is about the *logical form*, not about whether the premises are actually true.

**Valid argument with false premises:**
- P1: All robots are made of cheese. (false)
- P2: My laptop is a robot. (false)
- C: My laptop is made of cheese.

This is **valid** — the conclusion *does* follow from the premises. If those premises *were* true, the conclusion would *have to be* true. It's just not **sound**.

A deductive argument is **sound** if it is (1) valid AND (2) all premises are actually true.

| | Premises true? | Valid form? | Result |
|---|---|---|---|
| Sound | ✓ | ✓ | Best case — conclusion guaranteed true |
| Valid but unsound | ✗ | ✓ | Conclusion may or may not be true |
| Invalid | — | ✗ | Nothing guaranteed, even if premises true |

**Engineering application:** When reviewing an RCA, check two things:
1. **Validity:** "Even if I grant all these facts, does the conclusion actually follow?"
2. **Soundness:** "Are the stated facts actually correct?"

An RCA can be logically valid but unsound (the reasoning is correct but a premise is wrong — e.g., "the firmware was v1.18" when it was actually v1.20). Or it can have all true premises but be invalid (the facts are right but the conclusion doesn't follow from them).

## 3.2 Inductive Arguments: Strength and Cogency

An inductive argument is **strong** if, assuming the premises are true, the conclusion is *probably* true.

An inductive argument is **cogent** if it is (1) strong AND (2) all premises are actually true.

**Strong:**
- P1: 95 out of 100 robots in the fleet had the same failure mode after the firmware update.
- C: The firmware update caused the failure mode.
(Strong — 95% is compelling evidence)

**Weak:**
- P1: 2 out of 100 robots had the failure mode after the firmware update.
- C: The firmware update caused the failure mode.
(Weak — 2% could easily be coincidence)

## 3.3 Argument Forms: A Preview

The *form* of an argument can be separated from its *content*. Consider:

**Form 1 (Modus Ponens — valid):**
- If P then Q
- P
- Therefore, Q

**Form 2 (Affirming the Consequent — invalid):**
- If P then Q
- Q
- Therefore, P

**Example of Form 2 (invalid):**
- If the SPI bus locked up, the sensorbar reports stale data.
- The sensorbar reports stale data.
- Therefore, the SPI bus locked up.

This is **invalid!** The sensorbar could report stale data for other reasons (loose connector, software bug, power glitch). Just because the consequent is true doesn't mean the antecedent caused it. This is one of the most common reasoning errors in debugging — seeing a symptom and jumping to a specific cause without ruling out alternatives.

**Form 3 (Modus Tollens — valid):**
- If P then Q
- Not Q
- Therefore, not P

**Example:**
- If the SPI bus locked up, the sensorbar *would* report stale data.
- The sensorbar is *not* reporting stale data.
- Therefore, the SPI bus did *not* lock up.

This one is valid. If the SPI lockup *always* causes stale data, and we don't see stale data, we can safely rule out SPI lockup.

## 3.4 Counterexample Method

To prove an argument form **invalid**, find a **counterexample**: a substitution of the variables that makes all premises true but the conclusion false.

**Example:** Is this form valid? "If P then Q. Q. Therefore P."

Counterexample: Let P = "I am a billionaire" and Q = "I am a mammal."
- If I am a billionaire, then I am a mammal. (True — billionaires are humans, humans are mammals)
- I am a mammal. (True)
- Therefore, I am a billionaire. (False!)

Premises true, conclusion false → **invalid form**.

---

# PART 4 — PUTTING IT TOGETHER

## 4.1 Argument Diagramming

Complex reasoning often has intermediate conclusions (subconclausions) that serve as premises for the final conclusion.

**Linear chain:**
```
P1 → Subconclusion 1 → Subconclusion 2 → Final Conclusion
```

**Convergent:**
```
P1 ─┐
    ├──→ Conclusion
P2 ─┘
```
(Independent premises each support the conclusion separately)

**Linked:**
```
P1 + P2 ──→ Conclusion
```
(Premises work *together* — neither supports the conclusion alone)

**RCA example (chain):**
```
"SPI clock jitter > 5ns"
        │
        ▼
"DMA transfer corrupted"
        │
        ▼
"Sensorbar reads wrong position"
        │
        ▼
"Estimator diverges from ground truth"
        │
        ▼
"Robot stops (DELOCALIZED)"
```

Each arrow is an argument. The output of one becomes the input to the next. If you break *any* link in the chain, the overall argument fails.

## 4.2 Common Patterns in Engineering Arguments

**Pattern: Elimination**
> "The failure is either in hardware, firmware, or software. We ruled out hardware (bench test passed). We ruled out software (same binary on working robots). Therefore, it's firmware."

This is a valid disjunctive syllogism — but only if the three options are *truly exhaustive*. What about environmental factors? Power supply? Network?

**Pattern: Causal chain**
> "The firmware update changed the SPI clock divider. Changing the clock divider alters bus timing. Altered bus timing causes DMA misalignment. Therefore, the firmware update causes DMA misalignment."

This is a hypothetical syllogism chain — valid in form. But check: does changing the clock divider *always* alter timing in a problematic way, or only under certain conditions?

**Pattern: Generalization from testing**
> "We tested 20 robots with the fix and none reproduced the issue. Therefore, the fix works."

This is induction. It's *strong* if the 20 robots represent a good sample (different hardware revisions, different environments). It's *weak* if they were all tested in the same ideal conditions.

---

## Key Takeaways

1. **An argument = premises + conclusion.** Learn to identify both, especially when indicator words are missing.
2. **Deductive = must follow; Inductive = probably follows.** Most engineering arguments are inductive.
3. **Valid ≠ True.** A valid argument can have false premises. A sound argument is valid *and* has true premises.
4. **Check the form.** Modus ponens is valid; affirming the consequent is not. Learn to recognize the difference.
5. **The counterexample method** is your most powerful tool for showing an argument is invalid.
6. **RCA chains** are only as strong as their weakest link. Verify each step independently.
