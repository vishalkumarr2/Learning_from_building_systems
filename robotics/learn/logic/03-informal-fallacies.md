# 03 — Informal Fallacies
### Recognizing bad reasoning in everyday arguments

**Covers:** Chapter 3 — Informal Fallacies (§3.1–3.5)
**Prerequisite:** Lessons 01–02
**Unlocks:** Lesson 04 (Categorical Logic)

---

## Why Should I Care?

Fallacies are errors in reasoning that *look* convincing but aren't. In debugging, design reviews, and incident analysis, fallacies waste time and lead to wrong conclusions. Learning to spot them is like learning to spot anti-patterns in code — once you see them, you can't unsee them.

---

# PART 1 — FALLACIES OF RELEVANCE

These fallacies occur when the premises are logically irrelevant to the conclusion.

## 1.1 Appeal to Force (*Argumentum ad Baculum*)

Using threats instead of evidence.

> "If you don't approve this firmware release, we'll miss the deadline and it'll be on you."

The threat of blame is not evidence that the firmware is ready. The firmware is either tested and stable or it isn't — pressure doesn't change that.

## 1.2 Appeal to Pity (*Argumentum ad Misericordiam*)

Using sympathy instead of evidence.

> "I know the code doesn't pass the linter, but I stayed up until 3 AM writing it."

The programmer's suffering doesn't make the code correct.

## 1.3 Appeal to the People (*Argumentum ad Populum*)

Using popularity or group pressure as evidence.

> "Everyone on the team thinks the bug is in the firmware, so it must be in the firmware."

Popularity doesn't determine truth. The entire team can be wrong. Check the evidence.

**Bandwagon variant:** "All the other robotics companies use this approach."
**Snob appeal variant:** "Only the best engineers use formal verification."

## 1.4 Argument Against the Person (*Ad Hominem*)

Attacking the person making the argument instead of the argument itself.

**Direct (abusive):**
> "You've only been here 6 months — you can't possibly understand the estimator code."

Experience doesn't determine whether an argument is valid. A junior engineer can make a sound argument; a senior engineer can make a fallacious one.

**Circumstantial:**
> "Of course the firmware team says the bug isn't in firmware — they don't want to do more work."

Maybe they're biased, but the argument still needs to be evaluated on its merits. Even biased people can be right.

**Tu quoque (you too):**
> "You're telling me my code has bugs? Your last PR had three regressions!"

Past mistakes don't invalidate current observations.

## 1.5 Straw Man

Distorting someone's argument to make it easier to attack.

> A: "We should add more unit tests to the estimator."
> B: "So you want us to stop all development and only write tests? That's impractical."

A didn't say "stop all development." B distorted the argument to an extreme version and attacked that instead.

**How to spot it:** Ask "Is B attacking what A actually said, or a caricatured version?"

## 1.6 Red Herring

Introducing an irrelevant topic to divert attention.

> "We need to fix the SPI timing bug."
> "But have you seen how outdated our CI pipeline is? We should fix that first."

The CI pipeline may need fixing, but it's irrelevant to the SPI bug discussion.

## 1.7 Missing the Point (*Ignoratio Elenchi*)

Drawing a conclusion different from what the premises support.

> "The robot has failed 15 times this week. Therefore, we need to buy new robots."

The premises support "the robot has a reliability problem." The conclusion jumps to "buy new robots" when the fix might be a firmware update, recalibration, or environmental change.

---

# PART 2 — FALLACIES OF WEAK INDUCTION

These fallacies occur when the premises provide insufficient support for the conclusion.

## 2.1 Appeal to Unqualified Authority (*Ad Verecundiam*)

Citing an authority who isn't an expert in the relevant field.

> "Our CEO says the motor driver design is fine, so it's fine."

Unless the CEO is an electrical engineer who reviewed the schematic, their opinion on motor driver design isn't authoritative.

**Legitimate use of authority:** "Dr. X, who has 20 years in motor control and published 30 papers on H-bridge design, reviewed the circuit and found no issues." — This is reasonable, because the authority is qualified in the relevant field.

## 2.2 Appeal to Ignorance (*Ad Ignorantiam*)

Arguing that something is true because it hasn't been proven false (or vice versa).

> "We haven't found any bugs in the new firmware, so it must be bug-free."

Absence of evidence is not evidence of absence. You might not have tested the right conditions.

> "Nobody has proven that the SPI timing fix works on all hardware revisions, so it probably doesn't."

The lack of proof doesn't mean it fails. It means you need to test it.

## 2.3 Hasty Generalization

Drawing a broad conclusion from too few examples.

> "We tested on two robots and both passed. The fix works across the fleet."

Two robots out of hundreds is not a representative sample. Maybe those two had different hardware revisions, different environments, or different usage patterns.

**Engineering rule of thumb:** Sample size matters. State it explicitly. "Tested on N robots across M hardware revisions over T hours" lets others judge the strength of your generalization.

## 2.4 False Cause

Assuming causation from correlation or temporal sequence.

**Post hoc ergo propter hoc** ("after this, therefore because of this"):
> "We deployed firmware v1.24 on Monday. On Tuesday, the robot crashed. Therefore v1.24 caused the crash."

Maybe it did. Or maybe the crash was caused by something else that happened Tuesday (new floor wax, temperature change, different payload). Temporal sequence alone doesn't prove causation.

**Non causa pro causa** (mistaking the cause):
> "The robot always crashes in aisle 7. There must be something wrong with aisle 7."

Maybe. Or maybe aisle 7 is where the robot makes a specific turn that triggers an estimator edge case. The location is correlated but the geometry is the cause.

**How to establish actual causation:** Use Mill's methods (Lesson 08) — method of difference, method of agreement, concomitant variation. Isolate variables.

## 2.5 Slippery Slope

Claiming that one step will inevitably lead to a chain of increasingly bad consequences, without evidence for the chain.

> "If we allow one exception to the coding standard, soon nobody will follow any standards, the codebase will become unmaintainable, and the product will fail."

Each step in the chain needs its own justification. One exception doesn't necessarily lead to total anarchy.

## 2.6 Weak Analogy

Drawing a conclusion based on a comparison that doesn't hold.

> "Debugging robots is like debugging cars — just check the engine light. We should add a single diagnostic indicator to the robot."

Robots have dozens of subsystems with complex interactions. Cars do too, but the "engine light" analogy oversimplifies. The comparison is too weak to support the specific conclusion.

---

# PART 3 — FALLACIES OF PRESUMPTION

These fallacies presuppose what they're trying to prove or assume something not in evidence.

## 3.1 Begging the Question (*Petitio Principii*)

The conclusion is assumed in the premises (circular reasoning).

> "This firmware is reliable because it was written by our most reliable programmer."

How do we know the programmer is reliable? Presumably because they write reliable firmware. The argument goes in a circle.

**Subtler form in debugging:**
> "The estimator is wrong because the sensorbar data is bad. How do we know the sensorbar data is bad? Because the estimator shows the wrong position."

This is circular — each claim depends on the other. You need independent evidence.

## 3.2 False Dichotomy (False Dilemma)

Presenting only two options when more exist.

> "Either we ship the firmware now with known bugs, or we delay the entire project by 6 months."

Other options: ship with a partial fix now and patch later; ship to a subset of robots; fix only the critical bugs and defer the minor ones.

**In debugging:**
> "The problem is either hardware or software."

What about firmware? Configuration? Environment? Network? Power supply? The real world rarely has only two options.

## 3.3 Complex Question

Asking a question that presupposes something unproven.

> "Why did the firmware update break the system?"

This presupposes that the firmware update *did* break the system. If that hasn't been established, the question is loaded. Better: "Did the firmware update affect system behavior? If so, how?"

## 3.4 Suppressed Evidence

Ignoring evidence that counts against the conclusion.

> "Firmware v1.24 is fully tested — it passed all 200 unit tests."

This suppresses the fact that there are no integration tests, no hardware-in-the-loop tests, and three known edge cases that unit tests don't cover.

**In RCA reports:** Always include negative results and ruled-out hypotheses, not just the evidence that supports your conclusion.

---

# PART 4 — FALLACIES OF AMBIGUITY AND GRAMMATICAL ANALOGY

## 4.1 Equivocation

Using a word with two different meanings in the same argument.

> "The sensorbar gives us a good **reading**. A good **reading** is important for education. Therefore, the sensorbar is important for education."

"Reading" means different things in the two premises. This is an extreme example, but subtler equivocations happen:

> "The system is **stable** — it hasn't crashed in 24 hours. Since it's **stable**, we can increase the gain."

"Stable" in the first sentence means "hasn't crashed" (operational stability). "Stable" in the second means "control-theoretic stability" (bounded output for bounded input). These are different properties.

## 4.2 Amphiboly

Ambiguous sentence structure leads to wrong conclusions.

> "The engineer told the manager that he needed to update the firmware."

Who needs to update the firmware — the engineer or the manager? The sentence structure is ambiguous.

## 4.3 Composition and Division

**Composition:** Assuming what's true of the parts is true of the whole.
> "Each component in the system works perfectly. Therefore, the system works perfectly."

Integration bugs exist precisely because this fallacy is false. Components can work individually but fail when combined.

**Division:** Assuming what's true of the whole is true of each part.
> "The fleet has a 99.9% uptime rate. Therefore, each robot has 99.9% uptime."

The fleet average can mask individual robots with much lower uptime.

---

## Quick Reference: Fallacy Cheat Sheet

| Fallacy | Pattern | Engineering Example |
|---|---|---|
| Ad Hominem | "You're wrong because of who you are" | "Junior devs don't understand the codebase" |
| Straw Man | "You said X (distorted)" | "So you want us to stop all development?" |
| False Cause | "A then B, so A caused B" | "Deployed Monday, crashed Tuesday" |
| Hasty Generalization | "N=2, so it's universal" | "Two robots passed, fix works everywhere" |
| Appeal to Ignorance | "Not proven false → true" | "No bugs found → bug-free" |
| False Dichotomy | "Only two options" | "Ship buggy or delay 6 months" |
| Begging the Question | "Conclusion in premise" | "It's reliable because the reliable dev wrote it" |
| Composition | "Parts true → whole true" | "Each module works → system works" |
| Equivocation | "Same word, different meaning" | "'Stable' = no crashes vs. control theory" |
| Suppressed Evidence | "Ignore contrary data" | "Passed unit tests (no integration tests)" |

---

## Key Takeaways

1. **Learn the names.** When you can name a fallacy, you can call it out precisely: "That's a hasty generalization — we tested two robots, not the fleet."
2. **Most debugging fallacies are false cause and hasty generalization.** Be especially vigilant about these.
3. **Straw man and ad hominem poison team discussions.** Watch for them in meetings.
4. **Composition is the integration test fallacy.** Individual tests passing doesn't prove system correctness.
5. **Always check for suppressed evidence** in RCA reports and test summaries.
6. **Not every fallacy makes the conclusion false** — it just means the *reasoning* is flawed. The conclusion might still be true, but for different reasons.
