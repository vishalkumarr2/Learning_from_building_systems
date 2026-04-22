# Exercise 3 — Informal Fallacies

**Covers:** Lesson 03
**Difficulty:** Foundation to Intermediate

---

## Problem 1: Name That Fallacy

Identify the fallacy in each argument.

**A.** "We should use the old firmware because the new one hasn't been proven safe."

**B.** "The intern says the bug is in the motor driver, but what does an intern know?"

**C.** "If we allow one developer to skip code review, soon everyone will skip reviews, the codebase will rot, and the product will fail."

**D.** "The robot crashed right after we changed the floor wax. Therefore, the floor wax caused the crash."

**E.** "Either we rewrite the entire navigation stack from scratch, or we accept that it will always have bugs."

**F.** "Our testing methodology must be good because it was developed by testing experts who are the best in the industry because they developed our methodology."

**G.** "Every component in the robot passed its individual bench test. Therefore, the robot system works correctly."

**H.** "The firmware update passed all 500 unit tests, so it's ready for production."

**I.** "We tested the fix on 2 robots and both passed. The fix works across the fleet."

**J.** "You're criticizing my code? Your last PR had 15 bugs!"

---

## Problem 2: Real-World Scenario

Read the following meeting transcript and identify all fallacies present.

> **Manager:** "We need to decide whether to delay the release or ship with the known SPI timing issue."
> 
> **Dev A:** "We should ship. Everyone else in the industry ships with known issues. That's just how it works." 
>
> **Dev B:** "That's because Dev A wants to go on vacation next week and doesn't want to stay for the fix."
>
> **Dev C:** "If we delay, the customer will lose confidence, they'll cancel the contract, and we'll have to lay off half the team."
>
> **Manager:** "Dev D, you suggested delaying — but last quarter you were the one who wanted to rush the release. So your judgment on timing is clearly unreliable."
>
> **Dev D:** "Look, the point is simple: either we delay and do it right, or we ship garbage."
>
> **Dev E:** "I've been working 80-hour weeks on this. Can we please just ship it?"

List each fallacy with the speaker and the fallacy name.

---

## Problem 3: Fallacy or Valid?

Some of these are fallacies, some are legitimate arguments. Determine which is which.

**A.** "The motor failed after 10,000 hours. The manufacturer rates it for 15,000 hours. Therefore, this motor is defective."

**B.** "Dr. Smith, who specializes in motor control and holds 12 patents in H-bridge design, reviewed the circuit and found no issues. So the circuit is probably fine."

**C.** "We haven't found the bug yet, so it probably doesn't exist."

**D.** "95% of robots that exhibited this symptom pattern had a failed capacitor on the motor board. This robot has the same symptom pattern. So it probably has a failed capacitor."

**E.** "The CEO says our fleet reliability is excellent, so it must be."

---

## Problem 4: Composition and Division

Determine whether each commits the fallacy of composition, division, or neither.

**A.** "Each line of code in the function has been reviewed. Therefore, the function is correct."

**B.** "The team has high expertise. Therefore, each team member has high expertise."

**C.** "Hydrogen is flammable. Oxygen supports combustion. Therefore, water (H₂O) is flammable and supports combustion."

**D.** "The fleet achieved 99.5% uptime this month. Robot-42 is in the fleet. Therefore, Robot-42 achieved 99.5% uptime."

---

## Problem 5: Fix the Fallacy

For each fallacious argument, rewrite it as a legitimate argument (you may need to add information or weaken the conclusion).

**A.** (Hasty generalization) "We tested on 2 robots and both passed. The fix works across the fleet."

**B.** (False cause) "We deployed the update Monday. Tuesday the robot crashed. The update caused the crash."

**C.** (Appeal to ignorance) "No one has proven the firmware has memory leaks, so it doesn't."

---

## Solutions

<details>
<summary>Click to reveal solutions</summary>

**Problem 1:**

**A.** Appeal to Ignorance (ad ignorantiam) — absence of proof of safety ≠ proof of danger.

**B.** Ad Hominem (abusive) — attacking the intern's status instead of evaluating the argument.

**C.** Slippery Slope — no evidence that one exception leads to total breakdown.

**D.** False Cause (post hoc) — temporal sequence doesn't prove causation.

**E.** False Dichotomy — other options exist (fix specific bugs, refactor critical sections, etc.).

**F.** Begging the Question (circular) — the expertise is "proven" by the methodology, which is "proven" by the expertise.

**G.** Composition — properties of individual parts don't necessarily apply to the whole system.

**H.** Suppressed Evidence — unit tests don't cover integration, hardware-in-the-loop, or edge cases.

**I.** Hasty Generalization — 2 out of a fleet of potentially hundreds is insufficient.

**J.** Ad Hominem (tu quoque / "you too") — past mistakes don't invalidate current criticism.

**Problem 2:**

- **Dev A:** Appeal to the People (ad populum / bandwagon) — "everyone does it" is not evidence it's right.
- **Dev B:** Ad Hominem (circumstantial) — attacking Dev A's motivation rather than the argument.
- **Dev C:** Slippery Slope — the chain from delay → lost confidence → cancelled contract → layoffs is unsupported.
- **Manager:** Ad Hominem (tu quoque) — Dev D's past inconsistency doesn't invalidate the current argument.
- **Dev D:** False Dichotomy — "do it right" vs. "ship garbage" are not the only options.
- **Dev E:** Appeal to Pity (ad misericordiam) — personal suffering doesn't make the firmware ready.

**Problem 3:**

**A.** Legitimate argument (inductive). The motor failed significantly before its rated lifetime. The conclusion "defective" is reasonable given the evidence, though other factors could contribute (overloading, overheating).

**B.** Legitimate appeal to authority. Dr. Smith is qualified in the relevant field. This is proper use of expert authority.

**C.** Appeal to Ignorance — fallacy. Not finding the bug doesn't prove it doesn't exist. Maybe you haven't looked in the right place.

**D.** Legitimate inductive argument (strong generalization from a large sample). 95% with the same symptoms had the same cause — this is a strong basis for probable diagnosis.

**E.** Appeal to Unqualified Authority — fallacy. The CEO is not an authority on fleet reliability metrics.

**Problem 4:**

**A.** Composition — reviewing each line doesn't guarantee the function is correct (logic errors, interaction between lines).

**B.** Division — the team's aggregate expertise doesn't mean every individual is an expert.

**C.** Composition — properties of parts (hydrogen, oxygen) don't transfer to their chemical compound (water).

**D.** Division — fleet-wide average uptime doesn't apply to individual robots. Robot-42 might have been down for 3 days.

**Problem 5:**

**A.** "We tested the fix on 2 robots and both passed. This is encouraging initial evidence, but we need to test on a larger sample (at least 20 robots across different hardware revisions and environments) before concluding the fix works fleet-wide."

**B.** "We deployed the update Monday. Tuesday the robot crashed. The timing is suspicious, so the update is one hypothesis. To test it: we should check if the crash log shows behavior related to the updated code, test whether reverting the update prevents the crash, and check if any other changes occurred between Monday and Tuesday."

**C.** "We haven't found evidence of memory leaks in 72 hours of profiling under production load. While this doesn't prove the firmware is leak-free, it provides reasonable confidence for the tested conditions. Extended testing under stress conditions would strengthen this conclusion."

</details>
