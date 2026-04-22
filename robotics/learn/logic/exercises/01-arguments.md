# Exercise 1 — Arguments, Premises & Conclusions

**Covers:** Lesson 01
**Difficulty:** Foundation

---

## Problem 1: Identify the Argument

For each passage, determine: (a) Is it an argument? (b) If yes, identify the premises and conclusion.

**A.** "The robot's battery is at 8%. Low battery causes reduced motor torque. Reduced torque causes wheel slip. Therefore, the robot is likely experiencing wheel slip."

**B.** "At 14:32, the robot entered tile C4. At 14:33, it stopped. The operator pressed the emergency stop button."

**C.** "Since all firmware versions before v1.20 have the SPI timing bug, and this robot is running v1.17, this robot has the SPI timing bug."

**D.** "If the ambient temperature exceeds 45°C, the thermal protection will engage."

**E.** "Renewable energy is important because fossil fuels contribute to climate change and renewable sources produce no direct emissions."

---

## Problem 2: Indicator Words

Identify the conclusion indicator or premise indicator in each argument, then state the conclusion.

**A.** "The motor current spiked to 15A. Consequently, the H-bridge protection circuit activated."

**B.** "The test suite should be expanded, given that three bugs were missed in the last release cycle."

**C.** "Owing to the fact that the sensorbar cable was loose, the position readings were intermittent. This means the estimator had insufficient data."

---

## Problem 3: Deductive or Inductive?

Classify each argument as deductive or inductive. Explain your reasoning.

**A.** "All robots with hardware revision C have the grounding issue. Robot-42 has hardware revision C. Therefore, Robot-42 has the grounding issue."

**B.** "The last 15 robots we updated to firmware v1.24 showed no SPI issues. Therefore, firmware v1.24 probably fixes the SPI issue."

**C.** "Either the fault is in the motor driver or the motor itself. Bench testing shows the motor is fine. Therefore, the fault is in the motor driver."

**D.** "Robot-87 has the same symptoms as Robot-42 did last month. Robot-42's problem was a bad capacitor on the motor board. So Robot-87 probably has a bad capacitor too."

---

## Problem 4: Validity vs. Soundness

For each argument, determine: (a) Is it valid? (b) Is it sound? Explain.

**A.** "All birds can fly. Penguins are birds. Therefore, penguins can fly."

**B.** "If water is heated to 100°C at sea level, it boils. This water was heated to 100°C at sea level. Therefore, it boiled."

**C.** "All cats are mammals. All mammals are animals. Therefore, all cats are animals."

**D.** "If it's raining, the ground is wet. The ground is wet. Therefore, it's raining."

---

## Problem 5: Counterexample Method

Use the counterexample method to show that each argument form is invalid.

**A.** "If P then Q. Q. Therefore P." (Provide specific P and Q that make premises true and conclusion false.)

**B.** "If P then Q. Not P. Therefore not Q."

---

## Problem 6: Argument Diagramming

Diagram the following argument, showing the logical structure (linear chain, convergent, or linked):

"The firmware update changed the clock prescaler value. Changing the prescaler affects SPI bus timing. When SPI timing is affected, DMA transfers can misalign. Misaligned DMA transfers produce corrupted sensor data. Meanwhile, the same firmware update also disabled a watchdog timer. Without the watchdog, the system cannot auto-recover from data corruption. Therefore, the firmware update led to unrecoverable sensor data corruption."

---

## Solutions

<details>
<summary>Click to reveal solutions</summary>

**Problem 1:**

**A.** Yes, it's an argument.
- P1: The robot's battery is at 8%.
- P2: Low battery causes reduced motor torque.
- P3: Reduced torque causes wheel slip.
- C: The robot is likely experiencing wheel slip.

**B.** Not an argument. This is a report/description — a sequence of facts with no claim that one supports another.

**C.** Yes, it's an argument.
- P1: All firmware versions before v1.20 have the SPI timing bug. (Premise indicator: "Since")
- P2: This robot is running v1.17.
- C: This robot has the SPI timing bug.

**D.** Not an argument. This is a single conditional statement. No premise is offered to support a conclusion.

**E.** Yes, it's an argument.
- P1: Fossil fuels contribute to climate change. (Premise indicator: "because")
- P2: Renewable sources produce no direct emissions.
- C: Renewable energy is important.

**Problem 2:**

**A.** Conclusion indicator: "Consequently." Conclusion: "The H-bridge protection circuit activated."

**B.** Premise indicator: "given that." Conclusion: "The test suite should be expanded."

**C.** Premise indicator: "Owing to the fact that." Conclusion indicator: "This means." The final statement — "the estimator had insufficient data" — is the conclusion.

**Problem 3:**

**A.** Deductive. It's a categorical syllogism (All M are P, S is M, therefore S is P). If the premises are true, the conclusion must be true.

**B.** Inductive. It generalizes from a sample (15 robots) to a broader claim. The word "probably" signals induction.

**C.** Deductive. It's a disjunctive syllogism (A or B, not B, therefore A). If the disjunction is truly exhaustive, the conclusion follows with necessity.

**D.** Inductive. It's an argument from analogy — similar symptoms suggest a similar cause, but only probably.

**Problem 4:**

**A.** Valid (if all birds could fly, penguins would fly). Not sound (P1 is false — not all birds can fly).

**B.** Valid and sound. The form is modus ponens, and both premises are true.

**C.** Valid and sound. This is Barbara (AAA-1), and both premises are true.

**D.** Invalid. This is affirming the consequent. The ground could be wet from a sprinkler. Not sound (invalid arguments cannot be sound).

**Problem 5:**

**A.** Let P = "I am the President" and Q = "I am a human." Then: If I am the President, then I am a human (T). I am a human (T). Therefore I am the President (F). Premises true, conclusion false → invalid.

**B.** Let P = "I am in Paris" and Q = "I am in Europe." Then: If I am in Paris, then I am in Europe (T). I am not in Paris (T). Therefore I am not in Europe (F — I could be in London). Premises true, conclusion false → invalid.

**Problem 6:**

This is a **convergent** structure with two independent chains meeting at the conclusion:

```
Chain 1:                              Chain 2:
"Changed prescaler"                   "Disabled watchdog"
        │                                     │
        ▼                                     │
"SPI timing affected"                         │
        │                                     │
        ▼                                     │
"DMA misaligned"                              │
        │                                     │
        ▼                                     │
"Corrupted sensor data" ◄────────────────────┘
        │                                     │
        └──────────── LINKED ─────────────────┘
                        │
                        ▼
        "Unrecoverable sensor corruption"
```

The conclusion requires both chains together — corrupted data (chain 1) AND inability to recover (chain 2) → unrecoverable corruption.

</details>
