# Exercise 2 — Language, Meaning & Definition

**Covers:** Lesson 02
**Difficulty:** Foundation

---

## Problem 1: Cognitive vs. Emotive

Rewrite each statement to maximize cognitive meaning and minimize emotive meaning.

**A.** "The firmware is an absolute trainwreck."

**B.** "Our amazing new algorithm crushes the competition."

**C.** "Management's reckless decision to ship early will destroy product quality."

**D.** "The lazy QA team didn't bother testing edge cases."

---

## Problem 2: Ambiguity

Identify the type of ambiguity (semantic, syntactic, or cross-reference) and provide two different interpretations.

**A.** "The robot hit the shelf at high speed."

**B.** "We need to replace defective boards with new screws."

**C.** "The estimator gets data from the sensorbar and the IMU. When it fails, the robot stops."

**D.** "The engineer tested the motor running the diagnostic script."

---

## Problem 3: Vagueness

Each statement is vague. Rewrite it with a precise, measurable version.

**A.** "The robot is too slow."

**B.** "The battery drains quickly."

**C.** "The firmware update has been tested extensively."

**D.** "There's a significant delay in the control loop."

**E.** "The robot crashes frequently."

---

## Problem 4: Verbal vs. Genuine Dispute

Determine whether each disagreement is verbal (about definitions) or genuine (about facts).

**A.** Engineer A: "The robot delocalized three times today." Engineer B: "No, those were just covariance spikes — the position error never exceeded 50cm."

**B.** Engineer A: "The firmware fix works." Engineer B: "No, the same failure occurred yesterday on Robot-53."

**C.** Engineer A: "This is a critical bug." Engineer B: "It's not critical — it only affects 2% of robots."

---

## Problem 5: Types of Definitions

Classify each definition as stipulative, lexical, précising, theoretical, or persuasive.

**A.** "For this investigation, 'STALL' means the sensorbar sequence counter did not increment for 5 or more consecutive cycles."

**B.** "A motor is a device that converts electrical energy into mechanical motion."

**C.** "A 'high priority' bug is one that affects more than 10% of the fleet or causes a safety stop."

**D.** "'Real testing' means testing on actual hardware under production conditions, not in simulation."

**E.** "In information theory, 'entropy' is a measure of the average amount of information produced by a stochastic source of data."

---

## Problem 6: Genus and Difference

Write genus-and-difference definitions for:

**A.** Sensorbar

**B.** Firmware

**C.** Deadlock (in software)

**D.** Root cause analysis

---

## Problem 7: Definition Evaluation

Each definition violates one or more rules of good definitions. Identify the violation.

**A.** "A bug is a problem with software." (too broad? too narrow? circular? negative? figurative? emotive?)

**B.** "A Kalman filter is a filter that uses the Kalman algorithm."

**C.** "A robot is not a toaster."

**D.** "Programming is the art of telling a computer your dreams."

**E.** "A sensor is a device used in robotics to measure physical quantities."

---

## Solutions

<details>
<summary>Click to reveal solutions</summary>

**Problem 1:**

**A.** "Firmware v2.3 has 12 unresolved bugs, including 3 that cause system crashes under normal operating conditions."

**B.** "Our new algorithm achieves 40ms average path planning time, compared to 120ms for the previous algorithm."

**C.** "Shipping 2 weeks ahead of the original schedule means skipping integration testing, which historically catches 30% of production bugs."

**D.** "The QA test plan did not include edge cases for sensor timeout and concurrent DMA access."

**Problem 2:**

**A.** Semantic ambiguity: "hit" could mean physically collided with OR reached/arrived at. Also "high speed" is relative.

**B.** Syntactic ambiguity: Replace defective boards using new screws? OR replace defective boards with ones that have new screws? OR replace defective boards and also replace screws?

**C.** Cross-reference ambiguity: "it" could refer to the estimator, the sensorbar, or the IMU.

**D.** Syntactic ambiguity: The engineer tested the motor while running the diagnostic script? OR the engineer tested the motor that was running the diagnostic script?

**Problem 3:**

**A.** "The robot's average traverse speed is 0.15 m/s, which is below the required 0.3 m/s threshold."

**B.** "Battery depletes from 100% to 20% in 4 hours of continuous operation, versus the 8-hour design target."

**C.** "The firmware update was tested on 25 robots across 3 hardware revisions over 72 hours of continuous operation, with 0 failures observed."

**D.** "The control loop latency is 8ms, exceeding the 5ms maximum specified in the requirements."

**E.** "The robot has experienced 7 unplanned stops in the past 24 hours of operation."

**Problem 4:**

**A.** Verbal dispute — they disagree on the definition of "delocalized." If they agree on a threshold (e.g., position error > 50cm for > 2 seconds), the factual disagreement may disappear.

**B.** Genuine dispute — they disagree about the factual claim that the fix prevents the failure. This is resolved by evidence.

**C.** Verbal dispute — they disagree on the definition of "critical." Once they agree on criteria (e.g., affects >10% of fleet, or causes safety event), the classification follows.

**Problem 5:**

**A.** Stipulative — assigns a specific technical meaning for a particular investigation.

**B.** Lexical — reports the standard dictionary meaning.

**C.** Précising — takes the vague term "high priority" and gives it precise criteria.

**D.** Persuasive — the word "real" smuggles in the value judgment that simulation-based testing isn't legitimate.

**E.** Theoretical — defines entropy within the framework of information theory.

**Problem 6:**

**A.** A sensorbar (genus: sensor module) is a linear array of optical encoders that measures incremental wheel displacement by detecting alternating stripes on a code strip, providing position feedback to the navigation system.

**B.** Firmware (genus: software) is low-level code stored in non-volatile memory that directly controls a device's hardware functions, typically running on a microcontroller without a general-purpose operating system.

**C.** A deadlock (genus: concurrency error) is a condition where two or more processes each hold a resource while waiting for a resource held by another, preventing any of them from proceeding.

**D.** Root cause analysis (genus: investigation methodology) is a systematic process for identifying the fundamental underlying cause of an incident or defect, as opposed to its symptoms or proximate triggers.

**Problem 7:**

**A.** Too broad — a performance bottleneck, a design flaw, and a missing feature could all be called "a problem with software," but not all are bugs.

**B.** Circular — defines "Kalman filter" using "Kalman algorithm." You need to know what the algorithm is to understand the filter.

**C.** Negative when positive is possible — tells you what a robot ISN'T, not what it IS.

**D.** Figurative/metaphorical — "art" and "dreams" are poetic, not informative.

**E.** Too narrow — sensors are used in many fields (medicine, automotive, manufacturing), not just robotics.

</details>
