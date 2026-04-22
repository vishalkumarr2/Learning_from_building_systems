# Exercise 4 — Categorical Logic

**Covers:** Lesson 04
**Difficulty:** Intermediate

---

## Problem 1: Standard Form

Convert each statement to standard categorical form (A, E, I, or O). Identify the subject term (S) and predicate term (P).

**A.** "Every robot in the fleet needs a firmware update."

**B.** "Some of the sensorbar readings are inaccurate."

**C.** "No device with hardware revision A passed the thermal test."

**D.** "There are OKS robots that don't have the latest calibration."

**E.** "Only robots with firmware ≥ v1.24 are approved for deployment."

**F.** "Wherever there's an SPI bus, there's a potential timing issue."

---

## Problem 2: Distribution

For each proposition, state which terms are distributed.

**A.** All SPI devices are susceptible to clock jitter. (A)

**B.** No defective motors are approved for use. (E)

**C.** Some test cases are failing. (I)

**D.** Some firmware versions are not compatible with Rev C hardware. (O)

---

## Problem 3: Square of Opposition

Given that each statement is TRUE, what can you conclude about the truth values of the related propositions?

**A.** "All robots in fleet X have the SPI bug." (A) — TRUE.
- What about E? I? O?

**B.** "Some sensors are miscalibrated." (I) — TRUE.
- What about A? E? O?

**C.** "No firmware build passed all integration tests." (E) — TRUE.
- What about A? I? O?

---

## Problem 4: Conversion, Obversion, Contraposition

Apply the specified operation and state whether the result is logically equivalent.

**A.** Convert: "No defective parts are usable parts."

**B.** Obvert: "All tested robots are operational robots."

**C.** Contrapose: "All devices with errors are devices needing repair."

**D.** Convert: "All programmers are problem solvers." — Is this valid?

---

## Problem 5: Syllogism Validity

Test each syllogism for validity using the five rules. Identify any fallacies.

**A.**
```
All devices with the SPI bug (M) are devices needing updates (P).
All robots in building 3 (S) are devices with the SPI bug (M).
∴ All robots in building 3 (S) are devices needing updates (P).
```

**B.**
```
All robots that crashed (P) are robots running Linux (M).
All OKS robots (S) are robots running Linux (M).
∴ All OKS robots (S) are robots that crashed (P).
```

**C.**
```
No defective sensors (M) are approved components (P).
Some warehouse devices (S) are defective sensors (M).
∴ Some warehouse devices (S) are not approved components (P).
```

**D.**
```
No firmware version (M) is bug-free (P).
No untested code (S) is a firmware version (M).
∴ No untested code (S) is bug-free (P).
```

**E.**
```
All motors that overheated (M) are motors that were replaced (P).
Some motors in the fleet (S) are not motors that overheated (M).
∴ Some motors in the fleet (S) are not motors that were replaced (P).
```

---

## Problem 6: Translate and Evaluate

Translate the following argument into standard form categorical syllogism, then test validity.

"Every robot with stale sensorbar data eventually delocalizes. Some robots in the Hamamatsu warehouse have stale sensorbar data. Therefore, some robots in the Hamamatsu warehouse eventually delocalize."

---

## Solutions

<details>
<summary>Click to reveal solutions</summary>

**Problem 1:**

**A.** A: All robots in the fleet (S) are things needing a firmware update (P).

**B.** I: Some sensorbar readings (S) are inaccurate readings (P).

**C.** E: No devices with hardware revision A (S) are things that passed the thermal test (P).

**D.** O: Some OKS robots (S) are not things with the latest calibration (P).

**E.** A: All things approved for deployment (S) are robots with firmware ≥ v1.24 (P). Note: "Only S are P" translates to "All P are S."

**F.** A: All things with an SPI bus (S) are things with a potential timing issue (P).

**Problem 2:**

**A.** A: S (SPI devices) is distributed. P (susceptible to clock jitter) is NOT distributed.

**B.** E: Both S (defective motors) and P (approved for use) are distributed.

**C.** I: Neither S (test cases) nor P (failing) is distributed.

**D.** O: S (firmware versions) is NOT distributed. P (compatible with Rev C hardware) IS distributed.

**Problem 3:**

**A.** A is TRUE → E is FALSE (contraries can't both be true), I is TRUE (A implies I in traditional logic — but in Boolean logic, only if the subject class is nonempty), O is FALSE (contradictory of A).

**B.** I is TRUE → E is FALSE (contradictory of I). A is UNDETERMINED (some ≠ all). O is UNDETERMINED.

**C.** E is TRUE → A is FALSE (contraries). I is FALSE (contradictory of E). O is TRUE (E implies O when class is nonempty).

**Problem 4:**

**A.** "No usable parts are defective parts." — Valid (E-conversion is always valid). Equivalent.

**B.** "No tested robots are non-operational robots." — Valid (obversion always valid for all forms). Equivalent.

**C.** "All devices not needing repair are devices without errors." — Valid (A-contraposition is valid). Equivalent.

**D.** "All problem solvers are programmers." — INVALID! A-conversion is NOT valid. Not all problem solvers are programmers.

**Problem 5:**

**A.** AAA-1 (Barbara). Check rules:
- Middle term (M) distributed? Yes — subject of first A and predicate of second... wait. M is subject of P1 (distributed in A) and predicate of P2 (NOT distributed in A). But M IS distributed at least once (in P1). ✓
- No terms distributed in conclusion but undistributed in premises: S is distributed in conclusion (A proposition), S is distributed in P2 (subject of A). ✓ P is distributed in conclusion? No (predicate of A). ✓
- Not two negative premises ✓. Negative premise/conclusion match ✓. Not universal premises with particular conclusion ✓.
- **VALID** ✓

**B.** AAA-2. Check rules:
- Middle term (M = robots running Linux): In P1, M is predicate of A → NOT distributed. In P2, M is predicate of A → NOT distributed. **M is never distributed** → **INVALID: Undistributed Middle**.

**C.** EIO-1. Check rules:
- M (defective sensors): Subject of E (distributed). ✓
- P (approved components): Predicate of E (distributed in E). In conclusion O, P is distributed. Distributed in premise? Yes (in E). ✓
- S (warehouse devices): Not distributed in conclusion (O). ✓
- One negative premise, negative conclusion ✓.
- **VALID** ✓ (This is Ferio.)

**D.** EEE-1 (or similar). Check rules:
- **Two negative premises** → **INVALID: Exclusive Premises**.

**E.** AOO-2? Let's check.
- P1: A (All M are P). P2: O (Some S are not M). Conclusion: O (Some S are not P).
- M (motors that overheated): In P1, M is subject of A (distributed). In P2, M is predicate of O (distributed). ✓
- P (motors that were replaced): In conclusion O, P is distributed. In P1, P is predicate of A (NOT distributed). → **INVALID: Illicit Major**.

**Problem 6:**

Standard form:
```
All robots with stale sensorbar data (M) are things that eventually delocalize (P).  [A]
Some robots in Hamamatsu (S) are robots with stale sensorbar data (M).               [I]
∴ Some robots in Hamamatsu (S) are things that eventually delocalize (P).             [I]
```

This is AII-1 (Darii). Check rules:
- M distributed? Subject of A → Yes ✓
- No illicit distribution in conclusion ✓ (neither S nor P distributed in I)
- No two negatives ✓
- No negative/affirmative mismatch ✓
- Not universal premises with particular conclusion (one is particular) ✓

**VALID** ✓

</details>
