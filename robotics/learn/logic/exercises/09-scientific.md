# Exercise 9 — Statistical & Scientific Reasoning

**Covers:** Lesson 09
**Difficulty:** Intermediate

---

## Problem 1: Sampling Bias

Identify the sampling bias in each scenario and suggest a better approach.

**A.** To assess fleet reliability, you analyze failure data only from robots that are currently operational.

**B.** To determine if a firmware update fixes a bug, you deploy it to the 10 newest robots and report results.

**C.** To measure customer satisfaction with robot performance, you survey warehouse managers during a period when the fleet had zero downtime.

**D.** To evaluate a new navigation algorithm, you test it exclusively in a well-lit, obstacle-free test corridor.

---

## Problem 2: Measures of Central Tendency

A fleet of 10 robots had the following number of unplanned stops in the past week:

{0, 0, 1, 1, 1, 2, 2, 3, 5, 45}

**A.** Calculate the mean, median, and mode.

**B.** Which measure best represents the "typical" robot's experience? Why?

**C.** A report states: "The average robot had 6 unplanned stops this week." Is this misleading? Explain.

**D.** What would you recommend reporting instead?

---

## Problem 3: Percentage Analysis

**A.** Last month: 3 failures out of 200 robots (1.5%). This month: 5 failures out of 200 robots (2.5%). A report says "failures increased by 67%." Is this accurate? Is it misleading?

**B.** A firmware update reduced critical bugs from 8 to 2. The team reports "75% reduction in critical bugs." Another team member says "we only fixed 6 bugs — that's nothing special for a 100,000-line codebase." Who is right?

**C.** A test passes 99.9% of the time. The fleet runs 10,000 tests per day. How many failures per day should you expect? Is "99.9% pass rate" good enough?

---

## Problem 4: Hypothesis Evaluation

Two engineers propose competing hypotheses for a robot failure pattern:

**H1 (Engineer A):** "The failures are caused by a firmware bug in the SPI driver that triggers when CPU temperature exceeds 70°C."
- Predicts: Failures correlate with CPU temperature > 70°C. Failures stop if you add a thermal throttle at 65°C.

**H2 (Engineer B):** "The failures are caused by bad luck and cosmic rays."
- Predicts: Failures are random and unpredictable.

**A.** Which hypothesis is more testable/falsifiable? Why?

**B.** Which is simpler? (Occam's Razor)

**C.** Design an experiment that could distinguish between the two hypotheses.

**D.** If you run the experiment and find that failures DO correlate with CPU temperature > 70°C, is H1 proven? Explain.

---

## Problem 5: Scientific Reasoning Errors

Identify the scientific reasoning error in each scenario.

**A.** An engineer hypothesizes that the motor driver is the problem. They examine only the motor driver logs, find a suspicious spike, and declare the root cause found — without examining other subsystems.

**B.** The team tested a fix on 50 robots. 48 passed, 2 failed. They modify the fix, test on 50 more robots, 47 pass, 3 fail. They modify again, test 50 more. Eventually, a version passes all 50 tests. They ship it.

**C.** "We've been using this vendor's components for 10 years. They've always been reliable. Therefore, we don't need incoming inspection for this new batch."

**D.** Original hypothesis: "The bug only affects firmware v1.18." Evidence: A v1.20 robot also exhibits the bug. Modified hypothesis: "The bug affects v1.18 and v1.20 but not v1.22+." Evidence: A v1.22 robot exhibits the bug. Modified hypothesis: "The bug affects v1.18, v1.20, and v1.22 on Rev C hardware only."

---

## Problem 6: Cognitive Bias Identification

Name the cognitive bias operating in each scenario.

**A.** After three SPI-related incidents in a row, a new incident comes in. Before looking at any evidence, the engineer says "it's probably SPI again."

**B.** The team lead proposed the initial hypothesis in the RCA meeting. Over the next 2 hours, the team found evidence both for and against the hypothesis, but everyone kept referring back to the team lead's original idea.

**C.** An engineer ran 20 tests. 18 passed, 2 failed. They report: "The fix works — 90% pass rate, and the 2 failures were probably environmental anomalies."

**D.** "We investigated for 3 days and found the root cause is X. We can't be wrong — we spent too much time on this to start over."

---

## Problem 7: Design a Complete Investigation

A fleet of 50 robots at a warehouse has been experiencing increased localization errors over the past 2 weeks. You're asked to investigate.

Outline a complete investigation plan that incorporates:
1. Hypothesis formation (at least 3 competing hypotheses)
2. Mill's methods (which would you use and how?)
3. Statistical considerations (sample size, measures, potential biases)
4. Falsifiable predictions for each hypothesis
5. How you would guard against cognitive biases

---

## Solutions

<details>
<summary>Click to reveal solutions</summary>

**Problem 1:**

**A.** Survivorship bias — by excluding robots that already failed and were removed, you undercount failures. Better: include all robots that were ever in the fleet, including decommissioned ones.

**B.** Selection bias — the 10 newest robots may have different hardware revisions, conditions, or run time. Better: randomly select robots across all hardware revisions, ages, and locations.

**C.** Selection bias (temporal) — surveying during an ideal period doesn't represent typical experience. Better: survey across a representative time period that includes typical downtime.

**D.** Selection bias (environmental) — the test corridor doesn't represent real conditions. Better: test in actual warehouse conditions with varying lighting, obstacles, traffic.

**Problem 2:**

**A.**
- Mean: (0+0+1+1+1+2+2+3+5+45)/10 = 60/10 = **6.0**
- Median: sorted = {0,0,1,1,1,2,2,3,5,45} → middle of 5th and 6th values = (1+2)/2 = **1.5**
- Mode: **1** (appears 3 times)

**B.** The **median (1.5)** or **mode (1)** best represents the typical robot. 9 out of 10 robots had 0–5 stops. Robot-10 with 45 stops is a clear outlier that dramatically skews the mean.

**C.** Yes, it's misleading. "Average 6 stops" suggests all robots are having significant problems, when actually 90% of robots had 3 or fewer stops. One outlier robot is responsible for the high mean.

**D.** Report: "Median unplanned stops: 1.5. Range: 0–45. Note: One robot (Robot-10) accounted for 45 of the 60 total stops and requires individual investigation. Excluding this outlier, the fleet mean is 1.67."

**Problem 3:**

**A.** The 67% relative increase is mathematically accurate (from 1.5% to 2.5% = 67% relative increase). But it's misleading because the absolute increase is just 2 additional failures out of 200. Both rates are very low. Better: "Failures increased from 3 to 5 (an additional 2 failures, absolute increase of 1 percentage point)."

**B.** Both are right — they're measuring different things. 75% reduction in critical bugs is meaningful for quality assessment. "Only 6 bugs in 100K lines" provides context about the scale. Report both: "Critical bugs reduced from 8 to 2 (75% reduction), representing a fix density of 0.006% of the codebase."

**C.** 10,000 × 0.001 = **10 failures per day**. Whether this is acceptable depends on context. If each failure causes a 5-minute robot stoppage, that's 50 minutes of downtime per day across the fleet, which might or might not be acceptable.

**Problem 4:**

**A.** H1 is far more testable. It makes specific, falsifiable predictions (temperature correlation, effect of thermal throttle). H2 is essentially unfalsifiable — any pattern of failures (or lack thereof) is compatible with "randomness."

**B.** H1 is simpler in the explanatory sense — it identifies a specific mechanism. H2 invokes an untestable, unfalsifiable external factor.

**C.** Experiment: Monitor CPU temperature on all robots continuously. Log temperatures at the time of each failure. If H1 is correct, failures should cluster above 70°C. Additionally, install thermal throttling at 65°C on half the fleet (randomly selected) and compare failure rates between the two groups (controlled experiment, method of difference).

**D.** No, H1 is not proven — it's supported. The temperature correlation is consistent with H1 but could also be explained by other factors that correlate with temperature (e.g., time of day, workload). The thermal throttle experiment provides stronger evidence because it's a controlled intervention. But even then, technically, there could be confounding factors. Evidence supports H1; it doesn't prove it with certainty.

**Problem 5:**

**A.** Confirmation bias — only examined evidence that could support the pre-existing hypothesis. Should have examined all subsystem logs and actively looked for evidence that contradicts the motor driver hypothesis.

**B.** Multiple testing problem — if you keep modifying and retesting, eventually you'll get a version that passes by chance (50 tests is not a large sample). The "final" version might not actually be better; it might just have been lucky. Better: define the fix, test it once with a pre-specified sample size and success criterion.

**C.** Appeal to tradition / insufficient induction — past reliability doesn't guarantee future reliability. The new batch may have different manufacturing conditions, components, or quality control. Incoming inspection is a necessary independent verification.

**D.** Ad hoc modification — each time the hypothesis is contradicted, it's patched with an additional exception. After three patches, the hypothesis is becoming unfalsifiable (it'll just keep adding exceptions). Time to reconsider: maybe the bug is broader than initially thought and a fundamentally different hypothesis is needed.

**Problem 6:**

**A.** Availability heuristic — overweighting recent, memorable events (three SPI incidents) when assessing probability.

**B.** Anchoring — the team lead's initial hypothesis serves as a mental anchor, and subsequent evidence is interpreted relative to it.

**C.** Confirmation bias — the engineer emphasizes the 18 passes and dismisses the 2 failures as anomalies rather than investigating them as potential genuine failures.

**D.** Sunk cost fallacy — the time invested doesn't make the conclusion more likely to be correct. If new evidence contradicts X, the right thing is to reconsider, regardless of time spent.

**Problem 7:**

**1. Hypotheses:**
- H1: Environmental change — the warehouse made modifications (new shelving, lighting changes, floor treatment) that affect LIDAR/camera performance.
- H2: Firmware degradation — a recent firmware update introduced a subtle bug that manifests over time.
- H3: Hardware aging — sensors (especially odometry encoders or LIDAR) are degrading with age/use.

**2. Mill's methods:**
- Method of Agreement: What do the affected robots have in common? (Same firmware? Same area? Same age?)
- Method of Difference: Compare affected vs. unaffected robots. What's different? (Different firmware version? Different area? Different hardware?)
- Concomitant Variation: Plot localization error rate over time vs. environmental changes, firmware updates, and robot age.

**3. Statistical considerations:**
- Sample size: Analyze all 50 robots, not just a subset.
- Measures: Report median and distribution of localization errors, not just mean.
- Biases: Avoid survivorship bias (include robots taken offline). Avoid selection bias (don't just analyze the worst robots). Time-window bias (ensure pre/post comparison uses same conditions).

**4. Falsifiable predictions:**
- H1: Errors concentrate in areas of the warehouse that were modified. Robots in unmodified areas should be unaffected. Moving a robot to an unmodified area should reduce errors.
- H2: Only robots with the recent firmware version are affected. Reverting firmware should fix the issue. Robots never updated should be fine.
- H3: Errors correlate with robot age/usage hours. Replacing the suspected sensor should fix the error. Newer robots should be less affected.

**5. Guarding against bias:**
- Form all three hypotheses before looking at data.
- Assign different team members to investigate each hypothesis.
- Pre-register the test criteria (what constitutes confirmation/disconfirmation).
- Actively seek disconfirming evidence for the leading hypothesis.
- Document negative results alongside positive ones.
- Have someone play "devil's advocate" for each hypothesis.

</details>
