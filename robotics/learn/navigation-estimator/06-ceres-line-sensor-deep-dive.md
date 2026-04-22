# 06 — Ceres Bell-Curve Solver: Internals, Failure Modes & Diagnosis
### Reflectance estimator deep-dive — source-verified, incident-proven

**Prerequisite:** `03-measurement-models.md` (line constraints), `05-failure-modes.md` (failure taxonomy)
**Source files:** `estimator_reflectance_bell.cpp` (268 lines), `ceres_solver.cpp` (~200 lines)
**Unlocks:** Diagnosing cold-start REJECT, geometric parking failures, hardware-vs-algorithmic triage

---

## Why This File Exists

Ticket #104718 (amr-robot-scl-site-a105, 2026-04-22) revealed a failure mode that the existing
learning material didn't cover: a robot parked at a **tape intersection** caused all 4 line-sensors
to REJECT simultaneously on cold-start, with no hardware fault whatsoever.

The investigation required understanding the Ceres solver internals deeply enough to:
- Explain *why* 3 sensors crossing tape triggers REJECT even with Cauchy loss active
- Explain *why* the robot can never self-recover once it enters this state
- Distinguish this geometric failure from the stale-data cascade pattern already documented

This note captures all of that so future investigations can reach the same conclusions in minutes.

---

## Part 1 — The Bell Curve Model

### Formula

The line-sensor estimator fits a bell curve (actually a super-Gaussian trough) to the reflectance profile:

```
y(x) = p0 + p1 × exp(-|p2 × (x - p3)|^p4)
```

| Parameter | Meaning | Init value | Notes |
|-----------|---------|------------|-------|
| `p0` | Baseline (floor ADC) | ~960 | High because black floor reflects little IR |
| `p1` | Amplitude | Negative (~-878) | **Negative** — curve is a downward trough, not a peak |
| `p2` | Sharpness | 50.0 | Wider init → narrower initial expected tape width |
| `p3` | Tape centre (m) | from previous | The value we want to extract for pose update |
| `p4` | Shape exponent | 2.0 | p4=2 = Gaussian; p4>2 = super-Gaussian (flat-top, steeper sides) |

**Critical: the curve is a TROUGH (dip), not a peak.**
- Tape (white) → low IR reflection → transistor saturates → **LOW ADC values (~65–150)**
- Floor (black) → high IR reflection → transistor barely conducts → **HIGH ADC values (~900–960)**

This is the opposite of what you'd intuitively expect. The pull-down circuit in the line-sensor
hardware inverts the response: more light → lower voltage → lower ADC reading.

### Parameter bounds (from `ceres_solver.cpp`)

```
p1  ≤ 0.7 × initial_p1      (amplitude can't shrink more than 30%)
p2  ≤ 100.0                  (sharpness cap)
p4  ∈ [1.2, 8.0]            (shape bounded between sub-Gaussian and steep super-Gaussian)
```

### What "good fit" looks like

With a robot crossing a single tape line at moderate speed:
- p3 converges to the tape centre position in metres
- p4 gradually increases from 2.0 toward 3–5 as the solver finds a better shape
- p2 adjusts to match actual tape width
- Residuals for all 12 sensors are small; 0 sensors disabled

---

## Part 2 — The Two Code Paths (CRITICAL)

There are **two completely different sensor disablement mechanisms** in `estimator_reflectance_bell.cpp`.
Which path runs depends on the YAML config flag `reflectance_use_loss_function`.

### Path A: Loss Function Active (DEFAULT in production)

```yaml
reflectance_use_loss_function: true
loss_function_constant: 40.0
```

Disabled-sensor check uses a **hardcoded constant**:

```cpp
constexpr double DISABLED_SENSOR_ERROR_PERCENTAGE = 0.25;
// Source: estimator_reflectance_bell.cpp
```

A sensor is disabled if:
```
|raw_residual| > |p1| × DISABLED_SENSOR_ERROR_PERCENTAGE
```

Where `raw_residual` = `y_predicted(xi) - y_measured(xi)`, **without** applying the Cauchy loss
(the `apply_loss_function = false` flag is set during the disabled-check step).

With p1 ≈ -878 (amplitude):
```
threshold = 878 × 0.25 = 219.5 ADC units
```

**This is the active path on all production robots.** The `reflectance_disabled_sensor_threshold: 0.15`
YAML value is **not used** when `reflectance_use_loss_function: true`.

### Path B: Loss Function Inactive (NOT in production)

```yaml
reflectance_use_loss_function: false
```

Uses `_disabled_sensor_threshold = 0.15` from YAML. A sensor is disabled if its reflectance
value is below `max_reflectance × 0.15`. This was the original mechanism.

### Why this matters for investigation

If you read the YAML and see `reflectance_disabled_sensor_threshold: 0.15`, do NOT assume that
is the active threshold. **In loss-function mode, the threshold is hardcoded at 0.25.**

The effective threshold ratio is 0.25 (not 0.15). This is 67% stricter — sensors are disabled
only if their raw residual exceeds 25% of the fitted amplitude, not 15% of max reflectance.

---

## Part 3 — The Cauchy Loss Function and Its Trap

### What Cauchy loss does

During optimization, Ceres applies `CauchyLoss(c=40.0)`:

```
ρ(r²) = c² × log(1 + r²/c²)
```

The effect on gradient weight:

```
Cauchy weight = c² / (c² + r²) = 1 / (1 + (r/c)²)
```

At residual `r = 254` (a sensor 3 sensor-positions off the tape centre):
```
weight = 1 / (1 + (254/40)²) = 1 / (1 + 40.3) ≈ 0.024
```

That sensor contributes only **2.4% of its normal gradient influence**. The optimizer effectively
ignores sensors with large residuals.

### How it creates a trap

On the initial Gaussian guess (p2=50, p4=2.0) with 3 sensors crossing tape:

1. The 3 tape sensors have residuals of ~235–255 ADC (far from the bell prediction of ~60)
2. Cauchy weights these at 2.4–2.8% of normal
3. The optimizer converges to a solution that fits the **9 floor sensors** well
4. The 3 tape sensors remain at large residuals throughout
5. The disabled-sensor check fires: `|raw_residual| > 219.5` → all 3 disabled
6. `disabled_sensors_count = 3 > max_disabled_sensor_count = 2` → **REJECT**

The Cauchy loss intended to make the solver robust to outliers is working correctly — but the
"outliers" here are the tape sensors, which are physically the most important ones.

### Why it doesn't always trap

Verification via Python IRLS simulation showed that for a robot **moving** over a single tape line:

```
Step  0: p2=50.0  p4=2.00  → 3 disabled → REJECT  (initial Gaussian too narrow)
Step  1: p2=63.1  p4=2.62  → 0 disabled → VALID    ← ONE step escapes!
Step 12: p2=72.1  p4=6.72  → 0 disabled → converged to super-Gaussian
```

The solver CAN escape in step 1 because after the first Gauss-Newton update, p2 and p4 adjust
enough that the initial REJECT sensors now fit within the threshold.

**The key difference:** when moving, `_previous_solutions` has a valid cached solution. When
cold-starting on a tape intersection, there is no cached solution and the initial guess is
always the narrow Gaussian.

---

## Part 4 — The Self-Perpetuating Failure Loop

This is the core mechanism of the #104718 failure.

### `_previous_solutions` cache

In `estimator_reflectance_bell.cpp`:

```cpp
std::map<std::string, Solution> _previous_solutions;
// Maps bar frame_id → last valid solution
```

**When a solution is VALID:** the solution is stored in `_previous_solutions[frame_id]`.
Next call: initialization uses cached `p3` (tape position). The solver starts close to the
answer and converges quickly.

**When a solution is REJECT:** the solution is NOT stored. The map entry is NOT updated.
Next call: if the map has no entry, initialization falls back to the narrow Gaussian default.

### The loop

```
[Robot restarts on tape intersection]
   │
   ▼
Call solver → no cached solution → init with narrow Gaussian (p2=50, p4=2)
   │
   ▼
3 tape sensors get large residuals at step 0 → disabled → REJECT
   │
   ▼
REJECT → do NOT store solution → _previous_solutions still empty
   │
   └──────────────────────────────────────────────────────────────────► (loop forever)
```

**The robot can NEVER self-recover from this state.** Every call to the solver starts from the
same narrow Gaussian and produces the same REJECT. The only exit is:
- Robot moves (changes sensor geometry)
- Manual intervention / restart with robot on clean floor

### Why all 4 bars fail simultaneously

The tape intersection geometry at tile (28,0,0) places 3 sensors from EACH bar on white tape.
With `max_disabled_sensor_count = 2` (designed for one broken sensor), 3 sensors on tape always
exceeds the limit. All 4 bars → REJECT simultaneously. The system has no fall-back.

---

## Part 5 — Real Data from Ticket #104718

### Sensor readings (mean from recording_001.mcap, ~5 min at 05:21–05:22 UTC)

| Bar | S00 | S01 | S02 | S03 | S04 | S05 | S06 | S07 | S08 | S09 | S10 | S11 |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| front | 959 | 930 | 898 | 569 | **71** | **66** | **76** | 635 | 878 | 924 | 944 | 956 |
| rear  | 999 | 972 | 992 | 719 | **142** | **59** | **63** | 252 | 859 | 940 | 995 | 958 |
| left  | 948 | 931 | 913 | 834 | 560 | **69** | **65** | **75** | 580 | 889 | 936 | 956 |
| right | 940 | 836 | 499 | **67** | **65** | **71** | 512 | 846 | 890 | 883 | 918 | 931 |

**Bold = on white tape (ADC 59–142, well below 219.5 threshold)**

Each bar has exactly 3 sensors on tape → exceeds `max_disabled_sensor_count=2` → all 4 REJECT.

### Hardware verdict (from 30-min rosbag, 04:10–04:40 UTC)

- `is_reliable` dropout rate: **0.02%** (175,487 frames/bar) — nominal
- Stuck-high frames: **0%**, stuck-low frames: **0%** — no hardware fault
- Tape sensors (59–143 ADC): correct for white tape
- Floor sensors (836–999 ADC): correct for black floor
- Same 3 sensors/bar on tape for 100% of 30-min window → robot stationary since ≤03:46 UTC

**Conclusion: HARDWARE IS HEALTHY. Failure is entirely geometric + algorithmic.**

### Event timeline

| Time (UTC) | Event |
|---|---|
| ≤03:46 | Robot already parked at tile (28,0,0) on tape intersection |
| 05:21:29 | All nodes kill (roslaunch restart — someone triggered a restart) |
| 05:21:41 | **`EstimatorReflectanceBell: Curve fitting disabled sensors for line-sensor: 4 (max: 2)`** |
| 05:21:46 | Line-sensor calibration: 4/4 bars HEALTHY (not the issue) |
| 05:21:50 | Map loads, tile (28,0,0) already in state=3 (disabled) |
| 05:22:00 | `InitializeProcessing::recoverErrors: Recovering error 1 by MCLBC_Navigation` |
| **05:22:06.916** | `RequestManualAssistance` — MCLBC_Navigation code 1 ⛔ |

---

## Part 6 — Diagnostic Protocol

### When to suspect this failure mode

Use this checklist when you see `MCLBC_Navigation code 1` or `EstimatorReflectanceBell: Curve fitting disabled sensors for line-sensor: N (max: 2)`:

```
□ Does /rosout show "Curve fitting disabled sensors" for ALL 4 bars simultaneously?
  → If yes: suspect tape intersection geometry (not hardware)
  → If only 1-2 bars: more likely single line-sensor hardware degradation

□ Was there a recent restart/reboot before the failure?
  → Cold-start required to trigger this pattern (no cached solution)

□ Check the actual sensor values from /robot<N>/line-sensors at failure time
  → If 3+ sensors have values < 200 per bar → robot is on tape intersection
  → If random sensors have values < 200 → could be sensor hardware fault

□ Run line-sensor health check (is_reliable dropout, stuck sensors)
  → If dropout < 1% and no stuck sensors → hardware healthy → geometric failure

□ Check tile position from /robot<N>/localization or RFID
  → Tile intersections are at map grid cross-points
  → Compare tile position against warehouse grid map
```

### Distinguishing from other failure modes

| Signal | Tape Intersection | Stale Data Cascade | Hardware Fault |
|--------|-------------------|--------------------|----------------|
| All 4 bars REJECT | Yes | Not usually | Sometimes |
| Failure on cold-start | Yes | No (occurs during motion) | No pattern |
| Hardware checks pass | Yes | Yes | **No** |
| Robot was stationary | Yes | No (requires motion) | Either |
| Sensor values | 3 sensors ~65–150 ADC | Normal pattern | Stuck/noisy |
| `is_reliable` dropout | Normal | Normal | **Elevated (>5%)** |

### Fix for the hardware-healthy tape intersection case

**Immediate:** Manually reposition robot 20-30cm off the intersection. Clear the tile disable.

**Long-term options:**
1. Increase `reflectance_max_disabled_sensor_count` from 2 to 3 (allows 3 sensors on tape)
2. Add warm-start fallback: if REJECT and no previous solution, try a wider initial p2 (e.g., p2=20)
3. Add parking-position constraint to prevent robots from stopping at tape intersections
4. Increase `DISABLED_SENSOR_ERROR_PERCENTAGE` from 0.25 to allow more sensor variation

**Tradeoff note:** Option 1 and 4 reduce sensitivity to actual broken sensors. The threshold
was designed for detecting 1 broken sensor (noise floor, not geometry). Changing it affects
all robots at all sites.

---

## Part 7 — Configuration Reference

From `config_default.yaml`:

```yaml
sensor_spacing_x: 0.0069           # 6.9mm between adjacent sensors (12 sensors = 82.8mm span)
reflectance_use_loss_function: true  # ENABLES PATH A (loss function mode)
loss_function_constant: 40.0         # c parameter in CauchyLoss(c)
weight_reduction_error_threshold: 20.0  # Weight reduction starts above this residual (not DISABLED)
reflectance_disabled_sensor_threshold: 0.15   # NOT USED when loss_function=true
reflectance_max_disabled_sensor_count: 2      # REJECT if more than this many sensors disabled
```

Hardcoded in source (NOT configurable via YAML):
```cpp
constexpr double DISABLED_SENSOR_ERROR_PERCENTAGE = 0.25;  // estimator_reflectance_bell.cpp
```

---

## Summary

| Topic | Key fact |
|-------|----------|
| Bell curve | Trough shape; p1 is negative; LOW ADC = white tape |
| Active disable threshold | 25% of fitted amplitude (~219 ADC) — NOT the 0.15 YAML value |
| Cauchy C=40 | Down-weights sensors with residual >40 ADC; creates trap for tape sensors |
| Cold-start failure | No cached solution → narrow Gaussian → tape sensors at large residual → REJECT |
| Self-perpetuating loop | REJECT doesn't update cache → next call identical → REJECT forever |
| Tape intersection | 3 sensors/bar on tape at grid cross-points → always > max_disabled=2 |
| Hardware check | 0% stuck, 0.02% unreliable → failure is geometric, not hardware |
| Fix | Move robot off intersection OR increase max_disabled OR add warm-start fallback |
