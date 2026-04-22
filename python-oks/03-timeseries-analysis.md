# 03 — Time-Series Analysis: pandas, polars, matplotlib, scipy
### From "I can see the data" to "I found the anomaly in 10 seconds"
**Prerequisite:** Python lists, basic numpy (arrays, indexing), familiarity with CSV files
**Unlocks:** Diagnosing covariance blow-ups, finding exact anomaly onset timestamps, producing publication-ready diagnostic plots

---

## Why Should I Care? (Context)

When a robot log is exported to CSV, you get time-series data: covariance values over time, velocity profiles, status code transitions, controller state changes. The raw data is often thousands of rows per second.

The difference between "I can see the data" and "I can identify the anomaly onset to within 100ms" is knowing which functions to reach for:
- **pandas**: convenient, familiar, slow on large files (>500k rows)
- **polars**: 5–20x faster, lazy evaluation, excellent for log scans
- **matplotlib**: the diagnostic plot that shows three signals on a shared time axis
- **scipy**: peak detection, numerical derivatives — answers "when did this start growing?"

This chapter builds up to a complete, reusable analysis script template.

---

# PART 1 — PANDAS REFRESHER (THE BASELINE)

---

## 1.1 Reading and Inspecting Data

```python
import pandas as pd
from pathlib import Path


# Reading
df = pd.read_csv("bag_export.csv")

# Inspect
print(df.shape)          # (rows, columns)
print(df.dtypes)         # column types
print(df.head(5))        # first 5 rows
print(df.tail(5))        # last 5 rows
print(df.describe())     # count, mean, std, min, 25%, 50%, 75%, max

# Check for missing values
print(df.isnull().sum())
```

---

## 1.2 Selecting Data

```python
# Single column → Series
ts = df["timestamp"]

# Multiple columns → DataFrame
sub = df[["timestamp", "cov_xx", "velocity"]]

# Row by position
first_row = df.iloc[0]         # first row (Series)
first_10 = df.iloc[:10]        # first 10 rows

# Row by index label
row = df.loc[42]               # row with index label 42
```

---

## 1.3 Filtering Rows

```python
# Simple comparison
errors = df[df["status"] != 0]

# Multiple conditions — use & and | with parentheses
anomalies = df[(df["cov_xx"] > 0.01) & (df["timestamp"] > 1714300000.0)]

# String matching
nav_events = df[df["event_type"].str.contains("NAV_")]

# Filter by list of values
selected = df[df["status"].isin([1, 2, 5])]

# Filter out NaN
clean = df[df["cov_xx"].notna()]
```

---

## 1.4 Timestamps

```python
# Convert epoch float to pandas Timestamp
df["time"] = pd.to_datetime(df["timestamp"], unit="s")

# Convert string timestamp to datetime
df["time"] = pd.to_datetime(df["time_str"])

# Set timestamp as index for time-based operations
df = df.set_index("time")

# Slice by time range
window = df["2024-04-28 10:00":"2024-04-28 10:05"]

# Relative to an event
incident_time = pd.Timestamp("2024-04-28 10:02:30")
before = df[df.index < incident_time]
after = df[df.index >= incident_time]
```

---

## 1.5 Rolling Windows and Basic Stats

```python
# Rolling mean (smoothing)
df["cov_smooth"] = df["cov_xx"].rolling(window=10).mean()

# Rolling standard deviation
df["cov_std"] = df["cov_xx"].rolling(window=20).std()

# Per-group stats
by_status = df.groupby("status")["cov_xx"].agg(["mean", "std", "max"])

# Overall stats
print(f"Mean cov_xx: {df['cov_xx'].mean():.6f}")
print(f"Max cov_xx:  {df['cov_xx'].max():.6f}")
print(f"Rows > threshold: {(df['cov_xx'] > 0.01).sum()}")
```

---

## 1.6 Why Pandas Can Be Slow

```python
# BAD: row-by-row Python loop (extremely slow for large datasets)
results = []
for _, row in df.iterrows():
    if row["cov_xx"] > 0.01:
        results.append(row["timestamp"])

# GOOD: vectorised operations (fast)
results = df.loc[df["cov_xx"] > 0.01, "timestamp"].tolist()

# BAD: apply() with Python lambda (slow)
df["label"] = df["status"].apply(lambda x: "error" if x > 0 else "ok")

# GOOD: vectorised comparison (fast)
df["label"] = df["status"].map({0: "ok"}).fillna("error")
# or
df["label"] = df["status"].gt(0).map({True: "error", False: "ok"})
```

**Key insight:** pandas is designed for vectorised operations. Any `iterrows()` loop is a red flag — there's almost always a vectorised equivalent that's 100–1000x faster.

---

# PART 2 — POLARS: WHY AND WHEN

---

## 2.1 The Core Difference from pandas

| Feature | pandas | polars |
|---------|--------|--------|
| Memory layout | Row-oriented (or mixed) | Columnar (Apache Arrow) |
| Index | Yes (can be confusing) | No index |
| Evaluation | Eager by default | Lazy by default (`scan_csv`) |
| Performance (scan) | Baseline | 5–20x faster |
| API style | Permissive, many ways to do one thing | Explicit, one correct way |
| Python memory | More copies | Fewer copies (zero-copy slices) |

**Columnar storage** means all values of `cov_xx` are stored contiguously in memory. CPU SIMD instructions can process 8–16 values per clock cycle. pandas's row-oriented storage breaks SIMD.

---

## 2.2 When to Use Which

| Situation | Use |
|-----------|-----|
| File >500k rows | polars |
| Scanning multiple large CSVs | polars (lazy, `scan_csv`) |
| Existing codebase uses pandas | pandas |
| Need time-zone-aware resampling | pandas |
| New analysis script from scratch | polars |
| Small dataset (<50k rows) | Either; pandas is fine |

---

## 2.3 Lazy vs Eager

```python
import polars as pl

# Eager: reads the entire file immediately
df = pl.read_csv("bag_export.csv")

# Lazy: builds a query plan, doesn't read the file yet
lf = pl.scan_csv("bag_export.csv")

# Lazy chain: describe the query
result = (
    lf
    .filter(pl.col("timestamp") > 1714300000.0)
    .filter(pl.col("cov_xx") > 0.005)
    .select(["timestamp", "cov_xx", "velocity"])
    .sort("timestamp")
    .collect()   # ← executes the entire plan now
)
# polars reads only the needed columns from disk, applies filters during read
```

**Key insight:** `scan_csv` + `collect()` is faster than `read_csv` for filtered queries because polars can skip reading columns you don't need and apply filters row-by-row during reading.

---

# PART 3 — POLARS: CORE API

---

## 3.1 Selecting, Filtering, Sorting

```python
import polars as pl


df = pl.read_csv("bag_export.csv")

# Select specific columns
sub = df.select([pl.col("timestamp"), pl.col("cov_xx"), pl.col("status")])

# Alternative: select by name list
sub = df.select(["timestamp", "cov_xx", "status"])

# Filter rows
errors = df.filter(pl.col("status") != 0)

# Chained filters
anomalies = df.filter(
    (pl.col("cov_xx") > 0.01) & (pl.col("timestamp") > 1714300000.0)
)

# Sort
sorted_df = df.sort("timestamp")
sorted_desc = df.sort("cov_xx", descending=True)
```

---

## 3.2 Adding Computed Columns

```python
# with_columns: non-mutating — returns a new DataFrame
df = df.with_columns(
    pl.col("cov_xx").abs().alias("abs_cov_xx"),
    (pl.col("cov_xx") > 0.01).alias("is_anomaly"),
    (pl.col("timestamp") - pl.col("timestamp").first()).alias("elapsed_s"),
)

# Multiple computations at once
df = df.with_columns([
    pl.col("cov_xx").rolling_mean(window_size=10).alias("cov_smooth"),
    pl.col("cov_xx").diff().alias("dcov_dt"),
    pl.col("velocity").abs().alias("speed"),
])
```

---

## 3.3 Group By and Aggregation

```python
# Aggregate by status code
by_status = (
    df
    .group_by("status")
    .agg([
        pl.col("cov_xx").mean().alias("mean_cov"),
        pl.col("cov_xx").max().alias("max_cov"),
        pl.len().alias("count"),
    ])
    .sort("status")
)

print(by_status)
# shape: (4, 4)
# ┌────────┬──────────┬──────────┬───────┐
# │ status ┆ mean_cov ┆ max_cov  ┆ count │
# ╞════════╪══════════╪══════════╪═══════╡
# │ 0      ┆ 0.001234 ┆ 0.008000 ┆ 4821  │
# │ 1      ┆ 0.012847 ┆ 0.034000 ┆  143  │
# └────────┴──────────┴──────────┴───────┘
```

---

## 3.4 Worked Example: Find Anomalous Rows

```python
import polars as pl
from pathlib import Path


DELOCALIZED_THRESHOLD = 0.010   # cov_xx above this = delocalized
INCIDENT_WINDOW_S = 30.0        # look at 30s around the incident


def find_covariance_anomalies(
    csv_path: Path,
    start_ts: float,
    end_ts: float,
    threshold: float = DELOCALIZED_THRESHOLD,
) -> pl.DataFrame:
    """Return rows where cov_xx exceeds threshold in [start_ts, end_ts]."""
    df = (
        pl.scan_csv(str(csv_path))
        .filter(pl.col("timestamp").is_between(start_ts, end_ts))
        .filter(pl.col("cov_xx") > threshold)
        .sort("timestamp")
        .collect()
    )
    return df


anomalies = find_covariance_anomalies(
    Path("bag_export.csv"),
    start_ts=1714300000.0,
    end_ts=1714300300.0,
)

if anomalies.is_empty():
    print("No anomalies found in window")
else:
    first = anomalies.row(0, named=True)
    print(f"First anomaly at t={first['timestamp']:.3f}s, cov_xx={first['cov_xx']:.6f}")
    print(f"Total anomalous rows: {len(anomalies)}")
```

---

# PART 4 — TIME WINDOWS WITH POLARS

---

## 4.1 Converting Timestamps

```python
import polars as pl


# Epoch float (seconds) → Datetime
df = df.with_columns(
    pl.from_epoch(pl.col("timestamp"), time_unit="s").alias("datetime")
)

# Or cast directly if the column is already microseconds/milliseconds
df = df.with_columns(
    pl.col("timestamp_us").cast(pl.Datetime("us"))
)

# String → Datetime
df = df.with_columns(
    pl.col("time_str").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f")
)
```

---

## 4.2 Time Binning with group_by_dynamic

```python
# Resample into 500ms bins — mean cov_xx per bin
binned = (
    df
    .sort("datetime")
    .group_by_dynamic("datetime", every="500ms")
    .agg([
        pl.col("cov_xx").mean().alias("cov_mean"),
        pl.col("velocity").mean().alias("vel_mean"),
        pl.col("status").max().alias("status_max"),
    ])
)

# 1-second bins
binned_1s = (
    df
    .sort("datetime")
    .group_by_dynamic("datetime", every="1s")
    .agg(pl.col("cov_xx").max())
)
```

---

## 4.3 Rolling Operations and Derivatives

```python
# Rolling mean (smoothing noise)
df = df.with_columns(
    pl.col("cov_xx").rolling_mean(window_size=20).alias("cov_smooth")
)

# First-order difference (rate of change per row)
df = df.with_columns(
    pl.col("cov_xx").diff().alias("dcov_per_row")
)

# Access the previous row's value
df = df.with_columns(
    pl.col("cov_xx").shift(1).alias("prev_cov_xx")
)

# Find rows where cov_xx is growing fast
fast_growth = df.filter(pl.col("dcov_per_row") > 0.001)
```

---

## 4.4 Finding the First Threshold Crossing

```python
DELOCALIZED_THRESHOLD = 0.010


def find_first_breach(df: pl.DataFrame, threshold: float) -> float | None:
    """Return timestamp of first cov_xx breach, or None if no breach."""
    first = (
        df
        .filter(pl.col("cov_xx") > threshold)
        .sort("timestamp")
        .head(1)
    )
    if first.is_empty():
        return None
    return first["timestamp"][0]


breach_ts = find_first_breach(df, DELOCALIZED_THRESHOLD)
if breach_ts:
    print(f"Localization breach detected at t = {breach_ts:.3f}s")
    # Find the last clean row before the breach
    last_clean_ts = (
        df
        .filter(pl.col("timestamp") < breach_ts)
        .filter(pl.col("cov_xx") <= DELOCALIZED_THRESHOLD)
        .sort("timestamp")
        .tail(1)["timestamp"][0]
    )
    latency = breach_ts - last_clean_ts
    print(f"Localization was good until t = {last_clean_ts:.3f}s")
    print(f"Onset-to-breach latency: {latency:.3f}s")
```

---

# PART 5 — MATPLOTLIB: PRACTICAL PLOTTING

---

## 5.1 The Canonical Setup

```python
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


# Always use this pattern — it gives you control over each axis
fig, ax = plt.subplots(figsize=(14, 6))

# Or multiple vertically stacked axes sharing the x-axis
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
```

---

## 5.2 Line Plots and Threshold Lines

```python
# Line plot
ax.plot(t, cov_xx, label="cov_xx", color="steelblue", linewidth=1.0)

# Smoothed overlay
ax.plot(t, cov_smooth, label="cov_xx (smoothed)", color="blue",
        linewidth=2.0, linestyle="--")

# Horizontal threshold line
ax.axhline(y=DELOCALIZED_THRESHOLD, color="red", linestyle="--",
           linewidth=1.5, label=f"threshold ({DELOCALIZED_THRESHOLD})")

# Vertical event marker
ax.axvline(x=incident_time, color="orange", linewidth=2.0,
           linestyle="-", label="incident time")

ax.set_ylabel("Covariance XX (m²)")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
```

---

## 5.3 Shading Anomalous Regions

```python
import numpy as np

t = np.array(df["timestamp"].to_numpy())
cov_xx = np.array(df["cov_xx"].to_numpy())
status = np.array(df["status"].to_numpy())

# Shade where status is non-zero (anomalous)
ax.fill_between(
    t,
    0,
    cov_xx.max() * 1.1,
    where=status > 0,
    alpha=0.2,
    color="red",
    label="status > 0",
)

# Shade where cov exceeds threshold
ax.fill_between(
    t,
    DELOCALIZED_THRESHOLD,
    cov_xx,
    where=cov_xx > DELOCALIZED_THRESHOLD,
    alpha=0.3,
    color="orange",
    label="cov above threshold",
)
```

---

## 5.4 Twin Y-Axes

When two signals have very different scales but you want them on the same time axis:

```python
fig, ax1 = plt.subplots(figsize=(14, 6))

# Left axis: covariance
color_cov = "steelblue"
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Covariance XX (m²)", color=color_cov)
ax1.plot(t, cov_xx, color=color_cov, label="cov_xx")
ax1.tick_params(axis="y", labelcolor=color_cov)
ax1.axhline(y=DELOCALIZED_THRESHOLD, color="red", linestyle="--", alpha=0.7)

# Right axis: velocity
ax2 = ax1.twinx()
color_vel = "forestgreen"
ax2.set_ylabel("Velocity (m/s)", color=color_vel)
ax2.plot(t, velocity, color=color_vel, label="velocity", alpha=0.7)
ax2.tick_params(axis="y", labelcolor=color_vel)

# Combine legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
```

---

## 5.5 Worked Example: Canonical Diagnostic Plot

```python
def plot_covariance_diagnostic(
    t: np.ndarray,
    cov_xx: np.ndarray,
    velocity: np.ndarray,
    status: np.ndarray,
    incident_ts: float | None = None,
    threshold: float = 0.010,
    output_path: str | None = None,
) -> None:
    """
    Two-panel diagnostic plot: covariance (top) and velocity (bottom).

    Top panel: cov_xx with threshold line, anomaly shading.
    Bottom panel: speed with anomalous-status shading.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Covariance and Velocity Diagnostic", fontsize=14)

    # — Top: covariance —
    ax1.plot(t, cov_xx, label="cov_xx", color="steelblue", linewidth=1.0)
    ax1.axhline(y=threshold, color="red", linestyle="--",
                linewidth=1.5, label=f"threshold={threshold:.3f}")
    ax1.fill_between(t, threshold, cov_xx,
                     where=cov_xx > threshold, alpha=0.3, color="orange")
    ax1.set_ylabel("Covariance XX (m²)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # — Bottom: velocity —
    speed = np.abs(velocity)
    max_speed = speed.max() if speed.max() > 0 else 1.0
    ax2.plot(t, speed, label="|velocity|", color="forestgreen", linewidth=1.0)
    ax2.fill_between(t, 0, max_speed,
                     where=status > 0, alpha=0.2, color="red", label="status>0")
    ax2.set_ylabel("Speed (m/s)")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    # — Event marker —
    if incident_ts is not None:
        for ax in (ax1, ax2):
            ax.axvline(x=incident_ts, color="orange", linewidth=2.0,
                       linestyle="-", label="incident" if ax is ax1 else "")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close(fig)
```

---

## 5.6 Saving Figures

```python
# High-res PNG for reports
plt.savefig("analysis.png", dpi=150, bbox_inches="tight")

# PDF for vector graphics
plt.savefig("analysis.pdf", bbox_inches="tight")

# Always close after saving (prevents memory leak in loops)
plt.close(fig)

# Suppress display when running in a script (not interactive)
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — put this before plt import
```

---

# PART 6 — SCIPY: PEAK DETECTION AND DERIVATIVES

---

## 6.1 Finding Peaks with find_peaks

```python
from scipy.signal import find_peaks
import numpy as np


# Find covariance spikes
# height: minimum peak value
# distance: minimum samples between peaks (prevents finding duplicate peaks)
# prominence: how much the peak stands above its surroundings
peaks, properties = find_peaks(
    cov_xx,
    height=0.005,       # only peaks above 5mm²
    distance=50,        # at least 50 samples apart
    prominence=0.003,   # must stick out at least 3mm²
)

print(f"Found {len(peaks)} covariance spikes")
for idx in peaks:
    print(f"  t={t[idx]:.3f}s, cov_xx={cov_xx[idx]:.6f}")

# Plot peaks
ax.plot(t[peaks], cov_xx[peaks], "v", color="red", markersize=8,
        label=f"{len(peaks)} peaks")
```

---

## 6.2 Low-Pass Filter: Smoothing Noisy Signals

```python
from scipy.signal import butter, sosfilt


def lowpass_filter(signal: np.ndarray, cutoff_hz: float,
                   sample_rate_hz: float, order: int = 4) -> np.ndarray:
    """Apply a Butterworth low-pass filter."""
    # Normalised cutoff: fraction of Nyquist frequency
    nyquist = sample_rate_hz / 2.0
    normal_cutoff = cutoff_hz / nyquist

    sos = butter(order, normal_cutoff, btype="low", analog=False, output="sos")
    return sosfilt(sos, signal)


# Example: sensor running at ~50Hz, filter noise above 5Hz
sample_rate = 50.0   # Hz
cov_filtered = lowpass_filter(cov_xx, cutoff_hz=5.0, sample_rate_hz=sample_rate)

ax.plot(t, cov_xx, alpha=0.4, label="raw")
ax.plot(t, cov_filtered, linewidth=2.0, label="filtered (5Hz cutoff)")
```

---

## 6.3 Numerical Derivatives with numpy

```python
import numpy as np


# First derivative: rate of change of cov_xx with respect to time
# np.gradient handles non-uniform spacing correctly
dcov_dt = np.gradient(cov_xx, t)

# Second derivative: acceleration of covariance
d2cov_dt2 = np.gradient(dcov_dt, t)

# Find the moment covariance started growing rapidly
GROWTH_THRESHOLD = 0.0005   # m²/s — tune based on sensor noise

growth_mask = dcov_dt > GROWTH_THRESHOLD
if growth_mask.any():
    growth_start_idx = np.argmax(growth_mask)   # index of first True
    growth_start_ts = t[growth_start_idx]
    print(f"Covariance started growing at t = {growth_start_ts:.3f}s")
    print(f"Growth rate at onset: {dcov_dt[growth_start_idx]:.6f} m²/s")
```

---

## 6.4 np.searchsorted: Finding Index for a Timestamp

```python
import numpy as np


# t must be sorted (it usually is — timestamps are monotonic)
t_sorted = np.sort(t)

# Find the array index corresponding to a specific timestamp
event_ts = 1714300012.3
idx = np.searchsorted(t_sorted, event_ts)
idx = min(idx, len(t_sorted) - 1)   # clamp to valid range

print(f"t[{idx}] = {t_sorted[idx]:.3f}s (target: {event_ts}s)")

# Extract a window: ±5 seconds around the event
window_start = np.searchsorted(t_sorted, event_ts - 5.0)
window_end = np.searchsorted(t_sorted, event_ts + 5.0)
window_cov = cov_xx[window_start:window_end]
window_t = t_sorted[window_start:window_end]
```

---

## 6.5 Worked Example: Finding Anomaly Onset

```python
import numpy as np
from scipy.signal import find_peaks
from dataclasses import dataclass
from typing import Optional


DELOCALIZED_THRESHOLD = 0.010
GROWTH_THRESHOLD = 0.0005


@dataclass
class AnomalyOnset:
    breach_ts: float              # when cov first crossed the threshold
    growth_start_ts: float        # when cov started growing toward threshold
    pre_breach_growth_s: float    # seconds of growth before breach
    max_cov: float                # peak cov_xx in the window
    num_peaks: int                # how many spikes found


def find_anomaly_onset(
    t: np.ndarray,
    cov_xx: np.ndarray,
    threshold: float = DELOCALIZED_THRESHOLD,
) -> Optional[AnomalyOnset]:
    """
    Find the onset of a covariance anomaly.

    Returns None if no anomaly detected.
    """
    # Step 1: Is there a breach?
    breach_mask = cov_xx > threshold
    if not breach_mask.any():
        return None

    breach_idx = int(np.argmax(breach_mask))
    breach_ts = float(t[breach_idx])

    # Step 2: When did cov start growing?
    dcov_dt = np.gradient(cov_xx, t)
    growth_mask = dcov_dt > GROWTH_THRESHOLD

    # Find the last growth start before the breach
    pre_breach_growth = growth_mask[:breach_idx]
    if pre_breach_growth.any():
        growth_start_idx = int(np.argmax(pre_breach_growth))
        growth_start_ts = float(t[growth_start_idx])
    else:
        growth_start_ts = breach_ts

    # Step 3: Find peaks in the breach region
    peaks, _ = find_peaks(cov_xx, height=threshold * 0.8, distance=10)

    return AnomalyOnset(
        breach_ts=breach_ts,
        growth_start_ts=growth_start_ts,
        pre_breach_growth_s=breach_ts - growth_start_ts,
        max_cov=float(cov_xx.max()),
        num_peaks=len(peaks),
    )


# Usage
onset = find_anomaly_onset(t, cov_xx)
if onset:
    print(f"Anomaly onset:")
    print(f"  Covariance started growing at: t = {onset.growth_start_ts:.3f}s")
    print(f"  Threshold breached at:         t = {onset.breach_ts:.3f}s")
    print(f"  Pre-breach growth window:      {onset.pre_breach_growth_s:.2f}s")
    print(f"  Peak covariance:               {onset.max_cov:.6f} m²")
    print(f"  Spikes found:                  {onset.num_peaks}")
```

---

# PART 7 — PUTTING IT TOGETHER: COMPLETE ANALYSIS SCRIPT TEMPLATE

---

## 7.1 The Full Script

```python
#!/usr/bin/env python3
"""
Bag CSV Analysis Script Template

Usage:
    python3 analyse_bag.py bag_export.csv --incident-ts 1714300012.3 --output analysis.png

Reads a time-series CSV export, detects anomalies, produces a diagnostic plot,
and prints a text summary.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")   # non-interactive — must come before plt import
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.signal import find_peaks

# ── Constants ────────────────────────────────────────────────────────────────

DELOCALIZED_THRESHOLD = 0.010   # cov_xx (m²) above this = delocalized
GROWTH_THRESHOLD = 0.0005       # dcov/dt (m²/s) above this = growing
MIN_PEAK_PROMINENCE = 0.003     # for find_peaks
MIN_PEAK_DISTANCE = 30          # samples between peaks


# ── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class AnalysisWindow:
    """Describes the time window to analyse."""
    start_ts: float
    end_ts: float
    incident_ts: Optional[float] = None

    @property
    def duration_s(self) -> float:
        return self.end_ts - self.start_ts


@dataclass
class AnomalyReport:
    """Results of anomaly detection."""
    window: AnalysisWindow
    breach_ts: Optional[float] = None
    growth_start_ts: Optional[float] = None
    max_cov: float = 0.0
    mean_cov: float = 0.0
    num_peaks: int = 0
    total_rows: int = 0
    anomalous_rows: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def breach_detected(self) -> bool:
        return self.breach_ts is not None

    @property
    def anomaly_fraction(self) -> float:
        if self.total_rows == 0:
            return 0.0
        return self.anomalous_rows / self.total_rows


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_bag_csv(
    path: Path,
    window: AnalysisWindow,
) -> pl.DataFrame:
    """Load and pre-process a bag CSV for analysis."""
    df = (
        pl.scan_csv(str(path))
        .filter(
            pl.col("timestamp").is_between(window.start_ts, window.end_ts)
        )
        .sort("timestamp")
        .collect()
    )

    if df.is_empty():
        raise ValueError(
            f"No data in window [{window.start_ts:.1f}, {window.end_ts:.1f}]"
        )

    return df


# ── Anomaly Detection ─────────────────────────────────────────────────────────

def detect_anomalies(
    t: np.ndarray,
    cov_xx: np.ndarray,
    window: AnalysisWindow,
) -> AnomalyReport:
    """Run anomaly detection on loaded arrays."""
    report = AnomalyReport(window=window)
    report.total_rows = len(t)
    report.max_cov = float(cov_xx.max())
    report.mean_cov = float(cov_xx.mean())

    # Count anomalous rows
    report.anomalous_rows = int((cov_xx > DELOCALIZED_THRESHOLD).sum())

    # Find breach
    breach_mask = cov_xx > DELOCALIZED_THRESHOLD
    if breach_mask.any():
        breach_idx = int(np.argmax(breach_mask))
        report.breach_ts = float(t[breach_idx])

        # Find when cov started growing before the breach
        dcov_dt = np.gradient(cov_xx, t)
        growth_mask = dcov_dt[:breach_idx] > GROWTH_THRESHOLD
        if growth_mask.any():
            report.growth_start_ts = float(t[int(np.argmax(growth_mask))])

    # Find peaks
    peaks, _ = find_peaks(
        cov_xx,
        height=DELOCALIZED_THRESHOLD * 0.7,
        distance=MIN_PEAK_DISTANCE,
        prominence=MIN_PEAK_PROMINENCE,
    )
    report.num_peaks = len(peaks)

    return report


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_diagnostic(
    t: np.ndarray,
    cov_xx: np.ndarray,
    velocity: np.ndarray,
    status: np.ndarray,
    report: AnomalyReport,
    output_path: Path,
) -> None:
    """Produce a two-panel diagnostic plot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    title = "Bag Export Diagnostic"
    if report.breach_detected:
        title += f" — BREACH at t={report.breach_ts:.2f}s"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # ── Panel 1: Covariance ──
    ax1.plot(t, cov_xx, color="steelblue", linewidth=0.8, label="cov_xx")
    ax1.plot(
        t,
        np.convolve(cov_xx, np.ones(20)/20, mode="same"),
        color="blue",
        linewidth=1.8,
        linestyle="--",
        label="smoothed (20-sample MA)",
    )
    ax1.axhline(
        y=DELOCALIZED_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"threshold={DELOCALIZED_THRESHOLD}",
    )
    ax1.fill_between(
        t, DELOCALIZED_THRESHOLD, cov_xx,
        where=cov_xx > DELOCALIZED_THRESHOLD,
        alpha=0.25, color="orange", label="above threshold",
    )
    if report.breach_ts:
        ax1.axvline(x=report.breach_ts, color="red", linewidth=2, alpha=0.8,
                    label=f"breach t={report.breach_ts:.2f}s")
    if report.growth_start_ts:
        ax1.axvline(x=report.growth_start_ts, color="goldenrod",
                    linewidth=1.5, linestyle=":", alpha=0.9,
                    label=f"growth start t={report.growth_start_ts:.2f}s")
    if report.window.incident_ts:
        ax1.axvline(x=report.window.incident_ts, color="purple",
                    linewidth=2, linestyle="-.",
                    label=f"incident t={report.window.incident_ts:.2f}s")
    ax1.set_ylabel("Covariance XX (m²)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.25)

    # ── Panel 2: Velocity ──
    speed = np.abs(velocity)
    max_speed = float(speed.max()) if speed.max() > 0 else 1.0
    ax2.plot(t, speed, color="forestgreen", linewidth=0.8, label="|velocity|")
    ax2.fill_between(
        t, 0, max_speed,
        where=status > 0,
        alpha=0.20, color="red", label="status > 0",
    )
    ax2.set_ylabel("Speed (m/s)")
    ax2.set_xlabel("Time (s relative to window start)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.25)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {output_path}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(report: AnomalyReport) -> None:
    """Print a human-readable text summary."""
    sep = "─" * 55
    print(f"\n{sep}")
    print("  ANALYSIS SUMMARY")
    print(sep)
    print(f"  Window:            {report.window.start_ts:.1f}s → {report.window.end_ts:.1f}s"
          f"  ({report.window.duration_s:.1f}s)")
    print(f"  Total rows:        {report.total_rows:,}")
    print(f"  Max cov_xx:        {report.max_cov:.6f} m²")
    print(f"  Mean cov_xx:       {report.mean_cov:.6f} m²")
    print(f"  Anomalous rows:    {report.anomalous_rows:,}"
          f"  ({report.anomaly_fraction*100:.1f}%)")
    print(f"  Peaks found:       {report.num_peaks}")
    if report.breach_ts:
        print(f"\n  BREACH DETECTED:")
        print(f"  Threshold crossed: t = {report.breach_ts:.3f}s")
        if report.growth_start_ts:
            lead = report.breach_ts - report.growth_start_ts
            print(f"  Growth started:    t = {report.growth_start_ts:.3f}s")
            print(f"  Pre-breach lead:   {lead:.2f}s")
    else:
        print("\n  No threshold breach detected.")
    if report.notes:
        print("\n  Notes:")
        for note in report.notes:
            print(f"    • {note}")
    print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv_path", type=Path, help="Path to bag export CSV")
    p.add_argument("--start-ts", type=float, default=None,
                   help="Window start (epoch seconds). Default: file start.")
    p.add_argument("--end-ts", type=float, default=None,
                   help="Window end (epoch seconds). Default: file end.")
    p.add_argument("--incident-ts", type=float, default=None,
                   help="Known incident timestamp (epoch seconds)")
    p.add_argument("--output", type=Path, default=Path("analysis.png"),
                   help="Output plot path (default: analysis.png)")
    p.add_argument("--threshold", type=float, default=DELOCALIZED_THRESHOLD,
                   help=f"cov_xx threshold (default: {DELOCALIZED_THRESHOLD})")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.csv_path.exists():
        print(f"ERROR: file not found: {args.csv_path}", file=sys.stderr)
        return 1

    # Step 1: Determine time window
    # Quick scan to find file's time range without loading everything
    meta = pl.scan_csv(str(args.csv_path)).select(
        pl.col("timestamp").min().alias("t_min"),
        pl.col("timestamp").max().alias("t_max"),
    ).collect()
    t_min = float(meta["t_min"][0])
    t_max = float(meta["t_max"][0])

    window = AnalysisWindow(
        start_ts=args.start_ts if args.start_ts else t_min,
        end_ts=args.end_ts if args.end_ts else t_max,
        incident_ts=args.incident_ts,
    )
    print(f"Analysing: {args.csv_path}")
    print(f"Window: {window.start_ts:.1f}s → {window.end_ts:.1f}s"
          f"  ({window.duration_s:.1f}s)")

    # Step 2: Load data
    df = load_bag_csv(args.csv_path, window)
    print(f"Loaded {len(df):,} rows")

    # Step 3: Extract arrays
    t = df["timestamp"].to_numpy().astype(float)
    t_rel = t - t[0]   # relative timestamps for plotting

    cov_xx = df["cov_xx"].to_numpy().astype(float)
    velocity = df["velocity"].to_numpy().astype(float) if "velocity" in df.columns \
               else np.zeros_like(t)
    status = df["status"].to_numpy().astype(float) if "status" in df.columns \
             else np.zeros_like(t)

    # Step 4: Detect anomalies
    report = detect_anomalies(t, cov_xx, window)

    # Step 5: Plot
    plot_diagnostic(t_rel, cov_xx, velocity, status, report, args.output)

    # Step 6: Print summary
    print_summary(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

## 7.2 Running the Script

```bash
# Basic usage
python3 analyse_bag.py bag_export.csv

# With known incident time
python3 analyse_bag.py bag_export.csv --incident-ts 1714300012.3

# Custom window and output
python3 analyse_bag.py bag_export.csv \
    --start-ts 1714300000 \
    --end-ts 1714300300 \
    --output plots/session_42_diagnostic.png

# Custom threshold
python3 analyse_bag.py bag_export.csv --threshold 0.005
```

---

## Summary — What to Remember

| Concept | Rule |
|---------|------|
| `pd.to_datetime(col, unit="s")` | Convert epoch float to pandas Timestamp |
| `df.rolling(window=N).mean()` | Smooth a signal; first N-1 values are NaN |
| `pl.scan_csv()` + `.collect()` | Lazy reading: 5–20x faster than `read_csv` for large filtered queries |
| `df.filter(pl.col("x") > v)` | polars filter; returns a new DataFrame |
| `df.with_columns(pl.col("x").diff())` | Add computed column; non-mutating |
| `group_by_dynamic("ts", every="500ms")` | Resample into fixed time bins |
| `fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)` | Two panels with a shared time axis |
| `ax.axhline(y=threshold, ...)` | Horizontal threshold reference line |
| `ax.fill_between(t, 0, v, where=mask, ...)` | Shade anomalous regions |
| `plt.savefig("out.png", dpi=150)` + `plt.close(fig)` | Save without displaying; always close to free memory |
| `np.gradient(signal, t)` | Numerical first derivative with non-uniform spacing |
| `find_peaks(signal, height=h, distance=d)` | Find spikes; tune `height`/`distance`/`prominence` |
| `np.searchsorted(t_sorted, ts)` | Find array index for a specific timestamp (fast) |

---

# QUICK REFERENCE CARD

```
┌──────────────────────────────── TIME-SERIES CHEAT SHEET ────────────────────────────────────┐
│                                                                                              │
│  PANDAS (small/medium datasets):                                                             │
│    pd.read_csv("f.csv")                                                                     │
│    df[df["col"] > v]                   filter rows                                          │
│    df.set_index("timestamp")           make timestamp the index                             │
│    df["col"].rolling(N).mean()         rolling average                                      │
│    pd.to_datetime(df["ts"], unit="s")  epoch float → Timestamp                             │
│                                                                                              │
│  POLARS (large files, scan):                                                                 │
│    pl.scan_csv("f.csv")                lazy: plan only, no read                             │
│    .filter(pl.col("x") > v)            row filter                                           │
│    .with_columns(pl.col("x").diff())   add computed column                                  │
│    .group_by_dynamic("ts", every="1s") time binning                                         │
│    .collect()                          execute the plan                                     │
│                                                                                              │
│  MATPLOTLIB (plotting):                                                                      │
│    fig, ax = plt.subplots(figsize=(14,6))                                                   │
│    ax.plot(t, y, label="...", color="...")                                                  │
│    ax.axhline(y=thresh, color="red", linestyle="--")                                        │
│    ax.axvline(x=ts, color="orange", linewidth=2)                                            │
│    ax.fill_between(t, 0, y, where=mask, alpha=0.3, color="red")                            │
│    ax2 = ax.twinx()                    dual y-axes                                          │
│    plt.savefig("out.png", dpi=150) ; plt.close(fig)                                        │
│                                                                                              │
│  SCIPY (signal analysis):                                                                    │
│    np.gradient(y, t)                   first derivative                                     │
│    find_peaks(y, height=h, distance=d) find spikes                                         │
│    butter(N, fc/nyq, btype="low") → sosfilt(sos, y)  low-pass filter                      │
│    np.searchsorted(t_sorted, ts)       index for timestamp                                  │
│    np.argmax(mask)                     index of first True in boolean array                 │
│                                                                                              │
│  PATTERN — THRESHOLD BREACH TIME:                                                            │
│    breach_idx = np.argmax(signal > threshold)                                               │
│    breach_ts = t[breach_idx]           (assumes sorted t)                                   │
│                                                                                              │
│  PATTERN — ANOMALY ONSET LEAD TIME:                                                          │
│    dcov_dt = np.gradient(cov_xx, t)                                                         │
│    growth_start_idx = np.argmax(dcov_dt[:breach_idx] > GROWTH_THRESHOLD)                   │
│    lead_s = breach_ts - t[growth_start_idx]                                                 │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```
