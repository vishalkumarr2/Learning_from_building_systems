# Exercise 03 — Log Analysis with polars, matplotlib, and scipy

### Companion exercises for `03-timeseries-analysis.md`

**Estimated time:** 60 minutes  
**Prerequisite:** [03-timeseries-analysis.md](../03-timeseries-analysis.md)

Part A tests conceptual understanding. Part B is code reading — trace the output before running anything. Part C is a complete free-build pipeline.

---

## Part A — Concepts Check

**1.** What is the key architectural difference between pandas and polars that makes polars faster on large files?

<details><summary>Answer</summary>

| Aspect | pandas | polars |
|---|---|---|
| Execution model | Eager — operations run immediately | Lazy — builds a query plan, executes on `.collect()` |
| Internals | Python + NumPy, row-major iteration common | Rust + Apache Arrow, columnar memory layout |
| Parallelism | Single-threaded by default | Multi-threaded by default (Rayon thread pool) |
| Memory | Can copy data between steps | Often operates in-place on Arrow buffers |

The biggest practical win is **lazy evaluation**: `pl.scan_csv(...).filter(...).group_by(...).agg(...)` generates an optimised execution plan before reading a single byte. polars can push the filter down to the CSV reader, skipping rows it will discard. pandas `read_csv` always loads the full file first.

For a 500 MB CSV where you only need 5% of rows, polars reads ~25 MB; pandas reads 500 MB then filters.

</details>

- [ ] Done

---

**2.** What does `.lazy()` / `scan_csv()` do that `read_csv()` does not?

<details><summary>Answer</summary>

`read_csv()` reads the entire file into memory immediately and returns a `DataFrame`.

`scan_csv()` (or `df.lazy()`) returns a `LazyFrame` — a query plan object. No data is read or processed until you call `.collect()`.

```python
# Eager — reads all 2 GB immediately
df = pl.read_csv("large_log.csv")
filtered = df.filter(pl.col("status") != 0)

# Lazy — reads only rows that pass the filter
filtered = (
    pl.scan_csv("large_log.csv")
    .filter(pl.col("status") != 0)
    .collect()
)
```

With `scan_csv`, polars can:
1. **Predicate pushdown** — apply `filter` conditions while parsing, skipping filtered rows
2. **Projection pushdown** — if you only `select(["timestamp", "status"])`, unused columns are never parsed
3. **Plan optimisation** — reorder operations for minimal memory usage

Use `scan_csv` when: files are large (> 50 MB), you filter to a small subset, or you chain multiple operations. Use `read_csv` for small files or when you need the full `DataFrame` immediately.

</details>

- [ ] Done

---

**3.** `df.filter(pl.col("status") != 0)` — what does `pl.col("status")` return?

<details><summary>Answer</summary>

`pl.col("status")` returns an **`Expr`** (expression) object — not a series of values. It is a symbolic reference to the `"status"` column that polars will evaluate when the expression is applied to a DataFrame.

```python
expr = pl.col("status") != 0
print(type(expr))   # <class 'polars.expr.expr.Expr'>

# Expressions can be composed:
anomaly = (pl.col("cov_xx") > 0.01) | (pl.col("cov_yy") > 0.01)
df.filter(anomaly)

# Expressions work in select, with_columns, group_by, agg:
df.with_columns(
    (pl.col("cov_xx") + pl.col("cov_yy")).alias("cov_sum")
)
```

The expression model is why polars can optimise queries: it inspects the `Expr` tree before executing. Compare to pandas where `df["status"] != 0` immediately computes and returns a Series.

</details>

- [ ] Done

---

**4.** Why is `rolling_mean` with `window_size=N` not the same as `group_by_dynamic` with `every="500ms"`?

<details><summary>Answer</summary>

They answer different questions:

**`rolling_mean(window_size=N)`** — a sliding window over **N rows**:
```python
df.with_columns(pl.col("cov_xx").rolling_mean(window_size=10))
```
- Window is row-count based: always the previous 10 rows
- If timestamps are irregular (e.g., sensor dropouts skip timestamps), the time span of the window varies
- Output has the same number of rows as the input

**`group_by_dynamic(every="500ms")`** — bins by **time interval**:
```python
df.group_by_dynamic("timestamp", every="500ms").agg(pl.col("cov_xx").mean())
```
- Window is time-based: always a 500 ms calendar slice
- Handles irregular sampling correctly (sparse windows have fewer rows contributing to the mean)
- Output has one row per time bin (fewer rows than input)

Use `rolling_mean` for smoothing noisy signals with regular sampling.  
Use `group_by_dynamic` for resampling, downsampling, or producing per-window statistics with irregular or variable-rate data.

</details>

- [ ] Done

---

**5.** When would you use `scipy.signal.find_peaks` vs a simple threshold filter?

<details><summary>Answer</summary>

**Simple threshold** (`df.filter(pl.col("cov_xx") > 0.01)`):
- Use when you want all points above a value (contiguous regions)
- Good for: "how long was cov above threshold?", "what fraction of the window is anomalous?"
- Detects sustained periods, not discrete events

**`scipy.signal.find_peaks(arr, height=0.01, distance=50)`**:
- Use when you want discrete local maxima — peaks that rise and fall
- Good for: "when did cov spike?", "find the 3 sharpest covariance explosions", "detect individual slip events"
- `distance=50` prevents detecting the same event twice (minimum 50 samples between peaks)
- `prominence` parameter distinguishes significant peaks from noise ripples

```python
import scipy.signal as signal
import numpy as np

cov = df["cov_xx"].to_numpy()
peaks, props = signal.find_peaks(cov, height=0.01, prominence=0.005, distance=100)
peak_timestamps = df["timestamp"].to_numpy()[peaks]
```

Practical guide:
- "Was there an anomaly in this window?" → threshold filter
- "When exactly did it spike, and how sharp was it?" → `find_peaks`
- "How long was the anomaly?" → threshold + `diff()` to find transitions

</details>

- [ ] Done

---

## Part B — Code Reading

Trace the output of each snippet **without running it**. Then check by running.

**1.** What does this produce? Describe the shape and columns of the result.

```python
import polars as pl

df = pl.DataFrame({
    "timestamp": [1.0, 1.1, 1.2, 2.0, 2.1],
    "status":    [0,   1,   0,   2,   0  ],
    "value":     [0.1, 0.5, 0.2, 0.8, 0.1],
})

result = (
    df.lazy()
    .filter(pl.col("status") != 0)
    .group_by("status")
    .agg(pl.count())
    .collect()
)
```

<details><summary>Answer</summary>

After `.filter(pl.col("status") != 0)`, two rows remain: `status=1` (row index 1) and `status=2` (row index 3).

`.group_by("status").agg(pl.count())` counts rows per status group.

Result:
```
shape: (2, 2)
┌────────┬───────┐
│ status ┆ count │
│ ---    ┆ ---   │
│ i64    ┆ u32   │
╞════════╪═══════╡
│ 1      ┆ 1     │
│ 2      ┆ 1     │
└────────┴───────┘
```

**Note:** `group_by` does **not** guarantee row order — you may see `status=2` before `status=1`. Add `.sort("status")` after `.agg()` if order matters.

</details>

- [ ] Done

---

**2.** What is the value of `dcov` at row 0, and why?

```python
df = pl.DataFrame({
    "timestamp": [0.0, 0.1, 0.2, 0.3],
    "cov_xx":    [0.001, 0.003, 0.008, 0.020],
})

result = df.with_columns(
    pl.col("cov_xx").diff().alias("dcov")
)
print(result)
```

<details><summary>Answer</summary>

```
shape: (4, 3)
┌───────────┬────────┬────────┐
│ timestamp ┆ cov_xx ┆ dcov   │
│ ---       ┆ ---    ┆ ---    │
│ f64       ┆ f64    ┆ f64    │
╞═══════════╪════════╪════════╡
│ 0.0       ┆ 0.001  ┆ null   │
│ 0.1       ┆ 0.003  ┆ 0.002  │
│ 0.2       ┆ 0.008  ┆ 0.005  │
│ 0.3       ┆ 0.020  ┆ 0.012  │
└───────────┴────────┴────────┘
```

Row 0 of `dcov` is **`null`** (not 0, not NaN). `.diff()` computes `current - previous` and there is no previous row for the first element, so polars uses `null`.

This matters when you chain further operations: `pl.col("dcov") > 0.005` on row 0 evaluates to `null` (not `False`). Filter conditions with `null` drop the row. If you need a 0 instead of null, use `.diff().fill_null(0)`.

</details>

- [ ] Done

---

**3.** Are these two expressions always equivalent? If not, when would they differ?

```python
# Expression A
result_a = df.sort("timestamp").filter(pl.col("cov_xx") > 0.01).head(1)

# Expression B
result_b = df.filter(pl.col("cov_xx") > 0.01).sort("timestamp").head(1)
```

<details><summary>Answer</summary>

**They are equivalent in result** — both return the row with the earliest `timestamp` among rows where `cov_xx > 0.01`. The `.head(1)` takes the first row after sorting, so both produce the same single row.

**They differ in performance:**
- Expression A sorts the full DataFrame first (all N rows), then filters
- Expression B filters first (to M rows, M ≤ N), then sorts only M rows

For large DataFrames where the filter is selective (M << N), Expression B is faster. Expression A does unnecessary work sorting rows that the filter discards.

**When they give different results:**
If the DataFrame has duplicate `timestamp` values and `cov_xx > 0.01` for multiple rows at that timestamp, the row order within the same timestamp group is undefined after `filter`. The `head(1)` could pick different rows depending on polars' internal ordering. To get deterministic results, add a secondary sort key: `.sort(["timestamp", "cov_xx"], descending=[False, True])`.

</details>

- [ ] Done

---

**4.** Translate this pandas rolling mean to polars.

```python
import pandas as pd

df_pd = pd.read_csv("data.csv")
df_pd["smooth_cov"] = df_pd["cov_xx"].rolling(window=10, min_periods=1).mean()
```

<details><summary>Answer</summary>

```python
import polars as pl

df_pl = pl.read_csv("data.csv")
df_pl = df_pl.with_columns(
    pl.col("cov_xx")
    .rolling_mean(window_size=10, min_periods=1)
    .alias("smooth_cov")
)
```

Key differences:
- `window=10` → `window_size=10`
- `min_periods=1` → `min_periods=1` (same parameter name)
- In polars the result is added via `.with_columns()` — there is no in-place column assignment
- polars `.rolling_mean()` computes a **trailing** window (uses the current row and the N-1 rows before it), matching pandas default `center=False`

If you want a centred window (pandas `center=True`):
```python
pl.col("cov_xx").rolling_mean(window_size=10, center=True)
```

</details>

- [ ] Done

---

**5.** What does `np.gradient(arr, timestamps)` approximate, and what are its edge-case properties?

```python
import numpy as np

timestamps = np.array([0.0, 0.1, 0.2, 0.4, 0.5])   # non-uniform spacing
cov_xx     = np.array([0.001, 0.002, 0.010, 0.015, 0.012])

dcov_dt = np.gradient(cov_xx, timestamps)
```

<details><summary>Answer</summary>

`np.gradient(f, x)` approximates the **first derivative** `df/dx` at each point using finite differences.

For interior points it uses the **central difference**:
```
f'[i] ≈ (f[i+1] - f[i-1]) / (x[i+1] - x[i-1])
```

For the first and last points it uses **one-sided differences**:
```
f'[0] ≈ (f[1] - f[0]) / (x[1] - x[0])     # forward difference
f'[-1] ≈ (f[-1] - f[-2]) / (x[-1] - x[-2]) # backward difference
```

For the example above:
- `dcov_dt[0]` ≈ (0.002 - 0.001) / 0.1 = 0.01 /s
- `dcov_dt[2]` ≈ (0.015 - 0.002) / (0.4 - 0.1) = 0.013 / 0.3 ≈ 0.043 /s  ← fast rise
- `dcov_dt[4]` ≈ (0.012 - 0.015) / (0.5 - 0.4) = -0.03 /s ← falling

**Edge-case properties:**
1. **Non-uniform spacing is handled** — `np.gradient` accepts the `x` coordinates as the second argument and accounts for variable `Δx`
2. **First/last values are less accurate** (one-sided vs. central) — avoid relying on derivatives at array boundaries
3. **Amplifies noise** — differentiation emphasises high-frequency components; smooth the signal first with a rolling mean if noise is present

</details>

- [ ] Done

---

## Part C — Build Exercises

### Exercise C1: Anomaly Detection Pipeline

Given a CSV file with columns: `timestamp` (float, unix epoch), `cov_xx` (float), `cov_yy` (float), `status` (int), `velocity` (float).

Write a complete polars pipeline that:
1. Loads the CSV lazily
2. Filters to the time window `[t_start, t_end]`
3. Adds a column `cov_max = max(cov_xx, cov_yy)` per row
4. Adds a boolean column `is_anomaly` where `cov_max > threshold`
5. Groups into 500 ms time bins and finds the maximum `cov_max` per bin
6. Returns a dict `{"first_anomaly_ts": float | None, "peak_cov": float}`

Use `.collect()` exactly once.

<details><summary>Answer</summary>

```python
from pathlib import Path
import polars as pl


def detect_anomalies(
    csv_path: Path,
    t_start: float,
    t_end: float,
    threshold: float = 0.01,
) -> dict:
    """
    Run anomaly detection over a time window.

    Returns:
        {"first_anomaly_ts": float | None, "peak_cov": float}
    """
    df = (
        pl.scan_csv(csv_path)
        .filter(
            (pl.col("timestamp") >= t_start) & (pl.col("timestamp") <= t_end)
        )
        .with_columns(
            pl.max_horizontal("cov_xx", "cov_yy").alias("cov_max")
        )
        .with_columns(
            (pl.col("cov_max") > threshold).alias("is_anomaly")
        )
        .collect()
    )

    if df.is_empty():
        return {"first_anomaly_ts": None, "peak_cov": 0.0}

    peak_cov = float(df["cov_max"].max())

    # Group into 500 ms bins (timestamp is unix epoch float -> use duration_ms)
    # Convert timestamp to milliseconds for group_by_dynamic
    df_bins = (
        df.with_columns(
            (pl.col("timestamp") * 1000).cast(pl.Int64).alias("ts_ms")
        )
        .sort("ts_ms")
        .group_by_dynamic("ts_ms", every="500i")   # 500 integer ms bins
        .agg(pl.col("cov_max").max().alias("bin_max_cov"))
    )

    anomaly_rows = df.filter(pl.col("is_anomaly")).sort("timestamp")
    first_anomaly_ts: float | None = (
        float(anomaly_rows["timestamp"][0])
        if len(anomaly_rows) > 0
        else None
    )

    return {"first_anomaly_ts": first_anomaly_ts, "peak_cov": peak_cov}
```

Usage:
```python
result = detect_anomalies(
    csv_path=Path("session_export.csv"),
    t_start=1745000000.0,
    t_end=1745000060.0,
    threshold=0.01,
)
print(result)
# {"first_anomaly_ts": 1745000012.34, "peak_cov": 0.087}
```

**Key polars notes:**
- `pl.max_horizontal("cov_xx", "cov_yy")` computes row-wise max (polars 0.19+). For older versions: `pl.concat_list(["cov_xx", "cov_yy"]).list.max()`
- `.collect()` is called once — the lazy chain processes the CSV in a single pass
- `group_by_dynamic` requires a sorted column; the `.sort("ts_ms")` before it is mandatory

</details>

- [ ] Done

---

### Exercise C2: Diagnostic Plot Function

Implement this function signature completely:

```python
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt


def plot_covariance_timeline(
    csv_path: Path,
    t_start: float,
    t_end: float,
    output_path: Path,
    threshold: float = 0.01,
) -> dict:
    """
    Plot cov_xx, velocity, and status over the time window.

    Returns:
        {"anomaly_start": float | None, "peak_cov": float}
    """
    ...
```

Requirements:
- 3 vertically stacked subplots with a shared x-axis (time): `cov_xx` with threshold line, `velocity`, `status` as scatter points
- Shade periods where `status != 0` in translucent red across all three panels using `axvspan`
- Save the figure to `output_path`
- Return the timestamp of the first threshold crossing and the peak covariance value
- Load and filter data with polars

<details><summary>Answer</summary>

```python
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def plot_covariance_timeline(
    csv_path: Path,
    t_start: float,
    t_end: float,
    output_path: Path,
    threshold: float = 0.01,
) -> dict:
    """
    Plot cov_xx, velocity, and status over the time window.

    Returns:
        {"anomaly_start": float | None, "peak_cov": float}
    """
    df = (
        pl.scan_csv(csv_path)
        .filter(
            (pl.col("timestamp") >= t_start) & (pl.col("timestamp") <= t_end)
        )
        .collect()
        .sort("timestamp")
    )

    if df.is_empty():
        return {"anomaly_start": None, "peak_cov": 0.0}

    ts = df["timestamp"].to_numpy()
    cov = df["cov_xx"].to_numpy()
    vel = df["velocity"].to_numpy()
    status = df["status"].to_numpy()

    # Compute return values before plotting
    peak_cov = float(np.max(cov))
    above_threshold = np.where(cov > threshold)[0]
    anomaly_start: Optional[float] = (
        float(ts[above_threshold[0]]) if len(above_threshold) > 0 else None
    )

    # Find anomalous spans (status != 0) for shading
    def get_spans(ts_arr: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
        """Return list of (start, end) pairs for contiguous True regions."""
        spans = []
        in_span = False
        span_start = 0.0
        for i, (t, m) in enumerate(zip(ts_arr, mask)):
            if m and not in_span:
                span_start = t
                in_span = True
            elif not m and in_span:
                spans.append((span_start, ts_arr[i - 1]))
                in_span = False
        if in_span:
            spans.append((span_start, ts_arr[-1]))
        return spans

    anomaly_spans = get_spans(ts, status != 0)

    # Build figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(
        f"Covariance Timeline  |  window: [{t_start:.1f}, {t_end:.1f}]",
        fontsize=12,
    )

    # --- Panel 1: cov_xx ---
    ax0 = axes[0]
    ax0.plot(ts, cov, linewidth=1.0, color="steelblue", label="cov_xx")
    ax0.axhline(threshold, color="red", linestyle="--", linewidth=0.8, label=f"threshold={threshold}")
    ax0.set_ylabel("cov_xx")
    ax0.legend(fontsize=8)

    # --- Panel 2: velocity ---
    ax1 = axes[1]
    ax1.plot(ts, vel, linewidth=1.0, color="darkorange", label="velocity")
    ax1.set_ylabel("velocity (m/s)")
    ax1.legend(fontsize=8)

    # --- Panel 3: status ---
    ax2 = axes[2]
    ax2.scatter(ts, status, s=4, c=np.where(status == 0, "grey", "crimson"), zorder=3)
    ax2.set_ylabel("status")
    ax2.set_xlabel("timestamp (s)")

    # Shade anomalous spans on all panels
    for ax in axes:
        for span_start, span_end in anomaly_spans:
            ax.axvspan(span_start, span_end, alpha=0.15, color="red", zorder=0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"anomaly_start": anomaly_start, "peak_cov": peak_cov}
```

</details>

- [ ] Done

---

### Exercise C3: End-to-End CLI Analysis Script

Write a complete CLI script `analysis.py` using `argparse` that:

- Arguments: `--csv PATH`, `--start FLOAT`, `--end FLOAT`, `--threshold FLOAT` (default 0.01), `--output-dir DIR`
- Loads CSV with polars, filters to `[--start, --end]`
- Runs anomaly detection: finds first threshold crossing, peak covariance, duration above threshold, % of window above threshold
- Prints a formatted text summary table to stdout
- Saves the diagnostic plot to `output_dir/analysis_plot.png`
- Handles errors gracefully: file not found, empty CSV, no data in window

All functions must have type annotations.

<details><summary>Answer</summary>

```python
#!/usr/bin/env python3
"""analysis.py — Covariance anomaly analysis from a CSV export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless runs
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_window(csv_path: Path, t_start: float, t_end: float) -> pl.DataFrame:
    """Load and filter the CSV to the given time window."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = (
        pl.scan_csv(csv_path)
        .filter(
            (pl.col("timestamp") >= t_start) & (pl.col("timestamp") <= t_end)
        )
        .collect()
        .sort("timestamp")
    )
    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_stats(df: pl.DataFrame, threshold: float) -> dict:
    """Compute anomaly statistics over a windowed DataFrame."""
    if df.is_empty():
        return {
            "row_count": 0,
            "anomaly_start": None,
            "peak_cov": 0.0,
            "duration_above_s": 0.0,
            "pct_above": 0.0,
        }

    cov = df["cov_xx"].to_numpy()
    ts = df["timestamp"].to_numpy()

    peak_cov = float(np.max(cov))
    above = cov > threshold

    above_indices = np.where(above)[0]
    anomaly_start: float | None = float(ts[above_indices[0]]) if len(above_indices) > 0 else None

    # Approximate duration: sum of timestep intervals where above threshold
    dt = np.diff(ts, prepend=ts[0])
    duration_above = float(np.sum(dt[above]))
    total_duration = float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0
    pct_above = 100.0 * float(np.sum(above)) / len(above)

    return {
        "row_count": len(df),
        "anomaly_start": anomaly_start,
        "peak_cov": peak_cov,
        "duration_above_s": duration_above,
        "pct_above": pct_above,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plot(
    df: pl.DataFrame,
    output_path: Path,
    threshold: float,
    t_start: float,
    t_end: float,
) -> None:
    """Generate a 3-panel diagnostic plot and save to output_path."""
    ts = df["timestamp"].to_numpy()
    cov = df["cov_xx"].to_numpy()
    vel = df["velocity"].to_numpy()
    status = df["status"].to_numpy()

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"Covariance Analysis  [{t_start:.1f}s – {t_end:.1f}s]", fontsize=11)

    axes[0].plot(ts, cov, linewidth=0.9, color="steelblue")
    axes[0].axhline(threshold, color="red", linestyle="--", linewidth=0.8,
                    label=f"threshold={threshold}")
    axes[0].set_ylabel("cov_xx")
    axes[0].legend(fontsize=8)

    axes[1].plot(ts, vel, linewidth=0.9, color="darkorange")
    axes[1].set_ylabel("velocity (m/s)")

    axes[2].scatter(ts, status, s=3,
                    c=np.where(status == 0, "grey", "crimson"), zorder=3)
    axes[2].set_ylabel("status")
    axes[2].set_xlabel("timestamp (s)")

    # Shade anomalous regions
    in_span = False
    span_start = 0.0
    for i, (t, s) in enumerate(zip(ts, status)):
        if s != 0 and not in_span:
            span_start, in_span = t, True
        elif s == 0 and in_span:
            for ax in axes:
                ax.axvspan(span_start, ts[i - 1], alpha=0.15, color="red", zorder=0)
            in_span = False
    if in_span:
        for ax in axes:
            ax.axvspan(span_start, ts[-1], alpha=0.15, color="red", zorder=0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(stats: dict, t_start: float, t_end: float, threshold: float) -> None:
    """Print a text summary table to stdout."""
    width = 46
    print("=" * width)
    print(f"  Covariance Analysis Report")
    print(f"  Window  : {t_start:.2f}s → {t_end:.2f}s  ({t_end - t_start:.1f}s)")
    print(f"  Threshold: {threshold}")
    print("=" * width)
    print(f"  Rows in window       : {stats['row_count']}")
    if stats["anomaly_start"] is not None:
        print(f"  First anomaly at     : {stats['anomaly_start']:.3f}s")
    else:
        print(f"  First anomaly at     : (none)")
    print(f"  Peak cov_xx          : {stats['peak_cov']:.6f}")
    print(f"  Duration above thresh: {stats['duration_above_s']:.2f}s")
    print(f"  % of window above    : {stats['pct_above']:.1f}%")
    print("=" * width)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Covariance anomaly analysis from a CSV log export."
    )
    parser.add_argument("--csv", required=True, type=Path, metavar="PATH",
                        help="Path to the CSV file")
    parser.add_argument("--start", required=True, type=float, metavar="TIMESTAMP",
                        help="Window start timestamp (unix epoch)")
    parser.add_argument("--end", required=True, type=float, metavar="TIMESTAMP",
                        help="Window end timestamp (unix epoch)")
    parser.add_argument("--threshold", type=float, default=0.01, metavar="FLOAT",
                        help="Covariance anomaly threshold (default: 0.01)")
    parser.add_argument("--output-dir", required=True, type=Path, metavar="DIR",
                        help="Directory where analysis_plot.png will be saved")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.start >= args.end:
        print(f"ERROR: --start must be < --end ({args.start} >= {args.end})", file=sys.stderr)
        sys.exit(1)

    try:
        df = load_window(args.csv, args.start, args.end)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    if df.is_empty():
        print(
            f"ERROR: No data in window [{args.start}, {args.end}] — "
            "check timestamps or CSV content.",
            file=sys.stderr,
        )
        sys.exit(1)

    stats = compute_stats(df, args.threshold)
    print_summary(stats, args.start, args.end, args.threshold)

    plot_path = args.output_dir / "analysis_plot.png"
    save_plot(df, plot_path, args.threshold, args.start, args.end)
    print(f"\nPlot saved → {plot_path}")


if __name__ == "__main__":
    main()
```

Usage:
```bash
python analysis.py \
  --csv session_export.csv \
  --start 1745000000 \
  --end   1745000060 \
  --threshold 0.01 \
  --output-dir ./output
```

Sample output:
```
==============================================
  Covariance Analysis Report
  Window  : 1745000000.00s → 1745000060.00s  (60.0s)
  Threshold: 0.01
==============================================
  Rows in window       : 6000
  First anomaly at     : 1745000012.340s
  Peak cov_xx          : 0.087431
  Duration above thresh: 14.82s
  % of window above    : 24.7%
==============================================

Plot saved → output/analysis_plot.png
```

</details>

- [ ] Done
