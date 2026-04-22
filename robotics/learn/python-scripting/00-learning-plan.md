# Python / Robot Scripting — Learning Plan
### For: Engineer writing daily RCA scripts who wants them testable, typed, and maintainable
### Goal: Write analysis scripts that future-you can understand in 6 months

---

## Why This Track Exists

You write Python daily: `knowledge_search.py`, `session_tracker.py`, `knowledge_builder.py`, log analysis scripts.
But they lack:
- Type annotations (hard to refactor safely)
- Tests (regressions silently break things)
- Structured data parsing (ad-hoc string manipulation in many places)

**Goal:** After 3 weeks, every new script you write is typed, has pytest coverage, and
uses dataclasses/Pydantic for data boundaries.

---

## Weekly Schedule

### Week 1: Type Hints + Dataclasses + Pydantic (2 hrs)
- Type annotations: `Optional`, `Union`, `TypedDict`, `Protocol`
- `@dataclass` vs `NamedTuple` vs `Pydantic BaseModel` — when to use each
- Validating config files (`.env` → Pydantic settings)

**Exercise:** Refactor `session_tracker.py` session state into a typed Pydantic model

### Week 2: Testing CLI Scripts with pytest (2 hrs)
- `pytest` fixtures, `tmp_path`, `monkeypatch`
- Mocking ADO API calls with `pytest-mock`
- Testing scripts that read from `.env`

**Exercise:** Write tests for `knowledge_search.py` — test the scoring logic without hitting ADO

### Week 3: Time-Series Log Analysis with polars (3 hrs)
- polars vs pandas: why polars is faster for large bag CSVs
- Lazy evaluation: `scan_csv().filter().collect()`
- Time-window aggregations: rolling max, group-by time bin
- Plotting with matplotlib: covariance vs time, velocity vs time

**Exercise:** Load a bag CSV and reproduce the covariance blow-up plot using polars + matplotlib

---

## Status: � Content complete — 3 lessons + 3 exercises
