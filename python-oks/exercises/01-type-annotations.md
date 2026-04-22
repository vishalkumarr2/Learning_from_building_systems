# Exercise 01 — Type Annotations, Dataclasses, and Pydantic

### Companion exercises for `01-types-dataclasses.md`

**Estimated time:** 45 minutes  
**Prerequisite:** [01-types-dataclasses.md](../01-types-dataclasses.md)

Write answers on paper or in a scratch file before expanding each `Answer` block. If you can answer Part A without hints you're solid on the concepts; Part C is the real test.

---

## Part A — Concepts Check

**1.** What does `Optional[str]` mean? Write the equivalent Python 3.10+ syntax.

<details><summary>Answer</summary>

`Optional[str]` is shorthand for `Union[str, None]` — the value is either a `str` or `None`.

Python 3.10+ union syntax (no import needed):
```python
# Old style (works everywhere)
from typing import Optional
def greet(name: Optional[str]) -> str: ...

# 3.10+ style
def greet(name: str | None) -> str: ...
```

**Critical gotcha:** `Optional[str]` does **not** mean "the argument is optional (has a default)". It means the value may be `None`. A required parameter can be typed `Optional[str]` — the caller must explicitly pass `None`.

</details>

- [ ] Done

---

**2.** When would you use `TypedDict` instead of a regular `dict`? Give a concrete example.

<details><summary>Answer</summary>

Use `TypedDict` when a dict has a fixed, known schema and you want the type checker to catch mistakes on key access and value types.

```python
from typing import TypedDict

class SearchResult(TypedDict):
    query: str
    confidence: float
    matched_pattern: str
    source_ids: list[int]

# mypy checks this:
result: SearchResult = {
    "query": "motor stall",
    "confidence": 0.92,
    "matched_pattern": "motor-overload-v2",
    "source_ids": [101, 207],
}

result["typo_key"]  # mypy ERROR: TypedDict has no key "typo_key"
```

Use a plain `dict` when keys are dynamic (come from user input, JSON payloads with variable keys). Use `TypedDict` when parsing a fixed-schema JSON API response, defining a config section, or modelling CSV-parsed rows.

</details>

- [ ] Done

---

**3.** What is the key difference between `@dataclass(frozen=True)` and a `NamedTuple`?

<details><summary>Answer</summary>

Both create immutable instances, but:

| Feature | `@dataclass(frozen=True)` | `NamedTuple` |
|---|---|---|
| Base class | Normal class | Subclass of `tuple` |
| Iteration / unpacking | Not iterable by default | `x, y = point` works |
| `isinstance(obj, tuple)` | False | **True** |
| `__post_init__` | ✓ Built-in | Use `__new__` (awkward) |
| Complex methods | Natural | Possible but awkward |
| Memory | Standard object | Tuple (marginally lighter) |

Prefer `@dataclass(frozen=True)` for rich value objects with methods.  
Prefer `NamedTuple` when you need tuple semantics (unpacking, positional indexing, CSV row parsing, compatibility with code expecting sequences).

</details>

- [ ] Done

---

**4.** When should you use `NewType` instead of just a type alias?

<details><summary>Answer</summary>

Use `NewType` when two values share the same underlying type but should NOT be interchangeable — you want the checker to catch mix-ups.

```python
from typing import NewType

# Type alias — mypy treats these as identical:
TicketId = int
EventId  = int

def process(tid: TicketId) -> None: ...

event: EventId = 42
process(event)   # mypy is SILENT — alias is just a name, not a type

# NewType — mypy treats these as distinct:
TicketId = NewType("TicketId", int)
EventId  = NewType("EventId",  int)

def process(tid: TicketId) -> None: ...

event: EventId = EventId(42)
process(event)   # mypy ERROR: Expected TicketId, got EventId
```

Classic use cases: domain IDs (ticket vs. event vs. device), physical units (metres vs. seconds), security-sensitive values (raw SQL vs. sanitised SQL).

</details>

- [ ] Done

---

**5.** What does mypy's `--strict` flag catch that `--ignore-missing-imports` misses?

<details><summary>Answer</summary>

`--ignore-missing-imports` silences errors about third-party packages without type stubs (e.g., `import polars`). It suppresses exactly one class of error and adds **zero** strictness.

`--strict` bundles several checks:

- `--disallow-untyped-defs` — every function must be fully annotated
- `--disallow-any-generics` — `list` is an error; must write `list[str]`
- `--warn-return-any` — functions returning `Any` are flagged
- `--no-implicit-optional` — `def f(x: str = None)` is an error (needs `str | None`)
- `--warn-unused-ignores` — dead `# type: ignore` comments are flagged

A file with `--ignore-missing-imports` can be full of unannotated functions that mypy silently skips. The two flags are orthogonal; a codebase can (and usually should) use both.

</details>

- [ ] Done

---

## Part B — Fix the Type Errors

**1.** Add correct type annotations. The function takes a list of raw CSV lines and returns a mapping from topic name to occurrence count.

```python
def count_topics(lines, include_empty=False):
    counts = {}
    for line in lines:
        if not line.strip() and not include_empty:
            continue
        parts = line.split(",")
        topic = parts[1] if len(parts) > 1 else "unknown"
        counts[topic] = counts.get(topic, 0) + 1
    return counts
```

<details><summary>Answer</summary>

```python
def count_topics(
    lines: list[str],
    include_empty: bool = False,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in lines:
        if not line.strip() and not include_empty:
            continue
        parts = line.split(",")
        topic = parts[1] if len(parts) > 1 else "unknown"
        counts[topic] = counts.get(topic, 0) + 1
    return counts
```

Key decisions:
- `list[str]` not bare `list` — `--strict` rejects bare generics
- Return is `dict[str, int]`, not `Optional[dict[str, int]]` — the function always returns a dict (possibly empty, never `None`)
- Local `counts` annotation is optional but prevents a mypy warning about inferred `dict[Any, Any]`

</details>

- [ ] Done

---

**2.** This dataclass contains a classic mutable default bug. Identify it and fix it.

```python
from dataclasses import dataclass

@dataclass
class AnalysisRun:
    run_id: str
    findings: list[str] = []
    tag_scores: dict[str, float] = {}
```

<details><summary>Answer</summary>

```python
from dataclasses import dataclass, field

@dataclass
class AnalysisRun:
    run_id: str
    findings: list[str] = field(default_factory=list)
    tag_scores: dict[str, float] = field(default_factory=dict)
```

**Why it's a bug:** With `findings: list[str] = []`, all instances share the **same list object**. Mutating `run1.findings` also mutates `run2.findings`. Python's `@dataclass` decorator actually raises `ValueError` at class-definition time if it detects a mutable default — so this code fails before you even instantiate it.

```python
# Verify fix is correct:
r1 = AnalysisRun(run_id="a")
r2 = AnalysisRun(run_id="b")
r1.findings.append("disk full")
assert r2.findings == []   # independent — would fail without field()
```

</details>

- [ ] Done

---

**3.** This function crashes at runtime on `None` input. Add the correct guard.

```python
from typing import Optional

def normalise_label(raw: Optional[str]) -> str:
    return raw.strip().lower()
```

<details><summary>Answer</summary>

The return type depends on your contract. Two valid options:

```python
# Option A: propagate None — callers handle absence
def normalise_label(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    return raw.strip().lower()

# Option B: provide a fallback — callers always get a string
def normalise_label(raw: Optional[str], default: str = "") -> str:
    if raw is None:
        return default
    return raw.strip().lower()
```

mypy with `--strict` catches `raw.strip()` on `Optional[str]` before runtime: _"Item 'None' of 'str | None' has no attribute 'strip'"_. The annotation told us about the crash; the guard fixes it.

</details>

- [ ] Done

---

**4.** This code compiles but crashes on certain inputs. Identify the problem and fix it.

```python
from typing import TypedDict

class BaseConfig(TypedDict):
    timeout: int
    retries: int

class ExtendedConfig(BaseConfig, total=False):
    proxy_url: str

def build_connection_string(cfg: ExtendedConfig) -> str:
    return f"timeout={cfg['timeout']};proxy={cfg['proxy_url'].upper()}"
```

<details><summary>Answer</summary>

`proxy_url` is declared with `total=False` — it may not be present in the dict. Accessing `cfg['proxy_url']` raises `KeyError` when the key is absent.

```python
def build_connection_string(cfg: ExtendedConfig) -> str:
    proxy = cfg.get("proxy_url")
    proxy_part = proxy.upper() if proxy is not None else "none"
    return f"timeout={cfg['timeout']};proxy={proxy_part}"
```

mypy correctly flags `cfg["proxy_url"]` on a `total=False` key — this is one case where running mypy would have caught the bug before writing a test.

</details>

- [ ] Done

---

**5.** This Pydantic model silently accepts values that should be invalid. Add a validator that rejects negative `score` and `match_count` values, then write a test demonstrating the rejection.

```python
from pydantic import BaseModel

class ScoringResult(BaseModel):
    query: str
    score: float
    match_count: int
```

<details><summary>Answer</summary>

```python
from pydantic import BaseModel, field_validator

class ScoringResult(BaseModel):
    query: str
    score: float
    match_count: int

    @field_validator("score")
    @classmethod
    def score_non_negative(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"score must be >= 0.0, got {v}")
        return v

    @field_validator("match_count")
    @classmethod
    def count_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"match_count must be >= 0, got {v}")
        return v
```

```python
# tests/test_scoring_result.py
import pytest
from pydantic import ValidationError

def test_rejects_negative_score():
    with pytest.raises(ValidationError, match="score must be >= 0.0"):
        ScoringResult(query="test", score=-0.5, match_count=3)

def test_rejects_negative_count():
    with pytest.raises(ValidationError, match="match_count must be >= 0"):
        ScoringResult(query="test", score=0.8, match_count=-1)

def test_string_score_is_coerced():
    # Pydantic v2 coerces "0.9" -> 0.9 for float fields by design
    result = ScoringResult(query="test", score="0.9", match_count=3)
    assert result.score == pytest.approx(0.9)

def test_negative_string_score_is_rejected():
    with pytest.raises(ValidationError):
        ScoringResult(query="test", score="-0.5", match_count=3)
```

Note: Pydantic v2 coerces `"0.9"` → `0.9` for `float` fields (lax mode by default). Add `model_config = ConfigDict(strict=True)` if you want to reject strings entirely.

</details>

- [ ] Done

---

## Part C — Build Exercises

### Exercise C1: Type a Log Parser

Given this untyped function:

```python
def parse_log_line(line):
    parts = line.strip().split(",")
    return {
        "timestamp": float(parts[0]),
        "topic": parts[1],
        "value": parts[2],
    }
```

**(a)** Define a `TypedDict` named `LogEntry` and add full type annotations to `parse_log_line`.

**(b)** The function crashes on malformed lines. Change the return type to `Optional[LogEntry]` and return `None` on any parsing error.

**(c)** Write the complete type signature for a batch parser that takes `list[str]` and returns `list[LogEntry]` (skipping malformed lines — never returns `None`). Implement it.

<details><summary>Answer</summary>

**(a) TypedDict + annotations:**
```python
from typing import TypedDict

class LogEntry(TypedDict):
    timestamp: float
    topic: str
    value: str

def parse_log_line(line: str) -> LogEntry:
    parts = line.strip().split(",")
    return {
        "timestamp": float(parts[0]),
        "topic": parts[1],
        "value": parts[2],
    }
```

**(b) Optional with error handling:**
```python
def parse_log_line(line: str) -> LogEntry | None:
    try:
        parts = line.strip().split(",")
        if len(parts) < 3:
            return None
        return {
            "timestamp": float(parts[0]),
            "topic": parts[1],
            "value": parts[2],
        }
    except (ValueError, IndexError):
        return None
```

**(c) Batch parser:**
```python
def parse_log_lines(lines: list[str]) -> list[LogEntry]:
    results: list[LogEntry] = []
    for line in lines:
        entry = parse_log_line(line)
        if entry is not None:
            results.append(entry)
    return results
```

The batch parser has return type `list[LogEntry]` (not `list[LogEntry] | None`) because it always returns a list — possibly empty, never `None`. An empty list is the correct representation of "no valid entries".

</details>

- [ ] Done

---

### Exercise C2: SessionState Dataclass

Build a `SessionState` dataclass with the following spec:

- Required fields: `session_id: str`, `ticket_id: int`, `started_at: datetime`
- Optional fields: `phase: Literal["orient", "investigate", "conclude"]` (default `"orient"`), `hypotheses: list[str]` (default empty list), `confidence: float` (default `0.0`)
- `frozen=False` — phase and hypotheses change over time
- `__post_init__` that raises `ValueError` if `confidence` is outside `[0.0, 1.0]`
- `advance_phase()` method: moves through phases in order; raises `RuntimeError` if already at `"conclude"`
- `from_dict(d: dict) -> "SessionState"` classmethod: builds an instance from a plain dict (handles `started_at` as an ISO string)

<details><summary>Answer</summary>

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

_PHASES: tuple[str, ...] = ("orient", "investigate", "conclude")

@dataclass
class SessionState:
    session_id: str
    ticket_id: int
    started_at: datetime
    phase: Literal["orient", "investigate", "conclude"] = "orient"
    hypotheses: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0], got {self.confidence}"
            )

    def advance_phase(self) -> None:
        idx = _PHASES.index(self.phase)
        if idx == len(_PHASES) - 1:
            raise RuntimeError(
                f"Already at final phase '{self.phase}', cannot advance"
            )
        self.phase = _PHASES[idx + 1]  # type: ignore[assignment]

    @classmethod
    def from_dict(cls, d: dict) -> "SessionState":
        started_at = d["started_at"]
        if isinstance(started_at, str):
            started_at = datetime.fromisoformat(started_at)
        return cls(
            session_id=str(d["session_id"]),
            ticket_id=int(d["ticket_id"]),
            started_at=started_at,
            phase=d.get("phase", "orient"),
            hypotheses=list(d.get("hypotheses", [])),
            confidence=float(d.get("confidence", 0.0)),
        )
```

Usage:
```python
from datetime import datetime, timezone

state = SessionState(
    session_id="sess-001",
    ticket_id=12345,
    started_at=datetime.now(timezone.utc),
)
state.hypotheses.append("power supply fault")
state.advance_phase()
assert state.phase == "investigate"

# Round-trip via dict
d = {"session_id": "sess-002", "ticket_id": "99999", "started_at": "2026-01-15T10:00:00"}
state2 = SessionState.from_dict(d)
assert state2.ticket_id == 99999   # int coercion happened
```

The `# type: ignore[assignment]` is needed because mypy cannot verify that `_PHASES[idx + 1]` is a valid `Literal["orient", "investigate", "conclude"]` at compile time. An alternative is to use a string cast: `self.phase = cast(Literal["orient", "investigate", "conclude"], _PHASES[idx + 1])`.

</details>

- [ ] Done

---

### Exercise C3: Pydantic Settings

Write a `Settings` class using `pydantic-settings` for a hypothetical analysis tool. Then write a pytest test module that verifies correct loading using `monkeypatch`.

**Requirements:**
- Required: `api_token: str`, `api_url: str`
- Optional with defaults: `timeout_s: int = 30`, `min_confidence: float = 0.7`, `debug: bool = False`
- Validator: `api_url` must start with `"https://"`
- Reads from a `.env` file automatically

<details><summary>Answer</summary>

**settings.py:**
```python
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    api_token: str
    api_url: str
    timeout_s: int = 30
    min_confidence: float = 0.7
    debug: bool = False

    @field_validator("api_url")
    @classmethod
    def api_url_must_be_https(cls, v: str) -> str:
        if not v.startswith("https://"):
            raise ValueError(f"api_url must start with 'https://', got: {v!r}")
        return v
```

**tests/test_settings.py:**
```python
import pytest
from pydantic import ValidationError


def test_loads_required_fields(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "test-token-abc")
    monkeypatch.setenv("API_URL", "https://api.example.com")

    from settings import Settings
    s = Settings()

    assert s.api_token == "test-token-abc"
    assert s.api_url == "https://api.example.com"
    assert s.timeout_s == 30        # default
    assert s.min_confidence == 0.7  # default
    assert s.debug is False         # default


def test_overrides_defaults(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "tok")
    monkeypatch.setenv("API_URL", "https://api.example.com")
    monkeypatch.setenv("TIMEOUT_S", "60")
    monkeypatch.setenv("MIN_CONFIDENCE", "0.85")
    monkeypatch.setenv("DEBUG", "true")

    from settings import Settings
    s = Settings()

    assert s.timeout_s == 60
    assert s.min_confidence == pytest.approx(0.85)
    assert s.debug is True


def test_rejects_http_url(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "tok")
    monkeypatch.setenv("API_URL", "http://insecure.example.com")

    from settings import Settings
    with pytest.raises(ValidationError, match="must start with 'https://'"):
        Settings()


def test_requires_api_token(monkeypatch):
    monkeypatch.delenv("API_TOKEN", raising=False)
    monkeypatch.setenv("API_URL", "https://api.example.com")

    from settings import Settings
    with pytest.raises(ValidationError):
        Settings()
```

**Important:** Instantiate `Settings()` inside each test function, not at module level. If you instantiate at import time, `monkeypatch` has not yet applied and the test reads the real environment.

</details>

- [ ] Done
