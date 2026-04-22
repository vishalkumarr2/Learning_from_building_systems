# 01 — Type Annotations, Dataclasses, and Pydantic
### Stop chasing runtime AttributeErrors — catch them before the script runs
**Prerequisite:** Python functions, dicts, classes. You've written scripts with argparse before.
**Unlocks:** mypy static analysis, validated config loading, self-documenting function signatures, Pydantic models

---

## Why Should I Care? (Context)

Analysis scripts that process large log files or call external APIs contain dozens of implicit assumptions:
- "This function always returns a list, never `None`"
- "The config dict definitely has a `confidence` key"
- "Ticket ID is an int, robot ID is a string — don't mix them up"

Without type annotations, every one of those assumptions is invisible until the script crashes at 2 AM on real data. With type annotations + mypy:
- A mismatched function argument is a **red underline in the editor**, not a runtime traceback
- Pydantic turns your `.env` file into a validated, typed Python object — missing `API_TOKEN`? Crash at startup with a clear error, not 40 lines into processing
- `NewType` makes `ticket_id` and `robot_id` incompatible types — the compiler prevents passing one where the other is expected

This chapter is a force multiplier for every script in this repo.

---

# PART 1 — PYTHON TYPE ANNOTATIONS BASICS

---

## 1.1 The Syntax

Type annotations are hints — they don't change runtime behavior, but tools like mypy and your editor use them to catch errors before the code runs.

```python
# Variable annotations
count: int = 0
name: str = "default"
ratio: float = 0.95
active: bool = True
result: None = None          # rarely useful as a variable annotation

# Function annotations
def greet(name: str) -> str:
    return f"Hello, {name}"

def process_log(path: str, verbose: bool = False) -> int:
    """Returns number of lines processed."""
    ...
    return 0

def log_event(message: str) -> None:   # -> None = does not return a value
    print(message)
```

**Key insight:** `-> None` is not the same as omitting the return type. Omitting means "I haven't annotated this yet." `-> None` says "I explicitly guarantee this returns nothing."

---

## 1.2 Collection Types (Python 3.9+)

Python 3.9 added built-in generic types — you no longer need to import from `typing`:

```python
# Python 3.9+ (preferred)
scores: list[float] = [0.8, 0.9, 0.5]
lookup: dict[str, int] = {"alpha": 1, "beta": 2}
coords: tuple[float, float] = (1.0, 2.0)
ids: set[int] = {1, 2, 3}

# Python 3.8 and earlier (still works, just verbose)
from typing import List, Dict, Tuple, Set
scores: List[float] = []
lookup: Dict[str, int] = {}
```

Fixed-length tuples are annotated with each element's type:

```python
Point3D = tuple[float, float, float]        # exactly 3 floats
LogLine = tuple[float, str, str]            # (timestamp, level, message)

def parse_log_line(raw: str) -> LogLine:
    parts = raw.split("|", 2)
    return float(parts[0]), parts[1], parts[2]
```

Variable-length tuples of uniform type use `...`:

```python
Numbers = tuple[int, ...]    # any number of ints
```

---

## 1.3 Optional and Union

```python
from typing import Optional, Union

# Optional[X] means "X or None" — equivalent to X | None (Python 3.10+)
def find_session(session_id: str) -> Optional[dict]:
    ...

# Python 3.10+ syntax (cleaner)
def find_session(session_id: str) -> dict | None:
    ...

# Union: accept multiple types
def format_id(value: Union[int, str]) -> str:
    return str(value)

# Python 3.10+
def format_id(value: int | str) -> str:
    return str(value)
```

**Common mistake:** returning `None` from a function annotated `-> str` is a type error. Always annotate with `str | None` if None is possible.

```python
# BAD — mypy will flag this
def get_error_code(log: str) -> str:
    if "ERROR" in log:
        return log.split("ERROR:")[1].strip()
    return None  # Type error: None is not str

# GOOD
def get_error_code(log: str) -> str | None:
    if "ERROR" in log:
        return log.split("ERROR:")[1].strip()
    return None
```

---

## 1.4 Worked Example: Annotating a Log Parser

```python
from pathlib import Path


def parse_incident_log(
    path: Path,
    error_only: bool = False,
    max_lines: int | None = None,
) -> list[tuple[float, str, str]]:
    """
    Parse a structured log file into (timestamp, level, message) tuples.

    Args:
        path: Path to the log file.
        error_only: If True, only return ERROR-level entries.
        max_lines: Stop after this many lines (None = read all).

    Returns:
        List of (timestamp_seconds, level, message) tuples.
    """
    entries: list[tuple[float, str, str]] = []

    with path.open() as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break

            line = line.strip()
            if not line:
                continue

            parts = line.split("|", 2)
            if len(parts) != 3:
                continue

            ts_str, level, message = parts
            if error_only and level.strip() != "ERROR":
                continue

            entries.append((float(ts_str), level.strip(), message.strip()))

    return entries


# The return type makes it clear: callers know they get a list of 3-tuples.
# Without the annotation, they'd have to read the whole function to know.
results: list[tuple[float, str, str]] = parse_incident_log(
    Path("session.log"),
    error_only=True,
    max_lines=1000,
)

for ts, level, msg in results:
    print(f"[{ts:.3f}] {level}: {msg}")
```

---

# PART 2 — ADVANCED TYPES

---

## 2.1 TypedDict: Typed Dictionary Keys

Plain `dict[str, Any]` loses all type information about what keys exist. `TypedDict` fixes this:

```python
from typing import TypedDict, Any


class IncidentSummary(TypedDict):
    ticket_id: int
    robot_id: str
    error_code: str
    duration_s: float


class SearchResult(TypedDict):
    query: str
    matches: list[dict[str, Any]]
    confidence: float
    source: str


def summarise_incident(raw: dict) -> IncidentSummary:
    return IncidentSummary(
        ticket_id=int(raw["id"]),
        robot_id=str(raw["robot"]),
        error_code=str(raw.get("code", "UNKNOWN")),
        duration_s=float(raw.get("duration", 0.0)),
    )


# Now the editor knows summary["ticket_id"] is an int
summary = summarise_incident({"id": "123", "robot": "bot-01", "code": "NAV_ERR"})
print(summary["ticket_id"] + 1)   # OK — int + int
print(summary["robot_id"] + 1)   # mypy ERROR: str + int
```

`TypedDict` also supports optional keys via `total=False`:

```python
class PartialResult(TypedDict, total=False):
    confidence: float    # this key may or may not be present
    source: str


class RequiredResult(TypedDict):
    query: str


class FullResult(RequiredResult, PartialResult):
    pass   # query is required, confidence/source are optional
```

---

## 2.2 Protocol: Structural Subtyping

A `Protocol` defines an interface by behaviour, not inheritance. Any class that has the right methods satisfies the protocol — no need to explicitly inherit.

```python
from typing import Protocol


class Searchable(Protocol):
    def search(self, query: str) -> list[str]: ...


class Serialisable(Protocol):
    def to_dict(self) -> dict: ...
    def from_dict(cls, data: dict) -> "Serialisable": ...


# Any class with a .search() method satisfies Searchable
class KbSearcher:
    def search(self, query: str) -> list[str]:
        return []    # real impl goes here


class MemorySearcher:
    def search(self, query: str) -> list[str]:
        return []    # different backend, same interface


def run_search(backend: Searchable, query: str) -> list[str]:
    return backend.search(query)


# Both work — no inheritance required
run_search(KbSearcher(), "motor stall")
run_search(MemorySearcher(), "slip event")
```

**Key insight:** `Protocol` lets you write generic functions that work with any object satisfying a contract. This is especially useful for dependency injection in tests — replace the real searcher with a fake one.

---

## 2.3 Callable, Literal, Final, ClassVar

```python
from typing import Callable, Literal, Final, ClassVar


# Callable[[ArgTypes...], ReturnType]
Scorer = Callable[[str, list[str]], float]

def apply_scorer(scorer: Scorer, query: str, candidates: list[str]) -> float:
    return scorer(query, candidates)


# Literal: restrict to specific values
LogLevel = Literal["DEBUG", "INFO", "WARN", "ERROR"]

def log(level: LogLevel, msg: str) -> None:
    print(f"[{level}] {msg}")

log("INFO", "started")       # OK
log("VERBOSE", "too much")   # mypy ERROR: "VERBOSE" not in Literal


# Final: a constant that cannot be reassigned
MAX_RETRIES: Final[int] = 3
KB_VERSION: Final[str] = "2.0"

MAX_RETRIES = 4   # mypy ERROR: Cannot assign to final name


# ClassVar: a class-level variable (not per-instance)
class SearchConfig:
    DEFAULT_LIMIT: ClassVar[int] = 20
    MIN_CONFIDENCE: ClassVar[float] = 0.5

    def __init__(self, limit: int = SearchConfig.DEFAULT_LIMIT) -> None:
        self.limit = limit
```

---

## 2.4 Type Aliases and NewType

```python
from typing import NewType


# Type alias: just a shorthand name
LogLine = tuple[float, str, str]
ScoreMap = dict[str, float]
Findings = list[dict[str, str]]

def score_results(lines: list[LogLine]) -> ScoreMap:
    ...


# NewType: create a *distinct* type that cannot be confused with its base
TicketId = NewType("TicketId", int)
RobotId = NewType("RobotId", str)

def fetch_ticket(ticket_id: TicketId) -> dict:
    ...

def fetch_robot_logs(robot_id: RobotId) -> list[str]:
    ...


ticket = TicketId(12345)
robot = RobotId("bot-07")

fetch_ticket(ticket)          # OK
fetch_ticket(robot)           # mypy ERROR: RobotId is not TicketId
fetch_ticket(12345)           # mypy ERROR: plain int is not TicketId (strict mode)

# Create NewType values explicitly
my_ticket = TicketId(67890)
my_robot = RobotId("bot-12")
```

**Key insight:** `NewType` costs nothing at runtime (it's the identity function) but prevents entire classes of mix-up bugs. Use it whenever two values have the same underlying type but different meanings — IDs are the classic case.

---

# PART 3 — @DATACLASS

---

## 3.1 Basics

`@dataclass` auto-generates `__init__`, `__repr__`, and `__eq__` from your field annotations.

```python
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Hypothesis:
    description: str
    confidence: float
    evidence: list[str]

# Generated __init__:
# def __init__(self, description: str, confidence: float, evidence: list[str])

h = Hypothesis("slip due to surface change", 0.8, ["velocity drop at t=12.3"])
print(h)
# Hypothesis(description='slip due to surface change', confidence=0.8,
#            evidence=['velocity drop at t=12.3'])
print(h == Hypothesis("slip due to surface change", 0.8, ["velocity drop at t=12.3"]))
# True — __eq__ compares all fields
```

---

## 3.2 field() — Defaults and Mutable Defaults

```python
from dataclasses import dataclass, field


@dataclass
class SearchRequest:
    query: str
    limit: int = 20                         # simple default
    sources: list[str] = field(            # mutable default MUST use field()
        default_factory=lambda: ["kb", "memory"]
    )
    metadata: dict[str, str] = field(
        default_factory=dict
    )
    created_at: datetime = field(
        default_factory=datetime.utcnow
    )

# BAD — will raise ValueError at class definition time:
# @dataclass
# class Bad:
#     items: list[str] = []   # Python: mutable default is dangerous

r1 = SearchRequest("motor stall")
r2 = SearchRequest("slip event")
r1.sources.append("ado")

print(r1.sources)  # ["kb", "memory", "ado"]
print(r2.sources)  # ["kb", "memory"] — r2 has its own list
```

---

## 3.3 frozen=True: Immutable Dataclasses

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class EventKey:
    """Immutable key identifying a specific event in a log."""
    session_id: str
    timestamp: float
    event_type: str

    def before(self, other: "EventKey") -> bool:
        return self.timestamp < other.timestamp


key = EventKey("sess-001", 1714300000.0, "NAV_ERR")
key.timestamp = 0.0   # raises FrozenInstanceError

# frozen dataclasses are hashable — can be used in sets and dict keys
seen: set[EventKey] = set()
seen.add(key)
cache: dict[EventKey, str] = {key: "delocalized"}
```

---

## 3.4 __post_init__: Computed Fields

```python
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class IncidentWindow:
    start_ts: float
    end_ts: float
    _duration_s: float = field(init=False)   # computed, not a constructor arg

    def __post_init__(self) -> None:
        if self.end_ts < self.start_ts:
            raise ValueError(
                f"end_ts ({self.end_ts}) must be >= start_ts ({self.start_ts})"
            )
        object.__setattr__(self, "_duration_s", self.end_ts - self.start_ts)

    @property
    def duration_s(self) -> float:
        return self._duration_s

    @property
    def duration_min(self) -> float:
        return self._duration_s / 60.0


window = IncidentWindow(start_ts=1714300000.0, end_ts=1714300300.0)
print(window.duration_min)   # 5.0
```

---

## 3.5 @dataclass vs NamedTuple

```python
from typing import NamedTuple


# NamedTuple: immutable, supports positional access and unpacking
class LogEntry(NamedTuple):
    timestamp: float
    level: str
    message: str

entry = LogEntry(1714300000.0, "ERROR", "localization failed")
ts, level, msg = entry      # unpacking works
print(entry[0])             # positional access works
print(entry.timestamp)      # named access works

# dataclass: mutable by default, better for complex entities
@dataclass
class SessionState:
    session_id: str
    phase: str = "orient"
    hypotheses: list[str] = field(default_factory=list)
    # ... many more fields

state = SessionState("sess-001")
state.phase = "analyse"    # mutation is natural
state.hypotheses.append("slip event at t=12.3")
```

| | NamedTuple | @dataclass |
|---|---|---|
| Immutable | Yes (always) | Only with `frozen=True` |
| Positional access | Yes (`x[0]`) | No |
| Unpacking | Yes | No |
| Default values | Yes | Yes |
| Mutable defaults | No (workaround needed) | Yes via `field()` |
| Inheritance | Limited | Full |
| Best for | Small value objects (points, keys) | Entities with many fields |

---

## 3.6 Worked Example: SessionState

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


Phase = Literal["orient", "scope", "analyse", "hypothesise", "close"]


@dataclass
class Finding:
    description: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


@dataclass
class SessionState:
    """Tracks progress of a root-cause analysis session."""
    session_id: str
    ticket_id: int
    title: str
    phase: Phase = "orient"
    hypotheses: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    closed_at: datetime | None = None
    notes: dict[str, str] = field(default_factory=dict)

    def advance_phase(self, next_phase: Phase) -> None:
        self.phase = next_phase

    def add_hypothesis(self, h: str) -> None:
        if h not in self.hypotheses:
            self.hypotheses.append(h)

    def add_finding(self, description: str, confidence: float,
                   evidence: list[str] | None = None) -> Finding:
        f = Finding(
            description=description,
            confidence=confidence,
            evidence=evidence or [],
        )
        self.findings.append(f)
        return f

    def close(self) -> None:
        self.phase = "close"
        self.closed_at = datetime.utcnow()

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "ticket_id": self.ticket_id,
            "phase": self.phase,
            "hypotheses": self.hypotheses,
            "findings": [
                {
                    "description": f.description,
                    "confidence": f.confidence,
                    "evidence": f.evidence,
                }
                for f in self.findings
            ],
        }


# Usage
state = SessionState(session_id="sess-042", ticket_id=99999, title="robot stopped mid-run")
state.advance_phase("scope")
state.add_hypothesis("localization diverged due to featureless corridor")
f = state.add_finding(
    "covariance xx exceeded threshold at t=12.3s",
    confidence=0.87,
    evidence=["cov_xx=0.018 > limit=0.010", "vel dropped from 0.8 to 0.0 in 200ms"],
)
```

---

# PART 4 — PYDANTIC BASEMODEL

---

## 4.1 Why Pydantic?

`@dataclass` is great for pure Python objects. Pydantic goes further:
- **Runtime validation**: invalid types raise `ValidationError`, not silent corruption
- **Type coercion**: `"42"` → `42`, `"true"` → `True`
- **Serialisation**: `.model_dump()` → dict, `.model_dump_json()` → JSON string
- **Parsing**: `.model_validate(raw_dict)` validates a dict from an API or file

```python
from pydantic import BaseModel, Field
from typing import Any, Literal


class SearchResult(BaseModel):
    query: str
    matches: list[dict[str, Any]]
    confidence: float = Field(ge=0.0, le=1.0, description="0–1 match confidence")
    source: Literal["kb", "ado", "memory"] = "kb"


# Pydantic coerces types and validates constraints
result = SearchResult(
    query="motor stall",
    matches=[{"title": "motor fault pattern", "score": 0.91}],
    confidence=0.91,
)

print(result.confidence)            # 0.91
print(result.model_dump())          # dict with all fields
print(result.model_dump_json())     # JSON string

# Validation failure — clear error, not a silent bug
try:
    bad = SearchResult(query="test", matches=[], confidence=1.5)
except Exception as e:
    print(e)
    # confidence: Input should be less than or equal to 1
```

---

## 4.2 Field() — Constraints, Defaults, Aliases

```python
from pydantic import BaseModel, Field
import re


class IncidentReport(BaseModel):
    ticket_id: int = Field(gt=0, description="Positive ticket identifier")
    title: str = Field(min_length=5, max_length=200)
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    tags: list[str] = Field(default_factory=list, max_length=20)
    summary: str | None = Field(default=None, alias="exec_summary")

    # alias: when the source dict uses a different key name
    # model_validate({"exec_summary": "..."}) → summary = "..."

    model_config = {"populate_by_name": True}   # allow both alias and field name


report = IncidentReport(
    ticket_id=99999,
    title="Robot stopped at turn",
    confidence=0.75,
    exec_summary="Slip event at waypoint 12",
)
print(report.summary)  # "Slip event at waypoint 12"
```

---

## 4.3 Validators

Pydantic v2 uses `@field_validator` for per-field validation:

```python
from pydantic import BaseModel, Field, field_validator


class KbSearchRequest(BaseModel):
    query: str
    limit: int = Field(default=10, ge=1, le=100)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be empty after stripping whitespace")
        if len(v) < 3:
            raise ValueError(f"query too short ({len(v)} chars); minimum 3")
        return v

    @field_validator("limit")
    @classmethod
    def limit_reasonable(cls, v: int) -> int:
        if v > 50:
            import warnings
            warnings.warn(f"limit={v} is large; consider reducing for performance")
        return v
```

---

## 4.4 model_validate(), model_dump(), Nested Models

```python
from pydantic import BaseModel
from typing import Any


class Evidence(BaseModel):
    description: str
    timestamp_s: float | None = None
    data: dict[str, Any] = {}


class RcaFinding(BaseModel):
    root_cause: str
    confidence: float
    evidence: list[Evidence] = []
    recommended_fix: str | None = None


# Parse from dict (e.g., loaded from a JSON file)
raw = {
    "root_cause": "localization diverged due to reflective floor",
    "confidence": 0.88,
    "evidence": [
        {"description": "cov_xx spike", "timestamp_s": 1714300012.3},
        {"description": "velocity drop to zero"},
    ],
}
finding = RcaFinding.model_validate(raw)
print(finding.evidence[0].timestamp_s)  # 1714300012.3

# Serialize back to dict
d = finding.model_dump()
print(d)

# Serialize to JSON
json_str = finding.model_dump_json(indent=2)

# Auto-generate JSON schema
schema = RcaFinding.model_json_schema()
print(schema)
```

---

# PART 5 — PYDANTIC SETTINGS (CONFIG FROM .env)

---

## 5.1 BaseSettings

`pydantic-settings` extends Pydantic to read values from environment variables and `.env` files automatically:

```bash
pip install pydantic-settings
```

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # Required — will raise ValidationError if not set in env or .env
    api_token: str

    # Optional — have defaults
    chat_token: str | None = None
    grafana_url: str = "http://localhost:3000"
    kb_min_confidence: float = 0.7
    max_search_results: int = 20
    debug: bool = False

    # Field with alias: env var ADO_ORG maps to .ado_org
    ado_org: str = Field(default="my-org", alias="ADO_ORG")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,      # API_TOKEN and api_token both work
        extra="ignore",            # ignore unknown env vars
    )


# Usage — typically at module level or in main()
def get_settings() -> Settings:
    """Singleton-style settings loader."""
    return Settings()   # raises ValidationError if API_TOKEN is missing
```

---

## 5.2 How it reads values

Priority order (highest wins):
1. Direct constructor argument: `Settings(api_token="override")`
2. Environment variable: `export API_TOKEN=abc123`
3. `.env` file: `API_TOKEN=abc123`
4. Field default

```python
# .env file
# API_TOKEN=abc123
# CHAT_TOKEN=xoxb-...
# KB_MIN_CONFIDENCE=0.8
# DEBUG=true            ← "true" is coerced to True

settings = Settings()
print(settings.api_token)              # "abc123"
print(settings.kb_min_confidence)    # 0.8 (float, not string)
print(settings.debug)                # True (bool, not string)
```

---

## 5.3 Worked Example: Settings for an Analysis Tool

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from pathlib import Path
import warnings


class AnalysisSettings(BaseSettings):
    """Settings for the incident analysis toolkit."""

    # Auth tokens
    api_token: str = Field(description="Azure DevOps Personal Access Token")
    chat_token: str | None = Field(default=None)

    # Service URLs
    grafana_url: str = "http://localhost:3000"
    monitoring_token: str | None = None
    ado_org: str = "my-org"
    ado_project: str = "my-project"

    # Search / KB settings
    kb_min_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    max_search_results: int = Field(default=20, ge=1, le=200)

    # Paths
    attachments_dir: Path = Path("attachments")
    kb_dir: Path = Path("scripts/kb")

    # Behaviour
    dry_run: bool = False
    verbose: bool = False

    @field_validator("grafana_url", "ado_org")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()

    @field_validator("attachments_dir", "kb_dir")
    @classmethod
    def dir_exists_warning(cls, v: Path) -> Path:
        if not v.exists():
            warnings.warn(f"Directory {v} does not exist yet")
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Fail fast at startup — better than failing 40 lines into processing
def load_settings() -> AnalysisSettings:
    try:
        return AnalysisSettings()
    except Exception as e:
        raise SystemExit(f"Configuration error: {e}") from e


# In main():
# settings = load_settings()
# print(f"KB confidence threshold: {settings.kb_min_confidence}")
# print(f"ADO org: {settings.ado_org}")
```

---

## 5.4 Testing Settings (Preview)

```python
import os
import pytest
from unittest.mock import patch


def test_settings_loads_from_env(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "test-token-123")
    monkeypatch.setenv("KB_MIN_CONFIDENCE", "0.9")

    settings = AnalysisSettings()
    assert settings.api_token == "test-token-123"
    assert settings.kb_min_confidence == 0.9


def test_settings_fails_without_api_token(monkeypatch):
    monkeypatch.delenv("API_TOKEN", raising=False)
    with pytest.raises(Exception):   # ValidationError
        AnalysisSettings()
```

---

# PART 6 — mypy / pyright BASICS

---

## 6.1 Running mypy

```bash
# Install
pip install mypy

# Check a single file
mypy scripts/knowledge_search.py

# Strict mode — catches more (recommended for new files)
mypy --strict scripts/knowledge_search.py

# Check all scripts
mypy scripts/ --ignore-missing-imports

# With config file (mypy.ini or pyproject.toml)
mypy .
```

Example output:
```
scripts/knowledge_search.py:42: error: Argument 1 to "fetch_ticket" has incompatible
    type "str"; expected "TicketId"  [arg-type]
scripts/knowledge_search.py:67: error: Item "None" of "str | None" has no attribute
    "split"  [union-attr]
```

---

## 6.2 Common Errors and How to Fix Them

```python
# Error: Item "None" of "str | None" has no attribute "split"
# Fix: guard against None before using the value
def process(value: str | None) -> list[str]:
    if value is None:
        return []
    return value.split(",")    # now mypy knows value is str


# Error: Argument 1 to "int" has incompatible type "str | int"
# Fix: cast or check type first
from typing import cast

def ensure_int(v: str | int) -> int:
    if isinstance(v, int):
        return v
    return int(v)


# Error: Function is missing a return type annotation
# Fix: add -> ReturnType
def compute(x: float) -> float:
    return x * 2.0


# Error: Need type annotation for ... (hint: use ...)
# Fix: annotate the variable
results: list[str] = []
lookup: dict[str, int] = {}
```

---

## 6.3 cast() and type: ignore

```python
from typing import cast
import json

# cast(): tell mypy "trust me, this is X"
# Use sparingly — it's a lie that can hide real bugs
raw = json.loads('{"id": 1}')
# raw has type dict[str, Any]
ticket_id = cast(int, raw["id"])   # mypy now treats ticket_id as int


# type: ignore — suppress a specific error
# Include the error code so it's clear what you're ignoring
result = some_third_party_function()  # type: ignore[no-any-return]


# Better: use a proper annotation on the third-party call
from typing import Any
result2: Any = some_third_party_function()
```

---

## 6.4 mypy.ini / pyproject.toml Configuration

```ini
# mypy.ini
[mypy]
python_version = 3.11
warn_return_any = True
warn_unused_ignores = True
disallow_untyped_defs = True
ignore_missing_imports = True

[mypy-boto3.*]
ignore_missing_imports = True
```

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
ignore_missing_imports = true
```

---

## 6.5 Pre-commit Integration

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.8.0
    hooks:
      - id: mypy
        args: [--strict, --ignore-missing-imports]
        additional_dependencies:
          - pydantic
          - pydantic-settings
```

---

## Summary — What to Remember

| Concept | Rule |
|---------|------|
| `Optional[X]` / `X \| None` | Use whenever a function can return None or a parameter is optional |
| `TypedDict` | Use instead of `dict[str, Any]` when you know the exact keys |
| `Protocol` | Define interfaces without inheritance; great for testable dependency injection |
| `NewType` | Create distinct ID types (`TicketId`, `RobotId`) to prevent mix-ups |
| `@dataclass` | Auto-generates `__init__`/`__repr__`/`__eq__`; use `field(default_factory=...)` for mutable defaults |
| `frozen=True` | Makes dataclass immutable and hashable; use for dict keys and set members |
| `BaseModel` (Pydantic) | Runtime validation + coercion + serialisation; use for API responses and parsed data |
| `BaseSettings` | Reads from env/`.env`, coerces types, fails fast at startup if required vars missing |
| `Literal[...]` | Restricts a string param to specific allowed values |
| `cast()` / `# type: ignore` | Use sparingly; always include the error code in ignore comments |

---

# QUICK REFERENCE CARD

```
┌──────────────────────────────── TYPE ANNOTATIONS CHEAT SHEET ───────────────────────────────┐
│                                                                                              │
│  BASIC:         x: int   y: str   z: float   b: bool   n: None                             │
│  COLLECTIONS:   list[int]   dict[str, float]   tuple[int, str]   set[str]                   │
│  OPTIONAL:      str | None    (or Optional[str] for 3.8)                                    │
│  UNION:         int | str     (or Union[int, str] for 3.8)                                  │
│                                                                                              │
│  FUNCTIONS:     def f(x: int, y: str = "a") -> float: ...                                  │
│                 def g() -> None: ...                                                         │
│                                                                                              │
│  ADVANCED:      TypedDict  — typed dict with known keys                                     │
│                 Protocol   — structural interface (duck typing)                              │
│                 NewType    — distinct type alias (zero runtime cost)                         │
│                 Literal    — value-restricted string/int                                     │
│                 Final      — constant, cannot be reassigned                                  │
│                 Callable[[A,B], R]  — function type                                          │
│                                                                                              │
│  DATACLASS:     @dataclass → auto __init__/__repr__/__eq__                                  │
│                 field(default_factory=list)  — mutable default                               │
│                 frozen=True  → immutable, hashable                                           │
│                 __post_init__  → computed fields after init                                  │
│                                                                                              │
│  PYDANTIC:      BaseModel → validated, serialisable model                                   │
│                 Field(ge=0, le=1)  → constraints                                             │
│                 .model_validate(dict) → parse from dict                                     │
│                 .model_dump()  → serialize to dict                                           │
│                 @field_validator → custom validation logic                                   │
│                                                                                              │
│  SETTINGS:      BaseSettings → reads .env + env vars automatically                          │
│                 SettingsConfigDict(env_file=".env")  → config                               │
│                 Settings() raises ValidationError if required field missing                  │
│                                                                                              │
│  mypy:          mypy --strict file.py                                                        │
│                 cast(Type, value)  — assert type (use sparingly)                             │
│                 # type: ignore[error-code]  — suppress specific error                        │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```
