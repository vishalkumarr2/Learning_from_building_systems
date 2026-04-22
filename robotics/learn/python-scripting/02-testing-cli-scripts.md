# 02 — Testing CLI Scripts with pytest
### "I'll add tests later" is how a buggy scorer silently mis-routes incidents for months
**Prerequisite:** Python functions, basic argparse scripts, `pip install pytest pytest-mock`
**Unlocks:** Reliable refactoring, confident changes to shared scripts, monkeypatching env vars and HTTP calls

---

## Why Should I Care? (Context)

Analysis scripts that run against real data and take action — posting comments, tagging squads, updating knowledge bases — have bugs with delayed consequences. A scoring function that ranks the wrong pattern as most similar will silently produce misleading diagnoses. You won't notice until someone asks why the same root cause keeps being misidentified.

pytest makes testing cheap enough that "I'll test this later" stops being an excuse:
- A fixture that creates a temp `.env` file takes 5 lines
- Monkeypatching an HTTP call takes 1 line
- Running your test suite takes 3 seconds

The investment: 20 minutes writing tests. The payoff: never wondering if your script is quietly broken.

---

# PART 1 — PYTEST BASICS

---

## 1.1 Test Discovery

pytest finds tests automatically based on naming conventions:

```
tests/
    conftest.py           # shared fixtures — auto-discovered by pytest
    test_knowledge_search.py     # any file matching test_*.py or *_test.py
    unit/
        test_scorer.py
    integration/
        test_session.py
```

Function naming rules:
```python
# pytest discovers these
def test_scorer_returns_float():
    ...

def test_scorer_handles_empty_query():
    ...

# pytest ignores these (wrong prefix)
def scorer_test():
    ...

def check_scorer():
    ...
```

---

## 1.2 Running pytest

```bash
# Run all tests
pytest

# Verbose output (shows each test name)
pytest -v

# Run a specific file
pytest tests/unit/test_scorer.py

# Run a specific test by name substring
pytest -k "test_score"
pytest -k "test_score and not empty"

# Stop after first failure
pytest -x

# Show local variable values on failure
pytest -l

# Run with coverage
pytest --cov=scripts --cov-report=term-missing

# Run and show slowest 5 tests
pytest --durations=5
```

---

## 1.3 Writing Assertions

pytest uses plain `assert` — it intercepts assertion failures and shows detailed diffs:

```python
def test_score_high_match():
    result = compute_score("motor stall", ["motor stall pattern"])
    assert result > 0.8


def test_parse_returns_list():
    lines = parse_log_lines("1.0|ERROR|fault\n2.0|INFO|ok\n")
    assert lines == [(1.0, "ERROR", "fault"), (2.0, "INFO", "ok")]
    # On failure, pytest shows:
    # AssertionError: assert [(1.0, 'WARN', ...)] == [(1.0, 'ERROR', ...)]
    # At index 0 diff: (1.0, 'WARN', 'fault') != (1.0, 'ERROR', 'fault')


def test_score_range():
    score = compute_score("test", ["test"])
    assert 0.0 <= score <= 1.0, f"Score {score} out of range [0, 1]"
```

---

## 1.4 pytest.raises: Testing Exceptions

```python
import pytest


def test_empty_query_raises():
    with pytest.raises(ValueError):
        compute_score("", ["pattern"])


def test_empty_query_message():
    with pytest.raises(ValueError, match="query must not be empty"):
        compute_score("", ["pattern"])


def test_invalid_confidence_raises():
    with pytest.raises(Exception) as exc_info:
        validate_confidence(1.5)
    assert "1.0" in str(exc_info.value)   # check the message
```

---

## 1.5 Parametrize: Multiple Inputs in One Test

```python
import pytest


@pytest.mark.parametrize("query,candidates,expected_min_score", [
    ("motor stall", ["motor stall pattern"], 0.8),
    ("slip event", ["wheel slip on wet surface"], 0.5),
    ("completely unrelated", ["motor stall pattern"], 0.0),
    ("", ["motor stall pattern"], 0.0),
])
def test_score_ranges(query, candidates, expected_min_score):
    score = compute_score(query, candidates)
    assert score >= expected_min_score


@pytest.mark.parametrize("level,expected_count", [
    ("ERROR", 3),
    ("INFO", 5),
    ("WARN", 1),
    ("DEBUG", 0),
])
def test_filter_by_level(level, expected_count, sample_log_lines):
    filtered = filter_by_level(sample_log_lines, level)
    assert len(filtered) == expected_count
```

Each parameter combination runs as a separate test case — you see them individually in `pytest -v` output.

---

## 1.6 Skip and XFail

```python
import pytest
import sys


@pytest.mark.skip(reason="ADO API not available in CI")
def test_fetch_real_ticket():
    ...


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="relies on /tmp — not portable to Windows"
)
def test_temp_file_handling():
    ...


@pytest.mark.xfail(reason="known bug in score normalisation, tracked in backlog")
def test_score_with_unicode():
    result = compute_score("résumé", ["resume pattern"])
    assert result > 0.5
```

`xfail` (expected failure) is different from `skip`: the test still runs. If it passes, it's marked `XPASS` — a signal that the bug may be fixed.

---

# PART 2 — FIXTURES

---

## 2.1 @pytest.fixture Basics

A fixture is a function that provides a value (or setup/teardown) to tests that declare it as a parameter:

```python
import pytest
from pathlib import Path


@pytest.fixture
def sample_log_lines():
    """A small representative set of log lines for testing."""
    return [
        (1714300000.0, "INFO", "session started"),
        (1714300005.0, "WARN", "velocity below threshold"),
        (1714300010.0, "ERROR", "localization failed"),
        (1714300012.0, "ERROR", "emergency stop triggered"),
        (1714300015.0, "INFO", "session closed"),
    ]


@pytest.fixture
def empty_log():
    return []


# Any test can request these fixtures by name
def test_filter_errors(sample_log_lines):
    errors = [line for line in sample_log_lines if line[1] == "ERROR"]
    assert len(errors) == 2


def test_empty_log_returns_empty(empty_log):
    result = filter_by_level(empty_log, "ERROR")
    assert result == []
```

---

## 2.2 Fixture Scope

```python
@pytest.fixture(scope="function")   # default: new instance per test
def db_connection():
    conn = create_test_db()
    yield conn
    conn.close()


@pytest.fixture(scope="module")     # once per test file
def loaded_kb():
    """Loading the KB is slow; reuse across tests in the file."""
    return load_knowledge_base("tests/fixtures/kb_sample.json")


@pytest.fixture(scope="session")    # once per entire test run
def settings():
    """Settings object shared across all tests."""
    import os
    os.environ["API_TOKEN"] = "test-token"
    return load_settings()
```

---

## 2.3 Yield Fixtures: Setup and Teardown

```python
import pytest
import json
from pathlib import Path


@pytest.fixture
def temp_session_file(tmp_path):
    """Create a session JSON file that tests can read/modify."""
    session_data = {
        "session_id": "test-001",
        "ticket_id": 99999,
        "phase": "orient",
        "hypotheses": [],
    }
    session_file = tmp_path / "session.json"
    session_file.write_text(json.dumps(session_data))

    yield session_file   # tests receive this path

    # Teardown: nothing needed here — tmp_path is auto-cleaned
    # But you could close DB connections, delete external resources, etc.


@pytest.fixture
def running_server():
    """Start a mock server, yield its URL, stop it after the test."""
    server = MockServer()
    server.start()
    yield server.url
    server.stop()   # always runs, even if test fails
```

---

## 2.4 Fixtures Using Other Fixtures

```python
@pytest.fixture
def base_settings(tmp_path):
    """A minimal .env file for testing."""
    env_file = tmp_path / ".env"
    env_file.write_text("API_TOKEN=test-token-123\nKB_MIN_CONFIDENCE=0.8\n")
    return env_file


@pytest.fixture
def settings_obj(base_settings):
    """Load Settings from the test .env file."""
    from scripts.settings import Settings
    return Settings(_env_file=str(base_settings))


# settings_obj automatically gets base_settings
def test_confidence_loaded(settings_obj):
    assert settings_obj.kb_min_confidence == 0.8
```

---

## 2.5 conftest.py: Shared Fixtures

pytest auto-imports `conftest.py` from the test directory and all parent directories. Put shared fixtures there:

```
tests/
    conftest.py           ← fixtures available to ALL tests
    unit/
        conftest.py       ← fixtures available to unit tests only
        test_scorer.py
    integration/
        conftest.py       ← fixtures available to integration tests only
        test_session.py
```

```python
# tests/conftest.py
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).parent.parent


@pytest.fixture
def sample_incident():
    return {
        "ticket_id": 99999,
        "title": "Robot stopped mid-run",
        "error_code": "NAV_ERR_001",
        "duration_s": 45.2,
    }
```

---

# PART 3 — tmp_path AND FILE-BASED TESTS

---

## 3.1 tmp_path Built-in Fixture

`tmp_path` is a `pathlib.Path` pointing to a unique temporary directory managed by pytest. It's cleaned up after each test.

```python
from pathlib import Path
import json


def test_save_and_load_session(tmp_path):
    from scripts.session_tracker import save_session, load_session, SessionState

    state = SessionState(
        session_id="test-001",
        ticket_id=99999,
        title="test incident",
        phase="analyse",
    )

    session_file = tmp_path / "session.json"
    save_session(session_file, state)

    # Verify the file was created
    assert session_file.exists()

    # Verify the file content
    loaded = load_session(session_file)
    assert loaded.session_id == "test-001"
    assert loaded.phase == "analyse"


def test_json_output_format(tmp_path):
    output_file = tmp_path / "results.json"
    write_results(output_file, [{"score": 0.9, "match": "pattern A"}])

    data = json.loads(output_file.read_text())
    assert isinstance(data, list)
    assert data[0]["score"] == 0.9
```

---

## 3.2 Testing Scripts That Write Files

```python
def test_csv_export(tmp_path):
    import csv

    output_csv = tmp_path / "export.csv"
    export_findings_to_csv(
        findings=[
            {"root_cause": "slip", "confidence": 0.9},
            {"root_cause": "localization", "confidence": 0.7},
        ],
        output_path=output_csv,
    )

    with output_csv.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["confidence"] == "0.9"


def test_creates_output_directory(tmp_path):
    nested = tmp_path / "deep" / "nested" / "output.txt"
    write_output(nested, "content")   # function should create parent dirs

    assert nested.exists()
    assert nested.read_text() == "content"
```

---

## 3.3 tmp_path_factory: Session-Scoped Temp Dirs

When a fixture needs a temp dir shared across the whole test session (not per-test):

```python
import pytest


@pytest.fixture(scope="session")
def shared_temp_dir(tmp_path_factory):
    """A temp dir that persists for the entire test session."""
    tmp_dir = tmp_path_factory.mktemp("shared")
    # Set up expensive shared resources here
    kb_file = tmp_dir / "kb.json"
    kb_file.write_text('{"patterns": []}')
    return tmp_dir
```

---

# PART 4 — MONKEYPATCH: ENV VARS AND FUNCTIONS

---

## 4.1 Environment Variables

```python
def test_loads_api_token(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "fake-token-for-test")
    monkeypatch.setenv("KB_MIN_CONFIDENCE", "0.9")

    settings = load_settings()
    assert settings.api_token == "fake-token-for-test"
    assert settings.kb_min_confidence == 0.9


def test_fails_without_required_env(monkeypatch):
    monkeypatch.delenv("API_TOKEN", raising=False)   # remove if present

    with pytest.raises(SystemExit):   # load_settings() calls sys.exit on failure
        load_settings()


def test_default_used_when_optional_absent(monkeypatch):
    monkeypatch.setenv("API_TOKEN", "fake-token")
    monkeypatch.delenv("KB_MIN_CONFIDENCE", raising=False)

    settings = load_settings()
    assert settings.kb_min_confidence == 0.7   # the default value
```

**Key insight:** monkeypatch changes are automatically reverted after each test. You don't need cleanup code — the fixture handles it.

---

## 4.2 Replacing Functions

```python
def test_search_uses_correct_query(monkeypatch):
    calls = []

    def fake_search(query: str, limit: int = 10) -> list[dict]:
        calls.append({"query": query, "limit": limit})
        return [{"match": "pattern A", "score": 0.9}]

    monkeypatch.setattr("scripts.knowledge_search.run_query", fake_search)

    results = search_and_format("motor stall", limit=5)

    assert len(calls) == 1
    assert calls[0]["query"] == "motor stall"
    assert calls[0]["limit"] == 5
    assert results[0]["match"] == "pattern A"
```

---

## 4.3 Changing Working Directory

```python
def test_script_finds_env_file(monkeypatch, tmp_path):
    """Script reads .env from the current working directory."""
    env_file = tmp_path / ".env"
    env_file.write_text("API_TOKEN=test-token\n")

    monkeypatch.chdir(tmp_path)   # change cwd to tmp_path

    settings = load_settings()   # will find .env in tmp_path
    assert settings.api_token == "test-token"
```

---

## 4.4 Worked Example: Testing a Config-Loading Function

```python
# scripts/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    api_token: str
    kb_min_confidence: float = 0.7
    model_config = SettingsConfigDict(env_file=".env")

def load_settings() -> Settings:
    try:
        return Settings()
    except Exception as e:
        raise SystemExit(f"Config error: {e}") from e


# tests/test_config.py
import pytest
from scripts.config import load_settings


class TestLoadSettings:
    def test_loads_required_fields(self, monkeypatch):
        monkeypatch.setenv("API_TOKEN", "my-pat-token")
        s = load_settings()
        assert s.api_token == "my-pat-token"

    def test_default_confidence(self, monkeypatch):
        monkeypatch.setenv("API_TOKEN", "any-token")
        monkeypatch.delenv("KB_MIN_CONFIDENCE", raising=False)
        s = load_settings()
        assert s.kb_min_confidence == 0.7

    def test_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv("API_TOKEN", "any-token")
        monkeypatch.setenv("KB_MIN_CONFIDENCE", "0.95")
        s = load_settings()
        assert s.kb_min_confidence == 0.95

    def test_missing_required_exits(self, monkeypatch):
        monkeypatch.delenv("API_TOKEN", raising=False)
        with pytest.raises(SystemExit):
            load_settings()

    def test_float_coercion(self, monkeypatch):
        monkeypatch.setenv("API_TOKEN", "token")
        monkeypatch.setenv("KB_MIN_CONFIDENCE", "0.85")   # string in env
        s = load_settings()
        assert isinstance(s.kb_min_confidence, float)     # coerced to float
```

---

# PART 5 — MOCKING WITH pytest-mock

---

## 5.1 Installing and Basic Usage

```bash
pip install pytest-mock
```

`pytest-mock` provides a `mocker` fixture with cleaner syntax than `unittest.mock`:

```python
def test_api_call_made(mocker):
    mock_get = mocker.patch("requests.get")
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"results": []}

    result = fetch_from_api("http://example.com/search", "motor stall")

    mock_get.assert_called_once()
    assert result == {"results": []}
```

---

## 5.2 Controlling Return Values

```python
def test_scorer_uses_embedding(mocker):
    # Simple return value
    mock_embed = mocker.patch("scripts.scorer.get_embedding")
    mock_embed.return_value = [0.1, 0.2, 0.3]

    score = compute_similarity("test query", "test candidate")
    mock_embed.assert_called()


def test_multiple_calls_return_different_values(mocker):
    mock_fetch = mocker.patch("scripts.session.fetch_ticket")
    # Each call returns a different value
    mock_fetch.side_effect = [
        {"id": 1, "title": "first"},
        {"id": 2, "title": "second"},
        {"id": 3, "title": "third"},
    ]

    r1 = fetch_ticket(1)
    r2 = fetch_ticket(2)

    assert r1["title"] == "first"
    assert r2["title"] == "second"
```

---

## 5.3 Simulating Failures

```python
def test_handles_api_timeout(mocker):
    import requests

    mock_get = mocker.patch("requests.get")
    mock_get.side_effect = requests.Timeout("Connection timed out")

    result = fetch_from_api("http://example.com", "query")

    # The function should handle the timeout gracefully
    assert result is None   # or assert result == []


def test_retries_on_failure(mocker):
    mock_get = mocker.patch("requests.get")
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"ok": True}

    # Fail twice, succeed on third attempt
    mock_get.side_effect = [
        Exception("network error"),
        Exception("network error"),
        mock_response,
    ]

    result = fetch_with_retry("http://example.com", max_retries=3)
    assert result == {"ok": True}
    assert mock_get.call_count == 3
```

---

## 5.4 Verifying Calls

```python
def test_posts_comment_with_correct_content(mocker):
    mock_post = mocker.patch("scripts.ado_client.post_comment")

    process_and_comment(ticket_id=99999, findings=["slip event detected"])

    # Verify the function was called exactly once
    mock_post.assert_called_once()

    # Verify it was called with specific arguments
    mock_post.assert_called_once_with(
        ticket_id=99999,
        comment="slip event detected",
    )

    # Check keyword argument specifically
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["ticket_id"] == 99999

    # Verify it was NOT called
    mock_other = mocker.patch("scripts.ado_client.create_ticket")
    mock_other.assert_not_called()
```

---

## 5.5 Worked Example: Testing a Search Function with HTTP

```python
# scripts/knowledge_search.py
import requests

def search_kb(query: str, min_confidence: float = 0.5) -> list[dict]:
    """Search the knowledge base via HTTP."""
    response = requests.get(
        "http://localhost:8000/search",
        params={"q": query, "min_confidence": min_confidence},
        timeout=5,
    )
    response.raise_for_status()
    return response.json()["results"]


# tests/test_knowledge_search.py
import pytest
from scripts.knowledge_search import search_kb


class TestSearchKb:
    def test_returns_results(self, mocker):
        mock_get = mocker.patch("scripts.knowledge_search.requests.get")
        mock_get.return_value.json.return_value = {
            "results": [
                {"match": "motor stall pattern", "confidence": 0.91},
            ]
        }
        mock_get.return_value.raise_for_status = lambda: None

        results = search_kb("motor stall")

        assert len(results) == 1
        assert results[0]["confidence"] == 0.91

    def test_passes_min_confidence(self, mocker):
        mock_get = mocker.patch("scripts.knowledge_search.requests.get")
        mock_get.return_value.json.return_value = {"results": []}
        mock_get.return_value.raise_for_status = lambda: None

        search_kb("test query", min_confidence=0.8)

        call_params = mock_get.call_args.kwargs["params"]
        assert call_params["min_confidence"] == 0.8

    def test_raises_on_http_error(self, mocker):
        import requests as req
        mock_get = mocker.patch("scripts.knowledge_search.requests.get")
        mock_get.return_value.raise_for_status.side_effect = req.HTTPError("404")

        with pytest.raises(req.HTTPError):
            search_kb("query")

    def test_returns_empty_on_no_results(self, mocker):
        mock_get = mocker.patch("scripts.knowledge_search.requests.get")
        mock_get.return_value.json.return_value = {"results": []}
        mock_get.return_value.raise_for_status = lambda: None

        results = search_kb("obscure unmatched query", min_confidence=0.99)
        assert results == []
```

---

# PART 6 — TESTING CLI SCRIPTS

---

## 6.1 Subprocess Approach

Use subprocess when the CLI script has complex argument handling or side effects that are hard to isolate:

```python
import subprocess
import sys
import json


def test_search_cli_returns_results():
    result = subprocess.run(
        [sys.executable, "scripts/knowledge_search.py", "search", "motor stall"],
        capture_output=True,
        text=True,
        cwd="/path/to/project",
        env={"PATH": "/usr/bin", "API_TOKEN": "test-token"},
    )
    assert result.returncode == 0
    assert "confidence" in result.stdout


def test_search_cli_json_output():
    result = subprocess.run(
        [sys.executable, "scripts/knowledge_search.py", "search", "--format=json", "slip"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    data = json.loads(result.stdout)
    assert isinstance(data, list)


def test_missing_required_arg_fails():
    result = subprocess.run(
        [sys.executable, "scripts/knowledge_search.py", "search"],  # no query
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "required" in result.stderr.lower()
```

---

## 6.2 Import Approach: Unit Testing Individual Functions

Better for testing specific logic without the overhead of subprocess:

```python
# Directly import and test the logic
from scripts.knowledge_search import compute_score, format_results, parse_args


def test_compute_score_perfect_match():
    score = compute_score("motor stall", ["motor stall pattern"])
    assert score > 0.85


def test_compute_score_no_match():
    score = compute_score("completely unrelated query", ["motor stall pattern"])
    assert score < 0.2


def test_format_results_truncates_long_titles():
    results = [{"title": "A" * 200, "score": 0.9}]
    formatted = format_results(results, max_title_len=80)
    assert len(formatted[0]["title"]) <= 80


def test_parse_args_defaults():
    args = parse_args(["search", "motor stall"])
    assert args.query == "motor stall"
    assert args.limit == 10   # default
    assert args.format == "text"   # default
```

---

## 6.3 capsys: Capturing stdout/stderr

```python
def test_prints_results_to_stdout(capsys):
    print_search_results([
        {"match": "motor stall", "confidence": 0.9},
        {"match": "slip event", "confidence": 0.7},
    ])

    captured = capsys.readouterr()
    assert "motor stall" in captured.out
    assert "0.9" in captured.out
    assert captured.err == ""   # nothing to stderr


def test_warns_to_stderr_on_low_confidence(capsys):
    import sys

    print_search_results([{"match": "weak match", "confidence": 0.3}])

    captured = capsys.readouterr()
    assert "low confidence" in captured.err.lower()
```

---

## 6.4 Choosing Subprocess vs Import

| Situation | Use |
|-----------|-----|
| Testing argument parsing and exit codes | subprocess |
| Testing that the script fails gracefully on bad input | subprocess |
| Unit testing scoring/parsing logic | import |
| Mocking HTTP calls and external services | import + mocker |
| Testing file output | import + tmp_path |
| CI smoke test ("does it run?") | subprocess |

---

# PART 7 — TEST COVERAGE

---

## 7.1 Running Coverage

```bash
# Run tests with coverage for the scripts directory
pytest --cov=scripts --cov-report=term-missing

# Generate HTML report (open htmlcov/index.html in browser)
pytest --cov=scripts --cov-report=html

# Fail if coverage drops below 80%
pytest --cov=scripts --cov-fail-under=80
```

Example output:
```
Name                    Stmts   Miss  Cover   Missing
-----------------------------------------------------
scripts/knowledge_search.py       45      7    84%   23, 47-51, 89
scripts/session_tracker.py     62     15    76%   34-38, 71-82
scripts/scorer.py          28      2    93%   45, 67
-----------------------------------------------------
TOTAL                     135     24    82%
```

"Missing" shows the line numbers not covered — these are your test gaps.

---

## 7.2 What to Cover (and What Not To)

**DO test:**
- Scoring/ranking logic
- Data parsing (log lines, JSON, CSV)
- Threshold comparisons and edge cases (`== limit`, `> limit`, `< 0`)
- Error handling paths (what happens when the API returns 404?)
- Config validation (missing fields, invalid values)

**DON'T test directly (mock instead):**
- File I/O — use `tmp_path` or mock the file operations
- HTTP calls — use `mocker.patch("requests.get")`
- Database queries — use a test database or mock the client
- External CLI tools — mock the `subprocess.run` call

```python
# DON'T do this — it makes your test depend on a real file
def test_bad():
    results = load_kb("/home/user/real_kb.json")   # fragile!
    assert len(results) > 0


# DO this — test with controlled data
def test_good(tmp_path):
    kb_file = tmp_path / "kb.json"
    kb_file.write_text('[{"pattern": "motor stall", "confidence": 0.9}]')

    results = load_kb(kb_file)
    assert len(results) == 1
    assert results[0]["pattern"] == "motor stall"
```

---

## 7.3 Coverage Targets

| Script type | Target | Rationale |
|------------|--------|-----------|
| Scoring / ranking functions | 90%+ | These directly affect output quality |
| Parsing functions | 85%+ | Edge cases here cause silent data corruption |
| Config loading | 80%+ | Missing coverage = untested failure modes |
| IO-heavy scripts (download, upload) | 60%+ | Mock the IO; test the logic around it |
| One-off utility scripts | 40%+ | Better than nothing |

---

## 7.4 Putting It Together: Test File Template

```python
# tests/unit/test_scorer.py
"""Tests for the scoring and ranking logic in scripts/scorer.py."""

import pytest
from typing import Any


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_patterns() -> list[dict[str, Any]]:
    return [
        {"id": 1, "description": "motor stall on ramp", "tags": ["motor", "ramp"]},
        {"id": 2, "description": "wheel slip on wet surface", "tags": ["slip", "wet"]},
        {"id": 3, "description": "localization failure in corridor", "tags": ["nav"]},
    ]


@pytest.fixture
def scorer(sample_patterns):
    from scripts.scorer import PatternScorer
    return PatternScorer(patterns=sample_patterns)


# ── Happy path ───────────────────────────────────────────────────────────────

class TestScore:
    def test_returns_float(self, scorer):
        result = scorer.score("motor fault", top_k=1)
        assert isinstance(result[0]["score"], float)

    def test_score_in_range(self, scorer):
        result = scorer.score("test query")
        for r in result:
            assert 0.0 <= r["score"] <= 1.0

    def test_top_k_limits_results(self, scorer, sample_patterns):
        result = scorer.score("any query", top_k=2)
        assert len(result) <= 2

    def test_exact_match_scores_high(self, scorer):
        result = scorer.score("motor stall", top_k=1)
        assert result[0]["score"] >= 0.7


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestScoreEdgeCases:
    def test_empty_query_raises(self, scorer):
        with pytest.raises(ValueError, match="empty"):
            scorer.score("")

    def test_empty_patterns_returns_empty(self):
        from scripts.scorer import PatternScorer
        s = PatternScorer(patterns=[])
        assert s.score("motor stall") == []

    @pytest.mark.parametrize("query", ["a", "  ", "\t"])
    def test_short_or_whitespace_query_raises(self, scorer, query):
        with pytest.raises(ValueError):
            scorer.score(query)
```

---

## Summary — What to Remember

| Concept | Rule |
|---------|------|
| Test discovery | Files: `test_*.py` or `*_test.py`; Functions: `test_*` |
| `pytest.raises(Exc)` | Context manager for asserting that code raises a specific exception |
| `@pytest.mark.parametrize` | Run one test with multiple input/expected pairs |
| `@pytest.fixture` | Setup/teardown; add as parameter to receive it |
| `scope="module"` | Share fixture across tests in one file (useful for slow setup) |
| `yield` in fixture | Code before yield = setup; code after yield = teardown |
| `tmp_path` | pytest-managed temp directory; cleaned up automatically |
| `monkeypatch.setenv` | Set env var for one test only; auto-reverted afterward |
| `monkeypatch.setattr` | Replace a function for one test only |
| `mocker.patch` | From pytest-mock; returns a MagicMock; cleaner than unittest.mock |
| `mock.side_effect = [v1, v2]` | Return different values on successive calls |
| `capsys.readouterr()` | Capture stdout/stderr from the tested code |
| subprocess vs import | subprocess for CLI/argparse tests; import for logic unit tests |
| Coverage targets | 90% scoring logic, 80% parsing, 60% IO-heavy |

---

# QUICK REFERENCE CARD

```
┌──────────────────────────────── pytest CHEAT SHEET ─────────────────────────────────────────┐
│                                                                                              │
│  RUN:       pytest -v                 pytest -k "test_score"                                │
│             pytest -x                 (stop on first failure)                               │
│             pytest --cov=scripts --cov-report=term-missing                                  │
│                                                                                              │
│  ASSERT:    assert result == expected                                                        │
│             assert 0.0 <= score <= 1.0                                                      │
│             pytest.raises(ValueError, match="msg")                                          │
│                                                                                              │
│  FIXTURES:  @pytest.fixture                  → function scope (default)                     │
│             @pytest.fixture(scope="module")  → one per test file                           │
│             yield value                      → setup before, teardown after                 │
│             tmp_path                         → managed temp dir (Path)                      │
│             capsys / caplog / monkeypatch    → built-in helpers                             │
│                                                                                              │
│  MONKEYPATCH:                                                                                │
│             monkeypatch.setenv("KEY", "val")                                                │
│             monkeypatch.delenv("KEY", raising=False)                                        │
│             monkeypatch.setattr(module, "fn_name", replacement)                             │
│             monkeypatch.chdir(tmp_path)                                                     │
│                                                                                              │
│  MOCKING (pytest-mock):                                                                      │
│             mocker.patch("module.function")   → MagicMock                                  │
│             mock.return_value = value                                                        │
│             mock.side_effect = [v1, v2, ...]                                                │
│             mock.side_effect = SomeException("msg")                                         │
│             mock.assert_called_once_with(arg1, kwarg=val)                                   │
│             mock.assert_not_called()                                                         │
│                                                                                              │
│  PARAMETRIZE:                                                                                │
│             @pytest.mark.parametrize("a,b,expected", [(1,2,3), (4,5,9)])                   │
│                                                                                              │
│  SKIP / XFAIL:                                                                               │
│             @pytest.mark.skip(reason="...")                                                  │
│             @pytest.mark.skipif(condition, reason="...")                                     │
│             @pytest.mark.xfail(reason="known bug")                                          │
│                                                                                              │
│  SUBPROCESS TEST PATTERN:                                                                    │
│             result = subprocess.run([sys.executable, "script.py", "arg"],                   │
│                                      capture_output=True, text=True)                        │
│             assert result.returncode == 0                                                    │
│                                                                                              │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```
