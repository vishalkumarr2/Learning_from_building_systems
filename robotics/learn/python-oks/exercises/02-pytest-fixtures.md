# Exercise 02 — Testing CLI Scripts with pytest

### Companion exercises for `02-testing-cli-scripts.md`

**Estimated time:** 50 minutes  
**Prerequisite:** [02-testing-cli-scripts.md](../02-testing-cli-scripts.md)

Write answers on paper before expanding `Answer` blocks. Part B exercises are the most important — debugging broken tests is a core skill. Part C is free-build.

---

## Part A — Concepts Check

**1.** What is the difference between `scope="function"` and `scope="session"` for a pytest fixture?

<details><summary>Answer</summary>

| Scope | Created | Destroyed | Use when |
|---|---|---|---|
| `"function"` (default) | Before each test | After each test | State must be isolated between tests (tmp files, in-memory state) |
| `"class"` | Once per test class | After the class | Shared setup within a class |
| `"module"` | Once per file | After the file | Expensive setup shared across a module (DB connection) |
| `"session"` | Once for the entire run | After all tests finish | Very expensive setup (start a server, compile a large dataset) |

```python
@pytest.fixture(scope="session")
def db_connection():
    conn = create_connection()   # expensive — do once
    yield conn
    conn.close()                 # teardown after all tests
```

**Gotcha:** `scope="session"` fixtures must not be modified by individual tests — the modified state leaks to subsequent tests. Use `scope="function"` for anything that tests write to.

</details>

- [ ] Done

---

**2.** How do you use a yield fixture to guarantee cleanup runs even if a test fails?

<details><summary>Answer</summary>

Anything after `yield` in a fixture runs as teardown, regardless of whether the test raised an exception:

```python
import pytest
from pathlib import Path

@pytest.fixture
def temp_config(tmp_path):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("timeout: 30\n")
    yield cfg_path
    # This runs even if the test fails:
    if cfg_path.exists():
        cfg_path.unlink()
```

pytest wraps the `yield` in a try/finally internally. If the test raises, pytest catches it, runs the teardown code, then re-raises the exception. This means:

- Resources are always released (file handles, DB connections, temp files)
- The teardown itself can `assert` or raise — but if teardown raises, the original test failure may be masked. Keep teardown minimal and infallible.

</details>

- [ ] Done

---

**3.** What is the difference between `mocker.patch` (pytest-mock) and `monkeypatch.setattr`?

<details><summary>Answer</summary>

Both replace a name temporarily for the duration of a test, but differ in style and scope:

```python
# monkeypatch.setattr — built-in to pytest
def test_with_monkeypatch(monkeypatch):
    monkeypatch.setattr("mymodule.requests.get", lambda url, **kw: mock_response)
    # auto-reverts after test

# mocker.patch — from pytest-mock (wraps unittest.mock.patch)
def test_with_mocker(mocker):
    mock_get = mocker.patch("mymodule.requests.get")
    mock_get.return_value.json.return_value = {"id": 1}
    mock_get.return_value.status_code = 200
    # auto-reverts after test
```

| | `monkeypatch.setattr` | `mocker.patch` |
|---|---|---|
| Source | pytest built-in | pytest-mock (requires install) |
| Returns | nothing | a `MagicMock` object |
| Assertion support | Manual | `mock_get.assert_called_once_with(...)` |
| Use for | Simple replacements, env vars, dict entries | HTTP calls, complex mocks, call assertions |

For HTTP calls and external services, `mocker.patch` is usually cleaner because the returned `MagicMock` has `.assert_called_with()`, `.call_count`, `.return_value`, etc.

</details>

- [ ] Done

---

**4.** When should you test a CLI script by importing it directly vs. running it with `subprocess`?

<details><summary>Answer</summary>

**Import directly** (preferred for unit/integration tests):
```python
from mymodule.cli import run_analysis

def test_run_analysis_empty_input(tmp_path):
    result = run_analysis(csv_path=tmp_path / "empty.csv", threshold=0.01)
    assert result["count"] == 0
```
- Fast (no subprocess overhead)
- Works with pytest fixtures, monkeypatching, coverage instrumentation
- Requires the CLI to be importable (functions extracted from `if __name__ == "__main__"`)

**Use subprocess** when:
```python
import subprocess

def test_cli_exit_code_on_missing_file():
    result = subprocess.run(
        ["python", "analysis.py", "--csv", "/nonexistent.csv"],
        capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "File not found" in result.stderr
```
- Testing argument parsing, exit codes, or stderr/stdout formatting
- The script has side effects you cannot easily mock (writes to system paths)
- Testing the `if __name__ == "__main__"` entry point specifically
- Coverage does NOT instrument subprocess by default — keep subprocess tests to the thin entry-point layer

**Best practice:** Keep the main logic in importable functions. Use subprocess only to test the CLI interface (argument parsing, exit codes).

</details>

- [ ] Done

---

**5.** What does `capsys.readouterr()` return, and when is it useful?

<details><summary>Answer</summary>

`capsys.readouterr()` returns a `CaptureResult` named tuple with two attributes:
- `.out` — everything printed to `sys.stdout` during the test
- `.err` — everything printed to `sys.stderr` during the test

After calling `readouterr()`, the internal buffers are cleared.

```python
def print_summary(findings: list[str]) -> None:
    print(f"Found {len(findings)} issues:")
    for f in findings:
        print(f"  - {f}")

def test_summary_output(capsys):
    print_summary(["disk full", "timeout"])
    captured = capsys.readouterr()
    assert "Found 2 issues" in captured.out
    assert "disk full" in captured.out
    assert captured.err == ""   # no error output
```

Useful for:
- Testing scripts that use `print()` instead of logging
- Verifying progress messages or summary tables output to stdout
- Confirming that error paths write to stderr, not stdout

Note: `capsys` captures at the Python level. Use `capfd` instead if you need to capture output from C extensions or subprocesses.

</details>

- [ ] Done

---

## Part B — Fix the Broken Tests

Each snippet below has a subtle mistake. Identify the problem and write the corrected version.

**1.** This test passes locally and fails in CI. Why?

```python
import json
from pathlib import Path

def save_report(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f)

def test_save_report_creates_file():
    path = Path("test_output.json")
    save_report({"status": "ok"}, path)
    assert path.exists()
    content = json.loads(path.read_text())
    assert content["status"] == "ok"
```

<details><summary>Answer</summary>

**Problem:** The test writes to `"test_output.json"` in the current working directory. Locally the CWD is the project root and the developer has write permission. In CI the CWD may be read-only, or a parallel test may clobber the same file.

**Fix:** Use the `tmp_path` fixture to get a unique writable directory per test:

```python
import json
from pathlib import Path

def test_save_report_creates_file(tmp_path):
    path = tmp_path / "report.json"
    save_report({"status": "ok"}, path)
    assert path.exists()
    content = json.loads(path.read_text())
    assert content["status"] == "ok"
```

`tmp_path` is a `pathlib.Path` to a fresh temporary directory that pytest creates, provides to the test, and cleans up automatically. Each test invocation gets a unique directory.

</details>

- [ ] Done

---

**2.** This test always passes even when the mock is not working. What is wrong?

```python
# mymodule/fetcher.py
import requests

def fetch_data(url: str) -> dict:
    return requests.get(url).json()

# tests/test_fetcher.py
import pytest
import requests
from unittest.mock import patch

def test_fetch_data_returns_json():
    with patch("requests.get") as mock_get:
        mock_get.return_value.json.return_value = {"id": 42}
        from mymodule.fetcher import fetch_data
        result = fetch_data("https://api.example.com/item/1")
    assert result == {"id": 42}
```

<details><summary>Answer</summary>

**Problem:** The patch target is `"requests.get"` — this patches the `get` function in the `requests` module itself. But `mymodule.fetcher` already imported `requests` at module load time and calls `requests.get` via its own reference. If the module was imported before the `with patch(...)` block, the reference is already bound and the patch has no effect.

**Fix:** Patch the name where it is **used**, not where it is defined:

```python
def test_fetch_data_returns_json():
    with patch("mymodule.fetcher.requests.get") as mock_get:
        mock_get.return_value.json.return_value = {"id": 42}
        from mymodule.fetcher import fetch_data
        result = fetch_data("https://api.example.com/item/1")
    assert result == {"id": 42}
```

**Rule:** Always patch `"the.module.where.it.is.used.name"`, not `"the.module.where.it.was.defined.name"`. The `unittest.mock` documentation calls this "patch where the name is looked up".

</details>

- [ ] Done

---

**3.** This fixture leaks a resource. Find the bug and fix it.

```python
import pytest
import sqlite3

@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE events (id INTEGER, message TEXT)")
    return conn   # <-- no yield, no teardown

def test_insert_event(db):
    db.execute("INSERT INTO events VALUES (1, 'started')")
    db.commit()
    row = db.execute("SELECT message FROM events WHERE id=1").fetchone()
    assert row[0] == "started"
```

<details><summary>Answer</summary>

**Problem:** The fixture uses `return conn` instead of `yield conn`. There is no teardown — the connection is never closed. With `:memory:` SQLite this is harmless, but with real files or connection pools it leaks resources.

**Fix:**
```python
@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE events (id INTEGER, message TEXT)")
    yield conn          # test runs here
    conn.close()        # teardown — runs even if the test fails
```

The `yield` turns the fixture into a generator. pytest calls `next()` to get the value, hands it to the test, then calls `next()` again (which hits the code after `yield`) for teardown. This is equivalent to a try/finally.

</details>

- [ ] Done

---

**4.** This parametrized test is structured incorrectly. The `pytest.raises` block does not catch what you think it does.

```python
import pytest

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

@pytest.mark.parametrize("a,b,expected", [
    (10, 2, 5.0),
    (9,  3, 3.0),
])
def test_divide_normal(a, b, expected):
    assert divide(a, b) == expected

@pytest.mark.parametrize("a,b", [
    (10, 0),
    (0,  0),
])
def test_divide_by_zero(a, b):
    with pytest.raises(ValueError):
        result = divide(a, b)
        assert result is None    # BUG: this line is inside pytest.raises
```

<details><summary>Answer</summary>

**Problem:** The `assert result is None` line is inside the `with pytest.raises(ValueError):` block. The `pytest.raises` context manager exits as soon as the `ValueError` is raised by `divide()`. The `assert` on the next line is **never reached** — not because it passes, but because execution jumps to the end of the `with` block the moment the exception fires.

This means you could replace `assert result is None` with `assert False` and the test would still "pass" — silently skipping your assertion.

**Fix:** Put the assertion outside the `with` block:

```python
@pytest.mark.parametrize("a,b", [
    (10, 0),
    (0,  0),
])
def test_divide_by_zero(a, b):
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(a, b)
    # Assertions here run after the exception is confirmed:
    # (nothing to assert on result since it was never assigned)
```

**Better pattern when you need to inspect the exception:**
```python
def test_divide_by_zero_message():
    with pytest.raises(ValueError) as exc_info:
        divide(10, 0)
    assert "Cannot divide by zero" in str(exc_info.value)
```

</details>

- [ ] Done

---

## Part C — Build Exercises

### Exercise C1: File-Based Test Suite

Given this function:

```python
import json
from pathlib import Path

def save_findings(findings: list[dict], output_path: Path) -> None:
    with open(output_path, "w") as f:
        json.dump({"findings": findings, "count": len(findings)}, f, indent=2)
```

Write a complete pytest test module that:
- Uses `tmp_path` for all file I/O
- Tests that 3 findings produce the correct JSON structure (keys, count, content)
- Tests that empty findings produce `{"findings": [], "count": 0}`
- Tests that the output file is actually created on disk

<details><summary>Answer</summary>

```python
# tests/test_save_findings.py
import json
import pytest
from pathlib import Path

# Assume save_findings is importable from your module:
# from mymodule.report import save_findings

def save_findings(findings: list[dict], output_path: Path) -> None:
    with open(output_path, "w") as f:
        json.dump({"findings": findings, "count": len(findings)}, f, indent=2)


@pytest.fixture
def sample_findings() -> list[dict]:
    return [
        {"id": 1, "severity": "high",   "message": "connection timeout"},
        {"id": 2, "severity": "medium", "message": "retry limit hit"},
        {"id": 3, "severity": "low",    "message": "slow response"},
    ]


def test_file_is_created(tmp_path, sample_findings):
    output = tmp_path / "findings.json"
    assert not output.exists()   # precondition
    save_findings(sample_findings, output)
    assert output.exists()


def test_count_matches_findings(tmp_path, sample_findings):
    output = tmp_path / "findings.json"
    save_findings(sample_findings, output)
    data = json.loads(output.read_text())
    assert data["count"] == 3


def test_findings_content(tmp_path, sample_findings):
    output = tmp_path / "findings.json"
    save_findings(sample_findings, output)
    data = json.loads(output.read_text())
    assert len(data["findings"]) == 3
    assert data["findings"][0]["severity"] == "high"
    assert data["findings"][2]["id"] == 3


def test_empty_findings(tmp_path):
    output = tmp_path / "empty.json"
    save_findings([], output)
    data = json.loads(output.read_text())
    assert data == {"findings": [], "count": 0}


def test_output_is_valid_json(tmp_path, sample_findings):
    output = tmp_path / "findings.json"
    save_findings(sample_findings, output)
    # json.loads raises if the file is not valid JSON
    parsed = json.loads(output.read_text())
    assert isinstance(parsed, dict)
    assert "findings" in parsed
    assert "count" in parsed
```

</details>

- [ ] Done

---

### Exercise C2: Mocking an External API Call

Given this function:

```python
import requests

def fetch_ticket_details(ticket_id: int, api_token: str) -> dict:
    response = requests.get(
        f"https://api.example.com/workitems/{ticket_id}",
        headers={"Authorization": f"Bearer {api_token}"},
    )
    response.raise_for_status()
    return response.json()
```

Write tests that:
1. Mock a successful 200 response and verify the returned dict
2. Mock a 404 response and verify `requests.HTTPError` is raised
3. Mock a `requests.exceptions.ConnectionError` and verify it propagates
4. Verify the correct URL and headers are passed to `requests.get`

<details><summary>Answer</summary>

```python
# tests/test_fetch_ticket.py
import pytest
import requests
from unittest.mock import MagicMock, patch

# from mymodule.client import fetch_ticket_details

def fetch_ticket_details(ticket_id: int, api_token: str) -> dict:
    response = requests.get(
        f"https://api.example.com/workitems/{ticket_id}",
        headers={"Authorization": f"Bearer {api_token}"},
    )
    response.raise_for_status()
    return response.json()


@pytest.fixture
def mock_successful_response():
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {"id": 42, "title": "Sensor dropout", "state": "Active"}
    mock.raise_for_status.return_value = None
    return mock


def test_successful_response_returns_dict(mock_successful_response):
    with patch("requests.get", return_value=mock_successful_response):
        result = fetch_ticket_details(ticket_id=42, api_token="tok-abc")
    assert result["id"] == 42
    assert result["title"] == "Sensor dropout"


def test_404_raises_http_error():
    mock = MagicMock()
    mock.status_code = 404
    mock.raise_for_status.side_effect = requests.HTTPError(
        "404 Client Error: Not Found"
    )
    with patch("requests.get", return_value=mock):
        with pytest.raises(requests.HTTPError, match="404"):
            fetch_ticket_details(ticket_id=99999, api_token="tok-abc")


def test_connection_error_propagates():
    with patch(
        "requests.get",
        side_effect=requests.exceptions.ConnectionError("Network unreachable"),
    ):
        with pytest.raises(requests.exceptions.ConnectionError):
            fetch_ticket_details(ticket_id=1, api_token="tok-abc")


def test_correct_url_and_headers_are_used(mock_successful_response):
    with patch("requests.get", return_value=mock_successful_response) as mock_get:
        fetch_ticket_details(ticket_id=42, api_token="my-secret-token")

    mock_get.assert_called_once_with(
        "https://api.example.com/workitems/42",
        headers={"Authorization": "Bearer my-secret-token"},
    )
```

Note: `mock.raise_for_status.side_effect = requests.HTTPError(...)` makes the mock raise when `.raise_for_status()` is called — this is how you simulate 4xx/5xx responses from `requests`.

</details>

- [ ] Done

---

### Exercise C3: Full Script Test Suite

Design a complete test suite for this search function:

```python
def search(
    query: str,
    items: list[dict],
    min_confidence: float = 0.5,
) -> list[dict]:
    """Return items whose 'text' field matches query, sorted by confidence descending."""
    results = []
    for item in items:
        score = compute_similarity(item["text"], query)
        if score >= min_confidence:
            results.append({**item, "confidence": score})
    return sorted(results, key=lambda x: x["confidence"], reverse=True)
```

Assume `compute_similarity(a: str, b: str) -> float` is importable from the same module and returns a float in `[0.0, 1.0]`.

Write:
1. A fixture providing a standard set of 5 test items
2. Tests for: correct item filtering, descending sort, threshold enforcement, empty query, empty items list
3. A parametrized test covering multiple `(query, min_confidence, expected_count)` combinations

<details><summary>Answer</summary>

```python
# tests/test_search.py
import pytest
from unittest.mock import patch


# from mymodule.search import search

def search(query, items, min_confidence=0.5):
    results = []
    for item in items:
        score = compute_similarity(item["text"], query)
        if score >= min_confidence:
            results.append({**item, "confidence": score})
    return sorted(results, key=lambda x: x["confidence"], reverse=True)


@pytest.fixture
def sample_items():
    return [
        {"id": 1, "text": "motor controller fault detected"},
        {"id": 2, "text": "encoder signal lost on axis 2"},
        {"id": 3, "text": "battery voltage below threshold"},
        {"id": 4, "text": "motor driver overheated"},
        {"id": 5, "text": "navigation estimator diverged"},
    ]


def _make_similarity_map(mapping: dict[str, float]):
    """Helper: return a function that looks up similarity from a fixed dict."""
    def sim(text: str, query: str) -> float:
        return mapping.get(text, 0.0)
    return sim


def test_returns_matching_items(sample_items):
    scores = {
        "motor controller fault detected": 0.9,
        "motor driver overheated": 0.7,
        "encoder signal lost on axis 2": 0.3,
        "battery voltage below threshold": 0.1,
        "navigation estimator diverged": 0.2,
    }
    with patch("__main__.compute_similarity", side_effect=_make_similarity_map(scores)):
        results = search("motor fault", sample_items, min_confidence=0.5)
    assert len(results) == 2
    result_ids = {r["id"] for r in results}
    assert result_ids == {1, 4}


def test_results_sorted_descending(sample_items):
    scores = {
        "motor controller fault detected": 0.9,
        "motor driver overheated": 0.7,
        "navigation estimator diverged": 0.6,
        "encoder signal lost on axis 2": 0.2,
        "battery voltage below threshold": 0.1,
    }
    with patch("__main__.compute_similarity", side_effect=_make_similarity_map(scores)):
        results = search("motor fault", sample_items, min_confidence=0.5)
    confidences = [r["confidence"] for r in results]
    assert confidences == sorted(confidences, reverse=True)


def test_threshold_excludes_low_scores(sample_items):
    scores = {item["text"]: 0.49 for item in sample_items}
    with patch("__main__.compute_similarity", side_effect=_make_similarity_map(scores)):
        results = search("anything", sample_items, min_confidence=0.5)
    assert results == []


def test_empty_query_returns_empty(sample_items):
    with patch("__main__.compute_similarity", return_value=0.0):
        results = search("", sample_items)
    assert results == []


def test_empty_items_list():
    with patch("__main__.compute_similarity", return_value=0.9):
        results = search("motor fault", [])
    assert results == []


@pytest.mark.parametrize("min_conf,expected_count", [
    (0.0,  5),   # all items pass
    (0.5,  3),   # 3 items above 0.5
    (0.8,  1),   # only 1 above 0.8
    (1.0,  0),   # none are perfect matches
    (0.99, 0),   # edge: just below 1.0
])
def test_threshold_parametrized(sample_items, min_conf, expected_count):
    # Items get scores: 0.9, 0.7, 0.6, 0.3, 0.1 (in order of sample_items)
    score_values = [0.9, 0.7, 0.6, 0.3, 0.1]
    scores = {item["text"]: score for item, score in zip(sample_items, score_values)}

    with patch("__main__.compute_similarity", side_effect=_make_similarity_map(scores)):
        results = search("test query", sample_items, min_confidence=min_conf)

    assert len(results) == expected_count
```

**Note on the `patch` target:** In a real project, replace `"__main__.compute_similarity"` with the actual module path, e.g. `"mymodule.search.compute_similarity"`. The patch target must be the name as it appears in the module being tested.

</details>

- [ ] Done
