# ex05: CI-Ready C++ Project

A complete CI pipeline for the Week 5 exercises, runnable locally with `make`.

## Prerequisites

| Tool       | Required | Purpose                        |
|------------|:--------:|--------------------------------|
| CMake 3.20+| Yes      | Build system                   |
| Ninja      | Yes      | Fast parallel builds           |
| g++ / clang++| Yes    | C++20 compiler                 |
| lcov       | Yes*     | Coverage reports               |
| genhtml    | Yes*     | HTML coverage output           |
| bc         | Yes*     | Coverage threshold comparison  |
| clang++    | No       | Required for `fuzz` target     |

\* Only needed for the `coverage` target.

Install on Ubuntu/Debian:
```bash
sudo apt install cmake ninja-build g++ lcov bc
# Optional: sudo apt install clang   # for fuzzing
```

## Targets

| Target          | Description                                             |
|-----------------|---------------------------------------------------------|
| `make build`        | Debug build with all warnings                       |
| `make build-release`| Release build (-O2, no debug)                       |
| `make test`         | Run all GTest tests (debug build)                   |
| `make test-asan`    | Build + test with ASan + UBSan                      |
| `make test-tsan`    | Build + test with TSan                              |
| `make test-ubsan`   | Build + test with UBSan (standalone)                |
| `make coverage`     | Build with coverage, run tests, generate HTML, gate ≥80% |
| `make fuzz`         | 60-second libFuzzer run (requires clang)            |
| `make clean`        | Remove all build artifacts                          |

## Quick Start

```bash
cd ex05_ci_project

# Run one check:
make test-asan

# Run the full pipeline:
chmod +x run_all_checks.sh
./run_all_checks.sh
```

## Pipeline Stages

```
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌─────────┐
│  Build   │→│  Test   │→│  ASan   │→│  TSan   │→│ Coverage │→│  Fuzz   │
│ Debug +  │ │ ctest   │ │ memory  │ │ races   │ │ lcov     │ │ 60-sec  │
│ Release  │ │         │ │ errors  │ │         │ │ ≥80%     │ │ corpus  │
└─────────┘ └─────────┘ └─────────┘ └─────────┘ └──────────┘ └─────────┘
```

All stages must pass (exit 0) for the pipeline to succeed.

## Why Each Stage Matters

- **Debug + Release builds**: Catches warnings treated as errors, ensures -O2 doesn't break anything
- **ASan**: Detects heap-buffer-overflow, use-after-free, stack overflow, memory leaks
- **TSan**: Detects data races, lock-order inversions (CANNOT run with ASan — mutually exclusive)
- **UBSan**: Detects signed integer overflow, null deref, alignment violations, shift errors
- **Coverage**: Ensures tests exercise ≥80% of lines (necessary but not sufficient — see puzzle01)
- **Fuzz**: Discovers edge cases in parsers/serializers that humans miss

## `run_all_checks.sh`

Runs all targets sequentially and prints a color-coded summary:

```
  PASS  Debug Build
  PASS  Release Build
  PASS  Unit Tests
  PASS  ASan + UBSan Tests
  PASS  TSan Tests
  PASS  UBSan Tests
  PASS  Code Coverage (≥80%)
  SKIP  Fuzzing (clang++ not found)

  Passed: 7  Failed: 0  Skipped: 1

PIPELINE PASSED
```

Exit code is 0 on success, 1 if any required check fails.
Fuzzing is optional (skipped if clang++ is not installed).
